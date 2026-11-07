"""案B 04（約3.7秒）「速さは、時速せんひゃくさんじゅっキロ。旅客機より速い。」

背景：03 の最後の地図（桂浜の点・放物線つき）。カメラの押し込みと破線の流れは 03 からそのまま続く（_bmap.drift、b03.dash_phase）。
最初の 8 フレームで約 40% 暗くし、メーターが出ているあいだは約 72% まで落とす。「桂浜」のラベルは同じ 8 フレームで消す。
地図は 10 フレーム目からは 1 回だけ描いた板を、押し込みの分だけ warpAffine で拡大して使う（弧と点は毎フレーム描く）。
0–6  ：03 の文字（「桂浜から」「10km先へ」）が上へ抜けて消え（ぼかし 0→4）、縮尺も消える。
4–23 ：メーターが組み上がる。半円が 0→1,500 まで expo_out で描かれ、目盛りが 0.35 フレームずつ外へ back_out で出て、
       目盛りの数字はそれぞれ 2 フレーム遅れて 6px 上がりながら出る。軸のキャップと針は back_out で 0→1。
ジ（じそく）：針は 3 フレーム沈んでから（−30km/h）跳ね上がり、キの少し前に 1,130 を越えて 1,160 あたりまで行き、
       ばね（ζ0.32・2.0Hz）で 1,130 に戻る。止まったあとも ±4km/h のノイズでわずかに震える。速く回るあいだは針の残像 3 本。
       「時速 約◯km」はオドメーター（針の値。1,130 を超えた数は出さない）。
キ（キロ）：オドメーターが 1,130 で決まる（落ち＋1.06→1.0）。1,130km を強調。
リョ（りょかくき）：900→1,130 の帯が明るく（SKY→FOAM 0.7）塗られていき（越えた分）、「旅客機」の目盛りが 24→34px にのびて明るくなる。
       下に「旅客機（約900km/h）より速い」が下から出る（行のマスク）。音速（1,224 km/h）には届かないので、音速の目盛りは描かない。
61–105：メーター全体が MC まわりに 1.00→1.06 へ押し込まれる（背景はその 0.3 倍＝奥行き）。
最後の 6 フレーム：メーターは 1.0→0.94 に縮みながらぼけて消え（expo_in）、文字は上へ抜けて消える。暗くした地図だけが残る。
2026-11-07 訂正：10km 版（v＝315 m/s＝時速約1,130km、音速の 0.93 倍）。
2026-11-07 動きの仕様（P0/P1）に合わせて作り直し。
"""
import math

import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W, H
from beats import _mapui as U
from beats import _geo as G
from cutB.beats import _bmap as B, b03, _motion as M
from cutB.beats._m34 import Board, blend_mask

MC = (540.0, 952.0)                 # メーターの中心（半円の底）
R = 340.0
VMAX, V = 1500, 1130
JET = 900.0                         # ジェット旅客機の巡航（約900 km/h）
DIM = 0.40
Y_MAIN, Y_NOTE = 1112, 1196
LIVE = 10                           # このフレームまでは地図を毎フレーム描く（暗くなりきるまで）
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
_r = np.hypot((_xx - MC[0]) / 470, (_yy - (MC[1] - 60)) / 380)
_VIG = (np.clip(1.25 - _r, 0, 1) ** 1.5).astype(np.float32)
# メーターの面（半円＋数字の帯）の後ろだけ、地図の線をやわらかく沈める
_rd = np.hypot(_xx - MC[0], np.minimum(_yy - MC[1], 0) * 1.0 + np.maximum(_yy - (MC[1] + 250), 0))
_DISC = (np.clip((R + 150 - _rd) / 130.0, 0, 1) ** 1.2).astype(np.float32)
del _yy, _xx, _r, _rd

# 針のばね
ZETA, HZ = 0.32, 2.0
W_LIN = 0.55                        # 上がりの終わりに残す速さ（expo_out と直線の混ぜ方）


def _th(v):
    return np.pi * (1 - min(max(v, -60), VMAX) / VMAX)


def timing(beat):
    mo = beat['moras']
    assert mo[4][0] == 'ジ' and mo[15][0] == 'キ' and mo[17][0] == 'リョ', mo   # ジュッキロ の キ／リョカクキ の リョ
    mf = lambda k: timeline.mora_frame(beat, k)
    f_ji, f_ki, f_o = mf(4), mf(15), mf(17)
    s0 = f_ji - 3                      # 針が沈みはじめる（ジで跳ね上がる）
    return dict(ji=f_ji, ki=f_ki, o=f_o, s0=s0, a=s0 + 3, fa=f_ki - 2, b0=max(3, f_ji - 19))


def needle(t, T):
    """フレーム t（小数可）の針の値（km/h）。"""
    s0, a, fa = T['s0'], T['a'], T['fa']
    if t <= s0:
        return 0.0
    if t <= a:
        return -30.0 * M.ease((t - s0) / float(a - s0), 'sine_inout')
    if t <= fa:
        s = (t - a) / float(fa - a)
        e = (1 - W_LIN) * M.ease(s, 'expo_out') + W_LIN * s
        return -30.0 + (V + 30.0) * e
    va = (V + 30.0) * W_LIN / (fa - a) * M.FPS          # 1,130 を越えるときの速さ（km/h／秒）
    w = 2 * math.pi * HZ
    wd = w * math.sqrt(1 - ZETA ** 2)
    k = (t - fa) / M.FPS
    v = V + va / wd * math.exp(-ZETA * w * k) * math.sin(wd * k)
    v += M.jitter(t, 4.0, 8.0, seed=11) * M.ease((t - fa - 8) / 10.0, 'sine_inout')
    return v


PUSH = (61, 105, 0.06)              # メーターの押し込み（はじめ, おわり, 量）：針が止まってから、最後の縮みの前まで


def meter_push(i, n):
    a, b, k = PUSH
    return 1.0 + k * M.ease((i - a) / float(b - a), 'sine_inout')


# ---------------------------------------------------------------- メーターの板
_PAD = 96


def _box():
    x0, y0 = int(MC[0] - R - _PAD), int(MC[1] - R - _PAD)
    w, h = int(2 * (R + _PAD)), int(R + _PAD + 60)
    return x0, y0, w, h


def meter_board(value, i, T, ghosts=(), snd=0.0, excess=0.0):
    """メーターの板（線・目盛り・針。数字の目盛りは別に重ねる）。i はビートの中のフレーム（組み上がりの動き）。"""
    x0, y0, w, h = _box()
    L = core.SS(s=3, w=w, h=h)
    d = L.draw
    cx, cy = MC[0] - x0, MC[1] - y0
    b0 = T['b0']
    # 半円：0→1,500 まで expo_out で描かれる
    ap = M.ease((i - b0) / 12.0, 'expo_out')
    if ap > 0.002:
        d.arc(L.box(cx - R, cy - R, cx + R, cy + R), 180, 180 + 180 * ap, fill=L.col(core.FOAM, 0.85), width=L.w(2.2))
    rr = R - 14
    if value > 0:
        a1 = 180 + 180 * min(value, VMAX) / VMAX
        d.arc(L.box(cx - rr, cy - rr, cx + rr, cy + rr), 180, a1, fill=L.col(core.SKY, 0.55), width=L.w(7))
    if excess > 0.002 and value > JET:
        e1 = JET + (min(value, V) - JET) * excess
        d.arc(L.box(cx - rr, cy - rr, cx + rr, cy + rr), 180 + 180 * JET / VMAX, 180 + 180 * e1 / VMAX,
              fill=L.col(core.FOAM, 0.7 * min(1.0, excess * 1.5)), width=L.w(7))
    # 目盛り：0.35 フレームずつ、外へ 8px の back_out で出る
    for k, v in enumerate(range(0, VMAX + 1, 50)):
        p = M.ease((i - (b0 + 1) - 0.35 * k) / 7.0, 'back_out')
        if p <= 0.01:
            continue
        th = _th(v)
        major = v % 300 == 0
        mid = v % 100 == 0
        off = -8 * (1 - p)
        r0 = R - (30 if major else 20 if mid else 12) + off
        r1 = R + off
        al = min(1.0, p * 1.6)
        col = L.col(core.FOAM, (0.85 if major else 0.45) * al)
        d.line(L.p([(cx + r0 * np.cos(th), cy - r0 * np.sin(th)), (cx + r1 * np.cos(th), cy - r1 * np.sin(th))]),
               fill=col, width=L.w(2.4 if major else 1.4 if mid else 1.0))
    # 旅客機の目盛り（弧の外側へ）：リョで 24→34px にのびる
    pj = M.ease((i - (b0 + 1) - 0.35 * 18) / 7.0, 'back_out')
    if pj > 0.01:
        th = _th(JET)
        sa = (0.55 + 0.45 * snd) * min(1.0, pj * 1.6)
        ln = 24 + 10 * M.ease((i - T['o'] + 1) / 7.0, 'back_out')
        r0, r1 = R + 6, R + 6 + ln * pj
        d.line(L.p([(cx + r0 * np.cos(th), cy - r0 * np.sin(th)), (cx + r1 * np.cos(th), cy - r1 * np.sin(th))]),
               fill=L.col(core.FOAM, sa), width=L.w(2.2))
    # 針（と残像）・軸のキャップ
    cs = M.ease((i - (b0 + 2)) / 8.0, 'back_out')
    if cs > 0.01:
        for gv, ga in list(ghosts) + [(value, 1.0)]:
            th = _th(gv)
            rn = (R - 44) * cs
            tip = (cx + rn * np.cos(th), cy - rn * np.sin(th))
            bx, by = cx - 26 * cs * np.cos(th), cy + 26 * cs * np.sin(th)
            nx, ny = -np.sin(th), -np.cos(th)
            hw = 5 * cs
            d.polygon(L.p([(bx + nx * hw, by + ny * hw), tip, (bx - nx * hw, by - ny * hw)]),
                      fill=L.col(core.FOAM, ga * min(1.0, cs)))
        d.ellipse(L.box(cx - 13 * cs, cy - 13 * cs, cx + 13 * cs, cy + 13 * cs), fill=L.col(core.FOAM, min(1.0, cs)))
        d.ellipse(L.box(cx - 5 * cs, cy - 5 * cs, cx + 5 * cs, cy + 5 * cs), fill=L.col(core.DEEP, min(1.0, cs)))
    return Board.from_straight(L.done(), x0, y0)


_LB = {}


def _label_boards():
    if not _LB:
        out = []
        for v in range(0, VMAX + 1, 300):
            th = _th(v)
            rl = R - 66
            p = U.Patch([dict(text='{:,}'.format(v), x=MC[0] + rl * np.cos(th), y=MC[1] - rl * np.sin(th) + 11, size=30,
                              weight='Medium', alpha=0.8, track=0)])
            out.append((v // 50, Board.from_patch(p)))
        th = _th(JET)
        r = R + 44
        x, y = MC[0] + r * np.cos(th), MC[1] - r * np.sin(th)
        jet = Board.from_patch(U.Patch([dict(text='旅客機', x=x + 2, y=y - 4, size=28, weight='Medium', anchor='lm',
                                             track=0.06)]))
        _LB.update(labels=out, jet=jet)
    return _LB


def meter_group(value, i, T, ghosts=(), snd=0.0, excess=0.0):
    """メーター＋目盛りの数字＋「旅客機」を1枚の板に。"""
    Bd = meter_board(value, i, T, ghosts, snd, excess)
    LB = _label_boards()
    b0 = T['b0']
    for k, lb in LB['labels']:
        p = M.ease((i - (b0 + 1) - 0.35 * k - 2) / 7.0, 'cubic_out')
        if p > 0.01:
            Bd.over(lb, alpha=p, dy=6 * (1 - p))
    k = 18
    p = M.ease((i - (b0 + 1) - 0.35 * k - 2) / 7.0, 'cubic_out')
    if p > 0.01:
        Bd.over(LB['jet'], alpha=p * (0.6 + 0.4 * snd), dy=6 * (1 - p))
    return Bd


# ---------------------------------------------------------------- 背景
def background(t, dim, label_a=1.0):
    """03 の続きの地図（毎フレーム描く版。互換のため残す）。"""
    f = b03.map_frame(t, 1.0)
    f = b03.overlay_static(f, t, 1.0, track_u=1.0, head=False, dot_a=1.0, label_a=label_a)
    return f * (1 - dim) + core.NAVY * dim


class _Bg:
    """10 フレーム目からの地図：1 回だけ描いた板（地図と桂浜の点）を、押し込みと流れの分だけ（相似変換で）動かす。弧は毎フレーム描く。"""

    PRE = 0.92                       # 板は少し引いて描く（流れと押し込みで画面の端の外が見えないように）

    def __init__(self, tc):
        self.tc = tc
        orig = B.cam
        B.cam = lambda t, extra=1.0: orig(t, extra * self.PRE)
        try:
            f = b03.map_frame(tc, 1.0)
            f = b03.overlay_static(f, tc, 1.0, track_u=0.0, head=False, dot_a=1.0, label_a=0.0, dot_s=self.PRE)
        finally:
            B.cam = orig
        self.base = f.astype(np.float32)
        self.zc = B.cam(tc, self.PRE)[1]
        self.sc = B.target_scr(tc, self.PRE)

    def frame(self, t, extra=1.0):
        ctr, z = B.cam(t, extra)
        s = z / self.zc
        st = B.target_scr(t, extra)
        (cx, cy), (tx, ty) = self.sc, st
        Mx = np.float32([[s, 0, tx - s * cx], [0, s, ty - s * cy]])
        f = cv2.warpAffine(self.base, Mx, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        L = G.Lines()
        B.draw_track(L, ctr, z, 1.0, alpha=1.0, head=True, dash_phase=b03.dash_phase(t), head_r=b03.land_r(t))
        return blend_mask(f, L.done(), core.FOAM)


def render(beat, ctx):
    n = beat['n']
    T = timing(beat)
    f_ji, f_ki, f_o = T['ji'], T['ki'], T['o']
    t0 = B.n03()                                   # 03 からの通しの時刻
    cnt = M.Odometer('1,130', W / 2, Y_MAIN, digit_size=104, side_size=84, prefix='時速 約', suffix='km', hl_suffix='km')
    note = M.KText([dict(text='旅客機（約900km/h）より速い', x=W / 2, y=Y_NOTE, size=42, weight='Medium', track=0.04)])
    sub3, cnt3 = b03.texts()
    bg = None
    for i in range(n):
        out = 1 - M.ease((i - (n - 6) + 1) / 6.0, 'cubic_in')     # 最後の 6 フレームで消える
        push = meter_push(i, n)
        extra = push ** 0.3                                      # 背景はメーターの押し込みの 0.3 倍
        dim = DIM * core.ease((i + 1) / 8.0) + 0.32 * core.ease((i + 1) / 10.0) * out
        if i < LIVE:
            f = b03.map_frame(t0 + i, 1.0)
            f = b03.overlay_static(f, t0 + i, 1.0, track_u=1.0, head=False, dot_a=1.0,
                                   label_a=1 - core.ease((i + 1) / 8.0))   # 「桂浜」はメーターの数字と重なるので消す
        else:
            if bg is None:
                bg = _Bg(t0 + LIVE)
            f = bg.frame(t0 + i, extra)
        # 03 の文字・縮尺は最初の 6 フレームで上へ抜けて消える
        if i < 6:
            q = M.ease((i + 1) / 6.0, 'cubic_in')
            f = B.scale_bar(f, B.cam(t0 + i)[1], alpha=0.9 * (1 - q))
            f = U.scrim(f, b03.Y_SUB - 60, b03.Y_BIG + 10, strength=0.42 * (1 - q), feather=150)
            f = sub3.exit(f, i, 0, dur=6)
            f = cnt3.put(f, float(b03.KM), 1 - q, dy=-0.15 * 108 * q, blur=4 * q)
        # 暗く（全体・まわり・メーターの面）を1回で
        sh = 0.55 * core.ease((i + 1) / 8.0) * out
        appear = core.ease((i + 1) / 8.0) * out
        A = 1 - (1 - dim) * (1 - _VIG * sh) * (1 - _DISC * (0.82 * appear))   # 2026-11-07 検証：弧・海岸線が針と重なるので 0.82
        f += (core.NAVY - f) * A[..., None]
        # メーター
        val = needle(i, T)
        ghosts = []
        d1 = abs(val - needle(i - 1, T)) * 180.0 / VMAX
        if d1 > 4.0:
            ghosts = [(needle(i - k / 3.0, T), a) for k, a in ((3, 0.1), (2, 0.2), (1, 0.35))]
        snd = M.ease((i - f_o + 1) / 6.0, 'sine_inout')
        excess = M.ease((i - f_o + 1) / 8.0, 'expo_out')
        grp = meter_group(val, i, T, ghosts, snd, excess)
        if i >= n - 6:
            q = M.ease((i - (n - 6) + 1) / 6.0, 'expo_in')      # 縮み・ぼけは expo_in（最後に加速して抜ける）
            grp = grp.affine(push * (1.0 - 0.06 * q), MC).blur(3 * q)
        elif push > 1.0005:
            grp = grp.affine(push, MC)
        grp.put(f, out)
        # 時速のオドメーター（針の値、1,130 を超えない）
        if i >= f_ji:
            a = M.ease((i - f_ji + 2) / 4.0, 'sine_inout')
            v = min(max(val, 0.0), float(V))
            dv = v - min(max(needle(i - 1, T), 0.0), float(V))
            if i >= n - 6:
                q = M.ease((i - (n - 6) + 1) / 6.0, 'cubic_in')
                f = cnt.put(f, v, a * (1 - q), i=i, f_lock=f_ki, dy=-0.15 * 104 * q, blur=4 * q)
            else:
                f = cnt.put(f, v, a, i=i, f_lock=f_ki, dv=0.6 * dv)
        if i >= f_o - 1:
            if i >= n - 6:
                f = note.exit(f, i, n - 6, dur=6)
            else:
                f = note.reveal(f, i, f_o - 1, 'line_mask')
        yield np.clip(f, 0, 1).astype(np.float32)
