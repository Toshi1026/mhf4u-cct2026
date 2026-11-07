"""案B 04（約3.5秒）「速さは、時速せんひゃくさんじゅっキロ。旅客機より速い。」

背景：03 の最後の地図（桂浜の点・放物線・ラベルつき。カメラはそのままごくゆっくり流れ続ける）を、
最初の 8 フレームで約 40% 暗くし、メーターが出ているあいだは約 72% まで落とす（03 で大きくした弧がメーターの後ろを通るため）。「桂浜」のラベルはメーターの数字と重なるので同じ 8 フレームで消す。
その上に、案A b08 と同じ線のスピードメーター（0〜1,500 km/h、細い半円）。
 ジ（じそく）：「時速 約◯km」が出て、針が上がりはじめる（数字は針と一緒に数える）。
 キ（キロ）：針が 1,130 で止まり、数字も「約1,130km」（1,130km を強調）。
 リョ（りょかくき）：弧の外の「旅客機」の小さな目盛り（ジェット旅客機の巡航 約900 km/h）が少し明るくなり、
   下に「旅客機（約900km/h）より速い」。音速（1,224 km/h）には届かないので、音速の目盛りは描かない。
2026-11-07 訂正：10km 版（v＝315 m/s＝時速約1,130km、音速の 0.93 倍）。旧「時速1,300km・音速の約1.06倍」。
最後の 6 フレーム：メーターと文字を消し、暗くした地図だけが残る（05 はどこからでも始められる）。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W, H
from beats import _mapui as U
from cutB.beats import _bmap as B, b03

MC = (540.0, 952.0)                 # メーターの中心（半円の底）
R = 340.0
VMAX, V = 1500, 1130
JET = 900.0                         # ジェット旅客機の巡航（約900 km/h）
DIM = 0.40
Y_MAIN, Y_NOTE = 1112, 1196
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
_r = np.hypot((_xx - MC[0]) / 470, (_yy - (MC[1] - 60)) / 380)
_VIG = (np.clip(1.25 - _r, 0, 1) ** 1.5)[..., None].astype(np.float32)
# メーターの面（半円＋数字の帯）の後ろだけ、地図の線をやわらかく沈める
_rd = np.hypot(_xx - MC[0], np.minimum(_yy - MC[1], 0) * 1.0 + np.maximum(_yy - (MC[1] + 250), 0))
_DISC = (np.clip((R + 150 - _rd) / 130.0, 0, 1) ** 1.2)[..., None].astype(np.float32)


def _th(v):
    return np.pi * (1 - min(max(v, 0), VMAX) / VMAX)


def meter(value, alpha=1.0, snd=0.0):
    """メーターの小さな板 (rgba, x0, y0)。snd：旅客機の目盛りの強さ（0..1 で 0.55→1.0 の明るさ）。"""
    pad = 70
    x0, y0 = int(MC[0] - R - pad), int(MC[1] - R - pad)
    w, h = int(2 * (R + pad)), int(R + pad + 40)
    L = core.SS(s=3, w=w, h=h)
    d = L.draw
    cx, cy = MC[0] - x0, MC[1] - y0
    col = L.col(core.FOAM, 0.85 * alpha)
    dim = L.col(core.FOAM, 0.45 * alpha)
    d.arc(L.box(cx - R, cy - R, cx + R, cy + R), 180, 360, fill=col, width=L.w(2.2))
    if value > 0:
        a1 = 180 + 180 * min(value, VMAX) / VMAX
        rr = R - 14
        d.arc(L.box(cx - rr, cy - rr, cx + rr, cy + rr), 180, a1, fill=L.col(core.SKY, 0.55 * alpha), width=L.w(7))
    for v in range(0, VMAX + 1, 50):
        th = _th(v)
        major = v % 300 == 0
        mid = v % 100 == 0
        r0 = R - (30 if major else 20 if mid else 12)
        d.line(L.p([(cx + r0 * np.cos(th), cy - r0 * np.sin(th)), (cx + R * np.cos(th), cy - R * np.sin(th))]),
               fill=col if major else dim, width=L.w(2.4 if major else 1.4 if mid else 1.0))
    # 旅客機の目盛り（弧の外側へ短く）
    th = _th(JET)
    sa = (0.55 + 0.45 * snd) * alpha
    r0, r1 = R + 6, R + 30
    d.line(L.p([(cx + r0 * np.cos(th), cy - r0 * np.sin(th)), (cx + r1 * np.cos(th), cy - r1 * np.sin(th))]),
           fill=L.col(core.FOAM, sa), width=L.w(2.2))
    # 針
    th = _th(value)
    rn = R - 44
    tip = (cx + rn * np.cos(th), cy - rn * np.sin(th))
    bx, by = cx - 26 * np.cos(th), cy + 26 * np.sin(th)
    nx, ny = -np.sin(th), -np.cos(th)
    hw = 5
    d.polygon(L.p([(bx + nx * hw, by + ny * hw), tip, (bx - nx * hw, by - ny * hw)]), fill=L.col(core.FOAM, alpha))
    d.ellipse(L.box(cx - 13, cy - 13, cx + 13, cy + 13), fill=L.col(core.FOAM, alpha))
    d.ellipse(L.box(cx - 5, cy - 5, cx + 5, cy + 5), fill=L.col(core.DEEP, alpha))
    return L.done(), x0, y0


def _labels():
    items = []
    for v in range(0, VMAX + 1, 300):
        th = _th(v)
        rl = R - 66
        items.append(dict(text='{:,}'.format(v), x=MC[0] + rl * np.cos(th), y=MC[1] - rl * np.sin(th) + 11, size=30,
                          weight='Medium', alpha=0.8, track=0))
    return U.Patch(items)


def _sound_label():
    th = _th(JET)
    r = R + 44
    x, y = MC[0] + r * np.cos(th), MC[1] - r * np.sin(th)
    return U.Patch([dict(text='旅客機', x=x + 2, y=y - 4, size=28, weight='Medium', anchor='lm', track=0.06)])


def background(t, dim, label_a=1.0):
    f = b03.map_frame(t, 1.0)
    f = b03.overlay_static(f, t, 1.0, track_u=1.0, head=False, dot_a=1.0, label_a=label_a)
    return f * (1 - dim) + core.NAVY * dim


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[4][0] == 'ジ' and mo[15][0] == 'キ' and mo[17][0] == 'リョ', mo   # ジュッキロ の キ／リョカクキ の リョ
    mf = lambda k: timeline.mora_frame(beat, k)
    f_ji, f_ki, f_o = mf(4), mf(15), mf(17)
    t0 = B.n03()                                   # 03 からの通しの時刻
    cnt = B.HCounter('1,130', W / 2, Y_MAIN, hl_suffix='km', digit_size=104, side_size=84, prefix='時速 約',
                     suffix='km')
    note = U.Patch([dict(text='旅客機（約900km/h）より速い', x=W / 2, y=Y_NOTE, size=42, weight='Medium', track=0.04)])
    labs, slab = _labels(), _sound_label()
    sub3 = U.Patch([dict(text='桂浜から', x=W / 2, y=b03.Y_SUB, size=46, weight='Medium', track=0.08)])
    cnt3 = B.HCounter(b03.KM, W / 2, b03.Y_BIG, hl_suffix='km', digit_size=108, side_size=96, prefix='', suffix='km先へ')
    s0 = f_ji - 2                                     # 針が上がりはじめる
    for i in range(n):
        out = 1 - core.ease((i - (n - 6) + 1) / 6.0, 'inout')
        dim = DIM * core.ease((i + 1) / 8.0) + 0.32 * core.ease((i + 1) / 10.0) * out   # メーターのあいだは地図と弧をさらに落とす（大きくなった弧がメーターと重なるため）
        f = background(t0 + i, dim, label_a=1 - core.ease((i + 1) / 8.0))   # 「桂浜」はメーターの数字と重なるので消す
        old = 1 - core.ease((i + 1) / 6.0)               # 03 の文字・縮尺は最初の 6 フレームで消す（境目で急に消えないように）
        if old > 0.003:
            f = B.scale_bar(f, B.cam(t0 + i)[1], alpha=0.9 * old)
            f = U.scrim(f, b03.Y_SUB - 60, b03.Y_BIG + 10, strength=0.42 * old, feather=150)
            f = sub3.put(f, old)
            f = cnt3.put(f, b03.KM, old)
        sh = 0.55 * core.ease((i + 1) / 8.0) * out
        if sh > 0:
            f = f * (1 - _VIG * sh) + core.NAVY * (_VIG * sh)
        appear = core.ease((i + 1) / 8.0) * out
        f = f * (1 - _DISC * 0.82 * appear) + core.NAVY * (_DISC * 0.82 * appear)   # 2026-11-07 検証：弧・海岸線が針と「旅客機」の目盛りに重なって読みにくかったので 0.6→0.82、半径も広げた
        val = V * core.ease((i - s0) / float(f_ki - s0), 'inout') if i > s0 else 0.0
        snd = core.ease((i - f_o + 1) / 6.0)
        rgba, x0, y0 = meter(val, 1.0, snd)
        f = U.paste(f, rgba, x0, y0, alpha=appear)
        f = labs.put(f, appear)
        f = slab.put(f, appear * (0.6 + 0.4 * snd))
        if i >= f_ji:
            shown = U.fmt(val, 10) if i < f_ki else '1,130'
            f = cnt.put(f, shown, core.ease((i - f_ji + 1) / 5.0) * out)
        if i >= f_o:
            f = note.put(f, snd * out)
        yield np.clip(f, 0, 1).astype(np.float32)
