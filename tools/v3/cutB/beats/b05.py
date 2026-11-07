"""B05（約4.3秒）「パンチは、やく、ひゃくきゅうじゅっトン。ゾウ、さんじゅうごとうぶん。」
濃紺の無地（中央だけごく弱く深い海の色）。上に「パンチの力」、その下に大きなカウンター「約190トン」
（190 を強調。ヤで出て数えはじめ、トンの「ン」で 190 に止まる）。注「体重60kg・パンチが当たる時間0.01秒・空気抵抗なし（試算）」。
「ゾウ、さんじゅうごとうぶん」：淡い青の塗りのアフリカゾウのシルエット 35 頭（7×5）が、上の列から順にぽんぽんと並ぶ。
「ゾウ 約35頭分」と注「アフリカゾウ（大人のオス）約5.5t として」（193 tf ÷ 5.5 t ≈ 35）。
2026-11-07 訂正：10km 版（F＝60kg×315m/s÷0.01s＝1.89×10⁶N＝約193トン重）。旧「約220トン・ゾウ40頭分」。
最初の 6 フレーム：04 の暗い地図からのズームのぼかしを 0 へ。最後の 6 フレーム：上へ振る縦の whip（06 が同じ向きで受ける）。

2026-11 動きの改善（director's motion spec §1-05）：
- カウンターは _motion.Odometer（数字が縦に回る）。8 を超えてから出し（「約0トン」は見せない）、ヤ→ン で expo_out。
  ンで 190 にロック（1.12→1.0 の back_out＋小さな揺れ 4px）。
- ゾウは上から落ちて（−42px・quad_in・5 フレーム。−70 は上の段に重なるため）、足もとを中心につぶれて（1.14×0.84）ばねで戻る（ζ0.45・4Hz）。
  まんなかの列から外へ、上の段から順に。段がそろうたびに全体が 2px 沈む（重さ）。1 頭ごとに砂ぼこり 2 粒。
- 「ゾウ 約35頭分」の 35 は「サ」の少し前から白で回り、「ゴ」で止まる（強調は 190 のまま）。ゾウは「ト」までに全部着く。
- 全体はごくゆっくり押し込む（1.00→1.05）。入りのズームのぼかしの中心は 04 のメーターの中心 (540, 900)。
- 出の whip：ずれ 0→300px（expo_in）、最後のぶれ 110px（06 の最初のフレームと同じ）。"""
import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W, H
from beats import _mapui as U
from cutB.beats import _motion as M

COLS, ROWS = 7, 5
DX, Y_ROW0, DY = 118.0, 790.0, 75.0
ESC = 0.82                       # ゾウの大きさ（1単位＝0.92px、全長約 100 単位）
EL_RGB = core.SKY
WHIP_OUT = 6
WHIP_PX, WHIP_BLUR = 300.0, 110.0      # 06 の入りと同じ
PUSH = 1.05

_C = {}


def background():
    if 'bg' in _C:
        return _C['bg']
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((x - W / 2) / (W * 0.70)) ** 2 + ((y - 820) / (H * 0.42)) ** 2
    bg = core.NAVY[None, None] + (core.DEEP - core.NAVY)[None, None] * (0.42 * np.exp(-r2 * 1.1))[..., None]
    bg = bg * (1 - 0.15 * np.clip(r2 - 0.8, 0, 1))[..., None]
    _C['bg'] = bg.astype(np.float32)
    return _C['bg']


def _spline(P, k=10):
    P = np.asarray(P, float)
    Q = np.concatenate([P[:1], P, P[-1:]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        t = np.linspace(0, 1, k, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 +
                          (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1:])
    return np.concatenate(out)


def elephant_sprite():
    """横向き（左向き）のアフリカゾウのシルエット（塗り＋耳と牙は細い抜きの線）。(rgba, ax, ay)：ax, ay は足もとの中心。"""
    if 'el' in _C:
        return _C['el']
    s, pad = 4, 6
    Wd, Hd = int(108 * ESC) + 2 * pad, int(74 * ESC) + 2 * pad
    L = core.SS(s=s, w=Wd, h=Hd)
    d = L.draw
    ox, oy = pad, Hd - pad                          # 単位 (0,0)＝左端・地面
    P = lambda pts: L.p([(ox + x * ESC, oy + y * ESC) for x, y in pts])
    B = lambda x0, y0, x1, y1: L.box(ox + x0 * ESC, oy + y0 * ESC, ox + x1 * ESC, oy + y1 * ESC)
    col = L.col(EL_RGB, 1.0)
    cut = (0, 0, 0, 0)
    # 胴（肩が高く、背中が少しくぼむ）
    back = _spline([(30, -40), (34, -60), (48, -65), (62, -60), (76, -61), (90, -55), (98, -42), (97, -26),
                    (86, -20), (60, -21), (40, -22), (30, -30)], 8)
    d.polygon(P(back), fill=col)
    # 脚（少し先細り、足先は少し広く）
    for x0, x1 in ((34, 45), (47, 57), (75, 85), (86, 96)):
        d.polygon(P([(x0, -32), (x1, -32), (x1 - 0.8, -3), (x1 + 0.6, 0), (x0 - 0.6, 0), (x0 + 0.8, -3)]), fill=col)
    # 頭
    d.ellipse(B(13, -70, 44, -36), fill=col)
    d.polygon(P([(16, -50), (28, -34), (40, -36), (38, -56)]), fill=col)
    # 鼻（太さが先へ細くなる。先は少し前へ巻く）
    tr = _spline([(19, -46), (12, -36), (9, -24), (9, -12), (11, -4), (15, -3)], 10)
    for k, p in enumerate(tr):
        r = 5.2 - 2.6 * k / (len(tr) - 1)
        x, y = p
        d.ellipse(B(x - r, y - r, x + r, y + r), fill=col)
    # 尾
    d.line(P([(97, -48), (101, -36), (102, -26)]), fill=col, width=L.w(1.6 * ESC), joint='curve')
    d.ellipse(B(100.5, -28, 104, -22), fill=col)
    # 耳（大きなアフリカゾウの耳）：抜きの線で形だけ
    ear = _spline([(36, -66), (50, -68), (58, -56), (56, -40), (50, -30), (43, -27), (38, -36), (35, -50), (36, -66)], 8)
    d.line(P(ear), fill=cut, width=L.w(1.6 * ESC), joint='curve')
    # 牙
    d.line(P(_spline([(22, -38), (18, -32), (12, -30), (7, -31)], 6)), fill=cut, width=L.w(2.0 * ESC), joint='curve')
    # 前脚と後ろ脚の間（奥の脚との境）を細く抜く
    for x in (46, 85.5):
        d.line(P([(x, -28), (x, 0)]), fill=cut, width=L.w(1.2 * ESC))
    rgba = L.done()
    _C['el'] = (rgba, pad + 55 * ESC, Hd - pad)
    return _C['el']


def put_sprite(f, rgba, ax, ay, x, y, scale=1.0, alpha=1.0, sy=None):
    """sy を渡すと横 scale・縦 sy（足もと (ax, ay) が中心）。"""
    if alpha <= 0.003:
        return f
    sx = scale
    sy = scale if sy is None else sy
    h, w = rgba.shape[:2]
    fx, fy = x - ax, y - ay
    ix, iy = int(np.floor(fx)), int(np.floor(fy))
    rx, ry = fx - ix, fy - iy
    pad = 8
    M2 = np.float32([[sx, 0, ax * (1 - sx) + rx + pad], [0, sy, ay * (1 - sy) + ry + pad]])
    rgba = cv2.warpAffine(rgba, M2, (w + 2 * pad, h + 2 * pad), flags=cv2.INTER_LINEAR)
    return U.paste(f, rgba, ix - pad, iy - pad, alpha)


ROW_D, COL_D = 3.8, 0.8   # 段ごと・列ごとの遅れ（最後のゾウが「ト」のころに着く。旧 4.5／0.9 は「ゴ」より 7 フレーム遅かった）


def el_delay(r, c):
    return r * ROW_D + abs(c - (COLS - 1) / 2) * COL_D


DROP = 5            # 落ちるフレーム数（DROP フレーム目に着く）
DROP_PX = 42.0      # 落ちる高さ（70 だと上の段・上の文字に重なって見えるので 42）


def elephants_at(i, f_zo):
    """i フレーム目に見えている各ゾウ (x, y, sx, sy, alpha)。"""
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            k = i - (f_zo + el_delay(r, c))
            if k < 0:
                continue
            x = W / 2 + (c - (COLS - 1) / 2) * DX
            y = Y_ROW0 + r * DY
            if k < DROP:
                q = M.ease((k + 1) / DROP, 'quad_in')
                y += -DROP_PX * (1 - q)
                sx, sy = 0.97, 1.05
                a = M.ease((k + 1) / 3.0, 'sine_inout')
            elif k < DROP + 2:
                sx, sy, a = 1.14, 0.84, 1.0
            else:
                sp = M.spring((k - DROP - 1) / core.FPS, 0.45, 4.0)
                sx, sy, a = 1.14 + (1 - 1.14) * sp, 0.84 + (1 - 0.84) * sp, 1.0
            out.append((x, y, sx, sy, a))
    return out


def landed(i, f_zo):
    """着いたゾウの数（小数：着く直前の 1 フレームで次の 1 頭へ回る）。"""
    v = 0.0
    for r in range(ROWS):
        for c in range(COLS):
            v += float(np.clip(i - (f_zo + el_delay(r, c) + DROP - 1), 0, 1))
    return v


def grid_nudge(i, f_zo):
    """段がそろうたびに 2px 沈んで戻る（重さ）。"""
    d = 0.0
    for r in range(ROWS):
        k = i - (f_zo + el_delay(r, 0) + DROP)
        if k >= 0:
            d += 2.0 * np.exp(-k / 4.0) * np.sin(np.pi * min(k, 12) / 4.0) * (k < 12)
    return d


_D = {}


def dust(f_zo):
    if f_zo not in _D:
        bs = []
        for r in range(ROWS):
            for c in range(COLS):
                x = W / 2 + (c - (COLS - 1) / 2) * DX
                y = Y_ROW0 + r * DY
                f0 = f_zo + el_delay(r, c) + DROP
                for side, rng in ((-1, (-175, -140)), (1, (-40, -5))):
                    bs.append(M.Burst(r * 31 + c * 7 + (side > 0), 1, (x + side * 30, y - 2), angle_range=rng,
                                      speed_range=(1.5, 3.2), gravity=0.12, drag=0.15, life=8, size=(2.2, 3.4),
                                      f0=int(np.ceil(f0)), rgb=core.SAND, alpha=0.35, stretch=0.3))
        _D[f_zo] = bs
    return _D[f_zo]


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[4][0] == 'ヤ' and mo[13][0] == 'ン' and mo[14][0] == 'ゾ' and mo[16][0] == 'サ' and mo[20][0] == 'ゴ', mo
    mf = lambda k: timeline.mora_frame(beat, k)
    f_pa, f_ya, f_n, f_zo = mf(0), mf(4), mf(13), mf(14)
    f_sa, f_go = mf(16), mf(20)
    bg = background()
    el, ax, ay = elephant_sprite()
    head = M.KText([dict(text='パンチの力', x=W / 2, y=410, size=48, weight='Medium', track=0.12)])
    ctr = M.Odometer('190', W / 2, 588, digit_size=150, side_size=96, prefix='約', suffix='トン', lock_scale=1.12,
                     lock_drop=6.0, lock_dur=8)
    sub = M.Odometer('35', W / 2, 686, digit_size=48, side_size=48, prefix='ゾウ 約', suffix='頭分', weight='Medium',
                     side_weight='Medium', gap=0.08, digit_color=core.FOAM, lock_dur=6, lock_scale=1.08, lock_drop=3.0, hide_zero=True,
                     blur_k=0.3)
    note1 = M.KText([dict(text='アフリカゾウ（大人のオス）約5.5t として', x=W / 2, y=1162, size=28, weight='Medium',
                          alpha=0.9, track=0.04)])
    note2 = M.KText([dict(text='体重60kg・パンチが当たる時間0.01秒・空気抵抗なし（試算）', x=W / 2, y=1206, size=28,
                          weight='Medium', alpha=0.8, track=0.02)])
    dusts = dust(f_zo)

    def val(t):
        return 190 * M.ease((t - f_ya) / float(f_n - f_ya), 'expo_out')
    for i in range(n):
        f = bg.copy()
        # ゾウ（段がそろうたびに少し沈む）
        gy = grid_nudge(i, f_zo)
        for x, y, sx, sy, a in elephants_at(i, f_zo):
            f = put_sprite(f, el, ax, ay, x, y + gy, sx, a * 0.95, sy=sy)
        for b in dusts:
            if 0 <= i - b.f0 < 9:
                f = b.draw(f, i)
        # 文字
        f = head.reveal(f, i, f_pa - 2, 'char_rise', stagger=1.3, dur=10)
        v = val(i)
        if v >= 8 or i >= f_n:
            a = float(np.clip((v - 8) / 22.0, 0, 1)) if i < f_n else 1.0
            f = ctr.put(f, v, a, i=i, f_lock=f_n, dv=val(i + 0.5) - val(i - 0.5))
            f = note2.reveal(f, i, f_ya + 1, 'line_mask')
        # 「ゾウ 約35頭分」：行は「サ（さんじゅうご）」の 3 フレーム前から出て、35 は「ゴ」で止まる（声と同じ時に数が読める）。
        #   以前は着いたゾウの数で回していたので、「さんじゅうご」と言い終わっても 25 前後だった。
        s0 = f_sa - 3
        if i >= s0:
            cnt = lambda t: 35 * M.ease(min(1.0, max(0.0, (t - s0 + 1) / float(f_go - s0 + 1))), 'quad_out')
            ac = M.ease((i - s0 + 1) / 4.0, 'sine_inout')
            f = sub.put(f, cnt(i), ac, i=i, f_lock=f_go, dv=cnt(i + 0.5) - cnt(i - 0.5))
        if i >= f_zo + 6:
            f = note1.reveal(f, i, f_zo + 6, 'line_mask')
        # 全体のゆっくりした押し込み＋ロックの小さな揺れ
        tq = i / float(n - 1)
        k = 1 + (PUSH - 1) * (0.5 * tq + 0.5 * M.ease(tq, 'sine_inout'))     # 頭から少し動いている
        sh = M.shake(i, f_n, amp=4.0, tau=2.0, rot_deg=0.0, seed=5)
        cx, cy = W / 2, 800.0
        Mw = np.float32([[k, 0, cx * (1 - k) + sh[0]], [0, k, cy * (1 - k) + sh[1]]])
        f = cv2.warpAffine(f, Mw, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # 入り：ズームのぼかしを 0 へ（04 のメーターの中心から）
        if i < 6:
            f = core.zoom_blur(f, 0.12 * (1 - i / 6.0) ** 2, cx=540, cy=900, steps=14)
        # 出：上へ振る whip（06 が同じ速さで受ける）
        j = i - (n - WHIP_OUT)
        if j >= 0:
            f = M.whip(f, (j + 1) / WHIP_OUT, axis=0, sign=-1, px=WHIP_PX, blur=WHIP_BLUR, direction='out')
        yield np.clip(f, 0, 1).astype(np.float32)
