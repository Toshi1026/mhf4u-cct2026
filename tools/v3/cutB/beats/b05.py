"""B05（12.1–16.3秒）「パンチは、約220トン。ゾウ40頭分。」
濃紺の無地（中央だけごく弱く深い海の色）。上に「パンチの力」、その下に大きなカウンター「約220トン」
（220 を強調。ヤで出て数えはじめ、トンの「ン」で 220 に止まる）。注「体重60kg・パンチが当たる時間0.01秒・空気抵抗なし（試算）」。
「ゾウ、よんじゅっとうぶん」：淡い青の塗りのアフリカゾウのシルエット 40 頭（8×5）が、上の列から順にぽんぽんと並ぶ。
「ゾウ 約40頭分」と注「アフリカゾウ（大人のオス）約5.5〜6t として」。
最初の 6 フレーム：04 の暗い地図からのズームのぼかしを 0 へ。最後の 6 フレーム：上へ振る縦の whip（06 が同じ向きで受ける）。"""
import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W, H
from beats import _mapui as U

COLS, ROWS = 8, 5
DX, Y_ROW0, DY = 110.0, 790.0, 75.0
ESC = 0.82                       # ゾウの大きさ（1単位＝0.92px、全長約 100 単位）
EL_RGB = core.SKY
WHIP_OUT = 6

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


def put_sprite(f, rgba, ax, ay, x, y, scale=1.0, alpha=1.0):
    if alpha <= 0.003:
        return f
    if abs(scale - 1) > 1e-3:
        h, w = rgba.shape[:2]
        M = np.float32([[scale, 0, ax * (1 - scale)], [0, scale, ay * (1 - scale)]])
        rgba = cv2.warpAffine(rgba, M, (w, h), flags=cv2.INTER_LINEAR)
    fx, fy = x - ax, y - ay
    ix, iy = int(np.floor(fx)), int(np.floor(fy))
    rx, ry = fx - ix, fy - iy
    if rx > 1e-3 or ry > 1e-3:
        h, w = rgba.shape[:2]
        rgba = cv2.warpAffine(rgba, np.float32([[1, 0, rx], [0, 1, ry]]), (w + 1, h + 1), flags=cv2.INTER_LINEAR)
    return U.paste(f, rgba, ix, iy, alpha)


def elephants_at(i, f_zo):
    """i フレーム目に見えている各ゾウ (x, y, scale, alpha)。上の列から順に、列の中は左から少しずつずらす。"""
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            f0 = f_zo + r * 5.5 + c * 0.6
            k = (i - f0 + 1) / 6.0
            if k <= 0:
                continue
            k = min(1.0, k)
            a = core.ease(k, 'out')
            sc = 0.72 + 0.28 * a + 0.06 * np.sin(np.pi * k) * (k < 1)
            x = W / 2 + (c - (COLS - 1) / 2) * DX
            y = Y_ROW0 + r * DY + 10 * (1 - a)
            out.append((x, y, sc, a))
    return out


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[4][0] == 'ヤ' and mo[13][0] == 'ン' and mo[14][0] == 'ゾ', mo
    mf = lambda k: timeline.mora_frame(beat, k)
    f_pa, f_ya, f_n, f_zo = mf(0), mf(4), mf(13), mf(14)
    bg = background()
    el, ax, ay = elephant_sprite()
    head = U.Patch([dict(text='パンチの力', x=W / 2, y=410, size=48, weight='Medium', track=0.12)])
    ctr = U.Counter('220', W / 2, 588, digit_size=150, side_size=96, prefix='約', suffix='トン')
    sub = U.Patch([dict(text='ゾウ 約40頭分', x=W / 2, y=686, size=48, weight='Medium', track=0.08)])
    note1 = U.Patch([dict(text='アフリカゾウ（大人のオス）約5.5〜6t として', x=W / 2, y=1162, size=28, weight='Medium',
                          alpha=0.9, track=0.04)])
    note2 = U.Patch([dict(text='体重60kg・パンチが当たる時間0.01秒・空気抵抗なし（試算）', x=W / 2, y=1206, size=28,
                          weight='Medium', alpha=0.8, track=0.02)])
    for i in range(n):
        f = bg.copy()
        # ゾウ
        for x, y, sc, a in elephants_at(i, f_zo):
            f = put_sprite(f, el, ax, ay, x, y, sc, a * 0.95)
        # 文字
        f = head.put(f, core.ease((i - f_pa + 2) / 6.0))
        if i >= f_ya - 1:
            a = core.ease((i - f_ya + 2) / 5.0)
            v = 220 * core.ease((i - f_ya) / float(f_n - f_ya), 'out')
            f = ctr.put(f, str(int(round(v))), a)
            f = note2.put(f, core.ease((i - f_ya - 2) / 8.0))
        if i >= f_zo - 1:
            a = core.ease((i - f_zo + 2) / 6.0)
            f = sub.put(f, a, dy=8 * (1 - a))
            f = note1.put(f, core.ease((i - f_zo - 8) / 8.0))
        # 入り：ズームのぼかしを 0 へ
        if i < 6:
            f = core.zoom_blur(f, 0.12 * (1 - i / 6.0) ** 2, cy=H * 0.45, steps=14)
        # 出：上へ振る whip
        j = i - (n - WHIP_OUT)
        if j >= 0:
            q = (j + 1) / WHIP_OUT
            dy = 260 * q ** 2
            f = cv2.warpAffine(f, np.float32([[1, 0, 0], [0, 1, -dy]]), (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
            f = core.whip(f, 110 * q ** 1.4, axis=0)
        yield np.clip(f, 0, 1).astype(np.float32)
