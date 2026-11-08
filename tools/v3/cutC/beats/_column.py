"""案C 06・07 の背景：高さの柱（悪役を追って上がるカメラ。画面の縦＝高さ、対数の目盛り）。

  y(高さ) = Y_V − K·(ln(1 + 高さ/A0) − ln(1 + カメラの高さ/A0))   … 悪役（カメラの高さ）はいつも y = Y_V
  空の色は高さで決まる（地上の明るい空色 → 10km の青 → 50km の紺 → 100km より上はほぼ黒＝宇宙）。
  積雲の層（1.5–2.6km、_beach の体積の雲を手前・奥の 3 段の奥行きで）、すじ雲（9–11km）、旅客機（10km）、
  カーマン・ライン（100km の細い線と「ここから宇宙」）、高さの目盛り（左、10km・50km・100km）。
  60km より上で星がうすく出る（C08 の最初のカメラと同じ星の並び）。

高さの道のり（km、ビートの中のフレーム）：
  C06：0.10 → 1.30（マッハ 6.7 の縦の速さ ≈ 0.05km/フレーム をもとに、雲の手前まで）
  C07：1.30 → 132（「キ」で 132 に着いて止まる＝頂点）。始まりの速さは C06 の終わりの速さ（対数の速さ）を受け継ぐ。
"""
import math

import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W, H
from cutB.beats import _render as R, _beach as BE, _motion as M
from beats import _geo as G

A0 = 0.30            # km
K = 520.0            # px／ln
Y_V = 710.0          # 悪役（カメラ）の画面の高さ
SKY_KEYS = [(0.0, '9CCDEE'), (1.0, '8CC3EA'), (3.0, '6BAADF'), (8.0, '4688C4'), (15.0, '2F6BA8'), (30.0, '1C4D86'),
            (55.0, '113260'), (80.0, '0A1E3D'), (100.0, '061326'), (125.0, '030A15'), (200.0, '02060C')]
CLOUD_LO, CLOUD_HI = 1.5, 2.6
_S = {}


def lnA(a):
    return np.log1p(np.asarray(a, np.float64) / A0)


def y_of(a, cam_a, depth=1.0):
    """高さ a（km）の画面の y。depth＞1 は奥（動きが小さい）。"""
    return Y_V - K / depth * (lnA(a) - lnA(cam_a))


def alt_of_y(y, cam_a):
    return A0 * np.expm1((Y_V - np.asarray(y, np.float64)) / K + lnA(cam_a))


def sky_rows(cam_a):
    """各行の空の色（線形、(H, 3)）。"""
    ys = np.arange(H, dtype=np.float64) + 0.5
    a = alt_of_y(ys, cam_a)
    ka = np.array([k for k, _ in SKY_KEYS])
    cols = np.array([R.hexl(c) for _, c in SKY_KEYS])
    la = np.log1p(np.maximum(a, 0) / 2.0)
    lk = np.log1p(ka / 2.0)
    out = np.stack([np.interp(la, lk, cols[:, c]) for c in range(3)], 1)
    return out.astype(np.float32), a


def _side():
    if 'side' not in _S:
        xx = (np.arange(W, dtype=np.float32) - W / 2) / (W / 2)
        _S['side'] = (1 - 0.10 * xx * xx).astype(np.float32)[None, :, None]   # 左右の端をわずかに暗く（奥行き）
    return _S['side']


# ---------------------------------------------------------------- 雲
CLOUDS = [  # (高さ km, x, 幅 px, seed, 塔の数, 奥行き)
    (1.62, 180, 520, 31, 4, 1.0), (1.95, 860, 610, 32, 5, 1.0), (2.35, 420, 470, 33, 3, 1.0),
    (1.75, 640, 300, 34, 3, 1.8), (2.20, 120, 280, 35, 3, 1.8), (2.50, 930, 330, 36, 4, 1.8),
    (1.70, 300, 180, 37, 3, 3.0), (2.05, 760, 170, 38, 3, 3.0), (2.45, 560, 160, 39, 2, 3.0),
    (1.55, 1000, 700, 40, 5, 0.7), (2.30, -80, 660, 41, 4, 0.7)]


def _cloud(i):
    key = 'cl%d' % i
    if key not in _S:
        a, x, w, sd, tw, d = CLOUDS[i]
        p, al = BE._cloud(w, sd, tw, detail=0.6)
        if d >= 1.8:                                     # 奥の雲：かすませる（空の色へ）
            pass
        _S[key] = (p, al)
    return _S[key]


def _put_prem(f_lin, prem, a, x0, y0, haze_col=None, haze=0.0):
    h, w = a.shape
    X0, Y0, X1, Y1 = max(int(x0), 0), max(int(y0), 0), min(int(x0) + w, W), min(int(y0) + h, H)
    if X1 <= X0 or Y1 <= Y0:
        return f_lin
    sx, sy = X0 - int(x0), Y0 - int(y0)
    pp = prem[sy:sy + Y1 - Y0, sx:sx + X1 - X0]
    aa = a[sy:sy + Y1 - Y0, sx:sx + X1 - X0][..., None]
    if haze > 0:
        pp = pp * (1 - haze) + haze_col * aa * haze
    sub = f_lin[Y0:Y1, X0:X1]
    sub *= 1 - aa
    sub += pp
    return f_lin


def clouds(f_lin, cam_a, vy=0.0):
    """積雲の層。vy：1 フレームの縦の動き（px、手前の段ほど大きくぶらす）。"""
    order = sorted(range(len(CLOUDS)), key=lambda i: -CLOUDS[i][5])
    for i in order:
        a, x, w, sd, tw, d = CLOUDS[i]
        yb = y_of(a, cam_a, d)                           # 雲の底の高さ
        p, al = _cloud(i)
        h = al.shape[0]
        y0 = yb - h * 0.86
        if y0 > H + 10 or y0 + h < -10:
            continue
        blur = abs(vy) / d * 0.5
        hz = {0.7: 0.0, 1.0: 0.12, 1.8: 0.32, 3.0: 0.5}.get(d, 0.2)
        if blur >= 2 or d <= 0.7:
            k = int(blur) | 1
            pb = cv2.blur(p, (1, k)) if k > 1 else p
            ab = cv2.blur(al, (1, k)) if k > 1 else al
            if d <= 0.7:                                 # いちばん手前：ぼかす（ピントの外）
                pb = cv2.GaussianBlur(pb, (0, 0), 4.0)
                ab = cv2.GaussianBlur(ab, (0, 0), 4.0)
            p, al = pb, ab
        fa = 1.0 if d <= 1.0 else 1.0 - float(R.smoothstep(2.8, 5.0, cam_a))     # 奥の雲は層を出ると消える（遠くへかすむ）
        if fa <= 0.01:
            continue
        f_lin = _put_prem(f_lin, p * fa, al * fa, x - w / 2, y0, R.hexl('8CC3EA'), hz)
    return f_lin


# ---------------------------------------------------------------- すじ雲・旅客機
def _cirrus():
    if 'ci' not in _S:
        w, h = 1400, 220
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        n = R.fbm(xx / 260.0, yy / 34.0, 5, seed=88)
        a = R.smoothstep(0.05, 0.6, n) * np.exp(-((yy - h / 2) / (h * 0.28)) ** 2) * 0.55
        _S['ci'] = a.astype(np.float32)
    return _S['ci']


def cirrus(f_lin, cam_a):
    a = _cirrus()
    h, w = a.shape
    for alt, x0, sc in ((9.5, -260, 1.0), (10.8, 100, 0.8)):
        y = y_of(alt, cam_a, 1.0) - h / 2
        if y > H or y + h < 0:
            continue
        col = R.hexl('D2E4F2')
        prem = np.broadcast_to(col, (h, w, 3)) * a[..., None] * sc
        f_lin = _put_prem(f_lin, prem.astype(np.float32), a * sc, x0, y)
    return f_lin


def airliner(f, cam_a, x=820.0):
    """10km の旅客機（横から、白い小さなシルエット＋短い飛行機雲）。f は sRGB。戻り値：(f, (x, y))。"""
    y = float(y_of(10.0, cam_a, 1.0))
    if y < -60 or y > H + 60:
        return f, (x, y)
    L = core.SS(s=4, w=W, h=H)
    d = L.draw
    s = 1.6
    body = [(-22, -1.6), (16, -2.2), (22, -0.6), (22, 0.8), (16, 2.2), (-22, 2.0)]
    wing = [(-2, 0), (6, 0), (-6, 9), (-10, 9)]
    tail = [(-22, -1.6), (-17, -1.6), (-21, -9), (-24, -9)]
    for poly in (body, wing, tail):
        d.polygon(L.p([(x + px * s, y + py * s) for px, py in poly]), fill=L.col(core.FOAM, 0.92))
    m = L.done()
    con = np.zeros((H, W), np.float32)
    x1 = int(x - 22 * s)
    yy = int(round(y))
    if 0 <= yy < H:
        g = np.clip(np.linspace(0.0, 1.0, max(x1 - 200, 1)), 0, 1) ** 1.5
        con[yy, max(x1 - 200, 0):x1] = g[-len(con[yy, max(x1 - 200, 0):x1]):] * 0.5
        con = cv2.GaussianBlur(con, (0, 0), 1.6) * 2.0
    a = np.maximum(m[..., 3], np.clip(con, 0, 0.5))[..., None]
    return f * (1 - a) + np.asarray(core.FOAM, np.float32) * 0.9 * a, (x, y)


# ---------------------------------------------------------------- 目盛り・カーマン・ライン
RULER_X = 96
_TXT = {}


def _patch(key, items):
    if key not in _TXT:
        from beats import _mapui as U
        _TXT[key] = U.Patch(items)
    return _TXT[key]


def ruler(f, cam_a, alpha=1.0):
    """左の高さの目盛り：細い縦線＋10km・50km・100km（数字のラベル）＋小さな目盛り（1, 2, 5, 20）。"""
    if alpha <= 0.01:
        return f
    Lm = G.Lines()
    Lm.poly(np.array([[RULER_X, -10.0], [RULER_X, H + 10.0]]), 1.2, int(255 * 0.35 * alpha))
    for v in (1, 2, 5, 10, 20, 50, 100):
        y = float(y_of(v, cam_a))
        if -20 < y < H + 20:
            ln_ = 22 if v in (10, 50, 100) else 12
            Lm.poly(np.array([[RULER_X, y], [RULER_X + ln_, y]]), 1.6 if ln_ > 12 else 1.1, int(255 * 0.8 * alpha))
    m = Lm.done()
    f = f + (np.asarray(core.FOAM, np.float32) - f) * m[..., None]
    for v in (10, 50, 100):
        y = float(y_of(v, cam_a))
        if 240 < y < 1300:
            p = _patch('r%d' % v, [dict(text='%dkm' % v, x=RULER_X + 30, y=0 + 10, size=30, weight='Medium', anchor='ls',
                                        track=0.02, alpha=0.85, shadow=0.7)])
            f = p.put(f, alpha * edge_fade(y), dy=y)
    return f


def edge_fade(y):
    """セーフゾーンの上下の外へ行く文字を消す（y 280–1240 の外で 0 へ）。"""
    return float(np.clip((y - 280) / 40.0, 0, 1) * np.clip((1240 - y) / 40.0, 0, 1))


def karman(f, cam_a, alpha=1.0):
    """カーマン・ライン（100km）：画面の幅いっぱいの細い線（1.4px・0.7）と「ここから宇宙」（左寄せ、線の上）。"""
    y = float(y_of(100.0, cam_a))
    if y < -40 or y > H + 40 or alpha <= 0.01:
        return f
    Lm = G.Lines()
    Lm.poly(np.array([[RULER_X, y], [W + 10.0, y]]), 1.4, int(255 * 0.7 * alpha))
    m = Lm.done()
    f = f + (np.asarray(core.FOAM, np.float32) - f) * m[..., None]
    p = _patch('karman', [dict(text='ここから宇宙', x=RULER_X + 150, y=-16, size=40, weight='Bold', anchor='ls',
                               track=0.04, shadow=0.9)])
    return p.put(f, alpha * edge_fade(y), dy=y)


# ---------------------------------------------------------------- 星
def stars(cam_a, star_alpha=1.0):
    """60km より上の行に、C08 の最初のカメラと同じ星（動かない＝無限遠）。(H, W) の明るさ。"""
    if 'stars' not in _S:
        from cutC.beats import _space as SPC, _globe as GL
        k08, _ = SPC.offsets()
        _S['stars'] = GL.star_layer(SPC.cam(k08), 1.0)
    ys = np.arange(H, dtype=np.float64) + 0.5
    a = alt_of_y(ys, cam_a)
    rowk = np.clip((a - 55.0) / 50.0, 0, 1).astype(np.float32)[:, None]
    return _S['stars'] * rowk * star_alpha


# ---------------------------------------------------------------- 1 枚
def frame(cam_a, vy=0.0, star_alpha=1.0):
    """sRGB の背景（悪役・文字なし）。vy：カメラの縦の速さ（px/フレーム、手前の雲のぶれ）。"""
    rows, _ = sky_rows(cam_a)
    f = np.broadcast_to(rows[:, None, :], (H, W, 3)) * _side()
    f = np.ascontiguousarray(f, np.float32)
    st = stars(cam_a, star_alpha)
    if float(st.max()) > 0:
        f = f + st[..., None] * R.hexl('DCE8F4')
    f = cirrus(f, cam_a)
    f = clouds(f, cam_a, vy)
    return R.srgb(f).astype(np.float32)


def speed_lines(f, x, y, k, amount, d=(0.0, 1.0), spread=170.0):
    """悪役のうしろ（下）へ流れる速さの線（_vill.wind_lines と同じ描き方）。"""
    from cutC.beats import _vill as V
    return V.wind_lines(f, x, y, k, d, amount=amount, n=14, spread=spread, speed=95.0, seed=9)
