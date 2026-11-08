"""06・07 の舞台（美術監督の指示 2026-11-08）：横から見た断面を「ジオラマの切り口」として。_xsec.py の形・縮尺・弧はそのまま。

- 空：01 と同じ色の台本（天頂 #2F6FB0 → 地平 #BFE0F5）。左上の太陽のにじみは冷たい白 #EAF6FF を 0.12 足すだけ（色相 205–212°）。
  地平の近くに小さな体積の積雲（_clouds3d）を 50% かすませて。
- 水：水面 #1F6FB5 → 深い所 #0C3560。左上から右下へ斜めの弱い光の筋（ゆっくり）、ただよう粒（ぼかした点）。
- 陸（桂浜）と海底：岩 #5E7385、地層は ±6% L。陸の上は細い砂の帯、海底の上の縁は光の当たった面取り（線にしない）。
  海底のコースティクス：ゆっくり（周期 ≥2 秒）・うすく（ΔL ≤0.04）。
- 水面：細い 1.2px・α0.40 の線（にじませない）。
- 富士山（同じ縮尺のガラスの山）：雪は 2,600m あたりまで、10 本の谷すじの指（先が細い、ノイズでゆらす）。
  ガラスの縁のフレネルのつやは太陽の側（左）だけ、中は #3E77AE へ吸われる。すそは霞に溶ける（地平の空より明るくしない）。
  輪郭の線の描かれ方（stroke）と塗りの入り方（fill）は 06 のまま。

  background(t) → sRGB (H, W, 3)        t：秒
  fuji(fill, f)  → f に富士山のガラスの体を重ねる（輪郭の線は _xsec.fuji_layer2）
"""
import math

import cv2
import numpy as np

import core
from core import W, H, WL
from cutB.beats import _render as R, _xsec as X, _clouds3d as C3

ZEN, HOR = R.hexl('2F6FB0'), R.hexl('BFE0F5')
HAZE_C = R.hexl('BFDDF2')
GLOW = R.hexl('EAF6FF')
W_TOP, W_DEEP = R.hexl('1F6FB5'), R.hexl('0C3560')
ROCK = R.hexl('5E7385')
CLOUDS = [(330.0, WL - 10.0, 120, 61), (512.0, WL - 22.0, 92, 67), (905.0, WL - 14.0, 132, 71),
          (60.0, WL - 30.0, 80, 73)]
_C = {}


def _land_mask(ss=3):
    P = X._land_poly()
    m = np.zeros((H * ss, W * ss), np.uint8)
    cv2.fillPoly(m, [np.round(P * ss * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    return cv2.resize(m.astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_AREA)


def _surface_y():
    P = X._land_poly()[1:-1]
    xs = np.arange(W, dtype=np.float32) + 0.5
    return np.interp(xs, P[:, 0], P[:, 1]).astype(np.float32)


def _sky(h):
    yy, xx = np.mgrid[0:h, 0:W].astype(np.float32)
    e = (WL - yy) / 1500.0                                          # 仰角（ざっくり）
    t = 1 - np.exp(-np.maximum(e, 0) * 3.4)
    col = HOR + (ZEN - HOR) * t[..., None]
    hz = np.exp(-np.maximum(e, 0) / 0.012)[..., None]
    col = col + (HAZE_C * 1.04 - col) * hz * 0.45
    r = np.hypot(xx + 260, yy + 300) / 1100.0                       # 左上の画面の外
    col = col + GLOW * (0.12 * np.exp(-r * 1.6))[..., None]
    return col.astype(np.float32)


def _cloud(w, seed):
    h = int(w * 0.62)
    prem, a = C3.cloud_sprite(w, h, seed=seed, towers=3, flat_k=0.8, sun_k=2.6, amb_k=0.72, detail=0.6)
    st = prem / np.maximum(a, 1e-4)[..., None]
    st = np.minimum(st, 0.80)
    st = st * 0.5 + HAZE_C * 0.5                                   # 50% かすむ
    a = a * 0.75
    return (st * a[..., None]).astype(np.float32), a.astype(np.float32)


def static():
    if _C:
        return _C
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    _C['sky'] = _sky(WL)
    _C['clouds'] = [_cloud(w, sd) for (_, _, w, sd) in CLOUDS]
    # ---- 水（線形）
    d = np.clip(yy[WL:] - WL, 0, None)
    k = 1 - np.exp(-d / 420.0)
    water = W_TOP * (1 - k)[..., None] + W_DEEP * k[..., None]
    water = water + R.hexl('8EC3EA') * (0.16 * np.exp(-d / 12.0))[..., None]     # 水面のすぐ下（下から見た全反射）
    _C['water'] = water.astype(np.float32)
    # ---- 陸・海底の切り口（線形）
    sy = _surface_y()
    ds = yy - sy[None]
    land = _land_mask()
    wob = R.fbm(xx / 260.0, yy / 90.0, 4, seed=31) * 10 + R.fbm(xx / 60.0, yy / 25.0, 3, seed=32) * 3
    q = ds + wob
    # 地層：±6% L の層（細い線にしない＝帯の明暗）
    strata = np.sin(q / 14.0 + R.fbm(xx / 300.0, yy / 300.0, 2, seed=33) * 2.5)
    fine = np.sin(q / 3.2 + R.fbm(xx / 90.0, yy / 40.0, 2, seed=39) * 4.0)
    grain = R.sample(R.fbm_tile(512, 128, 2, 0.6, 41), xx * 1.4, yy * 1.4)
    peb = R.fbm(xx / 6.0, yy / 4.5, 2, seed=40)
    lk = 1 + 0.05 * strata + 0.025 * fine + 0.03 * R.fbm(xx / 30.0, yy / 12.0, 2, seed=34) + 0.035 * grain + \
        0.05 * R.smoothstep(0.35, 0.6, peb) - 0.05 * R.smoothstep(-0.35, -0.6, peb)
    rock = ROCK * lk[..., None]
    wk = R.smoothstep(WL - 2, WL + 30, sy)[None] * np.ones((H, 1), np.float32)   # 水の中の海底ほど 1
    # 陸の上の細い砂の帯（水の上の陸だけ）
    sand = R.hexl('B9C3CB') * (1 + 0.04 * R.fbm(xx / 6.0, yy / 3.0, 2, seed=35))[..., None]
    sk = R.smoothstep(9.0, 5.0, q) * (1 - wk)
    col = rock * (1 - sk[..., None]) + sand * sk[..., None]
    # 海底の上の面：水の色が少しかかる
    kk = (wk * np.exp(-np.clip(ds, 0, None) / 50.0))[..., None]
    col = col * (1 - 0.30 * kk) + W_TOP * 0.35 * (0.30 * kk)
    # 下ほど濃紺へ（切り口の奥行き）
    deep = R.smoothstep(30, 520, ds) ** 0.8
    col = col * (1 - deep[..., None] * 0.86) + R.hexl('0B2238') * (deep[..., None] * 0.86)
    # 上の縁：光の当たった面取り（6px：上から明るく→ふつう）とそのすぐ下のやわらかい影（AO）
    bev = np.exp(-np.clip(ds, 0, None) / 2.4) * (ds > -0.5)
    col = col * (1 + (0.16 - 0.06 * wk) * bev)[..., None]
    col = col * (1 - 0.18 * np.exp(-np.clip(ds - 5, 0, None) / 14.0) * (ds > 5))[..., None]
    _C['land'], _C['land_a'], _C['sy'], _C['wk'] = col.astype(np.float32), land, sy, wk
    # 線：目盛り（_xsec.background と同じ位置・うすさ）と水面（1.2px・0.40）
    L = core.SS(s=3)
    for m in (1000, 2000):
        y = X.y_of(m)
        L.draw.line(L.p([(225, y), (1010, y)]), fill=L.col(core.FOAM, 0.11), width=L.w(1.2))
    L.draw.line(L.p([(X.SHORE_X, WL), (W, WL)]), fill=L.col(core.FOAM, 0.40), width=L.w(1.2))
    _C['lines'] = L.done()
    # ただよう粒
    rng = np.random.default_rng(5)
    n = 260
    _C['motes'] = np.stack([rng.uniform(0, W, n), rng.uniform(WL + 8, H, n), rng.uniform(0.6, 1.5, n),
                            rng.uniform(0.15, 0.6, n), rng.uniform(0, 6.28, n)], 1)
    u = np.arange(4096, dtype=np.float32)
    _C['shaft'] = (R.fbm(u / 110.0, np.zeros_like(u) + 3.0, 4, seed=36) * 0.5 + 0.5).astype(np.float32)
    # コースティクスの帯（海底の上の面の近く）
    near = np.exp(-np.clip(yy - sy[None], 0, None) / 7.0) * (yy > sy[None] - 0.5) * (wk > 0.5)
    rows = np.nonzero(near.max(1) > 0.01)[0]
    _C['cband'] = (rows[0], rows[-1] + 1, near[rows[0]:rows[-1] + 1].astype(np.float32))
    return _C


def _put_prem(L, prem, a, x0, y0):
    ix, iy = int(math.floor(x0)), int(math.floor(y0))
    h, w = a.shape
    pa = cv2.warpAffine(np.dstack([prem, a]), np.float32([[1, 0, x0 - ix], [0, 1, y0 - iy]]), (w + 1, h + 1),
                        flags=cv2.INTER_LINEAR)
    xa, ya, xb, yb = max(ix, 0), max(iy, 0), min(ix + w + 1, L.shape[1]), min(iy + h + 1, L.shape[0])
    if xb <= xa or yb <= ya:
        return L
    sub = pa[ya - iy:yb - iy, xa - ix:xb - ix]
    L[ya:yb, xa:xb] = L[ya:yb, xa:xb] * (1 - sub[..., 3:4]) + sub[..., :3]
    return L


def background(t):
    """断面の背景（sRGB）。t：秒。"""
    C = static()
    out = np.empty((H, W, 3), np.float32)
    sky = C['sky'].copy()
    for (x, yb, w, _), (p, a) in zip(CLOUDS, C['clouds']):
        h = a.shape[0]
        sky = _put_prem(sky, p, a, x + t * 3.0 - w / 2, yb - h * 0.86)
    # 雲は水面の上だけ（地平の 4px 上から下はもや）
    out[:WL] = sky
    wat = C['water'].copy()
    yy = np.arange(WL, H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None]
    d = yy - WL
    uu = (xx - d * 0.55 + t * 7.0) % 4096
    sh = np.interp(uu.ravel(), np.arange(4096), C['shaft']).reshape(uu.shape)
    sh = R.smoothstep(0.55, 0.85, sh) * np.exp(-d / 380.0) * R.smoothstep(0, 50, d)
    wat = wat + R.hexl('A8D6F2') * (sh * 0.10)[..., None]
    out[WL:] = wat
    land = C['land']
    r0, r1, near = C['cband']
    # コースティクス（ゆっくり：t·0.05、うすく）
    xq = xx * np.ones((r1 - r0, 1), np.float32)
    yq = np.arange(r0, r1, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    w1 = R.fbm(xq / 80.0 + t * 0.06, yq / 80.0, 1, seed=37) * 0.6
    cst = 1 - np.abs(R.fbm(xq / 42.0 + w1, yq / 24.0 - t * 0.08, 2, seed=38))
    cst = R.smoothstep(0.70, 0.95, cst) * near
    lb = land[r0:r1] * (1 + 0.09 * cst)[..., None]
    la = C['land_a'][..., None]
    out = out * (1 - la) + land * la
    out[r0:r1] = out[r0:r1] * (1 - la[r0:r1]) + lb * la[r0:r1]
    f = R.srgb(out)
    # ただよう粒（小さなぼかした点）
    m = C['motes']
    x = (m[:, 0] + t * 6.0 * m[:, 3] + 4 * np.sin(t * 0.7 + m[:, 4])) % W
    y = m[:, 1] - t * 3.0 * m[:, 3]
    y = WL + 8 + (y - WL - 8) % (H - WL - 8)
    lay = np.zeros((H, W), np.float32)
    ok = C['land_a'][y.astype(int).clip(0, H - 1), x.astype(int).clip(0, W - 1)] < 0.5
    xi, yi = x[ok], y[ok]
    wv = m[ok, 3] * np.exp(-(yi - WL) / 700.0) * m[ok, 2] ** 2 * 0.8
    x0i, y0i = np.floor(xi).astype(int), np.floor(yi).astype(int)
    fx_, fy_ = xi - x0i, yi - y0i
    for ddx, ddy, ww in ((0, 0, (1 - fx_) * (1 - fy_)), (1, 0, fx_ * (1 - fy_)), (0, 1, (1 - fx_) * fy_),
                         (1, 1, fx_ * fy_)):
        np.add.at(lay, (np.clip(y0i + ddy, 0, H - 1), np.clip(x0i + ddx, 0, W - 1)), wv * ww)
    lay = cv2.GaussianBlur(lay, (0, 0), 1.1) * 6.0
    f = f + (core.FOAM - f) * (lay * 0.25)[..., None]
    f = core.over(f, C['lines'])
    return np.clip(f, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- ガラスの富士山
def _fuji_static():
    if 'fuji' in _C:
        return _C['fuji']
    P = X.fuji_outline()
    ss = 3
    m = np.zeros((H * ss, W * ss), np.uint8)
    cv2.fillPoly(m, [np.round(np.vstack([P, [[P[-1, 0], WL], [P[0, 0], WL]]]) * ss * 16).astype(np.int32)],
                 255, cv2.LINE_AA, shift=4)
    mask = cv2.resize(m.astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    hp = 3776 * X.PXM
    hw = np.clip((WL - yy) / hp, 0, 1)
    half = 30 + 172 * (1 - hw) ** 1.5
    u = np.clip((xx - X.FUJI_X) / half, -1, 1)
    nzc = np.sqrt(np.clip(1 - u * u, 0, 1))
    dhalf = 172 * 1.5 * (1 - hw) ** 0.5 / hp
    nx, ny, nz = u, -dhalf * nzc * 0.9, nzc + 0.25
    nn = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nn, ny / nn, nz / nn
    Lw = np.array([-0.55, -0.65, 0.52], np.float32)
    Lw /= np.linalg.norm(Lw)
    ndl = np.clip(nx * Lw[0] + ny * Lw[1] + nz * Lw[2], 0, 1)
    fres = (1 - nz) ** 3 * R.smoothstep(0.1, -0.6, u)                 # 太陽の側（左）の縁だけ
    # 雪：2,600m あたりまで。10 本の谷すじの指（先が細い）
    alt = (WL - yy) / X.PXM
    ang = np.arcsin(u)
    # 雪：2,650m あたりまで。谷すじの指 11 本（尾根に残る雪が下へ細く伸びる。先は細く、ノイズでゆらす）
    jit = R.fbm(ang * 2.0, np.zeros_like(ang) + 1.7, 3, seed=44) * 0.8
    ridge = 0.5 + 0.5 * np.cos(ang * 22.0 + 0.4 + jit)
    ln_ = 0.35 + 0.65 * (R.fbm(ang * 5.0, np.zeros_like(ang) + 5.0, 2, seed=45) * 0.5 + 0.5) ** 1.5
    fing = ridge ** 6 * ln_
    line = 3050 - 260 * (R.fbm(ang * 3.0, np.zeros_like(ang) + 9.0, 2, seed=47) * 0.5 + 0.5) - 1150 * fing
    line = line + R.fbm(xx / 7.0, yy / 7.0, 2, seed=46) * 30
    snow = R.smoothstep(line - 12, line + 12, alt) * mask
    # 雪の面の細かい谷（縦の細い陰）
    gul = 0.55 + 0.45 * np.clip(0.5 - 0.5 * u, 0, 1) ** 0.8          # 左（光の側）が明るい
    thick = nzc * half / 202.0                                         # ガラスの厚み（まんなかほど厚い）
    base = R.smoothstep(WL - 4, WL - 150, yy)                          # すそは霞に溶ける
    ys, xs = np.nonzero(mask > 0.002)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cut = lambda a: np.ascontiguousarray(a[y0:y1, x0:x1]).astype(np.float32)
    _C['fuji'] = dict(mask=cut(mask), ndl=cut(ndl), fres=cut(fres), snow=cut(snow), thick=cut(thick), gul=cut(gul),
                      hh=cut(np.clip(alt / 3776.0, 0, 1)),
                      base=cut(base), box=(y0, y1, x0, x1))
    return _C['fuji']


def fuji(fill, f):
    """富士山のガラスの体（輪郭の線は X.fuji_layer2）。fill：0..1。f：その下の画（sRGB）。新しい画を返す。"""
    if fill <= 0.003:
        return f
    F = _fuji_static()
    y0, y1, x0, x1 = F['box']
    out = f.copy()
    L = R.lin(f[y0:y1, x0:x1])
    m = F['mask'] * fill * F['base']
    # ガラス：厚いほど #3E77AE へ吸われる（明るさは下がる）
    # 山の体：うすいガラス。左（光の側）は明るく、右は暗く、厚いほど #3E77AE へ吸われる
    ab = (0.30 + 0.25 * np.clip(F['thick'], 0, 1)) * m * (0.25 + 0.75 * F['hh'] ** 0.8)
    L = L * (1 - ab[..., None]) + R.hexl('3E77AE') * (ab * (0.70 + 0.85 * F['ndl'] ** 1.3))[..., None]
    L = L + R.hexl('EAF5FD') * (F['fres'] * m * 0.30)[..., None]
    # 雪（くっきり、光の側が明るい。白の上限 ≤0.92）
    sn = F['snow'] * fill
    scol = R.hexl('DCE8F2') * (0.92 * F['gul'] * (0.85 + 0.15 * F['ndl']))[..., None]
    L = L * (1 - sn[..., None] * 0.92) + scol * (sn[..., None] * 0.92)
    out[y0:y1, x0:x1] = R.srgb(L)
    return out


def knockout(rgba, x0, y0, boxes, pad=6):
    """線の板（rgba、世界の座標の (x0, y0)）から、文字の箱（世界の座標）の所を pad px ぬく。"""
    if not boxes:
        return rgba
    a = rgba[..., 3].copy()
    h, w = a.shape
    m = np.ones((h, w), np.float32)
    for (bx0, by0, bx1, by1) in boxes:
        xa, ya = int(math.floor(bx0 - pad - x0)), int(math.floor(by0 - pad - y0))
        xb, yb = int(math.ceil(bx1 + pad - x0)), int(math.ceil(by1 + pad - y0))
        xa, ya, xb, yb = max(xa, 0), max(ya, 0), min(xb, w), min(yb, h)
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = 0
    m = cv2.GaussianBlur(m, (0, 0), 1.2)
    out = rgba.copy()
    out[..., 3] = a * m
    return out
