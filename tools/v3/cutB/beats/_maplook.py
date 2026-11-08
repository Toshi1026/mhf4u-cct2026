"""案B 03・04・T の地図の見た目（2026-11 look-dev v2：美術監督のレビューに合わせて作り直し）。

方針（カメラ・レイアウト・動きは _bmap / b11 のまま。絵だけを差し替える）
- 地形：国土地理院の標高タイル（dem_png）を、地図の px（_tosa.to_screen、4.6 px/km）の格子で 5 段の解像度
  （1・2・4・8・16 テクセル／地図の px）に焼いたテクスチャ。毎フレームはその 2 段を相似変換（03・04）か
  射影変換（T、b11 の斜めのカメラ）で引き伸ばして混ぜるだけ（1 フレーム 0.1 秒ほど）。
- 立体感：プラン・オブリーク（plan oblique relief）。標高に比例して地形を画面の上（北）へずらし、手前（南）の
  地形が奥を隠す。標高 0 の海岸線・桂浜・着水点は動かない（弧・点・ラベルの位置は線画の版と同じ）。
  弧の高さを「画面の上へ起こす」描き方と同じ約束なので、山と弧の高さの見え方がそろう。
- 光：太陽はすべてのショットで同じ（左上・方位 315°・高さ 40°）。キー（ランバート）＋地形の落とす影（4 方向に
  ずらした半影）＋反対側（135°）からの補助光 0.25（#9FB8D0）＋空の光（AO つき、#1B3550 寄りの青）。
- 色（夕方の紺の地図）：陸 #4B6A86、高い所 #8FB0C8、平野は #5E7F90 を 10% 暗く（ごく弱い田畑のむら）。
  海は岸からの距離で #123A60（浅い）→ #0E2E4E（深い）。浅い所の明るさは連続した exp（しきい値・縁なし、ΔL≤0.06）。
  湾・川・湖も同じ水の色（明るい切り抜きにしない）。海岸線は 1.25px・0.40 の線だけ（にじみ・光なし）。
- 地図の位置合わせ：Natural Earth 50m の海岸線（桂浜・着水点・河口の点の基準）は本当の海岸より約 1.2km 南東に
  ずれている。DEM を東西の位置に応じて平行移動（桂浜と河口のずれを線形につなぐ）して、桂浜の点と
  MAZE の前の海の点が本当の海岸線の上・沖 0.3km に来るようにする（_shift）。
- 出典：「国土地理院（標高タイル）を加工して作成」（R.GSI_CREDIT）を 03・04・T の画面に出す（credit()）。

キャッシュ：exports/v3/cache/render/maplook_v*_r{r}.png（RGBA の uint8。A＝水のマスク）、海岸線は .npz。
"""
import math
import os

import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W, H
from beats import _tosa, _geo as G
from beats import _mapui as U
from cutB.beats import _render as R

try:
    import numba as nb
except Exception:       # pragma: no cover
    nb = None

VER = 'v2c'
# (テクセル／地図の px, 範囲 (x0, x1, y0, y1)（地図の px）, DEM のズーム)
LEVELS = [(1, (-200, 1300, -300, 2200), 9), (2, (-20, 1100, -20, 1940), 10), (4, (129, 991, 305, 1813), 11),
          (8, (362, 669, 624, 1087), 12), (16, (439, 600, 702, 934), 13)]
EXO = 1.5               # プラン・オブリークのずらしの強調（標高 1km → 地図の 4.6×1.5 px 上へ）
EXN = 2.2               # 陰影の法線の強調（粗い DEM で山がのっぺりしないように）
SUN_AZ, SUN_EL = 315.0, 40.0
C_LAND = '4B6A86'
C_HIGH = '8FB0C8'
C_PLAIN = '5B7D95'           # #5E7F90 を色相 205° 寄りに（隣の青い丘と並んでオリーブに見えないように）
C_SEA_SH = '123A60'
C_SEA_DP = '0E2E4E'
C_FILL = '9FB8D0'
C_AMB = '1B3550'
SEA_SPEC = 0.04
AMB = 0.22               # 空の光の強さ（平らな所の明るさのうち）
HI_MIX = 0.6
MIN_LOOP = 14.0          # これより小さい（画面 px）島・池の縁の線は描かない（引きの画が点々でうるさくならないように）
PLAIN_K = 0.38            # 平野（線形の倍率）：丘よりわずかに暗く、霧や雪に見えないように
KEY = 1.40

C0, C1 = _tosa.CENTER
KX = 111.32 * math.cos(math.radians(C0)) * _tosa.PX_KM
KY = 111.2 * _tosa.PX_KM
_S = {}


def ll_of(X, Y):
    """地図の px → (lat, lon)（_tosa.to_screen の逆）。"""
    return C0 - (np.asarray(Y, np.float64) - _tosa.CY) / KY, C1 + (np.asarray(X, np.float64) - _tosa.CX) / KX


# ---------------------------------------------------------------- 位置合わせ（DEM → NE 50m の地図）
def _shift_anchors():
    """(x_a, D_a, x_b, D_b)：地図の x が x_a 以下は D_a、x_b 以上は D_b、その間は線形（D＝地図の点 − 本当の点）。
    桂浜：NE に寄せた点 − 本当の桂浜（33.4971N 133.5745E。DEM の海岸線の上）。
    河口側：NE に寄せた河口 − 本当の河口（33.4593N 133.4815E）。着水点（MAZE の前の海）は河口の約 0.45km 西で、同じずれ。"""
    if 'anch' in _S:
        return _S['anch']
    from cutB.beats import _bmap as B
    from beats import b11
    S = B.setup()
    tk = np.array(_tosa.to_screen(*B.KAT_LL))
    tm = np.array(_tosa.to_screen(*b11.MOUTH_LL))
    Dk = S['kat'] - tk
    Dm = S['M'] - tm
    _S['anch'] = (float(S['M'][0]), Dm, float(S['kat'][0]), Dk)
    return _S['anch']


def shift_of(X):
    xa, Da, xb, Db = _shift_anchors()
    t = np.clip((np.asarray(X, np.float64) - xa) / (xb - xa), 0, 1)[..., None]
    return Da * (1 - t) + Db * t


# ---------------------------------------------------------------- numba：プラン・オブリーク
if nb is not None:
    @nb.njit(parallel=True, cache=True, fastmath=True)
    def _oblique_k(col, sh, out, vis_y):
        h, w, c = col.shape
        for x in nb.prange(w):
            M = np.empty(h, np.float64)
            m = 1e18
            for y in range(h - 1, -1, -1):
                g = y - sh[y, x]
                if g < m:
                    m = g
                M[y] = m
            j = 0
            for yo in range(h):
                while j + 1 < h and M[j + 1] <= yo:
                    j += 1
                if M[j] > yo:
                    yy = 0.0
                elif j + 1 < h and M[j + 1] > M[j]:
                    yy = j + (yo - M[j]) / (M[j + 1] - M[j])
                else:
                    yy = float(j)
                if yy > h - 1:
                    yy = h - 1.0
                y0 = int(yy)
                fr = yy - y0
                y1 = min(y0 + 1, h - 1)
                for ch in range(c):
                    out[yo, x, ch] = col[y0, x, ch] * (1 - fr) + col[y1, x, ch] * fr
                vis_y[yo, x] = yy


def oblique(col, shift_tex):
    """col：(h, w, c) float32（平面の色）、shift_tex：(h, w) 画面の上へのずらし（テクセル）。手前（南＝下）が奥を隠す。"""
    out = np.empty_like(col)
    vy = np.empty(col.shape[:2], np.float32)
    _oblique_k(np.ascontiguousarray(col, np.float32), np.ascontiguousarray(shift_tex, np.float32), out, vy)
    return out, vy


# ---------------------------------------------------------------- 1 段のテクスチャを焼く
def _dem_grid(r, rect, z, pad_s):
    x0, x1, y0, y1 = rect
    y1e = y1 + pad_s                                   # 南へ余分に（プラン・オブリークで下から持ち上がる分）
    nx, ny = int(round((x1 - x0) * r)), int(round((y1e - y0) * r))
    X = x0 + (np.arange(nx) + 0.5) / r
    Y = y0 + (np.arange(ny) + 0.5) / r
    d = shift_of(X)                                    # (nx, 2)
    lat_r, lon_r = ll_of(X[None] - d[None, :, 0], Y[:, None] - d[None, :, 1])
    lat_r, lon_r = np.broadcast_arrays(lat_r, lon_r)
    la0, la1 = float(lat_r.min()) - 0.02, float(lat_r.max()) + 0.02
    lo0, lo1 = float(lon_r.min()) - 0.02, float(lon_r.max()) + 0.02
    E, info = R.gsi_dem(la0, la1, lo0, lo1, z=z)
    out = np.empty((ny, nx), np.float32)
    for k in range(0, ny, 1024):
        out[k:k + 1024] = R.gsi_sample_ll(E, info, lat_r[k:k + 1024], lon_r[k:k + 1024])
    return out, nx, ny


def _water_masks(E, r):
    """標高の NaN → 水（海＋湾＋湖＋川）。小さな穴・小島（約 30 画面 px 相当）は消す。"""
    nanm = np.isnan(E).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(nanm, connectivity=4)
    amin = max(6, int(30 * (r / 16.0) ** 2 * 4))
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= amin
    water = keep[lab] & (nanm > 0)
    land = (~water).astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(land, connectivity=4)
    keep2 = np.zeros(n2, bool)
    keep2[1:] = st2[1:, cv2.CC_STAT_AREA] >= amin
    land = keep2[lab2]
    water = ~land
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    water = cv2.morphologyEx(water.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    return water


def _fill_nan(E, water):
    """水の所の標高：海は 0、内陸の水（川・湖）はまわりの陸の標高をなめらかにのばした値（穴にしない）。"""
    Z = np.where(np.isnan(E), 0.0, np.maximum(E, 0.0)).astype(np.float32)
    m = (~water).astype(np.float32)
    num, den = Z * m, m.copy()
    for s in (2.0, 6.0, 18.0):
        nb_ = cv2.GaussianBlur(num, (0, 0), s)
        db_ = cv2.GaussianBlur(den, (0, 0), s)
        fill = nb_ / np.maximum(db_, 1e-4)
        hole = (m < 0.5) & (db_ > 0.02)
        Z = np.where(hole & (den < 0.5), fill, Z)
        num = np.where(hole & (den < 0.5), fill, num)
        den = np.maximum(den, (hole & (den < 0.5)).astype(np.float32))
    return Z


def _sea_mask(water):
    """海（南の端につながる水）。"""
    n, lab = cv2.connectedComponents(water.astype(np.uint8), connectivity=4)
    ids = np.unique(lab[-1][water[-1]])
    ids = ids[ids > 0]
    return np.isin(lab, ids)


def _bake(r, rect, z):
    pad_s = 14.0                                        # 地図の px（標高 2km×EXO で 13.8）
    E, nx, ny = _dem_grid(r, rect, z, pad_s)
    water = _water_masks(E, r)
    sea = _sea_mask(water)
    Z = _fill_nan(E, water)
    Z = np.where(sea, 0.0, Z).astype(np.float32)
    cell = 1000.0 / (_tosa.PX_KM * r)                   # 1 テクセルの m
    Zs = cv2.GaussianBlur(Z, (0, 0), 1.0)
    # ---- 法線（σ=1 のぼかしのあと）
    gy, gx = np.gradient(Zs * EXN, cell)
    nrm = np.stack([-gx, gy, np.ones_like(gx)], -1)    # x 東、y 北（行は南へ増えるので符号を反転）、z 上
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)

    def sun(az, el):
        a, e = math.radians(az), math.radians(el)
        return np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)], np.float32)
    Ls = sun(SUN_AZ, SUN_EL)
    Lf = sun(SUN_AZ - 180, 40.0)
    key = np.clip(nrm @ Ls, 0, 1)
    fill = np.clip(nrm @ Lf, 0, 1)
    # ---- 地形の落とす影（4 方向にずらした半影）
    shd = np.zeros_like(Z)
    hgt = Zs * EXN
    maxd = 3000.0 / cell * 1.5
    for daz, del_ in ((-2.0, -1.5), (2.0, -1.5), (-2.0, 1.5), (2.0, 1.5)):
        shd += R.terrain_shadow(hgt + 0.5 * EXN, cell, az=SUN_AZ + daz, el=SUN_EL + del_, maxd=maxd)
    shd = cv2.GaussianBlur(shd / 4.0, (0, 0), 0.7)
    # ---- AO（約 12 画面 px：寄りきった所で 150m、引きでは広く）
    ao = R.terrain_ao(Zs, cell, radii=(max(150.0, 2.5 * cell), max(600.0, 8 * cell)), exag=EXN)
    ao = 1 - 0.35 * (1 - ao)
    # ---- 陸の色（線形）
    e = Zs
    k_hi = R.smoothstep(600.0, 1700.0, e)[..., None] * HI_MIX
    k_pl = (1 - R.smoothstep(15.0, 90.0, e))[..., None]
    slope = np.sqrt(np.clip(1 - nrm[..., 2] ** 2, 0, 1))
    k_pl = k_pl * (1 - R.smoothstep(0.08, 0.25, slope))[..., None]
    land_c = R.hexl(C_LAND) * (1 - k_hi) + R.hexl(C_HIGH) * k_hi
    plain_c = R.hexl(C_PLAIN) * PLAIN_K
    X = (np.arange(nx, dtype=np.float32)[None] + 0.5) / r + rect[0]
    Yy = (np.arange(ny, dtype=np.float32)[:, None] + 0.5) / r + rect[2]
    fields = R.fbm(X * 3.2, Yy * 3.2, 3, seed=71) * 0.7 + R.fbm(X * 9.0, Yy * 9.0, 2, seed=72) * 0.3
    plain_c = plain_c * (1 + 0.07 * fields)[..., None]
    # 平野の細かい起伏（自然堤防・川の跡・段丘）：DEM の高い周波数（約 150m）を明暗に
    hp = Zs - cv2.GaussianBlur(Zs, (0, 0), max(1.5, 150.0 / cell))
    micro = np.tanh(hp / 4.0)
    plain_c = plain_c * (1 + 0.18 * micro)[..., None]
    alb = land_c * (1 - k_pl) + plain_c * k_pl
    Lr = sun(45.0, 25.0)
    rim = np.clip(nrm @ Lr, 0, 1) ** 3 * (1 - nrm[..., 2]) * 2.0
    flat = math.sin(math.radians(SUN_EL)) * KEY + 0.25 * math.sin(math.radians(40.0)) + AMB
    alb = alb / flat
    amb = R.hexl(C_AMB) / max(float(R.luma(R.hexl(C_AMB))), 1e-3) * AMB      # 空の光：色は #1B3550 の色み
    rad = (KEY * key * shd)[..., None] * R.SUN_RGB + (0.25 * fill)[..., None] * (R.hexl(C_FILL) / float(R.luma(R.hexl(C_FILL)))) \
        + amb * (ao * (0.55 + 0.45 * nrm[..., 2]))[..., None] + (0.35 * rim)[..., None] * R.hexl('DDEFFF')
    land_lin = alb * rad
    # ---- 水（海・湾・湖・川は同じ材質）：岸からの距離で深さの色。しきい値なしの exp だけ
    wu = water.astype(np.uint8)
    dist = cv2.distanceTransform(np.pad(wu, 1, constant_values=1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    dkm = dist / (_tosa.PX_KM * r)
    dkm = cv2.GaussianBlur(dkm.astype(np.float32), (0, 0), max(1.0, 0.4 * _tosa.PX_KM * r))
    dkm = np.where(water, dkm, 0)
    deep = 1 - np.exp(-dkm / 7.0)
    sea_lin = R.hexl(C_SEA_SH) * (1 - deep)[..., None] + R.hexl(C_SEA_DP) * deep[..., None]
    shal = np.exp(-dkm / 0.9)
    sea_lin = sea_lin * (1 + 0.10 * shal)[..., None]
    # 陸と水の境目はなめらかに（1 テクセルの AA）
    wf = cv2.GaussianBlur(water.astype(np.float32), (0, 0), 0.6)[..., None]
    col = land_lin * (1 - wf) + sea_lin * wf
    # ---- プラン・オブリーク
    sh = (Zs / 1000.0 * _tosa.PX_KM * EXO * r).astype(np.float32)
    sh = np.where(water, 0.0, sh).astype(np.float32)
    rgba = np.concatenate([col.astype(np.float32), wf.astype(np.float32)], -1)
    ob, vy = oblique(rgba, sh)
    nyc = int(round((rect[3] - rect[2]) * r))
    ob = ob[:nyc]
    # sRGB の uint8 へ（三角ディザで縞を防ぐ）
    rng = np.random.default_rng(r)
    out = np.empty(ob.shape, np.uint8)
    srgb = R.srgb(ob[..., :3])
    dn = (rng.random(srgb.shape[:2], dtype=np.float32) + rng.random(srgb.shape[:2], dtype=np.float32) - 1.0)[..., None]
    out[..., :3] = np.clip(srgb * 255 + dn + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(ob[..., 3] * 255 + 0.5, 0, 255).astype(np.uint8)
    sea_c = sea[:nyc]
    return out, sea_c, water[:nyc]


def _coast_polys(water, sea, r, rect):
    """水の縁の折れ線（地図の px）。海（湾を含む）の縁と、内陸の水の縁を分けて返す。"""
    out = {}
    for name, m in (('sea', sea), ('inland', water & ~sea)):
        mm = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)
        mu = (mm > 0.5).astype(np.uint8)
        cs, _ = cv2.findContours(mu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        polys = []
        minlen = 12 if name == 'sea' else 30
        for c in cs:
            if len(c) < minlen:
                continue
            p = c[:, 0, :].astype(np.float64)
            kk = 2
            pp = np.concatenate([p[-kk:], p, p[:kk]])
            ker = np.ones(2 * kk + 1) / (2 * kk + 1)
            p = np.stack([np.convolve(pp[:, j], ker, 'valid') for j in range(2)], 1)
            p = cv2.approxPolyDP(p.astype(np.float32)[:, None, :], 0.35, True)[:, 0, :].astype(np.float64)
            p = np.concatenate([p, p[:1]])
            polys.append((p + 0.5) / r + np.array([rect[0], rect[2]]))
        out[name] = polys
    return out


def level(k):
    """k 番目の段：dict(r, rect, tex (h, w, 4) uint8, coast)。"""
    key = 'lv%d' % k
    if key in _S:
        return _S[key]
    r, rect, z = LEVELS[k]
    base = os.path.join(R.CACHE, 'maplook_%s_r%d' % (VER, r))
    if os.path.exists(base + '.png') and os.path.exists(base + '.npz'):
        tex = cv2.imread(base + '.png', cv2.IMREAD_UNCHANGED)
        tex = cv2.cvtColor(tex, cv2.COLOR_BGRA2RGBA)
        zc = np.load(base + '.npz', allow_pickle=False)
        coast = {}
        for name in ('sea', 'inland'):
            pts, idx = zc[name + '_pts'], zc[name + '_idx']
            coast[name] = [pts[idx[i]:idx[i + 1]] for i in range(len(idx) - 1)]
    else:
        tex, sea, water = _bake(r, rect, z)
        coast = _coast_polys(water, sea, r, rect)
        cv2.imwrite(base + '.png', cv2.cvtColor(tex, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 6])
        kw = {}
        for name, polys in coast.items():
            kw[name + '_pts'] = np.concatenate(polys) if polys else np.zeros((0, 2))
            kw[name + '_idx'] = np.cumsum([0] + [len(p) for p in polys])
        np.savez_compressed(base + '.npz', **kw)
    _S[key] = dict(r=r, rect=rect, tex=tex, coast=coast)
    return _S[key]


def _levels_for(z):
    """拡大率 z（画面 px／地図の px）に使う段と重み。粗い段は最大 1.41 倍まで引き伸ばし、そこから細かい段へ溶かす。"""
    L = math.log2(max(z, 1e-3))
    lo = int(math.floor(L))
    fr = L - lo
    ks = [k for k, (r, _, _) in enumerate(LEVELS)]
    rs = [LEVELS[k][0] for k in ks]
    lr = [int(round(math.log2(v))) for v in rs]
    if lo < lr[0]:
        return [(0, 1.0)]
    if lo >= lr[-1]:
        return [(len(ks) - 1, 1.0)]
    i_lo = lr.index(lo)
    w_hi = float(R.smoothstep(0.0, 0.5, fr))
    out = []
    if w_hi < 0.999:
        out.append((i_lo, 1 - w_hi))
    if w_hi > 0.001:
        out.append((i_lo + 1, w_hi))
    return out


# ---------------------------------------------------------------- 毎フレームの地図
def _warp(k, A, size=(W, H), persp=False, border=None):
    """A：画面 → 地図の px の 2x3（persp なら 3x3）。テクスチャの段 k から引き伸ばした RGBA（float32 0..1）。"""
    lv = level(k)
    r, rect = lv['r'], lv['rect']
    T = np.array([[r, 0, -rect[0] * r - 0.5], [0, r, -rect[2] * r - 0.5], [0, 0, 1]], np.float64)
    A3 = A if persp else np.vstack([A, [0, 0, 1]])
    Mx = T @ A3
    if border is None:
        sc = np.clip(np.asarray(R.srgb(R.hexl(C_SEA_DP))) * 255 + 0.5, 0, 255)
        bv = (float(sc[0]), float(sc[1]), float(sc[2]), 255.0)       # テクスチャの外は深い海（段の境で暗くならない）
    else:
        bv = border
    if persp:
        o = cv2.warpPerspective(lv['tex'], Mx, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=bv)
    else:
        o = cv2.warpAffine(lv['tex'], Mx[:2], size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=bv)
    return o


def _sea_fill(shape, ctr=None):
    """テクスチャの外（引きの南の端など）の海の色（深い紺）。"""
    return np.asarray(R.srgb(R.hexl(C_SEA_DP)), np.float32)


def render_affine(ctr, z, t=0.0, anchor=None, size=(W, H), extra_levels=0):
    """真上のカメラ（_bmap.cam）：screen = (P − ctr)·z + ANCHOR。戻り値：(rgb sRGB float32, water (H, W))。"""
    ax, ay = (B_ANCHOR() if anchor is None else anchor)
    A = np.array([[1.0 / z, 0, ctr[0] - ax / z], [0, 1.0 / z, ctr[1] - ay / z]], np.float64)
    return _compose([(k, w) for k, w in _levels_for(z)], A, False, size, t, z)


def B_ANCHOR():
    from cutB.beats import _bmap as B
    return float(B.ANCHOR[0]), float(B.ANCHOR[1])


def _compose(lw, A, persp, size, t, z, wmap=None):
    acc = None
    for k, w in lw:
        o = _warp(k, A, size, persp).astype(np.float32) * (1.0 / 255)
        if wmap is not None:
            w = w * wmap[k] if isinstance(wmap, dict) else w
            w = w[..., None] if np.ndim(w) == 2 else w
        acc = o * w if acc is None else acc + o * w
    rgb, wa = acc[..., :3], acc[..., 3]
    # テクスチャの外（アルファも 0）は深い海の色で埋める（引きの画の南の端だけ）
    cov = acc[..., :3].max(axis=2)
    out_m = (cov < 1e-4)
    if out_m.any():
        rgb[out_m] = _sea_fill(None)
        wa[out_m] = 1.0
    return np.ascontiguousarray(rgb), np.ascontiguousarray(wa)


# ---------------------------------------------------------------- 斜めのカメラ（b11.Cam）
def ground_homography(cam, PX=W / 2, PY=H / 2):
    """b11.Cam の地面（地図の px, z=0）→ 画面 の 3x3。"""
    C, F, Rr, Uu, f = cam.C, cam.F, cam.R, cam.U, cam.f
    rz = np.array([F[0], F[1], -(F @ C)])
    rx = PX * rz + f * np.array([Rr[0], Rr[1], -(Rr @ C)])
    ry = PY * rz - f * np.array([Uu[0], Uu[1], -(Uu @ C)])
    return np.stack([rx, ry, rz])


def render_cam(cam, t=0.0, size=(W, H)):
    """b11 のカメラ（真上〜斜め）で地図を描く。戻り値：(rgb, water, valid)。valid：地面が写っている割合（水平線より上は 0）。"""
    Hg = ground_homography(cam)
    A = np.linalg.inv(Hg)
    A = A / A[2, 2] if abs(A[2, 2]) > 1e-12 else A
    # 画面の粗い格子で、その場所の拡大率（画面 px／地図の px）を出す
    st = 24
    gx, gy = np.meshgrid(np.arange(0, size[0] + st, st, dtype=np.float64), np.arange(0, size[1] + st, st, dtype=np.float64))
    P = np.stack([gx, gy, np.ones_like(gx)], -1) @ A.T
    wv = P[..., 2]
    ok = wv > 1e-9
    X, Y = P[..., 0] / np.where(ok, wv, 1), P[..., 1] / np.where(ok, wv, 1)
    dxX, dxY = np.gradient(X, st, axis=1), np.gradient(Y, st, axis=1)
    dyX, dyY = np.gradient(X, st, axis=0), np.gradient(Y, st, axis=0)
    det = np.abs(dxX * dyY - dxY * dyX)
    zl = 1.0 / np.sqrt(np.maximum(det, 1e-12))
    # 地平線の近く（伸びの強い所）は、短い方の向きの拡大率で（ぼかし側に倒す）
    sv = np.sqrt(np.maximum(dxX ** 2 + dxY ** 2, dyX ** 2 + dyY ** 2))
    zl = np.minimum(zl, 1.0 / np.maximum(sv, 1e-9))
    zl = np.where(ok, zl, 1e-3)
    Lg = np.log2(np.maximum(zl, 1e-3)).astype(np.float32)
    Lg = cv2.resize(Lg, size, interpolation=cv2.INTER_LINEAR)
    valid = cv2.resize(ok.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    lr = np.array([math.log2(LEVELS[k][0]) for k in range(len(LEVELS))])
    # 各段の重み：粗い段から細かい段へ（_levels_for と同じ溶かし方）
    lo = np.floor(Lg)
    fr = Lg - lo
    whi = R.smoothstep(0.0, 0.5, fr)
    wts = {}
    for k in range(len(LEVELS)):
        wk = np.zeros(Lg.shape, np.float32)
        if k == 0:
            wk += (Lg < lr[0]).astype(np.float32)
        if k == len(LEVELS) - 1:
            wk += (lo >= lr[-1]).astype(np.float32)
        wk += ((lo == lr[k]) & (lo < lr[-1])).astype(np.float32) * (1 - whi)
        if k > 0:
            wk += ((lo == lr[k - 1]) & (lo >= lr[0])).astype(np.float32) * whi
        if float(wk.max()) > 1e-3:
            wts[k] = wk
    acc = None
    for k, wk in wts.items():
        o = _warp(k, A, size, persp=True).astype(np.float32) * (1.0 / 255)
        acc = o * wk[..., None] if acc is None else acc + o * wk[..., None]
    rgb, wa = acc[..., :3], acc[..., 3]
    cov = acc[..., :3].max(axis=2)
    out_m = (cov < 1e-4) & (valid > 0.5)
    if out_m.any():
        rgb[out_m] = _sea_fill(None)
        wa[out_m] = 1.0
    return np.ascontiguousarray(rgb), np.ascontiguousarray(wa), valid


# ---------------------------------------------------------------- 毎フレームの仕上げ（地図の層）
_F = {}


def _screen_fields():
    if 'yy' not in _F:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        # 海の広いつや（太陽は左上）：画面の左上ほど水面がわずかに明るい（静的、0.04）
        sp = np.exp(-(((xx + 150) / 900.0) ** 2 + ((yy - 380) / 1100.0) ** 2))
        # 上の空気遠近（北＝奥）：画面の上 40% を #6F8DA8 へ、コントラストを下げる
        hz = (1 - R.smoothstep(0.0, 820.0, yy)) ** 1.4
        # ティルトシフト：上 25%・下 15% だけごく弱くぼかす
        ts = np.maximum(1 - R.smoothstep(0.0, 520.0, yy), R.smoothstep(H - 380.0, H + 60.0, yy))
        _F.update(yy=yy, xx=xx, sp=sp.astype(np.float32), hz=hz.astype(np.float32), ts=ts.astype(np.float32))
    return _F


def sea_and_air(rgb, water, t, ctr=None, z=None, haze=0.30, tilt=1.6):
    """地図の層に、海のつや（静的な広い明るみ＋世界に固定したごくゆっくりのむら）、上の空気遠近、ティルトシフトをかける。"""
    F = _screen_fields()
    L = R.lin(rgb)
    # 海：広いつや（左上）＋ 世界の座標に固定した大きなむら（約 3km、ゆっくり流れる。ΔL≲0.015）
    sm = (W // 4, H // 4)
    if ctr is not None:
        xs = (np.arange(sm[0], dtype=np.float32) * 4 + 2 - B_ANCHOR()[0]) / z + ctr[0]
        ys = (np.arange(sm[1], dtype=np.float32) * 4 + 2 - B_ANCHOR()[1]) / z + ctr[1]
        u = xs[None] / 60.0 + t * 0.0010
        v = ys[:, None] / 60.0 + t * 0.0006
        n = R.fbm(np.broadcast_to(u, (sm[1], sm[0])), np.broadcast_to(v, (sm[1], sm[0])), 3, seed=91)
        n = cv2.resize(n.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        n = 0.0
    k = water * (SEA_SPEC * F['sp'] + 0.004 * n)
    L = L + R.hexl('9FC3E0') * k[..., None]
    # 空気遠近（上）
    if haze > 0:
        a = (haze * F['hz'])[..., None]
        fog = R.hexl('6F8DA8')
        L = L * (1 - a) + fog * a
    out = R.srgb(L)
    if tilt > 0:
        bl = cv2.GaussianBlur(out, (0, 0), tilt)
        m = F['ts'][..., None]
        out = out * (1 - m) + bl * m
    return out.astype(np.float32)


def coast_lines(ctr, z, alpha=0.40, width=1.25, inland_a=0.22, ls=None):
    """海岸線（1.25px・0.40、にじみなし）のマスク。拡大率に合う段の折れ線を使う。ls：変換（地図の px → 画面）。"""
    lw = _levels_for(z)
    k = max(lw, key=lambda a: a[1])[0]
    lv = level(k)
    ax, ay = B_ANCHOR()
    x0, x1 = ctr[0] - ax / z - 4, ctr[0] + (W - ax) / z + 4
    y0, y1 = ctr[1] - ay / z - 4, ctr[1] + (H - ay) / z + 4
    Lc = G.Lines()
    for name, a in (('sea', alpha), ('inland', inland_a)):
        for P in lv['coast'][name]:
            if P[:, 0].max() < x0 or P[:, 0].min() > x1 or P[:, 1].max() < y0 or P[:, 1].min() > y1:
                continue
            if max(np.ptp(P[:, 0]), np.ptp(P[:, 1])) * z < MIN_LOOP:
                continue
            Q = (P - ctr) * z + (ax, ay) if ls is None else ls(P)
            Lc.poly(Q, width, 255 * a, closed=False)
    return Lc.done()


def map_layer(t, ctr, z, haze=0.16, tilt=1.6, coast_a=0.40):
    """03・04 の地図（sRGB float32、文字・弧なし）。"""
    rgb, wa = render_affine(ctr, z, t)
    f = sea_and_air(rgb, wa, t, ctr, z, haze=haze, tilt=tilt)
    if coast_a > 0:
        m = coast_lines(ctr, z, alpha=coast_a)
        f += (np.asarray(core.FOAM, np.float32) - f) * m[..., None]
    return f, wa


def radial_scrim(f, cx, cy, rx, ry=None, strength=0.25, col=None):
    """小さな文字の後ろのやわらかい暗がり（半径 rx・ry の楕円、ガウスの形）。"""
    ry = rx if ry is None else ry
    x0, x1 = int(max(0, cx - 2.2 * rx)), int(min(W, cx + 2.2 * rx))
    y0, y1 = int(max(0, cy - 2.2 * ry)), int(min(H, cy + 2.2 * ry))
    if x1 <= x0 or y1 <= y0:
        return f
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    a = np.exp(-0.5 * (((xx - cx) / (rx * 0.55)) ** 2 + ((yy - cy) / (ry * 0.55)) ** 2)) * strength
    c = np.asarray(core.NAVY if col is None else col, np.float32)
    sub = f[y0:y1, x0:x1]
    sub += (c - sub) * a[..., None]
    return f


def label_scrims(f, a=1.0, cg=True, scale_bar=False, credit_a=0.0):
    """CG ラベル（左上）・縮尺（右上）・出典（右下）の後ろの暗がり（0.25）。"""
    if cg:
        f = radial_scrim(f, 200, 290, 200, 70, 0.25 * a)
    if scale_bar:
        f = radial_scrim(f, 930, 362, 150, 60, 0.25 * a)
    if credit_a > 0:
        f = radial_scrim(f, 830, 1234, 260, 50, 0.25 * credit_a)
    return f


_CR = []


def credit(f, alpha=1.0):
    """出典「国土地理院（標高タイル）を加工して作成」：23px・白 0.6・やわらかい影・右寄せ（x 1012、y 1242）。"""
    if alpha <= 0.002:
        return f
    if not _CR:
        _CR.append(U.Patch([dict(text=R.GSI_CREDIT, x=1012, y=1242, size=23, weight='Medium', anchor='rs', track=0.02,
                                 alpha=0.6, shadow=0.7)]))
    return _CR[0].put(f, alpha)


def drop_shadow(f, m, dx=2.0, dy=3.0, blur=2.5, strength=0.35):
    """線・点のマスク m の下に、右下へ落ちるやわらかい影（#1B3550 を掛ける。黒にしない）。"""
    if float(m.max()) <= 0:
        return f
    sh = cv2.warpAffine(m, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H))
    sh = cv2.GaussianBlur(sh, (0, 0), blur) * strength
    c = np.asarray(core.c(C_AMB), np.float32)
    f *= 1 - sh[..., None] * (1 - c)
    return f


SHADOW_K = 0.35          # 弧の影のずれ（物理の長さの 0.35 倍：画面の上へ起こす描き方では、そのままだと離れすぎて別の線に見える）


def track_ground_shadow(f, ctr, z, u, alpha=0.16, n=120):
    """弧（高さ R/4）が地面（海）に落とす影：太陽（左上・高さ 40°）の反対の右下へ、高さ×cot40° ずらした線。
    やわらかい（半影 σ = 4px + 0.08×ずれ）。"""
    if u <= 0.002:
        return f
    from cutB.beats import _bmap as B
    S = B.setup()
    s = np.linspace(0, u, max(3, int(n * u) + 2))
    g = S['kat'][None] + (S['dest'] - S['kat'])[None] * s[:, None]
    length = float(np.hypot(*(S['dest'] - S['kat'])))
    h = 4 * s * (1 - s) * length * 0.25
    off = (h / math.tan(math.radians(SUN_EL)))[:, None] * np.array([0.7071, 0.7071])[None]
    # 画面の上へ起こす描き方なので、影は地面の点から右下へ（真上から見た太陽の反対向き）
    P = (g + off * SHADOW_K - ctr) * z + B_ANCHOR()
    Lm = G.Lines()
    Lm.poly(P, 3.2, 255)
    m = Lm.done()
    dmax = float(np.max(np.hypot(*(off * z).T))) if len(off) else 0
    sg = 6.0 + 0.08 * dmax
    m = cv2.GaussianBlur(m, (0, 0), sg) * (alpha * math.sqrt(2 * math.pi) * sg / 3.2)
    m = np.clip(m, 0, alpha)
    c = np.asarray(core.c(C_AMB), np.float32)
    f *= 1 - m[..., None] * (1 - c)
    return f
