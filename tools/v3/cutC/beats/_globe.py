"""案C 08〜10 の地球（宇宙から見た地球 → 日本 → 四国 → 土佐湾）。CPU・numba。

C10 は宇宙から仁淀川の河口の前の海まで「一本のズーム」で降りて、最後は案B の T（bT.py：着水・地図 → b11 → 実写）へ渡す。
そのために地図の色・光・プラン・オブリークは _maplook（03・04・T）と同じ焼き方にそろえ、足りない広さだけ段を足す：

  段 c0a：日本全体（北緯 24–46.5°・東経 122–150°）、0.125 テクセル／地図の px（1 テクセル≈1.7km）、標高タイル z7
  段 c0b：四国のまわり（地図の px で x −1180..2220・y −2078..3722）、0.5 テクセル／地図の px、標高タイル z8
  その先は _maplook の段（1・2・4・8・16）をそのまま。地図の px は _tosa.to_screen（4.6 px/km）。

  陸と海：日本（国土地理院の標高がある所から 25km 以内）は _maplook と同じく標高の欠測＝水。
          日本の外（朝鮮半島・大陸など）は国土地理院の標高がないので、Natural Earth 50m の陸を平らな陸（標高 0）として塗る
          （標高がない＝海、にすると朝鮮半島が消えるため）。起伏は作らない（ない地形を描かない）。
  地球全体：Natural Earth 50m の陸（正距円筒 4096 幅）を同じ色の平らな陸・深い海で（日本の外の遠い所だけ）。

描き方
  球（宇宙〜日本の引き）：画素ごとのレイと球の交点 → 緯度経度 → 地図の px → 段のテクスチャ。縁はレイの最接近の高さで
        1px のやわらかい輪郭。大気：地表の上のかすみ（空気の道のりの長さ）と、縁の外の薄い青い層（最接近の高さの exp）。
        星（世界に固定した方向）・太陽（やわらかい円盤とにじみ、点滅しない）。
  平面（四国より寄り）：真上のカメラの地図（段を相似変換で引き伸ばして重ねる）＝ T の地図と同じ手順。
  球 → 平面：同じ画面の点の「地図の px」を、球の写し方から平面の写し方へなめらかに寄せる（地図がゆがみながら合う。
        寄るほど差は小さく、切り替えの所で 1% 以下）。
"""
import math
import os

import cv2
import numpy as np
import numba as nb

from cutB.beats import _useA  # noqa: F401
import core
from core import W, H
from beats import _tosa, _geo as G
from cutB.beats import _render as R, _maplook as ML

RE = 6371.0
VER = 'c2'
# (名前, テクセル／地図の px, (x0, x1, y0, y1) 地図の px, 標高タイルのズーム)
LV_C = [('c0a', 0.125, (-4405, 7582, -5872, 5637), 7), ('c0b', 0.5, (-1180, 2220, -2078, 3722), 8)]
JP_ZONE_KM = 25.0
FAR_LAND = R.hexl(ML.C_LAND) * 0.62          # 日本の外の陸（平ら・起伏なし、線形）。段と地球全体で同じ色
_S = {}


def mp_of(lat, lon):
    x, y = _tosa.to_screen(lat, lon)
    return np.asarray(x, np.float64), np.asarray(y, np.float64)


# ---------------------------------------------------------------- 段を焼く（_maplook._bake と同じ焼き方。陸の判定だけ日本の外を NE で補う）
def _bake_c(r, rect, z):
    M = ML
    pad_s = 14.0
    E, nx, ny = M._dem_grid(r, rect, z, pad_s)
    water = M._water_masks(E, r)
    # 日本の外（国土地理院の標高がない所）：Natural Earth の陸を平らな陸に
    X = rect[0] + (np.arange(nx) + 0.5) / r
    Y = rect[2] + (np.arange(ny) + 0.5) / r
    lat, lon = M.ll_of(X[None, :], Y[:, None])
    lat, lon = np.broadcast_arrays(lat, lon)
    ne = G.land_at(lat.astype(np.float32), lon.astype(np.float32), 16384) > 0.5
    valid = (~np.isnan(E)).astype(np.uint8)
    dist = cv2.distanceTransform(1 - valid, cv2.DIST_L2, 5)               # テクセル
    zone = dist < JP_ZONE_KM * _tosa.PX_KM * r
    water = water & ~(ne & ~zone)
    sea = M._sea_mask(water)
    Z = M._fill_nan(E, water)
    Z = np.where(sea | ~zone, 0.0, Z).astype(np.float32)
    cell = 1000.0 / (_tosa.PX_KM * r)
    Zs = cv2.GaussianBlur(Z, (0, 0), 1.0)
    gy, gx = np.gradient(Zs * M.EXN, cell)
    nrm = np.stack([-gx, gy, np.ones_like(gx)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    Zl = cv2.GaussianBlur(np.minimum(Z, M.LOW_TOP), (0, 0), 0.8)
    gyl, gxl = np.gradient(Zl * M.EX_LOW, cell)
    nrm_l = np.stack([-gxl, gyl, np.ones_like(gxl)], -1)
    nrm_l /= np.linalg.norm(nrm_l, axis=-1, keepdims=True)
    k_m = R.smoothstep(20.0, 120.0, Zs)[..., None]
    nrm = nrm_l * (1 - k_m) + nrm * k_m
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)

    def sun(az, el):
        a, e = math.radians(az), math.radians(el)
        return np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)], np.float32)
    Ls = sun(M.SUN_AZ, M.SUN_EL)
    Lf = sun(M.SUN_AZ - 180, 40.0)
    key = np.clip(nrm @ Ls, 0, 1)
    fill = np.clip(nrm @ Lf, 0, 1)
    shd = np.zeros_like(Z)
    hgt = Zs * M.EXN
    maxd = 3000.0 / cell * 1.5
    for daz, del_ in ((-2.0, -1.5), (2.0, -1.5), (-2.0, 1.5), (2.0, 1.5)):
        shd += R.terrain_shadow(hgt + 0.5 * M.EXN, cell, az=M.SUN_AZ + daz, el=M.SUN_EL + del_, maxd=maxd)
    shd = cv2.GaussianBlur(shd / 4.0, (0, 0), 0.7)
    ao = R.terrain_ao(Zs, cell, radii=(max(150.0, 2.5 * cell), max(600.0, 8 * cell)), exag=M.EXN)
    ao = 1 - 0.35 * (1 - ao)
    e = Zs
    k_hi = R.smoothstep(600.0, 1700.0, e)[..., None] * M.HI_MIX
    k_pl = (1 - R.smoothstep(20.0, 120.0, e))[..., None]
    land_c = R.hexl(M.C_LAND) * (1 - k_hi) + R.hexl(M.C_HIGH) * k_hi
    plain_c = R.hexl(M.C_PLAIN2) * M.PLAIN_K2
    Xf = X.astype(np.float32)[None]
    Yf = Y.astype(np.float32)[:, None]
    fields = R.fbm(Xf * 11.0, Yf * 11.0, 2, seed=71) * 0.6 + R.fbm(Xf * 40.0, Yf * 40.0, 2, seed=72) * 0.4
    plain_c = plain_c * (1 + 0.025 * np.clip(fields, -1.5, 1.5))[..., None]
    alb = land_c * (1 - k_pl) + plain_c * k_pl
    Lr = sun(45.0, 25.0)
    rim = np.clip(nrm @ Lr, 0, 1) ** 3 * (1 - nrm[..., 2]) * 2.0
    flat = math.sin(math.radians(M.SUN_EL)) * M.KEY + 0.25 * math.sin(math.radians(40.0)) + M.AMB
    alb = alb / flat
    amb = R.hexl(M.C_AMB) / max(float(R.luma(R.hexl(M.C_AMB))), 1e-3) * M.AMB
    rad = (M.KEY * key * shd)[..., None] * R.SUN_RGB + (0.25 * fill)[..., None] * (R.hexl(M.C_FILL) / float(R.luma(R.hexl(M.C_FILL)))) \
        + amb * (ao * (0.55 + 0.45 * nrm[..., 2]))[..., None] + (0.35 * rim)[..., None] * R.hexl('DDEFFF')
    land_lin = alb * rad
    far = (~zone & ~water)[..., None]
    land_lin = np.where(far, FAR_LAND[None, None] * (1 + 0.03 * np.clip(fields, -1.5, 1.5))[..., None], land_lin)
    wu = water.astype(np.uint8)
    dist = cv2.distanceTransform(np.pad(wu, 1, constant_values=1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    dkm = dist / (_tosa.PX_KM * r)
    dkm = cv2.GaussianBlur(dkm.astype(np.float32), (0, 0), max(1.0, 0.4 * _tosa.PX_KM * r))
    dkm = np.where(water, dkm, 0)
    deep = 1 - np.exp(-dkm / 14.0)
    sea_lin = R.hexl(M.C_SEA_SH) * (1 - deep)[..., None] + R.hexl(M.C_SEA_DP) * deep[..., None]
    shal = np.exp(-dkm / 0.9)
    sea_lin = sea_lin * (1 + 0.10 * shal)[..., None]
    wf = cv2.GaussianBlur(water.astype(np.float32), (0, 0), 0.6)[..., None]
    col = land_lin * (1 - wf) + sea_lin * wf
    sh = (Zs / 1000.0 * _tosa.PX_KM * M.EXO * r).astype(np.float32)
    sh = np.where(water, 0.0, sh).astype(np.float32)
    rgba = np.concatenate([col.astype(np.float32), wf.astype(np.float32)], -1)
    ob, vy = M.oblique(rgba, sh)
    nyc = int(round((rect[3] - rect[2]) * r))
    ob = ob[:nyc]
    rng = np.random.default_rng(int(r * 1000))
    out = np.empty(ob.shape, np.uint8)
    srgb = R.srgb(ob[..., :3])
    dn = (rng.random(srgb.shape[:2], dtype=np.float32) + rng.random(srgb.shape[:2], dtype=np.float32) - 1.0)[..., None]
    out[..., :3] = np.clip(srgb * 255 + dn + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(ob[..., 3] * 255 + 0.5, 0, 255).astype(np.uint8)
    return out, sea[:nyc], water[:nyc]


def level_c(k):
    """C の段 k（0：c0a、1：c0b）。dict(r, rect, tex, coast)。"""
    key = 'lvc%d' % k
    if key in _S:
        return _S[key]
    name, r, rect, z = LV_C[k]
    base = os.path.join(R.CACHE, 'globe_%s_%s' % (VER, name))
    if os.path.exists(base + '.png') and os.path.exists(base + '.npz'):
        tex = cv2.cvtColor(cv2.imread(base + '.png', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        zc = np.load(base + '.npz', allow_pickle=False)
        coast = {}
        for nm in ('sea', 'inland'):
            pts, idx = zc[nm + '_pts'], zc[nm + '_idx']
            coast[nm] = [pts[idx[i]:idx[i + 1]] for i in range(len(idx) - 1)]
    else:
        tex, sea, water = _bake_c(r, rect, z)
        coast = ML._coast_polys(water, sea, r, rect)
        cv2.imwrite(base + '.png', cv2.cvtColor(tex, cv2.COLOR_RGBA2BGRA), [cv2.IMWRITE_PNG_COMPRESSION, 6])
        kw = {}
        for nm, polys in coast.items():
            kw[nm + '_pts'] = np.concatenate(polys) if polys else np.zeros((0, 2))
            kw[nm + '_idx'] = np.cumsum([0] + [len(p) for p in polys])
        np.savez_compressed(base + '.npz', **kw)
    _S[key] = dict(r=r, rect=rect, tex=tex, coast=coast)
    return _S[key]


def all_levels():
    """粗い順の段のリスト：[c0a, c0b, ML0, ML1, ML2, ML3, ML4]（各 dict(r, rect, tex, coast)）。"""
    if 'all' not in _S:
        _S['all'] = [level_c(0), level_c(1)] + [ML.level(k) for k in range(len(ML.LEVELS))]
    return _S['all']


def global_tex():
    """地球全体（正距円筒、経度 -180..180 → x、緯度 90..-90 → y）の RGBA uint8：平らな陸＋深い海（日本の外の遠い所用）。"""
    if 'glob' not in _S:
        m = G.land_mask(4096)
        land = R.srgb(FAR_LAND)
        sea = R.srgb(R.hexl(ML.C_SEA_DP))
        col = sea[None, None] * (1 - m[..., None]) + land[None, None] * m[..., None]
        t = np.empty(m.shape + (4,), np.uint8)
        t[..., :3] = np.clip(col * 255 + 0.5, 0, 255)
        t[..., 3] = np.clip((1 - m) * 255 + 0.5, 0, 255)
        _S['glob'] = t
    return _S['glob']


# ---------------------------------------------------------------- 球のレイ
@nb.njit(parallel=True, cache=True, fastmath=True)
def _rays_k(C, F, Rv, U, f, px, py, re, ha, scale, sun, lat, lon, cov, mu, hmin, path, ndl, spec):
    h, w = lat.shape
    cc = C[0] * C[0] + C[1] * C[1] + C[2] * C[2]
    dcam = math.sqrt(cc)
    ra = re + ha
    for yy in nb.prange(h):
        for xx in range(w):
            sx = ((xx + 0.5) * scale - px) / f
            sy = ((yy + 0.5) * scale - py) / f
            dx = F[0] + sx * Rv[0] - sy * U[0]
            dy = F[1] + sx * Rv[1] - sy * U[1]
            dz = F[2] + sx * Rv[2] - sy * U[2]
            dn = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= dn
            dy /= dn
            dz /= dn
            b = C[0] * dx + C[1] * dy + C[2] * dz
            tc = -b
            if tc < 0:
                tc = 0.0
            qx, qy, qz = C[0] + tc * dx, C[1] + tc * dy, C[2] + tc * dz
            dmin = math.sqrt(qx * qx + qy * qy + qz * qz)
            hmin[yy, xx] = dmin - re
            # 画素の大きさ（接点までの距離 / f）で 1px のやわらかい縁
            dt = math.sqrt(max(cc - re * re, 1.0))
            pix = dt / f * scale
            a = (re - dmin) / pix + 0.5
            cov[yy, xx] = min(max(a, 0.0), 1.0)
            disc = b * b - (cc - re * re)
            if disc > 0 and -b - math.sqrt(disc) > 0:
                t = -b - math.sqrt(disc)
                X, Y, Z = C[0] + t * dx, C[1] + t * dy, C[2] + t * dz
                # 大気の中の道のり（殻の入口から地面まで）
                da = b * b - (cc - ra * ra)
                t_in = -b - math.sqrt(da) if da > 0 else 0.0
                if t_in < 0:
                    t_in = 0.0
                path[yy, xx] = t - t_in
            else:
                # 当たらない：最接近の点を球の上へ（縁の AA 用）
                k = re / max(dmin, 1e-6)
                X, Y, Z = qx * k, qy * k, qz * k
                da = b * b - (cc - ra * ra)
                path[yy, xx] = 2.0 * math.sqrt(da) if da > 0 else 0.0
            nx, ny, nz = X / re, Y / re, Z / re
            dd = dx * nx + dy * ny + dz * nz
            mu[yy, xx] = -dd
            ndl[yy, xx] = nx * sun[0] + ny * sun[1] + nz * sun[2]
            rx, ry, rz = dx - 2 * dd * nx, dy - 2 * dd * ny, dz - 2 * dd * nz
            spec[yy, xx] = max(rx * sun[0] + ry * sun[1] + rz * sun[2], 0.0)
            lat[yy, xx] = math.degrees(math.asin(min(max(nz, -1.0), 1.0)))
            lon[yy, xx] = math.degrees(math.atan2(ny, nx))


class SCam:
    """宇宙のカメラ（地球の中心が原点の km、x＝(0N,0E)、z＝北極）。F・R・U は単位ベクトル（R＝画面の右、U＝画面の上）。"""

    def __init__(self, C, F, U_hint, f, px=W / 2, py=H / 2):
        self.C = np.asarray(C, np.float64)
        F = np.asarray(F, np.float64)
        self.F = F / np.linalg.norm(F)
        Rv = np.cross(self.F, np.asarray(U_hint, np.float64))
        self.R = Rv / np.linalg.norm(Rv)
        self.U = np.cross(self.R, self.F)
        self.f, self.px, self.py = float(f), float(px), float(py)

    def proj(self, P):
        """世界の点 (N,3) → 画面 (N,2)（後ろは NaN）。"""
        v = np.asarray(P, np.float64) - self.C
        z = v @ self.F
        x = self.px + self.f * (v @ self.R) / np.where(z > 1e-9, z, np.nan)
        y = self.py - self.f * (v @ self.U) / np.where(z > 1e-9, z, np.nan)
        return np.stack([x, y], -1)

    def dir_proj(self, D):
        """方向 (N,3)（星・太陽）→ 画面 (N,2)。"""
        D = np.asarray(D, np.float64)
        z = D @ self.F
        x = self.px + self.f * (D @ self.R) / np.where(z > 1e-9, z, np.nan)
        y = self.py - self.f * (D @ self.U) / np.where(z > 1e-9, z, np.nan)
        return np.stack([x, y], -1)


def enu(lat, lon):
    """(東, 北, 上) の単位ベクトル（世界の座標）。"""
    la, lo = math.radians(lat), math.radians(lon)
    up = np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    east = np.array([-math.sin(lo), math.cos(lo), 0.0])
    north = np.cross(up, east)
    return east, north, up


def ecef(lat, lon, alt=0.0):
    return G.xyz(lat, lon) * (RE + alt)


def rays(cam, sun=None, scale=1):
    """画面（W/scale × H/scale）の各画素の (lat, lon, cov, mu, hmin, path, ndl, spec)。"""
    w, h = W // scale, H // scale
    out = [np.empty((h, w), np.float32) for _ in range(8)]
    s = np.asarray(SUN_DEFAULT if sun is None else sun, np.float64)
    _rays_k(cam.C, cam.F, cam.R, cam.U, cam.f, cam.px, cam.py, RE, HA, float(scale), s / np.linalg.norm(s), *out)
    return out


def _sun_default():
    e, n, u = enu(33.4, 133.5)
    a, el = math.radians(ML.SUN_AZ), math.radians(52.0)
    return math.cos(el) * (math.sin(a) * e + math.cos(a) * n) + math.sin(el) * u


SUN_DEFAULT = _sun_default()      # 土佐湾で方位 315°（左上）・高さ 52°（地図の焼いた光と同じ向き）


HA = 70.0           # 大気の殻（見た目の厚さ、km）


# ---------------------------------------------------------------- テクスチャを「地図の px」で引く
def sample_levels(MX, MY, zl, levels=None, k_max=None):
    """地図の px (MX, MY)（float32 の画面の形）で段を引く。zl：その場所の拡大率（画面 px／地図の px、スカラー）。
    粗い段から順に重ね、細かい段は覆う所（テクスチャの内側）だけ、拡大率に合わせて溶かす。戻り値：RGBA float32 0..1。"""
    lv = all_levels() if levels is None else levels
    L2 = math.log2(max(zl, 1e-4))
    out = None
    for k, d in enumerate(lv):
        if k_max is not None and k > k_max:
            break
        r, rect = d['r'], d['rect']
        lr = math.log2(r)
        # この段の重み：拡大率が r/2^0.75 で出はじめ r/2^0.25 で 1（粗い段は覆わない所の下地）
        wk = 1.0 if k == 0 else float(R.smoothstep(lr - 1.0, lr - 0.5, L2))
        if wk <= 0.001:
            break
        u = ((MX - rect[0]) * r - 0.5).astype(np.float32)
        v = ((MY - rect[2]) * r - 0.5).astype(np.float32)
        o = cv2.remap(d['tex'], u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        o = o.astype(np.float32) * (1.0 / 255)
        th, tw = d['tex'].shape[:2]
        # 覆う所（縁から 6 テクセルでやわらかく）
        cm = np.minimum(np.minimum(u + 0.5, tw - 0.5 - u), np.minimum(v + 0.5, th - 0.5 - v))
        cm = R.smoothstep(6.0, 40.0, cm)
        if out is None:
            out = o
            out_cov = cm
        else:
            a = (cm * wk)[..., None]
            out = out * (1 - a) + o * a
            out_cov = np.maximum(out_cov, cm)
    return out, out_cov


def sample_global(lat, lon):
    t = global_tex()
    hh, ww = t.shape[:2]
    u = ((lon + 180.0) % 360.0) * (ww / 360.0) - 0.5
    v = (90.0 - lat) * (hh / 180.0) - 0.5
    return cv2.remap(t, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP).astype(np.float32) * (1.0 / 255)


# ---------------------------------------------------------------- 雲（遠い引きだけ）
CLEAR_LL = (33.6, 133.9)          # 晴れの中心（四国〜紀伊水道）
LON0, LON1 = 40.0, 230.0


def _cloud_tex():
    def mk():
        w, h = 4096, 2048
        lon = np.linspace(LON0, LON1, w, dtype=np.float32)[None, :]
        lat = np.linspace(80, -80, h, dtype=np.float32)[:, None]
        cl = np.cos(np.radians(lat))
        u = np.broadcast_to(lon * cl / 4.0, (h, w)).astype(np.float32)
        v = np.broadcast_to(lat / 4.0 + 0 * lon, (h, w)).astype(np.float32)
        wu = R.fbm(u * 0.5 + 3.1, v * 0.5, 3, seed=301) * 0.9
        wv = R.fbm(u * 0.5, v * 0.5 + 7.7, 3, seed=302) * 0.9
        n = R.fbm(u + wu, v + wv, 7, gain=0.55, seed=303)
        band = 0.25 * np.cos(np.radians(lat) * 6.0) + 0.0 * lon
        a = R.smoothstep(0.18, 0.75, n + band)
        fine = R.fbm(u * 9.0, v * 9.0, 3, seed=304)
        a = np.clip(a * (0.75 + 0.35 * fine), 0, 1) * 0.85
        la0, lo0 = math.radians(CLEAR_LL[0]), math.radians(CLEAR_LL[1])
        la, lo = np.radians(lat), np.radians(lon)
        d = np.arccos(np.clip(np.sin(la) * math.sin(la0) + np.cos(la) * math.cos(la0) * np.cos(lo - lo0), -1, 1)) * RE
        a = a * R.smoothstep(650.0, 1300.0, d)
        edge = R.smoothstep(LON0, LON0 + 12, lon) * (1 - R.smoothstep(LON1 - 12, LON1, lon))
        return (np.broadcast_to(a * edge, (h, w))).astype(np.float32)
    return R.cached('c_clouds_v2', mk)


def sample_clouds(lat, lon):
    t = _cloud_tex()
    hh, ww = t.shape
    u = ((lon % 360.0) - LON0) * (ww / (LON1 - LON0)) - 0.5
    v = (80.0 - lat) * (hh / 160.0) - 0.5
    return cv2.remap(t, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


# ---------------------------------------------------------------- 星・太陽
def stars():
    if 'stars' not in _S:
        rng = np.random.default_rng(1977)
        n = 2600
        v = rng.standard_normal((n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        mag = rng.random(n) ** 3.2                  # ほとんど暗い、少しだけ明るい
        tint = rng.random(n)
        _S['stars'] = (v, mag, tint)
    return _S['stars']


def star_layer(cam, alpha=1.0):
    """星の明るさ（H, W）。1px 前後のやわらかい点（点滅しない）。"""
    v, mag, _ = stars()
    q = cam.dir_proj(v)
    ok = np.isfinite(q).all(1) & (q[:, 0] > -3) & (q[:, 0] < W + 2) & (q[:, 1] > -3) & (q[:, 1] < H + 2)
    img = np.zeros((H + 8, W + 8), np.float32)
    x, y, m = q[ok, 0], q[ok, 1], mag[ok]
    b = (0.10 + 0.75 * m) * alpha
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - ix, y - iy
    for oy, wy in ((0, 1 - fy), (1, fy)):
        for ox, wx in ((0, 1 - fx), (1, fx)):
            np.add.at(img, (iy + oy + 4, ix + ox + 4), b * wx * wy)
    img = img[4:-4, 4:-4]
    big = cv2.GaussianBlur(img, (0, 0), 0.9) * 2.2
    return np.minimum(np.maximum(img * 0.8, big), 1.0)


def sun_glow(cam, D_sun, strength=1.0):
    """太陽（方向 D_sun）の円盤＋にじみ（線形の加算の明るさ (H, W)）。画面の外でもにじみは入る。"""
    q = cam.dir_proj(np.asarray(D_sun)[None])[0]
    if not np.isfinite(q).all():
        return None, None
    yy, xx = _grid()
    r = np.hypot(xx - q[0], yy - q[1])
    disc = np.clip(17.0 - r + 0.5, 0, 1)
    g = 0.40 * np.exp(-(r / 34.0) ** 2) + 0.10 * np.exp(-r / 110.0) + 0.018 * np.exp(-r / 380.0)
    return (disc * 1.1 + g) * strength, q


def _grid():
    if 'grid' not in _S:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        _S['grid'] = (yy, xx)
    return _S['grid']


# ---------------------------------------------------------------- 球の地球の 1 枚
GLOW_IN = R.hexl('9CCBEA')          # 地平の近くの大気（明るい空色）
GLOW_OUT = R.hexl('2F6FB0')         # 上の薄い層（深い青）
HAZE = R.hexl('4C8FC8')             # 地表の上のかすみ
SPACE = R.hexl('02060C')


def globe_frame(cam, t=0.0, zl=None, scale=1, haze_k=1.0, glow_k=1.0, star_a=1.0, sun=None, sun_k=1.0,
                land_tint=0.0, planar=None, planar_w=0.0, light=None, cloud_k=1.0, globe_k=1.0):
    """球の地球（線形ではなく sRGB float32 (H, W, 3) を返す）＋ water（H, W）＋ earth の覆い（H, W）。
    zl：狙いの点での拡大率（画面 px／地図の px、段の選び方に使う）。planar：(A 2x3：画面 → 地図の px)、planar_w：0..1。"""
    D_sun = SUN_DEFAULT if light is None else light
    lat, lon, cov, mu, hmin, path, ndl, spec = rays(cam, D_sun, scale)
    if scale != 1:
        up = lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)
        lat, lon, cov, mu, hmin, path, ndl, spec = (up(a) for a in (lat, lon, cov, mu, hmin, path, ndl, spec))
    MX, MY = mp_of(lat, lon)
    MX, MY = MX.astype(np.float32), MY.astype(np.float32)
    if planar is not None and planar_w > 0:
        yy, xx = _grid()
        A = planar
        PXm = (A[0, 0] * xx + A[0, 1] * yy + A[0, 2]).astype(np.float32)
        PYm = (A[1, 0] * xx + A[1, 1] * yy + A[1, 2]).astype(np.float32)
        MX = MX * (1 - planar_w) + PXm * planar_w
        MY = MY * (1 - planar_w) + PYm * planar_w
    lv, lcov = sample_levels(MX, MY, zl if zl is not None else 0.05)
    gl = sample_global(lat, lon)
    ca = sample_clouds(lat, lon) if cloud_k > 0 else np.zeros((H, W), np.float32)
    st = star_layer(cam, star_a) if star_a > 0 else np.zeros((H, W), np.float32)
    q = np.array([-1e6, -1e6])
    sk = 0.0
    if sun is not None:
        qq = cam.dir_proj(np.asarray(sun)[None])[0]
        if np.isfinite(qq).all():
            q, sk = qq, sun_k
    out = np.empty((H, W, 3), np.float32)
    water = np.empty((H, W), np.float32)
    _shade_k(np.ascontiguousarray(lv, np.float32), np.ascontiguousarray(lcov, np.float32),
             np.ascontiguousarray(gl, np.float32), ca, st, cov, path, hmin, ndl, spec,
             float(globe_k), float(cloud_k), float(haze_k), float(glow_k), float(q[0]), float(q[1]), float(sk),
             _C_GLINT, _C_CLOUD, HAZE.astype(np.float32), GLOW_IN.astype(np.float32), GLOW_OUT.astype(np.float32),
             SPACE.astype(np.float32), _C_STAR, R.SUN_RGB.astype(np.float32), out, water)
    return out, water, cov


_C_GLINT = R.hexl('BFD9EC').astype(np.float32)
_C_CLOUD = R.hexl('D7E5F1').astype(np.float32)
_C_STAR = R.hexl('DCE8F4').astype(np.float32)


@nb.njit(cache=True, fastmath=True, inline='always')
def _l1(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


@nb.njit(cache=True, fastmath=True, inline='always')
def _s1(x):
    x = min(max(x, 0.0), 1.0)
    return x * 12.92 if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055


@nb.njit(parallel=True, cache=True, fastmath=True)
def _shade_k(lv, lcov, gl, ca, st, cov, path, hmin, ndl, spec, gk, ck, hk, glk, qx, qy, sk,
             c_glint, c_cloud, c_haze, c_gin, c_gout, c_space, c_star, c_sun, out, water):
    h, w = cov.shape
    for y in nb.prange(h):
        for x in range(w):
            k = lcov[y, x]
            if k < 0:
                k = 0.0
            rgb = np.empty(3, np.float32)
            for c in range(3):
                rgb[c] = _l1(gl[y, x, c] * (1 - k) + lv[y, x, c] * k)
            wa = gl[y, x, 3] * (1 - k) + lv[y, x, 3] * k
            nd = ndl[y, x]
            if gk > 0:
                lam = min(max(nd * 0.8 + 0.35, 0.0), 1.15)
                m = 1 + gk * (lam - 0.85) * 0.9
                g = spec[y, x] ** 600 * 0.10 * wa * gk * (1.0 if nd > 0 else 0.0)
                for c in range(3):
                    rgb[c] = rgb[c] * m + c_glint[c] * g
            a = ca[y, x] * ck
            if a > 0:
                lc = min(max(0.30 + 0.75 * max(nd, 0.0), 0.0), 1.0)
                for c in range(3):
                    rgb[c] = rgb[c] * (1 - a) + c_cloud[c] * lc * a
            hz = (1 - math.exp(-path[y, x] / 520.0)) * hk
            for c in range(3):
                rgb[c] = rgb[c] * (1 - hz * 0.85) + c_haze[c] * hz * 0.85 + c_gin[c] * hz * hz * hz * 0.6
            # 宇宙
            hm = max(hmin[y, x], 0.0)
            gw = (math.exp(-hm / 9.0) * 0.85 + math.exp(-hm / 28.0) * 0.25) * glk
            e14 = math.exp(-hm / 14.0)
            gwc = min(max(gw, 0.0), 1.0)
            cv = cov[y, x]
            r2 = 0.0
            if sk > 0:
                d = math.sqrt((x + 0.5 - qx) ** 2 + (y + 0.5 - qy) ** 2)
                disc = min(max(17.0 - d + 0.5, 0.0), 1.0)
                r2 = (disc * 1.1 + 0.40 * math.exp(-(d / 34.0) ** 2) + 0.10 * math.exp(-d / 110.0)
                      + 0.018 * math.exp(-d / 380.0)) * sk * (1 - 0.7 * cv)
            for c in range(3):
                sp = c_space[c] + st[y, x] * c_star[c]
                gc = c_gin[c] * e14 + c_gout[c] * (1 - e14)
                sp = sp * (1 - gwc) + gc * gw
                o = rgb[c] * cv + sp * (1 - cv) + c_sun[c] * r2
                out[y, x, c] = _s1(o)
            water[y, x] = wa * cv


# ---------------------------------------------------------------- 平面の地図（真上のカメラ）
def planar_frame(A, zl):
    """A：画面 → 地図の px（2x3）。段を引き伸ばして重ねた sRGB と水のマスク（_maplook.render_affine と同じ材料）。"""
    yy, xx = _grid()
    MX = (A[0, 0] * xx + A[0, 1] * yy + A[0, 2]).astype(np.float32)
    MY = (A[1, 0] * xx + A[1, 1] * yy + A[1, 2]).astype(np.float32)
    tex, _ = sample_levels(MX, MY, zl)
    return np.ascontiguousarray(tex[..., :3]), np.ascontiguousarray(tex[..., 3])


def coast_mask(A, zl, alpha=0.40, width=1.25, inland_a=0.20):
    """拡大率に合う段の海岸線（1.25px・0.40）。A：画面 → 地図の px（相似）。"""
    lv = all_levels()
    L2 = math.log2(max(zl, 1e-4))
    k = 0
    for i, d in enumerate(lv):
        if L2 >= math.log2(d['r']) - 0.5:
            k = i
    d = lv[k]
    rx0, rx1, ry0, ry1 = d['rect']
    eps = 2.5 / d['r']                                   # テクスチャの縁（海の輪郭が縁に沿う所）は線にしない
    Ai = np.linalg.inv(np.vstack([A, [0, 0, 1]]))[:2]
    corners = np.array([[0, 0, 1], [W, 0, 1], [0, H, 1], [W, H, 1]], np.float64) @ A.T
    x0, y0 = corners.min(0) - 4
    x1, y1 = corners.max(0) + 4
    Lc = G.Lines()
    for name, a in (('sea', alpha), ('inland', inland_a)):
        for P in d['coast'][name]:
            if P[:, 0].max() < x0 or P[:, 0].min() > x1 or P[:, 1].max() < y0 or P[:, 1].min() > y1:
                continue
            if max(np.ptp(P[:, 0]), np.ptp(P[:, 1])) * zl < ML.MIN_LOOP:
                continue
            edge = (P[:, 0] < rx0 + eps) | (P[:, 0] > rx1 - eps) | (P[:, 1] < ry0 + eps) | (P[:, 1] > ry1 - eps)
            Q = P @ Ai[:, :2].T + Ai[:, 2]
            if edge.any():
                Q = Q.copy()
                Q[edge] = np.nan
            Lc.poly(Q, width, 255 * a, closed=False)
    return Lc.done()
