"""地図・地球の共通部品（b03／b04／b07／b08／b09 で使う）。

- Natural Earth 50m の陸（tools/models/ne/ne_50m_land）を正距円筒の陸マスクに焼いて exports/v3/cache にキャッシュ
- Globe：正射図法の地球（海は DEEP→NAVY のグラデーション＋縁のやわらかい光、陸は LAND の平塗り、海岸線 1.5px FOAM 60%）
- 大円の補間、角を丸めた経路、破線、進み具合つきの線、2倍で描いて縮める線レイヤー（cv2）
"""
import os

import cv2
import numpy as np

import core
from core import W, H

CACHE = os.path.join(core.OUT, 'cache')
NE_LAND = os.path.join(core.MODELS, 'ne', 'ne_50m_land', 'ne_50m_land.shp')
os.makedirs(CACHE, exist_ok=True)
EARTH_KM = 6371.0


# ---------------------------------------------------------------- 球面の基本
def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def latlon(p):
    p = p / np.linalg.norm(p, axis=-1, keepdims=True)
    return np.degrees(np.arcsin(np.clip(p[..., 2], -1, 1))), np.degrees(np.arctan2(p[..., 1], p[..., 0]))


def slerp(a, b, t):
    """単位ベクトル a→b の大円上の t（0..1）の点。t は配列でもよい。"""
    t = np.asarray(t, np.float64)[..., None]
    om = np.arccos(np.clip(np.dot(a, b), -1, 1))
    if om < 1e-9:
        return np.broadcast_to(a, t.shape[:-1] + (3,)).copy()
    return (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / np.sin(om)


def gc_line(a, b, step_deg=0.5):
    """(lat, lon) a→b の大円を step_deg ごとの単位ベクトル列に。"""
    A, B = xyz(*a), xyz(*b)
    om = np.degrees(np.arccos(np.clip(np.dot(A, B), -1, 1)))
    n = max(2, int(np.ceil(om / step_deg)) + 1)
    return slerp(A, B, np.linspace(0, 1, n))


def gc_km(a, b):
    return float(np.arccos(np.clip(np.dot(xyz(*a), xyz(*b)), -1, 1)) * EARTH_KM)


def route(waypoints, step_deg=0.4, round_deg=4.0):
    """経由点を大円で結び、角を丸めた（Chaikin＋正規化）単位ベクトルの列。"""
    pts = [gc_line(waypoints[i], waypoints[i + 1], step_deg)[(1 if i else 0):] for i in range(len(waypoints) - 1)]
    P = np.concatenate(pts)
    # 角の丸め：移動平均（窓は round_deg 相当）を数回。両端は固定
    k = max(1, int(round(round_deg / step_deg)))
    for _ in range(3):
        Q = P.copy()
        cs = np.cumsum(np.vstack([np.zeros((1, 3)), P]), axis=0)
        for i in range(1, len(P) - 1):
            r = min(k, i, len(P) - 1 - i)
            Q[i] = (cs[i + r + 1] - cs[i - r]) / (2 * r + 1)
        P = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    return P


def arclen(P):
    """単位ベクトル列の累積弧長（ラジアン）。"""
    d = np.arccos(np.clip(np.sum(P[1:] * P[:-1], axis=1), -1, 1))
    return np.concatenate([[0], np.cumsum(d)])


def cut(P, s, frac):
    """累積長 s を持つ点列 P を全長の frac までで切った点列（最後の点は補間）と先端の点。"""
    L = s[-1] * float(np.clip(frac, 0, 1))
    i = int(np.searchsorted(s, L))
    if i <= 0:
        return P[:1].copy(), P[0]
    if i >= len(P):
        return P.copy(), P[-1]
    t = (L - s[i - 1]) / max(s[i] - s[i - 1], 1e-12)
    tip = P[i - 1] * (1 - t) + P[i] * t
    if P.shape[1] == 3:
        tip = tip / np.linalg.norm(tip)
    return np.vstack([P[:i], tip[None]]), tip


# ---------------------------------------------------------------- Natural Earth の陸
_RINGS = None


def rings():
    """陸の輪郭（lon, lat）の配列のリスト。"""
    global _RINGS
    if _RINGS is None:
        npz = os.path.join(CACHE, 'ne50_rings.npz')
        if os.path.exists(npz):
            z = np.load(npz)
            pts, idx = z['pts'], z['idx']
            _RINGS = [pts[idx[i]:idx[i + 1]] for i in range(len(idx) - 1)]
        else:
            import shapefile
            sf = shapefile.Reader(NE_LAND)
            out = []
            for sh in sf.shapes():
                pp = np.array(sh.points, np.float64)
                parts = list(sh.parts) + [len(pp)]
                for a, b in zip(parts[:-1], parts[1:]):
                    if b - a >= 3:
                        out.append(pp[a:b])
            idx = np.cumsum([0] + [len(r) for r in out])
            np.savez_compressed(npz, pts=np.concatenate(out), idx=idx)
            _RINGS = out
    return _RINGS


_MASKS = {}


def land_mask(width):
    """正距円筒（lon -180..180 → x、lat 90..-90 → y）の陸マスク float32 0..1。width は 2 のべき（512..16384）。"""
    if width in _MASKS:
        return _MASKS[width]
    path = os.path.join(CACHE, 'ne50_land_%d.png' % width)
    if os.path.exists(path):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        if width == 16384:
            h = width // 2
            m = np.zeros((h, width), np.uint8)
            sc = width / 360.0
            polys = [np.round(np.stack([(r[:, 0] + 180) * sc, (90 - r[:, 1]) * sc], 1) * 16).astype(np.int32)
                     for r in rings()]
            cv2.fillPoly(m, polys, 255, lineType=cv2.LINE_AA, shift=4)
        else:
            big = land_mask(width * 2)
            m = cv2.resize((big * 255).astype(np.uint8), (width, width // 2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(path, m)
    _MASKS[width] = m.astype(np.float32) / 255
    return _MASKS[width]


def land_at(lat, lon, width=4096):
    """lat/lon 配列での陸マスク（双一次）。"""
    key = ('pad', width)
    if key not in _MASKS:
        m = land_mask(width)
        _MASKS[key] = np.ascontiguousarray(np.concatenate([m[:, -1:], m, m[:, :1]], axis=1))  # 経度方向の折り返し用
    mx = _MASKS[key]
    u = ((np.asarray(lon, np.float32) + 180) % 360) * np.float32(width / 360) + np.float32(0.5)
    v = (90 - np.asarray(lat, np.float32)) * np.float32((width // 2) / 180) - np.float32(0.5)
    return remap_flat(mx, u, v, cv2.BORDER_REPLICATE)


def remap_flat(src, u, v, border=cv2.BORDER_REPLICATE, value=0):
    """任意の形の座標配列 u, v で src を双一次サンプリング（cv2.remap の大きさの制限を避けて 2D に組み直す）。"""
    shp = u.shape
    u, v = u.reshape(-1), v.reshape(-1)
    n = len(u)
    cols = 2048
    rows = (n + cols - 1) // cols
    pad = rows * cols - n
    if pad:
        u = np.concatenate([u, np.zeros(pad, np.float32)])
        v = np.concatenate([v, np.zeros(pad, np.float32)])
    out = cv2.remap(src, u.reshape(rows, cols).astype(np.float32), v.reshape(rows, cols).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=border, borderValue=value)
    return out.reshape(-1)[:n].reshape(shp)


_COAST = None


def coast_xyz():
    """海岸線の単位ベクトル（全リングを NaN 区切りで連結）。"""
    global _COAST
    if _COAST is None:
        parts = []
        for r in rings():
            parts.append(xyz(r[:, 1], r[:, 0]))
            parts.append(np.full((1, 3), np.nan))
        _COAST = np.concatenate(parts).astype(np.float64)
    return _COAST


# ---------------------------------------------------------------- 2倍で描いて縮める線（cv2）
class Lines:
    """1枚の単色マスクに cv2 で 2 倍の解像度で線を描き、INTER_AREA で縮める。太さは小数 px でよい。"""
    S = 2

    def __init__(self, w=W, h=H):
        self.w, self.h = w, h
        self.m = np.zeros((h * self.S, w * self.S), np.uint8)

    def poly(self, pts, width=1.5, value=255, closed=False):
        """pts：(N,2) の画面座標、または (N,2) のリスト。NaN を含む行で線を切る。"""
        if isinstance(pts, np.ndarray):
            pts = split_nan(pts)
        segs = [np.round(p * self.S * 16).astype(np.int32) for p in pts if len(p) >= 2]
        if segs:
            cv2.polylines(self.m, segs, closed, int(value), thickness=max(1, int(round(width * self.S))),
                          lineType=cv2.LINE_AA, shift=4)

    def dot(self, x, y, r, value=255):
        cv2.circle(self.m, (int(round(x * self.S * 16)), int(round(y * self.S * 16))), int(round(r * self.S * 16)),
                   int(value), -1, cv2.LINE_AA, 4)

    def done(self):
        return cv2.resize(self.m, (self.w, self.h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def split_nan(P):
    """NaN の行で区切った連続部分のリスト。"""
    bad = ~np.isfinite(P).all(axis=1)
    if not bad.any():
        return [P]
    out, start = [], 0
    for i in np.flatnonzero(bad):
        if i - start >= 2:
            out.append(P[start:i])
        start = i + 1
    if len(P) - start >= 2:
        out.append(P[start:])
    return out


def dashed(P, dash=14, gap=10, phase=0.0):
    """画面座標の折れ線 P（NaN 区切り可）を破線にした部分のリスト。"""
    out = []
    for seg in split_nan(P):
        d = np.hypot(*np.diff(seg, axis=0).T)
        s = np.concatenate([[0], np.cumsum(d)])
        per = dash + gap
        k0 = int(np.floor((0 - phase) / per)) - 1
        k1 = int(np.ceil((s[-1] - phase) / per)) + 1
        for k in range(k0, k1):
            a, b = phase + k * per, phase + k * per + dash
            a, b = max(a, 0), min(b, s[-1])
            if b - a < 0.5:
                continue
            ia, ib = np.searchsorted(s, a), np.searchsorted(s, b)
            pa = np.array([np.interp(a, s, seg[:, 0]), np.interp(a, s, seg[:, 1])])
            pb = np.array([np.interp(b, s, seg[:, 0]), np.interp(b, s, seg[:, 1])])
            out.append(np.vstack([pa, seg[ia:ib], pb]))
    return out


def bbox(mask, thr=1e-3):
    rows = np.flatnonzero(mask.max(axis=1) > thr)
    if not len(rows):
        return None
    cols = np.flatnonzero(mask[rows[0]:rows[-1] + 1].max(axis=0) > thr)
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def tint(frame, mask, rgb, alpha=1.0):
    """frame に mask（0..1）× alpha の強さで単色 rgb を重ねる（frame をその場で書き換えて返す）。"""
    b = bbox(mask)
    if b is None or alpha <= 0:
        return frame
    y0, y1, x0, x1 = b
    a = mask[y0:y1, x0:x1, None] * np.float32(alpha)
    sub = frame[y0:y1, x0:x1]
    sub += (np.asarray(rgb, np.float32) - sub) * a
    return frame


def glow(mask, sigma):
    return cv2.GaussianBlur(mask, (0, 0), sigma)


# ---------------------------------------------------------------- 地球（正射図法）
OCEAN_C = core.lerp(core.DEEP, core.SEA, 0.18)     # 中心の海
OCEAN_E = core.lerp(core.NAVY, core.DEEP, 0.35)    # 縁の海
RIM = core.lerp(core.SEA, core.SKY, 0.5)           # 縁の光
BG_TOP = core.lerp(core.NAVY, core.DEEP, 0.30)
BG_BOT = core.NAVY


def background():
    return core.vgrad(BG_TOP, BG_BOT, gamma=0.9)


class View:
    """lat0/lon0：画面中心に来る点、R：半径 px、(cx, cy)：その点の画面位置。"""

    def __init__(self, lat0, lon0, R, cx=W / 2, cy=H / 2):
        self.lat0, self.lon0, self.R, self.cx, self.cy = lat0, lon0, R, cx, cy
        la, lo = np.radians(lat0), np.radians(lon0)
        self.c = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
        self.e = np.array([-np.sin(lo), np.cos(lo), 0.0])
        self.n = np.cross(self.c, self.e)

    def proj(self, P):
        """単位ベクトル（…,3）→ 画面 x, y と奥行き z（>0 なら見える）。"""
        x = self.cx + self.R * (P @ self.e)
        y = self.cy - self.R * (P @ self.n)
        return x, y, P @ self.c

    def proj_ll(self, lat, lon):
        return self.proj(xyz(lat, lon))

    def screen(self, P, hide=True):
        """(N,3) → (N,2) 画面座標。見えない点は NaN（線がそこで切れる）。"""
        x, y, z = self.proj(P)
        S = np.stack([x, y], 1)
        if hide:
            S[z < 0.0] = np.nan
        return S


def view_lerp(a, b, t):
    """2つの View の間（中心は大円で、半径は対数で補間）。a, b は (lat, lon, R, cx, cy)。"""
    A, B = xyz(a[0], a[1]), xyz(b[0], b[1])
    la, lo = latlon(slerp(A, B, t))
    R = float(np.exp(np.log(a[2]) * (1 - t) + np.log(b[2]) * t))
    return View(float(la), float(lo), R, core.lerp(a[3], b[3], t), core.lerp(a[4], b[4], t))


class Globe:
    def __init__(self, coast_alpha=0.6, grat_alpha=0.06):
        self.bg = background()
        self.coast_alpha, self.grat_alpha = coast_alpha, grat_alpha
        self.coast = coast_xyz()
        g = []
        for lat in range(-60, 61, 30):
            g.append(xyz(np.full(361, lat), np.arange(-180, 181)))
            g.append(np.full((1, 3), np.nan))
        for lon in range(-180, 180, 30):
            g.append(xyz(np.linspace(-90, 90, 181), np.full(181, lon)))
            g.append(np.full((1, 3), np.nan))
        self.grat = np.concatenate(g)
        self._detail = None
        self._rad = None

    # 拡大したとき用の高解像度の陸（ポルトガル周辺）
    def detail(self):
        if self._detail is None:
            path = os.path.join(CACHE, 'ne50_land_portugal.png')
            box = (-14.0, -4.0, 35.0, 45.0)  # lon0, lon1, lat0, lat1
            ppd = 400
            w, h = int((box[1] - box[0]) * ppd), int((box[3] - box[2]) * ppd)
            if os.path.exists(path):
                m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            else:
                m = np.zeros((h, w), np.uint8)
                polys = []
                for r in rings():
                    if (r[:, 0].max() < box[0] or r[:, 0].min() > box[1] or r[:, 1].max() < box[2]
                            or r[:, 1].min() > box[3]):
                        continue
                    polys.append(np.round(np.stack([(r[:, 0] - box[0]) * ppd, (box[3] - r[:, 1]) * ppd], 1) * 16)
                                 .astype(np.int32))
                cv2.fillPoly(m, polys, 255, lineType=cv2.LINE_AA, shift=4)
                cv2.imwrite(path, m)
            self._detail = (m.astype(np.float32) / 255, box, ppd)
        return self._detail

    def _radial(self, R, cx, cy):
        """半径と中心だけで決まる部分（海のグラデーション・縁の光・大気）をキャッシュ。"""
        key = (round(R, 3), round(cx, 3), round(cy, 3))
        if self._rad is not None and self._rad[0] == key:
            return self._rad[1]
        m = 1.45
        x0, x1 = int(max(0, cx - R * m)), int(min(W, cx + R * m + 1))
        y0, y1 = int(max(0, cy - R * m)), int(min(H, cy + R * m + 1))
        if x1 <= x0 or y1 <= y0:
            d = None
        else:
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            X = (xx + 0.5 - cx) / np.float32(R)
            Y = -(yy + 0.5 - cy) / np.float32(R)
            r2 = X * X + Y * Y
            r = np.sqrt(r2)
            out = np.clip(r - 1, 0, None)
            halo = (np.exp(-out / 0.035) * 0.40 + np.exp(-out / 0.10) * 0.12) * (r > 1)
            halo *= np.clip((m - r) / 0.1, 0, 1)
            aa = np.clip((1 - r) * R + 0.5, 0, 1)
            ins = r < 1.0 + 1.0 / R
            Z = np.sqrt(np.clip(1 - r2, 0, 1))
            t = np.clip(1 - Z, 0, 1) ** 0.8
            lam = np.clip(X * -0.35 + Y * 0.45 + Z * 0.82, 0, 1)
            oc = OCEAN_C * (1 - t[..., None]) + OCEAN_E * t[..., None]
            oc = oc * (0.88 + 0.16 * lam[..., None])
            rim = np.exp(-(Z / 0.10)) * 0.55
            oc = oc + (RIM - oc) * rim[..., None]
            shade = (0.90 + 0.10 * lam) * (1 - 0.18 * t)
            iy, ix = np.nonzero(ins)
            d = dict(box=(y0, y1, x0, x1), halo=halo[..., None].astype(np.float32), aa=aa[..., None],
                     oc=oc.astype(np.float32), shade=shade[..., None].astype(np.float32),
                     iy=iy, ix=ix, X=X[iy, ix], Y=Y[iy, ix], Z=Z[iy, ix])
        self._rad = (key, d)
        return d

    def render(self, v, land_rgb=None, bright=1.0):
        """v：View。地球の画（H,W,3）を返す。"""
        land_rgb = core.LAND if land_rgb is None else land_rgb
        f = self.bg.copy()
        R = v.R
        d = self._radial(R, v.cx, v.cy)
        if d is None:
            return f
        y0, y1, x0, x1 = d['box']
        sub = f[y0:y1, x0:x1]
        sub += (RIM - sub) * d['halo']
        # 見えている各画素の緯度経度 → 陸マスク
        e, n, c = (np.asarray(a, np.float32) for a in (v.e, v.n, v.c))
        X, Y, Z = d['X'], d['Y'], d['Z']
        px = X * e[0] + Y * n[0] + Z * c[0]
        py = X * e[1] + Y * n[1] + Z * c[1]
        pz = X * e[2] + Y * n[2] + Z * c[2]
        lat = np.degrees(np.arcsin(np.clip(pz, -1, 1)))
        lon = np.degrees(np.arctan2(py, px))
        width = 1024
        while width < 16384 and width < 2 * np.pi * R:
            width *= 2
        land = land_at(lat, lon, width)
        if R > 3000:
            m, box, ppd = self.detail()
            u = ((lon - box[0]) * ppd - 0.5).astype(np.float32)
            vv = ((box[3] - lat) * ppd - 0.5).astype(np.float32)
            dl = remap_flat(m, u, vv, cv2.BORDER_CONSTANT, -1)
            land = np.where(dl >= 0, dl, land)
        L = np.zeros(sub.shape[:2], np.float32)
        L[d['iy'], d['ix']] = land
        L = L[..., None]
        g = d['oc'] * (1 - L) + land_rgb * d['shade'] * L
        if bright != 1.0:
            g = g * bright
        a = d['aa']
        f[y0:y1, x0:x1] = sub * (1 - a) + g * a
        # 経緯線と海岸線
        if self.grat_alpha > 0:
            L2 = Lines()
            L2.poly(self._clip(v, self.grat), 1.2)
            f = tint(f, L2.done(), core.FOAM, self.grat_alpha * bright)
        L1 = Lines()
        L1.poly(self._clip(v, self.coast), 1.5)
        f = tint(f, L1.done(), core.FOAM, self.coast_alpha * bright)
        return f

    @staticmethod
    def _clip(v, P):
        x, y, z = v.proj(P)
        S = np.stack([x, y], 1)
        bad = (z < 0.0) | (x < -200) | (x > W + 200) | (y < -200) | (y > H + 200)
        S[bad] = np.nan
        return S


# ---------------------------------------------------------------- 線を描く（画面座標で）
def draw_line(frame, pts, rgb, width=2.0, alpha=1.0, glow_sigma=0, glow_alpha=0.0):
    """pts：(N,2)（NaN 区切り可）またはそのリスト。"""
    L = Lines()
    if isinstance(pts, list):
        for p in pts:
            L.poly(p, width)
    else:
        L.poly(pts, width)
    m = L.done()
    if glow_sigma and glow_alpha:
        frame = tint(frame, np.clip(glow(m, glow_sigma) * 2.2, 0, 1), rgb, glow_alpha)
    return tint(frame, m, rgb, alpha)


def draw_dots(frame, pts, r, rgb, alpha=1.0, glow_sigma=0, glow_alpha=0.0):
    L = Lines()
    for x, y in pts:
        if np.isfinite(x) and np.isfinite(y):
            L.dot(x, y, r)
    m = L.done()
    if glow_sigma and glow_alpha:
        frame = tint(frame, np.clip(glow(m, glow_sigma) * 2.5, 0, 1), rgb, glow_alpha)
    return tint(frame, m, rgb, alpha)
