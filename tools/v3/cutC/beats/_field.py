"""案C 03〜05 の「現実版」の世界（質感のある CG）：草原・空・雲の層・パンチの衝撃（衝撃波・雲の穴・なぎ倒れる草・クレーター・
砂煙・飛ぶ土くれ）。地震・建物の倒壊・津波を思わせるもの（地割れ・水の壁・水の輪・光る海岸）は描かない。光らせない。

座標（m）：x 左（Cam3D の向きのため世界の +x が画面の左）、y 上、z 奥（カメラ A から見て）。パンチが当たる所＝原点。地面 y=0。光は左うしろ上（影は右奥へ）。
色はすべて線形の光で計算して最後に sRGB（_render.srgb）。カメラは _render.Cam3D。

  sky_deck(cam, hole_r=0, ...)              空＋雲の層（高さ ALT の平面の fBm、原点の真上に丸い穴）。線形 (H, W, 3)
  ground_tex(X, Z, fp, flat=0, view=None)   草原の地面の色（線形）。flat：なぎ倒れた度合い 0..1
  Blades                                    手前の草の葉（スクリーンで散らして世界に置く 3D の線）。立つ／なぎ倒れる
  crater_height(r_scale) / CraterView       クレーター（高さの場）と、カメラ C からのレイマーチ（numba）
  dust_puffs()                              砂煙の板（体積の積雲の描き方を砂の色で）
  Clods                                     飛ぶ土くれ（3D の放物線、ぶれの線）
  refract(frame, cam, R, center)            衝撃波の殻の屈折（画面のずらし）
"""
import math

import cv2
import numpy as np

import core
from core import W, H
from cutB.beats import _render as R

hx = R.hexl
SUN = R.sun_vec(140.0, 36.0)                        # 左うしろ上（Cam3D は世界の +x が画面の左）
SUN_RGB = np.array([1.0, 0.985, 0.965], np.float32) * 2.6
ZEN, HOR, HAZE = hx('2468AE'), hx('B5DAF3'), hx('C4E0F2')
ALT = 1300.0                                        # 雲の層の高さ（m）
# 草（青みのある緑。黄にしない）
G_DARK, G_MID, G_LIGHT = hx('1F5232'), hx('3C8A4C'), hx('8FC48F')
G_FLAT, G_FLAT_HI = hx('4C8A55'), hx('98C49A')       # なぎ倒れた草（葉の裏の銀色のつや）
SOIL, SOIL_LIT, SOIL_DK = hx('5A5856'), hx('8C8A86'), hx('2F3133')
DUST_LIT, DUST_SH = hx('C2C9CE'), hx('6F7C88')
_C = {}


def fbm1(u, v, *a, **k):
    """1 次元の長い配列でも使える R.fbm（cv2.remap の 32767 の制限をよける）。"""
    u = np.asarray(u, np.float32)
    n = u.size
    m = int(math.ceil(n / 256.0)) * 256
    U = np.zeros(m, np.float32)
    V = np.zeros(m, np.float32)
    U[:n] = u.ravel()
    V[:n] = np.asarray(v, np.float32).ravel()
    return R.fbm(U.reshape(-1, 256), V.reshape(-1, 256), *a, **k).ravel()[:n].reshape(u.shape)


# =============================================================== 空と雲の層
def sky_rad(D):
    return R.sky_radiance(D[..., 0], D[..., 1], D[..., 2], SUN, zen=ZEN, hor=HOR, haze=HAZE, mie=0.10,
                          sun_disk=False)


def _deck_density(X, Z, seed=31):
    wx = R.fbm(X * 0.55, Z * 0.55, 3, seed=seed + 1) * 0.35
    d0 = R.fbm(X + wx, Z * 1.25 + wx, 6, gain=0.52, seed=seed)
    return d0 * 0.5 + 0.5


def deck(cam, res=0.5, hole_r=0.0, hole_c=(0.0, 0.0), cov=0.52, t=0.0, rim_push=0.0, D=None, sc=2400.0, sharp=3.0):
    """雲の層（線形 RGB, α）を res 倍で作って拡大。hole_r：原点の真上の丸い穴の半径（m）。
    穴の縁はわずかにちぎれ（角度のノイズ 3%）、外側に押しのけられた雲が少し厚くなる（rim_push 0..1）。"""
    hh, ww = int(cam.h * res), int(cam.w * res)
    if D is None:
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        D = cam.rays((yy + 0.5) / res, (xx + 0.5) / res).astype(np.float32)
    dy = D[..., 1]
    up = dy > 2e-3
    dyc = np.maximum(dy, 2e-3)
    dist = (ALT - cam.eye[1]) / dyc
    Xw = cam.eye[0] + D[..., 0] * dist
    Zw = cam.eye[2] + D[..., 2] * dist
    X = Xw / sc + t * 0.004
    Z = Zw / sc
    d = _deck_density(X, Z)
    soft = 0.20
    dd = np.clip(R.smoothstep(1 - cov - soft * 0.3, 1 - cov + soft, d) * (d - (1 - cov) + soft * 0.3), 0, None) * sharp
    if hole_r > 0:
        rx, rz = Xw - hole_c[0], Zw - hole_c[1]
        r = np.sqrt(rx * rx + rz * rz)
        th = np.arctan2(rz, rx)
        edge = hole_r * (1 + 0.025 * R.fbm(np.cos(th) * 3.0, np.sin(th) * 3.0, 3, seed=77) + 0.012 * R.fbm(np.cos(th) * 24.0, np.sin(th) * 24.0, 2, seed=78))
        wdt = max(hole_r * 0.02, 4.0)
        keep = R.smoothstep(edge - wdt, edge + wdt, r)
        rn = R.fbm(np.cos(th) * 9.0 + r / 300.0, np.sin(th) * 9.0, 3, seed=79) * 0.5 + 0.5
        bump = 1 + rim_push * 0.9 * np.exp(-((r - edge * 1.06) / (hole_r * 0.07 + 1)) ** 2) * (0.3 + 0.9 * rn)
        dd = dd * keep * bump
    # 影：太陽の方へずらした 2 回目
    off = 0.05
    d1 = _deck_density(X + SUN[0] * off, Z + SUN[2] * off)
    d1 = np.clip(R.smoothstep(1 - cov - soft * 0.3, 1 - cov + soft, d1) * (d1 - (1 - cov) + soft * 0.3), 0, None) * 3.0
    shadow = np.exp(-np.clip(d1 - dd * 0.5, 0, None) * 2.2)
    a = 1 - np.exp(-dd * 1.8)
    a = a * R.smoothstep(0.0, 0.10, dyc) * up
    lit = np.array([1.0, 1.0, 1.0], np.float32) * 0.86
    shade = hx('8AA4BF')
    colr = shade * (1 - shadow)[..., None] + lit * shadow[..., None]
    colr = colr * (1 - 0.30 * np.clip(dd - 0.4, 0, 1))[..., None]          # 厚い所の底は暗く青い
    hz = np.exp(-np.maximum(dy, 0) / 0.05)[..., None]
    colr = colr * (1 - 0.6 * hz) + HAZE * 0.6 * hz
    if res != 1.0:
        colr = cv2.resize(colr, (cam.w, cam.h), interpolation=cv2.INTER_CUBIC)
        a = np.clip(cv2.resize(a, (cam.w, cam.h), interpolation=cv2.INTER_CUBIC), 0, 1)
    return colr.astype(np.float32), a.astype(np.float32)


def sky_deck(cam, res=0.5, D=None, **kw):
    """空＋雲の層（線形、画面全体。地平線より下も空の色で埋める）。"""
    yy, xx = np.mgrid[0:cam.h, 0:cam.w].astype(np.float32)
    Df = cam.rays(yy + 0.5, xx + 0.5).astype(np.float32) if D is None else D
    s = sky_rad(np.concatenate([Df[..., :1], np.maximum(Df[..., 1:2], 0), Df[..., 2:]], -1))
    c, a = deck(cam, res, **kw)
    return s * (1 - a[..., None]) + c * a[..., None], Df


# =============================================================== 地面（草原）
def ground_tex(X, Z, fp, flat=None, rad=None, view=None, seed=0):
    """草原の色（線形）。fp：1px の地面の大きさ（m）。flat（0..1）：なぎ倒れた度合い、rad：なぎ倒れの向き（単位ベクトルの x,z）、
    view：視線の水平の向き（x,z）。倒れた葉は向こうへ倒れたものほど明るい（葉の裏のつや）。"""
    n1 = R.fbm(X / 22.0, Z / 22.0, 4, seed=seed + 3) * 0.7 + R.fbm(X / 90.0, Z / 90.0, 3, seed=seed + 4) * 0.6
    n2 = R.fbm(X / 4.0, Z / 4.0, 3, seed=seed + 5)
    fine_k = np.clip(1.2 - fp / 0.05, 0, 1)
    n3 = R.fbm(X / 0.45, Z / 0.45, 2, seed=seed + 7) * fine_k
    k = np.clip(0.5 + 0.30 * n1 + 0.18 * n2 + 0.20 * n3, 0, 1)[..., None]
    col = G_DARK * (1 - k) + G_MID * k
    col = col + (G_LIGHT - col) * np.clip(0.18 * (n2 * 0.5 + 0.5) ** 3 + 0.25 * np.clip(n3, 0, 1), 0, 1)[..., None]
    if flat is not None:
        fl = np.clip(flat, 0, 1)
        # 放射の筋（倒れた向きに長い）：極座標のノイズ
        r = np.sqrt(X * X + Z * Z) + 1e-3
        th = np.arctan2(Z, X)
        st = R.fbm(r / 9.0, th * 60.0, 3, seed=seed + 11) * 0.5 + 0.5
        if rad is not None and view is not None:
            sheen = np.clip(rad[0] * view[0] + rad[1] * view[1], -1, 1)      # 向こうへ倒れた葉 +1
        else:
            sheen = np.zeros_like(X)
        kf = np.clip(0.45 + 0.35 * sheen + 0.35 * (st - 0.5) + 0.12 * n2, 0, 1)[..., None]
        fcol = G_FLAT * (1 - kf) * 0.80 + G_FLAT_HI * kf * 0.80
        fcol = fcol * 0.55 + col * 0.45
        col = col * (1 - fl[..., None]) + fcol * fl[..., None]
    return col


def aerial(col, dist, k=1 / 2600.0):
    a = (1 - np.exp(-dist * k))[..., None]
    return col * (1 - a) + HAZE * a


def light_ground(col, shade=1.0):
    """草の照明（太陽＋空）。shade：影 0..1。"""
    sky_amb = hx('B4C8DA') * 0.40
    return col * (SUN_RGB * 0.50 * SUN[1] * np.asarray(shade)[..., None] + sky_amb)


# =============================================================== 手前の草の葉
class Blades:
    """手前の草の葉を画面で散らして（下ほど少なく大きい）、地面 (X, Z) に根を置いた 3D の線分にする。
    draw(cam, t, flat_fn) → (rgba 線形, 深さ)。flat_fn(X, Z) → 0..1（なぎ倒れの度合い）。倒れた葉は原点から外へ寝る。"""

    def __init__(self, cam, n=60000, zmax=26.0, seed=3):
        rng = np.random.default_rng(seed)
        hz_y = cam.project(np.array([[0.0, cam.eye[1], 1e6]]))[0][0, 1]
        ys = hz_y + (H + 40 - hz_y) * rng.uniform(0, 1, n * 2) ** 0.55
        xs = rng.uniform(-40, W + 40, n * 2)
        D = cam.rays(ys, xs)
        t = -cam.eye[1] / np.minimum(D[:, 1], -1e-4)
        X, Z = cam.eye[0] + D[:, 0] * t, cam.eye[2] + D[:, 2] * t
        dep = np.hypot(X - cam.eye[0], Z - cam.eye[2])
        ok = dep < zmax
        X, Z, dep = X[ok][:n], Z[ok][:n], dep[ok][:n]
        m = len(X)
        self.X, self.Z, self.dep = X, Z, dep
        self.h = rng.uniform(0.10, 0.30, m) * (1 + 0.35 * (fbm1(X / 3.0, Z / 3.0, 2, seed=8) * 0.5 + 0.5))
        self.lean = rng.uniform(0, 2 * np.pi, m)
        self.lk = rng.uniform(0.05, 0.35, m)
        self.w = rng.uniform(0.003, 0.007, m)
        patch = fbm1(X / 2.5, Z / 2.5, 3, seed=12)
        self.tone = np.clip(0.48 + 0.22 * rng.standard_normal(m) + 0.35 * patch, 0, 1)
        self.ph = rng.uniform(0, 2 * np.pi, m)
        order = np.argsort(-dep)
        for k in ('X', 'Z', 'dep', 'h', 'lean', 'lk', 'w', 'tone', 'ph'):
            setattr(self, k, getattr(self, k)[order])

    def draw(self, cam, t=0.0, flat_fn=None, wind=1.0, ss=2, sel_fn=None, q=16):
        """(img 線形, α, 深さ)。sel_fn(dep) → bool の配列：描く葉を選ぶ（奥／手前に分けるとき）。"""
        m = len(self.X)
        h = self.h
        lx = np.cos(self.lean) * self.lk + 0.10 * wind * np.sin(t * 1.3 + self.ph + self.X * 0.4)
        lz = np.sin(self.lean) * self.lk
        if flat_fn is not None:
            fl = np.clip(flat_fn(self.X, self.Z), 0, 1)
            r = np.hypot(self.X, self.Z) + 1e-3
            ux, uz = self.X / r, self.Z / r
            lx = lx * (1 - fl) + ux * 1.6 * fl
            lz = lz * (1 - fl) + uz * 1.6 * fl
            hh = h * (1 - 0.82 * fl)
        else:
            fl = np.zeros(m, np.float32)
            hh = h
        use = np.ones(m, bool) if sel_fn is None else sel_fn(self.dep)
        root = np.stack([self.X, np.zeros(m), self.Z], 1)[use]
        mid = (root + np.stack([lx * h * 0.35, hh * 0.55, lz * h * 0.35], 1)[use])
        tip = (root + np.stack([lx * h, hh, lz * h], 1)[use])
        fl, dep_b = fl[use], self.dep[use]
        P0, _ = cam.project(root)
        P1, _ = cam.project(mid)
        P2, _ = cam.project(tip)
        px_w = self.w[use] * cam.fpx / np.maximum(dep_b, 0.3)
        img = np.zeros((H * ss, W * ss, 3), np.float32)
        al = np.zeros((H * ss, W * ss), np.float32)
        dep = np.zeros((H * ss, W * ss), np.float32)
        tone = np.clip(self.tone[use] * (1 - fl) + fl, 0, 1)
        kb = np.clip(tone * (q - 1), 0, q - 1).round().astype(int)
        wb = np.clip(np.round(px_w * ss), 1, 12).astype(int)
        wb = np.where(wb > 4, (wb // 2) * 2, wb)
        shift = 3
        mul = float(ss * (1 << shift))
        S0 = np.round(np.stack([P0, P1], 1) * mul).astype(np.int32)
        S1 = np.round(np.stack([P1, P2], 1) * mul).astype(np.int32)
        n = len(P0)
        bands = np.array_split(np.arange(n), 4)
        for idx in bands:
            if len(idx) == 0:
                continue
            dv = float(np.mean(dep_b[idx]))
            key = kb[idx] * 100 + wb[idx]
            for kv in np.unique(key):
                sel = idx[key == kv]
                b, wv = divmod(int(kv), 100)
                tq = b / (q - 1)
                c_lo = G_DARK * (0.55 + 0.35 * tq) + G_FLAT * 0.25 * tq
                c_hi = G_MID * (1 - tq) + G_LIGHT * tq if tq < 0.92 else G_FLAT_HI * 0.92
                c_hi = c_hi * (0.72 + 0.28 * tq)
                dk = float(np.clip(0.62 + 0.38 * (dv - 2.0) / 10.0, 0.62, 1.0))   # 手前の葉は陰が多い
                c_lo, c_hi = c_lo * dk, c_hi * dk
                p0, p1 = list(S0[sel]), list(S1[sel])
                cv2.polylines(img, p0, False, tuple(float(v) for v in c_lo), wv, cv2.LINE_AA, shift)
                cv2.polylines(img, p1, False, tuple(float(v) for v in c_hi), max(1, int(wv * 0.7)), cv2.LINE_AA, shift)
                cv2.polylines(al, p0, False, 1.0, wv, cv2.LINE_AA, shift)
                cv2.polylines(al, p1, False, 1.0, max(1, int(wv * 0.7)), cv2.LINE_AA, shift)
                cv2.polylines(dep, p0, False, dv, wv, cv2.LINE_8, shift)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        al = np.clip(cv2.resize(al, (W, H), interpolation=cv2.INTER_AREA), 0, 1)
        dep = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST)
        img = img / np.maximum(al, 1e-4)[..., None]
        return img, al, dep


# =============================================================== クレーター（高さの場）とレイマーチ
import numba as _nb


def crater_height(N=640, cell=0.5, Rr=48.0, depth=13.0, seed=0):
    """原点のまわり N×N（1 マス cell m）の高さ（m）と、色に使う場（ray：放出物の筋の濃さ、bowl：内側の度合い）。
    碗形の内側・盛り上がった縁・外へうすくなる放出物の毛布（放射の筋）。地割れは描かない。"""
    c = (np.arange(N, dtype=np.float32) - N / 2 + 0.5) * cell
    X, Z = np.meshgrid(c, c)
    r = np.hypot(X, Z)
    th = np.arctan2(Z, X)
    u = r / Rr
    n1 = R.fbm(X / 9.0, Z / 9.0, 4, seed=seed + 21)
    n2 = R.fbm(X / 2.2, Z / 2.2, 3, seed=seed + 23)
    rays = np.clip(R.fbm(np.cos(th) * 16.0 + r / 400.0, np.sin(th) * 16.0, 4, seed=seed + 25) * 0.5 + 0.5, 0, 1)
    rays = R.smoothstep(0.40, 0.80, rays) * (0.6 + 0.4 * (R.fbm(X / 14.0, Z / 14.0, 3, seed=seed + 27) * 0.5 + 0.5))
    bowl = np.where(u < 1, -depth * np.clip(1 - u * u, 0, 1) ** 0.75, 0.0)
    rim = 5.5 * np.exp(-((u - 1.0) / 0.14) ** 2) * (1 + 0.25 * n1)
    blanket = np.where(u > 1, 1.6 * np.exp(-(u - 1) / 0.5) * (0.9 + 0.2 * (n1 * 0.5 + 0.5)), 0.0)
    h = bowl + rim + blanket + (0.5 * n1 + 0.15 * n2) * np.clip(1.6 - u * 0.6, 0.2, 1.2)
    h = h * R.smoothstep(4.6, 3.2, u)                     # 外はまっすぐ平らに
    blot = R.fbm(X / 11.0, Z / 11.0, 5, seed=seed + 29) * 0.5 + 0.5
    ejecta = np.clip(np.where(u > 0.9, np.exp(-np.clip(u - 1, 0, None) / 0.35) * (0.55 + 0.7 * blot) + 0.35 * rays * np.exp(-np.clip(u - 1, 0, None) / 1.1), 1.0), 0, 1)
    return h.astype(np.float32), ejecta.astype(np.float32), (u < 1.0).astype(np.float32), X, Z


@_nb.njit(parallel=True, fastmath=True, cache=True)
def _crater_march(D, eye, Hm, x0, z0, cell, hmax, hmin, rb, out):
    hh, ww = D.shape[0], D.shape[1]
    n = Hm.shape[0]
    for j in _nb.prange(hh):
        for i in range(ww):
            dx, dy, dz = D[j, i, 0], D[j, i, 1], D[j, i, 2]
            if dy > -1e-5:
                out[j, i, 0] = -1.0
                continue
            tp = -eye[1] / dy                              # 平面 y=0
            # 高さの範囲の板の中だけ歩く
            ta = (eye[1] - hmax) / -dy
            tb = (eye[1] - hmin) / -dy
            hit = -1.0
            # 円柱 r<rb と交わるか
            ox, oz = eye[0], eye[2]
            a = dx * dx + dz * dz
            b = 2 * (ox * dx + oz * dz)
            c = ox * ox + oz * oz - rb * rb
            disc = b * b - 4 * a * c
            if disc > 0 and a > 1e-12:
                sq = math.sqrt(disc)
                t0 = (-b - sq) / (2 * a)
                t1 = (-b + sq) / (2 * a)
                lo = max(ta, t0)
                hi = min(tb, t1)
                if hi > lo:
                    hor = math.sqrt(a)
                    dt = cell * 0.5 / hor
                    t = lo
                    prev = lo
                    while t <= hi:
                        px, py, pz = ox + dx * t, eye[1] + dy * t, oz + dz * t
                        u = (px - x0) / cell
                        v = (pz - z0) / cell
                        if u < 0:
                            u = 0.0
                        if v < 0:
                            v = 0.0
                        if u > n - 1.001:
                            u = n - 1.001
                        if v > n - 1.001:
                            v = n - 1.001
                        i0, j0 = int(u), int(v)
                        fu, fv = u - i0, v - j0
                        hg = (Hm[j0, i0] * (1 - fu) + Hm[j0, i0 + 1] * fu) * (1 - fv) + \
                            (Hm[j0 + 1, i0] * (1 - fu) + Hm[j0 + 1, i0 + 1] * fu) * fv
                        if py <= hg:
                            lo2, hi2 = prev, t
                            for _ in range(7):
                                m = 0.5 * (lo2 + hi2)
                                px, py, pz = ox + dx * m, eye[1] + dy * m, oz + dz * m
                                u = min(max((px - x0) / cell, 0.0), n - 1.001)
                                v = min(max((pz - z0) / cell, 0.0), n - 1.001)
                                i0, j0 = int(u), int(v)
                                fu, fv = u - i0, v - j0
                                hg = (Hm[j0, i0] * (1 - fu) + Hm[j0, i0 + 1] * fu) * (1 - fv) + \
                                    (Hm[j0 + 1, i0] * (1 - fu) + Hm[j0 + 1, i0 + 1] * fu) * fv
                                if py <= hg:
                                    hi2 = m
                                else:
                                    lo2 = m
                            hit = 0.5 * (lo2 + hi2)
                            break
                        prev = t
                        t += dt
                    if hit < 0 and t1 < tp:
                        hit = -2.0                         # 円柱を抜けたあと平面へ
            if hit < 0:
                hit = tp
            out[j, i, 0] = hit
            out[j, i, 1] = ox + dx * hit
            out[j, i, 2] = oz + dz * hit
            out[j, i, 3] = eye[1] + dy * hit


def crater_hits(cam, D, Hm, cell, rb):
    """各画素の (t, x, z, y)。空は t<0。"""
    n = Hm.shape[0]
    x0 = -(n / 2 - 0.5) * cell
    out = np.zeros(D.shape[:2] + (4,), np.float64)
    _crater_march(np.ascontiguousarray(D, np.float64), cam.eye.astype(np.float64), np.ascontiguousarray(Hm), x0, x0,
                  float(cell), float(Hm.max()) + 0.01, float(Hm.min()) - 0.01, float(rb), out)
    return out
