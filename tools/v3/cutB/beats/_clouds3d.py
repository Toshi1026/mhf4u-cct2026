"""体積の積雲（美術監督の指示 2026-11-08：01-1）。numba、CPU。1 つの雲を 1 回だけ焼いて、毎フレームはずらすだけ。

作り方
  形：4〜7 個の楕円体のなめらかな和（底は平ら）× もこもこの 3 段のノイズ（1-|perlin| のビロウ）で縁をけずる。
  光：各ボクセルから太陽の方へ 8 歩（Beer–Lambert、σ≈1.2）→ 太陽の透過。パウダーの項 1-exp(-2d) を弱めに、
      Henyey–Greenstein（g=0.6 の前方＋後方の 2 つ）の位相、空からの環境の光（上は明るい空色、底は #8FA9C6）。
  視線：正射影で手前から奥へ。1 画素あたりのレイは ss×ss（2× で焼いて面積で縮める）。縁の細かいノイズは視線の計算の中で足す。
  アルファ＝1-exp(-∫σ·密度)。縁は密度の場のまま自然にやわらかい（暗い縁の線は出ない）。

  cloud_sprite(w, h, seed, sun=..., ...) → (prem 線形 RGB (h, w, 3), alpha (h, w))   ※ ディスクに uint16 でキャッシュ
"""
import hashlib
import math
import os

import cv2
import numba as nb
import numpy as np

import core

CACHE = os.path.join(core.OUT, 'cache', 'clouds3d')
os.makedirs(CACHE, exist_ok=True)
LK = 0.7
SUN_SCREEN = (-0.55, -0.65, 0.52)          # 画面の向き（x 右、y 下、z 手前）。全ショット共通のキーの光


def _perm(seed):
    rng = np.random.default_rng(seed)
    p = rng.permutation(256).astype(np.int32)
    return np.concatenate([p, p]).astype(np.int32)


@nb.njit(fastmath=True, cache=True, inline='always')
def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


@nb.njit(fastmath=True, cache=True, inline='always')
def _grad(h, x, y, z):
    h = h & 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if (h == 12 or h == 14) else z)
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


@nb.njit(fastmath=True, cache=True)
def perlin3(x, y, z, P):
    xi, yi, zi = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
    xf, yf, zf = x - xi, y - yi, z - zi
    xi &= 255
    yi &= 255
    zi &= 255
    u, v, w = _fade(xf), _fade(yf), _fade(zf)
    a = P[xi] + yi
    aa = P[a] + zi
    ab = P[a + 1] + zi
    b = P[xi + 1] + yi
    ba = P[b] + zi
    bb = P[b + 1] + zi
    x1 = _grad(P[aa], xf, yf, zf) + u * (_grad(P[ba], xf - 1, yf, zf) - _grad(P[aa], xf, yf, zf))
    x2 = _grad(P[ab], xf, yf - 1, zf) + u * (_grad(P[bb], xf - 1, yf - 1, zf) - _grad(P[ab], xf, yf - 1, zf))
    y1 = x1 + v * (x2 - x1)
    x3 = _grad(P[aa + 1], xf, yf, zf - 1) + u * (_grad(P[ba + 1], xf - 1, yf, zf - 1) - _grad(P[aa + 1], xf, yf, zf - 1))
    x4 = _grad(P[ab + 1], xf, yf - 1, zf - 1) + u * (_grad(P[bb + 1], xf - 1, yf - 1, zf - 1) -
                                                     _grad(P[ab + 1], xf, yf - 1, zf - 1))
    y2 = x3 + v * (x4 - x3)
    return y1 + w * (y2 - y1)


@nb.njit(fastmath=True, cache=True, inline='always')
def _billow(x, y, z, P, oct_):
    s = 0.0
    a = 1.0
    tot = 0.0
    for o in range(oct_):
        s += (1.0 - abs(perlin3(x, y, z, P))) * a
        tot += a
        x *= 2.03
        y *= 2.03
        z *= 2.03
        a *= 0.5
    return s / tot


@nb.njit(fastmath=True, cache=True, parallel=True)
def _density(E, nx, ny, nz, ybase, freq, erosion, P, out):
    """E：(k, 6) の楕円体（中心・半径、ボクセル）。なめらかな最大（log-sum-exp）で形の場 s（中で >0）。"""
    k = E.shape[0]
    for yi in nb.prange(ny):
        y = yi + 0.5
        for xi in range(nx):
            x = xi + 0.5
            for zi in range(nz):
                z = zi + 0.5
                yb2 = ybase + perlin3(x * 0.05, z * 0.05, 7.7, P) * 4.0      # 底の高さも少しゆらぐ
                flat = min(max((yb2 - y) / 7.0, 0.0), 1.0)                     # 平らな底（7 ボクセルでやわらかく）
                acc = 0.0
                for j in range(k):
                    qx = (x - E[j, 0]) / E[j, 3]
                    qy = (y - E[j, 1]) / E[j, 4]
                    qz = (z - E[j, 2]) / E[j, 5]
                    acc += math.exp(14.0 * (1.0 - math.sqrt(qx * qx + qy * qy + qz * qz)) * E[j, 3] / (0.09 * nx))
                s = math.log(acc + 1e-12) / 14.0 if acc > 0 else -10.0
                if s < -0.35 or flat <= 0:
                    out[xi, yi, zi] = 0.0
                    continue
                b = _billow(x * freq, y * freq, z * freq, P, 3)
                d = (s + 0.10 - erosion * (1.0 - b)) / 0.70
                out[xi, yi, zi] = min(max(d, 0.0), 1.0) * flat


@nb.njit(fastmath=True, cache=True, inline='always')
def _tri(G, x, y, z):
    nx, ny, nz = G.shape[0], G.shape[1], G.shape[2]
    x -= 0.5
    y -= 0.5
    z -= 0.5
    if x < 0 or y < 0 or z < 0 or x >= nx - 1 or y >= ny - 1 or z >= nz - 1:
        return 0.0
    i, j, k = int(x), int(y), int(z)
    fx, fy, fz = x - i, y - j, z - k
    c00 = G[i, j, k] * (1 - fx) + G[i + 1, j, k] * fx
    c10 = G[i, j + 1, k] * (1 - fx) + G[i + 1, j + 1, k] * fx
    c01 = G[i, j, k + 1] * (1 - fx) + G[i + 1, j, k + 1] * fx
    c11 = G[i, j + 1, k + 1] * (1 - fx) + G[i + 1, j + 1, k + 1] * fx
    return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz


@nb.njit(fastmath=True, cache=True, parallel=True)
def _light(D, Lx, Ly, Lz, sigma, out_sun, out_amb):
    """各ボクセルの太陽の透過（8 歩、だんだん大きく）と、上からの環境の光の透過（4 歩）。"""
    nx, ny, nz = D.shape[0], D.shape[1], D.shape[2]
    for xi in nb.prange(nx):
        for yi in range(ny):
            for zi in range(nz):
                if D[xi, yi, zi] <= 0.0:
                    out_sun[xi, yi, zi] = 1.0
                    out_amb[xi, yi, zi] = 1.0
                    continue
                x, y, z = xi + 0.5, yi + 0.5, zi + 0.5
                od = 0.0
                hsh = math.sin(xi * 12.9898 + yi * 78.233 + zi * 37.719) * 43758.5453
                hsh = hsh - math.floor(hsh)
                st = 0.8
                t = 0.3 + 0.8 * hsh
                for s in range(8):
                    od += _tri(D, x + Lx * t, y + Ly * t, z + Lz * t) * st
                    t += st
                    st *= 1.45
                out_sun[xi, yi, zi] = math.exp(-sigma * od * LK)
                oa = 0.0
                t = 1.0
                for s in range(4):
                    oa += _tri(D, x, y - t, z) * (1.0 + s)
                    t += 1.0 + s
                out_amb[xi, yi, zi] = math.exp(-sigma * oa * 0.12)


@nb.njit(fastmath=True, cache=True, parallel=True)
def _view(D, TS, TA, ss, sigma, phase, sun_r, sun_g, sun_b, top_r, top_g, top_b, und_r, und_g, und_b,
          ytop, ybase, dfreq, dero, P2, Lx, Ly, out):
    """正射影（手前 z=nz から奥へ）。out：(H, W, 4) の prem 線形 RGB＋A（H＝ny·ss、W＝nx·ss）。"""
    nx, ny, nz = D.shape[0], D.shape[1], D.shape[2]
    Hh, Ww = out.shape[0], out.shape[1]
    dt = 0.6
    for py in nb.prange(Hh):
        y = (py + 0.5) / ss
        hfrac = min(max((ybase - y) / max(ybase - ytop, 1.0), 0.0), 1.0)
        for px in range(Ww):
            x = (px + 0.5) / ss
            T = 1.0
            cr = 0.0
            cg = 0.0
            cb = 0.0
            z = nz - 0.5
            while z > 0.5:
                d = _tri(D, x, y, z)
                if d > 0.002:
                    # 縁の細かいけずり（密度の小さい所ほど強く）
                    if d < 0.6:
                        b = _billow(x * dfreq, y * dfreq, z * dfreq, P2, 2)
                        d = max(d - dero * (1.0 - b) * (1.0 - d / 0.6), 0.0)
                    if d > 0.0:
                        ts = _tri(TS, x, y, z)
                        ta = _tri(TA, x, y, z)
                        pw = 1.0 - 0.45 * math.exp(-2.0 * d * 3.0)           # パウダー（弱め：暗い縁を作らない）
                        sun = ts * phase * pw
                        # 太陽に向いた薄い所：わずかに銀色に（縁の内側だけ。外へはみ出さない）
                        amb = (0.45 + 0.55 * ta) * (0.75 + 0.25 * hfrac)
                        ar = und_r + (top_r - und_r) * hfrac
                        ag = und_g + (top_g - und_g) * hfrac
                        ab = und_b + (top_b - und_b) * hfrac
                        sr = sun_r * sun + ar * amb
                        sg = sun_g * sun + ag * amb
                        sb = sun_b * sun + ab * amb
                        a = 1.0 - math.exp(-sigma * d * dt)
                        cr += T * a * sr
                        cg += T * a * sg
                        cb += T * a * sb
                        T *= 1.0 - a
                        if T < 0.004:
                            break
                z -= dt
            out[py, px, 0] = cr
            out[py, px, 1] = cg
            out[py, px, 2] = cb
            out[py, px, 3] = 1.0 - T


def _hg(c, g):
    return (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * c) ** 1.5)


def _lin(h):
    c = np.asarray(core.c(h), np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _layout(nx, ny, nz, seed, towers, flat_k):
    """楕円体の並び（ボクセル）。土台の横長の 2 個＋上へもり上がる塔 towers 個＋塔の上半分に小さなこぶ（カリフラワー）。"""
    rng = np.random.default_rng(seed)
    yb = ny * 0.86
    E = []
    cx = nx * 0.5
    E.append((cx + rng.uniform(-0.04, 0.04) * nx, yb - ny * 0.13, nz * 0.5, nx * 0.38, ny * 0.18 * flat_k, nz * 0.30))
    E.append((cx + rng.uniform(-0.14, 0.14) * nx, yb - ny * 0.20, nz * 0.5 + rng.uniform(-0.1, 0.1) * nz,
              nx * 0.27, ny * 0.22 * flat_k, nz * 0.30))
    for j in range(4):                               # 土台の上のこぶ（土台と塔の段をなくす）
        bx = cx + rng.uniform(-0.32, 0.32) * nx
        br = nx * rng.uniform(0.07, 0.10)
        E.append((bx, yb - ny * 0.26 + rng.uniform(-0.03, 0.05) * ny, nz * 0.5 + rng.uniform(-0.15, 0.15) * nz,
                  br, br * 0.9, br))
    towers = int(towers)
    T = []
    for k in range(towers):
        q = (k + 0.5) / towers
        x = nx * (0.22 + 0.56 * q + rng.uniform(-0.04, 0.04))
        tall = math.sin(math.pi * (0.12 + 0.76 * q)) * rng.uniform(0.7, 1.0)
        r = min(nx * rng.uniform(0.10, 0.145) * (0.8 + 0.45 * tall), ny * 0.24)
        y = yb - ny * 0.26 - (ny * 0.26) * tall * flat_k
        z = nz * 0.5 + rng.uniform(-0.12, 0.12) * nz
        T.append((x, y, z, r))
        E.append((x, y, z, r, r * rng.uniform(0.88, 1.0), r * 0.95))
    # こぶ：塔の上半分（とくに太陽の側＝左上）に、半径 0.35〜0.55 倍の球を半分うめて
    for (x, y, z, r) in T:
        for j in range(int(rng.integers(3, 6))):
            th = rng.uniform(-math.pi * 0.95, -math.pi * 0.05)          # 上半分（y 下向きなので sin<0 が上）
            ph = rng.uniform(-0.9, 0.9)
            rr = r * rng.uniform(0.34, 0.52)
            px = x + math.cos(th) * math.cos(ph) * r * 0.82
            py = y + math.sin(th) * math.cos(ph) * r * 0.82
            pz = z + math.sin(ph) * r * 0.82
            if py - rr < ny * 0.03:
                continue
            E.append((px, py, pz, rr, rr * 0.92, rr))
    return np.array(E, np.float64)


def cloud_sprite(w, h, seed=0, sun=SUN_SCREEN, ss=2, towers=4, flat_k=1.0, sigma=1.2, erosion=0.55,
                 sun_rgb=(1.0, 0.99, 0.975), sun_k=3.2, top='DCEBF8', under='8FA9C6', amb_k=0.92, detail=1.0,
                 tag='v1'):
    """積雲 1 つ（w×h px）。(prem 線形 RGB, alpha)。光の強さは「白の上限 0.92」に収まるように後で合わせる。"""
    key = repr((w, h, seed, tuple(np.round(sun, 3)), ss, towers, flat_k, sigma, erosion, sun_k, top, under, amb_k,
                detail, tag))
    path = os.path.join(CACHE, hashlib.md5(key.encode()).hexdigest()[:16] + '.png')
    if os.path.exists(path):
        q = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        q = cv2.cvtColor(q, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 65535.0
        return q[..., :3] * 1.5, q[..., 3]
    vox = 2.0                                       # 1 ボクセル＝出力の 2px
    nx, ny = int(math.ceil(w / vox)), int(math.ceil(h / vox))
    nz = int(nx * 0.62)
    E = _layout(nx, ny, nz, seed, towers, flat_k)
    D = np.zeros((nx, ny, nz), np.float32)
    P = _perm(seed + 101)
    _density(E, nx, ny, nz, ny * 0.86, 13.0 / nx, erosion, P, D)
    L = np.array(sun, np.float64)
    L /= np.linalg.norm(L)
    TS = np.zeros_like(D)
    TA = np.zeros_like(D)
    _light(D, L[0], L[1], L[2], sigma, TS, TA)
    cos_t = -L[2]                                   # 入る光（-L）と出る光（+z）の角度
    phase = (0.7 * _hg(cos_t, 0.6) + 0.3 * _hg(cos_t, -0.3)) * 4 * math.pi
    sr = np.array(sun_rgb) * sun_k
    tp, un = _lin(top) * amb_k, _lin(under) * amb_k
    ys = np.nonzero(D.max(axis=(0, 2)) > 0.01)[0]
    ytop = float(ys.min()) if len(ys) else 0.0
    out = np.zeros((ny * ss * int(vox), nx * ss * int(vox), 4), np.float32)
    P2 = _perm(seed + 202)
    _view(D, TS, TA, ss * vox, sigma, phase, sr[0], sr[1], sr[2], tp[0], tp[1], tp[2], un[0], un[1], un[2],
          ytop, ny * 0.86, 34.0 / nx, 0.22 * detail, P2, L[0], L[1], out)
    small = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
    rgb, a = small[..., :3], small[..., 3]
    q = np.concatenate([np.clip(rgb / 1.5, 0, 1), np.clip(a, 0, 1)[..., None]], 2)
    cv2.imwrite(path, cv2.cvtColor((q * 65535 + 0.5).astype(np.uint16), cv2.COLOR_RGBA2BGRA))
    return rgb.astype(np.float32), a.astype(np.float32)
