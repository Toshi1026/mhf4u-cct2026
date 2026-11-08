"""案B の CG の人形を「本当の 3D」で描く小さな SDF レイマーチャ（numba、CPU 4 スレッド）。look-dev 2026-11。

なぜ：高さの場（_render.Relief）は正面からの光にしか強くない。SDF なら、自分の影（腕が胴に落とす影）、
やわらかい環境の遮へい（AO）、正しい輪郭のリム、つや（GGX）、布の起伏、スクワッシュ（2D のアフィン）がそのまま出る。
スプライトの大きさ（〜400px 四方）なら 1 枚 0.03〜0.3 秒。

座標：画面の向き（x 右、y 下、z 手前）。正射影（beats の 2D の配置とそのまま重なる）。
形は「ローカル単位」で組み、Xform（2×2 の A・中心 c・奥行きの倍率）で画面へ。A は回転・スクワッシュ・反転を含んでよい
（レイはローカル空間で z に沿ってまっすぐ進むので、距離はゆがまない）。

API
  Scene()                                     形を足していく入れ物
    .cone(a, b, ra, rb, k=0, mat=0)           太さの変わる丸い棒（a, b：(x, y, z)）
    .sphere(c, r, k=0, mat=0)
    .ellipsoid(c, (rx, ry, rz), k=0, mat=0)
    .rbox(c, (hx, hy, hz), r, k=0, mat=0)     角の丸い箱
    .cyl(a, b, r, rr=0, k=0, mat=0)            角の丸い円柱（袖口・リブ）
    .slab(poly, z, half, r, k=0, mat=0, fold=(amp, period, phase, dirx, diry), bend=0)
                                              多角形を押し出した板（布・耳）。fold：板の z を sin で波打たせる。
                                              bend：板の中心からの距離²で z を反らせる（耳の反り）
    .sub(...)                                 （上の各関数に op='sub' を渡すと、それまでの形から削る＝溝）
  Mat(rgb, spec, rough, sss, rim, coat, sheen)   材質（rgb は sRGB 0..1）
  render(scene, mats, w, h, A, c, zs=1.0, ss=3, light=..., env=...)
      → (rgba (h, w, 4) ストレートアルファの sRGB float32, depth (h, w)）
      env：dict(sun=線形の色, sky=, ground=, rim=, bounce=)。shadow_k：自分の影のやわらかさ（大きいほどくっきり）。
"""
import math

import numpy as np
import numba as nb
import cv2

import core

T_CONE, T_ELL, T_SLAB, T_RBOX, T_CYL = 0, 1, 2, 3, 4


@nb.njit(fastmath=True, cache=True)
def _sd_poly(px, py, V, o, n):
    d = (px - V[o, 0]) ** 2 + (py - V[o, 1]) ** 2
    s = 1.0
    j = o + n - 1
    for i in range(o, o + n):
        ex, ey = V[j, 0] - V[i, 0], V[j, 1] - V[i, 1]
        wx, wy = px - V[i, 0], py - V[i, 1]
        h = (wx * ex + wy * ey) / (ex * ex + ey * ey + 1e-12)
        h = min(max(h, 0.0), 1.0)
        bx, by = wx - ex * h, wy - ey * h
        d = min(d, bx * bx + by * by)
        c1 = py >= V[i, 1]
        c2 = py < V[j, 1]
        c3 = ex * wy > ey * wx
        if (c1 and c2 and c3) or ((not c1) and (not c2) and (not c3)):
            s = -s
        j = i
    return s * math.sqrt(d)


@nb.njit(fastmath=True, cache=True)
def _prim(P, i, x, y, z, V):
    t = int(P[i, 0])
    if t == 0:                                          # round cone（太さの変わるカプセル）
        ax, ay, az, bx, by, bz, ra, rb = P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6], P[i, 7], P[i, 8]
        bax, bay, baz = bx - ax, by - ay, bz - az
        pax, pay, paz = x - ax, y - ay, z - az
        h = (pax * bax + pay * bay + paz * baz) / (bax * bax + bay * bay + baz * baz + 1e-12)
        h = min(max(h, 0.0), 1.0)
        dx, dy, dz = pax - bax * h, pay - bay * h, paz - baz * h
        return math.sqrt(dx * dx + dy * dy + dz * dz) - (ra + (rb - ra) * h)
    if t == 1:                                          # ellipsoid（近似の距離）
        cx, cy, cz, rx, ry, rz = P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6]
        qx, qy, qz = (x - cx) / rx, (y - cy) / ry, (z - cz) / rz
        k0 = math.sqrt(qx * qx + qy * qy + qz * qz)
        k1 = math.sqrt((qx / rx) ** 2 + (qy / ry) ** 2 + (qz / rz) ** 2)
        return k0 * (k0 - 1.0) / (k1 + 1e-9)
    if t == 2:                                          # 押し出した多角形の板
        o, n = int(P[i, 1]), int(P[i, 2])
        zc, half, r = P[i, 3], P[i, 4], P[i, 5]
        amp, per, ph, fdx, fdy = P[i, 6], P[i, 7], P[i, 8], P[i, 13], P[i, 14]
        bend, bcx, bcy = P[i, 15], P[i, 16], P[i, 17]
        zz = zc
        if amp != 0.0:
            zz += amp * math.sin((x * fdx + y * fdy) / per + ph)
        if bend != 0.0:
            zz += bend * ((x - bcx) ** 2 + (y - bcy) ** 2)
        d2 = _sd_poly(x, y, V, o, n) + r
        dz = abs(z - zz) - half + r
        ox, oz = max(d2, 0.0), max(dz, 0.0)
        return (math.sqrt(ox * ox + oz * oz) + min(max(d2, dz), 0.0) - r) * 0.8
    if t == 4:                                          # 角の丸い円柱（a → b、半径 r、丸み rr）
        ax, ay, az, bx, by, bz, r, rr = P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6], P[i, 7], P[i, 8]
        bax, bay, baz = bx - ax, by - ay, bz - az
        L = math.sqrt(bax * bax + bay * bay + baz * baz) + 1e-12
        ux, uy, uz = bax / L, bay / L, baz / L
        pax, pay, paz = x - (ax + bx) * 0.5, y - (ay + by) * 0.5, z - (az + bz) * 0.5
        h = pax * ux + pay * uy + paz * uz
        qx, qy, qz = pax - ux * h, pay - uy * h, paz - uz * h
        dr = math.sqrt(qx * qx + qy * qy + qz * qz) - (r - rr)
        dh = abs(h) - (L * 0.5 - rr)
        ox, oz = max(dr, 0.0), max(dh, 0.0)
        return math.sqrt(ox * ox + oz * oz) + min(max(dr, dh), 0.0) - rr
    # rounded box
    cx, cy, cz, hx, hy, hz, r = P[i, 1], P[i, 2], P[i, 3], P[i, 4], P[i, 5], P[i, 6], P[i, 7]
    qx, qy, qz = abs(x - cx) - hx + r, abs(y - cy) - hy + r, abs(z - cz) - hz + r
    ox, oy, oz = max(qx, 0.0), max(qy, 0.0), max(qz, 0.0)
    return math.sqrt(ox * ox + oy * oy + oz * oz) + min(max(qx, max(qy, qz)), 0.0) - r


@nb.njit(fastmath=True, cache=True)
def _map(P, V, x, y, z):
    """(距離, 材質)。P[i, 9]=k（なめらかな和の幅）、P[i, 10]=材質、P[i, 11]=1 で削る。"""
    d = 1e9
    m = 0.0
    for i in range(P.shape[0]):
        # 包む球で早めに捨てる（遠い形は計算しない）
        bx, by, bz, br = P[i, 18], P[i, 19], P[i, 20], P[i, 21]
        db = math.sqrt((x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2) - br
        k = P[i, 9]
        if P[i, 11] > 0.5:
            if db > 0.0:
                continue
            di = _prim(P, i, x, y, z, V)
            # なめらかな差
            if k > 0.0:
                h = min(max(0.5 - 0.5 * (d + di) / k, 0.0), 1.0)
                d = d * (1 - h) + (-di) * h + k * h * (1 - h)
            else:
                d = max(d, -di)
            continue
        if db > d + k:
            continue
        di = _prim(P, i, x, y, z, V)
        if k > 0.0:
            h = max(k - abs(d - di), 0.0) / k
            if di < d:
                m = P[i, 10]
            d = min(d, di) - h * h * k * 0.25
        else:
            if di < d:
                d = di
                m = P[i, 10]
    return d, m


@nb.njit(fastmath=True, cache=True)
def _normal(P, V, x, y, z, e):
    a, _ = _map(P, V, x + e, y - e, z - e)
    b, _ = _map(P, V, x - e, y - e, z + e)
    c, _ = _map(P, V, x - e, y + e, z - e)
    d, _ = _map(P, V, x + e, y + e, z + e)
    nx = a - b - c + d
    ny = -a - b + c + d
    nz = -a + b - c + d
    n = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
    return nx / n, ny / n, nz / n


@nb.njit(parallel=True, fastmath=True, cache=True)
def _march(P, V, w, h, Ai, cx, cy, zs, z0, z1, eps, out_pos, out_m, out_hit):
    """各画素（画面の板 w×h）のレイ（ローカル空間で -z へ）を進めて、当たったローカルの位置・材質を記録。"""
    for j in nb.prange(h):
        for i in range(w):
            sx, sy = i + 0.5 - cx, j + 0.5 - cy
            lx = Ai[0, 0] * sx + Ai[0, 1] * sy
            ly = Ai[1, 0] * sx + Ai[1, 1] * sy
            z = z1
            hit = False
            m = 0.0
            for k in range(160):
                d, m = _map(P, V, lx, ly, z)
                if d < eps:
                    hit = True
                    break
                z -= max(d, eps * 0.5)
                if z < z0:
                    break
            if hit:
                out_pos[j, i, 0] = lx
                out_pos[j, i, 1] = ly
                out_pos[j, i, 2] = z
                out_m[j, i] = m
                out_hit[j, i] = 1


@nb.njit(parallel=True, fastmath=True, cache=True)
def _shade(P, V, w, h, pos, mm, hit, AiT, zs, L, MATS, env, shadow_k, ao_d, eps, out):
    """当たった画素の照明（線形）。L：ローカルの光の向き（単位）、Lw：画面の向き。env：(15,) の環境の光。"""
    sun = (env[0], env[1], env[2])
    sky = (env[3], env[4], env[5])
    gnd = (env[6], env[7], env[8])
    rim = (env[9], env[10], env[11])
    Lw = (env[12], env[13], env[14])
    for j in nb.prange(h):
        for i in range(w):
            if hit[j, i] == 0:
                continue
            x, y, z = pos[j, i, 0], pos[j, i, 1], pos[j, i, 2]
            nlx, nly, nlz = _normal(P, V, x, y, z, eps * 0.75)
            # ローカルの法線 → 画面の法線（A^-T、z は 1/zs）
            nx = AiT[0, 0] * nlx + AiT[0, 1] * nly
            ny = AiT[1, 0] * nlx + AiT[1, 1] * nly
            nz = nlz / zs
            nn = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
            nx, ny, nz = nx / nn, ny / nn, nz / nn
            mi = int(mm[j, i])
            ar, ag, ab = MATS[mi, 0], MATS[mi, 1], MATS[mi, 2]
            spec, rough, sss, rimk, coat, sheen = MATS[mi, 3], MATS[mi, 4], MATS[mi, 5], MATS[mi, 6], MATS[mi, 7], MATS[mi, 8]
            # 自分の影（やわらかい）：ローカル空間で光の向きへ
            res = 1.0
            s = eps * 4.0
            px, py, pz = x + nlx * eps * 2, y + nly * eps * 2, z + nlz * eps * 2
            for k in range(48):
                d, _ = _map(P, V, px + L[0] * s, py + L[1] * s, pz + L[2] * s)
                res = min(res, shadow_k * d / s)
                s += min(max(d, eps), ao_d * 2)
                if res < 0.002 or s > ao_d * 40:
                    break
            res = max(res, 0.0)
            res = res * res * (3 - 2 * res)
            # 環境の遮へい（法線の方向に 5 点）
            occ = 0.0
            wgt = 1.0
            for k in range(1, 6):
                hh = ao_d * k
                d, _ = _map(P, V, x + nlx * hh, y + nly * hh, z + nlz * hh)
                occ += (hh - d) * wgt
                wgt *= 0.6
            ao = min(max(1.0 - occ / (ao_d * 2.2), 0.0), 1.0)
            ndl = nx * Lw[0] + ny * Lw[1] + nz * Lw[2]
            dif = max((ndl + sss) / (1 + sss), 0.0)
            dif_sh = dif * (res * (1 - sss * 0.35) + sss * 0.35 * min(1.0, res + 0.35))
            up = min(max(-ny * 0.5 + 0.5, 0.0), 1.0)
            # 画面の上が空、下が地面（照り返し）
            amb_r = (sky[0] * up + gnd[0] * (1 - up)) * ao
            amb_g = (sky[1] * up + gnd[1] * (1 - up)) * ao
            amb_b = (sky[2] * up + gnd[2] * (1 - up)) * ao
            cr = ar * (sun[0] * dif_sh + amb_r)
            cg = ag * (sun[1] * dif_sh + amb_g)
            cb = ab * (sun[2] * dif_sh + amb_b)
            # GGX のつや（視線は +z）
            hx, hy, hz = Lw[0], Lw[1], Lw[2] + 1.0
            hn = math.sqrt(hx * hx + hy * hy + hz * hz) + 1e-12
            ndh = max((nx * hx + ny * hy + nz * hz) / hn, 0.0)
            a2 = max(rough * rough, 0.002) ** 2
            dd = ndh * ndh * (a2 - 1) + 1
            D = a2 / (3.14159 * dd * dd)
            ndv = max(nz, 0.0)
            F = 0.04 + 0.96 * (1 - ndv) ** 5
            G = 1.0 / (1.0 + rough * 2.0)
            spv = spec * D * F * G * max(ndl, 0.0) * res * 0.25
            cr += sun[0] * spv
            cg += sun[1] * spv
            cb += sun[2] * spv
            # 上塗り（クリアコート）：空の映り込み（上を向く面ほど明るい）
            if coat > 0:
                ry = -ny * 2 * nz         # 反射の上向き成分（ざっくり）
                env_t = min(max(0.5 - 0.5 * ry, 0.0), 1.0)
                cr += coat * F * (sky[0] * env_t + gnd[0] * (1 - env_t)) * 1.6 * ao
                cg += coat * F * (sky[1] * env_t + gnd[1] * (1 - env_t)) * 1.6 * ao
                cb += coat * F * (sky[2] * env_t + gnd[2] * (1 - env_t)) * 1.6 * ao
            # リム（輪郭の光）：視線に横向きの面、光の側ほど強い
            if env.shape[0] > 15:                                   # リムの向き（画面の xy）を別に指定
                side = max(nx * env[15] + ny * env[16], 0.0) / (math.sqrt(env[15] ** 2 + env[16] ** 2) + 1e-6)
            else:
                side = max(nx * Lw[0] + ny * Lw[1], 0.0) / (math.sqrt(Lw[0] ** 2 + Lw[1] ** 2) + 1e-6)
            rmin = 0.08 if env.shape[0] > 15 else 0.45
            rr = (1 - ndv) ** 3 * (rmin + (1 - rmin) * side) * rimk * (0.6 + 0.4 * ao)
            # 布のつや（sheen）：浅い角度でうすく明るい
            sh = sheen * (1 - ndv) ** 2 * 0.5
            cr += rim[0] * rr + sky[0] * sh * ar
            cg += rim[1] * rr + sky[1] * sh * ag
            cb += rim[2] * rr + sky[2] * sh * ab
            out[j, i, 0] = cr
            out[j, i, 1] = cg
            out[j, i, 2] = cb
            out[j, i, 3] = (z * zs)


class Scene:
    """形を足していく。単位はローカル（呼ぶ側の都合でよい）。"""

    def __init__(self):
        self.P = []
        self.V = []

    def _add(self, row, bound, k, mat, op):
        r = np.zeros(22)
        r[:len(row)] = row
        r[9], r[10], r[11] = k, mat, 1.0 if op == 'sub' else 0.0
        r[18:22] = bound[0], bound[1], bound[2], bound[3] + k
        self.P.append(r)
        return self

    def cone(self, a, b, ra, rb=None, k=0.0, mat=0, op='add'):
        rb = ra if rb is None else rb
        a, b = np.asarray(a, float), np.asarray(b, float)
        c = (a + b) / 2
        return self._add([T_CONE, *a, *b, ra, rb], (*c, np.linalg.norm(b - a) / 2 + max(ra, rb)), k, mat, op)

    def cyl(self, a, b, r, rr=0.0, k=0.0, mat=0, op='add'):
        a, b = np.asarray(a, float), np.asarray(b, float)
        c = (a + b) / 2
        return self._add([T_CYL, *a, *b, r, rr], (*c, math.hypot(np.linalg.norm(b - a) / 2, r)), k, mat, op)

    def sphere(self, c, r, k=0.0, mat=0, op='add'):
        return self.ellipsoid(c, (r, r, r), k, mat, op)

    def ellipsoid(self, c, rr, k=0.0, mat=0, op='add'):
        return self._add([T_ELL, *c, *rr], (*c, max(rr)), k, mat, op)

    def rbox(self, c, hh, r, k=0.0, mat=0, op='add'):
        return self._add([T_RBOX, *c, *hh, r], (*c, float(np.linalg.norm(hh))), k, mat, op)

    def slab(self, poly, z, half, r, k=0.0, mat=0, fold=None, bend=0.0, op='add'):
        poly = np.asarray(poly, float)
        o = len(self.V)
        self.V.extend(poly.tolist())
        amp, per, ph, fx, fy = fold if fold is not None else (0.0, 1.0, 0.0, 1.0, 0.0)
        c = poly.mean(0)
        rad = float(np.max(np.hypot(*(poly - c).T)))
        row = np.zeros(18)
        row[:9] = [T_SLAB, o, len(poly), z, half, r, amp, per, ph]
        row[13:18] = [fx, fy, bend, c[0], c[1]]
        zb = abs(amp) + abs(bend) * rad * rad
        r_ = np.zeros(22)
        r_[:18] = row
        r_[9], r_[10], r_[11] = k, mat, 1.0 if op == 'sub' else 0.0
        r_[18:22] = c[0], c[1], z, math.hypot(rad, half + zb) + k
        self.P.append(r_)
        return self

    def arrays(self):
        P = np.array(self.P, np.float64) if self.P else np.zeros((0, 22))
        V = np.array(self.V, np.float64) if self.V else np.zeros((1, 2))
        return P, V

    def bounds(self):
        P, _ = self.arrays()
        b = P[:, 18:22]
        return (b[:, 0] - b[:, 3]).min(), (b[:, 1] - b[:, 3]).min(), (b[:, 0] + b[:, 3]).max(), \
            (b[:, 1] + b[:, 3]).max(), (b[:, 2] - b[:, 3]).min(), (b[:, 2] + b[:, 3]).max()


def Mat(rgb, spec=0.3, rough=0.4, sss=0.3, rim=1.0, coat=0.0, sheen=0.0):
    c = np.asarray(rgb, np.float32)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return np.array([c[0], c[1], c[2], spec, rough, sss, rim, coat, sheen, 0, 0, 0], np.float64)


def _lin(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


DEFAULT_ENV = dict(sun=np.array([1.0, 0.99, 0.975]) * 1.25, sky=_lin(core.c('B8D8F0')) * 0.55,
                   ground=_lin(core.c('6F8FAA')) * 0.38, rim=_lin(core.c('DDF0FF')) * 0.85)


def render(scene, mats, w, h, A, c, zs=1.0, ss=3, light=(-0.6, -0.5, 0.45), env=None, shadow_k=6.0, ao_d=None,
           eps=None, linear=False):
    """scene を画面の板（w×h px）に。A：ローカル → 画面の 2×2（px／ローカル単位）、c：ローカルの原点の板の中の位置（px）、
    zs：ローカルの z → 画面の px（ふつうは sqrt|det A|）。ss：画素あたりのレイの数（縦横）。
    戻り値：(rgba (h, w, 4) ストレートアルファ sRGB, z (h, w) 手前ほど大きい px、当たっていない所は -inf)。"""
    P, V = scene.arrays()
    env = dict(DEFAULT_ENV, **(env or {}))
    A = np.asarray(A, np.float64)
    Ai = np.linalg.inv(A * ss)
    det = math.sqrt(abs(np.linalg.det(A)))
    L = np.asarray(light, np.float64)
    L = L / np.linalg.norm(L)
    # ローカルの光の向き：xy は A^-1、z は 1/zs
    Lxy = np.linalg.inv(A) @ L[:2]
    Ll = np.array([Lxy[0], Lxy[1], L[2] / zs])
    Ll /= np.linalg.norm(Ll)
    x0, y0, x1, y1, z0, z1 = scene.bounds()
    sz = 1.0 / det                                    # 1px のローカルの大きさ
    eps = (0.25 * sz / ss) if eps is None else eps
    ao_d = (max(x1 - x0, y1 - y0) * 0.035) if ao_d is None else ao_d
    hw, hh = w * ss, h * ss
    pos = np.zeros((hh, hw, 3), np.float64)
    mm = np.zeros((hh, hw), np.float64)
    hit = np.zeros((hh, hw), np.uint8)
    _march(P, V, hw, hh, Ai, c[0] * ss, c[1] * ss, zs, z0 - 1.0, z1 + 1.0, eps, pos, mm, hit)
    MATS = np.stack([mats[k] for k in range(len(mats))], 0) if isinstance(mats, (list, tuple)) else \
        np.stack([mats[k] for k in sorted(mats)], 0)
    envv = np.concatenate([np.asarray(env['sun'], np.float64), np.asarray(env['sky'], np.float64),
                           np.asarray(env['ground'], np.float64), np.asarray(env['rim'], np.float64), L] +
                          ([np.asarray(env['rim_dir'], np.float64)[:2]] if 'rim_dir' in env else []))
    out = np.zeros((hh, hw, 4), np.float64)
    AiT = np.linalg.inv(A).T
    _shade(P, V, hw, hh, pos, mm, hit, AiT, zs, Ll, MATS, envv, shadow_k, ao_d, eps, out)
    a = hit.astype(np.float32)
    rgb = out[..., :3].astype(np.float32)
    if not linear:
        rgb = np.clip(rgb, 0, None)
        rgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1 / 2.4) - 0.055).astype(np.float32)
    zbuf = np.where(hit > 0, out[..., 3], -1e9).astype(np.float32)
    if ss > 1:
        prem = np.concatenate([rgb * a[..., None], a[..., None]], 2)
        small = cv2.resize(prem, (w, h), interpolation=cv2.INTER_AREA)
        al = small[..., 3:4]
        rgba = np.concatenate([small[..., :3] / np.maximum(al, 1e-5), al], 2)
        zbuf = cv2.resize(np.where(hit > 0, zbuf, z0 * zs).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        zbuf[al[..., 0] < 0.02] = -1e9
        return rgba.astype(np.float32), zbuf
    return np.concatenate([rgb, a[..., None]], 2).astype(np.float32), zbuf
