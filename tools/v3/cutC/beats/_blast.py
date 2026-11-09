"""案C 03〜05 の「一撃の迫力」の部品：飛ぶ土・岩・芝のかたまり（3D の放物線、回転、サブフレームのぶれ、左上からの光）と、
色をつけた砂煙の板。どれも sRGB のフレームに、奥から順に重ねる（depth_sort）。

  Chunks(...)          かたまりの群れ。items(frame 用の (奥行き, 描く関数) のリスト) を返す。
  DustLayer            砂煙の板（_fx の 6 種）に土の色をつけて、半分の解像度の層に描く。
  draw_sorted(f, its)  (奥行き, 種類, fn) のリストを奥から描く。

光は左上（画面）から：かたまりの左上が明るく、右下が暗い。白く光らせない（明るい面でも sRGB 0.8 以下）。
"""
import math

import cv2
import numba as _nb
import numpy as np

import core
from core import W, H
from cutC.beats import _fx as FX

LDIR = np.array([-0.62, -0.78])
TINY = 3.0                                        # これより小さい（半径 px）かたまりは線で描く                   # 画面での光の来る向き（左上）
# かたまりの色（sRGB）：0 土くれ（こげ茶）、1 岩（灰）、2 芝（上：緑／裏：土）、3 粘土（明るい土）
COL = {0: core.c('4A3D31'), 1: core.c('77736C'), 2: core.c('3F6B3A'), 3: core.c('7A6650')}
TURF_UNDER = core.c('4E3F31')
TINTS = {'soil': np.array([1.0, 0.93, 0.84], np.float32), 'dust': np.array([1.0, 0.97, 0.93], np.float32),
         'grey': np.array([1.0, 1.0, 1.0], np.float32), 'dark': np.array([0.80, 0.74, 0.67], np.float32)}
_RS = {}


@_nb.njit(fastmath=True, cache=True)
def _over(L, A, spr, ix, iy, alpha, t0, t1, t2):
    """前乗せの層 (L, A) に、ストレートアルファの板 spr を色 (t0,t1,t2) 倍で「上に」重ねる。"""
    h, w = spr.shape[0], spr.shape[1]
    fh, fw = L.shape[0], L.shape[1]
    ya, yb = max(iy, 0), min(iy + h, fh)
    xa, xb = max(ix, 0), min(ix + w, fw)
    al = np.float32(alpha)
    c0, c1, c2 = np.float32(t0), np.float32(t1), np.float32(t2)
    for y in range(ya, yb):        # 並列にしない（小さな板が多く、スレッドの立ち上げのほうが重い）
        sr = spr[y - iy]
        lr = L[y]
        ar = A[y]
        for x in range(xa, xb):
            sx = x - ix
            a = sr[sx, 3] * al
            k = 1 - a
            lr[x, 0] = lr[x, 0] * k + sr[sx, 0] * c0 * a
            lr[x, 1] = lr[x, 1] * k + sr[sx, 1] * c1 * a
            lr[x, 2] = lr[x, 2] * k + sr[sx, 2] * c2 * a
            ar[x] = ar[x] * k + a


class DustLayer:
    """砂煙だけを半分の解像度の前乗せの層に描いて、最後にまとめて重ねる（やわらかいので半分で十分。重ね描きが速い）。"""
    S = 2

    def __init__(self):
        self.L = np.zeros((H // self.S, W // self.S, 3), np.float32)
        self.A = np.zeros((H // self.S, W // self.S), np.float32)
        self.n = 0

    def puff(self, k, x, y, size, alpha=1.0, tint='soil', dark=1.0):
        if size < 3 or alpha <= 0.004:
            return
        size = min(size, 1700.0)
        if x + size < -60 or x - size > W + 60 or y + size < -60 or y - size > H + 60:
            return
        s = self.S
        spr = FX.dust_sprites()[k % 6]
        q = int(round(math.log(max(size / s, 2.0)) * 24))
        key = (k % 6, q)
        r = _RS.get(key)
        if r is None:
            if _RS.get('_bytes', 0) > 6e8:
                _RS.clear()
            ss = math.exp(q / 24.0) / spr.shape[1]
            r = np.ascontiguousarray(cv2.resize(spr, None, fx=ss, fy=ss,
                                                interpolation=cv2.INTER_AREA if ss < 1 else cv2.INTER_LINEAR), np.float32)
            _RS[key] = r
            _RS['_bytes'] = _RS.get('_bytes', 0) + r.nbytes
        t = TINTS[tint] * dark
        _over(self.L, self.A, r, int(math.floor(x / s - r.shape[1] / 2)), int(math.floor(y / s - r.shape[0] * 0.55)),
              float(alpha), float(t[0]), float(t[1]), float(t[2]))
        self.n += 1

    def composite(self, frame):
        if self.n == 0:
            return frame
        A = cv2.resize(self.A, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
        L = cv2.resize(self.L, (W, H), interpolation=cv2.INTER_LINEAR)
        return frame * (1 - A) + L


def draw_sorted(frame, items, split_z=None):
    """items：(奥行き, 'd'（砂煙：fn(layer)）| 'c'（かたまり：fn(frame)）, fn)。砂煙は 1 枚の層にまとめて奥から描き、
    かたまりは split_z より奥のものを層の前に、手前のものを層のあとに（だいたいの前後関係）。"""
    lay = DustLayer()
    its = sorted(items, key=lambda q: -q[0])
    for z, kind, fn in its:
        if kind == 'd':
            fn(lay)
        elif split_z is not None and z > split_z:
            frame = fn(frame)
    frame = lay.composite(frame)
    for z, kind, fn in its:
        if kind == 'c' and (split_z is None or z <= split_z):
            frame = fn(frame)
    return frame


def _unit_shapes(rng, n, nv=10):
    ang = np.linspace(0, 2 * np.pi, nv, endpoint=False)[None] + rng.uniform(-0.25, 0.25, (n, nv))
    rr = rng.uniform(0.74, 1.0, (n, nv)) * (1 + 0.18 * np.cos(ang * 2 + rng.uniform(0, 6.3, (n, 1))))
    return np.stack([np.cos(ang) * rr, np.sin(ang) * rr], -1)


class Chunks:
    """n 個のかたまり。P0：飛び出す所 (n,3)、V：初速 (n,3)（m/フレーム）、T0：飛び出すフレーム、size：大きさ（m、半径）、
    kind：0 土 1 岩 2 芝 3 粘土。g：重力（m/フレーム²）。地面 y=0 に落ちたら消える（land_time で砂ぼこりを立てる側が使う）。"""

    def __init__(self, P0, V, T0, size, kind, seed=0, g=0.14, spin=(0.05, 0.45)):
        rng = np.random.default_rng(seed)
        self.P0, self.V = np.asarray(P0, float), np.asarray(V, float)
        self.T0, self.size, self.kind = np.asarray(T0, float), np.asarray(size, float), np.asarray(kind, int)
        n = len(self.T0)
        self.n, self.g = n, g
        self.shape = _unit_shapes(rng, n)
        self.spin = rng.uniform(*spin, n) * rng.choice([-1, 1], n)
        self.ph = rng.uniform(0, 2 * np.pi, n)
        self.flip = rng.uniform(0.18, 0.55, n) * rng.choice([-1, 1], n)
        self.fph = rng.uniform(0, 2 * np.pi, n)
        self.tone = rng.uniform(0.80, 1.18, n)
        self.aspect = np.where(self.kind == 2, rng.uniform(0.35, 0.55, n), rng.uniform(0.65, 1.0, n))
        vy, y0 = self.V[:, 1], self.P0[:, 1]
        self.t_land = self.T0 + (vy + np.sqrt(np.maximum(vy * vy + 2 * g * np.maximum(y0, 0), 0))) / g

    def pos(self, t):
        tau = np.clip(np.asarray(t, float) - self.T0, 0, None)
        p = self.P0 + self.V * tau[..., None]
        p[..., 1] -= 0.5 * self.g * tau * tau
        return p, tau

    def items(self, t, cam, M2=None, shutter=0.55, near=0.6, alpha=1.0, blur_fn=None, scale=1.0, max_px=260.0,
              fade_fn=None):
        """時刻 t（小数のフレーム）の、描く物のリスト [(奥行き, fn)]。"""
        ns = 4
        ts = t - shutter * np.linspace(1.0, 0.0, ns)
        live = (t > self.T0) & (t < self.t_land)
        idx = np.nonzero(live)[0]
        if len(idx) == 0:
            return []
        P = np.stack([self.pos(np.full(self.n, tt))[0][idx] for tt in ts], 0)        # (ns, m, 3)
        P[..., 1] = np.maximum(P[..., 1], 0.0)
        S, z = cam.project(P.reshape(-1, 3))
        S, z = S.reshape(ns, -1, 2), z.reshape(ns, -1)
        if M2 is not None:
            S = S @ M2[:, :2].T + M2[:, 2]
            sc = float(M2[0, 0])
        else:
            sc = 1.0
        out = []
        zl = z[-1]
        rpx = self.size[idx] * cam.fpx * sc * scale / np.maximum(zl, 1e-3)
        tiny = []
        for m, j in enumerate(idx):
            if zl[m] < near or rpx[m] > max_px:
                continue
            c = S[-1, m]
            ext = rpx[m] + np.abs(S[0, m] - c).max() + 4
            if c[0] + ext < -20 or c[0] - ext > W + 20 or c[1] + ext < -20 or c[1] - ext > H + 20:
                continue
            a = alpha if fade_fn is None else alpha * fade_fn(self.P0[j], P[-1, m], t - self.T0[j])
            if a <= 0.01:
                continue
            sig = blur_fn(zl[m]) if blur_fn is not None else 0.0
            if rpx[m] < TINY and sig < 1.0:
                tiny.append((float(zl[m]), j, S[0, m], S[-1, m], float(rpx[m]), a))
                continue
            out.append((float(zl[m]), 'c', self._drawer(j, S[:, m], ts, float(rpx[m]), a, sig)))
        if tiny:                                           # 小さいものは線にして、奥・手前の 2 つの束でまとめて描く
            tiny.sort(key=lambda q: -q[0])
            zs = [q[0] for q in tiny]
            zm = float(np.median(zs))
            for grp in ([q for q in tiny if q[0] >= zm], [q for q in tiny if q[0] < zm]):
                if grp:
                    out.append((float(np.mean([q[0] for q in grp])), 'c', self._liner(grp)))
        return out

    def _liner(self, grp):
        def fn(frame):
            lay = np.zeros((H, W, 4), np.uint8)
            for z, j, p0, p1, r, a in grp:
                kind = self.kind[j]
                ln = max(1.0, float(np.hypot(*(p1 - p0))))
                ak = a * min(1.0, 2.4 * r / ln + 0.45)
                col = COL[kind if kind != 2 else (0 if j % 3 else 2)] * self.tone[j] * 0.9 * 255 * ak
                cv2.line(lay, (int(p0[0] * 8), int(p0[1] * 8)), (int(p1[0] * 8), int(p1[1] * 8)),
                         (float(col[0]), float(col[1]), float(col[2]), 255.0 * ak), max(1, int(round(r * 2))),
                         cv2.LINE_AA, 3)
            A = lay[..., 3:].astype(np.float32) / 255.0
            return frame * (1 - A) + lay[..., :3].astype(np.float32) / 255.0
        return fn

    def _drawer(self, j, Ss, ts, r, a, sig):
        def fn(frame):
            return self._draw(frame, j, Ss, ts, r, a, sig)
        return fn

    def _draw(self, frame, j, Ss, ts, r, a, sig):
        kind = self.kind[j]
        tone = self.tone[j]
        if r < 1.1:                                       # 小さい：線（ぶれ）だけ
            p0, p1 = Ss[0], Ss[-1]
            x0, y0 = int(min(p0[0], p1[0]) - 3), int(min(p0[1], p1[1]) - 3)
            x1, y1 = int(max(p0[0], p1[0]) + 4), int(max(p0[1], p1[1]) + 4)
            xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
            if xb <= xa or yb <= ya:
                return frame
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.line(m, (int((p0[0] - x0) * 8), int((p0[1] - y0) * 8)), (int((p1[0] - x0) * 8), int((p1[1] - y0) * 8)),
                     255, max(1, int(round(r * 2))), cv2.LINE_AA, 3)
            ln = max(1.0, float(np.hypot(*(p1 - p0))))
            al = (m[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32) / 255.0 * a * min(1.0, 2.2 * r / ln + 0.35))[..., None]
            col = COL[kind if kind != 2 else 0] * tone * 0.9
            frame[ya:yb, xa:xb] = frame[ya:yb, xa:xb] * (1 - al) + col * al
            return frame
        # 形：各サンプルの回転・反転した多角形の凸包
        sh = self.shape[j]
        verts = []
        last = None
        for s_, tt in zip(Ss, ts):
            ang = self.ph[j] + self.spin[j] * tt
            ca, sa = math.cos(ang), math.sin(ang)
            fy = self.aspect[j]
            if kind == 2:
                fy = max(0.18, abs(math.cos(self.fph[j] + self.flip[j] * tt))) * 0.75
            v = sh * np.array([1.0, fy])
            v = v @ np.array([[ca, sa], [-sa, ca]]) * r + s_
            verts.append(v)
            last = v
        allv = np.concatenate(verts, 0)
        pad = int(math.ceil(2 + 2.5 * sig))
        x0, y0 = int(math.floor(allv[:, 0].min())) - pad, int(math.floor(allv[:, 1].min())) - pad
        x1, y1 = int(math.ceil(allv[:, 0].max())) + pad + 1, int(math.ceil(allv[:, 1].max())) + pad + 1
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if xb <= xa or yb <= ya:
            return frame
        hull = cv2.convexHull(((allv - (x0, y0)) * 8).astype(np.int32))
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [hull], 255, cv2.LINE_AA, 3)
        area_s = max(1.0, cv2.contourArea(last.astype(np.float32)))
        area_h = max(1.0, cv2.contourArea(hull.astype(np.float32)) / 64.0)
        k_blur = min(1.0, max(0.55, (area_s / area_h) ** 0.55 * 1.15))
        al = m.astype(np.float32) / 255.0 * (a * k_blur)
        # 色と光：左上ほど明るい（最後の位置の中心から）
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        c = Ss[-1]
        ramp = ((xx - c[0]) * LDIR[0] + (yy - c[1]) * LDIR[1]) / max(r, 1.0)
        lit = np.clip(0.78 + 0.55 * ramp, 0.40, 1.35)
        if kind == 2:
            top = math.cos(self.fph[j] + self.flip[j] * ts[-1]) > 0
            base = COL[2] if top else TURF_UNDER
        else:
            base = COL[kind]
        col = base * tone * lit[..., None]
        if r > 2.5:                                       # 縁を暗く（丸みと厚み。芝は縁に土の層）
            e = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1 + int(r > 9))
            edge = ((m.astype(np.float32) - e.astype(np.float32)) / 255.0)[..., None]
            ec = TURF_UNDER * 0.75 if kind == 2 else base * 0.55
            col = col * (1 - 0.75 * edge) + ec * tone * 0.75 * edge
        col = np.clip(col, 0, 0.82)
        if sig > 0.6:                                     # ピントの外（近すぎる）：前乗せでぼかす
            prem = cv2.GaussianBlur(np.ascontiguousarray(col * al[..., None]), (0, 0), sig)
            al = cv2.GaussianBlur(al, (0, 0), sig)
            col = prem / np.maximum(al[..., None], 1e-4)
        al = al[ya - y0:yb - y0, xa - x0:xb - x0, None]
        col = col[ya - y0:yb - y0, xa - x0:xb - x0]
        frame[ya:yb, xa:xb] = frame[ya:yb, xa:xb] * (1 - al) + col * al
        return frame
