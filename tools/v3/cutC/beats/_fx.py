"""案C 03〜05 の効果：衝撃波の屈折、砂煙の板、飛ぶ土くれ、湯気のような飛行機雲。どれも sRGB のフレームに重ねる。
光らせない：白い面積の大きいものは α を低く（凝結の殻 ≤0.14）、明るさは数フレームかけて変える。"""
import math

import cv2
import numpy as np

import core
from core import W, H
from cutB.beats import _render as R, _beach as BE, _motion as M

_C = {}
DUST_TINT = np.array([0.93, 0.95, 0.98], np.float32)


def put(frame, rgba, x0, y0, alpha=1.0):
    """ストレートアルファの板を小数の位置に。"""
    if alpha <= 0.003:
        return frame
    ix, iy = int(math.floor(x0)), int(math.floor(y0))
    h, w = rgba.shape[:2]
    xa, ya, xb, yb = max(ix, 0), max(iy, 0), min(ix + w, frame.shape[1]), min(iy + h, frame.shape[0])
    if xb <= xa or yb <= ya:
        return frame
    sub = rgba[ya - iy:yb - iy, xa - ix:xb - ix]
    a = sub[..., 3:4] * alpha
    frame[ya:yb, xa:xb] = frame[ya:yb, xa:xb] * (1 - a) + sub[..., :3] * a
    return frame


# ---------------------------------------------------------------- 衝撃波の屈折
def refract(frame, cx, cy, r, amp=10.0, width=16.0, shell_a=0.0, ry_k=1.0, ground_y=None):
    """中心 (cx, cy)・半径 r（縦は ry_k 倍）の殻で画面を外へずらす（レンズのような屈折）。shell_a：凝結の殻の白さ（≤0.14）。
    ground_y：これより下は殻を描かない（地面の下）。"""
    if r < 2 or (amp < 0.2 and shell_a < 0.003):
        return frame
    pad = r + 3 * width + amp + 4
    x0, x1 = int(max(0, cx - pad)), int(min(W, cx + pad))
    y0, y1 = int(max(0, cy - pad * ry_k)), int(min(H, cy + pad * ry_k))
    if x1 <= x0 or y1 <= y0:
        return frame
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - cx, (yy - cy) / ry_k
    rr = np.sqrt(dx * dx + dy * dy) + 1e-3
    q = (rr - r) / width
    prof = np.exp(-q * q) * (q * -1.6 + 0.4)          # 縁の内側で外へ・外側で内へ（なめらかな S）
    d = amp * prof
    ux, uy = dx / rr, dy / rr * ry_k
    mx = (xx - ux * d).astype(np.float32)
    my = (yy - uy * d).astype(np.float32)
    sub = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if shell_a > 0:
        s = shell_a * (np.exp(-q * q * 0.6) * 0.8 + 0.25 * (rr < r))
        if ground_y is not None:
            s = s * R.smoothstep(ground_y + 4, ground_y - 10, yy)
        sub = sub + (core.c('EEF4F8') - sub) * s[..., None]
    frame[y0:y1, x0:x1] = sub
    return frame


# ---------------------------------------------------------------- 砂煙の板
def dust_sprites():
    """砂煙の板（6 種、sRGB ストレート、256px）：やわらかいガウスのかたまりの和を fBm でけずり、左上から照らす
    （上が明るい青灰、下と奥が暗い）。縁はうすく透ける。"""
    if 'dust' not in _C:
        out = []
        n = 256
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
        for k in range(6):
            rng = np.random.default_rng(500 + k)
            acc = np.zeros((n, n), np.float32)
            for j in range(9):
                cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.35, 0.68)
                r = rng.uniform(0.10, 0.20)
                acc += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) * rng.uniform(0.6, 1.0)
            nz = R.fbm(xx * 5 + k * 3.1, yy * 5, 5, gain=0.55, seed=40 + k)
            d = acc / acc.max() + 0.32 * nz
            a = R.smoothstep(0.30, 0.85, d) * R.smoothstep(0.0, 0.10, np.minimum.reduce([xx, yy, 1 - xx, 1 - yy]))
            hb = cv2.GaussianBlur(d, (0, 0), 6)
            gx = cv2.Sobel(hb, cv2.CV_32F, 1, 0, ksize=5)
            gy = cv2.Sobel(hb, cv2.CV_32F, 0, 1, ksize=5)
            lit = np.clip(0.5 + (gx * 0.6 + gy * 0.8) * -40.0, 0, 1)          # 左上が明るい
            top = np.clip(1.2 - yy * 1.2, 0, 1)
            kk = np.clip(0.25 + 0.55 * lit * top + 0.2 * (nz * 0.5 + 0.5), 0, 1)[..., None]
            col = core.c('6E7780') * (1 - kk) + core.c('C9CFD3') * kk
            out.append(np.dstack([col, a * 0.92]).astype(np.float32))
        _C['dust'] = out
    return _C['dust']


_RS = {}


def put_puff(frame, k, x, y, size, alpha=1.0, dark=1.0, rot=0.0):
    """砂煙の板 k を中心 (x, y)・幅 size px で。dark：明るさの倍率。"""
    if size < 3 or alpha <= 0.003:
        return frame
    size = min(size, 1500.0)
    if x + size < -50 or x - size > W + 50 or y + size < -50 or y - size > H + 50:
        return frame
    spr = dust_sprites()[k % 6]
    s = size / spr.shape[1]
    q = int(round(math.log(max(size, 4.0)) * 24))           # 大きさは約 4% きざみ（キャッシュが増えすぎない）
    key = (k % 6, q)
    if key not in _RS:
        if _RS.get('_bytes', 0) > 6e8:
            _RS.clear()
        ss = math.exp(q / 24.0) / spr.shape[1]
        _RS[key] = cv2.resize(spr, None, fx=ss, fy=ss, interpolation=cv2.INTER_AREA if ss < 1 else cv2.INTER_LINEAR)
        _RS['_bytes'] = _RS.get('_bytes', 0) + _RS[key].nbytes
    r = _RS[key]
    if abs(dark - 1) > 1e-3:
        r = r.copy()
        r[..., :3] *= dark
    return put(frame, r, x - r.shape[1] / 2, y - r.shape[0] * 0.55, alpha)


# ---------------------------------------------------------------- 飛ぶ土くれ（3D）
class Clods:
    """原点から飛ぶ土くれ（n 個）。v：初速（m/フレーム）、仰角 el_range（度）。cam で投影し、前のフレームからの線（ぶれ）で描く。"""

    def __init__(self, n=260, f0=0, origin=(0.0, 0.0, 0.0), speed=(0.25, 1.1), el_range=(25, 78), seed=5, g=0.032,
                 size=(0.04, 0.22)):
        rng = np.random.default_rng(seed)
        self.f0, self.o = f0, np.asarray(origin, float)
        az = rng.uniform(0, 2 * np.pi, n)
        el = np.radians(rng.uniform(*el_range, n))
        sp = rng.uniform(*speed, n) * rng.uniform(0.6, 1.0, n) ** 2 * 1.4
        self.v = np.stack([np.cos(az) * np.cos(el), np.sin(el), np.sin(az) * np.cos(el)], 1) * sp[:, None]
        self.size = rng.uniform(*size, n) * (1.3 - 0.5 * sp / sp.max())
        self.g = g
        self.tone = rng.uniform(0.75, 1.15, n)
        self.dt0 = rng.uniform(0, 2.0, n)

    def pos(self, t):
        tt = np.clip(t - self.dt0, 0, None)[:, None]
        p = self.o + self.v * tt * (1 - 0.006 * tt) + np.array([0, -0.5 * self.g, 0]) * tt * tt
        land = p[:, 1] < 0
        p[:, 1] = np.maximum(p[:, 1], 0)
        return p, land, tt[:, 0]

    def draw(self, frame, cam, i, alpha=1.0, M2=None):
        t = i - self.f0
        if t <= 0:
            return frame
        p1, land, tt = self.pos(t)
        p0, _, _ = self.pos(t - 0.3)
        P1, z1 = cam.project(p1)
        P0, z0 = cam.project(p0)
        if M2 is not None:
            P1 = P1 @ M2[:, :2].T + M2[:, 2]
            P0 = P0 @ M2[:, :2].T + M2[:, 2]
        ok = (z1 > 0.5) & (tt > 0) & ~land
        if not ok.any():
            return frame
        sel = np.nonzero(ok & (P1[:, 0] > -100) & (P1[:, 0] < W + 100) & (P1[:, 1] > -100) & (P1[:, 1] < H + 100))[0]
        if len(sel) == 0:
            return frame
        bx0 = int(max(0, min(P1[sel, 0].min(), P0[sel, 0].min()) - 40))
        by0 = int(max(0, min(P1[sel, 1].min(), P0[sel, 1].min()) - 40))
        bx1 = int(min(W, max(P1[sel, 0].max(), P0[sel, 0].max()) + 40))
        by1 = int(min(H, max(P1[sel, 1].max(), P0[sel, 1].max()) + 40))
        if bx1 <= bx0 or by1 <= by0:
            return frame
        lay = np.zeros((by1 - by0, bx1 - bx0), np.float32)
        hi = np.zeros_like(lay)
        for j in sel[np.argsort(-z1[sel])]:
            w = max(0.9, self.size[j] * cam.fpx / z1[j])
            if w > 50:
                continue
            a0 = (int((P0[j, 0] - bx0) * 8), int((P0[j, 1] - by0) * 8))
            a1 = (int((P1[j, 0] - bx0) * 8), int((P1[j, 1] - by0) * 8))
            v = float(min(1.0, w / 1.5)) * self.tone[j] * 0.85
            cv2.line(lay, a0, a1, v * 0.6, max(1, int(w * 0.7)), cv2.LINE_AA, 3)
            cv2.circle(lay, a1, max(4, int(w * 0.55 * 8)), v, -1, cv2.LINE_AA, 3)
            if w > 3:
                cv2.circle(hi, (a1[0] - int(w * 2), a1[1] - int(w * 2)), max(3, int(w * 0.25 * 8)), 0.5, -1, cv2.LINE_AA, 3)
        a = np.clip(lay, 0, 1)[..., None] * alpha
        hm = np.clip(hi, 0, 1)[..., None]
        col = core.c('4E4A46') * (1 - hm) + core.c('8A857E') * hm
        frame[by0:by1, bx0:bx1] = frame[by0:by1, bx0:bx1] * (1 - a) + col * a
        return frame


# ---------------------------------------------------------------- 飛ぶ悪役のうしろの白い筋（湯気のような空気の筋）
def vapor(frame, p0, p1, w0, w1, alpha, blur=3.0):
    """p0→p1 の先細りの白い筋（ぼかし）。"""
    if alpha <= 0.003:
        return frame
    x0 = int(max(0, min(p0[0], p1[0]) - w0 * 2 - 20))
    y0 = int(max(0, min(p0[1], p1[1]) - w0 * 2 - 20))
    x1 = int(min(W, max(p0[0], p1[0]) + w0 * 2 + 20))
    y1 = int(min(H, max(p0[1], p1[1]) + w0 * 2 + 20))
    if x1 <= x0 or y1 <= y0:
        return frame
    L = core.SS(s=2, w=x1 - x0, h=y1 - y0)
    d = np.array(p1, float) - p0
    ln = np.hypot(*d) + 1e-6
    n = np.array([-d[1], d[0]]) / ln
    a0, a1 = np.array(p0) - (x0, y0), np.array(p1) - (x0, y0)
    pts = [a0 + n * w0 / 2, a1 + n * w1 / 2, a1 - n * w1 / 2, a0 - n * w0 / 2]
    L.draw.polygon(L.p([tuple(q) for q in pts]), fill=L.col(core.c('E9F0F5'), 1.0))
    lay = L.done()
    prem = M._to_prem(lay)
    prem = cv2.GaussianBlur(prem, (0, 0), blur)
    lay = M._from_prem(prem)
    return put(frame, lay, x0, y0, alpha)
