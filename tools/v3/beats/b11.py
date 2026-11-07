"""b11（ナレーションなし・約1.2秒）CG → 実写の見せ場。

b10 の最後（濃紺に、土佐湾の細い線画がごくうすく）から始まる。
 0–12：線画が濃くなり、仁淀川（ACC）が上流から河口まで引かれ、河口に小さな点。カメラは真上から河口へ寄りながら、
        向きを海の側（南南東）へ回し始める。
12–35：カメラが河口へ「降りる」。真上の視点から起き上がって、低い横向き（5倍望遠＝f 6400px）の視点に。
        地図は1枚の地面として本当に透視投影している（線はベクトルのまま投影）ので、沖の等距離線は
        水平線へ吸い込まれていき、海岸線はそのまま b12 の1フレーム目の波打ち際に重なる。
        近くの線（波頭・泡の上下・波打ち際・沖のうねり）は、b12 の1フレーム目から取り出した輪郭を、
        最後のカメラで地面に逆投影したもの。カメラが下がるにつれて下から上がってきて、最後のフレームで
        実写の輪郭にぴったり重なる（同じカメラで投影し直すので画素単位で一致する）。
        背景は濃紺から、実写の1フレーム目の行ごとの平均色（空→海→泡→砂）へ。カメラの角度に合わせて
        行を対応させるので、水平線が上から降りてきて、そこから上が空の色になる。明るさは単調に上がるだけ。
"""
import os

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

import core
from core import W, H
from beats import _geo as G, _tosa, _real, b10

PX, PY = W / 2, H / 2                      # 主点（画面の中心）
F1 = 6400.0                                # 5倍望遠（120mm 相当）の焦点距離（縦 1920px に対して）
H1 = 0.046                                 # 最後のカメラの高さ（地図の px。1px≈217m なので約10m）
MOUTH_LL = (33.465, 133.43)                # 仁淀川の河口（指定の値）。地図の海岸線（NE 50m）に寄せて使う

# 仁淀川（上流→河口）。おおよその流れを手で置いたもの（河口からの相対で、地図の海岸線に合わせて平行移動する）
RIVER_LL = [(33.700, 132.960), (33.655, 133.020), (33.635, 133.100), (33.600, 133.170), (33.562, 133.215), (33.530, 133.268), (33.545, 133.320),
            (33.533, 133.370), (33.552, 133.405), (33.540, 133.432), (33.518, 133.442), (33.498, 133.444),
            (33.480, 133.440)]

_S = {}


# ---------------------------------------------------------------- 地図（地面）の線
def _world(lat, lon):
    x, y = _tosa.to_screen(lat, lon)
    return np.stack([np.asarray(x, np.float64), np.asarray(y, np.float64)], -1)


def _densify(P, step):
    out = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        t = np.arange(1, n + 1)[:, None] / n
        out.append(a + (b - a) * t)
    return np.concatenate(out)


def _coast_rings():
    out = []
    for r in G.rings():
        if r[:, 0].max() < 128 or r[:, 0].min() > 139 or r[:, 1].max() < 29 or r[:, 1].min() > 37:
            continue
        out.append(_world(r[:, 1], r[:, 0]))
    return out


def _mouth(coast):
    """指定の河口の点に最も近い、地図の海岸線上の点と、その場所の海岸線の向き（単位ベクトル）。"""
    q = _world(*MOUTH_LL)
    best = (1e18, None, None)
    for P in coast:
        A, B = P[:-1], P[1:]
        AB = B - A
        L2 = (AB ** 2).sum(1) + 1e-12
        t = (((q - A) * AB).sum(1) / L2).clip(0, 1)
        C = A + AB * t[:, None]
        d = ((C - q) ** 2).sum(1)
        k = int(np.argmin(d))
        if d[k] < best[0]:
            best = (d[k], C[k], AB[k] / np.sqrt(L2[k]))
    return best[1], best[2]


def _offsets():
    """_tosa と同じ作り方（陸からの距離のなめらかな等値線）を、2倍の解像度でベクトルの折れ線にしたもの。"""
    path = os.path.join(G.CACHE, 'b11_offsets_v2.npz')
    if os.path.exists(path):
        z = np.load(path)
        pts, idx, lev = z['pts'], z['idx'], z['lev']
        return [(lev[i], pts[idx[i]:idx[i + 1]]) for i in range(len(idx) - 1)]
    S, pad = 2, 400
    land = np.zeros(((H + 2 * pad) * S, (W + 2 * pad) * S), np.uint8)
    polys = []
    for P in _coast_rings():
        polys.append(np.round((P + pad) * S * 16).astype(np.int32))
    cv2.fillPoly(land, polys, 255, lineType=cv2.LINE_AA, shift=4)
    dist = cv2.distanceTransform((land < 128).astype(np.uint8), cv2.DIST_L2, 5) / S
    dist = cv2.GaussianBlur(dist, (0, 0), 6 * S)
    out = []
    for k, d in enumerate(_tosa.STEPS):
        m = (dist < d).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            if len(c) < 20:
                continue
            p = c[:, 0, :].astype(np.float64)
            # 1px の階段を均す（輪郭に沿った移動平均、閉じた線）
            kk = 7
            pp = np.concatenate([p[-kk:], p, p[:kk]])
            ker = np.ones(2 * kk + 1) / (2 * kk + 1)
            p = np.stack([np.convolve(pp[:, j], ker, 'valid') for j in range(2)], 1)
            p = cv2.approxPolyDP(p.astype(np.float32)[:, None, :], 0.6, True)[:, 0, :].astype(np.float64)
            p = np.concatenate([p, p[:1]]) / S - pad
            out.append((k, p))
    idx = np.cumsum([0] + [len(p) for _, p in out])
    np.savez_compressed(path, pts=np.concatenate([p for _, p in out]), idx=idx, lev=np.array([k for k, _ in out]))
    return out


def _river(mouth):
    ll = np.array(RIVER_LL)
    P = _world(ll[:, 0], ll[:, 1])
    P = P + (mouth - P[-1])                         # 河口を地図の海岸線の点へ
    # なめらかに（Catmull-Rom）
    Q = np.concatenate([P[:1], P, P[-1:]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        t = np.linspace(0, 1, 12, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 +
                          (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1:])
    R = np.concatenate(out)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(R, axis=0).T))])
    return R, s / s[-1]


# ---------------------------------------------------------------- カメラ
class Cam:
    def __init__(self, C, phi, beta, f):
        self.C, self.f = np.asarray(C, np.float64), float(f)
        self.phi, self.beta = phi, beta
        p, b = np.radians(phi), np.radians(beta)
        g = np.array([np.sin(b), -np.cos(b), 0.0])
        self.F = np.array([g[0] * np.cos(p), g[1] * np.cos(p), -np.sin(p)])
        self.U = np.array([g[0] * np.sin(p), g[1] * np.sin(p), np.cos(p)])
        self.R = np.array([np.cos(b), np.sin(b), 0.0])
        self.h = self.C[2]

    def proj(self, P2):
        """地面の点 (N,2) → 画面 (N,2)。カメラの後ろ（または近すぎる）点は NaN。"""
        v = np.concatenate([P2, np.zeros((len(P2), 1))], 1) - self.C
        z = v @ self.F
        x = PX + self.f * (v @ self.R) / np.where(z > 0, z, 1)
        y = PY - self.f * (v @ self.U) / np.where(z > 0, z, 1)
        bad = (z < self.h * 0.02) | (np.abs(x) > 2e4) | (np.abs(y) > 2e4)
        out = np.stack([x, y], 1)
        out[bad] = np.nan
        return out

    def ground(self, xy):
        """画面 (N,2) → 地面 (N,2)（水平線より下の点だけ）。"""
        d = (self.F[None] + ((xy[:, 0:1] - PX) / self.f) * self.R[None] - ((xy[:, 1:2] - PY) / self.f) * self.U[None])
        s = -self.C[2] / d[:, 2]
        return self.C[None, :2] + d[:, :2] * s[:, None]

    def horizon(self):
        if self.phi > 89.0:
            return -1e9
        return PY - self.f * np.tan(np.radians(self.phi))


def _setup():
    if _S:
        return _S
    real = _real.wave_first_frame()
    ln = _real.lines_of(real)
    yh1 = ln['horizon']
    shore_y = float(np.median(ln['main'][-1][:, 1]))           # 波打ち際（最後の要素）
    phi1 = np.degrees(np.arctan((PY - yh1) / F1))
    coast = _coast_rings()
    M, tang = _mouth(coast)
    nrm = np.array([tang[1], -tang[0]])                        # 海の側の法線（地図の y は南が +）
    if nrm[1] < 0:
        nrm = -nrm
    beta1 = np.degrees(np.arctan2(nrm[0], -nrm[1]))            # 方位（北=0、東=90）
    a_s = np.radians(phi1) + np.arctan((shore_y - PY) / F1)
    D_s = H1 / np.tan(a_s)                                     # 波打ち際までの地面の距離
    C1 = np.array([M[0] - nrm[0] * D_s, M[1] - nrm[1] * D_s, H1])
    cam1 = Cam(C1, phi1, beta1, F1)

    # 実写の線を地面に逆投影（最後のカメラで投影し直すと元の位置に戻る）
    def back(polys):
        return [cam1.ground(p) for p in polys]

    # 海岸線は河口のまわりだけ細かく（近くで大きく写るため）
    co = []
    for P in coast:
        d = np.hypot(*(P - M).T)
        if d.min() < 60:
            # 河口から 6px 以内は 0.04px 刻み、60px 以内は 0.5px 刻み（その外は 6px）
            Q = _densify(P, 6.0)
            out = [Q[:1]]
            for a, b in zip(Q[:-1], Q[1:]):
                dm = min(np.hypot(*(a - M)), np.hypot(*(b - M)), np.hypot(*((a + b) / 2 - M)))
                step = 0.04 if dm < 8 else 0.5 if dm < 60 else 6.0
                out.append(_densify(np.stack([a, b]), step)[1:])
            co.append(np.concatenate(out))
        else:
            Q = _densify(P, 6.0)
            far = (Q[:, 0] < -2500) | (Q[:, 0] > 3600) | (Q[:, 1] < -2500) | (Q[:, 1] > 4400)
            if far.all():
                continue
            Q[far] = np.nan                        # 遠い所（大陸の輪郭の大部分）は捨てる
            co.append(Q)
    river, rs = _river(M)
    rm = _real.row_means(real, yh1, shore_y)

    # カメラの道のり。真上（北が上）→ 北を向いたまま起き上がって、海の上から河口へ降りる（波がたどり着く向き）
    # → 低い所で海の側へ振り向く（素早いパン。水平線は動かさない）→ b12 の1フレーム目の画角で止まる。
    YH = PY - yh1                                   # 水平線を画面の中心からこれだけ上に保つ（パンのあいだ）
    fh = 1200.0                                     # ここ（t=18）から水平線の高さを固定する
    keys = dict(
        cx=[(0, PX), (10, M[0]), (18, M[0]), (24, C1[0]), (35, C1[0])],
        cy=[(0, PY), (10, M[1] + 60), (15, M[1] + 95), (18, M[1] + 45), (24, C1[1] + 6), (30, C1[1]), (35, C1[1])],
        lh=[(0, np.log(1000)), (10, np.log(600)), (15, np.log(140)), (18, np.log(30)), (21, np.log(16)),
            (24, np.log(10)), (27, np.log(2.4)), (28.5, np.log(0.5)), (29.5, np.log(H1 * 2.9)), (30.5, np.log(H1 * 1.9)),
            (31.5, np.log(H1 * 1.4)), (32.5, np.log(H1 * 1.12)), (33.5, np.log(H1 * 1.03)), (35, np.log(H1))],
        lf=[(0, np.log(1000)), (12, np.log(1000)), (18, np.log(fh)), (24, np.log(1500)), (28, np.log(2600)),
            (30, np.log(4200)), (32, np.log(6000)), (33.5, np.log(F1)), (35, np.log(F1))],
        phi=[(0, 90.0), (9, 90.0), (14, 55.0), (18, np.degrees(np.arctan(YH / fh)))],
        beta=[(0, 0.0), (17.5, 0.0), (19.5, 30.0), (21.5, 105.0), (23.5, beta1 - 2.0), (26, beta1), (35, beta1)],
    )
    tr = {k: PchipInterpolator([a for a, _ in v], [b for _, b in v]) for k, v in keys.items()}

    def cam(t):
        t = float(np.clip(t, 0, 35))
        f = float(np.exp(tr['lf'](t)))
        if t <= 18:
            phi = float(np.clip(tr['phi'](t), 0, 90))
        else:
            phi = float(np.degrees(np.arctan(YH / f)))     # 水平線の高さを保つ
        return Cam((float(tr['cx'](t)), float(tr['cy'](t)), float(np.exp(tr['lh'](t)))), phi,
                   float(tr['beta'](t)), f)

    _S.update(real=real, ln=ln, yh1=yh1, phi1=phi1, cam=cam, cam1=cam1, coast=co, offs=_offsets(),
              river=river, rs=rs, M=M, rm=rm, main=back(ln['main']), swell=back(ln['swell']),
              bgA=b10._background())
    gA = _S['bgA'].mean(axis=2, keepdims=True)
    dA = _S['bgA'] - gA
    _S['bgS'] = lambda s: gA + dA * np.float32(s)
    return _S


# ---------------------------------------------------------------- 1フレーム
def _ramp(i, a, b, kind='inout'):
    return core.ease((i - a) / float(b - a), kind) if b > a else float(i >= a)


_ZERO = np.zeros((H, W), np.float32)


class _L2(G.Lines):
    """何も描かなかったときは縮小を省く。"""
    used = False

    def poly(self, *a, **k):
        self.used = True
        super().poly(*a, **k)

    def dot(self, *a, **k):
        self.used = True
        super().dot(*a, **k)

    def done(self):
        return super().done() if self.used else _ZERO


class _Lines1(_L2):
    S = 1                                          # 川の動きのぼかし用（枚数が多いので等倍で描く）


def _lines(t, S, river_only=False):
    """時刻 t（フレーム、小数可）の線のマスク：地図（白）、手前の線（白・くっきり）、川（ACC）。"""
    cam = S['cam'](t)
    yh = cam.horizon()
    a_map = (0.10 + 0.22 * _ramp(t, 0, 7)) * (1 - _ramp(t, 27, 33))
    a_coast = (0.16 + 0.44 * _ramp(t, 0, 7)) * (1 - _ramp(t, 30.5, 34.5))
    a_riv = _ramp(t, 1, 4) * (1 - _ramp(t, 17, 21))
    a_real = _ramp(t, 28.5, 32.5)
    Lr = _Lines1() if river_only else _L2()
    if not river_only:
        Lm, Lc = _L2(), _L2()
    if a_map > 0.003 and not river_only:
        for k, P in S['offs']:
            Lm.poly(cam.proj(P), 1.25, 255 * (0.85 - 0.12 * k) * a_map, closed=False)
    if a_coast > 0.003 and not river_only:
        for P in S['coast']:
            Lm.poly(cam.proj(P), 1.9, 255 * 0.95 * a_coast)
    # 川：上流から河口へ描いていく
    if a_riv > 0.003:
        u = core.ease((t - 1) / 8.5, 'inout')
        if u > 0:
            R, rs = S['river'], S['rs']
            n = int(np.searchsorted(rs, u))
            pts = R[:max(n, 2)]
            Lr.poly(cam.proj(pts), 3.6, 255 * a_riv)
            if u > 0.97:
                p = cam.proj(S['M'][None])[0]
                if np.isfinite(p).all():
                    Lr.dot(p[0], p[1], 6.0 * _ramp(t, 9, 11.5, 'out'), 255 * a_riv)
    if river_only:
        return Lr.done()
    # 水平線と手前の線
    if yh > -50:
        a_h = 0.92 * _ramp(t, 13, 21) * float(np.clip((yh + 40) / 160, 0, 1))
        Lc.poly(np.array([[-10, yh], [W + 10, yh]]), 2.0, 255 * a_h)
    if a_real > 0.003:
        for P in S['main']:
            Lc.poly(cam.proj(P), 2.0, 255 * 0.92 * a_real)
        for P in S['swell']:
            Lc.poly(cam.proj(P), 1.4, 255 * 0.42 * a_real)
    m = Lm.done()
    # 水平線の近くの地図の線は霧で消す（遠くで縮んだ線のちらつきを防ぐ）
    if yh > -1e8:
        yy = np.arange(H, dtype=np.float32)
        fog = np.clip((yy - yh - 4) / 70.0, 0, 1) ** 1.5
        m = m * fog[:, None]
    return m, Lc.done(), Lr.done(), cam


def _background(i, cam, S):
    """行の対応：カメラの俯角 α の行 → 最後のカメラで同じ地面の距離（空は同じ仰角）に写る行の色。"""
    rows = np.arange(H, dtype=np.float64)
    a = np.radians(cam.phi) + np.arctan((rows - PY) / cam.f)
    a = np.clip(a, -1.4, 1.45)
    a1 = np.where(a > 0, np.arctan(np.tan(a) * (H1 / cam.h)), a)
    y1 = PY + F1 * np.tan(a1 - np.radians(S['phi1']))
    # 高いところからは、地面はすべて「沖の海」の色（実写の近くの泡や砂の色は、低くなってから）
    lam = float(np.clip(np.log(cam.h / H1) / np.log(60.0), 0, 1))
    y1 = np.where(a > 0, y1 * (1 - lam) + (S['yh1'] + 3) * lam, y1)
    y1 = np.clip(y1, 0, H - 1)
    k0 = np.floor(y1).astype(int)
    k1 = np.minimum(k0 + 1, H - 1)
    fr = (y1 - k0)[:, None]
    col = S['rm'][k0] * (1 - fr) + S['rm'][k1] * fr
    w = core.ease((i - 7) / 28.0, 'smooth')
    col = col[:, None, :].astype(np.float32)
    # 濃紺と明るい青の途中で灰色っぽくならないように、途中だけ彩度を少し上げる（線形なので混ぜる前に分けて計算）
    s = 1 + 1.6 * w * (1 - w)
    gc = col.mean(axis=2, keepdims=True)
    colS = gc + (col - gc) * s
    return S['bgS'](s) * (1 - w) + colS * w


def render(beat, ctx):
    S = _setup()
    n = beat['n']
    assert n == 36, n
    for i in range(n):
        # 動きの速いところ：前半（寄り・起き上がり）は 3 枚の平均、パンのあいだは横の動きのぼかし
        if 12 <= i <= 17 or 28 <= i <= 32:
            subs = tuple(i + d for d in (-0.4, -0.2, 0.0, 0.2, 0.4))
        elif 2 <= i <= 27 and not 18 <= i <= 25 or i == 33:
            subs = (i - 0.33, i, i + 0.33)
        else:
            subs = (i,)
        acc = None
        for t in subs:
            m, c, r, cam = _lines(t, S)
            acc = [m, c, r] if acc is None else [a + b for a, b in zip(acc, (m, c, r))]
        m, c, r = [a / len(subs) for a in acc]
        if 8 <= i <= 21:                           # 川（太い ACC の線）はもっと細かく平均する
            r = sum(_lines(i + d, S, river_only=True) for d in np.linspace(-0.45, 0.45, 9)) / 9
        cam = S['cam'](i)
        if 18 <= i <= 25:
            v = cam.f * abs(np.radians(S['cam'](i + 0.5).beta - S['cam'](i - 0.5).beta))
            v = min(v, 360.0)
            m, c, r = (core.whip(x, v * 0.8) for x in (m, c, r))
        f = _background(i, cam, S)
        lm = np.clip(np.maximum(m, c), 0, 1)[..., None]
        f = f * (1 - lm) + core.FOAM * lm
        lr = r[..., None]
        f = f * (1 - lr) + core.ACC * lr
        yield np.clip(f, 0, 1).astype(np.float32)
