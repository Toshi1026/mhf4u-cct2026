"""案C ショット C の地割れ（2026-11-08 クライアント修正 4「丸い穴ができるだけでなく、クレーターから派生して周りの地面にヒビが入っていく」）。

クレーターの縁（半径 約 50m）から外へ、枝分かれしながら草原と放出物の上を走る放射状の地割れ。
  ・1 段目（G1＝ショット C の頭＋3）：主な割れ目 13 本が縁から一気に走り出し（はじめ 1 フレーム 9m、だんだん遅く）、途中で枝分かれ。
  ・2 段目（G2＝04 の「1,400」が止まるフレーム）：先がもう一段のびて、細い枝が増える（数字が止まる「ドン」に合わせて）。
  ・見た目：暗い割れ目（奥は影）＋日の当たる側の内壁がうすく明るい＋まわりのめくれた土（茶色のぼかし）。
    割れ目が通ったところで小さな砂ぼこりが上がる（crack_items）。
世界の上から見た格子（0.3m）に線を引いておき（到達の時刻・割れ目・土）、ショット C の板の各画素の (X, Z) で引く。
地割れは開けた草原の中だけ（建物・道・水はない）。光らせない。
"""
import math

import cv2
import numpy as np

import core
from cutC.beats import _field as F

CELL, EXT = 0.30, 205.0                 # 格子の 1 マス（m）・半分の広さ（m）
NG = int(round(2 * EXT / CELL))
R0 = 50.0                               # 割れ目の出どころ（縁の外側）
V0, TAU = 9.0, 14.0                     # 走る速さ（m/フレーム）と遅くなる時間（フレーム）
NONE = 1e9
_C = {}

COL_CORE = core.c('1E1813')
COL_WALL = core.c('8C7862')
COL_SOIL = core.c('5B4938')


def _arrive(d, t0, v0=V0, tau=TAU):
    """割れ目の前線が出どころから d m の所に着く時刻（フレーム）。速さは v0 から指数的に落ちる（最大 v0·tau m）。"""
    x = np.clip(d / (v0 * tau), 0, 0.985)
    return t0 + tau * -np.log(1 - x)


def _walk(rng, p, ang, length, step=2.4, wiggle=0.20, pull=0.25, radial=True):
    """ゆらぎながら進む折れ線（radial：外向きへ引き戻す）。"""
    pts = [np.array(p, float)]
    a = ang
    for _ in range(int(length / step)):
        q = pts[-1]
        if radial:
            ra = math.atan2(q[1], q[0])
            da = (ra - a + math.pi) % (2 * math.pi) - math.pi
            a += pull * da
        a += rng.normal(0, wiggle)
        pts.append(q + step * np.array([math.cos(a), math.sin(a)]))
    return np.array(pts)


def paths(G1, G2):
    """(折れ線 (N,2), 各点の到達時刻 (N,), 各点の幅 m (N,), 段) のリスト。"""
    key = (G1, G2)
    if key in _C:
        return _C[key]
    rng = np.random.default_rng(1401)
    out = []
    n_main = 13
    for j in range(n_main):
        th = 2 * math.pi * (j + rng.uniform(-0.3, 0.3)) / n_main + 0.2
        L = rng.uniform(58, 118)
        p0 = R0 * 1.0 * np.array([math.cos(th), math.sin(th)])
        P = _walk(rng, p0, th, L)
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        t = _arrive(d, G1 + rng.uniform(0, 2.0))
        w = 0.35 + 2.5 * (1 - d / d[-1]) ** 1.3
        out.append((P, t, w, 1))
        # 2 段目：先がのびる
        ext = _walk(rng, P[-1], math.atan2(*(P[-1] - P[-3])[::-1]), rng.uniform(12, 26))
        de = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ext, axis=0), axis=1))]
        out.append((ext, _arrive(de, G2 + rng.uniform(0, 1.5), v0=5.0, tau=8.0), 0.35 + 0.35 * (1 - de / de[-1]), 2))
        # 枝（1 段目：親が通ったあとに分かれる）と、その枝の枝
        for k in range(len(P)):
            if d[k] < 8 or d[k] > d[-1] - 10 or rng.uniform() > 0.075:
                continue
            side = rng.choice([-1, 1])
            fa = math.atan2(*(P[min(k + 1, len(P) - 1)] - P[k])[::-1]) + side * math.radians(rng.uniform(22, 42))
            stage = 1 if rng.uniform() < 0.65 else 2
            lb = (d[-1] - d[k]) * rng.uniform(0.3, 0.65) * (0.7 if stage == 2 else 1.0)
            B = _walk(rng, P[k], fa, lb, wiggle=0.24, pull=0.10)
            db = np.r_[0, np.cumsum(np.linalg.norm(np.diff(B, axis=0), axis=1))]
            tb0 = t[k] if stage == 1 else G2 + rng.uniform(0, 3)
            wb = 0.25 + 0.6 * w[k] * (1 - db / max(db[-1], 1e-3))
            tb = _arrive(db, tb0, v0=7.0 if stage == 1 else 4.5, tau=12.0 if stage == 1 else 8.0)
            out.append((B, tb, wb, stage))
            for m in range(4, len(B) - 3):
                if rng.uniform() > 0.06:
                    continue
                ca = math.atan2(*(B[m + 1] - B[m])[::-1]) + rng.choice([-1, 1]) * math.radians(rng.uniform(25, 50))
                Cc = _walk(rng, B[m], ca, rng.uniform(5, 14), wiggle=0.28, pull=0.0, radial=False)
                dc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Cc, axis=0), axis=1))]
                tc0 = max(tb[m], G2 + rng.uniform(0, 4)) if rng.uniform() < 0.5 else tb[m]
                out.append((Cc, _arrive(dc, tc0, v0=4.0, tau=8.0), 0.22 + 0.25 * (1 - dc / max(dc[-1], 1e-3)),
                            2 if tc0 >= G2 else 1))
    # 縁のすぐ外の短い細かいひび（1 段目、縁に沿って少し斜め）
    for j in range(40):
        th = rng.uniform(0, 2 * math.pi)
        p0 = rng.uniform(49, 56) * np.array([math.cos(th), math.sin(th)])
        Sx = _walk(rng, p0, th + rng.normal(0, 0.5), rng.uniform(4, 11), step=1.5, wiggle=0.3, pull=0.1)
        ds = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Sx, axis=0), axis=1))]
        out.append((Sx, _arrive(ds, G1 + rng.uniform(1, 6), v0=3.0, tau=6.0), 0.18 + 0.3 * (1 - ds / ds[-1]), 1))
    _C[key] = out
    return out


def _g(P):
    return P / CELL + NG / 2 - 0.5


def maps(G1, G2):
    """上から見た格子：(到達の時刻, 割れ目 0..1, 影の側, めくれた土 0..1)。"""
    key = ('maps', G1, G2)
    if key in _C:
        return _C[key]
    segs = []
    for P, t, w, st in paths(G1, G2):
        for k in range(len(P) - 1):
            segs.append((float(t[k]), P[k], P[k + 1], float(w[k])))
    segs.sort(key=lambda s: -s[0])                    # 遅いものから描く → 早い到達が上書き（最小）
    A = np.full((NG, NG), NONE, np.float32)
    core_m = np.zeros((NG, NG), np.uint8)
    soil_m = np.zeros((NG, NG), np.uint8)
    sh = 4
    for t, a, b, w in segs:
        pa, pb = _g(a), _g(b)
        ia = (int(round(pa[0] * 16)), int(round(pa[1] * 16)))
        ib = (int(round(pb[0] * 16)), int(round(pb[1] * 16)))
        cv2.line(A, (int(round(pa[0])), int(round(pa[1]))), (int(round(pb[0])), int(round(pb[1]))), t,
                 max(1, int(round(w * 3.2 / CELL))))
        cv2.line(core_m, ia, ib, 255, max(1, int(round(w / CELL))), cv2.LINE_AA, sh)
        cv2.line(soil_m, ia, ib, 255, max(1, int(round(w * 2.4 / CELL))), cv2.LINE_AA, sh)
    c = core_m.astype(np.float32) / 255
    # ぎざぎざの縁：ノイズでけずる
    yy, xx = np.mgrid[0:NG, 0:NG].astype(np.float32)
    nz = cv2.resize(np.random.default_rng(7).uniform(0, 1, (NG // 6, NG // 6)).astype(np.float32), (NG, NG),
                    interpolation=cv2.INTER_CUBIC)
    c = cv2.GaussianBlur(c, (0, 0), 0.8)
    c = np.clip((c + 0.35 * (nz - 0.5) - 0.30) / 0.35, 0, 1)
    # 日の当たる向きの反対側へずらした割れ目＝影の側（暗い）、残り＝日の当たる内壁（うすく明るい）
    sx, sz = F.SUN[0], F.SUN[2]
    n = math.hypot(sx, sz)
    dx, dz = sx / n * 2.0, sz / n * 2.0
    Ms = np.float32([[1, 0, dx], [0, 1, dz]])
    dark = np.minimum(c, cv2.warpAffine(c, Ms, (NG, NG)))
    s = cv2.GaussianBlur(soil_m.astype(np.float32) / 255, (0, 0), 2.2)
    s = np.clip(s * (0.75 + 0.5 * nz), 0, 1)
    _C[key] = (A, c, dark, s)
    return _C[key]


def sample(X, Z, G1, G2):
    """板の各画素の (X, Z) → (到達, 割れ目, 影, 土)。空の画素は到達 NONE。"""
    A, c, dark, s = maps(G1, G2)
    u = (X / CELL + NG / 2 - 0.5).astype(np.float32)
    v = (Z / CELL + NG / 2 - 0.5).astype(np.float32)
    rA = cv2.remap(A, u, v, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=NONE)
    rs = [cv2.remap(m, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for m in (c, dark, s)]
    return (rA,) + tuple(rs)


def shade(src, smp, g):
    """板 src（sRGB）に、時刻 g までに走った地割れを描く（smp は sample() の結果を行の範囲に切ったもの）。"""
    A, c, dark, s = smp
    age = g - A
    if not np.any(age > 0):
        return src
    rc = np.clip(age / 3.0, 0, 1)                     # 割れ目は 3 フレームで開く
    rs = np.clip(age / 6.0, 0, 1) * (0.55 + 0.45 * np.clip(age / 18.0, 0, 1))
    ms = (s * rs * 0.62)[..., None]
    out = src * (1 - ms) + COL_SOIL * ms
    mw = ((c - dark) * rc * 0.55)[..., None]
    out = out * (1 - mw) + COL_WALL * mw
    md = (np.maximum(dark, c * 0.6) * rc * 0.88)[..., None]
    return out * (1 - md) + COL_CORE * md


def fronts(G1, G2, g, step=4.0):
    """いま前線が通っている（通って 14 フレーム以内の）点：(x, z, 通ってからのフレーム, 幅)。砂ぼこりに使う。"""
    out = []
    for P, t, w, st in paths(G1, G2):
        kk = max(1, int(round(step / 2.4)))
        for k in range(0, len(P), kk):
            a = g - t[k]
            if 0 <= a < 14:
                out.append((P[k, 0], P[k, 1], a, w[k]))
    return out
