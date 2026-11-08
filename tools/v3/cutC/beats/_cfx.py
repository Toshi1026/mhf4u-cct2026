"""案C ショット C（03 の最後・04・05 の通し）の時間・カメラ・動く物：遠くへ広がる砂煙の壁、クレーターの上の砂煙、落ちてくる大きな土くれ。
時間は「当たってからの通しのフレーム」gt（03 の launch＝0）。カメラは 03 の WC0 から 05 の最後まで、1 本のゆっくりした押し込み。"""
import math

import cv2
import numpy as np

import core
from core import W, H
import timeline_c as timeline
from cutB.beats import _motion as M
from cutC.beats import _shots as SH, _fx as FX

ZOOM = (1.0, 1.13)
_C = {}


def anchors():
    """(03 の f0, launch の絶対フレーム, ショット C の始まりの絶対フレーム, 05 の終わりの絶対フレーム)。"""
    if 'a' not in _C:
        from cutC.beats import c03
        bs = {b['key']: b for b in timeline.load()}
        T = c03.timing(bs['03'])
        f0 = bs['03']['f0']
        _C['a'] = (f0, f0 + T['launch'], f0 + T['wc0'], bs['05']['f1'])
    return _C['a']


def gt(beat, i):
    """当たって（打ち上がって）からの通しのフレーム。"""
    return beat['f0'] + i - anchors()[1]


def zoom_at(beat, i):
    _, _, c0, c1 = anchors()
    q = float(np.clip((beat['f0'] + i - c0) / float(c1 - c0), 0, 1))
    return ZOOM[0] + (ZOOM[1] - ZOOM[0]) * (0.3 * q + 0.7 * M.ease(q, 'sine_inout'))


def project(cam, Mm, P):
    """世界の点 (N,3) → 画面 (N,2)・奥行き・1m の px（画面）。"""
    S, z = cam.project(np.asarray(P, float))
    S = S @ Mm[:, :2].T + Mm[:, 2]
    return S, z, cam.fpx * Mm[0, 0] / np.maximum(z, 1e-3)


# ---------------------------------------------------------------- 砂煙
def _rng_ring():
    if 'ring' not in _C:
        rng = np.random.default_rng(17)
        n = 150
        _C['ring'] = dict(th=rng.uniform(0, 2 * np.pi, n), k=rng.integers(0, 6, n), s=rng.uniform(0.7, 1.3, n),
                          r=rng.uniform(0.8, 1.05, n), lag=rng.uniform(0, 1, n))
        m = 46
        _C['haze'] = dict(x=rng.normal(0, 22, m), z=rng.normal(0, 22, m), k=rng.integers(0, 6, m), s=rng.uniform(0.8, 1.3, m),
                          rise=rng.uniform(0.25, 0.6, m), ph=rng.uniform(0, 1, m))
    return _C['ring'], _C['haze']


def ring_R(g):
    """砂煙の壁の半径（m）：ショット C のころ 150m → 05 の終わり 700m（だんだん遅く）。"""
    return 25.0 + 420.0 * (1 - math.exp(-max(g, 0) / 80.0))


def dust_C(f, cam, Mm, g, alpha=1.0):
    """遠くへ広がる低い砂煙の壁＋クレーターの上に残る砂煙（ゆっくり上がって流れる）。奥から描く。"""
    ring, haze = _rng_ring()
    items = []
    Rd = ring_R(g)
    for th, k, s, r, lag in zip(ring['th'], ring['k'], ring['s'], ring['r'], ring['lag']):
        rr = Rd * r
        sz = (26.0 + 0.10 * rr) * s
        P = np.array([[rr * math.cos(th), sz * 0.25, rr * math.sin(th)]])
        S, z, ppm = project(cam, Mm, P)
        px = float(sz * ppm[0] * 1.7)
        if z[0] < 60 or px > 1400 or not (-700 < S[0, 0] < W + 700 and -700 < S[0, 1] < H + 700):
            continue
        a = 0.62 * alpha * math.exp(-max(g - 40, 0) / 220.0) * min(1.0, (1400 - px) / 500.0)
        items.append((float(z[0]), int(k), float(S[0, 0]), float(S[0, 1]), px, a, 0.98))
    for x, z_, k, s, rise, ph in zip(haze['x'], haze['z'], haze['k'], haze['s'], haze['rise'], haze['ph']):
        hh = 6.0 + rise * g * 0.45 + 10 * ph
        sz = (30.0 + 0.16 * g) * s
        P = np.array([[x + 0.06 * g, hh, z_]])
        S, z, ppm = project(cam, Mm, P)
        a = 0.50 * alpha * min(1.0, g / 8.0) * math.exp(-max(g - 40, 0) / 120.0)
        items.append((float(z[0]), int(k), float(S[0, 0]), float(S[0, 1]), float(sz * ppm[0] * 1.6), a, 0.90))
    for (z, k, x, y, s, a, dk) in sorted(items, key=lambda q: -q[0]):
        f = FX.put_puff(f, k, x, y, s, a, dk)
    return f


class BigClods(FX.Clods):
    """ショット C の大きな土くれ（m の大きさ、m/フレームの速さ）。"""

    def draw_C(self, f, cam, Mm, g, alpha=1.0):
        return self.draw(f, cam, g + self.f0, alpha=alpha, M2=Mm)


def clods_C():
    if 'cl' not in _C:
        _C['cl'] = BigClods(n=220, f0=0, origin=(0.0, 0.0, 0.0), speed=(1.0, 4.2), el_range=(40, 75), seed=11,
                            g=0.11, size=(0.25, 1.4))
    return _C['cl']


def frame_C(beat, i, extra=None):
    """ショット C の 1 フレーム（sRGB）と、そのときの (cam, Mm)。extra(f, cam, Mm, g)：砂煙の手前に描く物（05 のゾウ）。"""
    plate, dep, cam = SH.plate_C()
    g = gt(beat, i)
    f, Mm = SH.view_C(plate, zoom_at(beat, i))
    f = f.copy()
    if extra is not None:
        f = extra(f, cam, Mm, g)
    f = clods_C().draw_C(f, cam, Mm, g)
    f = dust_C(f, cam, Mm, g)
    return f, cam, Mm, g
