"""案C 07〜10 の悪役（顔のない白いピクトグラム＋マント）を SDF の 3D で（_figures3d と同じ材質・同じ大きさの単位）。

案B の _figures3d.villain_scene は決まった 3 つの形（stand / tumble / fly）だけなので、ここでは手足の角度から形を組む：
  pose(kind, t)：'tumble'（回って飛ぶ・ひじひざを曲げる）、'flail'（真っ逆さまにジタバタ：腕を振り回し、脚をばたつかせる）、
                 'spread'（頭から落ちる：腕を前へ、脚をそろえて伸ばす＝落下の終わり）。w で 2 つの形を混ぜる。
  rgba(x, y, scale, ang, caps_cape, t, ...) → (rgba, x0, y0)   _figures3d.villain_rgba と同じ戻り値
白い体（BODY）＋淡い青のマント（CAPE）。光は全ショット共通の左上、リムは右上。白の上限は cap_white（≤0.92）。
"""
import math

import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from cutB.beats import _sdf as S
from cutB.beats import _figures3d as F3

SH = np.array([0.0, -6.0])        # 肩のまんなか
HIP = np.array([0.0, 4.2])


def _limb(base, a1, l1, a2, l2):
    """付け根 base から角度 a1（度、0＝下＝+y、正＝時計回り＝画面の左へ…ローカル）で l1、さらに a2（ひじ・ひざの曲げを足した角度）で l2。"""
    r1, r2 = math.radians(a1), math.radians(a1 + a2)
    j = base + l1 * np.array([-math.sin(r1), math.cos(r1)])
    e = j + l2 * np.array([-math.sin(r2), math.cos(r2)])
    return j, e


def pose(kind, t=0.0, seed=0):
    """(caps, cape)：caps は _figures._villain_parts と同じ (p0, p1, r0, r1, z0, k)。t は秒（ジタバタの位相）。"""
    if kind == 'tumble':
        from cutB.beats import _figures as F
        caps, _, cape = F._villain_parts('tumble')
        return list(caps), list(cape)
    ph = t * 2 * math.pi
    if kind == 'flail':
        # 腕：肩から上（頭の方＝落ちる向き）へ大きく振り回す。左右で位相をずらす（2.6Hz・3.1Hz）
        aL = 122 + 48 * math.sin(ph * 2.6 + seed)
        aR = -122 + 48 * math.sin(ph * 3.1 + 1.7 + seed)
        bL = -40 + 35 * math.sin(ph * 2.6 + 1.0 + seed)
        bR = 40 + 35 * math.sin(ph * 3.1 + 2.4 + seed)
        # 脚：ばたばた（自転車こぎ）
        lL = 30 + 30 * math.sin(ph * 3.4 + seed)
        lR = -30 + 30 * math.sin(ph * 3.4 + math.pi + seed)
        kL = -50 + 40 * math.sin(ph * 3.4 + 1.2 + seed)
        kR = 50 + 40 * math.sin(ph * 3.4 + math.pi + 1.2 + seed)
    elif kind == 'spread':
        aL, aR, bL, bR = 168, -168, -8, 8
        lL, lR, kL, kR = 6, -6, 0, 0
    else:
        raise ValueError(kind)
    shL, shR = SH + (-1.0, 0.0), SH + (1.0, 0.0)
    hpL, hpR = HIP + (-1.2, 0.0), HIP + (1.2, 0.0)
    eL, hL = _limb(shL, aL, 6.8, bL, 6.2)
    eR, hR = _limb(shR, aR, 6.8, bR, 6.2)
    kLp, fL = _limb(hpL, lL, 6.4, kL, 6.6)
    kRp, fR = _limb(hpR, lR, 6.4, kR, 6.6)
    t2 = lambda v: (float(v[0]), float(v[1]))
    caps = [(t2(fL), t2(kLp), 1.2, 1.35, -0.4, 0.0),            # すね（足先が p0）
            (t2(fR), t2(kRp), 1.2, 1.35, -0.4, 0.6),
            ((0.0, 4.6), (0.0, -6.8), 1.75, 2.15, 0.0, 0.9),     # 胴
            ((0.0, -6.8), (0.0, -9.0), 0.95, 0.95, 0.0, 0.5),    # 首
            (t2(hL), t2(eL), 0.95, 1.05, 0.6, 0.7),              # 前腕（手が p0）
            (t2(hR), t2(eR), 0.95, 1.05, 0.6, 0.7),
            (t2(kLp), t2(hpL), 1.35, 1.55, -0.4, 0.6),           # もも
            (t2(kRp), t2(hpR), 1.35, 1.55, -0.4, 0.6),
            (t2(eL), t2(shL), 1.05, 1.15, 0.6, 0.7),             # 上腕
            (t2(eR), t2(shR), 1.05, 1.15, 0.6, 0.7)]
    # マント：肩から足の方（落ちる向きの反対）へ流れてはためく
    fl = math.sin(ph * 5.0 + seed) * 1.6
    fl2 = math.sin(ph * 6.3 + 1.1 + seed) * 1.2
    cape = [(-2.6, -7.6), (2.6, -7.6), (7.5 + fl * 0.4, 6.0), (5.0 + fl, 17.0), (1.5 + fl2, 14.0), (-2.5 + fl, 18.0),
            (-7.0 + fl * 0.4, 7.0)]
    return caps, cape


def mix(pa, pb, w):
    """2 つの形 (caps, cape) を w（0→a、1→b）で混ぜる（端点の線形補間）。"""
    ca, ka = pa
    cb, kb = pb
    caps = []
    for A, B in zip(ca, cb):
        p0 = tuple(np.add(np.multiply(A[0], 1 - w), np.multiply(B[0], w)))
        p1 = tuple(np.add(np.multiply(A[1], 1 - w), np.multiply(B[1], w)))
        caps.append((p0, p1) + tuple(a * (1 - w) + b * w for a, b in zip(A[2:], B[2:])))
    cape = [tuple(np.add(np.multiply(a, 1 - w), np.multiply(b, w))) for a, b in zip(ka, kb)]
    return caps, cape


def scene(caps, cape, t):
    sc = S.Scene()
    cl = np.asarray(cape, float)
    ax_ = cl[3] - cl[0]
    an = ax_ / (np.hypot(*ax_) + 1e-6)
    sc.slab(cl, z=-2.6, half=0.30, r=0.28, k=0.0, mat=1, fold=(0.75, 1.1, t * 4.0, an[1], -an[0]), bend=-0.004)
    for (p0, p1, r0, r1, z0, k) in caps:
        sc.cone((p0[0], p0[1], z0 * 0.8), (p1[0], p1[1], z0 * 0.8), r0, r1, k=0.9 + k * 0.6, mat=0)
    sc.ellipsoid((0.0, -4.4, 0.25), (2.55, 3.0, 2.1), k=1.4, mat=0)
    sc.ellipsoid((0.0, 3.6, 0.0), (2.1, 1.6, 1.8), k=1.2, mat=0)
    for (p0, p1, r0, r1, z0, k) in caps[4:6]:
        sc.sphere((p0[0], p0[1], z0 * 0.8), r0 * 1.22, k=0.6, mat=0)
    for (p0, p1, r0, r1, z0, k) in caps[:2]:
        sc.ellipsoid((p0[0], p0[1] + 0.4, z0 * 0.8 + 0.5), (1.5, 1.0, 1.9), k=0.8, mat=0)
    sc.ellipsoid((0.0, -12.2, 0.35), (3.7 * 0.94, 3.7 * 1.04, 3.7 * 0.92), k=1.0, mat=0)
    sc.ellipsoid((0.0, -7.4, -0.3), (2.6, 0.9, 2.2), k=0.7, mat=1)
    return sc


SPACE_ENV = dict(sun=np.array([1.0, 0.99, 0.975]) * 1.55, sky=F3._lin('7FA6CC') * 0.16, ground=F3._lin('3A6A98') * 0.22,
                 rim=F3._lin('BFE6FF') * 1.0, rim_dir=(0.70, -0.70))
SKY_ENV = dict(F3.ENV_FLY)


def env_mix(w):
    """w：0＝空の中（SKY_ENV）→ 1＝宇宙（SPACE_ENV）。"""
    out = {}
    for k_ in SKY_ENV:
        a, b = SKY_ENV[k_], SPACE_ENV.get(k_, SKY_ENV[k_])
        out[k_] = tuple(np.add(np.multiply(a, 1 - w), np.multiply(b, w))) if isinstance(a, tuple) else a * (1 - w) + b * w
    return out


def rgba(x, y, scale, ang, pc, t=0.0, squash=1.0, env=None, ss=None):
    """(x, y)：胴のまんなか（画面）、scale：1 で全長 約 34px、ang：度（時計回り）。pc＝(caps, cape)。"""
    Rr = int(math.ceil(25 * scale * max(1.0, squash) + 10))
    fx, fy = x - math.floor(x), y - math.floor(y)
    cx, cy = Rr + fx, Rr + fy
    r = math.radians(ang)
    Rm = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    Sq = np.array([[1.0, 0], [0, squash]])
    A = Rm @ Sq * scale
    det = math.sqrt(abs(np.linalg.det(A)))
    sc = scene(pc[0], pc[1], t)
    if ss is None:
        ss = 2 if det >= 2.5 else 3 if det >= 1.5 else int(np.clip(round(10.0 / max(det, 0.5)), 2, 4))
    out, _z = S.render(sc, [F3.BODY, F3.CAPE], 2 * Rr, 2 * Rr, A, (cx, cy), zs=det, ss=ss, light=F3.KEY,
                       env=env or SPACE_ENV, shadow_k=7.0, ao_d=0.9)
    return F3.cap_white(out), int(math.floor(x)) - Rr, int(math.floor(y)) - Rr


# ---------------------------------------------------------------- 風の線（落ちる悪役のまわりを、落ちる向きの反対へ流れる）
def wind_lines(f, cx, cy, k, d, amount=1.0, n=26, spread=260.0, speed=70.0, seed=5, col=None):
    """(cx, cy)：悪役のまわりの中心、d：風が流れる向き（画面の単位ベクトル。落ちる向きの反対）、k：フレーム（小数可）。
    閉じた式（どのフレームから描いても同じ）。先が細い白い線（太さ 1–2.2px、長さ 80–260px、α≤0.5）。"""
    if amount <= 0.01:
        return f
    from beats import _geo as G
    rng = np.random.default_rng(seed)
    off = rng.uniform(-1, 1, n) * spread
    off = np.sign(off) * (np.abs(off) * 0.75 + spread * 0.18)        # 体の上は避ける
    ph = rng.uniform(0, 1, n)
    ln = rng.uniform(80, 260, n)
    wd = rng.uniform(1.0, 2.2, n)
    al = rng.uniform(0.25, 0.55, n)
    sp = speed * rng.uniform(0.8, 1.3, n)
    d = np.asarray(d, float) / np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]])
    span = 900.0
    L = G.Lines(f.shape[1], f.shape[0])
    for j in range(n):
        s = ((ph[j] * span + k * sp[j]) % span) - span * 0.45            # 中心からの位置（流れる向き）
        tail = np.array([cx, cy]) + nrm * off[j] + d * s
        head = tail + d * ln[j]
        # 先が細い：3 つの区間で太さ・明るさを落とす
        for q0, q1, wk, ak in ((0.0, 0.45, 1.0, 1.0), (0.45, 0.8, 0.7, 0.6), (0.8, 1.0, 0.45, 0.3)):
            a_ = tail + (head - tail) * q0
            b_ = tail + (head - tail) * q1
            L.poly(np.stack([a_, b_]), wd[j] * wk, int(255 * al[j] * ak * amount))
    m = L.done()
    return f + (np.asarray(core.FOAM, np.float32) * 0.92 - f) * m[..., None]
