"""案B の登場する形を、_render.Relief（高さの場）で「やわらかい 3D の人形」として描く（look-dev 2026-11）。

- villain_rgba(...)：_art.villain_rgba と同じ引数・同じ戻り値 (rgba, x0, y0)。顔のない悪役を、丸いカプセルの手足＋球の頭＋
  布のマント（しわのふくらみ）で。白いビニールの人形のような材質（つや＋やわらかい陰＋縁の光）。既存のキャラクターに似せない
  （触角・角・顔・衣装の柄はなし）。
- fist_sprite(ang, scale)：_art._fist_shapes の形を 3D に：手の甲（丸い箱）、前に盛り上がる 4 本の指、横切る親指、布の袖口（リブ）。
- elephant_sprite(esc)：b05 のアフリカゾウのシルエットを、胴・脚・頭・鼻（太さの変わるカプセルの列）・耳（重なる板）・牙で。
光は LIGHT（画面の向き：左上・少し手前）。各シーンは light= で変えられる。
"""
import math

import numpy as np

import core
from cutB.beats import _render as R

LIGHT = (-0.62, -0.50, 0.42)
BODY = dict(rgb=core.c('EAF3FA'), spec=0.38, rough=0.30, sss=0.55, rimk=1.0)
CAPE = dict(rgb=core.c('B9D9F0'), spec=0.10, rough=0.65, sss=0.45, rimk=0.7)


def _R2(a):
    r = math.radians(a)
    return np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])


# ---------------------------------------------------------------- 悪役
def _villain_parts(pose):
    """(capsules, head, cape)：ローカル単位（_art.villain と同じ。胴のまんなかが原点、+y 下）。
    capsule：(p0, p1, r0, r1, z0, k)。"""
    if pose == 'stand':
        caps = [((-4.2, 16.0), (0.0, 4.2), 1.25, 1.55, -0.4, 0.0),     # 脚
                ((4.2, 16.0), (0.0, 4.2), 1.25, 1.55, -0.4, 0.6),
                ((0.0, 4.6), (0.0, -6.8), 1.75, 2.15, 0.0, 0.9),      # 胴（胸が少し厚い）
                ((0.0, -6.8), (0.0, -9.0), 0.95, 0.95, 0.0, 0.5),     # 首
                ((0.0, -6.0), (-6.2, -1.2), 1.15, 1.05, 0.9, 0.7),    # 腰に手
                ((-6.2, -1.2), (-1.4, 2.6), 1.05, 1.0, 1.2, 0.6),
                ((0.0, -6.0), (6.2, -1.2), 1.15, 1.05, 0.9, 0.7),
                ((6.2, -1.2), (1.4, 2.6), 1.05, 1.0, 1.2, 0.6)]
        cape = [(-2.4, -8.0), (2.4, -8.0), (8.0, 0.0), (13.0, 10.5), (9.0, 9.4), (5.5, 11.6), (1.5, 10.6),
                (-3.5, 11.2), (-7.0, 10.0), (-6.0, 1.0)]
    elif pose == 'tumble':
        # 2026-11-08 r2：飛ばされて回る形（ひじ・ひざを曲げる → 空の小ささでも頭・手足が読める。飛行機の形にならない）
        caps = [((-2.6, 16.4), (-6.2, 10.2), 1.2, 1.35, -0.4, 0.0),    # すね（足先が p0）
                ((10.6, 12.2), (6.4, 8.8), 1.2, 1.35, -0.4, 0.6),
                ((0.0, 4.6), (0.0, -6.8), 1.75, 2.15, 0.0, 0.9),
                ((0.0, -6.8), (0.0, -9.0), 0.95, 0.95, 0.0, 0.5),
                ((-10.8, -3.0), (-7.4, -9.6), 0.95, 1.05, 0.6, 0.7),   # 前腕（手が p0）
                ((11.0, -15.4), (7.6, -10.6), 0.95, 1.05, 0.6, 0.7),
                ((-6.2, 10.2), (-1.2, 4.2), 1.35, 1.55, -0.4, 0.6),     # もも
                ((6.4, 8.8), (1.2, 4.2), 1.35, 1.55, -0.4, 0.6),
                ((-7.4, -9.6), (-1.0, -6.0), 1.05, 1.15, 0.6, 0.7),    # 上腕
                ((7.6, -10.6), (1.0, -6.0), 1.05, 1.15, 0.6, 0.7)]
        cape = [(-2.6, -7.6), (2.6, -7.6), (8.0, 9.0), (6.0, 18.0), (2.0, 15.5), (-2.0, 19.0), (-6.5, 12.0)]
    else:
        caps = [((-7.0, 14.5), (0.0, 4.2), 1.25, 1.55, -0.4, 0.0),
                ((6.5, 15.0), (0.0, 4.2), 1.25, 1.55, -0.4, 0.6),
                ((0.0, 4.6), (0.0, -6.8), 1.75, 2.15, 0.0, 0.9),
                ((0.0, -6.8), (0.0, -9.0), 0.95, 0.95, 0.0, 0.5),
                ((-9.5, -12.5), (0.0, -6.0), 1.0, 1.15, 0.6, 0.7),
                ((9.5, -12.0), (0.0, -6.0), 1.0, 1.15, 0.6, 0.7)]
        cape = [(-2.6, -7.6), (2.6, -7.6), (8.0, 9.0), (6.0, 18.0), (2.0, 15.5), (-2.0, 19.0), (-6.5, 12.0)]
    return caps, ((0.0, -12.2), 3.7), cape


def villain_relief(x, y, scale, ang=0.0, pose='fly', t=0.0, squash=1.0, M2=None, cape_ang=None, s=3,
                   light=LIGHT, env=None):
    """_art.villain_rgba と同じ (rgba, x0, y0)。env：dict(sky=, ground=, sun=, rim=)（線形の環境の光）。"""
    ext = 1.0 if M2 is None else max(1.0, float(np.abs(np.linalg.eigvals(np.asarray(M2, float))).max()))
    Rr = int(np.ceil((24 * scale + 6) * ext + 4))
    fx, fy = x - np.floor(x), y - np.floor(y)
    cx, cy = Rr + fx, Rr + fy
    S = np.array([[squash, 0], [0, 1 / max(squash, 0.6) ** 0.5]])
    Rm = _R2(ang)
    if M2 is not None:
        Rm = np.asarray(M2, float) @ Rm
    A = Rm @ S
    det = math.sqrt(abs(np.linalg.det(A)))
    T = lambda q: A @ np.asarray(q, float) * scale + (cx, cy)
    ss = int(np.clip(round(18.0 / max(scale, 0.5)), 2, 4)) if scale < 4 else s
    rel = R.Relief(2 * Rr, 2 * Rr, ss)
    caps, (hc, hr), cape = _villain_parts(pose)
    fl = math.sin(t * 9.0) * 1.2 + math.sin(t * 15.0 + 1.0) * 0.6
    # マント（いちばん奥）：布のしわ
    if pose == 'stand':
        cp = [(px + (fl * 0.3 if k in (2,) else fl if k == 3 else fl * 0.7 if k == 4 else fl * 0.5 if k == 5 else 0.0), py)
              for k, (px, py) in enumerate(cape)]
        P = np.array([T(q) for q in cp])
    else:
        cp = [(px + (fl if k in (2, 3) else fl * 0.6 if k == 4 else fl * 0.4 if k == 5 else fl * 0.3 if k == 6 else 0.0), py)
              for k, (px, py) in enumerate(cape)]
        if cape_ang is None:
            P = np.array([T(q) for q in cp])
        else:
            sh = T((0.0, -7.6))
            Rc = (np.asarray(M2, float) if M2 is not None else np.eye(2)) @ _R2(cape_ang - 90.0)
            P = np.array([sh + Rc @ (np.array(q) - (0.0, -7.6)) * scale for q in cp])
    sc = scale * det
    ax_ = P[3] - P[0]
    an = ax_ / (np.hypot(*ax_) + 1e-6)

    def folds(X, Y):
        u = (X - P[0][0]) * an[1] - (Y - P[0][1]) * an[0]
        return 0.55 * sc * np.sin(u / (1.3 * sc) + t * 4.0) ** 2
    rel.blob(P, 1.6 * sc, z0=-4.0 * sc, mat=2, bevel=4.0 * sc, bump=folds)
    # 体
    for (p0, p1, r0, r1, z0, k) in caps:
        rel.capsule(T(p0), T(p1), r0 * sc, r1 * sc, z0 * sc, mat=1, k=k * sc)
    hcs = T(hc)
    rel.ellipse(hcs, hr * scale, hr * scale, 0.0, z0=0.4 * sc, mat=1, k=0.0,
                A=None if M2 is None else np.asarray(M2, float) / max(det, 1e-6) ** 0 / 1.0)
    env = env or {}
    rgba = rel.render({1: BODY, 2: CAPE}, light=light, ao=0.55, ao_r=max(2.0, 2.2 * sc) * ss, **env)
    return rgba, int(np.floor(x)) - Rr, int(np.floor(y)) - Rr


# ---------------------------------------------------------------- 拳
FIST_SKIN = dict(rgb=core.c('E6F1FA'), spec=0.32, rough=0.34, sss=0.6, rimk=1.0)
FIST_NAIL = dict(rgb=core.c('F4F9FD'), spec=0.7, rough=0.18, sss=0.6, rimk=0.8)
FIST_CUFF = dict(rgb=core.c('B4D0E6'), spec=0.06, rough=0.75, sss=0.4, rimk=0.6)


def fist_relief(ang, scale=1.0, pad=40, s=3, light=LIGHT, env=None):
    """_art._fist_shapes を 3D に。(rgba, ax, ay)：ax, ay は拳のまんなか（ローカル原点）の板の中の位置。"""
    from cutB.beats import _art as A
    sh = A._fist_shapes()
    Rm = _R2(ang)
    pts = np.vstack(list(sh.values())) @ Rm.T * scale
    x0, y0 = pts[:, 0].min() - pad, pts[:, 1].min() - pad
    w, h = int(np.ceil(pts[:, 0].max() + pad - x0)), int(np.ceil(pts[:, 1].max() + pad - y0))
    T = lambda q: Rm @ np.asarray(q, float) * scale + (-x0, -y0)
    rel = R.Relief(w, h, s)
    sc = scale
    # 袖口：布（細いリブ）
    cuff = np.array([T(q) for q in sh['cuff']])
    d = Rm @ np.array([1.0, 0.0])

    def rib(X, Y):
        u = (X + x0) * d[0] + (Y + y0) * d[1]
        return 1.6 * sc * np.cos(u / (3.2 * sc)) ** 2
    rel.blob(cuff, 24 * sc, z0=-6 * sc, mat=2, bevel=22 * sc, bump=rib)
    # 手の甲（丸い箱）
    body = np.array([T(q) for q in sh['body']])
    rel.blob(body, 34 * sc, z0=0.0, mat=1, bevel=30 * sc, k=4 * sc)
    # 曲げた指（前の関節の山）：短いカプセルを手の甲の前に。指どうしの間はなめらかにしない（溝が出る）
    for j in range(4):
        yy = (-37.5, -12.5, 12.5, 37.5)[j]
        a0, a1 = np.array([20.0, yy]), np.array([41 - 3.5 * abs(j - 1.5), yy])
        zj = (10, 13, 13, 10)[j] * sc
        rel.capsule(T(a0), T(a1), 13.0 * sc, 12.6 * sc, z0=zj, mat=1, k=0.0)
    # 親指（いちばん手前）と爪
    rel.capsule(T((-30, 31)), T((20, 22)), 12.5 * sc, 11.5 * sc, z0=24 * sc, mat=1, k=5 * sc)
    rel.ellipse(T((22, 20.5)), 6.5 * sc, 5.0 * sc, ang + 8, z0=30 * sc, mat=3, flat=0.35)
    env = env or {}
    rgba = rel.render({1: FIST_SKIN, 2: FIST_CUFF, 3: FIST_NAIL}, light=light, ao=0.7, ao_r=9 * sc * s, **env)
    return rgba, -x0, -y0


# ---------------------------------------------------------------- ゾウ
ELE = dict(rgb=core.c('A9D6F0'), spec=0.22, rough=0.42, sss=0.45, rimk=1.0)
TUSK = dict(rgb=core.c('F2F8FC'), spec=0.5, rough=0.25, sss=0.5, rimk=0.6)


def _spline(P, k=10):
    P = np.asarray(P, float)
    Q = np.concatenate([P[:1], P, P[-1:]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        t = np.linspace(0, 1, k, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 +
                          (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1:])
    return np.concatenate(out)


def elephant_relief(esc=0.82, pad=8, s=4, light=LIGHT, env=None):
    """b05.elephant_sprite と同じ形・同じ戻り値 (rgba, ax, ay)（ax, ay は足もとの中心）。"""
    Wd, Hd = int(108 * esc) + 2 * pad, int(74 * esc) + 2 * pad
    ox, oy = pad, Hd - pad
    P = lambda pts: np.array([(ox + x * esc, oy + y * esc) for x, y in pts], float)
    rel = R.Relief(Wd, Hd, s)
    e = esc
    # 奥の脚（暗く・奥）
    for x0, x1 in ((47, 57), (86, 96)):
        rel.blob(P([(x0, -34), (x1, -34), (x1 - 0.8, -3), (x1 + 0.6, 0), (x0 - 0.6, 0), (x0 + 0.8, -3)]), 3.5 * e,
                 z0=-3 * e, mat=1, bevel=3.5 * e)
    # 胴
    back = _spline([(30, -40), (34, -60), (48, -65), (62, -60), (76, -61), (90, -55), (98, -42), (97, -26),
                    (86, -20), (60, -21), (40, -22), (30, -30)], 8)
    rel.blob(P(back), 15 * e, z0=0, mat=1, bevel=15 * e)
    # 手前の脚
    for x0, x1 in ((34, 45), (75, 85)):
        rel.blob(P([(x0, -36), (x1, -36), (x1 - 0.8, -3), (x1 + 0.6, 0), (x0 - 0.6, 0), (x0 + 0.8, -3)]), 5.0 * e,
                 z0=2 * e, mat=1, bevel=5.0 * e, k=2 * e)
    # 尾
    rel.capsule(P([(97, -48)])[0], P([(101.5, -30)])[0], 0.9 * e, 0.7 * e, z0=2 * e, mat=1)
    # 頭
    rel.ellipse(P([(28.5, -53)])[0], 15.5 * e, 17 * e, 0, z0=1 * e, mat=1, k=3 * e, flat=0.8)
    rel.blob(P([(16, -50), (28, -34), (40, -36), (38, -56)]), 9 * e, z0=1 * e, mat=1, bevel=8 * e, k=3 * e)
    # 鼻（太さが先へ細くなる）
    tr = _spline([(19, -46), (12, -36), (9, -24), (9, -12), (11, -4), (15, -3)], 10)
    Q = P(tr)
    for k in range(len(Q) - 1):
        r0 = (5.2 - 2.6 * k / (len(Q) - 1)) * e
        r1 = (5.2 - 2.6 * (k + 1) / (len(Q) - 1)) * e
        rel.capsule(Q[k], Q[k + 1], r0, r1, z0=6 * e, mat=1, k=1.5 * e)
    # 耳（重なる板：縁が少し反る）
    ear = _spline([(36, -66), (50, -68), (58, -56), (56, -40), (50, -30), (43, -27), (38, -36), (35, -50), (36, -66)], 8)
    rel.blob(P(ear), 3.0 * e, z0=13 * e, mat=1, bevel=4.0 * e)
    # 牙
    tk = P(_spline([(22, -38), (18, -32), (12, -30), (7, -31)], 6))
    for k in range(len(tk) - 1):
        rel.capsule(tk[k], tk[k + 1], 1.5 * e * (1 - 0.1 * k), 1.5 * e * (1 - 0.1 * (k + 1)), z0=11 * e, mat=2)
    env = env or {}
    rgba = rel.render({1: ELE, 2: TUSK}, light=light, ao=0.65, ao_r=5 * e * s, **env)
    return rgba, pad + 55 * esc, Hd - pad
