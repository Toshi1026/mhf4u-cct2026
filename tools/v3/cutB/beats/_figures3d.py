"""案B の人形を SDF（_sdf.py）で「本当の 3D」に：顔のない悪役・拳だけのヒーロー・ゾウ。look-dev 2026-11。

どれも既存の 2D の形（_figures._villain_parts、_art._fist_shapes、b05 のゾウ）と同じ大きさ・同じ向きに重なるように組んだ。
オリジナルの形（顔・触角・角・柄なし。拳は袖口＋グー）。白いビニールのような材質＋布のマント。

  villain_rgba(x, y, scale, ang=0, pose='fly', t=0, squash=1, M2=None, cape_ang=None, light=, env=, ss=None)
      → (rgba, x0, y0)   _art.villain_rgba と同じ引数・戻り値（そのまま差し替えられる）
  fist_rgba(ang, scale, light=, env=, ss=3) → (rgba, ax, ay)    ax, ay：拳のまんなか（ローカル原点）の板の中の位置
  elephant_rgba(esc=0.82, yaw=-22, pitch=8, light=, env=, ss=4) → (rgba, ax, ay)   ax, ay：足もとの中心
"""
import math

import numpy as np

import core
from cutB.beats import _sdf as S
from cutB.beats import _figures as F

LIGHT = (-0.62, -0.50, 0.45)
BODY = S.Mat(core.c('EEF5FB'), spec=0.55, rough=0.30, sss=0.30, rim=1.0, coat=0.35)
CAPE = S.Mat(core.c('B4D6EE'), spec=0.12, rough=0.70, sss=0.45, rim=0.75, sheen=0.7)


def _R2(a):
    r = math.radians(a)
    return np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])


# ---------------------------------------------------------------- 悪役
def villain_scene(pose, t, cape_local=None):
    """ローカル単位（_figures._villain_parts と同じ：胴のまんなかが原点、+y 下、全長 約 34）。"""
    sc = S.Scene()
    caps, (hc, hr), cape = F._villain_parts(pose)
    fl = math.sin(t * 9.0) * 1.2 + math.sin(t * 15.0 + 1.0) * 0.6
    # マント（いちばん奥の布）：しわは時間で流れる
    if cape_local is None:
        if pose == 'stand':
            wob = {2: 0.3, 3: 1.0, 4: 0.7, 5: 0.5}
        else:
            wob = {2: 1.0, 3: 1.0, 4: 0.6, 5: 0.4, 6: 0.3}
        cape_local = np.array([(px + fl * wob.get(k, 0.0), py) for k, (px, py) in enumerate(cape)])
    ax_ = cape_local[3] - cape_local[0]
    an = ax_ / (np.hypot(*ax_) + 1e-6)
    sc.slab(cape_local, z=-2.5, half=0.32, r=0.28, k=0.0, mat=1, fold=(0.55, 1.3, t * 4.0, an[1], -an[0]),
            bend=-0.006)
    # 体：手足は太さの変わるカプセル（関節はなめらかにつなぐ）
    for (p0, p1, r0, r1, z0, k) in caps:
        sc.cone((p0[0], p0[1], z0 * 0.8), (p1[0], p1[1], z0 * 0.8), r0, r1, k=0.9 + k * 0.6, mat=0)
    # 胸（少し厚い）・腰
    sc.ellipsoid((0.0, -4.4, 0.25), (2.55, 3.0, 2.1), k=1.4, mat=0)
    sc.ellipsoid((0.0, 3.6, 0.0), (2.1, 1.6, 1.8), k=1.2, mat=0)
    # 手・足の先（小さな丸）
    for (p0, p1, r0, r1, z0, k) in caps:
        if pose == 'stand' and abs(p0[1] - 16.0) < 0.1:
            sc.ellipsoid((p0[0] + (0.6 if p0[0] > 0 else -0.6), p0[1] + 0.6, 0.6), (1.7, 0.95, 1.9), k=0.8, mat=0)
    if pose != 'stand':
        for (p0, p1, r0, r1, z0, k) in caps[4:6]:
            sc.sphere((p0[0], p0[1], z0 * 0.8), r0 * 1.22, k=0.6, mat=0)
        for (p0, p1, r0, r1, z0, k) in caps[:2]:
            sc.ellipsoid((p0[0], p0[1] + 0.4, z0 * 0.8 + 0.5), (1.5, 1.0, 1.9), k=0.8, mat=0)
    # 頭：顔のないなめらかな卵形＋首もとのえり（マントの留め）
    sc.ellipsoid((hc[0], hc[1], 0.35), (hr * 0.94, hr * 1.04, hr * 0.92), k=1.0, mat=0)
    sc.ellipsoid((0.0, -7.4, -0.3), (2.6, 0.9, 2.2), k=0.7, mat=1)
    return sc


def villain_rgba(x, y, scale, ang=0.0, pose='fly', t=0.0, squash=1.0, M2=None, cape_ang=None, light=LIGHT,
                 env=None, ss=None, s=None):
    """_art.villain_rgba と同じ (rgba, x0, y0)。"""
    ext = 1.0 if M2 is None else max(1.0, float(np.abs(np.linalg.eigvals(np.asarray(M2, float))).max()))
    Rr = int(np.ceil((25 * scale + 6) * ext + 4))
    fx, fy = x - np.floor(x), y - np.floor(y)
    cx, cy = Rr + fx, Rr + fy
    Sq = np.array([[squash, 0], [0, 1 / max(squash, 0.6) ** 0.5]])
    Rm = _R2(ang)
    if M2 is not None:
        Rm = np.asarray(M2, float) @ Rm
    A = Rm @ Sq * scale
    det = math.sqrt(abs(np.linalg.det(A)))
    cape_local = None
    if cape_ang is not None and pose != 'stand':
        # 2D 版と同じ：マントは肩の点を軸に画面の向きで回す → ローカルへ戻す
        caps, _, cape = F._villain_parts(pose)
        fl = math.sin(t * 9.0) * 1.2 + math.sin(t * 15.0 + 1.0) * 0.6
        wob = {2: 1.0, 3: 1.0, 4: 0.6, 5: 0.4, 6: 0.3}
        cp = np.array([(px + fl * wob.get(k, 0.0), py) for k, (px, py) in enumerate(cape)])
        Mi = np.linalg.inv(A)
        shl = np.array([0.0, -7.6])
        sh_s = A @ shl
        Rc = (np.asarray(M2, float) if M2 is not None else np.eye(2)) @ _R2(cape_ang - 90.0)
        scr = np.array([sh_s + Rc @ (q - shl) * scale for q in cp])
        cape_local = scr @ Mi.T
    sc = villain_scene(pose, t, cape_local)
    if ss is None:
        ss = int(np.clip(round(10.0 / max(det, 0.5)), 2, 4))
    env = env or {}
    rgba, _z = S.render(sc, [BODY, CAPE], 2 * Rr, 2 * Rr, A, (cx, cy), zs=det, ss=ss, light=light, env=env,
                        shadow_k=7.0, ao_d=0.9)
    return rgba, int(np.floor(x)) - Rr, int(np.floor(y)) - Rr


# ---------------------------------------------------------------- 拳
SKIN = S.Mat(core.c('E9F3FB'), spec=0.45, rough=0.34, sss=0.45, rim=1.0, coat=0.25)
NAIL = S.Mat(core.c('F6FAFE'), spec=0.9, rough=0.16, sss=0.3, rim=0.8, coat=0.6)
CUFF = S.Mat(core.c('A9CBE6'), spec=0.10, rough=0.75, sss=0.35, rim=0.6, sheen=0.8)


def fist_scene():
    """_art._fist_shapes と同じローカル単位（+x が殴る向き、y 下、z 手前）。"""
    sc = S.Scene()
    # 袖口：太い布の筒（リブは z の波）。手首から外へ少し広がる
    cuff = [(-44, -44), (-44, 46), (-80, 60), (-88, 54), (-88, -52), (-80, -58)]
    sc.slab(cuff, z=-2.0, half=34.0, r=24.0, k=0.0, mat=2, fold=(1.8, 3.4 / 1.0, 0.0, 0.0, 1.0))
    # 手の甲：やわらかい箱
    sc.rbox((-12.0, 1.0, 0.0), (40.0, 46.0, 30.0), 24.0, k=0.0, mat=0)
    # 曲げた 4 本の指：根元（手の甲の中）→ 第二関節（前の山）→ 指先（下へ巻き込む）
    for j, yy in enumerate((-37.5, -12.5, 12.5, 37.5)):
        tip = 46 - 3.5 * abs(j - 1.5)
        zj = (8.0, 11.0, 11.0, 8.0)[j]
        sc.cone((6.0, yy, zj), (tip, yy, zj + 1.0), 13.0, 12.4, k=1.6, mat=0)
        sc.cone((tip, yy, zj + 1.0), (tip - 10.0, yy + 1.0, zj + 15.0), 12.0, 10.5, k=2.2, mat=0)
    # 指のあいだの細い溝（削る）
    for yy in (-25.0, 0.0, 25.0):
        sc.cone((10.0, yy, 26.0), (52.0, yy, 22.0), 1.6, 1.6, k=1.4, mat=0, op='sub')
    # 親指（いちばん手前）と爪
    sc.cone((-34.0, 30.0, 18.0), (20.0, 21.0, 30.0), 13.0, 11.5, k=6.0, mat=0)
    sc.ellipsoid((23.5, 20.0, 39.0), (7.0, 5.6, 2.4), k=0.6, mat=1)
    return sc


def fist_rgba(ang, scale=1.0, pad=36, light=LIGHT, env=None, ss=3):
    sc = fist_scene()
    Rm = _R2(ang)
    A = Rm * scale
    x0, y0, x1, y1, _, _ = sc.bounds()
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) @ A.T
    mn, mx = corners.min(0) - pad, corners.max(0) + pad
    w, h = int(np.ceil(mx[0] - mn[0])), int(np.ceil(mx[1] - mn[1]))
    c = (-mn[0], -mn[1])
    rgba, _z = S.render(sc, [SKIN, NAIL, CUFF], w, h, A, c, zs=scale, ss=ss, light=light, env=env or {},
                        shadow_k=5.0, ao_d=7.0)
    return rgba, c[0], c[1]


# ---------------------------------------------------------------- ゾウ
ELE = S.Mat(core.c('A6D3EE'), spec=0.30, rough=0.46, sss=0.35, rim=1.0, coat=0.15)
TUSK = S.Mat(core.c('F3F8FC'), spec=0.7, rough=0.22, sss=0.5, rim=0.6, coat=0.4)


def _rot(P, yaw, pitch, piv):
    """ローカルの点（x, y, z）を縦の軸（y）まわりに yaw、横の軸（x）まわりに pitch 回す（piv が中心）。"""
    P = np.asarray(P, float) - piv
    a, b = math.radians(yaw), math.radians(pitch)
    x = P[..., 0] * math.cos(a) + P[..., 2] * math.sin(a)
    z = -P[..., 0] * math.sin(a) + P[..., 2] * math.cos(a)
    y = P[..., 1] * math.cos(b) - z * math.sin(b)
    z2 = P[..., 1] * math.sin(b) + z * math.cos(b)
    return np.stack([x, y, z2], -1) + piv


def elephant_scene(yaw=-22.0, pitch=8.0):
    """b05 のゾウ（足もと＝y 0、上が -y、全長 約 100 単位、頭が左）。yaw／pitch で少し斜めから。"""
    sc = S.Scene()
    piv = np.array([55.0, -35.0, 0.0])
    R = lambda p: _rot(p, yaw, pitch, piv)

    def cone(a, b, ra, rb, k=0.0, mat=0, op='add'):
        sc.cone(R(a), R(b), ra, rb, k=k, mat=mat, op=op)

    def ell(c, rr, k=0.0, mat=0, op='add'):
        sc.ellipsoid(R(c), rr, k=k, mat=mat, op=op)
    # 胴・背中・おしり
    ell((63.0, -42.0, 0.0), (34.0, 21.0, 19.0), k=0.0)
    ell((50.0, -47.0, 0.0), (22.0, 19.0, 18.0), k=8.0)
    ell((86.0, -44.0, 0.0), (16.0, 17.0, 17.0), k=8.0)
    # 脚（手前の 2 本は z+、奥の 2 本は z-）＋足の裏
    for (xa, zz, r0) in ((40.0, 9.0, 6.8), (52.0, -9.0, 6.4), (80.0, 9.0, 7.0), (90.0, -9.0, 6.6)):
        cone((xa, -38.0, zz), (xa + 0.5, -2.5, zz), r0, r0 * 0.86, k=5.0)
        ell((xa + 0.8, -2.2, zz), (r0 * 1.02, 2.4, r0 * 1.0), k=1.5)
    # 尾と房
    cone((99.0, -50.0, 0.0), (103.0, -28.0, 2.0), 1.3, 0.8, k=1.0)
    ell((103.2, -26.5, 2.0), (1.6, 2.6, 1.6), k=0.6)
    # 頭（額のふくらみ）
    ell((28.5, -53.0, 0.0), (15.0, 16.0, 14.0), k=6.0)
    ell((25.0, -61.0, 0.0), (11.0, 8.0, 11.0), k=5.0)
    # 鼻：太さが先へ細くなる、先は少し前へ巻く
    pts = F._spline([(19, -46), (12, -36), (9, -24), (9, -12), (11, -4), (15, -3)], 6)
    n = len(pts)
    for kk in range(n - 1):
        r0 = 5.6 - 3.0 * kk / (n - 1)
        r1 = 5.6 - 3.0 * (kk + 1) / (n - 1)
        z0 = 2.0 + 3.0 * kk / (n - 1)
        z1 = 2.0 + 3.0 * (kk + 1) / (n - 1)
        cone((pts[kk][0], pts[kk][1], z0), (pts[kk + 1][0], pts[kk + 1][1], z1), r0, r1, k=1.8)
    # 鼻のしわ（細い溝をいくつか）
    for kk in range(3, n - 3, 2):
        p = pts[kk]
        cone((p[0] - 6.5, p[1], 9.0), (p[0] + 6.5, p[1] + 0.6, 9.0), 0.55, 0.55, k=0.8, op='sub')
    # 目（小さなくぼみ）
    ell((22.0, -56.0, 12.2), (1.3, 1.1, 1.3), k=0.8, op='sub')
    # 耳：手前の大きな耳（少しお椀型に反る板）
    ear = F._spline([(36, -66), (50, -68), (58, -56), (56, -40), (50, -30), (43, -27), (38, -36), (35, -50),
                     (36, -66)], 4)
    ear3 = np.array([R((px, py, 13.0)) for px, py in ear])
    sc.slab(ear3[:, :2], z=float(ear3[:, 2].mean()), half=1.1, r=1.0, k=1.5, mat=0, bend=-0.0045)
    # 牙
    tk = F._spline([(22, -38), (18, -32), (12, -30), (7, -31)], 3)
    for kk in range(len(tk) - 1):
        cone((tk[kk][0], tk[kk][1], 9.0), (tk[kk + 1][0], tk[kk + 1][1], 10.0), 1.8 * (1 - 0.12 * kk),
             1.8 * (1 - 0.12 * (kk + 1)), k=0.6, mat=1)
    return sc


def elephant_rgba(esc=0.82, yaw=-22.0, pitch=8.0, pad=8, light=LIGHT, env=None, ss=4):
    sc = elephant_scene(yaw, pitch)
    x0, y0, x1, y1, _, _ = sc.bounds()
    A = np.eye(2) * esc
    w = int(np.ceil((x1 - x0) * esc)) + 2 * pad
    h = int(np.ceil((y1 - y0) * esc)) + 2 * pad
    c = (pad - x0 * esc, pad - y0 * esc)
    rgba, z = S.render(sc, [ELE, TUSK], w, h, A, c, zs=esc, ss=ss, light=light, env=env or {}, shadow_k=6.0,
                       ao_d=2.6)
    # 足もとの中心（y=0 の高さ、x=55 のあたり）
    foot = _rot(np.array([55.0, 0.0, 0.0]), yaw, pitch, np.array([55.0, -35.0, 0.0]))
    return rgba, c[0] + foot[0] * esc, c[1] + foot[1] * esc
