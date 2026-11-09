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
# 2026-11-08 美術監督：白い体は砂の上で読めない＋「白い禿頭にマント」は有名な型に近い → スレートのサテン（GGX 0.45）＋短いスカーフ
SLATE = S.Mat(core.c('8AA2BC'), spec=0.55, rough=0.45, sss=0.04, rim=1.0, coat=0.12)
SCARF = S.Mat(core.c('CFE6F5'), spec=0.10, rough=0.75, sss=0.40, rim=0.8, sheen=0.6)
RIM_ENV = dict(rim_dir=(0.70, -0.70))     # リムは右上から（キーの光は左上）


def _R2(a):
    r = math.radians(a)
    return np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])


# ---------------------------------------------------------------- 悪役
def villain_scene(pose, t, cape_local=None, scarf=True):
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
    if scarf:
        # 短いスカーフ：首に巻いた輪（下の「えり」）＋うしろへ 2 本の短い端。向きはマントの多角形の向き（速さの反対）のまま
        sh = np.array([0.0, -7.6])
        tip = np.asarray(cape_local[2:], float).mean(0)
        d = tip - sh
        ln = float(np.hypot(*d)) + 1e-6
        d = d / ln
        L_ = min(ln * 0.6, 10.0) if pose != 'tumble' else min(ln * 0.3, 4.5)   # 回って飛ぶときは短く（翼に見えない）
        for side, dev, ww in ((-1, -13.0, 1.0), (1, 9.0, 0.85)):
            r_ = math.radians(dev)
            dd = np.array([d[0] * math.cos(r_) - d[1] * math.sin(r_), d[0] * math.sin(r_) + d[1] * math.cos(r_)])
            nrm = np.array([-dd[1], dd[0]])
            us = np.linspace(0, 1, 9)
            wav = np.sin(us * 4.0 + t * 11.0 + side) * 0.55 * us
            cen = sh + nrm * side * 0.7 + dd[None] * (us[:, None] * L_ * ww) + nrm[None] * wav[:, None]
            wdt = (1.35 - 0.55 * us) * 0.95
            poly = np.vstack([cen + nrm[None] * wdt[:, None], (cen - nrm[None] * wdt[:, None])[::-1]])
            sc.slab(poly, z=-1.4 - 0.5 * side, half=0.22, r=0.22, k=0.0, mat=1, fold=(0.35, 1.3, t * 5.0, dd[1], -dd[0]))
    else:
        ax_ = cape_local[3] - cape_local[0]
        an = ax_ / (np.hypot(*ax_) + 1e-6)
        sc.slab(cape_local, z=-2.6, half=0.30, r=0.28, k=0.0, mat=1, fold=(0.75, 1.1, t * 4.0, an[1], -an[0]),
                bend=-0.004)
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
    sc.ellipsoid((0.0, -7.4, -0.3), (2.6, 0.9, 2.2) if not scarf else (2.5, 1.15, 2.35), k=0.7, mat=1)
    return sc


def villain_rgba(x, y, scale, ang=0.0, pose='fly', t=0.0, squash=1.0, M2=None, cape_ang=None, light=LIGHT,
                 env=None, ss=None, s=None, mats=None, scarf=False):
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
    sc = villain_scene(pose, t, cape_local, scarf=scarf)
    if ss is None:
        ss = int(np.clip(round(10.0 / max(det, 0.5)), 2, 4))
    env = env or {}
    rgba, _z = S.render(sc, mats or [BODY, CAPE], 2 * Rr, 2 * Rr, A, (cx, cy), zs=det, ss=ss, light=light, env=env,
                        shadow_k=7.0, ao_d=0.9)
    return rgba, int(np.floor(x)) - Rr, int(np.floor(y)) - Rr


# ---------------------------------------------------------------- 拳
SKIN = S.Mat(core.c('E9F3FB'), spec=0.45, rough=0.34, sss=0.45, rim=1.0, coat=0.25)
NAIL = S.Mat(core.c('F6FAFE'), spec=0.9, rough=0.16, sss=0.3, rim=0.8, coat=0.6)
CUFF = S.Mat(core.c('A9CBE6'), spec=0.10, rough=0.75, sss=0.35, rim=0.6, sheen=0.8)


def fist_scene():
    """_art._fist_shapes と同じローカル単位（+x が殴る向き、y 下、z 手前）。"""
    sc = S.Scene()
    # 袖口：太い布の筒（輪を重ねてリブ）。手首から外へ少し広がる
    for kk in range(7):
        xa = -46.0 - 6.2 * kk
        r = 47.0 + 1.2 * kk + (1.6 if kk % 2 else 0.0)
        sc.cyl((xa, 1.0, -4.0), (xa - 6.6, 1.0, -4.0), r, rr=3.0, k=2.2, mat=2)
    # 手の甲：やわらかい箱＋ふくらみ
    sc.rbox((-12.0, 1.0, 0.0), (38.0, 45.0, 28.0), 26.0, k=0.0, mat=0)
    sc.ellipsoid((-14.0, 1.0, 6.0), (40.0, 44.0, 30.0), k=10.0, mat=0)
    # 曲げた 4 本の指：根元（手の甲の中）→ 第二関節（前の山）→ 指先（下へ巻き込む）
    for j, yy in enumerate((-37.5, -12.5, 12.5, 37.5)):
        tip = 46 - 3.5 * abs(j - 1.5)
        zj = (8.0, 11.0, 11.0, 8.0)[j]
        sc.cone((4.0, yy, zj), (tip, yy, zj + 1.0), 12.6, 12.2, k=3.0, mat=0)
        sc.cone((tip, yy, zj + 1.0), (tip - 12.0, yy + 1.0, zj + 14.0), 11.8, 10.6, k=2.0, mat=0)
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


def elephant_scene(yaw=20.0, pitch=8.0, trunk=0):
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
        cone((xa, -38.0, zz), (xa + 0.5, -4.0, zz), r0, r0 * 0.9, k=5.0)
        sc.cyl(R((xa + 0.5, -6.0, zz)), R((xa + 0.6, 0.0, zz)), r0 * 0.98, rr=1.6, k=2.0, mat=0)
    # 尾と房
    cone((99.0, -50.0, 0.0), (103.0, -28.0, 2.0), 1.3, 0.8, k=1.0)
    ell((103.2, -26.5, 2.0), (1.6, 2.6, 1.6), k=0.6)
    # 頭（額のふくらみ）
    ell((28.5, -53.0, 0.0), (15.0, 16.0, 14.0), k=6.0)
    ell((25.0, -61.0, 0.0), (11.0, 8.0, 11.0), k=5.0)
    # 鼻：太さが先へ細くなる、先は少し前へ巻く
    TR = {0: [(19, -46), (12, -36), (9, -24), (9, -12), (11, -4), (15, -3)],
          1: [(19, -46), (11, -37), (7, -26), (6, -15), (9, -8), (14, -9)],          # 先を少し巻き上げる
          2: [(19, -46), (13, -37), (11, -25), (12, -13), (15, -6), (19, -5)]}       # 少し前へ
    pts = F._spline(TR.get(trunk, TR[0]), 6)
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
    ear3 = np.array([R((px, py, 21.0)) for px, py in ear])
    sc.slab(ear3[:, :2], z=float(ear3[:, 2].mean()), half=1.1, r=1.0, k=1.5, mat=0, bend=-0.0045)
    # 牙
    tk = F._spline([(22, -38), (18, -32), (12, -30), (7, -31)], 3)
    for kk in range(len(tk) - 1):
        cone((tk[kk][0], tk[kk][1], 9.0), (tk[kk + 1][0], tk[kk + 1][1], 10.0), 1.8 * (1 - 0.12 * kk),
             1.8 * (1 - 0.12 * (kk + 1)), k=0.6, mat=1)
    return sc


def elephant_rgba(esc=0.82, yaw=20.0, pitch=8.0, pad=8, light=LIGHT, env=None, ss=4, trunk=0):
    sc = elephant_scene(yaw, pitch, trunk)
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


# ---------------------------------------------------------------- 2026-11-08 美術監督の版（01・02・06・07 で共通）
KEY = (-0.55, -0.65, 0.52)                     # 全ショット共通のキーの光（画面の向き、左上・方位 315°・高さ 40°）


def _lin(h):
    c = np.asarray(core.c(h), np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


ENV_DAY = dict(sun=np.array([1.0, 0.99, 0.975]) * 1.45, sky=_lin('A9CCEA') * 0.36, ground=_lin('B7C3CC') * 0.16,
               rim=_lin('CFEFFF') * 0.85, rim_dir=(0.70, -0.70))


def cap_white(rgba, top=0.91, knee=0.70):
    """CG の白の上限（≤0.92 sRGB）：knee より上をやわらかく top へ寄せる（色相は保つ：3 チャンネルの最大で縮める）。"""
    rgb = rgba[..., :3]
    m = rgb.max(-1, keepdims=True)
    mc = np.where(m > knee, knee + (top - knee) * (1 - np.exp(-(m - knee) / (top - knee))), m)
    out = rgba.copy()
    out[..., :3] = rgb * (mc / np.maximum(m, 1e-5))
    return out


# 2026-11-08 r2：空を背にして飛ぶとき（体の明るさは地面と同じ、空の側の冷たいリムを強く）
ENV_FLY = dict(ENV_DAY, sky=_lin('B6D6F0') * 0.46, rim=_lin('D8F2FF') * 1.35, rim_dir=(0.15, -1.0))


def villain_v2(x, y, scale, ang=0.0, pose='fly', t=0.0, squash=1.0, M2=None, cape_ang=None, s=None, env=None,
               ss=None):
    """_art.villain_rgba と同じ引数・戻り値。スレートのサテンの体＋短いスカーフ、右上からの冷たいリム。"""
    rgba, x0, y0 = villain_rgba(x, y, scale, ang, pose, t, squash, M2, cape_ang, light=KEY, env=env or ENV_DAY,
                                ss=ss, mats=[SLATE, SCARF], scarf=True)
    return cap_white(rgba), x0, y0


FIST_SKIN = S.Mat(core.c('DCE8F2'), spec=0.45, rough=0.35, sss=0.40, rim=1.0, coat=0.20)
FIST_NAIL = S.Mat(core.c('E8F0F7'), spec=0.8, rough=0.18, sss=0.3, rim=0.8, coat=0.5)
FIST_CUFF = S.Mat(core.c('9DC0DE'), spec=0.10, rough=0.75, sss=0.35, rim=0.6, sheen=0.8)


def fist_v2(ang, scale=1.0, pad=36, env=None, ss=3):
    """拳（2026-11-08）：指のすき間・袖口のリブに細かい環境の遮へい（3px）、F0 0.04・粗さ 0.35、右上のリム。"""
    sc = fist_scene()
    Rm = _R2(ang)
    A = Rm * scale
    x0, y0, x1, y1, _, _ = sc.bounds()
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) @ A.T
    mn, mx = corners.min(0) - pad, corners.max(0) + pad
    w, h = int(np.ceil(mx[0] - mn[0])), int(np.ceil(mx[1] - mn[1]))
    c = (-mn[0], -mn[1])
    rgba, _z = S.render(sc, [FIST_SKIN, FIST_NAIL, FIST_CUFF], w, h, A, c, zs=scale, ss=ss, light=KEY,
                        env=env or ENV_DAY, shadow_k=6.0, ao_d=3.2)
    return cap_white(rgba), c[0], c[1]


# ---------------------------------------------------------------- 2026-11-08 美術監督 r2：本物の拳（fist_v3）
def _Rxyz(yaw, pitch, roll=0.0):
    """3×3 回転：roll（z 軸）→ pitch（x 軸）→ yaw（y 軸）の順に掛ける。"""
    a, b, c = math.radians(yaw), math.radians(pitch), math.radians(roll)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


FIST3_YAW, FIST3_PITCH = 24.0, -34.0
FIST3_G = 2.0         # 手の甲と 4 つの拳頭の列が見える 3/4（上から少し、打つ面が少しこちらへ）


def fist3_scene(yaw=None, pitch=None):
    """手の甲を上にした水平の拳（+x が打つ向き、y 下、z 手前＝親指の側）。単位：拳の高さ ≈ 100。
    4 本の基節（向きのある角の丸い箱、r ≈ 0.13×指の太さの比で丸め、なめらかな和 k ≈ 7）が平らな打つ面（10° 傾ける）をつくり、
    中節は下へ折り込み（関節のしわは細い溝）、親指は人差し指・中指の中節の上を横切る。袖は無地の筒にしわ 2 本。"""
    sc = S.Scene()
    Rg = _Rxyz(FIST3_YAW if yaw is None else yaw, FIST3_PITCH if pitch is None else pitch)
    G = FIST3_G                               # 組む単位（拳の高さ ≈ 48）→ 2D の拳と同じ大きさ（≈ 100）
    P = lambda p: Rg @ (np.asarray(p, float) * G)
    Rb = lambda R_=None: Rg @ (np.eye(3) if R_ is None else R_)
    tilt = math.radians(10.0)                 # 打つ面の傾き（小指の側が少しうしろ）

    def box(c, hh, r, R_=None, k=0.0, mat=0, op='add'):
        sc.obox(P(c), tuple(np.asarray(hh) * G), r * G, Rb(R_), k=k * G, mat=mat, op=op)

    def cone(a, b, ra, rb, k=0.0, mat=0, op='add'):
        sc.cone(P(a), P(b), ra * G, rb * G, k=k * G, mat=mat, op=op)
    zs = (25.5, 8.8, -8.0, -23.5)             # 人差し指 → 小指（z：手前 → 奥）
    wd = (8.0, 8.2, 7.8, 6.8)                 # 指の半分の太さ
    # 手の甲（中手骨）：手首へ少し下がる厚い板＋手のひらのふくらみ
    box((-14.0, 0.5, 0.5), (25.0, 12.5, 33.0), 9.0, _Rxyz(0.0, 0.0, -4.0), k=0.0)
    box((-15.0, 12.0, 6.0), (20.0, 9.0, 28.0), 8.0, k=6.0)
    for j, (z, w) in enumerate(zip(zs, wd)):
        xo = -math.tan(tilt) * (z - 8.8) - (1.5 if j == 3 else 0.0)
        # 拳頭（MCP の頭）：打つ面の上の角に丸い山（手の甲の線より少し高い）
        sc.sphere(P((15.0 + xo, -10.0, z)), (w - 0.2) * G, k=4.0 * G, mat=0)
        # 基節：前の平らな打つ面（拳頭 → PIP、下へ）
        box((17.5 + xo, 3.0, z), (7.0, 15.0, w), 0.14 * 2 * w, k=2.6)
        # 中節：下へ折り込んで手のひらの方へ（うしろ）
        box((9.5 + xo, 20.0, z), (10.5, 6.2, w * 0.95), 0.15 * 2 * w, k=2.4)
        # しわ：中節と手のひらのあいだ、PIP の折れ目
        cone((-1.0 + xo, 24.5, z - w * 0.9), (-1.0 + xo, 24.5, z + w * 0.9), 1.4, 1.4, k=1.0, op='sub')
        cone((22.5 + xo, 14.5, z - w * 0.9), (22.5 + xo, 14.5, z + w * 0.9), 0.8, 0.8, k=0.8, op='sub')
    # 親指：付け根（手のひらの手前の下）→ 基節 → 末節が人差し指・中指の中節の上を横切る
    cone((-24.0, 12.0, 33.0), (-6.0, 22.0, 36.0), 11.0, 9.2, k=5.0)
    cone((-6.0, 22.0, 36.0), (10.0, 28.0, 29.0), 8.8, 8.0, k=2.5)
    cone((10.0, 28.0, 29.0), (15.0, 28.5, 13.0), 7.8, 6.8, k=2.0)
    cone((2.0, 24.0, 38.5), (5.5, 31.5, 31.0), 0.9, 0.9, k=0.7, op='sub')        # 親指の関節のしわ
    # 親指の爪（つやのある平たい箱、末節の手前の面）
    nail = Rb(_Rxyz(-75.0, 15.0, 0.0))
    sc.obox(P((14.0, 26.5, 18.5)), (4.6 * G, 0.7 * G, 3.8 * G), 0.6 * G, nail, k=0.3 * G, mat=1)
    # 手首 → 袖（無地の筒、ゆるいしわ 2 本、袖口は平ら）
    cone((-34.0, 1.0, 1.0), (-48.0, 0.5, 0.0), 14.5, 15.5, k=4.0)
    sc.cyl(P((-43.0, 0.5, 0.0)), P((-135.0, 3.0, -1.5)), 20.5 * G, rr=3.0 * G, k=1.0 * G, mat=2)   # 画面の外まで続く袖
    sc.cyl(P((-52.0, 0.3, 0.0)), P((-57.0, 0.8, 1.0)), 21.0 * G, rr=2.5 * G, k=3.5 * G, mat=2)    # しわ 1（手首のたるみ）
    sc.cyl(P((-74.0, 1.0, -1.0)), P((-80.0, 2.2, 1.5)), 20.9 * G, rr=2.5 * G, k=4.0 * G, mat=2)   # しわ 2（少し斜め）
    return sc


FIST3_SKIN = S.Mat(core.c('D3E0EC'), spec=0.40, rough=0.40, sss=0.12, rim=1.0, coat=0.15)
FIST3_SLEEVE = S.Mat(core.c('7E9DBB'), spec=0.08, rough=0.80, sss=0.10, rim=0.7, sheen=0.7)
FIST3_LIGHT = (0.48, -0.66, 0.58)             # キーは右上（打つ面と拳頭の列を読ませる）
ENV_FIST = dict(sun=np.array([1.0, 0.99, 0.975]) * 1.55, sky=_lin('A9CCEA') * 0.26, ground=_lin('C9D1D7') * 0.14,
                rim=_lin('BFE6FF') * 1.10, rim_dir=(-0.80, 0.45))      # 冷たいリムは左下の輪郭（砂から離す）


def fist_v3(ang, scale=1.0, pad=36, env=None, ss=3):
    """拳（2026-11-08 r2）：fist_v2 と同じ (rgba, ax, ay)。ax, ay は拳のまんなか（打つ面のうしろ）。"""
    sc = fist3_scene()
    A = _R2(ang) * scale
    x0, y0, x1, y1, _, _ = sc.bounds()
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) @ A.T
    mn, mx = corners.min(0) - pad, corners.max(0) + pad
    w, h = int(np.ceil(mx[0] - mn[0])), int(np.ceil(mx[1] - mn[1]))
    c = (-mn[0], -mn[1])
    rgba, _z = S.render(sc, [FIST3_SKIN, FIST_NAIL, FIST3_SLEEVE], w, h, A, c, zs=scale, ss=ss, light=FIST3_LIGHT,
                        env=env or ENV_FIST, shadow_k=7.0, ao_d=3.0)
    # 袖のうしろ（手首から 2.6 倍の所より先）は速さの中へ消える（腕は描かない）
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))], np.float32)
    u = ((xx - c[0]) * d[0] + (yy - c[1]) * d[1]) / scale              # 打つ向きのローカルの距離
    fade = np.clip((u + 215.0) / 95.0, 0, 1)
    rgba[..., 3] *= fade * fade * (3 - 2 * fade)
    return cap_white(rgba), c[0], c[1]


# ---------------------------------------------------------------- 2026-11-08 r2：ゾウ（脚の先細り・ひざ・足の裏の広がり・厚みのある反った耳・鼻のしわ 2 本）
def elephant_scene_v2(yaw=20.0, pitch=8.0, trunk=0, ear=0):
    """elephant_scene と同じ単位・向き。ear：耳の形 0..2（後ろの縁の反り・開き）。"""
    sc = S.Scene()
    piv = np.array([55.0, -35.0, 0.0])
    R = lambda p: _rot(p, yaw, pitch, piv)

    def cone(a, b, ra, rb, k=0.0, mat=0, op='add'):
        sc.cone(R(a), R(b), ra, rb, k=k, mat=mat, op=op)

    def ell(c, rr, k=0.0, mat=0, op='add'):
        sc.ellipsoid(R(c), rr, k=k, mat=mat, op=op)
    ell((63.0, -42.0, 0.0), (34.0, 21.0, 19.0), k=0.0)
    ell((50.0, -47.0, 0.0), (22.0, 19.0, 18.0), k=8.0)
    ell((86.0, -44.0, 0.0), (16.0, 17.0, 17.0), k=8.0)
    ell((62.0, -27.0, 0.0), (26.0, 8.0, 15.0), k=8.0)                    # 腹のたるみ
    # 脚：付け根 r → ひざ 0.9r（少し前へ）→ 足首 0.8r、足の裏は少し広がる
    for (xa, zz, r0, kn) in ((40.0, 9.0, 6.8, 1.2), (52.0, -9.0, 6.4, 1.0), (80.0, 9.0, 7.0, -0.8),
                             (90.0, -9.0, 6.6, -0.6)):
        cone((xa, -38.0, zz), (xa + kn, -19.0, zz), r0, r0 * 0.88, k=5.0)
        cone((xa + kn, -19.0, zz), (xa + 0.3, -5.0, zz), r0 * 0.86, r0 * 0.80, k=2.5)
        sc.cyl(R((xa + 0.3, -5.5, zz)), R((xa + 0.3, -0.6, zz)), r0 * 0.92, rr=1.4, k=2.2, mat=0)   # 足の裏の広がり
        for dz in (-0.55, 0.0, 0.55):                                    # 足の爪（小さなふくらみ）
            sc.sphere(R((xa + 0.3 - r0 * 0.8, -1.6, zz + dz * r0)), 1.3, k=0.6, mat=1)
    cone((99.0, -50.0, 0.0), (103.0, -28.0, 2.0), 1.3, 0.8, k=1.0)
    ell((103.2, -26.5, 2.0), (1.6, 2.6, 1.6), k=0.6)
    ell((28.5, -53.0, 0.0), (15.0, 16.0, 14.0), k=6.0)
    ell((25.0, -61.0, 0.0), (11.0, 8.0, 11.0), k=5.0)
    TR = {0: [(19, -46), (12, -36), (9, -24), (9, -12), (11, -4), (15, -3)],
          1: [(19, -46), (11, -37), (7, -26), (6, -15), (9, -8), (14, -9)],
          2: [(19, -46), (13, -37), (11, -25), (12, -13), (15, -6), (19, -5)]}
    pts = F._spline(TR.get(trunk, TR[0]), 6)
    n = len(pts)
    for kk in range(n - 1):
        r0 = 5.6 - 3.0 * kk / (n - 1)
        r1 = 5.6 - 3.0 * (kk + 1) / (n - 1)
        z0 = 2.0 + 3.0 * kk / (n - 1)
        z1 = 2.0 + 3.0 * (kk + 1) / (n - 1)
        cone((pts[kk][0], pts[kk][1], z0), (pts[kk + 1][0], pts[kk + 1][1], z1), r0, r1, k=1.8)
    # 鼻のしわ 2 本（はっきりした溝：法線のくぼみ）
    for kk in (int(n * 0.35), int(n * 0.55)):
        p, q = pts[kk], pts[min(kk + 1, n - 1)]
        d = np.array(q) - np.array(p)
        d = d / (np.hypot(*d) + 1e-6)
        nrm = np.array([-d[1], d[0]])
        rr = 5.6 - 3.0 * kk / (n - 1)
        a = np.array(p) - nrm * (rr * 0.75)
        b = np.array(p) + nrm * (rr * 0.75)
        zc = 2.0 + 3.0 * kk / (n - 1)
        cone((a[0], a[1], zc + rr * 0.92), (b[0], b[1], zc + rr * 0.92), 0.6, 0.6, k=0.6, op='sub')
    ell((22.0, -56.0, 12.2), (1.3, 1.1, 1.3), k=0.8, op='sub')
    # 耳：厚みのあるお椀型の板＋後ろの縁が反り返る（縁に沿った細い丸い棒を手前へ少し出す）
    EARS = {0: [(36, -66), (50, -68), (58, -56), (56, -40), (50, -30), (43, -27), (38, -36), (35, -50), (36, -66)],
            1: [(36, -67), (52, -70), (61, -58), (59, -41), (52, -29), (44, -25), (38, -35), (35, -50), (36, -67)],
            2: [(36, -65), (48, -66), (56, -55), (55, -41), (49, -32), (43, -29), (38, -37), (35, -50), (36, -65)]}
    ear_ = F._spline(EARS.get(ear, EARS[0]), 4)
    ear3 = np.array([R((px, py, 21.0)) for px, py in ear_])
    sc.slab(ear3[:, :2], z=float(ear3[:, 2].mean()), half=1.7, r=1.4, k=1.5, mat=0, bend=-0.0060)
    m = len(ear_)
    for kk in range(int(m * 0.12), int(m * 0.70)):                       # 後ろ（右）の縁
        a, b = ear_[kk], ear_[kk + 1]
        cone((a[0], a[1], 22.4), (b[0], b[1], 22.4), 1.35, 1.35, k=0.0)
    tk = F._spline([(22, -38), (18, -32), (12, -30), (7, -31)], 3)
    for kk in range(len(tk) - 1):
        cone((tk[kk][0], tk[kk][1], 9.0), (tk[kk + 1][0], tk[kk + 1][1], 10.0), 1.8 * (1 - 0.12 * kk),
             1.8 * (1 - 0.12 * (kk + 1)), k=0.6, mat=1)
    return sc


def elephant_rgba_v2(esc=0.82, yaw=20.0, pitch=8.0, pad=8, light=LIGHT, env=None, ss=4, trunk=0, ear=0):
    """elephant_rgba と同じ戻り値（rgba, ax, ay）。"""
    sc = elephant_scene_v2(yaw, pitch, trunk, ear)
    x0, y0, x1, y1, _, _ = sc.bounds()
    A = np.eye(2) * esc
    w = int(np.ceil((x1 - x0) * esc)) + 2 * pad
    h = int(np.ceil((y1 - y0) * esc)) + 2 * pad
    c = (pad - x0 * esc, pad - y0 * esc)
    rgba, z = S.render(sc, [ELE, TUSK], w, h, A, c, zs=esc, ss=ss, light=light, env=env or {}, shadow_k=6.0,
                       ao_d=2.6)
    foot = _rot(np.array([55.0, 0.0, 0.0]), yaw, pitch, np.array([55.0, -35.0, 0.0]))
    return rgba, c[0] + foot[0] * esc, c[1] + foot[1] * esc
