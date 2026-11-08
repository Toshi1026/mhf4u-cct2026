"""案C 03〜05 のカメラと、動かない「板」（一度だけ描く）。

ショット A（03 の前半）：草原に立つ悪役を 12m 先から、目の高さ 0.9m。01 のアニメ版と同じ構図（地平 y≈1062・足もと y≈1212・x≈606）。
ショット B（03 の中ほど）：真上を見上げる（雲の層にまんまるの穴）。
ショット C（03 の最後〜05）：高い斜めの引き（クレーター・放射状になぎ倒れた草原）。
"""
import math

import cv2
import numpy as np

import core
from core import W, H
from cutB.beats import _render as R, _beach as BE
from cutC.beats import _field as F

hx = R.hexl
CAM_A = R.Cam3D((0.35, 0.9, -12.0), (0.35, 1.445, 0.0), fov=46)
FOCUS_A = 12.0
_C = {}


def rays(cam, scale=1.0):
    h, w = int(cam.h * scale), int(cam.w * scale)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cam.rays((yy + 0.5) / scale, (xx + 0.5) / scale).astype(np.float32)


def horizon_y(cam, x=W / 2):
    P, _ = cam.project(np.array([[cam.eye[0], cam.eye[1], cam.eye[2] + 1e6]]))
    return float(P[0, 1])


def _put_prem(L, prem, a, x0, y0):
    return BE._put_prem(L, prem, a, x0, y0)


def far_hills(img, hz):
    """地平の上の遠い丘（空気で青くかすむ 2 段のシルエット）。img は線形。"""
    xs = np.arange(W, dtype=np.float32)
    for (amp, per, base, seed, c, k) in ((30.0, 260.0, 10.0, 3, hx('8FB4CC'), 0.55), (14.0, 120.0, 3.0, 9, hx('7FA6BE'), 0.65)):
        n = F.fbm1(xs / per, np.full_like(xs, seed * 1.7), 4, seed=seed) * 0.5 + 0.5
        top = hz - base - amp * n
        yy = np.arange(H, dtype=np.float32)[:, None]
        m = np.clip(yy - top[None] + 0.5, 0, 1) * (yy < hz + 2)
        col = c * (1 - k) + F.HAZE * k
        img = img * (1 - m[..., None]) + col * m[..., None]
    return img


def cumulus(img, specs):
    """積雲（_beach._cloud：体積の雲の板）を置く。specs：(x 中心, y 底, 幅, seed, towers, かすみ 0..1)。"""
    for (x, yb, w, sd, tw, hz) in specs:
        prem, a = BE._cloud(int(w), sd, tw, detail=0.6)
        st = prem / np.maximum(a, 1e-4)[..., None]
        st = st * (1 - hz) + F.HAZE * hz
        a = a * (1 - 0.3 * hz)
        img = _put_prem(img, (st * a[..., None]).astype(np.float32), a, x - w / 2, yb - a.shape[0] * 0.86)
    return img


def dof_levels(img, sig=(0.0, 1.6, 3.6, 7.0)):
    return [img if s <= 0 else cv2.GaussianBlur(img, (0, 0), s) for s in sig]


def dof_mix(levels, sigma, sig=(0.0, 1.6, 3.6, 7.0)):
    """画素ごとのぼけ σ（sigma マップ）で、ぼかした板を線形に混ぜる。"""
    out = np.zeros_like(levels[0])
    s = np.clip(sigma, sig[0], sig[-1])
    for k in range(len(sig) - 1):
        a, b = sig[k], sig[k + 1]
        w = np.clip((s - a) / (b - a), 0, 1)
        inside = (s >= a) & (s <= b) if k < len(sig) - 2 else (s >= a)
        wk = np.where(inside, 1 - w, 0)
        wk1 = np.where(inside, w, 0)
        out += levels[k] * wk[..., None] + levels[k + 1] * wk1[..., None]
    return out


def coc(dep, focus, k=24.0, lim=7.0):
    return np.clip(k * np.abs(1.0 / np.maximum(dep, 0.3) - 1.0 / focus), 0, lim)


# ---------------------------------------------------------------- ショット A
SKY_A_CLOUDS = [(150.0, 930.0, 330, 7, 4, 0.25), (930.0, 905.0, 250, 12, 3, 0.30), (560.0, 1012.0, 170, 19, 3, 0.45),
                (380.0, 1030.0, 120, 31, 3, 0.55), (860.0, 1036.0, 110, 37, 3, 0.55), (90.0, 300.0, 210, 41, 3, 0.15),
                (1000.0, 345.0, 240, 43, 4, 0.15)]


def ground_A(cam, D):
    """地面（線形）・深さ・地面のマスク。"""
    tt = -cam.eye[1] / np.minimum(D[..., 1], -1e-5)
    X = cam.eye[0] + D[..., 0] * tt
    Z = cam.eye[2] + D[..., 2] * tt
    fp = tt / cam.fpx / np.maximum(-D[..., 1], 1e-3)
    return X, Z, tt, fp


def plate_A():
    """(back 線形, front_rgba 線形, back の深さ, blades)：back＝空・雲・遠い丘・地面・奥の草（悪役より奥）。
    front＝悪役より手前の草（悪役の上に重ねる）。どちらもピントの外のぼけ込み。"""
    if 'A' in _C:
        return _C['A']
    cam = CAM_A
    D = rays(cam)
    sky = F.sky_rad(np.concatenate([D[..., :1], np.maximum(D[..., 1:2], 0), D[..., 2:]], -1))
    hz = horizon_y(cam)
    sky = cumulus(sky, SKY_A_CLOUDS)
    X, Z, tt, fp = ground_A(cam, D)
    g = (D[..., 1] < -1e-5)
    col = F.light_ground(F.ground_tex(X, Z, fp))
    col = F.aerial(col, tt, 1 / 1800.0)
    img = np.where(g[..., None], col, sky)
    img = far_hills(img, hz)
    dep = np.where(g, tt, 5000.0).astype(np.float32)
    bl = F.Blades(cam, n=70000, zmax=30.0)
    _C['blades'] = bl
    bimg, ba, bd = bl.draw(cam, 0.0, sel_fn=lambda d: d > FOCUS_A)
    bimg = F.light_ground(bimg)
    img = img * (1 - ba[..., None]) + bimg * ba[..., None]
    dep = np.where(ba > 0.5, bd, dep)
    # ピント（悪役 12m）：奥はほとんどぼけない、手前は大きくぼける
    lv = dof_levels(img)
    back = dof_mix(lv, coc(dep, FOCUS_A, lim=7.0))
    front = front_A(0.0)
    _C['A'] = (back.astype(np.float32), front, dep)
    return _C['A']


def front_A(t, flat_fn=None):
    """悪役より手前の草（ぼけ込み、線形 RGBA）。"""
    if 'blades' not in _C:
        plate_A()
    bl = _C['blades']
    fimg, fa, fd = bl.draw(CAM_A, t, flat_fn=flat_fn, sel_fn=lambda d: d <= FOCUS_A, ss=2 if flat_fn is None else 1)
    fimg = F.light_ground(fimg)
    prem = np.dstack([fimg * fa[..., None], fa])
    sig = coc(np.where(fa > 0.02, fd, FOCUS_A), FOCUS_A, lim=7.0)
    # 手前のぼけは前乗せで（縁がにじむ）。ぼかしは半分の大きさで作って戻す（速さ）
    sig_b = cv2.dilate(sig, np.ones((5, 5), np.uint8))
    small = cv2.resize(prem, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    lv = [prem] + [cv2.resize(cv2.GaussianBlur(small, (0, 0), s_ / 2), (W, H), interpolation=cv2.INTER_LINEAR)
                   for s_ in (1.6, 3.6, 7.0)]
    pm = dof_mix(lv, sig_b)
    a = np.clip(pm[..., 3:], 0, 1)
    return np.dstack([pm[..., :3] / np.maximum(a, 1e-4), a]).astype(np.float32)


def flat_A():
    """ショット A の「なぎ倒れた草原」の板（sRGB、ピントのぼけ込み）と、地面の各画素の原点からの距離 r（空は大きな値）。"""
    if 'flatA' in _C:
        return _C['flatA']
    cam = CAM_A
    D = rays(cam)
    X, Z, tt, fp = ground_A(cam, D)
    g = D[..., 1] < -1e-5
    r = np.hypot(X, Z)
    vx, vz = X - cam.eye[0], Z - cam.eye[2]
    vn = np.hypot(vx, vz) + 1e-6
    col = F.ground_tex(X, Z, fp, flat=np.ones_like(X), rad=(X / (r + 1e-3), Z / (r + 1e-3)), view=(vx / vn, vz / vn))
    col = F.aerial(F.light_ground(col), tt, 1 / 1800.0)
    col = R.srgb(col)
    lv = dof_levels(col)
    col = dof_mix(lv, coc(np.where(g, tt, 5000.0), FOCUS_A, lim=7.0))
    rmap = np.where(g, r, 1e6).astype(np.float32)
    _C['flatA'] = (col.astype(np.float32), rmap)
    return _C['flatA']


# ---------------------------------------------------------------- ショット C（高い斜めの引き：クレーター）
C_EYE, C_TGT, C_FOV = (48.0, 118.0, -292.0), (0.0, -4.0, 22.0), 40.0
C_OVER = 1.2                                      # 押し込み用に 1.2 倍の大きさで描く
CR = dict(N=800, cell=0.56, Rr=48.0, depth=21.0)


def cam_C(over=C_OVER):
    fov = math.degrees(2 * math.atan(math.tan(math.radians(C_FOV) / 2) * over))
    return R.Cam3D(C_EYE, C_TGT, fov=fov, w=int(round(W * over)), h=int(round(H * over)))


def crater():
    """(高さ, 放出物, 碗, GX, GZ, 岩のマスク)。F.crater_height に、縁と放出物の毛布・碗の中に散らばる岩（半分うまった丸い岩）を足す。
    05 のゾウの輪（半径 54・62m）の上には大きな岩を置かない。"""
    if 'crater' not in _C:
        Hm, ej, bowl, GX, GZ = F.crater_height(CR['N'], CR['cell'], CR['Rr'], CR['depth'])
        Hm = Hm.copy()
        rock = np.zeros_like(Hm)
        rng = np.random.default_rng(1400)
        n, cell, Rr = CR['N'], CR['cell'], CR['Rr']
        groups = ((150, 0.97, 1.8, 0.9, 3.6), (120, 1.8, 3.6, 0.6, 1.6), (45, 0.1, 0.55, 0.9, 3.0), (90, 3.6, 4.6, 0.4, 1.0))
        for cnt, u0, u1, s0, s1 in groups:
            for _ in range(cnt):
                u = rng.uniform(u0, u1) ** 1.0
                th = rng.uniform(0, 2 * np.pi)
                rad = s0 + (s1 - s0) * rng.uniform(0, 1) ** 2.4
                r = u * Rr
                if 50.5 < r < 66.0 and rad > 0.8:
                    continue
                x, z = r * math.cos(th), r * math.sin(th)
                ci, cj = int(round(z / cell + n / 2 - 0.5)), int(round(x / cell + n / 2 - 0.5))
                k = int(math.ceil(rad / cell)) + 1
                if not (k < ci < n - k and k < cj < n - k):
                    continue
                sub = Hm[ci - k:ci + k + 1, cj - k:cj + k + 1]
                gx, gz = GX[ci - k:ci + k + 1, cj - k:cj + k + 1], GZ[ci - k:ci + k + 1, cj - k:cj + k + 1]
                el = rng.uniform(0.75, 1.0)
                ang = rng.uniform(0, np.pi)
                dx, dz = gx - x, gz - z
                ex = (dx * math.cos(ang) + dz * math.sin(ang)) / rad
                ez = (-dx * math.sin(ang) + dz * math.cos(ang)) / (rad * el)
                d = np.sqrt(ex * ex + ez * ez)
                lump = 1 + 0.18 * np.sin(np.arctan2(ez, ex) * 3 + rng.uniform(0, 6))
                bump = rad * 0.85 * np.sqrt(np.clip(1 - (d / lump) ** 2, 0, 1))
                base = float(sub[k, k]) - 0.3 * rad
                Hm[ci - k:ci + k + 1, cj - k:cj + k + 1] = np.maximum(sub, np.where(bump > 0, base + bump, -1e3))
                rock[ci - k:ci + k + 1, cj - k:cj + k + 1] = np.maximum(rock[ci - k:ci + k + 1, cj - k:cj + k + 1],
                                                                         np.clip((1 - d / lump) * 4, 0, 1))
        _C['crater'] = (Hm.astype(np.float32), ej, bowl, GX, GZ, rock.astype(np.float32))
    return _C['crater']


def _grid_sample(a, X, Z):
    n = a.shape[0]
    cell = CR['cell']
    u = (X / cell + n / 2 - 0.5).astype(np.float32)
    v = (Z / cell + n / 2 - 0.5).astype(np.float32)
    return cv2.remap(a, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# 土の層（sRGB の 16 進）：上から 表土・粘土・砂まじり・れき・岩。どれも灰色がかった土の色（だいだい・黄にしない）
STRATA = [(0.6, '43382D'), (-2.4, '7A6A57'), (-5.2, '948573'), (-6.8, '665F56'), (-9.8, '847C70'), (-11.2, '57524C'),
          (-13.8, '76706A'), (-15.0, '4C4844')]
LOAM, CLAY, ROCK, RUBBLE = hx('463A2F'), hx('7A6C5A'), hx('6E6A64'), hx('382F28')


def _strata(Y):
    """高さ Y（m）の層の色（線形）。境目は少しなめらかに。"""
    col = np.broadcast_to(hx(STRATA[0][1]), Y.shape + (3,)).astype(np.float32)
    for (y0, c) in STRATA[1:]:
        k = R.smoothstep(y0 + 0.35, y0 - 0.35, Y)[..., None]
        col = col * (1 - k) + hx(c) * k
    return col


def plate_C():
    """(sRGB の板 1.2 倍, 深さ, cam)。クレーター（層の見える内壁・盛り上がった縁・岩）・放出物の毛布・放射状になぎ倒れた草原・
    地平の空。_C['C_stand']：草が立ったままの板（衝撃波の前線が通る前）、_C['C_r']：各画素の原点からの距離（m、空は大きな値）。"""
    if 'C' in _C:
        return _C['C']
    cam = cam_C()
    Hm, ej, bowl, GX, GZ, rock = crater()
    cell = CR['cell']
    D = rays(cam)
    hit = F.crater_hits(cam, D, Hm, cell, rb=CR['N'] * cell / 2 - 2)
    t, X, Z, Y = hit[..., 0], hit[..., 1].astype(np.float32), hit[..., 2].astype(np.float32), hit[..., 3]
    g = t > 0
    X = np.where(g, X, 0).astype(np.float32)
    Z = np.where(g, Z, 1e5).astype(np.float32)
    tt = np.where(g, t, 1e5).astype(np.float32)
    fp = tt / cam.fpx / np.maximum(-D[..., 1], 1e-3)
    # 法線・影・くぼみの陰
    gz_, gx_ = np.gradient(Hm, cell)
    nx = _grid_sample(-gx_.astype(np.float32), X, Z)
    nz = _grid_sample(-gz_.astype(np.float32), X, Z)
    nn = np.sqrt(nx * nx + nz * nz + 1)
    ndl = (nx * F.SUN[0] + F.SUN[1] + nz * F.SUN[2]) / nn
    slope = 1 - 1 / nn
    sh = F.R.terrain_shadow(Hm, cell, az=40.0, el=36.0, maxd=300.0)
    shadow = _grid_sample(sh, X, Z)
    cav = cv2.GaussianBlur(Hm, (0, 0), 6.0) - Hm
    cav2 = cv2.GaussianBlur(Hm, (0, 0), 1.6) - Hm                    # 岩の根もとの細かい陰
    ao = np.clip(1 - 0.10 * _grid_sample(np.clip(cav, 0, None).astype(np.float32), X, Z), 0.55, 1)
    ao = ao * np.clip(1 - 0.35 * _grid_sample(np.clip(cav2, 0, None).astype(np.float32), X, Z), 0.5, 1)
    ao = ao * np.clip(1 + 0.03 * _grid_sample(Hm, X, Z), 0.45, 1.0)
    r = np.hypot(X, Z) + 1e-3
    vx, vz = X - cam.eye[0], Z - cam.eye[2]
    vn = np.hypot(vx, vz) + 1e-6
    grass = F.ground_tex(X, Z, fp, flat=R.smoothstep(2600.0, 2200.0, r), rad=(X / r, Z / r), view=(vx / vn, vz / vn))
    stand = F.ground_tex(X, Z, fp)
    e = _grid_sample(ej, X, Z)
    b = _grid_sample(bowl, X, Z)
    rk = _grid_sample(rock, X, Z)
    n1 = R.fbm(X / 6.0, Z / 6.0, 4, seed=91) * 0.5 + 0.5
    n2 = R.fbm(X / 1.4, Z / 1.4, 3, seed=93) * 0.5 + 0.5
    n3 = R.fbm(X / 0.5, Z / 0.5, 2, seed=94) * 0.5 + 0.5
    yv = np.where(g, Y, 0).astype(np.float32)
    # 内壁：ゆがんだ層（斜面ほどはっきり）。底：くずれた暗い土とがれき
    warp = (n1 - 0.5) * 2.4 + (n2 - 0.5) * 0.6
    wall = _strata(yv + warp) * (0.86 + 0.28 * n2)[..., None]
    floor_k = R.smoothstep(0.30, 0.12, slope) * R.smoothstep(-4.0, -8.0, yv)
    rub = RUBBLE * (0.75 + 0.6 * n2)[..., None] * (1 - 0.4 * R.smoothstep(0.6, 0.75, n3))[..., None]
    rub = rub * (1 - 0.35 * n1[..., None]) + CLAY * 0.7 * 0.35 * n1[..., None]
    inner = wall * (1 - floor_k[..., None]) + rub * floor_k[..., None]
    # 縁の外：放出物（表土と粘土のまだら、放射の筋）
    ejc = LOAM * (1 - n1[..., None]) + CLAY * n1[..., None]
    ejc = ejc * (0.75 + 0.45 * n2)[..., None]
    rimk = R.smoothstep(0.6, 1.5, yv) * (1 - b)                     # 盛り上がった縁：ひっくり返った粘土の層
    ejc = ejc * (1 - 0.5 * rimk[..., None]) + CLAY * 1.1 * 0.5 * rimk[..., None]
    soil = np.where((b > 0.5)[..., None], inner, ejc)
    cover = np.clip(np.maximum(b, R.smoothstep(0.35, 0.80, e * (0.55 + 0.8 * n1))), 0, 1)
    spots = R.smoothstep(0.70, 0.80, n2) * np.clip(e * 1.5, 0, 1) * (1 - b)          # 散らばった土くれ
    # 放射状になぎ倒れた草：向こうへ倒れた葉は明るく（葉の裏のつや）、手前へ倒れた葉は暗い。細い放射の筋
    th = np.arctan2(Z, X)
    st = R.fbm(r / 40.0, th * 90.0, 3, seed=95) * 0.5 + 0.5
    sheen = (X / r) * (vx / vn) + (Z / r) * (vz / vn)
    rad_k = R.smoothstep(2600.0, 2200.0, r) * R.smoothstep(40.0, 70.0, r)
    grass = grass * (1 + rad_k * (0.20 * sheen + 0.26 * (st - 0.5)))[..., None]
    rockc = ROCK * (0.7 + 0.6 * n3)[..., None] * (0.8 + 0.4 * n2)[..., None]
    rk = np.clip(rk * 1.4, 0, 1)[..., None]
    cs = R.smoothstep(-0.15, 0.35, R.fbm(X / 420.0 + 3.0, Z / 420.0, 3, seed=97))      # 雲の影
    lit = np.clip(ndl, 0, 1) * shadow * (1 - 0.45 * cs)
    sky_amb = R.hexl('B4C8DA') * 0.36
    ao = ao * np.where(b > 0.5, 0.70, 1.0)
    light = F.SUN_RGB * 0.56 * lit[..., None] + sky_amb * ao[..., None]
    sky = F.sky_rad(np.concatenate([D[..., :1], np.maximum(D[..., 1:2], 0), D[..., 2:]], -1))
    outs = []
    for gr in (grass, stand):
        col = gr * (1 - cover[..., None]) + soil * cover[..., None]
        col = col * (1 - 0.55 * spots[..., None]) + F.SOIL_DK * 0.6 * spots[..., None]
        col = col * (1 - rk) + rockc * rk
        col = F.aerial(col * light, tt, 1 / 8000.0)
        img = np.where(g[..., None], col, sky)
        out = np.clip(R.srgb(img), 0, 1).astype(np.float32)
        outs.append(core.grade(out, con=0.16, sat=1.10))
    _C['C_stand'] = outs[1]
    _C['C_r'] = np.where(g, r, 1e6).astype(np.float32)
    _C['C'] = (outs[0], tt, cam)
    return _C['C']


def view_C(plate, zoom=1.0, dx=0.0, dy=0.0):
    """1.2 倍の板 → 画面（倍率 zoom は 1.0＝基本の画角）。戻り値 (フレーム, 2×3：板の座標 → 画面)。"""
    s = zoom / C_OVER
    ph, pw = plate.shape[:2]
    Mm = np.float32([[s, 0, W / 2 - s * pw / 2 + dx], [0, s, H / 2 - s * ph / 2 + dy]])
    f = cv2.warpAffine(plate, Mm, (W, H), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    return f, Mm


# ---------------------------------------------------------------- ショット B（真上の雲の層）
CAM_B = R.Cam3D((26.0, 1.2, -230.0), (0.0, F.ALT, 0.0), fov=64, up=(0, 0, 1))


def frame_B(hole_r, rim_push, t=0.0):
    """空＋雲の層（穴つき）の sRGB。"""
    if 'DB' not in _C:
        _C['DB'] = rays(CAM_B, 0.5)
        _C['DBf'] = rays(CAM_B)
        _C['skyB'] = F.sky_rad(_C['DBf'])
    c, a = F.deck(CAM_B, 0.5, hole_r=hole_r, cov=0.80, t=t, rim_push=rim_push, D=_C['DB'], sc=520.0, sharp=4.0)
    img = _C['skyB'] * (1 - a[..., None]) + c * a[..., None]
    return np.clip(R.srgb(img), 0, 1).astype(np.float32)
