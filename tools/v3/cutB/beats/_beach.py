"""01・02 の舞台（美術監督の指示 2026-11-08 で作り直し）：朝の空・体積の積雲・海・砂。_sky.py と同じ座標（HZ・DOT・FEET・shore_y）。

  world(gi, extra=0, hold_blur=0.0) → sRGB (H+extra, W, 3)   gi：01 の頭からの通しのフレーム番号（02 でも続けて数える）
  screen_post(f, hitstop=0.0, horizon_y=None)                 カメラのあとに掛ける：下 20% を 12% 暗く＋σ2 のぼけ、
                                                              ヒットストップ中は水平線と雲を σ1.5 でぼかす（奥行き）
色の台本（01/02 昼の外）：天頂 #2F6FB0 → 地平 #BFE0F5、海 #2C7DBE → #5FA8D8、砂 #D8DADB → #C9CED2。
太陽は左上（画面の外、方位 315°・高さ 40°）。影は右下へ。空の色相は 205–212°（ラベンダーや暖色にしない）。
雲：主役の 3 つは左の 1/3、見出しの箱（x 135–929・y 429–608）から 60px 以上はなれる（01・02 のカメラ・流れを通して）。
遠い雲 5 つは地平線の近く（高さに比例した大きさ、地平の色 #BFDDF2 へ 40–60% かすむ）。筋雲は 2 本だけ（α ≤0.15）。
"""
import math

import cv2
import numpy as np

import core
from core import W, H
from cutB.beats import _render as R, _sky as S, _clouds3d as C3

HZ = S.HZ
EXTRA_MAX = 300
HH = H + EXTRA_MAX
CAM = R.HorizonCam(HZ, f=1660.0, cam_h=3.2)
SUN = R.sun_vec(-40.0, 40.0)                       # 左上・画面の外
SUN_SEA = R.sun_vec(-40.0, 13.0)                   # 海のつやだけは低い角度で（左側にだけ光の筋が出る）
ZEN, HOR = R.hexl('2F78B0'), R.hexl('BFE0F5')          # 2026-11-08 r2：天頂の色相 210→206°（画面の上で 約208°）
HAZE_C = R.hexl('BFDDF2')
GLOW = R.hexl('EAF6FF')
XS = np.arange(W, dtype=np.float32) + 0.5
SHORE = S.shore_y(XS).astype(np.float32)
Y_SEA1 = int(SHORE.max()) + 16
Y_SAND0 = int(SHORE.min()) - 2
DAMP_D = 88.0                                      # しめった砂の幅（波打ち際から、px）
SWASH = (30.0, 60.0)                               # 寄せる水の膜の高さ（px）
_C = {}

# 主役の雲（世界の座標、x 中心・y 底・幅）。左の 1/3、見出しより上（カメラが上へ振る分も含めて離す）
HERO = [(170.0, 238.0, 231, 7, 4), (330.0, 128.0, 154, 12, 3), (6.0, 176.0, 147, 19, 3)]   # 2026-11-08 r2：0.7 倍（遠くへ）
HERO_DEFOCUS = 3.2                                  # σ px（直径 約 7px のぼけ）
HERO_DRIFT = (0.22, 0.16, 0.19)                     # px/フレーム（右へ。DOT の方へは行かない量）
# 遠い雲（x 中心・y 底・幅）：地平線に近いほど小さい
FAR = [(560.0, 1098.0, 150, 31), (812.0, 1112.0, 112, 37), (980.0, 1090.0, 176, 41), (90.0, 1104.0, 130, 43),
       (700.0, 1076.0, 210, 47)]
FAR_DRIFT = 0.22                                   # 2026-11-08 r2：0.10 → 0.22 px/フレーム（02 が止まって見えないように）


def _sky():
    yy, xx = np.mgrid[0:HH, 0:W].astype(np.float32)
    dx, dy, dz = CAM.dirs(yy, xx)
    e = np.arcsin(np.clip(dy, -1, 1))
    t = 1 - np.exp(-np.maximum(e, 0) * 3.6)
    col = HOR + (ZEN - HOR) * t[..., None]
    # 地平のごく近くのもや（白に近い青、色相はそのまま）
    hz = np.exp(-np.maximum(e, 0) / 0.012)[..., None]
    col = col + (HAZE_C * 1.06 - col) * hz * 0.5
    # 2026-11-08 r2：水平線の上の空気のもや（3–4% 持ち上げ。海と空の線が空気の中に座る）
    col = col + (HAZE_C - col) * (0.035 * np.exp(-np.maximum(e, 0) / 0.07))[..., None]
    # 太陽のにじみ：冷たい白を足す（0.12）
    cs = np.clip(dx * SUN[0] + dy * SUN[1] + dz * SUN[2], -1, 1)
    ang = np.arccos(cs)
    col = col + GLOW * (0.12 * np.exp(-ang / 0.32) + 0.035 * np.exp(-ang / 0.9))[..., None]
    return col.astype(np.float32)


def _cirrus():
    """2 本の筋雲（ノイズを 8:1 に伸ばし、10° 傾ける。α ≤0.15）。右上の空（DOT より上・見出しから離れた所）。"""
    h, w = 420, W + 400
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    th = math.radians(-10.0)
    u = (xx * math.cos(th) - yy * math.sin(th))
    v = (xx * math.sin(th) + yy * math.cos(th))
    n = R.fbm(u / 900.0, v / 112.0, 5, gain=0.55, seed=71) * 0.5 + 0.5
    a = R.smoothstep(0.58, 0.82, n)
    band = np.exp(-((v - 150) / 60.0) ** 2) + 0.6 * np.exp(-((v - 300) / 45.0) ** 2)
    a = np.clip(a * band, 0, 1) * 0.15
    a = cv2.GaussianBlur(a, (0, 0), 1.2)
    return a.astype(np.float32)


def worley(xx, yy, cell, seed=0):
    """2D Worley（F1、セルの大きさ cell px）。0（点の上）..約1。"""
    rng = np.random.default_rng(seed)
    gx, gy = np.floor(xx / cell).astype(int), np.floor(yy / cell).astype(int)
    nxc, nyc = gx.max() + 3, gy.max() + 3
    jit = rng.uniform(0, 1, (nyc + 2, nxc + 2, 2))
    d = np.full(xx.shape, 9.0, np.float32)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cx, cy = gx + ox, gy + oy
            j = jit[np.clip(cy + 1, 0, nyc + 1), np.clip(cx + 1, 0, nxc + 1)]
            px, py = (cx + j[..., 0]) * cell, (cy + j[..., 1]) * cell
            d = np.minimum(d, np.hypot(xx - px, yy - py) / cell)
    return d.astype(np.float32)


def cumulus_post(prem, a, seed=0, base=0.86, under_k=0.18, worley_cell=None):
    """2026-11-08 r2（美術監督）：積雲の仕上げ。平らな底の面で切り、底の側を 15–20% 冷たい灰青へ暗く、
    細かい毛羽の代わりに 2 つの大きさの Worley（大 1.0・小 0.25）で縁を削る。(prem, a) 線形 premultiplied。"""
    h, w = a.shape
    st = prem / np.maximum(a, 1e-4)[..., None]
    a = cv2.GaussianBlur(a, (0, 0), 0.9)                        # 細かい毛羽をならす
    st = cv2.GaussianBlur(st, (0, 0), 1.4)
    st = st * 0.6 + cv2.GaussianBlur(st, (0, 0), max(3.0, w * 0.025)) * 0.4      # 一様に明るい縁を弱める
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yb = h * base
    a = a * R.smoothstep(yb + 1.0, yb - 1.0, yy)                 # 平らな底（1 本の面）
    c1 = worley_cell or max(10.0, w * 0.11)
    wl = worley(xx, yy, c1, seed + 1)
    ws = worley(xx, yy, c1 * 0.25, seed + 2)
    er = 0.65 * wl + 0.35 * ws                                    # 大 1.0・小 0.25 の重ね
    edge = 1.0 - R.smoothstep(0.35, 1.0, cv2.GaussianBlur(a, (0, 0), max(2.0, w * 0.02)))   # 縁の帯（芯はそのまま）
    a = np.clip(a - edge * 1.1 * R.smoothstep(0.30, 0.80, er), 0, 1)
    a = R.smoothstep(0.03, 0.85, a) * a.max()
    # 底の側：冷たい灰青で 15–20% 暗く（底から 30% の高さまで）
    uk = R.smoothstep(yb - h * 0.30, yb - h * 0.02, yy) * under_k
    cool = R.hexl('A9B9CC')
    st = st * (1 - uk[..., None]) + st * cool * uk[..., None]
    return (st * a[..., None]).astype(np.float32), a.astype(np.float32)


def _cloud(w, seed, towers, flat_k=1.0, detail=1.0):
    h = int(w * 0.62)
    prem, a = C3.cloud_sprite(w, h, seed=seed, towers=towers, flat_k=flat_k, sun_k=2.6, amb_k=0.72, detail=detail)
    prem, a = cumulus_post(prem, a, seed=seed)
    # 白の上限（≤0.92 sRGB ≒ 線形 0.83）：やわらかく押さえる
    st = prem / np.maximum(a, 1e-4)[..., None]
    l = R.luma(st)
    k = 0.80
    lc = np.where(l > 0.55, 0.55 + (k - 0.55) * (1 - np.exp(-(l - 0.55) / (k - 0.55))), l)
    st = st * (lc / np.maximum(l, 1e-5))[..., None]
    return (st * a[..., None]).astype(np.float32), a.astype(np.float32)


def static():
    if _C:
        return _C
    _C['sky'] = _sky()
    _C['cirrus'] = _cirrus()
    hero = []
    for (_, _, w, sd, tw) in HERO:
        p, a = _cloud(w, sd, tw, detail=0.5)
        pad = 12
        p = cv2.GaussianBlur(np.pad(p, ((pad, pad), (pad, pad), (0, 0))), (0, 0), HERO_DEFOCUS)
        a = cv2.GaussianBlur(np.pad(a, pad), (0, 0), HERO_DEFOCUS)
        hero.append((p, a))
    _C['hero'] = hero
    far = []
    for (x, yb, w, sd) in FAR:
        prem, a = _cloud(w, sd, 3, flat_k=0.75, detail=0.6)
        kz = 0.40 + 0.20 * (1 - (HZ - yb) / 80.0)              # 地平に近いほどかすむ（40–60%）
        st = prem / np.maximum(a, 1e-4)[..., None]
        st = st * (1 - kz) + HAZE_C * kz
        a = a * (1 - 0.35 * kz)
        far.append(((st * a[..., None]).astype(np.float32), a))
    _C['far'] = far
    _C['sand'] = _sand()
    return _C


def _put_prem(L, prem, a, x0, y0):
    """線形の板 L に prem（線形・premultiplied）を小数の位置 (x0, y0) で重ねる。"""
    ix, iy = int(math.floor(x0)), int(math.floor(y0))
    fx, fy = x0 - ix, y0 - iy
    h, w = a.shape
    M_ = np.float32([[1, 0, fx], [0, 1, fy]])
    pa = np.dstack([prem, a])
    pa = cv2.warpAffine(pa, M_, (w + 1, h + 1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    xa, ya, xb, yb = max(ix, 0), max(iy, 0), min(ix + w + 1, L.shape[1]), min(iy + h + 1, L.shape[0])
    if xb <= xa or yb <= ya:
        return L
    sub = pa[ya - iy:yb - iy, xa - ix:xb - ix]
    L[ya:yb, xa:xb] = L[ya:yb, xa:xb] * (1 - sub[..., 3:4]) + sub[..., :3]
    return L


def hero_boxes(gi):
    """主役の雲の世界の座標の箱 [(x0, y0, x1, y1)]（見出しとの距離の確かめ用）。"""
    out = []
    for (x, yb, w, _, _), dv, (p, a) in zip(HERO, HERO_DRIFT, static()['hero']):
        h = a.shape[0] - 24
        x0 = x + dv * gi - w / 2 - 12
        ys, xs = np.nonzero(a > 0.02)
        out.append((x0 + xs.min(), yb - h * 0.86 - 12 + ys.min(), x0 + xs.max(), yb - h * 0.86 - 12 + ys.max()))
    return out


# =============================================================== 砂
def _sand():
    """砂（線形、行 Y_SAND0..HH）。2 倍で作って面積で縮める（風紋のちらつき・ぎざぎざなし）。"""
    ss = 2
    rows = HH - Y_SAND0
    yy, xx = np.mgrid[0:rows * ss, 0:W * ss].astype(np.float32)
    yy = Y_SAND0 + (yy + 0.5) / ss
    xx = (xx + 0.5) / ss
    X, Z, fz, fx = CAM.ground(yy, xx)
    # 砂の色：遠い #D8DADB → 手前 #C9CED2（青みはごく少し）
    tdist = np.clip((yy - HZ) / (HH - HZ), 0, 1) ** 0.9
    base = R.hexl('D8DADB') * (1 - tdist)[..., None] + R.hexl('C9CED2') * tdist[..., None]
    # 風紋：のこぎりの形（風上はゆるく、風下は急）。波長は世界の座標（遠いほど細かく見える）
    lam = 0.17
    th = math.radians(14.0)
    warp = R.fbm(X * 0.22, Z * 0.22, 3, seed=81) * 2.6 + R.fbm(X * 0.9, Z * 0.9, 2, seed=82) * 0.35
    ph = (Z * math.cos(th) + X * math.sin(th)) / lam + warp
    u = 1.0 - (ph - np.floor(ph))                 # 急な風下の面がカメラの側（影の筋が見える）
    # なめらかな非対称の波形（風下が急）：h = sin(2πu) + 0.32·sin(4πu)。傾き dh/du を 1 周の高さで割る
    tp = 2 * np.pi * u
    slope = (np.cos(tp) + 0.64 * np.cos(2 * tp)) * 2 * np.pi / 2.6
    # 風紋のある所／ない所（40–60% はなめらか）
    mk = R.smoothstep(-0.45, 0.10, R.fbm(X * 0.45, Z * 0.45, 3, seed=83))
    amp = 0.024 * mk
    # 2026-11-08 r2：風紋を地平の方まで。法線の強さは 1/(1+Z/Z0)（Z0＝もとの切れ目の約 2 倍）、
    # 細かさはガウスのフィルタ（1 サブ画素の大きさ ÷ 波長）でなめらかに弱める（急に消さない）。
    fpn = (np.abs(fz) * math.cos(th) + np.abs(fx) * math.sin(th)) / ss       # 波の向きの 1 サブ画素の大きさ
    aa = np.exp(-2.0 * (math.pi * fpn / lam) ** 2) * (1.0 / (1.0 + Z / 26.0)) * 1.25
    aa = np.clip(aa, 0, 1)
    g = slope * amp / lam * aa
    gx, gz = g * math.sin(th), g * math.cos(th)
    # 大きなゆるい起伏（足あとのないなだらかな浜）
    big = R.fbm(X * 0.35, Z * 0.35, 3, seed=84)
    gbx = np.gradient(big, axis=1) / np.maximum(fx, 1e-3) / ss * 0.010
    gbz = np.gradient(big, axis=0) / np.maximum(fz, 1e-3) / ss * 0.010
    nx, ny, nz = -(gx + gbx), np.ones_like(gx), -(gz + gbz)
    nn = np.sqrt(nx * nx + ny * ny + nz * nz)
    ndl = (nx * SUN[0] + ny * SUN[1] + nz * SUN[2]) / nn
    flat = float(SUN[1])
    shade = np.clip(ndl / flat, 0.2, 1.8)
    col = base * (0.62 + 0.38 * shade)[..., None]
    # 遠い所：細かい風紋が消えていく代わりに、波長 4 倍の明るさの帯（輝度 3.5%、同じフィルタ）へ受けわたす
    lam2 = lam * 4.0
    ph2 = (Z * math.cos(th) + X * math.sin(th)) / lam2 + warp * 0.5
    band = np.cos(2 * np.pi * ph2) * np.exp(-2.0 * (math.pi * fpn / lam2) ** 2)
    far_k = (1 - np.clip(aa / 0.6, 0, 1)) * (0.4 + 0.6 * mk)
    col = col * (1 + 0.035 * band * far_k)[..., None]
    # 風紋の頂の小さなつや（太陽の側の面）
    crest = np.clip(shade - 1.05, 0, None) * aa * mk
    col = col + R.hexl('F2F6F8') * (crest * 0.06)[..., None]
    # 1px の粒（法線の細かいゆらぎ）：近い所だけ
    rng = np.random.default_rng(85)
    gn = rng.standard_normal(X.shape).astype(np.float32)
    ga = np.clip(1.3 - fz / 0.012, 0, 1)
    col = col * (1 + gn * 0.05 * ga)[..., None]
    sm = cv2.resize(col, (W, rows), interpolation=cv2.INTER_AREA)
    # 貝殻・小石（0.1%）＋右下への接地の影
    ys0 = np.arange(rows, dtype=np.float32)[:, None] + Y_SAND0 + 0.5
    dlin = np.zeros((rows, W), np.float32)
    lay_b = np.zeros((rows * 4, W * 4), np.float32)
    lay_d = np.zeros((rows * 4, W * 4), np.float32)
    lay_s = np.zeros((rows * 4, W * 4), np.float32)
    n = int(rows * W * 0.00028)
    py = rng.uniform(0, 1, n) ** 0.6
    py = (S.shore_y(540.0) + 90) + py * (HH - S.shore_y(540.0) - 90)
    px = rng.uniform(0, W, n)
    for x, y in zip(px, py):
        Zp = CAM.h * CAM.f / max(y - HZ, 1)
        r = float(np.clip(0.008 / (Zp / CAM.f), 0.5, 2.4) * rng.uniform(0.6, 1.3))
        if r < 0.7:
            continue
        cx, cy = int(x * 4 * 16), int((y - Y_SAND0) * 4 * 16)
        ax_, ay_ = int(r * 4 * 16), int(r * 0.6 * 4 * 16)
        cv2.ellipse(lay_s, (cx + int(r * 0.9 * 64), cy + int(r * 0.5 * 64)), (ax_, ay_), 0, 0, 360, 1.0, -1,
                    cv2.LINE_AA, shift=4)
        tgt = lay_b if rng.random() < 0.55 else lay_d
        cv2.ellipse(tgt, (cx, cy), (ax_, ay_), float(rng.uniform(-30, 30)), 0, 360, 1.0, -1, cv2.LINE_AA, shift=4)
    lay_s = cv2.GaussianBlur(lay_s, (0, 0), 3.0)
    small = lambda a: cv2.resize(a, (W, rows), interpolation=cv2.INTER_AREA)
    sb, sd, sh = small(lay_b), small(lay_d), small(lay_s)
    sm = sm * (1 - 0.22 * sh * (1 - np.maximum(sb, sd)))[..., None]
    sm = sm * (1 - sb[..., None] * 0.8) + R.hexl('E6EAEC') * sb[..., None] * 0.8
    sm = sm * (1 - sd[..., None] * 0.8) + R.hexl('9AA5AF') * sd[..., None] * 0.8
    # 2026-11-08 r2（美術監督）：灰色の帯・ぼけた影の境をやめて 3 層に。
    # (b) しめった砂：アルベド ×0.78 に 8–12% の値のむら。(c) 乾いた砂との境はドメインワープ（振幅 10–14px）、ぼかしの帯なし。
    # (a) 寄せる水の膜（30–60px、フレネルの空の映り込み・泡のレース）は world() で時間とともに描く。
    dsh = ys0 - SHORE[None]
    xw = XS[None] + 9.0 * R.fbm(XS[None] / 70.0, ys0 / 40.0, 2, seed=91)
    yw = 12.0 * R.fbm(xw / 95.0, ys0 / 55.0 + 3.0, 3, seed=92) + 2.0 * R.fbm(xw / 34.0, ys0 / 20.0, 2, seed=93)
    edge = DAMP_D + 10.0 * R.fbm(XS[None] / 260.0, np.zeros((1, 1), np.float32) + 6.0, 2, seed=86)
    damp = R.smoothstep(0.75, -0.75, dsh + yw - edge) * (dsh >= -2)
    vn = R.fbm(XS[None] / 9.0, ys0 / 4.0, 3, seed=94) * 0.5 + R.fbm(XS[None] / 40.0, ys0 / 14.0, 2, seed=95) * 0.5
    dcol = sm * (0.78 * (1 + 0.10 * vn))[..., None]
    # 水ぎわのすぐ上はまだ少しつや（空をうすく映す、5% まで）
    dx, dy, dz = CAM.dirs(ys0 * np.ones((1, W), np.float32), XS[None] * np.ones((rows, 1), np.float32))
    refl = R.lin(np.clip(R.srgb(_C_sky_row(-dy)), 0, 1))
    gl = 0.05 * np.exp(-np.clip(dsh, 0, None) / 40.0)
    dcol = dcol * (1 - gl[..., None]) + refl * gl[..., None]
    sm = sm * (1 - damp[..., None]) + dcol * damp[..., None]
    return sm.astype(np.float32)


def _C_sky_row(e_dy):
    """反射に使う空（下向きの視線の反射＝上向き、仰角 asin(e_dy)）。"""
    e = np.arcsin(np.clip(e_dy, 0, 1))
    t = 1 - np.exp(-e * 3.6)
    return HOR + (ZEN - HOR) * t[..., None]


# =============================================================== 海
def _sky_fn(dx, dy, dz):
    e = np.arcsin(np.clip(dy, -1, 1))
    t = 1 - np.exp(-np.maximum(e, 0) * 3.6)
    col = HOR + (ZEN - HOR) * t[..., None]
    hz = np.exp(-np.maximum(e, 0) / 0.012)[..., None]
    return col + (HAZE_C * 1.06 - col) * hz * 0.5


_SEA_T = {}


def _sea_targets():
    if 't' not in _SEA_T:
        yy = np.arange(HZ, Y_SEA1, dtype=np.float32)[:, None] + 0.5
        q = np.clip((yy - HZ) / (SHORE.mean() - HZ), 0, 1) ** 0.8
        _SEA_T['t'] = (R.hexl('2C7DBE') * (1 - q) + R.hexl('5FA8D8') * q).astype(np.float32)
    return _SEA_T['t']


def _sea(t):
    yy = np.arange(HZ, Y_SEA1, dtype=np.float32)[:, None] + 0.5
    Zr = CAM.ground(yy, XS[None])[1]
    Zs = CAM.h * CAM.f / np.maximum(SHORE[None] - HZ, 1)
    shallow = np.exp(-np.clip(Zr - Zs, 0, None) / 3.5) * 0.6
    sea, info = R.sea(CAM, HZ, Y_SEA1, W, t, SUN_SEA, _sky_fn, amp=0.85, shallow=shallow, glitter=0.0, spec=0.0)
    # 行ごとの平均を色の台本へ合わせる（波の細かさは残す）
    tgt = _sea_targets()
    m = sea.mean(1, keepdims=True)
    sea = sea * (tgt[:, None, :] / np.maximum(m, 1e-4))
    # 太陽の側（左）だけの、波の頂のきらめき（波と一緒に動く＝ちらつかない）
    hgt = info['h']
    side = R.smoothstep(760.0, 120.0, XS)[None]
    near = R.smoothstep(HZ + 2, HZ + 40, yy) * R.smoothstep(SHORE.mean() + 5, SHORE.mean() - 60, yy)
    gl = R.smoothstep(0.52, 0.80, hgt) * side * near
    sea = sea + R.SUN_RGB * (gl * 0.22)[..., None]
    # 2026-11-08 r2：波の頂のつやのまたたき（時間で動く細かいむら × 波の高さ。ちらつかないよう弱く・ゆっくり）
    tw = R.fbm(XS[None] / 14.0 + t * 0.9, yy / 3.0 - t * 0.6, 2, seed=58) * 0.5 + 0.5
    sh = R.smoothstep(0.60, 0.85, hgt) * R.smoothstep(0.55, 0.80, tw) * near * (0.35 + 0.65 * side)
    sea = sea + R.SUN_RGB * (sh * 0.10)[..., None]
    # 水平線のごく細い明るい線（にじませない）
    sea[:2] = sea[:2] * 0.6 + HAZE_C * 0.4
    return sea


# =============================================================== 1 フレーム
def world(gi, extra=0):
    """sRGB の世界の板 (H+extra, W, 3)。"""
    C = static()
    t = gi / core.FPS
    Hh = H + extra
    L = C['sky'][:Hh].copy()
    # 筋雲（右上、ゆっくり右へ）
    ci = C['cirrus']
    sh = 200 - 0.15 * gi
    M_ = np.float32([[1, 0, -sh], [0, 1, 0]])
    ca = cv2.warpAffine(ci, M_, (W, ci.shape[0]), flags=cv2.INTER_LINEAR)
    reg = L[:ci.shape[0]]
    L[:ci.shape[0]] = reg + (R.hexl('F2F7FB') - reg) * ca[..., None]
    # 遠い雲（地平の近く）
    for (x, yb, w, _), (p, a) in zip(FAR, C['far']):
        h = a.shape[0]
        L = _put_prem(L, p, a, x + FAR_DRIFT * gi - w / 2, yb - h * 0.86)
    # 主役の雲
    for (x, yb, w, _, _), dv, (p, a) in zip(HERO, HERO_DRIFT, C['hero']):
        h = a.shape[0] - 24
        L = _put_prem(L, p, a, x + dv * gi - w / 2 - 12, yb - h * 0.86 - 12)
    # 海
    sea = _sea(t)
    L[HZ:Y_SEA1] = sea[:min(Y_SEA1, Hh) - HZ]
    # 波打ち際の泡（寄せて返す）＋寄せた水の膜
    foam = R.shore_foam(CAM, HZ, Y_SEA1, W, SHORE, t)
    reg = L[HZ:Y_SEA1]
    L[HZ:Y_SEA1] = reg * (1 - foam[..., None] * 0.80) + R.hexl('E9F1F6') * (foam[..., None] * 0.80)
    sand = C['sand'][:Hh - Y_SAND0]
    ys = np.arange(Y_SAND0, Hh, dtype=np.float32)[:, None] + 0.5
    below = (ys >= SHORE[None]).astype(np.float32)
    reg = L[Y_SAND0:Hh]
    new = sand * below[..., None] + reg * (1 - below[..., None])
    # 2026-11-08 r2：(a) 寄せる水の膜（30–60px）。フレネルで空を映し（F0 0.02 → 浅い角度で 0.30）、空の色を 35% 混ぜ、
    # 先に細い泡のレース（fbm 2 オクターブ・0.015/px・しきい値 0.55・0.4px/フレームで流れる）。
    r0, r1 = int(SHORE.min()) - Y_SAND0, min(int(SHORE.max() + SWASH[1] + 14) - Y_SAND0, Hh - Y_SAND0)
    yb = ys[r0:r1]
    ph = 0.5 + 0.5 * math.sin(t * 1.6)
    sw = SWASH[0] + (SWASH[1] - SWASH[0]) * (0.25 + 0.75 * ph) * (0.75 + 0.25 * (R.fbm(XS / 180.0 + t * 0.05,
                                                                                         np.zeros_like(XS) + 1.0, 2,
                                                                                         seed=96) * 0.5 + 0.5))
    edge = SHORE + sw[None] * 1.0 + 3.0 * np.sin(XS[None] / 37.0 + t * 2.2)
    film_a = np.clip(edge - yb + 0.5, 0, 1) * (yb >= SHORE[None])
    dx, dy, dz = CAM.dirs(yb * np.ones((1, W), np.float32), XS[None] * np.ones((r1 - r0, 1), np.float32))
    cos_i = np.clip(-dy, 0, 1)                                      # 下向きの視線と水面の法線の角度
    F = np.clip(0.02 + 0.98 * (1 - cos_i) ** 5, 0.02, 0.30)
    sky_r = _sky_fn(dx, -dy, dz)
    under = new[r0:r1]
    thin = R.smoothstep(0.0, 1.0, (edge - yb) / np.maximum(sw[None], 1))   # 先ほど薄い
    film = under * (1 - 0.35 * thin[..., None]) + sky_r * (0.35 * thin[..., None])
    film = film * (1 - F[..., None]) + sky_r * F[..., None]
    # 先のレース
    lx = XS[None] * 0.015 + 0.4 * 0.015 * gi
    ln = R.fbm(lx, yb * 0.015 * 3.0 + t * 0.1, 2, seed=97) * 0.5 + 0.5
    fine = R.fbm(XS[None] / 5.0 + gi * 0.4 / 5.0, yb / 2.2, 2, seed=98) * 0.5 + 0.5      # 泡の網目
    thk = 1.0 + 1.6 * (R.fbm(XS[None] / 23.0 + gi * 0.02, np.zeros((1, 1), np.float32) + 2.0, 2, seed=99) * 0.5 + 0.5)
    lace = R.smoothstep(0.50, 0.62, ln) * np.exp(-((yb - edge + 1.5) / thk) ** 2) * (0.45 + 0.55 * fine)
    lace += 0.45 * R.smoothstep(0.58, 0.72, fine) * R.smoothstep(-1.0, -4.0, yb - edge) * \
        R.smoothstep(-14.0, -5.0, yb - edge) * R.smoothstep(0.40, 0.62, ln)
    lace = np.clip(lace, 0, 1) * 0.70
    film = film + (R.hexl('E8F0F5') - film) * lace[..., None]
    fa = np.clip(film_a + lace * (yb >= SHORE[None]), 0, 1)[..., None]
    new[r0:r1] = under * (1 - fa) + film * fa
    L[Y_SAND0:Hh] = new
    return R.srgb(L)


def screen_post(f, hitstop=0.0, horizon_y=None):
    """カメラのあと：下 12% にごく弱い手前のぼけ（σ1）。hitstop>0 のあいだ水平線から上を σ1.5 でぼかす。
    2026-11-08 r2：下 20% を暗くする帯はやめた（美術監督：灰色の帯・霧に見える）。"""
    y0 = int(H * 0.88)
    ramp = np.clip((np.arange(H - y0, dtype=np.float32) + 0.5) / (H - y0), 0, 1)[:, None, None]
    sub = f[y0:]
    bl = cv2.GaussianBlur(sub, (0, 0), 1.0)
    k = R.smoothstep(0.0, 0.8, ramp)
    f[y0:] = sub * (1 - k) + bl * k
    if hitstop > 0 and horizon_y is not None:
        yh = int(min(max(horizon_y + 8, 0), H))
        if yh > 0:
            top = f[:yh]
            b = cv2.GaussianBlur(top, (0, 0), 1.5)
            f[:yh] = top * (1 - hitstop) + b * hitstop
    return f
