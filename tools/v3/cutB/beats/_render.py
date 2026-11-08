"""案B の CG を「仕上げた絵」にするための描画の道具箱（look-dev 2026-11）。numpy / cv2 だけで、CPU で 1 フレーム 1.5 秒以内。

方針：物理にそった近似（空の散乱・水のフレネル・ランバート＋リム・環境の遮へい）を、2.5D（高さの場＝heightfield）で描く。
重い部分は 1 回だけ作ってキャッシュする（ノイズのタイル、DEM、地形の色、雲の板、スプライト）。
色はすべて「線形の光」で計算して、最後に sRGB（0..1）へ戻す（core のフレームは sRGB の float32）。

公開 API（ざっくり）
  ノイズ
    perlin_tile(n, period, seed)            周期つき（つなぎ目なし）のグラジエントノイズ (n,n)。period は int か (px, py)
    fbm_tile(n, period, octaves, gain, seed, ridged=False)   周期つき fBm（-1..1 程度）。メモリ＋ディスクにキャッシュ
    sample(tex, u, v)                       周期つきの双一次サンプリング（u, v はテクセル単位、任意の形）
    fbm(u, v, octaves, seed, ...)           任意の座標の fBm（タイルを回して重ねる＝タイルの繰り返しが見えない）
  色
    lin(srgb) / srgb(lin) / luma(x) / smoothstep(a, b, x)
  空・雲（地平線のある画：01・02・06・07）
    HorizonCam(hz, f, cx, cam_h)            ピンホール。.dirs(y, x) → 視線、.ground(y, x) → 平面の (X, Z)
    sky_radiance(dx, dy, dz, sun, ...)      方向ごとの空の放射（線形）：レイリー風の天頂→地平のグラデーション＋ミー（HG）の太陽のにじみ
    sky(cam, h, w, sun, ...)                空の板（線形）
    clouds(cam, h, w, sun, t, ...)          雲の層（線形の色, アルファ）：平面に投影した fBm、太陽側の明るい面と影の面、銀色の縁
  海・砂
    sea(cam, y0, y1, w, t, sun, sky_fn, ...)          遠近の波（法線マップのアニメ＋フットプリントで粗さへ逃がす）、
                                                       太陽のきらめき、フレネルで空を映す、浅いところの色
    shore_foam(cam, y0, y1, w, shore_y, t, ...)       波打ち際の泡（寄せて返す）
    sand(cam, y0, y1, w, t, sun, shore_y, ...)        砂（細かい粒・風紋・ぬれた帯は暗く空を映す）
  2.5D の形（悪役・拳・ゾウ・富士山）
    Relief(w, h, s)                          高さの場のキャンバス：capsule / sphere / blob（多角形をふくらませる）を
                                             なめらかな和でまとめ、法線から照明（キー＋空＋照り返し＋リム＋スペキュラ＋AO）
    inflate(mask, radius)                    マスク → 丸い高さ（距離変換）
    shade_height(z, alpha, mat, light)       任意の高さの場を同じ照明で
    cast_shadow(alpha, ...) / contact_shadow(...)
  空気・仕上げ
    aerial(rgb, depth, fog, density)        空気遠近（奥ほど霞の色へ）
    finish(srgb_frame, i, ...)              全部の CG のフレームにかける仕上げ：ハイライトのにじみ（bloom）、トーンカーブ、
                                             周辺減光、ごく細かい粒（0.006–0.012）、縁の色収差（≤0.6px）、わずかなたる型のゆがみ
  地図（03・04・T）
    gsi_dem(lat0, lat1, lon0, lon1, z)      国土地理院の標高タイル（dem_png）を取ってきて 1 枚に。tools/models/gsi/ にキャッシュ
                                             ※ 出典の表示が必要：「国土地理院（標高タイル）を加工して作成」
    hillshade(z, cell, az, el) / terrain_ao(z, cell)
    Cam3D(eye, target, fov, w, h)           見下ろしの3Dカメラ（project / ground_rays）
    heightfield_render(cam, color_tex, height_tex, tex_to_world, ...)  地形の高さの場をレイマーチ（遮へいつき）
  計器（04）
    glass_dial(...)                         ガラス・金属の縁・映り込み・針の影（b04 の look-dev で使う部品）
"""
import hashlib
import math
import os

import cv2
import numpy as np

import core
from core import W, H

CACHE = os.path.join(core.OUT, 'cache', 'render')
GSI_DIR = os.path.join(core.MODELS, 'gsi')
os.makedirs(CACHE, exist_ok=True)
_MEM = {}


def cached(key, fn, disk=True):
    """key（文字列）で fn() の結果（ndarray）をメモリとディスク（exports/v3/cache/render）にとっておく。"""
    if key in _MEM:
        return _MEM[key]
    path = os.path.join(CACHE, hashlib.md5(key.encode()).hexdigest()[:16] + '.npy')
    if disk and os.path.exists(path):
        a = np.load(path)
    else:
        a = fn()
        if disk:
            np.save(path, a)
    _MEM[key] = a
    return a


# =============================================================== 色
def lin(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def luma(x):
    return x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def mix(a, b, t):
    return a + (b - a) * t


def hexl(h):
    """'#RRGGBB' → 線形の色。"""
    return lin(core.c(h))


# =============================================================== ノイズ
def perlin_tile(n=256, period=8, seed=0):
    """周期つきのグラジエント（Perlin）ノイズ。(n, n) float32、だいたい -1..1。period：格子の数（int か (px, py)）。"""
    px, py = (period, period) if np.isscalar(period) else period
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, (py, px))
    gx, gy = np.cos(ang), np.sin(ang)
    X = (np.arange(n) + 0.5) / n * px
    Y = (np.arange(n) + 0.5) / n * py
    X, Y = np.meshgrid(X, Y)
    xi, yi = np.floor(X).astype(int), np.floor(Y).astype(int)
    xf, yf = X - xi, Y - yi

    def g(ix, iy, dx, dy):
        ix %= px
        iy %= py
        return gx[iy, ix] * dx + gy[iy, ix] * dy

    n00 = g(xi, yi, xf, yf)
    n10 = g(xi + 1, yi, xf - 1, yf)
    n01 = g(xi, yi + 1, xf, yf - 1)
    n11 = g(xi + 1, yi + 1, xf - 1, yf - 1)
    u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
    v = yf * yf * yf * (yf * (yf * 6 - 15) + 10)
    return (((n00 * (1 - u) + n10 * u) * (1 - v) + (n01 * (1 - u) + n11 * u) * v) * 1.41421).astype(np.float32)


def fbm_tile(n=512, period=4, octaves=6, gain=0.5, seed=0, ridged=False):
    """周期つきの fBm。ridged=True は尾根のノイズ（1-|n|）。だいたい -1..1 に正規化。キャッシュ（メモリ＋ディスク）。"""
    px, py = (period, period) if np.isscalar(period) else period
    key = 'fbm_tile|%d|%d,%d|%d|%.3f|%d|%d' % (n, px, py, octaves, gain, seed, ridged)

    def mk():
        acc = np.zeros((n, n), np.float32)
        a, tot = 1.0, 0.0
        for k in range(octaves):
            p = perlin_tile(n, (px * 2 ** k, py * 2 ** k), seed * 101 + k)
            if ridged:
                p = 1 - np.abs(p) * 1.4
                p = p * p * 2 - 1
            acc += a * p
            tot += a
            a *= gain
        acc /= tot
        acc = (acc - acc.mean()) / (acc.std() * 2.2 + 1e-6)
        return np.clip(acc, -1.5, 1.5).astype(np.float32)
    return cached(key, mk)


def blur_wrap(a, sigma):
    """周期つきの画像のガウスぼかし（つなぎ目を出さない）。"""
    p = int(np.ceil(sigma * 3.5)) + 1
    b = cv2.GaussianBlur(np.pad(a, p, mode='wrap'), (0, 0), sigma)
    return b[p:-p, p:-p]


def sample(tex, u, v, interp=cv2.INTER_LINEAR):
    """周期つきの双一次サンプリング。u, v はテクセル単位（任意の形・大きな値でもよい）。"""
    h, w = tex.shape[:2]
    u = np.mod(np.asarray(u, np.float64), w).astype(np.float32)
    v = np.mod(np.asarray(v, np.float64), h).astype(np.float32)
    shp = u.shape
    if u.ndim == 1:
        u, v = u[None], v[None]
    out = cv2.remap(tex, u, v, interp, borderMode=cv2.BORDER_WRAP)
    return out.reshape(shp + tex.shape[2:])


def _base_noise():
    return cached('perlin|256|16|7', lambda: perlin_tile(256, 16, 7))


def fbm(u, v, octaves=5, gain=0.5, lac=2.03, seed=0, scale=1.0):
    """任意の座標（単位＝特徴の大きさが約 1）の fBm。タイルを回して重ねるので繰り返しが見えない。だいたい -1..1。"""
    tex = _base_noise()
    rng = np.random.default_rng(seed + 12345)
    acc = 0.0
    a, tot, f = 1.0, 0.0, 16.0 * scale       # 1 単位＝格子 1 つ
    for k in range(octaves):
        th = rng.uniform(0, 2 * np.pi)
        c, s = math.cos(th), math.sin(th)
        ox, oy = rng.uniform(0, 256, 2)
        uu = (u * c - v * s) * f + ox
        vv = (u * s + v * c) * f + oy
        acc = acc + a * sample(tex, uu, vv)
        tot += a
        a *= gain
        f *= lac
    return acc / tot * 1.6


# =============================================================== カメラ（地平線のある画）
class HorizonCam:
    """水平に前を見るピンホール（ロール・ピッチなし）。座標：x 右、y 上、z 奥。地面（海・砂）は y=0、カメラは高さ cam_h。
    hz：地平線の画面の y（px）、f：焦点距離（px）、cx：画面の中心の x。"""

    def __init__(self, hz, f=1660.0, cx=W / 2, cam_h=3.2):
        self.hz, self.f, self.cx, self.h = float(hz), float(f), float(cx), float(cam_h)

    def dirs(self, y, x):
        dx = (x - self.cx) / self.f
        dy = (self.hz - y) / self.f
        n = np.sqrt(dx * dx + dy * dy + 1)
        return dx / n, dy / n, 1 / n

    def ground(self, y, x):
        """画面の点 → 地面の (X, Z)（m）と 1px あたりの奥行き方向・横方向の大きさ（m）。y は地平線より下。"""
        dy = np.maximum(y - self.hz, 0.25)
        Z = self.h * self.f / dy
        X = (x - self.cx) * Z / self.f
        return X, Z, Z / dy, Z / self.f

    def y_of_Z(self, Z):
        return self.hz + self.h * self.f / Z


def sun_vec(az_deg, el_deg):
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)], np.float32)


# =============================================================== 空
SKY_ZEN = hexl('124F8E')          # 天頂（core.XS_SKY_TOP 1F5E96 よりわずかに深い）
SKY_HOR = hexl('5FA6DC')          # 地平の近く
SKY_HAZE = hexl('B9DAF0')         # 地平のもや（白に近い青）
SUN_RGB = np.array([1.0, 0.985, 0.965], np.float32)   # 太陽：ほぼ白（黄・金にしない）


def _hg(cos_t, g):
    return (1 - g * g) / (4 * np.pi * np.power(np.maximum(1 + g * g - 2 * g * cos_t, 1e-5), 1.5))


def sky_radiance(dx, dy, dz, sun, zen=SKY_ZEN, hor=SKY_HOR, haze=SKY_HAZE, mie=0.16, sun_disk=True, k_ray=2.6,
                 k_haze=34.0):
    """方向 (dx, dy, dz)（単位ベクトル、y 上）の空の放射（線形 RGB）。
    レイリー風：天頂 zen → 地平 hor（高さの指数）＋地平のもや haze。ミー：太陽のまわりの HG（g=0.76 と 0.3）のにじみ、
    空気の厚み（地平ほど厚い）で強める。sun：sun_vec()。"""
    e = np.arcsin(np.clip(dy, -1, 1))
    ep = np.maximum(e, 0)
    t = np.exp(-ep * k_ray)
    col = zen * (1 - t)[..., None] + hor * t[..., None]
    hz = np.exp(-ep * k_haze)[..., None]
    col = col + (haze - col) * hz * 0.7
    cs = dx * sun[0] + dy * sun[1] + dz * sun[2]
    air = (1.0 / (np.maximum(dy, 0) + 0.18))[..., None] * 0.18
    glow = (_hg(cs, 0.80) * 0.9 + _hg(cs, 0.25) * 0.35)[..., None] * mie * (0.5 + 0.6 * air)
    col = col + SUN_RGB * glow * 0.9
    if sun_disk:
        ang = np.arccos(np.clip(cs, -1, 1))
        col = col + SUN_RGB * (smoothstep(0.0105, 0.0075, ang) * 30.0)[..., None]
    # 地平線より下を見る方向（海に映すとき）：地平の色のまま
    return col.astype(np.float32)


def sky(cam, h, w, sun, y0=0, **kw):
    """空の板（線形）。行 y0..y0+h、地平線より下の行も地平の色で埋める。"""
    yy, xx = np.mgrid[y0:y0 + h, 0:w].astype(np.float32)
    dx, dy, dz = cam.dirs(yy, xx)
    return sky_radiance(dx, dy, dz, sun, **kw)


# =============================================================== 雲
def clouds(cam, h, w, sun, t=0.0, y0=0, alt=1400.0, scale=2600.0, coverage=0.42, wind=(9.0, 0.0), seed=3,
           sharp=3.2, dens=1.0, cirrus=0.35, horizon_fade=0.012, lit=None, shade=None, res=0.5):
    """雲（res 倍の解像度で計算して拡大。雲はやわらかいので 0.5 で十分、時間は約 1/4）。"""
    if res != 1.0:
        hh, ww = max(2, int(round(h * res))), max(2, int(round(w * res)))
        sc = _ScaledCam(cam, res, y0)
        col, a = clouds(sc, hh, ww, sun, t, 0, alt, scale, coverage, wind, seed, sharp, dens, cirrus, horizon_fade,
                        lit, shade, res=1.0)
        col = cv2.resize(col, (w, h), interpolation=cv2.INTER_CUBIC)
        a = np.clip(cv2.resize(a, (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)
        return col.astype(np.float32), a.astype(np.float32)
    return _clouds1(cam, h, w, sun, t, y0, alt, scale, coverage, wind, seed, sharp, dens, cirrus, horizon_fade, lit,
                    shade)


class _ScaledCam:
    """縮めた板のためのカメラ（板の画素 → 元の画面の座標）。"""

    def __init__(self, cam, res, y0):
        self.cam, self.res, self.y0 = cam, res, y0
        self.h, self.f, self.hz = cam.h, cam.f, cam.hz

    def dirs(self, y, x):
        return self.cam.dirs(self.y0 + (y + 0.5) / self.res - 0.5, (x + 0.5) / self.res - 0.5)


def _clouds1(cam, h, w, sun, t, y0, alt, scale, coverage, wind, seed, sharp, dens, cirrus, horizon_fade, lit, shade):
    """雲の層。(線形 RGB, アルファ) を返す（行 y0..y0+h）。
    高さ alt（m）の平面に投影した fBm（遠いほど小さく・薄く見える）＋ 太陽の方へずらした密度で影の面（2 回目のサンプル）、
    太陽の近くは縁が銀色に光る（前方散乱）。cirrus：上空の細い筋雲の量。t：秒（風で流れる）。"""
    yy, xx = np.mgrid[y0:y0 + h, 0:w].astype(np.float32)
    dx, dy, dz = cam.dirs(yy, xx)
    up = dy > 1e-3
    dyc = np.maximum(dy, 1e-3)
    dist = alt / dyc
    X = dx * dist / scale + wind[0] * t / scale
    Z = dz * dist / scale + wind[1] * t / scale
    # 大きな形（ゆがめた fBm）
    wx = fbm(X * 0.6, Z * 0.6, 3, seed=seed + 1) * 0.35
    d0 = fbm(X + wx, Z * 1.6 + wx, 6, gain=0.52, seed=seed)
    cov = coverage
    soft = 0.22                                                    # 縁のやわらかさ（しきい値のまわりを smoothstep で）
    d = smoothstep(1 - cov - soft * 0.3, 1 - cov + soft, d0 * 0.5 + 0.5) * (d0 * 0.5 + 0.5 - (1 - cov) + soft * 0.3)
    d = np.clip(d, 0, None) * sharp * dens
    # 影：太陽の方へ少しずらした密度
    sxz = np.array([sun[0], sun[2]])
    sxz = sxz / (np.hypot(*sxz) + 1e-6)
    off = 0.06
    d1 = fbm(X + wx + sxz[0] * off, Z * 1.6 + wx + sxz[1] * off * 1.6, 5, gain=0.52, seed=seed)
    d1 = np.clip(smoothstep(1 - cov - soft * 0.3, 1 - cov + soft, d1 * 0.5 + 0.5) *
                 (d1 * 0.5 + 0.5 - (1 - cov) + soft * 0.3), 0, None) * sharp * dens
    shadow = np.exp(-np.clip(d1 - d * 0.55, 0, None) * 2.4)
    # 透明度：密度 → 1-exp
    a = 1 - np.exp(-d * 1.7)
    # 遠いほど薄く（地平に向けて消える）、真上ほどくっきり
    fade = smoothstep(0.0, 0.09, dyc) * up
    a = a * fade
    cs = dx * sun[0] + dy * sun[1] + dz * sun[2]
    silver = (_hg(cs, 0.6) * 2.2)[..., None] * (1 - np.clip(d, 0, 1))[..., None]
    lit = np.array([1.0, 1.0, 1.0], np.float32) * 1.05 if lit is None else lit
    shade = hexl('9DB9D6') if shade is None else shade
    col = shade * (1 - shadow)[..., None] + lit * shadow[..., None]
    col = col + SUN_RGB * silver * 0.55
    # 底は少し暗く青い（厚いところ）
    col = col * (1 - 0.22 * np.clip(d - 0.35, 0, 1))[..., None]
    if cirrus > 0:
        cx_ = fbm(X * 0.35, Z * 3.2, 5, gain=0.6, seed=seed + 9)
        ci = np.clip(cx_ * 0.5 + 0.5 - 0.55, 0, None) * 1.6 * cirrus * smoothstep(0.05, 0.4, dyc)
        ci = ci * fade
        col = col * a[..., None] + np.array([0.95, 0.97, 1.0], np.float32) * (ci * (1 - a))[..., None]
        a = a + ci * (1 - a)
        col = col / np.maximum(a, 1e-4)[..., None]
    # 地平近くの空気遠近：雲を空の色へ寄せる
    hz = np.exp(-np.maximum(dy, 0) / max(horizon_fade * 4, 1e-3))[..., None]
    col = col * (1 - 0.55 * hz) + SKY_HAZE * 0.55 * hz
    return col.astype(np.float32), np.clip(a, 0, 1).astype(np.float32)


# =============================================================== 海
WATER_DEEP = hexl('0A3E6E')
WATER_MID = hexl('1D6AA6')
WATER_SHALLOW = hexl('5FB2DA')


def _wave_tiles():
    """海の高さのタイル 3 種（うねり・風の波・さざ波）と、その傾き（∂h/∂u, ∂h/∂v）とぼかした版（遠い所用）。"""
    def mk():
        out = []
        for (pp, seed) in (((3, 8), 21), ((6, 12), 22), ((10, 14), 23)):
            hgt = fbm_tile(512, pp, 5, 0.55, seed)
            gx = (np.roll(hgt, -1, 1) - np.roll(hgt, 1, 1)) * 0.5
            gy = (np.roll(hgt, -1, 0) - np.roll(hgt, 1, 0)) * 0.5
            out.append(np.stack([gx, gy, hgt], 0))
        return np.stack(out, 0).astype(np.float32)
    T = cached('wave_tiles_v1', mk)
    levels = []
    for k in range(T.shape[0]):
        lv = [T[k]]
        for s in (3.0, 9.0):
            lv.append(np.stack([blur_wrap(T[k, j], s) for j in range(3)], 0))
        levels.append(lv)
    return levels


_WT = {}


def wave_tiles():
    if 'w' not in _WT:
        _WT['w'] = _wave_tiles()
    return _WT['w']


def sea(cam, y0, y1, w, t, sun, sky_fn, amp=1.0, octs=((14.0, 0.55, (0.0, 0.9)), (4.2, 0.35, (0.6, 1.4)),
                                                         (1.3, 0.18, (-0.5, 1.1))),
        rough=0.05, spec=1.0, glitter=1.0, shallow=None):
    """海面（線形 RGB）の板（行 y0..y1）。
    octs：(タイルの大きさ m, 傾きの強さ, 流れる速さ m/s) の 3 段。1px の大きさ（フットプリント）が波より大きい所は
    傾きを弱めて、そのぶん粗さ（ハイライトの広がり）を増やす（遠くのきらめきが自然に帯になる）。
    sky_fn(dx, dy, dz) → 空の放射（反射に使う）。shallow：(rows, w) の 0..1（1＝浅い、砂の色が透ける）。"""
    yy, xx = np.mgrid[y0:y1, 0:w].astype(np.float32) + 0.5
    X, Z, fz, fx = cam.ground(yy, xx)
    tiles = wave_tiles()
    sx = np.zeros_like(X)
    sz = np.zeros_like(X)
    hsum = np.zeros_like(X)
    var_lost = np.zeros_like(X)
    for k, (L, g, vel) in enumerate(octs):
        n = 512.0
        fp = np.maximum(fz, fx) / L * n                     # 1px が何テクセルか
        u = (X + vel[0] * t) / L * n
        v = (Z + vel[1] * t) / L * n * 0.55                 # 奥行き方向に少し縮めて、横に長い波頭
        lv = tiles[k]
        w0 = np.clip(1 - fp / 3.0, 0, 1)                    # くっきりの版
        w2 = np.clip((fp - 3.0) / 9.0, 0, 1)                # 一番ぼかした版
        w1 = np.clip(1 - w0 - w2, 0, 1)
        acc = None
        att = np.minimum(1.0, 12.0 / np.maximum(fp, 1e-3)) ** 1.3       # 一番ぼかした版よりさらに粗い所は傾きを弱める
        for wk, lvl, ak in ((w0, lv[0], 1.0), (w1, lv[1], 1.0), (w2, lv[2], att)):
            if wk.max() <= 0:
                continue
            s = np.stack([sample(lvl[j], u, v) for j in range(3)], 0) * (wk * ak)[None]
            acc = s if acc is None else acc + s
        kk = g * amp * n / L * 0.02
        sx += acc[0] * kk
        sz += acc[1] * kk / 0.55 * 0.55
        hsum += acc[2] * g
        var_lost += (g * amp) ** 2 * (1 - np.exp(-fp / 6.0)) * 0.035
    # 法線（y 上）
    nx, ny, nz = -sx, np.ones_like(sx), -sz
    nn = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / nn, ny / nn, nz / nn
    dx, dy, dz = cam.dirs(yy, xx)                           # 視線（下向き）
    dn = dx * nx + dy * ny + dz * nz
    rx, ry, rz = dx - 2 * dn * nx, dy - 2 * dn * ny, dz - 2 * dn * nz
    ry = np.abs(ry) * 0.9 + 0.004                           # 下へ反射する所（波の裏）は浅い角度の空にする
    rn = np.sqrt(rx * rx + ry * ry + rz * rz)
    rx, ry, rz = rx / rn, ry / rn, rz / rn
    cosv = np.clip(-dn, 0, 1)
    F = 0.02 + 0.98 * (1 - cosv) ** 5
    refl = sky_fn(rx, ry, rz)
    # 水の色（波の頂上は少し明るく＝光が抜ける）
    crest = np.clip(hsum * 1.2 + 0.2, 0, 1)
    depth_k = smoothstep(0, 40, Z)                          # 手前（浜の近く）ほど明るい水
    body = WATER_MID * (1 - depth_k)[..., None] + WATER_DEEP * depth_k[..., None]
    body = body * (0.75 + 0.5 * crest)[..., None]
    sdot = max(float(sun[1]), 0.0)
    body = body * (0.55 + 0.6 * sdot)
    if shallow is not None:
        body = body * (1 - shallow[..., None]) + WATER_SHALLOW * shallow[..., None]
    col = body * (1 - F)[..., None] + refl * F[..., None]
    # 太陽のきらめき（ブリン・フォン。失った傾きの分だけ広げる）
    rl = np.clip(rx * sun[0] + ry * sun[1] + rz * sun[2], 0, 1)
    r2 = rough * rough + var_lost
    p = 2.0 / np.maximum(r2, 1e-5)
    sp = np.power(rl, p) * (p + 2) / (2 * np.pi) * F * 0.02 * spec
    if glitter:
        # きらめき：水面（世界の座標）の細かい面のばらつき。1px より細かい所は平均（1）に溶ける
        gl = sample(fbm_tile(256, 32, 2, 0.5, 31), X / 0.9 * 256 / 8 + t * 30, Z / 0.9 * 256 / 8 * 0.6) * 0.5 + 0.5
        fpg = np.maximum(fz, fx) / 0.9 * 4
        gk = np.clip(1 - fpg / 2.0, 0, 1) * glitter
        sp = sp * (1 - gk + gk * (0.2 + 2.4 * gl ** 4))
    col = col + SUN_RGB * sp[..., None]
    return col.astype(np.float32), dict(X=X, Z=Z, h=hsum, F=F)


def shore_foam(cam, y0, y1, w, shore, t, wash=6.0, width=0.6, seed=5):
    """波打ち際の泡のアルファ（行 y0..y1）。shore：各列の波打ち際の y（画面）。
    寄せる線は sin(t) で行き来し、うしろに網目のような泡（ノイズのしきい値）が残る。"""
    yy, xx = np.mgrid[y0:y1, 0:w].astype(np.float32) + 0.5
    X, Z, fz, fx = cam.ground(yy, xx)
    Zs = cam.h * cam.f / np.maximum(shore[None, :] - cam.hz, 1)       # 列ごとの波打ち際の奥行き
    dz = Z - Zs                                                        # 波打ち際から沖へ（m）
    ph = 0.5 + 0.5 * math.sin(t * 1.6)
    edge = 0.25 + 0.6 * ph + 0.15 * math.sin(t * 2.3 + 1)
    nz = fbm(X * 0.9, Z * 2.2 + t * 0.25, 4, seed=seed) * 0.5 + 0.5
    lace = fbm(X * 2.5, Z * 5.0 - t * 0.4, 4, seed=seed + 1) * 0.5 + 0.5
    line = np.exp(-((dz - edge) / (0.18 + 0.12 * nz)) ** 2)
    trail = np.clip((dz - edge) / 1.0, 0, 1) * np.exp(-np.clip(dz - edge, 0, None) / (width * 3.0))
    a = line * (0.55 + 0.45 * nz) + trail * smoothstep(0.55, 0.75, lace) * 0.55
    a *= (dz > -0.3)
    a *= np.clip(1 - fz / 0.6, 0.2, 1)
    return np.clip(a, 0, 1).astype(np.float32)


# =============================================================== 砂
SAND_DRY = hexl('B9C9D6')
SAND_DARK = hexl('7790A4')
SAND_WET = hexl('6F8DA6')


def sand(cam, y0, y1, w, t, sun, shore, sky_fn, wet_m=1.6, seed=8, base=None):
    """砂（線形 RGB、行 y0..y1）。細かい粒（遠近に合わせて細かく）、風紋（横長の小さな起伏をフォンで照らす）、
    波打ち際から wet_m（m）はぬれて暗く、空を映して青く光る。shore：各列の波打ち際の y。"""
    yy, xx = np.mgrid[y0:y1, 0:w].astype(np.float32) + 0.5
    X, Z, fz, fx = cam.ground(yy, xx)
    Zs = cam.h * cam.f / np.maximum(shore[None, :] - cam.hz, 1)
    d = np.clip(Zs - Z, 0, None)                                      # 波打ち際から手前へ（m）
    # 起伏（風紋）
    rip = fbm(X * 0.8, Z * 3.5, 4, gain=0.5, seed=seed) * 0.6 + fbm(X * 0.12, Z * 0.25, 3, seed=seed + 1) * 0.4
    gxr = np.gradient(rip, axis=1) / np.maximum(fx, 1e-3) * 0.004
    gzr = np.gradient(rip, axis=0) / np.maximum(fz, 1e-3) * 0.004
    nx, ny, nz = -gxr, np.ones_like(gxr), gzr
    nn = np.sqrt(nx * nx + ny * ny + nz * nz)
    ndl = np.clip((nx * sun[0] + ny * sun[1] + nz * sun[2]) / nn, 0, 1)
    tdist = np.clip((yy - cam.hz) / (H - cam.hz), 0, 1)
    col = (SAND_DRY * (1 - tdist ** 0.8)[..., None] + SAND_DARK * (tdist ** 0.8)[..., None]) if base is None else base
    col = col * (0.82 + 0.35 * ndl / max(float(sun[1]), 0.2))[..., None]
    # 粒（1px 以下になる所は平均に溶ける）
    gr = sample(fbm_tile(512, 128, 2, 0.6, 41), X / 0.05 * 4, Z / 0.05 * 4)
    ga = np.clip(1 - fz / 0.08, 0, 1)
    col = col * (1 + gr * 0.07 * ga)[..., None]
    # ぬれた帯：暗く、空をフレネルで映す
    wet = np.exp(-d / wet_m) * (0.8 + 0.2 * (fbm(X * 0.4, Z * 1.5, 3, seed=seed + 3) * 0.5 + 0.5))
    wet = wet[..., None]
    dx, dy, dz = cam.dirs(yy, xx)
    ry = -dy
    refl = sky_fn(dx, ry, dz)
    cosv = np.clip(-dy, 0, 1)
    F = (0.02 + 0.98 * (1 - cosv) ** 5)[..., None]
    wcol = SAND_WET * (1 - F) * 0.85 + refl * F * 0.9
    col = col * (1 - wet) + wcol * wet
    return col.astype(np.float32), dict(X=X, Z=Z, d=d)


# =============================================================== 2.5D の形（高さの場）
class Relief:
    """高さの場のキャンバス（出力 w×h px、内部は s 倍）。形を足していき、render() で照明して RGBA（ストレートアルファ）にする。
    座標は出力の px。z は手前が +（px 単位）。mat：材質の番号（render の mats の鍵）。
    - capsule(p0, p1, r0, r1, z0, mat, k)：太さの変わる丸い棒（カプセル）。k>0 でなめらかな和（関節がなじむ）
    - sphere(c, r, z0, mat)、ellipse(c, rx, ry, ang, z0, mat)
    - blob(poly, height, z0, mat, bevel)：多角形をふくらませる（距離変換。bevel px で縁が丸く落ちる）
    - offset(dz)：これまでの形を奥へ押す（重なりの順を作る）
    """

    def __init__(self, w, h, s=3):
        self.w, self.h, self.s = int(w), int(h), int(s)
        self.Z = np.full((self.h * s, self.w * s), -1e4, np.float32)
        self.M = np.zeros((self.h * s, self.w * s), np.int16)
        self.bump = np.zeros_like(self.Z)

    def _grid(self, x0, y0, x1, y1):
        s = self.s
        i0, j0 = max(int(math.floor(y0 * s)), 0), max(int(math.floor(x0 * s)), 0)
        i1, j1 = min(int(math.ceil(y1 * s)), self.h * s), min(int(math.ceil(x1 * s)), self.w * s)
        if i1 <= i0 or j1 <= j0:
            return None
        ys = (np.arange(i0, i1, dtype=np.float32) + 0.5) / s
        xs = (np.arange(j0, j1, dtype=np.float32) + 0.5) / s
        return (slice(i0, i1), slice(j0, j1)), xs[None, :], ys[:, None]

    def _merge(self, sl, z, mat, k):
        Zo = self.Z[sl]
        if k > 0:
            m = z > -1e3
            both = m & (Zo > -1e3)
            zn = np.where(m, z, Zo)
            # なめらかな最大（多項式の smooth max）
            hh = np.clip(0.5 + 0.5 * (z - Zo) / k, 0, 1)
            sm = Zo * (1 - hh) + z * hh + k * hh * (1 - hh)
            zn = np.where(both, sm, zn)
        else:
            zn = np.maximum(Zo, z)
        self.M[sl] = np.where(z >= Zo, mat, self.M[sl])
        self.Z[sl] = zn

    def capsule(self, p0, p1, r0, r1=None, z0=0.0, mat=1, k=0.0, flat=1.0):
        r1 = r0 if r1 is None else r1
        p0, p1 = np.asarray(p0, np.float32), np.asarray(p1, np.float32)
        rm = max(r0, r1)
        g = self._grid(min(p0[0], p1[0]) - rm - 1, min(p0[1], p1[1]) - rm - 1,
                       max(p0[0], p1[0]) + rm + 1, max(p0[1], p1[1]) + rm + 1)
        if g is None:
            return self
        sl, X, Y = g
        d = p1 - p0
        L2 = float(d @ d) + 1e-9
        tt = np.clip(((X - p0[0]) * d[0] + (Y - p0[1]) * d[1]) / L2, 0, 1)
        r = r0 + (r1 - r0) * tt
        qx, qy = X - (p0[0] + d[0] * tt), Y - (p0[1] + d[1] * tt)
        q2 = qx * qx + qy * qy
        inside = q2 < r * r
        z = np.where(inside, z0 + np.sqrt(np.maximum(r * r - q2, 0)) * flat, -1e4).astype(np.float32)
        self._merge(sl, z, mat, k)
        return self

    def sphere(self, c, r, z0=0.0, mat=1, k=0.0, flat=1.0):
        return self.capsule(c, c, r, r, z0, mat, k, flat)

    def ellipse(self, c, rx, ry, ang=0.0, z0=0.0, mat=1, k=0.0, flat=1.0, A=None):
        """楕円体（画面で rx×ry、ang 度回す。A を渡すと 2×2 の変形）。高さは min(rx, ry)。"""
        c = np.asarray(c, np.float32)
        Am = np.array([[rx, 0], [0, ry]], np.float32)
        th = math.radians(ang)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], np.float32)
        Am = (R @ Am) if A is None else (np.asarray(A, np.float32) @ R @ Am)
        ext = float(np.abs(Am).sum(1).max()) + 1
        g = self._grid(c[0] - ext, c[1] - ext, c[0] + ext, c[1] + ext)
        if g is None:
            return self
        sl, X, Y = g
        Ai = np.linalg.inv(Am)
        u = Ai[0, 0] * (X - c[0]) + Ai[0, 1] * (Y - c[1])
        v = Ai[1, 0] * (X - c[0]) + Ai[1, 1] * (Y - c[1])
        q = u * u + v * v
        rz = math.sqrt(abs(np.linalg.det(Am)))
        z = np.where(q < 1, z0 + rz * np.sqrt(np.maximum(1 - q, 0)) * flat, -1e4).astype(np.float32)
        self._merge(sl, z, mat, k)
        return self

    def blob(self, poly, height, z0=0.0, mat=1, bevel=None, k=0.0, bump=None):
        """多角形（出力 px）をふくらませる。bevel：縁から内側へ丸く上がる幅（px、None なら height）。
        bump：(rows, cols) の関数 f(X, Y) → 足す高さ（布のしわなど）。"""
        P = np.asarray(poly, np.float32)
        bevel = height if bevel is None else bevel
        g = self._grid(P[:, 0].min() - 2, P[:, 1].min() - 2, P[:, 0].max() + 2, P[:, 1].max() + 2)
        if g is None:
            return self
        sl, X, Y = g
        s = self.s
        hh, ww = Y.shape[0], X.shape[1]
        m = np.zeros((hh, ww), np.uint8)
        Q = (P * s - np.array([sl[1].start, sl[0].start], np.float32)) * 16
        cv2.fillPoly(m, [np.round(Q).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
        dist = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1] / s
        # 距離をぼかしてから丸める（距離変換の「骨」の折れ目＝稜線を消して、まくらのようになめらかに）
        dist = cv2.GaussianBlur(dist.astype(np.float32), (0, 0), max(0.8, 0.45 * bevel) * s) * (m > 0)
        q = np.clip(dist / max(bevel, 1e-3), 0, 1)
        prof = np.sqrt(np.maximum(1 - (1 - q) ** 2, 0)) * height
        if bump is not None:
            prof = prof + bump(X, Y) * q
        z = np.where(m > 0, z0 + prof, -1e4).astype(np.float32)
        self._merge(sl, z, mat, k)
        return self

    def offset(self, dz):
        m = self.Z > -1e3
        self.Z[m] -= dz
        return self

    def render(self, mats, light=(-0.55, -0.62, 0.56), sky=None, ground=None, sun=None, rim=None, ao=0.6,
               ao_r=None, aa=True, rim_power=2.6, rim_side=0.65):
        """照明して (h, w, 4) のストレートアルファの sRGB を返す。
        mats：{番号: dict(rgb=sRGB の色, spec=0..1, rough=0..1, sss=0..1, rimk=1.0)}。
        light：画面の向きの光（x 右、y 下、z 手前）。sky／ground：上・下から来る環境の光（線形）。sun：キーの光の色（線形）。
        rim：輪郭の光（線形、None で空の色）。ao：くぼみ・重なりの暗さ。"""
        Z = self.Z
        valid = Z > -1e3
        if not valid.any():
            return np.zeros((self.h, self.w, 4), np.float32)
        s = self.s
        base = np.where(valid, Z, 0).astype(np.float32)
        zmin = float(base[valid].min())
        base = np.where(valid, Z, zmin - 2.0 * s / s)
        zz = cv2.GaussianBlur(base, (0, 0), 0.6) * s           # 内部の px 単位の傾き
        gx = cv2.Sobel(zz, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(zz, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        nx, ny, nz = -gx, -gy, np.ones_like(gx)
        nn = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / nn, ny / nn, nz / nn
        L = np.asarray(light, np.float32)
        L = L / np.linalg.norm(L)
        sky = hexl('B8D8F0') * 0.55 if sky is None else sky
        ground = hexl('6F8FAA') * 0.35 if ground is None else ground
        sun = np.array([1.0, 0.99, 0.97], np.float32) * 1.25 if sun is None else sun
        rim = hexl('DDF0FF') * 0.9 if rim is None else rim
        # 環境の遮へい（まわりより低い所が暗い）
        r = ao_r if ao_r is not None else max(2.0, 0.06 * max(self.w, self.h)) * s
        bl = cv2.GaussianBlur(base, (0, 0), r)
        occ = np.clip((bl - base) / (r / s * 0.9), 0, 1)
        AO = 1 - ao * occ ** 0.9
        AO = AO * (0.75 + 0.25 * np.clip(nz * 1.2, 0, 1))
        ndl = nx * L[0] + ny * L[1] + nz * L[2]
        H_ = L + np.array([0, 0, 1], np.float32)
        H_ /= np.linalg.norm(H_)
        ndh = np.clip(nx * H_[0] + ny * H_[1] + nz * H_[2], 0, 1)
        up = np.clip(-ny * 0.5 + 0.5, 0, 1)
        out = np.zeros(Z.shape + (3,), np.float32)
        for k, m in mats.items():
            sel = (self.M == k) & valid
            if not sel.any():
                continue
            alb = lin(np.asarray(m.get('rgb', core.FOAM), np.float32))
            wrap = m.get('sss', 0.35)
            dif = np.clip((ndl[sel] + wrap) / (1 + wrap), 0, 1)
            amb = sky * up[sel][:, None] + ground * (1 - up[sel])[:, None]
            col = alb * (sun * dif[:, None] + amb) * AO[sel][:, None]
            rough = m.get('rough', 0.35)
            p = 2 / max(rough * rough, 1e-3)
            fres = 0.04 + 0.96 * (1 - np.clip(nz[sel], 0, 1)) ** 5
            sp = m.get('spec', 0.3) * np.power(ndh[sel], p) * (p + 2) / 8 * (0.25 + fres)
            col += sun * sp[:, None] * AO[sel][:, None]
            # リム：視線に対して横を向く面。光の側ほど強く
            side = np.clip(nx[sel] * L[0] + ny[sel] * L[1], 0, None) / (np.hypot(L[0], L[1]) + 1e-6)
            rr = (1 - np.clip(nz[sel], 0, 1)) ** rim_power * (rim_side + (1 - rim_side) * side) * m.get('rimk', 1.0)
            col += rim * rr[:, None]
            out[sel] = col
        a = valid.astype(np.float32)
        rgb = srgb(out)
        if aa and s > 1:
            prem = np.concatenate([rgb * a[..., None], a[..., None]], 2)
            small = cv2.resize(prem, (self.w, self.h), interpolation=cv2.INTER_AREA)
            al = small[..., 3:4]
            return np.concatenate([small[..., :3] / np.maximum(al, 1e-5), al], 2).astype(np.float32)
        return np.concatenate([rgb, a[..., None]], 2).astype(np.float32)

    def alpha(self):
        a = (self.Z > -1e3).astype(np.float32)
        return cv2.resize(a, (self.w, self.h), interpolation=cv2.INTER_AREA)


def inflate(mask, radius):
    """0..1 のマスク → 縁から radius px で丸く上がる高さ（0..radius）。"""
    m = (mask > 0.5).astype(np.uint8)
    d = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    q = np.clip(d / radius, 0, 1)
    return (np.sqrt(np.maximum(1 - (1 - q) ** 2, 0)) * radius).astype(np.float32)


def shade_height(z, alpha, rgb, light=(-0.5, -0.6, 0.62), spec=0.25, rough=0.4, ao=0.5, sky=None, ground=None,
                 sun=None, rim=None, s=1):
    """任意の高さの場 z（px）を Relief と同じ照明で。(h, w, 4)。"""
    R = Relief(z.shape[1] // s, z.shape[0] // s, s)
    R.Z = np.where(alpha > 0.5, z, -1e4).astype(np.float32)
    R.M = np.ones(z.shape, np.int16)
    return R.render({1: dict(rgb=rgb, spec=spec, rough=rough)}, light=light, sky=sky, ground=ground, sun=sun,
                    rim=rim, ao=ao)


def cast_shadow(alpha, dx, dy, blur, opacity, squash=1.0, pivot_y=None):
    """アルファから地面への影（ずらし・縦のつぶれ・ぼかし）。(h, w) の 0..1。pivot_y：足もとの行（そこを軸に縦につぶす）。"""
    h, w = alpha.shape
    py = h if pivot_y is None else pivot_y
    M = np.float32([[1, -dx / max(h, 1) * 0, dx], [0, squash, py * (1 - squash) + dy]])
    sh = cv2.warpAffine(alpha, M, (w, h), flags=cv2.INTER_LINEAR)
    if blur > 0:
        sh = cv2.GaussianBlur(sh, (0, 0), blur)
    return np.clip(sh * opacity, 0, 1)


def contact_shadow(frame, cx, cy, rx, ry, opacity=0.35, rgb=None, blur=None):
    """足もとの接地の影（やわらかい楕円）。frame は sRGB。"""
    rgb = core.c('0A2540') if rgb is None else rgb
    blur = max(1.0, ry * 0.8) if blur is None else blur
    pad = int(rx + 3 * blur + 2)
    x0, y0 = int(cx) - pad, int(cy) - pad
    m = np.zeros((2 * pad, 2 * pad), np.float32)
    cv2.ellipse(m, (int((cx - x0) * 4), int((cy - y0) * 4)), (int(rx * 4), int(ry * 4)), 0, 0, 360, 1.0, -1,
                cv2.LINE_AA, shift=2)
    m = cv2.GaussianBlur(m, (0, 0), blur) * opacity
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + 2 * pad, frame.shape[1]), min(y0 + 2 * pad, frame.shape[0])
    if xb <= xa or yb <= ya:
        return frame
    mm = m[ya - y0:yb - y0, xa - x0:xb - x0, None]
    frame[ya:yb, xa:xb] = frame[ya:yb, xa:xb] * (1 - mm) + rgb * mm
    return frame


# =============================================================== 空気遠近
def aerial(rgb_lin, depth, fog, density):
    """線形の色に、奥行き depth（任意の単位）に応じて霞の色 fog を混ぜる（1-exp(-depth·density)）。"""
    k = (1 - np.exp(-np.asarray(depth, np.float32) * density))[..., None]
    return rgb_lin * (1 - k) + np.asarray(fog, np.float32) * k


# =============================================================== 仕上げ
_FIN = {}


def _radial():
    if 'r' not in _FIN:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        nx, ny = (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)
        r2 = (nx * nx * (W / H) ** 2 + ny * ny)          # 長辺で 1
        _FIN['r'] = (xx, yy, nx, ny, r2)
    return _FIN['r']


def tone(x, shoulder=0.82, lift=0.006, contrast=0.05):
    """sRGB 0..∞ → 0..1 のフィルムっぽいカーブ：shoulder より上はやわらかく 1 へ（スペキュラ・にじみを白く飛ばさない）、
    暗部はわずかに持ち上げ、中間はごく弱い S 字。"""
    x = np.asarray(x, np.float32)
    k = shoulder
    hi = k + (1 - k) * (1 - np.exp(-(x - k) / (1 - k)))
    y = np.where(x > k, hi, x)
    y = y + contrast * (y - 0.5) * (1 - np.abs(2 * y - 1)) * 1.0
    return np.clip(lift + (1 - lift) * y, 0, 1)


def bloom(lin_rgb, threshold=0.75, knee=0.3, strength=0.10, levels=5):
    """ハイライトのにじみ（線形の色で）：しきい値をやわらかく（knee）越えた分をピラミッドでぼかして足す。"""
    l = luma(lin_rgb)
    soft = np.clip(l - threshold + knee, 0, 2 * knee)
    soft = soft * soft / (4 * knee + 1e-6)
    k = np.maximum(soft, l - threshold)
    src = (lin_rgb * (k / np.maximum(l, 1e-4))[..., None]).astype(np.float32)
    sizes = []
    cur = src
    for _ in range(levels):
        cur = cv2.pyrDown(cur)
        sizes.append(cur)
    up = None
    for i in range(levels - 1, -1, -1):
        g = cv2.GaussianBlur(sizes[i], (0, 0), 1.2)
        up = g if up is None else g + cv2.pyrUp(up, dstsize=(g.shape[1], g.shape[0]))
    up = cv2.resize(up, (lin_rgb.shape[1], lin_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return lin_rgb + up * (strength / levels * 2.0)


def finish(frame, i=0, bloom_k=0.10, bloom_thr=0.72, vignette=0.16, grain=0.009, ca_px=0.5, distort=0.0,
           curve=True, seed=0, bloom_src=None, grain_size=0.65):
    """全部の CG のフレームにかける仕上げ（sRGB の float32 (H, W, 3) を受けて返す）。文字はこの後に重ねる。
    - bloom：ハイライトのやわらかいにじみ（線形の光で。bloom_src を渡すとそこから作る＝文字を光らせない）
    - tone：フィルムっぽいカーブ（肩・わずかなトー）
    - vignette：周辺減光（線形で掛ける。中心の明るさは変えない）
    - ca_px：縁の色収差（R と B を中心から逆向きにわずかに拡大縮小。角で ca_px）
    - distort：たる型のゆがみ（角で distort px）
    - grain：輝度の細かい粒（平均 0、σ＝grain。フレームごとに変わる。中間調に強く、暗部・明部で弱い）
    平均の明るさはほぼ変わらない（点滅しない）。"""
    f = np.asarray(frame, np.float32)
    xx, yy, nx, ny, r2 = _radial()
    L = lin(f)
    if bloom_k > 0:
        src = L if bloom_src is None else lin(bloom_src)
        L = L + (bloom(src, bloom_thr, 0.3, bloom_k) - src)
    if vignette > 0:
        L = L * (1 - vignette * np.clip(r2, 0, 1.6) ** 1.35)[..., None]
    out = srgb(L)
    if curve:
        out = tone(out)
    if ca_px > 0 or distort:
        rp2 = (xx - W / 2) ** 2 + (yy - H / 2) ** 2
        rc2 = (W / 2) ** 2 + (H / 2) ** 2
        rc = math.sqrt(rc2)
        base_s = 1 + (distort * rp2 / (rc2 * rc) if distort else 0.0)   # 角で distort px（たる型）
        chans = []
        for ch, sgn in ((0, 1), (1, 0), (2, -1)):
            if sgn == 0 and not distort:
                chans.append(out[..., ch])
                continue
            s = base_s + sgn * ca_px * np.sqrt(rp2) / rc2              # ずれ＝ca_px·(r/r角)²
            mx = (W / 2 + (xx - W / 2) * s).astype(np.float32)
            my = (H / 2 + (yy - H / 2) * s).astype(np.float32)
            chans.append(cv2.remap(np.ascontiguousarray(out[..., ch]), mx, my, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT))
        out = np.stack(chans, 2)
    if grain > 0:
        rng = np.random.default_rng(seed * 100003 + i * 7919 + 17)
        g = rng.standard_normal((H, W)).astype(np.float32)
        if grain_size > 0:
            g = cv2.GaussianBlur(g, (0, 0), grain_size)
            g /= g.std() + 1e-6
        l = luma(out)
        wgt = 0.45 + 2.2 * l * (1 - l)
        out = out + (g * grain * wgt)[..., None]
    return np.clip(out, 0, 1).astype(np.float32)


# =============================================================== 地図：国土地理院の標高タイル
GSI_URL = 'https://cyberjapandata.gsi.go.jp/xyz/dem_png/{z}/{x}/{y}.png'
GSI_CREDIT = '国土地理院（標高タイル）を加工して作成'


def _tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    lr = math.radians(lat)
    y = (1 - math.log(math.tan(lr) + 1 / math.cos(lr)) / math.pi) / 2 * n
    return x, y


def _tile_ll(x, y, z):
    n = 2 ** z
    lon = x / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def gsi_tile(z, x, y):
    """1 枚の dem_png を (256, 256) の標高（m、海・欠測は NaN）で。tools/models/gsi/dem_png/z/x/y.png にキャッシュ。"""
    path = os.path.join(GSI_DIR, 'dem_png', str(z), str(x), '%d.png' % y)
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with urllib.request.urlopen(GSI_URL.format(z=z, x=x, y=y), timeout=30) as r:
                data = r.read()
        except Exception:
            data = b''
        with open(path, 'wb') as fp:
            fp.write(data)
    if os.path.getsize(path) == 0:
        return np.full((256, 256), np.nan, np.float32)
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return np.full((256, 256), np.nan, np.float32)
    b, g, r = [im[..., k].astype(np.int64) for k in range(3)]
    v = r * 65536 + g * 256 + b
    out = np.where(v < 2 ** 23, v * 0.01, (v - 2 ** 24) * 0.01).astype(np.float32)
    out[v == 2 ** 23] = np.nan
    return out


def gsi_dem(lat0, lat1, lon0, lon1, z=12):
    """範囲の標高をつないで 1 枚に。戻り値：(elev (rows, cols) m・海は NaN, info)。
    info['ll'](row, col) → (lat, lon)。Web メルカトルの画素の並び（北が上）。"""
    xa, ya = _tile_xy(lat1, lon0, z)
    xb, yb = _tile_xy(lat0, lon1, z)
    tx0, ty0, tx1, ty1 = int(xa), int(ya), int(xb), int(yb)
    key = 'gsi|%d|%d|%d|%d|%d' % (z, tx0, ty0, tx1, ty1)

    def mk():
        rows = []
        for ty in range(ty0, ty1 + 1):
            rows.append(np.concatenate([gsi_tile(z, tx, ty) for tx in range(tx0, tx1 + 1)], 1))
        return np.concatenate(rows, 0)
    E = cached(key, mk)

    def ll(r, c):
        return _tile_ll(tx0 + (np.asarray(c) + 0.5) / 256.0, ty0 + (np.asarray(r) + 0.5) / 256.0, z)
    return E, dict(z=z, tx0=tx0, ty0=ty0, ll=ll)


def gsi_sample_ll(E, info, lat, lon):
    """(lat, lon)（配列可）での標高（双一次、海は NaN→0 にしない。呼ぶ側で扱う）。"""
    z = info['z']
    n = 2 ** z
    x = (np.asarray(lon, np.float64) + 180) / 360 * n
    lr = np.radians(np.asarray(lat, np.float64))
    y = (1 - np.log(np.tan(lr) + 1 / np.cos(lr)) / np.pi) / 2 * n
    c = (x - info['tx0']) * 256 - 0.5
    r = (y - info['ty0']) * 256 - 0.5
    return cv2.remap(E.astype(np.float32), c.astype(np.float32), r.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=float('nan'))


def hillshade(z, cell, az=315.0, el=40.0, exag=1.0):
    """陰影起伏（0..1）。z：標高（m）、cell：1 画素の大きさ（m）。az：光の来る方位（度、北＝0、時計回り）。"""
    gy, gx = np.gradient(z * exag, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azr, elr = math.radians(360 - az + 90), math.radians(el)
    hs = np.sin(elr) * np.cos(slope) + np.cos(elr) * np.sin(slope) * np.cos(azr - aspect)
    return np.clip(hs, 0, 1).astype(np.float32)


def terrain_ao(z, cell, radii=(60, 240, 900), exag=1.0):
    """地形の環境の遮へい（谷が暗い）。いくつかの大きさのぼかしとの差。0..1（1＝開けている）。"""
    acc = np.zeros_like(z, dtype=np.float32)
    zz = z.astype(np.float32) * exag
    for rm in radii:
        s = max(rm / cell, 0.8)
        bl = cv2.GaussianBlur(zz, (0, 0), s)
        acc += np.clip((bl - zz) / (rm * 0.6), 0, 1)
    return np.clip(1 - acc / len(radii) * 1.6, 0, 1).astype(np.float32)


# =============================================================== 3D カメラ（地図）
class Cam3D:
    """注視点を見るピンホール。世界：x 東、y 上、z 南（＝地図の px の y と同じ向き）。fov は縦の画角（度）。"""

    def __init__(self, eye, target, fov=34.0, w=W, h=H, up=(0, 1, 0), pp=None):
        self.cx, self.cy = (w / 2, h / 2) if pp is None else pp      # 主点（注視点が画面のどこに来るか）
        self.eye = np.asarray(eye, np.float64)
        self.target = np.asarray(target, np.float64)
        self.w, self.h = w, h
        f = self.target - self.eye
        f /= np.linalg.norm(f)
        r = np.cross(f, np.asarray(up, np.float64))
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.R = np.stack([r, u, f], 0)                      # 世界 → カメラ（x 右、y 上、z 前）
        self.fpx = 0.5 * h / math.tan(math.radians(fov) / 2)

    def project(self, P):
        """(N, 3) → (N, 2) 画面 px と奥行き。"""
        P = np.asarray(P, np.float64)
        c = (P - self.eye) @ self.R.T
        z = np.maximum(c[..., 2], 1e-6)
        x = self.cx + c[..., 0] / z * self.fpx
        y = self.cy - c[..., 1] / z * self.fpx
        return np.stack([x, y], -1), c[..., 2]

    def rays(self, yy, xx):
        dx = (xx - self.cx) / self.fpx
        dy = -(yy - self.cy) / self.fpx
        d = dx[..., None] * self.R[0] + dy[..., None] * self.R[1] + self.R[2]
        return d / np.linalg.norm(d, axis=-1, keepdims=True)


def heightfield_render(cam, height, origin, cell, y_scale=1.0, steps=72, refine=6, scale=0.5, sea_level=0.0):
    """高さの場（height：(rows, cols)、世界の y（上）での高さ、origin＝(x0, z0) が [0,0] の画素の世界の位置、
    1 画素 cell）を、cam から見た画面の各点で、レイマーチ（最大の高さの平面から海面まで steps 歩＋二分法 refine 回）して、
    当たった所の世界の (x, z, y) を返す（scale 倍の解像度で計算して、なめらかに拡大）。地形の遮へいが正しく出る。
    戻り値：dict(x, z, y, t, hit)（すべて (H, W)）。"""
    hh, ww = int(cam.h * scale), int(cam.w * scale)
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    D = cam.rays((yy + 0.5) / scale, (xx + 0.5) / scale)
    e = cam.eye
    hmax = float(np.nanmax(height)) * y_scale + 1e-3
    dy = np.minimum(D[..., 1], -1e-4)
    t0 = np.maximum((e[1] - hmax) / -dy, 0)
    t1 = (e[1] - sea_level) / -dy
    Hm = np.nan_to_num(height, nan=sea_level).astype(np.float32) * y_scale

    def hgt(px, pz):
        u = ((px - origin[0]) / cell).astype(np.float32)
        v = ((pz - origin[1]) / cell).astype(np.float32)
        return cv2.remap(Hm, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    lo = t0.copy()
    hi = t1.copy()
    found = np.zeros(t0.shape, bool)
    prev = t0.copy()
    for k in range(1, steps + 1):
        tk = t0 + (t1 - t0) * (k / steps)
        px, py, pz = e[0] + D[..., 0] * tk, e[1] + D[..., 1] * tk, e[2] + D[..., 2] * tk
        below = (py <= hgt(px, pz)) & ~found
        lo = np.where(below, prev, lo)
        hi = np.where(below, tk, hi)
        found |= below
        prev = tk
    lo = np.where(found, lo, t1)
    hi = np.where(found, hi, t1)
    for _ in range(refine):
        mid = (lo + hi) * 0.5
        px, py, pz = e[0] + D[..., 0] * mid, e[1] + D[..., 1] * mid, e[2] + D[..., 2] * mid
        b = py <= hgt(px, pz)
        hi = np.where(b, mid, hi)
        lo = np.where(b, lo, mid)
    t = (lo + hi) * 0.5
    out = dict(t=t, x=e[0] + D[..., 0] * t, z=e[2] + D[..., 2] * t, y=e[1] + D[..., 1] * t)
    for k in list(out):
        out[k] = cv2.resize(out[k].astype(np.float32), (cam.w, cam.h), interpolation=cv2.INTER_LINEAR)
    return out


# =============================================================== 計器（ガラス・金属・針の影）
def disc_normals(h, w, cx, cy, r_in, r_out, profile='torus'):
    """リング（r_in..r_out）の法線（画面の向き、z 手前）と内外のマスク。profile='torus' は丸い断面。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy) + 1e-6
    mid, hw = (r_in + r_out) / 2, (r_out - r_in) / 2
    q = (r - mid) / hw
    inside = np.abs(q) < 1
    nzr = np.sqrt(np.clip(1 - q * q, 0, 1))
    nx, ny = q * dx / r, q * dy / r
    return nx, ny, nzr, inside, r


def env_reflect(nx, ny, nz, top=None, bottom=None, horizon=None, sharp=6.0):
    """簡単なスタジオの映り込み（線形）：上は明るい窓、地平の細い明るい帯、下は暗い。光沢のある金属・ガラス用。"""
    top = hexl('CFE6F7') if top is None else top
    bottom = hexl('061626') if bottom is None else bottom
    horizon = hexl('8FC3E6') if horizon is None else horizon
    ry = -ny                                 # 上向きの成分（画面は y 下）
    t = np.clip(ry * 0.5 + 0.5, 0, 1)
    col = bottom * (1 - t)[..., None] + top * t[..., None]
    band = np.exp(-(ry * sharp) ** 2)[..., None]
    col = col + horizon * band * 0.6
    return col
