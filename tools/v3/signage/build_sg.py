"""v3 サイネージ：映画館で流れる CM のような、しんみりした 30 秒／15 秒（edl/v3/EDL_signage_v3.md）。

  python3 tools/v3/signage/build_sg.py 30 15            # 書き出し → exports/v3/MAZE_v3_sg_{30,15}s.mp4
  python3 tools/v3/signage/build_sg.py 30 --stills 2.5 7 # 指定した秒のコマだけ PNG（exports/v3/sg_preview/）

横 1920×1080・30fps・無音。2.00:1 の黒帯（上下 60px）。LP の言葉「海と、時間と、おいしいもの。」を 1 句ずつ。
タイムラインは輪（ループ）として扱う：最後のカット（夜）と最初のカット（朝の海）のディゾルブは、ファイルの終わりと
頭にまたがる（最後のコマ → 最初のコマが途切れない）。
処理の順：素材（トーンマップ済み・表示値）→ カットごとの下ごしらえ → リニア光でディゾルブ → 色（シャドウ）→
にじみ → 周辺減光 → 粒子 → 黒帯 → 文字。
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
FOOT = os.path.join(ROOT, 'footage')
OUT = os.path.join(ROOT, 'exports', 'v3')
FONTS = os.path.join(ROOT, 'assets', 'fonts')
W, H, FPS = 1920, 1080, 30
BAR = 60                                      # 2.00:1 の黒帯（上下）
PIC = (BAR, H - BAR)                          # 映像の部分の y 範囲
HORIZON_Y = BAR + 0.38 * (H - 2 * BAR)        # 水平線をそろえる高さ（映像の部分の上から 38%）= 425
OFFWHITE = np.array([0xF4, 0xF1, 0xEA], np.float32) / 255
NAVY_SH = np.array([0x0B, 0x1A, 0x24], np.float32) / 255
NAVY_DIP = np.array([0x0E, 0x1B, 0x26], np.float32) / 255   # ループの継ぎ目：夜 → 紺 → 朝（黒へのフェードは使わない）
FLOOR = np.array([10, 20, 30], np.float32) / 255     # いちばん暗い所はこの紺で止める（真っ黒にしない）
TONEMAP = ('zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,'
           'zscale=t=bt709:m=bt709:r=full')
PHOTO_DAY = os.path.join(FOOT, 'facility', '76C67856-9213-47E0-ACDD-8C5ECD799F4C.JPEG')     # 写真 28
PHOTO_NIGHT = os.path.join(FOOT, 'facility', '2DE02582-9463-4C5F-A6D1-8D440EA2CF2E.JPEG')   # 写真 12
PASTA = os.path.join(FOOT, 'new_1107', 'C176E666.jpg')
CAKE = os.path.join(FOOT, 'new_1107', 'IMG_0946.JPG')


def ease(t, kind='inout'):
    t = float(np.clip(t, 0, 1))
    if kind == 'in':
        return t * t
    if kind == 'out':
        return 1 - (1 - t) ** 2
    return t * t * (3 - 2 * t)


def linear_move(u, edge=0.10):
    """一定の速さの動き。最初と最後の edge だけなめらかに加減速する（台車の移動のように）。"""
    u = float(np.clip(u, 0, 1))
    e = edge
    v = 1.0 / (1 - e)                         # 中間の速さ（全体で 0→1）
    if u < e:
        return v * u * u / (2 * e)
    if u > 1 - e:
        return 1 - v * (1 - u) ** 2 / (2 * e)
    return v * (u - e / 2)


def srgb_to_lin(a):
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(a):
    a = np.clip(a, 0, 1)
    return np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055).astype(np.float32)


def luma(a):
    return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722


# ---------------------------------------------------------------------------
# 素材
# ---------------------------------------------------------------------------
class Video:
    """iPhone の HLG 4K/60 → SDR、0.5 倍（60 コマ → 30 コマ／秒、コマは 1 枚ずつそのまま）。
    rot(i)・zoom(i)・cx/cy(i)：コマごとの回転（度）・倍率・中心（素材の比）で 1920×1080 に切り出す。"""

    def __init__(self, name, t0, rot=0.0, zoom=1.0, cx=0.5, cy=0.5, speed=0.5, dw=2560, dh=1440):
        self.src = os.path.join(FOOT, name)
        self.t0, self.speed = t0, speed
        self.rot, self.zoom, self.cx, self.cy = [v if callable(v) else (lambda i, v=v: v) for v in (rot, zoom, cx, cy)]
        self.dw, self.dh = dw, dh
        self.p, self.k = None, None

    def _open(self, i):
        if self.p:
            self.p.kill()
        t = self.t0 + i / FPS * self.speed
        vf = TONEMAP + ',scale=%d:%d:flags=lanczos,setpts=%g*(PTS-STARTPTS),fps=%d,format=rgb48le' % (
            self.dw, self.dh, 1 / self.speed, FPS)
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', '%.4f' % t, '-i', self.src, '-an', '-vf', vf,
                                   '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-'], stdout=subprocess.PIPE)
        self.k = i

    def frame(self, i):
        if self.p is None or self.k != i:
            self._open(i)
        b = self.p.stdout.read(self.dw * self.dh * 6)
        self.k = i + 1
        if len(b) < self.dw * self.dh * 6:
            b = self.last
        self.last = b
        a = np.frombuffer(b, '<u2').reshape(self.dh, self.dw, 3).astype(np.float32) / 65535
        return affine(a, self.cx(i), self.cy(i), self.zoom(i), self.rot(i))


def affine(img, cx, cy, zoom, rot):
    """img の (cx, cy)（比）を画面の中心に、幅が画面の幅の zoom 倍になるように、rot 度回して 1920×1080 へ。"""
    h, w = img.shape[:2]
    s = zoom * W / w
    M = cv2.getRotationMatrix2D((cx * w, cy * h), rot, s)
    M[0, 2] += W / 2 - cx * w
    M[1, 2] += H / 2 - cy * h
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


def load_photo(path, long_side=None):
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    if long_side and max(im.size) > long_side:
        r = long_side / max(im.size)
        im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255


# ---------------------------------------------------------------------------
# カット
# ---------------------------------------------------------------------------
class Shot:
    """frame(i, n)：カットの i コマ目（n はカットの全コマ数。ディゾルブの重ねを含む）。表示値（sRGB）で返す。"""
    def frame(self, i, n):
        raise NotImplementedError


class Beach(Shot):
    """IMG_9658：浜。サーファーと崩れる白波（5.0 秒以降ほぼ固定）。+0.3° で水平線を水平に、38% の高さへ。"""
    def __init__(self, t0=5.10):
        # 素材の水平線は上から約 41%（0.41）。zoom 1.10 で下のフェンスを外し、水平線を HORIZON_Y へ
        z = 1.10
        cy = 0.41 + (H / 2 - HORIZON_Y) / (z * W / 2560) / 1440
        self.v = Video('IMG_9658.MOV', t0, rot=0.3, zoom=lambda i: z * (1 + 0.02 * i / 190.0), cx=0.5, cy=cy)

    def frame(self, i, n):
        return self.v.frame(i)


class Window(Shot):
    """IMG_9674：2F の客席越しに右へパンして、窓の海へ。窓から入る光と窓枠の影がテーブルと床をゆっくり動く。"""
    def __init__(self, t0=6.20):
        self.v = Video('IMG_9674.MOV', t0, rot=0.0, zoom=1.06, cx=0.5, cy=0.47)
        self.prev, self.X = None, 0.0
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.xx, self.yy = xx, yy
        # 光の落ちる所：画面の下（床とテーブル）。上 2/3 にはかけない（窓・天井）
        self.floor = np.clip((yy / H - 0.66) / 0.10, 0, 1)

    def frame(self, i, n):
        f = self.v.frame(i)
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (480, 270))
        if self.prev is not None:
            (dx, _), _ = cv2.phaseCorrelate(self.prev, g)
            self.X += dx * 4.0                     # カメラのパンに光の模様を貼りつける
        self.prev = g
        # 窓枠の影：斜めの太い帯（周期 520px、縁はやわらかく）。光の差す向きがゆっくり動く（0.5px/コマ）
        th = np.deg2rad(62)
        u = ((self.xx - self.X - 0.5 * i) * np.cos(th) + self.yy * np.sin(th)) / 520.0
        band = 0.5 + 0.5 * np.cos(2 * np.pi * u)
        band = np.clip((band - 0.30) / 0.40, 0, 1)
        band = cv2.GaussianBlur(band, (0, 0), 6)
        L = luma(f)
        m = self.floor * (1 - np.clip((L - 0.70) / 0.15, 0, 1))      # 白飛びしている所にはかけない
        lit = 1 + m * (band * 0.07 - (1 - band) * 0.10)
        f = f * lit[..., None]
        # 窓の白飛びをカーブで抑える
        hi = np.clip((L - 0.80) / 0.20, 0, 1)[..., None]
        return np.clip(f * (1 - 0.08 * hi), 0, 1)


class PhotoTilt(Shot):
    """縦の写真を横の画面で、上（窓の海）から下（料理）へ一定の速さで見下ろす。"""
    def __init__(self, path, y0, y1, zoom0=1.0, zoom1=1.0, x=0.5):
        self.img = load_photo(path)
        self.y0, self.y1, self.z0, self.z1, self.x = y0, y1, zoom0, zoom1, x

    def frame(self, i, n):
        u = linear_move(i / max(1, n - 1))
        cy = self.y0 + (self.y1 - self.y0) * u
        z = self.z0 + (self.z1 - self.z0) * u
        return np.clip(affine(self.img, self.x, cy, z, 0.0), 0, 1)


class RiverMouth(Shot):
    """IMG_9665：高い所から見た砂州・河口・外洋。水平線の傾き（約 1.3°→2.2° 右下がり）をコマごとに直す。"""
    def __init__(self, t0=10.10):
        z = 1.18
        # 素材の水平線は上から約 50%。水平線を HORIZON_Y へ上げる
        cy = 0.50 + (H / 2 - HORIZON_Y) / (z * W / 2560) / 1440
        self.v = Video('IMG_9665.MOV', t0, rot=lambda i: -(1.3 + 0.9 * min(1, i / 140.0)), zoom=z, cx=0.55, cy=cy)

    def frame(self, i, n):
        return self.v.frame(i)


class DayToNight(Shot):
    """写真 28（昼）→ 12（夜）。同じ位置から撮った 2 枚を 6 点で合わせ、空が暮れ、軒の電球が左から順に灯り、
    窓が点き、最後に看板「南風」が光る（tools/v2/relight.py を作り直したもの。白く抜けずに夜で止まる）。"""
    DAY_PTS = [(425, 135), (457, 413), (510, 413), (423, 345), (152, 480), (748, 488)]
    NIGHT_PTS = [(430, 80), (457, 322), (505, 322), (427, 258), (155, 385), (693, 389)]

    def __init__(self, t_start=0.0, t_hold=0.80, k0=0.0, k1=1.0):
        day = load_photo(PHOTO_DAY, 2400)
        night = load_photo(PHOTO_NIGHT, 2048)
        h, w = day.shape[:2]
        nh, nw = night.shape[:2]
        sd = np.float32(self.DAY_PTS) * (w / 900.0)
        sn = np.float32(self.NIGHT_PTS) * (nw / 900.0)
        Hm, _ = cv2.findHomography(sn, sd, 0)
        nw_ = cv2.warpPerspective(night, Hm, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpPerspective(np.ones((nh, nw), np.float32), Hm, (w, h))
        valid = cv2.GaussianBlur(cv2.erode(valid, np.ones((31, 31), np.uint8)), (0, 0), w / 40)
        nw_ = np.clip(nw_ * 1.30, 0, 1)
        ld, ln = day.mean(axis=2), nw_.mean(axis=2)
        light = np.clip((ln - ld * 0.6 - 0.18) / 0.35, 0, 1) * valid
        light = cv2.GaussianBlur(light, (0, 0), w / 900)
        glow = cv2.GaussianBlur(nw_ * light[..., None], (0, 0), w / 120)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        order = 0.30 + 0.35 * xs / w
        ex, ey = sd[3]
        sign = np.exp(-(((xs - ex) / (w * 0.09)) ** 2 + ((ys - ey) / (h * 0.16)) ** 2))
        order = order * (1 - sign) + 0.72 * sign
        sky = np.median(nw_[(light < 0.05) & (valid > 0.9)][::50], axis=0).astype(np.float32)
        self.P = dict(day=day, night=nw_, valid=valid, light=light, glow=glow, order=order, sky=sky)
        self.t_start, self.t_hold, self.k0, self.k1 = t_start, t_hold, k0, k1
        self.cx, self.cy = 0.47, 0.50

    def look(self, t):
        P = self.P
        day, night, light, order, valid = P['day'], P['night'], P['light'], P['order'], P['valid']
        dusk = ease(np.clip(t / 0.65, 0, 1))
        v = valid[..., None]
        lum = day.mean(axis=2, keepdims=True)
        dark = day * (1 - dusk) + (P['sky'] * (0.6 + 0.8 * lum)) * dusk
        Lm = light[..., None]
        unlit = night * (1 - Lm) + dark * Lm
        base = dark * (1 - dusk * v) + unlit * (dusk * v)
        on = np.clip((t - order) / 0.10, 0, 1)
        on = on * on * (3 - 2 * on)                  # ふっと灯る（ちらつかせない）
        L = (light * on)[..., None]
        out = base * (1 - L) + night * L
        out = out + P['glow'] * (on[..., None] * 0.12) * ease(np.clip((t - 0.5) / 0.5, 0, 1))
        # ブルーアワーへ：明かりのない所（空・壁）を持ち上げ、深い青を残す（真っ暗だと昼の屋外で見えないため）
        amb = (1 - L) * dusk
        out = out * (1 + 0.9 * amb) + np.array([0.035, 0.075, 0.15], np.float32) * amb
        # 看板の光を白く飛ばさない（光の芯を 0.85 で丸める）
        hi = np.clip(out - 0.70, 0, None)
        out = np.where(out > 0.70, 0.70 + 0.15 * (1 - np.exp(-hi / 0.15)), out)
        return np.clip(out, 0, 1)

    def frame(self, i, n):
        u = i / max(1, n - 1)
        t = self.k0 + (self.k1 - self.k0) * np.clip((u - self.t_start) / max(1e-6, self.t_hold - self.t_start), 0, 1)
        img = self.look(float(t))
        z = 1.00 + 0.05 * linear_move(u)               # ごくゆっくり寄る
        return affine(img, self.cx, self.cy, z, 0.0)


# ---------------------------------------------------------------------------
# 色（シャドウ）と質感
# ---------------------------------------------------------------------------
def hue_weights(a):
    """色相ごとの彩度の倍率：全体 0.88、青（190〜225°）1.00、肌・料理（10〜50°）0.95、緑（80〜160°）0.85。"""
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)           # float32：H は 0..360
    hh = hsv[..., 0]

    def bump(c, wdt):
        d = np.minimum(np.abs(hh - c), 360 - np.abs(hh - c))
        return np.clip(1 - d / wdt, 0, 1)
    k = np.full(hh.shape, 0.88, np.float32)
    k += (1.00 - 0.88) * np.clip(bump(207, 30) * 1.6, 0, 1)
    k += (0.95 - 0.88) * np.clip(bump(30, 30) * 1.6, 0, 1)
    k += (0.85 - 0.88) * np.clip(bump(120, 50) * 1.4, 0, 1)
    return k


def grade(a):
    """表示値 → 表示値。暗部を沈めて締め（いちばん暗い所は紺で止める）、ハイライトは 0.95 で丸める。"""
    L = luma(a)[..., None]
    # 1) 暗部を沈める：0.25 以下を最大 3 割下げる（トゥ）
    toe = np.where(L < 0.25, 0.25 * (np.maximum(L, 1e-5) / 0.25) ** 1.35, L)
    a = a * (toe / np.maximum(L, 1e-5))
    # 2) ゆるい S 字（+10%、中心 0.45）
    x = np.clip(a, 0, 1)
    a = x + 0.10 * (x - 0.45) * (1 - np.abs(2 * x - 1) ** 1.5) * 1.2
    # 3) ハイライトの肩：0.78 から 0.95 へ丸める
    k0, top = 0.78, 0.95
    over = np.maximum(a - k0, 0)
    a = np.where(a > k0, k0 + (top - k0) * (1 - np.exp(-over / (top - k0))), a)
    # 4) 色相ごとの彩度
    a = np.clip(a, 0, 1).astype(np.float32)
    g = luma(a)[..., None]
    a = g + (a - g) * hue_weights(a)[..., None]
    # 5) 色の分け方：ハイライトに少し暖かさ、暗部は紺（B だけ）
    g = luma(a)[..., None]
    hi = np.clip((g - 0.55) / 0.35, 0, 1)
    lo = np.clip((0.35 - g) / 0.35, 0, 1)
    a = a + hi * np.array([0.015, 0.005, 0], np.float32) + lo * np.array([-0.01, 0, 0.02], np.float32)
    # 6) いちばん暗い所は紺で止める（真っ黒にしない）
    g = luma(a)[..., None]
    w = np.clip(1 - g / 0.12, 0, 1) ** 2
    a = a * (1 - w * 0.5) + np.maximum(a, FLOOR) * (w * 0.5)
    a = np.maximum(a, FLOOR * 0.8)
    return np.clip(a, 0, 1).astype(np.float32)


class Look:
    """にじみ・広い光・周辺減光・粒子。にじみのマスクは時間方向になめらかに（海面のきらめきをちらつかせない）。"""
    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)
        self.m_prev = None
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        # 周辺減光：映像の部分の半対角線を基準に、r=0.55 から四隅で ×0.86（リニア光）
        cy = H / 2
        hy = (H - 2 * BAR) / 2
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - cy) / hy) ** 2) / np.sqrt(2)
        self.vig = (1 - 0.14 * np.clip((r - 0.55) / 0.45, 0, 1) ** 2).astype(np.float32)
        self.active = ((yy >= BAR) & (yy < H - BAR)).astype(np.float32)

    def apply(self, a):
        lin = srgb_to_lin(a)
        Y = luma(lin)
        m = np.clip((Y - 0.78) / 0.15, 0, 1) ** 1.5
        if self.m_prev is not None:
            m = 0.6 * m + 0.4 * self.m_prev
        self.m_prev = m
        small = cv2.resize(lin * m[..., None], (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        hal = cv2.resize(cv2.GaussianBlur(small, (0, 0), 10 / 4), (W, H))
        blo = cv2.resize(cv2.GaussianBlur(small, (0, 0), 60 / 4), (W, H))
        lin = lin + hal * np.array([1.0, 0.50, 0.38], np.float32) * 0.10 + blo * 0.04
        lin = lin * self.vig[..., None]
        a = lin_to_srgb(lin)
        # 粒子：明るさだけ、中間調で強く（強さ 0.014）。映像の部分だけ
        n = cv2.GaussianBlur(self.rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.7)
        n /= n.std() + 1e-6
        Yd = luma(a)
        wgt = 0.35 + 0.65 * 4 * Yd * (1 - Yd)
        a = a + (0.014 * wgt * n * self.active)[..., None]
        return np.clip(a, 0, 1).astype(np.float32)


def bars(a):
    a = a.copy()
    a[:BAR] = 0
    a[H - BAR:] = 0
    return a


# ---------------------------------------------------------------------------
# 文字
# ---------------------------------------------------------------------------
_F = {}


def font(name, size):
    k = (name, size)
    if k not in _F:
        _F[k] = ImageFont.truetype(os.path.join(FONTS, name), size)
    return _F[k]


MINCHO = 'ShipporiMinchoB1-SemiBold.ttf'
MINCHO_M = 'ShipporiMinchoB1-Medium.ttf'


def text_layer(lines):
    """lines：(text, x, y, size, font, track_em, anchor 'm'|'r'|'l')。オフホワイトの文字と、紺のやわらかい 2 層の影。"""
    im = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(im)
    for text, x, y, size, fn, tr, an in lines:
        f = font(fn, size)
        ws = [f.getlength(ch) for ch in text]
        total = sum(ws) + tr * size * (len(text) - 1)
        x0 = x - total / 2 if an == 'm' else x - total if an == 'r' else x
        for ch, wd in zip(text, ws):
            d.text((x0, y), ch, font=f, fill=255, anchor='ls')
            x0 += wd + tr * size
    a = np.asarray(im).astype(np.float32) / 255
    size = max(l[3] for l in lines)
    sh1 = np.roll(cv2.GaussianBlur(a, (0, 0), max(1.0, size * 0.12)), 2, axis=0)
    sh2 = cv2.GaussianBlur(a, (0, 0), max(2.0, size * 0.35))
    return a, np.clip(sh1 * 0.45 + sh2 * 0.20 * 2.0, 0, 1)


def put_text(frame, layer, alpha, dy=0.0):
    if alpha <= 0:
        return frame
    a, sh = layer
    if dy:
        M = np.float32([[1, 0, 0], [0, 1, dy]])
        a = cv2.warpAffine(a, M, (W, H))
        sh = cv2.warpAffine(sh, M, (W, H))
    s = (sh * alpha)[..., None]
    f = frame * (1 - s) + NAVY_SH * s
    t = (a * alpha)[..., None]
    return f * (1 - t) + OFFWHITE * t


def text_alpha(t, t0, t1, fade=0.5):
    """0.5 秒でふわっと出て、t1 まで置き、0.5 秒で消える。dy は 4px の浮き上がり。"""
    if t < t0 or t > t1 + fade:
        return 0.0, 0.0
    if t < t0 + fade:
        u = ease((t - t0) / fade, 'out')
        return u, 4 * (1 - u)
    if t > t1:
        return 1 - ease((t - t1) / fade, 'in'), 0.0
    return 1.0, 0.0


# ---------------------------------------------------------------------------
# タイムライン
# ---------------------------------------------------------------------------
PHRASE_Y = BAR + 0.80 * (H - 2 * BAR)        # 句の高さ（映像の部分の下 1/3 の中央 ≒ 828）
# END：夜の建物の屋根より上の暗い空に置く（看板の光と、軒の「Blue River Meets…」の帯に重ねない）
END_LINES = [('Seaside MAZE（南風・まぜ）', W / 2, 168, 64, MINCHO, 0.06, 'm'),
             ('高知県土佐市新居・国道56号沿い', W / 2, 236, 40, MINCHO_M, 0.12, 'm')]


def plan(kind):
    """cuts：(名前, 頭のコマ, 終わりのコマ, Shot を作る関数)。境界のディゾルブ：dis[i]＝カット i と i+1 の間の長さ（最後は輪）。
    texts：(行, 出る秒, 消え始める秒)。"""
    if kind == '30':
        total = 900
        cuts = [('beach', 0, 150, lambda: Beach(5.10)),
                ('window', 150, 300, lambda: Window(6.20)),
                ('pasta', 300, 435, lambda: PhotoTilt(PASTA, 0.255, 0.66)),
                ('cake', 435, 555, lambda: PhotoTilt(CAKE, 0.52, 0.55, 1.00, 1.03)),
                ('river', 555, 660, lambda: RiverMouth(10.10)),
                ('night', 660, 900, lambda: DayToNight(t_start=0.12, t_hold=0.80))]
        dis = [24, 24, 24, 24, 45, 60]
        texts = [([('海と、', W / 2, PHRASE_Y, 84, MINCHO, 0.14, 'm')], 1.0, 3.6),
                 ([('時間と、', W / 2, PHRASE_Y, 84, MINCHO, 0.14, 'm')], 6.0, 8.6),
                 ([('おいしいもの。', W / 2, PHRASE_Y, 84, MINCHO, 0.14, 'm')], 15.0, 17.6),
                 (END_LINES, 25.0, 28.8)]
        corner = (0.6, 21.0)
    else:
        total = 450
        cuts = [('beach', 0, 120, lambda: Beach(5.10)),
                ('window', 120, 240, lambda: Window(6.50)),
                ('pasta', 240, 345, lambda: PhotoTilt(PASTA, 0.36, 0.64)),
                ('night', 345, 450, lambda: DayToNight(t_start=0.0, t_hold=0.75, k0=0.30, k1=1.0))]
        dis = [24, 24, 30, 60]
        texts = [([('海と、', W / 2, PHRASE_Y, 84, MINCHO, 0.14, 'm')], 0.8, 3.0),
                 ([('時間と、', W / 2, PHRASE_Y, 84, MINCHO, 0.14, 'm')], 4.6, 6.9),
                 ([('おいしいもの。', W / 2, PHRASE_Y, 84, MINCHO, 0.14, 'm')], 8.6, 10.7),
                 (END_LINES, 12.0, 14.0)]
        corner = (0.5, 11.0)
    return total, cuts, dis, texts, corner


class Timeline:
    def __init__(self, kind):
        self.total, cuts, self.dis, texts, self.corner = plan(kind)
        nc = len(cuts)
        self.cuts = []
        for k, (name, s, e, mk) in enumerate(cuts):
            hin = self.dis[k - 1] // 2 if k > 0 else self.dis[-1] // 2
            hout = self.dis[k] - self.dis[k] // 2
            a0, a1 = s - hin, e + hout                      # このカットが画面にある範囲（頭は負、最後は total を超えうる）
            self.cuts.append(dict(name=name, s=s, e=e, a0=a0, a1=a1, mk=mk, shot=None, n=a1 - a0))
        self.texts = [(text_layer(lines), t0, t1) for lines, t0, t1 in texts]
        self.corner_layer = text_layer([('Seaside MAZE（南風・まぜ）', W - 96, H - BAR + 38, 26, MINCHO_M, 0.08, 'r')])
        self.look = Look()

    def weights(self, f):
        """ファイルのコマ f に出ているカットと重み（輪として、f と f±total の両方を見る）。"""
        out = []
        for k, c in enumerate(self.cuts):
            for ff in (f, f + self.total, f - self.total):
                if c['a0'] <= ff < c['a1']:
                    w = 1.0
                    hin = c['s'] - c['a0']
                    hout = c['a1'] - c['e']
                    if ff < c['s'] + hin:                      # 入りのディゾルブ（境界 s を中心に）
                        w = min(w, ease((ff - c['a0'] + 0.5) / (2 * hin)) if hin else 1.0)
                    if ff >= c['e'] - hout:                    # 出のディゾルブ
                        w = min(w, 1 - ease((ff - (c['e'] - hout) + 0.5) / (2 * hout)) if hout else 1.0)
                    out.append((k, ff - c['a0'], w))
        s = sum(w for _, _, w in out)
        return [(k, i, w / s) for k, i, w in out]

    def shot(self, k):
        c = self.cuts[k]
        if c['shot'] is None:
            c['shot'] = c['mk']()
        return c['shot']

    def loop_dip(self, f):
        """ループの継ぎ目なら、夜 → 紺 → 朝の位置 u（0..1）を返す。それ以外は None。"""
        ho = self.cuts[-1]['a1'] - self.cuts[-1]['e']
        hi = self.cuts[0]['s'] - self.cuts[0]['a0']
        if f >= self.total - ho:
            p = f - (self.total - ho)
        elif f < hi:
            p = f + ho
        else:
            return None
        return (p + 0.5) / (ho + hi)

    def render(self, f):
        u = self.loop_dip(f)
        if u is not None:
            if u < 0.5:          # 夜が紺へ沈む
                k, i = len(self.cuts) - 1, (f if f >= self.total // 2 else f + self.total) - self.cuts[-1]['a0']
                fr, w = self.shot(k).frame(i, self.cuts[k]['n']), ease(u * 2)
            else:                # 紺から朝の海が上がる
                k, i = 0, (f if f < self.total // 2 else f - self.total) - self.cuts[0]['a0']
                fr, w = self.shot(k).frame(i, self.cuts[k]['n']), 1 - ease((u - 0.5) * 2)
            lin = srgb_to_lin(fr) * (1 - w) + srgb_to_lin(NAVY_DIP) * w
        else:
            lin = None
            for k, i, w in self.weights(f):
                fr = self.shot(k).frame(i, self.cuts[k]['n'])
                x = srgb_to_lin(fr) * w
                lin = x if lin is None else lin + x
        a = lin_to_srgb(lin)
        a = grade(a)
        a = self.look.apply(a)
        a = bars(a)
        t = f / FPS
        for layer, t0, t1 in self.texts:
            al, dy = text_alpha(t, t0, t1)
            a = put_text(a, layer, al, dy)
        c0, c1 = self.corner
        al, _ = text_alpha(t, c0, c1, fade=0.6)
        a = put_text(a, self.corner_layer, al * 0.85)
        return np.clip(a, 0, 1)


def encode(path, total, frames):
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-s', '%dx%d' % (W, H),
                          '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '15',
                          '-tune', 'grain', '-profile:v', 'high', '-level', '4.1', '-pix_fmt', 'yuv420p',
                          '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
                          '-movflags', '+faststart', path], stdin=subprocess.PIPE)
    rng = np.random.default_rng(3)
    stats = []
    for f, a in frames:
        stats.append(float(luma(a[BAR:H - BAR]).mean()))
        q = np.clip(a * 65535 + rng.uniform(-0.5, 0.5, a.shape), 0, 65535).astype('<u2')
        p.stdin.write(q.tobytes())
        if f % 30 == 0:
            print('  %5.1fs  APL %.3f' % (f / FPS, stats[-1]), flush=True)
    p.stdin.close()
    p.wait()
    return stats


def check(stats):
    d = np.abs(np.diff(stats + stats[:1]))
    flashes = [i for i, v in enumerate(d) if v > 0.10]
    return dict(apl_min=round(min(stats), 3), apl_max=round(max(stats), 3), max_step=round(float(d.max()), 4),
                over_0_1=len(flashes), loop_step=round(abs(stats[-1] - stats[0]), 4))


def main():
    args = sys.argv[1:]
    stills = None
    if '--stills' in args:
        j = args.index('--stills')
        stills = [float(x) for x in args[j + 1:]]
        args = args[:j]
    for kind in args:
        tl = Timeline(kind)
        if stills is not None:
            d = os.path.join(OUT, 'sg_preview')
            os.makedirs(d, exist_ok=True)
            for s in stills:
                f = int(round(s * FPS)) % tl.total
                a = tl.render(f)
                Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(os.path.join(d, 'sg%s_%05.2f.png' % (kind, s)))
                print('still', kind, s)
            continue
        path = os.path.join(OUT, 'MAZE_v3_sg_%ss.mp4' % kind)
        stats = encode(path, tl.total, ((f, tl.render(f)) for f in range(tl.total)))
        rep = check(stats)
        json.dump(dict(stats=stats, **rep), open(os.path.join(OUT, 'sg%s_check.json' % kind), 'w'))
        print('wrote', path, rep)


if __name__ == '__main__':
    main()
