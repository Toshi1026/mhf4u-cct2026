"""案C 16（約2.4秒・声なし）エンドカード：夜に光る看板（写真 41）→ カフェのロゴ（assets/logo_white.png）と施設名。

 0–：15 の夜（露出を落とした浜）から、夜の看板（南風のエンブレム）。縦の窓（写真の高さいっぱい）からエンブレムへゆっくり寄る。
     看板の明かりは暖色なので、案C のパレット（オレンジ・黄・金を使わない）に合わせて、明かりの色を冷たい白へ寄せる
     （彩度を落として、わずかに青へ）。窓の黄色い明かりも同じ。15 の最後（平均の明るさ 約 0.2）から段差 0.1 未満で始まる。
 BLUR0–BLUR1：看板がゆっくりぼけて（σ 0→44px）暗くなり（×0.50）、深い海の青へ寄せて、ロゴの背景のやわらかい光になる。
 LOGO0–：ロゴ（白の線画）が、中心から外へ広がるやわらかい円のマスクで現れ、1.035→1.0 にわずかに縮み、ぼかし 5→0。
 NAME0–：「Seaside MAZE（南風・まぜ）」（64px Bold、Seaside MAZE＝ACC。12・15 と同じ強調）が 1 文字ずつ上がる。
 ADDR0–：「高知県土佐市新居」（40px Medium）が左から右へやわらかいマスクで出る。
 すべて 48 フレームまでに止まり、最後の約 0.8 秒はそのまま置く。
文字とロゴはセーフゾーンの中（ロゴ y 455〜835、施設名 y≈950、住所 y≈1022）。
"""
import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W, H
from cutB.beats import _fac as F
from cutB.beats import _motion as Mo
from cutC.beats import _maze as Z

SIGN = dict(cx=0.52, cy0=0.50, cy1=0.47, z1=1.16, emblem=(0.52, 0.45))
BLUR0, BLUR1, BLUR_MAX, DIM = 14, 38, 44.0, 0.50
LOGO0, LOGO_DUR = 24, 16
NAME0, ADDR0 = 30, 38
TINT = np.array([0.80, 0.93, 1.12], np.float32)     # ぼけた背景を深い海の青へ（案C のパレット）
LOGO_W, LOGO_CY = 400, 645
NAME_Y, ADDR_Y = 950, 1022
START_GAIN = 0.92


def _cool(a):
    """看板の暖色の明かりを冷たい白へ：彩度を 0.25 倍にして、ほんの少し青へ。"""
    l = a.mean(2, keepdims=True)
    a = l + (a - l) * 0.25
    return np.clip(a * np.array([0.96, 0.99, 1.05], np.float32), 0, 1)


def _logo():
    lg = Z.logo_white()
    h, w = lg.shape[:2]
    s = LOGO_W / float(w)
    lg = cv2.resize(lg, (LOGO_W, round(h * s)), interpolation=cv2.INTER_AREA)
    rgb = lg[..., :3] * 0 + core.FOAM * 0.97
    a = lg[..., 3:4]
    return np.concatenate([rgb, a], 2).astype(np.float32)


def _put_logo(f, lg, i):
    q = Mo.ease((i - LOGO0) / float(LOGO_DUR), 'cubic_out')
    if q <= 0:
        return f
    sc = 1.035 - 0.035 * Mo.ease((i - LOGO0) / float(LOGO_DUR + 6), 'quart_out')
    h, w = lg.shape[:2]
    cx, cy = W / 2, LOGO_CY
    M = np.float32([[sc, 0, cx - sc * w / 2], [0, sc, cy - sc * h / 2]])
    L = cv2.warpAffine(lg, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    bl = 5.0 * (1 - q)
    if bl > 0.3:
        L = cv2.GaussianBlur(L, (0, 0), bl)
    # 中心から外へ広がるやわらかい円のマスク
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    R = 40 + q * 320
    m = np.clip((R - r) / 90.0, 0, 1)[..., None]
    a = L[..., 3:4] * m * min(1.0, q * 1.6)
    # ロゴの後ろにごく弱い影（暗い背景でも線が沈まないように）
    sh = cv2.GaussianBlur(a[..., 0], (0, 0), 10)[..., None]
    f = f * (1 - 0.35 * sh)
    return f * (1 - a) + L[..., :3] * a


def render(beat, ctx):
    n = beat['n']
    img = _cool(F.grade_photo(F.load(Z.NIGHT_SIGN, 2048)))
    vmax = F.max_vw(img)
    lg = _logo()
    kn = Mo.KText([dict(text='Seaside MAZE（南風・まぜ）', x=W / 2 + 12, y=NAME_Y, size=64, weight='Bold', hl='Seaside MAZE',
                        track=0.02, shadow=0.9)])
    ka = Mo.KText([dict(text='高知県土佐市新居', x=W / 2, y=ADDR_Y, size=40, weight='Medium', track=0.10, shadow=0.9)])
    for i in range(n):
        u = Mo.ease(i / float(n - 1), 'sine_out')
        z = SIGN['z1'] ** u
        f = F.view(img, SIGN['cx'], core.lerp(SIGN['cy0'], SIGN['cy1'], u), vmax / z, interp=cv2.INTER_CUBIC)
        if i < 6:
            f = f * core.lerp(START_GAIN, 1.0, Mo.ease(i / 6.0, 'sine_out'))
        b = Mo.ease((i - BLUR0) / float(BLUR1 - BLUR0), 'sine_inout')
        if b > 0:
            f = cv2.GaussianBlur(f, (0, 0), 0.5 + BLUR_MAX * b) * (1 - (1 - DIM) * b)
            f = f * (1 + (TINT - 1) * b)
        f = _put_logo(f, lg, i)
        f = kn.reveal(f, i, NAME0, 'char_rise', stagger=1.0, dur=10, dy_em=16.0 / 64, blur=5.0, track_from=0.06)
        f = ka.reveal(f, i, ADDR0, 'wipe', dur=12, feather=60.0)
        yield np.clip(f, 0, 1).astype(np.float32)
