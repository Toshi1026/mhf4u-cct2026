"""案C 11〜16（MAZE のパート：実写と写真）の共通部品。案B の 08〜13（cutB/beats/b08〜b13）の作り方をそのまま使う。

- mf(beat, k, mora)：k 番目のモーラ（moras.json）のフレーム（ビートの頭から）。読みが変わったら assert で止まる。
- LP の空撮（edl/lp/img/about_s02.jpg：仁淀川河口大橋と河口、奥の橋のたもとが MAZE）と、夜の看板（写真 41）、
  カフェのロゴ（assets/logo_white.png）の読み込み。
- 色は案A/B の実写と同じ grade（_fac.grade_photo：寒色寄り・すっきり）。
- 階と階のつなぎ（12→13→14）は案B と同じ縦の振り（_p25.whip_out / whip_in、+1＝カメラが上へ）。
"""
import os

import cv2
import numpy as np
from PIL import Image

from cutB.beats import _useA  # noqa: F401
import core
from cutB.beats import _fac as F

LP = os.path.join(core.ROOT, 'edl', 'lp', 'img')
CACHE = os.path.join(core.OUT, 'cache')
NIGHT_SIGN = 'A6247107-2DEC-4BED-B889-68ADE9E350CF.JPEG'      # 写真 41：夜に光る南風のエンブレム
BUILDING_SIDE = 'CBE2B753-4892-4525-BBEF-6B47AAAAD4BD.JPEG'   # 写真 56：建物の横と避難タワー


def mf(beat, k, mora=None):
    if mora is not None:
        assert beat['moras'][k][0] == mora, (beat['key'], k, beat['moras'])
    return round(beat['moras'][k][1] * core.FPS) - beat['f0']


def lum(f):
    return F.luma(f)


def load_lp(name, scale=1.0):
    """LP の画像（float RGB 0..1）。scale>1 は Lanczos で拡大して、ごく弱いアンシャープ（拡大のぼけを少し戻す）。"""
    p = os.path.join(CACHE, 'lp_%s_%.2f.npy' % (os.path.splitext(name)[0], scale))
    if os.path.exists(p):
        return np.load(p).astype(np.float32) / 255
    im = Image.open(os.path.join(LP, name)).convert('RGB')
    if scale != 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255
    if scale > 1.0:
        b = cv2.GaussianBlur(a, (0, 0), 1.2 * scale)
        a = np.clip(a + 0.35 * (a - b), 0, 1)
    np.save(p, (a * 255 + 0.5).astype(np.uint8))
    return a


def logo_white():
    """assets/logo_white.png（白の線画・透明背景）の RGBA float。"""
    im = Image.open(os.path.join(core.ROOT, 'assets', 'logo_white.png')).convert('RGBA')
    return np.asarray(im).astype(np.float32) / 255
