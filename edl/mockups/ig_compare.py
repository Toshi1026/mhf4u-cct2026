#!/usr/bin/env python3
"""Instagramの2本（IG30・IG15）の End Card を左右に並べる確認用の画像。

どちらも共通の数値表 ig_endcard_spec.py から描く（IG30＝ig30_render、IG15＝ig15_render）。
  python3 edl/mockups/ig_compare.py
    ig_compare_endcards.png            2本の End Card（全部が出そろった瞬間）を半分の大きさで左右に並べる
    ig_compare_endcards_safezones.png  同じ2枚に、共通のセーフゾーンを重ねたもの
"""
import os
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ig_endcard_spec as C  # noqa: E402
import ig15_render as R15  # noqa: E402
import ig30_render as R30  # noqa: E402

OUT = R15.OUT
T30, T15 = 27.20, 14.50


def main():
    c30 = os.path.join(tempfile.gettempdir(), 'maze_ig30_frames')
    c15 = os.path.join(tempfile.gettempdir(), 'maze_ig15_frames')
    os.makedirs(c30, exist_ok=True)
    os.makedirs(c15, exist_ok=True)
    p30, _ = R30.layout()
    L30 = C.stage_layers(p30)
    L30['copy'] = R30.copy_layer()[0]
    p15, _ = R15.layout()
    L15 = C.stage_layers(p15)
    a, _ = R30.render(T30, c30, L30)
    b, _, _ = R15.render(T15, L15, c15)
    f = ImageFont.truetype(C.font_path('Medium'), 24)
    hw, hh = C.W // 2, C.H // 2
    for name, conv in (('ig_compare_endcards.png', lambda im: im),
                       ('ig_compare_endcards_safezones.png', C.safezones)):
        cmp = Image.new('RGB', (hw * 2 + 30, hh + 60), (18, 18, 18))
        cmp.paste(conv(a).resize((hw, hh), Image.LANCZOS), (0, 60))
        cmp.paste(conv(b).resize((hw, hh), Image.LANCZOS), (hw + 30, 60))
        d = ImageDraw.Draw(cmp)
        d.text((12, 16), 'IG30 %.2f秒（IMG_9661。空にロゴ・海に2行）' % T30, font=f, fill=(255, 230, 120))
        d.text((hw + 42, 16), 'IG15 %.2f秒（IMG_9633。海の上に横並び）' % T15, font=f, fill=(255, 230, 120))
        cmp.save(os.path.join(OUT, name))
        print(os.path.join(OUT, name))


if __name__ == '__main__':
    main()
