#!/usr/bin/env python3
"""サイネージ2本（15秒版・30秒版）の、同じ種類の瞬間を並べる確認用の画像。

どちらも共通の数値表 signage_spec.py から描く（15秒版＝signage15_render、30秒版＝signage30_render）。
ENDは2026-09-23に決まった③（中央に正式ロゴ＋「MAZE WIND~RETREAT」＋「高知県土佐市」）だけ。
  python3 edl/mockups/signage_compare.py
    signage_compare_15s_30s.png   上：右下がいちばん明るい瞬間、下：ENDが完全に見えている瞬間
    signage_compare_corner_2x.png 右下のマークを1080pの画素のまま2倍に拡大して上下に並べる（大きさ・位置が同じことの確認）
    signage_compare_end.png       2本のENDを左右に並べる（ENDの1組の大きさ・位置が同じことの確認）
"""
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import signage_spec as S  # noqa: E402,F401
import signage15_render as R15  # noqa: E402
import signage30_render as R30  # noqa: E402

OUT = R15.OUT
W, H = 1920, 1080
T15_BRIGHT, T15_END = 2.00, 12.00
T30_BRIGHT, T30_END = 2.00, 26.00


def half(im, text):
    return R15.label(im.resize((W // 2, H // 2), Image.LANCZOS), text, 22)


def main():
    a15 = R15.render(T15_BRIGHT)[0]
    b15 = R15.render(T15_END)[0]
    a30 = R30.render(T30_BRIGHT)[0]
    b30 = R30.render(T30_END)[0]
    R15.grid([half(a15, '15秒版 %.2f秒（右下がいちばん明るい）' % T15_BRIGHT),
              half(a30, '30秒版 %.2f秒（右下がいちばん明るい）' % T30_BRIGHT),
              half(b15, '15秒版 %.2f秒（END）' % T15_END),
              half(b30, '30秒版 %.2f秒（END）' % T30_END)], 2).save(os.path.join(OUT, 'signage_compare_15s_30s.png'))

    box = (1280, 900, 1920, 1060)
    crops = [R15.label(im.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.NEAREST), txt, 20)
             for im, txt in ((a15, '15秒版 %.2f秒（IMG_9665）' % T15_BRIGHT),
                             (a30, '30秒版 %.2f秒（IMG_9631）' % T30_BRIGHT))]
    R15.grid(crops, 1).save(os.path.join(OUT, 'signage_compare_corner_2x.png'))

    R15.grid([half(b15, '15秒版 END %.2f秒（IMG_9631、ENDのカットの頭から3.00秒）' % T15_END),
              half(b30, '30秒版 END %.2f秒（IMG_9658、ENDのカットの頭から5.30秒）' % T30_END)], 2).save(
        os.path.join(OUT, 'signage_compare_end.png'))


if __name__ == '__main__':
    main()
