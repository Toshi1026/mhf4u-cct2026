#!/usr/bin/env python3
"""サイネージ2本（15秒版・30秒版）の、同じ種類の瞬間を並べる確認用の画像。

どちらも共通の数値表 signage_spec.py から描く（15秒版＝signage15_render、30秒版＝signage30_render）。
  python3 edl/mockups/signage_compare.py
    signage_compare_15s_30s.png   上：右下がいちばん明るい瞬間、下：ENDの中央が完全に見えている瞬間（既定）
    signage_compare_corner_2x.png 右下のマークを1080pの画素のまま2倍に拡大して上下に並べる（大きさ・位置が同じことの確認）
    signage_compare_end_AB.png    ENDの見せ方の選択肢。行＝既定（常時表示）／代案A／代案A＋店名の行（提案）、列＝15秒版／30秒版
"""
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import signage_spec as S  # noqa: E402
import signage15_render as R15  # noqa: E402
import signage30_render as R30  # noqa: E402

OUT = R15.OUT
W, H = 1920, 1080
T15_BRIGHT, T15_END = 2.00, 12.00
T30_BRIGHT, T30_END = 2.00, 26.00


def r30(t, mode='default', proposal=False):
    if not proposal:
        return R30.render(t, W, H, mode=mode)[0]
    fr, _ = R30.plate(t, W, H, mode)
    tm = S.timing(mode, R30.E, R30.T)
    fr = S.comp(fr, 'corner', S.envelope(t, tm['corner']))
    ink, sh, _ = R15.proposal_planes(W, H)
    return R15.comp_planes(fr, ink, sh, S.envelope(t, tm['end']))


def half(im, text):
    return R15.label(im.resize((W // 2, H // 2), Image.LANCZOS), text, 22)


def main():
    a15 = R15.render(T15_BRIGHT)[0]
    b15 = R15.render(T15_END)[0]
    a30 = r30(T30_BRIGHT)
    b30 = r30(T30_END)
    R15.grid([half(a15, '15秒版 %.2f秒（右下がいちばん明るい）' % T15_BRIGHT),
              half(a30, '30秒版 %.2f秒（右下がいちばん明るい）' % T30_BRIGHT),
              half(b15, '15秒版 %.2f秒（END・既定）' % T15_END),
              half(b30, '30秒版 %.2f秒（END・既定）' % T30_END)], 2).save(os.path.join(OUT, 'signage_compare_15s_30s.png'))

    box = (1280, 900, 1920, 1060)
    crops = [R15.label(im.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.NEAREST), txt, 20)
             for im, txt in ((a15, '15秒版 %.2f秒' % T15_BRIGHT), (a30, '30秒版 %.2f秒' % T30_BRIGHT),
                             (b15, '15秒版 %.2f秒' % T15_END), (b30, '30秒版 %.2f秒' % T30_END))]
    R15.grid(crops, 1).save(os.path.join(OUT, 'signage_compare_corner_2x.png'))

    rows = [half(b15, '既定：右下を常時表示＋中央「高知県土佐市」（15秒版）'),
            half(b30, '既定（30秒版）'),
            half(R15.render(T15_END, mode='alt_a')[0], '代案A：右下から中央のロゴ＋所在地へ（15秒版）'),
            half(r30(T30_END, 'alt_a'), '代案A（30秒版）'),
            half(R15.render(T15_END, mode='alt_a', proposal=True)[0], '代案A＋店名の行（提案・15秒版）'),
            half(r30(T30_END, 'alt_a', proposal=True), '代案A＋店名の行（提案・30秒版）')]
    R15.grid(rows, 2).save(os.path.join(OUT, 'signage_compare_end_AB.png'))


if __name__ == '__main__':
    main()
