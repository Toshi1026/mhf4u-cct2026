#!/usr/bin/env python3
"""OffthreadVideo の色ずれを打ち消す表（src/colorFix.json）を作る。ふだんは実行しなくてよい。

なぜ必要か：Remotion は OffthreadVideo のコマを、同梱の ffmpeg（swscale の速い経路）で
YUV→RGB に直してからブラウザに渡す。この経路は丸めが切り捨て寄りで、RGBが平均で
R −1.0／G −1.5／B −1.0（0〜255）暗くなる（測定値。見本の zscale / accurate_rnd と比べて）。
書き出しで BT.709 に戻しても、Y が約 −1.15 下がったまま残る。
そこで、各チャンネルの「速い経路の値 → 正確な変換での平均値」の表を素材から実測し、
クリップの層にだけ SVG の feComponentTransfer（type=table、256段＝8bitの完全な表）でかける。
黒（0）は0のまま（仮スレートの黒を持ち上げない）。

  python3 scripts/calibrate_color.py
"""
import glob
import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RM = os.path.dirname(HERE)
ROOT = os.path.dirname(RM)
FF = os.environ.get('VERIFY_FFMPEG',
                    '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')


def rgb(path, flags, n):
    r = subprocess.run([FF, '-v', 'error', '-i', path, '-vf',
                        "select='eq(n\\,%d)',scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=%s,format=rgb24"
                        % (n, flags), '-vsync', '0', '-frames:v', '1', '-f', 'image2pipe', '-c:v', 'ppm', '-'],
                       capture_output=True, check=True)
    from io import BytesIO
    from PIL import Image
    return np.asarray(Image.open(BytesIO(r.stdout)).convert('RGB')).astype(int)


def main():
    cap = os.path.join(ROOT, 'exports', 'capcut')
    files = [f for f in sorted(glob.glob(os.path.join(cap, 'signage*', '*.mp4')) +
                               glob.glob(os.path.join(cap, 'proposals', '*', '*.mp4')))
             if 'placeholder' not in f.lower() and 'option' not in f]
    S = np.zeros((3, 256))
    N = np.zeros((3, 256))
    before = []
    for f in files:
        for n in (0, 20):
            # swscale の既定（Remotion のコンポジタと同じ結果になることを確認済み）と、正確な変換
            a, b = rgb(f, 'bilinear', n), rgb(f, 'bilinear+accurate_rnd+full_chroma_int', n)
            before.append(np.abs(a - b).mean())
            for c in range(3):
                np.add.at(S[c], a[..., c].ravel(), b[..., c].ravel())
                np.add.at(N[c], a[..., c].ravel(), 1)
    x = np.arange(256)
    lut = []
    for c in range(3):
        est = np.where(N[c] > 50, S[c] / np.maximum(N[c], 1), np.nan)
        off = np.nanmedian(est - x)
        est = np.where(np.isnan(est), x + off, est)
        t = np.clip(np.round(est), 0, 255).astype(int)
        t[0] = 0
        t = np.maximum.accumulate(t)  # 単調にする
        lut.append(t.tolist())
    doc = dict(_note='scripts/calibrate_color.py が作る。OffthreadVideo（swscaleの速い経路）の切り捨てを打ち消す 8bit の表',
               clips=len(files), mad_before=round(float(np.mean(before)), 3),
               mean_offset=[round(float(np.mean(np.array(lut[c]) - x)), 3) for c in range(3)], r=lut[0], g=lut[1], b=lut[2])
    json.dump(doc, open(os.path.join(RM, 'src', 'colorFix.json'), 'w'), indent=0)
    print('素材%d本・変換前のずれ（平均絶対差）%.3f  表の平均の持ち上げ R/G/B = %s' % (len(files), doc['mad_before'], doc['mean_offset']))


if __name__ == '__main__':
    main()
