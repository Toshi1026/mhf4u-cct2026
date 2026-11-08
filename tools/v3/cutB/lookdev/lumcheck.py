"""ビートの mp4（exports/v3/preview_B/bXX.mp4）から、となりのフレームの平均の明るさの差を測る（点滅の規則 ≤0.1、目標 ≤0.05）。
ビートの中の最大と、つなぎ目（前のビートの最後 → 次のビートの最初）の差。python3 cutB/lookdev/lumcheck.py 01 02 03 ..."""
import os
import subprocess
import sys

import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'exports', 'v3', 'preview_B')


def lumas(key):
    p = os.path.join(D, 'b%s.mp4' % key)
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', p, '-vf', 'scale=270:480', '-f', 'rawvideo', '-pix_fmt',
                          'rgb24', '-'], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, 480, 270, 3).astype(np.float32) / 255
    return (a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722).mean((1, 2)), os.path.getmtime(p)


keys = sys.argv[1:] or ['01', '02', '03', '04', '05', '06', '07', 'T', '08']
prev = None
for k in keys:
    l, mt = lumas(k)
    d = np.abs(np.diff(l))
    j = int(d.argmax())
    s = 'b%s  n=%d  mean %.3f  max|Δ| in beat %.4f (f%d→%d)' % (k, len(l), l.mean(), d.max(), j, j + 1)
    if prev is not None:
        s += '   seam from b%s: %.4f' % (prev[0], abs(l[0] - prev[1]))
    print(s)
    prev = (k, l[-1])
