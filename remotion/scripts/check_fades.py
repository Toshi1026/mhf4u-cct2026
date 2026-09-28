#!/usr/bin/env python3
"""透過PNGのフェードの確認：キーフレームのコマと前後1コマで、PNGの見える所（アルファ>0）だけの差を見る。

  python3 scripts/check_fades.py [Composition ...]
出力と見本の「PNGの所の明るさ」を並べ、フェードが1コマでもずれていれば大きな差（数段）として出る。
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify as V  # noqa: E402

tl = json.load(open(os.path.join(V.RM, 'src', 'timeline.json'), encoding='utf-8'))
d = os.path.join(V.ROOT, 'exports', 'final')
for c in tl['compositions']:
    if sys.argv[1:] and c['id'] not in sys.argv[1:]:
        continue
    W, H = c['width'], c['height']
    out, ref = os.path.join(d, c['output']), os.path.join(V.ROOT, c['reference'])
    for o in c['overlays']:
        alpha = np.asarray(Image.open(os.path.join(V.ROOT, o['file'])).convert('RGBA'))[..., 3]
        m = alpha > 128
        ks = sorted({f + k for f, _ in o['keyframes'] for k in (-1, 0, 1) if 0 <= f + k < c['durationInFrames']})
        A, B = V.frames_rgb(out, ks, W, H), V.frames_rgb(ref, ks, W, H)
        worst = 0
        cells = []
        for i, f in enumerate(ks):
            a, b = A[i][m].astype(float).mean(), B[i][m].astype(float).mean()
            worst = max(worst, abs(a - b))
            cells.append('%d:%.1f/%.1f' % (f, a, b))
        print('%-14s %-22s 最大差%.2f  ' % (c['id'], o['label'], worst) + ' '.join(cells))
