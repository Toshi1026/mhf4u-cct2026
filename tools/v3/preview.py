"""1つのビートだけ描いて確認する。
  python3 preview.py 03            → exports/v3/preview/b03.png（約12枚の縮小を並べた1枚）
  python3 preview.py 03 --mp4      → さらに exports/v3/preview/b03.mp4（そのビートだけの動画、無音）
  python3 preview.py 03 --frames 0,30,63 → 指定フレームを原寸 PNG で exports/v3/preview/b03_f0030.png など
"""
import importlib
import os
import sys

import numpy as np

import core
import timeline

key = sys.argv[1].zfill(2)
beat = next(b for b in timeline.load() if b['key'] == key)
mod = importlib.import_module('beats.b%s' % key)
want = None
if '--frames' in sys.argv:
    want = set(int(v) for v in sys.argv[sys.argv.index('--frames') + 1].split(','))
pick = set(np.linspace(0, beat['n'] - 1, 12).round().astype(int))
enc = core.Encoder(os.path.join(core.OUT, 'preview', 'b%s.mp4' % key), beat['n']) if '--mp4' in sys.argv else None
shots = []
for i, f in enumerate(mod.render(beat, {})):
    if int(key) <= 11:
        f = core.cg_label(f)
    if enc:
        enc.write(f)
    if i in pick:
        shots.append(f)
    if want and i in want:
        core.save_png(f, os.path.join(core.OUT, 'preview', 'b%s_f%04d.png' % (key, i)))
if enc:
    enc.close()
core.contact(shots, os.path.join(core.OUT, 'preview', 'b%s.png' % key), cols=6, scale=0.25)
print('ok', key, beat['n'], 'frames')
