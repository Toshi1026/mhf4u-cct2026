"""案B の1ビートだけ描いて確認： python3 preview.py 05 [--mp4] [--frames 0,30]  → exports/v3/preview_B/"""
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core       # noqa: E402
import timeline_b as timeline   # noqa: E402  （cutB の timeline）

CG = {'01', '02', '03', '04', '05', '06', '07', 'T'}
key = sys.argv[1] if sys.argv[1] == 'T' else sys.argv[1].zfill(2)
beat = next(b for b in timeline.load() if b['key'] == key)
mod = importlib.import_module('cutB.beats.b%s' % key)
OUTD = os.path.join(core.OUT, 'preview_B')
want = set(int(v) for v in sys.argv[sys.argv.index('--frames') + 1].split(',')) if '--frames' in sys.argv else set()
pick = set(np.linspace(0, beat['n'] - 1, 12).round().astype(int))
enc = core.Encoder(os.path.join(OUTD, 'b%s.mp4' % key), beat['n']) if '--mp4' in sys.argv else None
shots = []
for i, f in enumerate(mod.render(beat, {})):
    if key in CG:
        f = core.cg_label(f)
    if enc:
        enc.write(f)
    if i in pick:
        shots.append(f)
    if i in want:
        core.save_png(f, os.path.join(OUTD, 'b%s_f%04d.png' % (key, i)))
if enc:
    enc.close()
core.contact(shots, os.path.join(OUTD, 'b%s.png' % key), cols=6, scale=0.25)
print('ok', key, beat['n'], 'frames')
