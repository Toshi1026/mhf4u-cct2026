"""ビートの 1 フレームの実時間（render() の各フレームの間隔。最初の 2 フレーム＝静的な層・numba の読み込みは除く）。
python3 cutB/lookdev/bench_beats.py 01 [a:b]   （a:b を渡すとそのフレームの範囲だけ数える。前のフレームも描く）"""
import importlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import core  # noqa: E402,F401
import timeline_b as TL  # noqa: E402

key = sys.argv[1]
rng = sys.argv[2] if len(sys.argv) > 2 else None
beat = next(b for b in TL.load() if b['key'] == key)
mod = importlib.import_module('cutB.beats.b%s' % key)
a, b = (int(v) for v in rng.split(':')) if rng else (2, beat['n'])
ts = []
t0 = time.perf_counter()
for i, f in enumerate(mod.render(beat, {})):
    t1 = time.perf_counter()
    if a <= i < b:
        ts.append((t1 - t0, i))
    t0 = t1
    if i + 1 >= b:
        break
v = np.array([t for t, _ in ts])
print('b%s frames %d..%d  mean %.2f s/frame  max %.2f (f%d)' % (key, a, b - 1, v.mean(), v.max(), ts[int(v.argmax())][1]))
