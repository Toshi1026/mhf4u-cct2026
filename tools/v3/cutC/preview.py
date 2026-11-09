"""案C の1ビートだけ描いて確認： python3 cutC/preview.py 05 [--mp4] [--frames 0,30]  → exports/v3/preview_C/
（tools/v3 から実行。プレビューは上書き。ディスクが少ないので --mp4 は必要な時だけ）

CG ラベル「CGシミュレーション」：03〜10（現実版の CG）に付ける。
  01・02 は付けない：アニメ版（平面的な絵柄）で、誰が見ても現実の映像ではないうえ、フックの大テロップ
  「アニメだと」「現実だと」の画面をすっきり保つため。「現実だと」の CG が始まる 03 から付ける。
  10 は最後に実写の海へ切り替わるので、最後の 8 フレームでラベルを消す（案B の T と同じ）。
"""
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cutB.beats import _useA  # noqa: E402,F401  最上位の `beats` を案A（tools/v3/beats）に固定（cutB の _motion などが使う）
import core       # noqa: E402
import timeline_c as timeline   # noqa: E402

CG = set()   # 2026-11-08 クライアントの指示で「CGシミュレーション」の小さなラベルは出さない（以前：03〜10）


def label(key, f, k, n):
    if key not in CG:
        return f
    a = 0.7 * (float(np.clip((n - 1 - k) / 8, 0, 1)) if key == '10' else 1.0)
    return core.cg_label(f, a)


if __name__ == '__main__':
    key = sys.argv[1].zfill(2)
    beat = next(b for b in timeline.load() if b['key'] == key)
    mod = importlib.import_module('cutC.beats.c%s' % key)
    OUTD = os.path.join(core.OUT, 'preview_C')
    want = set(int(v) for v in sys.argv[sys.argv.index('--frames') + 1].split(',')) if '--frames' in sys.argv else set()
    pick = set(np.linspace(0, beat['n'] - 1, 12).round().astype(int))
    enc = core.Encoder(os.path.join(OUTD, 'c%s.mp4' % key), beat['n']) if '--mp4' in sys.argv else None
    shots = []
    for i, f in enumerate(mod.render(beat, {})):
        f = label(key, f, i, beat['n'])
        if enc:
            enc.write(f)
        if i in pick:
            shots.append(f)
        if i in want:
            core.save_png(f, os.path.join(OUTD, 'c%s_f%04d.png' % (key, i)))
    if enc:
        enc.close()
    core.contact(shots, os.path.join(OUTD, 'c%s.png' % key), cols=6, scale=0.25)
    print('ok', key, beat['n'], 'frames')
