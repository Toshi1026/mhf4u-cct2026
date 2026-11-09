"""ビートの雛形。render(beat, ctx) は beat['n'] 枚の sRGB float フレーム（1920×1080×3... 実際は H=1920, W=1080）を yield する。
beat：timeline.load() の要素（key, f0, f1, n, vo=ナレーション開始秒, moras=[(モーラ, 絶対秒)]）。
timeline.mora_frame(beat, i) で i 番目のモーラが鳴るフレーム（ビート内）を得られる。"""
import numpy as np

import core
from core import W, H


def render(beat, ctx):
    n = beat['n']
    txt = core.text_layer([dict(text='見本', x=W / 2, y=800, size=96, weight='Bold', hl='本')])
    for i in range(n):
        f = core.vgrad(core.SEA, core.DEEP)
        yield core.put_text(f, txt, core.fade(i, n))
