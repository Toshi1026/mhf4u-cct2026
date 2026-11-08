"""仮のビート（プレースホルダ）。本番の絵ができるまで、全体のパイプライン（timeline → preview/build → 音 → mp4）を通すため。
背景のグラデーション（案C のパレット）、ビート番号と内容、テロップ案（ACC は1語）、ナレーションのモーラが来たら下の帯が光る。"""
import numpy as np

import core
from core import W, H, SAFE

_BG = {}


def bg(top, bottom):
    k = (tuple(top), tuple(bottom))
    if k not in _BG:
        _BG[k] = core.vgrad(top, bottom)
    return _BG[k]


def render(beat, title, telop='', hl=None, top=core.DEEP, bottom=core.NAVY, note=''):
    n = beat['n']
    base = bg(top, bottom)
    cx, cy = W / 2, (SAFE[1] + SAFE[3]) / 2
    items = [dict(text='C%s  STUB' % beat['key'], x=cx, y=SAFE[1] + 90, size=44, weight='Bold', track=0.08),
             dict(text=title, x=cx, y=SAFE[1] + 160, size=34, weight='Medium')]
    if telop:
        items.append(dict(text=telop, x=cx, y=cy + 40, size=96 if len(telop) <= 8 else 64, weight='Bold', hl=hl, shadow=0.9))
    if note:
        items.append(dict(text=note, x=cx, y=SAFE[3] - 40, size=30))
    lay = core.text_layer(items)
    mf = [round(s * core.FPS) - beat['f0'] for _, s in beat['moras']]
    for i in range(n):
        f = base.copy()
        f = core.put_text(f, lay)
        # 進み具合の帯（セーフゾーンの下端）とモーラの点
        y0 = SAFE[3] + 30
        f[y0:y0 + 8, SAFE[0]:SAFE[0] + int((SAFE[2] - SAFE[0]) * (i + 1) / n)] = core.SEA
        if any(0 <= i - m < 3 for m in mf):
            f[y0 + 16:y0 + 40, int(cx) - 12:int(cx) + 12] = core.ACC
        yield f
