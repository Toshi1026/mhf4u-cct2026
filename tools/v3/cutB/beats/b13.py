"""案B 13（ナレーションなし・約1.8秒）エンドカード：穏やかな水平線に施設名。

12 の IMG_9660 のそのまま続き（素材のフレームが途切れない）。空は 12 と同じ ND グラデーションで落としたまま、
全体をほんの少しずつ暗くして（最大 12%）、文字の帯の後ろの空もやわらかく落として文字を立たせる。セーフゾーンの中央に
「Seaside MAZE（南風・まぜ）」（44px Medium）／「高知県土佐市新居・国道56号沿い」（34px）をゆっくり出して、最後まで置く。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W
from beats import _real
from cutB.beats import _fac as F
from cutB.beats import b12
import timeline_b

CY = (core.SAFE[1] + core.SAFE[3]) / 2          # セーフゾーンの縦の中央（≈758）
_y = np.arange(core.H, dtype=np.float32)
SPOT = (0.22 * np.exp(-((_y - (CY + 20)) / 160.0) ** 2))[:, None, None]   # 文字の帯の後ろの空だけ少し落とす


def render(beat, ctx):
    n = beat['n']
    prev = next(b for b in timeline_b.load() if b['key'] == '12')
    spec = dict(F.DECK, t0=F.DECK['t0'] + prev['n'] / 60.0)       # 12 の最後の次の素材フレームから（0.5倍スロー）
    src = F.fast_clip(spec, n)
    shade = b12.sky_shade()
    tl = F.Text(core.text_layer([dict(text='Seaside MAZE（南風・まぜ）', x=W / 2, y=CY - 8, size=44, weight='Medium', track=0.03,
                               shadow=0.95),
                          dict(text='高知県土佐市新居・国道56号沿い', x=W / 2, y=CY + 52, size=34, weight='Regular',
                               track=0.08, shadow=0.95)]))
    for i in range(n):
        f = next(src) * shade
        d = core.ease(min(1.0, i / 24.0), 'smooth')
        f = f * (1 - 0.12 * d) * (1 - SPOT * d)
        a = core.ease(float(np.clip((i - 4) / 16.0, 0, 1)), 'smooth')
        f = tl.put(f, a)
        yield np.clip(f, 0, 1).astype(np.float32)
