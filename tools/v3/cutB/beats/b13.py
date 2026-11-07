"""案B 13（ナレーションなし・約1.8秒）エンドカード：穏やかな水平線に施設名。

12 の IMG_9660 のそのまま続き（素材のフレームが途切れない）。空は 12 と同じ ND グラデーションで落としたまま、
全体をほんの少しずつ暗くして（最大 12%）、文字の帯の後ろの空もやわらかく落として文字を立たせる。セーフゾーンの中央に
「Seaside MAZE（南風・まぜ）」（64px Bold）／「高知県土佐市新居・国道56号沿い」（40px Medium）をゆっくり出して、最後まで置く。
2026-11-07：文字を大きく（旧 44px Medium／34px Regular）。68px 以上だと施設名がセーフゾーンの幅（950px）からはみ出すため 64px。
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
NAME_PX, ADDR_PX = 64, 40                       # 施設名 64px Bold（幅 約904px＝セーフゾーンの幅 950px に収まる上限）・住所 40px Medium
_y = np.arange(core.H, dtype=np.float32)
SPOT = (0.22 * np.exp(-((_y - (CY + 20)) / 160.0) ** 2))[:, None, None]   # 文字の帯の後ろの空だけ少し落とす


def render(beat, ctx):
    n = beat['n']
    prev = next(b for b in timeline_b.load() if b['key'] == '12')
    spec = dict(F.DECK, t0=F.DECK['t0'] + prev['n'] / 60.0)       # 12 の最後の次の素材フレームから（0.5倍スロー）
    src = F.fast_clip(spec, n)
    shade = b12.sky_shade()
    # 施設名は全角の閉じかっこの右の余白のぶん +12px（見た目の中央へ）
    tl = F.Text(core.text_layer([dict(text='Seaside MAZE（南風・まぜ）', x=W / 2 + 12, y=CY - 18, size=NAME_PX, weight='Bold', track=0.02,
                                      shadow=0.95),
                                 dict(text='高知県土佐市新居・国道56号沿い', x=W / 2, y=CY + 62, size=ADDR_PX, weight='Medium',
                                      track=0.06, shadow=0.95)]))
    for i in range(n):
        f = next(src) * shade
        d = core.ease(min(1.0, i / 24.0), 'smooth')
        f = f * (1 - 0.12 * d) * (1 - SPOT * d)
        a = core.ease(float(np.clip((i - 4) / 16.0, 0, 1)), 'smooth')
        f = tl.put(f, a)
        yield np.clip(f, 0, 1).astype(np.float32)
