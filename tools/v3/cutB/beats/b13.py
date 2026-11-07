"""案B 13（ナレーションなし・約1.8秒）エンドカード：穏やかな水平線に施設名。

12 の IMG_9660 のそのまま続き（素材のフレームが途切れない）。空は 12 と同じ ND グラデーションで落としたまま、
全体をほんの少しずつ暗くして（最大 12%）、文字の帯の後ろの空もやわらかく落として文字を立たせる。セーフゾーンの中央に
「Seaside MAZE（南風・まぜ）」（64px Bold）／「高知県土佐市新居・国道56号沿い」（40px Medium）をゆっくり出して、最後まで置く。
2026-11-07：文字を大きく（旧 44px Medium／34px Regular）。68px 以上だと施設名がセーフゾーンの幅（950px）からはみ出すため 64px。
2026-11-07 動きの改訂：
 - カメラは 12 からの 1 本の曲線の続き（b12.cam、通しのフレーム＝12 の長さ＋i）。カットで動きが始まり直さない。
 - 施設名：1 文字ずつ（18 文字、1.1 フレームずつ遅れて、1 文字 10 フレーム）、ぼかし 5→0、下から 16px、
   字間 0.07em→0.02em（quart_out。仕様の 0.10em だと出始めの左端がセーフゾーンの外に出るため 0.07em）。
 - 住所：施設名の 10 フレーム後から、左から右へやわらかいマスク（ぼかし幅 60px）。
 - 30 フレームまでにすべて止まり、0.8 秒以上そのまま置く。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W
from beats import _real
from cutB.beats import _fac as F
from cutB.beats import _motion as Mo
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
    total = b12.cam_total()
    # 施設名は全角の閉じかっこの右の余白のぶん +12px（見た目の中央へ）
    kn = Mo.KText([dict(text='Seaside MAZE（南風・まぜ）', x=W / 2 + 12, y=CY - 18, size=NAME_PX, weight='Bold', track=0.02,
                        shadow=0.95)])
    ka = Mo.KText([dict(text='高知県土佐市新居・国道56号沿い', x=W / 2, y=CY + 62, size=ADDR_PX, weight='Medium', track=0.06,
                        shadow=0.95)])
    for i in range(n):
        f = b12.cam(next(src), prev['n'] + i, total) * shade
        d = core.ease(min(1.0, i / 24.0), 'smooth')
        f = f * (1 - 0.12 * d) * (1 - SPOT * d)
        f = kn.reveal(f, i, 0, 'char_rise', stagger=1.1, dur=10, dy_em=16.0 / NAME_PX, blur=5.0, track_from=0.07)
        f = ka.reveal(f, i, 10, 'wipe', dur=12, feather=60.0)
        yield np.clip(f, 0, 1).astype(np.float32)
