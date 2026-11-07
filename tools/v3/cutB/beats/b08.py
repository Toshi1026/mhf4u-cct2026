"""案B 08（約3.1秒）「落ちた先は、仁淀川の、河口の前の海。」 実写。案A の b12 の見た目をそのまま使う。

 0–(ニ)：T（＝案A b11）の最後のフレームから、IMG_9662（5倍、波が崩れる）。案A b12 と同じ読み出し・同じ grade・
         同じ0.5倍スロー。最初の 3 フレームは同じ線を重ねて消していく（線が実写に「着地」する。b12 と画素単位で同じ）。
 (ニ)–：「ニ」でハードカットして IMG_9671（タワーから見た河口。川・白い砂州・外海）、0.5倍スロー。
テロップ（空の部分）：「仁淀川の」が「ニ」で、「河口の前の海」（河口＝ACC）が「カ」で出て、最後の 6 フレームで消える。
空は b12 と同じハーフ ND で落とす。
2026-11-07 動きの改訂：
 - 河口のショット（ニ〜最後）：素材はほぼ固定なので、河口（x 0.55, y 0.62）を中心に 1.00→1.035 のデジタルの寄りと
   0.25° のわずかな回転（in_out_sine）。波のショット（T からの画素の一致）はそのまま。
 - 文字：「仁淀川の」は「ニ」の 1 フレーム前から行ごと下のマスクから上がる。「河口の前の海」は「カ」の 2 フレーム前から
   1 文字ずつ上がり、「河口」が FOAM→ACC の強調。最後の 6 フレームで全体が上へぼけながら抜ける。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W
from beats import _real, b12
from cutB.beats import _fac as F
from cutB.beats import _motion as Mo
import cv2

Y1, Y2, SZ = 390, 505, 88
PUSH, ROLL, PIVOT = 1.035, 0.25, (0.55 * W, 0.62 * core.H)     # 河口のショットの寄り・回転（度）・中心


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[6][0] == 'ニ' and mo[12][0] == 'カ', mo
    f_ni = round(mo[6][1] * core.FPS) - beat['f0']
    f_ka = round(mo[12][1] * core.FPS) - beat['f0']
    t1 = Mo.KText([dict(text='仁淀川の', x=W / 2, y=Y1, size=SZ, weight='Bold', shadow=0.85)])
    t2 = Mo.KText([dict(text='河口の前の海', x=W / 2, y=Y2, size=SZ, weight='Bold', hl='河口', shadow=0.85)])
    shade = b12._shade()
    wave = _real.clip(_real.WAVE, f_ni)
    river = F.fast_clip(_real.RIVER, n - f_ni)        # カットの後なので速い読み出しでよい（画は同じ）
    first = _real.lines_of(_real.wave_first_frame())
    for i in range(n):
        if i < f_ni:
            f = next(wave)
            if i < b12.LAND:
                ln = first if i == 0 else _real.lines_of(f)
                f = b12._overlay(f, ln, (1 - i / b12.LAND) ** 1.5)
        else:
            e = Mo.ease((i - f_ni) / float(n - 1 - f_ni), 'sine_inout')
            Mt = cv2.getRotationMatrix2D(PIVOT, -ROLL * e, 1 + (PUSH - 1) * e)
            f = cv2.warpAffine(next(river), Mt, (W, core.H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
            f = f * shade
            f = t1.reveal(f, i, f_ni - 1, 'line_mask', exit=n - 6, exit_dur=6)
            f = t2.reveal(f, i, f_ka - 2, 'char_rise', stagger=1.2, dur=10, accent=f_ka, exit=n - 6, exit_dur=6)
        yield np.clip(f, 0, 1).astype(np.float32)
