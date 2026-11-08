"""案C 11（約3.2秒）「落ちたのは、太平洋と、仁淀川が、出会う海。」 実写。案B の b08 と同じ作り（10 の着水から実写の海へ）。

 0–(ニ)：10（＝案B の T、最後の 18 フレームは案A の b11）の最後のフレームから、IMG_9662（5倍、波が崩れる）。
         案A b12・案B b08 と同じ読み出し・同じ grade・同じ 0.5倍スロー。最初の 3 フレームは同じ線を重ねて消していく
         （線が実写に「着地」する。b08 と画素単位で同じ）。「太平洋と」はこの外海の波の上。
 (ニ)–：「仁淀川」の「ニ」でハードカットして IMG_9671（タワーから見た河口。川・白い砂州・外海）、0.5倍スロー。
         河口（x 0.55, y 0.62）を中心に 1.00→1.035 のデジタルの寄りと 0.25° のわずかな回転（in_out_sine）。
テロップ（空の帯。案B の階の見出しと同じ 2 行の位置・大きさ）：
  「仁淀川の河口」（96px Bold、河口＝ACC）：「ニ」の 2 フレーム前から 1 文字ずつ上がる。「ガ（仁淀川が）」で河口が FOAM→ACC（出そろってから約 0.4 秒後。抜けるまで約 0.8 秒 ACC で読める）。
  「高知県土佐市」（46px Medium、小さく）：「ド（仁淀川）」の 1 フレーム前から行ごと下のマスクから上がる。
  最後の 6 フレームで全体が上へぼけながら抜けて、12 の空撮へ。空は b08 と同じハーフ ND で落とす。
"""
import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W
from beats import _real, b12
from cutB.beats import _fac as F
from cutB.beats import _motion as Mo
from cutC.beats import _maze as Z

TAIL, TAIL_DIM = 6, 0.10
PUSH, ROLL, PIVOT = 1.035, 0.25, (0.55 * W, 0.62 * core.H)     # 河口のショットの寄り・回転（度）・中心（b08 と同じ）


def render(beat, ctx):
    n = beat['n']
    f_ni = Z.mf(beat, 12, 'ニ')
    f_ga = Z.mf(beat, 15, 'ガ')
    f_do = Z.mf(beat, 14, 'ド')
    tb = Mo.KText([dict(text='仁淀川の河口', x=W / 2, y=F.Y_BIG, size=F.SZ_BIG, weight='Bold', hl='河口', track=0.03,
                        shadow=0.9)])
    ts = Mo.KText([dict(text='高知県土佐市', x=W / 2, y=F.Y_SUB, size=F.SZ_SUB, weight='Medium', track=0.08, shadow=0.9)])
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
            if i >= n - TAIL:                               # 12 の空撮（空が少し暗い）へ、カットの段差を 0.1 未満に
                f = f * (1 - TAIL_DIM * Mo.ease((i - (n - TAIL) + 1) / float(TAIL), 'sine_in'))
            f = tb.reveal(f, i, f_ni - 2, 'char_rise', stagger=1.2, dur=10, accent=f_ga, exit=n - 6, exit_dur=6)
            f = ts.reveal(f, i, f_do - 1, 'line_mask', exit=n - 6, exit_dur=6)
        yield np.clip(f, 0, 1).astype(np.float32)
