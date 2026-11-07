"""案B 08（約3.1秒）「落ちた先は、仁淀川の、河口の前の海。」 実写。案A の b12 の見た目をそのまま使う。

 0–(ニ)：T（＝案A b11）の最後のフレームから、IMG_9662（5倍、波が崩れる）。案A b12 と同じ読み出し・同じ grade・
         同じ0.5倍スロー。最初の 3 フレームは同じ線を重ねて消していく（線が実写に「着地」する。b12 と画素単位で同じ）。
 (ニ)–：「ニ」でハードカットして IMG_9671（タワーから見た河口。川・白い砂州・外海）、0.5倍スロー。
テロップ（空の部分）：「仁淀川の」が「ニ」で、「河口の前の海」（河口＝ACC）が「カ」で出て、最後の 6 フレームで消える。
空は b12 と同じハーフ ND で落とす。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W
from beats import _real, b12
from cutB.beats import _fac as F

Y1, Y2, SZ = 390, 505, 88


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[6][0] == 'ニ' and mo[12][0] == 'カ', mo
    f_ni = round(mo[6][1] * core.FPS) - beat['f0']
    f_ka = round(mo[12][1] * core.FPS) - beat['f0']
    t1 = F.Text(core.text_layer([dict(text='仁淀川の', x=W / 2, y=Y1, size=SZ, weight='Bold', shadow=0.85)]))
    t2 = F.Text(core.text_layer([dict(text='河口の前の海', x=W / 2, y=Y2, size=SZ, weight='Bold', hl='河口',
                                      shadow=0.85)]))
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
            f = next(river) * shade
            out = core.fade(i, n, fin=0, fout=6)
            a1 = min(1.0, (i - f_ni + 1) / 5) * out
            a2 = (min(1.0, (i - f_ka + 1) / 5) if i >= f_ka else 0.0) * out
            f = t1.put(f, a1)
            f = t2.put(f, a2)
        yield np.clip(f, 0, 1).astype(np.float32)
