"""b12（約3.4秒）「波が立つ浅瀬は、仁淀川が運んだ砂、かも。」 実写。

 0–(ニ)：b11 の線画からハードカットで IMG_9662（5倍、波が崩れる）。0.5倍スロー（素材の1フレーム＝出力1フレーム）。
         1フレーム目は b11 の最後のフレームと同じ画角・同じ水平線（b11 はこのフレームから線を起こしている）。
         最初の 3 フレームだけ、そのフレームの輪郭から起こした同じ線を重ねて消していく（線が実写に「着地」する）。
 (ニ)–：ナレーションの「ニ」でハードカットして IMG_9671（タワーから見た河口。川・白い砂州・外海）、0.5倍スロー。
         下の消波ブロックと柵は切り出しで外す（_real.RIVER）。
テロップ：「仁淀川が運んだ」が「ニ」で、「砂、かも。」が「ス」で出て、最後の 6 フレームで消える。
空を上からやわらかく暗くして（ハーフ ND のように）、白い文字のコントラストを 3:1 以上にする。
"""
import cv2
import numpy as np

import core
import timeline
from core import W, H
from beats import _geo as G, _real

Y1, Y2, SZ = 390, 505, 88
LAND = 3                                     # 線を重ねて消すフレーム数


def _shade():
    """空だけを上からやわらかく暗くする縦のグラデーション（ハーフ ND のように。約0.45段）。
    文字（y 390／505）の後ろの空で、白い文字とのコントラストが 3:1 以上になるように（実測 3.1〜4.0）。"""
    y = np.arange(H, dtype=np.float32)
    t = np.clip((y - 560) / 300.0, 0, 1)
    return (1 - 0.14 * (1 - t * t * (3 - 2 * t)))[:, None, None]


def _overlay(f, ln, a):
    L = G.Lines()
    for P in ln['main']:
        L.poly(P, 2.0, 255 * 0.92 * a)
    for P in ln['swell']:
        L.poly(P, 1.4, 255 * 0.42 * a)
    L.poly(np.array([[-10, ln['horizon']], [W + 10, ln['horizon']]]), 2.0, 255 * 0.92 * a)
    m = L.done()[..., None]
    return f * (1 - m) + core.FOAM * m


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[9][0] == 'ニ' and mo[19][0] == 'ス', mo
    f_ni = timeline.mora_frame(beat, 9)
    f_su = timeline.mora_frame(beat, 19)
    t1 = core.text_layer([dict(text='仁淀川が運んだ', x=W / 2, y=Y1, size=SZ, weight='Bold', shadow=0.85)])
    t2 = core.text_layer([dict(text='砂、かも。', x=W / 2, y=Y2, size=SZ, weight='Bold', hl='砂', shadow=0.85)])
    shade = _shade()
    wave = _real.clip(_real.WAVE, f_ni)
    river = _real.clip(_real.RIVER, n - f_ni)
    first = _real.lines_of(_real.wave_first_frame())
    for i in range(n):
        if i < f_ni:
            f = next(wave)
            if i < LAND:
                ln = first if i == 0 else _real.lines_of(f)
                f = _overlay(f, ln, (1 - i / LAND) ** 1.5)
        else:
            f = next(river)
            out = core.fade(i, n, fin=0, fout=6)
            a1 = min(1.0, (i - f_ni + 1) / 5) * out
            a2 = (min(1.0, (i - f_su + 1) / 5) if i >= f_su else 0.0) * out
            f = f * shade                          # 空を少し暗く（カットの頭からずっと同じ。明るさが途中で変わらないように）
            f = core.put_text(f, t1, a1)
            if a2 > 0:
                f = core.put_text(f, t2, a2)
        yield np.clip(f, 0, 1).astype(np.float32)
