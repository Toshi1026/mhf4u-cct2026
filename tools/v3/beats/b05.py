"""b05（14.2–16.9秒）：しかも波は、浅い所で崩れる。
真横の断面図。左の深い所から来た波が、右の浅瀬で立ち上がって巻き、白い泡になって崩れる。
サーファーは崩れる所でやわらかく消え（落ちない）、泡が引くと浜にボードを抱えて立っている（「あれ？」の1拍）。
最初の 8 フレームは b04（地球へのズームイン）からのズームのぼかしを 0 へ戻す。"""
import numpy as np

import core
import timeline
from core import W, H
from beats import _wave

ENTRY, ENTRY_DIM = 8, 0.72


def render(beat, ctx):
    n = beat['n']
    sc = _wave.XSScene(n)
    f_txt = timeline.mora_frame(beat, 6)            # 「ア」サイ
    txt = core.text_layer([dict(text='浅い所で 崩れる', x=W / 2, y=620, size=96, weight='Bold', hl='崩れる')])
    for i in range(n):
        f = sc.frame(i)
        sc.hold(f, i)                               # 泡が引くと、浜にボードを抱えて立っている
        if i >= f_txt:
            f = core.put_text(f, txt, min(1.0, (i - f_txt + 1) / 4))
        if i < 8:
            f = core.zoom_blur(f, 0.15 * (1 - i / 8) ** 2, cy=core.WL)
        if i < ENTRY:
            # b04 の最後（暗い地球）から明るい断面図へ：明るさの段差を 0.14 → 0.02 以下に抑え、8 フレームで戻す
            f = f * (ENTRY_DIM + (1 - ENTRY_DIM) * core.ease(i / ENTRY, 'inout'))
        yield np.clip(f, 0, 1).astype(np.float32)
