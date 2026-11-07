"""b13（約2.8秒）「気づけば、ここ。」 実写・おわり。

IMG_9636（2F の窓越しの海。固定）を 1.6 秒から等速（素材の2フレームに1枚）。水平を直し、空・水平線・白波・浜だけに
切り出す（駐車場・バイク・ナンバー・天井照明の映り込みは入らない。_real.WINDOW）。
空は上ほど少し深い青に（窓ガラス越しで白っぽい空を、CG の青と b12 の空にそろえる。白い文字のコントラスト 3:1 以上）。
「気づけば、」が「キ」で、「ここ。」が「コ」で同じ行に足される（ここ＝強調色）。話し終わりの約0.3秒後に消え、
そのあと施設名だけを小さく出して、終わりまで置く。ロゴは出さない。静かに終わる。
"""
import numpy as np

import core
import timeline
from core import W, H
from beats import _real

Y_LINE, SZ = 760, 96
Y_NAME, Y_ADDR = 1160, 1214


def _sky():
    """空を深くする色の掛け算（H,1,3）。上 0..1230 は (0.64, 0.73, 0.92)。水平線（実測 y=1413、傾き 0.02°）に向かって
    半分だけ戻し（水平線の近くの白っぽいもやを残す）、水平線の境目で 1 に（海はそのまま）。"""
    y = np.arange(H, dtype=np.float32)
    t = np.clip((y - 1230) / 175.0, 0, 1)
    t = 0.5 * t * t * (3 - 2 * t)
    t = np.where(y >= 1385, t + (1 - t) * np.clip((y - 1385) / 28.0, 0, 1) ** 2, t)
    k = np.array([0.64, 0.73, 0.92], np.float32)
    return (k[None, :] * (1 - t[:, None]) + t[:, None])[:, None, :]


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[0][0] == 'キ' and mo[4][0] == 'コ', mo
    f_ki = timeline.mora_frame(beat, 0)
    f_ko = timeline.mora_frame(beat, 4)
    f_end = round((mo[-1][1] + 0.21) * core.FPS) - beat['f0']        # 話し終わり（最後の「コ」の終わり）
    f_out = f_end + 9                                                # その約0.3秒後に消し始める
    f_name = f_out + 5
    a_txt, b_txt = '気づけば、', 'ここ。'
    tr = 0.04
    total = core.text_width(a_txt + b_txt, SZ, 'Bold', tr)
    x0 = W / 2 - total / 2
    xb = x0 + core.text_width(a_txt, SZ, 'Bold', tr) + tr * SZ
    tA = core.text_layer([dict(text=a_txt, x=x0, y=Y_LINE, size=SZ, weight='Bold', anchor='ls', shadow=0.9)])
    tB = core.text_layer([dict(text=b_txt, x=xb, y=Y_LINE, size=SZ, weight='Bold', anchor='ls', hl='ここ', shadow=0.9)])
    tN = core.text_layer([dict(text='Seaside MAZE（南風・まぜ）', x=W / 2, y=Y_NAME, size=40, weight='Medium', alpha=0.9,
                               track=0.03, shadow=0.9),
                          dict(text='高知県土佐市新居', x=W / 2, y=Y_ADDR, size=34, weight='Regular', alpha=0.9,
                               track=0.08, shadow=0.9)])
    sky = _sky()
    for i, f in enumerate(_real.clip(_real.WINDOW, n, step=2)):
        f = f * sky
        out = 1 - core.ease((i - f_out) / 5.0) if i >= f_out else 1.0
        if i >= f_ki and out > 0:
            f = core.put_text(f, tA, min(1.0, (i - f_ki + 1) / 5) * out)
        if i >= f_ko and out > 0:
            f = core.put_text(f, tB, min(1.0, (i - f_ko + 1) / 4) * out)
        if i >= f_name:
            f = core.put_text(f, tN, core.ease((i - f_name + 1) / 6.0))
        yield np.clip(f, 0, 1).astype(np.float32)
