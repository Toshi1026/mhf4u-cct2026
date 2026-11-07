"""B02（3.3–5.1秒）「あれ、本当にやったら？」
01 の続き（同じ空・同じ押し込みの中心 DOT）。キラーンの光は消えていき、悪役の小さな点だけが残る。
01 の文字が消え、「どれくらいの力なら」→（ホ）「あそこまで飛ぶ？」（あそこまで を強調）。
「あそこまで」と同時に、点のまわりに小さな四隅のかぎ（ねらいの印・輪ではない）。押し込みは 1.035 → 1.10。
最後の 12 フレーム：DOT へのズームのぼかし＋濃紺へ 9 割まで暗くして、03（土佐湾の線画・濃紺）へやわらかく渡す。"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W, H
from beats import _mapui as U
from cutB.beats import _art as A, _sky as S, b01

OUT = 12
DARK = 0.90          # 最後のフレームで NAVY へ寄せる割合


def brackets(L, cx, cy, half, arm, col, width):
    for sx in (-1, 1):
        for sy in (-1, 1):
            x0, y0 = cx + sx * half, cy + sy * half
            L.draw.line(L.p([(x0 - sx * arm, y0), (x0, y0), (x0, y0 - sy * arm)]), fill=col, width=width,
                        joint='curve')


def render(beat, ctx):
    n = beat['n']
    bs = timeline.load()
    b1 = next(b for b in bs if b['key'] == '01')
    T1 = b01.timing(b1)
    off = beat['f0'] - b1['f0']                     # 01 の頭からの通し番号
    mo = beat['moras']
    assert mo[2][0] == 'ホ', mo
    f_ho = timeline.mora_frame(beat, 2)
    old1, old2 = b01.texts()
    q1 = U.Patch([dict(text='どれくらいの力なら', x=W / 2, y=470, size=48, weight='Medium', track=0.10)])
    q2 = U.Patch([dict(text='あそこまで飛ぶ？', x=W / 2, y=600, size=100, weight='Bold', hl='あそこまで')])
    f_q1 = 5
    for i in range(n):
        gi = off + i
        f = S.frame(gi)
        # 点（ごく弱くまたたく）と、消えていくキラーン
        tw = 0.85 + 0.15 * np.cos((gi - T1['glint']) * 0.35)
        f = A.put_dot(f, S.DOT[0], S.DOT[1], 2.3, alpha=tw)
        g = A.glint_env(gi - T1['glint'])
        if g > 0.01:
            f = A.put_glint(f, S.DOT[0], S.DOT[1], 46 * g, ang=10 * (1 - g), alpha=min(1.0, g * 1.4))
        k = core.lerp(*S.PUSH02, core.ease(i / (n - 1), 'smooth'))
        f = S.push(f, k)
        # ねらいの印（あそこ）
        if i >= f_ho:
            a = core.ease((i - f_ho + 1) / 8.0)
            half = 30 + 26 * (1 - a)
            f = A.put(f, lambda L, cx, cy: brackets(L, cx, cy, half, 11, L.col(core.FOAM, 0.85), L.w(2.4)),
                      S.DOT[0], S.DOT[1], half + 16, s=3, alpha=a, shadow=0.25)
        # 文字：01 の文字が消えて、問いの文字
        j = i - (n - OUT)
        tout = 1 - core.ease((j - 1) / 7.0) if j >= 2 else 1.0
        a_old = 1 - core.ease((i + 1) / 5.0)
        f = old1.put(f, a_old)
        f = old2.put(f, a_old)
        for tl, f0 in ((q1, f_q1), (q2, f_ho)):
            if i >= f0 - 1:
                a = core.ease((i - f0 + 2) / 6.0)
                f = tl.put(f, a * tout, dy=10 * (1 - a))
        # 最後：DOT へズームのぼかし＋暗く
        if j >= 0:
            q = (j + 1) / OUT
            f = core.zoom_blur(f, 0.14 * q ** 1.3, cx=S.DOT[0], cy=S.DOT[1], steps=16)
            d = DARK * (0.5 * q + 0.5 * core.ease(q, 'smooth'))
            f = f * (1 - d) + core.NAVY * d
        yield np.clip(f, 0, 1).astype(np.float32)
