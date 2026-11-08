"""B02（3.3–5.1秒）「あれ、本当にやったら？」
01 の続き（同じ空・同じ DOT 中心の押し込み）。キラーンの光は消えていき、悪役の小さな点だけが残る。
01 の文字が消え、「どれくらいの力なら」→（ホ）「あそこまで飛ぶ？」（あそこまで を強調）。
「あそこまで」と同時に、点のまわりに小さな四隅のかぎ（ねらいの印・輪ではない）。

2026-11 動きの改善（director's motion spec §1-02）：
- 押し込みは 01 から続く 1 本の曲線（_sky.push_k、sine_in で速くなりながら 1.13 へ）。
- ねらいのかぎは back_out で飛び込む（半分の大きさ 70→30、左上・右上・右下・左下の順に 1 フレームずつ、各 7 フレーム）、
  そのあと ±1.5px・1.2Hz でゆっくり呼吸。
- 文字：1 行目は行ごとのマスク、「あそこまで飛ぶ？」は 1 文字ずつ、「？」は back_out で回りながら（10°→0）ポンと出る。
  01 の文字は上へぼけて抜ける（KText.exit）。最後の問いの文字も f40 から抜ける。
- 03 へのズームスルー（f46–55）：DOT を中心に expo_in で 1.13→1.9、ズームのぼかし 0→0.22、NAVY へ暗く。03 が同じ点から
  1.6→1.0 で受ける（zoom_through の 'in'）。
2026-11-08 見た目：舞台は 01 と同じ _beach。仕上げ（finish_ad）は文字の前。最後の濃紺へは DARK 0.80 のまっすぐな曲線（03 の最初と明るさをそろえる）。
"""
import math

import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W, H
from cutB.beats import _art as A, _sky as S, _motion as M, b01
from cutB.beats import _beach as BE, _render as R

ZT0 = 46                 # ズームスルーの始まり（ローカル）
ZT_END = 1.9             # 最後のフレームの倍率（DOT 中心）
ZT_BLUR = 0.22
DARK = 0.80              # 最後のフレームで NAVY へ寄せる割合（2026-11-08：03 の新しい最初のフレーム（平均 約0.20）と明るさをそろえる）
TXT_OUT = 45             # 問いの文字が抜けはじめるフレーム


def brackets(L, cx, cy, halfs, arm, col, width):
    """四隅のかぎ。halfs：各角（左上・右上・右下・左下）の (半分の大きさ, 不透明度)。"""
    order = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for (sx, sy), (half, a) in zip(order, halfs):
        if a <= 0.01:
            continue
        x0, y0 = cx + sx * half, cy + sy * half
        L.draw.line(L.p([(x0 - sx * arm, y0), (x0, y0), (x0, y0 - sy * arm)]),
                    fill=(col[0], col[1], col[2], int(col[3] * a)), width=width, joint='curve')


_TX = {}


def texts():
    if 't' not in _TX:
        q1 = M.KText([dict(text='どれくらいの力なら', x=W / 2, y=470, size=48, weight='Medium', track=0.10)])
        q2 = M.KText([dict(text='あそこまで飛ぶ？', x=W / 2, y=600, size=100, weight='Bold', hl='あそこまで')])
        _TX['t'] = (q1, q2)
    return _TX['t']


def put_q2(f, i, f0, exit_f):
    """「あそこまで飛ぶ」は 1 文字ずつ、最後の「？」は回りながらポン。"""
    q1, q2 = texts()
    qmark = q2.glyphs[-1]
    body = q2.glyphs[:-1]
    # 「？」以外：KText の char_rise（glyphs を一時的に絞る）
    full = q2.glyphs
    q2.glyphs = body
    f = q2.reveal(f, i, f0, 'char_rise', stagger=0.9, dur=10, dy_em=0.26, blur=6, accent=f0 + 7, exit=exit_f)
    q2.glyphs = full
    # 「？」
    fq = f0 + 0.9 * len(body) + 1
    if i < fq:
        return f
    k = i - fq
    s = 0.55 + 0.45 * M.ease((k + 1) / 7.0, 'back_out')
    rot = 10.0 * (1 - M.ease((k + 1) / 8.0, 'cubic_out'))
    o = M.ease((k + 1) / 3.0, 'sine_inout')
    ex_dy, ex_b = 0.0, 0.0
    if i >= exit_f:
        e = M.ease((i - exit_f + 1) / 6.0, 'cubic_in')
        o *= 1 - e
        ex_dy, ex_b = -0.15 * 100 * e, 4 * e
    if o <= 0.002:
        return f
    g = qmark
    cx, cy = g['gx'] - g['x0'], g['gy'] - g['y0']
    Mr = cv2.getRotationMatrix2D((cx, cy), -rot, s)
    arrs = []
    for a in (g['prem'], g['a'], g['sh']):
        if ex_b > 0.25:
            a = cv2.GaussianBlur(a, (0, 0), ex_b)
        arrs.append(cv2.warpAffine(a, Mr, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR))
    y0 = g['y0'] + int(round(ex_dy))
    M._composite_sh(f, arrs[2], g['x0'], y0, o)
    M._composite_col(f, arrs[0], arrs[1], g['x0'], y0, o)
    return f


def render(beat, ctx):
    n = beat['n']
    bs = timeline.load()
    b1 = next(b for b in bs if b['key'] == '01')
    T1 = b01.timing(b1)
    off = beat['f0'] - b1['f0']                     # 01 の頭からの通し番号
    gi_end = b01.push_end()
    mo = beat['moras']
    assert mo[2][0] == 'ホ', mo
    f_ho = timeline.mora_frame(beat, 2)
    old1, old2 = b01.texts()
    q1, q2 = texts()
    f_q1 = 5
    zt0 = min(ZT0, n - 10)
    for i in range(n):
        gi = off + i
        f = BE.world(gi)
        # 点（ごく弱くまたたく）と、消えていくキラーン
        tw = 0.85 + 0.15 * np.cos((gi - T1['glint']) * 0.35)
        f = A.put_dot(f, S.DOT[0], S.DOT[1], 2.3, alpha=tw)
        f = A.put_glint_fx(f, S.DOT[0], S.DOT[1], gi - T1['glint'])
        f = S.push(f, S.push_k(gi, gi_end))
        # ねらいの印（あそこ）：角ごとに 1 フレームずつずれて飛び込み、そのあと呼吸
        if i >= f_ho - 1:
            br = 1.5 * math.sin(2 * math.pi * 1.2 * (i - f_ho) / core.FPS)
            halfs = []
            for c in range(4):
                p = M.ease((i - (f_ho - 1) - c + 1) / 7.0, 'back_out')
                halfs.append((70 - 40 * p + br, float(np.clip((i - (f_ho - 1) - c + 1) / 2.0, 0, 1))))
            f = A.put(f, lambda L, cx, cy: brackets(L, cx, cy, halfs, 11, L.col(core.FOAM, 0.85), L.w(2.4)),
                      S.DOT[0], S.DOT[1], 100, s=3, alpha=1.0, shadow=0.25)
        k_ = S.push_k(gi, gi_end)
        f = BE.screen_post(f)
        f = R.finish_ad(f, gi, grain=0.005, vignette=0.10, ca_px=0.25)
        # 最後：DOT へのズームスルー（03 が同じ点で受ける）。文字はズームのぼかしの後に重ねる（文字はぶらさない）
        j = i - zt0
        if j >= 0:
            q = (j + 1) / float(n - zt0)
            k0 = S.push_k(gi, gi_end)
            f = M.zoom_through(f, q, S.DOT, 'out', s_out=(1.0, ZT_END / k0), blur=ZT_BLUR, steps=14)
        # 文字：01 の文字が抜けて、問いの文字
        f = old1.exit(f, i, -1, dur=6)
        f = old2.exit(f, i, -1, dur=6)
        f = q1.reveal(f, i, f_q1 - 1, 'line_mask', exit=TXT_OUT)
        f = put_q2(f, i, f_ho - 2, TXT_OUT + 1)
        # NAVY へ暗く（文字ごと）
        if j >= 0:
            d = DARK * q                       # 2026-11-08：まっすぐ（1 フレームの明るさの差 ≤0.055、03 の最初と ≤0.01）
            f = f * (1 - d) + core.NAVY * d
        yield np.clip(f, 0, 1).astype(np.float32)
