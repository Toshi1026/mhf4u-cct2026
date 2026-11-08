"""案C 04（約 1.6 秒）「約1,400トン。」  テロップ 約**1,400**トン（1,400＝強調）。

03 の最後の引き（ショット C）の続き：えぐれたクレーターの中にまだ砂煙がたまってゆっくり上がり、砂煙の壁は遠くへ広がる。
カメラは 03 から 05 まで 1 本のゆっくりした押し込み（_cfx.zoom_at）。03 の最後から続く砂煙のドームがゆっくり晴れて、
「ト」（1,400 で止まる）のころに深いクレーター（層の見える内壁・盛り上がった縁・岩・くすぶる煙）が現れる（_cfx）。
落ちてくる大きなかたまりのたびに画面がごく小さく揺れる。
数字はオドメーター（_motion.Odometer、やわらかい列）：「ヤ」の 1 フレーム前から回りはじめ（「約0トン」は見せない）、
「ト」で 1,400 に止まる（落ち＋1.10→1.00）。止まったときに画面がごく小さく揺れる（文字は揺らさない）。
文字の位置は 02・03 の「現実だと」と同じベースライン（y 600）。文字の後ろだけ空をごく弱く暗くして読みやすく。
最後の 5 フレームで数字は上へぼけて抜ける（05 の「ゾウ250頭分」が同じ所に出る）。
"""
import math

import cv2
import numpy as np

import core
from core import W, H
import timeline_c as timeline
from cutB.beats import _render as R, _motion as M
from cutC.beats import _cfx as CF

TEL_Y = 600
_C = {}


def timing(beat):
    mo = beat['moras']
    assert mo[0][0] == 'ヤ' and mo[8][0] == 'ト', mo
    mf = lambda k: timeline.mora_frame(beat, k)
    return dict(ya=mf(0), to=mf(8), out=beat['n'] - 5)


def odo():
    if 'o' not in _C:
        _C['o'] = M.Odometer('1,400', W / 2, TEL_Y, digit_size=150, side_size=96, prefix='約', suffix='トン',
                             lock_scale=1.10, lock_drop=6.0, lock_dur=8, hide_zero=True, feather_px=10)
    return _C['o']


def scrim():
    """文字の後ろの、ごく弱い暗いぼかし（縦 y 380–700、中央が濃い）。"""
    if 's' not in _C:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        k = np.exp(-((yy - 545) / 150.0) ** 2) * np.exp(-((xx - W / 2) / 520.0) ** 4)
        _C['s'] = (1 - 0.16 * k)[..., None].astype(np.float32)
    return _C['s']


def value(t, T):
    q = (t - (T['ya'] - 1)) / float(T['to'] - (T['ya'] - 1))
    return 1400.0 * M.ease(float(np.clip(q, 0, 1)), 'expo_out')


def put_text(f, i, T, beat):
    o = odo()
    if i < T['ya'] - 1:
        return f
    v = value(i, T)
    a = float(np.clip((v - 8) / 30.0, 0, 1)) if i < T['to'] else 1.0      # 「約0トン」「約 トン」は見せない
    dv = value(i + 0.5, T) - value(i - 0.5, T)
    if i < T['out']:
        return o.put(f, v, a, i=i, f_lock=T['to'], dv=dv)
    # 抜け：上へ 30px、ぼかし 0→6、不透明度 1→0（黒と白の上に描いて色とアルファを取り出す）
    q = M.ease((i - T['out'] + 1) / 5.0, 'cubic_in')
    outs = []
    for base in (np.zeros_like(f), np.ones_like(f)):
        outs.append(o.put(base, v, 1.0, i=i, f_lock=T['to'], dv=0.0, dy=-30 * q))
    ta = np.clip(1 - (outs[1] - outs[0]), 0, 1)
    t0 = outs[0]
    if q > 0.05:
        ta = cv2.GaussianBlur(ta, (0, 0), 6 * q)
        t0 = cv2.GaussianBlur(t0, (0, 0), 6 * q)
    ta, t0 = ta * (1 - q), t0 * (1 - q)
    return f * (1 - ta) + t0


def render(beat, ctx):
    for i in range(beat['n']):
        yield one(beat, i)


def one(beat, i):
    T = timing(beat)
    f, cam, Mm, g = CF.frame_C(beat, i)
    sh = M.shake(i, T['to'], amp=4.0, tau=2.0, rot_deg=0.0, seed=8)
    e = CF.thump(g) * 5.0                               # 大きなかたまりが落ちるたびの小さな揺れ
    sh = (sh[0] + e * M.vnoise(g * 0.7, 81), sh[1] + e * M.vnoise(g * 0.7 + 5, 82))
    if abs(sh[0]) + abs(sh[1]) > 0.05:
        f = cv2.warpAffine(f, np.float32([[1, 0, sh[0]], [0, 1, sh[1]]]), (W, H), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
    f = R.finish_ad(f, 2100 + i, grain=0.006, vignette=0.10, ca_px=0.25)
    ka = min(1.0, (i + 1) / 6.0)
    f = f * (1 - (1 - scrim()) * ka)
    f = put_text(f, i, T, beat)
    return np.clip(f, 0, 1).astype(np.float32)
