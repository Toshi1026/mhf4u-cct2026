"""案C 02（約 1.4 秒）03 の声「でも、現実だと……」をこの止まった絵の上で言う（timeline.VO_EARLY）。
テロップ「現実だと」（大）＋その下に小さく「同じパンチを本当にやったら」（「現実」の「ゲ」で下からふわっと出る）。
レコードを止めたような停止（2026-11-08 クライアント修正：いきなり「現実だとこうなる」では伝わらないので説明を足す）。

01 の最後の絵（c01.world(n-1)）をそのまま止める：最初のフレームで色が抜け（彩度 1→0.22、明るさは保つ輝度の混ぜ方）、
寒色へ少しだけ寄せ、3 フレームで 1.00→1.06 へズームが「カクッ」と寄って止まる（そのあと 1.075 までごくゆっくり）、
最初の 2 フレームだけ横に 6px→0 のずれ（止まった衝撃）。光らせない（平均の明るさは −6% 以内でなめらか）。
テロップは「アニメだと」（縁取りのマンガ文字）から「現実だと」（家の白い文字＋柔らかい影、現実＝強調）へ同じ位置で切り替わる
（最初のフレームから見える。1.10→1.00 の 3 フレームの「ドン」）。03 はこの文字をそのまま引き継ぐ（telop_layer）。
"""
import math

import cv2
import numpy as np

import core
from core import W
import timeline_c as timeline
from cutB.beats import _motion as M
from cutC.beats import c01

TEL_Y = c01.TEL_Y
SIZE = 150
SUB_Y = c01.SUB_Y + 4
SUB_SIZE = 48
SUB_TEXT = '同じパンチを本当にやったら'
_L = {}


def telop_layer():
    """「現実だと」（Black 150px、現実＝ACC）の text_layer。03 も同じものを使う。"""
    if 'l' not in _L:
        _L['l'] = core.text_layer([dict(text='現実だと', x=W / 2, y=TEL_Y, size=SIZE, weight='Black', hl='現実', track=0.04,
                                        shadow=1.0)])
    return _L['l']


def sub_layer():
    """小テロップ「同じパンチを本当にやったら」（Bold 46px・白・影。強調色なし）。"""
    if 's' not in _L:
        _L['s'] = core.text_layer([dict(text=SUB_TEXT, x=W / 2, y=SUB_Y, size=SUB_SIZE, weight='Bold', track=0.04,
                                        shadow=1.0)])
    return _L['s']


def _warp(layer, Mm, blur):
    t, sh = layer
    if Mm is not None:
        t = cv2.warpAffine(t, Mm, (t.shape[1], t.shape[0]), flags=cv2.INTER_LINEAR)
        sh = cv2.warpAffine(sh, Mm, (sh.shape[1], sh.shape[0]), flags=cv2.INTER_LINEAR)
    if blur > 0.3:
        t = cv2.GaussianBlur(t, (0, 0), blur)
        sh = cv2.GaussianBlur(sh, (0, 0), blur)
    return t, sh


def put_telop(f, s=1.0, alpha=1.0, dy=0.0, blur=0.0, sub=1.0, sub_dy=0.0):
    """スケール s（中心はテロップの中心）・上へのずれ dy・ぼかし blur でテロップ（大＋小）を重ねる。sub：小テロップの不透明度。"""
    Mm = None
    if abs(s - 1) > 1e-4 or abs(dy) > 1e-3:
        cy = TEL_Y - SIZE * 0.36
        Mm = np.float32([[s, 0, W / 2 * (1 - s)], [0, s, cy * (1 - s) + dy]])
    if sub > 0.004:
        Ms = Mm if abs(sub_dy) < 1e-3 else (np.float32([[1, 0, 0], [0, 1, sub_dy]]) if Mm is None else
                                            Mm + np.float32([[0, 0, 0], [0, 0, sub_dy]]))
        f = core.put_text(f, _warp(sub_layer(), Ms, blur), alpha * sub)
    return core.put_text(f, _warp(telop_layer(), Mm, blur), alpha)


def sub_frame():
    """小テロップが出はじめる 02 のフレーム（03 の声の「ゲ」＝「現実」の頭、02 の頭から数える）。"""
    bs = {b['key']: b for b in timeline.load()}
    b3 = bs['03']
    assert b3['moras'][2][0] == 'ゲ', b3['moras']
    return timeline.mora_frame(b3, 2) + b3['f0'] - bs['02']['f0']


def sub_alpha(i, k0):
    """(不透明度, 下からのずれ px)：「ゲ」の 1 フレーム前から 6 フレームで。"""
    q = M.ease((i - (k0 - 1)) / 6.0, 'cubic_out')
    return q, 14.0 * (1 - q)


_F = {}


def frozen():
    if 'f' not in _F:
        b1 = next(b for b in timeline.load() if b['key'] == '01')
        _F['f'] = c01.world(b1['n'] - 1, b1)
    return _F['f']


def grade_freeze(f, k):
    """彩度を抜き（輝度を保つ）、寒色へ寄せ、少しだけ暗く。k：0..1（最初のフレームで 0.85、2 フレームで 1）。"""
    l = (f * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1, keepdims=True)
    sat = 1 - 0.78 * k
    g = l + (f - l) * sat
    g = g + (core.c('7FA8C8') - g) * 0.06 * k
    return g * (1 - 0.06 * k)


def render(beat, ctx):
    n = beat['n']
    base = frozen()
    H, W_ = base.shape[:2]
    cx, cy = c01.STAR[0] * 0.5 + W / 2 * 0.5, 980.0
    k0 = sub_frame()
    for i in range(n):
        k = min(1.0, 0.85 + 0.15 * i)
        f = grade_freeze(base, k)
        s = 1 + 0.06 * M.ease((i + 1) / 3.0, 'expo_out') + 0.035 * M.ease(i / max(n - 1, 1), 'sine_inout')
        jolt = 6.0 * (1 - i / 2.0) if i < 2 else 0.0
        Mm = np.float32([[s, 0, cx * (1 - s) + jolt], [0, s, cy * (1 - s)]])
        f = cv2.warpAffine(f, Mm, (W_, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # 周辺を少し落とす（止まった感じ）
        f = f * vignette()
        ts = 1.10 - 0.10 * M.ease((i + 1) / 3.0, 'cubic_out') if i < 3 else 1.0
        sa, sdy = sub_alpha(i, k0)
        f = put_telop(f, ts, sub=sa, sub_dy=sdy)
        yield np.clip(f, 0, 1).astype(np.float32)


def vignette():
    if 'v' not in _F:
        yy, xx = np.mgrid[0:1920, 0:1080].astype(np.float32)
        r2 = ((xx - 540) / 540) ** 2 * 0.32 + ((yy - 960) / 960) ** 2
        _F['v'] = (1 - 0.10 * np.clip(r2, 0, 1.5) ** 1.5)[..., None].astype(np.float32)
    return _F['v']
