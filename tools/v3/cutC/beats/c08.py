"""案C 08（約 1.9 秒）「……宇宙まで、来ちゃいました！」（2026-11-09 クライアント：旧「……宇宙です。」） テロップ「宇宙」（全体が強調色）。

地球の丸み・薄い青い大気の層・星・太陽（左上、やわらかい円盤とにじみ。点滅しない）。地球の上に日本列島が見える
（国土地理院の標高タイルの起伏。出典を右下に）。カメラは _space の 1 本のカメラ（C09・C10 へそのまま続く）。
  0–12：C07 の上昇の続き。上（宇宙）を見ていたカメラが下へ振れて、地球が下から上がってくる（向きのぶれ）。
  悪役（白いピクトグラム）は画面のまんなかの少し上でくるくる回りながら上がり、頂点（C09 の頭の位置）で止まる。
  「ウ」（宇宙）の 2 フレーム前から「宇宙」が 1 文字ずつ上がる（150px、ACC）。最後の 5 フレームで上へ抜ける（C09 はテロップなし）。
"""
import math

import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W, H, SAFE
from cutB.beats import _motion as M, _maplook as ML
from cutC.beats import _space as SPC, _vill as V, _fall as FA
from cutC.beats.c10 import dir_blur, cam_flow
import timeline_c as TL

TEL_Y = 520
TEL_SIZE = 150
_T = {}


def telop():
    if 'k' not in _T:
        _T['k'] = M.KText([dict(text='宇宙', x=W / 2, y=TEL_Y, size=TEL_SIZE, weight='Bold', hl='宇宙', track=0.10,
                                shadow=0.9)])
    return _T['k']


def villain_xy(k):
    k08, k09 = SPC.offsets()
    q = (k - k08) / float(k09 - k08)
    y = FA.APEX[1] + 70.0 * (1 - M.ease(q, 'cubic_out'))
    return FA.APEX[0], y


def render(beat, ctx):
    n = beat['n']
    f_u = TL.mora_frame(beat, 0)                 # 「ウ」
    for i in range(n):
        k = SPC.k_of(beat, i)
        rgb, info = SPC.earth(k)
        (vx, vy), dz = cam_flow(k)
        rgb = dir_blur(rgb, vx, vy, 0.5)
        f = SPC.map_post(rgb, k, info, 500 + i)
        x, y = villain_xy(k)
        x1, y1 = villain_xy(k - 1)
        r, x0, y0 = V.rgba(x, y, 3.4, FA.tumble_ang(k), V.pose('tumble'), k / 30.0)
        r = M.smear(r, (x - x1) * 0.5, (y - y1) * 0.5 - vy * 0.0)
        f = M.put_rgba(f, r, x0, y0)
        pass  # 出典（国土地理院）は画面に出さず、キャプションに書く（2026-11-08 クライアントの指示）
        f = telop().reveal(f, i, f_u - 2, 'char_rise', stagger=2.0, exit=n - 5, exit_dur=5)
        yield np.clip(f, 0, 1).astype(np.float32)
