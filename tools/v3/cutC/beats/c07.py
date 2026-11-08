"""案C 07（2.0 秒）「高さ、132キロ。」 テロップ「132km」（132 を強調）。

高さの柱（_column：画面の縦＝高さ、対数の目盛り）を、悪役を追うカメラが上がっていく。悪役は画面の y=710 で回り続ける。
  0–10 ：積雲の層（1.5–2.6km）を突き抜ける（手前の雲ほど速く大きくぶれる）。
  すじ雲（約 10km）と旅客機（10km、「旅客機」の小さなラベル）を追い越し、空は紺 → 黒へ。60km から星。
  カーマン・ライン（100km の細い線と「ここから宇宙」）を越えて、「キ」（132キロのキ）で 132km に着いて止まる（頂点）。
  高さのカウンター（数字が回る）は「タ」で出て、いまの高さを数え、「キ」で 132 に決まる。
  最後の 6 フレーム：カメラが下へ振れはじめ（中身が上へ、縦のぶれ）、C08 の宇宙のカメラ（同じ速さで下へ振る）へ溶ける。
  高さの道のり：C06 の終わり（1.3km、対数の速さ 0.0295/フレーム）を受け継いで、「キ」で 132km・速さ 0。
"""
import math

import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W, H
from cutB.beats import _motion as M, _render as RR, _maplook as ML
from beats import _mapui as U
from cutC.beats import _column as CO, _vill as V, _fall as FA, _space as SPC
import timeline_c as TL

A_START, A_TOP = 1.30, 132.0
V06 = (CO.lnA(1.30) - CO.lnA(0.10)) / 46.0          # C06 の対数の速さ（1 フレーム）
N_X = 6                                              # 最後の重ね（C08 の宇宙のカメラへ）
CNT_Y = 1110
_T = {}


def f_lock(beat):
    return TL.mora_frame(beat, 10)                   # 「キ」


def alt(i, beat):
    """C07 の i フレーム目の高さ（km）。"""
    fl = f_lock(beat)
    L0, L1 = float(CO.lnA(A_START)), float(CO.lnA(A_TOP))
    if i < 0:                                         # C06 の速さの続き（ぶれの量をつなぐため）
        return float(CO.A0 * math.expm1(L0 + V06 * i))
    x = min(i / float(fl), 1.0)
    v0 = V06 * fl / (L1 - L0)
    u = v0 * x + (3 - 2 * v0) * x * x + (v0 - 2) * x ** 3
    return float(CO.A0 * math.expm1(L0 + (L1 - L0) * u))


def counter():
    if 'c' not in _T:
        _T['c'] = M.Odometer('132', W / 2, CNT_Y, digit_size=150, side_size=96, prefix='', suffix='km', hl_suffix='',
                             feather_px=10)
        _T['jet'] = U.Patch([dict(text='旅客機', x=0, y=0, size=28, weight='Medium', anchor='ms', track=0.06, alpha=0.9,
                                  shadow=0.8)])
    return _T['c']


def scene(i, beat):
    """C07 の i フレーム目（文字なし・悪役なし・仕上げ済みでない sRGB）と、そのときの縦の速さ。"""
    a, a1 = alt(i, beat), alt(i - 1, beat)
    vy = CO.K * (float(CO.lnA(a)) - float(CO.lnA(a1)))
    f = CO.frame(a, vy)
    f = CO.karman(f, a)
    f, (jx, jy) = CO.airliner(f, a)
    counter()
    if -40 < jy < H:
        f = _T['jet'].put(f, CO.edge_fade(jy + 44), dx=jx, dy=jy + 44)
    f = CO.ruler(f, a)
    if vy > 2:
        f = core.whip(f, vy * 0.35, axis=0)
    return f, a, vy


def render(beat, ctx):
    n = beat['n']
    fl = f_lock(beat)
    f_on = TL.mora_frame(beat, 0)                    # 「タ」
    k08, _ = SPC.offsets()
    od = counter()
    for i in range(n):
        f, a, vy = scene(i, beat)
        # 悪役：回り続ける（C08 の回転の式の続き）。速いときは下へ速さの線
        k = k08 - (n - i)
        x, y = FA.APEX[0], 710.0
        amt = float(np.clip(vy / 60.0, 0, 1))
        f = CO.speed_lines(f, x, y, i, amt * 0.8)
        r, x0, y0 = V.rgba(x, y, 3.4, FA.tumble_ang(k), V.pose('tumble'), k / 30.0, env=V.env_mix(float(np.clip((math.log(a) - math.log(5)) / (math.log(80) - math.log(5)), 0, 1))))
        if vy > 3:
            r = M.smear(r, 0.0, -vy * 0.15)
        f = M.put_rgba(f, r, x0, y0)
        f = RR.finish_ad(ML.label_scrims(f, 1.0, cg=False), 400 + i, grain=0.006, vignette=0.10, bloom_k=0.06)
        # 最後の N_X フレーム：C08 の宇宙のカメラ（k08 より前＝同じ速さで下へ振っている）へ溶ける
        q = (i - (n - N_X)) / float(N_X)
        if q >= 0:
            kk = k08 - (n - i)
            rgb, info = SPC.earth(kk)
            from cutC.beats.c10 import dir_blur, cam_flow
            (fx, fy), _ = cam_flow(kk)
            rgb = dir_blur(rgb, fx, fy, 0.5)
            g = SPC.map_post(rgb, kk, info, 480 + i)
            from cutC.beats.c08 import villain_xy
            vx_, vy_ = villain_xy(kk)
            rr, x0, y0 = V.rgba(vx_, vy_, 3.4, FA.tumble_ang(kk), V.pose('tumble'), kk / 30.0)
            g = M.put_rgba(g, rr, x0, y0)
            w = M.ease((q * N_X + 1) / (N_X + 1), 'sine_inout')
            f = core.whip(f, 40 * w, axis=0) * (1 - w) + g * w
        # カウンター：「タ」で出て、いまの高さ。「キ」で 132 に決まる。最後の 6 フレームで上へ抜ける
        if i >= f_on - 2:
            al = min(1.0, (i - f_on + 3) / 4.0)
            ex = M.ease((i - (n - N_X)) / float(N_X), 'cubic_in') if i >= n - N_X else 0.0
            dv = a - alt(i - 1, beat)
            f = od.put(f, a, alpha=al * (1 - ex), i=i, f_lock=fl, dv=dv, dy=-40 * ex)
        yield np.clip(f, 0, 1).astype(np.float32)
