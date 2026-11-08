"""案C 09・10 の落ちる悪役の動き（閉じた式。C09 と C10 で同じ関数を使い、つなぎ目で位置・速さ・形がそろう）。

k は C10 の頭からのフレーム（C09 は負）。戻り値 state(k)：dict(x, y, scale, ang, pc（形）, t（はためきの時刻）, kind)。
  C09（k −49..−1）：頂点でくるくる（C08 の続き、回転がゆっくりになる）→ −40..−30 で「あれ？」と止まり、ばねで頭が下（180°）へ
       → ジタバタしながら落ちはじめる（だんだん速く下へ）。最後の 14 フレームはカメラが下を向きはじめる分だけ上へ戻る。
  C10（k 0..65）：画面のやや上で、狙いの点の右上に浮いてジタバタ（風の線）→ 46 から体を伸ばして小さくなりながら
       着水点へ（T の 0 フレーム目の白い筋の位置・速さ 27.5px/フレームにつながる）。
"""
import math

import numpy as np

from cutB.beats import _useA  # noqa: F401
from cutB.beats import _motion as M
from cutC.beats import _vill as V

DIRV = np.array([0.42, -0.91])                     # T の筋が来る向き（画面の右上）
P_BT = np.array([520.50114644, 822.41328895])      # T の着水点（画面）
P65 = P_BT + DIRV * 110.0                          # k=65 の位置（T の筋の式で t=−1）
V65 = -DIRV * 27.5                                 # そのときの速さ（px/フレーム）
HOLD = np.array([600.0, 690.0])
S_HOLD = 5.4
APEX = np.array([540.0, 640.0])


def _e(x, kind='cubic_inout'):
    return M.ease(min(max(x, 0.0), 1.0), kind)


def state(k):
    k = float(k)
    if k < -30:
        # 頂点（C08 の続き）：回転は C08 の終わりの速さ（約 9°/フレーム）から −40 でほぼ止まる
        x, y = APEX[0], APEX[1] + 4.0 * math.sin((k + 49) * 0.12)
        if k < -40:
            ang = tumble_ang(k)
            return dict(x=x, y=y, scale=3.4, ang=ang, pc=V.pose('tumble'), t=k / 30.0, kind='tumble')
        # −40..−30：「あれ？」→ ばね（ζ0.45・3Hz）で頭が下へ。形はタンブル → ジタバタ
        a0 = tumble_ang(-40)
        a = a0 + (_target_down(a0) - a0) * M.spring((k + 40) / 30.0, 0.45, 3.0)
        w = _e((k + 40) / 6.0)
        pc = V.mix(V.pose('tumble'), V.pose('flail', k / 30.0), w)
        return dict(x=x, y=y - 6.0 * math.sin(min(max((k + 40) / 10.0, 0), 1) * math.pi), scale=3.4, ang=a, pc=pc,
                    t=k / 30.0, kind='flail')
    if k < 0:
        # −30..0：落ちはじめる（下へ quad_in、最後の 14 フレームはカメラが追うので画面では少し戻る）
        q = (k + 30) / 30.0
        fall = 330.0 * q * q
        cam_back = -70.0 * _e((k + 14) / 14.0, 'quad_in') if k > -14 else 0.0
        a0 = _target_down(tumble_ang(-40))
        ang = a0 + 12.0 * math.sin((k + 30) * 0.35) * min(1.0, (k + 30) / 8.0)
        return dict(x=APEX[0] + 10.0 * q * q, y=APEX[1] + fall + cam_back, scale=3.4 + 0.5 * q, ang=ang,
                    pc=V.pose('flail', k / 30.0), t=k / 30.0, kind='flail')
    # C10
    p0 = np.array([APEX[0] + 10.0, APEX[1] + 330.0 - 70.0])         # k=0（C09 の最後と同じ）
    s0 = 3.9
    if k <= 46:
        # C09 の最後の速さ（下へ約 12px/フレーム）を受け継いで HOLD へ（20 フレーム、終わりの速さ 0）
        q = min(k / 20.0, 1.0)
        v0 = np.array([0.67, 12.0]) * 20.0
        h00, h10, h01 = 2 * q ** 3 - 3 * q ** 2 + 1, q ** 3 - 2 * q ** 2 + q, -2 * q ** 3 + 3 * q ** 2
        p = h00 * p0 + h10 * v0 + h01 * HOLD
        u = _e(q)
        jx, jy = M.jitter(k, 3.0 * u, 9.0, seed=11), M.jitter(k, 3.0 * u, 9.0, seed=12)
        sc = s0 + (S_HOLD - s0) * u
        ang = _hold_ang(k)
        return dict(x=p[0] + jx, y=p[1] + jy, scale=sc, ang=ang, pc=V.pose('flail', k / 30.0), t=k / 30.0, kind='flail')
    # 46..65（＋その先）：体を伸ばして小さくなりながら P65 へ（終わりの速さ V65）
    q = min((k - 46) / 19.0, 1.0)
    jx, jy = M.jitter(46, 3.0, 9.0, seed=11), M.jitter(46, 3.0, 9.0, seed=12)
    A0 = HOLD + np.array([jx, jy])
    # エルミート（始まりの速さ 0、終わりの速さ V65・19 フレームで）
    h00, h10, h01, h11 = 2 * q ** 3 - 3 * q ** 2 + 1, q ** 3 - 2 * q ** 2 + q, -2 * q ** 3 + 3 * q ** 2, q ** 3 - q ** 2
    p = h00 * A0 + h01 * P65 + h11 * V65 * 19.0
    sc = S_HOLD * math.exp(math.log(0.22 / S_HOLD) * q ** 1.6)
    a_hold = _hold_ang(46)
    a_end = _target_down(tumble_ang(-40)) + 25.0
    ang = a_hold + (a_end - a_hold) * _e(q / 0.6)
    w = _e(q / 0.55)
    pc = V.mix(V.pose('flail', k / 30.0), V.pose('spread'), w)
    return dict(x=p[0], y=p[1], scale=sc, ang=ang, pc=pc, t=k / 30.0, kind='spread' if w > 0.5 else 'flail')


def _hold_ang(k):
    """C10 の頭〜46 の角度：C09 の最後のゆれ（180 ± 12°）から、少し傾いたジタバタのゆれ（192 ± 16°）へ。"""
    a0 = _target_down(tumble_ang(-40))
    u = _e(k / 20.0)
    w9 = 12.0 * math.sin((k + 30) * 0.35)
    wh = 12.0 + 10.0 * math.sin(k * 0.31) + 6.0 * math.sin(k * 0.77)
    return a0 + w9 * (1 - u) + wh * u


def tumble_ang(k):
    """C08 から続く回転（度）。C08 の頭（k08）から一定の速さ 9°/フレーム → C09 の頭から −40 まででゆっくり止まる。"""
    from cutC.beats import _space as SPC
    k08, k09 = SPC.offsets()
    rate = 9.0
    if k <= k09:
        return 30.0 + rate * (k - k08)
    a9 = 30.0 + rate * (k09 - k08)
    T = -40.0 - k09                                     # 止まるまでのフレーム
    x = min((k - k09) / T, 1.0)
    return a9 + rate * T * (x - 0.5 * x * x)            # 速さが一直線に 0 へ


def _target_down(a):
    """角度 a から一番近い「頭が下」（180° + 360n）。"""
    return 180.0 + 360.0 * round((a - 180.0) / 360.0)
