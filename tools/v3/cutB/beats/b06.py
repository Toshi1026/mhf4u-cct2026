"""B06（約3.9秒）「高さは、にせんごひゃくメートル。下からは、ほぼ点。」＋キラーン（効果音）
横から見た断面。左の陸が桂浜。うすい点線の放物線の上を、悪役が回りながら上がっていき、頂点で止まる（アニメのお約束の一瞬の宙づり）。
 ニ（にせん）：「最高 2,500m」（2,500m を強調）。ル（メートル）のころ頂点に着き、頂点から右へ「2,500m」の点線、下へ高さの点線。
 その後：同じ縮尺の富士山（3,776m）のうすい輪郭が、すそから描かれ、「富士山の約3分の2」（2,525÷3,776＝0.67）。
 頂点の悪役（身長1.6m）を真下から見た視角は 1.6/2,525 rad ≈ 2.2分（人の目の限界 約1分の2倍ほど＝「ほぼ点」）。
2026-11-07 訂正：10km 版。旧「最高 3,300m・富士山の9割近く」。
 シ（したからは）：桂浜から頂点へ見上げる点線。ホ（ほぼ点）：悪役が縮んで点になる。キラーン（効果音 ki）：1回だけ小さく光る。
最初の 6 フレーム：05 の上への whip を受ける（下から上がってくる＋明るさを 12 フレームで戻す）。07 はこの最後の状態から続ける。

2026-11 動きの改善（director's motion spec §1-06）：
- カメラ（世界のレイヤーだけ）：6–80 で (700, 760) を中心に 1.00→1.08（sine_inout）、80–92「ほぼ点」で頂点へ 1.13（expo_out）、
  キラーンのあいだは止める。目盛り・桂浜・富士山・2,500m のラベルは Cam2D.project で置き（大きさは変えない）、セーフゾーンに収める。
- 「最高 X m」：X は悪役のいまの高さ（_motion.Odometer）。ルで 2,500 にロック（back_out のポップ）。
- 悪役：上がりはじめ（4–14）は弧の向きに 1.3→1.0 伸びてサブフレームのぶれ、マントは画面の向きで速さの反対へ。
  頂点では回転が止まり、3px・0.8Hz でゆれる。80–90 で expo_in で縮み、点はポンと決まる（1.4→1.0）。
- キラーン：_art.put_glint_fx。一番大きいフレームが効果音の頭（timing()['ki']）。
- 見上げる点線は上へ流れる（0.8px/フレーム）。富士山は輪郭が quart_out で描かれ、塗りが 4 フレーム遅れ、名前は字間を詰めて入る。
- 頂点から右への点線は ACC ではなく FOAM 0.75（1 画面に強調は 1 つ）。
- 入り：05 と同じ速さの whip（_motion.whip 'in'、300px・ぶれ 110）。
2026-11-08 見た目：舞台は _xsec3d（ジオラマの切り口・水の光の筋・ゆっくりのコースティクス・ガラスの富士山と雪の指）。
富士山の輪郭の線と未来の点線は、目盛りのラベルの所を 6px ぬく（label_boxes）。悪役は villain_v2 を 1 回描いてぶらす。
仕上げ（finish_ad、ゆがみなし）はカメラのあと、ラベル・文字はその後（ぶれ・ゆれの外）。
"""
import math

import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W, H, WL
from beats import _mapui as U
from cutB.beats import _art as A, _xsec as X, _motion as M
from cutB.beats import _xsec3d as X3, _figures3d as F3, _render as R

ENTRY, ENTRY_DIM = 12, 0.36
WHIP_IN = 6
PIV1 = (700.0, 760.0)


def timing(beat):
    mo = beat['moras']
    # タカサワ ニセンゴヒャク メエトル シタカラワ ホボテン（キラーンは効果音）
    # 2026-11-07：「キラーン」はナレーションから外し、効果音にした。光るのは「ほぼ点」の言い終わりの約0.35秒後
    assert mo[4][0] == 'ニ' and mo[13][0] == 'ル' and mo[14][0] == 'シ' and mo[19][0] == 'ホ', mo
    mf = lambda k: timeline.mora_frame(beat, k)
    ki = min(mf(len(mo) - 1) + 10, beat['n'] - 14)
    return dict(start=4, san=mf(4), ru=mf(13), shi=mf(14), ho=mf(19), ki=ki, n=beat['n'])


def villain_u(i, T):
    tau = np.clip((i - T['start']) / float(T['ru'] - T['start']), 0, 1)
    return 0.5 * (1 - (1 - tau) ** 1.7)


def villain_scale(i, T):
    k = M.ease((i - T['ho']) / 10.0, 'expo_in')
    return X.VS * (1 - k) + 0.12 * k


def altitude_m(i, T):
    u = villain_u(i, T)
    return 4 * 2525 * u * (1 - u)


# ---------------------------------------------------------------- カメラ
def cam_zooms(i, T):
    z1 = 1.0 + 0.08 * M.ease((i - 6) / float(max(1, T['ho'] - 6)), 'sine_inout')
    z2 = 1.0 + (1.13 / 1.08 - 1.0) * M.ease((i - T['ho']) / 12.0, 'expo_out')
    return z1, z2


def cam_M(i, T):
    """世界 → 画面（2×3）。PIV1 まわりの z1 のあと、頂点まわりの z2。07 はこの最後の値から戻す。"""
    z1, z2 = cam_zooms(i, T)
    A1 = np.array([[z1, 0, PIV1[0] * (1 - z1)], [0, z1, PIV1[1] * (1 - z1)], [0, 0, 1]])
    ax, ay = X.APEX
    A2 = np.array([[z2, 0, ax * (1 - z2)], [0, z2, ay * (1 - z2)], [0, 0, 1]])
    return (A2 @ A1)[:2]


def project(Mc, x, y, clamp=core.SAFE, margin=(0, 0, 0, 0)):
    px = Mc[0, 0] * x + Mc[0, 1] * y + Mc[0, 2]
    py = Mc[1, 0] * x + Mc[1, 1] * y + Mc[1, 2]
    if clamp is not None:
        px = min(max(px, clamp[0] + margin[0]), clamp[2] - margin[2])
        py = min(max(py, clamp[1] + margin[1]), clamp[3] - margin[3])
    return px, py


def warp(f, Mc):
    if np.allclose(Mc, [[1, 0, 0], [0, 1, 0]], atol=1e-6):
        return f
    return cv2.warpAffine(f, Mc.astype(np.float32), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- 文字・ラベル
_TX = {}


def texts():
    """(高さのカウンター, 「富士山の約3分の2」)。"""
    if 't' not in _TX:
        ctr = M.Odometer('2,500', W / 2, 430, digit_size=100, side_size=100, prefix='最高 ', suffix='m',
                         gap=0.02, suffix_color=core.ACC, lock_scale=1.08, lock_drop=5.0, lock_dur=8, blur_k=0.5)
        sub = M.KText([dict(text='富士山の約3分の2', x=W / 2, y=505, size=48, weight='Medium', track=0.08)])
        fj = M.KText([dict(text='富士山 3,776m', x=X.FUJI_X, y=X.y_of(3776) - 22, size=28, weight='Medium',
                           alpha=0.85)])
        _TX['t'] = (ctr, sub, fj)
    return _TX['t']


def put_counter(f, i, T, alpha=1.0, extra_dy=0.0, blur=0.0):
    """「最高 X m」（サで出て、悪役の高さと一緒に回り、ルで 2,500 に止まる）。"""
    ctr = texts()[0]
    if i < T['san'] - 2:
        return f
    a = M.ease((i - T['san'] + 3) / 4.0, 'sine_inout') * alpha
    dy = 14 * (1 - M.ease((i - T['san'] + 3) / 8.0, 'expo_out'))
    v = altitude_m(i, T)
    dv = altitude_m(i + 0.5, T) - altitude_m(i - 0.5, T)
    return ctr.put(f, v, a, i=i, f_lock=T['ru'], dv=dv, dy=dy + extra_dy, blur=blur)


def put_labels(f, i, T, Mc, labels_a=1.0, fuji_a=None, apex_a=None, fuji_i=None):
    """世界の点に付くラベル（大きさは変えず、位置だけカメラで動かしてセーフゾーンに収める）。"""
    lab = X.labels()
    for key, (x, y) in (('grid', (1010.0, X.y_of(1000) - 10)), ('kat', (X.X0 - 10, WL + 62))):
        p = lab[key]
        # 「桂浜」は中央ぞろえ（幅 約74px）なので、左の余白は半分の幅＋少し（寄りで左へ押し出されてもセーフゾーンの外に出ない）
        px, py = project(Mc, x, y, margin=(45 if key == 'kat' else 0, 30, 0, 0))
        if key == 'grid':
            px = min(px, 1010.0)
        f = p.put(f, labels_a, dx=px - x, dy=py - y)      # 2,000m も grid の板に入っている（1,000m の点で動かす）
    if apex_a is not None and apex_a > 0.003:
        ax, ay = 1010.0, X.APEX[1] - 12
        px, py = project(Mc, ax, ay)
        f = lab['apex'].put(f, apex_a, dx=min(px, 1010.0) - ax, dy=py - ay)
    fa = 0.0 if fuji_a is None else fuji_a
    if fa > 0.003:
        fj = texts()[2]
        x, y = X.FUJI_X, X.y_of(3776) - 22
        px, py = project(Mc, x, y)
        f = fj.reveal(f, i if fuji_i is None else fuji_i, T['ru'] + 9, 'track', track_from=0.30, dur=12,
                      alpha=fa, dx=px - x, dy=py - y)
    return f


# ---------------------------------------------------------------- 世界の線
def apex_lines(f, a, phase=0.0):
    """頂点から右への点線（FOAM 0.75）と、水面までの高さの点線。"""
    if a <= 0.003:
        return f
    ax, ay = X.APEX
    y0 = int(ay) - 4
    L = core.SS(s=2, w=W, h=int(WL) + 4 - y0)
    k = M.ease(a, 'cubic_out')
    A.dashed(L, [(ax + 14, ay - y0), (ax + 14 + (1010 - ax - 14) * k, ay - y0)], L.col(core.FOAM, 0.75), L.w(2.0),
             dash=8, gap=7, phase=phase)
    A.dashed(L, [(ax, ay + 16 - y0), (ax, ay + 16 + (WL - ay - 16) * k - y0)], L.col(core.FOAM, 0.38), L.w(1.6),
             dash=6, gap=8, phase=phase)
    return U.paste(f, L.done(), 0, y0)


def sight_line(f, a, phase=0.0):
    """桂浜から頂点を見上げる点線（点は上へ流れる）。"""
    if a <= 0.003:
        return f
    ax, ay = X.APEX
    p0 = np.array([X.X0 + 6, WL - 10.0])
    p1 = p0 + (np.array([ax - 10, ay + 14]) - p0) * M.ease(min(1, a * 1.2), 'cubic_out')
    y0 = int(ay) - 4
    L = core.SS(s=2, w=W, h=int(WL) + 4 - y0)
    A.dashed(L, [(p0[0], p0[1] - y0), (p1[0], p1[1] - y0)], L.col(core.FOAM, 0.55), L.w(1.8), dash=5, gap=7,
             phase=phase)
    return U.paste(f, L.done(), 0, y0, min(1.0, a * 1.5))


def fuji_state(i, T):
    stroke = M.ease((i - T['ru'] - 3) / 16.0, 'quart_out')
    fill = M.ease((i - T['ru'] - 7) / 14.0, 'sine_inout')
    return stroke, fill


def label_boxes(i, T, Mc):
    """画面の上の目盛りのラベル（1,000m・2,000m）と 2,500m の箱 → 世界の座標の箱（富士山の輪郭の線をぬく所）。"""
    lab = X.labels()
    out = []
    Ai = np.linalg.inv(np.vstack([Mc, [0, 0, 1]]))

    def add(p, dx, dy):
        a = p.rgba[..., 3]
        rows = np.nonzero(a.max(1) > 0.05)[0]
        cols = np.nonzero(a.max(0) > 0.05)[0]
        if not len(rows):
            return
        # 行のすき間で分ける（1 つの板に 2 つのラベルが入っているとき）
        groups = np.split(rows, np.nonzero(np.diff(rows) > 4)[0] + 1)
        for g in groups:
            cc = np.nonzero(a[g].max(0) > 0.05)[0]
            x0, x1 = p.x0 + dx + cc.min(), p.x0 + dx + cc.max()
            y0, y1 = p.y0 + dy + g.min(), p.y0 + dy + g.max()
            P = np.array([[x0, y0, 1], [x1, y1, 1]]) @ Ai.T
            out.append((P[0, 0], P[0, 1], P[1, 0], P[1, 1]))
    x, y = 1010.0, X.y_of(1000) - 10
    px, py = project(Mc, x, y, margin=(0, 30, 0, 0))
    add(lab['grid'], min(px, 1010.0) - x, py - y)
    ax, ay = 1010.0, X.APEX[1] - 12
    px, py = project(Mc, ax, ay)
    add(lab['apex'], min(px, 1010.0) - ax, py - ay)
    return out


def scene(i, T, f=None, fuji_a=None, fuji=None, boxes=None):
    """06 の世界の背景（ジオラマの空と水・桂浜・ガラスの富士山）。ラベルは put_labels（カメラのあと）。fuji=(stroke, fill) で上書き、
    fuji_a で全体の濃さ（07 が消すとき）。boxes：富士山の輪郭の線をぬく文字の箱（世界の座標）。"""
    f = X3.background(i / core.FPS) if f is None else f
    st, fl = fuji_state(i, T) if fuji is None else fuji
    if fuji_a is not None:
        fl *= fuji_a
    if st > 0.003 and (fuji_a is None or fuji_a > 0.003):
        f = X3.fuji(fl, f)
        rgba, x0, y0 = X.fuji_layer2(st, 0.0)
        rgba = X3.knockout(rgba, x0, y0, boxes or [])
        f = U.paste(f, rgba, x0, y0, 1.0 if fuji_a is None else fuji_a)
    return f


# ---------------------------------------------------------------- 悪役
def villain_draw(t, T):
    """小数のフレーム t の悪役の板（世界の座標）。"""
    u = villain_u(t, T)
    x, y = X.arc(u)
    sc = villain_scale(t, T)
    sec = t / core.FPS
    ang = -20 + 560 * u / 0.5
    hang = max(0.0, t - T['ru'])
    bob = -3 * math.sin(2 * math.pi * 0.8 * hang / core.FPS) * (hang > 0)
    tg = X.tangent_ang(min(u, 0.499))
    s_al = 1.3 - 0.3 * M.ease((t - T['start']) / 10.0, 'cubic_out')
    M2 = M.squash_matrix(tg, s_al, 1 / math.sqrt(s_al)) if s_al > 1.002 else None
    # マント：上がるあいだは速さの反対（2 フレーム遅れ）、頂点では左うしろへ少し垂れてはためく（真下だと逆さの頭に重なる）
    ul = villain_u(t - 2, T)
    tl = X.tangent_ang(min(max(ul, 0.001), 0.499))
    w_hang = M.ease(hang / 10.0, 'sine_inout')
    cape = (tl + 180) * (1 - w_hang) + 160 * w_hang + 8 * math.sin(sec * 9.0)     # 頂点：左うしろへ少し垂れる
    return F3.villain_v2(float(x), float(y) - 14 * sc + bob, sc, ang, 'fly', sec, M2=M2, cape_ang=cape)


def put_villain(f, i, T):
    sc = villain_scale(i, T)
    if sc * 34 >= 5:
        x0, y0 = X.arc(villain_u(i - 0.5, T))
        x1, y1 = X.arc(villain_u(i + 0.5, T))
        return put_smeared(f, villain_draw(float(i), T), x1 - x0, y1 - y0, shutter=0.4)
    # 点：ポンと決まる（1.4→1.0）
    k0 = next((k for k in range(int(T['ho']), int(T['ho']) + 14) if villain_scale(k, T) * 34 < 5), int(T['ho']) + 8)
    pr = 1.4 - 0.4 * M.ease((i - k0 + 1) / 4.0, 'back_out')
    return A.put_dot(f, X.APEX[0], X.APEX[1] - 2, 2.4 * pr)


def put_smeared(f, r, vx, vy, shutter=0.4):
    """1 回だけ描いた板を、1 フレームの移動 (vx, vy) × shutter の長さで速さの向きにぶらして重ねる（サブフレームの代わり）。"""
    if r is None:
        return f
    rgba, x0, y0 = r
    sp = math.hypot(vx, vy)
    if sp * shutter > 1.5:
        ln = sp * shutter
        pad = int(ln / 2) + 2
        rgba = M.smear(np.pad(rgba, ((pad, pad), (pad, pad), (0, 0))), vx / sp * ln, vy / sp * ln)
        x0, y0 = x0 - pad, y0 - pad
    return M.put_rgba(f, rgba, x0, y0)


def glint_k(i, T):
    """キラーンの k（一番大きいフレームが ki＝効果音の頭）。"""
    return i - (T['ki'] - A.GLINT_PEAK)


def render(beat, ctx):
    n = beat['n']
    T = timing(beat)
    ctr, t_sub, _ = texts()
    for i in range(n):
        Mc = cam_M(i, T)
        bx = label_boxes(i, T, Mc)
        f = scene(i, T, boxes=bx)
        u = villain_u(i, T)
        # 放物線（この先はうすい点線、通った所は実線）
        rgba, x0, y0 = X.trail_layer(u, ghost=M.ease((i + 1) / 10.0, 'sine_inout'))
        f = U.paste(f, X3.knockout(rgba, x0, y0, bx), x0, y0)
        # 頂点の印
        a_ap = float(np.clip((i - T['ru'] + 6) / 10.0, 0, 1))
        f = apex_lines(f, a_ap)
        f = sight_line(f, float(np.clip((i - T['shi'] + 1) / 12.0, 0, 1)) * (1 - 0.5 * M.ease((i - T['ki']) / 12.0)),
                       phase=-0.8 * i)
        f = put_villain(f, i, T)
        f = A.put_glint_fx(f, X.APEX[0], X.APEX[1] - 2, glint_k(i, T), r=42)
        # カメラ（世界 → 画面）、仕上げ、ラベルは画面で（ぶれ・ゆれの外）
        f = warp(f, Mc)
        f = R.finish_ad(f, 475 + i, grain=0.005, vignette=0.10)
        st, _fl = fuji_state(i, T)
        f = put_labels(f, i, T, Mc, fuji_a=M.ease((i - T['ru'] - 9) / 6.0, 'sine_inout') if st > 0 else 0.0,
                       apex_a=M.ease((i - T['ru'] + 1) / 6.0, 'sine_inout'))
        # 文字
        f = put_counter(f, i, T)
        f = t_sub.reveal(f, i, T['ru'] + 4, 'char_rise', stagger=0.8, dur=10)
        # 入り：05 の上への whip を受ける
        if i < WHIP_IN:
            f = M.whip(f, i / float(WHIP_IN), axis=0, sign=-1, px=300.0, blur=110.0, direction='in')
        if i < ENTRY:
            q = i / float(ENTRY)
            f = f * (ENTRY_DIM + (1 - ENTRY_DIM) * (0.5 * q + 0.5 * core.ease(q, 'smooth')))
        yield np.clip(f, 0, 1).astype(np.float32)
