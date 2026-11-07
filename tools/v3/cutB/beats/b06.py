"""B06（約4.4秒）「高さは、にせんごひゃくメートル。下からは、ほぼ点。キラーン。」
横から見た断面。左の陸が桂浜。うすい点線の放物線の上を、悪役が回りながら上がっていき、頂点で止まる（アニメのお約束の一瞬の宙づり）。
 ニ（にせん）：「最高 2,500m」（2,500m を強調）。ル（メートル）のころ頂点に着き、頂点から右へ「2,500m」の点線、下へ高さの点線。
 その後：同じ縮尺の富士山（3,776m）のうすい輪郭が、すそから描かれ、「富士山の約3分の2」（2,525÷3,776＝0.67）。
 頂点の悪役（身長1.6m）を真下から見た視角は 1.6/2,525 rad ≈ 2.2分（人の目の限界 約1分の2倍ほど＝「ほぼ点」）。
2026-11-07 訂正：10km 版。旧「最高 3,300m・富士山の9割近く」。
 シ（したからは）：桂浜から頂点へ見上げる点線。ホ（ほぼ点）：悪役が縮んで点になる。キ（キラーン）：1回だけ小さく光る。
最初の 6 フレーム：05 の上への whip を受ける（下から上がってくる＋明るさを 12 フレームで戻す）。07 はこの最後の状態から続ける。"""
import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W, H, WL
from beats import _mapui as U
from cutB.beats import _art as A, _xsec as X

ENTRY, ENTRY_DIM = 12, 0.36


def timing(beat):
    mo = beat['moras']
    # タカサワ ニセンゴヒャク メエトル シタカラワ ホボテン キラアン
    assert mo[4][0] == 'ニ' and mo[13][0] == 'ル' and mo[14][0] == 'シ' and mo[19][0] == 'ホ' and mo[23][0] == 'キ', mo
    mf = lambda k: timeline.mora_frame(beat, k)
    return dict(start=4, san=mf(4), ru=mf(13), shi=mf(14), ho=mf(19), ki=mf(23), n=beat['n'])


def villain_u(i, T):
    tau = np.clip((i - T['start']) / float(T['ru'] - T['start']), 0, 1)
    return 0.5 * (1 - (1 - tau) ** 1.7)


def villain_scale(i, T):
    k = core.ease((i - T['ho']) / 10.0, 'inout')
    return X.VS * (1 - k) + 0.12 * k


def texts():
    return (U.Patch([dict(text='最高 2,500m', x=W / 2, y=430, size=100, weight='Bold', hl='2,500m')]),
            U.Patch([dict(text='富士山の約3分の2', x=W / 2, y=505, size=48, weight='Medium', track=0.08)]))


def apex_lines(f, a):
    """頂点から右への点線（ACC）と、水面までの高さの点線。"""
    if a <= 0.003:
        return f
    ax, ay = X.APEX
    y0 = int(ay) - 4
    L = core.SS(s=2, w=W, h=int(WL) + 4 - y0)
    k = core.ease(a)
    A.dashed(L, [(ax + 14, ay - y0), (ax + 14 + (1010 - ax - 14) * k, ay - y0)], L.col(core.ACC, 0.75), L.w(2.0),
             dash=8, gap=7)
    A.dashed(L, [(ax, ay + 16 - y0), (ax, ay + 16 + (WL - ay - 16) * k - y0)], L.col(core.FOAM, 0.38), L.w(1.6),
             dash=6, gap=8)
    return U.paste(f, L.done(), 0, y0)


def sight_line(f, a):
    """桂浜から頂点を見上げる点線。"""
    if a <= 0.003:
        return f
    ax, ay = X.APEX
    p0 = np.array([X.X0 + 6, WL - 10.0])
    p1 = p0 + (np.array([ax - 10, ay + 14]) - p0) * core.ease(min(1, a * 1.2))
    y0 = int(ay) - 4
    L = core.SS(s=2, w=W, h=int(WL) + 4 - y0)
    A.dashed(L, [(p0[0], p0[1] - y0), (p1[0], p1[1] - y0)], L.col(core.FOAM, 0.55), L.w(1.8), dash=5, gap=7)
    return U.paste(f, L.done(), 0, y0, min(1.0, a * 1.5))


def scene(i, T, f=None, title_a=1.0, fuji_a=None, labels_a=1.0):
    """06 の i フレーム目（07 は i を n06 以上で呼ばずに、自分で組み立てる）。"""
    bg = X.background()
    lab = X.labels()
    f = bg.copy() if f is None else f
    f = lab['grid'].put(f, labels_a)
    f = lab['kat'].put(f, labels_a)
    # 富士山
    fa = core.ease((i - T['ru'] - 3) / 16.0) if fuji_a is None else fuji_a
    if fa > 0:
        rgba, x0, y0 = X.fuji_layer(fa)
        f = U.paste(f, rgba, x0, y0)
        f = lab['fuji'].put(f, core.ease((i - T['ru'] - 10) / 8.0) if fuji_a is None else fuji_a)
    return f


def render(beat, ctx):
    n = beat['n']
    T = timing(beat)
    t_main, t_sub = texts()
    lab = X.labels()
    for i in range(n):
        t = i / core.FPS
        f = scene(i, T)
        u = villain_u(i, T)
        # 放物線（この先はうすい点線、通った所は実線）
        rgba, x0, y0 = X.trail_layer(u, ghost=core.ease((i + 1) / 10.0))
        f = U.paste(f, rgba, x0, y0)
        # 頂点の印
        a_ap = (i - T['ru'] + 6) / 10.0
        f = apex_lines(f, float(np.clip(a_ap, 0, 1)))
        f = lab['apex'].put(f, core.ease((i - T['ru'] + 1) / 6.0))
        f = sight_line(f, float(np.clip((i - T['shi'] + 1) / 12.0, 0, 1)) * (1 - 0.5 * core.ease((i - T['ki']) / 12.0)))
        # 悪役（頂点で宙づりのあいだは、ゆっくり回る）
        x, y = X.arc(u)
        sc = villain_scale(i, T)
        ang = -20 + 560 * u / 0.5 + 25 * max(0, i - T['ru']) / core.FPS
        bob = -3 * np.sin(max(0, i - T['ru']) * 0.18) * (u >= 0.4999)
        if sc * 34 >= 5:
            f = A.put_villain(f, float(x), float(y) - 14 * sc + bob, sc, ang=ang, pose='fly', t=t)
        else:
            f = A.put_dot(f, X.APEX[0], X.APEX[1] - 2, 2.4)
        # キラーン
        g = A.glint_env(i - T['ki'])
        if g > 0.01:
            f = A.put_glint(f, X.APEX[0], X.APEX[1] - 2, 42 * g, ang=10 * (1 - g), alpha=min(1.0, g * 1.4))
        # 文字
        if i >= T['san'] - 1:
            a = core.ease((i - T['san'] + 2) / 6.0)
            f = t_main.put(f, a, dy=10 * (1 - a))
        if i >= T['ru'] + 6:
            a = core.ease((i - T['ru'] - 5) / 6.0)
            f = t_sub.put(f, a, dy=8 * (1 - a))
        # 入り：05 の上への whip を受ける
        if i < ENTRY:
            q = i / float(ENTRY)
            if i < 6:
                dy = 200 * (1 - i / 6.0) ** 2
                f = cv2.warpAffine(f, np.float32([[1, 0, 0], [0, 1, dy]]), (W, H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
                f = core.whip(f, 100 * (1 - i / 6.0) ** 1.4, axis=0)
            f = f * (ENTRY_DIM + (1 - ENTRY_DIM) * (0.5 * q + 0.5 * core.ease(q, 'smooth')))
        yield np.clip(f, 0, 1).astype(np.float32)
