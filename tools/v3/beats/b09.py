"""09（27.4–29.2秒）「休まず進むと……」
地球儀（b04 と同じ描き方・明るさ一定・回転は約87°）。サーファーを乗せた光のリボンが海の道を進む。
日数のカウンター「0日」→「25日」はだんだん速く（リボンも同じ進み方）。日本の南で黒潮（細い流れの線）を横切るときに「黒潮」。
左上には b08 のスピードメーターが小さなアイコンで残る。最後の6フレーム：全体を NAVY へなめらかに暗くする（b10 へ）。
"""
import numpy as np

import core
import timeline
from core import W, H
from beats import _geo as G
from beats import _mapui as U
from beats import b04

KUROSHIO = [(21.5, 121.8), (24.5, 123.2), (27.5, 126.8), (30.2, 130.2), (32.3, 133.4), (33.8, 137.0),
            (34.8, 140.4), (35.6, 144.5)]
EXP = 1.55
RIBBON_COL = core.lerp(core.FOAM, core.ACC, 0.45)


def _p(i, f0, f1):
    t = float(np.clip((i - f0) / (f1 - f0), 0, 1))
    return t ** EXP


def render(beat, ctx):
    n = beat['n']
    f0, f1 = 3, n - 9                  # リボンが動く範囲（最後の数フレームは25日で止まる）
    P, s = G.sea_route()
    # 地球の中心：進み具合 p に対するキーを、フレームに直して並べる
    fr = lambda p: f0 + (f1 - f0) * p ** (1 / EXP)
    keys = [(0, 10.0, 22.0), (fr(0.40), -9.0, 40.0), (fr(0.56), -12.0, 62.0), (fr(0.76), -3.0, 95.0),
            (f1, 19.0, 109.0), (n - 1, 19.5, 109.5)]
    cs = G.spline_centres(keys, n)
    globe = G.Globe()
    K = G.route(KUROSHIO, step_deg=0.25, round_deg=1.5)
    Ks = G.arclen(K)
    icon = U.icon_layer(alpha=0.75)
    cnt = U.Counter('25', W / 2, 762, digit_size=170, side_size=92, prefix='', suffix='日')
    f_k = None
    t_k = None
    for i in range(n):
        v = G.View(cs[i][0], cs[i][1], b04.R_G, b04.DISC[0], b04.DISC[1])
        f = globe.render(v)
        p = _p(i, f0, f1)
        # 海の道（全体をうすく）
        f = G.draw_line(f, v.screen(P), core.FOAM, 1.6, 0.28)
        # 黒潮：先端が北緯15°を越えたころから、細い流れの線（流れる点つき）
        part, tip = G.cut(P, s, p)
        tla, tlo = G.latlon(tip)
        if f_k is None and p > 0.70:
            f_k = i
        if f_k is not None:
            ka = core.ease((i - f_k) / 6)
            SK = v.screen(K)
            f = G.draw_line(f, SK, core.SKY, 8.0, 0.34 * ka, glow_sigma=5, glow_alpha=0.2 * ka)
            f = G.draw_line(f, G.dashed(SK, 12, 14, phase=i * 2.5), core.FOAM, 2.6, 0.95 * ka, under=0.4)
            if t_k is None:
                vv = G.View(cs[-1][0], cs[-1][1], b04.R_G, b04.DISC[0], b04.DISC[1])
                x, y, _ = vv.proj(G.xyz(25.5, 139.0))
                t_k = U.Patch([dict(text='黒潮', x=float(x) + 10, y=float(y) + 34, size=36, weight='Medium',
                                    anchor='ls')])
            f = t_k.put(f, ka)
        # 光のリボン（先端から後ろへ全長の 16% ぶん、だんだん薄く）
        if p > 0:
            tail = 0.16
            pa = max(0.0, p - tail)
            ia, ib = np.searchsorted(s, s[-1] * pa), np.searchsorted(s, s[-1] * p)
            seg = np.vstack([P[ia:ib], tip[None]])
            S = v.screen(seg)
            # 後ろほど細く薄く：3つの区間に分けて重ねる
            m = len(S)
            for a0, a1, w, al in ((0.0, 1.0, 2.0, 0.45), (0.35, 1.0, 3.4, 0.7), (0.7, 1.0, 5.0, 0.95)):
                sub = S[int(a0 * (m - 1)):max(int(a1 * m), int(a0 * (m - 1)) + 2)]
                f = G.draw_line(f, sub, RIBBON_COL, w, al, glow_sigma=6, glow_alpha=0.30 * al)
            # サーファー（進む向きにボードを向ける）
            x, y, z = v.proj(tip)
            if z > 0:
                k2 = min(len(seg) - 1, max(0, len(seg) - 6))
                xb, yb, _ = v.proj(seg[k2])
                # サーファーはいつも立ったまま（傾けるのは進む向きに合わせて ±12° まで。倒れて見えないように）
                dxy = np.hypot(x - xb, y - yb)
                ang = float(np.clip(12 * (y - yb) / dxy * np.sign(x - xb + 1e-6), -12, 12)) if dxy > 0.5 else 0.0
                f = G.draw_dots(f, [(x, y)], 7.0, core.FOAM, 0.9, glow_sigma=10, glow_alpha=0.45)
                L = core.SS(s=3, w=160, h=160)
                core.surfer(L, 80, 86, scale=1.5, ang=ang)
                lay = L.done()
                x0, y0 = int(round(x)) - 80, int(round(y)) - 92
                f = _paste(f, lay, x0, y0)
        # 最初：b08 のズームのぼかしが収まる
        if i < 6:
            f = core.zoom_blur(f, 0.22 * (1 - core.ease((i + 1) / 6, 'out')), cx=W / 2, cy=H / 2)
        # 日数（ぼかしのあとに重ねる）・左上のメーター
        days = int(np.floor(25 * p + 1e-6)) if p < 0.999 else 25
        f = cnt.put(f, str(days), core.ease((i + 1) / 4))
        f = U.paste(f, *icon)
        # 最後：NAVY へ
        k = core.ease((i - (n - 6) + 1) / 6, 'inout')
        if k > 0:
            f = f * (1 - k) + core.NAVY * k
        yield np.clip(f, 0, 1).astype(np.float32)


def _paste(f, rgba, x0, y0):
    h, w = rgba.shape[:2]
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if X1 <= X0 or Y1 <= Y0:
        return f
    a = rgba[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    sub = f[Y0:Y1, X0:X1]
    sub += (a[..., :3] - sub) * a[..., 3:4]
    return f
