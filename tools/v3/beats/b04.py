"""04（10.4–14.2秒）「問題は距離。海の道だと、約2万5千キロ。」
地球儀。ナザレのそばに寄った所（b03 のズームアウトを受ける）から地球全体へ引く。
ナザレと土佐市新居に白い点。まず大円の点線（ユーラシアを横切る）と「直線なら約11,000km（陸を通る）」→ 消える。
次に海だけの道を明るい線で描き、地球を回して線の先を画面の中央付近に保つ。カウンター「約25,000km」が線と一緒に増える。
最後の8フレーム：線の先へ寄るズームのぼかしで b05 へ。
"""
import cv2
import numpy as np

import core
import timeline
from core import W, H
from beats import _geo as G
from beats import _mapui as U
from beats import b03

R_G = 400.0                 # 地球全体のときの半径
DISC = (540.0, 1196.0)      # 地球全体のときの画面上の中心
R0 = 30000.0                # 最初（ナザレに寄った所）の半径
ROUTE_COL = core.lerp(core.FOAM, core.ACC, 0.35)


def _timing(beat):
    mf = lambda k: timeline.mora_frame(beat, k)
    n = beat['n']
    f_umi = mf(7)                    # ウ（うみの みちだと）
    f_end = mf(-2)                   # 最後の「キロ」のキ
    r0 = f_umi + 3                   # 道を描きはじめる
    return n, f_umi, r0, f_end


def _centres(beat):
    """各フレームの地球の中心（lat, lon）。経度は単調に増やし、回転の合計は約119°。"""
    n, f_umi, r0, f_end = _timing(beat)
    P, s = G.sea_route()

    def at(p):
        return int(round(r0 + (f_end - r0) * _inv_ease(p)))

    keys = [(0, 39.6, -9.1), (38, 31.0, 43.0), (r0, 28.0, 44.0),
            (at(0.40), -9.0, 46.0), (at(0.56), -12.0, 64.0), (at(0.76), -3.0, 96.0),
            (f_end, 19.0, 109.0), (n - 1, 20.0, 110.0)]
    keys = sorted(dict((k[0], k) for k in keys).values())
    return G.spline_centres(keys, n)


def _prog(i, r0, f_end):
    return core.ease((i - r0) / max(1, f_end - r0), 'inout')


def _inv_ease(p):
    # ease 'inout'（3次）の逆
    lo, hi = 0.0, 1.0
    for _ in range(40):
        m = (lo + hi) / 2
        if core.ease(m, 'inout') < p:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def views(beat):
    n, f_umi, r0, f_end = _timing(beat)
    cs = _centres(beat)
    out = []
    for i in range(n):
        k = core.ease(i / 40.0, 'out')
        R = float(np.exp(np.log(R0) * (1 - k) + np.log(R_G) * k))
        # 画面の中心点：b03 の浜の位置 → 地球全体の中心
        kk = core.ease(i / 34.0, 'inout')
        cx = core.lerp(b03.PX, DISC[0], kk)
        cy = core.lerp(b03.PY, DISC[1], kk)
        roll = b03.ROLL * (1 - core.ease(i / 30.0, 'inout'))
        la, lo = cs[i]
        out.append(G.View(la, lo, R, cx, cy, roll))
    return out


def render(beat, ctx):
    n, f_umi, r0, f_end = _timing(beat)
    globe = G.Globe()
    vs = views(beat)
    P, s = G.sea_route()
    GC = G.gc_line(G.NAZARE, G.SHINKAI, 0.3)
    gcs = G.arclen(GC)
    A, B = G.xyz(*G.NAZARE), G.xyz(*G.SHINKAI)
    t_umi = U.Patch([dict(text='海の道で', x=W / 2, y=620, size=44, weight='Medium')])
    cnt = U.Counter('25,000', W / 2, 762, digit_size=170, side_size=92)
    t_gai = U.Patch([dict(text='概算', x=cnt.right, y=820, size=30, weight='Medium', anchor='rs', alpha=0.9)])
    gc_lab = None
    for i in range(n):
        v = vs[i]
        f = globe.render(v)
        # --- 大円の点線（陸を通る）：f10〜f32 で伸び、f44〜f56 で消える
        gp = core.ease((i - 10) / 22, 'inout')
        ga = 1 - core.ease((i - 44) / 12)
        if gp > 0 and ga > 0:
            part, _ = G.cut(GC, gcs, gp)
            S = v.screen(part)
            f = G.draw_line(f, G.dashed(S, 13, 11), core.FOAM, 2.4, 0.9 * ga, under=0.55)
            if gc_lab is None and i >= 20:
                # 弧のいちばん上のあたりの、少し上にラベル（引き終わった所の見え方で決める）
                Sf = vs[34].screen(GC)
                ok = np.isfinite(Sf[:, 1])
                j = np.flatnonzero(ok)[np.argmin(Sf[ok, 1])]
                lx = float(np.clip(Sf[j, 0], 330, 700))
                ly = float(min(Sf[j, 1] - 34, 700))
                gc_lab = U.Patch([dict(text='直線なら約11,000km（陸を通る）', x=lx, y=ly, size=34, weight='Medium')])
            if gc_lab is not None:
                f = gc_lab.put(f, core.ease((i - 20) / 8) * ga)
        # --- 海の道
        p = _prog(i, r0, f_end)
        if p > 0:
            part, tip = G.cut(P, s, p)
            S = v.screen(part)
            f = G.draw_line(f, S, ROUTE_COL, 3.4, 1.0, glow_sigma=7, glow_alpha=0.35, under=0.45)
            tx, ty, tz = v.proj(tip)
            if tz > 0:
                f = G.draw_dots(f, [(tx, ty)], 6.5, core.FOAM, 1.0, glow_sigma=9, glow_alpha=0.5)
        # --- 2つの白い点
        pts = []
        for Q, a0 in ((A, 4), (B, 18)):
            x, y, z = v.proj(Q)
            al = core.ease((i - a0) / 8) * float(np.clip(z / 0.08, 0, 1))
            if al > 0:
                f = G.draw_dots(f, [(x, y)], 7.0, core.FOAM, al, glow_sigma=6, glow_alpha=0.4 * al)
        # --- 文字
        ta = core.ease((i - f_umi + 1) / 6) * core.ease((n - 2 - i) / 6)
        if ta > 0:
            f = U.scrim(f, 560, 790, 0.30 * ta, feather=130)
            f = t_umi.put(f, ta)
            val = 25000 * p
            sv = U.fmt(val, 100) if p < 0.999 else '25,000'
            f = cnt.put(f, sv, ta)
            f = t_gai.put(f, ta * core.ease((i - r0 - 6) / 8))
        # --- b03 からのズームのぼかしが収まる（b03 と同じ上の霧も最初だけ重ねる）
        if i < 14:
            fa = 1 - core.ease(i / 14.0)
            yy = np.arange(H, dtype=np.float32)
            far = np.clip((b03.PY - 30 - yy) / 420, 0, 1)
            far = (far * (2 - far) * 0.96 * fa)[:, None, None]
            f = f * (1 - far) + b03.FOG * far
        if i < 10:
            f = core.zoom_blur(f, 0.30 * (1 - core.ease((i + 1) / 10, 'out')), cx=v.cx, cy=v.cy)
        # --- 最後：線の先へ寄るズームのぼかし → b05
        if i >= n - 8:
            k = (i - (n - 8) + 1) / 8
            _, tip = G.cut(P, s, 1.0)
            tx, ty, _ = v.proj(tip)
            sc = 1 + 0.6 * core.ease(k, 'in')
            M = np.float32([[sc, 0, tx * (1 - sc)], [0, sc, ty * (1 - sc)]])
            f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
            f = core.zoom_blur(f, 0.25 * k ** 1.4, cx=tx, cy=ty)
        yield np.clip(f, 0, 1).astype(np.float32)
