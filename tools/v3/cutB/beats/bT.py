"""案B T（ナレーションなし・36フレーム＝1.2秒）着水 → CG から実写へ。

案A の b11 をそのまま使う（土佐湾の線画 → 仁淀川を河口（2026-11-07 に本当の河口 33.4593N 133.4815E へ訂正）まで → 河口へ降りて、IMG_9662 の1フレーム目に
画素単位で重なる）。その上に、最初の約 12 フレームだけ小さな白いしぶきを重ねる：
 0–3 ：悪役（の点）が東北東の上から短い白い筋になって落ちてくる（03 の放物線と同じ向き）。
 3–13：MAZE の前の海の点（03 の放物線の終点と同じ地図の点。本当の仁淀川の河口の約 0.45km 西）に、上向きの扇に跳ねる数粒のしぶきと小さな泡。
       輪（同心円）や光は使わない。しぶきの大きさは画面の px で決め、寄りに合わせて少しだけ大きくなる。
最後のフレームは b11 と同じ（08 は案A b12 の 0 フレーム目と同じ IMG_9662 から始められる）。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from beats import b11
from beats import _mapui as U, _geo as G
from cutB.beats import _bmap as B

FALL = (0, 3)                 # 筋が落ちてくる
SPL = (3, 13)                 # しぶき
RNG = np.random.default_rng(7)
# しぶきの粒：角度（度、画面の上が −90）、速さ、大きさ
DROPS = [(-150, 1.00, 1.0), (-128, 1.25, 1.2), (-108, 1.45, 1.3), (-92, 1.60, 1.4), (-74, 1.40, 1.2),
         (-56, 1.20, 1.1), (-34, 0.95, 0.9), (-116, 0.85, 0.8), (-66, 0.80, 0.8)]


def _spray(cx, cy, i, k=1.0):
    """しぶきの小さな板 (rgba, x0, y0)。i：フレーム（SPL の中）、k：大きさ。"""
    pw = int(160 * k) + 20
    x0, y0 = int(cx - pw / 2), int(cy - pw * 0.72)
    L = core.SS(s=4, w=pw, h=pw)
    d = L.draw
    px, py = cx - x0, cy - y0
    tau = (i - SPL[0]) / float(SPL[1] - SPL[0])          # 0..1
    fade = 1 - core.ease((tau - 0.45) / 0.55)
    u = 10.5 * k                                         # しぶきの単位（px）
    T = tau * 3.2
    for ang, sp, sz in DROPS:
        a = np.radians(ang)
        vx, vy = np.cos(a) * sp * 2.6, np.sin(a) * sp * 2.6
        x = px + u * (vx * T)
        y = py + u * (vy * T + 0.5 * 1.55 * T * T)
        r = u * 0.42 * sz * (1 - 0.55 * tau)
        # 少し縦長の粒（動きの向きへ）
        d.ellipse(L.box(x - r * 0.8, y - r * 1.15, x + r * 0.8, y + r * 1.15), fill=L.col(core.FOAM, 0.95 * fade))
        # 粒の後ろの小さな尾（跳ねた向きの逆へ）
        tx, ty = x - vx * u * 0.35, y - (vy + 1.55 * T) * u * 0.35
        rt = r * 0.5
        d.ellipse(L.box(tx - rt, ty - rt, tx + rt, ty + rt), fill=L.col(core.FOAM, 0.6 * fade))
    # まんなかの冠（立ち上がってすぐ沈む白い山）
    hgt = u * 2.6 * np.sin(np.pi * min(1.0, tau * 1.6)) if tau < 0.62 else 0.0
    if hgt > 0.5:
        wd = u * 1.35                                    # 3本の角の冠（なめらかな山にすると噴水のアイコンに見えるため）
        d.polygon(L.p([(px - wd, py), (px - wd * 0.78, py - hgt * 0.62), (px - wd * 0.42, py - hgt * 0.30),
                       (px - wd * 0.12, py - hgt), (px + wd * 0.18, py - hgt * 0.34), (px + wd * 0.52, py - hgt * 0.80),
                       (px + wd * 0.70, py - hgt * 0.26), (px + wd, py)]), fill=L.col(core.FOAM, 0.9 * fade))
    # 水面の小さな泡（つぶれた楕円。広がらない）
    fw, fh = u * (1.5 + 0.5 * tau), u * 0.42
    d.ellipse(L.box(px - fw, py - fh, px + fw, py + fh), fill=L.col(core.FOAM, 0.75 * fade))
    return L.done(), x0, y0


def _streak(cx, cy, i):
    """落ちてくる白い筋（悪役）。東北東の上から着水点へ。"""
    tau = (i + 1) / float(FALL[1] - FALL[0] + 1)          # 0.25..0.75
    dirv = np.array([0.42, -0.91])                        # 来る方向（画面の右上）
    head = np.array([cx, cy]) + dirv * 110 * (1 - tau)
    tail = head + dirv * 64
    L = G.Lines()
    pts = np.linspace(tail, head, 12)
    for j in range(11):
        L.poly(pts[j:j + 2], 2.6, 255 * 0.9 * (j + 1) / 11)
    L.dot(head[0], head[1], 3.6, 255)
    return L.done()


def render(beat, ctx):
    n = beat['n']
    assert n == 36, n
    S = b11._setup()
    P = B.setup()['dest']
    for i, f in enumerate(b11.render(beat, ctx)):
        if i < SPL[1]:
            cam = S['cam'](i)
            p = cam.proj(P[None])[0]
            z = 1000.0 / cam.h * (cam.f / 1000.0)
            k = float(np.clip(z ** 0.35, 1.0, 1.6))
            if FALL[0] <= i < FALL[1]:
                m = _streak(p[0], p[1], i)[..., None]
                f = f * (1 - m) + core.FOAM * m
            if SPL[0] <= i < SPL[1]:
                rgba, x0, y0 = _spray(p[0], p[1], i, k)
                f = U.paste(f, rgba, x0, y0)
        yield np.clip(f, 0, 1).astype(np.float32)
