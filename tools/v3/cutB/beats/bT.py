"""案B T（ナレーションなし・36フレーム＝1.2秒）着水 → CG から実写へ。

案A の b11 をそのまま使う（土佐湾の線画 → 仁淀川を河口（2026-11-07 に本当の河口 33.4593N 133.4815E へ訂正）まで → 河口へ降りて、IMG_9662 の1フレーム目に
画素単位で重なる）。その上に、最初の約 12 フレームだけ小さな白いしぶきを重ねる：
 0–3 ：悪役（の点）が東北東の上から短い白い筋になって落ちてくる（03 の放物線と同じ向き）。
 3–13：MAZE の前の海の点（03 の放物線の終点と同じ地図の点。本当の仁淀川の河口の約 0.45km 西）に、上向きの扇に跳ねる数粒のしぶきと小さな泡。
       輪（同心円）や光は使わない。しぶきの大きさは画面の px で決め、寄りに合わせて少しだけ大きくなる。
最後のフレームは b11 と同じ（08 は案A b12 の 0 フレーム目と同じ IMG_9662 から始められる）。
2026-11-07 動きの仕様（P2）：筋はサブフレームのぶれをかけたカプセル（先が太く明るい）。しぶきの粒はそのときの速さの向きに
のびた楕円（長さ ∝ 速さ×0.5 フレーム）。2 フレーム遅れて、0.6 倍の速さの小さな粒 5 つの 2 波目。冠は 3 本の角のまま。
2026-11-08 絵の作り直し（美術監督の look-dev v2）：b11 の線画の地図の代わりに、03・04 と同じ立体の地図（_maplook、
  b11 のカメラで射影）を見せ、カメラが起き上がるあいだ（11→18 フレーム）に b11 の絵へ溶かす。b11 の中身（_lines・
  _background）を同じ手順で合成し直すので、18 フレーム目からは b11 と画素単位で同じ（最後のフレームは b12 の 1 フレーム目）。
  地図を見せているあいだは、沖の等距離線（海岸に平行な線）は出さず、川は強調色ではなく #6FB8E8 で描く。
  最初の 7 フレームは 04 の最後の暗さから明るくなる（b11 の線画の出と同じ）。しぶき・筋には海に落ちるやわらかい影。
"""
import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from beats import b11
from beats import _mapui as U, _geo as G
from cutB.beats import _bmap as B, _motion as M, _maplook as ML, _render as RR

FALL = (0, 3)                 # 筋が落ちてくる
SPL = (3, 13)                 # しぶき
RNG = np.random.default_rng(7)
# しぶきの粒：角度（度、画面の上が −90）、速さ、大きさ
DROPS2 = [(-140, 0.6, 0.7), (-112, 0.6, 0.75), (-90, 0.6, 0.8), (-70, 0.6, 0.75), (-44, 0.6, 0.7)]   # 2 波目
WAVE2 = 2                     # 2 波目の遅れ（フレーム）
DROPS = [(-150, 1.00, 1.0), (-128, 1.25, 1.2), (-108, 1.45, 1.3), (-92, 1.60, 1.4), (-74, 1.40, 1.2),
         (-56, 1.20, 1.1), (-34, 0.95, 0.9), (-116, 0.85, 0.8), (-66, 0.80, 0.8)]


def _drop(d, L, x, y, vx, vy, r, a):
    """速さ (vx, vy)（px/フレーム）の向きにのびた粒（楕円の多角形）。長さ ∝ 速さ×0.5 フレーム。"""
    sp = float(np.hypot(vx, vy))
    ux, uy = (vx / sp, vy / sp) if sp > 1e-3 else (0.0, -1.0)
    la = r * 0.95 + sp * 0.25                            # 長い半径（のびた分は前後に半分ずつ）
    lb = r * 0.8 * (1 - 0.25 * min(1.0, sp / 20.0))      # 速いほど少し細く
    th = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    ca, cb = np.cos(th) * la, np.sin(th) * lb
    pts = [(x + ca[k] * ux - cb[k] * uy, y + ca[k] * uy + cb[k] * ux) for k in range(len(th))]
    d.polygon(L.p(pts), fill=L.col(core.FOAM, a))


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
    dT = 3.2 / float(SPL[1] - SPL[0])                    # T の 1 フレームあたりの進み
    for drops, delay in ((DROPS, 0), (DROPS2, WAVE2)):
        tw = (i - SPL[0] - delay) / float(SPL[1] - SPL[0])
        if tw <= 0:
            continue
        T = tw * 3.2
        for ang, sp, sz in drops:
            a = np.radians(ang)
            vx, vy = np.cos(a) * sp * 2.6, np.sin(a) * sp * 2.6
            x = px + u * (vx * T)
            y = py + u * (vy * T + 0.5 * 1.55 * T * T)
            r = u * 0.42 * sz * (1 - 0.55 * tw)
            _drop(d, L, x, y, u * vx * dT, u * (vy + 1.55 * T) * dT, r, 0.95 * fade * (0.85 if delay else 1.0))
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


DIRV = np.array([0.42, -0.91])                           # 筋が来る方向（画面の右上）


def _streak_at(cx, cy, t):
    """フレーム t（小数）の筋の板 (rgba, x0, y0)：先が太く明るいカプセル（尾へ細く・うすく）。"""
    tau = (t + 1) / float(FALL[1] - FALL[0] + 1)
    head = np.array([cx, cy]) + DIRV * 110 * (1 - tau)
    tail = head + DIRV * 64
    pad = 8
    xs, ys = (head[0], tail[0]), (head[1], tail[1])
    x0, y0 = int(np.floor(min(xs))) - pad, int(np.floor(min(ys))) - pad
    w, h = int(np.ceil(max(xs))) + pad - x0, int(np.ceil(max(ys))) + pad - y0
    L = core.SS(s=4, w=w, h=h)
    d = L.draw
    nrm = np.array([-DIRV[1], DIRV[0]])
    m = 10
    for j in range(m):
        q0, q1 = j / float(m), (j + 1) / float(m)            # 0＝尾、1＝先
        p0, p1 = tail + (head - tail) * q0, tail + (head - tail) * q1
        w0, w1 = 0.35 + 1.45 * q0, 0.35 + 1.45 * q1         # 半分の太さ（px）
        pts = [p0 + nrm * w0 - (x0, y0), p1 + nrm * w1 - (x0, y0), p1 - nrm * w1 - (x0, y0), p0 - nrm * w0 - (x0, y0)]
        d.polygon(L.p([tuple(v) for v in pts]), fill=L.col(core.FOAM, 0.9 * (0.15 + 0.85 * q1)))
    hx, hy = head - (x0, y0)
    d.ellipse(L.box(hx - 2.6, hy - 2.6, hx + 2.6, hy + 2.6), fill=L.col(core.FOAM, 1.0))
    return L.done(), x0, y0


def _streak(f, cx, cy, i):
    """筋をサブフレームのぶれ（シャッター 180°、速さ 27px/フレーム → 5 点）で描く。"""
    rgba, x0, y0 = M.subframe(lambda t: _streak_at(cx, cy, t), i, shutter=0.5, vel_px=110 / 4.0 * 1.1)
    return U.paste(f, rgba, x0, y0)


MAP_W = (11, 18)               # 立体の地図 → b11 の絵へ溶かす区間（フレーム）
RIVER = core.c('6FB8E8')
DIMC = core.c('0A1D33')


def map_weight(i):
    return 1 - core.ease((i - MAP_W[0]) / float(MAP_W[1] - MAP_W[0]))


def _b11_frame(i, S):
    """b11.render の i フレーム目と同じ手順（同じ結果）。地図の合成に使う部品も返す。"""
    if 12 <= i <= 17 or 28 <= i <= 32:
        subs = tuple(i + d for d in (-0.4, -0.2, 0.0, 0.2, 0.4))
    elif 2 <= i <= 27 and not 18 <= i <= 25 or i == 33:
        subs = (i - 0.33, i, i + 0.33)
    else:
        subs = (i,)
    acc = None
    for t in subs:
        m, c, r, cam = b11._lines(t, S)
        acc = [m, c, r] if acc is None else [a + b for a, b in zip(acc, (m, c, r))]
    m, c, r = [a / len(subs) for a in acc]
    if 8 <= i <= 21:
        r = sum(b11._lines(i + d, S, river_only=True) for d in np.linspace(-0.45, 0.45, 9)) / 9
    cam = S['cam'](i)
    if 18 <= i <= 25:
        v = cam.f * abs(np.radians(S['cam'](i + 0.5).beta - S['cam'](i - 0.5).beta))
        v = min(v, 360.0)
        m, c, r = (core.whip(x, v * 0.8) for x in (m, c, r))
    bg = b11._background(i, cam, S)
    lm = np.clip(np.maximum(m, c), 0, 1)[..., None]
    f = bg * (1 - lm) + core.FOAM * lm
    lr = r[..., None]
    f = f * (1 - lr) + core.ACC * lr
    return f, dict(m=m, c=c, r=r, cam=cam, bg=bg)


def _map_frame(i, parts):
    """立体の地図（b11 のカメラ）＋海岸線＋手前の線（水平線など）＋川（#6FB8E8）。仕上げまで。"""
    cam = parts['cam']
    rgb, wa, valid = ML.render_cam(cam, i)
    f = ML.sea_and_air(rgb, wa, i, haze=0.16 * min(1.0, cam.phi / 90.0), tilt=1.2)
    v = valid[..., None]
    f = f * v + parts['bg'] * (1 - v)
    z = cam.f / cam.h
    lw = ML._levels_for(z)
    lv = ML.level(max(lw, key=lambda a: a[1])[0])
    Lc = G.Lines()
    for name, a in (('sea', 0.40), ('inland', 0.20)):
        for P in lv['coast'][name]:
            Q = cam.proj(P)
            ok = np.isfinite(Q).all(axis=1)
            if not ok.any() or max(np.ptp(Q[ok, 0]), np.ptp(Q[ok, 1])) < ML.MIN_LOOP:
                continue
            Lc.poly(Q, 1.25, 255 * a, closed=False)
    m = np.maximum(Lc.done(), parts['c'])
    f += (np.asarray(core.FOAM, np.float32) - f) * m[..., None]
    lr = parts['r'][..., None]
    f = f * (1 - lr) + RIVER * lr
    # 04 の最後の暗さから（b11 の線画が濃くなるのと同じ 0→7 フレーム）
    vis = 0.42 + 0.58 * core.ease(i / 7.0)
    f = f * vis + DIMC * (1 - vis)
    f = ML.label_scrims(f, 1.0, cg=True, credit_a=1.0)
    return RR.finish_ad(f, 643 + i, grain=0.007, vignette=0.10, bloom_k=0.05)


def _paste_shadowed(f, rgba, x0, y0, sh_a=0.30):
    """白い粒の板を、海に落ちるやわらかい影（右下）つきで貼る。"""
    h, w = rgba.shape[:2]
    a = rgba[..., 3]
    pad = 8
    sh = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
    sh[pad:pad + h, pad:pad + w] = a
    sh = cv2.GaussianBlur(cv2.warpAffine(sh, np.float32([[1, 0, 2], [0, 1, 3]]), (w + 2 * pad, h + 2 * pad)), (0, 0), 1.6)
    X0, Y0 = x0 - pad, y0 - pad
    fx0, fy0, fx1, fy1 = max(X0, 0), max(Y0, 0), min(X0 + w + 2 * pad, core.W), min(Y0 + h + 2 * pad, core.H)
    if fx1 > fx0 and fy1 > fy0:
        sub = f[fy0:fy1, fx0:fx1]
        k = sh[fy0 - Y0:fy1 - Y0, fx0 - X0:fx1 - X0][..., None] * sh_a
        sub *= 1 - k * (1 - np.asarray(core.c('1B3550'), np.float32))
    # 粒は真っ白にせず、わずかに青い白（上が明るい）
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    col = rgba[..., :3] * (1.0 - 0.10 * yy) * np.asarray(core.c('EEF6FC'), np.float32) / np.asarray(core.FOAM, np.float32)
    return U.paste(f, np.concatenate([col, rgba[..., 3:4]], -1), x0, y0)


def render(beat, ctx):
    n = beat['n']
    assert n == 36, n
    S = b11._setup()
    P = B.setup()['dest']
    for i in range(n):
        f, parts = _b11_frame(i, S)
        w = map_weight(i)
        if w > 0.001:
            fm = _map_frame(i, parts)
            f = fm * w + f * (1 - w)
            f = ML.credit(f, w)
        if i < SPL[1]:
            cam = S['cam'](i)
            p = cam.proj(P[None])[0]
            z = 1000.0 / cam.h * (cam.f / 1000.0)
            k = float(np.clip(z ** 0.35, 1.0, 1.6))
            if FALL[0] <= i < FALL[1]:
                f = _streak(f, p[0], p[1], i)
            if SPL[0] <= i < SPL[1]:
                rgba, x0, y0 = _spray(p[0], p[1], i, k)
                f = _paste_shadowed(f, rgba, x0, y0)
        yield np.clip(f, 0, 1).astype(np.float32)
