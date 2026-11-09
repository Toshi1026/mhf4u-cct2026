"""look-dev 07：06 と同じジオラマ（ldx）で、落ちて着水する所。しぶきは b07 の粒（動きはそのまま）に、
水面の白い泡のふくらみ（横に低く広がる。壁にしない・輪にしない）と、水の中へ入った跡の細かい泡を足す。
python3 cutB/lookdev/ld07.py [frames...]"""
import math
import sys

from _common import np, cv2, core, tick, report, save_pair, W, H  # noqa: F401
from cutB.beats import _render as R, _figures3d as F3, _art as A, _xsec as X, b06, b07, _motion as M
from beats import _mapui as U, _tosa, b10
from core import WL
import timeline_b as timeline
import ldx
import ld06  # noqa: F401  （悪役の差し替え）


def foam_and_bubbles(f, i, f_land):
    """着水点（X.X1, WL）の泡：水面の上に低い白いふくらみ（数フレームで広がって消える）、水の中へ泡の列。"""
    k = i - f_land
    if k < 0 or k > 14:
        return f
    x0, y0 = X.X1, WL
    pad = 90
    xa, ya = int(x0 - pad), int(y0 - 40)
    w, h = 2 * pad, 40 + 150
    yy, xx = np.mgrid[ya:ya + h, xa:xa + w].astype(np.float32)
    g = M.ease((k + 1) / 5.0, 'expo_out')
    fade = 1 - M.ease((k - 2) / 12.0, 'sine_inout')
    # 水面の上：横に長い低い泡（半楕円、縁は fBm でけずる）
    rx, ry = 16 + 34 * g, 5 + 9 * g * (1 - 0.4 * M.ease(k / 10.0))
    d = ((xx - x0) / rx) ** 2 + ((yy - y0) / ry) ** 2
    n = R.fbm(xx / 7.0, yy / 7.0 + k * 0.4, 3, seed=51)
    foam = R.smoothstep(1.1, 0.6, d + n * 0.35) * (yy <= y0 + 1)
    # 水の中：泡の列（着水点から下へ。上るにつれて小さく）
    rng = np.random.default_rng(7)
    bub = np.zeros_like(foam)
    for j in range(26):
        by = y0 + 8 + rng.uniform(0, 120) * g - k * 2.2 * rng.uniform(0.5, 1.0)
        bx = x0 + rng.normal(0, 6 + 3 * k) + 3 * math.sin(k * 0.8 + j)
        br = rng.uniform(1.2, 3.6) * (1 - 0.03 * k)
        if by < y0 + 3:
            continue
        dd = np.sqrt((xx - bx) ** 2 + (yy - by) ** 2)
        ring = np.exp(-((dd - br) / 0.8) ** 2) * 0.8 + R.smoothstep(br, 0, dd) * 0.25
        bub = np.maximum(bub, ring)
    reg = f[ya:ya + h, xa:xa + w]
    a = (np.clip(foam, 0, 1) * 0.92 * fade)[..., None]
    reg = reg * (1 - a) + core.FOAM * a
    b = (bub * 0.55 * fade)[..., None]
    reg = reg + (core.c('D8EEFA') - reg) * b
    f[ya:ya + h, xa:xa + w] = reg
    return f


def frame_at(i, beat):
    n = beat['n']
    mo = beat['moras']
    f_so, f_go = timeline.mora_frame(beat, 0), timeline.mora_frame(beat, 7)
    b6 = next(b for b in timeline.load() if b['key'] == '06')
    T6 = b06.timing(b6)
    n6 = b6['n']
    _ctr6, t_sub, _ = b06.texts()
    ctr = M.Odometer('45', W / 2, 450, digit_size=110, side_size=110, prefix='', suffix='秒後', gap=0.02,
                     suffix_color=core.ACC, lock_scale=1.10, lock_drop=5.0, lock_dur=6, blur_k=0.6)
    nav = b10._background()
    lines = _tosa.contours()
    f_land = min(f_go + b07.LAND_AFTER_GO, n - b07.DARKEN - 2)
    out = n - (f_land + 1)
    M_end = b06.cam_M(n6 - 1, T6)
    splash = M.Burst(23, 5, (X.X1, WL - 2), angle_range=(-124, -56), speed_range=(8.0, 13.0), gravity=1.3,
                     drag=0.04, life=8, size=(2.8, 4.2), f0=f_land, alpha=0.9, stretch=0.5, life_jitter=0.2)
    f_exit = f_land + b07.LOCK_HOLD
    gi = n6 + i
    fade6 = 1.0 - M.ease((i + 1) / 9.0, 'sine_inout')
    with tick('scene'):
        f = ld06.scene(gi, T6, fuji_a=fade6, fuji=(1.0, 1.0))
    with tick('lines+figure'):
        u, tau = b07.fall_u(i, f_land)
        rgba, x0, y0 = X.trail_layer(u, ghost=1.0 - core.ease(i / 30.0) * 0.4)
        f = U.paste(f, rgba, x0, y0)
        f = b06.apex_lines(f, 1.0 - 0.4 * core.ease(i / 12.0))
        f = b06.sight_line(f, 0.5 * (1 - core.ease(i / 8.0)), phase=-0.8 * gi)
        va = 1 - core.ease((i - f_land + 1) / 3.0)
        if va > 0.003:
            f = b07.speed_lines(f, i, f_land)
            g = core.ease((i - 1) / 10.0, 'inout')
            sc = 0.12 + (X.VS - 0.12) * g
            if sc * 34 >= 5:
                ua, _ = b07.fall_u(i - 0.5, f_land)
                ub, _ = b07.fall_u(i + 0.5, f_land)
                sp = float(np.hypot(*(np.array(X.arc(ub)) - np.array(X.arc(ua)))))
                nn = int(np.clip(math.ceil(sp * 0.4 / 3.0), 1, 8)) if sp > 12 else 1
                r = M.subframe(lambda t: b07.villain_draw(t, f_land, n6), i, shutter=0.4, n=nn)
                if r:
                    f = M.put_rgba(f, *r, alpha=va, shadow=0.35, shadow_px=max(1.0, 1.1 * sc))
        f = foam_and_bubbles(f, i, f_land)
        f = splash.draw(f, i)
        f = A.put_glint_fx(f, X.APEX[0], X.APEX[1] - 2, b06.glint_k(gi, T6), r=42)
    with tick('camera'):
        Mc = b07.cam_M(i, M_end)
        f = b06.warp(f, Mc)
    with tick('finish'):
        f = R.finish(f, i, bloom_k=0.10, bloom_thr=0.74, vignette=0.16, grain=0.009, ca_px=0.45, distort=0.4)
    with tick('text'):
        f = b06.put_labels(f, gi, T6, Mc, fuji_a=fade6, apex_a=1.0 - 0.4 * core.ease(i / 12.0), fuji_i=10 ** 6)
        e = M.ease((i + 1) / 4.0, 'cubic_in')
        if e < 0.999:
            f = b06.put_counter(f, gi, T6, alpha=1 - e, extra_dy=-15 * e, blur=4 * e)
        f = t_sub.exit(f, i, -1, dur=4)
        j = i - (f_land + 1)
        jd = i - (n - b07.DARKEN)
        if j >= 0:
            q = (j + 1) / out
            lx, ly = b06.project(Mc, X.X1, WL, clamp=None)
            f = core.zoom_blur(f, 0.24 * q ** 1.2, cx=lx, cy=ly, steps=18)
        if jd >= 0:
            qd = (jd + 1) / b07.DARKEN
            d = 0.6 * qd + 0.4 * core.ease(qd, 'smooth')
            f = f * (1 - d) + nav * d
        if j >= 0:
            la = (0.14 * core.ease(q, 'in') * lines)[..., None]
            f = f * (1 - la) + core.FOAM * la
        if i >= f_so - 1:
            a = M.ease((i - f_so + 2) / 4.0, 'sine_inout')
            dy, bl = 0.0, 0.0
            if i >= f_exit:
                ee = M.ease((i - f_exit + 1) / 6.0, 'cubic_in')
                a *= 1 - ee
                dy, bl = -10 * ee, 5 * ee
            v = b07.SECS * u
            dv = b07.SECS * (b07.fall_u(i + 0.5, f_land)[0] - b07.fall_u(i - 0.5, f_land)[0])
            f = ctr.put(f, v, a, i=i, f_lock=f_land, dv=dv, dy=dy, blur=bl)
    return np.clip(f, 0, 1).astype(np.float32)


def main(frames):
    beat = next(b for b in timeline.load() if b['key'] == '07')
    ldx.static()
    report('07 static')
    for i in frames:
        out = frame_at(i, beat)
        report('07 f%d' % i)
        save_pair('07', i, out)


if __name__ == '__main__':
    main([int(v) for v in sys.argv[1:]] or [36])
