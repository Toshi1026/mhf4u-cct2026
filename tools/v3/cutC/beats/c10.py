"""案C 10（約 3.4 秒・声なし・テロップなし）：宇宙から地上へ一気に → 仁淀川の河口の前の海へ着水 → 実写の海。
悪役の悲鳴（exports/v3/vo_C/scream.wav）は 10 の頭から 0.1 秒で始まり、落ちるにつれて大きく（音は audio_c.py）。

  0–15 ：C09 の終わりから、カメラが真下を向く（北が上）。宇宙の地球 → 日本列島。速い振りは向きのぶれ。
  16–65：狙いの点（T の着水点＝MAZE の前の海、画面 (520, 822)）へまっすぐ降りる（ズームのぶれ）。日本列島 → 四国 → 土佐湾。
         拡大率 0.30→0.55 で球の写し方から T の地図の写し方へ寄せ、65 で T の 0 フレーム目の 1 フレーム手前（同じ速さ）。
         悪役は狙いの点の右上でジタバタ（風の線が右上へ流れる）→ 46 から体を伸ばして小さくなり、着水点へ。
  66–101（最後の 36 フレーム）：案B の T（cutB/beats/bT.py）をそのまま使う：落ちてくる白い筋 → しぶき → 地図から b11 の線へ →
         河口へ降りて IMG_9662 の 1 フレーム目に画素単位で重なる。最後の 18 フレームは案A の b11 と同じ。
         違いは 1 つだけ：T は 04 の暗い地図から入るので最初の 7 フレームで暗い所から明るくなるが、C は明るい地図から
         続くのでその暗くする分を外した（bT._map_frame から dim の 2 行を除いた _map_frame_c）。ほかの手順は bT.render と同じ。
  CG ラベルは preview.label が最後の 8 フレームで消す。出典（国土地理院）は地図が見えているあいだ右下に（T と同じ位置）。
  見せ場の前後で明るさは 1 フレーム 0.1 以下の変化（白い光・点滅なし）。
"""
import math

import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W, H
from beats import _mapui as U, _geo as G
from cutB.beats import bT, _maplook as ML, _render as RR, _motion as M
from cutC.beats import _space as SPC, _globe as GL, _vill as V, _fall as FA

NB = 36                          # T の長さ
NZ_SPEC = 66                     # ズームの長さ（T の 0 フレーム目＝k 66）


# ---------------------------------------------------------------- ぶれ
def dir_blur(f, vx, vy, k=0.5):
    """(vx, vy) px／フレームの動きの、シャッター 180° の向きのぶれ（線のカーネル）。"""
    L = math.hypot(vx, vy) * k
    if L < 1.5:
        return f
    n = int(math.ceil(L)) | 1
    ker = np.zeros((n, n), np.float32)
    c = n // 2
    ux, uy = vx / math.hypot(vx, vy), vy / math.hypot(vx, vy)
    for s in np.linspace(-L / 2, L / 2, 2 * n):
        x, y = c + ux * s, c + uy * s
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < n and 0 <= iy < n:
            ker[iy, ix] += 1
    ker /= ker.sum()
    return cv2.filter2D(f, -1, ker, borderType=cv2.BORDER_REFLECT)


def cam_flow(k):
    """カメラの見かけの動き：狙いの点の画面の速さ (vx, vy) と、拡大の速さ（log、1 フレーム）。"""
    p0, p1 = SPC.target_screen(k - 0.5), SPC.target_screen(k + 0.5)
    z0, z1 = SPC.zl_of(k - 0.5), SPC.zl_of(k + 0.5)
    return (p1 - p0), math.log(z1 / z0)


# ---------------------------------------------------------------- 悪役
def put_villain(f, k, prev=None):
    st = FA.state(k)
    st1 = FA.state(k - 1)
    vx, vy = st['x'] - st1['x'], st['y'] - st1['y']
    if st['scale'] >= 0.55:
        r, x0, y0 = V.rgba(st['x'], st['y'], st['scale'], st['ang'], st['pc'], st['t'])
        if math.hypot(vx, vy) > 2:
            r = M.smear(r, vx * 0.5, vy * 0.5)
        a = 1.0 if st['scale'] >= 0.8 else (st['scale'] - 0.55) / 0.25
        f = M.put_rgba(f, r, x0, y0, alpha=a)
    if st['scale'] < 0.8:
        # 小さくなったら T と同じ白い筋（頭が明るいカプセル＋尾）
        a = 1.0 if st['scale'] < 0.55 else (0.8 - st['scale']) / 0.25
        rgba, x0, y0 = _streak_at(st['x'], st['y'], a)
        f = U.paste(f, rgba, x0, y0)
    return f, st


def _streak_at(hx, hy, alpha):
    """bT._streak_at と同じ形（太さ 4px のカプセルの頭＋ 12px の尾）を、頭の位置 (hx, hy) に。"""
    head = np.array([hx, hy])
    tail = head + FA.DIRV * 12.0
    pad = 8
    x0, y0 = int(np.floor(min(head[0], tail[0]))) - pad, int(np.floor(min(head[1], tail[1]))) - pad
    w, h = int(np.ceil(max(head[0], tail[0]))) + pad - x0, int(np.ceil(max(head[1], tail[1]))) + pad - y0
    L = core.SS(s=4, w=w, h=h)
    d = L.draw
    nrm = np.array([-FA.DIRV[1], FA.DIRV[0]])
    m = 12
    for j in range(m):
        q0, q1 = j / float(m), (j + 1) / float(m)
        p0, p1 = tail + (head - tail) * q0, tail + (head - tail) * q1
        w0, w1 = 0.3 + 1.6 * q0 ** 1.3, 0.3 + 1.6 * q1 ** 1.3
        pts = [p0 + nrm * w0 - (x0, y0), p1 + nrm * w1 - (x0, y0), p1 - nrm * w1 - (x0, y0), p0 - nrm * w0 - (x0, y0)]
        d.polygon(L.p([tuple(v) for v in pts]), fill=L.col(core.FOAM, (0.15 + 0.75 * q1) * alpha))
    lx, ly = head - (x0, y0)
    d.ellipse(L.box(lx - 1.95, ly - 1.95, lx + 1.95, ly + 1.95), fill=L.col(core.FOAM, alpha))
    return L.done(), x0, y0


def wind_amount(k):
    return float(RR.smoothstep(4.0, 18.0, k)) * (1 - float(RR.smoothstep(52.0, 61.0, k)))


# ---------------------------------------------------------------- ズームの 1 枚
def zoom_frame(k):
    rgb, info = SPC.earth(k)
    (vx, vy), dz = cam_flow(k)
    if k < SPC.K_NADIR + 2:
        rgb = dir_blur(rgb, vx, vy, 0.5)
    if abs(dz) > 0.004:
        p = SPC.target_screen(k)
        rgb = core.zoom_blur(rgb, min(abs(dz) * 0.5, 0.06), cx=float(p[0]), cy=float(p[1]), steps=8)
    f = SPC.map_post(rgb, k, info, 700 + k)
    st = FA.state(k)
    f = V.wind_lines(f, st['x'], st['y'], k, FA.DIRV, amount=wind_amount(k) * 0.9, n=18, spread=240.0, speed=80.0)
    f, st = put_villain(f, k)
    f = ML.credit(f, 1.0)
    return np.clip(f, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- T（bT.render と同じ手順。最初の暗さだけ外す）
def _map_frame_c(i, cam, bg):
    """bT._map_frame から「04 の最後の暗さから明るくする」2 行を除いたもの。"""
    rgb, wa, valid = ML.render_cam(cam, i)
    f = ML.sea_and_air(rgb, wa, i, haze=0.16 * min(1.0, cam.phi / 90.0), tilt=1.2)
    v = valid[..., None]
    f = f * v + bg * (1 - v)
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
    m = Lc.done()
    f += (np.asarray(core.FOAM, np.float32) - f) * m[..., None]
    f = ML.label_scrims(f, 1.0, cg=True, credit_a=1.0)
    return RR.finish_ad(f, 643 + i, grain=0.007, vignette=0.10, bloom_k=0.05), wa * valid


def t_frames():
    """bT.render と同じ 36 枚（_map_frame だけ _map_frame_c）。"""
    l9 = None
    S, S2 = bT._setup()
    for i in range(NB):
        if i >= bT.HAND:
            f, _ = bT._b11_frame(i, S)
            yield np.clip(f, 0, 1).astype(np.float32)
            continue
        parts = bT._b11_parts(i, S2)
        cam = parts['cam']
        k = core.ease((i - bT.RELIEF[0]) / float(bT.RELIEF[1] - bT.RELIEF[0]))
        water = None
        if k < 1:
            fm, water = _map_frame_c(i, cam, parts['bg'])
            f = bT._relief_to_bg(fm, parts['bg'], k)
        else:
            f = parts['bg'].astype(np.float32) + np.zeros((core.H, core.W, 3), np.float32)
        wl = core.ease((i - bT.LINES[0]) / float(bT.LINES[1] - bT.LINES[0]))
        if wl > 0 and parts['m'] is not None:
            lm = (np.clip(np.maximum(parts['m'], parts['c']), 0, 1) * wl)[..., None]
            f = f * (1 - lm) + core.FOAM * lm
        rm = bT._river(i, S2)
        if rm is not None:
            lr = rm[..., None]
            f = f * (1 - lr) + bT.RIVER * lr
        if i == bT.RELIEF[0]:
            l9 = bT._lum(f)
        elif bT.RELIEF[0] < i < bT.HAND and l9 is not None:
            tgt = l9 + (bT._l18(S) - l9) * (i - bT.RELIEF[0]) / float(bT.HAND - bT.RELIEF[0])
            f = f * np.float32(tgt / max(bT._lum(f), 1e-4))
        f = ML.credit(f, 1 - k)
        if i < bT.SPL[1]:
            p, sea, along = bT._splash_point(cam)
            z = cam.f / cam.h
            kk = float(np.clip(z ** 0.35, 1.0, 1.6))
            if bT.FALL[0] <= i < bT.FALL[1]:
                f = bT._streak(f, p[0], p[1], i)
            if bT.SPL[0] <= i < bT.SPL[1]:
                foam, air, x0, y0 = bT._splash(p[0], p[1], i, kk, cam.phi, sea, along, water)
                f = U.paste(f, foam, x0, y0)
                f = bT._paste_shadowed(f, air, x0, y0)
        yield np.clip(f, 0, 1).astype(np.float32)


def render(beat, ctx):
    n = beat['n']
    nz = n - NB
    assert nz == NZ_SPEC, ('C10 のズームは 66 フレームの前提（timeline の 10 は 3.40 秒）', n)
    for k in range(nz):
        yield zoom_frame(k)
    yield from t_frames()
