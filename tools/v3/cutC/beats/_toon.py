"""案C 01・02 の「アニメ版」の部品（平面的なマンガの絵柄）。ここだけ太い輪郭線を使ってよい（台本 C01）。

色：明るい平面の色（空の青・ミントの草・白）。オレンジ・赤・黄・金は使わない。輪郭は濃紺 OUT。
  bg_static()                      空・遠い山・丘・草原（動かない板、sRGB）
  clouds()                         マンガの雲の板 [(rgba, x, y)]（ゆっくり流す）
  villain_patch(...)               顔のない白いピクトグラムの悪役（マントつき）を輪郭線＋セルの影で → (rgba, x0, y0)
  fist_patch(...)                  拳（ヒーローは拳だけ）を輪郭線で → (rgba, x0, y0)
  burst / mini_star / focus_lines  当たりの「ポコッ」の星形・小さな星・集中線
  outlined_text(...)               縁取りの太い文字（「アニメだと」「ポコッ」）
"""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

import core
from core import W, H
from cutB.beats import _art as A

OUT = core.c('0E2F57')            # 輪郭（濃紺）
SKY_T = core.c('1C86E0')          # 空の上
SKY_B = core.c('A4E0FB')          # 地平の近く
CLOUD = core.c('FFFFFF')
CLOUD_SH = core.c('CFE9FA')
MTN = core.c('8DC9EC')            # 遠い山（空気で青い）
MTN2 = core.c('6FB8E6')
HILL = core.c('4ACB9A')           # 丘（ミント）
HILL_SH = core.c('37B487')
GRASS_T = core.c('47CE92')        # 草原（地平側）
GRASS_B = core.c('26A874')        # 草原（手前）
TUFT = core.c('1E8E62')
TUFT_HI = core.c('86E6B6')
BODY = core.c('FFFFFF')
BODY_SH = core.c('C9E0F4')        # 体のセルの影（淡い青）
CAPE = core.c('5FB8F5')           # マント（青）
CAPE_SH = core.c('3E97DD')
FIST = core.c('FAFDFF')
FIST_SH = core.c('CFE4F6')
SLEEVE = core.c('2F72D2')         # 袖口（青）
BURST_IN = core.c('BDEBFF')

HZ = 1062                          # 地平線
_C = {}


# ---------------------------------------------------------------- 小道具
def _ss(w, h, s=3):
    return core.SS(s=s, w=int(math.ceil(w)), h=int(math.ceil(h)))


def col(rgb, a=1.0):
    return tuple(int(round(v * 255)) for v in rgb) + (int(round(a * 255)),)


def shift_mask(a, dx, dy):
    """a(x+dx, y+dy)（範囲の外は 0）。"""
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def cel(rgba, mask, d, shade_rgb, k=1.0):
    """mask（体の塗りのアルファ）の右下の縁に幅 d のセルの影（光は左上）。rgba を書き換えて返す。"""
    if d < 0.6:
        return rgba
    sh = mask * (1 - shift_mask(mask, d, d * 0.9))
    sh = np.clip(sh * k, 0, 1)[..., None]
    rgba[..., :3] = rgba[..., :3] * (1 - sh) + shade_rgb * sh
    return rgba


def paste(frame, rgba, x0, y0, alpha=1.0):
    """ストレートアルファの rgba を frame の (x0, y0) に（はみ出しは切る）。"""
    if alpha <= 0.002:
        return frame
    h, w = rgba.shape[:2]
    x0, y0 = int(round(x0)), int(round(y0))
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, frame.shape[1]), min(y0 + h, frame.shape[0])
    if xb <= xa or yb <= ya:
        return frame
    sub = rgba[ya - y0:yb - y0, xa - x0:xb - x0]
    a = sub[..., 3:4] * alpha
    frame[ya:yb, xa:xb] = frame[ya:yb, xa:xb] * (1 - a) + sub[..., :3] * a
    return frame


def paste_f(frame, rgba, x0, y0, alpha=1.0):
    """小数の位置に貼る（ゆっくり動く物がガタつかない）。"""
    ix, iy = int(math.floor(x0)), int(math.floor(y0))
    fx, fy = x0 - ix, y0 - iy
    if abs(fx) > 1e-3 or abs(fy) > 1e-3:
        h, w = rgba.shape[:2]
        prem = np.dstack([rgba[..., :3] * rgba[..., 3:], rgba[..., 3:]])
        prem = cv2.warpAffine(prem, np.float32([[1, 0, fx], [0, 1, fy]]), (w + 1, h + 1), flags=cv2.INTER_LINEAR)
        a = prem[..., 3:]
        rgba = np.dstack([prem[..., :3] / np.maximum(a, 1e-5), a])
    return paste(frame, rgba, ix, iy, alpha)


# ---------------------------------------------------------------- 背景
def _cloud_poly(L, cx, yb, w, seed, fill, line, ow, shade=None):
    """マンガの積雲：円の集まり＋平らな底（輪郭 → 塗り）。影の帯は clouds() で塗りのマスクから作る。"""
    rng = np.random.default_rng(seed)
    n = 6
    circ = []
    for k in range(n):
        q = (k + 0.5) / n
        r = w * (0.13 + 0.10 * math.sin(math.pi * q)) * rng.uniform(0.9, 1.1)
        x = cx - w * 0.42 + w * 0.84 * q + rng.uniform(-0.03, 0.03) * w
        y = yb - r * rng.uniform(0.35, 0.65) - (w * 0.10 * math.sin(math.pi * q))
        circ.append((x, y, r))
    d = L.draw
    if line is not None:
        for x, y, r in circ:
            d.ellipse(L.box(x - r - ow, y - r - ow, x + r + ow, y + r + ow), fill=line)
        d.rounded_rectangle(L.box(cx - w * 0.47 - ow, yb - w * 0.12, cx + w * 0.47 + ow, yb + ow),
                            radius=L.w(w * 0.05), fill=line)
    for x, y, r in circ:
        d.ellipse(L.box(x - r, y - r, x + r, y + r), fill=fill)
    d.rounded_rectangle(L.box(cx - w * 0.47, yb - w * 0.12 - ow, cx + w * 0.47, yb), radius=L.w(w * 0.04), fill=fill)


def clouds():
    """[(rgba, x0, y0, 流れる速さ px/フレーム)]。主役の雲は左と右上（見出し・キラーンの所をあける）。"""
    if 'clouds' in _C:
        return _C['clouds']
    spec = [(150.0, 905.0, 300.0, 3, 0.35), (930.0, 250.0, 260.0, 7, 0.28), (690.0, 1000.0, 150.0, 11, 0.18),
            (330.0, 1022.0, 110.0, 13, 0.15), (1010.0, 1010.0, 120.0, 17, 0.16), (-40.0, 300.0, 200.0, 19, 0.30)]
    out = []
    for (cx, yb, w, sd, sp) in spec:
        ow = 5.0 if w > 200 else 3.6
        pw, ph = int(w * 1.15 + 20), int(w * 0.62 + 20)
        L = _ss(pw, ph, 3)
        _cloud_poly(L, pw / 2, ph - 10, w, sd, L.col(CLOUD), L.col(OUT), ow)
        Lm = _ss(pw, ph, 3)
        _cloud_poly(Lm, pw / 2, ph - 10, w, sd, Lm.col(CLOUD), None, ow)
        rgba, m = L.done(), Lm.done()[..., 3]
        d_ = w * 0.07
        band = m * (1 - shift_mask(m, -d_ * 0.25, d_))          # 下と右下の縁の影の帯
        band = np.clip(band, 0, 1)[..., None]
        rgba[..., :3] = rgba[..., :3] * (1 - band) + CLOUD_SH * band
        out.append((rgba, cx - pw / 2, yb - ph + 10, sp))
    _C['clouds'] = out
    return out


def bg_static():
    """空・遠い山・丘・草原（sRGB）。雲は別（clouds）。"""
    if 'bg' in _C:
        return _C['bg']
    f = np.empty((H, W, 3), np.float32)
    t = (np.arange(HZ, dtype=np.float32)[:, None] / HZ) ** 1.25
    f[:HZ] = (SKY_T * (1 - t) + SKY_B * t)[:, None, :]
    # 空のごく淡い斜めの光の帯（アニメの空の「抜け」）
    yy, xx = np.mgrid[0:HZ, 0:W].astype(np.float32)
    band = np.exp(-(((xx * 0.42 + yy) - 820) / 70.0) ** 2) * 0.05 + np.exp(-(((xx * 0.42 + yy) - 1010) / 34.0) ** 2) * 0.03
    f[:HZ] = f[:HZ] + (1 - f[:HZ]) * band[..., None]
    gt = (np.arange(H - HZ, dtype=np.float32)[:, None] / (H - HZ)) ** 0.8
    f[HZ:] = (GRASS_T * (1 - gt) + GRASS_B * gt)[:, None, :]
    L = core.SS(s=3)
    d = L.draw
    xs = np.arange(-20, W + 40, 10, dtype=np.float32)
    # 遠い山（輪郭なし・空気の青）
    m1 = HZ - 70 - 52 * np.sin(xs / 190.0 + 0.6) ** 2 - 18 * np.sin(xs / 61.0 + 1.3)
    d.polygon(L.p([(x, y) for x, y in zip(xs, m1)] + [(W + 40, HZ + 4), (-20, HZ + 4)]), fill=L.col(MTN))
    m2 = HZ - 26 - 30 * np.sin(xs / 140.0 + 2.2) ** 2
    d.polygon(L.p([(x, y) for x, y in zip(xs, m2)] + [(W + 40, HZ + 4), (-20, HZ + 4)]), fill=L.col(MTN2))
    # 丘（ミント・細い輪郭）
    h1 = HZ - 8 - 22 * np.sin(xs / 230.0 + 0.2) ** 2
    ow = 3.0
    d.line(L.p([(x, y) for x, y in zip(xs, h1)]), fill=L.col(OUT), width=L.w(2 * ow), joint='curve')
    d.polygon(L.p([(x, y) for x, y in zip(xs, h1)] + [(W + 40, HZ + 30), (-20, HZ + 30)]), fill=L.col(HILL))
    lay = L.done()
    f = f * (1 - lay[..., 3:]) + lay[..., :3] * lay[..., 3:]
    # 草原：横の明るい帯（セル）と草のふさ
    L = core.SS(s=3)
    d = L.draw
    for (y0, y1, a) in ((HZ + 60, HZ + 96, 0.10), (HZ + 230, HZ + 300, 0.08), (HZ + 520, HZ + 640, 0.06)):
        d.polygon(L.p([(-10, y0), (W + 10, y0 - 18), (W + 10, y1 - 18), (-10, y1)]), fill=L.col(core.c('FFFFFF'), a))
    rng = np.random.default_rng(5)
    for k in range(95):
        y = HZ + 20 + (H - HZ - 20) * rng.uniform(0, 1) ** 1.6
        x = rng.uniform(-10, W + 10)
        s = 0.5 + 1.8 * (y - HZ) / (H - HZ)
        c_ = TUFT if rng.uniform() < 0.72 else TUFT_HI
        for dx, ln, lean in ((-5, 13, -6), (0, 18, 0), (5, 12, 7)):
            d.line(L.p([(x + dx * s, y), (x + dx * s + lean * s * 0.6, y - ln * s)]), fill=L.col(c_), width=L.w(2.2 * s),
                   joint='curve')
    lay = L.done()
    f = f * (1 - lay[..., 3:]) + lay[..., :3] * lay[..., 3:]
    _C['bg'] = f.astype(np.float32)
    return _C['bg']


def fg_grass(i):
    """手前の左右の角の大きな草（ゆれる）。rgba（画面全体）。"""
    key = ('fg', i % 60)
    if key in _C:
        return _C[key]
    L = core.SS(s=2)
    d = L.draw
    t = i / core.FPS
    for (bx, n, side, seed) in ((40, 7, 1, 1), (1040, 6, -1, 2)):
        rng = np.random.default_rng(seed)
        for k in range(n):
            x = bx + side * rng.uniform(0, 120)
            ln = rng.uniform(150, 260)
            lean = side * rng.uniform(10, 60)
            sw = 6 * math.sin(2 * math.pi * 0.9 * t + k)
            tip = (x + lean + sw, H - ln)
            mid = (x + lean * 0.4 + sw * 0.3, H - ln * 0.55)
            wb = rng.uniform(18, 26)
            poly = [(x - wb / 2, H + 8), (mid[0] - wb * 0.28, mid[1]), tip, (mid[0] + wb * 0.28, mid[1]),
                    (x + wb / 2, H + 8)]
            d.polygon(L.p(poly), fill=L.col(OUT))
            d.line(L.p(poly + [poly[0]]), fill=L.col(OUT), width=L.w(7), joint='curve')
            d.polygon(L.p(poly), fill=L.col(TUFT if k % 2 else core.c('2BB07A')))
    _C[key] = L.done()
    return _C[key]


# ---------------------------------------------------------------- 悪役（輪郭線＋セル）
def _vil_parts(pose, fl):
    """_art.villain と同じ形（ローカル単位：胴のまんなかが原点、+y 下、全長 約 32）。(lines, head, cape)。
    lines：[(pts, 太さ)]。"""
    if pose == 'stand':
        cape = [(-2.4, -8.0), (2.4, -8.0), (8.0 + fl * 0.3, 0.0), (13.0 + fl, 10.5), (9.0 + fl * 0.7, 9.4),
                (5.5 + fl * 0.5, 11.6), (1.5, 10.6), (-3.5, 11.2), (-7.0, 10.0), (-6.0, 1.0)]
        lines = [([(-4.2, 16), (0, 4), (4.2, 16)], 2.9), ([(0, 4.5), (0, -7.5)], 3.6),
                 ([(0, -6.0), (-6.2, -1.2), (-1.4, 2.6)], 2.3), ([(0, -6.0), (6.2, -1.2), (1.4, 2.6)], 2.3)]
    elif pose == 'hit':                                # 当たってのけぞる（手足が後ろへ）
        cape = [(-2.4, -8.0), (2.4, -8.0), (9.0 + fl, -2.0), (16.0 + fl, 6.5), (11.0, 8.4), (6.0, 11.0), (1.0, 10.0),
                (-4.0, 9.0)]
        lines = [([(-1.0, 16), (1.0, 4), (7.5, 14.5)], 2.9), ([(1.0, 4.5), (0, -7.5)], 3.6),
                 ([(0, -6.0), (7.0, -11.0), (11.5, -9.0)], 2.3), ([(0, -6.0), (8.0, -3.0), (12.5, -5.0)], 2.3)]
    else:                                              # 'fly'：手足を広げて飛ばされる
        cape = [(-2.6, -7.6), (2.6, -7.6), (8.0 + fl, 9.0), (6.0 + fl, 18.0), (2.0 + fl * 0.6, 15.5),
                (-2.0 + fl * 0.4, 19.0), (-6.5 + fl * 0.3, 12.0)]
        lines = [([(-7.0, 14.5), (0, 4), (6.5, 15.0)], 2.9), ([(0, 4.5), (0, -7.5)], 3.6),
                 ([(-9.5, -12.5), (0, -6.0), (9.5, -12.0)], 2.3)]
    return lines, ((0.0, -12.2), 3.9), cape


def villain_patch(x, y, scale, ang=0.0, pose='stand', t=0.0, M2=None, cape_ang=None, ow=None, alpha=1.0):
    """(x, y)＝胴のまんなか（画面）。(rgba, x0, y0)。輪郭 ow px（既定 0.62·scale、最小 1.6）。"""
    ow = max(1.6, 0.62 * scale) if ow is None else ow
    fl = math.sin(t * 9.0) * 1.2 + math.sin(t * 15.0 + 1.0) * 0.6
    lines, (hc, hr), cape = _vil_parts(pose, fl)
    Rm = A._R(ang)
    if M2 is not None:
        Rm = np.asarray(M2, float) @ Rm
    ext = 1.0 if M2 is None else max(1.0, float(np.abs(np.linalg.eigvals(np.asarray(M2, float))).max()))
    R = int(math.ceil((26 * scale + ow + 6) * ext))
    s = 3 if scale > 2 else 4
    L = core.SS(s=s, w=2 * R, h=2 * R)
    Lb = core.SS(s=s, w=2 * R, h=2 * R)              # 体だけ（セルの影のマスク）
    cx, cy = R + (x - math.floor(x)), R + (y - math.floor(y))

    def T(pts):
        return [tuple(Rm @ np.array(q, float) * scale + (cx, cy)) for q in pts]

    if cape_ang is not None and pose == 'fly':
        sh = Rm @ np.array([0.0, -7.6]) * scale + (cx, cy)
        Rc = (np.asarray(M2, float) if M2 is not None else np.eye(2)) @ A._R(cape_ang - 90.0)
        cp = [tuple(sh + Rc @ (np.array(q) - (0.0, -7.6)) * scale) for q in cape]
    else:
        cp = T(cape)
    hx, hy = T([hc])[0]
    rr = hr * scale
    d = L.draw
    # 1) マントの輪郭 → 塗り
    d.polygon(L.p(cp), fill=L.col(OUT))
    d.line(L.p(cp + [cp[0]]), fill=L.col(OUT), width=L.w(2 * ow), joint='curve')
    d.polygon(L.p(cp), fill=L.col(CAPE))
    # 2) 体の輪郭（マントの上にも）→ 塗り
    for pts, wd in lines:
        d.line(L.p(T(pts)), fill=L.col(OUT), width=L.w(wd * scale + 2 * ow), joint='curve')
        for q in T([pts[0], pts[-1]]):
            r_ = wd * scale / 2 + ow
            d.ellipse(L.box(q[0] - r_, q[1] - r_, q[0] + r_, q[1] + r_), fill=L.col(OUT))
    d.ellipse(L.box(hx - rr - ow, hy - rr - ow, hx + rr + ow, hy + rr + ow), fill=L.col(OUT))
    for LL in (L, Lb):
        dd = LL.draw
        for pts, wd in lines:
            dd.line(LL.p(T(pts)), fill=LL.col(BODY), width=LL.w(wd * scale), joint='curve')
            for q in T([pts[0], pts[-1]]):
                r_ = wd * scale / 2
                dd.ellipse(LL.box(q[0] - r_, q[1] - r_, q[0] + r_, q[1] + r_), fill=LL.col(BODY))
        dd.ellipse(LL.box(hx - rr, hy - rr, hx + rr, hy + rr), fill=LL.col(BODY))
    rgba = L.done()
    body = Lb.done()[..., 3]
    rgba = cel(rgba, body, max(0.0, 0.85 * scale), BODY_SH)
    # マントのセルの影（体の外のマントだけ）
    if scale > 1.5:
        capem = np.clip(rgba[..., 3] - body, 0, 1) * (np.abs(rgba[..., :3] - CAPE).sum(-1) < 0.08)
        rgba = cel(rgba, capem.astype(np.float32), 0.9 * scale, CAPE_SH, k=0.9)
    if alpha < 1:
        rgba[..., 3] *= alpha
    return rgba, int(math.floor(x)) - R, int(math.floor(y)) - R


# ---------------------------------------------------------------- 拳（輪郭線）
def fist_patch(x, y, ang, scale=1.0, squash=(1.0, 1.0)):
    """拳のまんなか (x, y)、殴る向き ang（度）。squash：(打つ向き, 横) の伸び縮み。(rgba, x0, y0)。
    形は _art._fist_shapes（手の甲・曲げた 4 本の指・親指・袖口）。"""
    sh = A._fist_shapes()
    Rm = A._R(ang) @ np.diag(squash)
    pts = np.vstack(list(sh.values()))
    P = pts @ Rm.T * scale
    pad = 14
    x0, y0 = int(math.floor(x + P[:, 0].min() - pad)), int(math.floor(y + P[:, 1].min() - pad))
    x1, y1 = int(math.ceil(x + P[:, 0].max() + pad)), int(math.ceil(y + P[:, 1].max() + pad))
    L = core.SS(s=3, w=x1 - x0, h=y1 - y0)
    Lm = core.SS(s=3, w=x1 - x0, h=y1 - y0)

    def T(Q, LL):
        Q = np.asarray(Q, float) @ Rm.T * scale + (x - x0, y - y0)
        return LL.p([tuple(q) for q in Q])

    d = L.draw
    lw = 3.6 * scale
    for k, Q in sh.items():
        d.polygon(T(Q, L), fill=L.col(OUT))
        d.line(T(np.vstack([Q, Q[:1]]), L), fill=L.col(OUT), width=L.w(2 * lw), joint='curve')
        d.polygon(T(Q, L), fill=L.col(SLEEVE if k == 'cuff' else FIST))
        if k != 'cuff':
            Lm.draw.polygon(T(Q, Lm), fill=Lm.col(FIST))
    det = L.w(lw * 0.7)
    nail = np.array([(18, 13.5), (27, 14.5), (30, 20), (27, 25.5), (19, 26)], float)
    d.line(T(nail, L), fill=L.col(OUT), width=det, joint='curve')
    d.line(T([(-58, -48), (-58, 50)], L), fill=L.col(OUT), width=det)
    rgba = L.done()
    m = Lm.done()[..., 3]
    m = m * (np.abs(rgba[..., :3] - FIST).sum(-1) < 0.06)
    rgba = cel(rgba, m.astype(np.float32), 9.0 * scale, FIST_SH, k=0.9)
    return rgba, x0, y0


# ---------------------------------------------------------------- 当たりの絵
def burst(r, spikes=11, rot=0.0, ow=5.0, seed=4):
    """「ポコッ」の星形（白・内側に淡い青・濃紺の輪郭）。(rgba, 中心の板の中の位置)。"""
    rng = np.random.default_rng(seed)
    R = int(r * 1.25 + ow + 6)
    L = core.SS(s=3, w=2 * R, h=2 * R)
    d = L.draw
    pts, inner = [], []
    for k in range(2 * spikes):
        th = math.radians(rot) + math.pi * k / spikes
        rr = r * (1.0 + 0.18 * rng.uniform(-1, 1)) if k % 2 == 0 else r * 0.56
        pts.append((R + rr * math.cos(th), R + rr * math.sin(th)))
        ri = rr * 0.55
        inner.append((R + ri * math.cos(th), R + ri * math.sin(th)))
    d.polygon(L.p(pts), fill=L.col(OUT))
    d.line(L.p(pts + [pts[0]]), fill=L.col(OUT), width=L.w(2 * ow), joint='curve')
    d.polygon(L.p(pts), fill=L.col(CLOUD))
    d.polygon(L.p(inner), fill=L.col(BURST_IN))
    return L.done(), R


def mini_star(r, rot=0.0, ow=2.6):
    """小さな 5 つ角の星（白・濃紺の輪郭）。"""
    R = int(r + ow + 4)
    L = core.SS(s=3, w=2 * R, h=2 * R)
    pts = []
    for k in range(10):
        th = math.radians(rot - 90) + math.pi * k / 5
        rr = r if k % 2 == 0 else r * 0.45
        pts.append((R + rr * math.cos(th), R + rr * math.sin(th)))
    d = L.draw
    d.polygon(L.p(pts), fill=L.col(OUT))
    d.line(L.p(pts + [pts[0]]), fill=L.col(OUT), width=L.w(2 * ow), joint='curve')
    d.polygon(L.p(pts), fill=L.col(CLOUD))
    return L.done(), R


def focus_lines(center, r0, n=46, seed=9, alpha=0.8, width=(5, 14)):
    """集中線（画面の縁から中心へ向かう細いくさび、中心の r0 の内側はあける）。rgba（画面全体）。"""
    rng = np.random.default_rng(seed)
    L = core.SS(s=2)
    d = L.draw
    cx, cy = center
    far = 2400.0
    for k in range(n):
        th = 2 * math.pi * (k + rng.uniform(-0.35, 0.35)) / n
        ri = r0 * rng.uniform(1.0, 1.55)
        wd = rng.uniform(*width)
        c, s_ = math.cos(th), math.sin(th)
        nx, ny = -s_, c
        p_in = (cx + c * ri, cy + s_ * ri)
        po = (cx + c * far, cy + s_ * far)
        d.polygon(L.p([p_in, (po[0] + nx * wd * 3, po[1] + ny * wd * 3), (po[0] - nx * wd * 3, po[1] - ny * wd * 3)]),
                  fill=L.col(CLOUD, alpha))
    return L.done()


def outlined_text(text, size, fills, stroke, weight='Black', track=0.02, shadow=(0, 0.0), rot=0.0, pad=40):
    """縁取りの太い文字の板。fills：文字ごとの色（1 つなら全部）。stroke：縁の太さ px。shadow：(下へのずれ px, α)。
    戻り値 (rgba, 基準点の板の中の位置 (中心 x, ベースライン y))。"""
    s = 2
    f = core.font(size * s, weight)
    ws = [f.getlength(ch) for ch in text]
    tot = sum(ws) + track * size * s * (len(text) - 1)
    Wd = int(tot / s + 2 * pad + 2 * stroke)
    Hd = int(size * 1.45 + 2 * pad + 2 * stroke)
    im = Image.new('RGBA', (Wd * s, Hd * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    bx, by = Wd * s / 2, (pad + stroke + size * 1.08) * s
    if not isinstance(fills, (list, tuple)) or (len(fills) == 3 and np.isscalar(fills[0])):
        fills = [fills] * len(text)
    # 影（ずらした濃紺）→ 縁 → 塗り
    for layer in ('shadow', 'stroke', 'fill'):
        x = bx - tot / 2
        if layer == 'shadow' and shadow[1] <= 0:
            continue
        for ch, wd, fc in zip(text, ws, fills):
            if layer == 'shadow':
                d.text((x, by + shadow[0] * s), ch, font=f, fill=col(OUT, shadow[1]), anchor='ls',
                       stroke_width=int(stroke * s), stroke_fill=col(OUT, shadow[1]))
            elif layer == 'stroke':
                d.text((x, by), ch, font=f, fill=col(OUT), anchor='ls', stroke_width=int(stroke * s),
                       stroke_fill=col(OUT))
            else:
                d.text((x, by), ch, font=f, fill=col(fc), anchor='ls')
            x += wd + track * size * s
    if rot:
        im = im.rotate(-rot, resample=Image.BICUBIC, center=(bx, by))
    im = im.resize((Wd, Hd), Image.LANCZOS)
    rgba = np.asarray(im).astype(np.float32) / 255
    # ストレートアルファに（PIL の RGBA はストレートのまま）
    return rgba, (Wd / 2, by / s)
