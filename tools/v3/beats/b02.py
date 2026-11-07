"""b02（3.8–8.0秒）「人が乗った、最大の波。26.21メートル。」
真横の断面図（core.xs_background・水面 WL）。左：波（b05 と同じ色・線・白い水面の線）が沖で立ち上がる。
右：白い線画のマンション（9階・1階 3m）を比べるための目盛りとして置く（陸も浜も描かない。波は建物へ進まない）。
波の高さ＝カウンター（0.00→26.21m）。高さが各階の半分を越えると、その階がうすい ACC で点灯する。
カウンターは「ニイイチ」の「チ」のフレームでぴたりと止まる。サーファーは波の上の方の斜面に小さく（大きさの対比）。
最初の 8 フレーム：b01 の右への whip を受けて横のぶれが収まる。最後の 8 フレーム：下へのティルト（縦の whip）→ b03。
"""
import cv2
import numpy as np

import core
import timeline
from core import W, H, WL
from beats import _wave as V
from beats import _mapui as U

PXM = 13.0                       # 1m あたりの px（27m の建物＝351px）
REC = 26.21
FLOORS, FLOOR_M = 9, 3.0
BX0, BX1 = 770, 950              # 建物の左右
BTOP = WL - FLOORS * FLOOR_M * PXM
XC = 300.0                       # 波の頂の x（ほぼその場で立ち上がる）
WIN = 8                          # はじめ／おわりのつなぎ
Y0 = int(WL - 420)               # 波を描く帯の上端

WATER_LIT = V.XS_LIT


# ---------------------------------------------------------------- 建物（静止したレイヤーを一度だけ作る）
def _building():
    pad = 30
    x0, y0 = BX0 - pad, int(BTOP) - 60
    w, h = BX1 - BX0 + 2 * pad, int(WL - y0) + 8
    L = core.SS(s=4, w=w, h=h)
    d = L.draw
    X0, X1 = BX0 - x0, BX1 - x0
    T, B = BTOP - y0, WL - y0
    fh = FLOOR_M * PXM
    line = L.col(core.FOAM, 0.92)
    thin = L.col(core.FOAM, 0.50)
    # 屋上：塔屋と手すり
    d.rectangle(L.box(X1 - 58, T - 30, X1 - 18, T), outline=line, width=L.w(2.0))
    d.line(L.p([(X0 + 10, T - 12), (X1 - 66, T - 12)]), fill=thin, width=L.w(1.4))
    for xx in np.linspace(X0 + 10, X1 - 66, 7):
        d.line(L.p([(xx, T - 12), (xx, T)]), fill=thin, width=L.w(1.2))
    # 外形
    d.rectangle(L.box(X0, T, X1, B), outline=line, width=L.w(2.6))
    # 各階の床・窓（ベランダの手すりのような細い横線）
    wins = 4
    gw = (X1 - X0 - 24) / wins
    for k in range(FLOORS):
        yb = B - k * fh
        yt = yb - fh
        if k > 0:
            d.line(L.p([(X0, yb), (X1, yb)]), fill=L.col(core.FOAM, 0.70), width=L.w(1.6))
        for q in range(wins):
            wx0 = X0 + 12 + q * gw + 5
            wx1 = wx0 + gw - 10
            if k == 0 and q == 1:          # 1階の入口
                d.rectangle(L.box(wx0 + 4, yt + 10, wx1 - 4, yb), outline=thin, width=L.w(1.3))
                continue
            d.rectangle(L.box(wx0, yt + 9, wx1, yb - 12), outline=thin, width=L.w(1.2))
        if k > 0:
            d.line(L.p([(X0 + 8, yb - 7), (X1 - 8, yb - 7)]), fill=L.col(core.FOAM, 0.32), width=L.w(1.0))
    lay = L.done()
    # 各階の点灯マスク（外形の内側）
    masks = []
    for k in range(FLOORS):
        m = np.zeros((h, w), np.float32)
        yb = B - k * fh
        yt = yb - fh
        m[int(round(yt)) + 2:int(round(yb)) - 1, X0 + 2:X1 - 1] = 1.0
        masks.append(cv2.GaussianBlur(m, (0, 0), 0.6))
    return lay, masks, x0, y0


# ---------------------------------------------------------------- カウンター（等幅の数字・小数点はせまく・右端を固定）
class Counter:
    def __init__(self, cx, y, size=190, unit_size=120):
        self.y, self.size, self.us = y, size, unit_size
        f = core.font(size, 'Bold')
        self.cell = max(f.getlength(str(d)) for d in range(10)) * 0.96
        self.dot = f.getlength('.') * 1.15
        fu = core.font(unit_size, 'Bold')
        self.uw = fu.getlength('m')
        self.gap = 0.10 * size
        tot = 4 * self.cell + self.dot + self.gap + self.uw
        self.right = cx + tot / 2
        self.cache = {}

    def items(self, s):
        x = self.right - self.uw
        out = [dict(text='m', x=x, y=self.y, size=self.us, weight='Bold', anchor='ls', track=0)]
        x -= self.gap
        for ch in reversed(s):
            w = self.dot if ch == '.' else self.cell
            out.append(dict(text=ch, x=x - w / 2, y=self.y, size=self.size, weight='Bold', anchor='ms', track=0,
                            color=core.ACC))
            x -= w
        return out

    def put(self, f, v, alpha=1.0):
        s = '%.2f' % v
        p = self.cache.get(s)
        if p is None:
            p = self.cache[s] = U.Patch(self.items(s))
        return p.put(f, alpha)


# ---------------------------------------------------------------- 波
def wave_state(hm, p):
    """hm：波の高さ（m）、p：0..1 の進み具合 → 断面の点 (N,2)、形のパラメータ。"""
    m = -1.0 + 1.12 * core.ease(p, 'inout')          # 沖のなだらかな山 → 前が立った斜面（巻き込む前）
    Wd = 118 + 42 * p
    P = V.shape_xs(m)
    top = float(P[:, 1].max())
    Hh = hm * PXM / top                                 # 一番高い点がちょうど hm になるように
    xc = XC + 30 * p
    pts = np.stack([xc + P[:, 0] * Wd, WL - P[:, 1] * Hh], 1)
    return pts, P, m, xc, Wd, Hh


def face_spot(P, m, frac=0.66):
    """前の斜面で、高さが frac のあたりの点（単位系）と接線。"""
    if m >= 0:
        best = None
        for s in np.linspace(0.0, 1.0, 81):
            q, d = V.face_point(min(m, 1.0), s)
            e = abs(q[1] - frac)
            if best is None or e < best[0]:
                best = (e, q, d)
        return best[1], best[2]
    top = P[1:-1]
    i0 = int(np.argmax(top[:, 1]))
    fr = top[i0:]
    j = int(np.argmin(np.abs(fr[:, 1] - frac)))
    j = min(max(j, 1), len(fr) - 2)
    return fr[j], fr[j + 1] - fr[j - 1]


class Scene:
    def __init__(self, n):
        self.bg = core.xs_background(WL)
        self.blay, self.bmask, self.bx0, self.by0 = _building()
        self.bbody = np.clip(sum(self.bmask), 0, 1)
        hh = WL + 4 - Y0
        yy = np.arange(Y0, WL + 4, dtype=np.float32)[:, None]
        t = np.clip((WL - yy) / 360, 0, 1)
        self.wcol = (core.XS_WATER_TOP * (1 - t) + WATER_LIT * t)[:, None, :] * np.ones((1, W, 1), np.float32)
        self.hh = hh
        rng = np.random.default_rng(21)
        N = 700
        self.sp_ph = rng.uniform(0, 1, N).astype(np.float32)
        self.sp_u = rng.uniform(-0.25, 0.25, N).astype(np.float32)
        self.sp_v = np.stack([rng.normal(-3.2, 1.0, N), rng.normal(-1.6, 0.8, N)], 1).astype(np.float32)
        self.sp_w = rng.uniform(0.25, 1.0, N).astype(np.float32)

    def wave(self, out, pts, P, m, xc, Wd, Hh):
        reg = out[Y0:WL + 4]
        if Hh > 0.5:
            body = V.mask_poly(pts, self.hh, W, Y0)
            lines = np.zeros_like(body)
            top = P[1:-1]
            for s_ in (0.86, 0.72, 0.58, 0.44, 0.30, 0.16):
                q = np.stack([xc + top[:, 0] * Wd, WL - top[:, 1] * Hh * s_], 1)
                lines = np.maximum(lines, V.mask_line(q, self.hh, W, 1, Y0) * (0.08 + 0.22 * s_))
            b3 = body[..., None]
            reg[:] = reg * (1 - b3) + self.wcol * b3
            la = (lines * cv2.erode(body, np.ones((3, 3), np.uint8)))[..., None]
            reg[:] = reg * (1 - la) + core.FOAM * la
            top = pts[1:-1]
            top = top[(top[:, 0] > -40) & (top[:, 0] < W + 40)]
            line = np.concatenate([[(-40, WL)], top, [(W + 40, WL)]])
        else:
            line = np.array([(-40, WL), (W + 40, WL)], np.float32)
        lm = V.mask_line(line, self.hh, W, 3, Y0)[..., None]
        reg[:] = reg * (1 - lm) + core.FOAM * lm

    def spray(self, out, fi, m, xc, Wd, Hh, amt):
        """頂から後ろ（左）へ風で流れる細かいしぶき。"""
        if amt <= 0.01:
            return
        age = (fi / 24.0 + self.sp_ph) % 1.0
        tip_u = 0.05 + 0.08 * max(m, 0)
        x = xc + (tip_u + self.sp_u) * Wd + self.sp_v[:, 0] * age * 26
        y = WL - Hh * 1.0 + self.sp_v[:, 1] * age * 22 + 30 * age * age
        a = self.sp_w * np.sin(np.pi * age) ** 1.5 * amt * 0.55
        ya, yb = int(WL - Hh - 140), int(WL - Hh + 60)
        ya = max(ya, 0)
        buf = V.splat(np.zeros((yb - ya, W), np.float32), x, y - ya, a)
        al = np.clip(cv2.GaussianBlur(buf, (0, 0), 1.1) * 1.8 + cv2.GaussianBlur(buf, (0, 0), 4.0) * 1.2, 0, 0.9)
        out[ya:yb] = out[ya:yb] * (1 - al[..., None]) + core.FOAM * al[..., None]

    def building(self, out, lit):
        h, w = self.blay.shape[:2]
        sub = out[self.by0:self.by0 + h, self.bx0:self.bx0 + w]
        # 消えている階はわずかに沈んだ青（点灯した階との差がはっきり見えるように）
        base = self.bbody[..., None] * 0.30
        sub[:] = sub * (1 - base) + core.DEEP * base
        for k, a in enumerate(lit):
            if a > 0.002:
                m = self.bmask[k][..., None] * a
                sub[:] = sub * (1 - m) + core.ACC * m
        al = self.blay[..., 3:4]
        sub[:] = sub * (1 - al) + self.blay[..., :3] * al


def dashed_h(out, x0, x1, y, alpha, dash=12, gap=9, th=2):
    if alpha <= 0.01 or x1 <= x0:
        return
    ya = int(np.floor(y)) - 3
    m = np.zeros((8, W), np.float32)
    x = x0
    while x < x1:
        xe = min(x + dash, x1)
        cv2.line(m, (int(round(x * 4)), int(round((y - ya) * 4))), (int(round(xe * 4)), int(round((y - ya) * 4))),
                 1.0, th, cv2.LINE_AA, 2)
        x += dash + gap
    m = (m * alpha)[..., None]
    out[ya:ya + 8] = out[ya:ya + 8] * (1 - m) + core.FOAM * m


def render(beat, ctx):
    n = beat['n']
    sc = Scene(n)
    f_stop = timeline.mora_frame(beat, 23)          # 「ニイイチ」の「チ」
    assert beat['moras'][23][0] == 'チ', beat['moras'][23]
    f_go = 12                                       # 立ち上がり始め（whip のあと）
    f_note = f_stop - 8                             # 8〜9階が点くころ（読む時間を 1 秒ほど確保）
    cnt = Counter(W / 2, 744)
    t_g = U.Patch([dict(text='ギネス世界記録', x=W / 2, y=566, size=44, weight='Medium', track=0.14)])
    t_n = U.Patch([dict(text='マンションなら8〜9階くらい', x=W / 2, y=1226, size=34, weight='Medium', track=0.06)])
    lit_on = [None] * FLOORS
    for i in range(n):
        # 波の高さ＝カウンター：はじめはゆっくり、途中で速く、最後はぴたりと止まる
        t = np.clip((i - f_go) / (f_stop - f_go), 0, 1)
        p = 1 - (1 - t) ** 1.9
        p = p * (0.35 + 0.65 * core.ease(min(1.0, t * 4), 'out'))
        hm = REC * p if i < f_stop else REC
        pp = min(1.0, hm / REC)
        pts, P, m, xc, Wd, Hh = wave_state(max(hm, 1.2 * (1 - pp) + hm * pp), pp)
        out = sc.bg.copy()
        sc.wave(out, pts, P, m, xc, Wd, Hh)
        sc.spray(out, i, m, xc, Wd, Hh, V.smooth(pp, 0.55, 0.95) * (1.0 + 0.0 * i))
        # 各階：その階の半分の高さを越えたら点灯（3 フレームで）
        lit = []
        for k in range(FLOORS):
            if lit_on[k] is None and hm >= k * FLOOR_M + 1.5:
                lit_on[k] = i
            if lit_on[k] is None:
                lit.append(0.0)
            else:
                g = i - lit_on[k]
                lit.append(0.40 * core.ease((g + 1) / 3, 'out') + 0.18 * np.exp(-g / 5.0) * (g >= 0))
        sc.building(out, lit)
        # 頂の高さ → 建物まで細い点線（いまの高さ）
        ytop = WL - hm * PXM
        if hm > 1.0:
            xtop = float(pts[np.argmin(pts[:, 1]), 0])
            dashed_h(out, xtop + 26, BX1 + 18, ytop, 0.55 * V.smooth(hm, 1.0, 4.0))
        # サーファー（斜面の上のほう。小さく＝波の大きさが伝わる）
        q, d = face_spot(P, m, 0.62)
        sx, sy = xc + q[0] * Wd, WL - q[1] * Hh
        ang = np.degrees(np.arctan2(-d[1] * Hh, d[0] * Wd))
        sa = core.ease((i - 4) / 8)
        bob = 1.2 * np.sin(i / 30 * 2 * np.pi * 0.7)
        V.draw_surfer(out, sx, sy - 2 + bob, scale=1.35, ang=float(np.clip(0.45 * ang, -6, 22)), alpha=sa,
                      shadow=0.45)
        # 最初：b01 の右への whip が収まる（絵は右から左へ流れ込んで止まる）
        if i < WIN:
            k = core.ease(i / WIN, 'out')            # 0 フレーム目は b01 の最後と同じぶれ（120px）
            dx = 220 * (1 - k)
            M = np.float32([[1, 0, dx], [0, 1, 0]])
            out = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
            kk = int(round(120 * (1 - k) ** 1.5))
            if kk >= 2:
                out = cv2.blur(out, (kk, 1), borderType=cv2.BORDER_REPLICATE)
        # 文字
        j = i - (n - WIN)
        q_out = (j + 1) / WIN if j >= 0 else 0.0
        ta = 1 - core.ease((i - (n - WIN - 5)) / 6)   # 文字は whip が強くなる前に消す（ぶれた文字の残像を出さない）
        out = t_g.put(out, core.ease((i - WIN + 1) / 6) * ta)
        out = cnt.put(out, hm, core.ease((i - WIN + 1) / 6) * ta)
        out = t_n.put(out, core.ease((i - f_note + 1) / 6) * ta)
        # 最後：下へのティルト（絵は上へ流れる）→ b03 の最初（下から上へ収まる）へ
        if j >= 0:
            dy = -240 * q_out ** 2
            M = np.float32([[1, 0, 0], [0, 1, dy]])
            out = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
            kk = int(round(150 * q_out ** 1.5))
            if kk >= 2:
                out = cv2.blur(out, (1, kk), borderType=cv2.BORDER_REPLICATE)
        yield np.clip(out, 0, 1).astype(np.float32)
