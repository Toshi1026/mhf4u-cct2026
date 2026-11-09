"""案C 15（約2.3秒）「悪役も、ここなら悪くないかも。」 実写の波打ち際に、ずぶ濡れの悪役（顔のない白いピクトグラム）が座って海を眺める。

素材：IMG_9662（11 の頭と同じ浜。悪役が落ちた海の、その波打ち際）。4.6 秒から 0.5倍スロー。上寄りに 1.32 倍で切り出し、
     水平線の傾きは 11 と同じ +1.34°。砂の縁（波打ち際）は画面の y≈1258。テロップなし。
悪役（_wet.py の SDF）：砂の縁のすぐ手前（足もと y≈1296、x≈430）に、カメラに背を向けて斜めに座る（yaw −50°：ひざとすねの形が
     右に見える）。座高 約 150px（画面の高さの 8%＝「小さく」）。
  芝居：最初は肩を落としてうつむく（ずぶ濡れ）。「コ（ここなら）」で頭を上げて水平線を見て、肩の力が抜ける（ease 14 フレーム）。
        ゆっくりした息（肩が 0.25 単位上下、1.9 秒周期。素材が 0.5倍スローなので動きもゆっくり）。
  なじませ方：
   - 光：素材は海（南）を向いた逆光ぎみ → 光は右上の奥から。背中は影の側（空の青い光）、頭と肩の上・右の縁にリム。
   - 影：お尻と足もとの接地の影（AO、やわらかい楕円）＋光の反対（手前・左下）へ落ちる影（シルエットを縦 0.22 倍につぶして
     左へずらし、ぼかす）。濡れた砂の黒ずみ（体の下の楕円、ゆっくり広がる）。
   - 色：素材と同じ _real.grade を通す。σ0.55px のぼけ（5倍の望遠の甘さ）と、素材と同じくらいの粒（σ0.010）。
     縁に背景を 18% 回り込ませる（ライトラップ）。
   - しずく：両ひじの下とマントの裾の左右から、ふくらんでは落ちる（重力、素材と同じ 0.5倍スローの速さ）。落ちた所の砂に小さな黒ずみが残る。
最後の 14 フレーム：夕方から夜へ（露出 1.0→0.40、上ほど暗く・青く）。16 の夜の看板（平均の明るさ 約 0.2）へ、1 フレームの段差 0.1 未満で渡す。
カメラ：全体（素材＋悪役）に 1.00→1.045 の寄り（悪役の少し上を中心に、sine_inout）。
"""
import math

import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W, H
from beats import _real
from cutB.beats import _motion as Mo
from cutB.beats import _fac as F
from cutC.beats import _maze as Z
from cutC.beats import _wet as WT

PLATE = dict(name='IMG_9662.MOV', t0=4.6, rot=1.34, cx=0.5, cy=0.621, zoom=1.32)
VX, VY, SCALE, YAW = 430.0, 1296.0, 6.9, -50.0
PUSH, PIVOT = 1.045, (VX + 40.0, VY - 90.0)
DUSK, DUSK_EXP = 14, 0.40
GRAIN = 0.010
_rng = np.random.default_rng(15)


ROI = (240, 980, 680, 1400)          # x0, y0, x1, y1：悪役まわりの処理はこの中だけ（速さのため）


def _ell(cx, cy, rx, ry, blur):
    """やわらかい楕円のマスク（ROI の大きさ）。"""
    x0, y0, x1, y1 = ROI
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.ellipse(m, (int(round((cx - x0) * 4)), int(round((cy - y0) * 4))), (int(round(rx * 4)), int(round(ry * 4))), 0, 0,
                360, 1.0, -1, cv2.LINE_AA, 2)
    return cv2.GaussianBlur(m, (0, 0), blur)


def _cast(rgba, x0, y0, base_y, squash=0.22, shear=-1.3):
    """光の反対（手前・左下）へ落ちる影：シルエットを base_y を軸に下へ折り返し、縦につぶして左へずらす（ROI の大きさ）。"""
    a = rgba[..., 3]
    rx0, ry0, rx1, ry1 = ROI
    by = base_y - ry0
    # 入力（板の中の x, y）→ 出力（ROI の中）：y' = by + (by − Y)·squash、x' = X + shear·squash·(by − Y)、X = x + x0 − rx0
    ox, oy = x0 - rx0, y0 - ry0
    M = np.float32([[1, -shear * squash, ox + shear * squash * (by - oy)],
                    [0, -squash, by + squash * (by - oy)]])
    out = cv2.warpAffine(a, M, (rx1 - rx0, ry1 - ry0), flags=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(out, (0, 0), 2.4)


class Drips:
    """しずく：点 p（画面）でふくらみ（grow フレーム）、落ちて砂（ground_y）に当たると黒ずみを残す。閉じた式ではなく順に進める。"""

    def __init__(self, sources, ground_y, seed=4):
        self.rng = np.random.default_rng(seed)
        self.src = sources                       # [(x, y, period, phase)]
        self.ground = ground_y
        self.g = 0.55                            # px／フレーム²（0.5倍スローの重力：座高 150px≈1m なので 150*9.8/30²/4）

    def state(self, i):
        out = []
        for (x, y, per, ph) in self.src:
            k = (i + ph) % per
            grow = 7
            if k < grow:
                out.append(('bead', x, y + 1.2 * k / grow, 0.6 + 0.9 * k / grow))
            else:
                t = k - grow
                yy = y + 1.2 + 0.5 * self.g * t * t
                if yy < self.ground:
                    out.append(('fall', x, yy, self.g * t))
        return out

    def spots_until(self, i):
        """i までに砂に落ちたしずくの位置（黒ずみ）。"""
        sp = []
        for (x, y, per, ph) in self.src:
            tf = 7 + math.sqrt(2 * max(1.0, self.ground - y - 1.2) / self.g)
            k0 = -ph
            while k0 + tf <= i:
                if k0 + tf >= 0:
                    sp.append((x + 0.6 * math.sin(k0 * 1.7), self.ground + 1.5 * math.cos(k0)))
                k0 += per
        return sp


def _draw_drops(r, items, spots):
    """r：ROI の切り出し（ここに描く）。しずくは水の玉：暗い縁＋明るい芯（背景の砂でも泡でも読める）。"""
    x0, y0 = ROI[0], ROI[1]
    h, w = r.shape[:2]
    if spots:
        L = core.SS(s=3, w=w, h=h)
        d = L.draw
        for (x, y) in spots:
            d.ellipse(L.box(x - x0 - 2.8, y - y0 - 1.1, x - x0 + 2.8, y - y0 + 1.1), fill=(0, 0, 0, 90))
        r = r * (1 - L.done()[..., 3:4] * 0.9)
    if items:
        L = core.SS(s=3, w=w, h=h)
        d = L.draw
        for kind, x, y, v in items:
            x, y = x - x0, y - y0
            if kind == 'bead':
                rr = v
                d.ellipse(L.box(x - rr - 0.5, y - rr * 0.9 - 0.5, x + rr + 0.5, y + rr * 1.15 + 0.5), fill=(40, 60, 80, 120))
                d.ellipse(L.box(x - rr, y - rr * 0.9, x + rr, y + rr * 1.15), fill=L.col(core.FOAM, 0.75))
            else:
                ln = min(9.0, 1.6 + v * 0.5)
                d.ellipse(L.box(x - 1.7, y - ln - 0.5, x + 1.7, y + 1.8), fill=(40, 60, 80, 110))
                d.ellipse(L.box(x - 1.15, y - ln, x + 1.15, y + 1.3), fill=L.col(core.FOAM, 0.85))
        r = core.over(r, L.done())
    return r


def _comp_figure(f, rgba, x0, y0):
    """素材の色にそろえて、ぼかし・粒・ライトラップをして重ねる。"""
    h, w = rgba.shape[:2]
    ys, xs = slice(y0, y0 + h), slice(x0, x0 + w)
    bg = f[ys, xs]
    a = rgba[..., 3:4]
    col = _real.grade(np.clip(rgba[..., :3], 0, 1))
    col = cv2.GaussianBlur(col, (0, 0), 0.55)
    a = cv2.GaussianBlur(a, (0, 0), 0.55)[..., None]
    col = col + _rng.normal(0, GRAIN, col.shape).astype(np.float32)
    # ライトラップ：縁（アルファの内側の勾配）に背景のぼかしを混ぜる
    bgb = cv2.GaussianBlur(bg, (0, 0), 5.0)
    inner = cv2.GaussianBlur(a[..., 0], (0, 0), 2.5)[..., None]
    edge = np.clip((1 - inner) * 2.2, 0, 1) * a
    col = col * (1 - 0.18 * edge) + bgb * 0.18 * edge
    out = f.copy()
    out[ys, xs] = bg * (1 - a) + col * a
    return out


def _dusk(f, q):
    """夕方から夜へ：露出を下げ、上ほど暗く・青く。q：0→1。"""
    if q <= 0:
        return f
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    e = 1 - (1 - DUSK_EXP) * q * (1.08 - 0.16 * y)
    g = f * e
    lum = g.mean(2, keepdims=True)
    g = lum + (g - lum) * (1 - 0.20 * q)
    g = g * (1 + q * np.array([-0.14, -0.04, 0.10], np.float32))
    return np.clip(g, 0, 1)


def _push(f, z):
    if z < 1 + 1e-5:
        return f
    px, py = PIVOT
    Mt = np.float32([[z, 0, px * (1 - z)], [0, z, py * (1 - z)]])
    return cv2.warpAffine(f, Mt, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def render(beat, ctx):
    n = beat['n']
    f_ko = Z.mf(beat, 5, 'コ')
    plate = F.fast_clip(PLATE, n)          # _real.clip と同じ画（回転・切り出しを変換の前にする速い読み出し）
    # しずくの出どころ（画面）：両ひじの下＋マントの裾の左右の少し上
    src = []
    for k, (x, y, z) in enumerate(WT.elbows(YAW, 0.5)):
        src.append((VX + x * SCALE, VY - (WT.GROUND_Y - y) * SCALE, 23 + 6 * k, 5 + 11 * k))
    src.append((VX - 3.6 * SCALE, VY - 3.2 * SCALE, 29, 17))
    drips = Drips(src, VY + 2.0)
    wet0, wet1 = _ell(VX - 6, VY + 3, 74, 13, 6.0), _ell(VX - 6, VY + 3, 84, 15, 6.0)
    ao = _ell(VX - 2, VY + 1, 46, 6.5, 3.0)
    ao2 = _ell(VX - 2, VY + 1, 30, 3.5, 1.5)
    x0r, y0r, x1r, y1r = ROI
    for i in range(n):
        f = next(plate)
        u = Mo.ease((i - f_ko + 2) / 14.0, 'sine_inout')
        head, slump = 0.8 * u, 0.7 * (1 - u)
        breath = 0.5 + 0.5 * math.sin(2 * math.pi * i / 57.0 - 1.2)
        rgba, x0, y0, A, c = WT.rgba(VX, VY, SCALE, yaw=YAW, breath=breath, head=head, slump=slump, t=i / 30.0)
        r = f[y0r:y1r, x0r:x1r]
        # 砂：濡れた黒ずみ → 接地の影 → 落ちる影 → 砂に落ちたしずくの跡
        g = i / float(n - 1)
        wet = (wet0 * (1 - g) + wet1 * g)[..., None]
        r = r * (1 - 0.18 * wet) * (1 - 0.05 * wet * np.array([1, 0.6, 0], np.float32))
        r = r * (1 - 0.50 * ao[..., None]) * (1 - 0.35 * ao2[..., None])
        r = r * (1 - 0.50 * _cast(rgba, x0, y0, int(VY + 2))[..., None])
        r = _draw_drops(r, [], drips.spots_until(i))
        f = f.copy()
        f[y0r:y1r, x0r:x1r] = r
        f = _comp_figure(f, rgba, x0, y0)
        f[y0r:y1r, x0r:x1r] = _draw_drops(f[y0r:y1r, x0r:x1r], drips.state(i), [])
        f = _push(f, 1 + (PUSH - 1) * Mo.ease(i / float(n - 1), 'sine_inout'))
        f = _dusk(f, Mo.ease((i - (n - DUSK) + 1) / float(DUSK), 'sine_in'))
        yield np.clip(f, 0, 1).astype(np.float32)
