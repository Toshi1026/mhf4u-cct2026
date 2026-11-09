"""案C 06（約 1.6 秒）「マッハ6.7。」 テロップ「マッハ6.7」（6.7 を強調）。

上昇の続き：高さの柱（_column、0.1→1.3km、対数の一定の速さ）を背に、悪役は画面の上の方を小さく上がっていく。
ガラスのスピードメーター（案B 04 の _dial.Dial：ガラスの文字盤・クロームの縁・浮き彫りの目盛り）を「マッハ」の目盛り
（0〜5、0.25 ごと、整数が大きい目盛り）にしたもの。メーターの最大はマッハ 5 なので、6.7 は測れない ＝ 振り切れて針が折れる（マンガ）：
  0–9  ：入り（C05 から上への振りの続き：中身が下へ流れるぶれ 90→0）。メーターが組み上がる（半円が描かれ、目盛りがぽんと出る）。
  「マ」：針が跳ね上がる（quad_in）。カウンター「マッハ X.X」（数字が回る）は針の値で、1.0 を越えてから出る。
  17   ：針が 5 に着いて、止めのピンに押しつけられ、ふるえながら曲がる（カウンターは 5 を越えて回り続ける）。
  26   ：針が折れる（ピンの所から先が回りながら右上へ飛んでいく）。ガラスの小さなかけら 11 枚（重力つき）と、ガラスにひび。
        メーターが小さくはねる（揺れ）。白い光・点滅はなし（形と動きだけ）。
  「ナ」：カウンターが 6.7 に決まる（落ち＋拡大のポップ）。
  最後の 6 フレーム：メーターは下へ落ちながらぼけて消え、カウンターは上へ抜ける。悪役は y 710（C07 の位置）へ。
"""
import math

import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W, H
from beats import _mapui as U
from cutB.beats import _motion as M, _render as RR, _maplook as ML, _dial
from cutB.beats._m34 import Board
from cutC.beats import _column as CO, _vill as V, _fall as FA, _space as SPC
import timeline_c as TL

MC = (540.0, 940.0)
RD = 330.0
VMAX = 500                     # マッハ × 100
PEG = 504.0                    # 針が止めのピンに当たる値（マッハ 5.04）
CNT_Y = 1150
A0_, A1_ = 0.10, 1.30
_T = {}


class MachDial(_dial.Dial):
    """マッハの目盛り（0.25 ごと、1 ごとに大きい目盛り、0.5 に中くらい）。針は自分で描く（曲がる・折れるため）。"""

    def th(self, v):
        return np.pi * (1 - np.clip(v, -60, self.vmax * 1.12) / self.vmax)

    def _ticks(self, prog, jet, jet_len, jet_a, jet_p):
        key = None
        if all(p >= 0.999 for _, p in prog):
            key = 'all'
            if self._tick_cache is not None and self._tick_cache[0] == key:
                return self._tick_cache[1]
        Rr = self.R
        Lm, Ld, Ll = (core.SS(s=3, w=self.w, h=self.h) for _ in range(3))
        cx, cy = self.cx - self.x0, self.cy - self.y0
        for v, p in prog:
            if p <= 0.01:
                continue
            t_ = self.th(v)
            major, mid = v % 100 == 0, v % 50 == 0
            off = -8 * (1 - p)
            r0 = Rr - (30 if major else 20 if mid else 12) + off
            r1 = Rr + off
            al = min(1.0, p * 1.6) * (0.85 if major else 0.50)
            wd = 2.4 if major else 1.4 if mid else 1.0
            pts = [(cx + r0 * np.cos(t_), cy - r0 * np.sin(t_)), (cx + r1 * np.cos(t_), cy - r1 * np.sin(t_))]
            Lm.draw.line(Lm.p(pts), fill=Lm.col((1, 1, 1), al), width=Lm.w(wd))
            Ld.draw.line(Ld.p([(x + 0.9, y + 1.0) for x, y in pts]), fill=Ld.col((1, 1, 1), min(1, al * 1.4)),
                         width=Ld.w(wd))
            Ll.draw.line(Ll.p([(x - 0.7, y - 0.7) for x, y in pts]), fill=Ll.col((1, 1, 1), al * 0.5),
                         width=Ll.w(max(0.8, wd - 0.6)))
        out = (Lm.done()[..., 3], cv2.GaussianBlur(Ld.done()[..., 3], (0, 0), 0.6), Ll.done()[..., 3])
        if key is not None:
            self._tick_cache = (key, out)
        return out


def dial():
    if 'd' not in _T:
        _T['d'] = MachDial(MC, RD, vmax=VMAX, pad=96)
        labs = []
        for v in range(0, VMAX + 1, 100):
            th = np.pi * (1 - v / VMAX)
            rl = RD - 66
            p = U.Patch([dict(text='%d' % (v // 100), x=MC[0] + rl * np.cos(th), y=MC[1] - rl * np.sin(th) + 12, size=34,
                              weight='Medium', alpha=0.85, track=0)])
            labs.append((v // 25, Board.from_patch(p)))
        _T['labels'] = labs
        _T['mach'] = Board.from_patch(U.Patch([dict(text='MACH', x=MC[0], y=MC[1] - 120, size=26, weight='Medium',
                                                    alpha=0.55, track=0.3)]))
    return _T['d']


# ---------------------------------------------------------------- 時刻
def timing(beat):
    if 'tm' not in _T:
        m = beat['moras']
        f_ma = TL.mora_frame(beat, 0)
        f_na = TL.mora_frame(beat, 7)
        _T['tm'] = dict(go=f_ma, peg=f_ma + 14, snap=f_ma + 23, lock=f_na, exit=beat['n'] - 6)
    return _T['tm']


def needle(i, T):
    """(値, 曲がり（度）)。値は針の付け根の角度の値（マッハ×100）。"""
    if i < T['go']:
        return 0.0, 0.0
    if i < T['peg']:
        q = (i - T['go']) / float(T['peg'] - T['go'])
        return PEG * q * q, 0.0
    if i < T['snap']:
        q = (i - T['peg']) / float(T['snap'] - T['peg'])
        v = PEG + 6.0 * math.sin((i - T['peg']) * 2.6) * (1 - 0.3 * q)          # ピンに押しつけてふるえる
        bend = 42.0 * M.ease(q, 'quad_in') + 4.0 * math.sin((i - T['peg']) * 3.3)  # 先が曲がっていく
        return v, bend
    # 折れたあと：残った根元はばねで少し戻る
    v = PEG - 40.0 * M.spring((i - T['snap']) / 30.0, 0.35, 3.0)
    return v, 0.0


def counter_v(i, T):
    """カウンターの値（マッハ×10）：針が 5 に着くまでは針の値、そのあとも 67 まで回り続ける（針はもう測れない）。"""
    if i < T['peg']:
        return needle(i, T)[0] / 10.0
    v0 = PEG / 10.0
    q = (i - T['peg']) / float(T['lock'] - T['peg'])
    return v0 + (67.0 - v0) * M.ease(q, 'quad_out')


def counter():
    if 'c' not in _T:
        od = M.Odometer('6,7', W / 2 + 40, CNT_Y, digit_size=150, side_size=92, prefix='マッハ', suffix='', hl_suffix='',
                        feather_px=10)
        dot = M._patch_of([dict(text='.', x=0, y=0, size=150, weight='Bold', anchor='ms', track=0, color=core.ACC)])
        od.dg[','] = dot
        od.comma_w = core.font(150, 'Bold').getlength('.')
        _T['c'] = od
    return _T['c']


# ---------------------------------------------------------------- 針・キャップ（_dial.Dial と同じ見た目、曲がる・折れる）
def _needle_poly(cx, cy, ang, r_from, r_to, bend, hb=6.5, ht=1.1, r_bend=None):
    """針の左右の多角形（画面の向き）。bend：r_bend から先を曲げる角度（度、マイナス＝下へ）。"""
    ux, uy = math.cos(ang), -math.sin(ang)
    pts_c = []
    rb = r_bend if r_bend is not None else r_to
    for r in np.linspace(r_from, r_to, 14):
        if r <= rb or bend == 0:
            pts_c.append((cx + r * ux, cy + r * uy, ang))
        else:
            a2 = ang - math.radians(bend) * (r - rb) / max(r_to - rb, 1e-3)
            px, py = cx + rb * ux, cy + rb * uy
            # 曲がった先：弧に沿って
            seg = (r - rb)
            pts_c.append((px + seg * math.cos((ang + a2) / 2), py - seg * math.sin((ang + a2) / 2), a2))
    left, right = [], []
    n = len(pts_c)
    for j, (x, y, a) in enumerate(pts_c):
        t = j / (n - 1.0)
        hw = hb + (ht - hb) * t
        nx, ny = math.sin(a), math.cos(a)
        left.append((x + nx * hw, y + ny * hw))
        right.append((x - nx * hw, y - ny * hw))
    return left, right


def draw_pin(Bd, alpha=1.0):
    """止めのピン（文字盤の 5 のすぐ先、半径 R−70 の小さな金属の丸）。"""
    d = dial()
    if 'pin' not in _T:
        th = float(d.th(PEG + 6.0))
        r = RD - 70
        x, y = d.cx - d.x0 + r * math.cos(th), d.cy - d.y0 - r * math.sin(th)
        L = core.SS(s=3, w=d.w, h=d.h)
        L.draw.ellipse(L.box(x - 8, y - 8, x + 8, y + 8), fill=L.col(RR.srgb(RR.hexl('6F8AA2')), 1.0))
        L.draw.ellipse(L.box(x - 6, y - 7, x + 3, y + 2), fill=L.col(RR.srgb(RR.hexl('E4EEF6')), 1.0))
        _T['pin'] = L.done()
    m = _T['pin']
    a = m[..., 3] * alpha
    Bd.rgb = Bd.rgb * (1 - a[..., None]) + m[..., :3] * a[..., None]
    Bd.a = np.maximum(Bd.a, a)
    return Bd


def draw_needle(Bd, ang, bend, stub=None, cs=1.0):
    """Board に針（2 色の面取り＋影）とキャップ。stub：折れたあとの根元の長さ（None なら全部）。"""
    d = dial()
    cx, cy = d.cx - d.x0, d.cy - d.y0
    r_to = (RD - 44) * cs if stub is None else stub
    left, right = _needle_poly(cx, cy, ang, -26 * cs, r_to, bend, 6.5 * cs, 1.1 * cs if stub is None else 3.2 * cs,
                               r_bend=(RD - 44) * 0.55)
    L1, L2 = core.SS(s=3, w=d.w, h=d.h), core.SS(s=3, w=d.w, h=d.h)
    cen = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(left, right)]
    L1.draw.polygon(L1.p(left + cen[::-1]), fill=(255, 255, 255, 255))
    L2.draw.polygon(L2.p(cen + right[::-1]), fill=(255, 255, 255, 255))
    m1, m2 = L1.done()[..., 3], L2.done()[..., 3]
    mm = np.clip(m1 + m2, 0, 1)
    lit_left = (math.sin(ang) * _dial.LV[0] + math.cos(ang) * _dial.LV[1]) > 0
    cA, cB = (RR.srgb(RR.hexl('E4EEF6')), RR.srgb(RR.hexl('8EA7BC')))
    if not lit_left:
        cA, cB = cB, cA
    sh = cv2.GaussianBlur(cv2.warpAffine(mm, np.float32([[1, 0, 6], [0, 1, 6]]), (d.w, d.h)), (0, 0), 2.5) * 0.35
    Bd.rgb *= (1 - sh[..., None])
    Bd.rgb = Bd.rgb * (1 - mm[..., None]) + cA * m1[..., None] + cB * m2[..., None]
    Bd.a = np.maximum(Bd.a, mm)
    _cap(Bd, cs)
    return Bd


def _cap(Bd, cs):
    d = dial()
    if 'cap' not in _T:
        dx, dy, r = d._dx, d._dy, d._r
        rc = 26.0
        dd = r / rc
        capm = np.clip((rc - r) + 0.5, 0, 1)
        nzc = np.sqrt(np.clip(1 - dd * dd, 0, 1))
        nyc = dy / rc
        ryc = -(2 * nzc * nyc)
        envc = _dial._env(ryc) * 0.9
        Hh = _dial.LV + np.array([0, 0, 1], np.float32)
        Hh /= np.linalg.norm(Hh)
        spc = np.clip(dx / rc * Hh[0] + dy / rc * Hh[1] + nzc * Hh[2], 0, 1)
        capc = envc + RR.SUN_RGB * (spc ** 80 * 0.9)[..., None]
        ring = np.clip(1.2 - np.abs(r - rc * 0.42), 0, 1) * capm * 0.25
        capc = capc * (1 - ring[..., None])
        capsh = cv2.GaussianBlur(np.clip((rc - np.hypot(dx - 5, dy - 7)), 0, 1).astype(np.float32), (0, 0), 4) * 0.45
        _T['cap'] = (RR.srgb(capc).astype(np.float32), capm.astype(np.float32), capsh.astype(np.float32))
    col, m, sh = _T['cap']
    Bd.rgb = Bd.rgb * (1 - sh[..., None]) * (1 - m[..., None]) + col * m[..., None]
    Bd.a = np.maximum(Bd.a, np.maximum(m, sh))


# ---------------------------------------------------------------- 折れた先・かけら・ひび
SHARDS = [(-62, 9.0, 12.0, 0), (-35, 11.0, 9.0, 1), (-12, 8.5, 11.0, 0), (8, 10.0, 8.0, 1), (-80, 7.0, 10.0, 2),
          (-50, 13.0, 7.0, 0), (-22, 6.0, 14.0, 1), (15, 7.5, 7.5, 2), (-95, 9.5, 6.5, 1), (-5, 12.0, 6.0, 0),
          (-70, 5.5, 12.0, 2), (-40, 15.0, 5.5, 1), (25, 6.5, 9.0, 0), (-110, 6.0, 8.0, 2)]


def peg_xy():
    th = np.pi * (1 - PEG / VMAX)
    r = RD - 44
    return MC[0] + r * math.cos(th), MC[1] - r * math.sin(th)


def put_broken(f, i, T, dx=0.0, dy=0.0):
    """折れた針の先（回りながら右上へ）と、ガラスのかけら（重力つき）。f：画面（sRGB）。"""
    k = i - T['snap']
    if k < 0:
        return f
    px, py = peg_xy()
    # 先のかけら：長さ 0.45·(R−44)、ピンの所から右上へ（14, −20）px/フレーム、重力 1.4、回転 −38°/フレーム
    ln = (RD - 44) * 0.45
    cx = px + 14.0 * k + dx
    cy = py - 20.0 * k + 0.7 * k * k + dy
    ang = math.radians(-8.0 - 38.0 * k)
    ux, uy = math.cos(ang), -math.sin(ang)
    L = core.SS(s=3, w=W, h=H)
    a0 = (cx - ux * ln / 2, cy - uy * ln / 2)
    a1 = (cx + ux * ln / 2, cy + uy * ln / 2)
    nx, ny = math.sin(ang), math.cos(ang)
    poly = [(a0[0] + nx * 3.4, a0[1] + ny * 3.4), (a1[0] + nx * 1.1, a1[1] + ny * 1.1),
            (a1[0] - nx * 1.1, a1[1] - ny * 1.1), (a0[0] - nx * 3.4, a0[1] - ny * 3.4)]
    L.draw.polygon(L.p(poly), fill=L.col(RR.srgb(RR.hexl('D9E6F0')), 1.0))
    # ガラスのかけら：小さな三角（明るい縁、半透明の面）
    rng = np.random.default_rng(5)
    for j, (a_deg, sp, sz, sd) in enumerate(SHARDS):
        life = 16 + 3 * sd
        if k > life:
            continue
        th = math.radians(a_deg)
        vx, vy = math.cos(th) * sp, math.sin(th) * sp
        x = px + dx + vx * k * (1 - 0.02 * k)
        y = py + dy + vy * k + 0.8 * k * k
        rot = k * (17 + 9 * sd) * (1 if j % 2 else -1)
        al = 0.85 * (1 - M.ease((k - life + 5) / 5.0, 'quad_in'))
        tri = []
        for q in range(3):
            aa = math.radians(rot + q * 120 + rng.uniform(-25, 25))
            rr = sz * (0.7 + 0.5 * rng.random())
            tri.append((x + rr * math.cos(aa), y + rr * math.sin(aa)))
        L.draw.polygon(L.p(tri), fill=L.col(RR.srgb(RR.hexl('BFDDF2')), al * 0.55), outline=L.col(core.FOAM, al))
    m = L.done()
    a = m[..., 3:4]
    sh = cv2.GaussianBlur(cv2.warpAffine(m[..., 3], np.float32([[1, 0, 3], [0, 1, 4]]), (W, H)), (0, 0), 2.0) * 0.3
    f = f * (1 - sh[..., None])
    return f * (1 - a) + m[..., :3] * a


def cracks(Bd, i, T):
    """ガラスのひび：ピンのまわりに細い白い線 6 本（折れた瞬間に 3 フレームでのびる）。"""
    k = i - T['snap']
    if k < 0:
        return Bd
    d = dial()
    px, py = peg_xy()
    px, py = px - d.x0 - 30, py - d.y0 + 10
    g = M.ease((k + 1) / 3.0, 'expo_out')
    L = core.SS(s=3, w=d.w, h=d.h)
    rng = np.random.default_rng(12)
    for j in range(6):
        a = math.radians(-180 + j * 34 + rng.uniform(-10, 10))
        ln = (60 + 70 * rng.random()) * g
        mid = (px + math.cos(a) * ln * 0.5 + rng.uniform(-5, 5) * g, py + math.sin(a) * ln * 0.5 + rng.uniform(-5, 5) * g)
        end = (px + math.cos(a) * ln, py + math.sin(a) * ln)
        L.draw.line(L.p([(px, py), mid, end]), fill=L.col(core.FOAM, 0.8), width=L.w(1.5))
    m = L.done()[..., 3] * d.face_a
    Bd.rgb = Bd.rgb * (1 - m[..., None]) + np.asarray(core.FOAM, np.float32) * 0.85 * m[..., None]
    return Bd


# ---------------------------------------------------------------- 背景・悪役
def alt(i, n):
    L0, L1 = float(CO.lnA(A0_)), float(CO.lnA(A1_))
    return float(CO.A0 * math.expm1(L0 + (L1 - L0) * i / (n - 1.0)))


def villain(f, i, n, k08):
    """悪役：画面の上の方（y 430）を小さく上がる → 最後の 8 フレームで C07 の位置（y 710・3.4 倍）へ。"""
    q = M.ease((i - (n - 9)) / 8.0, 'sine_inout')
    x, y = FA.APEX[0], 430.0 + (710.0 - 430.0) * q
    sc = 2.4 + (3.4 - 2.4) * q
    # C07 の頭のフレームの回転の続き（C07 の n フレーム前）
    n07 = _n07()
    kk = k08 - n07 - (n - i)
    r, x0, y0 = V.rgba(x, y, sc, FA.tumble_ang(kk), V.pose('tumble'), kk / 30.0, env=V.SKY_ENV)
    f = CO.speed_lines(f, x, y, i, 0.7, spread=120.0)
    return M.put_rgba(f, r, x0, y0)


def _n07():
    if 'n07' not in _T:
        _T['n07'] = next(b for b in TL.load() if b['key'] == '07')['n']
    return _T['n07']


# ---------------------------------------------------------------- 1 枚
HEAD_FROM, HEAD_N = 0.60, 3       # 05.tail_whip の終わりの明るさ（実測 約 0.60）


def render(beat, ctx):
    n = beat['n']
    T = timing(beat)
    d = dial()
    od = counter()
    k08, _ = SPC.offsets()
    prog_all = [(v, 1.0) for v in range(0, VMAX + 1, 25)]
    for i in range(n):
        a = alt(i, n)
        vy = CO.K * (float(CO.lnA(alt(i, n))) - float(CO.lnA(alt(i - 1, n))))
        f = CO.frame(a, vy)
        f = core.whip(f, vy * 0.35, axis=0)
        f = villain(f, i, n, k08)
        f = RR.finish_ad(ML.label_scrims(f, 1.0, cg=False), 300 + i, grain=0.006, vignette=0.10, bloom_k=0.06)
        # 入りの振り（中身が下へ）
        if i < 8:
            f = M.whip(f, (i + 1) / 8.0, axis=0, sign=+1, px=160.0, blur=90.0, direction='in')
        if i < HEAD_N:                     # 05 の最後（空へ寄せて約 0.60）から明るさをつなぐ（1 フレーム ≤0.04）
            L = float((f * np.float32([0.2126, 0.7152, 0.0722])).sum(axis=2).mean())
            tgt = HEAD_FROM + (L - HEAD_FROM) * (i + 1) / float(HEAD_N + 1)
            f = f * (tgt / max(L, 1e-3))
        # メーター
        ap = M.ease((i + 1) / 10.0, 'expo_out')
        prog = prog_all if i > 14 else [(v, M.ease((i - 1 - 0.3 * kk) / 7.0, 'back_out'))
                                        for kk, v in enumerate(range(0, VMAX + 1, 25))]
        cs = M.ease((i - 2) / 8.0, 'back_out')
        v, bend = needle(i, T)
        Bd = d.frame(min(v, VMAX), ap=ap, prog=prog, cs=0.0)          # 値の帯（針は下で自分で描く）
        for kk, lb in _T['labels']:
            p = M.ease((i - 2 - 0.3 * kk) / 7.0, 'cubic_out')
            if p > 0.01:
                Bd.over(lb, alpha=p, dy=6 * (1 - p))
        Bd.over(_T['mach'], alpha=ap)
        Bd = draw_pin(Bd, ap)
        Bd = cracks(Bd, i, T)
        if cs > 0.01:
            stub = None if i < T['snap'] else (RD - 44) * 0.55
            Bd = draw_needle(Bd, float(d.th(v)), bend, stub=stub, cs=cs)
        # 折れた瞬間の揺れ・出の落下
        sx, sy, _ = M.shake(i, T['snap'], amp=10.0, tau=3.0, hz=16.0) if i >= T['snap'] else (0.0, 0.0, 0.0)
        ex = M.ease((i - T['exit'] + 1) / 6.0, 'cubic_in') if i >= T['exit'] else 0.0
        if ex > 0:
            Bd = Bd.affine(s=1.0 - 0.08 * ex, pivot=(MC[0], MC[1] - RD / 2))
            Bd = Bd.blur(1.0 + 10 * ex)
        f = Bd.put(f, alpha=1.0 - ex, dx=sx, dy=sy + 160 * ex)
        f = put_broken(f, i, T, dx=sx, dy=sy + 160 * ex)
        # カウンター（1.0 を越えてから）
        cv = counter_v(i, T)
        if cv >= 10.0:
            al = min(1.0, (cv - 10.0) / 6.0)
            exc = M.ease((i - T['exit'] + 1) / 6.0, 'cubic_in') if i >= T['exit'] else 0.0
            f = od.put(f, cv, alpha=al * (1 - exc), i=i, f_lock=T['lock'], dv=cv - counter_v(i - 1, T), dy=-40 * exc)
        yield np.clip(f, 0, 1).astype(np.float32)
