"""案B 04 のスピードメーター（ガラスの文字盤＋クロームの縁）。look-dev v2（美術監督のレビュー 04 に合わせて）。

b04 の組み上がりの動き（半円が 0→1,500 へ描かれる・目盛りが外へぽんと出る・キャップと針が back_out で出る）を
そのまま受け取れるように、frame() に進み（ap、目盛りごとの p、cs）を渡す。戻り値は _m34.Board（前乗せの sRGB）。
- 縁：丸い断面のクローム（環境の帯の映り込み＝空のグラデーション＋やわらかい明るい帯 2 本、地面側は暗く、
  境に暗い芯の線）、円周の向きのヘアライン（異方性）、両端は丸いキャップ。いちばん明るい所でも 0.80 以下。
- 文字盤：濃い紺のガラス（中心ほどわずかに明るい・縁へ暗く）、ごく細い同心のヘアライン。
- ガラスの映り込み：左上の 4 分の 1 だけ、ドームに沿って曲がったやわらかい帯（α≤0.08）。
- 値の帯：根元の深い青 → 先 #6FB8E8（強調色は使わない）。にじみは半径 10px・0.15。越えた分（900→）は白へ寄せる。
- 目盛り：浮き彫り（左上に 1px の明るい線、右下に 1px の暗い線）。
- 針：先細り・左右 2 色の面取り、文字盤に落ちる 6px ずれた影（0.35）、軸のキャップは 2 倍（半径 26）でつやの点。
- 計器の下に落ちるやわらかい影（地図から浮かせる）。
太陽は全部のショットと同じ左上。"""
import math

import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from cutB.beats import _render as R
from cutB.beats._m34 import Board

LV = np.array([-0.55, -0.65, 0.52], np.float32)
LV /= np.linalg.norm(LV)
BAND_TIP = '6FB8E8'
BAND_BASE = '24639C'


def _env(ry, rx=None):
    """クロームに映る環境（線形）。ry：反射の上向きの成分（-1..1）。空のグラデーション＋やわらかい帯 2 本、
    地平の下は暗い地面、境に暗い芯。"""
    sky = R.hexl('7FA6C6') * (1 - np.clip(ry, 0, 1))[..., None] + R.hexl('B3CDE2') * np.clip(ry, 0, 1)[..., None]
    gnd = R.hexl('0A1828') * (1 - np.clip(-ry, 0, 1))[..., None] + R.hexl('06101C') * np.clip(-ry, 0, 1)[..., None]
    k = R.smoothstep(-0.06, 0.06, ry)[..., None]
    col = gnd * (1 - k) + sky * k
    core_line = np.exp(-((ry - 0.0) / 0.035) ** 2)[..., None]
    col = col * (1 - 0.65 * core_line)
    b1 = np.exp(-((ry - 0.55) / 0.10) ** 2)[..., None]
    b2 = np.exp(-((ry - 0.22) / 0.05) ** 2)[..., None]
    col = col + R.hexl('DDEBF6') * (0.35 * b1 + 0.22 * b2)
    return col


class Dial:
    def __init__(self, center, R_, vmax=1500, pad=96, ss=2):
        self.cx, self.cy, self.R, self.vmax = float(center[0]), float(center[1]), float(R_), vmax
        self.pad = pad
        self.x0, self.y0 = int(self.cx - R_ - pad), int(self.cy - R_ - pad)
        self.w, self.h = int(2 * (R_ + pad)), int(R_ + pad + 60)
        self.ss = ss
        self._tick_cache = None
        self._static()

    def th(self, v):
        return np.pi * (1 - np.clip(v, -60, self.vmax) / self.vmax)

    # ------------------------------------------------------------ 動かない部分
    def _grid(self, s):
        yy, xx = np.mgrid[0:self.h * s, 0:self.w * s].astype(np.float32)
        xx = (xx + 0.5) / s
        yy = (yy + 0.5) / s
        dx, dy = xx - (self.cx - self.x0), yy - (self.cy - self.y0)
        r = np.sqrt(dx * dx + dy * dy) + 1e-6
        return dx, dy, r, np.arctan2(-dy, dx)

    def _static(self):
        s = self.ss
        Rr = self.R
        dx, dy, r, ang = self._grid(s)
        # ---- 文字盤（上の半円＋下の 6px の縁まで）
        Rf = Rr + 4
        cut = np.clip(6 - dy + 0.5, 0, 1)
        fm = np.clip(Rf - r + 0.5, 0, 1) * cut
        base = R.hexl('06192C') * (1.0 + 0.9 * np.exp(-(r / (Rr * 0.62)) ** 2))[..., None]
        base = base * (0.62 + 0.38 * R.smoothstep(Rf, Rf - 90, r))[..., None]
        hair = np.sin(r * 2.4) * 0.5 + 0.5
        base = base * (1 + 0.035 * hair)[..., None]
        # ガラスの厚み（縁のすぐ内側の細い明るい線、上だけ）
        base = base + R.hexl('8FB9DA') * (0.05 * np.exp(-((r - (Rf - 3)) / 1.4) ** 2) * (dy < 0))[..., None]
        col = base * fm[..., None]
        a = fm.copy()
        # ---- 縁（クローム）：中心線の半径 Rc、半分の太さ hw、0..π の弧＋両端の丸いキャップ（SDF）
        Rc, hw = Rr + 12, 9.0
        on_arc = dy <= 0
        d_arc = np.abs(r - Rc)
        ex, ey = np.where(dx >= 0, Rc, -Rc), 0.0
        d_end = np.sqrt((dx - ex) ** 2 + (dy - ey) ** 2)
        d = np.where(on_arc, d_arc, d_end)
        # 断面の向き（面の中の単位ベクトル）
        ux = np.where(on_arc, dx / r * np.sign(r - Rc + 1e-6), (dx - ex) / np.maximum(d_end, 1e-6))
        uy = np.where(on_arc, dy / r * np.sign(r - Rc + 1e-6), (dy - ey) / np.maximum(d_end, 1e-6))
        q = np.clip(d / hw, 0, 1)
        nz = np.sqrt(np.clip(1 - q * q, 0, 1))
        nx, ny = ux * q, uy * q
        ry = -(2 * nz * ny)                                     # 反射（視線 0,0,1）の上向き成分（画面の y は下）
        env = _env(ry)
        # 円周の向きのヘアライン（異方性の筋）
        streak = R.fbm(ang * 180.0, q * 2.0 + r * 0.02, 2, seed=83)
        env = env * (1 + 0.07 * streak)[..., None]
        Hh = LV + np.array([0, 0, 1], np.float32)
        Hh /= np.linalg.norm(Hh)
        ndh = np.clip(nx * Hh[0] + ny * Hh[1] + nz * Hh[2], 0, 1)
        spec = ndh ** 60 * 0.55 + ndh ** 12 * 0.08
        metal = env * 0.85 + R.SUN_RGB * spec[..., None]
        # 縁の AO（文字盤との境で暗く）
        metal = metal * (0.80 + 0.20 * nz)[..., None]
        ringa = np.clip((hw - d) * 1.0 + 0.5, 0, 1)
        # 縁が文字盤に落とす内側の影
        inner = np.exp(-np.clip(Rc - hw - r, 0, None) / 6.0) * (r < Rc - hw) * (dy < 0) * 0.45
        col = col * (1 - inner[..., None])
        # 明るさの上限（0.80 sRGB）
        lim = R.lin(np.float32(0.80))
        l = R.luma(metal)
        metal = metal * np.minimum(1.0, lim / np.maximum(l, 1e-5))[..., None]
        # ---- ガラスの映り込み（2026-11-08 美術監督のレビュー：やわらかい帯は汚れに見えたので、縁の弧に沿った
        #      くっきりした 2 本の反射＋上の縁のフレネルの線に）。帯 1：1.5px・8%、帯 2：6px・3%（左上の弧）
        def arc_band(rc, wpx, a0, a1, fade):
            prof = np.clip(wpx / 2.0 - np.abs(r - rc) + 0.5, 0, 1)            # くっきりした縁（1px の AA）
            angw = R.smoothstep(a0, a0 + fade, ang) * (1 - R.smoothstep(a1 - fade, a1, ang))
            return prof * angw * (dy < 0)
        b1 = arc_band(Rr * 0.88, 1.5, np.pi * 0.56, np.pi * 0.86, np.pi * 0.08)
        b2 = arc_band(Rr * 0.83, 6.0, np.pi * 0.60, np.pi * 0.80, np.pi * 0.07)
        fres = np.clip(0.9 - np.abs(r - (Rf - 4.5)) + 0.5, 0, 1) * R.smoothstep(np.pi * 0.08, np.pi * 0.30, ang) * \
            (1 - R.smoothstep(np.pi * 0.70, np.pi * 0.92, ang)) * (dy < 0)
        self.glass = cv2.resize(((0.08 * b1 + 0.03 * b2 + 0.05 * fres) * fm).astype(np.float32), (self.w, self.h),
                                interpolation=cv2.INTER_AREA)
        # ---- 計器の下の影（地図から浮かせる）
        sh = cv2.GaussianBlur(np.clip((Rc + hw - r) * 1.0, 0, 1) * cut, (0, 0), 18 * s)
        sh = cv2.warpAffine(sh.astype(np.float32), np.float32([[1, 0, 10 * s], [0, 1, 16 * s]]), (sh.shape[1], sh.shape[0]))
        # 縮めて保存
        dn = lambda x: cv2.resize(x.astype(np.float32), (self.w, self.h), interpolation=cv2.INTER_AREA)
        self.face_col = dn(col * a[..., None])           # 前乗せ（線形）
        self.face_a = dn(a)
        self.bez_col = dn(metal * ringa[..., None])
        self.bez_a = dn(ringa)
        self.bez_ang = None
        self.drop = dn(sh) * 0.30
        d1, e1, r1, a1 = self._grid(1)
        self._dx, self._dy, self._r, self._ang = d1, e1, r1, a1

    # ------------------------------------------------------------ 目盛り（浮き彫り）
    def _ticks(self, prog, jet, jet_len, jet_a, jet_p):
        """prog：目盛りごとの (v, p)。戻り値：(明るい線の α, 影の α, 光の線の α)。"""
        key = None
        if all(p >= 0.999 for _, p in prog):
            key = (round(jet_len, 2), round(jet_a, 3), round(jet_p, 3))
            if self._tick_cache is not None and self._tick_cache[0] == key:
                return self._tick_cache[1]
        Rr = self.R
        S3 = 3
        Lm, Ld, Ll = (core.SS(s=S3, w=self.w, h=self.h) for _ in range(3))
        cx, cy = self.cx - self.x0, self.cy - self.y0
        for v, p in prog:
            if p <= 0.01:
                continue
            t_ = self.th(v)
            major, mid = v % 300 == 0, v % 100 == 0
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
        if jet_p > 0.01:
            t_ = self.th(jet)
            r0, r1 = Rr + 24, Rr + 24 + jet_len * jet_p
            pts = [(cx + r0 * np.cos(t_), cy - r0 * np.sin(t_)), (cx + r1 * np.cos(t_), cy - r1 * np.sin(t_))]
            Lm.draw.line(Lm.p(pts), fill=Lm.col((1, 1, 1), jet_a * min(1.0, jet_p * 1.6)), width=Lm.w(2.2))
            Ld.draw.line(Ld.p([(x + 0.9, y + 1.2) for x, y in pts]), fill=Ld.col((1, 1, 1), jet_a * min(1.0, jet_p * 1.6)),
                         width=Ld.w(2.4))
        out = (Lm.done()[..., 3], cv2.GaussianBlur(Ld.done()[..., 3], (0, 0), 0.6), Ll.done()[..., 3])
        if key is not None:
            self._tick_cache = (key, out)
        return out

    # ------------------------------------------------------------ 針のぶれ・旅客機の形
    def _smear(self, col, a, smear, cs):
        """針が v_from → v_to へ動いた半フレームの間を、1 枚のくさびで（針の所 0.6 → 後ろ 0）。"""
        v0, v1 = smear
        t0, t1 = self.th(v0), self.th(v1)
        span = t0 - t1
        if abs(span) < 1e-4:
            return col, a
        dy, r, ang = self._dy, self._r, self._ang
        q = (ang - t1) / span                                      # 0＝今の針、1＝半フレーム前
        rn = (self.R - 44) * cs
        rad = np.clip(rn - r + 0.5, 0, 1) * R.smoothstep(14.0, 30.0, r)
        w = np.clip(1 - q, 0, 1) ** 1.8 * (1 - R.smoothstep(0.0, 0.03, -q))
        m = (0.6 * w * rad * (dy <= 0.5) * self.face_a).astype(np.float32)
        nc = R.hexl('BCCFDE')
        col = col * (1 - m[..., None]) + nc * m[..., None]
        return col, np.maximum(a, m)

    def _jet_glyph(self, col, a, jet, jet_len, jet_a, jet_p):
        """旅客機の目盛り（外へのびる線）の先に、真上から見た小さな旅客機の形（約 30px、機首は値が増える向き＝時計回り）。"""
        t_ = self.th(jet)
        ux, uy = math.cos(t_), -math.sin(t_)                     # 外向き
        tx, ty = -uy, ux                                         # 時計回り（値が増える向き）
        rr = self.R + 24 + jet_len * jet_p + 20
        cx, cy = self.cx - self.x0 + rr * ux, self.cy - self.y0 + rr * uy
        s_ = 15.0 * min(1.0, jet_p * 1.2)
        # 機体の形（機首 +x、右 +y の単位）：胴体・後退翼・水平尾翼
        half = [(1.00, 0.00), (0.80, 0.10), (0.18, 0.12), (-0.18, 0.88), (-0.34, 0.88), (-0.14, 0.12), (-0.66, 0.10),
                (-0.86, 0.38), (-0.98, 0.38), (-0.90, 0.00)]
        pts = half + [(x, -y) for x, y in half[-2:0:-1]]
        pts = [(cx + s_ * (x * tx + y * ux), cy + s_ * (x * ty + y * uy)) for x, y in pts]
        L = core.SS(s=3, w=self.w, h=self.h)
        L.draw.polygon(L.p(pts), fill=(255, 255, 255, 255))
        m = L.done()[..., 3] * (jet_a * min(1.0, jet_p * 1.6) * 0.9)
        sh = cv2.GaussianBlur(cv2.warpAffine(m, np.float32([[1, 0, 2], [0, 1, 3]]), (self.w, self.h)), (0, 0), 1.5) * 0.35
        col = col * (1 - sh[..., None])
        a = np.maximum(a, sh)
        c = R.lin(np.asarray(core.FOAM, np.float32)) * 0.80
        col = col * (1 - m[..., None]) + c * m[..., None]
        return col, np.maximum(a, m)

    # ------------------------------------------------------------ 1 フレーム
    def frame(self, value, ap=1.0, prog=(), cs=1.0, ghosts=(), excess=0.0, jet=900.0, jet_len=34.0, jet_a=1.0,
              jet_p=1.0, smear=None, jet_glyph=False):
        # smear：(v_from, v_to) 針の動きのぶれ（1 枚のくさび、針の所 0.6 → 後ろ 0）。jet_glyph：旅客機の目盛りの先に旅客機の形
        """ap：半円の描かれた割合（0..1、左から）。prog：目盛りの (v, p)。cs：針とキャップの大きさ（back_out）。"""
        Rr = self.R
        dx, dy, r, ang = self._dx, self._dy, self._r, self._ang
        # 縁と文字盤は左から ap まで（4° のやわらかい前縁）。文字盤の濃さも ap で
        frac = (np.pi - np.clip(ang, 0, np.pi)) / np.pi
        frac = np.where(dy > 0, np.where(dx < 0, 0.0, 1.0), frac)
        if ap < 0.999:
            m = np.clip((ap - frac) / 0.022 + 0.5, 0, 1) * (1.0 if ap > 0.002 else 0.0)
            fa = float(R.smoothstep(0.0, 0.8, ap))
        else:
            m, fa = None, 1.0
        # 文字盤はふわっと出る（ap）、縁は左から描かれる（m）
        bm_ = self.bez_a if m is None else self.bez_a * m
        bc_ = self.bez_col if m is None else self.bez_col * m[..., None]
        a = self.face_a * fa
        col = self.face_col * fa
        col = col * (1 - bm_[..., None]) + bc_
        a = a + bm_ * (1 - a)
        col = col / np.maximum(a, 1e-4)[..., None]
        # 値の帯（内側、半径 R-14、太さ 7）
        if value > 0:
            rr = Rr - 14
            band = np.clip(3.6 - np.abs(r - rr) + 0.5, 0, 1) * (dy <= 0.5)
            t_v = self.th(value)
            endw = np.clip((ang - t_v) * r + 0.5, 0, 1)
            bm = band * endw
            g = np.clip(frac / max(value / self.vmax, 1e-3), 0, 1) ** 1.2
            bandc = R.hexl(BAND_BASE) * (1 - g)[..., None] + R.hexl(BAND_TIP) * g[..., None]
            if excess > 0.002 and value > jet:
                tj = self.th(jet)
                te = self.th(jet + (min(value, 1130.0) - jet) * excess)
                ex = np.clip((tj - ang) * r + 0.5, 0, 1) * np.clip((ang - te) * r + 0.5, 0, 1)
                k_ex = (ex * min(1.0, excess * 1.5) * 0.75)[..., None]
                bandc = bandc * (1 - k_ex) + R.hexl('DCEEFA') * k_ex
            glow = cv2.GaussianBlur((bm * (0.3 + 0.7 * g)).astype(np.float32), (0, 0), 10.0) * 0.15
            col = col + R.hexl(BAND_TIP) * (glow * self.face_a)[..., None]
            col = col * (1 - bm[..., None] * 0.92) + bandc * (bm * 0.92)[..., None]
        # 目盛り（浮き彫り）
        tm, td, tl = self._ticks(prog, jet, jet_len, jet_a, jet_p)
        col = col * (1 - (td * 0.55)[..., None])
        col = col + (R.hexl('D6E6F2') - col) * (tl * 0.35)[..., None]
        col = col + (R.lin(np.asarray(core.FOAM, np.float32)) * 0.80 - col) * tm[..., None]
        a = np.maximum(a, np.maximum(tm, td * 0.55))
        # 針（影 → 本体 2 色）・軸のキャップ
        cx, cy = self.cx - self.x0, self.cy - self.y0
        if smear is not None and cs > 0.01:
            col, a = self._smear(col, a, smear, cs)
        if jet_glyph and jet_p > 0.01:
            col, a = self._jet_glyph(col, a, jet, jet_len, jet_a, jet_p)
        if cs > 0.01:
            for gv, ga in list(ghosts) + [(value, 1.0)]:
                t_ = self.th(gv)
                rn = (Rr - 44) * cs
                ux, uy = math.cos(t_), -math.sin(t_)
                nx, ny = -uy, ux
                tip = (cx + rn * ux, cy + rn * uy)
                bse = (cx - 26 * cs * ux, cy - 26 * cs * uy)
                hb, ht = 6.5 * cs, 1.1 * cs
                L1 = core.SS(s=3, w=self.w, h=self.h)
                L2 = core.SS(s=3, w=self.w, h=self.h)
                left = [(bse[0] + nx * hb, bse[1] + ny * hb), (tip[0] + nx * ht, tip[1] + ny * ht), tip, bse]
                right = [bse, tip, (tip[0] - nx * ht, tip[1] - ny * ht), (bse[0] - nx * hb, bse[1] - ny * hb)]
                L1.draw.polygon(L1.p(left), fill=(255, 255, 255, 255))
                L2.draw.polygon(L2.p(right), fill=(255, 255, 255, 255))
                m1, m2 = L1.done()[..., 3], L2.done()[..., 3]
                mm = np.clip(m1 + m2, 0, 1)
                # 光（左上）に向く側が明るい
                lit_left = (nx * LV[0] + ny * LV[1]) > 0
                cA, cB = (R.hexl('E4EEF6'), R.hexl('8EA7BC')) if lit_left else (R.hexl('8EA7BC'), R.hexl('E4EEF6'))
                if ga >= 0.999:
                    sh = cv2.warpAffine(mm, np.float32([[1, 0, 6], [0, 1, 6]]), (self.w, self.h))
                    sh = cv2.GaussianBlur(sh, (0, 0), 2.5) * 0.35 * self.face_a
                    col = col * (1 - sh[..., None])
                    nc = cA * m1[..., None] + cB * m2[..., None]
                    col = col * (1 - mm[..., None]) + nc
                    a = np.maximum(a, mm)
                else:
                    nc = (R.hexl('BCCFDE') * mm[..., None]) * ga
                    col = col * (1 - (mm * ga)[..., None]) + nc
                    a = np.maximum(a, mm * ga)
            # キャップ：半径 26 の金属のドーム＋つやの点（左上）
            rc = 26.0 * cs
            dd = r / rc
            capm = np.clip((rc - r) + 0.5, 0, 1)
            nzc = np.sqrt(np.clip(1 - dd * dd, 0, 1))
            nxc, nyc = dx / rc, dy / rc
            ryc = -(2 * nzc * nyc)
            envc = _env(ryc) * 0.9
            Hh = LV + np.array([0, 0, 1], np.float32)
            Hh /= np.linalg.norm(Hh)
            spc = np.clip(nxc * Hh[0] + nyc * Hh[1] + nzc * Hh[2], 0, 1)
            capc = envc + R.SUN_RGB * (spc ** 80 * 0.9)[..., None]
            capsh = cv2.GaussianBlur(np.clip((rc - np.hypot(dx - 5, dy - 7)), 0, 1).astype(np.float32), (0, 0), 4) * 0.45
            col = col * (1 - capsh[..., None])
            col = col * (1 - capm[..., None]) + capc * capm[..., None]
            a = np.maximum(a, capm)
            ring = np.clip(1.2 - np.abs(r - rc * 0.42), 0, 1) * capm * 0.25
            col = col * (1 - ring[..., None])
        # ガラスの映り込み（いちばん上）
        gl = self.glass * fa
        col = col + R.hexl('DDEFFB') * gl[..., None]
        rgb = R.srgb(col)
        # 計器の下の影（α を持つ黒＝前乗せの rgb は 0）
        dr = self.drop * fa
        a_tot = a + dr * (1 - a)
        return Board(rgb * a[..., None], a_tot, self.x0, self.y0)
