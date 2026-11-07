"""波グループ（b01・b05・b06）の共通部品。

profile(c)         … 波1枚の断面（高さ1・単位系）。c=0 立ち上がった斜面、0.5 リップが出る、1 チューブ、1.4 崩れて着水。
Wave3D             … b01：斜め横から見た、巻いて崩れる巨大な波（断面を奥行き方向に何十枚も並べて描く）。
XSScene            … b05・b06：真横の断面図。深い所から浅瀬へ進んで崩れる波と、砂の斜面。
splat()            … たくさんの点をやわらかい白い粒として描く（泡・しぶき・水の粒）。
"""
import cv2
import numpy as np

import core
from core import W, H


# ---------------------------------------------------------------- 小道具
def smooth(x, a, b):
    t = np.clip((np.asarray(x, np.float32) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def bez(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, np.float32) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def splat(buf, x, y, w):
    """buf（h, w）に点 (x, y) を重み w で双線形に足す（サブピクセルでなめらかに動く）。"""
    h, wd = buf.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx, fy = x - x0, y - y0
    for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        m = (xi >= 0) & (xi < wd) & (yi >= 0) & (yi < h)
        np.add.at(buf, (yi[m], xi[m]), (w * ww)[m])
    return buf


# ---------------------------------------------------------------- 波の断面
# 前面を 3 本のベジェ（外側のリップ・リップの裏・斜面）で表し、キーの間を補間する。
_KEYS = [0.0, 0.5, 1.0, 1.4]
_FRONT = [
    # c = 0：立ち上がった斜面（まだ巻いていない）
    [[(0, 1), (0.12, 1.0), (0.25, 0.93), (0.33, 0.80)],
     [(0.33, 0.80), (0.38, 0.72), (0.42, 0.62), (0.46, 0.52)],
     [(0.46, 0.52), (0.55, 0.30), (0.78, 0.06), (1.45, 0.0)]],
    # c = 0.5：リップが前へ出始める
    [[(0, 1), (0.30, 1.04), (0.55, 0.93), (0.62, 0.72)],
     [(0.62, 0.72), (0.56, 0.73), (0.46, 0.76), (0.38, 0.66)],
     [(0.38, 0.66), (0.31, 0.50), (0.42, 0.10), (1.25, 0.0)]],
    # c = 1：巻いてチューブになる
    [[(0, 1), (0.45, 1.08), (0.86, 0.86), (0.84, 0.40)],
     [(0.84, 0.40), (0.77, 0.62), (0.50, 0.78), (0.31, 0.62)],
     [(0.31, 0.62), (0.19, 0.45), (0.26, 0.08), (1.10, 0.0)]],
    # c = 1.4：リップが水面に落ちる
    [[(0, 0.93), (0.62, 1.00), (1.16, 0.74), (1.22, 0.03)],
     [(1.22, 0.03), (1.10, 0.36), (0.72, 0.60), (0.40, 0.50)],
     [(0.40, 0.50), (0.24, 0.34), (0.30, 0.05), (1.00, 0.0)]],
]
_FRONT = np.array(_FRONT, np.float32)       # (4 keys, 3 pieces, 4 ctrl, 2)
NB, NP = 26, 18                              # 背面・各ベジェの点の数


def front_ctrl(c):
    c = float(np.clip(c, 0, 1.4))
    i = min(int(np.searchsorted(_KEYS, c, side='right')) - 1, len(_KEYS) - 2)
    t = (c - _KEYS[i]) / (_KEYS[i + 1] - _KEYS[i])
    t = t * t * (3 - 2 * t)
    return _FRONT[i] * (1 - t) + _FRONT[i + 1] * t


def profile(c, back=3.6, ext=6.0, under=0.0):
    """閉じた断面（N, 2）＝(u, h)。u は進む向き（右）、h は高さ（波高=1）。底辺は h=0。"""
    k = front_ctrl(c)
    ub = np.linspace(-back, 0, NB, dtype=np.float32)
    hb = ((ub + back) / back) ** 2.4
    hb *= 1 - 0.07 * smooth(c, 1.0, 1.4)
    pts = [np.array([(-ext, -under), (-ext, 0.0)], np.float32), np.stack([ub, hb], 1)[:-1]]
    for p in k:
        pts.append(bez(*p, NP)[:-1])
    pts.append(np.array([k[2][3], (ext, 0.0), (ext, -under)], np.float32))
    return np.concatenate(pts, 0)


def lip_tip(c):
    return front_ctrl(c)[0][3]


def face_point(c, s):
    """斜面（3本目のベジェ）上の点。s=0 リップの付け根、1 波の前の谷。戻り値 (u, h), 接線 (du, dh)。"""
    p = front_ctrl(c)[2]
    t = float(s)
    q = (1 - t) ** 3 * p[0] + 3 * (1 - t) ** 2 * t * p[1] + 3 * (1 - t) * t * t * p[2] + t ** 3 * p[3]
    d = 3 * (1 - t) ** 2 * (p[1] - p[0]) + 6 * (1 - t) * t * (p[2] - p[1]) + 3 * t * t * (p[3] - p[2])
    return q, d


# ================================================================ b01：斜め横から見た巨大な波
SKY_HI = core.XS_SKY_TOP
SKY_LO = core.c('8CC3EA')
HAZE = core.c('9FCBEA')         # 水平線のかすみ
SEA_FAR = core.c('4E95CC')
SEA_NEAR = core.c('0E4A80')
W_DEEP = core.c('0B3F73')
W_MID = core.c('1F6FB0')
W_LIT = core.c('4FA6DE')
W_LIP = core.c('9FD6F2')


def curl3d(z, zp):
    """奥行き z の断面の巻き具合。zp でちょうどチューブ（c=1）、奥ほど崩れ、手前ほど立ち上がる前。"""
    return np.clip(1.0 + (np.asarray(z, np.float32) - zp) * 0.42, 0.18, 1.4).astype(np.float32)


def amp3d(z):
    return (0.05 + 0.95 * smooth(z, 0.9, 2.2)) * (1 - 0.35 * smooth(z, 6.0, 9.0))


class Wave3D:
    """奥行き z（近い 1.15 〜 遠い 9）に K 枚の断面を並べ、遠い方から塗る（画家のアルゴリズム）。
    各画素がどの断面かを idx バッファ（2倍の解像度）に描き、その断面の中の (u, h) から色を決める。
    断面のさかい目が、版画のような細い線になる。"""

    def __init__(self, K=230, yh=1150, camh=0.5, f=800.0, cx=W * 0.5, znear=0.78):
        self.K, self.yh, self.camh, self.f, self.cx = K, yh, camh, f, cx
        # 1/z で並べる（画面上でほぼ等間隔）。チューブのあたり（z 1.5〜3.8）は細かく
        inv = np.concatenate([np.linspace(1 / 9.0, 1 / 3.8, 36, endpoint=False),
                              np.linspace(1 / 3.8, 1 / 1.5, K - 126, endpoint=False),
                              np.linspace(1 / 1.5, 1 / znear, 90)]).astype(np.float32)
        self.z = 1 / inv
        z = self.z
        self.A = amp3d(z)   # 手前の肩は低く
        self.us = self.A ** 0.55                                    # 低い所は横にも短い
        self.haze = (smooth(z, 2.5, 9.0) ** 1.2 * 0.62).astype(np.float32)
        self.bg = self._background()
        rng = np.random.default_rng(7)
        # 海面の細い線（奥行きごとの短い線分）
        n = 520
        self.st_z = 1 / rng.uniform(1 / 20, 1 / 0.9, n).astype(np.float32)
        self.st_u = rng.uniform(-8, 8, n).astype(np.float32)
        self.st_l = rng.uniform(0.08, 0.45, n).astype(np.float32)
        self.st_a = rng.uniform(0.04, 0.12, n).astype(np.float32)
        # しぶき
        m = 2600
        self.sp_z = rng.uniform(0, 1, m).astype(np.float32)
        self.sp_ph = rng.uniform(0, 1, m).astype(np.float32)
        self.sp_v = np.stack([rng.normal(0.35, 0.18, m), rng.normal(0.30, 0.16, m)], 1).astype(np.float32)
        self.sp_kind = (rng.uniform(0, 1, m) < 0.38)                # True：風で後ろへ流れるしぶき
        self.sp_w = rng.uniform(0.25, 1.0, m).astype(np.float32)
        # 泡（崩れた所）
        q = 3400
        self.fo_z = rng.uniform(0, 1, q).astype(np.float32)
        self.fo_r = rng.uniform(0, 1, (q, 2)).astype(np.float32)
        self.fo_ph = rng.uniform(0, 1, q).astype(np.float32)
        self.fo_w = rng.uniform(0.3, 1.0, q).astype(np.float32)

    # -------------------------------------------------- 投影
    def scale(self, z):
        return self.f / z

    def proj(self, u, h, z, uoff):
        s = self.f / z
        return self.cx + (u + uoff) * s, self.yh + (self.camh - h) * s

    def _background(self):
        y = np.arange(H, dtype=np.float32)
        t_sky = np.clip(y / self.yh, 0, 1)[:, None] ** 1.35
        sky = SKY_HI * (1 - t_sky) + SKY_LO * t_sky
        # 海：水平線から手前へ
        zz = np.clip(self.camh * self.f / np.maximum(y - self.yh, 1e-3), 0.3, 60)[:, None]
        t_sea = np.clip(1 - np.log(zz / 0.3) / np.log(200), 0, 1) ** 1.3
        sea = SEA_FAR * (1 - t_sea) + SEA_NEAR * t_sea
        sea = sea * 0.92 + HAZE * 0.08 * (1 - t_sea)
        col = np.where((y > self.yh)[:, None], sea, sky)
        # 水平線のかすみ（白くは光らせない）
        g = np.exp(-((y - self.yh) / 26.0) ** 2)[:, None]
        col = col * (1 - 0.35 * g) + HAZE * 0.35 * g
        bg = np.broadcast_to(col[:, None, :], (H, W, 3)).astype(np.float32).copy()
        # ごく弱い光（上のほうを少し明るく）
        xx = (np.arange(W, dtype=np.float32) - W * 0.7) / W
        yy = (y - 300) / H
        r2 = xx[None, :] ** 2 + yy[:, None] ** 2
        bg[:self.yh] += (np.exp(-r2 / 0.12) * 0.05)[:self.yh, :, None]
        # 海面のやわらかい照り返し（右奥）
        sheen = np.exp(-(((np.arange(W, dtype=np.float32) - W * 0.78) / 260.0) ** 2))[None, :] * \
            np.exp(-((y - self.yh - 70) / 120.0) ** 2)[:, None]
        bg += (sheen * (y > self.yh)[:, None] * 0.07)[..., None] * (HAZE - bg)
        # 遠くの岬（ナザレの灯台のある崖）をかすんだ線画で
        L = core.SS(s=3, w=W, h=200)
        d = L.draw
        yb = 200 - 1
        pts = [(760, yb), (800, yb - 6), (850, yb - 16), (905, yb - 30), (930, yb - 34), (990, yb - 36), (1040, yb - 37), (1090, yb - 38), (1090, yb)]
        d.polygon(L.p(pts), fill=L.col(core.c('7FB3DD'), 1.0))
        hc = L.col(core.c('7FB3DD'), 1.0)
        d.polygon(L.p([(944, yb - 34), (946.5, yb - 55), (951.5, yb - 55), (954, yb - 34)]), fill=hc)   # 灯台
        d.rectangle(L.box(945, yb - 60, 953, yb - 55), fill=hc)
        d.polygon(L.p([(926, yb - 34), (929, yb - 41), (972, yb - 41), (975, yb - 34)]), fill=hc)     # 砦
        d.line(L.p(pts[:-1]), fill=L.col(core.c('B7DAF3'), 0.7), width=L.w(1.2))
        hd = L.done()
        a = hd[..., 3:4]
        bg[self.yh - 200 + 1:self.yh + 1] = bg[self.yh - 200 + 1:self.yh + 1] * (1 - a) + hd[..., :3] * a
        return bg

    def sea_color_rows(self, y0):
        return self.bg[y0:, W // 2]

    # -------------------------------------------------- 状態（時刻 t 秒）
    def state(self, t):
        z = self.z
        zp = 2.95 - 0.10 * t                                # 崩れていく点が手前へ進む
        c = curl3d(z, zp)
        uoff = -0.75 + 0.045 * t                            # 波はゆっくり右へ（カメラもほぼ並走）
        return dict(t=t, c=c, uoff=uoff, zp=zp)

    # -------------------------------------------------- 描画
    def render(self, t):
        st = self.state(t)
        K, z, A, us = self.K, self.z, self.A, self.us
        c, uoff = st['c'], st['uoff']
        S2 = 2
        polys, ymin = [], H
        for k in range(K):
            pr = profile(float(c[k]), under=0.04 / max(float(A[k]), 0.05))
            s = self.f / z[k]
            X = self.cx + (pr[:, 0] * us[k] + uoff) * s
            Y = self.yh + (self.camh - pr[:, 1] * A[k]) * s
            # 底辺は水面 h=0 の高さ
            polys.append(np.stack([X, Y], 1))
            ymin = min(ymin, float(Y.min()))
        y0 = max(0, int(ymin) - 40) & ~1
        rows = H - y0
        idx2 = np.zeros((rows * S2, W * S2), np.uint8)
        for k in range(K):
            P = polys[k].copy()
            P[:, 1] -= y0
            P = np.round(P * S2 * 4).astype(np.int32)
            cv2.fillPoly(idx2, [P], k + 1, lineType=cv2.LINE_8, shift=2)
        # 覆い（1倍）と、色を決めるための半分の解像度の idx（縁は隣の断面の色を使う）
        cov = cv2.resize((idx2 > 0).astype(np.float32), (W, rows), interpolation=cv2.INTER_AREA)
        idxh = cv2.dilate(idx2, np.ones((5, 5), np.uint8))[1::4, 1::4].astype(np.int32)
        # 断面のさかい目（線）
        e = np.zeros_like(idx2, np.float32)
        e[:, 1:] += (idx2[:, 1:] != idx2[:, :-1])
        e[1:, :] += (idx2[1:, :] != idx2[:-1, :])
        e = cv2.resize(np.minimum(e, 1), (W, rows), interpolation=cv2.INTER_AREA)

        hr, hw = idxh.shape
        k1 = np.maximum(idxh - 1, 0)
        ys = (np.arange(hr, dtype=np.float32) * 2 + 0.5 + y0)[:, None]
        xs = (np.arange(hw, dtype=np.float32) * 2 + 0.5)[None, :]
        s_k = (self.f / z).astype(np.float32)[k1]
        u = ((xs - self.cx) / s_k - uoff) / us[k1]
        h = (self.camh - (ys - self.yh) / s_k) / A[k1]
        ck = c[k1]
        hh = np.clip(h, 0, 1.1)
        face = smooth(u, -0.15, 0.25)                           # 前の斜面ほど明るい
        lip = smooth(hh, 0.55, 0.95) * smooth(u, 0.05, 0.6) * smooth(ck, 0.35, 0.9)
        whw = smooth(ck, 1.12, 1.4) * smooth(hh, 0.0, 0.55) * smooth(u, -0.4, 0.4)
        streak = 0.5 + 0.5 * np.cos(h * 34.0 + u * 3.0)
        blend = smooth(h, 0.0, 0.16) * smooth(h * A[k1], 0.0, 0.03)
        hz = self.haze[k1]
        lmask = (0.35 + 0.65 * face) * (1 - hz * 0.9)
        # 色（スカラーの重みを先に作ってから 3 色へ）
        F = face[..., None]
        col = W_DEEP * (1 - hh[..., None]) + W_MID * hh[..., None]
        col = col * (1 - 0.55 * F) + (W_MID * 0.5 + W_LIT * 0.5) * (0.55 * F)
        col += (W_LIT - W_MID) * (face * smooth(hh, 0.25, 0.8) * 0.8 + 0.035 * face * smooth(hh, 0.05, 0.5) * streak)[..., None]
        lp = (0.65 * lip)[..., None]
        col = col * (1 - lp) + W_LIP * lp
        wh = (0.75 * whw)[..., None]
        col = col * (1 - wh) + core.FOAM * wh
        bl = blend[..., None]
        hz3 = hz[..., None]
        seac = self.bg[y0 + 1::2, ::2][:hr, :hw]
        col = (seac * (1 - bl) + col * bl) * (1 - hz3) + HAZE * hz3
        col = cv2.resize(col, (W, rows), interpolation=cv2.INTER_LINEAR)
        lmask = cv2.resize(lmask.astype(np.float32), (W, rows), interpolation=cv2.INTER_LINEAR)
        # 線（水面ぎわの線は消す：高さは 1倍の idx で正確に求める）
        i1 = np.maximum(idx2[1::2, 1::2].astype(np.int32) - 1, 0)
        s1 = (self.f / z).astype(np.float32)[i1]
        h1 = (self.camh - ((np.arange(rows, dtype=np.float32) + y0)[:, None] - self.yh) / s1) / A[i1]
        lmask *= smooth(h1, 0.04, 0.2) * smooth(h1 * A[i1], 0.012, 0.05)
        la = np.clip(e * lmask * 0.51, 0, 0.5)[..., None]
        col = col * (1 - la) + core.FOAM * la

        out = self.bg.copy()
        self._sea_streaks(out, st)
        cv_ = cov[..., None]
        out[y0:] = out[y0:] * (1 - cv_) + col * cv_
        self._spray(out, st)
        return out, st

    def _sea_streaks(self, out, st):
        z = self.st_z
        drift = -0.35 * st['t']                     # 海面の模様は左へ流れる（カメラが右へ並走している）
        u = (self.st_u + drift + 8) % 16 - 8
        s = self.f / z
        Xc = self.cx + (u + self.st_l / 2) * s
        Lh = np.minimum(self.st_l * s, 18 + 0.12 * s) / 2           # 画面での長さを抑える
        Y = self.camh * s
        L = np.zeros((H - self.yh, W), np.float32)
        near = np.clip((s - 200) / 900, 0, 1)
        for xc, lh, y, a, nr in zip(Xc, Lh, Y, self.st_a, near):
            if y > H - self.yh + 2 or xc + lh < -10 or xc - lh > W + 10:
                continue
            a = float(a * (1 - 0.5 * nr)) / 3
            for q in (1.0, 0.66, 0.33):                                # 両端を細く（先細りの線）
                cv2.line(L, (int((xc - lh * q) * 4), int(y * 4)), (int((xc + lh * q) * 4), int(y * 4)), a, 1, cv2.LINE_AA, shift=2)
        o = out[self.yh:]
        o += (L[..., None] * (core.FOAM - o) * 0.9)

    def _spray(self, out, st):
        t = st['t']
        zr = self.z
        # しぶきは崩れかけ〜チューブの区間（c 0.6〜1.35）のリップの先から出る
        zs = st['zp'] - 0.95 + 1.8 * self.sp_z
        cz = curl3d(zs, st['zp'])
        A = amp3d(zs)
        us = A ** 0.55
        life = 1.3
        age = ((t / life + self.sp_ph) % 1.0) * life
        tip = np.stack([lip_tip(float(cc)) for cc in cz], 0)
        crest = np.zeros_like(tip)
        crest[:, 1] = 1.0
        base = np.where(self.sp_kind[:, None], crest + np.array([0.05, 0.0], np.float32), tip)
        v = self.sp_v.copy()
        v[self.sp_kind, 0] = -np.abs(v[self.sp_kind, 0]) * 1.6 - 0.15     # 後ろへ（風）
        v[self.sp_kind, 1] = np.abs(v[self.sp_kind, 1]) * 0.9 + 0.12
        uu = base[:, 0] + v[:, 0] * age
        hh = base[:, 1] + v[:, 1] * age - 0.42 * age * age * (~self.sp_kind) - 0.08 * age * age * self.sp_kind
        s = self.f / zs
        X = self.cx + (uu * us + st['uoff']) * s
        Y = self.yh + (self.camh - hh * A) * s
        live = np.sin(np.pi * age / life) ** 1.5 * smooth(cz, 0.55, 0.85) * (1 - smooth(zs, 6.5, 9))
        w = self.sp_w * live * np.where(self.sp_kind, 0.8, 1.0)
        buf = splat(np.zeros((H, W), np.float32), X, Y, w * 0.9)
        # 崩れた所の泡のかたまり
        fz = st['zp'] + 0.55 + 3.4 * self.fo_z
        fcz = curl3d(fz, st['zp'])
        fA = amp3d(fz)
        fus = fA ** 0.55
        r = self.fo_r
        boil = 0.5 + 0.5 * np.sin(2 * np.pi * (self.fo_ph + t * 0.6))
        fu = 0.15 + 1.25 * r[:, 0] + 0.08 * np.sin(6.28 * (self.fo_ph * 3 + t * 0.4))
        fh = (r[:, 1] ** 1.6) * (0.75 - 0.35 * r[:, 0]) * (0.75 + 0.25 * boil)
        s = self.f / fz
        X = self.cx + (fu * fus + st['uoff']) * s
        Y = self.yh + (self.camh - fh * fA) * s
        w = self.fo_w * smooth(fcz, 1.2, 1.4) * (1 - smooth(fz, 7.0, 9.5)) * 0.8
        splat(buf, X, Y, w)
        soft = cv2.GaussianBlur(buf, (0, 0), 1.1)
        glow = cv2.GaussianBlur(buf, (0, 0), 4.0)
        a = np.clip(soft * 1.4 + glow * 0.9, 0, 1)[..., None]
        out[:] = out * (1 - a) + core.FOAM * a

    # -------------------------------------------------- サーファー
    def surfer_at(self, t, z, sf):
        """断面 z（c は state(t) から補間）の斜面のパラメータ sf の点に立つサーファーの画面位置と角度。"""
        st = self.state(t)
        zp = st['zp']
        c = float(curl3d(z, zp))
        A = float(amp3d(z))
        us = A ** 0.55
        (u, h), (du, dh) = face_point(c, sf)
        s = self.f / z
        X = self.cx + (u * us + st['uoff']) * s
        Y = self.yh + (self.camh - h * A) * s
        ang = np.degrees(np.arctan2(-dh * A, du * us))
        return float(X), float(Y), float(ang), s * A


# ---------------------------------------------------------------- サーファーを小さなレイヤーで描いて重ねる
def draw_surfer(frame, x, y, scale=2.0, ang=0.0, alpha=1.0, pose='ride', head=0.0, shadow=0.35, box=None):
    """frame に core.surfer を重ねる（小さな SS レイヤーで描くので速い）。head：頭の傾き（度、pose='hold' のとき）。
    影：ごく弱い暗いにじみ（白いピクトグラムを明るい背景でも読めるように）。"""
    if alpha <= 0.003:
        return frame
    R = int(box or 60 * scale)
    L = core.SS(s=4, w=2 * R, h=2 * R)
    core.surfer(L, R, R, scale=scale, ang=ang, pose=pose)
    if head and pose == 'hold':
        # 頭だけ首の付け根を中心に傾ける
        r = np.deg2rad(ang)
        Rm = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
        hx, hy = Rm @ np.array([-1, -25.5]) * scale + (R, R)
        nx, ny = Rm @ np.array([-1, -21.0]) * scale + (R, R)
        rr = 3.4 * scale
        L.draw.ellipse(L.box(hx - rr - 1, hy - rr - 1, hx + rr + 1, hy + rr + 1), fill=(0, 0, 0, 0))
        core.surfer.__globals__  # noqa（同じ見た目の頭を描き直す）
        L.draw.line(L.p([(nx, ny), (nx, ny + 0.0)]), fill=L.col(core.FOAM), width=L.w(3.2 * scale))
        q = np.deg2rad(head)
        vx, vy = hx - nx, hy - ny
        hx2 = nx + vx * np.cos(q) - vy * np.sin(q)
        hy2 = ny + vx * np.sin(q) + vy * np.cos(q)
        L.draw.ellipse(L.box(hx2 - rr, hy2 - rr, hx2 + rr, hy2 + rr), fill=L.col(core.FOAM))
    lay = L.done()
    x0, y0 = int(round(x)) - R, int(round(y)) - R
    fx, fy = x - round(x), y - round(y)
    if abs(fx) > 1e-3 or abs(fy) > 1e-3:
        M = np.array([[1, 0, fx], [0, 1, fy]], np.float32)
        lay = cv2.warpAffine(lay, M, (2 * R, 2 * R), flags=cv2.INTER_LINEAR)
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + 2 * R, frame.shape[1]), min(y0 + 2 * R, frame.shape[0])
    if xb <= xa or yb <= ya:
        return frame
    sub = lay[ya - y0:yb - y0, xa - x0:xb - x0]
    a = sub[..., 3:4] * alpha
    reg = frame[ya:yb, xa:xb]
    if shadow:
        sa = cv2.GaussianBlur(sub[..., 3], (0, 0), 2.2 * scale / 2)[..., None] * alpha * shadow
        reg = reg * (1 - sa) + core.c('062A4E') * sa
    frame[ya:yb, xa:xb] = reg * (1 - a) + sub[..., :3] * a
    return frame
