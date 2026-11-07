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
        self.cx0 = cx
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
        uoff = -0.78 + 0.06 * t                            # 波はゆっくり右へ（カメラもほぼ並走）
        return dict(t=t, c=c, uoff=uoff, zp=zp)

    # -------------------------------------------------- 描画
    def render(self, t, pan=0.0):
        """pan：カメラを右へ振った量（px）。画は左へ動き、右から新しい海と空が入る。"""
        self.cx = self.cx0 - pan
        bg = self.bg
        if pan:
            M = np.array([[1, 0, -pan], [0, 1, 0]], np.float32)
            bg = cv2.warpAffine(self.bg, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
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
        seac = bg[y0 + 1::2, ::2][:hr, :hw]
        col = seac * (1 - bl) + (col * (1 - hz3) + HAZE * hz3) * bl
        col = cv2.resize(col, (W, rows), interpolation=cv2.INTER_LINEAR)
        lmask = cv2.resize(lmask.astype(np.float32), (W, rows), interpolation=cv2.INTER_LINEAR)
        # 線（水面ぎわの線は消す：高さは 1倍の idx で正確に求める）
        i1 = np.maximum(idx2[1::2, 1::2].astype(np.int32) - 1, 0)
        s1 = (self.f / z).astype(np.float32)[i1]
        h1 = (self.camh - ((np.arange(rows, dtype=np.float32) + y0)[:, None] - self.yh) / s1) / A[i1]
        lmask *= smooth(h1, 0.04, 0.2) * smooth(h1 * A[i1], 0.012, 0.05)
        la = np.clip(e * lmask * 0.51, 0, 0.5)[..., None]
        col = col * (1 - la) + core.FOAM * la

        out = bg.copy()
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
    def surfer_at(self, t, z, sf, pan=0.0):
        """断面 z（c は state(t) から補間）の斜面のパラメータ sf の点に立つサーファーの画面位置と角度。"""
        st = self.state(t)
        self.cx = self.cx0 - pan
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


# ================================================================ 断面図（真横）：浅瀬で崩れる波（b05、b06 の巻き戻し）
def mask_poly(pts, h, w, y0=0):
    """多角形の AA マスク（float 0..1、形 (h, w)、上端は y0）。"""
    m = np.zeros((h, w), np.uint8)
    P = np.round((np.asarray(pts, np.float32) - (0, y0)) * 4).astype(np.int32)
    cv2.fillPoly(m, [P], 255, lineType=cv2.LINE_AA, shift=2)
    return m.astype(np.float32) / 255


def mask_line(pts, h, w, th=1, y0=0):
    m = np.zeros((h, w), np.uint8)
    P = np.round((np.asarray(pts, np.float32) - (0, y0)) * 4).astype(np.int32)
    cv2.polylines(m, [P], False, 255, thickness=th, lineType=cv2.LINE_AA, shift=2)
    return m.astype(np.float32) / 255


def keys(x, ks, vs, ease_=True):
    """キーフレーム補間（区間ごとに smoothstep）。"""
    x = float(x)
    if x <= ks[0]:
        return float(vs[0])
    if x >= ks[-1]:
        return float(vs[-1])
    i = int(np.searchsorted(ks, x, side='right')) - 1
    t = (x - ks[i]) / (ks[i + 1] - ks[i])
    if ease_:
        t = t * t * (3 - 2 * t)
    return float(vs[i] + (vs[i + 1] - vs[i]) * t)


_P0 = profile(0.0)
_P0_TOP = _P0[1:-1]          # c=0 は上面が u について一価


def shape_xs(m):
    """m ∈ [-1, 1.4]。-1：沖のなだらかな山（左右対称）→ 0：前が立った斜面 → 1：チューブ → 1.4：着水。"""
    if m >= 0:
        return profile(m)
    u = np.linspace(-3.6, 3.6, 140, dtype=np.float32)
    hs = 1 / np.cosh(u / 0.9) ** 2
    hp = np.interp(u, _P0_TOP[:, 0], _P0_TOP[:, 1]).astype(np.float32)
    k = float(-m)
    k = k * k * (3 - 2 * k)
    h = hp * (1 - k) + hs * k
    return np.concatenate([[(-6, 0)], np.stack([u, h], 1), [(6, 0)]]).astype(np.float32)


def surf_point_xs(m, us=0.56):
    """サーファーの立つ点（単位系）。m<0 は一価の面の u=us、m>=0 は斜面のベジェ上（つながるように）。"""
    if m < 0:
        P = shape_xs(m)[1:-1]
        u0 = us
        h = float(np.interp(u0, P[:, 0], P[:, 1]))
        h2 = float(np.interp(u0 + 0.04, P[:, 0], P[:, 1]))
        return np.array([u0, h], np.float32), np.array([0.04, h2 - h], np.float32)
    sf = 0.25 + 0.40 * float(smooth(m, 0.0, 1.0))
    (q, d) = face_point(min(m, 1.0), sf)
    return q, d


XS_LIT = core.c('4A9ADA')       # 波の上のほうの水の色
SHORE = 860                     # 水面と砂浜の交わる x


def bed_y(x):
    x = np.asarray(x, np.float32)
    deep = core.WL + 700 * np.clip((SHORE - x) / SHORE, 0, None) ** 1.15
    beach = core.WL - (x - SHORE) * 0.42
    return np.where(x < SHORE, deep, beach)


class XSScene:
    """真横の断面図。frame(fi) で fi（小数可、0..n-1）の絵を返す。波のパラメータは keys() のキーフレーム。"""
    KF = [0, 20, 33, 40, 46, 56, 70]

    def __init__(self, n=80):
        self.n = n
        WL = core.WL
        self.bg = core.xs_background(WL)
        # 砂（青灰色）：斜面＋細い等間隔の線（地層のような線画）
        xs = np.arange(-20, W + 21, 6, dtype=np.float32)
        top = bed_y(xs)
        poly = np.concatenate([np.stack([xs, top], 1), [(W + 20, H + 20), (-20, H + 20)]])
        sm = mask_poly(poly, H, W)
        yy = np.arange(H, dtype=np.float32)[:, None]
        depth = np.clip((yy - bed_y(np.arange(W))[None, :]) / 500, 0, 1)
        sand = core.SAND[None, None] * (1 - 0.35 * depth[..., None]) + core.c('3E5A73') * 0.35 * depth[..., None]
        lines = np.zeros((H, W), np.float32)
        for k in range(1, 9):
            lines = np.maximum(lines, mask_line(np.stack([xs, top + k * 34 + 6 * k * k * 0.1], 1), H, W, 1) * (0.22 - k * 0.02))
        sand = sand * (1 - lines[..., None]) + core.FOAM * lines[..., None]
        edge = mask_line(np.stack([xs, top], 1), H, W, 2) * 0.75
        self.sand = np.concatenate([sand * (1 - edge[..., None]) + core.FOAM * edge[..., None], sm[..., None]], 2)
        # 水の色（波の山の部分：水面より上ほど明るい）
        self.y_top = WL - 300
        yy = np.arange(self.y_top, WL + 4, dtype=np.float32)[:, None]
        t = np.clip((WL - yy) / 280, 0, 1)
        self.wcol = (core.XS_WATER_TOP * (1 - t) + XS_LIT * t)[:, None, :] * np.ones((1, W, 1), np.float32)
        rn = np.random.default_rng(5).normal(0, 1, (WL + 4 - self.y_top, W)).astype(np.float32)
        nz = cv2.GaussianBlur(rn, (0, 0), 5) + 0.5 * cv2.GaussianBlur(rn, (0, 0), 1.6)
        self.noise = nz / (nz.std() * 2.5)
        self._sim_foam()

    # -------------------------------------------------- 波のキーフレーム
    def params(self, fi):
        K = self.KF
        m = keys(fi, [0, 22, 33, 40], [-1.0, 0.0, 1.0, 1.4])
        xc = keys(fi, [0, 20, 33, 40, 58], [60, 270, 390, 432, 690], ease_=False)
        Hh = keys(fi, K, [62, 150, 250, 232, 90, 26, 0])
        Wd = keys(fi, K, [240, 200, 240, 232, 180, 170, 170])
        return m, xc, Hh, Wd

    def wave_poly(self, fi):
        m, xc, Hh, Wd = self.params(fi)
        P = shape_xs(m)
        if m > 1.22:
            # 着水したあとは巻いた中の空気を泡が埋める（輪っかを残さない）：外側の線だけでふさぐ
            it = 2 + (NB - 1) + (NP - 1)          # リップの先の点
            P = np.concatenate([P[:it + 1], [(P[it, 0] + 0.05, 0.0), (6.0, 0.0), (6.0, 0.0)]]).astype(np.float32)
        return np.stack([xc + P[:, 0] * Wd, core.WL - P[:, 1] * Hh], 1), (m, xc, Hh, Wd)

    def surfer(self, fi):
        """乗っているサーファーの位置・角度・不透明度。崩れる所でやわらかく消える。"""
        m, xc, Hh, Wd = self.params(min(fi, 40))
        q, d = surf_point_xs(m)
        x = xc + q[0] * Wd
        y = core.WL - q[1] * Hh
        ang = np.degrees(np.arctan2(-d[1] * Hh, d[0] * Wd))
        a = 1 - smooth(fi, 36, 44)
        return x, y, float(np.clip(0.45 * ang, -6, 24)), float(a)

    # -------------------------------------------------- 泡（崩れた所）：一度だけシミュレーションしておく
    def _sim_foam(self):
        rng = np.random.default_rng(11)
        N = 4200
        t_em = rng.uniform(35.0, 48, N)                 # 生まれるフレーム
        kind = rng.uniform(0, 1, N)
        steps = self.n + 4
        P = np.full((steps, N, 2), np.nan, np.float32)
        A = np.zeros((steps, N), np.float32)
        pos = np.zeros((N, 2), np.float32)
        vel = np.zeros((N, 2), np.float32)
        alive = np.zeros(N, bool)
        landed = np.full(N, -1.0, np.float32)
        lift = rng.uniform(0, 1, N)
        for fi in range(steps):
            # 生まれる
            born = (~alive) & (t_em <= fi) & (landed < 0)
            if born.any():
                for j in np.where(born)[0]:
                    m, xc, Hh, Wd = self.params(t_em[j])
                    tip = lip_tip(m)
                    if kind[j] < 0.88:          # リップの先から
                        u0, h0 = tip[0] + rng.normal(0, 0.06), tip[1] * rng.uniform(0.2, 1.0)
                    else:                       # 山の上から（しぶき）
                        u0, h0 = rng.uniform(0.0, tip[0]), rng.uniform(0.75, 1.0)
                    pos[j] = (xc + u0 * Wd, core.WL - h0 * Hh)
                    vel[j] = (rng.normal(7.5, 2.4), -abs(rng.normal(6.0, 4.0)) * (0.8 if kind[j] >= 0.88 else 1.0))
                alive |= born
            # 動く
            fl = alive & (landed < 0)
            vel[fl, 1] += 2.4
            vel[fl] *= 0.985
            pos[fl] += vel[fl]
            sea = alive & (landed < 0) & (pos[:, 1] >= core.WL - 2) & (vel[:, 1] > 0)
            landed[sea] = fi
            on = alive & (landed >= 0)
            if on.any():
                x = pos[on, 0]
                v = vel[on, 0]
                age = fi - landed[on]
                # 水の上：前へ流れながら遅くなる。砂浜：駆け上がって、戻る
                onbeach = x > SHORE
                v = np.where(onbeach, v * 0.90 - 0.55, v * 0.955 + 0.18)
                v = np.clip(v, -6, 12)
                x = x + v
                thick = (4 + 46 * lift[on] ** 1.5) * np.exp(-age / 13.0)
                surf = np.where(x > SHORE, bed_y(x) - 1, core.WL)
                pos[on, 0] = x
                vel[on, 0] = v
                pos[on, 1] = surf - thick
            P[fi, alive] = pos[alive]
            a = np.ones(N, np.float32)
            a[landed >= 0] = np.exp(-(fi - landed[landed >= 0]) / 16.0)
            A[fi] = np.where(alive, a, 0)
        self.fP, self.fA = P, A
        self.f_w = rng.uniform(0.35, 1.0, N).astype(np.float32)

    def foam(self, fi):
        i0 = int(np.floor(fi))
        i1 = min(i0 + 1, self.fP.shape[0] - 1)
        t = fi - i0
        P = self.fP[i0] * (1 - t) + self.fP[i1] * t
        a = self.fA[i0] * (1 - t) + self.fA[i1] * t
        ok = np.isfinite(P[:, 0]) & (a > 0.01)
        return P[ok], (a * self.f_w)[ok]

    HOLD_X = 935

    def hold(self, f, fi):
        """泡が引いたあと（最後の約 0.9 秒）、浜にボードを抱えて立つピクトグラム。小さく首をかしげる。"""
        hin = self.n - 27
        a = core.ease((fi - hin) / 8, 'out') if fi >= hin else 0.0
        if a > 0:
            tilt = -28 * core.ease((fi - hin - 9) / 7, 'inout') if fi > hin + 9 else 0.0
            draw_surfer(f, self.HOLD_X, float(bed_y(self.HOLD_X)) - 1, scale=2.6, ang=0, alpha=a, pose='hold', head=tilt)
        return f

    # -------------------------------------------------- 描画
    def frame(self, fi, surfer=True, foam=True):
        WL = core.WL
        out = self.bg.copy()
        poly, (m, xc, Hh, Wd) = self.wave_poly(fi)
        y0 = self.y_top
        hh = WL + 4 - y0
        if Hh > 0.5:
            body = mask_poly(poly, hh, W, y0)
            # 波の中の細い線（b01 の版画の線と同じ考え方：高さを縮めた断面を何本か）
            P = np.stack([(poly[:, 0] - xc) / Wd, (WL - poly[:, 1]) / Hh], 1)
            lines = np.zeros_like(body)
            for s_ in (0.82, 0.64, 0.46, 0.28):
                top = P[1:-1]
                pts = np.stack([xc + top[:, 0] * Wd, WL - top[:, 1] * Hh * s_], 1)
                lines = np.maximum(lines, mask_line(pts, hh, W, 1, y0) * (0.10 + 0.22 * s_))
            body3 = body[..., None]
            reg = out[y0:WL + 4]
            col = self.wcol
            wk = float(smooth(m, 1.0, 1.4))
            if wk > 0:
                # 崩れたところは白い泡の色に（リップの側から）
                xx = np.arange(W, dtype=np.float32)[None, :]
                yy = np.arange(y0, WL + 4, dtype=np.float32)[:, None]
                ww = wk * smooth(xx, xc + 0.15 * Wd, xc + 0.85 * Wd) * (0.35 + 0.65 * smooth(yy, WL - 0.9 * Hh, WL - 0.1 * Hh))
                nz = np.roll(self.noise, int(fi * 3), axis=1)
                ww = (ww * np.clip(0.45 + 0.9 * nz, 0, 1))[..., None]
                col = col * (1 - ww) + core.FOAM * ww
            reg[:] = reg * (1 - body3) + col * body3
            la = (lines * cv2.erode(body, np.ones((3, 3), np.uint8)))[..., None]
            reg[:] = reg * (1 - la) + core.FOAM * la
        # 水面の白い線（3px）
        top = poly[1:-1]
        top = top[(top[:, 0] > -40) & (top[:, 0] < SHORE + 200)]
        line = np.concatenate([[(-40, WL)], top, [(SHORE + 200, WL)]]) if Hh > 0.5 else np.array([(-40, WL), (SHORE + 200, WL)], np.float32)
        lm = mask_line(line, hh, W, 3, y0)[..., None]
        reg = out[y0:WL + 4]
        reg[:] = reg * (1 - lm) + core.FOAM * lm
        # 砂
        sa = self.sand[..., 3:4]
        out = out * (1 - sa) + self.sand[..., :3] * sa
        # 泡
        if foam:
            P, a = self.foam(fi)
            if len(a):
                ya, yb = WL - 420, WL + 60
                buf = splat(np.zeros((yb - ya, W), np.float32), P[:, 0], P[:, 1] - ya, a)
                soft = cv2.GaussianBlur(buf, (0, 0), 1.3)
                glow = cv2.GaussianBlur(buf, (0, 0), 5.0)
                al = np.clip(soft * 1.6 + glow * 1.6, 0, 1)[..., None]
                out[ya:yb] = out[ya:yb] * (1 - al) + core.FOAM * al
        if surfer:
            x, y, ang, a = self.surfer(fi)
            if a > 0:
                draw_surfer(out, x, y - 3, scale=2.2, ang=ang, alpha=a)
        return out


# ================================================================ b06：水の粒の断面図（沖の水はその場で回るだけ）
GLOW = core.c('62CCFF')          # エネルギーの光の帯（明るい青）


def draw_buoy(frame, x, y, alpha=1.0, scale=1.0):
    """白いブイ（浮き＋短い柱＋小さな玉）。(x, y) は水面の位置。"""
    R = int(60 * scale)
    L = core.SS(s=4, w=2 * R, h=2 * R)
    d = L.draw
    col = L.col(core.FOAM, 1.0)
    cx, cy = R, R
    d.ellipse(L.box(cx - 17 * scale, cy - 12 * scale, cx + 17 * scale, cy + 10 * scale), fill=col)
    d.line(L.p([(cx, cy - 10 * scale), (cx, cy - 40 * scale)]), fill=col, width=L.w(3 * scale))
    d.ellipse(L.box(cx - 5 * scale, cy - 48 * scale, cx + 5 * scale, cy - 38 * scale), fill=col)
    lay = L.done()
    M = np.array([[1, 0, x - round(x)], [0, 1, y - round(y)]], np.float32)
    lay = cv2.warpAffine(lay, M, (2 * R, 2 * R), flags=cv2.INTER_LINEAR)
    x0, y0 = int(round(x)) - R, int(round(y)) - R
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + 2 * R, W), min(y0 + 2 * R, H)
    if xb <= xa or yb <= ya:
        return frame
    sub = lay[ya - y0:yb - y0, xa - x0:xb - x0]
    a = sub[..., 3:4] * alpha
    frame[ya:yb, xa:xb] = frame[ya:yb, xa:xb] * (1 - a) + sub[..., :3] * a
    return frame


class ParticleSea:
    """水の粒の断面図。粒は (x0, y0) のまわりを半径 r = A·exp(-深さ/D)·(0.3 + 0.7·帯) で回るだけ（横へは流れない）。
    明るい青の帯（エネルギー）は群速度 vg で右へ進み、波の山はその 2 倍の速さで帯の中を通り抜ける（深い海の性質）。
    world 座標で持ち、cam（px）だけ右を見る。"""

    def __init__(self, wl=700, lam=360.0, vg=11.9, A0=24.0, D=115.0, band_x0=-200.0, f_band0=20, sig=170.0):
        self.wl, self.k = wl, 2 * np.pi / lam
        self.vg, self.vp = vg, 2 * vg                                # px / frame
        self.om = self.k * self.vp
        self.A0, self.D, self.sig = A0, D, sig
        self.bx0, self.fb0 = band_x0, f_band0
        gx, gy = 34.0, 34.0
        xs = np.arange(-400, 3600, gx, dtype=np.float32)
        ys = np.arange(wl + 22, H + 30, gy, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        X = X + (np.arange(len(ys))[:, None] % 2) * gx / 2               # 互い違い
        self.x0, self.y0 = X.ravel(), Y.ravel()
        self.dep = self.y0 - wl
        sky = core.vgrad(core.XS_SKY_TOP, core.XS_SKY_WL, h=wl)
        water = core.vgrad(core.XS_WATER_TOP, core.XS_WATER_BOT, h=H - wl)
        self.bg = np.concatenate([sky, water], 0)
        self.trail = [0.0] + [0.045 * (j + 1) for j in range(15)]       # 秒ではなく「周期の割合」で軌跡を残す
        # 光の帯の縦の減衰（1/4 解像度）
        self.qy = np.arange(0, H - wl, 4, dtype=np.float32)

    def band_x(self, fi):
        return self.bx0 + self.vg * (fi - self.fb0)

    def env(self, x, fi):
        return np.exp(-((x - self.band_x(fi)) / self.sig) ** 2)

    def orbit(self, x0, dep, fi, base=0.3):
        r = self.A0 * np.exp(-dep / self.D) * (base + (1 - base) * self.env(x0, fi))
        th = self.k * x0 - self.om * fi
        return x0 - r * np.sin(th), self.wl + dep - r * np.cos(th)

    def surface(self, x0, fi):
        return self.orbit(x0, 0.0 * x0, fi)

    def render(self, fi, cam=0.0, dots=1.0):
        out = self.bg.copy()
        wl = self.wl
        # 光の帯（エネルギー）：水の中だけ、深いほど弱い
        qx = np.arange(0, W, 4, dtype=np.float32) + cam
        g = np.exp(-((qx - self.band_x(fi)) / (self.sig * 0.85)) ** 2)[None, :] * np.exp(-self.qy / 300.0)[:, None]
        g = cv2.resize(g.astype(np.float32), (W, H - wl), interpolation=cv2.INTER_LINEAR)
        gl = (g * 0.75)[..., None]
        o = out[wl:]
        o[:] = o * (1 - gl) + GLOW * gl
        # 粒と軌跡
        vis = (self.x0 - cam > -60) & (self.x0 - cam < W + 60)
        x0, dep, y0 = self.x0[vis], self.dep[vis], self.y0[vis]
        buf = np.zeros((H - wl + 40, W), np.float32)
        r = self.A0 * np.exp(-dep / self.D) * (0.3 + 0.7 * self.env(x0, fi))
        for j, ph in enumerate(self.trail):
            th = self.k * x0 - self.om * fi + ph * 2 * np.pi
            X = x0 - r * np.sin(th) - cam
            Y = dep - r * np.cos(th) + 20
            w = (1.0 if j == 0 else 0.22 * (1 - j / len(self.trail))) * dots
            splat(buf, X, Y, np.full(len(X), w, np.float32))
        dot = np.clip(cv2.GaussianBlur(buf, (0, 0), 1.25) * 4.2, 0, 0.9)[20:20 + H - wl]
        dot = dot[..., None]
        o[:] = o * (1 - dot) + core.FOAM * dot
        # 水面（白 3px）
        sx = np.arange(-40, W + 40, 6, dtype=np.float32) + cam
        X, Y = self.surface(sx, fi)
        pts = np.stack([X - cam, Y], 1)
        # 水面より上は空：水面の上側を空の色で塗り直す
        top = np.concatenate([[(-40, 0)], pts, [(W + 40, 0)]])
        sm = mask_poly(top, wl + 60, W)[..., None]
        out[:wl + 60] = out[:wl + 60] * (1 - sm) + self._skyext() * sm
        # 水面の下に少しはみ出た部分（谷）は水の色
        lm = mask_line(pts, wl + 60, W, 3)[..., None]
        out[:wl + 60] = out[:wl + 60] * (1 - lm) + core.FOAM * lm
        return out

    def _skyext(self):
        if not hasattr(self, '_se'):
            self._se = core.vgrad(core.XS_SKY_TOP, core.XS_SKY_WL, h=self.wl + 60) 
            self._se[self.wl:] = core.XS_SKY_WL
        return self._se
