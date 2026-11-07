"""03（8.0–10.4秒）「秘密は、海の底の谷。」
ナザレ海底谷の等深線図（EMODnet の実際の水深）。少し斜めから見下ろす。西北西から来た波の線（光の点が流れる）が、
谷の上で北へ曲がり、北の大陸棚の上の線と一緒にプライア・ド・ノルテの1点に集まる（屈折の模式図）。
最初の8フレーム：b02 の下へのティルトを受けて縦の whip が収まる。最後の8フレーム：ズームアウトのぼかしで b04 へ。
"""
import os

import cv2
import numpy as np
from PIL import Image

import core
import timeline
from core import W, H
from beats import _geo as G
from beats import _mapui as U

TIF = os.path.join(core.MODELS, 'nazare_emodnet.tif')
LON0, DLON, LAT0, DLAT = -10.6, 0.002125, 39.9, -0.0021212121212121
PDN = (39.605, -9.082)                 # Praia do Norte（波が集まる浜）
KX = 111.32 * np.cos(np.radians(39.55))  # km / 経度1度
KY = 111.2
# 画面への写し方：沖（西）から陸（東）の方を、少し斜めに見下ろす。画面の上＝東、右＝南
S_KM = 14.0              # 浜の奥行きでの px/km
TILT = np.radians(30)
FOC = 2100.0
PX, PY = 470.0, 900.0    # 浜の画面位置（b04 の最初の画面位置もここ）
ROLL = 90.0              # 画面の上が東（b04 はこの向きから北が上へ戻る）

SHALLOW = core.lerp(core.SEA, core.DEEP, 0.30)
DEEPC = core.lerp(core.NAVY, core.DEEP, 0.30)
FOG = core.lerp(core.NAVY, core.DEEP, 0.36)


def to_km(lat, lon):
    return (np.asarray(lon) - PDN[1]) * KX, (np.asarray(lat) - PDN[0]) * KY


def km_to_screen(X, Y):
    """X：東へ km、Y：北へ km → 画面。"""
    Xp, Yp = -np.asarray(Y) * S_KM, np.asarray(X) * S_KM
    z = FOC + Yp * np.sin(TILT)
    return PX + FOC * Xp / z, PY - FOC * Yp * np.cos(TILT) / z


def ll_to_screen(lat, lon):
    return km_to_screen(*to_km(lat, lon))


def screen_to_ll(x, y):
    dy = y - PY
    Yp = -dy * FOC / (FOC * np.cos(TILT) + dy * np.sin(TILT))
    z = FOC + Yp * np.sin(TILT)
    Xp = (x - PX) * z / FOC
    return PDN[0] - Xp / S_KM / KY, PDN[1] + Yp / S_KM / KX


def _grid():
    a = np.array(Image.open(TIF)).astype(np.float32)
    up = 3
    big = cv2.resize(a, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    sm = cv2.GaussianBlur(big, (0, 0), 3.2)
    return a, sm, up


def _thalweg(a):
    """谷の軸：谷頭から西へ、列ごとに近くでいちばん深い行をたどる。"""
    s = cv2.GaussianBlur(a, (0, 0), 2.0)
    c = int(round((-9.118 - LON0) / DLON))
    r = int(round((39.596 - LAT0) / DLAT))
    pts = []
    while c > int(round((-10.25 - LON0) / DLON)):
        lo, hi = max(0, r - 3), min(a.shape[0], r + 4)
        r = lo + int(np.argmin(s[lo:hi, c]))
        pts.append((LAT0 + r * DLAT, LON0 + c * DLON))
        c -= 1
    P = np.array(pts)
    k = 21
    ker = np.ones(k) / k
    lat = np.convolve(np.pad(P[:, 0], k // 2, mode='edge'), ker, 'valid')
    lon = P[:, 1]
    return np.stack([lat, lon], 1)


def _rays(th_km):
    """波の線（km 座標の点列のリスト）。西北西からまっすぐ来て、谷を越えたところ（北の線は浅瀬にかかったところ）から
    曲がり、浜の1点へ集まる（屈折の模式図）。th_km：谷の軸（谷頭→西）の km 座標。"""
    d = np.array([np.cos(np.radians(-22.5)), np.sin(np.radians(-22.5))])   # 進む向き（東南東）
    ts = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(th_km, axis=0).T))])
    starts = []
    for dist in (34.0, 23.0, 14.0, 7.0):                    # 谷を越える点（谷頭からの距離 km）
        starts.append((np.array([np.interp(dist, ts, th_km[:, 0]), np.interp(dist, ts, th_km[:, 1])]), 0.36))
    for off in ((-17.0, 7.0), (-23.0, 13.5), (-30.0, 20.5)):  # 北の大陸棚の上
        starts.append((np.array(off), 0.30))
    out = []
    n = len(starts)
    for k, (C, kk) in enumerate(starts):
        T = np.array([0.0, 0.32 * (k - 3.4) / 3.4])
        P0 = C - d * 160.0
        straight = P0[None] + (C - P0)[None] * np.linspace(0, 1, 160)[:, None]
        Q = C + d * kk * np.hypot(*(T - C))
        t = np.linspace(0, 1, 120)[1:, None]
        curve = (1 - t) ** 2 * C + 2 * (1 - t) * t * Q + t * t * T
        out.append(np.vstack([straight, curve]))
    return out


class _Static:
    def __init__(self):
        a, sm, up = _grid()
        self.a = a
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        lat, lon = screen_to_ll(xx + 0.5, yy + 0.5)
        gx = ((lon - LON0) / DLON * up).astype(np.float32)
        gy = ((lat - LAT0) / DLAT * up).astype(np.float32)
        dep = cv2.remap(sm, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # データの外は、強くぼかした（端を延ばした）水深で埋める
        far_sm = cv2.GaussianBlur(sm, (0, 0), 40)
        dep_far = cv2.remap(far_sm, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        inx = np.clip(np.minimum(gx, sm.shape[1] - 1 - gx) / (up * 26), 0, 1)
        iny = np.clip(np.minimum(gy, sm.shape[0] - 1 - gy) / (up * 22), 0, 1)
        vig = (inx * iny)
        vig = vig * vig * (3 - 2 * vig)
        dep = dep * vig + dep_far * (1 - vig)
        # 陰影（北西からの光）
        gyd, gxd = np.gradient(cv2.GaussianBlur(sm, (0, 0), 1.5))
        shade_g = np.clip(0.5 + (-gxd * 0.5 + gyd * 0.7) / 40.0, 0, 1)
        shade = cv2.remap(shade_g.astype(np.float32), gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        shade = shade * vig + 0.5 * (1 - vig)
        depth = np.clip(-dep, 0, None)
        t = np.clip(depth / 4800, 0, 1) ** 0.42
        sea = SHALLOW * (1 - t[..., None]) + DEEPC * t[..., None]
        sea = sea * (0.80 + 0.40 * shade[..., None])
        land = np.clip(dep / 6.0 + 0.5, 0, 1)[..., None]
        f = sea * (1 - land) + core.LAND * (0.96 + 0.06 * shade[..., None]) * land
        # 奥（東・画面の上）ほど霧で暗く
        far = np.clip((PY - 30 - yy) / 420, 0, 1)[..., None]
        far = far * (2 - far) * 0.96
        f = f * (1 - far) + FOG * far
        v = vig[..., None] * (1 - far)
        self.base_no_lines = f.astype(np.float32)
        # 等深線（250 m ごと）と海岸線
        L = core.SS(s=2)
        dr = L.draw
        for lev in range(250, 4800, 250):
            m = (sm < -lev).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            q = lev / 4750
            col = core.lerp(core.lerp(core.SEA, core.FOAM, 0.55), core.lerp(core.SKY, core.ACC, 0.55), q)
            al = 0.30 + 0.45 * q
            for cc in cs:
                if len(cc) < 12:
                    continue
                cc = cc[:, 0, :].astype(np.float64)
                lat_c = LAT0 + cc[:, 1] / up * DLAT
                lon_c = LON0 + cc[:, 0] / up * DLON
                x, y = ll_to_screen(lat_c, lon_c)
                pts = list(zip(x, y))
                if len(pts) > 2:
                    dr.line(L.p(pts), fill=L.col(col, al), width=L.w(1.15 + 0.5 * q), joint='curve')
        m = (sm > 0).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cc in cs:
            if len(cc) < 12:
                continue
            cc = cc[:, 0, :].astype(np.float64)
            x, y = ll_to_screen(LAT0 + cc[:, 1] / up * DLAT, LON0 + cc[:, 0] / up * DLON)
            dr.line(L.p(list(zip(x, y))), fill=L.col(core.FOAM, 0.6), width=L.w(1.5), joint='curve')
        lines = L.done()
        a_l = lines[..., 3:4] * v
        self.base = (f * (1 - a_l) + lines[..., :3] * a_l).astype(np.float32)
        self.vig = vig
        # 谷の軸
        th = _thalweg(a)
        x, y = ll_to_screen(th[:, 0], th[:, 1])
        self.thal = np.stack([x, y], 1)
        self.thal_s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(self.thal, axis=0).T))])
        # 波の線
        self.rays = []
        tk = np.stack(to_km(th[:, 0], th[:, 1]), 1)
        for B in _rays(tk):
            x, y = km_to_screen(B[:, 0], B[:, 1])
            P = np.stack([x, y], 1)
            on = np.flatnonzero((P[:, 1] < H + 60) & (P[:, 0] > -60))
            P = P[max(0, on[0] - 1):]
            s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
            self.rays.append((P, s))
        self.pdn = np.array(ll_to_screen(*PDN))


_ST = None


def static():
    global _ST
    if _ST is None:
        _ST = _Static()
    return _ST


def _partial(P, s, frac):
    L = s[-1] * frac
    i = int(np.searchsorted(s, L))
    if i <= 0:
        return P[:1]
    if i >= len(P):
        return P
    t = (L - s[i - 1]) / max(s[i] - s[i - 1], 1e-9)
    return np.vstack([P[:i], P[i - 1] * (1 - t) + P[i] * t])


def scene(i, n, st, beat):
    """文字と転換を除いた絵。"""
    f = st.base.copy()
    # 谷の軸：f10〜f30 で谷頭から西へ伸びる
    tp = core.ease((i - 8) / 22, 'out')
    if tp > 0:
        Pt = _partial(st.thal, st.thal_s, tp)
        al = np.interp(np.arange(len(Pt)), [0, len(st.thal) * 0.75, len(st.thal)], [1, 0.85, 0.0])
        Lm = G.Lines()
        Lm.poly(Pt, 2.6)
        m = Lm.done()
        # 西の端ほど薄く（軸の長さ方向の減衰は画面 x で近似）
        xfade = np.clip((H + 40 - np.arange(H, dtype=np.float32)) / 420, 0, 1)[:, None]
        m = m * xfade
        f = G.tint(f, np.clip(G.glow(m, 7) * 1.8, 0, 1), core.ACC, 0.35)
        f = G.tint(f, m, core.ACC, 0.95)
    # 波の線：f6 から西の端で生まれ、浜へ伸びる。そのあと光の点が流れる
    base = G.Lines()
    dash = G.Lines()
    arrived = 0.0
    for k, (P, s) in enumerate(st.rays):
        pr = core.ease((i - 6 - k * 1.5) / 28, 'inout')
        if pr <= 0:
            continue
        Pp = _partial(P, s, pr)
        base.poly(Pp, 1.6)
        ph = (i * 12.0 + k * 37) % 130
        segs = G.dashed(Pp, 34, 96, phase=ph)
        for sg in segs:
            dash.poly(sg, 2.6)
        arrived += pr >= 0.999
    bm, dm = base.done(), dash.done()
    f = G.tint(f, bm, core.FOAM, 0.24)
    f = G.tint(f, np.clip(G.glow(dm, 4) * 1.6, 0, 1), core.SKY, 0.30)
    f = G.tint(f, dm, core.FOAM, 0.85)
    # 集まる点
    ga = core.ease((i - 30) / 12)
    if ga > 0:
        f = G.draw_dots(f, [tuple(st.pdn)], 6.5, core.FOAM, 0.95 * ga, glow_sigma=10, glow_alpha=0.35 * ga)
    return f


def render(beat, ctx):
    n = beat['n']
    st = static()
    mf = lambda k: timeline.mora_frame(beat, k)
    f_umi = mf(4)            # ウ（うみの そこの たに）
    f_line2 = f_umi + 4       # 「波を集める」
    t1 = U.Patch([dict(text='海の底の谷が', x=W / 2, y=548, size=92, weight='Bold', hl='谷')])
    t2 = U.Patch([dict(text='波を集める', x=W / 2, y=660, size=92, weight='Bold')])
    # 「長さ 約210km」：谷の軸のそば（西寄り）
    j = int(np.argmin(np.abs(st.thal[:, 1] - 1120)))
    lx, ly = st.thal[j]
    t_len = U.Patch([dict(text='長さ 約210km', x=lx + 46, y=ly + 12, size=34, weight='Medium', anchor='ls')])
    t_cr = U.Patch([dict(text='水深データ：EMODnet', x=75, y=1236, size=28, weight='Regular', anchor='ls',
                         alpha=0.85)])
    for i in range(n):
        f = scene(i, n, st, beat)
        f = U.scrim(f, 470, 680, 0.30 * core.ease((i - f_umi + 6) / 10), feather=170)
        out = core.ease((n - 1 - i) / 6)          # 最後は文字を先に消す（ズームアウトの前）
        f = t_cr.put(f, 0.85 * core.ease((i - 4) / 10) * out)
        f = t_len.put(f, core.ease((i - 16) / 8) * out)
        f = t1.put(f, core.ease((i - f_umi + 1) / 6) * out)
        f = t2.put(f, core.ease((i - f_line2 + 1) / 6) * out)
        # 最初：b02 の下へのティルトを受けて、縦のぶれが収まる
        if i < 8:
            k = core.ease((i + 1) / 8, 'out')
            dy = 240 * (1 - k)
            M = np.float32([[1, 0, 0], [0, 1, dy]])
            f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
            f = core.whip(f, 150 * (1 - k), axis=0)
        # 最後：浜を中心にズームアウトしながらぼかす → b04
        if i >= n - 8:
            k = (i - (n - 8) + 1) / 8
            sc = 1 - 0.40 * core.ease(k, 'in')
            cx, cy = st.pdn
            M = np.float32([[sc, 0, cx * (1 - sc)], [0, sc, cy * (1 - sc)]])
            f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
            f = core.zoom_blur(f, 0.32 * k ** 1.3, cx=cx, cy=cy)
        yield np.clip(f, 0, 1).astype(np.float32)
