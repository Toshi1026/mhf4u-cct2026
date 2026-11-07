"""07（21.8–25.0秒）「1万6000キロ旅した、うねりもある。」
南インド洋 → 太平洋 → アラスカの地図（メルカトルの地図を、南から少し斜めに見下ろす）。
ハード島の近くに小さな白い渦（嵐のアイコン）。うねりの道：東へ、オーストラリアの南、ニュージーランドの南から太平洋へ入り、
北へ（サモア・パルミラ・ハワイのそば）→ アラスカのヤクタット。光のパルスが道を進み、小さな印を順に灯す。
パルスの先には小さなサーファー。高知・日本への線は引かない。
最初の8フレーム：b06 からのズームアウトが収まる。final_background() は最後の地図（b08 が暗くして使う）。
"""
import cv2
import numpy as np

import core
import timeline
from core import W, H
from beats import _geo as G
from beats import _mapui as U

LON_L, LON_R, LAT_B, LAT_T = 20.0, 300.0, -84.0, 82.0
PPD = 9.0                      # 地図の板の px / 度（経度）
TILT = np.radians(48)
FOC = 2000.0
PIV = (-12.0, 140.0)           # この点が画面の (PCX, PCY) に来る
PCX, PCY = 520.0, 980.0
K = 0.63                       # 板の px → 画面 px（ピボットの奥行きで）

PATH = [(-53, 73.5), (-50.5, 100), (-49, 130), (-51.5, 170), (-42, 190), (-14, 190), (5.9, 198), (21, 203),
        (45, 212), (59.5, 220.3)]
HEARD = (-53.0, 73.5)
MARKS = [(-14.0, 190.0), (5.9, 198.0), (21.0, 203.0), (59.5, 220.3)]   # サモア・パルミラ・ハワイ・ヤクタット
OCEAN_N = core.lerp(core.NAVY, core.DEEP, 0.55)
OCEAN_S = core.lerp(core.DEEP, core.SEA, 0.10)
FOG = core.lerp(core.NAVY, core.DEEP, 0.30)
PATH_COL = core.lerp(core.FOAM, core.ACC, 0.40)


def merc(lat):
    return np.degrees(np.log(np.tan(np.pi / 4 + np.radians(np.clip(lat, -85, 85)) / 2)))


YM_T = merc(LAT_T)
YM_B = merc(LAT_B)
TW = int((LON_R - LON_L) * PPD)
TH = int((YM_T - YM_B) * PPD)


def to_tex(lat, lon):
    lon = np.mod(np.asarray(lon, np.float64) - LON_L, 360.0)
    return lon * PPD, (YM_T - merc(np.asarray(lat, np.float64))) * PPD


def homography():
    up, vp = to_tex(*PIV)
    s, c, k = np.sin(TILT), np.cos(TILT), K
    D = FOC + s * k * vp
    Hm = np.array([[FOC * k, -PCX * s * k, PCX * D - FOC * k * up],
                   [0.0, -PCY * s * k + FOC * c * k, PCY * D - FOC * c * k * vp],
                   [0.0, -s * k, D]], np.float64)
    return Hm


HM = homography()


def to_screen(lat, lon):
    u, v = to_tex(lat, lon)
    p = np.stack([np.atleast_1d(u), np.atleast_1d(v), np.ones(np.size(u))], 0)
    q = HM @ p
    return q[0] / q[2], q[1] / q[2]


class _Static:
    def __init__(self):
        # ---- 板（テクスチャ）：海のグラデーション・陸・経緯線・海岸線
        vv, uu = np.mgrid[0:TH, 0:TW].astype(np.float32)
        lon = LON_L + (uu + 0.5) / PPD
        ym = YM_T - (vv + 0.5) / PPD
        lat = np.degrees(2 * np.arctan(np.exp(np.radians(ym))) - np.pi / 2)
        land = G.land_at(lat.astype(np.float32), ((lon + 180) % 360 - 180).astype(np.float32), 8192)
        land = land * np.clip((lat + 64) / 4, 0, 1)        # 南極は描かない（手前の暗い海へ）
        t = np.clip((lat + 60) / 140, 0, 1)[..., None]
        ocean = OCEAN_S * (1 - t) + OCEAN_N * t
        tex = ocean * (1 - land[..., None]) + core.LAND * 0.97 * land[..., None]
        L = np.zeros((TH * 2, TW * 2), np.uint8)
        segs = []
        for r in G.rings():
            u, v = to_tex(r[:, 1], r[:, 0])
            # 経度の切れ目（板の左右の端）で線を切る
            br = np.flatnonzero(np.abs(np.diff(u)) > TW * 0.5) + 1
            for part_u, part_v in zip(np.split(u, br), np.split(v, br)):
                if len(part_u) >= 2:
                    segs.append(np.round(np.stack([part_u, part_v], 1) * 2 * 16).astype(np.int32))
        cv2.polylines(L, segs, False, 255, thickness=3, lineType=cv2.LINE_AA, shift=4)
        coast = cv2.resize(L, (TW, TH), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        Lg = np.zeros((TH, TW), np.uint8)
        for la in range(-60, 61, 30):
            _, v = to_tex(la, 0)
            cv2.line(Lg, (0, int(v * 16)), (TW * 16, int(v * 16)), 255, 1, cv2.LINE_AA, 4)
        for lo in range(30, 300, 30):
            u, _ = to_tex(0, lo)
            cv2.line(Lg, (int(u * 16), 0), (int(u * 16), TH * 16), 255, 1, cv2.LINE_AA, 4)
        grat = Lg.astype(np.float32) / 255
        tex = tex + (core.FOAM - tex) * (grat[..., None] * 0.07)
        tex = tex + (core.FOAM - tex) * (coast[..., None] * 0.6)
        # 南極のあたり（南緯60°より南）は手前の暗い海へ溶かす
        sfade = np.clip((-55 - lat) / 15, 0, 1)
        sfade = (sfade * sfade * (3 - 2 * sfade))[..., None]
        tex = tex * (1 - sfade) + core.NAVY * sfade
        # ---- 画面へ
        warped = cv2.warpPerspective(tex.astype(np.float32), HM, (W, H), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(-1, -1, -1))
        inside = (warped[..., 0] >= 0).astype(np.float32)
        inside = cv2.GaussianBlur(inside, (0, 0), 3)
        bg = core.vgrad(FOG, core.NAVY)
        f = np.where(warped >= 0, warped, 0) * inside[..., None] + bg * (1 - inside[..., None])
        # 奥ほど霧（画面の y だけで決まる）と、左右の端を少し暗く
        yy = np.arange(H, dtype=np.float32)
        hy = self.horizon_y()
        fog = np.clip((800 - yy) / 260, 0, 1)
        fog = (fog * fog * (3 - 2 * fog)) * 0.88
        near = np.clip((yy - 1225) / 420, 0, 1)
        near = near * (2 - near) * 0.85
        fog = fog[:, None, None]
        f = f * (1 - fog) + FOG * fog
        f = f * (1 - near[:, None, None]) + core.NAVY * near[:, None, None]
        xx = np.linspace(-1, 1, W, dtype=np.float32)
        side = (1 - 0.22 * xx ** 4)[None, :, None]
        self.map = (f * side).astype(np.float32)
        # ---- 道
        P = G.route(PATH, step_deg=0.25, round_deg=3.0)
        la, lo = G.latlon(P)
        x, y = to_screen(la, np.mod(lo, 360))
        self.path = np.stack([x, y], 1)
        self.ps = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(self.path, axis=0).T))])
        self.marks = []
        for m in MARKS:
            mx, my = to_screen(*m)
            j = int(np.argmin(np.hypot(self.path[:, 0] - mx[0], self.path[:, 1] - my[0])))
            self.marks.append((float(mx[0]), float(my[0]), self.ps[j] / self.ps[-1]))
        hx, hy2 = to_screen(*HEARD)
        self.heard = (float(hx[0]), float(hy2[0]))

    @staticmethod
    def horizon_y():
        # 板の北の端（LAT_T）の画面 y
        _, y = to_screen(LAT_T, 180.0)
        return float(y[0])


_ST = None


def static():
    global _ST
    if _ST is None:
        _ST = _Static()
    return _ST


def storm(f, x, y, ang, alpha=1.0, r=30.0):
    """小さな白い渦（2本の腕）。同心円にはしない。"""
    L = core.SS(s=3, w=int(4 * r), h=int(4 * r))
    d = L.draw
    c0 = 2 * r
    for arm in (0, 120, 240):
        pts = []
        for t in np.linspace(0, 1, 40):
            th = np.radians(ang + arm) - t * 2.2
            rr = r * (0.28 + 0.72 * t)
            pts.append((c0 + rr * np.cos(th), c0 + rr * np.sin(th)))
        for q in range(len(pts) - 1):        # 外へ向かって細く・薄く
            t = q / (len(pts) - 1)
            d.line(L.p(pts[q:q + 2]), fill=L.col(core.FOAM, (0.95 - 0.55 * t) * alpha), width=L.w(3.4 - 2.0 * t))
    rc = r * 0.26
    d.ellipse(L.box(c0 - rc, c0 - rc, c0 + rc, c0 + rc), outline=L.col(core.FOAM, 0.95 * alpha), width=L.w(2.6))
    return U.paste(f, L.done(), int(round(x - c0)), int(round(y - c0)))


def _partial(P, s, frac):
    Lx = s[-1] * frac
    i = int(np.searchsorted(s, Lx))
    if i <= 0:
        return P[:1], P[0]
    if i >= len(P):
        return P, P[-1]
    t = (Lx - s[i - 1]) / max(s[i] - s[i - 1], 1e-9)
    tip = P[i - 1] * (1 - t) + P[i] * t
    return np.vstack([P[:i], tip[None]]), tip


def _timing(beat):
    n = beat['n']
    return n, 9, n - 18        # パルスが動く範囲


def map_frame(st, i, p, surfer_on=True):
    """文字なしの地図（パルスの進み p）。"""
    f = st.map.copy()
    # まだ通っていない所：うすい点線
    f = G.draw_line(f, G.dashed(st.path, 4, 9), core.FOAM, 1.6, 0.30)
    if p > 0:
        part, tip = _partial(st.path, st.ps, p)
        f = G.draw_line(f, part, PATH_COL, 3.0, 0.85, glow_sigma=6, glow_alpha=0.30)
        # パルスの頭：短い明るい区間
        hd, _ = _partial(st.path, st.ps, p)
        k0 = np.searchsorted(st.ps, st.ps[-1] * max(0, p - 0.05))
        f = G.draw_line(f, np.vstack([st.path[k0:len(hd) - 1], tip[None]]), core.FOAM, 4.6, 0.9,
                        glow_sigma=9, glow_alpha=0.45)
    # 印
    for mx, my, mp in st.marks:
        lit = core.ease((p - mp) / 0.035 + 1) if p > 0 else 0.0
        f = G.draw_dots(f, [(mx, my)], 6.0, core.FOAM, 0.35 + 0.0 * lit)
        if lit > 0:
            f = G.draw_dots(f, [(mx, my)], 6.0, core.FOAM, lit, glow_sigma=8, glow_alpha=0.45 * lit)
    # 嵐
    f = storm(f, st.heard[0], st.heard[1], ang=-i * 4.0, alpha=1.0)
    if p > 0 and surfer_on:
        _, tip = _partial(st.path, st.ps, p)
        f = G.draw_dots(f, [tuple(tip)], 6.0, core.FOAM, 1.0, glow_sigma=10, glow_alpha=0.5)
        L = core.SS(s=3, w=120, h=120)
        core.surfer(L, 60, 64, scale=1.25)
        f = U.paste(f, L.done(), int(round(tip[0])) - 60, int(round(tip[1])) - 72)
    return f


def final_background(beat=None):
    """b07 の最後のフレームの地図（文字なし）。b08 が暗くして使う。"""
    st = static()
    if beat is None:
        beat = next(b for b in timeline.load() if b['key'] == '07')
    n = beat['n']
    return map_frame(st, n - 1, 1.0)


def render(beat, ctx):
    n, p0, p1 = _timing(beat)
    st = static()
    mf = lambda k: timeline.mora_frame(beat, k)
    f_cnt, f_cnt_end = mf(0), mf(9)            # イチマン … キロ
    f_sub, f_note = mf(10), mf(14)            # タビシタ／ウネリモアル
    cnt = U.Counter('16,000', W / 2, 580, digit_size=150, side_size=78, prefix='', suffix='km以上')
    t_sub = U.Patch([dict(text='南の海の嵐 → アラスカまで', x=W / 2, y=410, size=44, weight='Medium')])
    t_note = U.Patch([dict(text='1963年 W.ムンクらの観測', x=W / 2, y=1238, size=30, weight='Medium', alpha=0.92)])
    for i in range(n):
        p = core.ease((i - p0) / (p1 - p0), 'inout')
        f = map_frame(st, i, p)
        out = core.ease((n - 1 - i) / 5)
        # 最初：b06 からのズームアウトが収まる
        if i < 8:
            k = core.ease((i + 1) / 8, 'out')
            sc = 1 + 0.30 * (1 - k)
            M = np.float32([[sc, 0, W / 2 * (1 - sc)], [0, sc, H / 2 * (1 - sc)]])
            f = cv2.warpAffine(f, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
            f = core.zoom_blur(f, 0.22 * (1 - k), cx=W / 2, cy=H / 2)
        ta = core.ease((i - f_cnt + 1) / 5) * out
        f = U.scrim(f, 350, 610, 0.25 * ta, feather=120)
        v = 16000 * core.ease((i - f_cnt) / max(1, f_cnt_end - f_cnt), 'out')
        f = cnt.put(f, U.fmt(v, 100) if i < f_cnt_end else '16,000', ta)
        f = t_sub.put(f, core.ease((i - f_sub + 1) / 6) * out)
        f = t_note.put(f, core.ease((i - f_note + 1) / 6) * out)
        yield np.clip(f, 0, 1).astype(np.float32)
