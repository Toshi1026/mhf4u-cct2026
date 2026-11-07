"""B07（約1.8秒）「そして、よんじゅうごびょうご。」
06 の続き（同じ断面）。06 の文字と富士山が消え、点だった悪役が元の大きさに戻りながら、放物線の右半分を回りながら落ちていく
（だんだん速く）。秒のカウンター（全体を強調）はソで「23秒後」（頂点＝飛んでいる時間 45.4 秒の半分）で出て、悪役の横の進みと同じに数え、
「ヨンジュウゴ」のゴの 3 フレーム後（「ゴビョオゴ」の頭）、水面に着くところで「45秒後」に止まる。
2026-11-07 訂正：10km 版（t＝2v·sin45°/g＝45.4 秒）。旧「52秒後」（26→52）。
悪役は水面で消え、着水（小さなしぶき）は T で土佐湾の線画の上に見せる。
着水の次のフレームから最後まで：着水する所へのズームのぼかし、最後の 11 フレームで濃紺（案A b10 と同じ背景）へ暗くし、土佐湾の線画をごくうすく出す（T は線画から）。

2026-11 動きの改善（director's motion spec §1-07）：
- カメラ：06 の最後（頂点へ 1.13 寄った状態）から、落ちるあいだに sine_inout で引いて全体へ戻る（着水点がセーフゾーンに入る）。
- 秒のカウンターは _motion.Odometer（回る速さ＝悪役の横の進み、落ちるほど速く）。ズームのぼかしと暗くするのより後に、画面に固定で描く。
  着水（f_land）で「45」にロック（1.10→1.0 の back_out）、8 フレームはくっきり見せ、そのあと 6 フレームで上へぼけて消える。
- 落ちる悪役：着く前の 6 フレームは速さの向きに 1.0→1.35 伸び、サブフレームのぶれと 2 本の速さの線。マントは速さの反対。
  着水で縦の小さなしぶき 5 粒（上と外へ、重力つき）。壁や輪は描かない。
- 06 の文字は上へぼけて抜ける。06 のキラーンの名残は _art.put_glint_fx。"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W, H, WL
from beats import _mapui as U
from beats import _tosa, b10
from cutB.beats import _art as A, _xsec as X, _motion as M, b06
import cv2
import math

LAND_AFTER_GO = 3                             # 着水（＝45秒）はナレーションの「ヨンジュウゴ」のゴの 3 フレーム後
SECS = 45.4                                   # 飛んでいる時間（試算）。表示は 23 → 45
DARKEN = 11                                   # 濃紺へ暗くするのは最後の 11 フレーム（1フレームの明るさの差を小さく）


class SecCounter:
    """「N秒後」。数字は同じ幅のマス、全体を ACC。右端の「秒後」を最終の文字列の中心で固定。"""

    def __init__(self, cx, y, size=110):
        self.y, self.size = y, size
        f = core.font(size, 'Bold')
        self.cell = max(f.getlength(str(d)) for d in range(10)) * 0.98
        self.sw = core.text_width('秒後', size, 'Bold', 0.02)
        tot = 2 * self.cell + 0.04 * size + self.sw
        self.left2 = cx - tot / 2
        self.cache = {}

    def put(self, f, v, alpha):
        s = str(int(v))
        p = self.cache.get(s)
        if p is None:
            x = self.left2 + (2 - len(s)) * self.cell / 2          # 1桁の間は少し右へ寄せて中心を保つ
            items = []
            for ch in s:
                items.append(dict(text=ch, x=x + self.cell / 2, y=self.y, size=self.size, weight='Bold', track=0,
                                  color=core.ACC))
                x += self.cell
            items.append(dict(text='秒後', x=x + 0.04 * self.size, y=self.y, size=self.size, weight='Bold',
                              anchor='ls', track=0.02, color=core.ACC))
            p = self.cache[s] = U.Patch(items)
        return p.put(f, alpha)


CAM_BACK = 26                                  # 06 の寄りから全体へ戻すフレーム数
LOCK_HOLD = 8                                  # 45 に止まってからくっきり見せるフレーム数


def cam_M(i, M_end):
    q = M.ease(i / float(CAM_BACK), 'sine_inout')
    return M_end * (1 - q) + np.array([[1.0, 0, 0], [0, 1.0, 0]]) * q


def fall_u(i, f_land):
    tau = np.clip((i - 2) / float(f_land - 2), 0, 1)
    return 0.5 + 0.5 * tau ** 1.7, float(tau)


def villain_draw(t, f_land, gi0):
    u, tau = fall_u(t, f_land)
    g = core.ease((t - 1) / 10.0, 'inout')
    sc = 0.12 + (X.VS - 0.12) * g
    if sc * 34 < 5:
        return None
    x, y = X.arc(u)
    sec = (gi0 + t) / core.FPS
    ang = -20 + 560 + 700 * tau ** 1.5
    ua, _ = fall_u(t - 0.5, f_land)
    ub, _ = fall_u(t + 0.5, f_land)
    va = math.degrees(math.atan2(*(np.array(X.arc(ub))[::-1] - np.array(X.arc(ua))[::-1])))
    s_al = 1 + 0.35 * M.ease((t - (f_land - 6)) / 6.0, 'quad_in')
    M2 = M.squash_matrix(va, s_al, 1 / math.sqrt(s_al)) if s_al > 1.002 else None
    ul, _ = fall_u(t - 2.5, f_land)
    ul2, _ = fall_u(t - 1.5, f_land)
    d = np.array(X.arc(ul2)) - np.array(X.arc(ul))
    cape = math.degrees(math.atan2(-d[1], -d[0])) if np.hypot(*d) > 0.3 else 100.0
    cape += 8 * math.sin(sec * 10.0)
    return A.villain_rgba(float(x), float(y) - 14 * sc, sc, ang, 'fly', sec, M2=M2, cape_ang=cape)


def speed_lines(f, i, f_land):
    """着く前の 6 フレーム：悪役のうしろに 2 本の短い速さの線。"""
    k = i - (f_land - 6)
    if k < 0 or i > f_land:
        return f
    u, _ = fall_u(i, f_land)
    u0, _ = fall_u(i - 1, f_land)
    p1, p0 = np.array(X.arc(u)), np.array(X.arc(u0))
    v = p1 - p0
    sp = float(np.hypot(*v))
    if sp < 4:
        return f
    v /= sp
    nrm = np.array([-v[1], v[0]])
    L = core.SS(s=2)
    a = 0.55 * M.ease((k + 1) / 3.0, 'sine_inout')
    for off, kk in ((-1, 1.0), (1, 0.75)):
        a0 = p1 - v * 22 + nrm * off * 7 - (0, 14 * X.VS)
        a1 = a0 - v * min(170.0, sp * 2.6) * kk
        L.draw.line(L.p([tuple(a0), tuple(a1)]), fill=L.col(core.FOAM, a), width=L.w(2.0))
    return core.over(f, L.done())


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[0][0] == 'ソ' and mo[7][0] == 'ゴ' and mo[8][0] == 'ビョ', mo    # ソシテ ヨンジュウ「ゴ」ビョオゴ
    f_so, f_go = timeline.mora_frame(beat, 0), timeline.mora_frame(beat, 7)
    b6 = next(b for b in timeline.load() if b['key'] == '06')
    T6 = b06.timing(b6)
    n6 = b6['n']
    _ctr6, t_sub, _ = b06.texts()
    ctr = M.Odometer('45', W / 2, 450, digit_size=110, side_size=110, prefix='', suffix='秒後', gap=0.02,
                     suffix_color=core.ACC, lock_scale=1.10, lock_drop=5.0, lock_dur=6, blur_k=0.6)
    nav = b10._background()
    lines = _tosa.contours()
    f_land = min(f_go + LAND_AFTER_GO, n - DARKEN - 2)   # 水面に着くフレーム（45 秒＝着水。ゴのすぐ後）
    out = n - (f_land + 1)                       # 着水の次のフレームから最後まで：着水点へのズームのぼかし
    M_end = b06.cam_M(n6 - 1, T6)
    splash = M.Burst(23, 5, (X.X1, WL - 2), angle_range=(-124, -56), speed_range=(8.0, 13.0), gravity=1.3,
                     drag=0.04, life=8, size=(2.8, 4.2), f0=f_land, alpha=0.9, stretch=0.5, life_jitter=0.2)
    f_exit = f_land + LOCK_HOLD
    for i in range(n):
        gi = n6 + i                              # 06 の頭からの通し番号
        fade6 = 1.0 - M.ease((i + 1) / 9.0, 'sine_inout')
        f = b06.scene(gi, T6, fuji_a=fade6, fuji=(1.0, 1.0))   # 富士山は落ちる弧と重なるので、06 の文字と一緒に消す
        u, tau = fall_u(i, f_land)
        rgba, x0, y0 = X.trail_layer(u, ghost=1.0 - core.ease(i / 30.0) * 0.4)
        f = U.paste(f, rgba, x0, y0)
        f = b06.apex_lines(f, 1.0 - 0.4 * core.ease(i / 12.0))
        f = b06.sight_line(f, 0.5 * (1 - core.ease(i / 8.0)), phase=-0.8 * gi)
        # 悪役：点から元の大きさへ戻りながら落ちる（水面に着いたら 3 フレームで消える。しぶきは T の地図の上でも見せる）
        va = 1 - core.ease((i - f_land + 1) / 3.0)
        if va > 0.003:
            f = speed_lines(f, i, f_land)
            g = core.ease((i - 1) / 10.0, 'inout')
            sc = 0.12 + (X.VS - 0.12) * g
            if sc * 34 >= 5:
                ua, _ = fall_u(i - 0.5, f_land)
                ub, _ = fall_u(i + 0.5, f_land)
                sp = float(np.hypot(*(np.array(X.arc(ub)) - np.array(X.arc(ua)))))
                nn = int(np.clip(math.ceil(sp * 0.4 / 3.0), 1, 8)) if sp > 12 else 1
                r = M.subframe(lambda t: villain_draw(t, f_land, n6), i, shutter=0.4, n=nn)
                if r:
                    f = M.put_rgba(f, *r, alpha=va, shadow=0.35, shadow_px=max(1.0, 1.1 * sc))
            else:
                x, y = X.arc(u)
                f = A.put_dot(f, float(x), float(y) - 2, 2.4, alpha=va)
        f = splash.draw(f, i)
        # 06 のキラーンの名残
        f = A.put_glint_fx(f, X.APEX[0], X.APEX[1] - 2, b06.glint_k(gi, T6), r=42)
        # カメラ（06 の寄りから戻す）とラベル
        Mc = cam_M(i, M_end)
        f = b06.warp(f, Mc)
        f = b06.put_labels(f, gi, T6, Mc, fuji_a=fade6, apex_a=1.0 - 0.4 * core.ease(i / 12.0), fuji_i=10 ** 6)
        # 06 の文字：上へぼけて抜ける
        e = M.ease((i + 1) / 4.0, 'cubic_in')
        if e < 0.999:
            f = b06.put_counter(f, gi, T6, alpha=1 - e, extra_dy=-15 * e, blur=4 * e)
        f = t_sub.exit(f, i, -1, dur=4)
        # 最後：着水点へのズームのぼかし＋濃紺へ＋線画をうすく
        j = i - (f_land + 1)
        jd = i - (n - DARKEN)
        if j >= 0:
            q = (j + 1) / out
            lx, ly = b06.project(Mc, X.X1, WL, clamp=None)
            f = core.zoom_blur(f, 0.24 * q ** 1.2, cx=lx, cy=ly, steps=18)
        if jd >= 0:
            qd = (jd + 1) / DARKEN
            d = 0.6 * qd + 0.4 * core.ease(qd, 'smooth')
            f = f * (1 - d) + nav * d
        if j >= 0:
            la = (0.14 * core.ease(q, 'in') * lines)[..., None]
            f = f * (1 - la) + core.FOAM * la
        # 秒のカウンター（画面に固定。ぼかしと暗くするのより後）
        if i >= f_so - 1:
            a = M.ease((i - f_so + 2) / 4.0, 'sine_inout')
            dy, bl = 0.0, 0.0
            if i >= f_exit:
                ee = M.ease((i - f_exit + 1) / 6.0, 'cubic_in')
                a *= 1 - ee
                dy, bl = -10 * ee, 5 * ee
            v = SECS * u
            dv = SECS * (fall_u(i + 0.5, f_land)[0] - fall_u(i - 0.5, f_land)[0])
            f = ctr.put(f, v, a, i=i, f_lock=f_land, dv=dv, dy=dy, blur=bl)
        yield np.clip(f, 0, 1).astype(np.float32)
