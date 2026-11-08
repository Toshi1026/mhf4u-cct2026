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
- 06 の文字は上へぼけて抜ける。06 のキラーンの名残は _art.put_glint_fx。
2026-11-08 見た目：着水は Splash（横へ低い冠の水滴・水面の泡のレース・塗りの泡の球。柱・壁・輪なし）。悪役は水面で切る。
着水点へのズームのぼかし → 仕上げ → ラベル（文字はぼかさない）→ 濃紺へ。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W, H, WL
from beats import _mapui as U
from beats import _tosa, b10
from cutB.beats import _art as A, _xsec as X, _motion as M, b06
from cutB.beats import _figures3d as F3, _render as R
import cv2
import math

LAND_AFTER_GO = 3                             # 着水（＝45秒）はナレーションの「ヨンジュウゴ」のゴの 3 フレーム後
SECS = 45.4                                   # 飛んでいる時間（試算）。表示は 23 → 45
DARKEN = 11                                   # 濃紺へ暗くするのは最後の 11 フレーム（1フレームの明るさの差を小さく）
DARK_MAX = 0.83                               # 2026-11-08：T の新しい最初のフレーム（立体の地図、平均 約0.16）と明るさをそろえる（全部濃紺にすると 0.09 で段差 0.07）


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
    return F3.villain_v2(float(x), float(y) - 14 * sc, sc, ang, 'fly', sec, M2=M2, cape_ang=cape)


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


def _sphere(c4, px, py, r, a, s):
    """4 倍の板に、光の当たった小さな球（塗り。輪郭の線なし）：体＋左上の明るい所＋1px のつやの点。"""
    body, hi, sp = c4
    P = lambda v: int(round(v * 16))
    cv2.circle(body, (P(px * s), P(py * s)), P(r * s), a, -1, cv2.LINE_AA, shift=4)
    cv2.circle(hi, (P((px - r * 0.3) * s), P((py - r * 0.3) * s)), P(r * 0.62 * s), a, -1, cv2.LINE_AA, shift=4)
    cv2.circle(sp, (P((px - r * 0.42) * s), P((py - r * 0.42) * s)), P(max(0.5, r * 0.22) * s), a, -1, cv2.LINE_AA,
               shift=4)


def _layer(f, x0, y0, w, h, s, c4, col_body, col_hi, a_body=1.0, smear=None):
    body, hi, sp = [cv2.resize(q, (w, h), interpolation=cv2.INTER_AREA) for q in c4]
    if smear is not None:
        k = smear
        body, hi = cv2.blur(body, k), cv2.blur(hi, k)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, f.shape[1]), min(y0 + h, f.shape[0])
    if xb <= xa or yb <= ya:
        return f
    sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    reg = f[ya:yb, xa:xb]
    ab = np.clip(body[sl] * a_body, 0, 1)[..., None]
    reg = reg * (1 - ab) + col_body * ab
    ah = np.clip(hi[sl], 0, 1)[..., None] * ab
    reg = reg * (1 - ah * 0.7) + col_hi * (ah * 0.7)
    asp = np.clip(sp[sl], 0, 1)[..., None]
    reg = reg * (1 - asp * 0.8) + core.c('E6F0F7') * (asp * 0.8)
    f[ya:yb, xa:xb] = reg
    return f


class Splash:
    """着水（美術監督 2026-11-08）：横へ低く開く冠の水滴（光の当たった球＋速さの向きの短い筋、高さ ≤60px）、
    水面を横へ広がる白い泡のレース、水の中へ入った泡（塗りの球 2–9px、ゆれながら上る）。縦の柱・壁・輪は描かない。"""

    def __init__(self, f0, x, y):
        rng = np.random.default_rng(23)
        self.f0, self.x, self.y = f0, x, y
        n = 26
        ang = np.concatenate([rng.uniform(-124, -56, 5), rng.uniform(-172, -132, 11), rng.uniform(-48, -8, 10)])
        sp = np.where(np.abs(ang + 90) < 40, rng.uniform(8.0, 11.5, n), rng.uniform(5.0, 10.0, n))
        self.v = np.stack([np.cos(np.radians(ang)) * sp, np.sin(np.radians(ang)) * sp], 1)
        self.r = rng.uniform(1.4, 3.6, n)
        self.life = rng.uniform(8, 13, n)
        self.g = 1.3
        nb = 34
        self.bx = rng.normal(0, 7, nb)
        self.by = rng.uniform(8, 95, nb)
        self.br = rng.uniform(1.0, 4.5, nb)
        self.bv = rng.uniform(1.2, 3.2, nb) * (0.6 + 0.4 * self.br / 4.5)
        self.bph = rng.uniform(0, 6.28, nb)
        self.bd = rng.uniform(0, 3, nb)

    def draw(self, f, i):
        k = i - self.f0
        if k < 0 or k > 30:
            return f
        s = 4
        x0, y0, w, h = int(self.x - 150), int(self.y - 80), 300, 230
        # 泡のレース（水面の上 ≤6px、下 3px。横へ広がって消える）
        if k <= 18:
            hw = 16 + 105 * M.ease((k + 1) / 12.0, 'expo_out')
            fade = 1 - M.ease((k - 3) / 15.0, 'sine_inout')
            xs = np.arange(w, dtype=np.float32)[None] + x0
            ys = np.arange(h, dtype=np.float32)[:, None] + y0
            dx = (xs - self.x) / hw
            env = np.clip(1 - dx * dx, 0, 1) ** 0.6
            th = (2.0 + 4.0 * env * (1 - 0.5 * M.ease(k / 14.0)))
            lace = R.fbm(xs / 7.0 + k * 0.15, ys / 3.0, 3, seed=52) * 0.5 + 0.5
            band = R.smoothstep(-th - 1, -th + 1.5, ys - self.y) * R.smoothstep(3.5, 1.0, ys - self.y)
            a = band * env * R.smoothstep(0.38, 0.62, lace + 0.15 * env) * fade * 0.9
            xb = min(x0 + w, f.shape[1])
            a = a[:, :xb - x0]
            reg = f[y0:y0 + h, x0:xb]
            col = core.c('DCE9F2') * (0.92 + 0.08 * (ys < self.y))[..., None]
            f[y0:y0 + h, x0:xb] = reg * (1 - a[..., None]) + col * a[..., None]
        # 水の中の泡（塗りの球、α0.5、ゆれながら上る）
        c4 = [np.zeros((h * s, w * s), np.float32) for _ in range(3)]
        g = M.ease((k + 1) / 5.0, 'expo_out')
        any_b = False
        for j in range(len(self.br)):
            kk = k - self.bd[j]
            if kk < 0:
                continue
            by = self.y + self.by[j] * g - self.bv[j] * kk
            if by < self.y + 4:
                continue
            bx = self.x + self.bx[j] * (1 + 0.15 * kk) + 2.5 * math.sin(kk * 0.7 + self.bph[j])
            a = 0.5 * min(1.0, (kk + 1) / 2.0) * float(np.clip((by - self.y - 4) / 10.0, 0, 1))
            _sphere(c4, bx - x0, by - y0, self.br[j], a, s)
            any_b = True
        if any_b:
            f = _layer(f, x0, y0, w, h, s, c4, core.c('9FCBEA'), core.c('D5EBF8'))
        # 冠の水滴（水面より上）
        c4 = [np.zeros((h * s, w * s), np.float32) for _ in range(3)]
        any_d = False
        for j in range(len(self.r)):
            if k > self.life[j]:
                continue
            px = self.x + self.v[j, 0] * k * (1 - 0.02 * k)
            py = self.y - 2 + self.v[j, 1] * k + 0.5 * self.g * k * k
            if py > self.y:
                continue
            a = float(np.clip(1 - k / self.life[j], 0, 1) ** 0.7)
            vy = self.v[j, 1] + self.g * k
            # 速さの向きの短い筋（3 つの球を少しずつずらして重ねる）
            for q in (0.0, 0.33, 0.66):
                _sphere(c4, px - self.v[j, 0] * q * 0.5 - x0, py - vy * q * 0.5 - y0, self.r[j] * (1 - 0.25 * q),
                        a * (1 - 0.35 * q), s)
            any_d = True
        if any_d:
            f = _layer(f, x0, y0, w, h, s, c4, core.c('5E9CCF'), core.c('DDEDF8'), a_body=0.9)
        return f


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
    splash = Splash(f_land, X.X1, WL)
    f_exit = f_land + LOCK_HOLD
    for i in range(n):
        gi = n6 + i                              # 06 の頭からの通し番号
        fade6 = 1.0 - M.ease((i + 1) / 9.0, 'sine_inout')
        Mc = cam_M(i, M_end)
        bx = b06.label_boxes(gi, T6, Mc)
        f = b06.scene(gi, T6, fuji_a=fade6, fuji=(1.0, 1.0), boxes=bx)
        u, tau = fall_u(i, f_land)
        rgba, x0, y0 = X.trail_layer(u, ghost=1.0 - core.ease(i / 30.0) * 0.4)
        f = U.paste(f, b06.X3.knockout(rgba, x0, y0, bx), x0, y0)
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
                d = np.array(X.arc(ub)) - np.array(X.arc(ua))
                r = villain_draw(float(i), f_land, n6)
                if r is not None:
                    rgba_v, vx0, vy0 = r
                    # 水面より下は描かない（水に入った所で切る）
                    cut = int(WL - vy0)
                    if 0 <= cut < rgba_v.shape[0]:
                        rgba_v = rgba_v.copy()
                        rgba_v[cut:, :, 3] = 0
                    if va > 0.99:            # 1 回描いて速さの向きにぶらす（サブフレームの代わり）
                        f = b06.put_smeared(f, (rgba_v, vx0, vy0), float(d[0]), float(d[1]), shutter=0.4)
                    else:
                        f = M.put_rgba(f, rgba_v, vx0, vy0, alpha=va)
            else:
                x, y = X.arc(u)
                f = A.put_dot(f, float(x), float(y) - 2, 2.4, alpha=va)
        f = splash.draw(f, i)
        # 06 のキラーンの名残
        f = A.put_glint_fx(f, X.APEX[0], X.APEX[1] - 2, b06.glint_k(gi, T6), r=42)
        # カメラ（06 の寄りから戻す）→ 着水点へのズームのぼかし → 仕上げ → ラベル（文字はぼかさない）
        f = b06.warp(f, Mc)
        j = i - (f_land + 1)
        jd = i - (n - DARKEN)
        if j >= 0:
            q = (j + 1) / out
            lx, ly = b06.project(Mc, X.X1, WL, clamp=None)
            f = core.zoom_blur(f, 0.24 * q ** 1.2, cx=lx, cy=ly, steps=18)
        f = R.finish_ad(f, 592 + i, grain=0.005, vignette=0.10)
        f = b06.put_labels(f, gi, T6, Mc, fuji_a=fade6, apex_a=1.0 - 0.4 * core.ease(i / 12.0), fuji_i=10 ** 6)
        # 06 の文字：上へぼけて抜ける
        e = M.ease((i + 1) / 4.0, 'cubic_in')
        if e < 0.999:
            f = b06.put_counter(f, gi, T6, alpha=1 - e, extra_dy=-15 * e, blur=4 * e)
        f = t_sub.exit(f, i, -1, dur=4)
        # 最後：濃紺へ＋線画をうすく
        if jd >= 0:
            qd = (jd + 1) / DARKEN
            d = (0.6 * qd + 0.4 * core.ease(qd, 'smooth')) * DARK_MAX
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
