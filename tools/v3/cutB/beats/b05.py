"""B05（約4.3秒）「パンチは、やく、ひゃくきゅうじゅっトン。ゾウ、さんじゅうごとうぶん。」
濃紺の無地（中央だけごく弱く深い海の色）。上に「パンチの力」、その下に大きなカウンター「約190トン」
（190 を強調。ヤで出て数えはじめ、トンの「ン」で 190 に止まる）。注「体重60kg・パンチが当たる時間0.01秒・空気抵抗なし（試算）」。
「ゾウ、さんじゅうごとうぶん」：淡い青の塗りのアフリカゾウのシルエット 35 頭（7×5）が、上の列から順にぽんぽんと並ぶ。
「ゾウ 約35頭分」と注「アフリカゾウ（大人のオス）約5.5t として」（193 tf ÷ 5.5 t ≈ 35）。
2026-11-07 訂正：10km 版（F＝60kg×315m/s÷0.01s＝1.89×10⁶N＝約193トン重）。旧「約220トン・ゾウ40頭分」。
最初の 6 フレーム：04 の暗い地図からのズームのぼかしを 0 へ。最後の 6 フレーム：上へ振る縦の whip（06 が同じ向きで受ける）。

2026-11 動きの改善（director's motion spec §1-05）：
- カウンターは _motion.Odometer（数字が縦に回る）。8 を超えてから出し（「約0トン」は見せない）、ヤ→ン で expo_out。
  ンで 190 にロック（1.12→1.0 の back_out＋小さな揺れ 4px）。
- ゾウは上から落ちて（−42px・quad_in・5 フレーム。−70 は上の段に重なるため）、足もとを中心につぶれて（1.14×0.84）ばねで戻る（ζ0.45・4Hz）。
  まんなかの列から外へ、上の段から順に。段がそろうたびに全体が 2px 沈む（重さ）。1 頭ごとに砂ぼこり 2 粒。
- 「ゾウ 約35頭分」の 35 は「サ」の少し前から白で回り、「ゴ」で止まる（強調は 190 のまま）。ゾウは「ト」までに全部着く。
- 全体はごくゆっくり押し込む（1.00→1.05）。入りのズームのぼかしの中心は 04 のメーターの中心 (540, 900)。
- 出の whip：ずれ 0→300px（expo_in）、最後のぶれ 110px（06 の最初のフレームと同じ）。
2026-11-08 見た目：床をやめ、濃紺のスタジオ（#0B2238→光だまり #1F4A72）に正面の棚のように並べる（段ごとの接地の影だけ、映り込みなし）。
ゾウは 3D（_figures3d.elephant_rgba）の 9 種（向き ±8°×鼻の形 3）＋大きさ ±3%、キーは左上・リムは右上。仕上げ（粒 0.006）のあとに文字。
2026-11-08 r2：暗い映り込みの床（地平 y 975、映り込み 12%・ぼかし 8px）。ゾウは elephant_rgba_v2（先細りの脚・ひざ・
足の裏・厚い反った耳・鼻のしわ）を 9 種、1 頭ごとに向き・形・大きさ ±4%・明るさ ±3%。床の上に遠近で 5 段（奥 0.82 倍→手前、
段ごとに半分ずらす）、奥から描く（重なりは手前が前）。注は y 1194/1238（セーフゾーンの中）。
出は画面全体を上へ振る whip-tilt（中身は下へ、ぶれ 55px）。06 が同じ向きで受ける。
"""
import math

import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W, H
from beats import _mapui as U
from cutB.beats import _motion as M
from cutB.beats import _render as R, _figures3d as F3

COLS, ROWS = 7, 5
DX, Y_ROW0, DY = 118.0, 790.0, 75.0
ESC = 0.82                       # ゾウの大きさ（1単位＝0.92px、全長約 100 単位）
EL_RGB = core.SKY
WHIP_OUT = 6
WHIP_PX, WHIP_BLUR = 300.0, 55.0       # 06 の入りと同じ（2026-11-08 r2：画面全体を上へ振る whip-tilt、ぶれ 110 → 55）
WHIP_SIGN = 1                          # +1：中身が下へ動く＝カメラが上（06 の空）へ振る
PUSH = 1.05
# 2026-11-08 r2（美術監督）：暗い映り込みの床（地平 y≈975）。ゾウはこの床の上に遠近で並ぶ（奥の段 0.82 倍 → 手前 1.0 倍）
HZ5 = 975.0
ROW_Y0, ROW_DY = 1003.0, 30.0
ROW_S0, ROW_DS = 0.82, 0.045
REFL_A, REFL_BLUR = 0.12, 8.0
NOTE_Y = (1194, 1238)

_C = {}


def background():
    """スタジオの濃紺（美術監督 2026-11-08）：#0B2238 → 中央のやわらかい光だまり #1F4A72。床・つなぎ目・まだらはなし。"""
    if 'bg' in _C:
        return _C['bg']
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((x - W / 2) / (W * 0.62)) ** 2 + ((y - 860) / (H * 0.36)) ** 2
    k = np.exp(-r2 * 1.25)
    bg = R.hexl('0B2238') * (1 - k)[..., None] + R.hexl('1F4A72') * k[..., None]
    bg = bg * (1 - 0.25 * np.clip(r2 - 1.0, 0, 1.5) / 1.5)[..., None]
    # 床：地平の少し明るい紺 → 手前の深い紺。光だまりの 12% の映り込み（ぼかし）。地平の線は光らせない（やわらかい明るさの差だけ）
    fl = (y >= HZ5).astype(np.float32)[..., None]
    d = np.clip((y - HZ5) / (H - HZ5), 0, 1)
    top = bg[int(HZ5) - 1][None] * 0.80                                 # 地平では奥の壁と 80% の明るさでつなぐ
    floor = top * (1 - d ** 0.5)[..., None] + R.hexl('06152A') * (d ** 0.5)[..., None]
    yr = 2 * HZ5 - y
    r2m = ((x - W / 2) / (W * 0.62)) ** 2 + ((yr - 860) / (H * 0.36)) ** 2
    floor = floor + R.hexl('1F4A72') * (REFL_A * 1.6 * np.exp(-r2m * 1.25) * np.exp(-(y - HZ5) / 260.0))[..., None]
    bg = bg * (1 - fl) + floor * fl
    near = (np.abs(y - HZ5) < 4)[..., None]
    bg = cv2.GaussianBlur(bg.astype(np.float32), (1, 9), 0.0, sigmaY=1.2) * near + bg * (1 - near)    # 地平の線はやわらかく
    _C['bg'] = R.srgb(bg).astype(np.float32)
    return _C['bg']


KEY5 = (-0.55, -0.65, 0.52)
ENV5 = dict(sky=R.hexl('7FB4DE') * 0.34, ground=R.hexl('123E66') * 0.30, sun=np.array([1.0, 0.99, 0.98]) * 1.40,
            rim=R.hexl('CFEFFF') * 1.25, rim_dir=(0.70, -0.70))
YAWS = (12.0, 20.0, 28.0)


def elephants3d():
    """3D のゾウ 9 種（向き 12/20/28°＝±8° × 鼻と耳の形 3 つ）。(rgba, ax, ay) のリスト。1 回だけ描く（r2：elephant_rgba_v2）。"""
    if 'el3' not in _C:
        out = []
        for yaw in YAWS:
            for tr in (0, 1, 2):
                rgba, ax, ay = F3.elephant_rgba_v2(ESC * 1.06, yaw=yaw, light=KEY5, env=ENV5, trunk=tr, ear=tr)
                out.append((F3.cap_white(rgba), ax, ay))
        _C['el3'] = out
    return _C['el3']


def variant(r, c):
    """1 頭ごとのばらつき（いつも同じ）：(種類, 大きさ ±4%, 明るさ ±3%)。"""
    rng = np.random.default_rng(1000 + r * 17 + c)
    return int(rng.integers(0, 9)), float(rng.uniform(0.96, 1.04)), float(rng.uniform(0.97, 1.03))


def instance(r, c):
    """(ゾウの板, 映り込みの板)：明るさのばらつきを入れた 1 頭ぶん（1 回だけ作る）。"""
    key = ('inst', r, c)
    if key not in _C:
        vk, _, tone = variant(r, c)
        rgba, ax, ay = elephants3d()[vk]
        t = rgba.copy()
        t[..., :3] = np.clip(t[..., :3] * tone, 0, 1)
        fl = t[::-1].copy()
        pad = int(REFL_BLUR * 3)
        fl = np.pad(fl, ((pad, pad), (pad, pad), (0, 0)))
        prem = np.dstack([fl[..., :3] * fl[..., 3:], fl[..., 3:]])
        prem = cv2.GaussianBlur(prem, (0, 0), REFL_BLUR)
        refl = np.dstack([prem[..., :3] / np.maximum(prem[..., 3:], 1e-5), prem[..., 3:] * REFL_A])
        h = rgba.shape[0]
        _C[key] = ((t, ax, ay), (refl.astype(np.float32), ax + pad, (h - ay) + pad))
    return _C[key]


def row_pos(r, c):
    """床の上の段 r（0＝奥）・列 c の足もとの位置と、遠近の大きさ。奇数の段は半分ずらす。"""
    s_ = ROW_S0 + ROW_DS * r
    y = ROW_Y0 + ROW_DY * r
    x = W / 2 + (c - (COLS - 1) / 2 + (0.25 if r % 2 else -0.25)) * DX * s_ * 0.98
    return x, y, s_


def put_elephant(f, el, x, y, sx, sy, a, refl=None, y_floor=None):
    """床への映り込み（12%・ぼかし 8px、浮いている高さの分だけ下へ）→ 接地の影（足もとの環境の遮へい）→ ゾウ。"""
    rgba, ax, ay = el
    yf = y if y_floor is None else y_floor
    if refl is not None:
        r_rgba, rax, ray = refl
        f = put_sprite(f, r_rgba, rax, ray, x, yf + (yf - y), sx, a, sy=sy)
    if a > 0.3:
        lift = max(0.0, yf - y)
        k_ = float(np.exp(-lift / 20.0))
        f = R.contact_shadow(f, x + 3, yf + 1, 44 * sx, 5.0, opacity=0.35 * a * k_, rgb=core.c('050F1C'), blur=5.0)
        f = R.contact_shadow(f, x + 1, yf, 36 * sx, 2.2, opacity=0.55 * a * k_, rgb=core.c('050F1C'), blur=1.6)
    return put_sprite(f, rgba, ax, ay, x, y, sx, a, sy=sy)


def _spline(P, k=10):
    P = np.asarray(P, float)
    Q = np.concatenate([P[:1], P, P[-1:]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        t = np.linspace(0, 1, k, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 +
                          (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1:])
    return np.concatenate(out)


def elephant_sprite():
    """横向き（左向き）のアフリカゾウのシルエット（塗り＋耳と牙は細い抜きの線）。(rgba, ax, ay)：ax, ay は足もとの中心。"""
    if 'el' in _C:
        return _C['el']
    s, pad = 4, 6
    Wd, Hd = int(108 * ESC) + 2 * pad, int(74 * ESC) + 2 * pad
    L = core.SS(s=s, w=Wd, h=Hd)
    d = L.draw
    ox, oy = pad, Hd - pad                          # 単位 (0,0)＝左端・地面
    P = lambda pts: L.p([(ox + x * ESC, oy + y * ESC) for x, y in pts])
    B = lambda x0, y0, x1, y1: L.box(ox + x0 * ESC, oy + y0 * ESC, ox + x1 * ESC, oy + y1 * ESC)
    col = L.col(EL_RGB, 1.0)
    cut = (0, 0, 0, 0)
    # 胴（肩が高く、背中が少しくぼむ）
    back = _spline([(30, -40), (34, -60), (48, -65), (62, -60), (76, -61), (90, -55), (98, -42), (97, -26),
                    (86, -20), (60, -21), (40, -22), (30, -30)], 8)
    d.polygon(P(back), fill=col)
    # 脚（少し先細り、足先は少し広く）
    for x0, x1 in ((34, 45), (47, 57), (75, 85), (86, 96)):
        d.polygon(P([(x0, -32), (x1, -32), (x1 - 0.8, -3), (x1 + 0.6, 0), (x0 - 0.6, 0), (x0 + 0.8, -3)]), fill=col)
    # 頭
    d.ellipse(B(13, -70, 44, -36), fill=col)
    d.polygon(P([(16, -50), (28, -34), (40, -36), (38, -56)]), fill=col)
    # 鼻（太さが先へ細くなる。先は少し前へ巻く）
    tr = _spline([(19, -46), (12, -36), (9, -24), (9, -12), (11, -4), (15, -3)], 10)
    for k, p in enumerate(tr):
        r = 5.2 - 2.6 * k / (len(tr) - 1)
        x, y = p
        d.ellipse(B(x - r, y - r, x + r, y + r), fill=col)
    # 尾
    d.line(P([(97, -48), (101, -36), (102, -26)]), fill=col, width=L.w(1.6 * ESC), joint='curve')
    d.ellipse(B(100.5, -28, 104, -22), fill=col)
    # 耳（大きなアフリカゾウの耳）：抜きの線で形だけ
    ear = _spline([(36, -66), (50, -68), (58, -56), (56, -40), (50, -30), (43, -27), (38, -36), (35, -50), (36, -66)], 8)
    d.line(P(ear), fill=cut, width=L.w(1.6 * ESC), joint='curve')
    # 牙
    d.line(P(_spline([(22, -38), (18, -32), (12, -30), (7, -31)], 6)), fill=cut, width=L.w(2.0 * ESC), joint='curve')
    # 前脚と後ろ脚の間（奥の脚との境）を細く抜く
    for x in (46, 85.5):
        d.line(P([(x, -28), (x, 0)]), fill=cut, width=L.w(1.2 * ESC))
    rgba = L.done()
    _C['el'] = (rgba, pad + 55 * ESC, Hd - pad)
    return _C['el']


def put_sprite(f, rgba, ax, ay, x, y, scale=1.0, alpha=1.0, sy=None):
    """sy を渡すと横 scale・縦 sy（足もと (ax, ay) が中心）。"""
    if alpha <= 0.003:
        return f
    sx = scale
    sy = scale if sy is None else sy
    h, w = rgba.shape[:2]
    fx, fy = x - ax, y - ay
    ix, iy = int(np.floor(fx)), int(np.floor(fy))
    rx, ry = fx - ix, fy - iy
    pad = 8
    M2 = np.float32([[sx, 0, ax * (1 - sx) + rx + pad], [0, sy, ay * (1 - sy) + ry + pad]])
    rgba = cv2.warpAffine(rgba, M2, (w + 2 * pad, h + 2 * pad), flags=cv2.INTER_LINEAR)
    return U.paste(f, rgba, ix - pad, iy - pad, alpha)


ROW_D, COL_D = 3.8, 0.8   # 段ごと・列ごとの遅れ（最後のゾウが「ト」のころに着く。旧 4.5／0.9 は「ゴ」より 7 フレーム遅かった）


def el_delay(r, c):
    return r * ROW_D + abs(c - (COLS - 1) / 2) * COL_D


DROP = 5            # 落ちるフレーム数（DROP フレーム目に着く）
DROP_PX = 42.0      # 落ちる高さ（70 だと上の段・上の文字に重なって見えるので 42）


def elephants_at(i, f_zo):
    """i フレーム目に見えている各ゾウ (x, y, sx, sy, alpha)。"""
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            k = i - (f_zo + el_delay(r, c))
            if k < 0:
                continue
            x, y, s_ = row_pos(r, c)
            if k < DROP:
                q = M.ease((k + 1) / DROP, 'quad_in')
                y += -DROP_PX * s_ * (1 - q)
                sx, sy = 0.97, 1.05
                a = M.ease((k + 1) / 3.0, 'sine_inout')
            elif k < DROP + 2:
                sx, sy, a = 1.14, 0.84, 1.0
            else:
                sp = M.spring((k - DROP - 1) / core.FPS, 0.45, 4.0)
                sx, sy, a = 1.14 + (1 - 1.14) * sp, 0.84 + (1 - 0.84) * sp, 1.0
            out.append((x, y, sx * s_, sy * s_, a))
    return out


def landed(i, f_zo):
    """着いたゾウの数（小数：着く直前の 1 フレームで次の 1 頭へ回る）。"""
    v = 0.0
    for r in range(ROWS):
        for c in range(COLS):
            v += float(np.clip(i - (f_zo + el_delay(r, c) + DROP - 1), 0, 1))
    return v


def grid_nudge(i, f_zo):
    """段がそろうたびに 2px 沈んで戻る（重さ）。"""
    d = 0.0
    for r in range(ROWS):
        k = i - (f_zo + el_delay(r, 0) + DROP)
        if k >= 0:
            d += 2.0 * np.exp(-k / 4.0) * np.sin(np.pi * min(k, 12) / 4.0) * (k < 12)
    return d


_D = {}


def dust(f_zo):
    if f_zo not in _D:
        bs = []
        for r in range(ROWS):
            for c in range(COLS):
                x, y, _s = row_pos(r, c)
                f0 = f_zo + el_delay(r, c) + DROP
                for side, rng in ((-1, (-175, -140)), (1, (-40, -5))):
                    bs.append(M.Burst(r * 31 + c * 7 + (side > 0), 1, (x + side * 30, y - 2), angle_range=rng,
                                      speed_range=(1.5, 3.2), gravity=0.12, drag=0.15, life=8, size=(2.2, 3.4),
                                      f0=int(np.ceil(f0)), rgb=core.SAND, alpha=0.35, stretch=0.3))
        _D[f_zo] = bs
    return _D[f_zo]


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[4][0] == 'ヤ' and mo[13][0] == 'ン' and mo[14][0] == 'ゾ' and mo[16][0] == 'サ' and mo[20][0] == 'ゴ', mo
    mf = lambda k: timeline.mora_frame(beat, k)
    f_pa, f_ya, f_n, f_zo = mf(0), mf(4), mf(13), mf(14)
    f_sa, f_go = mf(16), mf(20)
    bg = background()
    el, ax, ay = elephant_sprite()
    head = M.KText([dict(text='パンチの力', x=W / 2, y=410, size=48, weight='Medium', track=0.12)])
    ctr = M.Odometer('190', W / 2, 588, digit_size=150, side_size=96, prefix='約', suffix='トン', lock_scale=1.12,
                     lock_drop=6.0, lock_dur=8)
    sub = M.Odometer('35', W / 2, 686, digit_size=48, side_size=48, prefix='ゾウ 約', suffix='頭分', weight='Medium',
                     side_weight='Medium', gap=0.08, digit_color=core.FOAM, lock_dur=6, lock_scale=1.08, lock_drop=3.0, hide_zero=True,
                     blur_k=0.3)
    note1 = M.KText([dict(text='アフリカゾウ（大人のオス）約5.5t として', x=W / 2, y=NOTE_Y[0], size=28, weight='Medium',
                          alpha=0.9, track=0.04)])
    note2 = M.KText([dict(text='体重60kg・パンチが当たる時間0.01秒・空気抵抗なし（試算）', x=W / 2, y=NOTE_Y[1], size=28,
                          weight='Medium', alpha=0.8, track=0.02)])
    dusts = dust(f_zo)

    def val(t):
        return 190 * M.ease((t - f_ya) / float(f_n - f_ya), 'expo_out')
    els = elephants3d()
    for i in range(n):
        f = bg.copy()
        # ゾウ（段がそろうたびに少し沈む）
        gy = grid_nudge(i, f_zo)
        vis = sorted(zip(elephants_at(i, f_zo), _order(i, f_zo)), key=lambda e: (e[1][0], e[1][1]))   # 奥の段から
        for (x, y, sx, sy, a), (r, c) in vis:
            _vk, vs, _t = variant(r, c)
            body, refl = instance(r, c)
            yf = row_pos(r, c)[1]
            f = put_elephant(f, body, x, y + gy, sx * vs, sy * vs, a * 0.97, refl=refl, y_floor=yf + gy)
        for b in dusts:
            if 0 <= i - b.f0 < 9:
                f = b.draw(f, i)
        # 全体のゆっくりした押し込み＋ロックの小さな揺れ
        tq = i / float(n - 1)
        k = 1 + (PUSH - 1) * (0.5 * tq + 0.5 * M.ease(tq, 'sine_inout'))     # 頭から少し動いている
        sh = M.shake(i, f_n, amp=4.0, tau=2.0, rot_deg=0.0, seed=5)
        cx, cy = W / 2, 800.0
        Mw = np.float32([[k, 0, cx * (1 - k) + sh[0]], [0, k, cy * (1 - k) + sh[1]]])
        # 文字は揺れを ±2px までに抑える（美術監督の規則：文字は ±2px より揺らさない）
        Mt = np.float32([[k, 0, cx * (1 - k) + 0.5 * sh[0]], [0, k, cy * (1 - k) + 0.5 * sh[1]]])
        f = cv2.warpAffine(f, Mw, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        f = R.finish_ad(f, 380 + i, grain=0.006, vignette=0.10)
        # 文字（仕上げの後。同じ押し込みを掛ける：黒の上と白の上の 2 枚から色とアルファを取り出す）
        outs = []
        for base in (np.zeros((H, W, 3), np.float32), np.ones((H, W, 3), np.float32)):
            g = head.reveal(base, i, f_pa - 2, 'char_rise', stagger=1.3, dur=10)
            v = val(i)
            if v >= 8 or i >= f_n:
                a = float(np.clip((v - 8) / 22.0, 0, 1)) if i < f_n else 1.0
                g = ctr.put(g, v, a, i=i, f_lock=f_n, dv=val(i + 0.5) - val(i - 0.5))
                g = note2.reveal(g, i, f_ya + 1, 'line_mask')
            s0 = f_sa - 3
            if i >= s0:
                cnt = lambda t: 35 * M.ease(min(1.0, max(0.0, (t - s0 + 1) / float(f_go - s0 + 1))), 'quad_out')
                ac = M.ease((i - s0 + 1) / 4.0, 'sine_inout')
                g = sub.put(g, cnt(i), ac, i=i, f_lock=f_go, dv=cnt(i + 0.5) - cnt(i - 0.5))
            if i >= f_zo + 6:
                g = note1.reveal(g, i, f_zo + 6, 'line_mask')
            outs.append(cv2.warpAffine(g, Mt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        ta = np.clip(1 - (outs[1] - outs[0]), 0, 1)
        # 入り：ズームのぼかしを 0 へ（04 のメーターの中心から）。文字はぼかさない（ぼかしの後に重ねる）
        if i < 6:
            f = core.zoom_blur(f, 0.12 * (1 - i / 6.0) ** 2, cx=540, cy=900, steps=14)
        # 出：上へ振る whip（06 が同じ速さで受ける）。文字は同じずれで上へ動きながら消える（ぶれはかけない）
        j = i - (n - WHIP_OUT)
        if j >= 0:
            q = (j + 1) / WHIP_OUT
            f = M.whip(f, q, axis=0, sign=WHIP_SIGN, px=WHIP_PX, blur=WHIP_BLUR, direction='out')
            e = M.ease(q, 'expo_in')
            Md = np.float32([[1, 0, 0], [0, 1, WHIP_SIGN * WHIP_PX * e]])
            ta = cv2.warpAffine(ta, Md, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            t0 = cv2.warpAffine(outs[0], Md, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            k_ = (1 - e) ** 2
            ta, t0 = ta * k_, t0 * k_
        else:
            t0 = outs[0]
        f = f * (1 - ta) + t0
        yield np.clip(f, 0, 1).astype(np.float32)


def _order(i, f_zo):
    """elephants_at と同じ順の (段, 列)。"""
    out = []
    for r in range(ROWS):
        for c in range(COLS):
            if i - (f_zo + el_delay(r, c)) >= 0:
                out.append((r, c))
    return out
