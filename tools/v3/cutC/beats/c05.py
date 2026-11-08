"""案C 05（約 2.0 秒）「ゾウ250頭分。」  テロップ 小「威力 約1,400トン ＝」＋ ゾウ約**250**頭分（250＝強調）。

2026-11-08 クライアント修正 5：ゾウをクレーターのまわりに輪で囲むと見えにくい → クレーターの「中」に入れる。
威力 1,400 トンがゾウ何頭分かを見せるので、テロップの小さい行「威力 約1,400トン ＝」で 04 の数字とつなぐ。
画：04 の引きからカット。クレーターの中をのぞきこむ高い斜め（_shots ショット D、仰角 約 50°、1.00→1.14 の押し込み）。
アフリカゾウ 250 頭（_figures3d.elephant_rgba_v2 の 3D の板：上から見下ろす角度 PITCH で描いた 4 向き × 鼻と耳 2 ×
左右反転）が碗の底から内壁の等高線に沿って列になって詰まっていく（底の列から順に、水がたまるように）。大きさは遠近
（1 頭 長さ約 6m）。カメラから足もとが見える所にだけ置く（手前の縁に隠れる所には置かない）。
1 頭ずつ 6 フレームで上から降りて（quad_in）足もとでつぶれて戻る。地面の高さはクレーターの高さの場から。足もとに接地の影。
カウンターは「ゾ」から着いた頭数で回り、「ト」で 250 に止まる（最後のゾウもそのころ着く）。04 と同じベースライン。
最後の 5 フレームで文字は上へぼけて抜け、空へ振り上がって 06 へ（tail_whip）。くすぶる煙は薄く（ゾウを隠さない）。
"""
import math

import cv2
import numpy as np

import core
from core import W, H
import timeline_c as timeline
from cutB.beats import _render as R, _motion as M, _figures3d as F3
from cutC.beats import _cfx as CF, _shots as SH, c04

TEL_Y = c04.TEL_Y
N_EL = 250
EL_LEN = 6.2                                     # m（鼻から尾まで）
DROP = 6
PITCH = 46.0                                     # ゾウの板を見下ろす角度（ショット D の仰角に合わせる）
ROW_DR, ARC = 3.3, 4.7                           # 列の間（m、半径の向き）・列の中の間（m）
CAP = '威力 約1,400トン ＝'
_C = {}


def timing(beat):
    mo = beat['moras']
    assert mo[0][0] == 'ゾ' and mo[8][0] == 'ト', mo
    mf = lambda k: timeline.mora_frame(beat, k)
    return dict(zo=mf(0), to=mf(8), out=beat['n'] - 5)


def sprites():
    """ゾウの板 16 種（向き 4 × 鼻と耳 2 × 左右反転）。(rgba, ax, ay, 板の長さ px)。灰色の肌へ色をつけなおす。"""
    if 'sp' not in _C:
        out = []
        env = dict(F3.ENV_DAY)
        for yaw in (8.0, 24.0, 42.0, 64.0):
            for tr in (0, 2):
                rgba, ax, ay = F3.elephant_rgba_v2(0.62, yaw=yaw, pitch=PITCH, light=F3.KEY, env=env, trunk=tr, ear=tr)
                l = R.luma(rgba[..., :3])[..., None]
                rgba = rgba.copy()
                rgba[..., :3] = np.clip(l * core.c('A3A9AD') * 1.12, 0, 1)
                ln = 0.62 * 110.0                     # 板の 1 頭の長さ（モデルの単位 約 110 × esc）
                out.append((rgba, ax, ay, ln))
                out.append((rgba[:, ::-1].copy(), rgba.shape[1] - ax, ay, ln))
        _C['sp'] = out
    return _C['sp']


def layout():
    """各ゾウの (x, z, 降り立つ順の 0..1, 種類, 大きさのばらつき) と地面の高さ。
    碗の中の等高線の列（半径 ROW_DR おき・列の中は ARC おき）で、カメラ（ショット D）から足もとが見える所だけを、
    低い列（底）から順に N_EL 頭。同じ列の中は手前のまんなかから左右へ。"""
    if 'lay' not in _C:
        rng = np.random.default_rng(250)
        pl, dep, cam = SH.plate_D()
        Hm = SH.crater()[0]
        cand = []
        for ri, r in enumerate(np.arange(2.0, 46.0, ROW_DR)):
            n = max(1, int(2 * math.pi * r / ARC))
            th = (np.arange(n) + 0.5 * (ri % 2)) * 2 * math.pi / n + rng.normal(0, 0.02, n)
            rr = r + rng.normal(0, 0.35, n)
            X, Z = (rr * np.cos(th)).astype(np.float32), (rr * np.sin(th)).astype(np.float32)
            Y = SH._grid_sample(Hm, X.reshape(1, -1), Z.reshape(1, -1)).ravel()
            P = np.stack([X, Y + 0.3, Z], 1)
            S, _ = cam.project(P)
            d = np.linalg.norm(P - cam.eye, axis=1)
            xi = np.clip(S[:, 0].astype(int), 0, dep.shape[1] - 1)
            yi = np.clip(S[:, 1].astype(int), 0, dep.shape[0] - 1)
            ok = d <= dep[yi, xi] + 0.8
            front = np.abs(((th - math.atan2(cam.eye[2], cam.eye[0]) + math.pi) % (2 * math.pi)) - math.pi)
            for k in np.nonzero(ok)[0]:
                cand.append((float(Y[k]), ri, float(front[k]), float(X[k]), float(Z[k])))
        cand.sort(key=lambda c: (c[1], c[2]))
        assert len(cand) >= N_EL, len(cand)
        cand = cand[:N_EL]
        items = [[c[3], c[4], j / float(N_EL - 1), int(rng.integers(0, 16)), float(rng.uniform(0.93, 1.07))]
                 for j, c in enumerate(cand)]
        Y = np.array([c[0] for c in cand], np.float32)
        _C['lay'] = (items, Y)
    return _C['lay']


def land_frame(q, T):
    return T['zo'] + q * (T['to'] - DROP + 1 - T['zo'])


def landed(i, T):
    items, _ = layout()
    v = 0.0
    for it in items:
        v += float(np.clip(i - (land_frame(it[2], T) + DROP - 1), 0, 1))
    return v


def put_elephants(f, cam, Mm, i, T):
    items, Y = layout()
    sp = sprites()
    vis = []
    for it, y in zip(items, Y):
        k = i - land_frame(it[2], T)
        if k < 0:
            continue
        P = np.array([[it[0], y, it[1]]])
        S, z, ppm = CF.project(cam, Mm, P)
        vis.append((float(z[0]), it, float(S[0, 0]), float(S[0, 1]), float(ppm[0]), k))
    if not vis:
        return f
    vis.sort(key=lambda q: -q[0])
    # 接地の影（まとめて 1 回）
    xs = [v[2] for v in vis]
    ys = [v[3] for v in vis]
    bx0, by0 = int(max(0, min(xs) - 80)), int(max(0, min(ys) - 60))
    bx1, by1 = int(min(W, max(xs) + 80)), int(min(H, max(ys) + 40))
    if bx1 > bx0 and by1 > by0:
        sh = np.zeros((by1 - by0, bx1 - bx0), np.float32)
        for z, it, x, y, ppm, k in vis:
            if k >= DROP - 1:
                rx = EL_LEN * 0.42 * ppm * it[4]
                cv2.ellipse(sh, (int((x - bx0) * 4), int((y - by0) * 4)), (int(rx * 4), int(rx * 0.45 * 4)), 0, 0, 360,
                            1.0, -1, cv2.LINE_AA, shift=2)
        sh = cv2.GaussianBlur(sh, (0, 0), 2.0)[..., None] * 0.45
        f[by0:by1, bx0:bx1] = f[by0:by1, bx0:bx1] * (1 - sh * (1 - core.c('2A2620')))
    for z, it, x, y, ppm, k in vis:
        rgba, ax, ay, ln = sp[it[3]]
        s = EL_LEN * ppm * it[4] / ln
        dy, sx, sy, a = 0.0, 1.0, 1.0, 1.0
        if k < DROP:
            q = M.ease((k + 1) / DROP, 'quad_in')
            dy = -5.0 * ppm * (1 - q)
            sx, sy = 0.96, 1.05
            a = M.ease((k + 1) / 3.0, 'sine_inout')
        elif k < DROP + 2:
            sx, sy = 1.08, 0.88
        else:
            sp_ = M.spring((k - DROP - 1) / core.FPS, 0.45, 4.0)
            sx, sy = 1.08 + (1 - 1.08) * sp_, 0.88 + (1 - 0.88) * sp_
        f = put_sprite(f, rgba, ax, ay, x, y + dy, s * sx, s * sy, a)
    return f


def put_sprite(f, rgba, ax, ay, x, y, sx, sy, alpha):
    h, w = rgba.shape[:2]
    ow, oh = int(math.ceil(w * sx)) + 4, int(math.ceil(h * sy)) + 4
    x0, y0 = x - ax * sx, y - ay * sy
    ix, iy = int(math.floor(x0)), int(math.floor(y0))
    M2 = np.float32([[sx, 0, x0 - ix], [0, sy, y0 - iy]])
    r = cv2.warpAffine(rgba, M2, (ow, oh), flags=cv2.INTER_LINEAR)
    xa, ya, xb, yb = max(ix, 0), max(iy, 0), min(ix + ow, W), min(iy + oh, H)
    if xb <= xa or yb <= ya:
        return f
    sub = r[ya - iy:yb - iy, xa - ix:xb - ix]
    a = sub[..., 3:4] * alpha
    f[ya:yb, xa:xb] = f[ya:yb, xa:xb] * (1 - a) + sub[..., :3] * a
    return f


def odo():
    if 'o' not in _C:
        _C['o'] = M.Odometer('250', W / 2, TEL_Y, digit_size=150, side_size=96, prefix='ゾウ約', suffix='頭分',
                             lock_scale=1.10, lock_drop=6.0, lock_dur=8, hide_zero=True, feather_px=10)
    return _C['o']


def put_text(f, i, T):
    o = odo()
    # 小テロップ「威力 約1,400トン ＝」：頭から 5 フレームで下からふわっと（04 の「パンチの威力」と同じ所）、最後は一緒に抜ける
    qa = M.ease((i + 1) / 5.0, 'cubic_out')
    if i < T['out']:
        f = c04.put_cap(f, c04.cap_layer(CAP), qa, dy=12 * (1 - qa))
    else:
        q = M.ease((i - T['out'] + 1) / 5.0, 'cubic_in')
        f = c04.put_cap(f, c04.cap_layer(CAP), 1 - q, dy=-30 * q, blur=6 * q)
    if i < T['zo']:
        return f
    cnt = lambda t: min(float(N_EL), landed(t, T) * (N_EL / float(len(layout()[0]))))
    v = cnt(i)
    a = float(np.clip(v / 3.0, 0, 1)) if i < T['to'] else 1.0               # 1 頭目が着くまで出さない
    dv = cnt(i + 0.5) - cnt(i - 0.5)
    if i < T['out']:
        return o.put(f, v, a, i=i, f_lock=T['to'], dv=dv)
    q = M.ease((i - T['out'] + 1) / 5.0, 'cubic_in')
    outs = [o.put(base, v, 1.0, i=i, f_lock=T['to'], dv=0.0, dy=-30 * q) for base in (np.zeros_like(f), np.ones_like(f))]
    ta = np.clip(1 - (outs[1] - outs[0]), 0, 1)
    t0 = outs[0]
    if q > 0.05:
        ta = cv2.GaussianBlur(ta, (0, 0), 6 * q)
        t0 = cv2.GaussianBlur(t0, (0, 0), 6 * q)
    return f * (1 - ta * (1 - q)) + t0 * (1 - q)


TAIL = 4                                       # 06 への振り（最後の 4 フレーム）
SKY = np.float32([0.62, 0.78, 0.92])           # 06 の空の色（輝度 約 0.76）
LUM = np.float32([0.2126, 0.7152, 0.0722])


def tail_whip(f, i, n, st):
    """06 へ：カメラが空（悪役の飛んだ方）へ振り上がる。中身が下へ流れ（06 の入りの振りと同じ向き）、
    明るさを空の色へ少しずつ寄せる。草原（約 0.49）→ 空（約 0.69）の切り替えを 1 フレーム 0.035 ずつに分ける
    （06 の頭の 3 フレームが残りを受け持つ。06.HEAD_FROM）。"""
    k = i - (n - TAIL)
    if k < 0:
        return f
    if k == 0:
        st['L0'] = float((f * LUM).sum(axis=2).mean())
    f = M.whip(f, (k + 1) / float(TAIL + 1), axis=0, sign=+1, px=160.0, blur=90.0, direction='out')
    if k > 0:
        L = float((f * LUM).sum(axis=2).mean())
        tgt = st['L0'] + 0.035 * k
        Ls = float((SKY * LUM).sum())
        w = float(np.clip((tgt - L) / max(Ls - L, 1e-3), 0, 1))
        f = f * (1 - w) + SKY * w
    return f


def render(beat, ctx):
    n = beat['n']
    T = timing(beat)
    st = {}
    for i in range(n):
        f, cam, Mm, g = CF.frame_C(beat, i, extra=lambda f, cam, Mm, g, i=i: put_elephants(f, cam, Mm, i, T), shot='D',
                                   smoke=0.35)
        sh = M.shake(i, T['to'], amp=3.0, tau=2.0, rot_deg=0.0, seed=9)
        if abs(sh[0]) + abs(sh[1]) > 0.05:
            f = cv2.warpAffine(f, np.float32([[1, 0, sh[0]], [0, 1, sh[1]]]), (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        f = R.finish_ad(f, 2200 + i, grain=0.006, vignette=0.10, ca_px=0.25)
        f = f * c04.scrim()
        f = put_text(f, i, T)
        f = tail_whip(f, i, n, st)
        yield np.clip(f, 0, 1).astype(np.float32)
