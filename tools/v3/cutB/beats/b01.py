"""B01（0.0–3.3秒）「アニメの悪役は、パンチ一発で、空のかなたへ。」
朝の空と低い水平線、手前の浜。顔のない白い悪役（小さなマント、両手を腰に当てて胸を張る）。
「パンチ」で画面の左下の端から白い線画のマンガの拳（丸い指の山・親指・短い袖口。腕は描かず、うしろに速さの線）が入り、「チ」で当たる。
悪役は右上へ弧を描いて飛び、回りながら小さくなって（うしろに速さの線）点になり、「カナタ」でキラーンと1回光る。
文字：「パンチ一発で」（パ）、「空のかなたへ」（ソ）。

2026-11 動きの改善（director's motion spec §1-01）：
- 世界のレイヤー（空・海・浜・悪役・拳・跡）にカメラ（_motion.Cam2D）。はじめは 240px 上へずらし（足 y≈1190、当たる所 y≈1100：
  セーフゾーンの中）、0–30 で 1.10→1.00 に引いて見せる。当たり+3 からばね（ζ0.8・2.2Hz）で悪役を追って上へ振り、キラーンまでに 0 へ
  （DOT は 02 と同じ画面の位置）。世界は下に 300px 足した背景で描く。文字と CG ラベルは画面に固定。
- 01・02 を通した DOT 中心の押し込み（_sky.push_k、sine_in）。
- 殴る流れ：胸を張って弾む（つぶれと伸び）→ 37–42 で拳に気づいてのけぞる → 33–39 拳が 130px 顔を出す → 39–43 少し引く →
  43–46 打ち込む（サブフレームのぶれ）→ 46–48 ヒットストップ（打つ向きにつぶれる・衝撃の星・放射の線・火花）＋カメラの揺れ →
  49– 打ち出し（速さの向きに伸びてばねで戻る、足もとの砂ぼこり）→ 拳は少し押し込んでから back_in で去る。
- 飛ぶ悪役：回転は expo_out で 720°、マントは画面の向きで速さの反対へ（2 フレーム遅れ）、速いときはサブフレームのぶれ。
- キラーン：_art.put_glint_fx（行き過ぎ 1.15、回転、小さな星 2 つ、横の筋）。一番大きいのは glint+5（効果音の頭）。
- 文字：_motion.KText の 1 文字ずつの立ち上がり（「空のかなたへ」は強調のポップつき）。"""
import math

import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W, H
from beats import _geo as G
from cutB.beats import _art as A, _sky as S, _motion as M

FS = 1.15
DIR = -27.0                                  # 拳の向き（度）
DV = np.array([math.cos(math.radians(DIR)), math.sin(math.radians(DIR))])
CONTACT = (392.0, 1342.0)                    # 拳が当たる所（悪役の胴の左）（世界の座標）
BEZ_C = (1000.0, 900.0)                      # 飛んでいく弧の制御点
EXTRA = 300                                  # 世界の背景を下へ足す行
OFF0 = -240.0                                # はじめのカメラのずれ（世界を上へ）
HOLD = 3                                     # ヒットストップのフレーム数
HITC = np.array(CONTACT) - 68 * FS * DV      # 当たったときの拳のまんなか


def bez(t, p0, c, p2):
    t = np.asarray(t, np.float64)[..., None]
    return (1 - t) ** 2 * np.asarray(p0) + 2 * t * (1 - t) * np.asarray(c) + t ** 2 * np.asarray(p2)


def torso():
    return (S.FEET[0], S.FEET[1] - 16 * S.VSCALE)


def timing(beat):
    mf = lambda k: timeline.mora_frame(beat, k)
    mo = beat['moras']
    assert mo[9][0] == 'パ' and mo[11][0] == 'チ' and mo[17][0] == 'ソ' and mo[20][0] == 'カ', mo
    return dict(pa=mf(9), hit=mf(11) - 1, so=mf(17), glint=mf(20) + 3)


def push_end(beat=None):
    """01・02 の通しの押し込みが終わる（02 のズームスルーが始まる）通し番号。"""
    bs = timeline.load()
    b1 = next(b for b in bs if b['key'] == '01')
    b2 = next(b for b in bs if b['key'] == '02')
    return b2['f0'] - b1['f0'] + 46


# ---------------------------------------------------------------- 飛ぶ
def flight_tau(ta, T):
    """飛ぶ進み具合（0..1）。ta はヒットストップで付け替えた時間。キラーンのフレームで 1。"""
    return float(np.clip((ta - T['hit']) / float(T['glint'] - HOLD - T['hit']), 0, 1))


def flight(ta, T):
    """飛んでいる悪役の (x, y, scale, ang, 進み具合 tp)。"""
    tau = flight_tau(ta, T)
    tp = 1 - (1 - tau) ** 2.2
    x, y = bez(tp, torso(), BEZ_C, S.DOT)
    sc = S.VSCALE * np.exp(np.log(0.12 / S.VSCALE) * tau ** 1.25)
    ang = -30 + 600 * M.ease(tau, 'cubic_out')         # 速く回りはじめ、小さくなるにつれて落ち着く
    return float(x), float(y), float(sc), float(ang), float(tp)


def flight_vel(ta, T, h=0.5):
    a = np.array(flight(ta - h, T)[:2])
    b = np.array(flight(ta + h, T)[:2])
    return (b - a) / (2 * h)


def trail(f, tp, tau, alpha):
    """飛んだ跡のうすい弧（太さは手前ほど太い）と、悪役のうしろの速さの線。"""
    if alpha <= 0.003 or tp <= 0.002:
        return f
    ts = np.linspace(0, tp, 60)
    P = bez(ts, torso(), BEZ_C, S.DOT)
    x0, y0 = int(max(0, P[:, 0].min() - 40)), int(max(0, P[:, 1].min() - 40))
    x1, y1 = int(min(W, P[:, 0].max() + 40)), int(min(f.shape[0], P[:, 1].max() + 40))
    L = core.SS(s=2, w=x1 - x0, h=y1 - y0)
    Q = P - (x0, y0)
    for k in range(len(Q) - 1):
        u = ts[k] / max(tp, 1e-6)
        wd = 1.2 + 4.0 * (1 - ts[k]) ** 1.5
        a = alpha * 0.30 * (0.25 + 0.75 * u)
        L.draw.line(L.p([tuple(Q[k]), tuple(Q[k + 1])]), fill=L.col(core.FOAM, a), width=L.w(wd))
    if tp < 0.97:
        p1 = bez(tp, torso(), BEZ_C, S.DOT)
        p0 = bez(max(0, tp - 0.06), torso(), BEZ_C, S.DOT)
        v = p1 - p0
        sp = np.hypot(*v)
        if sp > 1:
            v /= sp
            nrm = np.array([-v[1], v[0]])
            sc = S.VSCALE * np.exp(np.log(0.12 / S.VSCALE) * tau ** 1.25)
            ln = min(260, sp * 3.2)
            for off, k in ((-1, 0.7), (0, 1.0), (1, 0.7)):
                a0 = p1 - v * (14 * sc) + nrm * off * 9 * sc
                a1 = a0 - v * ln * k
                L.draw.line(L.p([tuple(a0 - (x0, y0)), tuple(a1 - (x0, y0))]),
                            fill=L.col(core.FOAM, alpha * 0.55 * (1 - tp) ** 0.5), width=L.w(max(1.0, 1.6 * sc / 2)))
    lay = L.done()
    reg = f[y0:y1, x0:x1]
    a = lay[..., 3:4]
    f[y0:y1, x0:x1] = reg * (1 - a) + lay[..., :3] * a
    return f


# ---------------------------------------------------------------- 当たり
def star_pts(cx, cy, r, ang=0.0, n=8, waist=0.42, jag=0.0):
    pts = []
    for k in range(2 * n):
        th = math.radians(ang) + math.pi * k / n
        rr = r if k % 2 == 0 else r * waist
        if k % 2 == 0 and jag:
            rr *= 1 - jag * ((k * 7919) % 5) / 5.0
        pts.append((cx + rr * math.cos(th), cy + rr * math.sin(th)))
    return pts


def impact(f, k):
    """当たった瞬間（k=0..2：ヒットストップ）：とげとげの衝撃の星（半径 60→90）＋外へ伸びて細くなる放射の線。輪は描かない。
    星はヒットストップのあと 3 フレームで消える。画面全体は光らせない。"""
    if k < 0 or k >= 7:
        return f
    cx, cy = CONTACT
    R = 210

    def fn(L, ox, oy):
        if k < HOLD + 3:
            q = min(1.0, k / float(HOLD - 1))
            r = 60 + 30 * M.ease(q, 'expo_out')
            a = 0.92 if k < HOLD else 0.92 * (1 - (k - HOLD + 1) / 4.0)
            L.draw.polygon(L.p(star_pts(ox, oy, r, ang=DIR + 11, n=8, waist=0.40, jag=0.35)),
                           fill=L.col(core.FOAM, a))
            L.draw.polygon(L.p(star_pts(ox, oy, r * 0.52, ang=DIR + 33, n=8, waist=0.5, jag=0.2)),
                           fill=L.col(core.SKY, a * 0.55))
        back = math.radians(DIR + 180)
        e = M.ease((k + 1) / 7.0, 'expo_out')
        for j in range(10):
            th = back + math.radians(48 + j * 29 + (j % 3) * 4)
            r0 = 70 + 70 * e
            r1 = r0 + (30 + 8 * (j % 2)) * (0.6 + 0.8 * e)
            wd = 4.0 - 3.0 * (k / 6.0)
            L.draw.line(L.p([(ox + r0 * math.cos(th), oy + r0 * math.sin(th)),
                             (ox + r1 * math.cos(th), oy + r1 * math.sin(th))]),
                        fill=L.col(core.FOAM, 0.9 * (1 - k / 7.0) ** 0.8), width=L.w(wd))
    return A.put(f, fn, cx, cy, R, s=3)


SPARKS = None
DUST = None


def particles(T):
    global SPARKS, DUST
    if SPARKS is None or SPARKS.f0 != T['hit']:
        SPARKS = M.Burst(11, 6, CONTACT, angle_range=(DIR + 180 - 70, DIR + 180 + 70), speed_range=(9, 17),
                         gravity=0.35, drag=0.18, life=8, size=(1.6, 2.6), f0=T['hit'], alpha=0.95, stretch=0.9)
        DUST = M.Burst(5, 7, (S.FEET[0] + 4, S.FEET[1] - 4), angle_range=(-165, -15), speed_range=(2.5, 6.5),
                       gravity=0.42, drag=0.10, life=11, size=(3.0, 6.0), f0=T['hit'] + HOLD, rgb=core.LAND,
                       alpha=0.5, stretch=0.25)
    return SPARKS, DUST


# ---------------------------------------------------------------- 拳
_FIST = {}


def fist_sprite():
    """拳の板を 1 回だけ描く（向きは DIR 固定、影つき）。(rgba, ax, ay)：ax, ay は拳のまんなか。"""
    if 'f' not in _FIST:
        pad = 40
        pts = np.vstack(list(A._fist_shapes().values()))
        P = pts @ A._R(DIR).T * FS
        x0, y0 = P[:, 0].min() - pad, P[:, 1].min() - pad
        w, h = int(np.ceil(P[:, 0].max() + pad - x0)), int(np.ceil(P[:, 1].max() + pad - y0))
        L = core.SS(s=3, w=w, h=h)
        A.fist(L, -x0, -y0, DIR, FS, 1.0)
        lay = L.done()
        sa = np.clip(cv2_blur(lay[..., 3], 6) * 0.30, 0, 1)
        a = lay[..., 3]
        out_a = a + sa * (1 - a)
        rgb = (lay[..., :3] * a[..., None] + A.SHADOW * (sa * (1 - a))[..., None]) / np.maximum(out_a, 1e-5)[..., None]
        _FIST['f'] = (np.concatenate([rgb, out_a[..., None]], 2).astype(np.float32), -x0, -y0)
    return _FIST['f']


def cv2_blur(a, s):
    import cv2
    return cv2.GaussianBlur(np.ascontiguousarray(a), (0, 0), s)


def fist_s(i, T):
    """拳の進み s（DIR に沿った HITC からの距離、負＝手前）。None＝画面にいない。i は小数でもよい。"""
    h = T['hit']
    s_out = -900.0
    s_poke = -225.0
    if i < h - 13:
        return None
    if i < h - 7:                                  # 顔を出す（expo_out、6 フレーム）
        return s_out + (s_poke - s_out) * M.ease((i - (h - 13)) / 6.0, 'expo_out')
    if i < h - 3:                                  # 少し引く（ため）
        return s_poke - 35 * M.ease((i - (h - 7)) / 4.0, 'sine_inout')
    if i < h:                                      # 打ち込む
        return (s_poke - 35) * (1 - M.ease((i - (h - 3)) / 3.0, 'cubic_in'))
    if i < h + HOLD:                               # ヒットストップ：少し食い込んだまま止まる
        return 10.0
    if i < h + HOLD + 3:                           # 押し込み（フォロースルー）
        return 10 + 30 * M.ease((i - h - HOLD) / 3.0, 'quad_out')
    q = (i - h - HOLD - 3) / 9.0                   # 去る（back_in、9 フレーム）
    if q >= 1:
        return None
    return 40 - 980 * M.ease(q, 'back_in')


def put_fist(f, i, T):
    s = fist_s(i, T)
    if s is None:
        return f
    rgba, ax, ay = fist_sprite()
    p = HITC + s * DV
    s0, s1 = fist_s(i - 0.25, T), fist_s(i + 0.25, T)
    v = 0.0 if (s0 is None or s1 is None) else (s1 - s0) * 2       # 1 フレームあたり
    if T['hit'] <= i < T['hit'] + HOLD:
        v = 0.0                                                     # ヒットストップ中はぶれなし
    # 速さの線（前へ進んでいるとき）
    q = fist_s(i - 1, T)
    if v > 12:
        f = speed_lines(f, p, None if q is None else HITC + q * DV)
    spr = rgba
    pad = 0
    if abs(v) > 6:                                  # サブフレームのぶれ（180°シャッター＝1 フレームの移動の半分）
        ln = abs(v) * 0.5
        pad = int(ln / 2) + 2
        spr = np.pad(rgba, ((pad, pad), (pad, pad), (0, 0)))
        spr = M.smear(spr, DV[0] * ln, DV[1] * ln)
    x0f, y0f = p[0] - ax - pad, p[1] - ay - pad
    ix, iy = int(np.floor(x0f)), int(np.floor(y0f))
    import cv2
    spr = cv2.warpAffine(spr, np.float32([[1, 0, x0f - ix], [0, 1, y0f - iy]]), (spr.shape[1] + 1, spr.shape[0] + 1),
                         flags=cv2.INTER_LINEAR)
    return M.put_rgba(f, spr, ix, iy)


def speed_lines(f, p, q):
    """拳のうしろ（袖口の後ろ）に3本の短い白い線。q：前のフレームの位置（None なら画面の外から来たとみなす）。"""
    d = DV
    sp = 60.0 if q is None else float(np.dot(np.asarray(p) - q, d))
    if sp < 12:
        return f
    a = min(1.0, (sp - 12) / 30.0)
    n = np.array([-d[1], d[0]])
    L = G.Lines(w=f.shape[1], h=f.shape[0])
    for off, k in ((-34, 0.75), (0, 1.0), (34, 0.75)):
        a0 = np.asarray(p) - d * (96 * FS) + n * off * FS
        a1 = a0 - d * min(240.0, sp * 2.2) * k
        L.poly(np.array([a1, a0]), 3.2, 255 * 0.55 * a)
    m = L.done()[..., None]
    return f * (1 - m) + core.FOAM * m


# ---------------------------------------------------------------- 悪役（立つ → 当たる → 飛ぶ）
def villain_draw(t, T):
    """小数のフレーム t の悪役の板 (rgba, x0, y0)（世界の座標）。サブフレームのぶれに使う。"""
    h = T['hit']
    ta = M.hitstop(t, [(h, HOLD)])
    sec = t / core.FPS
    fx, fy = S.FEET
    tx, ty = torso()
    if t < h:
        # 胸を張って弾む（2.2Hz、足もとを中心に縦に伸び縮み）＋ 37–42 で拳に気づいてのけぞる
        ph = math.sin(2 * math.pi * 2.2 * sec)
        sy = 1.025 + 0.025 * ph
        sx = 1 / math.sqrt(sy)
        lean = 7.0 * M.ease((t - (h - 9)) / 6.0, 'back_out')
        bob = -2.0 * max(0.0, ph)
        Rl = A._R(lean)
        M2 = Rl @ np.diag([sx, sy])
        c = np.array([fx, fy]) + M2 @ (np.array([tx, ty]) - (fx, fy))
        return A.villain_rgba(c[0], c[1] + bob, S.VSCALE, 0.0, 'stand', sec, M2=M2)
    if t < h + HOLD:
        # ヒットストップ：打つ向きにつぶれて止まる
        M2 = M.squash_matrix(DIR, 0.72, 1.18) @ A._R(7.0)
        c = np.array([fx, fy]) + M2 @ (np.array([tx, ty]) - (fx, fy)) + DV * 8
        return A.villain_rgba(c[0], c[1], S.VSCALE, 0.0, 'stand', h / core.FPS, M2=M2)
    x, y, sc, ang, tp = flight(ta, T)
    if sc * 34 < 5:
        return None
    v = flight_vel(ta, T)
    va = math.degrees(math.atan2(v[1], v[0]))
    k = (ta - h) / core.FPS                       # 打ち出しからの秒
    s_al = 1.45 + (1.0 - 1.45) * M.spring(k, 0.5, 5.0)
    s_ac = 0.80 + (1.0 - 0.80) * M.spring(k, 0.5, 5.0)
    M2 = M.squash_matrix(va, s_al, s_ac)
    vl = flight_vel(ta - 2, T)                    # マントは 2 フレーム遅れて、速さの反対へ
    cape = math.degrees(math.atan2(-vl[1], -vl[0])) + 9 * math.sin(sec * 11.0)
    return A.villain_rgba(x, y, sc, ang, 'fly', sec, M2=M2, cape_ang=cape)


def put_villain(f, i, T):
    h = T['hit']
    ta = M.hitstop(i, [(h, HOLD)])
    if i >= h + HOLD:
        x, y, sc, ang, tp = flight(ta, T)
        if sc * 34 < 5:
            return A.put_dot(f, x, y, max(1.6, sc * 8))
        sp = float(np.hypot(*flight_vel(ta, T)))
        n = int(np.clip(math.ceil(sp * 0.35 / 3.0), 1, 10)) if sp > 40 else 1
        r = M.subframe(lambda t: villain_draw(t, T), i, shutter=0.35, n=n)
        return M.put_rgba(f, *r, shadow=0.35, shadow_px=max(1.0, 1.1 * sc)) if r else f
    r = villain_draw(float(i), T)
    return M.put_rgba(f, *r, shadow=0.35, shadow_px=max(1.0, 1.1 * S.VSCALE))


# ---------------------------------------------------------------- カメラ
def cam_matrix(i, T, gi_end):
    """世界（下に EXTRA 行足した板）→ 画面 の 2×3。"""
    h = T['hit']
    z = 1.10 - 0.10 * M.ease(i / 40.0, 'cubic_out')     # 引いて見せる（拳が来るころまで動き続ける）
    if i < h + 3:
        oy = OFF0
    else:
        oy = OFF0 * (1 - M.spring((i - h - 3) / core.FPS, 0.85, 1.1))   # ゆっくり追って、キラーンの直前に着く
        oy *= 1 - M.ease((i - (T['glint'] - 8)) / 8.0, 'sine_inout')      # キラーンでぴったり 0（残り 0.3px 以下）
    cam = M.Cam2D(xy=(0.0, oy), zoom=z, pivot=torso())
    Mf = np.vstack([cam.M(i, extra=M.shake(i, h, amp=9, tau=3, rot_deg=0.35)), [0, 0, 1]])
    k = S.push_k(i, gi_end)
    P = np.array([[k, 0, S.DOT[0] * (1 - k)], [0, k, S.DOT[1] * (1 - k)], [0, 0, 1]])
    return (P @ Mf)[:2]


# ---------------------------------------------------------------- 文字
_TX = {}


def texts():
    if 't' not in _TX:
        _TX['t'] = (M.KText([dict(text='パンチ一発で', x=W / 2, y=470, size=48, weight='Medium', track=0.10)]),
                    M.KText([dict(text='空のかなたへ', x=W / 2, y=600, size=100, weight='Bold', hl='空のかなたへ')]))
    return _TX['t']


def put_texts(f, i, T):
    t1, t2 = texts()
    f = t1.reveal(f, i, T['pa'] - 2, 'char_rise', stagger=1.1, dur=10, dy_em=0.32, blur=5)
    f = t2.reveal(f, i, T['so'] - 2, 'char_rise', stagger=1.0, dur=11, dy_em=0.26, blur=7, accent=T['so'] + 5)
    return f


def render(beat, ctx):
    import cv2
    n = beat['n']
    T = timing(beat)
    gi_end = push_end()
    sparks, dust = particles(T)
    for i in range(n):
        f = S.frame(i, extra=EXTRA)
        h = T['hit']
        # 跡（飛んでいるあいだ）
        if i >= h + HOLD:
            ta = M.hitstop(i, [(h, HOLD)])
            tau = flight_tau(ta, T)
            tp = 1 - (1 - tau) ** 2.2
            f = trail(f, tp, tau, 1 - core.ease((i - T['glint'] - 2) / 14.0))
        f = dust.draw(f, i)
        # 悪役（キラーンのあとは点）
        if i < T['glint']:
            f = put_villain(f, i, T)
        else:
            tw = 0.85 + 0.15 * np.cos((i - T['glint']) * 0.35)
            f = A.put_dot(f, S.DOT[0], S.DOT[1], 2.3, alpha=tw)
        f = impact(f, i - h)
        f = sparks.draw(f, i)
        f = put_fist(f, i, T)
        # キラーン
        f = A.put_glint_fx(f, S.DOT[0], S.DOT[1], i - T['glint'])
        # カメラ（世界 → 画面）
        Mc = cam_matrix(i, T, gi_end)
        f = cv2.warpAffine(f, Mc.astype(np.float32), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        f = put_texts(f, i, T)
        yield np.clip(f, 0, 1).astype(np.float32)
