"""案C 03（2.4–5.0 秒・78 フレーム）「現実だと、こうなる。」 テロップ（現実だと）は 02 から引き継いで、衝撃で上へ抜ける。

01 と同じパンチを、同じ構図のまま質感のある CG で（_shots ショット A：草原・12m 先の白い悪役・左下から拳）。
  0–HIT ：スローモーション。拳がゆっくり入ってくる（速さの段階づけ：最後の 3 フレームで実時間へ）。マントがはためき、
          悪役は拳に気づいてのけぞりかける。拳の先の空気がゆがむ。
  HIT（「コ」の 2 フレーム前）：当たる。2 フレームのヒットストップ（悪役がつぶれる）。光らせない：形とコントラストだけ。
  HIT+2–：悪役は右上へ打ち上がる（強いぶれ＋白い空気の筋）。当たった所から衝撃波の殻（屈折＋うすい凝結の殻 ≤0.14）が広がり、
          その下の草が外へなぎ倒れる波、殻のふもとに砂煙の壁、地面がえぐれて土くれが弧を描いて飛ぶ、まんなかに暗い砂煙の柱。
          カメラは大きく揺れ、殻がカメラに届いたところで上へ振られる（whip）。
  B（WB0–WB1）：真上の雲の層。悪役の点が上へ抜けると、雲にまんまるの穴があいて広がる（縁は少しちぎれ、外側に押しのけた雲）。
  C（WC0–）：下へ振って、高い斜めの引き（_shots ショット C）：えぐれたクレーター、放射状になぎ倒れた草原、遠くへ広がる砂煙の壁、
          落ちてくる土くれ。04・05 はこの絵の続き。
地震・建物・津波を思わせるもの（地割れ・水）は描かない。白い閃光なし（平均の明るさの差は 1 フレーム 0.1 以内）。
"""
import math

import cv2
import numpy as np

import core
from core import W, H
import timeline_c as timeline
from cutB.beats import _render as R, _motion as M, _figures3d as F3, _art as A
from cutC.beats import _shots as SH, _fx as FX, c02

CAM = SH.CAM_A
DIR = -15.0
DV = np.array([math.cos(math.radians(DIR)), math.sin(math.radians(DIR))])
HOLD = 2
LAUNCH_ANG = -63.0                                   # 打ち上がる向き（画面の度）
_C = {}


def timing(beat):
    mo = beat['moras']
    assert mo[6][0] == 'コ', mo
    ko = timeline.mora_frame(beat, 6)
    hit = ko - 2
    return dict(hit=hit, launch=hit + HOLD, wu0=hit + 11, wb0=hit + 15, wb1=hit + 29, wc0=hit + 33, ko=ko)


def geom():
    """悪役の足もと・大きさ・胴・当たる所・拳の止まる所（画面）。"""
    if 'g' not in _C:
        P, z = CAM.project(np.array([[0.0, 0.0, 0.0], [0.0, 1.7, 0.0], [0.0, 1.0, 0.0]]))
        feet = P[0]
        vs = (P[0, 1] - P[1, 1]) / 32.0
        torso = np.array([feet[0], feet[1] - 16 * vs])
        contact = torso + np.array([-3.6, -3.4]) * vs
        fs = 0.092 * vs
        hitc = contact - 68 * fs * DV
        _C['g'] = dict(feet=feet, vs=vs, torso=torso, contact=contact, fs=fs, hitc=hitc, dist=float(z[0]))
    return _C['g']


# ---------------------------------------------------------------- 悪役
def villain_sprite(key, fn):
    if key not in _C:
        _C[key] = fn()
    return _C[key]


def put_villain(f, i, T):
    g = geom()
    vs, feet, torso = g['vs'], g['feet'], g['torso']
    h = T['hit']
    if i < h:
        # スローモーション：マントのはためきは 1/4 の速さ。絵は 4 種を作って混ぜる（重いので）
        ts = i / core.FPS * 0.25
        lean = 5.0 * M.ease((i - (h - 10)) / 10.0, 'sine_inout')
        q = ts / 0.05
        k0 = int(math.floor(q))
        fr = q - k0
        outs = []
        for kk in (k0, k0 + 1):
            key = ('vs', kk)
            spr = villain_sprite(key, lambda kk=kk: _vil(torso, feet, vs, 'stand', kk * 0.05, None, F3.ENV_DAY))
            outs.append(spr)
        rgba, x0, y0 = outs[0]
        if fr > 0.02 and outs[1][0].shape == rgba.shape:
            rgba = rgba * (1 - fr) + outs[1][0] * fr
        if lean > 0.05:                               # のけぞり：足もとを中心に板を回す（描き直さない）
            Mr = cv2.getRotationMatrix2D((feet[0] - x0, feet[1] - y0), -lean, 1.0)
            prem = cv2.warpAffine(M._to_prem(rgba), Mr, (rgba.shape[1], rgba.shape[0]), flags=cv2.INTER_LINEAR)
            rgba = M._from_prem(prem)
        f = shadow(f, 1.0)
        return M.put_rgba(f, rgba, x0, y0)
    if i < h + HOLD:
        M2 = M.squash_matrix(DIR, 0.84, 1.10) @ A._R(8.0)
        rgba, x0, y0 = villain_sprite(('vh',), lambda: _vil(torso, feet, vs, 'stand', h / core.FPS * 0.25, M2,
                                                            F3.ENV_DAY, shift=DV * 6))
        f = shadow(f, 1.0)
        return M.put_rgba(f, rgba, x0, y0)
    k = i - T['launch']
    if k > 3:
        return f
    th = math.radians(LAUNCH_ANG)
    sp = 330.0
    p = torso + np.array([math.cos(th), math.sin(th)]) * sp * (k + 0.6)
    rgba, ax, ay = villain_sprite(('vt',), lambda: _vil_tumble(vs))
    ln = sp * 0.55
    pad = int(ln / 2) + 4
    spr = M.smear(np.pad(rgba, ((pad, pad), (pad, pad), (0, 0))), math.cos(th) * ln, math.sin(th) * ln)
    return M.put_rgba(f, spr, int(p[0] - ax - pad), int(p[1] - ay - pad))


def _vil(torso, feet, vs, pose, t, M2, env, shift=(0.0, 0.0)):
    c = np.array(torso, float)
    if M2 is not None:
        c = np.array(feet) + np.asarray(M2) @ (np.array(torso) - feet)
    c = c + np.asarray(shift)
    rgba, x0, y0 = F3.villain_rgba(c[0], c[1], vs, 0.0, pose, t, M2=M2, light=F3.KEY, env=env)
    return F3.cap_white(rgba), x0, y0


def _vil_tumble(vs):
    rgba, x0, y0 = F3.villain_rgba(300.0, 300.0, vs * 0.95, 35.0, 'tumble', 0.3, M2=M.squash_matrix(LAUNCH_ANG, 1.15, 0.9),
                                   cape_ang=LAUNCH_ANG + 180, light=F3.KEY, env=F3.ENV_FLY)
    return F3.cap_white(rgba), 300.0 - x0, 300.0 - y0


def shadow(f, a):
    g = geom()
    return R.contact_shadow(f, g['feet'][0] + 10, g['feet'][1] + 2, 26 * g['vs'] / 10, 5.0, opacity=0.45 * a,
                            rgb=core.c('183424'), blur=6.0)


# ---------------------------------------------------------------- 拳
def fist_s(i, T):
    h = T['hit']
    if i < h - 3:
        return -9.0 * (h - i) - 85.0
    if i < h:
        return {3: -112.0, 2: -62.0, 1: -22.0}[h - i]
    if i < h + HOLD:
        return 8.0
    k = i - h - HOLD
    return 8 + 46 * M.ease(min(1.0, k / 5.0), 'quad_out')


def put_fist(f, i, T):
    g = geom()
    if 'fist' not in _C:
        _C['fist'] = F3.fist_v3(DIR, g['fs'])
    rgba, ax, ay = _C['fist']
    s = fist_s(i, T)
    p = g['hitc'] + s * DV
    v = fist_s(i + 0.5, T) - fist_s(i - 0.5, T) if T['hit'] - 3 > i or i >= T['hit'] + HOLD else 0.0
    a = 1.0
    if i >= T['hit'] + HOLD:                          # 当たったあと：揺れと砂煙の中で少しずつ見えなくなる
        a = 1 - M.ease((i - T['hit'] - HOLD) / 9.0, 'sine_inout')
    spr, pad = rgba, 0
    if abs(v) > 3:
        ln = abs(v) * 0.5
        pad = int(ln / 2) + 2
        spr = M.smear(np.pad(rgba, ((pad, pad), (pad, pad), (0, 0))), DV[0] * ln, DV[1] * ln)
    return FX.put(f, spr, p[0] - ax - pad, p[1] - ay - pad, a)


def pre_air(f, i, T):
    """拳の先の空気のゆがみ（スローモーションの終わりに強まる、小さな屈折）。"""
    h = T['hit']
    if i < h - 12 or i >= h:
        return f
    g = geom()
    p = g['hitc'] + fist_s(i, T) * DV + DV * 70 * g['fs']
    k = (i - (h - 12)) / 12.0
    return FX.refract(f, p[0], p[1], 40 + 30 * k, amp=3.0 + 5.0 * k, width=12.0)


# ---------------------------------------------------------------- 衝撃
def shock_R(k):
    """当たってから k フレームの衝撃波の半径（m）。11 フレームでカメラ（12m）に届く。"""
    if k <= 0:
        return 0.0
    return 12.4 * (k / 11.0) ** 0.78


def project_r(Rm):
    """原点 1m の高さを中心にした半径 Rm の球の、画面の半径（px）。"""
    d = geom()['dist']
    a = math.asin(min(Rm / d, 0.985))
    return CAM.fpx * math.tan(a)


_CL = {}


def clods(T):
    if _CL.get('h') != T['hit']:
        _CL['h'] = T['hit']
        _CL['c'] = FX.Clods(n=360, f0=T['launch'], origin=(0.0, 0.05, 0.0), speed=(0.12, 0.75), el_range=(38, 72),
                            seed=7, size=(0.008, 0.05))
        rng = np.random.default_rng(3)
        _CL['ang'] = np.sort(rng.uniform(0, 2 * np.pi, 110))
        _CL['pk'] = rng.integers(0, 6, 110)
        _CL['pj'] = rng.uniform(0.7, 1.3, 110)
        _CL['pr'] = rng.uniform(0.85, 1.05, 110)
    return _CL


def dust_items(i, T):
    """(深さ, kind, x, y, size, alpha, dark) の砂煙のリスト（壁＋まんなかの柱）。"""
    k = i - T['launch']
    if k < 0:
        return []
    cl = clods(T)
    out = []
    Rd = shock_R(k) * 0.82
    for th, pk, pj, pr in zip(cl['ang'], cl['pk'], cl['pj'], cl['pr']):
        rr = Rd * pr
        sz = (0.35 + 0.12 * rr) * pj
        hgt = sz * 0.32
        P = np.array([[rr * math.cos(th), hgt, rr * math.sin(th)]])
        S, z = CAM.project(P)
        if z[0] < 3.0:
            continue
        px = min(sz * CAM.fpx / z[0], 520.0)
        a = 0.62 * min(1.0, k / 3.0) * min(1.0, (z[0] - 3.0) / 2.0)
        out.append((float(z[0]), int(pk), float(S[0, 0]), float(S[0, 1]), px * 1.6, a, 1.0))
    # 土のカーテン（えぐれた所から逆さの円すいに広がる土と砂煙。上ほど明るくうすい）
    hc = min(5.0, 0.6 * k)
    for lv in range(7):
        hq = hc * (lv + 0.5) / 7.0
        rc = 0.25 + hq * math.tan(math.radians(30))
        for j in range(14):
            ph = 2 * math.pi * (j + 0.5 * (lv % 2)) / 14.0
            S, z = CAM.project(np.array([[rc * math.cos(ph), hq, rc * math.sin(ph)]]))
            sz = 0.55 + 0.22 * hq + 0.03 * k
            a = 0.30 * min(1.0, k / 2.0) * (1 - 0.55 * lv / 6.0)
            out.append((float(z[0]), (j + lv) % 6, float(S[0, 0]), float(S[0, 1]), sz * CAM.fpx / z[0] * 1.6, a,
                        0.80 + 0.04 * lv))
    return out


def put_dust(f, items, near):
    d0 = geom()['dist']
    for (z, kind, x, y, s, a, dk) in sorted(items, key=lambda q: -q[0]):
        if (z < d0) != near:
            continue
        f = FX.put_puff(f, kind, x, y, s, a, dk)
    return f


# ---------------------------------------------------------------- テロップ
def put_telop(f, i, T):
    e0 = T['wu0'] - 2
    if i < e0:
        return c02.put_telop(f)
    q = M.ease((i - e0 + 1) / 6.0, 'cubic_in')
    if q >= 1:
        return f
    return c02.put_telop(f, 1.0, 1 - q, dy=-40 * q, blur=5 * q)


# ---------------------------------------------------------------- ショット A
def cam_A(i, T):
    h = T['hit']
    z = 1.0 + 0.045 * M.ease(i / float(h), 'sine_inout') if i < h else 1.045
    sh = (0.0, 0.0, 0.0)
    if i >= h:
        k = i - h
        e = 30.0 * math.exp(-k / 7.0) + 10.0
        sh = (e * M.vnoise(k * 0.55 + 0.3, 41), e * M.vnoise(k * 0.55 + 4.1, 42), 0.9 * M.vnoise(k * 0.5 + 9.0, 43))
    # 殻がカメラに届く（wu0 の少し前）：押される
    kb = i - (T['wu0'] - 2)
    if kb >= 0:
        z *= 1 - 0.05 * M.ease(kb / 4.0, 'expo_out')
    c = math.cos(math.radians(sh[2])) * z
    s = math.sin(math.radians(sh[2])) * z
    cx, cy = geom()['contact']
    return np.float32([[c, -s, cx - c * cx + s * cy + sh[0]], [s, c, cy - s * cx - c * cy + sh[1]]])


def shot_A(i, T):
    back, front, dep = SH.plate_A()
    if 'backs' not in _C:
        _C['backs'] = np.clip(R.srgb(back), 0, 1)
        fr = front
        _C['front0'] = np.dstack([R.srgb(fr[..., :3]), fr[..., 3:]])
    f = _C['backs'].copy()
    h = T['hit']
    k = i - T['launch']
    Rs = shock_R(k)
    if k >= 0:                                         # なぎ倒れた草（殻の下）
        flat, rmap = SH.flat_A()
        m = R.smoothstep(Rs + 0.6, Rs - 0.6, rmap)[..., None]
        f = f * (1 - m) + flat * m
    items = dust_items(i, T)
    f = put_dust(f, items, near=False)
    f = put_villain(f, i, T)
    f = put_fist(f, i, T)
    # 手前の草：当たる前は板、当たったあとは毎フレーム（なぎ倒れる）
    fr = _C['front0']
    if k >= 0 and Rs >= 0.5:
        # なぎ倒れた手前の草（1 回だけ描いた板）を、地面の距離のマスクで混ぜる（毎フレーム描き直すと重い）
        if 'front1' not in _C:
            f1 = SH.front_A(0.0, flat_fn=lambda X, Z: np.ones_like(X))
            _C['front1'] = np.dstack([R.srgb(f1[..., :3]), f1[..., 3:]])
        rmap = SH.flat_A()[1]
        m = R.smoothstep(Rs + 1.0, Rs - 1.0, rmap)[..., None]
        fr = _C['front0'] * (1 - m) + _C['front1'] * m
    f = f * (1 - fr[..., 3:]) + fr[..., :3] * fr[..., 3:]
    f = clods(T)['c'].draw(f, CAM, i)
    f = put_dust(f, items, near=True)
    # 打ち上がった悪役のうしろの空気の筋
    g = geom()
    if k >= 0:
        th = math.radians(LAUNCH_ANG)
        far = g['torso'] + np.array([math.cos(th), math.sin(th)]) * 330 * (min(k, 3) + 0.6)
        a = 0.55 * (1 - M.ease(k / 12.0, 'sine_in'))
        f = FX.vapor(f, g['torso'], far, 26 + 6 * k, 10 + 3 * k, a, blur=3 + 1.5 * k)
    f = pre_air(f, i, T)
    if k >= -HOLD:                                     # 衝撃波の殻
        rpx = project_r(max(Rs, 0.25))
        fade = 1 - M.ease(k / 12.0, 'sine_in')
        f = FX.refract(f, g['contact'][0], g['contact'][1], rpx, amp=14.0 * fade + 2, width=18 + rpx * 0.05,
                       shell_a=0.12 * fade, ground_y=None)
    f = cv2.warpAffine(f, cam_A(i, T), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return f


# ---------------------------------------------------------------- ショット B（真上の雲の層に穴）
def shot_B(i, T):
    k = i - T['wb0']
    n = T['wb1'] - T['wb0']
    k_in = 3                                         # 悪役の点が雲に届くフレーム
    hr = 0.0 if k < k_in else 400.0 * (1 - math.exp(-(k - k_in + 0.6) / 3.2))
    rp = min(1.0, max(0.0, (k - k_in) / 3.0))
    f = SH.frame_B(hr, rp, t=k / core.FPS).copy()
    # 悪役の点（下から雲の穴のまんなかへ。うしろに白い筋）
    c, _ = SH.CAM_B.project(np.array([[0.0, F_ALT(), 0.0]]))
    c = c[0]
    q = min(1.0, (k + 1) / float(k_in + 1))
    st = np.array([c[0] - 120, c[1] + 760])
    p = st + (c - st) * M.ease(q, 'quad_out')
    if k <= k_in + 1:
        f = FX.vapor(f, st + (p - st) * 0.0, p, 3, 9, 0.45 * (1 - max(0, k - k_in) / 2.0), blur=2.0)
        f = A.put_dot(f, p[0], p[1], 3.0 * (1 - 0.4 * q), alpha=0.9) if False else _speck(f, p, 4.0 - 1.5 * q)
    sh = 14.0 * math.exp(-k / 5.0)
    Mm = np.float32([[1, 0, sh * M.vnoise(k * 0.6, 51)], [0, 1, sh * M.vnoise(k * 0.6 + 3, 52)]])
    f = cv2.warpAffine(f, Mm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if k < 4:
        f = M.whip(f, (k + 1) / 4.0, axis=0, sign=1, px=420, blur=160, direction='in')
    j = i - (T['wb1'] - 4)
    if j >= 0:
        f = M.whip(f, (j + 1) / 4.0, axis=0, sign=-1, px=420, blur=160, direction='out')
    return f


def F_ALT():
    from cutC.beats import _field
    return _field.ALT


def _speck(f, p, r):
    """下から見上げた悪役：空に小さな暗い点（白い体の影の側）。"""
    return R.contact_shadow(f, p[0], p[1], r, r, opacity=0.75, rgb=core.c('4A5A6A'), blur=0.8)


# ---------------------------------------------------------------- ショット C（引き：クレーター）
def shot_C(i, T, beat):
    from cutC.beats import _cfx as CF
    f, cam, Mm, g = CF.frame_C(beat, i)
    k = i - T['wc0']
    if k < 4:
        f = M.whip(f, (k + 1) / 4.0, axis=0, sign=-1, px=420, blur=160, direction='in')
    sh = 6.0 * math.exp(-k / 6.0)
    if sh > 0.3:
        Mm2 = np.float32([[1, 0, sh * M.vnoise(k * 0.6, 61)], [0, 1, sh * M.vnoise(k * 0.6 + 3, 62)]])
        f = cv2.warpAffine(f, Mm2, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return f


def frame(i, T, beat):
    if i < T['wb0']:
        f = shot_A(i, T)
        if i >= T['wu0']:
            f = M.whip(f, (i - T['wu0'] + 1) / float(T['wb0'] - T['wu0']), axis=0, sign=1, px=420, blur=160,
                       direction='out')
    elif i < T['wc0']:
        f = shot_B(i, T)
    else:
        f = shot_C(i, T, beat)
    return f


def render(beat, ctx):
    n = beat['n']
    T = timing(beat)
    for i in range(n):
        f = frame(i, T, beat)
        f = R.finish_ad(f, 2000 + i, grain=0.006, vignette=0.10, ca_px=0.25)
        f = put_telop(f, i, T)
        yield np.clip(f, 0, 1).astype(np.float32)
