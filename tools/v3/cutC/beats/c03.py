"""案C 03（約 2.2 秒）「（でも、現実だと……）こうなる。」 声は 02 から始まっていて、このビートの頭は「……」の間（「コ」の 0.5 秒前）。
テロップ（現実だと＋小「同じパンチを本当にやったら」）は 02 から引き継いで、衝撃で上へ抜ける。

01 と同じパンチを、同じ構図のまま質感のある CG で（_shots ショット A：草原・12m 先の白い悪役・左下から拳）。
  0–HIT ：スローモーション。拳がゆっくり入ってくる（速さの段階づけ：最後の 3 フレームで実時間へ）。マントがはためき、
          悪役は拳に気づいてのけぞりかける。拳の先の空気がゆがむ。カメラはじわっと寄る。
  HIT（「こうなる」の「コ」の 2 フレーム前）：当たる。2 フレームのヒットストップ＝ため：悪役がつぶれ、足もとの地面がへこんで
          （暗いくぼみ＋外の盛り上がり）草が丸く押し倒され、カメラはぐっと寄る。光らせない：形・コントラスト・揺れだけ。
  LAUNCH–：爆発。悪役は右上へ打ち上がり、地面が裂けて土・岩・芝のかたまりが逆さの円すいのカーテンになって弧を描いて飛ぶ
          （_blast.Chunks：放物線・回転・ぶれ・左上の光。芝はひっくり返る）。カメラは大きく揺れて一瞬引き、
          そのあと速さの段階づけ：2 フレームは実時間 → 約 0.3 倍のスローで土のカーテンが空中に止まって見える → 一気に加速して
          土の壁と地面を転がる砂煙の輪がカメラへ押し寄せる。衝撃波の殻（屈折）の下で草が外へなぎ倒れる。
          殻がカメラに届いたところで（wu0−2）カメラがはじかれて上へ振られる（whip）。
  B（WB0–WB1）：真上の雲の層。悪役の点が上へ抜けると、雲にまんまるの穴があいて広がる（縁は少しちぎれ、外側に押しのけた雲）。
  C（WC0–）：下へ振って、高い斜めの引き（_shots ショット C／_cfx）：砂煙のドームと降ってくるかたまり、地平へ走る砂煙の輪。
          寄りから一気に引いて大きさを見せる。04・05 はこの絵の続き（ドームが晴れて深いクレーターが現れる）。
時刻はすべて timing() から（audio_c が同じ関数で効果音を置く）。
地震・建物・津波を思わせるもの（地割れが町へ走る・水）は描かない。白い閃光なし（平均の明るさの差は 1 フレーム 0.1 以内）。
"""
import math

import cv2
import numpy as np

import core
from core import W, H
import timeline_c as timeline
from cutB.beats import _render as R, _motion as M, _figures3d as F3, _art as A
from cutC.beats import _shots as SH, _fx as FX, _blast as BL, _cfx as CF, c02

CAM = SH.CAM_A
DIR = -15.0
DV = np.array([math.cos(math.radians(DIR)), math.sin(math.radians(DIR))])
HOLD = 2
WU = 14                                              # 当たってから上へ振るまで（爆発を見せる長さ）
LAUNCH_ANG = -63.0                                   # 打ち上がる向き（画面の度）
_C = {}


def timing(beat):
    mo = beat['moras']
    assert mo[timeline.KO_INDEX][0] == 'コ' and mo[0][0] == 'デ', mo
    ko = timeline.mora_frame(beat, timeline.KO_INDEX)
    assert ko >= 13, ko                             # スローの拳（h−12 からの空気のゆがみ）が入る長さ
    hit = ko - 2
    return dict(hit=hit, launch=hit + HOLD, wu0=hit + WU, wb0=hit + WU + 4, wb1=hit + WU + 16, wc0=hit + WU + 20, ko=ko)


# ---------------------------------------------------------------- 速さの段階づけ（打ち上がってからの「中の時間」）
def _rate(k):
    """実時間のフレーム k（launch＝0）での、中の時間の速さ。2 フレームは実時間 → 0.3 倍 → 1.8 倍。"""
    if k < 2.0:
        return 1.0
    if k < 3.0:
        return 1.0 - 0.7 * M.ease(k - 2.0, 'sine_inout')
    if k < 6.0:
        return 0.3
    if k < 9.0:
        return 0.3 + 1.5 * M.ease((k - 6.0) / 3.0, 'sine_inout')
    return 1.8


def ks(k):
    """実時間のフレーム k（launch から）→ 中の時間（フレーム）。"""
    if k <= 0:
        return float(k)
    key = round(k, 4)
    if key not in _KS:
        n = max(2, int(k * 40))
        xs = np.linspace(0, k, n + 1)
        r = np.array([_rate(x) for x in xs])
        _KS[key] = float(np.trapezoid(r, xs))
    return _KS[key]


_KS = {}


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
    k = ks(i - T['launch'])
    if k > 3.2:
        return f
    th = math.radians(LAUNCH_ANG)
    sp = 330.0
    p = torso + np.array([math.cos(th), math.sin(th)]) * sp * (k + 0.6)
    rgba, ax, ay = villain_sprite(('vt',), lambda: _vil_tumble(vs))
    ln = sp * 0.55 * _rate(i - T['launch'])
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
    """打ち上がってから k フレーム（実時間）の衝撃波の半径（m）。中の時間で広がり、k＝wu0−2−launch でカメラ（12m）に届く。"""
    if k <= 0:
        return 0.0
    return 12.4 * (ks(k) / ks(K_CAM)) ** 0.78


K_CAM = WU - 2 - HOLD                                # 殻がカメラに届く（wu0 − 2 − launch）


def project_r(Rm):
    """原点 1m の高さを中心にした半径 Rm の球の、画面の半径（px）。"""
    d = geom()['dist']
    a = math.asin(min(Rm / d, 0.985))
    return CAM.fpx * math.tan(a)


_CL = {}


def chunks_A():
    """ショット A の土・岩・芝のかたまり（中の時間で飛ぶ）。地面の裂け目（えぐれの前線）から順に飛び出す。"""
    if 'ch' not in _CL:
        rng = np.random.default_rng(303)
        n = 1900
        T0 = rng.uniform(0, 1, n) ** 1.5 * 3.2
        r0 = 0.25 + 1.9 * T0 ** 0.8 + rng.uniform(0, 0.3, n)
        az = rng.uniform(0, 2 * np.pi, n)
        el = np.radians(rng.uniform(40, 70, n))
        sp = (0.9 + 1.7 * rng.uniform(0, 1, n) ** 0.7) * (1 - 0.12 * T0)
        P0 = np.stack([r0 * np.cos(az), np.full(n, 0.05), r0 * np.sin(az)], 1)
        V = np.stack([np.cos(az) * np.cos(el), np.sin(el), np.sin(az) * np.cos(el)], 1) * sp[:, None]
        kind = rng.choice([0, 1, 2, 3], n, p=[0.48, 0.18, 0.22, 0.12])
        size = 0.008 + 0.15 * rng.uniform(0, 1, n) ** 3.2
        size = np.where(kind == 2, 0.05 + 0.15 * rng.uniform(0, 1, n) ** 2.0, size)
        _CL['ch'] = BL.Chunks(P0, V, T0, size, kind, seed=31, g=0.16, spin=(0.15, 0.7))
        # 砂煙の粒（重い空気抵抗：すぐ止まって漂う）
        m = 260
        T0d = rng.uniform(0, 1, m) ** 1.3 * 3.0
        r0d = 0.3 + 1.7 * T0d ** 0.8
        azd = rng.uniform(0, 2 * np.pi, m)
        eld = np.radians(rng.uniform(25, 78, m))
        spd = rng.uniform(0.5, 1.6, m)
        _CL['dust'] = dict(T0=T0d, P0=np.stack([r0d * np.cos(azd), np.full(m, 0.2), r0d * np.sin(azd)], 1),
                           V=np.stack([np.cos(azd) * np.cos(eld), np.sin(eld), np.sin(azd) * np.cos(eld)], 1) * spd[:, None],
                           k=rng.integers(0, 6, m), s=rng.uniform(0.75, 1.3, m))
        q = 120
        _CL['ring'] = dict(th=np.sort(rng.uniform(0, 2 * np.pi, q)), k=rng.integers(0, 6, q), s=rng.uniform(0.8, 1.25, q),
                           r=rng.uniform(0.82, 1.02, q), ph=rng.uniform(0, 6.3, q))
    return _CL


def _coc(z):
    return float(min(9.0, 26.0 * abs(1.0 / max(z, 0.3) - 1.0 / SH.FOCUS_A)))


def _dust_item(items, P, size_m, a, kind, tint, dark=1.0):
    S, z = CAM.project(np.asarray(P, float).reshape(1, 3))
    z = float(z[0])
    if z < 1.2 or a <= 0.004:
        return
    a = a * min(1.0, (z - 1.2) / 2.0)
    px = float(size_m * CAM.fpx / z * 1.6)
    x, y = float(S[0, 0]), float(S[0, 1])
    if not (-px < x < W + px and -px < y < H + px):
        return
    items.append((z, 'd', lambda L, k=kind, x=x, y=y, s=px, a=a, t=tint, d=dark: L.puff(k, x, y, s, a, t, d)))


def blast_items(i, T):
    """ショット A の爆発：かたまり・砂煙の粒・地面を転がる輪・まんなかの暗い砂煙（中の時間 s）。"""
    k = i - T['launch']
    cl = chunks_A()
    if k < 0:                                          # ため：足もとから押し出された空気が低い砂ぼこりの輪に
        items = []
        if i >= T['hit']:
            q = (i - T['hit'] + 1) / float(HOLD)
            for j in range(26):
                th = 2 * math.pi * j / 26.0
                rr = 0.9 + 1.3 * q
                _dust_item(items, (rr * math.cos(th), 0.12, rr * math.sin(th)), 0.35 + 0.25 * q, 0.55 * q, j, 'dust', 0.85)
        return items
    s = ks(k)
    items = cl['ch'].items(s, CAM, shutter=0.55 * _rate(k), near=1.0, blur_fn=_coc, max_px=300.0)
    # 砂煙の粒：空気抵抗で止まって、ふくらみながら漂う
    d = cl['dust']
    tau = np.clip(s - d['T0'], 0, None)
    drag = 0.32
    P = d['P0'] + d['V'] * ((1 - np.exp(-drag * tau)) / drag)[:, None]
    P[:, 1] = np.maximum(P[:, 1] - 0.02 * tau * tau, 0.3)
    for j in np.nonzero(tau > 0)[0]:
        sz = (0.35 + 0.30 * tau[j]) * d['s'][j]
        a = 0.78 * min(1.0, tau[j] / 1.4)
        dark = 0.64 + 0.34 * min(1.0, P[j, 1] / 5.0)
        _dust_item(items, P[j], sz, a, int(d['k'][j]), 'soil', dark)
    # 地面を転がる砂煙の輪（衝撃波のすぐ後ろ）
    Rs = shock_R(k)
    rg = cl['ring']
    for th, kk, sc, rr, ph in zip(rg['th'], rg['k'], rg['s'], rg['r'], rg['ph']):
        R0 = 0.88 * Rs * rr
        sz = (0.45 + 0.20 * R0) * sc
        hh = sz * 0.30 * (1 + 0.3 * math.sin(ph + s * 0.8))
        _dust_item(items, (R0 * math.cos(th), hh, R0 * math.sin(th)), sz, 0.85 * min(1.0, s / 1.5), int(kk), 'soil', 0.84)
    # まんなかの暗い砂煙（裂け目から立ちのぼる）
    for j in range(7):
        hq = (j + 0.5) / 7.0 * min(7.0, 1.4 * s)
        sz = 0.9 + 0.35 * s + 0.25 * hq
        _dust_item(items, (0.4 * math.sin(j * 2.1), hq, 0.4 * math.cos(j * 1.7)), sz, 0.80, j, 'soil', 0.62)
    return items


def _ell(f, cx, cy, rx, ry, opacity, rgb, blur):
    """やわらかい楕円で地面に色を置く（R.contact_shadow と同じ見た目。横長でも縦は ry の分だけの板で速い）。"""
    px, py = int(rx + 3 * blur + 2), int(ry + 3 * blur + 2)
    x0, y0 = int(cx) - px, int(cy) - py
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + 2 * px, W), min(y0 + 2 * py, H)
    if xb <= xa or yb <= ya:
        return f
    m = np.zeros((2 * py, 2 * px), np.float32)
    cv2.ellipse(m, (int((cx - x0) * 4), int((cy - y0) * 4)), (int(rx * 4), int(ry * 4)), 0, 0, 360, 1.0, -1,
                cv2.LINE_AA, shift=2)
    m = cv2.GaussianBlur(m, (0, 0), blur) * opacity
    mm = m[ya - y0:yb - y0, xa - x0:xb - x0, None]
    f[ya:yb, xa:xb] = f[ya:yb, xa:xb] * (1 - mm) + rgb * mm
    return f


def ground_fx(f, i, T):
    """足もとの地面：ため（ヒットストップ）でへこむ → 裂けて黒い穴が広がる。"""
    g = geom()
    k = i - T['launch']
    h = T['hit']
    if i < h:
        return f
    fx, fy = g['feet'][0] + 4, g['feet'][1] + 4
    ppm = CAM.fpx / g['dist']
    if k < 0:                                           # へこみ：暗い楕円＋外の明るい盛り上がり
        q = (i - h + 1) / float(HOLD)
        rx = (1.4 + 0.7 * q) * ppm
        f = _ell(f, fx, fy - 2, rx * 1.4, rx * 0.11, opacity=0.30 * q, rgb=core.c('B9D2A8'), blur=7.0)
        f = _ell(f, fx, fy + 2, rx, rx * 0.08, opacity=0.70 * q, rgb=core.c('18221A'), blur=4.0)
        return f
    s = ks(k)
    Rex = min(10.0, 1.6 + 2.4 * s ** 0.75)
    rx = Rex * ppm
    f = _ell(f, fx, fy, rx * 1.08, rx * 0.085, opacity=0.65, rgb=core.c('5A4A3A'), blur=5.0)
    f = _ell(f, fx, fy, rx, rx * 0.075, opacity=0.85, rgb=core.c('1C1712'), blur=3.0)
    return f


def _rows(Rr):
    """地面の距離 < Rr の画素がある行の範囲 (y0, y1)。"""
    if 'rowmin' not in _C:
        _C['rowmin'] = SH.flat_A()[1].min(1)
    idx = np.nonzero(_C['rowmin'] < Rr)[0]
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0)


def put_dimple_grass(f, i, T):
    """ため：足もとの草が丸く押し倒される（倒れた草の板を小さな半径で）。"""
    h = T['hit']
    if i < h or i >= T['launch']:
        return f
    flat, rmap = SH.flat_A()
    Rr = 1.2 + 1.0 * (i - h + 1) / float(HOLD)
    y0, y1 = _rows(Rr + 0.4)
    m = R.smoothstep(Rr + 0.4, Rr - 0.4, rmap[y0:y1])[..., None]
    f[y0:y1] = f[y0:y1] * (1 - m) + flat[y0:y1] * m
    return f


# ---------------------------------------------------------------- テロップ
def put_telop(f, i, T):
    e0 = T['wu0'] - 2
    if i < e0:
        return c02.put_telop(f)
    q = M.ease((i - e0 + 1) / 6.0, 'cubic_in')
    if q >= 1:
        return f
    return c02.put_telop(f, 1.0, 1 - q, dy=-40 * q, blur=5 * q, sub_dy=-20 * q)


# ---------------------------------------------------------------- ショット A
def cam_A(i, T):
    """寄り（スロー）→ ため（ぐっと寄る）→ 爆発（はじかれて引く・大きく揺れる。揺れも中の時間で）→ 殻が届いてはじかれる。"""
    h = T['hit']
    z = 1.0 + 0.045 * M.ease(i / float(h), 'sine_inout') if i < h else 1.045
    if h <= i < T['launch']:
        z *= 1 + 0.035 * M.ease((i - h + 1) / float(HOLD), 'cubic_out')     # ため：ぐっと寄る（1.08）
    sh = (0.0, 0.0, 0.0)
    k = i - T['launch']
    if k >= 0:                                         # 揺れのはみ出しが見えないよう、つねに 1.07 倍以上
        s = ks(k)
        z = 1.08 * (1 + 0.035 * math.exp(-k / 0.8)) - 0.012 * M.ease(min(1.0, s / 3.0), 'expo_out')
        e = 30.0 * math.exp(-s / 5.0) + 8.0
        sh = (e * M.vnoise(s * 0.9 + 0.3, 41), e * M.vnoise(s * 0.9 + 4.1, 42), 1.6 * M.vnoise(s * 0.8 + 9.0, 43))
    elif i >= h:
        sh = (6.0 * M.vnoise(i * 3.1, 44), 6.0 * M.vnoise(i * 3.1 + 2, 45), 0.0)
    kb = i - (T['wu0'] - 2)                             # 殻がカメラに届く：はじかれる
    if kb >= 0:
        z *= 1 - 0.06 * M.ease(kb / 3.0, 'expo_out')
    c = math.cos(math.radians(sh[2])) * z
    s_ = math.sin(math.radians(sh[2])) * z
    cx, cy = geom()['contact']
    return np.float32([[c, -s_, cx - c * cx + s_ * cy + sh[0]], [s_, c, cy - s_ * cx - c * cy + sh[1]]])


def shot_A(i, T):
    back, front, dep = SH.plate_A()
    if 'backs' not in _C:
        _C['backs'] = np.clip(R.srgb(back), 0, 1)
        fr = front
        _C['front0'] = np.dstack([R.srgb(fr[..., :3]), fr[..., 3:]])
    f = _C['backs'].copy()
    k = i - T['launch']
    Rs = shock_R(k)
    if k >= 0:                                         # なぎ倒れた草（殻の下）。半径の中にかかる行だけ
        flat, rmap = SH.flat_A()
        y0, y1 = _rows(Rs + 0.6)
        if y1 > y0:
            m = R.smoothstep(Rs + 0.6, Rs - 0.6, rmap[y0:y1])[..., None]
            f[y0:y1] = f[y0:y1] * (1 - m) + flat[y0:y1] * m
    f = put_dimple_grass(f, i, T)
    f = ground_fx(f, i, T)
    items = blast_items(i, T)
    d0 = geom()['dist']
    far = [it for it in items if it[0] >= d0]
    near = [it for it in items if it[0] < d0]
    f = BL.draw_sorted(f, far)
    f = put_villain(f, i, T)
    f = put_fist(f, i, T)
    # 手前の草：当たる前は板、当たったあとはなぎ倒れた板とまぜる
    fr = _C['front0']
    if k >= 0 and Rs >= 0.5:
        if 'front1' not in _C:
            f1 = SH.front_A(0.0, flat_fn=lambda X, Z: np.ones_like(X))
            _C['front1'] = np.dstack([R.srgb(f1[..., :3]), f1[..., 3:]])
        rmap = SH.flat_A()[1]
        y0, y1 = _rows(Rs + 1.0)
        fr = _C['front0'].copy()
        if y1 > y0:
            m = R.smoothstep(Rs + 1.0, Rs - 1.0, rmap[y0:y1])[..., None]
            fr[y0:y1] = _C['front0'][y0:y1] * (1 - m) + _C['front1'][y0:y1] * m
    ya = _C.setdefault('fr_y0', int(np.argmax(_C['front0'][..., 3].max(1) > 0.003)))
    f[ya:] = f[ya:] * (1 - fr[ya:, :, 3:]) + fr[ya:, :, :3] * fr[ya:, :, 3:]
    f = BL.draw_sorted(f, near)
    # 打ち上がった悪役のうしろの空気の筋
    g = geom()
    if k >= 0:
        s = ks(k)
        th = math.radians(LAUNCH_ANG)
        far_p = g['torso'] + np.array([math.cos(th), math.sin(th)]) * 330 * (min(s, 3.2) + 0.6)
        a = 0.55 * (1 - M.ease(k / 12.0, 'sine_in'))
        f = FX.vapor(f, g['torso'], far_p, 26 + 6 * s, 10 + 3 * s, a, blur=3 + 1.5 * s)
    f = pre_air(f, i, T)
    if k >= -HOLD:                                     # 衝撃波の殻
        rpx = project_r(max(Rs, 0.25))
        fade = 1 - M.ease(k / 12.0, 'sine_in')
        f = FX.refract(f, g['contact'][0], g['contact'][1], rpx, amp=16.0 * fade + 2, width=18 + rpx * 0.05,
                       shell_a=0.10 * fade, ground_y=None)
    if T['hit'] <= i < T['launch']:                    # ため：ほんの少し暗く・硬く（光らせない）
        f = core.grade(f * 0.95, con=0.10, sat=1.0)
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
    f = debris_up(f, k)
    sh = 14.0 * math.exp(-k / 5.0)
    Mm = np.float32([[1, 0, sh * M.vnoise(k * 0.6, 51)], [0, 1, sh * M.vnoise(k * 0.6 + 3, 52)]])
    f = cv2.warpAffine(f, Mm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if k < 4:
        f = M.whip(f, (k + 1) / 4.0, axis=0, sign=1, px=420, blur=160, direction='in')
    j = i - (T['wb1'] - 4)
    if j >= 0:
        f = M.whip(f, (j + 1) / 4.0, axis=0, sign=-1, px=420, blur=160, direction='out')
    return f


def debris_up(f, k):
    """ショット B の頭：まだ爆発の中から見上げている。土くれがぶれた筋になって下から上へ抜け、下の縁に砂煙のもや。"""
    if k >= 8:
        return f
    if 'dup' not in _C:
        rng = np.random.default_rng(44)
        n = 13
        _C['dup'] = dict(x=rng.uniform(-40, W + 40, n), y0=rng.uniform(H * 0.75, H * 1.5, n), v=rng.uniform(600, 950, n),
                         w=rng.uniform(14, 44, n), tone=rng.uniform(0.7, 1.15, n), lean=rng.uniform(-0.15, 0.15, n))
    d = _C['dup']
    lay = np.zeros((H // 2, W // 2), np.float32)
    for x, y0, v, w, tone, lean in zip(d['x'], d['y0'], d['v'], d['w'], d['tone'], d['lean']):
        y1 = y0 - v * (k + 1)
        y2 = y1 + v * 0.35
        if y2 < -40 or y1 > H + 40:
            continue
        cv2.line(lay, (int(x / 2 * 8), int(y1 / 2 * 8)), (int((x + lean * v) / 2 * 8), int(y2 / 2 * 8)), float(tone),
                 max(1, int(w / 2)), cv2.LINE_AA, 3)
    lay = cv2.resize(cv2.GaussianBlur(lay, (0, 0), 3.0), (W, H), interpolation=cv2.INTER_LINEAR)
    a = np.clip(lay, 0, 1)[..., None] * 0.85
    f = f * (1 - a) + core.c('4A3F35') * lay[..., None].clip(0, 1.2) * a
    # 下の縁の砂煙のもや（8 フレームで晴れる）
    q = 1 - M.ease(k / 8.0, 'sine_out')
    if 'veil' not in _C:
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        _C['veil'] = np.clip((yy - 0.35) / 0.65, 0, 1) ** 1.5
    v = _C['veil'] * (0.75 * q)
    return f * (1 - v) + core.c('9A8E80') * v


def F_ALT():
    from cutC.beats import _field
    return _field.ALT


def _speck(f, p, r):
    """下から見上げた悪役：空に小さな暗い点（白い体の影の側）。"""
    return R.contact_shadow(f, p[0], p[1], r, r, opacity=0.75, rgb=core.c('4A5A6A'), blur=0.8)


# ---------------------------------------------------------------- ショット C（引き：クレーター）
def shot_C(i, T, beat):
    f, cam, Mm, g = CF.frame_C(beat, i)
    k = i - T['wc0']
    if k < 4:
        f = M.whip(f, (k + 1) / 4.0, axis=0, sign=-1, px=420, blur=160, direction='in')
    sh = 16.0 * math.exp(-k / 6.0) + CF.thump(g) * 5.0
    if sh > 0.3:
        Mm2 = np.float32([[1, 0, sh * M.vnoise(k * 0.6, 61)], [0, 1, sh * M.vnoise(k * 0.6 + 3, 62)]])
        f = cv2.warpAffine(f, Mm2, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return f


LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _lum(f):
    return float((f[::4, ::4] * LUM).sum(-1).mean())


def _refs(T, beat):
    """明るさのつなぎに使う、となりのショットの明るさと平均の色（一度だけ描いて覚える）。"""
    key = ('refs', beat['f0'], T['hit'])
    if key not in _C:
        a = shot_A(T['wu0'] - 1, T)
        b0 = shot_B(T['wb0'], T)
        b1 = shot_B(T['wc0'] - 5, T)
        c0 = shot_C(T['wc0'], T, beat)
        _C[key] = dict(LA=_lum(a), LB0=_lum(b0), colB=b0.reshape(-1, 3).mean(0), LB1=_lum(b1), LC=_lum(c0),
                       colC=c0.reshape(-1, 3).mean(0))
    return _C[key]


def _bridge(f, Lt, col):
    """画面を平らな色 col へ混ぜて、平均の明るさを Lt にする（振りのぶれの中だけで使う。白く光らせないため）。"""
    L, Lc = _lum(f), float((col * LUM).sum())
    w = float(np.clip((Lt - L) / (Lc - L), 0, 1)) if abs(Lc - L) > 1e-3 else 0.0
    return f * (1 - w) + col * w if w > 0 else f


def frame(i, T, beat):
    """ショットの切り替わり（A→B、B→C）は、振りの 4 フレームで平均の明るさをとなりのショットへ少しずつ寄せる
    （ほこりの壁の暗さ → 雲の明るさ、雲 → クレーターの引き）。1 フレームの差は 0.1 以内。"""
    if i < T['wb0']:
        f = shot_A(i, T)
        if i >= T['wu0']:
            k = i - T['wu0']
            f = M.whip(f, (k + 1) / float(T['wb0'] - T['wu0']), axis=0, sign=1, px=420, blur=160,
                       direction='out')
            r = _refs(T, beat)
            f = _bridge(f, r['LA'] + (r['LB0'] - r['LA']) * (k + 1) / 5.0, r['colB'])
    elif i < T['wc0']:
        f = shot_B(i, T)
        k = i - (T['wc0'] - 4)
        if k >= 0:
            r = _refs(T, beat)
            f = _bridge(f, r['LB1'] + (r['LC'] - r['LB1']) * (k + 1) / 5.0, r['colC'])
    else:
        f = shot_C(i, T, beat)
    return f


def one(beat, i):
    T = timing(beat)
    f = frame(i, T, beat)
    f = R.finish_ad(f, 2000 + i, grain=0.006, vignette=0.10, ca_px=0.25)
    f = put_telop(f, i, T)
    return np.clip(f, 0, 1).astype(np.float32)


def render(beat, ctx):
    for i in range(beat['n']):
        yield one(beat, i)
