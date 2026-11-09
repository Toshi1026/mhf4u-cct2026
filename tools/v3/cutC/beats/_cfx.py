"""案C ショット C（03 の最後・04・05 の通し）の時間・カメラ・動く物。時間は「当たってからの通しのフレーム」gt（03 の launch＝0）。

一撃の迫力（2026-11-08 クライアント修正「ただ穴が開いているだけに見える」）：
  ・砂煙のドーム：ショット C の頭ではクレーターは土色の濃い砂煙のドームに隠れていて、04 の「1,400 トン」が止まるころまでに
    ゆっくり晴れて、深いクレーター（層の見える内壁・盛り上がった縁・岩）が現れる。晴れたあとも底と岩から細い煙。
  ・降ってくる土・岩・芝のかたまり（_blast.Chunks：放物線・回転・サブフレームのぶれ・左上の光）。落ちた所で小さな砂ぼこり。
  ・地面を外へ転がっていく砂煙の輪（2 段、上下にうねる）と、草をなぎ倒しながら地平へ走る衝撃波の前線（立った草の板 → 倒れた草の板、
    前線に低い砂ぼこりの帯）。
  ・カメラ：ショット C の頭は寄り（1.30 倍）から一気に引いて大きさを見せ（引きの速さは expo_out）、そのあと 05 の終わりまで
    ゆっくり押し込む。かたまりが落ちるたびに小さく揺れる。
  ・地割れ（2026-11-08 クライアント修正 4）：縁から外の草原へ、枝分かれしながら走る放射状のひび（_crack）。
    1 段目はショット C の頭＋3（crack_times()[0]）、2 段目は 04 の「1,400」が止まるフレーム。前線に小さな砂ぼこり。
白く光らせない。水・建物・道は描かない（地割れは開けた草原の中だけ）。
"""
import math

import cv2
import numpy as np

import core
from core import W, H
import timeline_c as timeline
from cutB.beats import _motion as M
from cutC.beats import _shots as SH, _fx as FX, _blast as BL, _crack as CK

ZOOM = (1.0, 1.13)
PULL = 0.38                                        # ショット C の頭の寄り（引いて大きさを見せる）
PULL_N = 34                                        # 引きにかけるフレーム
_C = {}


def anchors():
    """(03 の f0, launch の絶対フレーム, ショット C の始まりの絶対フレーム, 05 の終わりの絶対フレーム)。"""
    if 'a' not in _C:
        from cutC.beats import c03
        bs = {b['key']: b for b in timeline.load()}
        T = c03.timing(bs['03'])
        f0 = bs['03']['f0']
        _C['a'] = (f0, f0 + T['launch'], f0 + T['wc0'], bs['05']['f1'])
    return _C['a']


def gt(beat, i):
    """当たって（打ち上がって）からの通しのフレーム。"""
    return beat['f0'] + i - anchors()[1]


def g_c0():
    """ショット C の始まりの gt。"""
    a = anchors()
    return a[2] - a[1]


def crack_times():
    """(G1, G2)：地割れの 1 段目・2 段目が走りはじめる gt。G1＝ショット C の頭＋3、G2＝04 の「ト」（1,400 で止まる）。"""
    if 'ct' not in _C:
        from cutC.beats import c04
        b4 = next(b for b in timeline.load() if b['key'] == '04')
        a = anchors()
        _C['ct'] = (g_c0() + 3, b4['f0'] + c04.timing(b4)['to'] - a[1])
    return _C['ct']


def plate(shot):
    return SH.plate_D() if shot == 'D' else SH.plate_C()


def _crack_smp(shot='C'):
    """板の画素ごとの地割れの場（割れ目がかかる行だけ）。"""
    if ('crk', shot) not in _C:
        plate(shot)
        X, Z = SH._C[shot + '_XZ']
        G1, G2 = crack_times()
        smp = CK.sample(X, Z, G1, G2)
        rows = np.nonzero((smp[0] < CK.NONE * 0.5).any(1))[0]
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        _C[('crk', shot)] = (y0, y1, tuple(m[y0:y1] for m in smp))
    return _C[('crk', shot)]


def put_cracks(src, g, shot='C'):
    y0, y1, smp = _crack_smp(shot)
    if g < crack_times()[0]:
        return src
    src = src.copy()
    src[y0:y1] = CK.shade(src[y0:y1], smp, g)
    return src


def crack_items(cam, Mm, g):
    """地割れの前線が通った所の小さな砂ぼこり（14 フレームでふくらんで消える）。"""
    items = []
    G1, G2 = crack_times()
    for j, (x, z, a, w) in enumerate(CK.fronts(G1, G2, g)):
        q = a / 14.0
        sz = (2.2 + 1.4 * w) * (0.5 + 1.1 * q)
        if ('gy', round(x, 1), round(z, 1)) not in _C:
            _C[('gy', round(x, 1), round(z, 1))] = _ground_y(x, z)
        y0 = _C[('gy', round(x, 1), round(z, 1))]
        _puff_item(items, cam, Mm, (x, y0 + sz * 0.35 + 1.5 * q, z), sz, 0.55 * (1 - q) ** 1.2 * min(1.0, a + 0.5) ,
                   j, 'soil', 0.9)
    return items


def zoom_at(beat, i):
    _, _, c0, c1 = anchors()
    fa = beat['f0'] + i
    q = float(np.clip((fa - c0) / float(c1 - c0), 0, 1))
    z = ZOOM[0] + (ZOOM[1] - ZOOM[0]) * (0.3 * q + 0.7 * M.ease(q, 'sine_inout'))
    p = float(np.clip((fa - c0) / float(PULL_N), 0, 1))
    return z * (1 + PULL * (1 - M.ease(p, 'cubic_out')))


def project(cam, Mm, P):
    """世界の点 (N,3) → 画面 (N,2)・奥行き・1m の px（画面）。"""
    S, z = cam.project(np.asarray(P, float))
    S = S @ Mm[:, :2].T + Mm[:, 2]
    return S, z, cam.fpx * Mm[0, 0] / np.maximum(z, 1e-3)


def _puff_item(items, cam, Mm, P, size_m, a, kind, tint, dark=1.0, max_px=1500.0, zmin=40.0):
    S, z, ppm = project(cam, Mm, np.asarray(P, float).reshape(1, 3))
    px = float(size_m * ppm[0] * 1.7)
    if z[0] < zmin or a <= 0.004 or px > max_px:
        return
    x, y = float(S[0, 0]), float(S[0, 1])
    if not (-px < x < W + px and -px < y < H + px):
        return
    items.append((float(z[0]), 'd', lambda L, k=kind, x=x, y=y, s=px, a=a, t=tint, d=dark: L.puff(k, x, y, s, a, t, d)))


# ---------------------------------------------------------------- 衝撃波の前線（なぎ倒れる草が地平へ走る）
def sweep_R(g):
    return 60.0 + 2900.0 * (1 - math.exp(-max(g, 0) / 55.0))


def ring_R(g):
    """地面を転がる砂煙の輪の半径（m）。"""
    return 20.0 + 470.0 * (1 - math.exp(-max(g, 0) / 90.0))


def dome_R(g):
    return 22.0 + 62.0 * (1 - math.exp(-max(g, 0) / 24.0))


def _rng():
    if 'rng' not in _C:
        rng = np.random.default_rng(17)
        n = 210
        _C['ring'] = dict(th=rng.uniform(0, 2 * np.pi, n), k=rng.integers(0, 6, n), s=rng.uniform(0.7, 1.3, n),
                          r=rng.uniform(0.86, 1.04, n), ph=rng.uniform(0, 2 * np.pi, n), tier=rng.integers(0, 2, n))
        m = 120
        _C['front'] = dict(th=rng.uniform(0, 2 * np.pi, m), k=rng.integers(0, 6, m), s=rng.uniform(0.7, 1.3, m),
                           r=rng.uniform(0.97, 1.0, m))
        # ドーム：上半球の向き・中心からの割合
        d = 110
        el = np.radians(rng.uniform(0, 80, d))
        az = rng.uniform(0, 2 * np.pi, d)
        _C['dome'] = dict(dir=np.stack([np.cos(az) * np.cos(el), np.sin(el) * 1.05, np.sin(az) * np.cos(el)], 1),
                          u=rng.uniform(0.25, 1.0, d) ** 0.5, k=rng.integers(0, 6, d), s=rng.uniform(0.75, 1.3, d),
                          rise=rng.uniform(0.1, 0.35, d))
        # くすぶる煙の出どころ（縁のまわりの岩・底）
        nsrc = 26
        rr = np.concatenate([rng.uniform(52, 150, 18), rng.uniform(0, 26, 8)])
        th = rng.uniform(0, 2 * np.pi, nsrc)
        _C['smoke'] = dict(x=rr * np.cos(th), z=rr * np.sin(th), ph=rng.uniform(0, 1, nsrc), s=rng.uniform(0.7, 1.4, nsrc),
                           k=rng.integers(0, 6, nsrc), floor=np.arange(nsrc) >= 18)
        _C['rng'] = True
    return _C


def _ground_y(x, z):
    Hm = SH.crater()[0]
    return float(SH._grid_sample(Hm, np.float32([[x]]), np.float32([[z]]))[0, 0])


def dust_items(cam, Mm, g, alpha=1.0, smoke=1.0):
    D = _rng()
    items = []
    # 転がる砂煙の輪（下の段は大きく濃い、上の段はうすい。上下にうねる）
    ring = D['ring']
    Rr = ring_R(g)
    ar = 0.78 * alpha * math.exp(-max(g - 35, 0) / 95.0)
    for th, k, s, r, ph, tier in zip(ring['th'], ring['k'], ring['s'], ring['r'], ring['ph'], ring['tier']):
        rr = Rr * r
        sz = (24.0 + 0.085 * rr) * s * (1.0 if tier == 0 else 0.75)
        hh = sz * (0.22 if tier == 0 else 0.62) * (1 + 0.35 * math.sin(ph + g * 0.16))
        _puff_item(items, cam, Mm, (rr * math.cos(th), hh, rr * math.sin(th)), sz, ar * (1.0 if tier == 0 else 0.6), int(k),
                   'soil', 0.92 if tier == 0 else 1.0)
    # 衝撃波の前線の低い砂ぼこりの帯
    Rs = sweep_R(g)
    af = 0.40 * alpha * float(np.clip((2900 - Rs) / 500.0, 0, 1))
    if af > 0.01:
        fr = D['front']
        for th, k, s, r in zip(fr['th'], fr['k'], fr['s'], fr['r']):
            rr = Rs * r
            sz = (18.0 + 0.03 * rr) * s
            _puff_item(items, cam, Mm, (rr * math.cos(th), sz * 0.2, rr * math.sin(th)), sz, af, int(k), 'dust', 1.0, zmin=60)
    # 砂煙のドーム（クレーターを隠す → ゆっくり晴れる）
    dm = D['dome']
    Rd = dome_R(g)
    ad = 0.92 * alpha * float(np.clip(1 - (g - 42) / 52.0, 0, 1)) ** 1.4
    if ad > 0.01:
        for dv, u, k, s, rise in zip(dm['dir'], dm['u'], dm['k'], dm['s'], dm['rise']):
            P = dv * Rd * u + np.array([0.0, 6.0 + rise * g, 0.0])
            sz = (30.0 + 0.30 * g) * s * (1.25 - 0.4 * u)
            dark = (0.50 + 0.50 * float(np.clip(P[1] / 75.0, 0, 1))) * (0.88 + 0.20 * float(dv[0]))   # 上・日なた（+x）ほど明るい
            _puff_item(items, cam, Mm, P, sz, ad * (0.85 + 0.15 * u), int(k), 'soil', dark)
    # 晴れたあとも残る煙：底からうすく立ちのぼる、岩からくすぶる細い煙
    sm = D['smoke']
    a_s = alpha * smoke * float(np.clip((g - 30) / 25.0, 0, 1))
    if a_s > 0.01:
        for x, z, ph, s, k, fl in zip(sm['x'], sm['z'], sm['ph'], sm['s'], sm['k'], sm['floor']):
            if ('gy', x, z) not in _C:
                _C[('gy', x, z)] = _ground_y(x, z)
            y0 = _C[('gy', x, z)]
            for j in range(4):
                q = ((g / 26.0 + ph + j / 4.0) % 1.0)
                hh = y0 + (2.0 if not fl else 4.0) + q * (26.0 if not fl else 46.0)
                sz = ((5.0 if not fl else 16.0) + q * (12.0 if not fl else 30.0)) * s
                a = (0.30 if not fl else 0.17) * a_s * math.sin(math.pi * q) ** 1.2
                _puff_item(items, cam, Mm, (x + q * 9.0, hh, z + q * 3.0), sz, a, int(k + j), 'grey', 0.88)
    return items


# ---------------------------------------------------------------- 降ってくるかたまり
def chunks_C():
    if 'ch' not in _C:
        rng = np.random.default_rng(1400)
        n = 520
        T0 = rng.uniform(0, 1, n) ** 1.6 * 8.0
        r0 = 4.0 + 4.5 * T0 + rng.uniform(0, 6, n)
        az = rng.uniform(0, 2 * np.pi, n)
        el = np.radians(rng.uniform(46, 74, n))
        sp = rng.uniform(1.4, 4.6, n)
        P0 = np.stack([r0 * np.cos(az), np.full(n, 0.5), r0 * np.sin(az)], 1)
        V = np.stack([np.cos(az) * np.cos(el), np.sin(el), np.sin(az) * np.cos(el)], 1) * sp[:, None]
        size = 0.25 + 2.3 * rng.uniform(0, 1, n) ** 3.0
        kind = rng.choice([0, 1, 2, 3], n, p=[0.42, 0.25, 0.22, 0.11])
        size = np.where(kind == 2, size * 1.5 + 0.4, size)
        _C['ch'] = BL.Chunks(P0, V, T0, size, kind, seed=5, g=0.11, spin=(0.04, 0.22))
    return _C['ch']


def landing_items(cam, Mm, g):
    """かたまりが落ちた所の砂ぼこり（大きなものだけ、16 フレームで広がって消える）。"""
    ch = chunks_C()
    items = []
    age = g - ch.t_land
    sel = np.nonzero((age >= 0) & (age < 16) & (ch.size > 0.6))[0]
    if len(sel) == 0:
        return items
    P, _ = ch.pos(ch.t_land)
    for j in sel:
        q = age[j] / 16.0
        sz = (4.0 + 5.0 * ch.size[j]) * (0.6 + 0.9 * q)
        a = 0.55 * (1 - q) ** 1.3
        _puff_item(items, cam, Mm, (P[j, 0], sz * 0.25, P[j, 2]), sz, a, int(j), 'soil', 0.95)
    return items


def trail_items(cam, Mm, g):
    """大きなかたまりのうしろの砂の尾（弧を描いて飛ぶ放出物の筋に見える）。"""
    ch = chunks_C()
    if 'big' not in _C:
        _C['big'] = np.nonzero(ch.size > 1.0)[0][:80]
    items = []
    big = _C['big']
    for lag in (2.0, 5.0, 9.0, 14.0):
        t = g - lag
        P, tau = ch.pos(np.full(ch.n, t))
        for j in big:
            if tau[j] <= 0 or t >= ch.t_land[j] or g >= ch.t_land[j] + 10:
                continue
            a = 0.40 * (1 - lag / 18.0) * float(np.clip(1 - (g - 40) / 70.0, 0, 1))
            sz = (2.5 + 1.8 * ch.size[j]) * (1 + lag * 0.12)
            _puff_item(items, cam, Mm, P[j], sz, a, int(j + lag), 'soil', 0.9)
    return items


def thump(g):
    """大きなかたまりが落ちた時の小さな揺れの強さ（0..1）。"""
    ch = chunks_C()
    age = g - ch.t_land
    m = (age >= 0) & (age < 6) & (ch.size > 1.2)
    return 1 - math.exp(-float(np.sum(ch.size[m] * np.exp(-age[m] / 1.6))) / 10.0)


def zoom_D(beat, i):
    """ショット D（05）：1.00 → 1.14 の押し込み（ゾウが詰まっていくクレーターの中へ寄る）。"""
    q = i / float(max(beat['n'] - 1, 1))
    return 1.0 + 0.14 * (0.3 * q + 0.7 * M.ease(q, 'sine_inout'))


def frame_C(beat, i, extra=None, shot='C', smoke=1.0):
    """ショット C（または D：05 の中をのぞく画）の 1 フレーム（sRGB）と、そのときの (cam, Mm)。
    extra(f, cam, Mm, g)：砂煙の手前に描く物（05 のゾウ）。smoke：くすぶる煙の濃さ。"""
    pl, dep, cam = plate(shot)
    g = gt(beat, i)
    Rs = sweep_R(g)
    if Rs < 2700:                                       # 前線の外は草が立ったまま
        rsm = SH._C[shot + '_r']
        m = np.clip((rsm - Rs) / 40.0 + 0.5, 0, 1)[..., None]
        src = pl * (1 - m) + SH._C[shot + '_stand'] * m
        # 前線のすぐ内側：倒れたばかりの葉のつやが少し明るい帯
        band = np.exp(-((rsm - Rs + 60.0) / 70.0) ** 2)[..., None] * 0.07
        src = src + (0.85 - src) * band
    else:
        src = pl
    src = put_cracks(src, g, shot)
    f, Mm = SH.view_C(src, zoom_D(beat, i) if shot == 'D' else zoom_at(beat, i))
    f = f.copy()
    if extra is not None:
        f = extra(f, cam, Mm, g)
    items = dust_items(cam, Mm, g, smoke=smoke)
    items += landing_items(cam, Mm, g)
    items += trail_items(cam, Mm, g)
    items += crack_items(cam, Mm, g)
    items += chunks_C().items(g, cam, M2=Mm, near=20.0, max_px=120.0)
    f = BL.draw_sorted(f, items, split_z=float(np.linalg.norm(cam.eye)))
    return f, cam, Mm, g
