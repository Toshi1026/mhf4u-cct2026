"""案B T（ナレーションなし・36フレーム＝1.2秒）着水 → CG から実写へ。

案A の b11 をそのまま使う（土佐湾の線画 → 仁淀川を河口（2026-11-07 に本当の河口 33.4593N 133.4815E へ訂正）まで → 河口へ降りて、IMG_9662 の1フレーム目に
画素単位で重なる）。その上に、最初の約 12 フレームだけ小さな白いしぶきを重ねる：
 0–3 ：悪役（の点）が東北東の上から短い白い筋になって落ちてくる（03 の放物線と同じ向き）。
 3–13：MAZE の前の海の点（03 の放物線の終点と同じ地図の点。本当の仁淀川の河口の約 0.45km 西）に、上向きの扇に跳ねる数粒のしぶきと小さな泡。
       輪（同心円）や光は使わない。しぶきの大きさは画面の px で決め、寄りに合わせて少しだけ大きくなる。
最後のフレームは b11 と同じ（08 は案A b12 の 0 フレーム目と同じ IMG_9662 から始められる）。
2026-11-07 動きの仕様（P2）：筋はサブフレームのぶれをかけたカプセル（先が太く明るい）。
2026-11-08 絵の作り直し（美術監督の look-dev v2）：b11 の線画の地図の代わりに、03・04 と同じ立体の地図（_maplook、
  b11 のカメラで射影）を見せる。最初の 7 フレームは 04 の最後の暗さから明るくなる（b11 の線画の出と同じ）。
2026-11-08 美術監督のレビュー（T 5/10）の直し：
- しぶき：7 枚の対称な扇（冠・蓮のロゴに見えた）をやめ、着水点を海の側へ（寄りの途中でも海岸線から 18px 以上沖）。
  1 本の傾いた水の柱（入ってくる向き＝東北東→西南西へ 18°）＋大きさのちがう 6 粒（0.4–1.4、角度 ±12° のゆらぎ、
  寿命をずらす、動く範囲は 45px まで、最後の 4 フレームで 0.9→0）＋水面の不規則な泡（ノイズの縁、1.0→1.6 倍に広がって消える、
  水の所だけ）。輪・光なし。真上に近いカメラでは高さを縮めて描く（粒が陸の上に散らない）。
- 川（仁淀川）：2px・0.70・#5FA3D6、にじみなし（仕上げの後に描く）。河口の点は描かない（しぶきと競わない）。
- 地図 → b11 の受け渡しを 2 段に：9→13 で地図の起伏を b11 の背景の色と明るさへ寄せる（細かい起伏のコントラストを先に落とす
  明るさ合わせ。出典と地図の海岸線も 13 で消える）→ 13→18 で b11 の線をその上に描き足す（海岸線は 1 本だけ：地図の海岸線が
  消えてから b11 の海岸線が出るので二重にならない）。18 フレーム目からは b11 そのもの（画素単位で同じ）。
- 落ちてくる筋：太さ 4px のカプセルの頭（先ほど明るい）＋ 12px の尾、海に落ちる影つき。向きは 07 の弧と同じ。
- カメラ：0→17.5 フレームの寄りを、拡大率の対数が 1 本の 3 次式（静止から始まり、17.5 で b11 の速さにつながる）になるよう
  b11 のカメラの時刻だけを付け替える（b11 の線の出・川ののび方の時刻はそのまま）。18 フレーム目からは b11 のカメラそのもの。
"""
import math

import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from beats import b11
from beats import _mapui as U, _geo as G
from cutB.beats import _bmap as B, _maplook as ML, _render as RR

FALL = (0, 3)                 # 筋が落ちてくる
SPL = (3, 13)                 # しぶき
HAND = 18                     # このフレームからは b11 と画素単位で同じ
T_RT = 17.5                   # カメラの時刻の付け替えはここまで（b11 の川のサブフレーム i±0.45 が 18 にかからないように）
RELIEF = (9, 13)              # 地図の起伏 → b11 の背景
LINES = (13, 18)              # b11 の線を描き足す
RIVER = core.c('5FA3D6')      # 川（にじみなし・2px・0.70）
RIVER_W, RIVER_A = 2.0, 0.70
DIMC = core.c('0A1D33')
OFFSHORE = 18.0               # しぶきの点を地図の px でこれだけ沖へ（寄りの途中 f9 で海岸線から約 18px 以上）
LEAN = 18.0                   # しぶきの傾き（度、入ってくる向き＝画面の左へ）
MAX_R = 45.0                  # 粒の動く範囲（画面 px）
DIRV = np.array([0.42, -0.91])                           # 筋が来る方向（画面の右上）＝07 の弧の落ちる向き

# 粒：(角度のずれ°, 速さ, 大きさ, 遅れ（フレーム）, 寿命（フレーム）)。角度は傾いた柱の向きからのずれ
# 地面の向き：0°＝西南西（海岸に沿って、入ってくる向き）、+90°＝沖、180°＝東北東
DROPS = [(-12, 1.10, 1.40, 0, 9), (18, 1.25, 0.90, 0, 8), (48, 0.90, 1.10, 1, 7), (85, 0.65, 0.60, 1, 6),
         (130, 0.55, 0.40, 2, 5), (2, 0.70, 0.75, 1, 7), (168, 0.50, 0.55, 0, 6)]
_JIT = np.random.default_rng(7).uniform(-12, 12, len(DROPS))


# ---------------------------------------------------------------- カメラの時刻の付け替え
_RT = {}


def _flow(cam, t, dt=0.05):
    """カメラの見かけの速さ（画面 px／フレーム）：画面の格子の地面の点が dt のあいだに動く量の中央値。"""
    if 'grid' not in _RT:
        gx, gy = np.meshgrid(np.linspace(60, 1020, 9), np.linspace(100, 1820, 15))
        _RT['grid'] = np.stack([gx.ravel(), gy.ravel()], 1)
    grid = _RT['grid']
    c0, c1 = cam(t), cam(t + dt)
    ok = grid[:, 1] > c0.horizon() + 30
    q = c1.proj(c0.ground(grid[ok]))
    return float(np.nanmedian(np.hypot(*(q - grid[ok]).T))) / dt


def _retime(S):
    """b11 のカメラの時刻 g(t)（0..T_RT）。見かけの速さ（_flow）が 0 で b11 と同じ速さから T_RT で b11 と同じ速さへ、
    単調に速くなる 1 本の曲線になるように（起き上がりはじめ 10–12 の速さの山と、6→9 の段をならして 1 回の寄りに）。"""
    if 'g' in _RT:
        return _RT['g']
    cam = S['cam']
    ts = np.linspace(0.0, T_RT, 701)
    v = np.array([_flow(cam, t) for t in ts])
    D = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(ts))])
    D1, v0, v1 = D[-1], v[0], v[-1]
    s = ts / T_RT
    # 速さ v0 + (v1 − v0)·s^p（単調に速くなるだけ。途中で遅くならない）、積分が D1 になる p
    p = max(0.5, (v1 - v0) * T_RT / max(D1 - v0 * T_RT, 1e-6) - 1)
    tgt = v0 * ts + (v1 - v0) * T_RT / (p + 1) * s ** (p + 1)
    tgt = np.maximum.accumulate(np.clip(tgt, 0, D1))
    g = np.interp(tgt, D, ts)
    _RT['g'] = lambda t: float(np.interp(t, ts, g)) if t < T_RT else float(t)
    return _RT['g']


def _setup():
    S = b11._setup()
    if 'S2' not in _RT:
        g = _retime(S)
        S2 = dict(S)
        cam = S['cam']
        S2['cam'] = lambda t: cam(g(t))
        _RT['S2'] = S2
    return S, _RT['S2']


# ---------------------------------------------------------------- b11 の部品
def _b11_frame(i, S, river=core.ACC):
    """b11.render の i フレーム目と同じ手順（river=ACC のとき同じ結果）。"""
    if 12 <= i <= 17 or 28 <= i <= 32:
        subs = tuple(i + d for d in (-0.4, -0.2, 0.0, 0.2, 0.4))
    elif 2 <= i <= 27 and not 18 <= i <= 25 or i == 33:
        subs = (i - 0.33, i, i + 0.33)
    else:
        subs = (i,)
    acc = None
    for t in subs:
        m, c, r, cam = b11._lines(t, S)
        acc = [m, c, r] if acc is None else [a + b for a, b in zip(acc, (m, c, r))]
    m, c, r = [a / len(subs) for a in acc]
    if 8 <= i <= 21:
        r = sum(b11._lines(i + d, S, river_only=True) for d in np.linspace(-0.45, 0.45, 9)) / 9
    cam = S['cam'](i)
    if 18 <= i <= 25:
        v = cam.f * abs(np.radians(S['cam'](i + 0.5).beta - S['cam'](i - 0.5).beta))
        v = min(v, 360.0)
        m, c, r = (core.whip(x, v * 0.8) for x in (m, c, r))
    bg = b11._background(i, cam, S)
    lm = np.clip(np.maximum(m, c), 0, 1)[..., None]
    f = bg * (1 - lm) + core.FOAM * lm
    lr = r[..., None]
    f = f * (1 - lr) + river * lr
    return f, dict(m=m, c=c, r=r, cam=cam, bg=bg)


def _b11_parts(i, S2):
    """0–17 フレーム用：付け替えたカメラでの b11 の線（地図・手前）と背景。川は別に描く（_river）。"""
    if 12 <= i <= 17:
        subs = tuple(i + d for d in (-0.4, -0.2, 0.0, 0.2, 0.4))
    elif 2 <= i:
        subs = (i - 0.33, i, i + 0.33)
    else:
        subs = (i,)
    need = i >= LINES[0] - 1
    m = c = None
    if need:
        acc = None
        for t in subs:
            mm, cc, _, _ = b11._lines(t, S2)
            acc = [mm, cc] if acc is None else [acc[0] + mm, acc[1] + cc]
        m, c = acc[0] / len(subs), acc[1] / len(subs)
    cam = S2['cam'](i)
    bg = b11._background(i, cam, S2)
    return dict(m=m, c=c, cam=cam, bg=bg)


def _river(i, S2):
    """仁淀川（b11 と同じのび方・同じ時刻）。2px・#5FA3D6・0.70、河口の点なし。動きの速い所は 3 枚の平均。"""
    # 動きのぶれ：1 フレームの間に川の線が動く量（画面 px）に合わせて枚数を決める（細い線が何本にも分かれて見えないように）
    R_ = S2['river']
    sel = R_[::8]
    q0, q1 = S2['cam'](i - 0.15).proj(sel), S2['cam'](i + 0.15).proj(sel)
    ok = np.isfinite(q0).all(1) & np.isfinite(q1).all(1) & (q0[:, 1] > -50) & (q0[:, 1] < core.H + 50) & \
        (q0[:, 0] > -50) & (q0[:, 0] < core.W + 50)
    mv = float(np.percentile(np.hypot(*(q1[ok] - q0[ok]).T), 90)) if ok.any() else 0.0
    ns = int(np.clip(math.ceil(mv / 1.2), 1, 40))
    subs = tuple(i + d for d in np.linspace(-0.15, 0.15, ns)) if ns > 1 else (i,)       # シャッター約 110°
    acc = None
    for t in subs:
        a_riv = b11._ramp(t, 1, 4) * (1 - b11._ramp(t, 17, 21))
        u = core.ease((t - 1) / 8.5, 'inout')
        if a_riv <= 0.003 or u <= 0:
            continue
        rs = S2['rs']
        n = int(np.searchsorted(rs, u))
        L = G.Lines()
        L.poly(S2['cam'](t).proj(R_[:max(n, 2)]), RIVER_W, 255 * a_riv)
        m = L.done()
        acc = m if acc is None else acc + m
    if acc is None:
        return None
    # 起き上がりのあいだ（10→16）は 0.70 → 約 0.18 へ（ぶれた川が帯に見えないように。18 の b11 のうすい川へつながる）
    return acc / len(subs) * RIVER_A * (1 - 0.75 * core.ease((i - 10) / 6.0))


# ---------------------------------------------------------------- 立体の地図
def _map_frame(i, cam, bg):
    """立体の地図（付け替えたカメラ）＋海岸線（地図の）。仕上げまで。戻り値：(f, water)。"""
    rgb, wa, valid = ML.render_cam(cam, i)
    f = ML.sea_and_air(rgb, wa, i, haze=0.16 * min(1.0, cam.phi / 90.0), tilt=1.2)
    v = valid[..., None]
    f = f * v + bg * (1 - v)
    z = cam.f / cam.h
    lw = ML._levels_for(z)
    lv = ML.level(max(lw, key=lambda a: a[1])[0])
    Lc = G.Lines()
    for name, a in (('sea', 0.40), ('inland', 0.20)):
        for P in lv['coast'][name]:
            Q = cam.proj(P)
            ok = np.isfinite(Q).all(axis=1)
            if not ok.any() or max(np.ptp(Q[ok, 0]), np.ptp(Q[ok, 1])) < ML.MIN_LOOP:
                continue
            Lc.poly(Q, 1.25, 255 * a, closed=False)
    m = Lc.done()
    f += (np.asarray(core.FOAM, np.float32) - f) * m[..., None]
    # 04 の最後の暗さから（b11 の線画が濃くなるのと同じ 0→7 フレーム）
    vis = 0.42 + 0.58 * core.ease(i / 7.0)
    f = f * vis + np.asarray(DIMC, np.float32) * (1 - vis)
    f = ML.label_scrims(f, 1.0, cg=True, credit_a=1.0)
    return RR.finish_ad(f, 643 + i, grain=0.007, vignette=0.10, bloom_k=0.05), wa * valid


def _relief_to_bg(fm, bg, k):
    """地図の起伏 → b11 の背景（明るさ合わせ）：大きなむら（σ≈32px）は k で背景の色へ、細かい起伏のコントラストは
    (1−k)² で先に落とす（2 枚の絵が重なって見えない）。"""
    if k <= 0:
        return fm
    if k >= 1:
        return bg + np.zeros_like(fm)
    sm = cv2.resize(fm, (core.W // 8, core.H // 8), interpolation=cv2.INTER_AREA)
    lo = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 4.0), (core.W, core.H), interpolation=cv2.INTER_LINEAR)
    det = fm - lo
    return lo * (1 - k) + bg * k + det * (1 - k) ** 2


# ---------------------------------------------------------------- しぶき
def _splash_point(cam):
    """着水点（MAZE の前の海）を海の側へ OFFSHORE だけずらした点の画面座標と、海の側・海岸に沿う向き（画面の単位ベクトル）。"""
    S = B.setup()
    P = S['dest']
    nrm = S['nrm']
    Ps = P + nrm * OFFSHORE
    q = cam.proj(np.stack([Ps, Ps + nrm * 4.0, Ps + np.array([-nrm[1], nrm[0]]) * 4.0]))
    p = q[0]
    sea = (q[1] - q[0]) / (np.linalg.norm(q[1] - q[0]) + 1e-9)
    along = (q[2] - q[0]) / (np.linalg.norm(q[2] - q[0]) + 1e-9)
    if along[0] > 0:                                    # 西南西（画面の左）向きに
        along = -along
    return p, sea, along


def _noise1(x, y, seed, sc):
    return RR.fbm(x / sc, y / sc, 3, seed=seed)


def _splash(cx, cy, i, k, phi, sea, along, water=None):
    """しぶきの板（前乗せでない rgba, x0, y0）と水面の泡の板。i：フレーム、k：大きさ、phi：カメラの俯角（度）。
    sea / along：画面の海の側・西南西の単位ベクトル。water：画面の水のマスク（あれば泡を水の所だけに）。"""
    u = 10.5 * k
    pw = int(2 * MAX_R + 4 * u) + 8
    x0, y0 = int(cx - pw / 2), int(cy - pw / 2)
    yy, xx = np.mgrid[0:pw, 0:pw].astype(np.float32)
    px, py = cx - x0, cy - y0
    tau = (i - SPL[0]) / float(SPL[1] - SPL[0])                         # 0..1
    vf = 0.45 + 0.55 * math.cos(math.radians(phi))                    # 高さの見え方（真上 0.45 → 横 1.0）
    fy = max(0.45, math.sin(math.radians(phi)))                       # 水面の楕円のつぶれ
    # ---- 水面の泡：ノイズの縁の不規則な形（西南西へのびる）、1.0→1.6 倍に広がって消える（中を埋めた形。輪にしない）
    grow = 1.0 + 0.6 * core.ease(min(1.0, tau / 0.8), 'out')
    rf = u * 0.80 * grow
    fcx, fcy = px + along[0] * 0.30 * u + sea[0] * 0.10 * u, py + along[1] * 0.30 * u + sea[1] * 0.10 * u
    dx, dy = (xx - fcx), (yy - fcy) / fy
    ax, ay = along[0], along[1] / fy
    an = math.hypot(ax, ay) + 1e-6
    ax, ay = ax / an, ay / an
    pa = dx * ax + dy * ay
    pb = -dx * ay + dy * ax
    pa = np.where(pa > 0, pa / 1.45, pa / 0.95)                     # 西南西の側へ長く（非対称）
    d = np.sqrt(pa ** 2 + (pb / 0.85) ** 2) / rf
    ang = np.arctan2(pb, pa)
    edge = 1.0 + 0.32 * RR.fbm(np.cos(ang) * 1.1 + 3.1, np.sin(ang) * 1.1 + 0.5 * tau, 3, seed=41)
    core_ = RR.smoothstep(edge, edge - 0.45, d)
    lace = 0.72 + 0.28 * RR.smoothstep(-0.4, 0.4, _noise1(xx + 5.0 * tau, yy, 43, 2.2))   # 細かいむら（穴にしない）
    a_f = core_ * lace * 0.78 * (1 - RR.smoothstep(0.30, 1.0, tau))
    if water is not None:
        X0, Y0 = max(x0, 0), max(y0, 0)
        X1, Y1 = min(x0 + pw, core.W), min(y0 + pw, core.H)
        wm = np.zeros((pw, pw), np.float32)
        if X1 > X0 and Y1 > Y0:
            wm[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] = water[Y0:Y1, X0:X1]
        a_f = a_f * RR.smoothstep(0.35, 0.85, wm)
    foam = np.concatenate([np.broadcast_to(np.asarray(core.c('D9E8F3'), np.float32), (pw, pw, 3)),
                           a_f[..., None].astype(np.float32)], -1)
    # ---- 傾いた水の柱（立ち上がってすぐ崩れる。縁はノイズでぎざぎざ、上は粒へほどける）
    lr = math.radians(LEAN)
    axs = np.array([math.copysign(math.sin(lr), along[0]), -math.cos(lr)])     # 画面の上から入ってくる向き（左）へ 18°
    hmax = 3.1 * u * vf
    hp = hmax * math.sin(math.pi * min(1.0, tau / 0.55)) if tau < 0.55 else 0.0
    col_a = np.zeros((pw, pw), np.float32)
    if hp > 1.0:
        al = math.hypot(*axs)
        ux_, uy_ = axs[0] / al, axs[1] / al
        nx_, ny_ = -uy_, ux_
        ss = np.linspace(0.0, 1.0, 14)
        wob = RR.fbm(ss * 2.2 + 5.0, np.full_like(ss, 0.37 * i), 2, seed=47)
        wob2 = RR.fbm(ss * 2.2 + 9.0, np.full_like(ss, 0.37 * i), 2, seed=48)
        hb = u * (0.52 * (1 - ss) ** 0.85 + 0.16)                         # 根元が太く、先へ細く（先は丸い）
        hl = hb * (0.85 + 0.30 * wob)                                    # 左右で太さがちがう（対称にしない）
        hr = hb * (1.10 + 0.30 * wob2)
        cxs = px + ux_ * hp * ss + nx_ * u * 0.12 * np.sin(ss * 3.0)      # 少し曲がった柱
        cys = py + uy_ * hp * ss + ny_ * u * 0.12 * np.sin(ss * 3.0)
        left = [(cxs[k] - nx_ * hl[k], cys[k] - ny_ * hl[k]) for k in range(len(ss))]
        right = [(cxs[k] + nx_ * hr[k], cys[k] + ny_ * hr[k]) for k in range(len(ss))][::-1]
        Lp = core.SS(s=4, w=pw, h=pw)
        Lp.draw.polygon(Lp.p(left + right), fill=(255, 255, 255, 255))
        # 柱の先の丸い水の玉（ワージントンの噴流のように、先から離れていく）
        sep = max(0.0, tau - 0.18) * 1.6
        rt = u * 0.30
        tx, ty = cxs[-1] + ux_ * (hp * sep * 0.9 + rt * 0.6), cys[-1] + uy_ * (hp * sep * 0.9 + rt * 0.6)
        Lp.draw.ellipse(Lp.box(tx - rt, ty - rt, tx + rt, ty + rt), fill=(255, 255, 255, 255))
        col_a = Lp.done()[..., 3] * 0.95 * (1 - RR.smoothstep(0.35, 0.62, tau))
    # ---- 粒：地面の向き（西南西を中心に、海の側へ寄せる）× 放物線の高さ（画面の上へ vf 倍）。
    #      角度 ±12° のゆらぎ、大きさ・寿命ばらばら、地面の動きは 45px まで、最後の 4 フレームで 0.9→0、寿命の終わりに水面へ戻る
    L = core.SS(s=4, w=pw, h=pw)
    d_ = L.draw
    for (da, sp, sz, dl, life), jit in zip(DROPS, _JIT):
        T = (i - SPL[0] - dl)
        if T <= 0 or T > life:
            continue
        th = math.radians(da + jit)
        gx, gy = along[0] * math.cos(th) + sea[0] * math.sin(th), along[1] * math.cos(th) + sea[1] * math.sin(th)
        reach = min(MAX_R, MAX_R * 0.78 * sp)
        hgt = reach * 0.55 * vf * (1.15 - 0.3 * abs(math.sin(th)))
        q = T / float(life)
        q1 = (T + 0.5) / float(life)
        x = px + gx * reach * (1 - (1 - q) ** 2.2)
        y = py + gy * reach * (1 - (1 - q) ** 2.2) - 4 * hgt * q * (1 - q)
        xn = px + gx * reach * (1 - (1 - q1) ** 2.2)
        yn = py + gy * reach * (1 - (1 - q1) ** 2.2) - 4 * hgt * q1 * (1 - q1)
        r = u * 0.21 * sz * (1 - 0.3 * q)
        aa = 0.9 * (1 - RR.smoothstep(life - 4.0, float(life), float(T)))
        if aa > 0.01 and r > 0.3:
            _drop(d_, L, x, y, (xn - x) * 2, (yn - y) * 2, r, aa)
    dr = L.done()
    a_air = np.maximum(dr[..., 3], col_a)
    air = np.zeros((pw, pw, 4), np.float32)
    air[..., :3] = 1.0
    air[..., 3] = a_air
    return foam, air, x0, y0


def _drop(d, L, x, y, vx, vy, r, a):
    """速さ (vx, vy)（px/フレーム）の向きにのびた粒（楕円の多角形）。長さ ∝ 速さ×0.5 フレーム。"""
    sp = float(np.hypot(vx, vy))
    ux, uy = (vx / sp, vy / sp) if sp > 1e-3 else (0.0, -1.0)
    la = r * 0.95 + sp * 0.25
    lb = r * 0.8 * (1 - 0.25 * min(1.0, sp / 20.0))
    th = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    ca, cb = np.cos(th) * la, np.sin(th) * lb
    pts = [(x + ca[k] * ux - cb[k] * uy, y + ca[k] * uy + cb[k] * ux) for k in range(len(th))]
    d.polygon(L.p(pts), fill=L.col(core.FOAM, a))


# ---------------------------------------------------------------- 落ちてくる筋
def _streak_at(cx, cy, t):
    """フレーム t の筋の板 (rgba, x0, y0)：太さ 4px のカプセルの頭（先ほど明るい）＋ 12px の尾（細く・うすく）。"""
    tau = (t + 1) / float(FALL[1] - FALL[0] + 1)
    head = np.array([cx, cy]) + DIRV * 110 * (1 - tau)
    tail = head + DIRV * 12.0
    pad = 8
    xs, ys = (head[0], tail[0]), (head[1], tail[1])
    x0, y0 = int(np.floor(min(xs))) - pad, int(np.floor(min(ys))) - pad
    w, h = int(np.ceil(max(xs))) + pad - x0, int(np.ceil(max(ys))) + pad - y0
    L = core.SS(s=4, w=w, h=h)
    d = L.draw
    nrm = np.array([-DIRV[1], DIRV[0]])
    m = 12
    for j in range(m):
        q0, q1 = j / float(m), (j + 1) / float(m)            # 0＝尾、1＝先
        p0, p1 = tail + (head - tail) * q0, tail + (head - tail) * q1
        w0, w1 = 0.3 + 1.6 * q0 ** 1.3, 0.3 + 1.6 * q1 ** 1.3
        pts = [p0 + nrm * w0 - (x0, y0), p1 + nrm * w1 - (x0, y0), p1 - nrm * w1 - (x0, y0), p0 - nrm * w0 - (x0, y0)]
        d.polygon(L.p([tuple(v) for v in pts]), fill=L.col(core.FOAM, 0.15 + 0.75 * q1))
    hx, hy = head - (x0, y0)
    d.ellipse(L.box(hx - 1.95, hy - 1.95, hx + 1.95, hy + 1.95), fill=L.col(core.FOAM, 1.0))
    return L.done(), x0, y0


def _streak(f, cx, cy, i):
    rgba, x0, y0 = _streak_at(cx, cy, i)
    return _paste_shadowed(f, rgba, x0, y0, sh_a=0.40, lit=False)


# ---------------------------------------------------------------- 貼る
def _paste_shadowed(f, rgba, x0, y0, sh_a=0.30, lit=True):
    """白い粒の板を、海に落ちるやわらかい影（右下）つきで貼る。"""
    h, w = rgba.shape[:2]
    a = rgba[..., 3]
    pad = 8
    sh = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
    sh[pad:pad + h, pad:pad + w] = a
    sh = cv2.GaussianBlur(cv2.warpAffine(sh, np.float32([[1, 0, 2], [0, 1, 3]]), (w + 2 * pad, h + 2 * pad)), (0, 0), 1.6)
    X0, Y0 = x0 - pad, y0 - pad
    fx0, fy0, fx1, fy1 = max(X0, 0), max(Y0, 0), min(X0 + w + 2 * pad, core.W), min(Y0 + h + 2 * pad, core.H)
    if fx1 > fx0 and fy1 > fy0:
        sub = f[fy0:fy1, fx0:fx1]
        k = sh[fy0 - Y0:fy1 - Y0, fx0 - X0:fx1 - X0][..., None] * sh_a
        sub *= 1 - k * (1 - np.asarray(core.c('1B3550'), np.float32))
    return U.paste(f, _lit_water(rgba) if lit else rgba, x0, y0)


_KEY = np.array([-0.55, -0.65, 0.52], np.float32) / np.linalg.norm([-0.55, -0.65, 0.52])   # 共通の太陽（左上）
_LIT, _SHD = np.asarray(core.c('E6F1F9'), np.float32), np.asarray(core.c('6F98BC'), np.float32)


def _lit_water(rgba):
    """白い平らな粒の板 → 光の当たった水の粒（07 のしぶきと同じ、左上の太陽で陰影・小さなハイライト・縁はやや透ける）。"""
    a = np.ascontiguousarray(rgba[..., 3], np.float32)
    hgt = cv2.GaussianBlur(a, (0, 0), 1.3)
    gy, gx = np.gradient(hgt)
    nx, ny, nz = -gx * 3.0, -gy * 3.0, np.full_like(a, 0.35)
    nn = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    ndl = (nx * _KEY[0] + ny * _KEY[1] + nz * _KEY[2]) / nn
    lam = np.clip(0.5 + 0.5 * ndl, 0, 1)[..., None]
    col = _SHD + (_LIT - _SHD) * lam ** 0.8
    spec = np.clip(ndl, 0, 1) ** 24 * 0.25
    col = np.minimum(col + spec[..., None], 0.92)
    ae = a * (0.80 + 0.20 * np.clip(hgt * 1.4, 0, 1))
    return np.concatenate([col, ae[..., None]], -1).astype(np.float32)


def _lum(f):
    return float((f[::4, ::4] @ np.array([0.2126, 0.7152, 0.0722], np.float32)).mean())


def _l18(S):
    if 'l18' not in _RT:
        _RT['l18'] = _lum(np.clip(_b11_frame(HAND, S)[0], 0, 1))
    return _RT['l18']


def render(beat, ctx):
    l9 = None
    n = beat['n']
    assert n == 36, n
    S, S2 = _setup()
    for i in range(n):
        if i >= HAND:                                   # b11 そのもの（画素単位で同じ）
            f, _ = _b11_frame(i, S)
            yield np.clip(f, 0, 1).astype(np.float32)
            continue
        parts = _b11_parts(i, S2)
        cam = parts['cam']
        k = core.ease((i - RELIEF[0]) / float(RELIEF[1] - RELIEF[0]))
        water = None
        if k < 1:
            fm, water = _map_frame(i, cam, parts['bg'])
            f = _relief_to_bg(fm, parts['bg'], k)
        else:
            f = parts['bg'].astype(np.float32) + np.zeros((core.H, core.W, 3), np.float32)
        # b11 の線（地図の起伏が消えてから描き足す）
        wl = core.ease((i - LINES[0]) / float(LINES[1] - LINES[0]))
        if wl > 0 and parts['m'] is not None:
            lm = (np.clip(np.maximum(parts['m'], parts['c']), 0, 1) * wl)[..., None]
            f = f * (1 - lm) + core.FOAM * lm
        # 川（にじみなし）
        rm = _river(i, S2)
        if rm is not None:
            lr = rm[..., None]
            f = f * (1 - lr) + RIVER * lr
        # 明るさ合わせ：9→18 の平均の明るさを、9 フレーム目の地図から b11 の 18 フレーム目へ一直線に（暗い谷・山を作らない）
        if i == RELIEF[0]:
            l9 = _lum(f)
        elif RELIEF[0] < i < HAND and l9 is not None:
            tgt = l9 + (_l18(S) - l9) * (i - RELIEF[0]) / float(HAND - RELIEF[0])
            f = f * np.float32(tgt / max(_lum(f), 1e-4))
        f = ML.credit(f, 1 - k)
        # しぶき・筋
        if i < SPL[1]:
            p, sea, along = _splash_point(cam)
            z = cam.f / cam.h
            kk = float(np.clip(z ** 0.35, 1.0, 1.6))
            if FALL[0] <= i < FALL[1]:
                f = _streak(f, p[0], p[1], i)
            if SPL[0] <= i < SPL[1]:
                foam, air, x0, y0 = _splash(p[0], p[1], i, kk, cam.phi, sea, along, water)
                f = U.paste(f, foam, x0, y0)
                f = _paste_shadowed(f, air, x0, y0)
        yield np.clip(f, 0, 1).astype(np.float32)
