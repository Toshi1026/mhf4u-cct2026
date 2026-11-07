"""06・07 の舞台：横から見た断面（案A と同じ core.xs_background の空と水）。左に桂浜の陸、そこから右へ放物線。
縦の縮尺は実際の比（富士山 3,776m と最高点 2,500m が同じ縮尺）。横は 10.1km を画面の幅に縮めた（縦を約1.8倍に強調）。
2026-11-07 訂正：10km 版（最高点 h＝R/4＝2,525m。旧 13.3km・3,300m）。"""
import cv2
import numpy as np

import core
from core import W, H, WL
from beats import _mapui as U
from cutB.beats import _art as A

X0, X1 = 150.0, 980.0                # 桂浜（打ち出し）→ 着水
PXM = 550.0 / 3776                   # 縦：1m あたりの px（富士山の頂上が y=600）
HAPEX = 2525 * PXM                   # 試算の最高点（表示は「約2,500m」）
FUJI_X = 752.0                       # 頂点（x≈565）と重ならないよう右へ（2026-11-07：頂点が低くなったため 700→752、すそも少し細く）
VS = 1.55                            # 飛んでいる悪役の大きさ


def arc(u):
    u = np.asarray(u, np.float64)
    return X0 + (X1 - X0) * u, WL - 4 * HAPEX * u * (1 - u)


def y_of(m):
    return WL - m * PXM


APEX = (float(arc(0.5)[0]), float(arc(0.5)[1]))


SHORE_X = 205.0


def seabed_y(x):
    """海底（模式図。深さの縮尺は合わせていない：海であることが読めるだけの深さ）。"""
    x = np.asarray(x, np.float64)
    q = np.clip((x - SHORE_X) / 340.0, 0, 1)
    return WL + 2 + 290 * q * q * (3 - 2 * q) + 40 * np.clip((x - SHORE_X) / (W - SHORE_X), 0, 1)


def _land_poly():
    """桂浜の陸（左）と海底。上の線＝陸の地面 → 波打ち際 → 海底。"""
    xs = np.linspace(0, SHORE_X, 40)
    ys = WL - 16 * np.clip(1 - (xs / SHORE_X) ** 2.2, 0, 1) - 6 * np.exp(-((xs - 40) / 50) ** 2)
    xb = np.linspace(SHORE_X, W + 4, 60)
    top = np.vstack([np.stack([xs, ys], 1), np.stack([xb, seabed_y(xb)], 1)])
    return np.vstack([[[0, H]], top, [[W + 4, H]]])


def fuji_outline():
    """富士山の輪郭（左のすそ → 頂上の平ら → 右のすそ）。高さは 3,776m を同じ縮尺で。"""
    hp = 3776 * PXM
    hw = np.linspace(0, 1, 60)
    half = 30 + 172 * (1 - hw) ** 1.5          # 高さ hw（0..1）での半分の幅（右の 1,000m・2,000m の数字にかからない細さ）
    ys = WL - hp * hw
    left = np.stack([FUJI_X - half, ys], 1)
    right = np.stack([FUJI_X + half, ys], 1)[::-1]
    # 頂上は少しだけでこぼこ
    top = np.array([[FUJI_X - 30, WL - hp], [FUJI_X - 13, WL - hp - 2.5], [FUJI_X + 2, WL - hp + 2],
                    [FUJI_X + 15, WL - hp - 1.5], [FUJI_X + 30, WL - hp]])
    return np.vstack([left, top[1:-1], right])


_C = {}


def background():
    """動かない部分：空と水・桂浜の陸・水面の線・1,000m と 2,000m のうすい目盛り線。"""
    if 'bg' in _C:
        return _C['bg']
    f = core.xs_background(WL)
    L = core.SS(s=3)
    d = L.draw
    # 目盛り線（点線）
    for m in (1000, 2000):
        y = y_of(m)
        d.line(L.p([(225, y), (1010, y)]), fill=L.col(core.FOAM, 0.11), width=L.w(1.2))
    # 水面
    d.line(L.p([(205, WL), (W, WL)]), fill=L.col(core.FOAM, 0.75), width=L.w(2.0))
    # 陸（桂浜）
    P = _land_poly()
    d.polygon(L.p([tuple(p) for p in P]), fill=L.col(core.SAND, 1.0))
    d.line(L.p([tuple(p) for p in P[1:41]]), fill=L.col(core.FOAM, 0.8), width=L.w(2.0))
    d.line(L.p([tuple(p) for p in P[40:-1]]), fill=L.col(core.FOAM, 0.35), width=L.w(1.6))
    lay = L.done()
    f = core.over(f, lay)
    # 地面の中：下ほど暗く、濃紺へ（灰色の板が画面の下を占めないように）
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    m = lay[..., 3:4] * (0.30 + 0.65 * np.clip((yy - WL) / 560.0, 0, 1) ** 0.8)
    f = f * (1 - m) + core.NAVY * m
    _C['bg'] = f.astype(np.float32)
    return _C['bg']


def labels():
    if 'lab' in _C:
        return _C['lab']
    lab = U.Patch([dict(text='1,000m', x=1010, y=y_of(1000) - 10, size=28, weight='Medium', anchor='rs', alpha=0.7),
                   dict(text='2,000m', x=1010, y=y_of(2000) - 10, size=28, weight='Medium', anchor='rs', alpha=0.7)])
    kat = U.Patch([dict(text='桂浜', x=X0 - 10, y=WL + 62, size=32, weight='Medium', track=0.1)])
    fuji = U.Patch([dict(text='富士山 3,776m', x=FUJI_X, y=y_of(3776) - 22, size=28, weight='Medium', alpha=0.85)])
    apex = U.Patch([dict(text='2,500m', x=1010, y=APEX[1] - 12, size=30, weight='Bold', anchor='rs')])
    _C['lab'] = dict(grid=lab, kat=kat, fuji=fuji, apex=apex)
    return _C['lab']


def fuji_layer(a):
    """富士山のうすい輪郭（塗りはごくうすく）。a：0..1（下から上へ線が描かれていく）。(rgba, x0, y0)。"""
    key = round(a, 2)
    if key in _C.get('fj', {}):
        return _C['fj'][key]
    P = fuji_outline()
    x0, y0 = int(P[:, 0].min()) - 8, int(P[:, 1].min()) - 8
    x1, y1 = min(W, int(P[:, 0].max()) + 8), int(WL) + 4
    L = core.SS(s=3, w=x1 - x0, h=y1 - y0)
    Q = P - (x0, y0)
    L.draw.polygon(L.p([tuple(p) for p in Q]), fill=L.col(core.FOAM, 0.08 * a))
    # 輪郭：すそから頂上へ向かって描かれる（左右同時）
    n = len(Q)
    k = int(np.ceil(n / 2 * core.ease(min(1, a * 1.4))))
    if k >= 2:
        for S_ in (Q[:k], Q[::-1][:k]):
            L.draw.line(L.p([tuple(p) for p in S_]), fill=L.col(core.FOAM, 0.5 * a), width=L.w(1.8), joint='curve')
    r = (L.done(), x0, y0)
    _C.setdefault('fj', {})[key] = r
    return r


def fuji_layer2(stroke, fill):
    """富士山（2026-11）：輪郭は stroke（0..1）ですそから頂上へ描かれ、塗りは fill（0..1）で遅れてうすく入る。(rgba, x0, y0)。"""
    key = (round(stroke, 2), round(fill, 2))
    cc = _C.setdefault('fj2', {})
    if key in cc:
        return cc[key]
    P = fuji_outline()
    x0, y0 = int(P[:, 0].min()) - 8, int(P[:, 1].min()) - 8
    x1, y1 = min(W, int(P[:, 0].max()) + 8), int(WL) + 4
    L = core.SS(s=3, w=x1 - x0, h=y1 - y0)
    Q = P - (x0, y0)
    if fill > 0.005:
        L.draw.polygon(L.p([tuple(p) for p in Q]), fill=L.col(core.FOAM, 0.08 * fill))
    n = len(Q)
    # 輪郭の長さに沿って描く（左右同時。先は小数の分だけ線を伸ばす）
    half = Q[:(n + 1) // 2 + 1]
    seg = np.hypot(*np.diff(half, axis=0).T)
    s_ = np.concatenate([[0], np.cumsum(seg)])
    lim = s_[-1] * stroke
    if lim > 1:
        for S_ in (Q, Q[::-1]):
            H_ = S_[:(n + 1) // 2 + 1]
            k = int(np.searchsorted(s_, lim))
            pts = list(H_[:k])
            if k < len(H_):
                t = (lim - s_[k - 1]) / max(seg[k - 1], 1e-6)
                pts.append(H_[k - 1] + (H_[k] - H_[k - 1]) * t)
            if len(pts) >= 2:
                L.draw.line(L.p([tuple(p) for p in pts]), fill=L.col(core.FOAM, 0.5), width=L.w(1.8), joint='curve')
    r = (L.done(), x0, y0)
    if len(cc) > 400:
        cc.clear()
    cc[key] = r
    return r


def trail_layer(u, alpha=1.0, ghost=0.0):
    """放物線：通ったところは白い実線、この先はうすい点線（ghost）。(rgba, x0, y0)。"""
    y0 = int(APEX[1]) - 12
    L = core.SS(s=2, w=W, h=int(WL) + 6 - y0)
    if ghost > 0:
        us = np.linspace(u, 1, 120)
        x, y = arc(us)
        A.dashed(L, np.stack([x, y - y0], 1), L.col(core.FOAM, 0.28 * ghost), L.w(1.6), dash=7, gap=9)
    if u > 0.002:
        us = np.linspace(0, u, max(4, int(u * 200)))
        x, y = arc(us)
        L.draw.line(L.p(list(zip(x, y - y0))), fill=L.col(core.FOAM, 0.85 * alpha), width=L.w(3.0), joint='curve')
    return L.done(), 0, y0


def tangent_ang(u):
    x0, y0 = arc(max(0, u - 0.01))
    x1, y1 = arc(min(1, u + 0.01))
    return float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
