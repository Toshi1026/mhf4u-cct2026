"""案B の 03（桂浜から13km）・04（メーターの背景）・T（着水のしぶき）で共有する、土佐湾の線画まわり。

座標は案A と同じ「地図の px」（beats/_tosa.to_screen：4.6 px/km、b11 の 0 フレーム目＝画面と一致）。
- 海岸線：Natural Earth 50m を _tosa と同じく Chaikin で丸めたベクトル（03 は寄るので、マスクではなく線で描く）
- 沖の線：b11 と同じ等距離線（_tosa.STEPS、うすく・少なく）
- 桂浜：33.4971N 133.5745E を地図の海岸線へ寄せた点（NE 50m は 1km ほどずれるため）
- 着水点：b11 が降りていく河口（b11._mouth）の、海の側へ約 0.65km（地名は 08 まで出さない）
カメラは真上から。screen = (P − ctr)·z + ANCHOR。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401  （`beats` を案A に固定）
import core
from core import W, H
from beats import _geo as G, _tosa, b10, b11
from beats import _mapui as U

KAT_LL = (33.4971, 133.5745)
DEST_LL = (33.457, 133.440)               # 試算の行き先（新居海岸の沖）。地図では河口の前の海の点を使う
ANCHOR = np.array([540.0, 760.0])         # 画面のこの点に ctr を置く

_S = {}


def _snap(q, rings):
    best = (1e18, None, None)
    for P in rings:
        A, B = P[:-1], P[1:]
        AB = B - A
        L2 = (AB ** 2).sum(1) + 1e-12
        t = (((q - A) * AB).sum(1) / L2).clip(0, 1)
        C = A + AB * t[:, None]
        d = ((C - q) ** 2).sum(1)
        k = int(np.argmin(d))
        if d[k] < best[0]:
            best = (d[k], C[k], AB[k] / np.sqrt(L2[k]))
    return best[1], best[2]


def setup():
    if _S:
        return _S
    rings = []
    for P in b11._coast_rings():
        far = (P[:, 0] < -1500) | (P[:, 0] > 2600) | (P[:, 1] < -1500) | (P[:, 1] > 3400)
        if far.all():
            continue
        Q = _tosa._chaikin(np.asarray(P, np.float64), 3)
        Q = np.concatenate([Q, Q[:1]])
        rings.append(Q)
    kat, _ = _snap(b11._world(*KAT_LL), rings)
    raw = b11._coast_rings()
    M, tang = b11._mouth(raw)
    nrm = np.array([tang[1], -tang[0]])
    if nrm[1] < 0:
        nrm = -nrm
    dest = M + nrm * 3.0                                   # 河口の前の海（約 0.65km 沖）
    offs = [(k, P) for k, P in b11._offsets()]
    _S.update(rings=rings, kat=kat, dest=dest, M=M, nrm=nrm, offs=offs, bg=b10._background(),
              km=float(np.hypot(*(dest - kat)) / _tosa.PX_KM))
    return _S


# ---------------------------------------------------------------- カメラ（03 の 0 フレーム目からの通しの時刻 t）
N03 = 116
Z0, Z1 = 1.30, 4.80


def cam(t):
    """(ctr, z)。0→約 80 フレームで寄り、その後はごくゆっくり流れ続ける（04 の背景も同じ関数で続ける）。"""
    S = setup()
    q = (S['kat'] + S['dest']) / 2 + np.array([0.0, -8.0])
    c0 = np.array([560.0, 905.0])                          # はじめは土佐湾全体（室戸〜足摺）が入る引き
    k = core.ease(t / 84.0, 'inout')
    z = Z0 * (Z1 / Z0) ** k + 0.0011 * max(0.0, t - 84)
    ctr = c0 + (q - c0) * k + np.array([-0.020, 0.006]) * max(0.0, t - 84)
    return ctr, z


def to_scr(P, ctr, z):
    return (np.asarray(P, np.float64) - ctr) * z + ANCHOR


# ---------------------------------------------------------------- 地図の線
def map_mask(t, coast_a=0.78, off_a=0.26):
    ctr, z = cam(t)
    S = setup()
    L = G.Lines()
    for k, P in S['offs']:
        L.poly(to_scr(P, ctr, z), 1.25, 255 * off_a * (0.85 - 0.12 * k))
    for P in S['rings']:
        Q = to_scr(P, ctr, z)
        inside = (Q[:, 0] > -200) & (Q[:, 0] < W + 200) & (Q[:, 1] > -200) & (Q[:, 1] < H + 200)
        if not inside.any():
            continue
        L.poly(Q, 1.9, 255 * coast_a)
    return L.done()


LAND_TINT = core.c('0F2D4B')                      # 陸のごく弱い塗り（海と見分けるため。光らせない）


def land_mask(t):
    """陸の塗りのマスク（等倍・AA）。海岸線の線と同じ丸めた輪。"""
    import cv2
    ctr, z = cam(t)
    m = np.zeros((H, W), np.uint8)
    polys = []
    for P in setup()['rings']:
        Q = to_scr(P, ctr, z)
        Q = np.clip(Q, -4000, 4000)
        polys.append(np.round(Q * 16).astype(np.int32))
    cv2.fillPoly(m, polys, 255, lineType=cv2.LINE_AA, shift=4)
    return m.astype(np.float32) / 255


# ---------------------------------------------------------------- 打ち出しの軌跡
def arc_pts(ctr, z, u=1.0, n=90, lift=1.0):
    """桂浜→着水点の放物線（真上から見た地面の線＋高さを画面の上方向へ少し起こした線）。u は 0..1 の進み。
    戻り値：(ground, arc) の画面座標（u までの部分）。"""
    S = setup()
    s = np.linspace(0, u, max(2, int(n * u) + 2))
    g = S['kat'][None] + (S['dest'] - S['kat'])[None] * s[:, None]
    G2 = to_scr(g, ctr, z)
    length = np.hypot(*(S['dest'] - S['kat'])) * z
    h = 4 * s * (1 - s) * length * 0.25 * lift            # 最高点＝距離の1/4（45度で打ち出したときの h = R/4）
    A = G2.copy()
    A[:, 1] -= h
    return G2, A


def draw_track(L, ctr, z, u, alpha=1.0, head=True, dash_phase=0.0):
    """L：G.Lines。起こした放物線（白い破線）。地面の線は海岸線とほぼ重なるので描かない。
    先頭の小さな点は進んでいるあいだだけ。"""
    if u <= 0.002:
        return
    g, a = arc_pts(ctr, z, u, n=160)
    segs = G.dashed(a, dash=13, gap=8, phase=dash_phase)
    tot = max(1, len(segs) - 1)
    for j, seg in enumerate(segs):
        k = j / tot
        L.poly(seg, 3.0, 255 * alpha * (0.97 - 0.22 * k * u))    # 先端に向けて少しだけうすく（行き先を強く指さない）
    if head and u < 0.999:
        L.dot(a[-1, 0], a[-1, 1], 5.0, 255 * alpha)


def kat_scr(ctr, z):
    return to_scr(setup()['kat'][None], ctr, z)[0]


# ---------------------------------------------------------------- 縮尺
def scale_bar(f, z, alpha=1.0, km=5, x1=1000.0, y=372.0):
    if alpha <= 0.001:
        return f
    Lb = G.Lines()
    ln = km * _tosa.PX_KM * z
    x0 = x1 - ln
    Lb.poly(np.array([[x0, y], [x1, y]]), 2.0, 255 * 0.85 * alpha)
    for x in (x0, x1):
        Lb.poly(np.array([[x, y - 9], [x, y + 1]]), 2.0, 255 * 0.85 * alpha)
    m = Lb.done()[..., None]
    f = f * (1 - m) + core.FOAM * m
    lab = _patch(('%dkm' % km, 30))
    return lab.put(f, alpha * 0.9, dx=(x0 + x1) / 2 - 540, dy=y - 16 - 1000)


_P = {}


def _patch(key):
    if key not in _P:
        text, size = key
        _P[key] = U.Patch([dict(text=text, x=540, y=1000, size=size, weight='Medium', track=0.02)])
    return _P[key]


# ---------------------------------------------------------------- 強調つきのカウンター
class HCounter(U.Counter):
    """U.Counter の後ろ（suffix）を「強調する部分（ACC）＋残り（白）」に分ける版。例：suffix='km先へ', hl_suffix='km'。"""

    def __init__(self, final, cx, y, hl_suffix='', **kw):
        super().__init__(final, cx, y, **kw)
        self.hl_suffix = hl_suffix

    def items(self, s, alpha=1.0):
        out = super().items(s, alpha)
        if not self.suffix or not self.hl_suffix:
            return out
        suf = out[0]
        assert suf['text'] == self.suffix
        a, b = self.hl_suffix, self.suffix[len(self.hl_suffix):]
        wa = core.text_width(a, self.ss, self.side_weight, 0.02) + 0.02 * self.ss
        it1 = dict(suf, text=a, color=core.ACC)
        out[0] = it1
        if b:
            out.insert(1, dict(suf, text=b, x=suf['x'] + wa))
        return out
