"""案B の 03（桂浜から10km）・04（メーターの背景）・T（着水のしぶき）で共有する、土佐湾の線画まわり。

座標は案A と同じ「地図の px」（beats/_tosa.to_screen：4.6 px/km、b11 の 0 フレーム目＝画面と一致）。
- 海岸線：Natural Earth 50m を _tosa と同じく Chaikin で丸めたベクトル（03 は寄るので、マスクではなく線で描く）
- 沖の線：b11 と同じ等距離線（_tosa.STEPS、うすく・少なく）
- 桂浜：33.4971N 133.5745E を地図の海岸線へ寄せた点（NE 50m は 1km ほどずれるため）
- 着水点：MAZE の前の海（33.455N 133.478E。MAZE＝33.4590N 133.4775E の約 0.45km 南、浜から約 0.3km 沖）を
  地図の海岸線へ寄せ、海の側へ 0.3km 出した点。桂浜の点からの距離は約 10.1km（試算の R＝10.10km）。
  b11 が降りていく本当の仁淀川の河口（b11._mouth）は、この点の約 0.45km 東（地名は 08 まで出さない）。
  2026-11-07 訂正：以前は 33.457N 133.440E（MAZE の約 3.5km 西）を行き先にしていた（13km 版）。
カメラは真上から。screen = (P − ctr)·z + ANCHOR。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401  （`beats` を案A に固定）
import core
from core import W, H
from beats import _geo as G, _tosa, b10, b11
from beats import _mapui as U

KAT_LL = (33.4971, 133.5745)
DEST_LL = (33.455, 133.478)               # 試算の行き先＝MAZE の前の海（2026-11-07 訂正。旧 33.457N 133.440E は 3.5km 西）
DEST_OFF_KM = 0.30                         # 地図の海岸線から沖へ（実際の行き先も浜から約 0.3km）
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
    d0, dt = _snap(b11._world(*DEST_LL), [np.asarray(P, np.float64) for P in raw])
    dn = np.array([dt[1], -dt[0]])
    if dn[1] < 0:
        dn = -dn
    dest = d0 + dn * DEST_OFF_KM * _tosa.PX_KM            # MAZE の前の海
    offs = [(k, P) for k, P in b11._offsets()]
    _S.update(rings=rings, kat=kat, dest=dest, M=M, nrm=nrm, offs=offs, bg=b10._background(),
              km=float(np.hypot(*(dest - kat)) / _tosa.PX_KM))
    return _S


# ---------------------------------------------------------------- カメラ（03 の 0 フレーム目からの通しの時刻 t）
Z0 = 1.30
ARC_FRAC = 0.62                       # 寄りきったとき、放物線（桂浜→着水点）の横幅を画面の幅のこの割合に（2026-11-07：小さく見えたので 0.24→0.62）


def n03():
    """03 のフレーム数（04 の背景はこの続きの時刻から）。"""
    import timeline_b
    return next(b for b in timeline_b.load() if b['key'] == '03')['n']


def z1():
    S = setup()
    return ARC_FRAC * W / abs(float(S['dest'][0] - S['kat'][0]))


CAM_N = 84                            # 寄りの長さ（フレーム）
CAM_EASE = (0.65, 0.0, 0.35, 1.0)     # 寄りの曲線（cubic-bezier）
# 寄りのあとも止めない、弧の中心まわりのごくゆっくりした押し込み（ln の速さ。03 の 74→111 で 1.00→1.045）
DRIFT_R = 0.001225
DRIFT_IN = (56, 84)                   # 押し込みがなめらかに始まる区間（寄りの減速と重ねて、ズームの速さが 0 にならない）
DRIFT_DECAY = (111, 25.0, 0.35)       # 04 に入ってから（t0, tau, 残す割合）：メーターの後ろでは押し込みを 0.35 倍まで緩める
PAN_V = np.array([1.2, 0.4])          # 押し込みと同じ曲線で、弧の中心の画面の位置もゆっくり流す（px/フレーム。飛ぶ向き＝左へカメラが追う）


def _target():
    S = setup()
    ch = S['dest'] - S['kat']
    return (S['kat'] + S['dest']) / 2 + np.array([0.0, -0.125 * float(np.hypot(*ch))])   # 起こした弧の高さの半分だけ上


def drift(t):
    """押し込みの量（ln）。t は 03 の 0 フレーム目からの通しの時刻。"""
    r = DRIFT_R
    a, b = DRIFT_IN
    n = b - a
    if t <= a:
        return 0.0
    if t < b:
        u = (t - a) / n
        return r * n * (u ** 3 - u ** 4 / 2)
    d = r * n * 0.5
    t0, tau, keep = DRIFT_DECAY
    if t < t0:
        return d + r * (t - b)
    d += r * (t0 - b)
    k = t - t0
    return d + r * (keep * k + (1 - keep) * tau * (1 - np.exp(-k / tau)))


def pan(t):
    return PAN_V * (drift(t) / DRIFT_R)


def cam(t, extra=1.0):
    """(ctr, z)。0→84 フレームで、弧の中心（_target）を画面の ANCHOR へ運びながら寄る（log z と、目標点の画面の位置を
    同じ曲線で動かす＝目標点がまっすぐ ANCHOR へ向かう）。そのあとは弧の中心まわりのごくゆっくりした押し込み（drift）と
    横への流れ（pan）が続く。04 の背景も同じ関数で続ける。extra：弧の中心まわりの追加の拡大（04 の奥行き）。"""
    from cutB.beats import _motion as M
    S = setup()
    q = _target()
    c0 = np.array([560.0, 905.0])                          # はじめは土佐湾全体（室戸〜足摺）が入る引き
    Z1 = z1()
    k = M.cubic_bezier(*CAM_EASE)(min(max(t / float(CAM_N), 0.0), 1.0))
    z = Z0 * (Z1 / Z0) ** k * float(np.exp(drift(t))) * extra
    s0 = (q - c0) * Z0 + ANCHOR                            # 目標点のはじめの画面の位置
    s = s0 + (ANCHOR - s0) * k + pan(t)
    ctr = q - (s - ANCHOR) / z
    return ctr, z


def target_scr(t, extra=1.0):
    ctr, z = cam(t, extra)
    return to_scr(_target()[None], ctr, z)[0]


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


def draw_track(L, ctr, z, u, alpha=1.0, head=True, dash_phase=0.0, head_r=5.0, tail=0.0):
    """L：G.Lines。起こした放物線（白い破線）。地面の線は海岸線とほぼ重なるので描かない。
    dash_phase：破線の模様を飛ぶ向きへ流す量（px）。head：先頭の点（半径 head_r）。
    tail：先頭のうしろの彗星の尾（弧の全長に対する割合。根元 0→先頭 0.9、太さ 1→3px）。"""
    if u <= 0.002:
        return
    g, a = arc_pts(ctr, z, u, n=160)
    segs = G.dashed(a, dash=13, gap=8, phase=dash_phase)
    tot = max(1, len(segs) - 1)
    for j, seg in enumerate(segs):
        k = j / tot
        L.poly(seg, 3.0, 255 * alpha * (0.97 - 0.22 * k * u))    # 先端に向けて少しだけうすく（行き先を強く指さない）
    if tail > 0 and u > 0.01:
        s0 = max(0.0, u - tail)
        _, tl = arc_pts(ctr, z, u, n=160)
        ss = np.linspace(0, u, len(tl))
        sel = ss >= s0
        P = tl[sel]
        m = len(P)
        for j in range(m - 1):
            q = (j + 1) / float(m - 1)
            L.poly(P[j:j + 2], 1.0 + 2.0 * q, 255 * alpha * 0.9 * q)
    if head and head_r > 0.2:
        L.dot(a[-1, 0], a[-1, 1], head_r, 255 * alpha)


def kat_scr(ctr, z):
    return to_scr(setup()['kat'][None], ctr, z)[0]


# ---------------------------------------------------------------- 縮尺
def scale_bar(f, z, alpha=1.0, km=None, x1=1000.0, y=372.0):
    """縮尺。km=None のときは寄りに合わせて 5km → 2km に切り替える（右端をそろえて 8 フレームほどで溶かす。棒が長くなりすぎないように）。"""
    if alpha <= 0.001:
        return f
    if km is None:
        L5 = 5 * _tosa.PX_KM * z
        w2 = float(np.clip((L5 - 230.0) / 70.0, 0, 1))
        w2 = w2 * w2 * (3 - 2 * w2)
        if w2 < 1:
            f = scale_bar(f, z, alpha * (1 - w2), 5, x1, y)
        if w2 > 0:
            f = scale_bar(f, z, alpha * w2, 2, x1, y)
        return f
    Lb = G.Lines()
    ln = km * _tosa.PX_KM * z
    x0 = x1 - ln
    Lb.poly(np.array([[x0, y], [x1, y]]), 2.0, 255 * 0.85 * alpha)
    for x in (x0, x1):
        Lb.poly(np.array([[x, y - 9], [x, y + 1]]), 2.0, 255 * 0.85 * alpha)
    m2 = Lb.done()
    from cutB.beats import _maplook as ML
    f = ML.drop_shadow(f, m2, 1.5, 2.0, 2.0, 0.40 * alpha)
    m = m2[..., None]
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
