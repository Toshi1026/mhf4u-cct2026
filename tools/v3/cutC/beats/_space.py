"""案C 08・09・10 の宇宙のカメラ（1 本の連続したカメラ）と地球の 1 枚。

カメラは C10 の頭からのフレーム k（C08・C09 は負）で決まる。どのビートのどのフレームから描いても同じ絵になる（状態を持たない）。
  C08（宇宙です）：斜め上からの地球（地球の丸み・薄い青い大気の層・星・太陽）。最初の 12 フレームでカメラが下へ振れて
       地球が下から上がってくる（C07 の上昇の続き）。そのあとはごくゆっくり寄る。
  C09（そして、5分半後）：同じ構図。最後の 14 フレームで悪役を追ってカメラが下を向きはじめる。
  C10：0→16 で真下を向き（北が上）、焦点距離 1100→2400、狙いの点（T の着水点＝MAZE の前の海の点）を画面の
       (520.5, 822.4) に固定したまま降りていく。拡大率 0.30→0.55 で球の写し方から平面（T の地図と同じ真上の写し方）へ
       寄せ、k=66（T の 0 フレーム目）で T の最初のカメラ（地図の px＝画面）にぴったり重なる。
       そのときの速さも T の最初の速さ（拡大の速さ 0、中心が北へ 20.6 px/フレーム）にそろえる。
"""
import math

import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W, H
from cutB.beats import _maplook as ML, _render as RR
from cutC.beats import _globe as GL

KM_PX = 1.0 / 4.6                       # 地図の 1px の km
P_T = np.array([520.50114644, 822.41328895])      # T の 0 フレーム目の着水点（画面＝地図の px）
V_BT = np.array([-4.10, -20.62])        # T の 0 フレーム目のカメラの中心の速さ（地図の px／フレーム）
F_SPACE, F_DIVE = 1100.0, 2400.0
K_NADIR = 16                            # ここで真下を向き終わる
K_END = 66                              # T の 0 フレーム目
H_END = F_DIVE * KM_PX                  # 拡大率 1（地図の px＝画面）になる高さ（km）
PL0, PL1 = 0.30, 0.55                   # 球 → 平面の拡大率の範囲

# 宇宙の構図（C08）：見る点 L（四国の北東）を、そこから方位 BEAR・地上の距離 D_KM・高さ H_SPACE のカメラで、
# 画面の y=PY_SPACE に置く（地球の縁は y≈790、上は宇宙＝文字と悪役の場所、日本は y≈1080）
L_SPACE = (33.7, 134.0)
D_KM, BEAR, H_SPACE, PY_SPACE = 1500.0, 140.0, 2500.0, 1180.0
PITCH_IN, N_IN = 22.0, 16           # C08 の頭：この角度だけ上を向いた所から N_IN フレームで下へ（cubic_out：速く振り始めて止まる）
H16 = 2200.0                            # 真下を向き終わる所の高さ（km）
_S = {}


def offsets():
    """(k08, k09)：C08・C09 の頭の k（C10 の頭が 0）。"""
    if 'off' not in _S:
        import timeline_c as TL
        bs = {b['key']: b for b in TL.load()}
        a10 = bs['10']['f0']
        _S['off'] = (bs['08']['f0'] - a10, bs['09']['f0'] - a10)
    return _S['off']


def k_of(beat, i):
    import timeline_c as TL
    if 'a10' not in _S:
        _S['a10'] = next(b for b in TL.load() if b['key'] == '10')['f0']
    return beat['f0'] + i - _S['a10']


def _ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def _lat_lon_lerp(a, b, t):
    A, B = GL.G.xyz(*a), GL.G.xyz(*b)
    P = GL.G.slerp(A, B, t)
    la, lo = GL.G.latlon(P)
    return float(la), float(lo)


def _pose_cam(G, h, tau, psi, f):
    e, n, u = GL.enu(*G)
    t, p = math.radians(tau), math.radians(psi)
    hd = math.cos(p) * n + math.sin(p) * e
    C = GL.ecef(G[0], G[1], h)
    F = math.cos(t) * (-u) + math.sin(t) * hd
    Uh = math.sin(t) * u + math.cos(t) * hd
    return GL.SCam(C, F, Uh, f)


def target_ll():
    if 'tll' not in _S:
        la, lo = ML.ll_of(P_T[0], P_T[1])
        _S['tll'] = (float(la), float(lo))
    return _S['tll']


def _pinned_ground(h, f):
    """真下を向いたカメラで、狙いの点が画面の P_T に来る地上の点（2 回の修正）。"""
    T = target_ll()
    G = T
    for _ in range(3):
        cam = _pose_cam(G, h, 0.0, 0.0, f)
        q = cam.proj(GL.ecef(*T)[None])[0]
        err = q - P_T                                   # 画面 px
        e, n, u = GL.enu(*G)
        s = h / f
        P = GL.ecef(*G) + (err[0] * s) * e - (err[1] * s) * n
        la, lo = GL.G.latlon(P)
        G = (float(la), float(lo))
    return G


# ---------------------------------------------------------------- 高さ・拡大率（C10）
def h_of(k):
    """K_NADIR〜K_END の高さ（km）。始まりの速さは真下へ振る間の降り方を受け継ぎ（約 −2%/フレーム）、K_END で 0。"""
    x = min(max((k - K_NADIR) / float(K_END - K_NADIR), 0.0), 1.0)
    L0, L1 = math.log(H16), math.log(H_END)
    v0 = 0.020 * (K_END - K_NADIR) / (L0 - L1)            # 正規化した始まりの速さ
    # エルミート：u(0)=0, u'(0)=v0, u(1)=1, u'(1)=0
    u = v0 * x + (3 - 2 * v0) * x * x + (v0 - 2) * x ** 3
    return math.exp(L0 + (L1 - L0) * u)


def zl_of(k):
    if k >= K_NADIR:
        return F_DIVE * KM_PX / h_of(k)
    c = cam(k)
    return float(c.f * KM_PX / max(np.linalg.norm(c.C) - GL.RE, 1.0))


# ---------------------------------------------------------------- カメラ
def _rot(v, axis, ang):
    axis = axis / np.linalg.norm(axis)
    return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * (axis @ v) * (1 - math.cos(ang))


def _space_base():
    if 'base' not in _S:
        e, n, u = GL.enu(*L_SPACE)
        b = math.radians(BEAR)
        P = GL.ecef(*L_SPACE) + D_KM * (math.sin(b) * e + math.cos(b) * n)
        la, lo = GL.G.latlon(P)
        C = GL.ecef(float(la), float(lo), H_SPACE)
        F = GL.ecef(*L_SPACE) - C
        F /= np.linalg.norm(F)
        _S['base'] = (C, F, C / np.linalg.norm(C))
    return _S['base']


def space_pitch(k):
    """C08・C09 のカメラの上下の振り（度、＋が上向き）。"""
    k08, k09 = offsets()
    p = 0.0
    q = (k - k08) / float(N_IN)                # C08 の頭：上を見ていた所から下へ。C07 の最後の重ね（k<k08）は同じ速さで続ける
    if q < 0:
        p += PITCH_IN * (1 - 3 * q)
    elif q < 1:
        p += PITCH_IN * (1 - q) ** 3
    qd = (k + 14) / 14.0                       # C09 の最後の 14 フレーム：悪役を追って下へ（だんだん速く）
    if qd > 0:
        p -= 8.0 * min(qd, 1.0) ** 2
    return p


def space_cam(k):
    k08, _ = offsets()
    C, F, up = _space_base()
    tt = min(max((k - k08) / float(-k08), 0.0), 1.0)
    # ゆっくりの寄り（C08・C09 を通して地球の中心へ 3%）
    C = C * (1 - 0.030 * tt * (np.linalg.norm(C) - GL.RE) / np.linalg.norm(C))
    Rv = np.cross(F, up)
    Rv /= np.linalg.norm(Rv)
    F2 = _rot(F, Rv, math.radians(space_pitch(k)))
    return GL.SCam(C, F2, up, F_SPACE, py=PY_SPACE)


def nadir_cam(k):
    h = h_of(k)
    G = _pinned_ground(h, F_DIVE)
    return _pose_cam(G, h, 0.0, 0.0, F_DIVE)


def cam(k):
    """フレーム k の宇宙のカメラ（SCam）。"""
    if k <= 0:
        return space_cam(k)
    if k >= K_NADIR:
        return nadir_cam(k)
    a, b = space_cam(0), nadir_cam(K_NADIR)
    x = k / float(K_NADIR)
    # 始まりの速さ：C09 の最後の下への振り（16°/14 フレーム）を受け継ぐ。終わりは真下で止まる
    ang = math.degrees(math.acos(np.clip(a.F @ b.F, -1, 1)))
    v0 = min((2 * 8.0 / 14.0) * K_NADIR / max(ang, 1e-3), 2.5)
    s = v0 * x + (3 - 2 * v0) * x * x + (v0 - 2) * x ** 3
    s = min(max(s, 0.0), 1.0)
    e = _ease(x)
    C = a.C * (1 - e) + b.C * e
    F = a.F * (1 - s) + b.F * s
    U = a.U * (1 - s) + b.U * s
    f = math.exp(math.log(a.f) * (1 - e) + math.log(b.f) * e)
    py = a.py * (1 - e) + b.py * e
    return GL.SCam(C, F, U - F * (U @ F) / (F @ F), f, py=py)


def planar_A(k):
    """C10 の平面のカメラ：画面 → 地図の px（2x3）と拡大率。T の 0 フレーム目（k=66）で恒等写像・同じ速さ。"""
    zl = zl_of(k)
    hm = 1000.0 / zl
    t = k - K_END
    c = np.array([W / 2, H / 2])
    Cxy = P_T - (P_T - c) * hm / 1000.0 + V_BT * t * math.exp(t / 8.0)
    s = hm / 1000.0
    A = np.array([[s, 0, Cxy[0] - c[0] * s], [0, s, Cxy[1] - c[1] * s]], np.float64)
    return A, zl


def target_screen(k):
    """狙いの点（着水点）の画面の位置。"""
    if k >= K_NADIR:
        A, zl = planar_A(k)
        Ai = np.linalg.inv(np.vstack([A, [0, 0, 1]]))
        p = Ai @ np.array([P_T[0], P_T[1], 1.0])
        return p[:2]
    c = cam(k)
    return c.proj(GL.ecef(*target_ll())[None])[0]


# ---------------------------------------------------------------- 太陽
def sun_dir():
    """太陽の方向（世界）：C09 の構図で画面の左上（x≈300, y≈170。CG ラベルの上）に来る向き。"""
    if 'sun' not in _S:
        c = space_cam(offsets()[1])
        x, y = 300.0, 170.0
        d = c.F + (x - c.px) / c.f * c.R - (y - c.py) / c.f * c.U
        _S['sun'] = d / np.linalg.norm(d)
    return _S['sun']


# ---------------------------------------------------------------- 1 枚
def earth(k, star_a=1.0):
    """フレーム k の地球（文字・悪役なし、仕上げ前の sRGB）。戻り値：(f, info)。info：zl, water, plan_w, cam。"""
    zl = zl_of(k)
    pw = float(RR.smoothstep(PL0, PL1, zl)) if k >= K_NADIR else 0.0
    gk = 1 - float(RR.smoothstep(0.08, 0.25, zl))          # 球の明暗・つや：引きだけ
    ck = 1 - float(RR.smoothstep(0.06, 0.16, zl))          # 雲：引きだけ
    hk = 1 - float(RR.smoothstep(0.25, 0.70, zl))          # 大気のかすみ：寄ると 0（T の地図と同じ）
    if pw >= 0.999:
        A, zl = planar_A(k)
        rgb, wa = GL.planar_frame(A, zl)
        info = dict(zl=zl, water=wa, plan_w=1.0, A=A)
    else:
        c = cam(k)
        A = None
        if pw > 0:
            A, _ = planar_A(k)
        sk = 1.0 - float(RR.smoothstep(PITCH_IN * 0.25, PITCH_IN * 1.1, space_pitch(k))) * 0.9 if k <= 0 else 1.0
        rgb, wa, cov = GL.globe_frame(c, zl=zl, haze_k=hk, star_a=star_a, sun=sun_dir(), light=sun_dir(), sun_k=sk,
                                      cloud_k=ck, globe_k=gk, planar=A, planar_w=pw)
        info = dict(zl=zl, water=wa, plan_w=pw, A=A, cam=c, cov=cov)
    return rgb, info


def map_post(f, k, info, i_fin):
    """T の地図と同じ仕上げ（平面の重みで入る）：海のつや・上の空気遠近・ティルトシフト・海岸線 → 暗がり → finish_ad。"""
    pw = info['plan_w']
    if pw > 0:
        g = ML.sea_and_air(f, info['water'], k, haze=0.16, tilt=1.2)
        if info.get('A') is not None:
            cm = GL.coast_mask(info['A'], info['zl'])
            g = g + (np.asarray(core.FOAM, np.float32) - g) * cm[..., None]
        f = f * (1 - pw) + g * pw
    f = ML.label_scrims(f, 1.0, cg=True, credit_a=1.0)
    return RR.finish_ad(f, i_fin, grain=0.007, vignette=0.10, bloom_k=0.05)
