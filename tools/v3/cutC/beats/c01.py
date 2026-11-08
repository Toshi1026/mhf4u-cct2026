"""案C 01（0.0–2.0 秒）「アニメなら、キラーン。」  テロップ「アニメだと」（大・最初のフレームから）。

アニメ版（平面的なマンガの絵柄・太い輪郭はここだけ）：明るい空・マンガの雲・ミントの草原。
顔のない白い悪役（青いマント）が両手を腰に当てて胸を張る。左下から拳（ヒーローは拳だけ）が速さの線つきで入り、
f4 でポコッと当たる（拳も悪役もつぶれる 2 フレームのヒットストップ・星形の「ポコッ」・集中線・小さな星）。
悪役は右上へくるくる 3 回転しながら小さくなって空の点になり、「キ」でキラーン（_art.put_glint_fx を大きく）。
そのあと拳はぽよんと戻って小さくガッツポーズのように 2 回はずむ。最後まで星がまたたく。
最初のフレームから：大テロップ「アニメだと」（アニメ＝強調）・拳は画面の中で振り抜いている途中・速さの線。
カメラ：キラーンの点を中心に 1.00→1.05 のゆっくりした押し込み＋当たりの揺れ。テロップは画面に固定（当たりで小さくポップ）。
02 はこのビートの最後の絵（world(n-1)）を止めて使う。
"""
import math

import cv2
import numpy as np

import core
from core import W, H
import timeline_c as timeline
from cutB.beats import _art as A, _motion as M
from cutC.beats import _toon as T

FEET = (606.0, 1212.0)
VS = 10.0                                      # 悪役の大きさ（全長 約 320px）
TORSO = (FEET[0], FEET[1] - 16 * VS)
DIR = -24.0                                    # 拳の向き（度）
DV = np.array([math.cos(math.radians(DIR)), math.sin(math.radians(DIR))])
FS = 1.62                                      # 拳の大きさ
CONTACT = (TORSO[0] - 38.0, TORSO[1] + 8.0)    # 拳が当たる所（胴の左）
HITC = np.array(CONTACT) - 54 * FS * DV        # 当たったときの拳のまんなか
HIT, HOLD = 4, 2
STAR = (842.0, 716.0)                          # キラーンの点
BEZ_C = (1010.0, 900.0)
TEL_Y = 600                                    # テロップのベースライン
PUSH = 1.09


def timing(beat):
    mo = beat['moras']
    assert mo[5][0] == 'キ', mo
    ki = timeline.mora_frame(beat, 5)
    return dict(ki=ki, glint=ki - 5)            # 光の一番大きいフレーム（glint+5）が「キ」


# ---------------------------------------------------------------- 拳
def fist_s(i):
    """HITC からの打つ向きの距離（負＝手前）。f0 は振り抜きの途中（画面の中）。"""
    if i < HIT:
        q = (i + 2) / float(HIT + 2)
        return -330.0 * (1 - q ** 1.6)
    if i < HIT + HOLD:
        return 6.0
    if i < HIT + HOLD + 3:
        return 6 + 22 * M.ease((i - HIT - HOLD) / 3.0, 'quad_out')
    if i < HIT + HOLD + 12:                    # ぽよんと引く（back_out）
        q = (i - HIT - HOLD - 3) / 9.0
        return 28 + (-300 - 28) * M.ease(q, 'back_out')
    return -300.0


def fist_bob(i):
    """引いたあとの小さなはずみ（2 回、ガッツポーズのように上へ）と、ゆっくりした息。"""
    b = 3.0 * math.sin(2 * math.pi * 1.1 * i / core.FPS)
    for f0 in (30, 39):
        k = i - f0
        if 0 <= k < 9:
            b += -26 * math.sin(math.pi * k / 9.0) ** 1.5
    return b


def fist_pose(i):
    s = fist_s(i)
    p = HITC + s * DV
    p = p + np.array([0.0, fist_bob(i) if i >= HIT + HOLD + 12 else 0.0])
    sq = (1.0, 1.0)
    if HIT <= i < HIT + HOLD:
        sq = (0.82, 1.14)                      # 当たりでつぶれる
    elif HIT + HOLD <= i < HIT + HOLD + 4:
        k = (i - HIT - HOLD) / 4.0
        sp = M.spring(k * 0.25, 0.35, 5.0)
        sq = (0.82 + 0.18 * sp, 1.14 - 0.14 * sp)
    for f0 in (30, 39):                        # はずむときの伸び縮み
        k = i - f0
        if 0 <= k < 9:
            e = math.sin(math.pi * k / 9.0)
            sq = (1 - 0.06 * e, 1 + 0.08 * e)
    return p, sq


# ---------------------------------------------------------------- 悪役
def flight(i, T_):
    """飛ぶ進み（f HIT+HOLD → glint）。(x, y, scale, ang)。"""
    f0, f1 = HIT + HOLD, T_['glint']
    tau = float(np.clip((i - f0) / float(f1 - f0), 0, 1))
    tp = 1 - (1 - tau) ** 1.8
    p = (1 - tp) ** 2 * np.array(TORSO) + 2 * tp * (1 - tp) * np.array(BEZ_C) + tp ** 2 * np.array(STAR)
    sc = VS * math.exp(math.log(0.22 / VS) * tau ** 1.7)
    ang = -14 + 1080 * M.ease(tau, 'cubic_out')
    return float(p[0]), float(p[1]), sc, ang, tau


def put_villain(f, i, T_, t):
    if i < HIT:
        # 胸を張って弾む（足もとを中心に）＋最後に拳に気づいて少しのけぞる
        ph = math.sin(2 * math.pi * 2.0 * t)
        sy = 1.02 + 0.02 * ph
        lean = 6.0 * M.ease((i - 1) / 3.0, 'back_out')
        M2 = A._R(lean) @ np.diag([1 / math.sqrt(sy), sy])
        c = np.array(FEET) + M2 @ (np.array(TORSO) - FEET)
        f = shadow(f, 1.0)
        rgba, x0, y0 = T.villain_patch(c[0], c[1], VS, 0.0, 'stand', t, M2=M2)
        return T.paste(f, rgba, x0, y0)
    if i < HIT + HOLD:
        M2 = M.squash_matrix(DIR, 0.80, 1.15) @ A._R(10.0)
        c = np.array(FEET) + M2 @ (np.array(TORSO) - FEET) + DV * 10
        f = shadow(f, 1.0)
        rgba, x0, y0 = T.villain_patch(c[0], c[1], VS, 0.0, 'hit', t, M2=M2)
        return T.paste(f, rgba, x0, y0)
    x, y, sc, ang, tau = flight(i, T_)
    if i < HIT + HOLD + 4:
        f = shadow(f, 1 - (i - HIT - HOLD) / 4.0)
    if sc * 32 < 9:
        return A.put_dot(f, x, y, max(1.8, sc * 7))
    # 飛び出しで速さの向きに伸びて戻る（ばね）
    k = (i - HIT - HOLD) / core.FPS
    va = math.degrees(math.atan2(STAR[1] - TORSO[1], STAR[0] - TORSO[0]))
    s_al = 1.35 + (1.0 - 1.35) * M.spring(k, 0.45, 5.0)
    M2 = M.squash_matrix(va, s_al, 1 / math.sqrt(s_al))
    rgba, x0, y0 = T.villain_patch(x, y, sc, ang, 'fly', t, M2=M2, cape_ang=va + 180 + 12 * math.sin(t * 13))
    return T.paste(f, rgba, x0, y0)


def shadow(f, a):
    """足もとの平らな影（濃紺 0.22）。"""
    if a <= 0.01:
        return f
    m = np.zeros((60, 260), np.float32)
    cv2.ellipse(m, (130 * 4, 30 * 4), (92 * 4, 15 * 4), 0, 0, 360, 1.0, -1, cv2.LINE_AA, shift=2)
    x0, y0 = int(FEET[0] - 130 + 14), int(FEET[1] - 30 + 4)
    reg = f[y0:y0 + 60, x0:x0 + 260]
    mm = m[..., None] * 0.22 * a
    f[y0:y0 + 60, x0:x0 + 260] = reg * (1 - mm) + T.OUT * mm
    return f


def swirl_trail(f, i, T_):
    """くるくる飛ぶ跡：小さくなっていく白い点線の弧＋悪役のまわりの回転の線（マンガ）。"""
    f0, f1 = HIT + HOLD, T_['glint']
    if i < f0 + 1 or i > f1 + 10:
        return f
    a_all = 1.0 if i <= f1 else 1 - (i - f1) / 10.0
    tau_now = float(np.clip((i - f0) / float(f1 - f0), 0, 1))
    L = core.SS(s=2)
    d = L.draw
    n = 26
    for k in range(n):
        tq = tau_now * k / n
        tp = 1 - (1 - tq) ** 1.8
        p = (1 - tp) ** 2 * np.array(TORSO) + 2 * tp * (1 - tp) * np.array(BEZ_C) + tp ** 2 * np.array(STAR)
        sc = VS * math.exp(math.log(0.22 / VS) * tq ** 1.7)
        # 進む向きに直角のゆれ（らせんの跡）
        th = 1080 * M.ease(tq, 'cubic_out') * math.pi / 180
        p = p + np.array([math.cos(th), math.sin(th)]) * sc * 3.0
        r = 1.2 + 0.55 * sc
        a = a_all * 0.85 * (0.3 + 0.7 * k / n)
        d.ellipse(L.box(p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=L.col(core.FOAM, a))
    return T.paste(f, L.done(), 0, 0)


# ---------------------------------------------------------------- 当たりの効果
_FX = {}


def fx_assets():
    if not _FX:
        _FX['burst'] = T.burst(58, spikes=10, rot=8.0)
        _FX['star'] = T.mini_star(20)
        _FX['focus'] = T.focus_lines(CONTACT, 380, n=40, alpha=0.7)
        _FX['poko'] = T.outlined_text('ポコッ', 84, T.CLOUD, 7, rot=-12, shadow=(6, 0.5))
    return _FX


def put_impact(f, i, layer='back'):
    fx = fx_assets()
    k = i - HIT
    if k < 0 or k > 22:
        return f
    if layer == 'front':
        return put_impact_front(f, k, fx)
    # 集中線（f4–12、外から 0.8→0）
    if k < 9:
        a = 0.75 * (1 - k / 9.0) ** 1.3
        f = T.paste(f, fx['focus'], 0, 0, alpha=a)
    return f


def put_burst(f, k, fx):
    # 星形（ぽんと出てしぼむ。拳と悪役の上、当たる所に 6 フレームだけ）
    if k < 6:
        s = (0.55 + 0.6 * M.ease((k + 1) / 3.0, 'back_out')) * (1 - 0.6 * M.ease((k - 3) / 5.0, 'cubic_in'))
        a = 1.0 if k < 4 else 1 - (k - 3) / 3.0
        rgba, R = fx['burst']
        sp = cv2.resize(rgba, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        f = T.paste(f, sp, CONTACT[0] + 6 - R * s, CONTACT[1] - 8 - R * s, alpha=a)
    return f


def put_impact_front(f, k, fx):
    f = put_burst(f, k, fx)
    # 小さな星（3 つ、外へ飛んで回る）
    if k < 14:
        rgba, R = fx['star']
        for j, (ang, sp_) in enumerate(((-150, 15.0), (-95, 17.0), (-35, 13.0))):
            q = (k + 1) / 14.0
            dist = 60 + sp_ * k * (1 - 0.35 * q)
            x = CONTACT[0] + math.cos(math.radians(ang)) * dist
            y = CONTACT[1] + math.sin(math.radians(ang)) * dist + 0.6 * k * k
            s = 1.0 - 0.5 * q
            rr = cv2.resize(rgba, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            M_ = cv2.getRotationMatrix2D((rr.shape[1] / 2, rr.shape[0] / 2), -k * 25 * (1 if j % 2 else -1), 1.0)
            rr = cv2.warpAffine(rr, M_, (rr.shape[1], rr.shape[0]), flags=cv2.INTER_LINEAR)
            f = T.paste(f, rr, x - rr.shape[1] / 2, y - rr.shape[0] / 2, alpha=1 - q ** 2)
    # 「ポコッ」（左上に、ぽんと出て 18 から消える）
    rgba, (bx, by) = fx['poko']
    s = 0.6 + 0.4 * M.ease((k + 1) / 4.0, 'back_out') if k < 4 else 1.0 + 0.02 * math.sin(k * 0.8)
    a = 1.0 if k < 16 else max(0.0, 1 - (k - 15) / 6.0)
    sp = cv2.resize(rgba, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    px, py = 300.0, 905.0 - 3 * min(k, 16) * 0.3
    f = T.paste(f, sp, px - bx * s, py - by * s, alpha=a)
    return f


def speed_lines(f, i):
    """振り抜く拳のうしろの白い速さの線（f0–HIT）。"""
    if i >= HIT:
        return f
    p, _ = fist_pose(i)
    L = core.SS(s=2)
    d = L.draw
    n = np.array([-DV[1], DV[0]])
    back = -DV
    for off, ln, wd in ((-58, 300, 9), (-20, 380, 12), (22, 340, 10), (60, 260, 8), (95, 200, 6)):
        o = p + back * (110 * FS) + n * off * FS
        e = o + back * ln
        d.polygon(L.p([tuple(o + n * wd / 2), tuple(o - n * wd / 2), tuple(e)]), fill=L.col(core.FOAM, 0.9))
    return T.paste(f, L.done(), 0, 0)


# ---------------------------------------------------------------- 1 フレーム
_TX = {}


def telop():
    if 't' not in _TX:
        fills = [core.ACC] * 3 + [T.CLOUD] * 2
        _TX['t'] = T.outlined_text('アニメだと', 150, fills, 11, weight='Black', track=0.03, shadow=(10, 0.55), rot=-3.0)
    return _TX['t']


def put_telop(f, i):
    rgba, (bx, by) = telop()
    k = i - HIT
    s = 1.0
    if 0 <= k < 8:                              # 当たりで小さくポップ（1.0→1.05→1.0）
        s = 1 + 0.05 * math.sin(math.pi * k / 8.0)
    if abs(s - 1) > 1e-3:
        Mm = np.float32([[s, 0, bx * (1 - s)], [0, s, by * (1 - s)]])
        rgba = cv2.warpAffine(rgba, Mm, (rgba.shape[1], rgba.shape[0]), flags=cv2.INTER_LINEAR)
    return T.paste(f, rgba, W / 2 - bx, TEL_Y - by)


def cam(i, n):
    q = i / float(max(n - 1, 1))
    k = 1 + (PUSH - 1) * (0.35 * q + 0.65 * M.ease(q, 'sine_inout'))
    sh = M.shake(i, HIT, amp=11.0, tau=2.5, rot_deg=0.5, seed=21)
    c, s = math.cos(math.radians(sh[2])) * k, math.sin(math.radians(sh[2])) * k
    cx, cy = STAR
    return np.float32([[c, -s, cx - c * cx + s * cy + sh[0]], [s, c, cy - s * cx - c * cy + sh[1]]])


def world(i, beat, T_=None):
    """i フレーム目の絵（カメラのあと・テロップの前）。02 が最後のフレームを止めて使う。"""
    T_ = T_ or timing(beat)
    n = beat['n']
    t = i / core.FPS
    f = T.bg_static().copy()
    for rgba, x0, y0, sp in T.clouds():
        f = T.paste_f(f, rgba, x0 + sp * i, y0)
    f = swirl_trail(f, i, T_)
    f = put_impact(f, i, 'back')
    g = T_['glint']
    if i < g:
        f = put_villain(f, i, T_, t)
    else:
        tw = 0.8 + 0.2 * math.cos((i - g) * 0.45)
        f = A.put_dot(f, STAR[0], STAR[1], 2.6, alpha=tw)
    f = speed_lines(f, i)
    p, sq = fist_pose(i)
    rgba, x0, y0 = T.fist_patch(p[0], p[1], DIR, FS, sq)
    f = T.paste(f, rgba, x0, y0)
    f = put_impact(f, i, 'front')
    # キラーン（大きめ）＋そのあとの小さなまたたき（明るさは 1 回だけ上がって下がる）
    f = A.put_glint_fx(f, STAR[0], STAR[1], i - g, r=78.0, rot=24.0)
    k = i - g - 24
    if k >= 0:
        a = 0.55 * (0.5 - 0.5 * math.cos(2 * math.pi * k / 14.0))
        f = A.put_glint(f, STAR[0], STAR[1], 16.0 + 4 * a, ang=12.0 + k * 1.5, alpha=a)
    f = T.paste(f, T.fg_grass(i), 0, 0)
    f = cv2.warpAffine(f, cam(i, n), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return np.clip(f, 0, 1)


def render(beat, ctx):
    n = beat['n']
    T_ = timing(beat)
    for i in range(n):
        f = world(i, beat, T_)
        f = put_telop(f, i)
        yield np.clip(f, 0, 1).astype(np.float32)
