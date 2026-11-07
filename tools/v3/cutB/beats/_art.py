"""案B の CG の部品：オリジナルの悪役ピクトグラム、画面の端から入る拳、キラーンの光、小さな板（スプライト）の貼り付け。

悪役は core.surfer と同じ描き方（白の太い線＋丸い頭、顔なし）に、小さなマントだけを足したもの。
既存のキャラクターに似せない：触角・角・特定の色や衣装はなし。拳は白い線画のマンガの拳（丸い指の山・親指・短い袖口。腕は描かない）。
"""
import cv2
import numpy as np

import core
from core import W, H

SHADOW = core.c('062A4E')
CAPE = core.c('BFDCF0')            # マント：白に近い淡い青（体の白と見分ける）


def _R(ang_deg):
    r = np.deg2rad(ang_deg)
    return np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])


# ---------------------------------------------------------------- 小さな板に描いて貼る
def put(frame, draw_fn, x, y, R, s=4, alpha=1.0, shadow=0.0, shadow_px=2.0, glow=0.0, glow_px=6.0, glow_rgb=None):
    """(x, y) を中心に 2R×2R の小さな SS 板を作り、draw_fn(L, cx, cy) で描いて frame に重ねる（サブピクセルの位置に対応）。
    shadow：ごく弱い暗いにじみ（明るい空でも白が読めるように）。glow：同じ形をぼかした白いにじみ（キラーン用）。"""
    if alpha <= 0.003:
        return frame
    R = int(np.ceil(R))
    L = core.SS(s=s, w=2 * R, h=2 * R)
    fx, fy = x - round(x), y - round(y)
    draw_fn(L, R + fx, R + fy)
    lay = L.done()
    x0, y0 = int(round(x)) - R, int(round(y)) - R
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + 2 * R, frame.shape[1]), min(y0 + 2 * R, frame.shape[0])
    if xb <= xa or yb <= ya:
        return frame
    sub = lay[ya - y0:yb - y0, xa - x0:xb - x0]
    reg = frame[ya:yb, xa:xb]
    if shadow:
        sa = cv2.GaussianBlur(np.ascontiguousarray(sub[..., 3]), (0, 0), shadow_px)[..., None] * alpha * shadow
        reg = reg * (1 - sa) + SHADOW * sa
    if glow:
        ga = cv2.GaussianBlur(np.ascontiguousarray(sub[..., 3]), (0, 0), glow_px)[..., None] * alpha * glow
        g = core.FOAM if glow_rgb is None else glow_rgb
        reg = reg + (g - reg) * np.clip(ga, 0, 1)
    a = sub[..., 3:4] * alpha
    frame[ya:yb, xa:xb] = reg * (1 - a) + sub[..., :3] * a
    return frame


# ---------------------------------------------------------------- 悪役（顔のない白いピクトグラム＋小さなマント）
def villain(L, x, y, scale=1.0, ang=0.0, pose='stand', t=0.0, rgb=None, alpha=1.0, squash=1.0, M2=None,
            cape_ang=None):
    """(x, y) は胴のまんなか。scale=1 で全長約 32px（足 +16、頭のてっぺん -16）。ang は度（時計回り）。
    pose='stand'：両手を腰に当てて胸を張る（悪役のお約束のポーズ）／'fly'：手足を広げて飛ばされる。
    t：マントのはためき（秒）。squash：殴られた瞬間の横のつぶれ（1＝なし）。
    M2（任意の 2×2、画面の向き）：回したあとに (x, y) を中心にかける変形（つぶれ・伸び。_motion.squash_matrix）。
    cape_ang（任意、度、0＝右・90＝下）：'fly' のマントを体の回転と関係なく、画面のこの向きへたなびかせる。
    どちらも None なら今までと同じ絵。"""
    rgb = core.FOAM if rgb is None else rgb
    col = L.col(rgb, alpha)
    ccol = L.col(CAPE, alpha * 0.92)               # マントは少しだけ透かして、体と見分けられるように
    Rm = _R(ang)
    S = np.array([[squash, 0], [0, 1 / max(squash, 0.6) ** 0.5]])
    if M2 is not None:
        Rm = np.asarray(M2, float) @ Rm

    def T(pts):
        return L.p([tuple(Rm @ (S @ np.array(q, float)) * scale + (x, y)) for q in pts])

    d = L.draw
    lw = lambda v: L.w(v * scale)
    fl = np.sin(t * 9.0) * 1.2 + np.sin(t * 15.0 + 1.0) * 0.6        # はためき
    if pose == 'stand':
        # マント：肩から右後ろへ、ひざの高さまで（短め）
        d.polygon(T([(-2.4, -8.0), (2.4, -8.0), (8.0 + fl * 0.3, 0.0), (13.0 + fl, 10.5),
                     (9.0 + fl * 0.7, 9.4), (5.5 + fl * 0.5, 11.6), (1.5, 10.6), (-3.5, 11.2), (-7.0, 10.0),
                     (-6.0, 1.0)]), fill=ccol)
        d.line(T([(-4.2, 16), (0, 4), (4.2, 16)]), fill=col, width=lw(2.7), joint='curve')         # 脚
        d.line(T([(0, 4.5), (0, -7.5)]), fill=col, width=lw(3.4))                                 # 胴
        d.line(T([(0, -6.0), (-6.2, -1.2), (-1.4, 2.6)]), fill=col, width=lw(2.2), joint='curve')  # 腰に手
        d.line(T([(0, -6.0), (6.2, -1.2), (1.4, 2.6)]), fill=col, width=lw(2.2), joint='curve')
        hx, hy = Rm @ (S @ np.array([0.0, -12.2])) * scale + (x, y)
    else:
        # マント：体の下側へ長くたなびく
        cape = [(-2.6, -7.6), (2.6, -7.6), (8.0 + fl, 9.0), (6.0 + fl, 18.0), (2.0 + fl * 0.6, 15.5),
                (-2.0 + fl * 0.4, 19.0), (-6.5 + fl * 0.3, 12.0)]
        if cape_ang is None:
            d.polygon(T(cape), fill=ccol)
        else:
            # 肩の点は体について回り、マントの向きだけ画面の cape_ang へ（体の +y＝マントの軸）
            sh = Rm @ (S @ np.array([0.0, -7.6])) * scale + (x, y)
            Rc = (np.asarray(M2, float) if M2 is not None else np.eye(2)) @ _R(cape_ang - 90.0)
            d.polygon(L.p([tuple(sh + Rc @ (np.array(q) - (0.0, -7.6)) * scale) for q in cape]), fill=ccol)
        d.line(T([(-7.0, 14.5), (0, 4), (6.5, 15.0)]), fill=col, width=lw(2.7), joint='curve')
        d.line(T([(0, 4.5), (0, -7.5)]), fill=col, width=lw(3.4))
        d.line(T([(-9.5, -12.5), (0, -6.0), (9.5, -12.0)]), fill=col, width=lw(2.2), joint='curve')
        hx, hy = Rm @ (S @ np.array([0.0, -12.2])) * scale + (x, y)
    rr = 3.7 * scale
    if M2 is None:
        d.ellipse(L.box(hx - rr, hy - rr, hx + rr, hy + rr), fill=col)
    else:                                         # つぶれ・伸びに合わせて頭も楕円に
        th = np.linspace(0, 2 * np.pi, 28, endpoint=False)
        P = (np.asarray(M2, float) @ np.stack([np.cos(th), np.sin(th)]) * rr).T + (hx, hy)
        d.polygon(L.p([tuple(p) for p in P]), fill=col)


def villain_rgba(x, y, scale, ang=0.0, pose='fly', t=0.0, squash=1.0, M2=None, cape_ang=None, s=4):
    """悪役だけの小さな板 (rgba, x0, y0)（ストレートアルファ）。_motion.subframe / put_rgba で使う。"""
    ext = 1.0 if M2 is None else max(1.0, float(np.abs(np.linalg.eigvals(np.asarray(M2, float))).max()))
    R = int(np.ceil((24 * scale + 6) * ext + 4))
    L = core.SS(s=s, w=2 * R, h=2 * R)
    fx, fy = x - np.floor(x), y - np.floor(y)
    villain(L, R + fx, R + fy, scale, ang, pose, t, alpha=1.0, squash=squash, M2=M2, cape_ang=cape_ang)
    return L.done(), int(np.floor(x)) - R, int(np.floor(y)) - R


def put_villain(frame, x, y, scale, ang=0.0, pose='stand', t=0.0, alpha=1.0, squash=1.0, shadow=0.35, M2=None,
                cape_ang=None):
    if scale * 34 < 5:                          # ここまで小さいと形は読めない：点にする
        return put_dot(frame, x, y, max(1.6, scale * 8), alpha)
    ext = 1.0 if M2 is None else max(1.0, float(np.abs(np.linalg.eigvals(np.asarray(M2, float))).max()))
    R = (24 * scale + 6) * ext
    return put(frame, lambda L, cx, cy: villain(L, cx, cy, scale, ang, pose, t, alpha=1.0, squash=squash, M2=M2,
                                                cape_ang=cape_ang),
               x, y, R, alpha=alpha, shadow=shadow, shadow_px=max(1.0, 1.1 * scale))


def put_dot(frame, x, y, r, alpha=1.0):
    def fn(L, cx, cy):
        L.draw.ellipse(L.box(cx - r, cy - r, cx + r, cy + r), fill=L.col(core.FOAM))
    return put(frame, fn, x, y, r + 6, alpha=alpha, glow=0.35, glow_px=r * 1.2)


# ---------------------------------------------------------------- 拳（白い線画のマンガの拳。腕は描かず、袖口で終わる）
FIST_FILL = core.c('E4F0F9')
FIST_SHADE = core.c('B9D3E8')       # 袖口（少しだけ濃く）


def _superellipse(a, b, n=2.6, k=72, cx=0.0, cy=0.0):
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return np.stack([cx + a * np.sign(c) * np.abs(c) ** (2 / n), cy + b * np.sign(s) * np.abs(s) ** (2 / n)], 1)


def _ellipse(a, b, cx, cy, k=48):
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return np.stack([cx + a * np.cos(t), cy + b * np.sin(t)], 1)


_FIST = None


def _capsule(p0, p1, r, k=20):
    """p0→p1 の太さ 2r の丸い棒（閉じた輪）。"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    a = np.arctan2(d[1], d[0])
    t1 = np.linspace(a - np.pi / 2, a + np.pi / 2, k)
    t0 = np.linspace(a + np.pi / 2, a + 3 * np.pi / 2, k)
    return np.vstack([p1 + r * np.stack([np.cos(t1), np.sin(t1)], 1), p0 + r * np.stack([np.cos(t0), np.sin(t0)], 1)])


def _fist_shapes():
    """マンガの拳の形（ローカル座標、+x が殴る向き、y は下が +、単位 px）。描く順＝重なりの順（後のものが前）。
    横から見たグー：手の甲（丸い四角）、前に丸く盛り上がった4本の指（関節の山がはっきり）、
    その下を横切る親指（爪つき）、手首に短く広がる袖口。腕は描かない（2026-11-07：棒の先のミトンに見えたため）。"""
    global _FIST
    if _FIST is None:
        sh = {}
        sh['cuff'] = np.array([(-46, -42), (-46, 44), (-78, 58), (-84, 54), (-84, -52), (-78, -56)], float)
        sh['body'] = _superellipse(46, 50, 3.2, 120, cx=-14, cy=1)      # 手の甲
        for j, yy in enumerate((-37.5, -12.5, 12.5, 37.5)):              # 曲げた4本の指（前に丸い関節の山）
            sh['f%d' % j] = _capsule((6, yy), (48 - 3.5 * abs(j - 1.5), yy), 13.4)
        sh['thumb'] = _capsule((-32, 31), (22, 20), 13.0)                # 下の2本を横切る親指
        _FIST = sh
    return _FIST


def fist(L, x, y, ang, scale=1.0, alpha=1.0, ox=0.0, oy=0.0):
    """(x, y)：拳のまんなか（画面座標）、ang：殴る向き（度、0＝右、負＝上向き）。L の原点は画面の (ox, oy)。
    各部品を「白い太い輪郭 → 淡い青の塗り」の順に重ねる（前の部品の輪郭が後ろの部品の上に残り、指の境目になる）。"""
    sh = _fist_shapes()
    Rm = _R(ang)

    def T(P):
        P = np.asarray(P, float) @ Rm.T * scale + (x - ox, y - oy)
        return L.p([tuple(q) for q in P])

    d = L.draw
    lw = 4.4 * scale
    line = L.col(core.FOAM, alpha)
    fill = L.col(FIST_FILL, alpha)
    shade = L.col(FIST_SHADE, alpha)
    for k, P in sh.items():
        d.polygon(T(P), fill=line)
        d.line(T(np.vstack([P, P[:1]])), fill=line, width=L.w(2 * lw), joint='curve')
        d.polygon(T(P), fill=shade if k == 'cuff' else fill)
    det = L.w(lw * 0.75)
    # 親指の爪
    nail = np.array([(18, 13.5), (27, 14.5), (30, 20), (27, 25.5), (19, 26)], float)
    d.line(T(nail), fill=line, width=det, joint='curve')
    # 袖口の折り返しの線
    d.line(T([(-58, -48), (-58, 50)]), fill=line, width=det)


def put_fist(frame, x, y, ang, scale=1.0, alpha=1.0):
    """拳を画面に重ねる。腕は画面の外へ伸びるので、画面と交わる範囲だけの板を作る。"""
    if alpha <= 0.003:
        return frame
    Rm = _R(ang)
    pts = np.vstack([v for v in _fist_shapes().values()])
    pts = np.clip(pts, -900, None)
    P = pts @ Rm.T * scale + (x, y)
    x0, y0 = int(max(0, np.floor(P[:, 0].min() - 14))), int(max(0, np.floor(P[:, 1].min() - 14)))
    x1, y1 = int(min(W, np.ceil(P[:, 0].max() + 14))), int(min(H, np.ceil(P[:, 1].max() + 14)))
    if x1 <= x0 or y1 <= y0:
        return frame
    L = core.SS(s=3, w=x1 - x0, h=y1 - y0)
    fist(L, x, y, ang, scale, 1.0, ox=x0, oy=y0)
    lay = L.done()
    reg = frame[y0:y1, x0:x1]
    sa = cv2.GaussianBlur(np.ascontiguousarray(lay[..., 3]), (0, 0), 6)[..., None] * 0.30 * alpha
    reg = reg * (1 - sa) + SHADOW * sa
    a = lay[..., 3:4] * alpha
    frame[y0:y1, x0:x1] = reg * (1 - a) + lay[..., :3] * a
    return frame


# ---------------------------------------------------------------- キラーン（小さな4つ角の星。1回だけ、やわらかく）
def glint(L, cx, cy, r, ang=0.0, alpha=1.0):
    d = L.draw
    Rm = _R(ang)

    def star(rad, waist, col):
        pts = []
        for k in range(8):
            th = np.pi / 4 * k
            rr = rad if k % 2 == 0 else rad * waist
            pts.append(Rm @ np.array([np.cos(th), np.sin(th)]) * rr + (cx, cy))
        d.polygon(L.p([tuple(p) for p in pts]), fill=col)

    # 斜めの小さな星（ACC）→ 縦横の星（ACC）→ 白い芯
    Rd = _R(ang + 45)
    pts = []
    for k in range(8):
        th = np.pi / 4 * k
        rr = r * 0.42 if k % 2 == 0 else r * 0.06
        pts.append(Rd @ np.array([np.cos(th), np.sin(th)]) * rr + (cx, cy))
    d.polygon(L.p([tuple(p) for p in pts]), fill=L.col(core.ACC, alpha * 0.7))
    star(r, 0.11, L.col(core.ACC, alpha))
    star(r * 0.55, 0.16, L.col(core.FOAM, alpha))
    rc = max(1.2, r * 0.09)
    d.ellipse(L.box(cx - rc, cy - rc, cx + rc, cy + rc), fill=L.col(core.FOAM, alpha))


def put_glint(frame, x, y, r, ang=0.0, alpha=1.0):
    if r < 0.8 or alpha <= 0.003:
        return frame
    return put(frame, lambda L, cx, cy: glint(L, cx, cy, r, ang), x, y, r * 1.6 + 8, alpha=alpha,
               glow=0.45, glow_px=max(2.0, r * 0.22), glow_rgb=core.ACC)


def glint_env(k, rise=6, hold=3, decay=16.0):
    """光の大きさ（0..1）。k は光りはじめからのフレーム数。すっと出て、ゆっくり消える（点滅しない）。"""
    if k < 0:
        return 0.0
    if k < rise:
        return core.ease((k + 1) / rise, 'out')
    if k < rise + hold:
        return 1.0
    return float(np.exp(-(k - rise - hold) / decay))


# ---------------------------------------------------------------- キラーン 2026-11（動きの改善版）
GLINT_PEAK = 5          # 光りはじめから一番大きくなるまでのフレーム（効果音の頭をここに合わせる）


def glint_env2(k):
    """光の大きさ。back_out（行き過ぎ 1.15）で 9 フレームかけて出て k=5 で最大、3 フレームそのまま、τ=14 で消える。"""
    if k < 0:
        return 0.0
    if k <= 9:
        sb = 2.2                                  # 行き過ぎ 1.15 の back_out（最大は q≈0.54）
        u = k / 9.0 - 1
        return float(max(0.0, 1 + (sb + 1) * u ** 3 + sb * u ** 2))
    if k < 12:
        return 1.0
    return float(np.exp(-(k - 12) / 14.0))


def put_glint_fx(frame, x, y, k, r=46.0, alpha=1.0, rot=18.0):
    """キラーン（k＝光りはじめからのフレーム）：主の星（0→rot 度回る）＋小さな星 2 つ（遅れて出て 8 フレームで消える）
    ＋横の細い光の筋（6 フレーム）。点滅しない（明るさは 1 回だけ上がって下がる）。"""
    g = glint_env2(k)
    if g <= 0.01 and k > 20:
        return frame
    life = float(np.clip(k / 40.0, 0, 1))
    ang = rot * (1 - (1 - life) ** 2)
    # 横の光の筋（主の星の後ろ）
    if 0 <= k < 6:
        q = k / 6.0
        sa = alpha * (1 - q) ** 1.5 * min(1.0, (k + 1) / 2.0)
        ln = 2.2 * r * g * (0.7 + 0.3 * q)

        def streak(L, cx, cy):
            L.draw.line(L.p([(cx - ln, cy), (cx + ln, cy)]), fill=L.col(core.FOAM, 0.9), width=L.w(1.5))
        frame = put(frame, streak, x, y, ln + 6, s=3, alpha=sa)
    if g > 0.01:
        frame = put_glint(frame, x, y, r * g, ang=ang, alpha=alpha * min(1.0, g * 1.4))
    for (ox, oy, dl) in ((22, -14, 3), (-22, -14, 5)):
        kk = k - dl
        if 0 <= kk < 8:
            q = kk / 8.0
            gm = (1 - q) ** 1.4 * min(1.0, (kk + 1) / 2.0)
            frame = put_glint(frame, x + ox * r / 46.0, y + oy * r / 46.0, r * 0.3 * (0.6 + 0.4 * min(1, kk / 2.0)),
                              ang=ang * 1.5, alpha=alpha * gm)
    return frame


# ---------------------------------------------------------------- 線（点線・折れ線）
def dashed(L, P, col, width, dash=10.0, gap=8.0, phase=0.0):
    """折れ線 P（画面座標の (N,2)）を点線で。L は SS（原点は画面の左上）。"""
    P = np.asarray(P, float)
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    tot = s[-1]
    a = -phase % (dash + gap) - (dash + gap)
    while a < tot:
        b0, b1 = max(a, 0), min(a + dash, tot)
        if b1 > b0:
            xs = np.interp(np.linspace(b0, b1, max(2, int((b1 - b0) / 4) + 2)), s, P[:, 0])
            ys = np.interp(np.linspace(b0, b1, max(2, int((b1 - b0) / 4) + 2)), s, P[:, 1])
            L.draw.line(L.p(list(zip(xs, ys))), fill=col, width=width, joint='curve')
        a += dash + gap
