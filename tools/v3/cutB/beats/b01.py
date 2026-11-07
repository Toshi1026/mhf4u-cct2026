"""B01（0.0–3.3秒）「アニメの悪役は、パンチ一発で、空のかなたへ。」
朝の空と低い水平線、手前の浜。顔のない白い悪役（小さなマント、両手を腰に当てて胸を張る）。
「パンチ」で画面の左下の端から白い線画の拳が入り、「チ」で当たる（短い白い線が数本、すぐ消える）。
悪役は右上へ弧を描いて飛び、回りながら小さくなって（うしろに速さの線）点になり、「カナタ」でキラーンと1回光る。
全体はゆっくり DOT へ押し込む（02 がそのまま続ける）。文字：「パンチ一発で」（パ）、「空のかなたへ」（ソ）。"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W, H
from beats import _mapui as U
from cutB.beats import _art as A, _sky as S

FS = 1.15
DIR = -27.0                                  # 拳の向き（度）
CONTACT = (392.0, 1342.0)                    # 拳が当たる所（悪役の胴の左）
BEZ_C = (1000.0, 900.0)                      # 飛んでいく弧の制御点


def bez(t, p0, c, p2):
    t = np.asarray(t, np.float64)[..., None]
    return (1 - t) ** 2 * np.asarray(p0) + 2 * t * (1 - t) * np.asarray(c) + t ** 2 * np.asarray(p2)


def torso():
    return (S.FEET[0], S.FEET[1] - 16 * S.VSCALE)


def timing(beat):
    mf = lambda k: timeline.mora_frame(beat, k)
    mo = beat['moras']
    assert mo[9][0] == 'パ' and mo[11][0] == 'チ' and mo[17][0] == 'ソ' and mo[20][0] == 'カ', mo
    return dict(pa=mf(9), hit=mf(11) - 1, so=mf(17), glint=mf(20) + 3)


def flight(i, T):
    """飛んでいる悪役の (x, y, scale, ang, 進み具合 tp)。"""
    tau = np.clip((i - T['hit']) / float(T['glint'] - T['hit']), 0, 1)
    tp = 1 - (1 - tau) ** 2.2
    x, y = bez(tp, torso(), BEZ_C, S.DOT)
    sc = S.VSCALE * np.exp(np.log(0.12 / S.VSCALE) * tau ** 1.25)
    return float(x), float(y), float(sc), -30 + 820 * tp, float(tp)


def trail(f, tp, tau, alpha):
    """飛んだ跡のうすい弧（太さは手前ほど太い）と、悪役のうしろの速さの線。"""
    if alpha <= 0.003 or tp <= 0.002:
        return f
    ts = np.linspace(0, tp, 60)
    P = bez(ts, torso(), BEZ_C, S.DOT)
    x0, y0 = int(max(0, P[:, 0].min() - 40)), int(max(0, P[:, 1].min() - 40))
    x1, y1 = int(min(W, P[:, 0].max() + 40)), int(min(H, P[:, 1].max() + 40))
    L = core.SS(s=2, w=x1 - x0, h=y1 - y0)
    Q = P - (x0, y0)
    for k in range(len(Q) - 1):
        u = ts[k] / max(tp, 1e-6)
        wd = 1.2 + 4.0 * (1 - ts[k]) ** 1.5
        a = alpha * 0.30 * (0.25 + 0.75 * u)
        L.draw.line(L.p([tuple(Q[k]), tuple(Q[k + 1])]), fill=L.col(core.FOAM, a), width=L.w(wd))
    # 速さの線：悪役のすぐうしろに3本
    if tp < 0.97:
        p1 = bez(tp, torso(), BEZ_C, S.DOT)
        p0 = bez(max(0, tp - 0.06), torso(), BEZ_C, S.DOT)
        v = p1 - p0
        sp = np.hypot(*v)
        if sp > 1:
            v /= sp
            nrm = np.array([-v[1], v[0]])
            sc = S.VSCALE * np.exp(np.log(0.12 / S.VSCALE) * tau ** 1.25)
            ln = min(260, sp * 3.2)
            for off, k in ((-1, 0.7), (0, 1.0), (1, 0.7)):
                a0 = p1 - v * (14 * sc) + nrm * off * 9 * sc
                a1 = a0 - v * ln * k
                L.draw.line(L.p([tuple(a0 - (x0, y0)), tuple(a1 - (x0, y0))]),
                            fill=L.col(core.FOAM, alpha * 0.55 * (1 - tp) ** 0.5), width=L.w(max(1.0, 1.6 * sc / 2)))
    lay = L.done()
    reg = f[y0:y1, x0:x1]
    a = lay[..., 3:4]
    f[y0:y1, x0:x1] = reg * (1 - a) + lay[..., :3] * a
    return f


def impact(f, k):
    """当たった瞬間の短い白い線（6フレームで消える）。光らせない。"""
    if k < 0 or k >= 6:
        return f
    cx, cy = CONTACT
    a = (1 - k / 6.0) ** 1.3
    R = 150

    def fn(L, ox, oy):
        back = np.deg2rad(DIR + 180)
        for j in range(9):
            th = back + np.deg2rad(55 + j * 31)
            r0 = 34 + 16 * k
            r1 = r0 + 30 + 6 * (j % 2)
            p0 = (ox + r0 * np.cos(th), oy + r0 * np.sin(th))
            p1 = (ox + r1 * np.cos(th), oy + r1 * np.sin(th))
            L.draw.line(L.p([p0, p1]), fill=L.col(core.FOAM), width=L.w(4.2 - 0.4 * k))
    return A.put(f, fn, cx, cy, R, s=3, alpha=a * 0.9)


def texts():
    return (U.Patch([dict(text='パンチ一発で', x=W / 2, y=470, size=48, weight='Medium', track=0.10)]),
            U.Patch([dict(text='空のかなたへ', x=W / 2, y=600, size=100, weight='Bold', hl='空のかなたへ')]))


def fist_pos(i, T):
    """拳のまんなか（画面の外から → 当たる → 少し押し込む → 引く）。"""
    d = np.array([np.cos(np.deg2rad(DIR)), np.sin(np.deg2rad(DIR))])
    hitc = np.array(CONTACT) - 68 * FS * d
    s0 = T['hit'] - 10
    if i < s0:
        return None
    if i <= T['hit']:
        p = core.ease((i - s0) / 10.0, 'in')
        return hitc - d * 780 * (1 - p)
    if i <= T['hit'] + 5:
        return hitc + d * 26 * core.ease((i - T['hit']) / 5.0, 'out')
    q = core.ease((i - T['hit'] - 5) / 14.0, 'in')
    if q >= 1:
        return None
    return hitc + d * 26 - d * 820 * q


def render(beat, ctx):
    n = beat['n']
    T = timing(beat)
    t1, t2 = texts()
    for i in range(n):
        t = i / core.FPS
        f = S.frame(i)
        # 悪役
        if i < T['hit']:
            bob = -2.2 * abs(np.sin(t * 6.5))                    # 胸を張って笑う（小さく弾む）
            x, y = torso()
            f = A.put_villain(f, x, y + bob, S.VSCALE, ang=0, pose='stand', t=t)
        elif i <= T['hit'] + 1:
            x, y = torso()
            f = A.put_villain(f, x + 6, y, S.VSCALE, ang=8, pose='stand', t=t, squash=0.82)
        else:
            x, y, sc, ang, tp = flight(i, T)
            f = trail(f, tp, (i - T['hit']) / float(T['glint'] - T['hit']), 1 - core.ease((i - T['glint'] - 2) / 14.0))
            if i < T['glint']:
                f = A.put_villain(f, x, y, sc, ang=ang, pose='fly', t=t)
            else:
                tw = 0.85 + 0.15 * np.cos((i - T['glint']) * 0.35)
                f = A.put_dot(f, S.DOT[0], S.DOT[1], 2.3, alpha=tw)
        # 拳
        p = fist_pos(i, T)
        if p is not None:
            f = A.put_fist(f, p[0], p[1], DIR, FS)
        f = impact(f, i - T['hit'])
        # キラーン
        g = A.glint_env(i - T['glint'])
        if g > 0.01:
            f = A.put_glint(f, S.DOT[0], S.DOT[1], 46 * g, ang=10 * (1 - g), alpha=min(1.0, g * 1.4))
        # 押し込み（DOT を中心に）
        k = core.lerp(*S.PUSH01, core.ease(i / (n - 1), 'smooth'))
        f = S.push(f, k)
        # 文字
        for tl, f0 in ((t1, T['pa']), (t2, T['so'])):
            if i >= f0 - 1:
                a = core.ease((i - f0 + 2) / 6.0)
                f = tl.put(f, a, dy=10 * (1 - a))
        yield np.clip(f, 0, 1).astype(np.float32)
