"""B01（0.0–3.3秒）「アニメの悪役は、パンチ一発で、空のかなたへ。」
朝の空と低い水平線、手前の浜。顔のない白い悪役（小さなマント、両手を腰に当てて胸を張る）。
「パンチ」で画面の左下の端から白い線画のマンガの拳（丸い指の山・親指・短い袖口。腕は描かず、うしろに速さの線）が入り、「チ」で当たる。
悪役は右上へ弧を描いて飛び、回りながら小さくなって（うしろに速さの線）点になり、「カナタ」でキラーンと1回光る。
文字：「パンチ一発で」（パ）、「空のかなたへ」（ソ）。

2026-11 動きの改善（director's motion spec §1-01）：
- 世界のレイヤー（空・海・浜・悪役・拳・跡）にカメラ（_motion.Cam2D）。はじめは 240px 上へずらし（足 y≈1190、当たる所 y≈1100：
  セーフゾーンの中）、0–30 で 1.10→1.00 に引いて見せる。当たり+3 からばね（ζ0.8・2.2Hz）で悪役を追って上へ振り、キラーンまでに 0 へ
  （DOT は 02 と同じ画面の位置）。世界は下に 300px 足した背景で描く。文字と CG ラベルは画面に固定。
- 01・02 を通した DOT 中心の押し込み（_sky.push_k、sine_in）。
- 殴る流れ：胸を張って弾む（つぶれと伸び）→ 37–42 で拳に気づいてのけぞる → 33–39 拳が 130px 顔を出す → 39–43 少し引く →
  43–46 打ち込む（サブフレームのぶれ）→ 46–48 ヒットストップ（打つ向きにつぶれる・衝撃の星・放射の線・火花）＋カメラの揺れ →
  49– 打ち出し（速さの向きに伸びてばねで戻る、足もとの砂ぼこり）→ 拳は少し押し込んでから back_in で去る。
- 飛ぶ悪役：回転は expo_out で 720°、マントは画面の向きで速さの反対へ（2 フレーム遅れ）、速いときはサブフレームのぶれ。
- キラーン：_art.put_glint_fx（行き過ぎ 1.15、回転、小さな星 2 つ、横の筋）。一番大きいのは glint+5（効果音の頭）。
- 文字：_motion.KText の 1 文字ずつの立ち上がり（「空のかなたへ」は強調のポップつき）。
2026-11-08 見た目（美術監督の指示。時間・動き・文字はそのまま）：
- 舞台は _beach（空 #2F6FB0→#BFE0F5、体積の積雲＝主役 3 つは左上・見出しから 60px 以上、遠い雲 5 つ、筋雲 2 本、海、のこぎりの風紋の砂、ぬれた帯）。
- 悪役は _figures3d.villain_v2（スレートのサテン・短いスカーフ・右上のリム）、拳は fist_v2（3D）。砂へ右下に倒れる影（#1B3550 を掛ける）。
- 当たり：平らな星・破片・全周の線をやめ、打つ向き ±35° の先の細い光の筋＋足もとの砂の粒（光と影つき）＋低い砂ぼこり。
- 飛ぶ悪役は 1 回描いて速さの向きにぶらす（サブフレームの代わり）。ヒットストップの形は 1 回だけ描く。
- カメラのあと：下 20% を 12% 暗く σ2 でぼかす、ヒットストップ中は水平線から上を σ1.5。仕上げは _render.finish_ad。文字は最後。
2026-11-08 r2（美術監督の 2 回目。時間・動きはそのまま）：
- 拳は fist_v3（4 つの拳頭の平らな打つ面・下へ折った中節としわ・人差し指と中指の中節を横切る親指・しわ 2 本の無地の袖、
  キーは右上・冷たいリム）。袖のうしろは速さの中へ消える。拳の下の砂に環境の遮へい。
- 当たり：灰色の汚れのような粒をやめ、SandKick3D（50 粒が上へ・奥へ飛んで約 8 フレームで落ちる、ぶらした筋）。
  ヒットストップの中で悪役は 0.85/1.12 を 2 フレーム → 0.92/1.06。光らせない。
- 飛ぶ悪役は 'tumble'（ひじ・ひざを曲げた形、短いスカーフ）＋空の側のリム（ENV_FLY）、ぶれの上限 2.5＋1.2·大きさ px。
"""
import math

import cv2

import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W, H
from beats import _geo as G
from cutB.beats import _art as A, _sky as S, _motion as M
from cutB.beats import _beach as BE, _figures3d as F3, _render as R

FS = 1.15
DIR = -27.0                                  # 拳の向き（度）
DV = np.array([math.cos(math.radians(DIR)), math.sin(math.radians(DIR))])
CONTACT = (392.0, 1342.0)                    # 拳が当たる所（悪役の胴の左）（世界の座標）
BEZ_C = (1000.0, 900.0)                      # 飛んでいく弧の制御点
EXTRA = 300                                  # 世界の背景を下へ足す行
OFF0 = -240.0                                # はじめのカメラのずれ（世界を上へ）
HOLD = 3                                     # ヒットストップのフレーム数
HITC = np.array(CONTACT) - 68 * FS * DV      # 当たったときの拳のまんなか


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


def push_end(beat=None):
    """01・02 の通しの押し込みが終わる（02 のズームスルーが始まる）通し番号。"""
    bs = timeline.load()
    b1 = next(b for b in bs if b['key'] == '01')
    b2 = next(b for b in bs if b['key'] == '02')
    return b2['f0'] - b1['f0'] + 46


# ---------------------------------------------------------------- 飛ぶ
def flight_tau(ta, T):
    """飛ぶ進み具合（0..1）。ta はヒットストップで付け替えた時間。キラーンのフレームで 1。"""
    return float(np.clip((ta - T['hit']) / float(T['glint'] - HOLD - T['hit']), 0, 1))


def flight(ta, T):
    """飛んでいる悪役の (x, y, scale, ang, 進み具合 tp)。"""
    tau = flight_tau(ta, T)
    tp = 1 - (1 - tau) ** 2.2
    x, y = bez(tp, torso(), BEZ_C, S.DOT)
    sc = S.VSCALE * np.exp(np.log(0.12 / S.VSCALE) * tau ** 1.25)
    ang = -30 + 600 * M.ease(tau, 'cubic_out')         # 速く回りはじめ、小さくなるにつれて落ち着く
    return float(x), float(y), float(sc), float(ang), float(tp)


def flight_vel(ta, T, h=0.5):
    a = np.array(flight(ta - h, T)[:2])
    b = np.array(flight(ta + h, T)[:2])
    return (b - a) / (2 * h)


def trail(f, tp, tau, alpha):
    """飛んだ跡のうすい弧（太さは手前ほど太い）と、悪役のうしろの速さの線。"""
    if alpha <= 0.003 or tp <= 0.002:
        return f
    ts = np.linspace(0, tp, 60)
    P = bez(ts, torso(), BEZ_C, S.DOT)
    x0, y0 = int(max(0, P[:, 0].min() - 40)), int(max(0, P[:, 1].min() - 40))
    x1, y1 = int(min(W, P[:, 0].max() + 40)), int(min(f.shape[0], P[:, 1].max() + 40))
    L = core.SS(s=2, w=x1 - x0, h=y1 - y0)
    Q = P - (x0, y0)
    for k in range(len(Q) - 1):
        u = ts[k] / max(tp, 1e-6)
        wd = 1.2 + 4.0 * (1 - ts[k]) ** 1.5
        a = alpha * 0.30 * (0.25 + 0.75 * u)
        L.draw.line(L.p([tuple(Q[k]), tuple(Q[k + 1])]), fill=L.col(core.FOAM, a), width=L.w(wd))
    lay = L.done()
    lay = M._from_prem(cv2.GaussianBlur(M._to_prem(lay), (0, 0), 1.0))
    reg = f[y0:y1, x0:x1]
    a = lay[..., 3:4]
    f[y0:y1, x0:x1] = reg * (1 - a) + lay[..., :3] * a
    # 悪役のうしろの速さの筋（先が細い、ぶらした光の筋）
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
            back = math.degrees(math.atan2(-v[1], -v[0]))
            for off, k in ((-1, 0.7), (0, 1.0), (1, 0.7)):
                a0 = p1 - v * (14 * sc) + nrm * off * 9 * sc
                f = streaks(f, tuple(a0), [(back, 0, ln * k, max(2.0, 2.6 * sc / 2) * 1.6,
                                            alpha * 0.5 * (1 - tp) ** 0.5)], blur=1.0)
    return f


# ---------------------------------------------------------------- 当たり
def streaks(f, origin, items, s=3, blur=0.9, rgb=None):
    """先が細くなる光の筋（items：(角度°, r0, r1, 太さ px, α)）。筋の向きにぶらして重ねる。輪にはしない。"""
    rgb = core.FOAM if rgb is None else rgb
    if not items:
        return f
    R_ = max(it[2] for it in items) + 12
    ox, oy = origin
    x0, y0 = int(ox - R_), int(oy - R_)
    L = core.SS(s=s, w=2 * int(R_), h=2 * int(R_))
    for ang, r0, r1, wd, a in items:
        if a <= 0.003 or r1 <= r0:
            continue
        th = math.radians(ang)
        d = np.array([math.cos(th), math.sin(th)])
        n = np.array([-d[1], d[0]])
        c = np.array([ox - x0, oy - y0])
        p0, p1 = c + d * r0, c + d * r1
        pm = c + d * (r0 + 0.3 * (r1 - r0))
        pts = [p0, pm + n * wd / 2, p1, pm - n * wd / 2]
        L.draw.polygon(L.p([tuple(q) for q in pts]), fill=L.col(rgb, a))
    lay = L.done()
    if blur > 0:
        lay = M._from_prem(cv2_blur(M._to_prem(lay), blur))
    return M.put_rgba(f, lay, x0, y0)


def impact(f, k):
    """当たった瞬間（k=0..6）：打つ向き ±35° に、先の細い光の筋（40–110px、α0.6）が外へ伸びて消える＋ごく弱い空気のふくらみ。
    輪・放射の全周・平らな破片は描かない。画面全体は光らせない。"""
    if k < 0 or k >= 7:
        return f
    cx, cy = CONTACT
    e = M.ease((k + 1) / 7.0, 'expo_out')
    fade = (1 - k / 7.0) ** 0.9
    items = []
    for j, (da, ln) in enumerate(((-33, 70), (-21, 104), (-9, 82), (2, 110), (13, 90), (24, 64), (34, 48))):
        r0 = 34 + 46 * e + (j % 3) * 4
        items.append((DIR + da, r0, r0 + ln * (0.55 + 0.45 * e), 4.2 - 1.6 * (k / 6.0), 0.6 * fade))
    f = streaks(f, (cx, cy), items)
    if k < HOLD + 2:                                   # 空気のふくらみ（やわらかい楕円、α 0.22）
        a = 0.22 * (1 - k / float(HOLD + 2))
        r = 26 + 18 * e

        def puff(L, ox, oy):
            L.draw.ellipse(L.box(ox - r * 0.7, oy - r, ox + r * 0.7, oy + r), fill=L.col(core.FOAM, 1.0))
        f = A.put(f, puff, cx + 10 * DV[0], cy + 10 * DV[1], r + 30, s=2, alpha=a)
    return f


class SandKick:
    """足もとからの砂の粒（90 粒）：地面の上を低く外へ飛び、重力で落ちる。光の当たった小さな粒＋地面の影（高いほどうすくぼける）。"""

    def __init__(self, f0, origin, n=44, seed=12):
        rng = np.random.default_rng(seed)
        self.f0, self.o = f0, np.asarray(origin, float)
        ang = np.radians(np.concatenate([rng.normal(-10, 24, int(n * 0.7)), rng.normal(188, 16, n - int(n * 0.7))]))
        sp = rng.uniform(8.0, 30.0, n) * np.where(np.cos(ang) > 0, 1.0, 0.5)
        self.vx = np.cos(ang) * sp                       # 横（画面の x）
        self.vd = np.sin(ang) * sp * 0.35 + rng.normal(0, 1.2, n)   # 奥行き（画面の y：地面の上の位置）
        self.vh = rng.uniform(1.5, 6.5, n)               # 上へ（px/フレーム）。高さ ≤ 約 40px
        self.g = 0.55
        self.r = rng.uniform(2.2, 4.6, n)
        self.life = rng.uniform(10, 18, n)
        self.x0 = rng.normal(0, 8, n)

    def puff(self, f, i):
        """低く横へ広がる砂ぼこり（やわらかい、α ≤0.28）。"""
        dt = float(i - self.f0)
        if dt < 0 or dt > 22:
            return f
        e = M.ease((dt + 1) / 12.0, 'expo_out')
        a = 0.36 * min(1.0, (dt + 1) / 2.0) * (1 - M.ease(dt / 22.0, 'sine_in'))
        cx, cy = self.o[0] + 30 * e, self.o[1] - 6 - 10 * e
        rx, ry = 30 + 120 * e, 8 + 18 * e

        def fn(L, ox, oy):
            for k, (dx, dy, sx, sy) in enumerate(((-0.35, 0.1, 0.6, 0.8), (0.0, 0.0, 0.8, 1.0), (0.45, 0.15, 0.55, 0.7))):
                L.draw.ellipse(L.box(ox + dx * rx - sx * rx, oy + dy * ry - sy * ry, ox + dx * rx + sx * rx,
                                     oy + dy * ry + sy * ry), fill=L.col(core.c('DCE2E7'), 0.6))
        sub = np.zeros((int(2 * ry + 80), int(2 * rx + 80), 3), np.float32)
        x0, y0 = int(cx - rx - 40), int(cy - ry - 40)
        L = core.SS(s=2, w=sub.shape[1], h=sub.shape[0])
        fn(L, cx - x0, cy - y0)
        lay = L.done()
        lay = M._from_prem(cv2_blur(M._to_prem(lay), 9.0 + 6 * e))
        return M.put_rgba(f, lay, x0, y0, alpha=a)

    def draw(self, f, i):
        dt = float(i - self.f0)
        if dt < 0 or dt > 18:
            return f
        kd = (1 - np.exp(-dt * 0.12)) / 0.12
        gx = self.o[0] + self.x0 + self.vx * kd
        gy = self.o[1] + self.vd * kd
        hh = np.maximum(self.vh * dt - 0.5 * self.g * dt * dt, 0)
        a = np.clip(1 - dt / self.life, 0, 1) ** 0.8 * np.clip(dt + 1, 0, 1)
        live = a > 0.02
        if not live.any():
            return f
        x0, y0 = int(gx[live].min() - 20), int((gy - hh)[live].min() - 20)
        x1, y1 = int(gx[live].max() + 30), int(gy[live].max() + 30)
        s = 4
        w, h = x1 - x0, y1 - y0
        shd = np.zeros((h * s, w * s), np.float32)
        col = np.zeros((h * s, w * s), np.float32)
        hi = np.zeros((h * s, w * s), np.float32)
        al = np.zeros((h * s, w * s), np.float32)
        for k in np.nonzero(live)[0]:
            px, py = (gx[k] - x0) * s, (gy[k] - hh[k] - y0) * s
            vy = (self.vh[k] - self.g * dt)
            ln = max(1.0, min(4.0, abs(vy) * 0.35 + abs(self.vx[k]) * 0.3))
            ang = math.degrees(math.atan2(-vy, self.vx[k]))
            r = self.r[k] * s
            cv2.ellipse(al, (int(px * 16), int(py * 16)), (int(r * ln * 16), int(r * 16)), ang, 0, 360,
                        float(a[k]), -1, cv2.LINE_AA, shift=4)
            cv2.circle(hi, (int((px - r * 0.35) * 16), int((py - r * 0.35) * 16)), int(r * 0.55 * 16), float(a[k]),
                       -1, cv2.LINE_AA, shift=4)
            sx, sy = (gx[k] + hh[k] * 0.55 - x0) * s, (gy[k] + hh[k] * 0.12 + 1 - y0) * s
            cv2.ellipse(shd, (int(sx * 16), int(sy * 16)), (int(r * 1.3 * 16), int(r * 0.6 * 16)), 0, 0, 360,
                        float(a[k] * 0.45 * np.clip(1 - hh[k] / 45.0, 0.1, 1)), -1, cv2.LINE_AA, shift=4)
        dn = lambda q: cv2.resize(q, (w, h), interpolation=cv2.INTER_AREA)
        shd, al, hi = cv2_blur(dn(shd), 1.2), dn(al), dn(hi)
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, f.shape[1]), min(y1, f.shape[0])
        if xb <= xa or yb <= ya:
            return f
        sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
        reg = f[ya:yb, xa:xb]
        reg = reg * (1 - shd[sl][..., None] * (1 - core.c('1B3550')))
        base = core.c('97A3AE') * (1 - np.clip(hi[sl], 0, 1)[..., None]) + core.c('CDD5DB') * np.clip(hi[sl], 0, 1)[..., None]
        am = np.clip(al[sl], 0, 1)[..., None]
        f[ya:yb, xa:xb] = reg * (1 - am) + base * am
        return f


_PK = {}


class SandKick3D(SandKick):
    """2026-11-08 r2：当たりの砂の跳ね（50 粒）。地面の 3D（横 x・奥行き z・高さ h）で、上へ・カメラから遠ざかる向きへ飛び、
    約 8 フレームで落ちて止まる。各粒は 1 フレームの移動ぶんぶらした筋（エネルギーを保つ α）。影の円盤は描かない（汚れに見えない）。
    puff（低い砂ぼこり）は SandKick のまま。"""

    def __init__(self, f0, origin, n=50, seed=12):
        super().__init__(f0, origin, n=44, seed=seed)
        rng = np.random.default_rng(seed + 100)
        self.n = n
        a = np.radians(rng.normal(-8.0, 30.0, n))                 # 横の向き（打つ向きの右へ多め）
        sp = rng.uniform(2.0, 12.0, n)
        self.kx = np.cos(a) * sp + 2.0                            # px/フレーム（足もとの奥行きでの画面の x）
        self.kz = rng.uniform(0.006, 0.030, n)                    # 奥へ（足もとの奥行きを 1 として、1 フレームあたり）
        self.kh = rng.uniform(6.0, 16.0, n)                       # 上へ px/フレーム
        self.kg = 2.9
        self.kr = rng.uniform(1.5, 3.4, n)
        self.kx0 = rng.normal(0, 7, n)
        self.kd = rng.uniform(0, 1.6, n)                          # 出る時刻のずれ

    def _pos(self, dt):
        t = np.clip(dt - self.kd, 0, None)
        t_land = 2 * self.kh / self.kg
        tf = np.minimum(t, t_land)
        z = 1.0 + self.kz * tf * (1 - 0.02 * tf)
        k = 1.0 / z                                               # 遠いほど小さく、地平へ近づく
        gy = S.HZ + (self.o[1] - S.HZ) * k
        gx = self.o[0] + (self.kx0 + self.kx * tf * (1 - 0.03 * tf)) * k
        hh = np.maximum(self.kh * tf - 0.5 * self.kg * tf * tf, 0)
        return gx, gy - hh * k, k, t, t_land

    def draw(self, f, i):
        dt = float(i - self.f0)
        if dt < 0 or dt > 16:
            return f
        x1, y1, k, t, tl = self._pos(dt)
        x0_, y0_, _, _, _ = self._pos(dt - 0.5)                  # 180° シャッター
        a = np.clip((t - 0.0) * 2, 0, 1) * (1 - np.clip((t - tl) / 4.0, 0, 1))
        live = (a > 0.02) & (t > 0)
        if not live.any():
            return f
        s = 4
        bx0 = int(min(x1[live].min(), x0_[live].min()) - 12)
        by0 = int(min(y1[live].min(), y0_[live].min()) - 12)
        bx1 = int(max(x1[live].max(), x0_[live].max()) + 12)
        by1 = int(max(y1[live].max(), y0_[live].max()) + 12)
        w, h = bx1 - bx0, by1 - by0
        al = np.zeros((h * s, w * s), np.float32)
        hi = np.zeros((h * s, w * s), np.float32)
        P = lambda v: int(round(v * s * 16))
        for j in np.nonzero(live)[0]:
            r = max(0.6, self.kr[j] * k[j])
            ln = math.hypot(x1[j] - x0_[j], y1[j] - y0_[j])
            aa = float(a[j]) * min(1.0, (2 * r) / (2 * r + ln)) ** 0.6
            p0 = (P(x0_[j] - bx0), P(y0_[j] - by0))
            p1 = (P(x1[j] - bx0), P(y1[j] - by0))
            cv2.line(al, p0, p1, aa, max(1, int(round(2 * r * s))), cv2.LINE_AA, shift=4)
            cv2.circle(hi, (P(x1[j] - bx0 - 0.3 * r), P(y1[j] - by0 - 0.3 * r)), P(r * 0.6), aa, -1, cv2.LINE_AA,
                       shift=4)
        al = cv2.resize(al, (w, h), interpolation=cv2.INTER_AREA)
        hi = cv2.resize(hi, (w, h), interpolation=cv2.INTER_AREA)
        xa, ya, xb, yb = max(bx0, 0), max(by0, 0), min(bx1, f.shape[1]), min(by1, f.shape[0])
        if xb <= xa or yb <= ya:
            return f
        sl = (slice(ya - by0, yb - by0), slice(xa - bx0, xb - bx0))
        am = np.clip(al[sl], 0, 1)[..., None]
        hm = np.clip(hi[sl], 0, 1)[..., None]
        col = core.c('B4BDC5') * (1 - hm) + core.c('E8ECEE') * hm         # 砂粒：上が光る
        f[ya:yb, xa:xb] = f[ya:yb, xa:xb] * (1 - am) + col * am
        return f


def particles(T):
    if _PK.get('f0') != T['hit']:
        _PK['f0'] = T['hit']
        _PK['kick'] = SandKick3D(T['hit'] + HOLD - 1, (S.FEET[0] + 6, S.FEET[1] + 2))
    return _PK['kick']


# ---------------------------------------------------------------- 拳
_FIST = {}


def fist_sprite():
    """拳の板を 1 回だけ描く（3D、向きは DIR 固定）。(rgba, ax, ay)：ax, ay は拳のまんなか。影は地面へ別に落とす。"""
    if 'f' not in _FIST:
        _FIST['f'] = F3.fist_v3(DIR, FS)          # 2026-11-08 r2：本物の拳（拳頭の打つ面・折った中節・親指・無地の袖）
    return _FIST['f']


_HOLD = {}


def _hold_sprite(fn, key=0):
    """ヒットストップのあいだ同じ形の悪役を 1 回だけ描く（key ごと）。"""
    if key not in _HOLD:
        _HOLD[key] = fn()
    return _HOLD[key]


def cv2_blur(a, s):
    import cv2
    return cv2.GaussianBlur(np.ascontiguousarray(a), (0, 0), s)


def fist_s(i, T):
    """拳の進み s（DIR に沿った HITC からの距離、負＝手前）。None＝画面にいない。i は小数でもよい。"""
    h = T['hit']
    s_out = -900.0
    s_poke = -225.0
    if i < h - 13:
        return None
    if i < h - 7:                                  # 顔を出す（expo_out、6 フレーム）
        return s_out + (s_poke - s_out) * M.ease((i - (h - 13)) / 6.0, 'expo_out')
    if i < h - 3:                                  # 少し引く（ため）
        return s_poke - 35 * M.ease((i - (h - 7)) / 4.0, 'sine_inout')
    if i < h:                                      # 打ち込む
        return (s_poke - 35) * (1 - M.ease((i - (h - 3)) / 3.0, 'cubic_in'))
    if i < h + HOLD:                               # ヒットストップ：少し食い込んだまま止まる
        return 10.0
    if i < h + HOLD + 3:                           # 押し込み（フォロースルー）
        return 10 + 30 * M.ease((i - h - HOLD) / 3.0, 'quad_out')
    q = (i - h - HOLD - 3) / 9.0                   # 去る（back_in、9 フレーム）
    if q >= 1:
        return None
    return 40 - 980 * M.ease(q, 'back_in')


def put_fist(f, i, T):
    s = fist_s(i, T)
    if s is None:
        return f
    rgba, ax, ay = fist_sprite()
    p = HITC + s * DV
    s0, s1 = fist_s(i - 0.25, T), fist_s(i + 0.25, T)
    v = 0.0 if (s0 is None or s1 is None) else (s1 - s0) * 2       # 1 フレームあたり
    if T['hit'] <= i < T['hit'] + HOLD:
        v = 0.0                                                     # ヒットストップ中はぶれなし
    # 速さの線（前へ進んでいるとき）
    q = fist_s(i - 1, T)
    if v > 12:
        f = speed_lines(f, p, None if q is None else HITC + q * DV)
    f = ground_shadow(f, rgba, int(p[0] - ax), int(p[1] - ay), S.FEET[1] + 4, opacity=0.30,
                      lift=max(0.0, S.FEET[1] - p[1] - 30))
    # 拳の下の砂の環境の遮へい（やわらかい楕円。拳が低いほど濃い）
    gap = max(0.0, S.FEET[1] + 4 - (p[1] + 52 * FS))
    f = R.contact_shadow(f, p[0] + 18, S.FEET[1] + 6, 70 + 0.25 * gap, 9 + 0.05 * gap,
                         opacity=0.30 * float(np.exp(-gap / 90.0)), rgb=core.c('1B3550'), blur=10 + 0.1 * gap)
    spr = rgba
    pad = 0
    if abs(v) > 6:                                  # サブフレームのぶれ（180°シャッター＝1 フレームの移動の半分）
        ln = abs(v) * 0.5
        pad = int(ln / 2) + 2
        spr = np.pad(rgba, ((pad, pad), (pad, pad), (0, 0)))
        spr = M.smear(spr, DV[0] * ln, DV[1] * ln)
    x0f, y0f = p[0] - ax - pad, p[1] - ay - pad
    ix, iy = int(np.floor(x0f)), int(np.floor(y0f))
    import cv2
    spr = cv2.warpAffine(spr, np.float32([[1, 0, x0f - ix], [0, 1, y0f - iy]]), (spr.shape[1] + 1, spr.shape[0] + 1),
                         flags=cv2.INTER_LINEAR)
    return M.put_rgba(f, spr, ix, iy)


def speed_lines(f, p, q):
    """拳のうしろ（袖口の後ろ）に 3 本の先の細い光の筋（ぶらして重ねる）。q：前のフレームの位置（None なら画面の外から）。"""
    d = DV
    sp = 60.0 if q is None else float(np.dot(np.asarray(p) - q, d))
    if sp < 12:
        return f
    a = min(1.0, (sp - 12) / 30.0)
    n = np.array([-d[1], d[0]])
    back = math.degrees(math.atan2(-d[1], -d[0]))
    items = []
    org = np.asarray(p) - d * (92 * FS)
    for off, k in ((-34, 0.75), (0, 1.0), (34, 0.75)):
        o = org + n * off * FS
        items.append((o, min(240.0, sp * 2.2) * k))
    for o, ln in items:
        f = streaks(f, tuple(o), [(back, 0, ln, 5.0, 0.5 * a)], blur=1.2)
    return f


def ground_shadow(f, rgba, x0, y0, ground_y, opacity=0.42, lift=0.0):
    """悪役・拳の砂への影：太陽は左上（手前の上）→ 影は右下へ倒れる。半影は地面からの距離で広がる（σ 4px＋0.08·距離）。
    色は #1B3550 を掛ける（黒にしない）。ground_y：この物の地面の行。lift：物が地面から浮いている高さ（px）。"""
    a = rgba[..., 3]
    h, w = a.shape
    fy = ground_y - y0
    pad = 170
    big = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
    big[pad:pad + h, pad:pad + w] = a
    sq, sh = 0.30, 0.75
    Mx = np.float32([[1, -sh, sh * (fy + pad)], [0, -sq, (fy + pad) * (1 + sq)]])
    sm = cv2.warpAffine(big, Mx, (big.shape[1], big.shape[0]), flags=cv2.INTER_LINEAR)
    yy = np.arange(big.shape[0], dtype=np.float32)[:, None] - (fy + pad)
    s0 = 1.0 + 0.08 * lift
    b1 = cv2_blur(sm, max(0.8, s0 * 0.5))
    b2 = cv2_blur(sm, 4.0 + 0.08 * (60 + lift))
    k = np.clip(yy / 40.0 + lift / 60.0, 0, 1)
    sm = (b1 * (1 - k) + b2 * k) * opacity * (1 - 0.45 * k)
    X0, Y0 = x0 - pad, y0 - pad
    xa, ya, xb, yb = max(X0, 0), max(Y0, 0), min(X0 + big.shape[1], f.shape[1]), min(Y0 + big.shape[0], f.shape[0])
    if xb <= xa or yb <= ya:
        return f
    sub = sm[ya - Y0:yb - Y0, xa - X0:xb - X0, None]
    f[ya:yb, xa:xb] = f[ya:yb, xa:xb] * (1 - sub * (1 - core.c('1B3550')))
    return f


# ---------------------------------------------------------------- 悪役（立つ → 当たる → 飛ぶ）
def villain_draw(t, T):
    """小数のフレーム t の悪役の板 (rgba, x0, y0)（世界の座標）。サブフレームのぶれに使う。"""
    h = T['hit']
    ta = M.hitstop(t, [(h, HOLD)])
    sec = t / core.FPS
    fx, fy = S.FEET
    tx, ty = torso()
    if t < h:
        # 胸を張って弾む（2.2Hz、足もとを中心に縦に伸び縮み）＋ 37–42 で拳に気づいてのけぞる
        ph = math.sin(2 * math.pi * 2.2 * sec)
        sy = 1.025 + 0.025 * ph
        sx = 1 / math.sqrt(sy)
        lean = 7.0 * M.ease((t - (h - 9)) / 6.0, 'back_out')
        bob = -2.0 * max(0.0, ph)
        Rl = A._R(lean)
        M2 = Rl @ np.diag([sx, sy])
        c = np.array([fx, fy]) + M2 @ (np.array([tx, ty]) - (fx, fy))
        return F3.villain_v2(c[0], c[1] + bob, S.VSCALE, 0.0, 'stand', sec, M2=M2)
    if t < h + HOLD:
        # ヒットストップ：打つ向きにつぶれて止まる（2026-11-08 r2：0.85/1.12 を 2 フレーム → 最後の 1 フレームで少し戻す）
        k_ = 0 if t < h + HOLD - 1 else 1
        sa, sb = ((0.85, 1.12), (0.92, 1.06))[k_]
        M2 = M.squash_matrix(DIR, sa, sb) @ A._R(7.0)
        c = np.array([fx, fy]) + M2 @ (np.array([tx, ty]) - (fx, fy)) + DV * 8
        return _hold_sprite(lambda: F3.villain_v2(c[0], c[1], S.VSCALE, 0.0, 'stand', h / core.FPS, M2=M2), k_)
    x, y, sc, ang, tp = flight(ta, T)
    if sc * 34 < 5:
        return None
    v = flight_vel(ta, T)
    va = math.degrees(math.atan2(v[1], v[0]))
    k = (ta - h) / core.FPS                       # 打ち出しからの秒
    s_al = 1.45 + (1.0 - 1.45) * M.spring(k, 0.5, 5.0)
    s_ac = 0.80 + (1.0 - 0.80) * M.spring(k, 0.5, 5.0)
    M2 = M.squash_matrix(va, s_al, s_ac)
    vl = flight_vel(ta - 2, T)                    # マントは 2 フレーム遅れて、速さの反対へ
    cape = math.degrees(math.atan2(-vl[1], -vl[0])) + 9 * math.sin(sec * 11.0)
    return F3.villain_v2(x, y, sc, ang, 'tumble', sec, M2=M2, cape_ang=cape, env=F3.ENV_FLY)


def put_villain(f, i, T):
    h = T['hit']
    ta = M.hitstop(i, [(h, HOLD)])
    if i >= h + HOLD:
        x, y, sc, ang, tp = flight(ta, T)
        if sc * 34 < 5:
            return A.put_dot(f, x, y, max(1.6, sc * 8))
        r = villain_draw(float(i), T)
        if r is None:
            return f
        v = flight_vel(ta, T)
        sp = float(np.hypot(*v))
        rgba, x0, y0 = r
        if sp > 8:                                  # 1 回描いて速さの向きにぶらす（180° シャッターの 0.35）
            ln = min(sp * 0.35, 2.5 + 1.2 * sc)    # 2026-11-08 r2：ぶれの上限（頭・手足が読めるように）
            pad = int(ln / 2) + 2
            rgba = M.smear(np.pad(rgba, ((pad, pad), (pad, pad), (0, 0))), v[0] / sp * ln, v[1] / sp * ln)
            x0, y0 = x0 - pad, y0 - pad
        if i < h + HOLD + 6:                        # 打ち出しの直後：まだ地面の近く。影も少し
            f = ground_shadow(f, r[0], r[1], r[2], S.FEET[1], opacity=0.35 * (1 - (i - h - HOLD) / 6.0),
                              lift=max(0.0, S.FEET[1] - (r[2] + r[0].shape[0] * 0.8)))
        return M.put_rgba(f, rgba, x0, y0)
    r = villain_draw(float(i), T)
    f = ground_shadow(f, r[0], r[1], r[2], S.FEET[1])
    f = R.contact_shadow(f, S.FEET[0] + 2, S.FEET[1] + 1, 15, 3.0, opacity=0.55, rgb=core.c('1B3550'))
    return M.put_rgba(f, *r)


# ---------------------------------------------------------------- カメラ
def cam_matrix(i, T, gi_end):
    """世界（下に EXTRA 行足した板）→ 画面 の 2×3。"""
    h = T['hit']
    z = 1.10 - 0.10 * M.ease(i / 40.0, 'cubic_out')     # 引いて見せる（拳が来るころまで動き続ける）
    if i < h + 3:
        oy = OFF0
    else:
        oy = OFF0 * (1 - M.spring((i - h - 3) / core.FPS, 0.85, 1.1))   # ゆっくり追って、キラーンの直前に着く
        oy *= 1 - M.ease((i - (T['glint'] - 8)) / 8.0, 'sine_inout')      # キラーンでぴったり 0（残り 0.3px 以下）
    cam = M.Cam2D(xy=(0.0, oy), zoom=z, pivot=torso())
    Mf = np.vstack([cam.M(i, extra=M.shake(i, h, amp=9, tau=3, rot_deg=0.35)), [0, 0, 1]])
    k = S.push_k(i, gi_end)
    P = np.array([[k, 0, S.DOT[0] * (1 - k)], [0, k, S.DOT[1] * (1 - k)], [0, 0, 1]])
    return (P @ Mf)[:2]


# ---------------------------------------------------------------- 文字
_TX = {}


def texts():
    if 't' not in _TX:
        _TX['t'] = (M.KText([dict(text='パンチ一発で', x=W / 2, y=470, size=48, weight='Medium', track=0.10)]),
                    M.KText([dict(text='空のかなたへ', x=W / 2, y=600, size=100, weight='Bold', hl='空のかなたへ')]))
    return _TX['t']


def put_texts(f, i, T):
    t1, t2 = texts()
    f = t1.reveal(f, i, T['pa'] - 2, 'char_rise', stagger=1.1, dur=10, dy_em=0.32, blur=5)
    f = t2.reveal(f, i, T['so'] - 2, 'char_rise', stagger=1.0, dur=11, dy_em=0.26, blur=7, accent=T['so'] + 5)
    return f


def render(beat, ctx):
    n = beat['n']
    T = timing(beat)
    gi_end = push_end()
    kick = particles(T)
    _HOLD.clear()
    BE.static()
    for i in range(n):
        f = BE.world(i, extra=EXTRA)
        h = T['hit']
        # 跡（飛んでいるあいだ）
        if i >= h + HOLD:
            ta = M.hitstop(i, [(h, HOLD)])
            tau = flight_tau(ta, T)
            tp = 1 - (1 - tau) ** 2.2
            f = trail(f, tp, tau, 1 - core.ease((i - T['glint'] - 2) / 14.0))
        # 悪役（キラーンのあとは点）
        if i < T['glint']:
            f = put_villain(f, i, T)
        else:
            tw = 0.85 + 0.15 * np.cos((i - T['glint']) * 0.35)
            f = A.put_dot(f, S.DOT[0], S.DOT[1], 2.3, alpha=tw)
        f = kick.puff(f, i)
        f = kick.draw(f, i)
        f = impact(f, i - h)
        f = put_fist(f, i, T)
        # キラーン
        f = A.put_glint_fx(f, S.DOT[0], S.DOT[1], i - T['glint'])
        # カメラ（世界 → 画面）
        Mc = cam_matrix(i, T, gi_end)
        f = cv2.warpAffine(f, Mc.astype(np.float32), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        hz_y = float(Mc[1, 1] * S.HZ + Mc[1, 2])
        hs = 1.0 if h <= i < h + HOLD else 0.0
        f = BE.screen_post(f, hitstop=hs, horizon_y=hz_y)
        f = R.finish_ad(f, i, grain=0.005, vignette=0.10, ca_px=0.25)
        f = put_texts(f, i, T)
        yield np.clip(f, 0, 1).astype(np.float32)
