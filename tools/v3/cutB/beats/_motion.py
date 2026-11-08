"""案B の動きの道具箱（director's motion spec §T1–T9）。numpy / cv2 / PIL だけ。CPU で軽い。

すべての関数はフレーム番号（または時間）を引数にとり、隠れた状態を持たない：どのフレームをどの順で描いても同じ絵になる。
使う側は先に `from cutB.beats import _useA` を import すること（案A の beats を解決するため）。

API（安定。引数を足すときは既定値で今の動きを保つ）：
  T1 ease(t, kind) / cubic_bezier(x1,y1,x2,y2) / prog(i, f0, f1, kind) / Track([(frame, value, kind), ...]).at(i)
  T2 spring(t_sec, zeta, hz) / spring_to(i, f0, a, b, zeta, hz) / vnoise(x, seed) / jitter(i, amp, hz, seed)
  T3 stagger(i, f0, k, step, dur, kind) / hitstop(i, events) / carry(v_out, n, kind) / velocity(track, i)
  T4 subframe(draw_fn, i, shutter, n, vel_px) / smear(rgba, vx, vy) / squash_matrix(theta, s_along, s_across)
     put_rgba(frame, rgba, x0, y0, alpha, shadow, shadow_px)
  T5 KText(items).reveal(frame, i, f0, mode, ...) / .exit(frame, i, f0, dur) / .put(frame, alpha)
  T6 Odometer(final, cx, y, digit_size, side_size, prefix, suffix, hl_suffix, ...).put(frame, v, ..., i, f_lock)
  T7 Cam2D(xy, zoom, rot, pivot).M(i) / .apply(frame, i) / .project(pt, i) ; shake(i, f_hit, ...)
     zoom_through(frame, q, center, direction) / whip(frame, q, axis, sign, px, blur, direction)
  T8 Burst(seed, n, origin, angle_range, speed_range, gravity, drag, life, size, ...).draw(frame, i)
  T9 depth_map(img, key) / prep_depth(d) / force_far(d, mask) / parallax_view(img, dep, cx, cy, vw, ...) / dof(...)

時間の単位：フレーム（30fps）。spring の t だけは秒。
"""
import functools
import math
import os

import cv2
import numpy as np

import core
from core import W, H, FPS

# ================================================================ T1. Easing
_S = 1.70158


def _norm_kind(kind):
    """'in_out_sine' → 'sine_inout'、'sine_in_out' → 'sine_inout'、core.ease の 'in'/'out'/'inout'/'smooth' も受ける。"""
    k = kind.lower()
    if k.startswith('in_out_'):
        return k[7:] + '_inout'
    if k.endswith('_in_out'):
        return k[:-7] + '_inout'
    return k


def _ease_arr(t, kind):
    k = _norm_kind(kind)
    if k in ('linear', 'lin'):
        return t
    if k in ('in', 'out', 'inout', 'smooth'):          # core.ease と同じ
        if k == 'in':
            return t ** 3
        if k == 'out':
            return 1 - (1 - t) ** 3
        if k == 'smooth':
            return t * t * (3 - 2 * t)
        return np.where(t < 0.5, 4 * t ** 3, 1 - (-2 * t + 2) ** 3 / 2)
    fam, _, mode = k.rpartition('_')
    if fam == 'back':
        c1, c3 = _S, _S + 1
        if mode == 'in':
            return c3 * t ** 3 - c1 * t ** 2
        if mode == 'out':
            return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2
        c2 = _S * 1.525
        return np.where(t < 0.5, (2 * t) ** 2 * ((c2 + 1) * 2 * t - c2) / 2,
                        ((2 * t - 2) ** 2 * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2)
    if k == 'elastic_out':
        return np.where(t <= 0, 0.0, np.where(t >= 1, 1.0,
                        2 ** (-10 * t) * np.sin((t * 10 - 0.75) * (2 * np.pi / 3)) + 1))
    if k == 'bounce_out':
        n1, d1 = 7.5625, 2.75
        return np.where(t < 1 / d1, n1 * t * t,
                        np.where(t < 2 / d1, n1 * (t - 1.5 / d1) ** 2 + 0.75,
                                 np.where(t < 2.5 / d1, n1 * (t - 2.25 / d1) ** 2 + 0.9375,
                                          n1 * (t - 2.625 / d1) ** 2 + 0.984375)))
    if k == 'bounce_in':
        return 1 - _ease_arr(1 - t, 'bounce_out')
    base = {
        'sine': lambda x: 1 - np.cos(x * np.pi / 2),
        'quad': lambda x: x ** 2,
        'cubic': lambda x: x ** 3,
        'quart': lambda x: x ** 4,
        'quint': lambda x: x ** 5,
        'expo': lambda x: np.where(x <= 0, 0.0, 2.0 ** (10 * x - 10)),
        'circ': lambda x: 1 - np.sqrt(np.clip(1 - x * x, 0, 1)),
    }.get(fam)
    if base is None:
        raise ValueError('unknown ease kind: %r' % kind)
    if mode == 'in':
        r = base(t)
    elif mode == 'out':
        r = 1 - base(1 - t)
    elif mode == 'inout':
        r = np.where(t < 0.5, base(2 * t) / 2, 1 - base(2 - 2 * t) / 2)
    else:
        raise ValueError('unknown ease kind: %r' % kind)
    if fam == 'expo':                                   # 端をぴったり 0 / 1 に
        r = np.where(t >= 1, 1.0, np.where(t <= 0, 0.0, r))
    return r


def ease(t, kind='cubic_inout'):
    """t は 0..1 に切り詰める。kind：linear, {sine,quad,cubic,quart,quint,expo,circ}_{in,out,inout}, back_in/out/inout,
    elastic_out, bounce_out。'in_out_sine' の書き方も可。core.ease の 'in'/'out'/'inout'/'smooth' も同じ曲線。
    スカラーなら float、配列なら配列を返す。"""
    scalar = np.isscalar(t) or np.ndim(t) == 0
    a = np.clip(np.asarray(t, np.float64), 0.0, 1.0)
    r = _ease_arr(a, kind)
    return float(r) if scalar else r


@functools.lru_cache(maxsize=64)
def cubic_bezier(x1, y1, x2, y2):
    """CSS の cubic-bezier(x1, y1, x2, y2)。関数 f(t)→y を返す（t は 0..1 に切り詰め）。Newton 5 回＋二分法。"""
    cx, bx = 3 * x1, 3 * (x2 - x1) - 3 * x1
    ax = 1 - cx - bx
    cy, by = 3 * y1, 3 * (y2 - y1) - 3 * y1
    ay = 1 - cy - by

    def sx(u):
        return ((ax * u + bx) * u + cx) * u

    def sy(u):
        return ((ay * u + by) * u + cy) * u

    def dsx(u):
        return (3 * ax * u + 2 * bx) * u + cx

    def solve(x):
        u = x
        for _ in range(5):
            e = sx(u) - x
            if abs(e) < 1e-7:
                return u
            d = dsx(u)
            if abs(d) < 1e-6:
                break
            u -= e / d
        if 0 <= u <= 1 and abs(sx(u) - x) < 1e-5:
            return u
        lo, hi, u = 0.0, 1.0, x
        for _ in range(40):
            v = sx(u)
            if abs(v - x) < 1e-7:
                break
            if v < x:
                lo = u
            else:
                hi = u
            u = (lo + hi) / 2
        return u

    def f(t):
        if np.ndim(t):
            return np.array([f(float(v)) for v in np.ravel(t)]).reshape(np.shape(t))
        t = min(max(float(t), 0.0), 1.0)
        if t <= 0 or t >= 1:
            return t
        return float(sy(solve(t)))
    return f


def _apply(kind, t):
    """kind は ease の名前か、関数（cubic_bezier の戻り値など）。"""
    if callable(kind):
        return kind(min(max(float(t), 0.0), 1.0))
    return ease(t, kind)


def prog(i, f0, f1, kind='cubic_out'):
    """f0..f1 のあいだの進み具合（0..1、切り詰め、イージングつき）。kind は名前か関数。"""
    if f1 == f0:
        return 1.0 if i >= f1 else 0.0
    return _apply(kind, (i - f0) / float(f1 - f0))


class Track:
    """After Effects 風のキーフレーム。keys = [(frame, value, kind), ...]（frame の昇順）。
    kind はそのキーから次のキーまでの区間のイージング（最後のキーの kind は使わない）。value はスカラーかタプル。
    最初のキーより前は最初の値、最後より後は最後の値。"""

    def __init__(self, keys):
        self.keys = [(float(k[0]), np.asarray(k[1], np.float64), (k[2] if len(k) > 2 and k[2] else 'linear'))
                     for k in keys]
        self.scalar = np.ndim(self.keys[0][1]) == 0

    def at(self, i):
        ks = self.keys
        if i <= ks[0][0]:
            v = ks[0][1]
        elif i >= ks[-1][0]:
            v = ks[-1][1]
        else:
            for (fa, va, ka), (fb, vb, _) in zip(ks, ks[1:]):
                if fa <= i <= fb:
                    e = _apply(ka, (i - fa) / (fb - fa)) if fb > fa else 1.0
                    v = va + (vb - va) * e
                    break
        return float(v) if self.scalar else tuple(float(x) for x in v)

    __call__ = at


# ================================================================ T2. Springs and noise
def spring(t, zeta=0.4, hz=3.0):
    """減衰ばねの単位ステップ応答（t は秒、t<0 で 0、→1 に収まる）。ζ≥1 は臨界減衰。"""
    if np.ndim(t):
        return np.array([spring(float(v), zeta, hz) for v in np.ravel(t)]).reshape(np.shape(t))
    if t <= 0:
        return 0.0
    w = 2 * math.pi * hz
    if zeta >= 1:
        return 1 - math.exp(-w * t) * (1 + w * t)
    wd = w * math.sqrt(1 - zeta * zeta)
    return 1 - math.exp(-zeta * w * t) * (math.cos(wd * t) + zeta / math.sqrt(1 - zeta * zeta) * math.sin(wd * t))


def spring_to(i, f0, a, b, zeta=0.4, hz=3.0):
    """フレーム f0 で a から b へ動きはじめるばねの、フレーム i の値。"""
    return a + (b - a) * spring((i - f0) / FPS, zeta, hz)


def _hash(n, seed):
    v = np.sin(np.asarray(n, np.float64) * 127.1 + seed * 311.7 + 0.123) * 43758.5453
    return v - np.floor(v)


def vnoise(x, seed=0):
    """なめらかな 1 次元のバリューノイズ（-1..1、格子は整数、余弦補間）。"""
    x = np.asarray(x, np.float64)
    i0 = np.floor(x)
    f = x - i0
    a, b = _hash(i0, seed) * 2 - 1, _hash(i0 + 1, seed) * 2 - 1
    u = (1 - np.cos(f * np.pi)) / 2
    r = a + (b - a) * u
    return float(r) if np.ndim(r) == 0 else r


def jitter(i, amp=1.0, hz=8.0, seed=0):
    """2 オクターブのノイズ（フレーム i、hz は 1 秒あたりの格子の数）。おおむね ±amp。"""
    x = i / FPS * hz
    return amp * (0.7 * vnoise(x, seed) + 0.3 * vnoise(x * 2.03 + 5.1, seed + 17)) / 0.85


# ================================================================ T3. Timing helpers
def stagger(i, f0, k, step=1.2, dur=10, kind='expo_out'):
    """k 番目の要素（k*step フレーム遅れて始まる）の進み具合 0..1。"""
    return _apply(kind, (i - f0 - k * step) / float(dur))


def hitstop(i, events):
    """events=[(f, hold), ...]：フレーム f で hold フレームだけ止まるように時間を付け替える（戻り値は float のフレーム）。
    悪役や拳はこの時間で動かし、カメラの揺れは本当の時間 i で動かす。"""
    t = float(i)
    for f, hold in events:
        t -= float(np.clip(i - f, 0, hold))
    return t


def carry(v_out, n, kind='expo_out'):
    """前のショットの最後の速さ v_out（px/フレーム など）を受け継ぐ、次のショットのずれの曲線 off(j)（j=0 で 0、速さ v_out、
    n フレームで止まる）を返す。最初の 1 フレームの移動がちょうど v_out になるよう距離を決める。"""
    k1 = _apply(kind, 1.0 / n) - _apply(kind, 0.0)          # 最初の 1 フレームで進む割合
    D = v_out / max(k1, 1e-6)

    def off(j):
        return D * _apply(kind, j / float(n))
    off.distance = D
    return off


def velocity(track, i, h=0.5):
    """track（関数か Track）のフレーム i での 1 フレームあたりの変化（中心差分）。タプルにも対応。"""
    fn = track.at if isinstance(track, Track) else track
    a, b = np.asarray(fn(i - h), np.float64), np.asarray(fn(i + h), np.float64)
    v = (b - a) / (2 * h)
    return float(v) if np.ndim(v) == 0 else v


# ================================================================ T4. Motion blur
def _to_prem(rgba):
    return np.concatenate([rgba[..., :3] * rgba[..., 3:4], rgba[..., 3:4]], axis=2)


def _from_prem(p):
    a = p[..., 3:4]
    rgb = np.where(a > 1e-5, p[..., :3] / np.maximum(a, 1e-5), 0)
    return np.concatenate([np.clip(rgb, 0, 1), np.clip(a, 0, 1)], axis=2).astype(np.float32)


def subframe(draw_fn, i, shutter=0.5, n=None, vel_px=None):
    """サブフレームのモーションブラー。draw_fn(t)→(rgba, x0, y0)（t は小数のフレーム。rgba はストレートアルファの小さな板）。
    t ∈ i ± shutter/2 の n 点を premultiplied で平均する。n=None なら vel_px（1フレームの移動 px）から
    n = clip(ceil(vel_px*shutter/3), 1, 10)。戻り値 (rgba, x0, y0)。小さな板（悪役・拳・しぶき）だけに使う。"""
    if n is None:
        n = 1 if vel_px is None else int(np.clip(math.ceil(abs(vel_px) * shutter / 3.0), 1, 10))
    if n <= 1:
        return draw_fn(float(i))
    ts = [i - shutter / 2 + shutter * (k + 0.5) / n for k in range(n)]
    outs = [draw_fn(t) for t in ts]
    outs = [o for o in outs if o is not None and o[0] is not None]
    if not outs:
        return None
    x0 = min(o[1] for o in outs)
    y0 = min(o[2] for o in outs)
    x1 = max(o[1] + o[0].shape[1] for o in outs)
    y1 = max(o[2] + o[0].shape[0] for o in outs)
    acc = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
    for rgba, ox, oy in outs:
        h, w = rgba.shape[:2]
        acc[oy - y0:oy - y0 + h, ox - x0:ox - x0 + w] += _to_prem(rgba)
    return _from_prem(acc / len(ts)), x0, y0


def smear(rgba, vx, vy):
    """板（ストレートアルファの RGBA）を (vx, vy) の向き・長さの線でぼかす（安い方向のぶれ）。大きさは変えない。"""
    ln = math.hypot(vx, vy)
    if ln < 1.0:
        return rgba
    k = int(math.ceil(ln)) | 1
    ker = np.zeros((k, k), np.float32)
    c = k // 2
    ux, uy = vx / ln, vy / ln
    for s in np.linspace(-ln / 2, ln / 2, max(2, int(ln * 2))):
        ker[int(round(c + uy * s)), int(round(c + ux * s))] = 1
    ker /= ker.sum()
    p = cv2.filter2D(_to_prem(rgba.astype(np.float32)), -1, ker, borderType=cv2.BORDER_CONSTANT)
    return _from_prem(p)


def squash_matrix(theta_deg, s_along, s_across):
    """θ 方向に s_along、直角の方向に s_across 倍する 2×2 行列 R·diag·Rᵀ（画面座標、y は下向き、θ は度・時計回り）。"""
    r = math.radians(theta_deg)
    R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    return R @ np.diag([s_along, s_across]) @ R.T


def put_rgba(frame, rgba, x0, y0, alpha=1.0, shadow=0.0, shadow_px=2.0, shadow_rgb=None):
    """ストレートアルファの板を frame の (x0, y0) に重ねる（画面の外ははみ出し分を切る）。shadow：暗いにじみ。"""
    if rgba is None or alpha <= 0.003:
        return frame
    h, w = rgba.shape[:2]
    fh, fw = frame.shape[:2]
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, fw), min(y0 + h, fh)
    if xb <= xa or yb <= ya:
        return frame
    sub = rgba[ya - y0:yb - y0, xa - x0:xb - x0]
    reg = frame[ya:yb, xa:xb]
    if shadow:
        sa = cv2.GaussianBlur(np.ascontiguousarray(rgba[..., 3]), (0, 0), shadow_px)[ya - y0:yb - y0, xa - x0:xb - x0]
        sa = sa[..., None] * (alpha * shadow)
        reg = reg * (1 - sa) + (core.c('062A4E') if shadow_rgb is None else shadow_rgb) * sa
    a = sub[..., 3:4] * alpha
    frame[ya:yb, xa:xb] = reg * (1 - a) + sub[..., :3] * a
    return frame


# ================================================================ T5. Kinetic type
def _patch_of(items):
    """core.text_layer と同じ見た目の小さな板（_mapui.Patch）→ (prem rgb, a, sh, x0, y0)。"""
    from beats import _mapui as U
    p = U.Patch(items)
    a = p.rgba[..., 3].astype(np.float32)
    return p.rgba[..., :3] * a[..., None], a, p.sh.astype(np.float32), p.x0, p.y0


def _composite_sh(frame, sh, x0, y0, alpha):
    h, w = sh.shape
    fh, fw = frame.shape[:2]
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, fw), min(y0 + h, fh)
    if xb <= xa or yb <= ya:
        return
    frame[ya:yb, xa:xb] *= 1 - sh[ya - y0:yb - y0, xa - x0:xb - x0, None] * (0.55 * alpha)


def _composite_col(frame, prem, a, x0, y0, alpha):
    h, w = a.shape
    fh, fw = frame.shape[:2]
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, fw), min(y0 + h, fh)
    if xb <= xa or yb <= ya:
        return
    sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    sub = frame[ya:yb, xa:xb]
    sub *= 1 - a[sl][..., None] * alpha
    sub += prem[sl] * alpha


def _xform(arrs, sigma=0.0, scale=1.0, fx=0.0, fy=0.0, cx=0.0, cy=0.0):
    """小さな板（同じ大きさの配列のリスト）に ぼかし・中心 (cx,cy) まわりの拡大・小数のずらし をかける。"""
    out = []
    need_warp = abs(scale - 1) > 1e-4 or abs(fx) > 0.02 or abs(fy) > 0.02
    M = np.float32([[scale, 0, cx * (1 - scale) + fx], [0, scale, cy * (1 - scale) + fy]])
    for A in arrs:
        if sigma > 0.25:
            A = cv2.GaussianBlur(A, (0, 0), sigma)
        if need_warp:
            A = cv2.warpAffine(A, M, (A.shape[1], A.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out.append(A)
    return out


_MODE_DEF = {'char_rise': (10, 'expo_out'), 'line_mask': (9, 'quart_out'), 'wipe': (12, 'cubic_inout'),
             'track': (14, 'quart_out'), 'fade': (6, 'sine_inout')}


class KText:
    """1 文字ずつ動く文字（core.text_layer と同じ白い文字と柔らかい影）。items は text_layer と同じ dict のリスト
    （1 dict＝1 行。text, x, y, size, weight, anchor, track, alpha, hl, color, shadow）。各文字を 1 回だけ小さな板にする。

    reveal(frame, i, f0, mode, ...)：f0 から出る。mode：
      'char_rise'：1 文字ずつ（stagger フレームずつ遅れて）下から dy_em 上がり、ぼかし blur→0、不透明度 0→1（4 フレーム）
      'line_mask'：行ごと、ベースラインの下の見えない線から mask_em 上がってくる（9 フレーム、quart_out）
      'wipe'：左から右へやわらかいマスク（feather px）
      'track'：字間を track_from em から本来の字間へ（不透明度は 6 フレーム）
      'fade'：全体の不透明度だけ
    家のきまり：f0＝ことばのフレーム − 2（char_rise は最初の文字が 2 フレーム後＝ことばのフレームで 50%）。
    track_from を渡すと、どのモードでも字間を寄せる。accent=フレーム：強調（hl）の文字は FOAM から ACC へ 6 フレームで変わり、
    1.04→1.0 に縮む（back_out）。None なら最初から ACC。exit=フレーム：そこから exit_dur で全体が上へ消える。"""

    def __init__(self, items):
        self.items = [dict(it) for it in items if it.get('text')]
        self.glyphs = []
        for li, it in enumerate(self.items):
            text, size = it['text'], it['size']
            wt = it.get('weight', 'Regular')
            f = core.font(size, wt)
            tr = it.get('track', 0.04)
            anchor = it.get('anchor', 'ms')
            widths = [f.getlength(ch) for ch in text]
            hl = it.get('hl')
            hs = text.find(hl) if hl else -1
            it['_widths'], it['_track'] = widths, tr
            xs = self._xs(it, tr)
            base = it.get('color', core.FOAM)
            for k, ch in enumerate(text):
                is_hl = hs >= 0 and hs <= k < hs + len(hl)
                gi = dict(text=ch, x=xs[k], y=it['y'], size=size, weight=wt, anchor='l' + anchor[1], track=0,
                          alpha=it.get('alpha', 1.0), shadow=it.get('shadow', 0.6))
                if ch.strip() == '':
                    continue
                prem, a, sh, x0, y0 = _patch_of([dict(gi, color=core.ACC if is_hl else base)])
                g = dict(line=li, k=k, prem=prem, a=a, sh=sh, x0=x0, y0=y0, hl=is_hl, size=size,
                         gx=xs[k] + widths[k] / 2, gy=it['y'] - (0.38 * size if anchor[1] == 's' else 0))
                if is_hl:
                    g['prem_base'] = _patch_of([dict(gi, color=base)])[0]
                self.glyphs.append(g)
        for n, g in enumerate(self.glyphs):
            g['n'] = n
        hlg = [g for g in self.glyphs if g['hl']]
        self.hl_c = (np.mean([g['gx'] for g in hlg]), np.mean([g['gy'] for g in hlg])) if hlg else None

    @staticmethod
    def _xs(it, tr):
        widths, size = it['_widths'], it['size']
        total = sum(widths) + tr * size * (len(widths) - 1)
        anchor = it.get('anchor', 'ms')
        x = it['x'] - (total / 2 if anchor[0] == 'm' else total if anchor[0] == 'r' else 0)
        xs = []
        for wd in widths:
            xs.append(x)
            x += wd + tr * size
        return xs

    def bbox(self):
        """文字の見える範囲 (x0, y0, x1, y1)（影は除く、QA 用）。"""
        bs = []
        for g in self.glyphs:
            ys, xs = np.nonzero(g['a'] > 0.05)
            if len(xs):
                bs.append((g['x0'] + xs.min(), g['y0'] + ys.min(), g['x0'] + xs.max(), g['y0'] + ys.max()))
        return (min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs))

    def put(self, frame, alpha=1.0, dx=0.0, dy=0.0):
        """止まった状態で貼る（_mapui.Patch.put と同じ見た目）。"""
        return self.reveal(frame, 10 ** 6, 0, 'fade', alpha=alpha, dx=dx, dy=dy)

    def exit(self, frame, i, f0, dur=6, alpha=1.0, dx=0.0, dy=0.0):
        """出終わった文字が f0 から dur フレームで消える（上へ 0.15em・ぼかし 0→4・cubic_in）。"""
        return self.reveal(frame, i, -10 ** 6, 'fade', alpha=alpha, exit=f0, exit_dur=dur, dx=dx, dy=dy)

    def reveal(self, frame, i, f0, mode='char_rise', stagger=1.2, dur=None, dy_em=0.28, blur=6.0, kind=None,
               track_from=None, accent=None, alpha=1.0, exit=None, exit_dur=6, mask_em=0.9, feather=60.0,
               line_step=3.0, dx=0.0, dy=0.0, scale=1.0, pivot=None):
        """i フレーム目を frame に描く（frame を書き換えて返す）。scale/pivot：全体の拡大（pivot まわり、既定は全体の中心）。"""
        if alpha <= 0.002 or i < f0 - 0.001:
            return frame
        d0, k0 = _MODE_DEF[mode]
        dur = d0 if dur is None else dur
        kind = k0 if kind is None else kind
        ex_a, ex_dy, ex_blur = 1.0, 0.0, 0.0
        if exit is not None and i >= exit:
            q = ease((i - exit + 1) / float(exit_dur), 'cubic_in')
            ex_a, ex_dy, ex_blur = 1 - q, -0.15 * q, 4.0 * q
            if ex_a <= 0.002:
                return frame
        # 字間
        trk = None
        if track_from is not None:
            nn = len(self.glyphs)
            tq = ease((i - f0) / float(dur + stagger * max(0, nn - 1)), 'quart_out')
            trk = {}
            for li, it in enumerate(self.items):
                tr = track_from + (it['_track'] - track_from) * tq
                xs_now, xs_rest = self._xs(it, tr), self._xs(it, it['_track'])
                trk[li] = [a - b for a, b in zip(xs_now, xs_rest)]
        # 強調
        acc_m, acc_s = 1.0, 1.0
        if accent is not None:
            acc_m = ease((i - accent) / 6.0, 'sine_inout')
            acc_s = 1.04 - 0.04 * ease((i - accent) / 8.0, 'back_out')
        if pivot is None and abs(scale - 1) > 1e-4:
            gx = [g['gx'] for g in self.glyphs]
            gy = [g['gy'] for g in self.glyphs]
            pivot = ((min(gx) + max(gx)) / 2, (min(gy) + max(gy)) / 2)
        states = []
        for g in self.glyphs:
            size = g['size']
            o, gdx, gdy, sig, s = 1.0, 0.0, 0.0, 0.0, 1.0
            clip_y = None
            wipe_x = None
            if mode == 'char_rise':
                fk = f0 + g['n'] * stagger
                if i < fk:
                    continue
                p = ease((i - fk) / float(dur), kind)
                o = ease((i - fk) / 4.0, 'sine_inout')
                gdy = dy_em * size * (1 - p)
                sig = blur * (1 - p)
            elif mode == 'line_mask':
                fk = f0 + g['line'] * line_step
                if i < fk:
                    continue
                p = ease((i - fk) / float(dur), kind)
                o = ease((i - fk + 1) / 3.0, 'sine_inout')
                gdy = mask_em * size * (1 - p)
                if p < 1:
                    it = self.items[g['line']]
                    clip_y = it['y'] + (0.14 * size if it.get('anchor', 'ms')[1] == 's' else 0.62 * size)
            elif mode == 'wipe':
                fk = f0 + g['line'] * line_step
                p = ease((i - fk) / float(dur), kind)
                if p <= 0:
                    continue
                if p < 1:
                    lg = [h for h in self.glyphs if h['line'] == g['line']]
                    xl = min(h['gx'] - h['size'] * 0.6 for h in lg)
                    xr = max(h['gx'] + h['size'] * 0.6 for h in lg)
                    wipe_x = xl - feather + (xr - xl + 2 * feather) * p
            elif mode == 'track':
                o = ease((i - f0) / 6.0, 'sine_inout')
            else:
                o = ease((i - f0 + 1) / float(dur), kind) if f0 > -10 ** 5 else 1.0
            if trk is not None:
                gdx += trk[g['line']][g['k']]
            m_acc = 1.0
            if g['hl'] and accent is not None:
                m_acc = acc_m
                s = acc_s
                hx, hy = self.hl_c
                gdx += (g['gx'] - hx) * (s - 1)
                gdy += (g['gy'] - hy) * (s - 1)
            if abs(scale - 1) > 1e-4:
                gdx += (g['gx'] - pivot[0]) * (scale - 1)
                gdy += (g['gy'] - pivot[1]) * (scale - 1)
                s *= scale
            gdy += ex_dy * size + dy
            gdx += dx
            sig = math.hypot(sig, ex_blur)
            o *= ex_a * alpha
            if o <= 0.002:
                continue
            states.append((g, o, gdx, gdy, sig, s, m_acc, clip_y, wipe_x))
        if not states:
            return frame
        drawn = []
        for g, o, gdx, gdy, sig, s, m_acc, clip_y, wipe_x in states:
            prem = g['prem'] if m_acc >= 0.999 else g['prem_base'] + (g['prem'] - g['prem_base']) * m_acc
            ix, iy = math.floor(gdx), math.floor(gdy)
            fx, fy = gdx - ix, gdy - iy
            X0, Y0 = g['x0'] + ix, g['y0'] + iy
            cx, cy = g['gx'] - g['x0'], g['gy'] - g['y0']
            prem, a, sh = _xform([prem, g['a'], g['sh']], sig, s, fx, fy, cx, cy)
            if clip_y is not None:
                yy = np.arange(a.shape[0], dtype=np.float32) + Y0
                mk = np.clip((clip_y - yy) / 2.0 + 0.5, 0, 1)[:, None]
                mks = np.clip((clip_y - yy) / (0.25 * g['size']) + 0.5, 0, 1)[:, None]
                prem, a, sh = prem * mk[..., None], a * mk, sh * mks
            if wipe_x is not None:
                xx = np.arange(a.shape[1], dtype=np.float32) + X0
                mk = np.clip((wipe_x - xx) / feather + 0.5, 0, 1)[None, :]
                mk = mk * mk * (3 - 2 * mk)
                prem, a, sh = prem * mk[..., None], a * mk, sh * mk
            drawn.append((prem, a, sh, X0, Y0, o))
        for prem, a, sh, X0, Y0, o in drawn:
            _composite_sh(frame, sh, X0, Y0, o)
        for prem, a, sh, X0, Y0, o in drawn:
            _composite_col(frame, prem, a, X0, Y0, o)
        return frame


# ================================================================ T6. Odometer
class Odometer:
    """数字が縦に回るカウンター。並びは _mapui.Counter / _bmap.HCounter と同じ（前置き＋数字＋後ろ、全体を cx の中心に）。
    数字は 1 桁ずつの列（クリップの高さ 1.15em）。列の値は floor(v/10^c) mod 10 に小数の分だけ下から次の数字が上がる。
    桁が増えるときは 0 の代わりに空白から数字が上がり、幅もなめらかに広がる。速く回る列は縦にぶれる（最大 0.4em）。
    put(frame, v, alpha, i=None, f_lock=None, dv=None)：表示する値は final を超えない。i>=f_lock で final に止まり、
    lock_drop px の back_out の落ち込み＋lock_scale→1.0 の拡大（既定 6px・1.06）でカチッと決まる。dv（1フレームの値の変化）でぶれの量を決める。"""

    def __init__(self, final, cx, y, digit_size=150, side_size=96, prefix='約', suffix='km', hl_suffix='',
                 weight='Bold', side_weight='Bold', gap=0.06, digit_color=None, side_color=None, suffix_color=None,
                 lock_dur=8, decreasing=False, lock_scale=1.06, lock_drop=6.0, hide_zero=False, blur_k=0.8,
                 feather_px=None, blur_cap=None):
        self.final_s = str(final)
        # 2026-11-08 r3（美術監督 global_rules (2)）：feather_px を渡すと「やわらかい列」：列の窓の上下を feather_px でぼかし、
        # 縦のぶれは板の外まで（切れない）なめらかな山形のカーネル、上限は数字の高さ（≈0.72em）× blur_cap（既定 0.5）。
        # 渡さないときは前のまま（03・04 と同じ見た目）。
        self.feather_px = feather_px
        self.blur_cap = 0.5 if blur_cap is None else blur_cap
        self.final = float(self.final_s.replace(',', ''))
        self.cx, self.y, self.ds, self.ss = cx, y, digit_size, side_size
        self.gap = gap * digit_size
        f = core.font(digit_size, weight)
        self.cell = max(f.getlength(str(d)) for d in range(10)) * 0.98
        self.comma_w = f.getlength(',')
        self.lock_dur, self.lock_scale, self.lock_drop = lock_dur, lock_scale, lock_drop
        self.blur_k = blur_k                            # 回る速さに対する縦のぶれの量
        self.hide_zero = hide_zero                      # True：値が 1 未満のあいだ 1 の位も空白（0 を見せない）
        self.decreasing = decreasing
        dcol = core.ACC if digit_color is None else digit_color
        scol = core.FOAM if side_color is None else side_color
        sfx_col = scol if suffix_color is None else suffix_color
        self.pitch = 1.15 * digit_size
        # 数字 0–9 とカンマ（中心 x=0、ベースライン y=0 の板）
        self.dg = {}
        for ch in '0123456789,':
            self.dg[ch] = _patch_of([dict(text=ch, x=0, y=0, size=digit_size, weight=weight, anchor='ms', track=0,
                                          color=dcol)])
        # 前置きと後ろ（後ろは hl_suffix の部分だけ ACC）
        self.pw = core.text_width(prefix, side_size, side_weight, 0.02) if prefix else 0.0
        self.sw = core.text_width(suffix, side_size, side_weight, 0.02) if suffix else 0.0
        self.pre = _patch_of([dict(text=prefix, x=0, y=0, size=side_size, weight=side_weight, anchor='rs', track=0.02,
                                   color=scol)]) if prefix else None
        self.suf = None
        if suffix:
            items = []
            if hl_suffix and suffix.startswith(hl_suffix):
                wa = core.text_width(hl_suffix, side_size, side_weight, 0.02) + 0.02 * side_size
                items.append(dict(text=hl_suffix, x=0, y=0, size=side_size, weight=side_weight, anchor='ls',
                                  track=0.02, color=core.ACC))
                rest = suffix[len(hl_suffix):]
                if rest:
                    items.append(dict(text=rest, x=wa, y=0, size=side_size, weight=side_weight, anchor='ls',
                                      track=0.02, color=sfx_col))
            else:
                items.append(dict(text=suffix, x=0, y=0, size=side_size, weight=side_weight, anchor='ls', track=0.02,
                                  color=sfx_col))
            self.suf = _patch_of(items)
        # 列（右から）：final の文字列の各文字
        self.chars = list(self.final_s)
        self.ncol = sum(ch.isdigit() for ch in self.chars)

    def _cols(self, v):
        """各桁の列の位置 p_c（小数。整数部分が数字、小数部分が次の数字への回り）と、見える度合い vis_c。"""
        v = max(0.0, float(v))
        ps, vis = [], []
        for c in range(self.ncol):
            pc = v / 10 ** c
            if c > 0:
                lower = v - math.floor(pc) * 10 ** c           # 下の桁の値（0..10^c）
                pc = math.floor(pc) + float(np.clip(lower - (10 ** c - 1), 0, 1))
            ps.append(pc)
            vis.append(1.0 if c == 0 else float(np.clip(pc, 0, 1)))
        return ps, vis

    def width_at(self, v):
        ps, vis = self._cols(v)
        w, c = 0.0, 0
        for ch in reversed(self.chars):
            if ch == ',':
                w += self.comma_w * vis[min(c, self.ncol - 1)]
            else:
                w += self.cell * vis[c]
                c += 1
        return self.pw + 2 * self.gap + w + self.sw

    def put(self, frame, v, alpha=1.0, i=None, f_lock=None, dv=None, dx=0.0, dy=0.0, scale=1.0, blur=0.0):
        if alpha <= 0.002:
            return frame
        v = float(v)
        locked_k = None
        if f_lock is not None and i is not None and i >= f_lock:
            v, dv = self.final, 0.0
            locked_k = i - f_lock
        clamp = (lambda x: max(x, self.final)) if self.decreasing else (lambda x: min(x, self.final))
        v_prev = clamp(v - dv) if dv else None
        v = clamp(v)
        ps, vis = self._cols(v)
        if dv:
            ps0, _ = self._cols(v_prev)
            spd = [abs(a - b) for a, b in zip(ps, ps0)]
        else:
            spd = [0.0] * self.ncol
        # 止まりかけの列は「カチッ」と回る（小数の部分を急な S 字に）。速く回る列はそのまま
        for c in range(self.ncol):
            fl = math.floor(ps[c])
            fr = ps[c] - fl
            kk = 1.0 + 11.0 * float(np.clip(1 - spd[c] / 0.6, 0, 1))
            if kk > 1.05 and 0 < fr < 1:
                fr = 0.5 + 0.5 * math.tanh((fr - 0.5) * kk) / math.tanh(0.5 * kk)
            ps[c] = fl + fr
            if c > 0:
                vis[c] = float(np.clip(ps[c], 0, 1))
        # 決まりの動き
        ldy, lsc = 0.0, 1.0
        if locked_k is not None:
            q = ease((locked_k + 1) / float(self.lock_dur), 'back_out')
            ldy = -self.lock_drop * (1 - q)
            lsc = self.lock_scale - (self.lock_scale - 1) * q
        # 並び（右端から左へ）
        tot = self.width_at(v)
        right = self.cx + tot / 2
        pads = 70
        top = self.y - self.pitch * 0.5 - 0.38 * self.ds - pads
        bot = self.y + self.pitch * 0.5 - 0.38 * self.ds + pads
        x_left = right - tot - pads
        x_right = right + pads
        bw, bh = int(math.ceil(x_right - x_left)), int(math.ceil(bot - top))
        BX, BY = int(math.floor(x_left)), int(math.floor(top))
        prem = np.zeros((bh, bw, 3), np.float32)
        aa = np.zeros((bh, bw), np.float32)
        sh = np.zeros((bh, bw), np.float32)

        soft = self.feather_px is not None

        def blit(P, px, py, mask_fn=None, vblur=0.0):
            pr, a, s, x0, y0 = P
            ix, iy = math.floor(px), math.floor(py)
            fx, fy = px - ix, py - iy
            if fx > 0.02 or fy > 0.02:
                pr, a, s = _xform([pr, a, s], 0.0, 1.0, fx, fy)
            if soft and vblur >= 1.0:
                # 板の上下に余白を足してから、山形（三角）のカーネルで縦にぶらす（板の端で切れない・縞にならない）
                pv = int(math.ceil(vblur)) + 2
                pr = np.pad(pr, ((pv, pv), (0, 0), (0, 0)))
                a, s = np.pad(a, ((pv, pv), (0, 0))), np.pad(s, ((pv, pv), (0, 0)))
                y0 = y0 - pv
                k = int(vblur) | 1
                ker = (1.0 - np.abs(np.linspace(-1, 1, k + 2)[1:-1]))[:, None].astype(np.float32)
                ker /= ker.sum()
                pr = cv2.filter2D(pr, -1, ker, borderType=cv2.BORDER_CONSTANT)
                a = cv2.filter2D(a, -1, ker, borderType=cv2.BORDER_CONSTANT)
                s = cv2.filter2D(s, -1, ker, borderType=cv2.BORDER_CONSTANT)
            elif vblur >= 1.5:
                k = int(vblur) | 1
                ker = np.ones((k, 1), np.float32) / k
                pr = cv2.filter2D(pr, -1, ker, borderType=cv2.BORDER_CONSTANT)
                a = cv2.filter2D(a, -1, ker, borderType=cv2.BORDER_CONSTANT)
            X0, Y0 = x0 + ix - BX, y0 + iy - BY
            h, w = a.shape
            xa, ya, xb, yb = max(X0, 0), max(Y0, 0), min(X0 + w, bw), min(Y0 + h, bh)
            if xb <= xa or yb <= ya:
                return
            sl = (slice(ya - Y0, yb - Y0), slice(xa - X0, xb - X0))
            pa, aa_, ss = pr[sl], a[sl], s[sl]
            if mask_fn is not None:
                m, ms = mask_fn(np.arange(ya, yb, dtype=np.float32) + BY)
                pa, aa_, ss = pa * m[:, None, None], aa_ * m[:, None], ss * ms[:, None]
            sh[ya:yb, xa:xb] += ss
            sub = prem[ya:yb, xa:xb]
            sub *= 1 - aa_[..., None]
            sub += pa
            aa[ya:yb, xa:xb] = aa[ya:yb, xa:xb] * (1 - aa_) + aa_

        x = right
        if self.suf is not None:
            blit(self.suf, right - self.sw, self.y)
        x = right - self.sw - self.gap
        yc = self.y - 0.38 * self.ds                       # 数字の見た目の中心
        half = self.pitch / 2
        fe = 0.10 * self.ds if not soft else float(self.feather_px)

        def colmask(yy):
            m = np.clip(np.minimum(yy - (yc - half), (yc + half) - yy) / fe, 0, 1)
            m = m * m * (3 - 2 * m)
            return m, m
        c = 0
        for ch in reversed(self.chars):
            if ch == ',':
                vv = vis[min(c, self.ncol - 1)]
                w = self.comma_w * vv
                if vv > 0.01:
                    P = self.dg[',']
                    blit((P[0] * vv, P[1] * vv, P[2] * vv, P[3], P[4]), x - w / 2, self.y)
                x -= w
                continue
            w = self.cell * vis[c]
            pc = ps[c]
            d0 = int(math.floor(pc)) % 10
            fr = pc - math.floor(pc)
            lead_blank = (c > 0 or self.hide_zero) and math.floor(pc) == 0     # 上の桁がまだ 0：空白
            vb = min(0.4 * self.ds, max(0.0, spd[c] - 0.12) * self.pitch * self.blur_k)
            if soft:
                vb = min(vb, self.blur_cap * 0.72 * self.ds)
            xc = x - w / 2
            if not lead_blank:
                blit(self.dg[str(d0)], xc, self.y - fr * self.pitch, colmask, vb)
            if fr > 0.002:
                blit(self.dg[str((d0 + 1) % 10)], xc, self.y + (1 - fr) * self.pitch, colmask, vb)
            x -= w
            c += 1
        if self.pre is not None:
            blit(self.pre, x - self.gap, self.y)
        sh = np.clip(sh, 0, 1)
        s = lsc * scale
        if abs(s - 1) > 1e-4 or blur > 0.25:
            pcx, pcy = self.cx - BX, yc - BY
            prem, aa, sh = _xform([prem, aa, sh], blur, s, 0, 0, pcx, pcy)
        oy = dy + ldy
        ix, iy = math.floor(dx), math.floor(oy)
        if (dx - ix) > 0.02 or (oy - iy) > 0.02:
            prem, aa, sh = _xform([prem, aa, sh], 0, 1, dx - ix, oy - iy)
        _composite_sh(frame, sh, BX + ix, BY + iy, alpha)
        _composite_col(frame, prem, aa, BX + ix, BY + iy, alpha)
        return frame


# ================================================================ T7. Camera
def _val(tr, i):
    if tr is None:
        return None
    if isinstance(tr, Track):
        return tr.at(i)
    if callable(tr):
        return tr(i)
    return tr


class Cam2D:
    """2D のカメラ。xy：画面上のずれ (dx, dy)、zoom：倍率、rot：回転（度・時計回り）、pivot：拡大・回転の中心（画面座標）。
    どれも 定数・Track・関数 f(i) のいずれか。画面の点 = pivot + xy + R·zoom·(世界の点 − pivot)。
    描く順：世界のレイヤー → apply（カメラ）→ 画面に固定の文字。extra=(dx, dy, rot) は揺れ（shake の戻り値）を足す。"""

    def __init__(self, xy=(0.0, 0.0), zoom=1.0, rot=0.0, pivot=(W / 2, H / 2)):
        self.xy, self.zoom, self.rot, self.pivot = xy, zoom, rot, pivot

    def M(self, i, extra=None):
        tx, ty = _val(self.xy, i)
        z = _val(self.zoom, i)
        r = math.radians(_val(self.rot, i))
        px, py = _val(self.pivot, i)
        if extra is not None:
            tx, ty, r = tx + extra[0], ty + extra[1], r + math.radians(extra[2])
        c, s = math.cos(r) * z, math.sin(r) * z
        return np.array([[c, -s, px + tx - c * px + s * py], [s, c, py + ty - s * px - c * py]], np.float64)

    def apply(self, frame, i, extra=None, size=(W, H), border=cv2.BORDER_REFLECT, M=None):
        M = self.M(i, extra) if M is None else M
        if np.allclose(M, [[1, 0, 0], [0, 1, 0]], atol=1e-5) and frame.shape[1] == size[0] and frame.shape[0] == size[1]:
            return frame
        return cv2.warpAffine(frame, M.astype(np.float32), size, flags=cv2.INTER_LINEAR, borderMode=border)

    def project(self, pt, i, clamp=core.SAFE, margin=(0, 0, 0, 0), extra=None):
        """世界の点 → 画面の点。clamp=(x0,y0,x1,y1) の内側（margin=(左,上,右,下) だけ内へ）に収める。None なら収めない。"""
        M = self.M(i, extra)
        x = M[0, 0] * pt[0] + M[0, 1] * pt[1] + M[0, 2]
        y = M[1, 0] * pt[0] + M[1, 1] * pt[1] + M[1, 2]
        if clamp is not None:
            x = min(max(x, clamp[0] + margin[0]), clamp[2] - margin[2])
            y = min(max(y, clamp[1] + margin[1]), clamp[3] - margin[3])
        return float(x), float(y)


def shake(i, f_hit, amp=9.0, tau=3.0, hz=18.0, rot_deg=0.35, seed=3):
    """当たった瞬間の揺れ (dx, dy, rot_deg)。amp·e^(−(i−f_hit)/tau)·ノイズ、3·tau フレームで 0。当たりのフレームだけに使う。"""
    k = i - f_hit
    if k < 0 or k >= 3 * tau:
        return (0.0, 0.0, 0.0)
    e = amp * math.exp(-k / tau) * (1 - k / (3 * tau)) ** 0.5
    x = k / FPS * hz
    return (e * vnoise(x + 0.37, seed), e * vnoise(x + 3.1, seed + 1), rot_deg / amp * e * vnoise(x + 7.7, seed + 2))


def zoom_through(frame, q, center, direction='out', s_out=(1.0, 1.9), s_in=(1.6, 1.0), blur=0.22, steps=12):
    """カットをまたぐズーム。direction='out'（出ていく側）：q 0→1 で center まわりに s_out[0]→s_out[1]（expo_in）、
    ズームのぼかし 0→blur。'in'（入ってくる側）：q 0→1 で s_in[0]→s_in[1]（expo_out）、ぼかし blur→0。"""
    if direction == 'out':
        e = ease(q, 'expo_in')
        s = s_out[0] + (s_out[1] - s_out[0]) * e
        b = blur * e
    else:
        e = ease(q, 'expo_out')
        s = s_in[0] + (s_in[1] - s_in[0]) * e
        b = blur * (1 - e)
    cx, cy = center
    if abs(s - 1) > 1e-4:
        M = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
    return core.zoom_blur(frame, b, cx=cx, cy=cy, steps=steps)


def whip(frame, q, axis=0, sign=-1, px=220.0, blur=100.0, direction='out'):
    """階と階のあいだの縦（axis=0）／横（axis=1）の振り。sign：中身が動く向き（−1＝上へ＝カメラは下へ振る）。
    'out'：最後の数フレーム、q 0→1 でずれ 0→sign·px（expo_in）、ぶれ 0→blur。
    'in' ：最初の数フレーム、q 0→1 でずれ −sign·px→0（expo_out）、ぶれ blur→0。同じ向きに動き続けて見える。"""
    if direction == 'out':
        e = ease(q, 'expo_in')
        d, b = sign * px * e, blur * e
    else:
        e = ease(q, 'expo_out')
        d, b = -sign * px * (1 - e), blur * (1 - e)
    if abs(d) > 0.05:
        M = np.float32([[1, 0, d if axis == 1 else 0], [0, 1, d if axis == 0 else 0]])
        frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
    return core.whip(frame, b, axis=axis)


# ================================================================ T8. Particles
class Burst:
    """閉じた式で位置が決まる粒（どのフレームからでも描ける）。f0 に origin から出る。
    p = o + v·(1−e^(−dt·drag))/drag + ½·g·dt²（dt はフレーム）。速さの向きに伸びた楕円、寿命で薄くなる。
    angle_range（度、0＝右、−90＝上）、speed_range（px/フレーム）、gravity（px/フレーム²、下が +）、life（フレーム）、
    size（半径 px）、rgb、alpha、stretch（1フレームの移動に対する伸びの割合）。"""

    def __init__(self, seed, n, origin, angle_range=(-180, 0), speed_range=(4, 10), gravity=0.6, drag=0.08,
                 life=10, size=(2.0, 4.0), f0=0, rgb=None, alpha=0.6, stretch=0.5, life_jitter=0.3):
        rng = np.random.default_rng(seed)
        self.n, self.o, self.f0 = n, np.asarray(origin, np.float64), f0
        ang = np.radians(rng.uniform(*angle_range, n))
        sp = rng.uniform(*speed_range, n)
        self.v = np.stack([np.cos(ang) * sp, np.sin(ang) * sp], 1)
        self.g, self.drag = gravity, drag
        self.life = life * (1 - life_jitter * rng.random(n))
        self.size = rng.uniform(*size, n) if np.ndim(size) else np.full(n, float(size))
        self.rgb = core.FOAM if rgb is None else rgb
        self.alpha, self.stretch = alpha, stretch

    def state(self, i):
        dt = float(i - self.f0)
        if dt < 0:
            return None
        if self.drag > 1e-6:
            k = (1 - math.exp(-dt * self.drag)) / self.drag
            kv = math.exp(-dt * self.drag)
        else:
            k, kv = dt, 1.0
        p = self.o + self.v * k + np.array([0, 0.5 * self.g * dt * dt])
        v = self.v * kv + np.array([0, self.g * dt])
        a = np.clip(1 - dt / self.life, 0, 1) ** 1.2
        return p, v, a

    def draw(self, frame, i, alpha=1.0, offset=(0.0, 0.0)):
        st = self.state(i)
        if st is None:
            return frame
        p, v, a = st
        p = p + np.asarray(offset)
        live = a > 0.01
        if not live.any():
            return frame
        P = p[live]
        x0, y0 = int(max(0, P[:, 0].min() - 40)), int(max(0, P[:, 1].min() - 40))
        x1, y1 = int(min(frame.shape[1], P[:, 0].max() + 40)), int(min(frame.shape[0], P[:, 1].max() + 40))
        if x1 <= x0 or y1 <= y0:
            return frame
        L = core.SS(s=3, w=x1 - x0, h=y1 - y0)
        for (px, py), (vx, vy), aa, r in zip(p[live], v[live], a[live], self.size[live]):
            sp = math.hypot(vx, vy)
            ln = max(r, r + sp * self.stretch)
            ang = math.degrees(math.atan2(vy, vx))
            c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            pts = []
            for t in np.linspace(0, 2 * np.pi, 14, endpoint=False):
                ex, ey = math.cos(t) * ln, math.sin(t) * r * 0.8
                pts.append((px - x0 + c * ex - s * ey, py - y0 + s * ex + c * ey))
            L.draw.polygon(L.p(pts), fill=L.col(self.rgb, float(aa) * self.alpha))
        lay = L.done()
        return put_rgba(frame, lay, x0, y0, alpha)


# ================================================================ T9. 2.5D photo parallax
_DA = None
CACHE = os.path.join(core.OUT, 'cache')


def depth_map(img, key):
    """Depth Anything V2 Small（ONNX）の近さ（大きいほど手前、0..1）。img と同じ大きさ。
    exports/v3/cache/depth_<key>_<w>x<h>.npy（float16）にキャッシュ。img は float RGB 0..1。"""
    global _DA
    tag = str(key).replace(':', '_').replace('/', '_')
    h, w = img.shape[:2]
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, 'depth_%s_%dx%d.npy' % (tag, w, h))
    if os.path.exists(cp):
        return np.load(cp).astype(np.float32)
    import onnxruntime as ort
    if _DA is None:
        _DA = ort.InferenceSession(os.path.join(core.MODELS, 'depth_anything_v2_small.onnx'),
                                   providers=['CPUExecutionProvider'])
    s = 518 / min(h, w)
    nh, nw = int(round(h * s / 14)) * 14, int(round(w * s / 14)) * 14
    x = cv2.resize(img.astype(np.float32), (nw, nh), interpolation=cv2.INTER_CUBIC)
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    name = _DA.get_inputs()[0].name
    d = _DA.run(None, {name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0]
    d = np.squeeze(d)
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float16)
    np.save(cp, d)
    return d.astype(np.float32)


def prep_depth(d, dilate=0.008, blur=0.004):
    """手前を少し太らせて（最大値フィルタ、幅の dilate 倍）縁の裂けを防ぎ、ぼかす（幅の blur 倍）。"""
    w = d.shape[1]
    k = max(1, int(round(dilate * w))) * 2 + 1
    d = cv2.dilate(d.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    s = max(0.5, blur * w)
    return cv2.GaussianBlur(d, (0, 0), s)


def force_far(d, mask):
    """mask（True／1 の所）の奥行きを 0（いちばん遠い）にする（09 の足した空など、合成した所）。"""
    d = d.copy()
    m = mask.astype(np.float32) if mask.dtype != np.float32 else mask
    return d * (1 - m)


def parallax_view(img, dep, cx, cy, vw, par=0.0, lat=0.0, lat_y=0.0, rot=0.0, d0=0.5, out_w=W, out_h=H,
                  interp=cv2.INTER_CUBIC, max_disp=18.0):
    """写真の仮想カメラ（v2/common.view の移植）。画像の (cx, cy)（0..1）を中心に、幅 vw（画像の幅の比）を out_w×out_h に写す。
    par：奥行きの視差（ドリー。手前ほど大きく）、lat / lat_y：横／縦の視差（手前ほど大きくずれる）。
    1 画素あたりの視差は出力で max_disp px まで。戻り値 (画, A)。A は出力→画像の 2×3。dep=None なら普通の切り出し。"""
    h, w = img.shape[:2]
    use = dep is not None and (par or lat or lat_y)
    m = 1.0 + ((0.5 * abs(par) + 1.2 * (abs(lat) + abs(lat_y))) if use else 0.0) + 0.01
    vh = vw * w / out_w * out_h / h
    if vh * m > 1.0:
        vw = vw / (vh * m)
        vh = 1.0 / m
    if vw * m > 1.0:
        vh = vh / (vw * m)
        vw = 1.0 / m
    cx = float(np.clip(cx, vw * m / 2, 1 - vw * m / 2))
    cy = float(np.clip(cy, vh * m / 2, 1 - vh * m / 2))
    k = vw * w / out_w
    c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    A = np.array([[k * c, -k * s, cx * w - k * (c * out_w / 2 - s * out_h / 2)],
                  [k * s, k * c, cy * h - k * (s * out_w / 2 + c * out_h / 2)]], np.float32)
    if not use:
        return cv2.warpAffine(img, A, (out_w, out_h), flags=interp | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT), A
    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    mx = A[0, 0] * xs + A[0, 1] * ys + A[0, 2]
    my = A[1, 0] * xs + A[1, 1] * ys + A[1, 2]
    dd = cv2.remap(dep, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    f = 1.0 / (1.0 + par * (dd - d0))
    pcx, pcy = cx * w, cy * h
    nx = pcx + (mx - lat * k * out_w * (dd - d0) - pcx) * f
    ny = pcy + (my - lat_y * k * out_h * (dd - d0) - pcy) * f
    lim = max_disp * k
    nx = mx + np.clip(nx - mx, -lim, lim)
    ny = my + np.clip(ny - my, -lim, lim)
    out = cv2.remap(img, nx.astype(np.float32), ny.astype(np.float32), interp, borderMode=cv2.BORDER_REFLECT)
    return out, A


def view_depth(dep, A, out_w=W, out_h=H):
    """parallax_view の A で、奥行きを出力の画角に写す（dof 用）。"""
    return cv2.warpAffine(dep, A, (out_w, out_h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


def dof(frame, dep_view, far_thresh=0.35, sigma=1.4, feather=0.12):
    """被写界深度：奥行きが far_thresh より遠い所（dep_view が小さい所）を sigma px でぼかして混ぜる。"""
    if sigma < 0.2:
        return frame
    b = cv2.GaussianBlur(frame, (0, 0), sigma)
    m = np.clip((far_thresh - dep_view) / feather + 0.5, 0, 1)[..., None]
    return frame * (1 - m) + b * m
