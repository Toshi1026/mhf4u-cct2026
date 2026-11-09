"""案B 08〜13（実写と写真）の動きの共通部品（director's motion spec §T3 / §T7 / §T9 を 08〜13 用にまとめたもの）。

- depth(img, key, vw)：Depth Anything V2 Small の奥行き（_motion.depth_map のキャッシュ）を、手前を少し太らせてぼかす。
  太らせる量・ぼかす量は「画面に写る窓の幅」に対する比で決める（写真の一部しか写さない 10 のローソンでも同じ見え方に）。
  重い処理なので縮めた奥行きで行い、元の大きさに戻す。
- hermite(t, v0, v1)：0→1 の進み具合。始まりの速さ v0・終わりの速さ v1（どちらも「平均の速さ」に対する比）。
  前のショットの速さを受け継いで始まり（carry）、次のカットの前でも止まらない（速さを持ったままカット）ための曲線。
- bez_tail(x1, y1, x2, y2, frac)：CSS の cubic-bezier を、最後のフレームでもピークの速さの frac 倍が残るところで切って 0→1 に直したもの。
- whip_out / whip_in：階と階のあいだの縦の振り（最後の 4 フレームと次の最初の 4 フレーム）。ずれ ±220px、ぶれ 0→100→0。
  sign＝中身が動く向き：−1＝上へ（カメラは下へ振る：09→10）、+1＝下へ（カメラは上へ振る：10→11、11→12）。
"""
import functools

import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
from cutB.beats import _motion as M

WHIP_N, WHIP_PX, WHIP_BLUR = 4, 220.0, 100.0


def depth(img, key, vw=1.0, dilate=0.008, blur=0.004, small=1400):
    d = M.depth_map(img, key)
    h, w = d.shape
    s = min(1.0, small / max(h, w))
    ds = cv2.resize(d, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else d
    ds = M.prep_depth(ds, dilate=dilate * vw, blur=blur * vw)
    if s < 1:
        ds = cv2.resize(ds, (w, h), interpolation=cv2.INTER_LINEAR)
    return ds.astype(np.float32)


def hermite(t, v0=0.0, v1=0.0):
    """t∈[0,1] → u（u(0)=0, u(1)=1, u'(0)=v0, u'(1)=v1）。v0=v1=0 は smoothstep。"""
    t = min(max(float(t), 0.0), 1.0)
    t2, t3 = t * t, t * t * t
    return (t3 - 2 * t2 + t) * v0 + (-2 * t3 + 3 * t2) + (t3 - t2) * v1


def hermite_d(t, v0=0.0, v1=0.0):
    t = min(max(float(t), 0.0), 1.0)
    return (3 * t * t - 4 * t + 1) * v0 + (6 * t - 6 * t * t) + (3 * t * t - 2 * t) * v1


@functools.lru_cache(maxsize=16)
def _bez_T(x1, y1, x2, y2, frac):
    f = M.cubic_bezier(x1, y1, x2, y2)
    ts = np.linspace(0, 1, 2001)
    ys = np.array([f(t) for t in ts])
    dv = np.gradient(ys, ts)
    pk = int(np.argmax(dv))
    after = np.nonzero(dv[pk:] <= frac * dv[pk])[0]
    T = ts[pk + after[0]] if len(after) else 1.0
    return float(T), float(f(T))


def bez_tail(x1, y1, x2, y2, frac=0.35):
    """u(t)：cubic-bezier を最後の速さがピークの frac 倍のところで切り、0→1 に直した曲線。"""
    f = M.cubic_bezier(x1, y1, x2, y2)
    T, yT = _bez_T(x1, y1, x2, y2, frac)
    return lambda t: f(min(max(float(t), 0.0), 1.0) * T) / yT


def whip_out(frame, i, n, sign):
    """最後の WHIP_N フレーム：ずれ 0→sign·220（expo_in）、ぶれ 0→100。最後のフレームがいちばん速い。"""
    k = i - (n - WHIP_N)
    if k < 0:
        return frame
    return M.whip(frame, (k + 1) / WHIP_N, axis=0, sign=sign, px=WHIP_PX, blur=WHIP_BLUR, direction='out')


def whip_in(frame, i, sign):
    """最初の WHIP_N フレーム：ずれ −sign·220→0（expo_out）、ぶれ 100→0。0 フレーム目は前の最後と同じ速さ。"""
    if i >= WHIP_N:
        return frame
    return M.whip(frame, i / WHIP_N, axis=0, sign=sign, px=WHIP_PX, blur=WHIP_BLUR, direction='in')


def whip_q(i, n=None):
    """振りの強さ 0..1（文字を振りといっしょに薄くするため）。n を渡すと出口、渡さないと入口。"""
    if n is not None:
        k = i - (n - WHIP_N)
        return 0.0 if k < 0 else M.ease((k + 1) / WHIP_N, 'expo_in')
    return 0.0 if i >= WHIP_N else 1 - M.ease(i / WHIP_N, 'expo_out')
