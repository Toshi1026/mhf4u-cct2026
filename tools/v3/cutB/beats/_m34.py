"""03・04 の小さな部品（共有の _motion.py にないものだけ）。

- blend_mask：マスクの範囲だけ色を寄せる（画面全体を掛け算しないので速い）
- Board：前乗せアルファの小さな板。04 のメーター（線＋目盛りの数字）を1枚にまとめ、拡大・ぼかし・フェードを板の上でする。
"""
import math

import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core


# ---------------------------------------------------------------- 範囲を絞った合成
def blend_mask(frame, m, col, strength=1.0):
    """frame（H,W,3）を、マスク m（H,W）の範囲だけ col に寄せる（画面全体を掛け算しないので速い）。"""
    ys = np.flatnonzero(m.max(axis=1) > 1e-3)
    if not len(ys):
        return frame
    y0, y1 = ys[0], ys[-1] + 1
    sub = m[y0:y1]
    xs = np.flatnonzero(sub.max(axis=0) > 1e-3)
    x0, x1 = xs[0], xs[-1] + 1
    mm = sub[:, x0:x1, None] * strength
    reg = frame[y0:y1, x0:x1]
    reg += (np.asarray(col, np.float32) - reg) * mm
    return frame


# ---------------------------------------------------------------- 前乗せアルファの小さな板
class Board:
    """rgb：前乗せ（h,w,3）、a：（h,w）、(x0, y0)：画面での左上。"""

    def __init__(self, rgb, a, x0, y0):
        self.rgb, self.a, self.x0, self.y0 = rgb.astype(np.float32), a.astype(np.float32), float(x0), float(y0)

    @staticmethod
    def from_straight(rgba, x0, y0):
        a = rgba[..., 3]
        return Board(rgba[..., :3] * a[..., None], a, x0, y0)

    @staticmethod
    def from_patch(p, alpha=1.0, color_rgba=None):
        """U.Patch（文字＋影）を板に。影は黒（前乗せの rgb に 0）、強さは Patch.put と同じ 0.55。"""
        t = p.rgba if color_rgba is None else color_rgba
        s = p.sh * 0.55 * alpha
        ta = t[..., 3] * alpha
        a = ta + s * (1 - ta)
        return Board(t[..., :3] * ta[..., None], a, p.x0, p.y0)

    @staticmethod
    def empty(x0, y0, w, h):
        return Board(np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32), x0, y0)

    def copy(self):
        return Board(self.rgb.copy(), self.a.copy(), self.x0, self.y0)

    def over(self, b, alpha=1.0, dx=0.0, dy=0.0):
        """板 b をこの板の上に重ねる（整数の位置）。"""
        X, Y = int(round(b.x0 + dx - self.x0)), int(round(b.y0 + dy - self.y0))
        h, w = b.a.shape
        H0, W0 = self.a.shape
        fx0, fy0, fx1, fy1 = max(X, 0), max(Y, 0), min(X + w, W0), min(Y + h, H0)
        if fx1 <= fx0 or fy1 <= fy0:
            return self
        ps = (slice(fy0 - Y, fy1 - Y), slice(fx0 - X, fx1 - X))
        sl = (slice(fy0, fy1), slice(fx0, fx1))
        ba = b.a[ps] * alpha
        self.rgb[sl] = b.rgb[ps] * alpha + self.rgb[sl] * (1 - ba[..., None])
        self.a[sl] = ba + self.a[sl] * (1 - ba)
        return self

    def blur(self, sx, sy=None):
        sy = sx if sy is None else sy
        if max(sx, sy) < 0.3:
            return self
        pad = int(math.ceil(3 * max(sx, sy))) + 1
        rgb = cv2.copyMakeBorder(self.rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        a = cv2.copyMakeBorder(self.a, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        rgb = cv2.GaussianBlur(rgb, (0, 0), sigmaX=max(sx, 0.01), sigmaY=max(sy, 0.01))
        a = cv2.GaussianBlur(a, (0, 0), sigmaX=max(sx, 0.01), sigmaY=max(sy, 0.01))
        return Board(rgb, a, self.x0 - pad, self.y0 - pad)

    def vblur(self, px):
        """縦の動きのぼかし（箱形、長さ px）。"""
        k = int(round(px))
        if k < 2:
            return self
        pad = k
        rgb = cv2.copyMakeBorder(self.rgb, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
        a = cv2.copyMakeBorder(self.a, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
        return Board(cv2.blur(rgb, (1, k)), cv2.blur(a, (1, k)), self.x0, self.y0 - pad)

    def affine(self, s=1.0, pivot=None, dx=0.0, dy=0.0, rot=0.0):
        """画面座標の pivot まわりに s 倍・rot 度、(dx, dy) ずらした新しい板（小数の位置もなめらか）。"""
        h, w = self.a.shape
        if pivot is None:
            pivot = (self.x0 + w / 2, self.y0 + h / 2)
        px, py = pivot
        c, sn = math.cos(math.radians(rot)) * s, math.sin(math.radians(rot)) * s
        # 画面での四隅の行き先 → 新しい板の範囲
        corners = np.array([[self.x0, self.y0], [self.x0 + w, self.y0], [self.x0, self.y0 + h], [self.x0 + w, self.y0 + h]])
        q = np.stack([c * (corners[:, 0] - px) - sn * (corners[:, 1] - py) + px + dx,
                      sn * (corners[:, 0] - px) + c * (corners[:, 1] - py) + py + dy], 1)
        nx0, ny0 = math.floor(q[:, 0].min()) - 1, math.floor(q[:, 1].min()) - 1
        nw, nh = int(math.ceil(q[:, 0].max())) + 2 - nx0, int(math.ceil(q[:, 1].max())) + 2 - ny0
        # 板の画素 (u,v) → 画面 (x0+u, y0+v) → 変換 → 新しい板 (X−nx0, Y−ny0)
        M = np.array([[c, -sn, 0.0], [sn, c, 0.0]])
        tx = -(c * px - sn * py) + px + dx
        ty = -(sn * px + c * py) + py + dy
        M[0, 2] = c * self.x0 - sn * self.y0 + tx - nx0
        M[1, 2] = sn * self.x0 + c * self.y0 + ty - ny0
        M = M.astype(np.float32)
        rgb = cv2.warpAffine(self.rgb, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a = cv2.warpAffine(self.a, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return Board(rgb, a, nx0, ny0)

    def clip_rows(self, y_max=None, y_min=None, feather=0.0):
        """画面の y で切る（y_max より下を消す）。"""
        h = self.a.shape[0]
        yy = self.y0 + np.arange(h, dtype=np.float32) + 0.5
        m = np.ones(h, np.float32)
        if y_max is not None:
            m *= np.clip((y_max - yy) / max(feather, 1.0) + (0.5 if feather <= 0 else 0), 0, 1)
        if y_min is not None:
            m *= np.clip((yy - y_min) / max(feather, 1.0) + (0.5 if feather <= 0 else 0), 0, 1)
        return Board(self.rgb * m[:, None, None], self.a * m[:, None], self.x0, self.y0)

    def put(self, frame, alpha=1.0, dx=0.0, dy=0.0):
        if alpha <= 0.001:
            return frame
        b = self
        fx, fy = dx - round(dx), dy - round(dy)
        if abs(fx) > 0.06 or abs(fy) > 0.06:
            b = self.affine(1.0, (0, 0), fx, fy)
            dx, dy = dx - fx, dy - fy
        X, Y = int(round(b.x0 + dx)), int(round(b.y0 + dy))
        h, w = b.a.shape
        fx0, fy0, fx1, fy1 = max(X, 0), max(Y, 0), min(X + w, frame.shape[1]), min(Y + h, frame.shape[0])
        if fx1 <= fx0 or fy1 <= fy0:
            return frame
        ps = (slice(fy0 - Y, fy1 - Y), slice(fx0 - X, fx1 - X))
        sub = frame[fy0:fy1, fx0:fx1]
        sub *= 1 - b.a[ps][..., None] * alpha
        sub += b.rgb[ps] * alpha
        return frame
