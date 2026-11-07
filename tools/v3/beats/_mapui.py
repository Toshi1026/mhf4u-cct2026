"""地図グループ（b03/b04/b07/b08/b09）の文字・カウンター・スピードメーターの部品。

core.text_layer は画面全体の大きさで影をぼかすので、ここでは文字のまわりだけの小さな板で作って貼る（速い）。
"""
import numpy as np

import core
from core import W, H

PAD = 64


def _extent(it):
    size = it['size']
    tw = core.text_width(it['text'], size, it.get('weight', 'Regular'), it.get('track', 0.04))
    anchor = it.get('anchor', 'ms')
    x = it['x'] - (tw / 2 if anchor[0] == 'm' else tw if anchor[0] == 'r' else 0)
    y = it['y']
    if anchor[1] == 's':
        y0, y1 = y - size * 1.15, y + size * 0.35
    elif anchor[1] == 'm':
        y0, y1 = y - size * 0.75, y + size * 0.75
    else:
        y0, y1 = y - size * 0.2, y + size * 1.3
    return x, x + tw, y0, y1


class Patch:
    """文字の小さな板。items は core.text_layer と同じ dict（座標は画面座標）。"""

    def __init__(self, items):
        items = [it for it in items if it.get('text')]
        ex = [_extent(it) for it in items]
        x0 = int(np.floor(min(e[0] for e in ex))) - PAD
        x1 = int(np.ceil(max(e[1] for e in ex))) + PAD
        y0 = int(np.floor(min(e[2] for e in ex))) - PAD
        y1 = int(np.ceil(max(e[3] for e in ex))) + PAD
        self.x0, self.y0 = x0, y0
        loc = [dict(it, x=it['x'] - x0, y=it['y'] - y0) for it in items]
        self.rgba, self.sh = core.text_layer(loc, w=x1 - x0, h=y1 - y0)

    def put(self, frame, alpha=1.0, dx=0, dy=0):
        if alpha <= 0.001:
            return frame
        h, w = self.sh.shape
        X0, Y0 = self.x0 + int(round(dx)), self.y0 + int(round(dy))
        fx0, fy0, fx1, fy1 = max(X0, 0), max(Y0, 0), min(X0 + w, W), min(Y0 + h, H)
        if fx1 <= fx0 or fy1 <= fy0:
            return frame
        px0, py0 = fx0 - X0, fy0 - Y0
        sl = (slice(fy0, fy1), slice(fx0, fx1))
        ps = (slice(py0, py0 + fy1 - fy0), slice(px0, px0 + fx1 - fx0))
        sub = frame[sl]
        sub *= 1 - self.sh[ps][..., None] * 0.55 * alpha
        a = self.rgba[ps][..., 3:4] * alpha
        sub += (self.rgba[ps][..., :3] - sub) * a
        return frame


def text(frame, items, alpha=1.0):
    return Patch(items).put(frame, alpha)


# ---------------------------------------------------------------- 数字のカウンター（等幅の見た目）
class Counter:
    """「約25,000km」のような 前置き＋数字＋後ろ の表示。数字は1文字ずつ同じ幅のマスに置き（等幅）、ACC 色。
    右端（suffix の右）を固定して、桁が増えると左へ伸びる。cx はできあがり（最終の文字列）の中心。"""

    def __init__(self, final, cx, y, digit_size=170, side_size=96, prefix='約', suffix='km', weight='Bold',
                 side_weight='Bold', gap=0.06, center=True):
        self.y, self.ds, self.ss = y, digit_size, side_size
        self.prefix, self.suffix, self.weight, self.side_weight = prefix, suffix, weight, side_weight
        f = core.font(digit_size, weight)
        self.cell = max(f.getlength(str(d)) for d in range(10)) * 0.98
        self.comma = f.getlength(',') * 1.0
        self.gap = gap * digit_size
        self.pw = core.text_width(prefix, side_size, side_weight, 0.02) if prefix else 0
        self.sw = core.text_width(suffix, side_size, side_weight, 0.02) if suffix else 0
        tot = self.pw + self.gap + self._numw(final) + self.gap + self.sw
        self.right = cx + tot / 2
        self.cx, self.center = cx, center
        self._cache = {}

    def width(self, s):
        return self.pw + 2 * self.gap + self._numw(s) + self.sw

    def _numw(self, s):
        return sum(self.comma if ch == ',' else self.cell for ch in s)

    def items(self, s, alpha=1.0):
        right = self.cx + self.width(s) / 2 if self.center else self.right
        x = right - self.sw
        out = []
        if self.suffix:
            out.append(dict(text=self.suffix, x=x, y=self.y, size=self.ss, weight=self.side_weight, anchor='ls',
                            track=0.02, alpha=alpha))
        x -= self.gap
        for ch in reversed(s):
            w = self.comma if ch == ',' else self.cell
            out.append(dict(text=ch, x=x - w / 2, y=self.y, size=self.ds, weight=self.weight, anchor='ms', track=0,
                            alpha=alpha, color=core.ACC))
            x -= w
        x -= self.gap
        if self.prefix:
            out.append(dict(text=self.prefix, x=x, y=self.y, size=self.ss, weight=self.side_weight, anchor='rs',
                            track=0.02, alpha=alpha))
        return out

    def left(self, s):
        right = self.cx + self.width(s) / 2 if self.center else self.right
        return right - self.width(s)

    def put(self, frame, s, alpha=1.0, extra=()):
        key = (s, tuple(sorted((k, str(v)) for e in extra for k, v in e.items())))
        p = self._cache.get(key)
        if p is None:
            p = Patch(self.items(s) + list(extra))
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = p
        return p.put(frame, alpha)


def fmt(v, step=1):
    v = int(round(v / step) * step)
    return '{:,}'.format(v)


# ---------------------------------------------------------------- 文字の下に敷く暗い帯（読みやすさ）
def scrim(frame, y0, y1, strength=0.5, feather=160):
    """y0..y1 を中心に、上下にぼかした暗い帯（NAVY へ寄せる）。"""
    yy = np.arange(H, dtype=np.float32)
    a = np.clip(np.minimum((yy - (y0 - feather)) / feather, ((y1 + feather) - yy) / feather), 0, 1)
    a = (a * a * (3 - 2 * a)) * strength
    lo, hi = int(max(0, y0 - feather)), int(min(H, y1 + feather))
    sub = frame[lo:hi]
    sub += (core.NAVY - sub) * a[lo:hi, None, None]
    return frame


# ---------------------------------------------------------------- スピードメーター（b08 で描き、b09 では小さなアイコンとして残す）
ICON_POS = (150, 400)       # 小さくなったメーターの中心（左上、CG ラベルの下）
ICON_SCALE = 0.20


def speedometer(L, cx, cy, r, value, scale=1.0, vmax=60, alpha=1.0, labels=True, label_items=None):
    """L：core.SS。半円（左 0 → 右 vmax）。value の位置に針。目盛りの数字は label_items（list）に dict を追加する。"""
    d = L.draw
    col = L.col(core.FOAM, 0.85 * alpha)
    dim = L.col(core.FOAM, 0.45 * alpha)
    r = r * scale
    # 弧
    box = L.box(cx - r, cy - r, cx + r, cy + r)
    d.arc(box, 180, 360, fill=col, width=L.w(max(1.0, 2.2 * scale)))
    # 進んだ分の弧（ACC は使わず、明るい空色）
    if value > 0:
        a1 = 180 + 180 * min(value, vmax) / vmax
        rr = r - 14 * scale
        d.arc(L.box(cx - rr, cy - rr, cx + rr, cy + rr), 180, a1, fill=L.col(core.SKY, 0.55 * alpha),
              width=L.w(max(1.0, 7 * scale)))
    for v in range(0, vmax + 1, 5):
        th = np.pi * (1 - v / vmax)
        major = v % 10 == 0
        r0 = r - (30 if major else 18) * scale
        x0, y0 = cx + r0 * np.cos(th), cy - r0 * np.sin(th)
        x1, y1 = cx + r * np.cos(th), cy - r * np.sin(th)
        d.line(L.p([(x0, y0), (x1, y1)]), fill=col if major else dim, width=L.w(max(1.0, (2.4 if major else 1.4) * scale)))
        if major and labels and label_items is not None:
            rl = r - 62 * scale
            label_items.append(dict(text=str(v), x=cx + rl * np.cos(th), y=cy - rl * np.sin(th) + 11 * scale,
                                    size=max(10, 32 * scale), weight='Medium', alpha=0.8 * alpha, track=0))
    # 針
    th = np.pi * (1 - min(value, vmax) / vmax)
    rn = r - 44 * scale
    tipx, tipy = cx + rn * np.cos(th), cy - rn * np.sin(th)
    bx, by = cx - 26 * scale * np.cos(th), cy + 26 * scale * np.sin(th)
    nx, ny = -np.sin(th), -np.cos(th)
    hw = 5 * scale
    d.polygon(L.p([(bx + nx * hw, by + ny * hw), (tipx, tipy), (bx - nx * hw, by - ny * hw)]),
              fill=L.col(core.FOAM, alpha))
    hr = 13 * scale
    d.ellipse(L.box(cx - hr, cy - hr, cx + hr, cy + hr), fill=L.col(core.FOAM, alpha))
    hr2 = 5 * scale
    d.ellipse(L.box(cx - hr2, cy - hr2, cx + hr2, cy + hr2), fill=L.col(core.DEEP, alpha))


METER_R = 340


def meter_layer(cx, cy, value, scale=1.0, alpha=1.0, labels=True):
    """メーターだけの小さな板 (rgba, x0, y0, label_items)。label_items は画面座標の数字（Patch で貼る）。"""
    r = METER_R * scale
    pad = 24 * max(scale, 0.3) + 6
    x0, y0 = int(np.floor(cx - r - pad)), int(np.floor(cy - r - pad))
    w, h = int(np.ceil(2 * r + 2 * pad)) + 1, int(np.ceil(r + 30 * scale + 2 * pad)) + 1
    L = core.SS(s=3, w=w, h=h)
    items = []
    speedometer(L, cx - x0, cy - y0, METER_R, value, scale=scale, alpha=alpha, labels=labels, label_items=items)
    for it in items:
        it['x'] += x0
        it['y'] += y0
    return L.done(), x0, y0, items


def paste(f, rgba, x0, y0, alpha=1.0):
    h, w = rgba.shape[:2]
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if X1 <= X0 or Y1 <= Y0 or alpha <= 0:
        return f
    a = rgba[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    sub = f[Y0:Y1, X0:X1]
    sub += (a[..., :3] - sub) * (a[..., 3:4] * alpha)
    return f


def icon_layer(alpha=1.0, value=40):
    """b08 の最後に残る小さなメーター（ICON_POS・ICON_SCALE）。(rgba, x0, y0)。b09 で同じものを使う。"""
    rgba, x0, y0, _ = meter_layer(ICON_POS[0], ICON_POS[1], value, scale=ICON_SCALE, alpha=alpha, labels=False)
    return rgba, x0, y0
