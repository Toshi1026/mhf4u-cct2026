"""b10（29.3–31.7秒）「答え。約26日。」
濃紺（NAVY）の無地に、ごく弱い周辺減光。「答え」が「コ」で、「約25日」が「ヤ」で出て、
「ニジュウロク」の「ロ」で 25→26 にぱちんと変わる（1.04→1.0 の小さな拡大の戻り）。静かに。
最後の 8 フレーム：文字を消しながら、土佐湾の細い線画をごくうすく出し始める（b11 は NAVY から始まる）。
"""
import cv2
import numpy as np

import core
import timeline
from core import W, H
from beats import _tosa

WIN = 8


def _background():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((x - W / 2) / (W * 0.62)) ** 2 + ((y - 800) / (H * 0.55)) ** 2
    v = np.clip(r2, 0, 1.6)
    bg = core.NAVY[None, None] * (1.0 + 0.10 * (1 - np.clip(v, 0, 1))[..., None]) * \
        (1 - 0.18 * np.clip(v - 0.6, 0, 1)[..., None])
    # 中心をほんの少しだけ海の色へ
    bg = bg + (core.DEEP - core.NAVY)[None, None] * (0.07 * np.exp(-r2 * 1.2))[..., None]
    return bg.astype(np.float32)


class Block:
    """文字の小さな板（RGBA と影）。scale で中心から拡大して重ねる。"""

    def __init__(self, items, pad=70):
        ex = []
        for it in items:
            tw = core.text_width(it['text'], it['size'], it.get('weight', 'Regular'), it.get('track', 0.04))
            ex.append((it['x'] - tw / 2, it['x'] + tw / 2, it['y'] - it['size'] * 1.15, it['y'] + it['size'] * 0.35))
        self.x0 = int(min(e[0] for e in ex)) - pad
        self.y0 = int(min(e[2] for e in ex)) - pad
        x1 = int(max(e[1] for e in ex)) + pad
        y1 = int(max(e[3] for e in ex)) + pad
        loc = [dict(it, x=it['x'] - self.x0, y=it['y'] - self.y0) for it in items]
        t, sh = core.text_layer(loc, w=x1 - self.x0, h=y1 - self.y0)
        self.rgba, self.sh = t, sh
        self.w, self.h = x1 - self.x0, y1 - self.y0

    def put(self, f, alpha=1.0, scale=1.0, cx=None, cy=None):
        if alpha <= 0.002:
            return f
        t, sh = self.rgba, self.sh
        if abs(scale - 1) > 1e-4:
            cx = self.w / 2 if cx is None else cx - self.x0
            cy = self.h / 2 if cy is None else cy - self.y0
            M = np.float32([[scale, 0, cx * (1 - scale)], [0, scale, cy * (1 - scale)]])
            t = cv2.warpAffine(t, M, (self.w, self.h), flags=cv2.INTER_LINEAR)
            sh = cv2.warpAffine(sh, M, (self.w, self.h), flags=cv2.INTER_LINEAR)
        sub = f[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]
        sub *= 1 - sh[..., None] * 0.55 * alpha
        a = t[..., 3:4] * alpha
        sub += (t[..., :3] - sub) * a
        return f


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[0][0] == 'コ' and mo[3][0] == 'ヤ' and mo[8][0] == 'ロ', mo
    f_ko = timeline.mora_frame(beat, 0)
    f_ya = timeline.mora_frame(beat, 3)
    f_ro = timeline.mora_frame(beat, 8)
    bg = _background()
    lines = _tosa.contours()
    Y_ANS, Y_NUM, SZ = 600, 830, 220
    t_ans = Block([dict(text='答え', x=W / 2, y=Y_ANS, size=48, weight='Medium', track=0.20)])
    num = {v: Block([dict(text='約%d日' % v, x=W / 2, y=Y_NUM, size=SZ, weight='Bold', track=0.02, hl=str(v))])
           for v in (25, 26)}
    cy_num = Y_NUM - SZ * 0.36                       # 拡大の中心（文字の見た目の中心）
    for i in range(n):
        f = bg.copy()
        j = i - (n - WIN)
        q = (j + 1) / WIN if j >= 0 else 0.0
        # 最後：土佐湾の線画をごくうすく（線の最大でも 0.16）
        if q > 0:
            la = (0.16 * core.ease(q, 'in') * lines)[..., None]
            f = f * (1 - la) + core.FOAM * la
        out = 1 - core.ease(q)
        # 「答え」
        a = core.ease((i - f_ko + 1) / 6) * out
        f = t_ans.put(f, a)
        # 「約25日」→「約26日」
        if i >= f_ya:
            a = core.ease((i - f_ya + 1) / 5) * out
            if i < f_ro:
                f = num[25].put(f, a)
            else:
                k = core.ease((i - f_ro) / 6, 'out')
                f = num[26].put(f, a, scale=1.04 - 0.04 * k, cx=W / 2, cy=cy_num)
        yield np.clip(f, 0, 1).astype(np.float32)
