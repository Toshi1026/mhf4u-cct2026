"""案B 09（約2.3秒）「目の前に建つのが、Seaside MAZE。」 写真：建物の外観（昼、切妻のロゴ）。

外観の写真（3:2 の横）から、切妻とロゴ（MAZE・南風）の縦の窓を取る。
- 庇の看板（英文の一部が切れて読めてしまう）より上で切る。
- 上は、写真の空の上端を鏡写しにして強くぼかした空を足す（曇り空なので継ぎ目は見えない）。
  これで屋根の頂点が文字の帯より下に来て、空に文字を置ける。
- 動き：ロゴへゆっくり寄りながら、わずかに上へティルト。
- 曇り空は明るい灰色で白い文字が沈むので、空だけを上から ND グラデーションのように落とす（屋根より上だけ）。
テロップ：「Seaside MAZE」（ACC、Bold）が「シ」で、「（南風・まぜ）」が「マ」で出る。
"""
import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W, H
from cutB.beats import _fac as F

BOTTOM = 0.765          # 庇の看板の上端（写真の高さの比）より上で切る
PAD = 0.15              # 上に足す空（写真の高さの比）
LOGO = (0.468, 0.50)    # ロゴの中心（写真の比）


def _image():
    a = F.load(F.EXTERIOR, 3600)
    h, w = a.shape[:2]
    a = a[:int(h * BOTTOM)]
    p = int(h * PAD)
    top = a[:p][::-1].copy()                                   # 空の上端を鏡写し
    soft = cv2.GaussianBlur(top, (0, 0), 10)
    hard = cv2.GaussianBlur(top, (0, 0), 45)
    d = (np.arange(p, dtype=np.float32)[::-1] / p)[:, None, None]     # 0＝継ぎ目、1＝上端
    k = np.clip(d * 1.6, 0, 1)
    ext = soft * (1 - k) + hard * k
    ext = ext * (1 - 0.06 * d)                                  # 上ほどほんの少し暗く（空の自然な階調）
    out = np.concatenate([ext, a], 0)
    # 継ぎ目の数行を縦にだけ均す
    s = p
    band = out[s - 24:s + 24]
    out[s - 24:s + 24] = cv2.GaussianBlur(band, (1, 0), sigmaX=0.1, sigmaY=6)
    return F.grade_photo(out), p


TAIL, TAIL_DIM = 6, 0.08


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[9][0] == 'シ' and mo[14][0] == 'マ', mo
    f_si = round(mo[9][1] * core.FPS) - beat['f0']
    f_ma = round(mo[14][1] * core.FPS) - beat['f0']
    img, p = _image()
    h, w = img.shape[:2]
    vmax = F.max_vw(img)
    lx, ly = LOGO[0], (LOGO[1] * (h - p) + p) / h
    tb, ts = F.heading2('Seaside MAZE', '（南風・まぜ）', hl='Seaside MAZE', sz_big=88, sz_sub=46)
    y = np.arange(H, dtype=np.float32)
    t = np.clip((y - 250) / 520.0, 0, 1)
    k = (1 - t * t * (3 - 2 * t))[:, None, None]
    shade = 1 - k * np.array([0.46, 0.42, 0.34], np.float32)       # 上ほど暗く、ほんの少し青く（約0.6段）、y≈770 で 0
    for i in range(n):
        u = core.ease(i / (n - 1), 'out')
        vw = vmax * core.lerp(1.0, 0.93, u)
        cy = core.lerp(0.5, 0.5 - 0.012, u)                    # わずかに上へ
        cx = core.lerp(lx + 0.004, lx, u)
        f = F.view(img, cx, cy, vw)
        f = f * shade
        f = tb.put(f, F.appear(i, f_si, 5))
        f = ts.put(f, F.appear(i, f_ma, 5))
        if i >= n - TAIL:                                      # 10（ローソンの入口）は暗いので、カットの段差を半分に
            f = f * (1 - TAIL_DIM * core.ease((i - (n - TAIL) + 1) / TAIL))
        yield np.clip(f, 0, 1).astype(np.float32)
