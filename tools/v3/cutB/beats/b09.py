"""案B 09（約2.3秒）「目の前に建つのが、Seaside MAZE。」 写真：建物の外観（昼、切妻のロゴ）。

外観の写真（3:2 の横）から、切妻とロゴ（MAZE・南風）の縦の窓を取る。
- 庇の看板（英文の一部が切れて読めてしまう）より上で切る。
- 上は、写真の空の上端を鏡写しにして強くぼかした空を足す（曇り空なので継ぎ目は見えない）。
  これで屋根の頂点が文字の帯より下に来て、空に文字を置ける。
- 動き（2026-11-07 動きの改訂）：奥行き（Depth Anything V2）で 2.5D のドリー。建物が空に対して少し大きくなり、
  わずかに上へせり上がる（lat_y）。足した空は奥行き 0（いちばん遠い）。曲線は cubic-bezier(0.4,0,0.6,1) を
  最後のフレームでもピークの 35% の速さが残るところで切ったもの（止まらずにカットへ）。
- 最後の 4 フレームは 10 へ縦の振り（カメラが下へ＝中身が上へ）。10 の最初の 4 フレームが同じ速さで受ける。
- 曇り空は明るい灰色で白い文字が沈むので、空だけを上から ND グラデーションのように落とす（屋根より上だけ）。
テロップ：「Seaside MAZE」（ACC、Bold）が「シ」の 2 フレーム前から 1 文字ずつ上がる（字間 0.12em→0.03em）。
「（南風・まぜ）」は「マ」の 1 フレーム前から行ごと下のマスクから上がる。
"""
import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W, H
from cutB.beats import _fac as F
from cutB.beats import _motion as Mo
from cutB.beats import _p25 as P

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


TAIL, TAIL_DIM = 6, 0.10          # 振りの最後のフレームは下の明るい壁が映り込むので、旧 0.08 より少し深く
VW0, VW1 = 0.975, 0.91      # 窓の幅（縦いっぱいの幅に対する比）
PAR1, LATY1 = 0.06, -0.008  # 奥行きのドリーと、建物のせり上がり（最後の値。0 から曲線どおりに増える）
CY1 = 0.488
CURVE = P.bez_tail(0.4, 0, 0.6, 1, 0.35)


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[9][0] == 'シ' and mo[14][0] == 'マ', mo
    f_si = round(mo[9][1] * core.FPS) - beat['f0']
    f_ma = round(mo[14][1] * core.FPS) - beat['f0']
    img, p = _image()
    h, w = img.shape[:2]
    vmax = F.max_vw(img)
    dep = P.depth(img, 'b09ext', vw=vmax)
    dep[:p + 8] = 0.0                                          # 足した空（鏡写し）はいちばん遠い
    lx, ly = LOGO[0], (LOGO[1] * (h - p) + p) / h
    kb = Mo.KText([dict(text='Seaside MAZE', x=W / 2, y=F.Y_BIG, size=88, weight='Bold', hl='Seaside MAZE', track=0.03,
                        shadow=0.9)])
    ks = Mo.KText([dict(text='（南風・まぜ）', x=W / 2, y=F.Y_SUB, size=46, weight='Medium', track=0.06, shadow=0.9)])
    y = np.arange(H, dtype=np.float32)
    t = np.clip((y - 250) / 520.0, 0, 1)
    k = (1 - t * t * (3 - 2 * t))[:, None, None]
    shade = 1 - k * np.array([0.46, 0.42, 0.34], np.float32)       # 上ほど暗く、ほんの少し青く（約0.6段）、y≈770 で 0
    for i in range(n):
        u = CURVE(i / (n - 1))
        vw = vmax * core.lerp(VW0, VW1, u)
        cy = core.lerp(0.5, CY1, u)                            # わずかに上へ
        cx = core.lerp(lx + 0.004, lx, u)
        f, _ = Mo.parallax_view(img, dep, cx, cy, vw, par=PAR1 * u, lat_y=LATY1 * u)
        f = f * shade
        wa = 1 - P.whip_q(i, n)                                 # 振りの最後のフレームでは文字は消えている（鏡写しの残像を出さない）
        f = kb.reveal(f, i, f_si - 2, 'char_rise', stagger=1.2, dur=10, track_from=0.12, alpha=wa)
        f = ks.reveal(f, i, f_ma - 1, 'line_mask', alpha=wa)
        if i >= n - TAIL:                                      # 10（ローソンの入口）は暗いので、カットの段差を半分に
            f = f * (1 - TAIL_DIM * core.ease((i - (n - TAIL) + 1) / TAIL))
        f = P.whip_out(f, i, n, -1)                            # 10 へ：カメラが下へ振る
        yield np.clip(f, 0, 1).astype(np.float32)
