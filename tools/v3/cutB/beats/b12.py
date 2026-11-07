"""案B 12（約3.0秒）「屋上と、避難タワーから、太平洋。」 実写：屋上・タワーから見た水平線と河口（IMG_9660、0.5倍スロー）。

素材はほぼ固定の縦 1x。上寄りに 1.72 倍で切り出して、下の柵・遊歩道・車を外す（_fac.DECK）。
画面の下に河口（川・砂州）と外海、上に広い空。水平線の傾きは回転で直す。13 は同じ素材のこの続きから始まる。
テロップ（空の帯）は2画面：
  「オ（屋上）」：「3F・屋上」（屋上＝ACC）／「展望スペース」
  「ヒ（避難）」：「津波避難タワー」（タワー＝ACC）／「平時も開放・どなたでも上まで」（LP の言い回し。開放は 9〜18 時なので「いつでも」とは書かない）
1画面目は2画面目が出る直前の 4 フレームで消える（速めに立ち上げて、止まって読める間を作る）。2画面目は最後の 6 フレームで消えて、13 の施設名へ渡す。
空は明るいので、上から ND グラデーションのように落とす（13 も同じ）。
11 からのカットの明るさの段差を抑えるため、最初の数フレームは少し暗い所から戻す（カメラの露出が合うように）。
2026-11-07 動きの改訂：
 - 最初の 4 フレームは 11 からの縦の振り（カメラが上へ）を受ける。露出の戻り（0.80→1.0）は振りの中にたたむ（6 フレーム）。
 - カメラ：12 と 13 を通して 1 本の曲線（寄り 1.00→1.05、水平線の少し上を中心に、in_out_sine）。素材は続いているので、
   カットで動きが始まり直さない（cam(g) の g は 12 の頭からの通しのフレーム）。
 - 文字：2 画面とも、見出しは 1 文字ずつ上がって（強調の語は FOAM→ACC）、下の行は行ごとマスクから上がる。
   1 画面目は「ヒ」の前にぼかしながら上へ抜け、2 画面目は最後の 6 フレームで同じように抜けて 13 の施設名へ渡す。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import H
from beats import _real
from cutB.beats import _fac as F
from cutB.beats import _motion as Mo
from cutB.beats import _p25 as P
import cv2


RAMP = 6
CAM_Z, CAM_PIVOT = 1.05, (core.W / 2, 1180.0)      # 12〜13 を通した寄り（水平線 y≈1280 の少し上を中心に）


def sky_shade():
    """空を上から落とす（約0.5段）。水平線（y≈1280）より上だけ、y≈900 で 0。12 と 13 で共通。"""
    y = np.arange(H, dtype=np.float32)
    t = np.clip((y - 200) / 700.0, 0, 1)
    return (1 - 0.30 * (1 - t * t * (3 - 2 * t)))[:, None, None]


def cam_total():
    import timeline_b
    bs = {b['key']: b for b in timeline_b.load()}
    return bs['12']['n'] + bs['13']['n']


def cam(f, g, total):
    """12 の頭からの通しのフレーム g のカメラ（12 と 13 で共通、1 本の曲線）。"""
    z = 1 + (CAM_Z - 1) * Mo.ease(g / (total - 1), 'sine_inout')
    if z < 1 + 1e-5:
        return f
    px, py = CAM_PIVOT
    Mt = np.float32([[z, 0, px * (1 - z)], [0, z, py * (1 - z)]])
    return cv2.warpAffine(f, Mt, (f.shape[1], f.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def _items(big, sub, hl):
    kb = Mo.KText([dict(text=big, x=core.W / 2, y=F.Y_BIG, size=F.SZ_BIG, weight='Bold', hl=hl, track=0.03, shadow=0.9)])
    ks = Mo.KText([dict(text=sub, x=core.W / 2, y=F.Y_SUB, size=F.SZ_SUB, weight='Medium', track=0.06, shadow=0.9)])
    return kb, ks


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[0][0] == 'オ' and mo[5][0] == 'ヒ', mo
    f_o, f_hi = [round(mo[k][1] * core.FPS) - beat['f0'] for k in (0, 5)]
    ta, tas = _items('3F・屋上', '展望スペース', '屋上')
    tb, tbs = _items('津波避難タワー', '平時も開放・どなたでも上まで', 'タワー')
    shade = sky_shade()
    total = cam_total()
    src = F.fast_clip(F.DECK, n)
    ex1 = f_hi - 5                                     # 1 画面目が抜け始める（4 フレームで「ヒ」の 2 フレーム前＝2 画面目の出だしちょうどに消える）
    for i in range(n):
        f = cam(next(src), i, total) * shade
        if i < RAMP:                                   # 11（室内の写真）より明るいので、カットの段差を 0.1 未満にして
            f = f * core.lerp(0.80, 1.0, Mo.ease(i / RAMP, 'sine_out'))       # 露出が合うように振りの中で戻す
        f = P.whip_in(f, i, +1)                        # 11 から：カメラが上へ振ってきた
        # 1 画面目は「オ」から「ヒ」まで 17 フレームしかないので、速めに立ち上げて（約 11 フレームで出そろう）止まって読める間を作る。
        #   旧（1 フレームずつ・9 フレーム、抜けは ヒ−7）では出そろう前に抜け始めていた。
        f = ta.reveal(f, i, max(0, f_o - 3), 'char_rise', stagger=0.7, dur=8, accent=f_o, exit=ex1, exit_dur=4)
        f = tas.reveal(f, i, f_o, 'line_mask', exit=ex1, exit_dur=4)
        f = tb.reveal(f, i, f_hi - 2, 'char_rise', stagger=1.1, dur=10, accent=f_hi, exit=n - 6, exit_dur=6)
        f = tbs.reveal(f, i, f_hi + 2, 'line_mask', exit=n - 6, exit_dur=6)
        yield np.clip(f, 0, 1).astype(np.float32)
