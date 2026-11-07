"""案B 12（約3.0秒）「屋上と、避難タワーから、太平洋。」 実写：屋上・タワーから見た水平線と河口（IMG_9660、0.5倍スロー）。

素材はほぼ固定の縦 1x。上寄りに 1.72 倍で切り出して、下の柵・遊歩道・車を外す（_fac.DECK）。
画面の下に河口（川・砂州）と外海、上に広い空。水平線の傾きは回転で直す。13 は同じ素材のこの続きから始まる。
テロップ（空の帯）は2画面：
  「オ（屋上）」：「3F・屋上」（屋上＝ACC）／「展望スペース」
  「ヒ（避難）」：「津波避難タワー」（タワー＝ACC）／「平時も開放・どなたでも上まで」（LP の言い回し。開放は 9〜18 時なので「いつでも」とは書かない）
1画面目は2画面目が出る前の 5 フレームで消える。2画面目は最後の 6 フレームで消えて、13 の施設名へ渡す。
空は明るいので、上から ND グラデーションのように落とす（13 も同じ）。
11 からのカットの明るさの段差を抑えるため、最初の 10 フレームは少し暗い所から戻す（カメラの露出が合うように）。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import H
from beats import _real
from cutB.beats import _fac as F


RAMP = 10


def sky_shade():
    """空を上から落とす（約0.5段）。水平線（y≈1280）より上だけ、y≈900 で 0。12 と 13 で共通。"""
    y = np.arange(H, dtype=np.float32)
    t = np.clip((y - 200) / 700.0, 0, 1)
    return (1 - 0.30 * (1 - t * t * (3 - 2 * t)))[:, None, None]


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[0][0] == 'オ' and mo[5][0] == 'ヒ', mo
    f_o, f_hi = [round(mo[k][1] * core.FPS) - beat['f0'] for k in (0, 5)]
    ta = F.heading('3F・屋上', '展望スペース', hl='屋上')
    tb = F.heading('津波避難タワー', '平時も開放・どなたでも上まで', hl='タワー')
    shade = sky_shade()
    src = F.fast_clip(F.DECK, n)
    for i in range(n):
        f = next(src) * shade
        if i < RAMP:                                   # 11（室内の写真）より明るいので、カットの段差を 0.1 未満にして
            f = f * core.lerp(0.80, 1.0, core.ease((i + 1) / RAMP, 'out'))   # 露出が合うように 10 フレームで戻す
        out1 = float(np.clip((f_hi - 1 - i) / 5, 0, 1))
        a1 = F.appear(i, f_o, 5) * out1
        a2 = F.appear(i, f_hi + 1, 6, n=n, fout=6)
        f = ta.put(f, a1)
        f = tb.put(f, a2)
        yield np.clip(f, 0, 1).astype(np.float32)
