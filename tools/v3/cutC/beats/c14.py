"""案C 14（約2.9秒）「屋上から、太平洋。」 実写：屋上・タワーから見た水平線と河口（IMG_9660、0.5倍スロー）→ 声なしで 3 カット。

 0–(R−1)：IMG_9660（案B b12 と同じ読み出し _fac.DECK：上寄りに 1.72 倍、下の柵・車を外す、水平線の傾きを直す）。
         空は b12 と同じ ND で上から落とす。カメラは水平線の少し上を中心に 1.00→1.06 の寄り（hermite、最後も速さを残す）。
         最初の 4 フレームは 13 からの縦の振り（カメラが上へ）を受ける。13（室内の写真）より明るいので、露出を 0.80→1.0 で
         振りの中で戻す（b12 と同じ）。
 R–：ナレーションのあとの声なしの 1.2 秒に、0.4 秒（12 フレーム）ずつ 3 カット：
         1F ローソンの入口（写真 57。案B b10 と同じ切り出し）→ 産直の棚（写真 53：柑橘・にんじん・青もの。色のかたまりが 1 色にならない広めの窓）→ 避難タワー（写真 56 の右の塔）。
         3 カットを通して 1 本の寄り（1.00→1.16、対数で等速）。屋上の最後の寄りの速さを受け継ぐ。
         各カットの平均の明るさをそろえて（窓の平均を TARGET へ、0.8〜1.25 倍）、カットの段差を 1 フレーム 0.1 未満に（点滅にしない）。
         写真は案A/B の実写と同じ grade（寒色寄り）。値札などの細かい文字が読めないよう、3 カットは σ0.8px だけ柔らかく。
テロップ（案B の階の見出しと同じ位置・大きさ）：「3F・屋上」（屋上＝ACC）が「オ」の 2 フレーム前（振りの後）から 1 文字ずつ、
「オ」の 2 フレーム後に FOAM→ACC。「展望スペース」は「カ（から）」の 1 フレーム前から行ごと。3 カットの前の 6 フレームで抜ける
（3 カットはほかの階なので「3F」を残さない）。3 カットには文字を置かない（声なし・0.4 秒で読めない）。
LP の表記は「3階・屋上」だが、案C の台本（edl/v3/SCRIPT_C_ig.md）と案B b12 の「3F・屋上」に合わせる（2F と表記をそろえる）。
"""
import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W, H
from cutB.beats import _fac as F
from cutB.beats import _motion as Mo
from cutB.beats import _p25 as P
from cutB.beats import b10, b12
from cutC.beats import _maze as Z

PER, NCUT = 12, 3
RAMP = 6
CAM_Z, CAM_PIVOT, CAM_V1 = 1.06, (W / 2, 1180.0), 1.5
S1 = 1.16
TARGET = 0.52
SOFT = 0.8
# (写真, 長辺 px, cx, cy, vw)：窓の幅は写真の幅に対する比
CUTS = [('lawson', F.LAWSON, 4800, b10.LAW['cx'], b10.LAW['cy1'], b10.LAW['vw1'] * 1.06, b10.LAW['gain']),
        ('produce', 'C317DA24-C796-4202-AACA-081518E1EA07.JPEG', 3600, 0.56, 0.50, 0.36, 1.0),
        ('tower', Z.BUILDING_SIDE, 4800, 0.785, 0.50, 0.215, 1.0)]


def _cuts():
    out = []
    for key, name, ls, cx, cy, vw, gain in CUTS:
        img = F.grade_photo(F.load(name, ls), gain=gain)
        mid = F.view(img, cx, cy, vw * 0.93, interp=cv2.INTER_AREA)
        g = float(np.clip(TARGET / max(1e-3, F.luma(mid)), 0.8, 1.25))
        out.append(dict(key=key, img=np.clip(img * g, 0, 1), cx=cx, cy=cy, vw=vw))
    return out


def _zoom(f, z, pivot):
    if z < 1 + 1e-5:
        return f
    px, py = pivot
    Mt = np.float32([[z, 0, px * (1 - z)], [0, z, py * (1 - z)]])
    return cv2.warpAffine(f, Mt, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def render(beat, ctx):
    n = beat['n']
    f_o = Z.mf(beat, 0, 'オ')
    f_ka = Z.mf(beat, 4, 'カ')
    R = n - PER * NCUT
    ta, tas = b12._items('3F・屋上', '展望スペース', '屋上')
    shade = b12.sky_shade()
    src = F.fast_clip(F.DECK, R)
    cuts = _cuts()
    # 3 カットの寄りの最初の速さ＝屋上の最後の寄りの速さ（log の倍率／フレーム）
    lz = np.log(CAM_Z) / (R - 1) * CAM_V1
    lb = np.log(S1) / (PER * NCUT - 1)
    v0b = lz / lb
    ex = R - 6
    for i in range(n):
        if i < R:
            z = CAM_Z ** P.hermite(i / float(R - 1), 0.0, CAM_V1)
            f = _zoom(next(src), z, CAM_PIVOT) * shade
            if i < RAMP:
                f = f * core.lerp(0.80, 1.0, Mo.ease(i / RAMP, 'sine_out'))
            if i >= R - 6:                                  # 3 カット（写真）へ、カットの段差を 0.1 未満に
                f = f * (1 - 0.08 * Mo.ease((i - (R - 6) + 1) / 6.0, 'sine_in'))
            f = P.whip_in(f, i, +1)                             # 13 から：カメラが上へ振ってきた
            f = ta.reveal(f, i, max(f_o - 2, P.WHIP_N - 1), 'char_rise', stagger=1.1, dur=10, accent=f_o + 2,
                          exit=ex, exit_dur=6)
            f = tas.reveal(f, i, f_ka - 1, 'line_mask', exit=ex, exit_dur=6)
        else:
            k = (i - R) // PER
            c = cuts[k]
            s = S1 ** P.hermite((i - R) / float(PER * NCUT - 1), v0b, 0.9)
            f = F.view(c['img'], c['cx'], c['cy'], c['vw'] / s)
            f = cv2.GaussianBlur(f, (0, 0), SOFT)
        yield np.clip(f, 0, 1).astype(np.float32)
