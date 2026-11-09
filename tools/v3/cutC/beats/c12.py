"""案C 12（約2.0秒）「目の前に、Seaside MAZE。」 LP の空撮 → 建物の南風のエンブレム（写真 28）。

 0–(シ−2)：LP の空撮（edl/lp/img/about_s02.jpg：手前に河口の緑の岸、仁淀川河口大橋、奥の橋のたもとに MAZE）。
         縦の窓（写真の高さいっぱい）から、奥の橋のたもと（MAZE）へ 2.5D で押し込む（Depth Anything の奥行き、
         手前の岸と橋ほど大きく動く）。寄りは最後まで速さを持ったまま（hermite、終わりの速さ 1.8 倍）でカットへ突っ込む。
 (シ−2)–：「シーサイド」の 2 フレーム前でハードカットして、建物の切妻の南風のエンブレム（案B の b09 と同じ写真・同じ切り出し・
         同じ空の足し方と ND）。空撮の寄りの速さを受け継いで始まり（hermite の始まりの速さ 1.4 倍）、2.5D のドリーで
         建物が空に対して少し大きくなり、わずかにせり上がる。
 最後の 4 フレームは 13（2F）へ縦の振り（カメラが上へ＝+1。案B の 10→11→12 と同じ向き：階を上がっていく）。
テロップ（案B b09 と同じ位置・大きさ）：「Seaside MAZE」（88px Bold、ACC）が「シ」の 2 フレーム前から 1 文字ずつ上がる
（字間 0.12em→0.03em）。「（南風・まぜ）」は「マ」の 1 フレーム前から行ごと下のマスクから上がる。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W, H
from cutB.beats import _fac as F
from cutB.beats import _motion as Mo
from cutB.beats import _p25 as P
from cutB.beats import b09
from cutC.beats import _maze as Z

AER = dict(file='about_s02.jpg', up=2.0, cx0=0.600, cy0=0.50, cx1=0.648, cy1=0.455, z1=1.30, par=0.09, lat=0.010,
           v0=0.6, v1=1.8, gain=0.97)
HEAD, HEAD_GAIN = 6, 1.10
TAIL, TAIL_DIM = 6, 0.10
EXT = dict(v0=1.4, v1=0.5, vw0=0.985, vw1=0.915, cy1=0.488, par1=0.06, laty1=-0.008)


def _aerial():
    img = Z.load_lp(AER['file'], AER['up'])
    img = F.grade_photo(img, gain=AER['gain'])
    return img


def render(beat, ctx):
    n = beat['n']
    f_si = Z.mf(beat, 5, 'シ')
    f_ma = Z.mf(beat, 10, 'マ')
    cut = f_si - 2
    aer = _aerial()
    ah, aw = aer.shape[:2]
    vfull = F.max_vw(aer)
    adep = P.depth(aer, 'c12aerial', vw=vfull)
    ext, p = b09._image()
    h, w = ext.shape[:2]
    vmax = F.max_vw(ext)
    edep = P.depth(ext, 'b09ext', vw=vmax)
    edep[:p + 8] = 0.0
    lx, ly = b09.LOGO[0], (b09.LOGO[1] * (h - p) + p) / h
    kb = Mo.KText([dict(text='Seaside MAZE', x=W / 2, y=F.Y_BIG, size=88, weight='Bold', hl='Seaside MAZE', track=0.03,
                        shadow=0.9)])
    ks = Mo.KText([dict(text='（南風・まぜ）', x=W / 2, y=F.Y_SUB, size=46, weight='Medium', track=0.06, shadow=0.9)])
    y = np.arange(H, dtype=np.float32)
    t = np.clip((y - 250) / 520.0, 0, 1)
    k = (1 - t * t * (3 - 2 * t))[:, None, None]
    shade = 1 - k * np.array([0.46, 0.42, 0.34], np.float32)        # b09 と同じ（曇り空を上から落とす）
    ta = np.clip((y - 200) / 600.0, 0, 1)
    ashade = (1 - 0.16 * (1 - ta * ta * (3 - 2 * ta)))[:, None, None]  # 空撮の明るい空を少しだけ落とす
    for i in range(n):
        if i < cut:
            u = P.hermite(i / float(cut - 1), AER['v0'], AER['v1'])
            z = AER['z1'] ** u
            f, _ = Mo.parallax_view(aer, adep, core.lerp(AER['cx0'], AER['cx1'], u), core.lerp(AER['cy0'], AER['cy1'], u),
                                    vfull / z, par=AER['par'] * u, lat=AER['lat'] * u)
            f = f * ashade
            if i < HEAD:                                   # 11（明るい河口）から：最初は少し明るい所から戻す（露出が合うように）
                f = f * core.lerp(HEAD_GAIN, 1.0, Mo.ease(i / float(HEAD), 'sine_out'))
        else:
            u = P.hermite((i - cut) / float(n - 1 - cut), EXT['v0'], EXT['v1'])
            vw = vmax * core.lerp(EXT['vw0'], EXT['vw1'], u)
            f, _ = Mo.parallax_view(ext, edep, core.lerp(lx + 0.004, lx, u), core.lerp(0.5, EXT['cy1'], u), vw,
                                    par=EXT['par1'] * u, lat_y=EXT['laty1'] * u)
            f = f * shade
            wa = 1 - P.whip_q(i, n)
            f = kb.reveal(f, i, f_si - 2, 'char_rise', stagger=1.2, dur=10, track_from=0.12, alpha=wa)
            f = ks.reveal(f, i, f_ma - 1, 'line_mask', alpha=wa)
        if i >= n - TAIL:                                  # 13（室内の写真）は暗いので、カットの段差を半分に（案B b09 と同じ）
            f = f * (1 - TAIL_DIM * core.ease((i - (n - TAIL) + 1) / TAIL))
        f = P.whip_out(f, i, n, +1)                             # 13 へ：カメラが上へ振る
        yield np.clip(f, 0, 1).astype(np.float32)
