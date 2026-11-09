"""08（25.0–27.4秒）「速さは、時速40キロくらい。」
b07 の最後の地図を約30%暗くし、その上に線のスピードメーター（0〜60 km/h）。針は 0→40 と上がり、「キロ」のキで止まる。
「時速 約40km」（40km を強調）と小さな注「深い沖・周期15秒のうねりの場合 ≈42km/h」。
最後の10フレーム：メーターが左上の小さなアイコンへ縮み（b09 でも残る）、画面全体がズームのぼかしで b09 へ。
"""
import numpy as np

import core
import timeline
from core import W, H
from beats import _mapui as U
from beats import b07

MC = (540.0, 930.0)        # メーターの中心（半円の底）
SHADE = 0.45
_yy, _xx = np.mgrid[0:H, 0:W].astype(np.float32)
_r = np.hypot((_xx - MC[0]) / 470, (_yy - (MC[1] - 60)) / 380)
_VIG = (np.clip(1.25 - _r, 0, 1) ** 1.5)[..., None].astype(np.float32)


def render(beat, ctx):
    n = beat['n']
    mf = lambda k: timeline.mora_frame(beat, k)
    f_ji, f_ki = mf(4), mf(11)                  # ジ（じそく）／キ（キロ）
    f_note = mf(8)                              # 「ン」：注記を少し早く出して、縮む前に約 0.5 秒読めるように
    b7 = next(b for b in timeline.load() if b['key'] == '07')
    st = b07.static()
    t_main = U.Patch([dict(text='時速 約40km', x=W / 2, y=1078, size=100, weight='Bold', hl='40km')])
    t_note = U.Patch([dict(text='深い沖・周期15秒のうねりの場合 ≈42km/h', x=W / 2, y=1160, size=30, weight='Medium',
                           alpha=0.92)])
    _, _, _, lab_items = U.meter_layer(MC[0], MC[1], 0, 1.0, 1.0, labels=True)
    t_lab = U.Patch(lab_items)
    s0 = n - 10                                  # 縮みはじめ
    icon = None
    for i in range(n):
        # 背景：b07 の最後の地図（嵐の渦は回り続ける）を暗く
        f = b07.map_frame(st, b7['n'] - 1 + i, 1.0)
        dim = 0.34 * core.ease(i / 8.0)
        f = f * (1 - dim) + core.NAVY * dim
        # メーターの後ろに、ごく弱い丸い影（読みやすさ）
        sh = SHADE * core.ease(i / 8.0) * (1 - (core.ease((i - (n - 10)) / 9.0, 'inout') if i >= n - 10 else 0))
        if sh > 0:
            f = f * (1 - _VIG * sh) + core.NAVY * (_VIG * sh)
        # 針：f8 から上がりはじめ、キで 40 に止まる
        val = 40.0 * core.ease((i - 8) / max(1, f_ki - 8), 'inout')
        appear = core.ease((i + 1) / 8.0)
        k = core.ease((i - s0) / 9.0, 'inout') if i >= s0 else 0.0
        cx = core.lerp(MC[0], U.ICON_POS[0], k)
        cy = core.lerp(MC[1], U.ICON_POS[1], k)
        sc = core.lerp(1.0, U.ICON_SCALE, k) * (0.92 + 0.08 * appear)
        al = appear * core.lerp(1.0, 0.75, k)
        # 文字（メーターが縮むときに先に消す）
        tout = 1 - core.ease((i - s0 + 2) / 5.0)
        f = t_main.put(f, core.ease((i - f_ji + 1) / 6) * tout)
        f = t_note.put(f, core.ease((i - f_note + 1) / 6) * tout)
        # 最後の数フレーム：画面全体のズームのぼかし（アイコンはぼかさずに上へ重ねる）
        if i >= n - 5:
            kk = (i - (n - 5) + 1) / 5.0
            f = core.zoom_blur(f, 0.22 * kk ** 1.2, cx=W / 2, cy=H / 2)
        if k >= 0.999:
            if icon is None:
                icon = U.icon_layer(alpha=0.75, value=40)
            f = U.paste(f, *icon)
        else:
            rgba, x0, y0, _ = U.meter_layer(cx, cy, val, sc, al, labels=False)
            f = U.paste(f, rgba, x0, y0)
            if k <= 0:
                f = t_lab.put(f, appear)
            else:
                f = t_lab.put(f, max(0.0, 1 - k * 3))
        yield np.clip(f, 0, 1).astype(np.float32)
