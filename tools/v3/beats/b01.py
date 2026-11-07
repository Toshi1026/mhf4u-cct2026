"""b01（0.0–4.2秒）：つかみ。ナザレの巨大な波を斜め横から。白いピクトグラムのサーファーが斜面を降りる。
波は左→右へ進む（手前には迫らない）。ゆっくり 8% 押し込み、最後の 8 フレームで右へ whip（b02 へ）。"""
import cv2
import numpy as np

import core
from core import W, H
from beats import _wave

PUSH = 0.08
WHIP = 8


def render(beat, ctx):
    n = beat['n']
    wave = _wave.Wave3D()
    txt = core.text_layer([
        dict(text='人類初挑戦', x=W / 2, y=470, size=44, weight='Medium', track=0.16),
        dict(text='世界記録の波で', x=W / 2, y=600, size=96, weight='Bold', hl='世界記録の波'),
        dict(text='高知まで行けるか', x=W / 2, y=720, size=96, weight='Bold'),
    ])
    lab = core.text_layer([dict(text='ナザレ（ポルトガル）', x=core.SAFE[0] + 14, y=1222, size=30, weight='Medium',
                                anchor='ls', track=0.06, shadow=0.8)])
    pcx, pcy = W * 0.46, 1150.0                     # 押し込みの中心（チューブの少し左）
    for i in range(n):
        t = i / core.FPS
        j = i - (n - WHIP)
        q = (j + 1) / WHIP if j >= 0 else 0.0
        pan = 300 * q ** 2                          # 最後：カメラを右へ振る
        f, st = wave.render(t, pan=pan)
        # サーファー：チューブの手前の開いた斜面を、波の進む向き（右下）へ降りていく
        p = core.ease(t / (n / core.FPS), 'smooth')
        zs = 2.06 - 0.10 * p
        sf = 0.50 + 0.16 * p + 0.02 * np.sin(t * 2.3)
        x, y, ang, hpx = wave.surfer_at(t, zs, sf, pan=pan)
        _wave.draw_surfer(f, x, y - 3, scale=2.0, ang=float(np.clip(0.42 * ang, 8, 26)))
        # ゆっくり押し込み
        k = 1 + PUSH * core.ease(t / (n / core.FPS), 'smooth')
        M = np.array([[k, 0, pcx * (1 - k)], [0, k, pcy * (1 - k)]], np.float32)
        f = cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # 文字
        f = core.put_text(f, lab, min(1.0, (i + 1) / 8) * (1 - q))
        if i >= 3:
            f = core.put_text(f, txt, min(1.0, (i - 2) / 3) * (1 - q))
        # 最後：右への whip（ぶれは 0 → 120px）
        if j >= 0:
            f = core.whip(f, 120 * q ** 1.5)
        yield np.clip(f, 0, 1).astype(np.float32)
