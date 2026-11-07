"""実写パート（b11 の終わり〜b12〜b13）の共通部品。

- 3本の iPhone HLG 素材の読み出し条件（回転・切り出し・速さ）を1か所にまとめる
- 色は1つの grade（すっきりした寒色寄り。CG の青に合わせる。ティール＆オレンジにはしない）
- b11 が線画に起こす「b12 の1フレーム目」をキャッシュする（b11 と b12 で同じ処理を通すので、画素単位で一致する）

傾きは素材のフレームから水平線を当てはめて実測した値（ffmpeg の rotate は時計回りが正）：
  IMG_9662 2.10–2.77s：右上がり 1.31〜1.38° → +1.34°
  IMG_9671 6.30–7.40s：右下がり 0.43〜0.67° → −0.55°
  IMG_9636 1.60s     ：右上がり 0.94°      → +0.95°
"""
import os

import cv2
import numpy as np

import core
from core import W, H

CACHE = os.path.join(core.OUT, 'cache')
os.makedirs(CACHE, exist_ok=True)

# 寒色寄り・すっきり。彩度はほぼそのまま、わずかにコントラスト
GRADE = dict(con=0.07, sat=1.0, lift=0.0, gain=1.0, cool=0.035)

# 読み出し条件
WAVE = dict(name='IMG_9662.MOV', t0=2.10, rot=1.34, cx=0.5, cy=0.5, zoom=1.06)     # 5倍・波が崩れる（0.5倍スロー）
RIVER = dict(name='IMG_9671.MOV', t0=6.30, rot=-0.55, cx=0.5, cy=0.41, zoom=1.25)  # 河口（0.5倍スロー・下の消波ブロックを外す）
WINDOW = dict(name='IMG_9636.MOV', t0=1.60, rot=0.95, cx=0.44, cy=0.385, zoom=2.4)  # 2F の窓（1倍）


def grade(f):
    f = core.grade(f, **GRADE)
    # 海の青緑を少しだけ青へ（緑を数%抑える）。オレンジ方向には振らない
    g = f[..., 1:2]
    b = f[..., 2:3]
    teal = np.clip((g - f[..., 0:1]) * 2.0, 0, 1) * np.clip(1 - np.abs(g - b) * 4, 0, 1)
    f[..., 1:2] = g - 0.025 * teal
    return np.clip(f, 0, 1)


def clip(spec, n, step=1):
    """spec の素材を n 枚。step=1 は素材の1フレーム＝出力1フレーム（60fps→30fps で 0.5倍スロー）、step=2 は等速。"""
    t1 = spec['t0'] + (n * step + 2) / 60.0
    gen = core.footage(spec['name'], spec['t0'], t1, rot=spec['rot'], cx=spec['cx'], cy=spec['cy'], zoom=spec['zoom'])
    k = 0
    last = None
    for i, f in enumerate(gen):
        if i % step:
            continue
        last = grade(f)
        yield last
        k += 1
        if k >= n:
            break
    while k < n:            # 念のため（素材が足りないときは最後の1枚で埋める）
        yield last
        k += 1


def wave_first_frame():
    """b12 の1フレーム目（IMG_9662 を WAVE の条件で読み、grade したもの）。キャッシュする。"""
    key = 'real_wave0_%s' % '_'.join('%s' % WAVE[k] for k in ('t0', 'rot', 'cx', 'cy', 'zoom'))
    key += '_' + '_'.join('%s%s' % kv for kv in sorted(GRADE.items())) + '_v1'
    path = os.path.join(CACHE, key + '.npy')
    if os.path.exists(path):
        return np.load(path)
    f = next(clip(WAVE, 1))
    np.save(path, f.astype(np.float32))
    return f
