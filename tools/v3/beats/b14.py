"""b14（約2.7秒）2F カフェのパスタとケーキ（2026-11-07 追加：クライアントの指示「パスタとケーキを必ず使う」）。

「気づけば、ここ。」の窓越しの海（b13）から、同じ窓辺のテーブルへ。
  前半：明太子クリームパスタ（窓の外に海。footage/new_1107/C176E666.jpg）をゆっくり寄る。
  後半：ケーキの皿（ティラミスとブリュレ。窓の外に海。IMG_0946.JPG）。ここで施設名を小さく出して終わる。
料理名は画面に出さない（ケーキは LP のメニューに載っていない季節の品のため。メニュー名を断定しない）。
"""
import os

import numpy as np
from PIL import Image, ImageOps

import core
from core import W, H

SRC = os.path.join(core.FOOT, 'new_1107')
SHOTS = [dict(file='C176E666.jpg', cx=0.50, cy=0.52, z0=1.00, z1=1.07),      # パスタ（4K の縦、9:16 ちょうど）
         dict(file='IMG_0946.JPG', cx=0.50, cy=0.50, z0=1.02, z1=1.07)]      # ケーキ（960×1706）
SPLIT = 0.48                                                                 # 前半の割合
Y_NAME, Y_ADDR = 400, 454         # ケーキの皿に重ねないよう、窓の外（空）の部分に置く


def _load(spec, scale=1.12):
    im = ImageOps.exif_transpose(Image.open(os.path.join(SRC, spec['file']))).convert('RGB')
    w, h = round(W * scale), round(H * scale)
    r = max(w / im.width, h / im.height)
    im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
    x0, y0 = (im.width - w) // 2, (im.height - h) // 2
    return np.asarray(im.crop((x0, y0, x0 + w, y0 + h))).astype(np.float32) / 255


def _grade(a):
    # 実写の海のパートにそろえる：ハイライトを少し抑え、ほんの少し青へ。料理の暖かさは残す
    a = core.grade(a, con=0.06, sat=1.03, gain=0.97)
    return np.clip(a * np.array([0.985, 1.0, 1.02], np.float32), 0, 1)


def _view(img, z, cx, cy):
    import cv2
    h, w = img.shape[:2]
    s = (W / w) * z * (w / W)                    # img は W*1.12 の大きさ → z=1.12 で原寸
    s = z / (w / W)
    M = np.array([[s, 0, W / 2 - cx * w * s], [0, s, H / 2 - cy * h * s]], np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)


_y = np.arange(H, dtype=np.float32)
TOP = (0.30 * np.exp(-((_y - 430) / 150.0) ** 2))[:, None, None]


def render(beat, ctx):
    n = beat['n']
    n1 = round(n * SPLIT)
    imgs = [_grade(_load(s)) for s in SHOTS]
    tN = core.text_layer([dict(text='Seaside MAZE（南風・まぜ）', x=W / 2, y=Y_NAME, size=40, weight='Medium', alpha=0.92,
                               track=0.03, shadow=0.95),
                          dict(text='高知県土佐市新居', x=W / 2, y=Y_ADDR, size=34, weight='Regular', alpha=0.92,
                               track=0.08, shadow=0.95)])
    f_name = n1 + 6
    for i in range(n):
        k, j, m = (0, i, n1) if i < n1 else (1, i - n1, n - n1)
        s = SHOTS[k]
        u = core.ease(j / max(1, m - 1), 'out' if k == 0 else 'smooth')
        f = _view(imgs[k], core.lerp(s['z0'], s['z1'], u), s['cx'], s['cy'])
        if i >= f_name:
            a = core.ease((i - f_name + 1) / 6.0)
            f = f * (1 - TOP * a)                          # 文字の下の空だけ少し落としてコントラストを確保
            f = core.put_text(f, tN, a)
        yield np.clip(f, 0, 1).astype(np.float32)
