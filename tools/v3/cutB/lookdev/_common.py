"""look-dev の試作スクリプトの共通部分（パス・保存・左右比較・時間の計測）。"""
import os
import sys
import time

V3 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUTB = os.path.join(V3, 'cutB')
for p in (V3, CUTB):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import core  # noqa: E402
from core import W, H  # noqa: E402,F401
from cutB.beats import _useA  # noqa: E402,F401

OUTD = os.path.join(core.OUT, 'lookdev_B')
BEFORE = os.path.join(core.OUT, 'preview_B')
os.makedirs(OUTD, exist_ok=True)
TIMES = {}


class tick:
    """with tick('sky'): ...  → TIMES['sky'] に秒を足す。"""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()

    def __exit__(self, *a):
        TIMES[self.name] = TIMES.get(self.name, 0.0) + time.perf_counter() - self.t


def report(tag):
    tot = sum(TIMES.values())
    print('[%s] ' % tag + '  '.join('%s %.3fs' % (k, v) for k, v in TIMES.items()) + '  | total %.3fs' % tot)
    TIMES.clear()


def label(img, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    f = core.font(40, 'Bold')
    d.rectangle([0, 0, 260, 64], fill=(0, 0, 0))
    d.text((18, 10), text, font=f, fill=(255, 255, 255))
    return np.asarray(im).astype(np.float32) / 255


def save_pair(key, frame_idx, after, before=None, cg=True):
    """after（sRGB float）を保存し、BEFORE（preview_B の同じフレームの PNG）と左右に並べた比較も保存する。"""
    if cg:
        after = core.cg_label(after)
    pa = os.path.join(OUTD, 'b%s_f%04d_after.png' % (key, frame_idx))
    core.save_png(after, pa)
    if before is None:
        pb = os.path.join(BEFORE, 'b%s_f%04d.png' % (key, frame_idx))
        if os.path.exists(pb):
            before = cv2.cvtColor(cv2.imread(pb), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    if before is not None:
        core.save_png(before, os.path.join(OUTD, 'b%s_f%04d_before.png' % (key, frame_idx)))
        gap = np.full((before.shape[0], 24, 3), 0.05, np.float32)
        sbs = np.concatenate([label(before, 'BEFORE'), gap, label(after, 'AFTER')], 1)
        core.save_png(sbs, os.path.join(OUTD, 'b%s_f%04d_sbs.png' % (key, frame_idx)))
    print('saved', pa)
