"""案B 10（約2.6秒）「1階は、ローソンと、産直。」 写真：ローソンの入口 → 産直の色の連打。

 0–(連打の頭)：1F ローソンの入口（LAWSON の看板と自動ドア）へゆっくり寄る。上の灰色の庇の面に見出しを置く。
 連打：産直の写真 8 枚を 5 フレームずつ。黄 → 橙 → 赤 → 紫 → 緑（v2 の PRODUCE と同じ並び・同じ色相）。
       色のかたまりの重心（blob_center）がいつも画面の同じ位置（中心）に来るように切り出す。
       写真ごとの明るさの差でちらつかないよう、窓の平均の明るさをそろえる（カットの前後で 0.1 未満）。
テロップ：「1F」（ACC）が「イ」で、「ローソン」が「ロ」で、「＋産直マーケット」が「サ」で出る（1行を中央ぞろえのまま部分ごとに）。
文字の帯の後ろは、すりガラスのように少しぼかして暗くする（値札・シールの細かい文字を読めなくする）。
"""
import json
import os

import cv2
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from core import W, H
from cutB.beats import _fac as F

PER = 5
# (写真の番号, 色相 OpenCV 0..180, 横のずらし)。v2 の PRODUCE から、IG で値札が帯に接する 108・102・109・29 を外したもの
PRODUCE = [(7, 25, 0.0), (53, 25, 0.0), (46, 12, None), (65, 12, -0.14), (42, 2, 0.0), (55, 2, -0.28),
           (31, 150, 0.15), (26, 55, 0.0)]
VW = 0.30                  # 産直の窓の幅（写真の幅の比）
TARGET = 0.43              # 産直の窓の平均の明るさ（そろえる先）
LAW = dict(cx=0.425, cy=0.69, vw0=0.23, vw1=0.215, gain=1.16)         # ローソンの入口：看板とドアが真ん中、上が庇の面（ロゴは外す）
FIXED = {46: (0.20, 0.56)}    # 46 は木箱の茶色を橙と拾うので手で置く（左のはっさく。右の値札＝生産者名は窓の外）


def blob_center(img, hue, tol=14, smin=0.35):
    """色のかたまりの重心（画像の比）。tools/v2/build_v2.py の blob_center と同じ。"""
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hh = hsv[..., 0].astype(np.int16)
    d = np.minimum(abs(hh - hue), 180 - abs(hh - hue))
    m = ((d < tol) & (hsv[..., 1] > smin * 255) & (hsv[..., 2] > 60)).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), img.shape[1] / 40)
    y, x = np.unravel_index(np.argmax(m), m.shape)
    return x / img.shape[1], y / img.shape[0]


def _files():
    J = json.load(open(os.path.join(core.ROOT, 'edl', 'v3', 'facility_assets.json')))
    return {p['idx']: p['file'] for p in J['produce']}


def _cuts():
    files = _files()
    out = []
    for idx, hue, dx in PRODUCE:
        raw = F.load(files[idx], 3600)
        small = cv2.resize(raw, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        cx, cy = FIXED[idx] if idx in FIXED else blob_center(small, hue)
        h, w = raw.shape[:2]
        cx = float(np.clip(cx + (dx or 0.0), VW / 2, 1 - VW / 2))
        vh = VW * w / h * H / W
        cy = float(np.clip(cy, vh / 2, 1 - vh / 2))
        mid = F.view(small, cx, cy, VW * 0.97, interp=cv2.INTER_LINEAR)
        g = float(np.clip(TARGET / max(1e-3, F.luma(F.grade_photo(mid))), 0.8, 1.25))
        out.append(dict(img=F.grade_photo(raw, gain=g), cx=cx, cy=cy))
    return out


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[0][0] == 'イ' and mo[5][0] == 'ロ' and mo[10][0] == 'サ', mo
    fr = [round(mo[k][1] * core.FPS) - beat['f0'] for k in (0, 5, 10)]
    head = n - PER * len(PRODUCE)
    law = F.grade_photo(F.load(F.LAWSON, 4800), gain=LAW['gain'])
    cuts = _cuts()
    tb = F.heading('1F', None, hl='1F')
    p1, p2 = F.split_line(['ローソン', '＋産直マーケット'], F.Y_SUB, F.SZ_SUB)
    # 「ローソン」は最初は中央に出し、「＋産直マーケット」が出るときに左へ滑って1行の中央ぞろえになる
    full = core.text_width('ローソン＋産直マーケット', F.SZ_SUB, 'Medium', 0.06)
    lone = core.text_width('ローソン', F.SZ_SUB, 'Medium', 0.06)
    slide = (full - lone) / 2
    shade = F.shade(depth=0.18)
    for i in range(n):
        if i < head:
            u = core.ease(i / max(1, head - 1), 'out')
            f = F.view(law, LAW['cx'], LAW['cy'], core.lerp(LAW['vw0'], LAW['vw1'], u)) * shade
        else:
            k, j = divmod(i - head, PER)
            c = cuts[k]
            f = F.view(c['img'], c['cx'], c['cy'], VW * core.lerp(1.0, 0.94, j / (PER - 1)))
            f = F.frost(f)
        f = tb.put(f, F.appear(i, fr[0], 5))
        a2 = F.appear(i, fr[2], 6)
        f = p1.put(f, F.appear(i, fr[1], 5), dx=slide * (1 - a2))
        f = p2.put(f, a2)
        yield np.clip(f, 0, 1).astype(np.float32)
