"""案B 10（約2.6秒）「1階は、ローソンと、産直。」 写真：ローソンの入口 → 産直の色の連打。

 0–(連打の頭)：1F ローソンの入口（LAWSON の看板と自動ドア）へゆっくり寄る。上の灰色の庇の面に見出しを置く。
 連打：産直の写真 8 枚を 5 フレームずつ。黄 → 橙 → 赤 → 紫 → 緑（v2 の PRODUCE と同じ並び・同じ色相）。
       色のかたまりの重心（blob_center）がいつも画面の同じ位置（中心）に来るように切り出す。
       写真ごとの明るさの差でちらつかないよう、窓の平均の明るさをそろえる（カットの前後で 0.1 未満）。
テロップ：「1F」（ACC）が「イ」で、「ローソン」が「ロ」で、「＋産直マーケット」が「サ」で出る（1行を中央ぞろえのまま部分ごとに）。
2026-11-07 動きの改訂：
 - 最初の 4 フレームは 09 からの縦の振り（カメラが下へ）を受ける。最後の 4 フレームは 11 へ縦の振り（カメラが上へ）。
 - ローソン：奥行きで 2.5D（庇が手前）。寄り・視差・横の視差が、最後（33〜37）に速くなって連打へ突っ込む（hermite、終わりの速さ 2.6 倍）。
 - 連打：8 カットを通して 1 本の寄り（1.00→1.14）。色のかたまりの中心まわりに寄り続けるので、1 本の押し込みに見える。
   最初の速さはローソンの最後の寄りの速さ（対数の倍率で）を受け継ぐ。
 - 文字：1 文字ずつ下から上がる（「1F」は FOAM→ACC の強調）。「ローソン」の左への寄せは expo_out・8 フレーム。
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
from cutB.beats import _motion as Mo
from cutB.beats import _p25 as P

PER = 5
# (写真の番号, 色相 OpenCV 0..180, 横のずらし)。v2 の PRODUCE から、IG で値札が帯に接する 108・102・109・29 を外したもの
PRODUCE = [(7, 25, 0.0), (53, 25, 0.0), (46, 12, None), (65, 12, -0.14), (42, 2, 0.0), (55, 2, -0.28),
           (31, 150, 0.15), (26, 55, 0.0)]
VW = 0.30                  # 産直の窓の幅（写真の幅の比）
TARGET = 0.43              # 産直の窓の平均の明るさ（そろえる先）
LAW = dict(cx=0.425, cy0=0.69, cy1=0.674, vw0=0.23, vw1=0.205, gain=1.16, par=0.07, lat=0.015, v0=0.8, v1=2.6)         # ローソンの入口：看板とドアが真ん中、上が庇の面（ロゴは外す）
S1 = 1.14                  # 連打の寄り（8 カットを通して）
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
    dep = P.depth(law, 'b10law', vw=LAW['vw0'])
    cuts = _cuts()
    # 連打の寄りの最初の速さ＝ローソンの最後の寄りの速さ（log の倍率／フレーム）
    lz = np.log(LAW['vw0'] / LAW['vw1']) / max(1, head - 1) * LAW['v1']
    lb = np.log(S1) / (n - head - 1)
    v0b = lz / lb
    kb = Mo.KText([dict(text='1F', x=W / 2, y=F.Y_BIG, size=F.SZ_BIG, weight='Bold', hl='1F', track=0.03, shadow=0.9)])
    full = core.text_width('ローソン＋産直マーケット', F.SZ_SUB, 'Medium', 0.06)
    lone = core.text_width('ローソン', F.SZ_SUB, 'Medium', 0.06)
    x1 = W / 2 - full / 2
    x2 = x1 + lone + 0.06 * F.SZ_SUB
    sub = dict(y=F.Y_SUB, size=F.SZ_SUB, weight='Medium', anchor='ls', track=0.06, shadow=0.9)
    k1 = Mo.KText([dict(sub, text='ローソン', x=x1)])
    k2 = Mo.KText([dict(sub, text='＋産直マーケット', x=x2)])
    # 「ローソン」は最初は中央に出し、「＋産直マーケット」が出るときに左へ滑って1行の中央ぞろえになる
    slide = (full - lone) / 2
    shade = F.shade(depth=0.18)
    for i in range(n):
        if i < head:
            u = P.hermite(i / max(1, head - 1), LAW['v0'], LAW['v1'])
            vw = LAW['vw0'] * (LAW['vw1'] / LAW['vw0']) ** u
            f, _ = Mo.parallax_view(law, dep, LAW['cx'], core.lerp(LAW['cy0'], LAW['cy1'], u), vw,
                                    par=LAW['par'] * u, lat=LAW['lat'] * u)
            f = f * shade
        else:
            k, j = divmod(i - head, PER)
            c = cuts[k]
            s = S1 ** P.hermite((i - head) / (n - head - 1), v0b, 0.8)
            f = F.view(c['img'], c['cx'], c['cy'], VW / s)
            f = F.frost(f)
        f = P.whip_in(f, i, -1) * (1 + 0.07 * P.whip_q(i))   # 09 から：カメラが下へ振ってきた（振りの頭は上に暗い庇が映り込むので少し持ち上げる）
        wa = 1 - P.whip_q(i, n)                                 # 振りの最後のフレームでは文字は消えている
        f = kb.reveal(f, i, fr[0] - 2, 'char_rise', stagger=1.5, dur=10, accent=fr[0], alpha=wa)
        f0s = fr[2] - 2
        sl = slide * (1 - Mo.prog(i, f0s, f0s + 8, 'expo_out'))
        f = k1.reveal(f, i, fr[1] - 2, 'char_rise', stagger=1.0, dur=9, dx=sl, alpha=wa)
        f = k2.reveal(f, i, f0s, 'char_rise', stagger=0.8, dur=9, alpha=wa)
        f = P.whip_out(f, i, n, +1)                            # 11 へ：カメラが上へ振る
        yield np.clip(f, 0, 1).astype(np.float32)
