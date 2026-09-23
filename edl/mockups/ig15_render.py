#!/usr/bin/env python3
"""Instagram 15秒版（EDL v7・2026-09-23 確定版）End Card のモックアップと、CapCutに重ねる透過PNGを作るスクリプト。

End Cardの数値（ロゴのファイルと大きさ、文字、影、出方、セーフゾーン、測り方）は、IG30と共通の
ig_endcard_spec.py だけで決まる。このファイルが持つのは、IG15の画（IMG_9633）に合わせた「配置」だけ。
MAPは入れない（2026-09-23 決定）。ロゴは assets/logo_white.png だけを使う。

実素材 IMG_9633（HDR/HLG）から指定の瞬間をトーンマップして1枚取り出し、EDLの8番と同じ切り出し
（900x1600、キーフレームで水平線に追従）→ 1080x1920 をかけ、ロゴと2行を重ねる。

  python3 edl/mockups/ig15_render.py                  # モックアップを edl/mockups/ に書き出す（位置とコントラストも表示）
  python3 edl/mockups/ig15_render.py --overlays DIR   # 承認後：CapCutに100%で重ねる透過PNG（1080x1920、出る段ごとに2枚）
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ig_endcard_spec as C  # noqa: E402  IG30と共通の End Card の数値表

ROOT = C.ROOT
FOOT = os.path.join(ROOT, 'footage', '01_撮影素材')
OUT = os.path.join(ROOT, 'edl', 'mockups')
FFMPEG = os.environ.get(
    'FFMPEG', '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
TONEMAP = ('zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,'
           'zscale=t=bt709:m=bt709:r=tv,format=yuv420p')
W, H = C.W, C.H

# IG15だけの値（配置）。海の上に、ロゴ（左）と2行の列（右、左そろえ）を横に並べた1組を左右中央に置く
LAYOUT = dict(
    center_y=0.570,          # 1組の縦の中心 57.0%h（1094px。水平線は約45.4%）
    gap=0.045,               # ロゴの右端〜文字の列の左端 4.5%w（49px）
)
END_START, END = 12.20, 15.00          # ①の始まり、動画の終わり
TIMING = C.timing(END_START, END)      # ① 12.20→12.80「高知県土佐市」、② 12.80→13.30 ロゴ＋「@mazewind2026」


# ---------------------------------------------------------------------------
# 素材（EDLの8番：IMG_9633 素材1.5〜4.5秒、等速、900x1600をキーフレームで動かして1.2倍）
# ---------------------------------------------------------------------------
def grab(sec, cache):
    out = os.path.join(cache, 'IMG_9633_%.4f.png' % sec)
    if not os.path.exists(out):
        subprocess.run([FFMPEG, '-v', 'error', '-ss', '%.4f' % sec, '-i', os.path.join(FOOT, 'IMG_9633.MOV'),
                        '-frames:v', '1', '-vf', TONEMAP, '-y', out], check=True)
    return Image.open(out).convert('RGB')


def cut8(t, cache):
    sec = 1.5 + (t - 12.0)
    lt = sec - 1.5
    x = 1950 - 38 * lt
    y = 90 + 50.667 * lt if lt < 1.5 else 166 + 25.333 * (lt - 1.5)
    src = grab(sec, cache)
    return src.transform((W, H), Image.AFFINE, (900 / W, 0, x, 0, 1600 / H, y), resample=Image.BICUBIC), sec, (x, y)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def layout():
    """各パーツの全画面アルファ（W x H の 'L'）と外接枠（px）。"""
    lg = C.logo_alpha()
    pl = C.place_mask()
    hd = C.handle_mask()
    gap = LAYOUT['gap'] * W
    colw = max(pl.width, hd.width)
    x0 = (W - (lg.width + gap + colw)) / 2          # 1組全体を左右中央に置く
    cy = LAYOUT['center_y'] * H
    tx = x0 + lg.width + gap
    r1, r2 = cy - C.TEXT['row_pitch'] / 2 * H, cy + C.TEXT['row_pitch'] / 2 * H
    parts, boxes = {}, {}
    parts['logo'], boxes['logo'] = C.full(lg, (x0, cy - lg.height / 2))
    parts['place'], boxes['place'] = C.full(pl, (tx, r1 - pl.height / 2))
    parts['handle'], boxes['handle'] = C.full(hd, (tx, C.handle_top(r2)))
    return parts, boxes


def render(t, L, cache):
    fr, sec, xy = cut8(t, cache)
    for s in ('s1', 's2'):
        fr = C.comp(fr, L[s], C.envelope(t, TIMING[s]))
    return fr, sec, xy


def report(cache, L):
    parts, boxes = layout()
    for k in ('logo', 'place', 'handle'):
        x0, y0, x1, y1 = boxes[k]
        print('%-6s x %4d-%4d (%.1f-%.1f%%w)  y %4d-%4d (%.1f-%.1f%%h)  %dx%d' % (
            k, x0, x1, x0 / W * 100, x1 / W * 100, y0, y1, y0 / H * 100, y1 / H * 100, x1 - x0, y1 - y0))
    xs = [boxes[k][0] for k in boxes] + [boxes[k][2] for k in boxes]
    ys = [boxes[k][1] for k in boxes] + [boxes[k][3] for k in boxes]
    sb = C.safe_box()
    print('1組の外接枠 x %d-%d  y %d-%d   セーフゾーン x %d-%d y %d-%d  内側：%s' % (
        min(xs), max(xs), min(ys), max(ys), sb[0], sb[2], sb[1], sb[3],
        min(xs) >= sb[0] and max(xs) <= sb[2] and min(ys) >= sb[1] and max(ys) <= sb[3]))
    reg = C.logo_regions(parts['logo'], boxes['logo'])
    for t in (13.30, 13.70, 14.10, 14.50, 14.90, 14.967):
        fr, _, _ = cut8(t, cache)
        c = C.comp(C.comp(fr, L['s1'], 1.0), L['s2'], 1.0)
        print('t=%.3f  高知県土佐市 %.2f:1  @mazewind2026 %.2f:1  ロゴ：外周 %.2f:1  MAZE WIND %.2f:1  RETREAT %.2f:1' % (
            t, C.contrast(c, parts['place'], boxes['place']), C.contrast(c, parts['handle'], boxes['handle']),
            C.outer_contrast(c, reg), C.contrast(c, parts['logo'], reg['maze_wind']),
            C.contrast(c, parts['logo'], reg['retreat'])))


def export_overlays(d):
    os.makedirs(d, exist_ok=True)
    parts, _ = layout()
    L = C.stage_layers(parts)
    for s in ('s1', 's2'):
        p = os.path.join(d, 'ig15_end_%s_1080x1920.png' % C.STAGE_FILES[s])
        L[s].save(p)
        print(p)


def label(im, text, xy=(16, 12), size=24):
    d = ImageDraw.Draw(im)
    d.text(xy, text, font=ImageFont.truetype(C.font_path('Medium'), size), fill=(255, 230, 120))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == '--overlays':
        export_overlays(sys.argv[2])
        return
    cache = os.path.join(tempfile.gettempdir(), 'maze_ig15_frames')
    os.makedirs(cache, exist_ok=True)
    parts, _ = layout()
    L = C.stage_layers(parts)
    report(cache, L)
    for name, t, zones in (('a_place', 12.80, False), ('b_full', 14.50, True)):
        im, sec, xy = render(t, L, cache)
        print(name, 't=%.2f' % t, 'IMG_9633 %.3fs' % sec, 'crop x=%.1f y=%.1f' % xy)
        base = os.path.join(OUT, 'ig15_end_%s_t%05.2f' % (name, t))
        im.save(base + '.png')
        if zones:
            C.safezones(im).save(base + '_safezones.png')
        im.resize((W // 3, H // 3), Image.LANCZOS).save(base + '_third.png')
        if name == 'b_full':
            box = (170, 920, 910, 1270)
            im.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.NEAREST).save(
                os.path.join(OUT, 'ig15_end_c_detail_2x.png'))
    # 出方の確認：1/3に縮小したコマを時間順に並べる（「高知県土佐市」→ ロゴとアカウント名 の2段）
    seq = [12.00, 12.20, 12.50, 12.80, 13.05, 13.30, 14.00, 14.97]
    tw, th = W // 3, H // 3
    strip = Image.new('RGB', (tw * 4 + 5 * 12, (th + 40) * 2 + 12), (18, 18, 18))
    for i, t in enumerate(seq):
        im, _, _ = render(t, L, cache)
        x = 12 + (i % 4) * (tw + 12)
        y = (i // 4) * (th + 40 + 6)
        strip.paste(im.resize((tw, th), Image.LANCZOS), (x, y + 40))
        label(strip, '%.2f秒' % t, (x + 4, y + 8), 24)
    strip.save(os.path.join(OUT, 'ig15_end_e_sequence_third.png'))


if __name__ == '__main__':
    main()
