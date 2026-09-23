#!/usr/bin/env python3
"""Instagram 30秒版（EDL v7・2026-09-23 確定版）のモックアップと、CapCutに重ねる透過PNGを作るスクリプト。

End Cardの数値（ロゴのファイルと大きさ、文字、影、出方、セーフゾーン、測り方）は、IG15と共通の
ig_endcard_spec.py だけで決まる。このファイルが持つのは、IG30の画（IMG_9661）に合わせた「配置」と、
カット10のコピー・空の補正だけ。MAPは入れない（2026-09-23 決定）。ロゴは assets/logo_white.png だけを使う。

実素材（HDR/HLG）から指定の瞬間をトーンマップして1枚取り出し、EDLと同じ
切り出し・回転・空の補正をかけ、ロゴと文字をEDLの数値どおりに重ねる。

  python3 edl/mockups/ig30_render.py                   # モックアップを edl/mockups/ に書き出す（数値と測定値も表示）
  python3 edl/mockups/ig30_render.py --overlays DIR    # 承認後：CapCutに重ねる透過PNG（1080x1920）を DIR に書き出す
      ig30_end_1_place / ig30_end_2_logo_handle / ig30_c10_copy

注意（2026-09-23 時点）：assets/logo_white.png は、お店のPDF（白地に黒の版画風エンブレム）のインクを白に置き換えたもの。
  犬の目・マズル・首輪の明暗が元のロゴと逆になる。お店が白版で使ってよいと了承するまでは、
  --overlays で本番用のPNGを焼き込まない（比較画像：ig30_g_logo_face_compare.png）。
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ig_endcard_spec as C  # noqa: E402  IG15と共通の End Card の数値表

ROOT = C.ROOT
FOOT = os.path.join(ROOT, 'footage', '01_撮影素材')
LOGO_RAW = os.path.join(ROOT, 'footage', '_store', 'logo_raw.jpeg')
OUT = os.path.join(ROOT, 'edl', 'mockups')
FFMPEG = os.environ.get(
    'FFMPEG', '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
TONEMAP = ('zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,'
           'zscale=t=bt709:m=bt709:r=tv,format=yuv420p')
W, H = C.W, C.H

# ---------------------------------------------------------------------------
# IG30だけの値（配置・コピー・空の補正）
# ---------------------------------------------------------------------------
LAYOUT = dict(
    logo_cx=0.500,           # ロゴの横の中心 50%w
    logo_bottom=0.510,       # ロゴの下端 上から51.0%h（水平線は約54.7〜55.7%）。ロゴは空の上
    row1=0.6030,             # 「高知県土佐市」の縦の中心 60.3%h（水平線のすぐ下の海）
    cx=0.500,                # 2行とも横の中心 50%w（行ごとに中央そろえ）
)
COPY = dict(text='海のそばで、少しだけ。', track=0.10, cy=0.2400, cx=0.500)   # カット10のコピー（文字の値は End Card と同じ）
SKY = dict(stops=-0.7, plateau=0.42, zero=0.535)      # カット12（9661）：上端〜42%で−0.7段、53.5%で0
SKY_C10 = dict(stops=-0.4, plateau=0.30, zero=0.60)   # カット10（9636）：上端〜30%で−0.4段、60%で0

# カット（rec_in, rec_out, source, src_in, speed, 切り出し(x, y, w, h) or None, 時計回りの回転°, 空の補正）
CUTS = [
    (4.00, 5.60, 'IMG_9673', 2.70, 0.5, None, 0.0, None),                            # カット3（のれんの波柄と黒い裾だけ）
    (5.60, 7.90, 'IMG_9673', 5.00, 1.0, (180, 640, 1800, 3200), 0.0, None),         # カット4
    (18.80, 22.30, 'IMG_9636', 1.70, 1.0, (415, 538, 1200, 2133), 0.95, SKY_C10),   # カット10（コピー）
    (24.70, 30.00, 'IMG_9661', 3.20, 1.0, (180, 18, 1800, 3200), 0.70, SKY),        # カット12（End Card）
]

END_START, END = 25.30, 30.00
TIMING = C.timing(END_START, END)     # ① 25.30→25.90「高知県土佐市」、② 25.90→26.40 ロゴ＋「@mazewind2026」
TIMING_COPY = [(19.20, 0.0), (19.60, 1.0), (21.50, 1.0), (21.90, 0.0)]


# ---------------------------------------------------------------------------
# 素材
# ---------------------------------------------------------------------------
def grab(source, sec, cache):
    out = os.path.join(cache, '%s_%.4f.png' % (source, sec))
    if not os.path.exists(out):
        subprocess.run([FFMPEG, '-v', 'error', '-ss', '%.4f' % sec, '-i', os.path.join(FOOT, source + '.MOV'),
                        '-frames:v', '1', '-vf', TONEMAP, '-y', out], check=True)
    return Image.open(out).convert('RGB')


def place(src, crop, rot_cw):
    """切り出し枠の中心まわりに時計回りに回して（右上がりの水平線を戻す）、枠で切り出し、W x H に縮小する。"""
    if crop is None:
        crop = (0, 0, src.width, src.height)
    x, y, w, h = crop
    if rot_cw:
        src = src.rotate(-rot_cw, resample=Image.BICUBIC, center=(x + w / 2, y + h / 2), fillcolor=(255, 0, 0))
    return src.crop((x, y, x + w, y + h)).resize((W, H), Image.LANCZOS)


def sky_grade(img, g):
    """空の補正：上端〜g['plateau']で g['stops'] 段、そこから g['zero'] へなめらかに0（線形光で掛ける）。"""
    a = np.asarray(img).astype(np.float32) / 255.0
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    y = (np.arange(H, dtype=np.float32) / H)[:, None, None]
    wgt = np.clip((g['zero'] - y) / (g['zero'] - g['plateau']), 0, 1)
    wgt = wgt * wgt * (3 - 2 * wgt)
    lin = lin * (2.0 ** (g['stops'] * wgt))
    out = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(np.clip(lin, 0, 1), 1 / 2.4) - 0.055)
    return Image.fromarray((out * 255).clip(0, 255).round().astype(np.uint8), 'RGB')


def frame_at(t, cache, grade=True, sky=None):
    for rec_in, rec_out, source, src_in, speed, crop, rot, g in CUTS:
        if rec_in <= t < rec_out or (t == rec_out == 30.0):
            sec = src_in + (t - rec_in) * speed
            fr = place(grab(source, sec, cache), crop, rot)
            if grade and g:
                fr = sky_grade(fr, sky if (sky is not None and g is SKY) else g)
            return fr, (source, round(sec, 3))
    raise ValueError(t)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def layout():
    lg = C.logo_alpha()
    pl = C.place_mask()
    hd = C.handle_mask()
    parts, boxes = {}, {}
    parts['logo'], boxes['logo'] = C.full(lg, (LAYOUT['logo_cx'] * W - lg.width / 2, LAYOUT['logo_bottom'] * H - lg.height))
    r1 = LAYOUT['row1'] * H
    r2 = r1 + C.TEXT['row_pitch'] * H
    parts['place'], boxes['place'] = C.full(pl, (LAYOUT['cx'] * W - pl.width / 2, r1 - pl.height / 2))
    parts['handle'], boxes['handle'] = C.full(hd, (LAYOUT['cx'] * W - hd.width / 2, C.handle_top(r2)))
    return parts, boxes


def copy_layer():
    m = C.text_mask(COPY['text'], C.TEXT['size'] * H, COPY['track'])
    a, box = C.full(m, (COPY['cx'] * W - m.width / 2, COPY['cy'] * H - m.height / 2))
    return C.shadowed(a, C.TEXT['shadow'], C.TEXT['opacity']), a, box


def render(t, cache, L):
    fr, info = frame_at(t, cache)
    fr = C.comp(fr, L['copy'], C.envelope(t, TIMING_COPY) if t <= TIMING_COPY[-1][0] else 0.0)
    for s in ('s1', 's2'):
        fr = C.comp(fr, L[s], C.envelope(t, TIMING[s]))
    return fr, info


def report(cache, L, sky=None):
    parts, boxes = layout()
    for k in ('logo', 'place', 'handle'):
        x0, y0, x1, y1 = boxes[k]
        print('%-6s x %4d-%4d (%.1f-%.1f%%w)  y %4d-%4d (%.1f-%.1f%%h)  %dx%d' % (
            k, x0, x1, x0 / W * 100, x1 / W * 100, y0, y1, y0 / H * 100, y1 / H * 100, x1 - x0, y1 - y0))
    sb = C.safe_box()
    xs = [boxes[k][0] for k in boxes] + [boxes[k][2] for k in boxes]
    ys = [boxes[k][1] for k in boxes] + [boxes[k][3] for k in boxes]
    print('1組の外接枠 x %d-%d  y %d-%d  セーフゾーンの内側：%s' % (
        min(xs), max(xs), min(ys), max(ys),
        min(xs) >= sb[0] and max(xs) <= sb[2] and min(ys) >= sb[1] and max(ys) <= sb[3]))
    reg = C.logo_regions(parts['logo'], boxes['logo'])
    for t in (26.40, 27.20, 28.50, 29.50, 29.90):
        fr, _ = frame_at(t, cache, sky=sky)
        c = C.comp(C.comp(fr, L['s1'], 1.0), L['s2'], 1.0)
        print('t=%.2f  高知県土佐市 %.2f:1  @mazewind2026 %.2f:1  ロゴ：外周 %.2f:1  MAZE WIND %.2f:1  RETREAT %.2f:1' % (
            t, C.contrast(c, parts['place'], boxes['place']), C.contrast(c, parts['handle'], boxes['handle']),
            C.outer_contrast(c, reg), C.contrast(c, parts['logo'], reg['maze_wind']),
            C.contrast(c, parts['logo'], reg['retreat'])))
    ov, a, box = copy_layer()
    for t in (19.60, 21.40):
        fr, _ = frame_at(t, cache)
        c = C.comp(fr, ov, 1.0)
        print('t=%.2f  コピー %.2f:1  背景の平均 %.0f/255' % (
            t, C.contrast(c, a, box), np.asarray(fr.crop(box).convert('L')).mean()))


def export_overlays(d):
    os.makedirs(d, exist_ok=True)
    parts, _ = layout()
    L = C.stage_layers(parts)
    for s in ('s1', 's2'):
        L[s].save(os.path.join(d, 'ig30_end_%s_1080x1920.png' % C.STAGE_FILES[s]))
    copy_layer()[0].save(os.path.join(d, 'ig30_c10_copy_1080x1920.png'))


def logo_face_compare():
    """元のロゴ（スキャン）と白版（assets/logo_white.png）の顔の比較（等倍と2倍）。お店に白版の了承をもらうための資料。"""
    raw = Image.open(LOGO_RAW).convert('RGB')
    wl = Image.open(C.LOGO).convert('RGBA')
    ab = wl.getchannel('A').point(lambda v: 255 if v > 8 else 0).getbbox()
    gray = np.asarray(raw.convert('L'))
    ys, xs = np.where(gray[200:1400, 500:1800] < 110)              # スキャンの中のエンブレム（外側の輪）の外接枠
    rb = (xs.min() + 500, ys.min() + 200, xs.max() + 500, ys.max() + 200)
    sx = (rb[2] - rb[0]) / (ab[2] - ab[0])
    sy = (rb[3] - rb[1]) / (ab[3] - ab[1])

    def to_raw(b):
        return (int(round(rb[0] + (b[0] - ab[0]) * sx)), int(round(rb[1] + (b[1] - ab[1]) * sy)),
                int(round(rb[0] + (b[2] - ab[0]) * sx)), int(round(rb[1] + (b[3] - ab[1]) * sy)))
    bg = Image.new('RGB', wl.size, (70, 110, 150))
    bg.paste(wl.convert('RGB'), mask=wl.getchannel('A'))
    f = ImageFont.truetype(C.font_path('Medium'), 26)
    pad, top = 20, 60
    rows = []
    for box, scale, cap in (((270, 230, 750, 670), 1, '等倍'), ((380, 340, 620, 500), 2, '目・2倍')):
        a = raw.crop(to_raw(box))
        a = a.resize((a.width * scale, a.height * scale), Image.NEAREST)
        b = bg.crop(box).resize(a.size, Image.LANCZOS if scale == 1 else Image.NEAREST)
        c = Image.new('RGB', (a.width * 2 + pad * 3, a.height + top + pad), (18, 18, 18))
        c.paste(a, (pad, top))
        c.paste(b, (a.width + pad * 2, top))
        d = ImageDraw.Draw(c)
        d.text((pad, 16), '元のロゴ（お店のPDF）%s' % cap, font=f, fill=(255, 230, 120))
        d.text((a.width + pad * 2, 16), '白版（logo_white.png）%s' % cap, font=f, fill=(255, 230, 120))
        rows.append(c)
    out = Image.new('RGB', (max(r.width for r in rows), sum(r.height for r in rows)), (18, 18, 18))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    return out


def main():
    cache = os.path.join(tempfile.gettempdir(), 'maze_ig30_frames')
    os.makedirs(cache, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[1] == '--overlays':
        export_overlays(sys.argv[2])
        return
    parts, _ = layout()
    L = C.stage_layers(parts)
    L['copy'] = copy_layer()[0]
    report(cache, L)
    if '--compare-sky' in sys.argv:        # 空の補正を−0.6段にした場合（v6）との比較
        report(cache, L, sky=dict(SKY, stops=-0.6))

    for name, t in (('ig30_a_end_t27.20', 27.20), ('ig30_b_end_t29.90', 29.90)):
        fr, info = render(t, cache, L)
        print(name, info)
        fr.save(os.path.join(OUT, name + '.png'))
        C.safezones(fr).save(os.path.join(OUT, name + '_safezones.png'))
        fr.resize((W // 3, H // 3), Image.LANCZOS).save(os.path.join(OUT, name + '_third.png'))
        if t == 27.20:
            box = (270, 640, 810, 1300)
            fr.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.NEAREST).save(
                os.path.join(OUT, 'ig30_a_end_t27.20_detail_2x.png'))

    # 出方の確認：1/3に縮小したコマを時間順に並べる（IG15と同じ2段）
    seq = [24.70, 25.30, 25.60, 25.90, 26.15, 26.40, 28.00, 29.97]
    tw, th = W // 3, H // 3
    strip = Image.new('RGB', (tw * 4 + 5 * 12, (th + 40) * 2 + 12), (18, 18, 18))
    fl = ImageFont.truetype(C.font_path('Medium'), 24)
    for i, t in enumerate(seq):
        im, _ = render(t, cache, L)
        x = 12 + (i % 4) * (tw + 12)
        y = (i // 4) * (th + 40 + 6)
        strip.paste(im.resize((tw, th), Image.LANCZOS), (x, y + 40))
        ImageDraw.Draw(strip).text((x + 4, y + 8), '%.2f秒' % t, font=fl, fill=(255, 230, 120))
    strip.save(os.path.join(OUT, 'ig30_e_end_sequence_third.png'))

    # カット10のコピー
    for name, t in (('ig30_f_c10_copy_t19.60', 19.60), ('ig30_f_c10_copy_t21.40', 21.40)):
        fr, info = render(t, cache, L)
        print(name, info)
        fr.save(os.path.join(OUT, name + '.png'))
        fr.resize((W // 3, H // 3), Image.LANCZOS).save(os.path.join(OUT, name + '_third.png'))
        if t == 19.60:
            C.safezones(fr).save(os.path.join(OUT, name + '_safezones.png'))

    # カット3：のれんの波柄と黒い裾だけ。入り・中・終わりと、カット4の入り
    strip = [(4.00, 'c3 4.00（素材2.70）'), (4.60, 'c3 4.60（3.00）'), (5.20, 'c3 5.20（3.30）'),
             (5.58, 'c3 5.58（3.49）'), (5.60, 'c4 5.60（5.00）')]
    tw, th = 270, 480
    s = Image.new('RGB', (tw * len(strip), th + 36), (18, 18, 18))
    d = ImageDraw.Draw(s)
    f = ImageFont.truetype(C.font_path('Regular'), 20)
    for i, (t, lab) in enumerate(strip):
        fr, _ = frame_at(t, cache)
        s.paste(fr.resize((tw, th), Image.LANCZOS), (i * tw, 36))
        d.text((i * tw + 8, 8), lab, font=f, fill=(255, 230, 120))
    s.save(os.path.join(OUT, 'ig30_c_cut03_noren_strip.png'))

    # ロゴの顔の比較（お店への確認用。4本共通の質問の資料）
    logo_face_compare().save(os.path.join(OUT, 'ig30_g_logo_face_compare.png'))


if __name__ == '__main__':
    main()
