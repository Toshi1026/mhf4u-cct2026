#!/usr/bin/env python3
"""サイネージ15秒版（EDL v5）のモックアップ。

右下のマーク・ENDの中央・右下の減光・出方の決まりは、30秒版と共通の数値表
signage_spec.py だけを使う（このファイルには持たない）。ここにあるのは15秒版のカット表だけ。

実素材（HDR/HLG）から、EDLの区間の「何コマ目」を1枚取り出してトーンマップし、EDLと同じ
拡大・回転・位置をかけ、素材ごとの減光を焼き込み、共通のPNGと同じ面を重ねる。

  python3 edl/mockups/signage15_render.py            # モックアップを edl/mockups/ に書き出す
  python3 edl/mockups/signage15_render.py --motion   # 0〜3秒を30fpsの動画にする（減光の楕円の見え方の確認）
  python3 edl/mockups/signage15_render.py --margins  # 各カットの最初・中・最後のコマで、画面の外が出ていないかを数える
"""
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import signage_spec as S  # noqa: E402  30秒版と共通の数値表

ROOT = S.ROOT
FOOT = os.path.join(ROOT, 'footage', '01_撮影素材')
OUT = os.path.join(ROOT, 'edl', 'mockups')
FFMPEG = os.environ.get(
    'FFMPEG', '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
TONEMAP = ('zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,'
           'zscale=t=bt709:m=bt709:r=tv,format=yuv420p')
CACHE = os.path.join(tempfile.gettempdir(), 'maze_signage15_frames_v5')

E, T = 9.00, 15.00   # ENDのカットの頭、ファイルの長さ（共通の出方の決まりに渡す）

# (n, rec_in, rec_out, source, 最初のコマ番号, コマの間隔, 拡大, 回転[(素材の秒, CapCutの回転欄の値)], 右へ[w比], 下へ[h比])
# 素材は平均59.97fpsの可変フレームレート。0.5倍＝素材のコマを1枚ずつ、30fpsの1コマに並べる。
# 共有する素材の拡大・回転（素材の秒で指定）・位置は、30秒版（signage30_render.py）とまったく同じ値にする。
CUTS = [
    (1, 0.00, 3.00, 'IMG_9665', 621, 1, 1.15, [(10.35, -1.2), (11.00, -1.65), (11.95, -2.15)], 0.0, 0.036),
    (2, 3.00, 6.00, 'IMG_9674', 402, 1, 1.12, [(6.70, -2.4)], 0.012, -0.012),
    (3, 6.00, 9.00, None, None, None, None, None, None, None),      # 撮影待ち（黒地の仮スレート）
    (4, 9.00, 15.00, 'IMG_9631', 198, 1, 1.07, [(3.20, 0.70), (8.60, 0.81)], 0.0, -0.010),
]
_PTS = {}


def pts_of(source):
    """各コマの表示時刻（秒、表示順）。時間の単位は1/600秒。"""
    if source not in _PTS:
        out = subprocess.run([FFMPEG, '-v', 'error', '-i', os.path.join(FOOT, source + '.MOV'), '-map', '0:v:0',
                              '-c', 'copy', '-f', 'framecrc', '-'], capture_output=True, text=True).stdout
        p = sorted(int(l.split(',')[2]) for l in out.splitlines() if l and not l.startswith('#'))
        _PTS[source] = [v / 600.0 for v in p]
    return _PTS[source]


def grab(source, n):
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, '%s_n%04d.png' % (source, n))
    if not os.path.exists(out):
        t = pts_of(source)[n] - 0.0008
        subprocess.run([FFMPEG, '-v', 'error', '-ss', '%.4f' % t, '-i', os.path.join(FOOT, source + '.MOV'),
                        '-frames:v', '1', '-vf', TONEMAP, '-y', out], check=True)
    return Image.open(out).convert('RGB')


def capcut_transform(src, scale, rot_cw_deg, dx, dy, W, H):
    """画面に合わせて配置 → 中心で拡大 → 中心で回転（+が時計回り）→ 位置をずらす。はみ出しはマゼンタ。"""
    k = src.width / W
    th = math.radians(rot_cw_deg)
    c, s = math.cos(th), math.sin(th)
    cx, cy, tx, ty = W / 2, H / 2, dx * W, dy * H
    a, b, d, e = c / scale, s / scale, -s / scale, c / scale
    ox = -a * (cx + tx) - b * (cy + ty) + cx
    oy = -d * (cx + tx) - e * (cy + ty) + cy
    return src.transform((W, H), Image.AFFINE, (a * k, b * k, ox * k, d * k, e * k, oy * k),
                         resample=Image.BICUBIC, fillcolor=(255, 0, 255))


def lerp(keys, t):
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return keys[-1][1]


def cut_at(t):
    for c in CUTS:
        if c[1] <= t < c[2] or (t >= T and c[2] == T):
            return c
    raise ValueError(t)


def source_frame(t):
    c = cut_at(t)
    src, n0, step = c[3], c[4], c[5]
    if src is None:
        return c, None, None, None
    k = min(int(round(t * 30)), int(round(c[2] * 30)) - 1) - int(round(c[1] * 30))
    idx = n0 + step * k
    return c, src, idx, pts_of(src)[idx]


def plate(t, W=1920, H=1080, mode='default', dim=True, ev=None):
    """減光まで焼き込んだクリップの1コマ（重ねる前）。ev で減光の強さを上書きできる（比較用）。"""
    c, src, idx, sec = source_frame(t)
    if src is None:
        return Image.new('RGB', (W, H), (0, 0, 0)), (c[0], None, None, None, None)
    rot = lerp(c[7], sec)
    fr = capcut_transform(grab(src, idx), c[6], rot, c[8], c[9], W, H)
    if dim and t < S.timing(mode, E, T)['dim_until']:
        fr = S.apply_dim(fr, S.DIM_EV[src] if ev is None else ev)
    return fr, (c[0], src, idx, round(sec, 4), round(rot, 3))


# ---------------------------------------------------------------------------
# 提案（共通の表にはまだ無い）：代案Aを選ぶ場合のENDの中央＝ロゴ＋2行（店名の行を足す）
# 採用するときは signage_spec.py の END_LOGO に入れ、30秒版と同時に変える。
# ---------------------------------------------------------------------------
PROPOSAL_ENDLOGO_NAME = dict(
    logo_h=0.160, gap=0.018, center_y=0.530,
    line1=dict(text='MAZE WIND~RETREAT', weight='Regular', size=0.036, track=0.15),   # 大文字の高さ約2.6%h
    line2=dict(text='高知県土佐市', weight='Regular', size=0.027, track=0.30),
    leading=0.020,          # 1行目の下端〜2行目の上端（インク）
    opacity=1.0,
)
_PROP = {}


def proposal_planes(W, H):
    if (W, H) in _PROP:
        return _PROP[(W, H)]
    p = PROPOSAL_ENDLOGO_NAME
    W4, H4 = 3840, 2160
    ink = np.zeros((H4, W4), np.float32)
    la = S.logo_alpha(p['logo_h'] * H4)
    t1, _, b1 = S.text_alpha(p['line1']['text'], p['line1']['weight'], p['line1']['size'] * H4, p['line1']['track'])
    t2, _, b2 = S.text_alpha(p['line2']['text'], p['line2']['weight'], p['line2']['size'] * H4, p['line2']['track'])
    h1, h2 = b1[3] - b1[1], b2[3] - b2[1]
    block_h = h1 + p['leading'] * H4 + h2
    block_w = max(b1[2] - b1[0], b2[2] - b2[0])
    total = la.width + p['gap'] * W4 + block_w
    lx = (W4 - total) / 2
    ly = p['center_y'] * H4 - la.height / 2
    S._paste_max(ink, la, lx, ly)
    x_text = lx + la.width + p['gap'] * W4
    y1 = p['center_y'] * H4 - block_h / 2
    S._paste_max(ink, t1, x_text - b1[0], y1 - b1[1])
    S._paste_max(ink, t2, x_text - b2[0], y1 + h1 + p['leading'] * H4 - b2[1])
    boxes = {'logo': (lx, ly, lx + la.width, ly + la.height),
             'line1': (x_text, y1, x_text + b1[2] - b1[0], y1 + h1),
             'line2': (x_text, y1 + h1 + p['leading'] * H4, x_text + b2[2] - b2[0], y1 + block_h)}
    sh = S.shadow_of(ink, H4)
    if (W, H) != (W4, H4):
        ink = np.asarray(Image.fromarray(ink).resize((W, H), Image.BOX)).astype(np.float32)
        sh = np.asarray(Image.fromarray(sh).resize((W, H), Image.BOX)).astype(np.float32)
        boxes = {n: tuple(v * W / W4 for v in b) for n, b in boxes.items()}
    _PROP[(W, H)] = (ink, sh, boxes)
    return _PROP[(W, H)]


def comp_planes(frame, ink, sh, op):
    if op <= 0:
        return frame
    rgb = np.asarray(frame).astype(np.float32)
    rgb = rgb * (1 - (sh * op)[..., None])
    rgb = rgb * (1 - (ink * op)[..., None]) + 255 * (ink * op)[..., None]
    return Image.fromarray(rgb.round().clip(0, 255).astype(np.uint8))


def render(t, W=1920, H=1080, mode='default', force_end=None, proposal=False, ev=None):
    """mode: 'default'（右下は常時表示・中央は地名だけ）/ 'alt_a'（ENDで右下からロゴの1組へ受け渡す）。"""
    fr, info = plate(t, W, H, mode, ev=ev)
    tm = S.timing(mode, E, T)
    fr = S.comp(fr, 'corner', S.envelope(t, tm['corner']))
    a = S.envelope(t, tm['end']) if force_end is None else force_end
    if mode == 'alt_a' and proposal:
        ink, sh, _ = proposal_planes(W, H)
        fr = comp_planes(fr, ink, sh, PROPOSAL_ENDLOGO_NAME['opacity'] * a)
    else:
        fr = S.comp(fr, 'end' if mode == 'default' else 'endlogo', a)
    return fr, info


# ---------------------------------------------------------------------------
# 測定
# ---------------------------------------------------------------------------
def luma(img):
    a = np.asarray(img).astype(np.float32)
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def rel_lum(v):
    v = v / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def under_corner(t, mode='default', ev=None, W=1920, H=1080):
    """右下のマークの下の背景（減光後・重ねる前）：店名の行の下の平均と明るいほうの1割、ロゴの下の平均、白い文字とのコントラスト比。"""
    fr, _ = plate(t, W, H, mode, ev=ev)
    y = luma(fr)
    _, _, b = S.planes('corner', W, H)
    out = {}
    for n in ('line', 'logo'):
        x0, y0, x1, y1 = [int(round(v)) for v in b[n]]
        r = y[y0:y1, x0:x1]
        out[n] = (float(r.mean()), float(np.percentile(r, 90)))
    bg = rel_lum(np.array(out['line'][0]))
    bg90 = rel_lum(np.array(out['line'][1]))
    white = rel_lum(np.array(255.0 * 1.0))
    out['contrast'] = float((white + 0.05) / (bg + 0.05))
    out['contrast90'] = float((white + 0.05) / (bg90 + 0.05))
    return out


def label(img, text, size=28):
    im = img.copy()
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(S.font_path('Medium'), size)
    d.rectangle([0, 0, d.textlength(text, font=f) + 24, size + 18], fill=(0, 0, 0))
    d.text((12, 6), text, font=f, fill=(255, 255, 255))
    return im


def grid(ims, cols, gap=8, bg=(40, 40, 40)):
    w, h = ims[0].size
    rows = (len(ims) + cols - 1) // cols
    g = Image.new('RGB', (cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap), bg)
    for i, im in enumerate(ims):
        g.paste(im, ((i % cols) * (w + gap), (i // cols) * (h + gap)))
    return g


def main():
    W, H = 1920, 1080
    j = lambda n: os.path.join(OUT, n)  # noqa: E731
    a, ia = render(2.00)
    b, ib = render(12.00)
    b_alt, _ = render(12.00, mode='alt_a')
    b_prop, _ = render(12.00, mode='alt_a', proposal=True)
    d, idd = render(3.00)
    print('a', ia, 'b', ib, 'cut2', idd)
    a.save(j('signage15_a_corner_brightest_t02.00.png'))
    b.save(j('signage15_b_end_t12.00.png'))
    b_alt.save(j('signage15_b_alt-a_end_t12.00.png'))
    b_prop.save(j('signage15_b_alt-a-name_end_t12.00.png'))
    d.save(j('signage15_d_cut2_in_t03.00.png'))
    box = (1280, 880, 1920, 1080)
    for im, n in ((a, 'signage15_a_corner_detail_2x.png'), (b, 'signage15_b_corner_detail_2x.png')):
        im.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.NEAREST).save(j(n))
    d.crop((0, 0, 640, 360)).resize((1280, 720), Image.NEAREST).save(j('signage15_d_cut2_in_topleft_2x.png'))

    # 距離の目安：1/4（数メートル先）、1/6、1/8（道路の向こう）。左から a／b（既定）／b（代案A）／b（代案A＋店名の行）
    for div in (4, 6, 8):
        w, h = W // div, H // div
        sm = grid([im.resize((w, h), Image.LANCZOS) for im in (a, b, b_alt, b_prop)], 4, gap=4, bg=(0, 0, 0))
        sm.save(j('signage15_c_distance_1-%d.png' % div))
        if div == 8:
            sm.resize((sm.width * 4, sm.height * 4), Image.BICUBIC).save(j('signage15_c_distance_1-8_view4x.png'))

    # ENDに入る前後（上2段＝既定、下2段＝代案A）：8.70〜10.60秒。カット3は撮影待ちの黒地
    ts = [8.70, 8.97, 9.00, 9.30, 9.60, 9.90, 10.30, 10.60]
    rows = []
    for mode in ('default', 'alt_a'):
        for t in ts:
            fr, _ = render(t, mode=mode)
            rows.append(label(fr.resize((480, 270), Image.LANCZOS),
                              '%s %.2f秒' % ('既定' if mode == 'default' else '代案A', t), 18))
    grid(rows, 4).save(j('signage15_f_end_transition_strip.png'))

    # 右下の背景の測定（既定の減光）
    for t in (0.30, 1.30, 2.00, 2.30, 2.70, 3.10, 4.00, 4.50, 5.00, 5.50, 5.90, 9.10, 10.0, 11.0, 12.0, 13.0, 14.0, 14.9):
        m = under_corner(t)
        print('t=%5.2f  line mean %5.1f p90 %5.1f  logo mean %5.1f  contrast %.2f (p90 %.2f)' % (
            t, m['line'][0], m['line'][1], m['logo'][0], m['contrast'], m['contrast90']))


def margins():
    """各カットの最初・中・最後のコマで、画面の外（マゼンタ）が出ていないかを数える。"""
    for c in CUTS:
        if c[3] is None:
            continue
        for t in (c[1], (c[1] + c[2]) / 2, c[2] - 1 / 30.0):
            fr, info = plate(t, dim=False)
            a = np.asarray(fr).astype(int)
            mag = int(((a[..., 0] > 250) & (a[..., 1] < 5) & (a[..., 2] > 250)).sum())
            print('cut %d t=%.3f %s magenta px=%d' % (c[0], t, info, mag))


def motion():
    """0〜3秒（カット1）を30fpsの90コマで描き、動画にする。減光の楕円が砂州の上で「動かないしみ」に見えないかの確認用。"""
    W, H = 1920, 1080
    d = os.path.join(CACHE, 'motion')
    os.makedirs(d, exist_ok=True)
    for i in range(90):
        fr, _ = render(i / 30.0, W, H)
        fr.save(os.path.join(d, 'f%03d.png' % i))
    out = os.path.join(OUT, 'signage15_g_cut1_motion_0-3s.mp4')
    subprocess.run([FFMPEG, '-v', 'error', '-y', '-framerate', '30', '-i', os.path.join(d, 'f%03d.png'),
                    '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], check=True)
    print(out)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--motion':
        motion()
    elif len(sys.argv) > 1 and sys.argv[1] == '--margins':
        margins()
    else:
        main()
