#!/usr/bin/env python3
"""サイネージ15秒版（EDL v6・2026-09-23 確定版）のモックアップ。

右下のマーク・ENDの中央・右下の減光・出方の決まりは、30秒版と共通の数値表
signage_spec.py だけを使う（このファイルには持たない）。ここにあるのは15秒版のカット表だけ。
ENDは③（右下のマークをENDの直前に消し、中央に正式ロゴ＋「MAZE WIND~RETREAT」＋「高知県土佐市」）だけを描く。

実素材（HDR/HLG）から、EDLの区間の「何コマ目」を1枚取り出してトーンマップし、EDLと同じ
拡大・回転・位置をかけ、素材ごとの減光を焼き込み、共通のPNGと同じ面を重ねる。

  python3 edl/mockups/signage15_render.py            # モックアップを edl/mockups/ に書き出す
  python3 edl/mockups/signage15_render.py --motion   # 0〜3秒を30fpsの動画にする（減光の楕円の見え方の確認）
  python3 edl/mockups/signage15_render.py --measure  # 右下とENDのコントラスト比（4本共通の測り方）を表示する
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


def plate(t, W=1920, H=1080, dim=True, ev=None):
    """減光まで焼き込んだクリップの1コマ（重ねる前）。ev で減光の強さを上書きできる（比較用）。"""
    c, src, idx, sec = source_frame(t)
    if src is None:
        return Image.new('RGB', (W, H), (0, 0, 0)), (c[0], None, None, None, None)
    rot = lerp(c[7], sec)
    fr = capcut_transform(grab(src, idx), c[6], rot, c[8], c[9], W, H)
    if dim and t < S.timing(E, T)['dim_until']:
        fr = S.apply_dim(fr, S.DIM_EV[src] if ev is None else ev)
    return fr, (c[0], src, idx, round(sec, 4), round(rot, 3))


def render(t, W=1920, H=1080, ev=None, force_end=None):
    """1コマを描く：減光を焼き込んだクリップ → 右下のマーク → END（どれも共通の出方の決まりどおり）。"""
    fr, info = plate(t, W, H, ev=ev)
    tm = S.timing(E, T)
    fr = S.comp(fr, 'corner', S.envelope(t, tm['corner']))
    fr = S.comp(fr, 'end', S.envelope(t, tm['end']) if force_end is None else force_end)
    return fr, info


# ---------------------------------------------------------------------------
# 測定
# ---------------------------------------------------------------------------
def measure_at(t):
    """その時刻に出ているもの（右下のマーク、END）のコントラスト比（signage_spec.contrast と同じ測り方）。"""
    tm = S.timing(E, T)
    fr, info = plate(t)
    out = {}
    for kind in ('corner', 'end'):
        if S.envelope(t, tm[kind]) >= 0.999:
            out[kind] = S.measure(S.comp(fr, kind, 1.0), kind)
    return out, info


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
    d, idd = render(3.00)
    print('a', ia, 'b', ib, 'cut2', idd)
    a.save(j('signage15_a_corner_brightest_t02.00.png'))
    b.save(j('signage15_b_end_t12.00.png'))
    d.save(j('signage15_d_cut2_in_t03.00.png'))
    box = (1280, 880, 1920, 1080)
    a.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.NEAREST).save(
        j('signage15_a_corner_detail_2x.png'))
    ebox = (560, 440, 1360, 700)
    b.crop(ebox).resize(((ebox[2] - ebox[0]) * 2, (ebox[3] - ebox[1]) * 2), Image.NEAREST).save(
        j('signage15_b_end_detail_2x.png'))
    d.crop((0, 0, 640, 360)).resize((1280, 720), Image.NEAREST).save(j('signage15_d_cut2_in_topleft_2x.png'))

    # 距離の目安：1/4（数メートル先）、1/6、1/8（道路の向こう）。左から a（右下）／b（END）
    for div in (4, 6, 8):
        w, h = W // div, H // div
        sm = grid([im.resize((w, h), Image.LANCZOS) for im in (a, b)], 2, gap=4, bg=(0, 0, 0))
        sm.save(j('signage15_c_distance_1-%d.png' % div))
        if div == 8:
            sm.resize((sm.width * 4, sm.height * 4), Image.BICUBIC).save(j('signage15_c_distance_1-8_view4x.png'))

    # ENDに入る前後：8.20〜10.20秒。カット3は撮影待ちの黒地
    ts = [8.20, 8.40, 8.70, 8.97, 9.00, 9.30, 9.60, 9.90]
    rows = [label(render(t)[0].resize((480, 270), Image.LANCZOS), '%.2f秒' % t, 18) for t in ts]
    grid(rows, 4).save(j('signage15_f_end_transition_strip.png'))
    measure_report()


def measure_report():
    for t in (0.60, 1.30, 2.00, 2.30, 2.70, 3.10, 4.00, 4.50, 5.00, 5.50, 5.90, 8.40,
              9.90, 11.0, 12.0, 13.0, 13.6):
        m, info = measure_at(t)
        print('t=%5.2f %s  %s' % (t, info[1], '  '.join('%s:%s' % (k, ' '.join('%s %.2f' % (n, v) for n, v in d.items()))
                                                      for k, d in m.items())))


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
    elif len(sys.argv) > 1 and sys.argv[1] == '--measure':
        measure_report()
    elif len(sys.argv) > 1 and sys.argv[1] == '--margins':
        margins()
    else:
        main()
