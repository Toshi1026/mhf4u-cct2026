#!/usr/bin/env python3
"""サイネージ30秒版（EDL v5・2026-09-23 確定版）のモックアップ。

実素材（HDR/HLG）から、EDLの区間の「何コマ目」を1枚取り出してトーンマップし、EDLと同じ
拡大・回転・位置・減光をかけ、右下のマークとENDを共通の数値表（signage_spec.py）で重ねる。
ENDは③（右下のマークを20.10→20.70秒で消し、中央に正式ロゴ＋「MAZE WIND~RETREAT」＋「高知県土佐市」）だけを描く。

  python3 edl/mockups/signage30_render.py           # モックアップを edl/mockups/ に書き出す（コントラスト比も表示）
  python3 edl/mockups/signage30_render.py --measure # コントラスト比（4本共通の測り方）だけを表示する
"""
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import signage_spec as S  # noqa: E402

ROOT = S.ROOT
FOOT = os.path.join(ROOT, 'footage', '01_撮影素材')
OUT = os.path.join(ROOT, 'edl', 'mockups')
FFMPEG = os.environ.get(
    'FFMPEG', '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
TONEMAP = ('zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,'
           'zscale=t=bt709:m=bt709:r=tv,format=yuv420p')
CACHE = os.path.join(tempfile.gettempdir(), 'maze_signage30_frames')

E, T = 20.70, 30.00   # ENDのカットの頭、ファイルの長さ

# (n, rec_in, rec_out, source, 最初のコマ番号, コマの間隔, 拡大, 回転[(素材の秒, CapCutの回転欄の値)], 右へ[w比], 下へ[h比])
CUTS = [
    (1, 0.00, 5.40, 'IMG_9631', 192, 2, 1.07, [(3.20, 0.70), (8.60, 0.81)], 0.0, -0.010),
    (2, 5.40, 8.70, 'IMG_9665', 621, 1, 1.15, [(10.35, -1.2), (11.00, -1.65), (11.95, -2.15)], 0.0, 0.036),
    (3, 8.70, 11.70, 'IMG_9674', 402, 1, 1.12, [(6.70, -2.4)], 0.012, -0.012),
    (4, 11.70, 17.70, None, None, None, None, None, None, None),
    (5, 17.70, 20.70, None, None, None, None, None, None, None),
    (6, 20.70, 30.00, 'IMG_9658', 366, 1, 1.15, [(6.10, 0.3)], 0.0, 0.015),
]
_PTS = {}


def pts_of(source):
    """各コマの表示時刻（秒、表示順）。素材は平均59.97fpsの可変（時間の単位は1/600秒）。"""
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
    """画面に合わせて配置 → 中心で拡大 → 中心で回転（+が時計回り）→ 位置をずらす（CapCutと同じ順番）。"""
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
    """本編の時刻 t（秒）→ (カット, 素材, コマ番号, 素材の秒)。"""
    c = cut_at(t)
    n, rec_in, _, src, n0, step = c[:6]
    if src is None:
        return c, None, None, None
    k = min(int(round(t * 30)), int(round(c[2] * 30)) - 1) - int(round(rec_in * 30))
    idx = n0 + step * k
    return c, src, idx, pts_of(src)[idx]


def plate(t, W=1920, H=1080, dim=True, ev=None):
    """減光まで入れたクリップの1コマ（重ねる前）。"""
    c, src, idx, sec = source_frame(t)
    if src is None:
        return Image.new('RGB', (W, H), (0, 0, 0)), (c[0], None, None, None)
    fr = capcut_transform(grab(src, idx), c[6], lerp(c[7], sec), c[8], c[9], W, H)
    if dim and t < S.timing(E, T)['dim_until']:
        fr = S.apply_dim(fr, S.DIM_EV[src] if ev is None else ev)
    return fr, (c[0], src, idx, sec)


def render(t, W=1920, H=1080, force_end=None, ev=None):
    fr, info = plate(t, W, H, ev=ev)
    tm = S.timing(E, T)
    fr = S.comp(fr, 'corner', S.envelope(t, tm['corner']))
    a = S.envelope(t, tm['end']) if force_end is None else force_end
    fr = S.comp(fr, 'end', a)
    return fr, info


def measure_at(t, ev=None):
    tm = S.timing(E, T)
    fr, info = plate(t, ev=ev)
    out = {}
    for kind in ('corner', 'end'):
        if S.envelope(t, tm[kind]) >= 0.999:
            out[kind] = S.measure(S.comp(fr, kind, 1.0), kind)
    return out, info


def measure_report():
    for t in (0.60, 1.00, 2.00, 3.00, 4.00, 4.50, 5.00, 5.37, 5.40, 6.00, 6.50, 7.00, 7.60, 8.10, 8.67,
              8.70, 9.30, 10.20, 11.00, 11.40, 11.67, 21.60, 23.00, 26.00, 28.60):
        m, info = measure_at(t)
        print('t=%5.2f %s  %s' % (t, info[1], '  '.join('%s:%s' % (k, ' '.join('%s %.2f' % (n, v) for n, v in d.items()))
                                                      for k, d in m.items())))


CHECK_TIMES = [  # 右下の等倍チェック（本編の秒, 説明）
    (0.00, 'カット1 入り'), (2.00, 'カット1 白波（いちばん明るい）'), (3.50, 'カット1 中'), (5.367, 'カット1 最終コマ'),
    (5.40, 'カット2 入り'), (6.50, 'カット2 砂州の縁'), (8.10, 'カット2 人影が円の縁'), (8.667, 'カット2 最終コマ'),
    (8.70, 'カット3 入り（素材6.70秒）'), (10.20, 'カット3 中（7.45秒）'), (11.40, 'カット3 幹が行の後ろ'),
    (11.667, 'カット3 最終コマ（8.19秒）'),
]


def corner_check(path):
    from PIL import ImageDraw, ImageFont
    box = (1280, 880, 1920, 1080)
    cw, ch, lab, g = box[2] - box[0], box[3] - box[1], 30, 6
    f = ImageFont.truetype(S.font_path('Regular'), 20)
    rows = (len(CHECK_TIMES) + 3) // 4
    grid = Image.new('RGB', (4 * cw + 3 * g, rows * (ch + lab) + (rows - 1) * g), (20, 20, 20))
    d = ImageDraw.Draw(grid)
    for i, (t, label) in enumerate(CHECK_TIMES):
        im, info = render(t)
        x, y = (i % 4) * (cw + g), (i // 4) * (ch + lab + g)
        d.text((x + 6, y + 4), '%.2f秒  %s  %s n=%s' % (t, label, info[1], info[2]), fill=(255, 220, 0), font=f)
        grid.paste(im.crop(box), (x, y + lab))
    grid.save(path)


def main():
    W, H = 1920, 1080
    corner_check(os.path.join(OUT, 'signage_d_corner_check_1to1.png'))
    a, ia = render(2.00)
    b, ib = render(26.00)
    print('a', ia, 'b', ib)
    a.save(os.path.join(OUT, 'signage_a_corner_brightest_rec02.00.png'))
    b.save(os.path.join(OUT, 'signage_b_end_rec26.00.png'))
    box = (1280, 880, 1920, 1080)
    a.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.NEAREST).save(
        os.path.join(OUT, 'signage_a_corner_detail_2x.png'))
    ebox = (560, 440, 1360, 700)
    b.crop(ebox).resize(((ebox[2] - ebox[0]) * 2, (ebox[3] - ebox[1]) * 2), Image.NEAREST).save(
        os.path.join(OUT, 'signage_b_end_detail_2x.png'))
    # 距離の目安：1/4（数メートル先）と1/8（道路の向こう）。左から a（右下）／b（END）
    for div, name in ((4, 'signage_c_distance_1-4.png'), (8, 'signage_c_distance_1-8.png')):
        w, h, g = W // div, H // div, 4
        sm = Image.new('RGB', (w * 2 + g, h), (0, 0, 0))
        for i, im in enumerate((a, b)):
            sm.paste(im.resize((w, h), Image.LANCZOS), (i * (w + g), 0))
        sm.save(os.path.join(OUT, name))
        if div == 8:
            sm.resize((sm.width * 4, sm.height * 4), Image.BICUBIC).save(
                os.path.join(OUT, 'signage_c_distance_1-8_view4x.png'))
    measure_report()


def motion(t0=0.0, t1=5.40, name='signage_g_cut1_motion_0-5.4s.mp4'):
    """本編 t0〜t1 を30fpsの連番で描いて動画にする（白波の上で、減光の楕円が止まったしみに見えないかの確認用）。"""
    d = os.path.join(CACHE, 'motion')
    os.makedirs(d, exist_ok=True)
    n = int(round((t1 - t0) * 30))
    for i in range(n):
        fr, _ = render(t0 + i / 30.0)
        fr.save(os.path.join(d, 'f%04d.png' % i))
    out = os.path.join(OUT, name)
    subprocess.run([FFMPEG, '-v', 'error', '-y', '-framerate', '30', '-i', os.path.join(d, 'f%04d.png'),
                    '-frames:v', str(n), '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', out], check=True)
    print(out)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--motion':
        motion()
    elif len(sys.argv) > 1 and sys.argv[1] == '--measure':
        measure_report()
    else:
        main()
