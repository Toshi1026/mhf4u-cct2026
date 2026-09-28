#!/usr/bin/env python3
"""書き出した動画を、見本（CapCut用に作った参照プレビュー）と比べる。

  python3 scripts/verify.py                       # exports/final/ の6本すべて
  python3 scripts/verify.py IG30 --dir remotion/out --offset 600   # 試し書き（600コマ目からの一部）

調べること：コマ数・長さ・解像度・色のタグ・音声トラック、見本との画素の差（RGB 0〜255 の平均絶対差）。
比べるコマは、全体に散らした10コマ＋透過PNGのフェードの始まり・終わり（キーフレームのコマと、その1コマ前後）。
結果は <dir>/verify_report.json と、目視用の並べ画像 <dir>/_verify/<id>_*.jpg。
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RM = os.path.dirname(HERE)
ROOT = os.path.dirname(RM)


def comp_bins():
    base = os.path.join(RM, 'node_modules', '@remotion')
    for d in sorted(os.listdir(base)):
        if d.startswith('compositor-'):
            p = os.path.join(base, d)
            return os.path.join(p, 'ffmpeg'), os.path.join(p, 'ffprobe'), p
    raise SystemExit('Remotion の ffmpeg が見つかりません')


_, FFPROBE, LIB = comp_bins()
ENV = dict(os.environ, LD_LIBRARY_PATH=LIB)
# コマの取り出しは、フィルタがそろった ffmpeg（imageio-ffmpeg 同梱。見本を作ったのと同じもの）で行う
# （Remotion 同梱の ffmpeg は select や rawvideo 出力を持たない最小構成）
FFMPEG = os.environ.get('VERIFY_FFMPEG',
                        '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')


def probe(path):
    r = subprocess.run([FFPROBE, '-v', 'error', '-count_frames', '-show_streams', '-show_format', '-of', 'json', path],
                       capture_output=True, text=True, env=ENV, check=True)
    d = json.loads(r.stdout)
    v = [s for s in d['streams'] if s['codec_type'] == 'video'][0]
    a = [s for s in d['streams'] if s['codec_type'] == 'audio']
    return dict(
        width=v['width'], height=v['height'], frames=int(v['nb_read_frames']), fps=v['r_frame_rate'],
        codec=v['codec_name'], profile=v.get('profile'), pix_fmt=v['pix_fmt'],
        color_range=v.get('color_range'), color_space=v.get('color_space'),
        color_primaries=v.get('color_primaries'), color_transfer=v.get('color_transfer'),
        duration=float(d['format']['duration']), video_duration=float(v.get('duration', 0)),
        audio=[dict(codec=s['codec_name'], sample_rate=int(s['sample_rate']), channels=s['channels'],
                    duration=float(s.get('duration', 0))) for s in a],
        size_mb=round(os.path.getsize(path) / 1048576, 2))


def frames_rgb(path, idx, w, h):
    """指定コマを BT.709・limited として RGB に直して返す（両方を同じ変換で読むので公平）。"""
    sel = '+'.join('eq(n\\,%d)' % i for i in idx)
    vf = "select='%s',scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=bilinear+accurate_rnd+full_chroma_int,format=rgb24" % sel
    r = subprocess.run([FFMPEG, '-v', 'error', '-i', path, '-vf', vf, '-vsync', '0', '-f', 'rawvideo', '-'],
                       capture_output=True, check=True)
    arr = np.frombuffer(r.stdout, np.uint8)
    n = arr.size // (w * h * 3)
    return arr[:n * w * h * 3].reshape(n, h, w, 3)


def pcm(path):
    r = subprocess.run([FFMPEG, '-v', 'error', '-i', path, '-map', '0:a:0', '-ac', '2', '-ar', '48000', '-f', 'f32le', '-'],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).reshape(-1, 2).astype(np.float64)


def audio_compare(out, wav, offset_frames=0):
    """出力の音と、元の現地音 wav を比べる（位置ずれ・音量差・差分の大きさ）。"""
    a, b = pcm(out), pcm(wav)
    b = b[offset_frames * 1600:]
    L = min(48000, len(a) - 30000)
    seg = b[24000:24000 + L, 0]
    best = max(range(-4096, 4097), key=lambda k: float(np.dot(a[24000 + k:24000 + k + L, 0], seg))
               if 24000 + k + L <= len(a) else -1e9)
    x = a[max(0, best):]
    y = b[max(0, -best):]
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    rms = lambda v: 20 * np.log10(np.sqrt(np.mean(v ** 2)) + 1e-12)
    return dict(samples_out=len(a), offset_samples=best, rms_out_db=round(float(rms(x)), 2),
                rms_wav_db=round(float(rms(y)), 2), residual_db=round(float(rms(x - y) - rms(y)), 1),
                peak_out_dbfs=round(float(20 * np.log10(np.abs(a).max() + 1e-12)), 2))


def main():
    argv = sys.argv[1:]

    def opt(name, default):
        if name in argv:
            i = argv.index(name)
            v = argv[i + 1]
            del argv[i:i + 2]
            return v
        return default

    d = os.path.abspath(opt('--dir', os.path.join(ROOT, 'exports', 'final')))
    offset = int(opt('--offset', '0'))
    tl = json.load(open(os.path.join(RM, 'src', 'timeline.json'), encoding='utf-8'))
    comps = [c for c in tl['compositions'] if not argv or c['id'] in argv]
    vis = os.path.join(d, '_verify')
    os.makedirs(vis, exist_ok=True)
    report = {}
    for c in comps:
        out = os.path.join(d, c['output'])
        ref = os.path.join(ROOT, c['reference'])
        if not os.path.exists(out):
            print(c['id'], 'なし', out)
            continue
        po, pr = probe(out), probe(ref)
        n_out = po['frames']
        W, H = c['width'], c['height']
        # 比べるコマ（出力のコマ番号）
        even = [int(round(i * (n_out - 1) / 9)) for i in range(10)]
        fades = []
        for o in c['overlays']:
            for f, _ in o['keyframes']:
                fades += [f - 1, f, f + 1]
            fades += [o['from'] + (o['keyframes'][1][0] - o['keyframes'][0][0]) // 2]  # フェードの途中
        idx = sorted({i for i in even + [f - offset for f in fades] if 0 <= i < n_out})
        A = frames_rgb(out, idx, W, H)
        B = frames_rgb(ref, [i + offset for i in idx], W, H)
        # 素材のクリップそのもの（PNGが出ていないコマだけ）とも比べる：見本も出力も、クリップを
        # もう一度圧縮したものなので、「クリップ→見本」の差が、到達できる下限の目安になる
        clip_of = {}
        for i in idx:
            f = i + offset
            vis_ov = any(o['from'] <= f < o['to'] and
                         np.interp(f, [q[0] for q in o['keyframes']], [q[1] for q in o['keyframes']]) > 0
                         for o in c['overlays'])
            if vis_ov:
                continue
            for cl in c['clips']:
                if cl['from'] <= f < cl['from'] + cl['durationInFrames']:
                    clip_of.setdefault(cl['file'], []).append((i, f - cl['from']))
        C = {}
        for fpath, lst in clip_of.items():
            arr = frames_rgb(os.path.join(ROOT, fpath), [q[1] for q in lst], W, H)
            for (i, _), a in zip(lst, arr):
                C[i] = a
        rows = []
        for k, i in enumerate(idx):
            diff = np.abs(A[k].astype(np.int16) - B[k].astype(np.int16))
            extra = {}
            if i in C:
                extra = dict(out_vs_clip=round(float(np.abs(A[k].astype(np.int16) - C[i]).mean()), 4),
                             ref_vs_clip=round(float(np.abs(B[k].astype(np.int16) - C[i]).mean()), 4))
            rows.append(dict(frame=i + offset, mad=round(float(diff.mean()), 4), max=int(diff.max()), **extra,
                             p999=int(np.percentile(diff, 99.9)),
                             mean_out=round(float(A[k].mean()), 3), mean_ref=round(float(B[k].mean()), 3)))
        mads = [r['mad'] for r in rows]
        fade_rows = [r for r in rows if r['frame'] in set(fades)]
        cr = [r for r in rows if 'out_vs_clip' in r]
        clipcmp = dict(n=len(cr), out_vs_clip=round(float(np.mean([r['out_vs_clip'] for r in cr])), 4) if cr else None,
                       ref_vs_clip=round(float(np.mean([r['ref_vs_clip'] for r in cr])), 4) if cr else None)
        rep = dict(output=os.path.relpath(out, ROOT), reference=c['reference'], probe_out=po, probe_ref=pr,
                   frames_expected=c['durationInFrames'], mad_mean=round(float(np.mean(mads)), 4),
                   mad_max=round(float(np.max(mads)), 4), max_abs=int(max(r['max'] for r in rows)),
                   fade_mad_max=round(max([r['mad'] for r in fade_rows] or [0]), 4), vs_clip=clipcmp, samples=rows)
        if c['audio'] and po['audio']:
            rep['audio_vs_wav'] = audio_compare(out, os.path.join(ROOT, c['audio']['file']), offset)
            print('   音：', rep['audio_vs_wav'])
        report[c['id']] = rep
        print('%-14s %s %dx%d %dコマ(期待%d・見本%d) %.3f秒 %s/%s/%s/%s 音声%s  MAD平均%.3f 最大%.3f（フェード%.3f） 画素最大差%d  %.1fMB  [PNGなしのコマ%d枚でクリップ比：出力%.3f／見本%.3f]' % (
            c['id'], po['codec'], po['width'], po['height'], po['frames'], c['durationInFrames'], pr['frames'],
            po['duration'], po['color_space'], po['color_primaries'], po['color_transfer'], po['color_range'],
            ('%s %dHz %dch' % (po['audio'][0]['codec'], po['audio'][0]['sample_rate'], po['audio'][0]['channels'])
             if po['audio'] else 'なし'), rep['mad_mean'], rep['mad_max'], rep['fade_mad_max'], rep['max_abs'],
            po['size_mb'], clipcmp['n'], clipcmp['out_vs_clip'] or 0, clipcmp['ref_vs_clip'] or 0))
        # 目視用：いちばん差の大きいコマと、フェード途中のコマを「出力｜見本｜差×8」で並べる
        worst = max(range(len(rows)), key=lambda k: rows[k]['mad'])
        mids = {o['from'] + (o['keyframes'][1][0] - o['keyframes'][0][0]) // 2 for o in c['overlays']}
        picks = sorted({worst} | {k for k, r in enumerate(rows) if r['frame'] in mids})
        for k in picks:
            diff = np.clip(np.abs(A[k].astype(np.int16) - B[k].astype(np.int16)) * 8, 0, 255).astype(np.uint8)
            strip = np.concatenate([A[k], B[k], diff], axis=1)
            im = Image.fromarray(strip)
            im.thumbnail((2400, 2400))
            im.save(os.path.join(vis, '%s_f%04d.jpg' % (c['id'], rows[k]['frame'])), quality=85)
    json.dump(report, open(os.path.join(d, 'verify_report.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
