#!/usr/bin/env python3
"""BGM候補の解析と、IG30／IG15の試聴用プレビューづくり（OpenTracks の候補3曲用）。

使い方（mp3 は OpenTracks の各曲ページ →「音楽素材ダウンロードページへ」→「DOWNLOAD FILE」で
ブラウザから保存し、exports/bgm/candidates/<曲ID>.mp3 に置く）:

  python3 tools/bgm_preview.py analyze            # candidates/ の全mp3を解析 → exports/bgm/analysis.json / analysis.md
  python3 tools/bgm_preview.py build              # 候補3曲（CANDIDATES）の ig30 / ig15 プレビューと抜粋mp3を作る
  python3 tools/bgm_preview.py build --n 2 --ig30-start 12.34 --ig15-start 40.0   # 開始位置を手で指定

作るもの（exports/bgm/）:
  bgm_<n>_<short>_ig30.mp4 / _ig15.mp4   preview_<edit>.mp4 の映像＋（現地音＋BGM）
  bgm_<n>_<short>_ig30.mp3 / _ig15.mp3   BGMの抜粋だけ（タイムラインと同じ位置・音量・フェード）
  bgm_cutpoints.json                      曲の開始オフセット、フェード位置、ゲイン、測定値

音の設計（exports/capcut/ig_manifest.json の bgm_ducking_points に合わせる）:
  IG30: 0.00〜1.50 BGMなし → 1.50 最初の音（フェードインなし）→ 22.30〜23.00 波が開く瞬間は −3.5dB
        → 28.00〜29.70 フェードアウト（または曲の自然な終わりを29.70に合わせる）→ 29.70〜30.00 波音だけ
  IG15: 0.00〜1.20 BGMなし → 1.20〜1.50 フェードイン → 11.30〜12.60 波のJカットの下で −3.5dB
        → 13.00〜14.70 フェードアウト → 14.70〜15.00 波音だけ
  BGMだけの積分ラウドネス（鳴っている区間）を −21 LUFS にそろえ、最後に全体のピークを −1 dBFS 以下にする。
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

FFMPEG = '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
ROOT = '/home/user/maze'
CAND_DIR = os.path.join(ROOT, 'exports/bgm/candidates')
OUT_DIR = os.path.join(ROOT, 'exports/bgm')
CAPCUT = os.path.join(ROOT, 'exports/capcut')
SR = 48000

# 候補3曲（edl/capcut/BGM_CANDIDATES.md と同じ順番）
CANDIDATES = [
    dict(n=1, id=17543, short='kogen_cafe', title='高原の小さなカフェにて', creator='のる'),
    dict(n=2, id=10647, short='odayaka', title='穏やかな世界', creator='こばっと'),
    dict(n=3, id=9307, short='chill_ocean', title='Chill Ocean', creator='MFP【Marron Fields Production】'),
]

EDITS = {
    'ig30': dict(dur=30.0, bgm_in=1.50, fade_in=0.0, anchors=[1.50, 4.00, 13.40, 16.60, 24.70],
                 duck=[(22.30, 23.00, -3.5)], fade_out=(28.00, 29.70)),
    'ig15': dict(dur=15.0, bgm_in=1.20, fade_in=0.30, anchors=[1.50, 3.00, 4.00, 5.50, 6.50, 8.50, 12.00],
                 duck=[(11.30, 12.60, -3.5)], fade_out=(13.00, 14.70)),
}
TARGET_LUFS = -21.0
PEAK_CEIL_DB = -1.0
DUCK_RAMP = 0.20


# ---------------------------------------------------------------- 入出力
def decode(path):
    raw = subprocess.run([FFMPEG, '-v', 'error', '-i', path, '-vn', '-ac', '2', '-ar', str(SR), '-f', 'f32le', '-'],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)


def write_wav(path, a):
    subprocess.run([FFMPEG, '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-',
                    '-c:a', 'pcm_s24le', path], input=a.astype(np.float32).tobytes(), check=True)


def ebur128(a):
    """ffmpeg ebur128 で 積分LUFS・トゥルーピーク・短時間ラウドネス(0.1秒ごと) を返す。"""
    p = subprocess.run([FFMPEG, '-nostats', '-hide_banner', '-v', 'verbose', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-',
                        '-af', 'ebur128=framelog=verbose:peak=true', '-f', 'null', '-'],
                       input=a.astype(np.float32).tobytes(), stderr=subprocess.PIPE).stderr.decode('utf8', 'replace')
    num = r'(-?(?:[\d.]+|inf|nan))'
    fl = lambda v: -70.0 if ('nan' in v or 'inf' in v) else float(v)
    st = [(float(m[1]), fl(m[2]), fl(m[3])) for m in
          re.finditer(r't:\s*([\d.]+)\s+TARGET.*?M:\s*' + num + r'\s+S:\s*' + num, p)]
    I = float(re.findall(r'I:\s+(-?[\d.]+) LUFS', p)[-1])
    tp = re.findall(r'Peak:\s+(-?[\d.inf]+) dBFS', p)
    return dict(I=I, true_peak=float(tp[-1]) if tp else None, short_term=st)


# ---------------------------------------------------------------- 解析
def stft_mag(x, n=2048, hop=512):
    w = np.hanning(n)
    frames = 1 + max(0, (len(x) - n) // hop)
    idx = np.arange(n)[None, :] + hop * np.arange(frames)[:, None]
    return np.abs(np.fft.rfft(x[idx] * w, axis=1)), SR / hop


def onset_env(mag):
    lm = np.log1p(100 * mag)
    flux = np.maximum(0, np.diff(lm, axis=0)).sum(1)
    flux = np.concatenate([[0], flux])
    k = 16
    loc = np.convolve(flux, np.ones(2 * k + 1) / (2 * k + 1), 'same')
    o = np.maximum(0, flux - loc)
    return o / (o.max() + 1e-9)


def tempo(o, fps, lo=55, hi=180, prior=90):
    o = o - o.mean()
    ac = np.correlate(o, o, 'full')[len(o) - 1:]
    lags = np.arange(len(ac))
    bpm = 60 * fps / np.maximum(lags, 1)
    ok = (bpm >= lo) & (bpm <= hi)
    weight = np.exp(-0.5 * (np.log2(bpm / prior) / 0.9) ** 2)
    score = np.where(ok, ac * weight, -np.inf)
    best = int(np.argmax(score))
    # 放物線補間
    if 1 <= best < len(ac) - 1:
        a, b, c = ac[best - 1], ac[best], ac[best + 1]
        d = 0.5 * (a - c) / (a - 2 * b + c + 1e-12)
    else:
        d = 0
    lag = best + d
    cands = sorted([(score[i], 60 * fps / i) for i in range(2, len(ac) - 1)
                    if ok[i] and score[i] > score[i - 1] and score[i] > score[i + 1]], reverse=True)[:3]
    strength = float(ac[best] / (ac[0] + 1e-9))
    return 60 * fps / lag, [round(c[1], 1) for c in cands], strength


def beat_track(o, fps, bpm, tight=100.0):
    """Ellis の動的計画法。"""
    period = 60 * fps / bpm
    n = len(o)
    score = o.copy()
    back = -np.ones(n, int)
    lo, hi = int(round(0.5 * period)), int(round(2 * period))
    offs = np.arange(-hi, -lo + 1)
    pen = -tight * np.log(-offs / period) ** 2
    for i in range(hi, n):
        cand = score[i + offs] + pen
        j = int(np.argmax(cand))
        score[i] = o[i] + cand[j]
        back[i] = i + offs[j]
    i = int(np.argmax(score[-int(period * 2):])) + n - int(period * 2)
    beats = []
    while i >= 0:
        beats.append(i)
        i = back[i]
    return np.array(beats[::-1]) / fps


def chroma(mag):
    f = np.fft.rfftfreq(2 * (mag.shape[1] - 1), 1 / SR)
    ok = (f > 60) & (f < 5000)
    pc = np.round(12 * np.log2(f[ok] / 440.0)) % 12
    C = np.zeros((mag.shape[0], 12))
    for k in range(12):
        C[:, k] = mag[:, ok][:, pc == k].sum(1)
    return C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)


def novelty(feat, fps, win_s=6.0):
    """ビート非依存の 1秒単位の特徴で、チェッカーボード・カーネルの新規性。"""
    step = int(fps)
    F = np.array([feat[i:i + step].mean(0) for i in range(0, len(feat) - step, step)])
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    S = F @ F.T
    L = int(win_s)
    g = np.outer(np.hanning(2 * L), np.hanning(2 * L))
    sign = np.ones((2 * L, 2 * L))
    sign[:L, L:] = sign[L:, :L] = -1
    K = g * sign
    nov = np.zeros(len(F))
    for i in range(L, len(F) - L):
        nov[i] = (S[i - L:i + L, i - L:i + L] * K).sum()
    nov = np.maximum(nov, 0)
    nov /= nov.max() + 1e-9
    peaks = [i for i in range(1, len(nov) - 1) if nov[i] > 0.3 and nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1]]
    return nov, peaks   # peaks は秒


def vocal_cue(x, mag, fps):
    """声らしさの目安（0〜1）。声の帯域(300〜3400Hz)の、4〜8Hzの音節リズムでの揺れの強さ。楽器でも上がるので目安のみ。"""
    f = np.fft.rfftfreq(2 * (mag.shape[1] - 1), 1 / SR)
    band = mag[:, (f > 300) & (f < 3400)].sum(1)
    env = np.log1p(band / (band.mean() + 1e-9))
    env = env - np.convolve(env, np.ones(int(fps)) / int(fps), 'same')
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) ** 2
    fm = np.fft.rfftfreq(len(env), 1 / fps)
    syl = spec[(fm > 4) & (fm < 8)].sum()
    tot = spec[(fm > 0.5) & (fm < 12)].sum() + 1e-9
    # 倍音の平坦さ（声はフォルマントで中域が盛り上がる）
    mid = mag[:, (f > 800) & (f < 3000)].mean()
    low = mag[:, (f > 100) & (f < 800)].mean() + 1e-9
    return dict(syllabic_ratio=round(float(syl / tot), 3), mid_to_low=round(float(mid / low), 3))


def ending_shape(st, total):
    """短時間ラウドネス（0.1秒ごと）から、最後の15秒の形を分類する。"""
    t = np.array([s[0] for s in st])
    S = np.array([s[2] for s in st])
    S[~np.isfinite(S)] = -70
    body = np.median(S[(t > 10) & (t < total - 20)]) if total > 40 else np.median(S)
    last = S[t > total - 15]
    tl = t[t > total - 15]
    audible = tl[last > body - 20]
    end_audible = float(audible[-1]) if len(audible) else float(total)
    drop5 = float(np.median(S[(t > total - 12) & (t < total - 8)]) - np.median(S[(t > total - 3) & (t < total - 1)]))
    if total - end_audible > 1.5 or drop5 > 12:
        kind = '自然に終わる（減衰・フェード。終わりの約%.1f秒前から静か）' % (total - end_audible)
    elif drop5 > 5:
        kind = '終わりに向けて音量が下がる（約%.0fdB）' % drop5
    else:
        kind = 'ほぼ同じ音量のまま終わる（ループ向け／ブツ切り）'
    return dict(body_short_term_lufs=round(float(body), 1), last_audible_s=round(end_audible, 2),
                drop_db_last10s=round(drop5, 1), kind=kind)


def analyze(path):
    a = decode(path)
    x = a.mean(1)
    total = len(x) / SR
    mag, fps = stft_mag(x)
    o = onset_env(mag)
    bpm, bpm_alts, pulse = tempo(o, fps)
    beats = beat_track(o, fps, bpm)
    C = chroma(mag)
    rms = np.sqrt((mag ** 2).mean(1, keepdims=True))
    feat = np.hstack([C, np.log1p(rms / (rms.mean() + 1e-9))])
    nov, peaks = novelty(feat, fps)
    L = ebur128(a)
    first = float(np.argmax(np.abs(x) > 10 ** (-40 / 20)) / SR)
    return dict(file=os.path.basename(path), seconds=round(total, 2), first_sound_s=round(first, 2),
                bpm=round(bpm, 1), bpm_alternatives=bpm_alts, pulse_clarity=round(pulse, 3),
                beats=[round(b, 3) for b in beats], sections_s=peaks,
                integrated_lufs=L['I'], true_peak_dbfs=L['true_peak'],
                loudness_curve_1s=[round(s[2], 1) for s in L['short_term'][9::10]],
                ending=ending_shape(L['short_term'], total), vocal_cue=vocal_cue(x, mag, fps),
                _onset=o, _fps=fps, _audio=a)


# ---------------------------------------------------------------- 開始位置
def onset_at(o, fps, t):
    i = int(round(t * fps))
    w = 3      # ±32ms
    if i < 0 or i >= len(o):
        return 0.0
    return float(o[max(0, i - w):i + w + 1].max())


def choose_start(an, edit):
    """アンカー（編集点）に曲のアタックが重なり、かつ曲の区切り（セクション頭）から始まる開始オフセットを探す。"""
    e = EDITS[edit]
    o, fps = an['_onset'], an['_fps']
    total = an['seconds']
    need = e['fade_out'][1] - e['anchors'][0]
    rel = [t - e['anchors'][0] for t in e['anchors']]
    sections = set(an['sections_s'])
    scores = []
    for b in an['beats']:
        if b < an['first_sound_s'] or b + need > total - 0.5:
            continue
        s = sum(onset_at(o, fps, b + r) for r in rel) / len(rel)
        s += 0.6 * onset_at(o, fps, b)                          # 1.50 の最初の音ははっきり
        if any(abs(b - p) <= 0.6 for p in sections):
            s += 0.35                                            # フレーズ／セクションの頭
        if b < 2.0:
            s += 0.15                                            # 曲頭から使えるなら少し優先
        scores.append((b, s))
    if not scores:
        raise SystemExit('%s: %s に足りる長さがない' % (an['file'], edit))
    top = max(s for _, s in scores)
    b, s = next((b, s) for b, s in scores if s >= top - 0.08)   # ほぼ同点なら、曲の前の方（イントロ寄り）を使う
    return round(b, 3), round(s, 3)


def natural_end_start(an, edit):
    """曲の自然な終わりを フェード終わり位置（29.70／14.70）に合わせた場合の開始オフセット。"""
    e = EDITS[edit]
    t = an['ending']['last_audible_s'] - (e['fade_out'][1] - e['anchors'][0])
    beats = np.array(an['beats'])
    if len(beats):
        t = float(beats[np.argmin(np.abs(beats - t))])        # 1.50 の頭が拍に乗るように、近い拍へ寄せる
    return round(t, 3)


# ---------------------------------------------------------------- ミックス
def envelope(edit, n):
    e = EDITS[edit]
    t = np.arange(n) / SR
    g = np.ones(n)
    g[t < e['bgm_in']] = 0
    if e['fade_in'] > 0:
        m = (t >= e['bgm_in']) & (t < e['bgm_in'] + e['fade_in'])
        g[m] = ((t[m] - e['bgm_in']) / e['fade_in']) ** 2
    for t0, t1, db in e['duck']:
        d = 10 ** (db / 20)
        k = np.ones(n)
        k[(t >= t0) & (t < t1)] = d
        up = (t >= t0 - DUCK_RAMP) & (t < t0)
        k[up] = 1 + (d - 1) * (t[up] - (t0 - DUCK_RAMP)) / DUCK_RAMP
        dn = (t >= t1) & (t < t1 + DUCK_RAMP)
        k[dn] = d + (1 - d) * (t[dn] - t1) / DUCK_RAMP
        g *= k
    f0, f1 = e['fade_out']
    m = (t >= f0) & (t < f1)
    g[m] *= np.cos(0.5 * np.pi * (t[m] - f0) / (f1 - f0))      # 等パワーのフェード
    g[t >= f1] = 0
    return g


def place(an, edit, start):
    e = EDITS[edit]
    n = int(round(e['dur'] * SR))
    out = np.zeros((n, 2))
    i0 = int(round(e['anchors'][0] * SR))           # 曲の start 秒をタイムラインの 1.50 に置く
    s0 = int(round(start * SR)) - (i0 - int(round(e['bgm_in'] * SR)))  # フェードイン分だけ前から
    src_from = max(0, s0)
    dst_from = int(round(e['bgm_in'] * SR)) + (src_from - s0)
    seg = an['_audio'][src_from:src_from + (n - dst_from)]
    out[dst_from:dst_from + len(seg)] = seg
    return out * envelope(edit, n)[:, None]


def load_ambience(edit):
    wav = os.path.join(CAPCUT, '%s_ambience.wav' % edit)
    src = wav if os.path.exists(wav) else os.path.join(CAPCUT, 'preview_%s.mp4' % edit)
    a = decode(src)
    n = int(round(EDITS[edit]['dur'] * SR))
    return np.pad(a, ((0, max(0, n - len(a))), (0, 0)))[:n], src


def limit(mix):
    """ピークが −1 dBFS を超える所だけ、先読みのなめらかなゲインで下げる。"""
    ceil = 10 ** ((PEAK_CEIL_DB - 0.3) / 20)
    peak = np.abs(mix).max(1)
    need = np.minimum(1, ceil / np.maximum(peak, 1e-9))
    w = int(0.005 * SR)
    g = np.array([need[max(0, i - w):i + w].min() for i in range(0, len(need), w)])
    g = np.repeat(g, w)[:len(need)]
    k = np.hanning(4 * w)
    k /= k.sum()
    gp = np.pad(g, (2 * w, 2 * w), mode='edge')
    g = np.minimum(g, np.convolve(gp, k, 'same')[2 * w:-2 * w])
    return mix * g[:, None], float(20 * np.log10(g.min()))


def build_one(c, an, edit, start, mode):
    bgm = place(an, edit, start)
    e = EDITS[edit]
    t = np.arange(len(bgm)) / SR
    active = bgm[(t >= e['anchors'][0]) & (t < e['fade_out'][0])]
    I = ebur128(active)['I']
    gain_db = TARGET_LUFS - I
    bgm *= 10 ** (gain_db / 20)
    amb, amb_src = load_ambience(edit)
    mix, lim_db = limit(amb + bgm)
    m = ebur128(mix)
    tag = 'bgm_%d_%s_%s' % (c['n'], c['short'], edit)
    wav = os.path.join(OUT_DIR, tag + '.wav')
    write_wav(wav, mix)
    subprocess.run([FFMPEG, '-v', 'error', '-y', '-i', os.path.join(CAPCUT, 'preview_%s.mp4' % edit), '-i', wav,
                    '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k',
                    '-shortest', '-movflags', '+faststart', os.path.join(OUT_DIR, tag + '.mp4')], check=True)
    os.remove(wav)
    # BGMの抜粋だけ（タイムライン位置のまま）
    bg, _ = limit(bgm)
    tmp = os.path.join(OUT_DIR, tag + '_bgm.wav')
    write_wav(tmp, bg)
    subprocess.run([FFMPEG, '-v', 'error', '-y', '-i', tmp, '-c:a', 'libmp3lame', '-b:a', '256k',
                    os.path.join(OUT_DIR, tag + '.mp3')], check=True)
    os.remove(tmp)
    src_end = start + (e['fade_out'][1] - e['anchors'][0])
    return dict(file_mp4=tag + '.mp4', file_mp3=tag + '.mp3', mode=mode,
                track_start_s=start, placed_at_timeline_s=e['anchors'][0],
                bgm_audible_from_timeline_s=e['bgm_in'], fade_in_s=e['fade_in'],
                fade_out_timeline_s=list(e['fade_out']),
                fade_out_track_s=[round(start + e['fade_out'][0] - e['anchors'][0], 3), round(src_end, 3)],
                ducking=e['duck'], bgm_gain_db=round(gain_db, 2), bgm_lufs_before_gain=I,
                bgm_lufs_after=TARGET_LUFS, ambience_source=os.path.relpath(amb_src, ROOT),
                limiter_max_reduction_db=round(lim_db, 2), mix_integrated_lufs=m['I'],
                mix_true_peak_dbfs=m['true_peak'])


def strip(an):
    return {k: v for k, v in an.items() if not k.startswith('_') and k != 'beats'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['analyze', 'build'])
    ap.add_argument('--n', type=int, help='build: 候補番号だけ作る')
    ap.add_argument('--ig30-start', type=float)
    ap.add_argument('--ig15-start', type=float)
    ap.add_argument('--mode', choices=['auto', 'beat', 'end'], default='auto',
                    help='beat=編集点に合う位置から始めてフェードアウト、end=曲の自然な終わりを最後に合わせる')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.cmd == 'analyze':
        res = {}
        for p in sorted(glob.glob(os.path.join(CAND_DIR, '*.mp3'))):
            print('解析', p, file=sys.stderr)
            res[os.path.basename(p)] = strip(analyze(p))
        json.dump(res, open(os.path.join(OUT_DIR, 'analysis.json'), 'w'), ensure_ascii=False, indent=1)
        with open(os.path.join(OUT_DIR, 'analysis.md'), 'w') as f:
            f.write('| ファイル | 長さ | BPM（候補） | 拍の明瞭さ | セクションの変わり目（秒） | 積分LUFS | 終わり方 | 声の目安 |\n|---|---|---|---|---|---|---|---|\n')
            for k, r in res.items():
                f.write('| %s | %.1f | %.1f (%s) | %.2f | %s | %.1f | %s | 音節%.2f／中域%.2f |\n' % (
                    k, r['seconds'], r['bpm'], ', '.join(map(str, r['bpm_alternatives'])), r['pulse_clarity'],
                    ' '.join(map(str, r['sections_s'][:14])), r['integrated_lufs'], r['ending']['kind'],
                    r['vocal_cue']['syllabic_ratio'], r['vocal_cue']['mid_to_low']))
        print(open(os.path.join(OUT_DIR, 'analysis.md')).read())
        return

    log_path = os.path.join(OUT_DIR, 'bgm_cutpoints.json')
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    for c in CANDIDATES:
        if args.n and c['n'] != args.n:
            continue
        path = os.path.join(CAND_DIR, '%d.mp3' % c['id'])
        if not os.path.exists(path):
            print('見つからない:', path, '（OpenTracksのダウンロードページから保存して置く）', file=sys.stderr)
            continue
        an = analyze(path)
        entry = dict(title=c['title'], creator=c['creator'], id=c['id'], analysis=strip(an))
        for edit in ('ig30', 'ig15'):
            manual = getattr(args, edit + '_start')
            if manual is not None:
                start, mode = manual, 'manual'
            else:
                bstart, bscore = choose_start(an, edit)
                estart = natural_end_start(an, edit)
                natural = an['ending']['kind'].startswith('自然')
                if args.mode == 'end' or (args.mode == 'auto' and edit == 'ig30' and natural and estart >= 0):
                    start, mode = estart, 'end（曲の自然な終わりを29.70に合わせる）'
                else:
                    start, mode = bstart, 'beat（編集点のアタック一致 %.2f）' % bscore
            print(c['n'], c['short'], edit, 'start', start, mode, file=sys.stderr)
            entry[edit] = build_one(c, an, edit, start, mode)
        log[str(c['n'])] = entry
    json.dump(log, open(log_path, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: {e: {kk: v[e][kk] for kk in ('track_start_s', 'fade_out_track_s', 'mode',
                                                           'mix_integrated_lufs', 'mix_true_peak_dbfs')}
                          for e in ('ig30', 'ig15') if e in v} for k, v in log.items()},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
