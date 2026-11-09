"""耳の代わり：wav / mp4 / フォルダを「数値・画像・文字」にして、聞けないエージェントでも音を判断できるようにする。

  python3 tools/v3/listen.py exports/v3/mix.wav
  python3 tools/v3/listen.py exports/v3/vo/                # フォルダ：中の音声を1本ずつ（読みの確認つき）
  python3 tools/v3/listen.py exports/v3/MAZE_v3_ig_A.mp4 --name ig_A
  → exports/v3/listen/<name>/report.md と PNG（overview.png、10秒ごとの zoom_*.png）

出すもの
  (a) ラウドネス：ffmpeg ebur128（統合 LUFS・LRA・トゥルーピーク）＋ pyloudnorm の照合、短期（3秒）LUFS の推移、
      サンプルピーク、クリップ数、DC、左右の相関、頭と尻の音量（プチ音・ブツ切れ）
  (b) 波形＋メルスペクトログラム＋短期 LUFS の PNG（拍の線、オンセット、Whisper の語を重ねる）
  (c) AudioSet のラベル（YAMNet 521種、PANNs Cnn14 527種。どちらも ONNX・CPU）を時間の窓ごとに
  (d) CLAP（LAION clap-htsat-unfused の ONNX）で、英語の説明文との近さ（-1〜1）を窓ごとに
  (e) faster-whisper（既定 large-v3-turbo・int8・CPU）で語ごとの時刻つき文字起こし。
      同じフォルダに moras.json があれば想定の読み（カタカナ）と比べ、listen_expect.json のキーワードも確認

モデルは /home/user/vendor/audio-models（環境変数 LISTEN_MODELS で変更）。無いモデルの項目は飛ばして続ける。
CPU を占有しないよう、ONNX と Whisper は既定 2 スレッド（--threads）。重い処理はしない（39秒の音で2分弱）。
"""
import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
MODELS = os.environ.get('LISTEN_MODELS', '/home/user/vendor/audio-models')
SR = 48000
AUDIO_EXT = ('.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.mp4', '.mov', '.m4v', '.webm')

DEFAULT_PROMPTS = [
    'calm ambient synth pad', 'gentle ocean waves on a beach', 'a narrator speaking calmly',
    'soft kick drum beat', 'a clear bell chime', 'upbeat pop music', 'tense cinematic music',
    'wind noise on a microphone', 'harsh distorted noise', 'clipping digital distortion',
    'muffled low quality audio', 'silence',
]
BANDS = [('sub<40', 0, 40), ('low40-150', 40, 150), ('lowmid150-500', 150, 500), ('mid500-2k', 500, 2000),
         ('pres2k-5k', 2000, 5000), ('air5k-12k', 5000, 12000), ('top>12k', 12000, 24000)]


# ---------------------------------------------------------------- 読み込み
def decode(path):
    """ffmpeg で 48kHz の float に。mp4 でも可。戻り値 (N, ch)"""
    ch = int(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=channels',
                             '-of', 'csv=p=0', path], capture_output=True, text=True).stdout.strip() or 0)
    if ch == 0:
        raise SystemExit('音声トラックがありません: ' + path)
    ch = min(ch, 2)
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vn', '-ac', str(ch), '-ar', str(SR), '-f', 'f32le', '-'],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, '<f4').reshape(-1, ch).copy()


def db(x, floor=-120.0):
    return float(max(floor, 20 * np.log10(max(float(x), 1e-12))))


def resample(y, sr_to):
    import librosa
    return librosa.resample(y, orig_sr=SR, target_sr=sr_to, res_type='soxr_hq').astype(np.float32)


# ---------------------------------------------------------------- (a) ラウドネス
def ebur128(path):
    p = subprocess.run(['ffmpeg', '-nostats', '-v', 'info', '-i', path, '-vn', '-af', 'ebur128=peak=true:framelog=quiet',
                        '-f', 'null', '-'], capture_output=True, text=True)
    s = p.stderr[p.stderr.rfind('Summary:'):]
    g = lambda pat: (float(re.search(pat, s).group(1)) if re.search(pat, s) else None)
    return dict(I=g(r'I:\s+(-?[\d.]+) LUFS'), LRA=g(r'LRA:\s+(-?[\d.]+) LU'), TP=g(r'Peak:\s+(-?[\d.]+|-inf) dBFS'),
                LRA_low=g(r'LRA low:\s+(-?[\d.]+)'), LRA_high=g(r'LRA high:\s+(-?[\d.]+)'))


def loudness(x, path):
    import pyloudnorm as pyln
    meter = pyln.Meter(SR)
    mono = x.mean(axis=1)
    r = dict(ffmpeg=ebur128(path))
    try:
        r['pyln_I'] = float(meter.integrated_loudness(x))
    except Exception:
        r['pyln_I'] = None
    r['sample_peak'] = db(np.abs(x).max())
    r['clips'] = int((np.abs(x) >= 0.999).sum())
    r['dc'] = float(mono.mean())
    r['corr'] = None if x.shape[1] == 1 else float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) if x.shape[1] > 1 and x[:, 0].std() > 0 and x[:, 1].std() > 0 else None
    r['head_10ms_peak'] = db(np.abs(x[:int(.01 * SR)]).max())
    r['tail_50ms_rms'] = db(np.sqrt((x[-int(.05 * SR):] ** 2).mean()))
    # 短期（3秒）LUFS を 0.5 秒刻みで。窓の中心の時刻を返す
    st_t, st = [], []
    win, hop = 3.0, 0.5
    for t0 in np.arange(0, max(len(x) / SR - win, 0) + 1e-6, hop):
        seg = x[int(t0 * SR):int((t0 + win) * SR)]
        try:
            v = meter.integrated_loudness(seg)  # 3秒の窓1つ＝短期ラウドネスの近似（ゲートは窓内）
        except Exception:
            v = -np.inf
        st_t.append(t0 + win / 2)
        st.append(float(v) if np.isfinite(v) else -70.0)
    r['short_t'], r['short'] = st_t, st
    # 100ms ごとの RMS（無音の割合と PNG 用）
    n = int(.1 * SR)
    rms = np.sqrt((mono[:len(mono) // n * n].reshape(-1, n) ** 2).mean(axis=1)) if len(mono) >= n else np.array([0.0])
    r['silence_ratio'] = float((20 * np.log10(np.maximum(rms, 1e-9)) < -60).mean())
    return r


def region_lufs(x, spans):
    """スパン（秒）だけをつなげた統合 LUFS"""
    import pyloudnorm as pyln
    if not spans:
        return None
    seg = np.concatenate([x[int(a * SR):int(b * SR)] for a, b in spans if b > a] or [np.zeros((1, x.shape[1]))])
    if len(seg) < int(0.5 * SR):
        return None
    try:
        v = pyln.Meter(SR).integrated_loudness(seg)
        return float(v) if np.isfinite(v) else None
    except Exception:
        return None


# ---------------------------------------------------------------- スペクトル・拍
def spectral(mono):
    import librosa
    S = np.abs(librosa.stft(mono, n_fft=4096, hop_length=1024)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=4096)
    tot = S.sum() + 1e-12
    bands = {name: float(S[(f >= lo) & (f < hi)].sum() / tot * 100) for name, lo, hi in BANDS}
    cent = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=SR)[0]
    flat = librosa.feature.spectral_flatness(S=np.sqrt(S))[0]
    return dict(bands=bands, centroid=float(np.median(cent)), flatness=float(np.median(flat)))


def clicks(mono):
    """プチ音の候補：8kHz 以上のエネルギーが前後 20ms より 20dB 以上高い、1〜2 コマ（2.7ms）だけの山。
    子音（サ行・タ行）は 10ms 以上続くので拾わない。デジタルの完全な無音（50ms 以上の 0）も返す"""
    import librosa
    S = np.abs(librosa.stft(mono, n_fft=256, hop_length=128)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=256)
    e = 10 * np.log10(S[f > 8000].sum(0) + 1e-12)
    out = []
    for i in range(8, len(e) - 8):
        ref = max(np.median(e[i - 8:i - 1]), np.median(e[i + 2:i + 9]))
        if e[i] - ref >= 20 and e[i] >= e[i - 1] and e[i] >= e[i + 1] and e[i] > -40:
            out.append((round(i * 128 / SR, 3), round(float(e[i] - ref), 1)))
    z = np.abs(mono) < 1e-6
    sil, i, n = [], 0, len(z)
    edges = np.flatnonzero(np.diff(np.r_[0, z.astype(np.int8), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= 0.05 * SR and a > 0 and b < n:
            sil.append((round(a / SR, 3), round(b / SR, 3)))
    return dict(clicks=out, digital_silence=sil)


def beats(mono):
    import librosa
    y = resample(mono, 22050)
    if len(y) < 22050 * 2:
        return dict(tempo=None, beats=[], onsets=[], strong=[])
    oenv = librosa.onset.onset_strength(y=y, sr=22050)
    tempo, bt = librosa.beat.beat_track(onset_envelope=oenv, sr=22050)
    on = librosa.onset.onset_detect(onset_envelope=oenv, sr=22050, backtrack=False)
    ot = librosa.frames_to_time(on, sr=22050)
    strength = oenv[on] if len(on) else np.array([])
    order = np.argsort(-strength)[:12]
    return dict(tempo=float(np.atleast_1d(tempo)[0]), beats=[float(t) for t in librosa.frames_to_time(bt, sr=22050)],
                onsets=[float(t) for t in ot], strong=sorted((float(ot[i]), float(strength[i])) for i in order))


# ---------------------------------------------------------------- (c) AudioSet タグ
def _ort(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])


def _labels(csv_path, col):
    import csv
    with open(csv_path) as f:
        return [r[col] for r in csv.DictReader(f)]


def windows(dur, win):
    t, out = 0.0, []
    while t < dur - 0.25 or not out:
        out.append((t, min(t + win, dur)))
        t += win
    return out


def tag_yamnet(y16, dur, win, threads):
    p = os.path.join(MODELS, 'yamnet', 'yamnet.onnx')
    if not os.path.exists(p):
        return None
    names = _labels(os.path.join(MODELS, 'yamnet', 'yamnet_class_map.csv'), 'display_name')
    sc = _ort(p, threads).run(None, {'waveform': y16})[0]       # (frames, 521)  0.48 秒ごと、0.96 秒の窓
    ft = np.arange(len(sc)) * 0.48 + 0.48
    out = []
    for a, b in windows(dur, win):
        m = (ft >= a) & (ft < b + 1e-6)
        s = sc[m].mean(axis=0) if m.any() else sc[np.argmin(np.abs(ft - (a + b) / 2))]
        top = np.argsort(-s)[:4]
        out.append(dict(t0=a, t1=b, top=[(names[i], float(s[i])) for i in top]))
    speech = sc[:, names.index('Speech')] if 'Speech' in names else None
    return dict(windows=out, frame_t=ft.tolist(), speech=None if speech is None else speech.tolist(),
                overall=[(names[i], float(sc.mean(axis=0)[i])) for i in np.argsort(-sc.mean(axis=0))[:8]])


def tag_panns(y16, dur, win, threads):
    p = os.path.join(MODELS, 'panns', 'Cnn14_16k.onnx')
    lab = os.path.join(MODELS, 'ced', 'class_labels_indices.csv')       # AudioSet 527 の表（順番は PANNs と同じ）
    if not (os.path.exists(p) and os.path.exists(lab)):
        return None
    names = _labels(lab, 'display_name')
    sess = _ort(p, threads)
    out = []
    for a, b in windows(dur, win):
        seg = y16[int(a * 16000):int(b * 16000)]
        if len(seg) < 16000:                                   # 1秒未満は前に伸ばす
            seg = y16[max(0, int(b * 16000) - 16000):int(b * 16000)]
        s = sess.run(None, {'input_audio': seg[None].astype(np.float32)})[0][0]
        top = np.argsort(-s)[:4]
        out.append(dict(t0=a, t1=b, top=[(names[i], float(s[i])) for i in top]))
    return dict(windows=out)


# ---------------------------------------------------------------- (d) CLAP
def clap_feat(y):
    import librosa
    N = 480000
    if len(y) < N:                                            # HF ClapFeatureExtractor の repeatpad と同じ
        y = np.tile(y, int(N / max(len(y), 1)))
        y = np.pad(y, (0, N - len(y)))
    else:
        y = y[:N]
    S = librosa.feature.melspectrogram(y=y, sr=48000, n_fft=1024, hop_length=480, center=True, pad_mode='reflect',
                                       power=2.0, n_mels=64, fmin=50, fmax=14000, htk=False, norm='slaney')
    return (10 * np.log10(np.maximum(S, 1e-10))).T[None, None].astype(np.float32)


def clap(mono, dur, prompts, win, threads):
    d = os.path.join(MODELS, 'clap')
    pa, pt = os.path.join(d, 'onnx', 'audio_model_quantized.onnx'), os.path.join(d, 'onnx', 'text_model_quantized.onnx')
    if not (os.path.exists(pa) and os.path.exists(pt)):
        return None
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(d, 'tokenizer.json'))
    ts = _ort(pt, threads)
    T = []
    for pr in prompts:
        e = ts.run(None, {'input_ids': np.array([tok.encode(pr).ids], np.int64)})[0][0]
        T.append(e / np.linalg.norm(e))
    T = np.array(T)
    sa = _ort(pa, threads)
    rows = []
    for a, b in windows(dur, win):
        seg = mono[int(a * SR):int(b * SR)]
        if len(seg) < SR:
            seg = mono[max(0, int(b * SR) - SR):int(b * SR)]
        e = sa.run(None, {'input_features': clap_feat(seg)})[0][0]
        rows.append(dict(t0=a, t1=b, sim=(T @ (e / np.linalg.norm(e))).tolist()))
    return dict(prompts=prompts, rows=rows)


# ---------------------------------------------------------------- (e) Whisper と読みの確認
_WHISPER = {}


def whisper(path, model, threads):
    if model == 'none':
        return None
    from faster_whisper import WhisperModel
    if model not in _WHISPER:
        _WHISPER[model] = WhisperModel(model, device='cpu', compute_type='int8', cpu_threads=threads)
    # 想定の文を initial_prompt に入れない（入れると聞こえていない語まで書いてしまうので、確認にならない）
    segs, info = _WHISPER[model].transcribe(path, language='ja', word_timestamps=True, beam_size=5,
                                            vad_filter=False, condition_on_previous_text=False)
    words, text = [], ''
    for s in segs:
        text += s.text
        for w in s.words or []:
            words.append(dict(w=w.word, t0=float(w.start), t1=float(w.end), p=float(w.probability)))
    return dict(text=text.strip(), words=words, model=model)


_D = 'ぜろ いち に さん よん ご ろく なな はち きゅう'.split()


def num2kana(m):
    """数字の読み（だいたい）：26.21→にじゅうろくてんにいいち、16000→いちまんろくせん、2万5千はそのまま漢字に任せる"""
    s = m.group(0).replace(',', '')
    if '.' in s:
        a, b = s.split('.', 1)
        return num2kana(re.match(r'.*', a)) + 'てん' + ''.join({'2': 'にい', '5': 'ごお'}.get(c, _D[int(c)]) for c in b)
    n = int(s)
    if n == 0:
        return 'ぜろ'
    out = ''
    for unit, name in ((10 ** 8, 'おく'), (10 ** 4, 'まん'), (1, '')):
        q, n = divmod(n, unit)
        if q == 0:
            continue
        part = ''
        for u, un, sp in ((1000, 'せん', {3: 'さんぜん', 8: 'はっせん'}), (100, 'ひゃく', {3: 'さんびゃく', 6: 'ろっぴゃく', 8: 'はっぴゃく'}),
                          (10, 'じゅう', {}), (1, '', {})):
            d, q = divmod(q, u)
            if d:
                part += sp.get(d) or ((('' if d == 1 and u > 1 and not (u == 1000 and unit > 1) else _D[d]) + un) if u > 1 else _D[d])
        out += part + name
    return out


_TOK = None


def to_kana(s):
    """文字起こしを読み（ひらがな）に。数字は先に読みへ。形態素解析は SudachiPy があれば使い、無ければ pykakasi"""
    global _TOK
    s = re.sub(r'\d[\d,]*(?:\.\d+)?', num2kana, s)
    try:
        from sudachipy import dictionary, tokenizer
        if _TOK is None:
            _TOK = (dictionary.Dictionary().create(), tokenizer.Tokenizer.SplitMode.C)
        return ''.join(m.reading_form() if re.search(r'[\u4e00-\u9fff\u30a0-\u30ff]', m.surface()) else m.surface()
                       for m in _TOK[0].tokenize(s, _TOK[1]))
    except ImportError:
        pass
    try:
        import pykakasi
    except ImportError:
        return None
    return ''.join(x['hira'] for x in pykakasi.kakasi().convert(s))


def norm_kana(s):
    """読みの比べ方をそろえる：カタカナ→ひらがな、記号を消す、長音・助詞の書き方の揺れを寄せる"""
    s = ''.join(chr(ord(c) - 0x60) if 'ァ' <= c <= 'ヶ' else c for c in s)
    s = re.sub(r'[^\u3041-\u309f\u30fc0-9]', '', s)
    out = []
    vowel = {}
    for row, v in (('あかさたなはまやらわがざだばぱぁゃ', 'あ'), ('いきしちにひみりぎじぢびぴぃ', 'い'),
                   ('うくすつぬふむゆるぐずづぶぷぅゅっ', 'う'), ('えけせてねへめれげぜでべぺぇ', 'え'),
                   ('おこそとのほもよろをごぞどぼぽぉょ', 'お')):
        for c in row:
            vowel[c] = v
    for c in s:
        if c == 'ー' and out:
            c = vowel.get(out[-1], '')
        out.append(c)
    s = ''.join(out)
    s = re.sub(r'(?<=[おこそとのほもよろごぞどぼぽょ])う', 'お', s)     # ちょう→ちょお
    s = re.sub(r'(?<=[えけせてねへめれげぜでべぺ])い', 'え', s)         # めい→めえ
    s = s.replace('ぢ', 'じ').replace('づ', 'ず').replace('を', 'お')
    return s


def check_reading(tr, expect_kana, must):
    r = {}
    if expect_kana is not None and tr:
        heard = to_kana(tr['text'])
        if heard is not None:
            a, b = norm_kana(expect_kana), norm_kana(heard)
            # 助詞「は」（わ）の書き方の差は無視して比べる
            b2 = b.replace('は', 'わ')
            a2 = a.replace('は', 'わ')
            sm = difflib.SequenceMatcher(None, a2, b2)
            diffs = [(a2[i1:i2], b2[j1:j2]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != 'equal']
            r.update(expect=a2, heard=b2, ratio=round(sm.ratio(), 3), diffs=diffs,
                     digits=bool(re.search(r'\d', heard)))
    if must and tr:
        hk = norm_kana(to_kana(tr['text']) or '').replace('は', 'わ')
        r['must'] = [(alts, any(x in tr['text'] for x in alts) or
                      any(len(norm_kana(to_kana(x) or '')) >= 3 and norm_kana(to_kana(x) or '').replace('は', 'わ') in hk for x in alts))
                     for alts in must]
    return r


# ---------------------------------------------------------------- (b) 画像
def plot(path_png, x, res, t_range=None, title=''):
    import librosa
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fp in ('/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf', '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf'):
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
            break
    dur = len(x) / SR
    a, b = t_range or (0, dur)
    xs = x[int(a * SR):int(b * SR)]
    mono = xs.mean(axis=1)
    t = a + np.arange(len(mono)) / SR
    bt = [v for v in res['beats']['beats'] if a <= v <= b]
    on = [v for v in res['beats']['onsets'] if a <= v <= b]
    fig, ax = plt.subplots(4, 1, figsize=(18, 10), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 2.2, 1.1, 0.7]))
    step = max(1, len(mono) // 20000)
    for c in range(xs.shape[1]):
        ax[0].plot(t[::step], xs[::step, c], lw=0.4, alpha=0.8, label=('L', 'R')[c] if xs.shape[1] > 1 else 'mono')
    ax[0].set_ylim(-1, 1)
    ax[0].axhline(0.891, color='r', lw=0.5, ls=':')     # -1 dBFS
    ax[0].axhline(-0.891, color='r', lw=0.5, ls=':')
    ax[0].set_ylabel('波形')
    tr = res.get('whisper')
    if tr:
        for w in tr['words']:
            if a <= w['t0'] <= b:
                ax[0].text(w['t0'], 0.82, w['w'], fontsize=8, va='top', color='navy', rotation=0)
    S = librosa.feature.melspectrogram(y=mono, sr=SR, n_fft=2048, hop_length=512, n_mels=128, fmax=16000)
    ax[1].imshow(librosa.power_to_db(S, ref=np.max, top_db=80), origin='lower', aspect='auto', cmap='magma',
                 extent=[a, a + S.shape[1] * 512 / SR, 0, 128])
    ticks = [100, 300, 1000, 3000, 8000, 16000]
    mel_t = librosa.hz_to_mel(ticks) / librosa.hz_to_mel(16000) * 128
    ax[1].set_yticks(mel_t)
    ax[1].set_yticklabels(['100', '300', '1k', '3k', '8k', '16k'])
    ax[1].set_ylabel('メル（Hz）')
    for v in bt:
        for k in (0, 1):
            ax[k].axvline(v, color='c', lw=0.6, alpha=0.6)
    for v in on:
        ax[1].plot([v], [124], 'wv', ms=4)
    L = res['loud']
    ax[2].plot(L['short_t'], L['short'], color='k', lw=1.2, label='短期 LUFS（3秒）')
    ax[2].axhline(-14, color='g', ls='--', lw=0.8, label='-14 LUFS')
    if L['ffmpeg']['I'] is not None:
        ax[2].axhline(L['ffmpeg']['I'], color='orange', ls=':', lw=0.8, label='統合 %.1f' % L['ffmpeg']['I'])
    if res.get('speech_spans'):
        for s0, s1 in res['speech_spans']:
            ax[2].axvspan(s0, s1, color='y', alpha=0.25)
    ax[2].set_ylim(-45, -5)
    ax[2].set_ylabel('LUFS')
    ax[2].legend(loc='lower left', fontsize=8, ncol=3)
    ax[3].set_ylim(0, 1)
    ax[3].set_yticks([])
    ax[3].set_ylabel('タグ')
    yam = res.get('yamnet')
    if yam:
        for w in yam['windows']:
            if w['t1'] > a and w['t0'] < b:
                ax[3].text(max(w['t0'], a) + 0.05, 0.5, '\n'.join('%s %.2f' % (n[:16], s) for n, s in w['top'][:2]),
                           fontsize=7, va='center')
                ax[3].axvline(w['t0'], color='gray', lw=0.4)
    ax[3].set_xlim(a, b)
    ax[3].set_xlabel('秒')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path_png, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- まとめ
def speech_spans_from(res, dur):
    tr = res.get('whisper')
    spans = []
    if tr and tr['words']:
        for w in tr['words']:
            if spans and w['t0'] - spans[-1][1] < 0.35:
                spans[-1][1] = w['t1']
            else:
                spans.append([w['t0'], w['t1']])
    elif res.get('yamnet') and res['yamnet']['speech']:
        for ft, p in zip(res['yamnet']['frame_t'], res['yamnet']['speech']):
            if p > 0.5:
                a = max(0, ft - 0.48)
                if spans and a - spans[-1][1] < 0.1:
                    spans[-1][1] = ft
                else:
                    spans.append([a, ft])
    return [(round(a, 2), round(min(b, dur), 2)) for a, b in spans]


def flags(res, is_mix):
    out = []
    L, F = res['loud'], res['loud']['ffmpeg']
    if F['I'] is not None:
        if is_mix and abs(F['I'] + 14) > 2:
            out.append('統合 %.1f LUFS：Reels でよく言われる目安（-14 前後。Instagram は公式の数値を出していない）から %.1f dB 離れている。%s'
                       % (F['I'], F['I'] + 14, '小さいと他の投稿より静かに聞こえる' if F['I'] < -14 else '大きいと配信側で下げられる'))
    if F['TP'] is not None and F['TP'] > -1.0:
        out.append('トゥルーピーク %.1f dBTP：-1 dBTP を超える。AAC にしたとき歪む恐れ' % F['TP'])
    if L['clips']:
        out.append('クリップ（|x|≥0.999）%d サンプル' % L['clips'])
    if F['LRA'] is not None and is_mix and F['LRA'] > 12:
        out.append('LRA %.1f LU：大小の差が大きい。スマホのスピーカーでは小さい所が聞こえにくい' % F['LRA'])
    if L['head_10ms_peak'] > -30:
        out.append('頭の 10ms が %.0f dBFS（いきなり音が鳴る）：再生直後にプチッと鳴る恐れ（フェードインを数ms）' % L['head_10ms_peak'])
    if L['tail_50ms_rms'] > -45:
        out.append('最後の 50ms が %.0f dBFS：音が鳴ったまま切れている（ブツ切れ）。尻にフェードを' % L['tail_50ms_rms'])
    C = res.get('clicks') or {}
    if C.get('clicks'):
        out.append('プチ音の候補 %d か所：%s 秒（エンベロープの急な切れ・素材の継ぎ目・ゲートの開閉を疑う）'
                   % (len(C['clicks']), '、'.join('%.2f' % a for a, _ in C['clicks'][:12])))
    if C.get('digital_silence'):
        out.append('途中の完全な無音（デジタル 0）：' + '、'.join('%.2f–%.2f' % ab for ab in C['digital_silence']) +
                   ' 秒。意図した「間」でも、出入りをフェードしないとプチ音になる')
    if abs(L['dc']) > 0.005:
        out.append('DC オフセット %.4f' % L['dc'])
    if L['corr'] is not None and L['corr'] < 0.2:
        out.append('左右の相関 %.2f：モノラル再生（スマホ）で音が痩せる・消える恐れ' % L['corr'])
    B = res['spec']['bands']
    if B['sub<40'] > 15:
        out.append('40Hz 未満が全体の %.0f%%：スマホでは聞こえず、ラウドネスだけ食う。ハイパスを' % B['sub<40'])
    if B['pres2k-5k'] > 25:
        out.append('2〜5kHz が %.0f%%：耳に刺さる帯域が多い' % B['pres2k-5k'])
    if res.get('speech_vs_gap') is not None and is_mix and res.get('whisper'):   # YAMNet だけの区間は粗いので判定しない
        d = res['speech_vs_gap']
        if d < 4:
            out.append('声がある所と無い所の差が %.1f dB しかない：BGM が声にかぶっている可能性（ダッキングを深く）' % d)
    return out


def analyze(path, args, is_mix):
    t0 = time.time()
    x = decode(path)
    if x.shape[1] == 1:
        x2 = np.repeat(x, 2, axis=1)
    else:
        x2 = x
    dur = len(x) / SR
    mono = x.mean(axis=1).astype(np.float32)
    res = dict(path=path, dur=dur, ch=x.shape[1])
    res['loud'] = loudness(x, path)      # モノラルはモノラルのまま測る（ffmpeg ebur128 と同じ。2ch に複製すると +3dB になる）
    res['spec'] = spectral(mono)
    res['beats'] = beats(mono)
    res['clicks'] = clicks(mono)
    y16 = resample(mono, 16000)
    if not args.no_tags:
        res['yamnet'] = tag_yamnet(y16, dur, args.win, args.threads)
        res['panns'] = tag_panns(y16, dur, args.win, args.threads) if not args.no_panns else None
    if not args.no_clap:
        res['clap'] = clap(mono, dur, args.prompts, args.clap_win, args.threads)
    res['whisper'] = whisper(path, args.whisper, args.threads)
    res['speech_spans'] = speech_spans_from(res, dur)
    if res['speech_spans']:
        gaps, prev = [], 0.0
        for a, b in res['speech_spans']:
            if a - prev > 0.3:
                gaps.append((prev + 0.1, a - 0.1))
            prev = b
        if dur - prev > 0.3:
            gaps.append((prev + 0.1, dur))
        sp, gp = region_lufs(x, res['speech_spans']), region_lufs(x, gaps)
        res['speech_lufs'], res['gap_lufs'] = sp, gp
        res['speech_vs_gap'] = (sp - gp) if (sp is not None and gp is not None) else None
    res['elapsed'] = time.time() - t0
    return res, x2


# ---------------------------------------------------------------- 書き出し
def fmt(v, f='%.1f'):
    return '—' if v is None else (f % v)


def md_one(res, rel_png, expect, is_mix):
    L, F = res['loud'], res['loud']['ffmpeg']
    o = []
    o.append('| 項目 | 値 |\n|---|---|')
    o.append('| 長さ / ch | %.2f 秒 / %d |' % (res['dur'], res['ch']))
    o.append('| 統合ラウドネス（ffmpeg ebur128） | **%s LUFS**（pyloudnorm %s） |' % (fmt(F['I']), fmt(L['pyln_I'])))
    o.append('| LRA | %s LU（%s〜%s） |' % (fmt(F['LRA']), fmt(F['LRA_low']), fmt(F['LRA_high'])))
    o.append('| トゥルーピーク / サンプルピーク | %s dBTP / %.1f dBFS |' % (fmt(F['TP']), L['sample_peak']))
    o.append('| 短期 LUFS の最大 / 最小（3秒窓） | %.1f / %.1f |' % (max(L['short']), min(L['short'])))
    o.append('| クリップ / DC / 左右の相関 | %d / %.4f / %s |' % (L['clips'], L['dc'], fmt(L['corr'], '%.2f')))
    o.append('| 頭10msのピーク / 尻50msのRMS | %.1f / %.1f dBFS |' % (L['head_10ms_peak'], L['tail_50ms_rms']))
    o.append('| 無音（-60dBFS 未満）の割合 | %.0f%% |' % (L['silence_ratio'] * 100))
    if res.get('speech_lufs') is not None:
        o.append('| 声がある所 / 無い所の LUFS（%s） | %s / %s（差 %s dB） |' % ('Whisper の語の区間' if res.get('whisper') else 'YAMNet Speech>0.5、粗い', fmt(res['speech_lufs']), fmt(res.get('gap_lufs')),
                                                                     fmt(res.get('speech_vs_gap'))))
    B = res['spec']['bands']
    o.append('| 帯域の割合 | ' + '、'.join('%s %.1f%%' % (k, v) for k, v in B.items()) + ' |')
    o.append('| スペクトル重心（中央値）/ 平坦度 | %.0f Hz / %.3f |' % (res['spec']['centroid'], res['spec']['flatness']))
    bt = res['beats']
    o.append('| テンポ（librosa） | %s BPM、拍 %d 個、オンセット %d 個 |' % (fmt(bt['tempo']), len(bt['beats']), len(bt['onsets'])))
    o.append('')
    fl = flags(res, is_mix)
    o.append('**自動の気づき**：' + ('なし' if not fl else ''))
    for f in fl:
        o.append('- ' + f)
    o.append('')
    if rel_png:
        o.append('![overview](%s)\n' % rel_png)
    if is_mix and len(L['short']) > 1:
        o.append('**短期 LUFS の推移（1秒ごと、窓の中心の秒）**\n')
        o.append(' '.join('%.1fs:%.1f' % (t, v) for t, v in zip(L['short_t'][::2], L['short'][::2])))
        o.append('')
    if bt['beats']:
        o.append('**拍（秒）**：' + ' '.join('%.2f' % v for v in bt['beats'][:80]))
        o.append('\n**強いオンセット上位（秒, 強さ）**：' + ' '.join('%.2f(%.1f)' % s for s in bt['strong']))
        o.append('')
    for key, title in (('yamnet', 'YAMNet（AudioSet 521）'), ('panns', 'PANNs Cnn14（AudioSet 527）')):
        T = res.get(key)
        if not T:
            continue
        o.append('**%s・窓ごとの上位**\n\n| 秒 | 1位 | 2位 | 3位 |\n|---|---|---|---|' % title)
        for w in T['windows']:
            o.append('| %.1f–%.1f | ' % (w['t0'], w['t1']) + ' | '.join('%s %.2f' % (n, s) for n, s in w['top'][:3]) + ' |')
        o.append('')
    C = res.get('clap')
    if C:
        o.append('**CLAP（説明文との近さ。-1〜1、0.25 以上ならかなり当てはまる、0 前後は無関係）**\n')
        o.append('| 秒 | ' + ' | '.join(C['prompts']) + ' |')
        o.append('|---|' + '---|' * len(C['prompts']))
        for r in C['rows']:
            best = int(np.argmax(r['sim']))
            o.append('| %.0f–%.0f | ' % (r['t0'], r['t1']) +
                     ' | '.join(('**%.2f**' if i == best else '%.2f') % v for i, v in enumerate(r['sim'])) + ' |')
        o.append('')
    tr = res.get('whisper')
    if tr:
        o.append('**Whisper（%s）**：「%s」\n' % (tr['model'], tr['text']))
        o.append('語ごと（開始秒・確からしさ）：' + ' '.join('%s[%.2f,%.2f]' % (w['w'].strip(), w['t0'], w['p']) for w in tr['words']))
        o.append('')
        if expect:
            ck = check_reading(tr, expect.get('kana'), expect.get('must'))
            if 'ratio' in ck:
                o.append('- 想定の読み：`%s`\n- 聞こえた読み（Whisper の文字→かな。SudachiPy／pykakasi の読みなので、漢字の読み違いはこちらの変換の誤りのことがある）：`%s`\n- 一致率 %.2f%s' %
                         (ck['expect'], ck['heard'], ck['ratio'], '（数字は読みに直していないので、数字の行は一致率が下がる）' if ck['digits'] else ''))
                if ck['diffs']:
                    o.append('- 違い（想定→聞こえた）：' + '、'.join('「%s」→「%s」' % d for d in ck['diffs']))
            for alts, ok in ck.get('must', []):
                o.append('- %s キーワード %s' % ('OK' if ok else '**NG**', ' / '.join(alts)))
            o.append('')
    o.append('<sub>処理 %.1f 秒</sub>\n' % res['elapsed'])
    return '\n'.join(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--name')
    ap.add_argument('--out', default=os.path.join(ROOT, 'exports', 'v3', 'listen'))
    ap.add_argument('--whisper', default='large-v3-turbo', help='large-v3-turbo / small / medium / none')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--win', type=float, default=2.0, help='タグの窓（秒）')
    ap.add_argument('--clap-win', type=float, default=5.0)
    ap.add_argument('--prompts', help='CLAP の説明文（1行1つのテキストファイル）')
    ap.add_argument('--expect', default=os.path.join(HERE, 'listen_expect.json'))
    ap.add_argument('--no-tags', action='store_true')
    ap.add_argument('--no-panns', action='store_true')
    ap.add_argument('--no-clap', action='store_true')
    ap.add_argument('--zoom', type=float, default=10.0, help='拡大画像の幅（秒）。0 で作らない')
    args = ap.parse_args()
    args.prompts = [l.strip() for l in open(args.prompts) if l.strip()] if args.prompts else DEFAULT_PROMPTS
    try:
        os.nice(10)
    except OSError:
        pass

    files, folder = [], None
    for p in args.inputs:
        if os.path.isdir(p):
            folder = folder or p
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(AUDIO_EXT))
        else:
            files.append(p)
    name = args.name or os.path.basename(os.path.normpath(folder or os.path.splitext(files[0])[0]))
    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)
    expect_all = json.load(open(args.expect)) if os.path.exists(args.expect) else {}
    is_batch = len(files) > 1

    md = ['# 聞き取りレポート：%s\n' % name,
          '作成 %s ／ `tools/v3/listen.py`（ffmpeg ebur128・pyloudnorm・librosa・YAMNet/PANNs/CLAP の ONNX・faster-whisper）\n'
          % time.strftime('%Y-%m-%d %H:%M')]
    summary = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        is_mix = not is_batch
        print('…', f, file=sys.stderr)
        res, x = analyze(f, args, is_mix)
        exp = dict(expect_all.get(stem, {}))
        mj = os.path.join(os.path.dirname(f), 'moras.json')
        if os.path.exists(mj) and stem in json.load(open(mj)) and 'kana' not in exp:
            exp['kana'] = ''.join(m[0] for m in json.load(open(mj))[stem])
        png = '%s_overview.png' % stem if is_batch else 'overview.png'
        plot(os.path.join(out, png), x, res, title='%s  %.1fs  %s LUFS' % (os.path.basename(f), res['dur'], fmt(res['loud']['ffmpeg']['I'])))
        md.append('## %s\n' % os.path.relpath(f, ROOT))
        md.append(md_one(res, png, exp, is_mix))
        if args.zoom and res['dur'] > args.zoom * 1.5 and not is_batch:
            zl = []
            for i, a in enumerate(np.arange(0, res['dur'], args.zoom)):
                zp = 'zoom_%02d_%03.0f-%03.0fs.png' % (i, a, min(a + args.zoom, res['dur']))
                plot(os.path.join(out, zp), x, res, (a, min(a + args.zoom, res['dur'])), title='%s  %.0f–%.0fs' % (name, a, min(a + args.zoom, res['dur'])))
                zl.append('[%s](%s)' % (zp, zp))
            md.append('拡大：' + '、'.join(zl) + '\n')
        F = res['loud']['ffmpeg']
        ck = check_reading(res['whisper'], exp.get('kana'), exp.get('must')) if res.get('whisper') and exp else {}
        summary.append((stem, res['dur'], F['I'], F['TP'], F['LRA'], (res.get('whisper') or {}).get('text', ''),
                        ck.get('ratio'), [a for a, ok in ck.get('must', []) if not ok]))
        json.dump({k: v for k, v in res.items() if k != 'path'}, open(os.path.join(out, stem + '.json'), 'w'),
                  ensure_ascii=False, default=float)
    if is_batch:
        tbl = ['## 一覧\n', '| ファイル | 秒 | LUFS | dBTP | LRA | Whisper | 読み一致率 | NG キーワード |', '|---|---|---|---|---|---|---|---|']
        for s in summary:
            tbl.append('| %s | %.2f | %s | %s | %s | %s | %s | %s |' % (s[0], s[1], fmt(s[2]), fmt(s[3]), fmt(s[4]), s[5],
                                                                fmt(s[6], '%.2f'), ' '.join('/'.join(a) for a in s[7]) or '—'))
        md.insert(2, '\n'.join(tbl) + '\n')
    open(os.path.join(out, 'report.md'), 'w').write('\n'.join(md))
    print(os.path.join(out, 'report.md'))


if __name__ == '__main__':
    main()
