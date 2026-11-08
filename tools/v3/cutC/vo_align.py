"""ナレーションのモーラ時刻を実測する（強制アラインメント）→ <vo_dir>/moras.json

  <venv>/bin/python -I tools/v3/cutC/vo_align.py fit                 # 子音ごとのずれを VOICEVOX 版（vo_B）で測る → align_offsets.json
  <venv>/bin/python -I tools/v3/cutC/vo_align.py run exports/v3/vo_C  # reading.json のモーラ列を wav に合わせる → moras.json

案B（vo_B_aida/moras.json）で使ったやり方を道具にしたもの：
  1) 日本語ひらがなの wav2vec2（vumichien/wav2vec2-large-xlsr-japanese-hiragana、tools/models/w2v_ja に置く。
     Hugging Face から config.json・vocab.json・preprocessor_config.json・model.safetensors を取る）で CTC の対数確率を出し、
     モーラ列（ひらがな）を Viterbi で強制アラインメント。長音・助詞の書き方（ワ/は、オ/う/ー など）は得点の高い方を選ぶ。
  2) この方法は子音で少し遅く出る（サ・カ・キなど）。正解の時刻が分かっている VOICEVOX 版（exports/v3/vo_B：
     moras.json はエンジンが返した長さ）で子音の種類ごとの中央値を測り（fit）、それを引く。
  3) 行の頭と、間（無音 0.05 秒以上）の直後のモーラは、声が実際に立ち上がる所に寄せる（無音のしきい値は阿井田のハムより上）。
  VOICEVOX 版での誤差：だいたい 20 ms 以内、9割が 1 フレーム以内（案B のときの実測）。

必要なもの：torch（CPU）と transformers、soundfile、scipy。システムの python には torch が無いので venv で動かす：
  python3 -m venv --system-site-packages venv && venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
  venv/bin/pip install transformers safetensors
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
MODEL = os.environ.get('MAZE_W2V', os.path.join(ROOT, 'tools', 'models', 'w2v_ja'))
OFFSETS = os.path.join(HERE, 'align_offsets.json')
STRIDE = 320 / 16000
HOP = 0.005
PAUSE, LOOK = 0.12, 0.08
K2H = {chr(c): chr(c - 0x60) for c in range(0x30A1, 0x30F7)}


# ---------------------------------------------------------------- モーラの子音の種類
def cls(m, prev=''):
    c = m[0]
    if c in 'アイウエオ':
        return 'V' if prev not in ('', 'ッ') else 'V0'
    if c == 'ン':
        return 'N'
    if c == 'ッ':
        return 'Q'
    if c in 'サシスセソ':
        return 'S'
    if c in 'ハヒヘホフ':
        return 'H'
    if c in 'ツチ':
        return 'TS'
    if c in 'カキクケコタテトパピプペポ':
        return 'K' if prev != 'ッ' else 'KQ'
    if c in 'ナニヌネノマミムメモ':
        return 'NAS'
    if c in 'ラリルレロ':
        return 'R'
    if c in 'ヤユヨワ':
        return 'GL'
    if c in 'ガギグゲゴダデドバビブベボ':
        return 'VS'
    if c in 'ザジズゼゾ':
        return 'Z'
    return '?'


# ---------------------------------------------------------------- CTC 強制アラインメント
def hira(m):
    return ''.join(K2H.get(c, c) for c in m)


def variants(seq, i):
    """書いた字と実際の音が違いうるもの（助詞・長音）の候補"""
    m = seq[i]
    prev = seq[i - 1] if i else ''
    v = [hira(m)]
    if m == 'ワ':
        v.append('は')
    if m == 'エ' and i == len(seq) - 1:
        v.append('へ')
    if m in ('オ', 'エ', 'ア', 'ウ', 'イ') and i > 0 and prev not in ('ッ', 'ン'):
        v.append('ー')
        if m == 'オ':
            v.append('う')
        if m == 'エ':
            v.append('い')
    return v


def load16(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(1)
    g = np.gcd(sr, 16000)
    return resample_poly(x, 16000 // g, sr // g).astype(np.float32)


_M = {}


def logprobs(y):
    import torch
    from transformers import Wav2Vec2ForCTC
    if 'm' not in _M:
        torch.set_num_threads(int(os.environ.get('ALIGN_THREADS', '3')))
        _M['m'] = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()
        _M['vocab'] = json.load(open(os.path.join(MODEL, 'vocab.json')))
    yy = (y - y.mean()) / (y.std() + 1e-7)
    pad = np.zeros(int(0.3 * 16000), np.float32)
    yy = np.concatenate([pad, yy, pad])
    with torch.no_grad():
        lg = _M['m'](torch.from_numpy(yy)[None]).logits[0]
    return torch.log_softmax(lg, -1).numpy(), int(round(0.3 / STRIDE))


def viterbi(lp, tokens, blank):
    T = lp.shape[0]
    ext = [blank]
    for t in tokens:
        ext += [t, blank]
    S = len(ext)
    dp = np.full((T, S), -1e18)
    bp = np.zeros((T, S), np.int16)
    dp[0, 0] = lp[0, ext[0]]
    dp[0, 1] = lp[0, ext[1]]
    for t in range(1, T):
        prev = dp[t - 1]
        c0 = prev
        c1 = np.concatenate([[-1e18], prev[:-1]])
        c2 = np.concatenate([[-1e18, -1e18], prev[:-2]])
        ok2 = np.array([s >= 2 and ext[s] != blank and ext[s] != ext[s - 2] for s in range(S)])
        c2 = np.where(ok2, c2, -1e18)
        st = np.stack([c0, c1, c2])
        arg = st.argmax(0)
        dp[t] = st[arg, np.arange(S)] + lp[t, ext]
        bp[t] = arg
    s = S - 1 if dp[T - 1, S - 1] > dp[T - 1, S - 2] else S - 2
    total = dp[T - 1, s]
    path = [0] * T
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= bp[t, s]
    return total, path


def align(lp, mora_tok):
    vocab = _M['vocab']
    toks, owner = [], []
    for i, tt in enumerate(mora_tok):
        for c in tt:
            toks.append(vocab[c])
            owner.append(i)
    total, path = viterbi(lp, toks, vocab['[PAD]'])
    n = len(mora_tok)
    first, last = [None] * n, [None] * n
    for t, s in enumerate(path):
        if s % 2 == 1:
            i = owner[(s - 1) // 2]
            if first[i] is None:
                first[i] = t
            last[i] = t
    return total, first, last


def ctc(wav, seq):
    lp, n0 = logprobs(load16(wav))
    cur = [variants(seq, i)[0] for i in range(len(seq))]
    best, _, _ = align(lp, cur)
    for i in range(len(seq)):
        for v in variants(seq, i)[1:]:
            trial = cur[:]
            trial[i] = v
            sc, _, _ = align(lp, trial)
            if sc > best + 0.5:
                best, cur = sc, trial
    sc, first, last = align(lp, cur)
    return [[m, round((first[i] - n0) * STRIDE, 3), round((last[i] + 1 - n0) * STRIDE, 3), cur[i]]
            for i, m in enumerate(seq)], float(sc)


# ---------------------------------------------------------------- 無音と仕上げ
def env(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype='float32', always_2d=True)
    x = x.mean(1)
    h, w = int(sr * HOP), int(sr * 0.02)
    xp = np.pad(x, (w // 2, w // 2))
    n = len(x) // h
    e = np.array([np.sqrt(np.mean(xp[i * h:i * h + w] ** 2) + 1e-12) for i in range(n)])
    return 20 * np.log10(e)


def silent_runs(e, rel=-42, minlen=0.05):
    thr = max(e.max() + rel, -55, np.percentile(e, 5) + 9)
    s = e < thr
    runs, i = [], 0
    while i < len(s):
        if s[i]:
            j = i
            while j < len(s) and s[j]:
                j += 1
            if (j - i) * HOP >= minlen:
                runs.append((round(i * HOP, 3), round(j * HOP, 3)))
            i = j
        else:
            i += 1
    return runs, thr


def assemble(wav, al, off):
    e = env(wav)
    runs, thr = silent_runs(e)
    sp = np.where(e >= thr)[0]
    onset0 = sp[0] * HOP if len(sp) else 0.0
    seq = [m[0] for m in al]
    out = []
    for i, (m, a, b, _) in enumerate(al):
        t = a - off.get(cls(m, seq[i - 1] if i else ''), 0.0)
        if i == 0:
            if 0 <= t - onset0 <= 0.2 or t < onset0:
                t = onset0
        elif m == 'ッ':
            for s0, s1 in runs:
                if abs(s0 - t) <= 0.06 and s1 > t:
                    t = s0
        else:
            for s0, s1 in runs:
                if out[-1] + 0.05 < s1 <= b and (t < s1 or (s1 - s0 >= PAUSE and t - s1 <= LOOK)):
                    t = s1
        if i:
            t = max(t, out[-1] + 0.03)
        out.append(round(t, 3))
    return out


def fit():
    vo = os.path.join(ROOT, 'exports', 'v3', 'vo_B')
    gt = json.load(open(os.path.join(vo, 'moras.json')))
    R = {}
    raw = {}
    for k in sorted(gt):
        seq = [m for m, _ in gt[k]]
        al, _ = ctc(os.path.join(vo, k + '.wav'), seq)
        raw[k] = al
        for i, ((m, g), a) in enumerate(zip(gt[k], al)):
            R.setdefault(cls(m, seq[i - 1] if i else ''), []).append(a[1] - g)
    off = {c: float(np.median(v)) for c, v in sorted(R.items())}
    json.dump(off, open(OFFSETS, 'w'), indent=1)
    res = []
    for k in sorted(gt):
        r = assemble(os.path.join(vo, k + '.wav'), raw[k], off)
        res += [a - g for a, (m, g) in zip(r, gt[k])]
    res = np.abs(np.array(res))
    print('offsets', {c: round(v * 1000) for c, v in off.items()})
    print('vo_B check: median %.0f ms, p90 %.0f ms, max %.0f ms, >33 ms %d/%d' % (
        np.median(res) * 1000, np.percentile(res, 90) * 1000, res.max() * 1000, (res > 0.033).sum(), len(res)))


def run(vo):
    off = json.load(open(OFFSETS))
    rd = json.load(open(os.path.join(vo, 'reading.json')))
    out, raw = {}, {}
    for k in sorted(rd):
        seq = rd[k]['moras']
        wav = os.path.join(vo, k + '.wav')
        al, sc = ctc(wav, seq)
        ts = assemble(wav, al, off)
        out[k] = [[m, t] for m, t in zip(seq, ts)]
        raw[k] = dict(score=round(sc, 2), moras=al)
        print(k, '%.1f' % sc, ' '.join('%s%.2f' % (m, t) for m, t in out[k]), flush=True)
    json.dump(out, open(os.path.join(vo, 'moras.json'), 'w'), ensure_ascii=False)
    json.dump(raw, open(os.path.join(vo, 'moras_ctc_raw.json'), 'w'), ensure_ascii=False)


if __name__ == '__main__':
    if sys.argv[1] == 'fit':
        fit()
    else:
        run(sys.argv[2])
