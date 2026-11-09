"""ナレーションの「抑揚」を数値で測る（2026-11-08 クライアント「棒読みみたい」への対応で作った）。

  python3 tools/v3/cutC/vo_metrics.py exports/v3/vo_C [exports/v3/vo_C/_flat_backup ...]

1行（wav）ごとに：
  f0_range  … 声の高さの幅（半音）。有声フレームの F0（librosa.pyin）の 95 パーセンタイル − 5 パーセンタイル
  f0_sd     … 声の高さのばらつき（半音の標準偏差）
  f0_move   … 高さの動き（有声の連続区間で、10 ms ごとの |ΔF0| の平均 × 100 ＝ 半音/秒）
  en_range  … 声の大きさの幅（dB）。10 ms ごとの RMS を声の区間（最大 −35 dB 以内）で 95−5 パーセンタイル
  en_sd     … 声の大きさのばらつき（dB の標準偏差）
  f0_med    … 声の高さの中央値（Hz、声の続き具合の確認用）
  creak     … きしみ声（F0 < 80 Hz）の割合。高さの数値からは除いてある
  f0_jumps  … 10 ms で 4 半音より大きい高さの跳びの数（声の裏返り・合成の乱れの目安。0〜1 が普通）
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import listen  # noqa: E402

SR16 = 16000
CREAK = 80.0       # これより低い F0 はきしみ声（この話者の普通の声は 95 Hz 以上。pyin は 70 Hz の床に張りつく）
HOP = 160


def load16(path):
    import librosa
    x = listen.decode(path)
    x = (x[:, 0] if x.ndim == 2 else x).astype(np.float64)
    return librosa.resample(x, orig_sr=48000, target_sr=SR16)


def octave_fix(f, win=7, lim=8.0):
    """pyin のオクターブ誤り（半分・倍の高さに跳ぶ）を直す：となりの有声 ±win フレームの中央値から lim 半音より
    離れたフレームを 12 半音単位で戻す"""
    if len(f) < 3:
        return f
    st = 12 * np.log2(f)
    out = st.copy()
    for i in range(len(st)):
        loc = np.median(st[max(0, i - win):i + win + 1])
        d = st[i] - loc
        if abs(d) > lim:
            out[i] = st[i] - 12 * round(d / 12)
    return 2 ** (out / 12)


def liveliness(path):
    import librosa
    y = load16(path)
    f0, vf, _ = librosa.pyin(y, fmin=70, fmax=420, sr=SR16, frame_length=1024, hop_length=HOP)
    ok = vf & ~np.isnan(f0)
    f0 = f0.copy()
    f0[ok] = octave_fix(f0[ok])
    creak = ok & (f0 < CREAK)
    ok &= ~creak                              # きしみ声（vocal fry）は高さの動きに数えない
    f = f0[ok]
    out = {'creak': round(float(creak.sum()) / max(1, int((vf & ~np.isnan(f0)).sum() + creak.sum())), 3)}
    if len(f) > 5:
        st = 12 * np.log2(f / np.median(f))
        out['f0_range'] = float(np.percentile(st, 95) - np.percentile(st, 5))
        out['f0_sd'] = float(np.std(st))
        out['f0_med'] = float(np.median(f))
        s_all = np.where(ok, 12 * np.log2(np.where(ok, f0, 1) / np.median(f)), np.nan)
        d = np.abs(np.diff(s_all))
        out['f0_jumps'] = int(np.sum(d[~np.isnan(d)] > 4.0))   # 10 ms で 4 半音より大きい跳び（声の裏返り・推定の乱れ）
        d = d[~np.isnan(d) & (d < 2.0)]       # 有声どうしのとなり（オクターブ誤りの跳びは除く）
        out['f0_move'] = float(d.mean() * 100) if len(d) else 0.0
    n = len(y) // HOP
    r = 20 * np.log10(np.sqrt(np.mean(y[:n * HOP].reshape(n, HOP) ** 2, 1)) + 1e-9)
    v = r[r > r.max() - 35]
    out['en_range'] = float(np.percentile(v, 95) - np.percentile(v, 5))
    out['en_sd'] = float(np.std(v))
    out['dur'] = round(len(y) / SR16, 3)
    return {k: round(v, 2) if isinstance(v, float) else v for k, v in out.items()}


KEYS = ('f0_range', 'f0_sd', 'f0_move', 'en_range', 'en_sd')


def table(dirs):
    res = {}
    for d in dirs:
        res[d] = {os.path.basename(p)[:-4]: liveliness(p) for p in sorted(glob.glob(os.path.join(d, '[0-9][0-9].wav')))}
    return res


def main():
    dirs = sys.argv[1:]
    res = table(dirs)
    for d, r in res.items():
        print(d)
        for k, m in r.items():
            print(' ', k, ' '.join(f'{kk} {m.get(kk, 0):5.2f}' for kk in KEYS + ('f0_med', 'dur')))
        mean = {kk: np.mean([m.get(kk, 0) for m in r.values()]) for kk in KEYS}
        print('  mean', ' '.join(f'{kk} {v:5.2f}' for kk, v in mean.items()))
    if os.environ.get('JSON_OUT'):
        json.dump(res, open(os.environ['JSON_OUT'], 'w'), indent=1)


if __name__ == '__main__':
    main()
