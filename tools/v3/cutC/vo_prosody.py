"""ナレーションの声の高さの動き（抑揚）を広げる（Praat の PSOLA：parselmouth）。

  <venv>/bin/python tools/v3/cutC/vo_prosody.py <in.wav> <out.wav> <alpha> [end_st] [end_dur]

  alpha   … 中央値より上の高さの動き（アクセントの山）を何倍にするか（半音で×alpha。1.0 で変わらず、1.3〜1.5 がめやす）。
            中央値より下は ×1.1 だけ（下げすぎるときしみ声になる）、88 Hz より下には動かさない
  end_st  … 行の最後 end_dur 秒の高さを、なめらかに end_st 半音だけ上げ下げする（+ で語尾を上げる・− で落とす）

AivisSpeech（Style-Bert-VITS2）は mora ごとの高さを指定できない（audio_query の pitch は使われない）ので、
スタイル・強さ・表記で選んだテイクに、仕上げとしてこれをかける。声の高さの推定は 70〜350 Hz（男声）、
推定がとびはねた所（中央値から 10 半音以上）は動かさない。音色・長さは変わらない（PSOLA は高さだけ）。
必要なもの：praat-parselmouth（scratchpad の venv に入れてある：pip install praat-parselmouth）。
"""
import sys

import numpy as np
import parselmouth
from parselmouth.praat import call


def expand(src, dst, alpha=1.5, end_st=0.0, end_dur=0.35, floor=70, ceil=350, alpha_down=1.1, min_hz=88.0):
    snd = parselmouth.Sound(src)
    sr = snd.sampling_frequency
    man = call(snd, 'To Manipulation', 0.01, floor, ceil)
    pt = call(man, 'Extract pitch tier')
    n = int(call(pt, 'Get number of points'))
    ts = np.array([call(pt, 'Get time from index', i + 1) for i in range(n)])
    fs = np.array([call(pt, 'Get value at index', i + 1) for i in range(n)])
    med = float(np.median(fs))
    st = 12 * np.log2(fs / med)
    good = np.abs(st) < 10
    # 中央値より上（アクセント・強調の山）は alpha 倍、下は alpha_down 倍だけ。下げすぎるときしみ声（80 Hz 未満）に
    # 落ちて PSOLA の荒れも出るので、下は控えめにし、min_hz より下には動かさない
    new = np.where(good, np.where(st > 0, st * alpha, st * alpha_down), st)
    lo = 12 * np.log2(min_hz / med)
    new = np.where(good & (new < lo), np.minimum(st, lo), new)
    if end_st:
        t_end = ts.max()
        w = np.clip((ts - (t_end - end_dur)) / end_dur, 0, 1)
        new = new + end_st * (0.5 - 0.5 * np.cos(np.pi * w))
    pt2 = call('Create PitchTier', 'new', snd.xmin, snd.xmax)
    for t, s in zip(ts, new):
        call(pt2, 'Add point', float(t), float(med * 2 ** (s / 12)))
    call([pt2, man], 'Replace pitch tier')
    out = call(man, 'Get resynthesis (overlap-add)')
    y = out.values[0]
    pk = np.abs(y).max()
    if pk > 0.99:
        y = y * 0.99 / pk
    import wave
    with wave.open(dst, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes((np.clip(y, -1, 1) * 32767).astype('<i2').tobytes())
    return dict(alpha=alpha, alpha_down=alpha_down, end_st=end_st, end_dur=end_dur, med_hz=round(med, 1), points=n)


def gain_limited(path, db, ceil=0.97, attack=0.005, release=0.06):
    """path の wav を db だけ上げ、ceil を超える山だけを先読み 5 ms・戻り 60 ms のリミッターで抑える（その行だけ小さく
    聞こえるのを直す用。エンジンの出力はどれもピーク 0 dBFS にそろっているので、上げるにはピークを抑えるしかない）"""
    import wave
    with wave.open(path) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float64) / 32768
    y = x * 10 ** (db / 20)
    need = np.minimum(1.0, ceil / np.maximum(np.abs(y), 1e-9))
    a = max(1, int(attack * sr))
    from scipy.ndimage import minimum_filter1d
    g = minimum_filter1d(need, size=2 * a + 1)            # 先読み：山の前後 attack 秒は下げておく
    r = np.exp(-1.0 / (release * sr))
    sm = np.empty_like(g)
    cur = 1.0
    for i in range(len(g)):                               # 下げるのは即座、戻すのは release でゆっくり
        cur = g[i] if g[i] < cur else g[i] + (cur - g[i]) * r
        sm[i] = cur
    y = y * sm
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype('<i2').tobytes())
    return dict(gain_db=db, ceil=ceil, max_reduction_db=round(float(-20 * np.log10(sm.min())), 2))


if __name__ == '__main__':
    a = sys.argv[1:]
    print(expand(a[0], a[1], float(a[2]) if len(a) > 2 else 1.5, float(a[3]) if len(a) > 3 else 0.0,
                 float(a[4]) if len(a) > 4 else 0.35))
