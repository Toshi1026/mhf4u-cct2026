"""候補の声で案B のリールの音（ナレーション＋BGM＋効果音＋波）だけを作る。映像は作らない。

  nice -n 10 python3 tools/v3/vo_reel_audio.py vv_kenzaki_mesuo aivis_aida ...
  → exports/v3/voice_cands/reel_<候補>/vo/（NN.wav・durations.json・moras.json）と reel_<候補>/mix_B.wav

声は「声のある区間」の前 0.03 秒・後ろ 0.06 秒で切りそろえる（調整済み VOICEVOX と同じ余白）。
moras.json が無い候補（AivisSpeech）は、現状 vo_B のモーラ時刻を声の区間に比例で当てはめる（目安）。
06 の「キラーン」は、06 の後半で一番長い無音のあとの立ち上がりに合わせ直す（キラーンの効果音がここに付くため）。
"""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'cutB'))
import vo_cands_eval as E  # noqa: E402

SR = 48000
VO_B = os.path.join(E.ROOT, 'exports', 'v3', 'vo_B')


def write_wav(path, x):
    import wave
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype('<i2').tobytes())


def kira_onset(x):
    """後半で一番長い無音（-45 dBFS 未満が続く所）が終わる時刻"""
    h = int(0.01 * SR)
    n = len(x) // h
    r = 20 * np.log10(np.sqrt(np.mean(x[:n * h].reshape(n, h) ** 2, 1)) + 1e-9)
    loud = r > -45
    best, cur, end = 0, 0, None
    for i in range(n // 2, n):
        if not loud[i]:
            cur += 1
        else:
            if cur > best:
                best, end = cur, i
            cur = 0
    return end * h / SR if end else None


def build_vo(name, out):
    src = dict((n, d) for n, d, _ in E.SETS)[name]
    os.makedirs(out, exist_ok=True)
    has_moras = os.path.exists(os.path.join(src, 'moras.json'))
    mor_b = json.load(open(os.path.join(VO_B, 'moras.json')))
    durs, moras = {}, {}
    for k in E.KEYS:
        x = E.read(os.path.join(src, k + '.wav')).astype(np.float64)
        if has_moras:                 # 調整済み VOICEVOX：そのまま（すでに 0.03/0.06 の余白）
            shutil.copy(os.path.join(src, k + '.wav'), os.path.join(out, k + '.wav'))
            durs[k] = round(len(x) / SR, 3)
            continue
        t0, t1 = E.speech_span(x, thr=-50.0)   # -50：語尾の弱い音を残す（阿井田のハム -57 より上）
        a, b = max(0, int((t0 - E.PAD0) * SR)), min(len(x), int((t1 + E.PAD1) * SR))
        y = x[a:b].copy()
        f = int(0.004 * SR)
        y[:f] *= np.linspace(0, 1, f)
        y[-f:] *= np.linspace(1, 0, f)
        write_wav(os.path.join(out, k + '.wav'), y)
        durs[k] = round(len(y) / SR, 3)
        # vo_B のモーラ時刻を比例で
        xb = E.read(os.path.join(VO_B, k + '.wav'))
        b0, b1 = E.speech_span(xb, thr=-50.0)
        n0, n1 = t0 - a / SR, t1 - a / SR
        sc = (n1 - n0) / max(1e-6, b1 - b0)
        mm = [[m, round(n0 + (s - b0) * sc, 3)] for m, s in mor_b[k]]
        if k == '06':
            ko = kira_onset(y)
            i = max(i for i, (m, _) in enumerate(mm) if m == 'キ')
            if ko:
                old_i = mm[i][1]
                span = (n1 - ko) / max(1e-6, (n1 - old_i))
                for j in range(i, len(mm)):
                    mm[j][1] = round(ko + (mm[j][1] - old_i) * span, 3)
        moras[k] = mm
    if has_moras:
        shutil.copy(os.path.join(src, 'moras.json'), os.path.join(out, 'moras.json'))
    else:
        json.dump(moras, open(os.path.join(out, 'moras.json'), 'w'), ensure_ascii=False)
    json.dump(durs, open(os.path.join(out, 'durations.json'), 'w'))
    return durs


def main():
    import core
    import timeline_b as TL
    import audio_b
    if sys.argv[1] == '--lengths':        # 全候補の vo を作り、timeline_b の式でリールの長さだけ出す（音は作らない）
        out = {}
        for name, _, _ in E.SETS:
            vo = os.path.join(E.VC, 'reel_' + name, 'vo')
            durs = build_vo(name, vo)
            TL.VO = vo
            bs = TL.load()
            out[name] = dict(vo_total=round(sum(durs.values()), 2), reel_sec=round(bs[-1]['f1'] / core.FPS, 2))
            print(name, out[name])
        json.dump(out, open(os.path.join(E.VC, 'reel_lengths.json'), 'w'), indent=1)
        return
    for name in sys.argv[1:]:
        base = os.path.join(E.VC, 'reel_' + name)
        vo = os.path.join(base, 'vo')
        durs = build_vo(name, vo)
        TL.VO = vo
        old = core.OUT
        core.OUT = base                 # audio_b は core.OUT/mix_B.wav に書く（効果音の場所は読み込み時に決まっている）
        try:
            audio_b.main()
        finally:
            core.OUT = old
        bs = TL.load()
        print(name, 'speech %.2fs' % sum(durs.values()), 'reel %.2fs' % (bs[-1]['f1'] / core.FPS))


if __name__ == '__main__':
    main()
