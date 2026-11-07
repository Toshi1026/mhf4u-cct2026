"""v3・案A の音：ナレーション＋コードで作る BGM・ごく小さな効果音＋実写パートの現地の波の音 → exports/v3/mix.wav

BGM：D メジャーの静かなパッド（84BPM、D–Bm–G–A）。06 から柔らかいキック、10 の「26」で澄んだベル 1 回、
09 の終わりに 0.2 秒だけ全部の音を抜く。12 から BGM を下げて現地の波の音へ。最後は和音を伸ばして自然に切る。
"""
import json
import os
import subprocess
import wave

import numpy as np

import core
import timeline

SR = 48000
BPM = 84
BEAT = 60 / BPM


def read_wav(path):
    with wave.open(path) as f:
        sr, n, ch = f.getframerate(), f.getnframes(), f.getnchannels()
        a = np.frombuffer(f.readframes(n), '<i2').astype(np.float32) / 32768
    a = a.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        a = np.interp(np.arange(int(len(a) * SR / sr)) / SR, np.arange(len(a)) / sr, a)
    return a


def footage_audio(name, t0, t1):
    p = subprocess.run(['ffmpeg', '-v', 'error', '-ss', '%.3f' % t0, '-t', '%.3f' % (t1 - t0), '-i',
                        os.path.join(core.FOOT, name), '-ac', '1', '-ar', str(SR), '-f', 's16le', '-'],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, '<i2').astype(np.float32) / 32768


def env_adsr(n, a, r):
    e = np.ones(n, np.float32)
    na, nr = int(a * SR), int(r * SR)
    if na:
        e[:na] = np.linspace(0, 1, na) ** 2
    if nr:
        e[-nr:] *= np.linspace(1, 0, nr) ** 2
    return e


def onepole_lp(x, fc):
    a = np.exp(-2 * np.pi * fc / SR)
    y = np.empty_like(x)
    s = 0.0
    for i in range(0, len(x), 4096):          # ブロックごとに（Python ループは遅いので scipy なしの簡易版）
        blk = x[i:i + 4096]
        out = np.empty_like(blk)
        for j, v in enumerate(blk):
            s = (1 - a) * v + a * s
            out[j] = s
        y[i:i + 4096] = out
    return y


def hz(m):
    return 440 * 2 ** ((m - 69) / 12)


CHORDS = [[50, 57, 62, 66, 69], [47, 54, 59, 62, 66], [43, 50, 55, 59, 62], [45, 52, 57, 61, 64]]   # D Bm G A


def pad(total):
    t = np.arange(int(total * SR)) / SR
    out = np.zeros_like(t, dtype=np.float32)
    bar = 4 * BEAT
    k = 0
    while k * bar < total:
        s0 = int(k * bar * SR)
        n = min(int((bar + 1.2) * SR), len(t) - s0)
        if n <= 0:
            break
        tt = np.arange(n) / SR
        e = env_adsr(n, 0.9, 1.4)
        v = np.zeros(n, np.float32)
        for m in CHORDS[k % 4]:
            f = hz(m)
            for det in (-0.12, 0.12):        # わずかにずらした2本で広がりを出す
                v += np.sin(2 * np.pi * f * (1 + det / 100) * tt) * 0.5 + 0.15 * np.sin(4 * np.pi * f * tt)
        out[s0:s0 + n] += v * e / 12
        k += 1
    return out


def kick(n=0.35):
    tt = np.arange(int(n * SR)) / SR
    f = 50 + 70 * np.exp(-tt * 30)
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 9)).astype(np.float32)


def bell(n=2.4, f=hz(86)):
    tt = np.arange(int(n * SR)) / SR
    v = sum(a * np.sin(2 * np.pi * f * r * tt) * np.exp(-tt * d) for a, r, d in
            [(1, 1, 2.2), (0.5, 2.756, 3.5), (0.25, 5.404, 5), (0.12, 8.933, 7)])
    return (v * np.minimum(1, tt / 0.004)).astype(np.float32)


def sub_hit(n=1.6):
    tt = np.arange(int(n * SR)) / SR
    return (np.sin(2 * np.pi * 42 * tt) * np.exp(-tt * 2.2) * np.minimum(1, tt / 0.03)).astype(np.float32)


def tick(n=0.05, f=1800):
    tt = np.arange(int(n * SR)) / SR
    return (np.sin(2 * np.pi * f * tt) * np.exp(-tt * 90)).astype(np.float32)


def add(buf, x, t, g=1.0):
    s = int(t * SR)
    if s >= len(buf):
        return
    e = min(len(buf), s + len(x))
    buf[s:e] += x[:e - s] * g


def db(x):
    return 10 ** (x / 20)


def main():
    bs = timeline.load()
    B = {b['key']: b for b in bs}
    total = bs[-1]['f1'] / core.FPS
    n = int(total * SR)
    vo = np.zeros(n, np.float32)
    for b in bs:
        p = os.path.join(core.VO, b['key'] + '.wav')
        if b['vo'] is not None and os.path.exists(p):
            add(vo, read_wav(p), b['vo'])
    vo /= max(1e-6, np.abs(vo).max())

    music = pad(total)
    t = np.arange(n) / SR
    t06, t09e, t10, t12 = B['06']['f0'] / 30, B['09']['f1'] / 30, B['10']['f0'] / 30, B['12']['f0'] / 30
    k = kick()
    tb = t06
    while tb < t09e - 0.25:                    # 06〜09 は柔らかいキック
        add(music, k, tb, 0.35)
        tb += BEAT
    gain = np.ones(n, np.float32)
    t03a, t03b = B['03']['f0'] / 30, B['03']['f1'] / 30
    # 03（海の底）は水の中のようにこもらせる：ローパスをかけた版と混ぜる
    lp = onepole_lp(music, 600)
    w03 = np.clip(np.minimum((t - t03a) / 0.15, (t03b - t) / 0.2), 0, 1)
    music = music * (1 - w03) + lp * w03 * 1.3
    # 09 の終わりに 0.2 秒、全部の音を抜く（答えの前の間）
    gap = (t > t09e - 0.2) & (t < t09e + 0.02)
    gain[gap] = 0
    gain *= np.where(t > t12, db(-7), 1.0) * np.clip((total - t) / 1.2, 0, 1) ** 0.5
    music = music * gain

    sfx = np.zeros(n, np.float32)
    add(sfx, sub_hit(), 0.0, 0.5)
    b10 = B['10']
    ro = next((s for m, s in b10['moras'] if m == 'ロ'), b10['f0'] / 30 + 1.2)
    add(sfx, bell(), ro, 0.22)
    b02 = B['02']
    chi = [s for m, s in b02['moras'] if m == 'チ']
    if chi:
        add(sfx, tick(0.12, 900), chi[-1] + 0.08, 0.25)
    b06 = B['06']
    no = [s for m, s in b06['moras'] if m == 'ノ']
    if no:
        add(sfx, tick(0.08, 1300), no[0], 0.2)

    # 実写パートの現地の波の音（素材の音は小さいので持ち上げる）
    amb = np.zeros(n, np.float32)
    b12, b13 = B['12'], B['13']
    a1 = footage_audio('IMG_9662.MOV', 1.6, 4.6)
    a2 = footage_audio('IMG_9636.MOV', 1.4, 5.4)
    for a in (a1, a2):
        a -= a.mean()
    a1 /= max(1e-6, np.sqrt(np.mean(a1 ** 2)))
    a2 /= max(1e-6, np.sqrt(np.mean(a2 ** 2)))
    amb_t0 = b12['f0'] / 30 - 0.3
    seg = np.concatenate([a1, a2])
    add(amb, seg * env_adsr(len(seg), 0.35, 0.6), amb_t0, 0.06)
    amb *= np.clip((total - t) / 0.8, 0, 1)

    mix = vo * db(-3) + music * db(-17) + sfx * db(-12) + amb * db(-6)
    mix = np.tanh(mix * 1.1) / 1.1
    st = np.stack([mix, mix], axis=1)
    pk = np.abs(st).max()
    st = st / pk * db(-1.5)
    out = os.path.join(core.OUT, 'mix.wav')
    with wave.open(out, 'wb') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((st * 32767).astype('<i2').tobytes())
    print('wrote', out, '%.2fs' % total)


if __name__ == '__main__':
    main()
