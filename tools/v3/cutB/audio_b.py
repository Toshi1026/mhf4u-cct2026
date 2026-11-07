"""案B の音 → exports/v3/mix_B.wav。
ナレーション（vo_B）＋ BGM（案A と同じ静かなパッド、03〜07 に柔らかいキック）＋効果音（効果音ラボ：商用可・表記任意）
＋実写パートの波の音。効果音は earshot rank で候補から選んだ：
  拳と飛んでいく音 = 剣の素振り1（z +5.5 "fast whoosh"）、キラーン = キラッ2（z +4.8 "magic sparkle chime"）、
  着水 = お風呂でチャポン（z +3.5 "small splash"）、波 = 海岸4（z +5.8）。
"""
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio as A          # noqa: E402  案A の部品（pad, kick, decode, env_adsr, add, db, read_wav）
import core                # noqa: E402
import timeline_b as TL    # noqa: E402

SR = A.SR
LAB = os.path.join(core.OUT, 'sfx', 'lab')


def lab(rel, t0=0.0, t1=None):
    p = os.path.join(LAB, rel)
    if t1 is None:
        import subprocess
        d = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p],
                                 capture_output=True, text=True).stdout)
        t1 = d
    x = A.decode(p, t0, t1)
    x = x - x.mean()
    return x / max(1e-6, np.abs(x).max())


def mora_t(b, m, nth=0):
    ts = [s for mm, s in b['moras'] if mm == m]
    return ts[nth] if len(ts) > nth else None


def main():
    bs = TL.load()
    B = {b['key']: b for b in bs}
    total = bs[-1]['f1'] / core.FPS
    n = int(total * SR)
    t = np.arange(n) / SR
    vo = np.zeros(n, np.float32)
    for b in bs:
        p = os.path.join(TL.VO, b['key'] + '.wav')
        if b['vo'] is not None and os.path.exists(p):
            A.add(vo, A.read_wav(p), b['vo'])
    vo /= max(1e-6, np.abs(vo).max())

    music = A.pad(total)
    k = A.kick()
    tb, te = B['03']['f0'] / 30, B['07']['f1'] / 30
    while tb < te - 0.3:
        A.add(music, k, tb, 0.3)
        tb += A.BEAT
    tT, t08 = B['T']['f0'] / 30, B['08']['f0'] / 30
    gain = np.ones(n, np.float32)
    ramp = 0.01
    gain *= np.clip(np.maximum((tT - 0.15 - t) / ramp, (t - tT - 0.1) / ramp), 0, 1)     # 着水の直前に一瞬の間
    gain *= np.where(t > t08, A.db(-7), 1.0) * np.clip((total - t) / 1.2, 0, 1) ** 0.5
    music *= gain

    sfx = np.zeros(n, np.float32)
    A.add(sfx, A.sub_hit(), 0.0, 0.35)
    A.add(sfx, lab('whoosh/sword-gesture1.mp3'), 0.25, 0.55)            # 拳 → 飛んでいく
    k1 = mora_t(B['01'], 'カ')                                          # 「かなた」の頃にキラーン
    A.add(sfx, lab('chimes/kira2.mp3'), (k1 or 2.0) + 0.25, 0.35)
    k6 = mora_t(B['06'], 'キ')
    if k6:
        A.add(sfx, lab('chimes/kira2.mp3'), k6, 0.4)
    A.add(sfx, lab('water/bath-thapon1.mp3'), tT + 0.05, 0.45)          # 着水

    amb = np.zeros(n, np.float32)
    a1 = A.decode(os.path.join(core.FOOT, 'IMG_9662.MOV'), 1.6, 4.6)
    a2 = A.decode(os.path.join(LAB, 'ambience', 'wave4.mp3'), 2.0, 2.0 + total)
    for a in (a1, a2):
        a -= a.mean()
        a /= max(1e-6, np.sqrt(np.mean(a ** 2)))
    t0 = t08 - 0.3
    s1, s2 = np.zeros(n, np.float32), np.zeros(n, np.float32)
    A.add(s1, a1 * A.env_adsr(len(a1), 0.35, 0.05), t0)
    A.add(s2, a2, t0)
    w = np.clip((t - (t0 + 1.6)) / 1.0, 0, 1)
    amb = (s1 * np.cos(w * np.pi / 2) + s2 * np.sin(w * np.pi / 2) * 0.8) * 0.06
    amb *= np.clip((t - t0) / 0.35, 0, 1) * np.clip((total - t) / 0.8, 0, 1)

    mix = vo * A.db(-3) + music * A.db(-17) + sfx * A.db(-10) + amb * A.db(-6)
    mix = np.tanh(mix * 1.1) / 1.1
    fi = int(0.005 * SR)
    mix[:fi] *= np.linspace(0, 1, fi)
    st = np.stack([mix, mix], 1)
    st = st / np.abs(st).max() * A.db(-1.5)
    out = os.path.join(core.OUT, 'mix_B.wav')
    with wave.open(out, 'wb') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((st * 32767).astype('<i2').tobytes())
    print('wrote', out, '%.2fs' % total)


if __name__ == '__main__':
    main()
