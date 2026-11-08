"""案C 10 の悪役の悲鳴「うわあああああ……」を作る（ナレーターの阿井田茂とは別の声）。

  python3 tools/v3/cutC/scream_c.py takes      → exports/v3/voice_cands/C_scream/<name>.wav と takes.json（長さ・終わり方の数値）
  python3 tools/v3/cutC/scream_c.py make <name> [<name2> ...]
      → exports/v3/vo_C/scream.wav（48 kHz モノラル）と credits.json

声：AivisSpeech の TANAKA（ACML 1.0・商用可・クレジット任意。作者「田中」、Aivis Hub 470a41e2-…）。
    男性の声で、驚き（surprise）・おどけ（Silly）のスタイルがあり、顔のない白いピクトグラムの悪役のコミカルな悲鳴に合う。
長さ 2.5〜3.0 秒が要る。1テイクの「うわあああああ！」は 1〜1.4 秒。「あ」を16個書いて話速 0.7 にすると 2.4 秒前後まで伸びる
（読点を入れると 0.75 秒の無音が入るので使わない）。足りない分は make に2本目を渡すと、その安定した「あ」を1本目の母音の途中に
差し込む（等パワーのクロスフェード）。語尾は1本目の自然な減衰のまま。音量の変化（落ちるにつれて大きく）はミックス側で付ける。
2026-11-08 の選定：earshot match「a man screaming in panic, a long terrified yell」（対：落ち着いた話し声・歌・笑い・怒鳴り声）で
Anger スタイルが一番強い（L1_Anger_s0.7 z +5.2・対案の中で 94%、怒鳴り声 5%）。
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(V3, '..', '..'))
sys.path.insert(0, V3)
import vo_aivis as va      # noqa: E402

SR = 48000
CAND = os.path.join(ROOT, 'exports', 'v3', 'voice_cands', 'C_scream')
OUT = os.path.join(ROOT, 'exports', 'v3', 'vo_C')
SPEAKER = 'TANAKA'
MODEL = '470a41e2-863f-4bf2-8a20-bf7cde0d8f7b'
TEXTS = ['うわあああああ！', 'うわあああああああっ！', 'うわぁぁぁあああああ！']
STYLES = ['surprise', 'Silly', 'Anger']
SPEEDS = [0.8, 1.0]
INTONS = [1.0, 1.5]


def synth(text, style, speed, inton):
    sid = va.style_id(SPEAKER, style)
    q = json.loads(va.req('POST', '/audio_query', {'text': text, 'speaker': sid}))
    q.update(kana=text, speedScale=speed, intonationScale=inton, tempoDynamicsScale=1.0, pitchScale=0.0, volumeScale=1.0,
             prePhonemeLength=0.05, postPhonemeLength=0.3, pauseLengthScale=1.0, outputSamplingRate=SR,
             outputStereo=False)
    return va._wav_to_np(va.req('POST', '/synthesis', {'speaker': sid}, q)), q


def env_db(x, hop=240):
    n = len(x) // hop
    return 20 * np.log10(np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, 1)) + 1e-9)


def measure(x):
    """有声区間・最後の 150 ms の減衰（dB/秒）・ピッチの目安（自己相関）"""
    e = env_db(x)
    on = np.where(e > e.max() - 35)[0]
    t0, t1 = on[0] * 0.005, (on[-1] + 1) * 0.005
    tail = e[max(on[0], on[-1] - 30):on[-1] + 1]
    slope = float(np.polyfit(np.arange(len(tail)) * 0.005, tail, 1)[0]) if len(tail) > 4 else 0.0
    return dict(t0=round(t0, 3), t1=round(t1, 3), voiced=round(t1 - t0, 3), tail_slope_db_s=round(slope, 1),
                peak_db=round(float(e.max()), 1))


def takes():
    os.makedirs(CAND, exist_ok=True)
    res = {}
    for ti, text in enumerate(TEXTS):
        for st in STYLES:
            for sp in SPEEDS:
                for it in INTONS:
                    name = f't{ti}_{st}_s{sp:g}_i{it:g}'
                    t = time.time()
                    try:
                        x, _ = synth(text, st, sp, it)
                    except Exception as e:      # noqa: BLE001
                        print(name, 'FAILED', e)
                        continue
                    va._np_to_wav(x, os.path.join(CAND, name + '.wav'))
                    m = measure(x)
                    res[name] = dict(text=text, style=st, speed=sp, inton=it, **m)
                    print(name, m, '%.1fs' % (time.time() - t), flush=True)
                    json.dump(res, open(os.path.join(CAND, 'takes.json'), 'w'), ensure_ascii=False, indent=1)


def vowel_region(x):
    """伸ばした「あ」の安定した区間（音量が最大から 6 dB 以内で続く一番長い所）"""
    e = env_db(x)
    good = e > e.max() - 6
    best, cur, end = 0, 0, 0
    for i, g in enumerate(good):
        cur = cur + 1 if g else 0
        if cur > best:
            best, end = cur, i
    return (end - best + 1) * 240, (end + 1) * 240


def first_run(x, rel=-35, gap=0.1):
    """最初の有声の塊（読点などで 0.1 秒以上の無音が入ったら、そこまで）"""
    e = env_db(x)
    on = e > e.max() + rel
    i0 = int(np.argmax(on))
    i, quiet = i0, 0
    while i < len(on):
        quiet = 0 if on[i] else quiet + 1
        if quiet * 0.005 >= gap:
            break
        i += 1
    return max(0, i0 * 240 - 480), (i - quiet + 1) * 240 + int(0.08 * SR)


def make(names, target=2.8, xf=0.10):
    """1本目の「うわあああ」（最初の塊、自然な語尾つき）の母音の途中に、2本目以降の安定した「あ」を差し込んで target 秒に。"""
    tk = json.load(open(os.path.join(CAND, 'takes.json')))
    xs = [va._wav_to_np(open(os.path.join(CAND, n + '.wav'), 'rb').read()).astype(np.float64) for n in names]
    a, b = first_run(xs[0])
    y = xs[0][a:b]
    nx = int(xf * SR)
    tt = np.linspace(0, np.pi / 2, nx)
    used, k = [], 1
    while len(y) / SR < target - 0.02 and len(names) > 1 and k < 6:
        src = xs[1 + (k - 1) % (len(xs) - 1)]
        sa, sb = vowel_region(src)
        need = int((target - len(y) / SR) * SR) + nx
        seg = src[sa:sb][:need]
        if len(seg) < 3 * nx:
            break
        ya, yb = vowel_region(y)
        cut = ya + int(0.6 * (yb - ya))                   # 安定した母音の 6 割の所に差し込む
        g = np.sqrt(np.mean(y[cut - nx:cut] ** 2)) / max(1e-9, np.sqrt(np.mean(seg[:nx] ** 2)))
        seg = seg * g
        g2 = np.sqrt(np.mean(y[cut:cut + nx] ** 2)) / max(1e-9, np.sqrt(np.mean(seg[-nx:] ** 2)))
        seg = seg * np.linspace(1, g2, len(seg))
        y = np.concatenate([y[:cut - nx], y[cut - nx:cut] * np.cos(tt) + seg[:nx] * np.sin(tt), seg[nx:-nx],
                            seg[-nx:] * np.cos(tt) + y[cut:cut + nx] * np.sin(tt), y[cut + nx:]])
        used.append(names[1 + (k - 1) % (len(xs) - 1)])
        k += 1
    f = int(0.02 * SR)
    y[-f:] *= np.linspace(1, 0, f)
    y[:int(0.005 * SR)] *= np.linspace(0, 1, int(0.005 * SR))
    y = y / np.abs(y).max() * 0.89
    os.makedirs(OUT, exist_ok=True)
    va._np_to_wav(y, os.path.join(OUT, 'scream.wav'))
    cred = dict(
        file='exports/v3/vo_C/scream.wav', duration_sec=round(len(y) / SR, 3), use='案C 10：悪役の悲鳴（落下）',
        engine='AivisSpeech Engine (docker ghcr.io/aivis-project/aivisspeech-engine:cpu-latest), LGPL-3.0, credit not required',
        model=f'{SPEAKER} v1.0.0', model_uuid=MODEL, creator='田中',
        hub_url='https://hub.aivis-project.com/aivm-models/' + MODEL,
        license='ACML 1.0 (Aivis Common Model License) — 個人・法人・営利利用可、クレジット表記は任意。禁止：本人・公式と誤認させる利用、'
                '話者のイメージを損なう利用、誹謗中傷・虚偽広告・政治宗教・反社会的利用。https://github.com/Aivis-Project/ACML',
        commercial_ok=True, credit_required=False, credit_text='音声合成：AivisSpeech（TANAKA）',
        takes=[dict(name=n, **tk[n]) for n in names], vowel_extended_from=used,
        method='1本目の最初の有声の塊（うわ＋あ＋自然な語尾）を使い、長さが足りない分は2本目の安定した「あ」を1本目の母音の途中（6割の所）に等パワー・クロスフェード 0.10 秒で差し込む。語尾は1本目のまま（切らない）。ピーク -1 dBFS。',
        narrator=dict(model='阿井田 茂 v1.0.0 (47e53151-a378-46f3-abee-ce13aa07feb1)', license='ACML 1.0',
                      credit_required=False, credit_text='音声合成：AivisSpeech（阿井田 茂 / CV：古山キリヲ）'))
    json.dump(cred, open(os.path.join(OUT, 'credits.json'), 'w'), ensure_ascii=False, indent=1)
    print('scream', round(len(y) / SR, 2), 's from', names, 'extended with', used)


if __name__ == '__main__':
    if sys.argv[1] == 'takes':
        takes()
    else:
        make(sys.argv[2:])
