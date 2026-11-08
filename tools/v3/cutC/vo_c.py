"""案C のナレーション（edl/v3/SCRIPT_C_ig.md の13行）を AivisSpeech 阿井田茂（Calm）で作る。

  python3 tools/v3/cutC/vo_c.py takes [--lines 04,05] [--takes 4] [--threads 2]
      → exports/v3/voice_cands/C_takes/<NN>/<name>.wav と pick.json（テイクごとの Whisper 読み確認）
  python3 tools/v3/cutC/vo_c.py collect [04=t1 ...]
      → 行ごとに一番よいテイクを、声の区間の前 0.03 秒・後ろ 0.06 秒で切りそろえて exports/v3/vo_C/NN.wav に。
        durations.json、lines.json（字幕用の文）、reading.json（合成に使った読み・モーラ列）、candidate.json（選んだテイクと設定）
  その後 tools/v3/cutC/vo_align.py で moras.json（強制アラインメントの実測）を作る。

設定は案B の阿井田と同じ（tools/v3/vo_aivis_cands.json の aida：Calm・話速 1.12・抑揚 1.0）。アクセントは自動（案B の
手付けカナ vo_tune.KANA はキーが B の行なので使わない）。エンジンの起動方法は tools/v3/vo_aivis.py の冒頭。
数字は読み違いが出ないように、かなで書く（「やく、せんよんひゃくトン」など）。01 の「キラーン」は句末の「ン」が
消えやすい話者なので、案B と同じく「キラーンと。」で合成して「と」を切り、前半とつなぐ。
"""
import argparse
import itertools
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(V3, '..', '..'))
sys.path.insert(0, V3)
import vo_aivis as va      # noqa: E402
import listen              # noqa: E402

OUT = os.path.join(ROOT, 'exports', 'v3', 'vo_C')
TAKES = os.path.join(ROOT, 'exports', 'v3', 'voice_cands', 'C_takes')
SR = 48000
PAD0, PAD1, THR = 0.03, 0.06, -50.0       # 切りそろえ：-50 dBFS（阿井田のハム -57 より上）で声の区間を決める

# 字幕・台本の文（lines.json）
LINES = {
    '01': 'アニメなら、悪役は、パンチ一発で、空のかなたへ。',   # 2026-11-08 修正5：キラーンは効果音だけ（ナレーションから外す）
    '03': 'でも、現実だと……こうなる。',
    '04': 'パンチの威力は、約1,400トン！',
    '05': 'ゾウ250頭分。',
    '06': 'マッハ6.7。',
    '07': '高さ、132キロ。',
    '08': '……宇宙です。',
    '09': 'そして、5分半後。',
    '11': '落ちたのは、太平洋と仁淀川が出会う海。',
    '12': '目の前に、Seaside MAZE。',
    '13': '海を見ながら、パスタとスイーツ。',
    '14': '屋上から、太平洋。',
    '15': '悪役も、ここなら悪くないかも。',
}

# 合成に使う表記（第1候補、必要なら別表記も）。parts は部分ごとに合成してつなぐ（vo_aivis.synth_one の per_line 形式）
VARIANTS = {
    '01': [dict(text='アニメなら、悪役は、パンチ一発で、空のかなたへ。')],   # 修正5（旧：アニメなら、キラーンと。）
    '03': [dict(text='でも、げんじつだと……こうなる。')],
    '04': [dict(text='パンチのいりょくは、やく、せんよんひゃく、トン！')],
    '05': [dict(text='ゾウ、にひゃくごじゅっとうぶん。'), dict(text='ゾウ、にひゃくごじゅっ頭ぶん。'),
           dict(text='象、にひゃくごじゅっとうぶん。'), dict(text='ゾウ。にひゃくごじゅっとうぶん。')],
    '06': [dict(text='マッハ、ろくてんなな。'), dict(text='マッハ。ろくてんなな。'), dict(text='まっは、ろくてんなな。'),
           dict(text='マッハ、6.7。'), dict(text='マッハー、ろくてんなな。'),
           dict(parts=[dict(text='マッ。'), dict(text='ハ、ろくてんなな。')], gaps=[0.05]),
           dict(text='マハ、ろくてんなな。'),
           dict(parts=[dict(text='マッ。'), dict(text='ハ、ろくてんなな。')], gaps=[0.0]),
           dict(parts=[dict(text='マッ。'), dict(text='ハ、ろくてんなな。')], gaps=[0.02])],
    '07': [dict(text='たかさ、ひゃくさんじゅうにキロ。')],
    '08': [dict(text='……うちゅうです。'), dict(text='うちゅうです。')],
    '09': [dict(text='そして、ごふんはんご。')],
    '11': [dict(text='落ちたのは、太平洋と、によどがわが出会う海。'),
           dict(text='落ちたのは、太平洋と、仁淀川が、出会う海。')],
    '12': [dict(text='目の前に、シーサイドまぜ。'), dict(text='目の前に、シーサイド・まぜ。')],
    '13': [dict(text='海を見ながら、パスタとスイーツ。')],
    '14': [dict(text='屋上から、太平洋。')],
    '15': [dict(text='悪役も、ここなら悪くないかも。')],
}

# Whisper の文字起こしに必ず入っていてほしい語（どれか1つ）。かな3文字以上は読みでも照合される（listen.check_reading）
EXPECT = {
    '01': [['アニメ', 'あにめ'], ['悪役', 'あくやく'], ['パンチ', 'ぱんち'], ['一発', 'いっぱつ', '1発'],
           ['かなた', '彼方', 'カナタ']],
    '03': [['でも', 'デモ'], ['現実', 'げんじつ', 'ゲンジツ'], ['こうなる', 'こーなる', 'コウナル']],
    '04': [['パンチ', 'ぱんち'], ['威力', 'いりょく'], ['1400', '1,400', '千四百', 'せんよんひゃく'], ['トン', 't', 'とん']],
    '05': [['ゾウ', '象', 'ぞう', '像', '増'], ['250', '二百五十', 'にひゃくごじゅう']],
    '06': [['マッハ', 'まっは'], ['6.7', '6点7', '六点七', 'ろくてんなな', '6てん7']],
    '07': [['高さ', 'たかさ'], ['132', '百三十二', 'ひゃくさんじゅうに']],
    '08': [['宇宙', 'うちゅう']],
    '09': [['5分半', '五分半', 'ごふんはん', '5分は']],
    '11': [['太平洋', 'たいへいよう'], ['仁淀川', 'によどがわ', 'ニヨドガワ'], ['出会う', 'であう']],
    '12': [['シーサイド', 'しーさいど'], ['まぜ', 'マゼ', 'MAZE', '混ぜ']],
    '13': [['パスタ'], ['スイーツ']],
    '14': [['屋上', 'おくじょう'], ['太平洋', 'たいへいよう']],
    '15': [['悪役', 'あくやく'], ['悪くない', 'わるくない']],
}

# 聞こえてはいけない語（Whisper の文字起こしにあれば読み NG）。04 は「1,400」が「3,400」に聞こえないこと
NEG = {'04': ['3400', '3,400', '三千', 'さんぜん', '4400', '4,400']}

for _k, _v in VARIANTS.items():          # vo_aivis.synth_one は既定の表記を TEXT[k] から引くので、C の行も入れておく
    va.TEXT.setdefault(_k, _v[0].get('text') or ''.join(p['text'] for p in _v[0]['parts']))

BASE = dict(model='47e53151-a378-46f3-abee-ce13aa07feb1', speaker='阿井田 茂', style='Calm',
            speed=1.12, inton=1.0, tempo=1.0, pause_scale=1.0)


def reading(pl):
    """（hstretch した「マハ」は「マッハ」として返す）
    合成に使う表記の読み（カタカナのモーラ列）を audio_query から。parts は cut_last の最後の1モーラを除く"""
    if pl.get('hstretch'):
        ms = reading({k: v for k, v in pl.items() if k != 'hstretch'})
        return ms[:1] + ['ッ'] + ms[1:] if ms[:2] == ['マ', 'ハ'] else ms
    sid = va.style_id(BASE['speaker'], BASE['style'])
    parts = pl['parts'] if pl.get('parts') else [dict(text=pl['text'])]
    moras = []
    for p in parts:
        q = json.loads(va.req('POST', '/audio_query', {'text': p['text'], 'speaker': sid}))
        ms = [m['text'] for ap in q['accent_phrases'] for m in ap['moras'] if m['vowel'] != 'pau']
        if p.get('cut_last'):
            ms = ms[:-1]
            while ms and ms[-1] == 'ッ':      # 「キラーンっと。」の促音は「と」と一緒に消える
                ms = ms[:-1]
        moras += ms
    return moras


def score(path, k, kana, threads):
    tr = listen.whisper(path, 'large-v3-turbo', threads)
    r = listen.check_reading(tr, kana, EXPECT.get(k))
    neg = [w for w in NEG.get(k, []) if w in tr['text']]
    must_ok = all(ok for _, ok in r.get('must', [])) and not neg
    p = sum(w['p'] for w in tr['words']) / max(1, len(tr['words']))
    return dict(text=tr['text'], ratio=r.get('ratio'), must_ok=must_ok,
                must_ng=[a[0] for a, ok in r.get('must', []) if not ok] + ['NEG:' + w for w in neg], p=round(p, 3), diffs=r.get('diffs'))


def rank_key(v):
    return (v['must_ok'], v['ratio'] or 0, v['p'])


def takes(a):
    os.makedirs(TAKES, exist_ok=True)
    pj = os.path.join(TAKES, 'pick.json')
    res = json.load(open(pj)) if os.path.exists(pj) else {}
    keys = a.lines.split(',') if a.lines else list(LINES)
    speeds = [float(s) for s in a.speeds.split(',')] if a.speeds else [BASE['speed']]
    for k in keys:
        for (vi, pl), sp, tk in itertools.product(enumerate(VARIANTS[k]), speeds, range(a.takes)):
            if a.variants and str(vi) not in a.variants.split(','):
                continue
            pl = dict(pl, speed=sp)
            name = f't{vi}_s{sp:g}_k{tk}'
            path = os.path.join(TAKES, k, name + '.wav')
            cfg = {**BASE, 'per_line': {k: pl}}
            try:
                prm, tim = va.synth_one(cfg, k, path, accent='auto')
            except Exception as e:      # noqa: BLE001
                print(k, name, 'FAILED', e, flush=True)
                continue
            kana = ''.join(reading(pl))
            sc = score(path, k, kana, a.threads)
            res.setdefault(k, {})[name] = dict(settings=pl, dur=tim['dur'], kana=kana, **sc)
            print(f'{k} {name:14s} {tim["dur"]:5.2f}s ratio {sc["ratio"]} must '
                  f'{"OK" if sc["must_ok"] else "NG " + ",".join(sc["must_ng"])} p {sc["p"]}  {sc["text"]}', flush=True)
            json.dump(res, open(pj, 'w'), ensure_ascii=False, indent=1)


def hstretch(name, add=0.08, gl=0.024, seed=0):
    """06「マッハ」：この話者（と SBV2 の他の話者）は「ッハ」を「ッカ」に近く読み、Whisper も wav2vec2 も「マッカ」と聞く。
    「マハ、ろくてんなな。」で合成し、ハの h の摩擦（声の谷）を add 秒だけ伸ばす（24 ms の窓をランダムな位置から重ね足し）
    ＝ 実際の「マッハ」の発音 [mahha] に近づける。2026-11-08：t6_s1.12_k0 に 80 ms で Whisper「マッハ6.7」。"""
    import wave
    src = os.path.join(TAKES, '06', name + '.wav')
    x = read(src)
    h = int(SR * 0.005)
    e = 20 * np.log10(np.sqrt(np.mean(x[:len(x) // h * h].reshape(-1, h) ** 2, 1)) + 1e-9)
    on = int(np.argmax(e > e.max() - 30))
    lo, hi = on + 10, on + 70                       # 声の頭から 50〜350 ms の中で、マの母音のあとの谷
    thr = e[lo:hi].max() - 7
    low = e[lo:hi] < thr
    best, cur, end = 0, 0, 0
    for i, v in enumerate(low):
        cur = cur + 1 if v else 0
        if cur > best:
            best, end = cur, i
    a, b = lo + end - best + 1, lo + end + 1
    mid = (a + b) // 2 * h
    G = int(gl * SR)
    win = np.hanning(G)
    seg = x[a * h:b * h]
    n_add = int(add * SR)
    buf = np.zeros(n_add + G)
    rng = np.random.default_rng(seed)
    for k in range(0, n_add, G // 2):
        o = rng.integers(0, max(1, len(seg) - G))
        buf[k:k + G] += seg[o:o + G] * win
    y = np.concatenate([x[:mid], buf[:n_add], x[mid:]])
    out = name + '_h%d' % round(add * 1000)
    with wave.open(os.path.join(TAKES, '06', out + '.wav'), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype('<i2').tobytes())
    pj = os.path.join(TAKES, 'pick.json')
    res = json.load(open(pj))
    base = res['06'][name]
    kana = ''.join(reading(base['settings']))
    sc = score(os.path.join(TAKES, '06', out + '.wav'), '06', kana, 2)
    res['06'][out] = dict(settings=dict(base['settings'], hstretch=dict(add=add, dip=[a * 0.005, b * 0.005])),
                          dur=round(len(y) / SR, 3), kana=kana, **sc)
    # 読み（モーラ列）は「マッハ」：ッを足す
    json.dump(res, open(pj, 'w'), ensure_ascii=False, indent=1)
    print(out, 'dip %.3f-%.3f' % (a * 0.005, b * 0.005), sc['text'], sc['p'])


def read(path):
    x = listen.decode(path)
    return (x[:, 0] if x.ndim == 2 else x).astype(np.float64)


def speech_span(x, thr=THR):
    """声の区間。しきい値は -50 dBFS か、そのテイクのハム（無音部の 10 パーセンタイル）＋8 dB の高い方
    （テイクによってハムが -49 dB まで上がり、-50 固定だとハムまで声として残るため）"""
    h = int(0.01 * SR)
    n = len(x) // h
    r = 20 * np.log10(np.sqrt(np.mean(x[:n * h].reshape(n, h) ** 2, 1)) + 1e-9)
    nz = r[r > -120]
    thr = max(thr, float(np.percentile(nz, 10)) + 8) if len(nz) else thr
    idx = np.where(r > thr)[0]
    return idx[0] * h / SR, min(len(x), (idx[-1] + 1) * h) / SR


def trim(src, dst):
    import wave
    x = read(src)
    t0, t1 = speech_span(x)
    a, b = max(0, int((t0 - PAD0) * SR)), min(len(x), int((t1 + PAD1) * SR))
    y = x[a:b].copy()
    f = int(0.004 * SR)
    y[:f] *= np.linspace(0, 1, f)
    y[-f:] *= np.linspace(1, 0, f)
    with wave.open(dst, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype('<i2').tobytes())
    return round(len(y) / SR, 3)


def collect(prefer):
    res = json.load(open(os.path.join(TAKES, 'pick.json')))
    os.makedirs(OUT, exist_ok=True)
    durs, chosen, rd = {}, {}, {}
    for k in LINES:
        vs = res.get(k, {})
        if k in prefer:
            vs = {n: v for n, v in vs.items() if n.startswith(prefer[k]) or n == prefer[k]} or vs
        if not vs:
            raise SystemExit(f'no take for {k}')
        name, v = max(vs.items(), key=lambda kv: rank_key(kv[1]))
        durs[k] = trim(os.path.join(TAKES, k, name + '.wav'), os.path.join(OUT, k + '.wav'))
        rd[k] = dict(text=v['settings'].get('text') or ''.join(p['text'] for p in v['settings']['parts']),
                     moras=reading(v['settings']))
        chosen[k] = dict(take=name, dur_trimmed=durs[k], whisper=v['text'], reading_ratio=v['ratio'],
                         must_ok=v['must_ok'], must_ng=v['must_ng'], whisper_p=v['p'], settings=v['settings'],
                         n_takes=len(res.get(k, {})))
        print(f'{k} {name:14s} {durs[k]:5.2f}s ratio {v["ratio"]} must {"OK" if v["must_ok"] else "NG"} p {v["p"]} {v["text"]}')
    json.dump(durs, open(os.path.join(OUT, 'durations.json'), 'w'))
    json.dump(LINES, open(os.path.join(OUT, 'lines.json'), 'w'), ensure_ascii=False, indent=0)
    json.dump(rd, open(os.path.join(OUT, 'reading.json'), 'w'), ensure_ascii=False, indent=1)
    meta = dict(
        engine='AivisSpeech Engine (docker ghcr.io/aivis-project/aivisspeech-engine:cpu-latest, CPU), localhost:10101',
        render='tools/v3/cutC/vo_c.py (takes → collect), moras by tools/v3/cutC/vo_align.py',
        model='阿井田 茂 v1.0.0 (47e53151-a378-46f3-abee-ce13aa07feb1), style Calm, speedScale 1.12, accent auto',
        license='ACML 1.0 (commercial OK, credit optional)', credit_text='音声合成：AivisSpeech（阿井田 茂 / CV：古山キリヲ）',
        trim=f'speech span at max({THR} dBFS, hum p10 + 8 dB), head {PAD0} s / tail {PAD1} s, 4 ms fades', lines=chosen,
        total_sec=round(sum(durs.values()), 2))
    json.dump(meta, open(os.path.join(OUT, 'candidate.json'), 'w'), ensure_ascii=False, indent=1)
    print('total %.2fs' % sum(durs.values()))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'hstretch':        # vo_c.py hstretch t6_s1.12_k0 0.08
        hstretch(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.08)
        return
    if len(sys.argv) > 1 and sys.argv[1] == 'collect':
        collect(dict(x.split('=', 1) for x in sys.argv[2:]))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['takes'])
    ap.add_argument('--lines')
    ap.add_argument('--takes', type=int, default=3)
    ap.add_argument('--speeds')
    ap.add_argument('--variants', help='表記の番号だけ（例 2,3）')
    ap.add_argument('--threads', type=int, default=2)
    takes(ap.parse_args())


if __name__ == '__main__':
    main()
