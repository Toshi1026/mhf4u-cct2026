"""ナレーション候補（exports/v3/voice_cands/*/ と現状の vo_B）を同じ物差しで測る。聞けないので、これは「ふるい」で、決めるのは人の耳。

  nice -n 10 python3 tools/v3/vo_cands_eval.py            # → exports/v3/voice_cands/eval.json と eval.md
  nice -n 10 python3 tools/v3/vo_cands_eval.py --skip-earshot

測るもの（行ごと・候補ごと）
  - ファイル長と「声のある区間」の長さ（10ms RMS が -45 dBFS を超える最初〜最後）。リールの長さは timeline_b の式で
    （LEAD 0.08 + 声の区間 + 先頭0.03/末尾0.06 の余白 + GAP 0.08、MIN、無音ビート T/13）計算し直す
  - ラウドネス（pyloudnorm, BS.1770）・サンプルピーク・|x|>=0.999 のサンプル数・無音部のノイズ床（声の区間外/内の静かな 50ms の 10% 点）
  - faster-whisper large-v3-turbo の文字起こし → listen.check_reading で想定の読み（vo_B/moras.json）との一致率と必須語
  - earshot（別の venv）：realness の p_real、CLAP の match z（落ち着いた大人の男性/女性ナレーター、voice-over、
    ロボット的な TTS、かわいいアニメ声）。CLAP は日本語の抑揚・アクセントを判断できないので、参考の目安でしかない
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import listen  # noqa: E402

VC = os.path.join(ROOT, 'exports', 'v3', 'voice_cands')
SETS = [
    ('temp_suzumatsu', os.path.join(ROOT, 'exports', 'v3', 'vo_B'), '現状（VOICEVOX 雀松朱司）'),
    ('vv_kigashima_sourin', os.path.join(VC, 'vv_kigashima_sourin'), 'VOICEVOX 麒ヶ島宗麟（調整済み）'),
    ('vv_kenzaki_mesuo', os.path.join(VC, 'vv_kenzaki_mesuo'), 'VOICEVOX 剣崎雌雄（調整済み）'),
    ('vv_kenzaki_mesuo_s114', os.path.join(VC, 'vv_kenzaki_mesuo_s114'), 'VOICEVOX 剣崎雌雄（調整済み・話速1.14）'),
    ('aivis_aida', os.path.join(VC, 'aivis_aida'), 'AivisSpeech 阿井田茂'),
    ('aivis_honoka', os.path.join(VC, 'aivis_honoka'), 'AivisSpeech ほのか'),
    ('aivis_lux', os.path.join(VC, 'aivis_lux'), 'AivisSpeech Lux'),
    ('aivis_morioki', os.path.join(VC, 'aivis_morioki'), 'AivisSpeech morioki'),
]
KEYS = ['%02d' % i for i in range(1, 13)]
EXPECT = {
    '01': [['悪役'], ['パンチ'], ['彼方', 'かなた']],
    '02': [['本当', 'ほんとう']],
    '03': [['桂浜', 'かつらはま', 'カツラハマ', 'かつら浜'], ['10キロ', '十キロ', '10km']],
    '04': [['速さ', '早さ', 'はやさ'], ['1130', '1,130', '千百三十'], ['旅客機']],
    '05': [['パンチ'], ['190', '百九十'], ['35', '三十五'], ['ゾウ', '象', '増', 'ぞう']],
    '06': [['高さ', 'たかさ'], ['2500', '2,500', '二千五百'], ['ほぼ'], ['キラーン', 'きらーん', 'キラン', 'キラ〜ン']],
    '07': [['45', '四十五'], ['秒後', 'びょうご']],
    '08': [['仁淀川', 'によどがわ', 'ニヨドガワ'], ['河口']],
    '09': [['シーサイド'], ['まぜ', 'マゼ', 'MAZE', '混ぜ']],
    '10': [['ローソン'], ['産直', 'さんちょく', '3直', '3チョク']],
    '11': [['カフェ']],
    '12': [['屋上'], ['避難タワー', '避難たわー'], ['太平洋']],
}
# timeline_b と同じ式
TL_LEAD, TL_GAP = 0.08, 0.08
TL_FIXED = {'T': 1.20, '13': 1.80}
TL_MIN = {'08': 3.10, '10': 2.60, '11': 3.00, '12': 3.00}
PAD0, PAD1 = 0.03, 0.06           # 調整済み VOICEVOX と同じ前後の余白で揃えて比べる

Q_M = 'a calm warm adult man narrating a documentary in a low steady voice'
Q_F = 'a calm warm adult woman narrating a documentary in a steady voice'
Q_ALTS = ['a voice-over narration', 'a robotic text-to-speech voice', 'a cute anime girl voice']
EARSHOT_PY = '/home/user/vendor/earshot/.venv/bin/python'

EARSHOT_SNIPPET = r'''
import json, sys
from earshot import api
paths = json.load(open(sys.argv[1])); qm, qf, alts = json.load(open(sys.argv[2]))
out = {}
for p in paths:
    r = api.realness(p)
    m = api.match(p, qm, alts=[qf] + alts)
    z = {qm: m['z']}
    for a in m['alts']:
        z[a['query']] = a['z']
    out[p] = dict(p_real=r['p_real'], verdict=r['verdict'], z=z)
    print(p, round(r['p_real'], 2), {k[:24]: round(v, 2) for k, v in z.items()}, file=sys.stderr, flush=True)
json.dump(out, open(sys.argv[3], 'w'))
'''


def read(path):
    x = listen.decode(path)
    return x[:, 0] if x.ndim == 2 else x


def speech_span(x, sr=48000, thr=-45.0):
    h = int(0.01 * sr)
    n = len(x) // h
    r = 20 * np.log10(np.sqrt(np.mean(x[:n * h].reshape(n, h) ** 2, 1)) + 1e-9)
    idx = np.where(r > thr)[0]
    if len(idx) == 0:
        return 0.0, len(x) / sr
    return idx[0] * h / sr, min(len(x), (idx[-1] + 1) * h) / sr


def noise_floor(x, t0, t1, sr=48000):
    """声の区間の外（前後の余白）と中の、静かな 50ms 窓の RMS（dBFS）。完全な0は除く"""
    h = int(0.05 * sr)
    n = len(x) // h
    fr = x[:n * h].reshape(n, h)
    rms = np.sqrt(np.mean(fr ** 2, 1))
    keep = rms > 1e-7
    if not keep.any():
        return None
    return float(20 * np.log10(np.percentile(rms[keep], 10) + 1e-12))


def loud(x, sr=48000):
    import pyloudnorm as pyln
    try:
        return float(pyln.Meter(sr).integrated_loudness(x.astype(np.float64)))
    except Exception:
        return None


def reel_len(speech):
    t = 0.0
    for k in ['01', '02', '03', '04', '05', '06', '07', 'T', '08', '09', '10', '11', '12', '13']:
        if k in TL_FIXED:
            t += TL_FIXED[k]
        else:
            t += max(TL_MIN.get(k, 0), TL_LEAD + PAD0 + speech[k] + PAD1 + TL_GAP)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--skip-earshot', action='store_true')
    ap.add_argument('--skip-whisper', action='store_true')
    a = ap.parse_args()
    mor = json.load(open(os.path.join(ROOT, 'exports', 'v3', 'vo_B', 'moras.json')))
    exp = {k: ''.join(m for m, _ in mor[k]) for k in KEYS}
    cache_p = os.path.join(VC, 'eval_cache.json')
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    res = {}
    for name, d, label in SETS:
        rows = {}
        for k in KEYS:
            p = os.path.join(d, k + '.wav')
            x = read(p)
            t0, t1 = speech_span(x)
            row = dict(file=os.path.relpath(p, ROOT), file_sec=round(len(x) / 48000, 3), speech_sec=round(t1 - t0, 3),
                       t0=round(t0, 3), lufs=loud(x), peak_dbfs=round(float(20 * np.log10(np.abs(x).max() + 1e-12)), 2),
                       n_fullscale=int((np.abs(x) >= 0.999).sum()), floor_dbfs=noise_floor(x, t0, t1))
            st = os.stat(p)
            ck = f'{p}|{st.st_size}|{int(st.st_mtime)}'
            if not a.skip_whisper:
                if ck not in cache:
                    tr = listen.whisper(p, 'large-v3-turbo', a.threads)
                    cache[ck] = dict(text=tr['text'], p=round(float(np.mean([w['p'] for w in tr['words']] or [0])), 3))
                    json.dump(cache, open(cache_p, 'w'), ensure_ascii=False)
                tr = cache[ck]
                r = listen.check_reading(dict(text=tr['text']), exp[k], EXPECT.get(k))
                row.update(whisper=tr['text'], whisper_p=tr['p'], reading_ratio=r.get('ratio'), diffs=r.get('diffs'),
                           must_ng=[al[0] for al, ok in r.get('must', []) if not ok])
            rows[k] = row
            print(name, k, row['speech_sec'], row.get('whisper'), row.get('reading_ratio'), row.get('must_ng'), flush=True)
        res[name] = dict(label=label, dir=os.path.relpath(d, ROOT), lines=rows)

    if not a.skip_earshot:
        import tempfile
        paths = [os.path.join(ROOT, r['file']) for s in res.values() for r in s['lines'].values()]
        with tempfile.TemporaryDirectory() as td:
            pj, qj, oj = (os.path.join(td, n) for n in ('p.json', 'q.json', 'o.json'))
            json.dump(paths, open(pj, 'w'))
            json.dump([Q_M, Q_F, Q_ALTS], open(qj, 'w'))
            env = dict(os.environ, EARSHOT_THREADS=str(a.threads))
            subprocess.run([EARSHOT_PY, '-c', EARSHOT_SNIPPET, pj, qj, oj], check=True, env=env)
            es = json.load(open(oj))
        for s in res.values():
            for r in s['lines'].values():
                r['earshot'] = es[os.path.join(ROOT, r['file'])]

    # 候補ごとのまとめ
    for name, s in res.items():
        L = s['lines'].values()
        sp = {k: r['speech_sec'] for k, r in s['lines'].items()}
        sm = dict(file_total=round(sum(r['file_sec'] for r in L), 2), speech_total=round(sum(sp.values()), 2),
                  reel_sec=round(reel_len(sp), 2),
                  lufs_median=round(float(np.median([r['lufs'] for r in L if r['lufs'] is not None])), 1),
                  peak_max=max(r['peak_dbfs'] for r in L), n_fullscale=sum(r['n_fullscale'] for r in L),
                  floor_median=round(float(np.median([r['floor_dbfs'] for r in L if r['floor_dbfs'] is not None])), 1))
        if 'reading_ratio' in next(iter(L)):
            sm.update(reading_mean=round(float(np.mean([r['reading_ratio'] or 0 for r in L])), 3),
                      must_ng=[f'{k}:{w}' for k, r in s['lines'].items() for w in r['must_ng']])
        if 'earshot' in next(iter(L)):
            sm['p_real_mean'] = round(float(np.mean([r['earshot']['p_real'] for r in L])), 3)
            for q in [Q_M, Q_F] + Q_ALTS:
                sm['z:' + q] = round(float(np.mean([r['earshot']['z'][q] for r in L])), 2)
        s['summary'] = sm
    json.dump(res, open(os.path.join(VC, 'eval.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({n: s['summary'] for n, s in res.items()}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
