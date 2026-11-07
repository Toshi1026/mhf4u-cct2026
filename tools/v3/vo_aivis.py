"""案B ナレーションを AivisSpeech Engine で作る（edl/v3/SCRIPT_B_ig.md §3 の12行）。

  python3 tools/v3/vo_aivis.py <candidate> [--lines 01,06] [--out DIR]
  → exports/v3/voice_cands/aivis_<candidate>/NN.wav（48 kHz モノラル）、params.json、candidate.json

エンジン：localhost:10101（docker の ghcr.io/aivis-project/aivisspeech-engine:cpu-latest、コンテナ名 aivis）。
  docker run -d --name aivis --cpus 3 -p 10101:10101 \
    -v ~/.local/share/AivisSpeech-Engine:/home/user/.local/share/AivisSpeech-Engine-Dev \
    -v /root/.ccr/ca-bundle.crt:/etc/ccr-ca.crt:ro -e SSL_CERT_FILE=/etc/ccr-ca.crt -e REQUESTS_CA_BUNDLE=/etc/ccr-ca.crt \
    ghcr.io/aivis-project/aivisspeech-engine:cpu-latest
  （プロキシの CA を入れないと、起動時の BERT ダウンロードで SSL エラーになり exit 1 で落ちる）

AivisSpeech（Style-Bert-VITS2 JP-Extra）の要点
- audio_query の kana は「読み上げテキスト（漢字かな交じり）」で、BERT に入る＝抑揚の自然さはここで決まる。
- accent_phrases の読み・アクセントも合成に使われる。読みは TEXT（漢字の読み違いが出ない書き方）で決め、
  アクセントは vo_tune.py の手付けカナ（KANA）から、モーラが一致する句だけ差し替える（--accent hand）。
- intonationScale は「スタイルの感情の強さ」（VOICEVOX の抑揚とは別物）。ノーマルスタイルでは無視される。
- 句読点は句の中のモーラ（vowel=pau）として返り、長さの個別指定は効かない。間は pauseLengthScale（倍率）で行ごとに。
  長い間が欲しい所は TEXT 側で「。」にする。
"""
import argparse, io, json, os, sys, time, urllib.parse, urllib.request, wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
HOST = 'http://localhost:10101'

# 読み上げテキスト（BERT 用）。lines.json と同じ文で、読み違いが出ない表記にしてある。
#  桂浜・仁淀川・産直・河口は pyopenjtalk で正しく読める（audio_query で確認済み）。数字はアラビア数字で読ませる。
TEXT = {
    '01': 'アニメの悪役は、パンチ一発で、空のかなたへ。',
    '02': 'あれ、本当にやったら？',
    '03': 'たとえば、桂浜から殴って、10キロ先まで飛ばすなら。',
    '04': '速さは、時速1130キロ。旅客機より速い。',
    '05': 'パンチは、約、190トン。ゾウ、35頭分。',
    '06': '高さは、2500メートル。下からは、ほぼ点。キラーン。',
    '07': 'そして、45秒後。',
    '08': '落ちた先は、仁淀川の、河口の前の海。',
    '09': '目の前に建つのが、シーサイドまぜ。',
    '10': '1階は、ローソンと、産直。',
    '11': '2階は、海の見える、カフェ。',
    '12': '屋上と、避難タワーから、太平洋。',
}

# 候補ごとの設定。style は既定スタイル名、per_line で行ごとに上書き（style / speed / inton / tempo / pause_scale / text）。
CANDS = {
    'fumifumi': dict(model='71e72188-2726-4739-9aa9-39567396fb2a', speaker='fumifumi', style='ノーマル',
                     speed=1.04, inton=1.0, tempo=1.0, pause_scale=1.0, per_line={}),
    'aida': dict(model='47e53151-a378-46f3-abee-ce13aa07feb1', speaker='阿井田 茂', style='Calm',
                 speed=1.06, inton=1.0, tempo=1.0, pause_scale=1.0, per_line={}),
    'tanaka': dict(model='470a41e2-863f-4bf2-8a20-bf7cde0d8f7b', speaker='TANAKA', style='ノーマル',
                   speed=1.06, inton=1.0, tempo=1.0, pause_scale=1.0, per_line={}),
    'lux': dict(model='3dabea2b-cc10-41da-955b-4618ce84e716', speaker='Lux', style='ノーマル',
                speed=1.04, inton=1.0, tempo=1.0, pause_scale=1.0, per_line={}),
    'morioki': dict(model='baaae3c0-7b22-4605-8ba5-80c959b41a48', speaker='morioki', style='ノーマル',
                    speed=1.04, inton=1.0, tempo=1.0, pause_scale=1.0, per_line={}),
    'airi': dict(model='1b2830f4-8cf1-4184-a0d9-3a1bace3a844', speaker='みちのくあいり', style='標準',
                 speed=1.04, inton=1.0, tempo=1.0, pause_scale=1.0, per_line={}),
}
CANDS_FILE = os.path.join(HERE, 'vo_aivis_cands.json')   # 試聴で決めた設定（あれば CANDS を上書き）


def load_cands():
    cands = {k: dict(v) for k, v in CANDS.items()}
    if os.path.exists(CANDS_FILE):
        for k, v in json.load(open(CANDS_FILE)).items():
            if not k.startswith('_'):
                cands[k] = {**cands.get(k, {}), **v}
    return cands


def req(method, path, params=None, body=None):
    url = HOST + path + ('?' + urllib.parse.urlencode(params) if params else '')
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={'Content-Type': 'application/json'} if data else {})
    with urllib.request.urlopen(r, timeout=600) as f:
        return f.read()


def style_id(speaker, style):
    for s in json.loads(req('GET', '/speakers')):
        if s['name'] == speaker:
            for st in s['styles']:
                if st['name'] == style:
                    return st['id']
    raise SystemExit(f'style not found: {speaker}/{style}')


def mora_str(ap):
    # AivisSpeech は句読点（, . ?）を句の中のモーラ（vowel=pau）として返すので除く
    return ''.join(m['text'] for m in ap['moras'] if m['vowel'] != 'pau')


def hand_accents(key, sid, aps):
    """vo_tune.py の手付けカナのアクセント核を、読みが一致する句に移す（句の切り方が違う所は自動のまま）"""
    try:
        from vo_tune import KANA
    except Exception:
        return aps, []
    hand = json.loads(req('POST', '/accent_phrases', {'text': KANA[key], 'speaker': sid, 'is_kana': 'true'}))
    hmap = {}
    for h in hand:
        hmap.setdefault(mora_str(h), h['accent'])
    changed = []
    for ap in aps:
        s = mora_str(ap)
        if s in hmap and hmap[s] != ap['accent']:
            changed.append(f'{s}:{ap["accent"]}->{hmap[s]}')
            ap['accent'] = hmap[s]
    return aps, changed


def _wav_to_np(b):
    with wave.open(io.BytesIO(b)) as w:
        assert w.getframerate() == 48000 and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float32) / 32768


def _np_to_wav(x, path):
    y = np.clip(np.round(x * 32768), -32768, 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(48000); w.writeframes(y.tobytes())


def _env_db(x, hop=240):
    n = len(x) // hop
    e = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(e)                                    # 5 ms ごと


def _speech_bounds(x, rel=-40.0):
    e = _env_db(x)
    on = np.where(e > e.max() + rel)[0]
    return on[0] * 240, (on[-1] + 1) * 240


def _cut_last_syllable(x, rel=-40.0):
    """最後の1音節（「キラーンと。」の「と」など、句末の「ン」を言わせるために足した字）を切る。
    最後の有音フレームから 40〜350 ms 前の範囲で一番静かな所（「と」の t の閉鎖）で切り、10 ms でフェードアウト。"""
    e = _env_db(x)
    loud = np.where(e > e.max() + rel)[0]
    end = loud[-1]
    lo, hi = max(0, end - 70), max(1, end - 8)
    cut = (lo + int(np.argmin(e[lo:hi]))) * 240 + 120
    y = x[:cut].copy()
    f = min(len(y), 480)
    y[-f:] *= np.linspace(1, 0, f)
    return y


def synth_query(cfg, k, pl, text, style, acc):
    sid = style_id(cfg['speaker'], style)
    q = json.loads(req('POST', '/audio_query', {'text': text, 'speaker': sid}))
    changed = []
    if acc == 'hand':
        q['accent_phrases'], changed = hand_accents(k, sid, q['accent_phrases'])
    q.update(kana=text, speedScale=pl.get('speed', cfg['speed']), intonationScale=pl.get('inton', cfg['inton']),
             tempoDynamicsScale=pl.get('tempo', cfg['tempo']), pitchScale=0.0, volumeScale=1.0,
             prePhonemeLength=0.08, postPhonemeLength=0.20, pauseLengthScale=pl.get('pause_scale', cfg['pause_scale']),
             outputSamplingRate=48000, outputStereo=False)
    return sid, q, changed, req('POST', '/synthesis', {'speaker': sid}, q)


def synth_one(cfg, k, path, accent='hand'):
    """1行を合成して path に書く。cfg は候補の設定（per_line[k] があれば上書き）。戻り値 (params, timing)

    per_line[k]['parts'] があれば、部分ごとに合成してつなぐ（キラーンなど、決め台詞だけ別スタイル・別テイクにする用）：
      parts=[{text, style?, accent?, cut_last?}, ...], gaps=[秒, ...]（部分と部分の間の無音）
      cut_last=true：text の最後に足した1音節（例「キラーンと。」の「と」）を切る。句末の「ン」が消える話者用。
    """
    pl = dict(cfg.get('per_line', {}).get(k, {}))
    style = pl.get('style', cfg['style'])
    acc = pl.get('accent', accent)
    t0 = time.time()
    if not pl.get('parts'):
        text = pl.get('text', TEXT[k])
        sid, q, changed, wav = synth_query(cfg, k, pl, text, style, acc)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        open(path, 'wb').write(wav)
        x = _wav_to_np(wav)
        prm = dict(style=style, style_id=sid, accent=acc, accent_changes=changed, query=q)
    else:
        segs, prm = [], dict(style=style, accent=acc, parts=[])
        for pi, part in enumerate(pl['parts']):
            ppl = {**pl, **{kk: v for kk, v in part.items() if kk in ('speed', 'inton', 'tempo', 'pause_scale')}}
            st = part.get('style', style)
            sid, q, changed, wav = synth_query(cfg, k, ppl, part['text'], st, part.get('accent', acc))
            x = _wav_to_np(wav)
            if part.get('cut_last'):
                x = _cut_last_syllable(x)
            a, b = _speech_bounds(x)
            segs.append(x[max(0, a - 240):min(len(x), b + 480)])
            prm['parts'].append(dict(text=part['text'], style=st, style_id=sid, cut_last=bool(part.get('cut_last')),
                                     accent_changes=changed, query=q))
        gaps = pl.get('gaps', [0.4] * (len(segs) - 1))
        out = [np.zeros(int(0.08 * 48000), np.float32)]
        for i, sgm in enumerate(segs):
            out.append(sgm)
            out.append(np.zeros(int((gaps[i] if i < len(gaps) else 0.20) * 48000), np.float32))
        x = np.concatenate(out)
        prm['gaps'] = gaps
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        _np_to_wav(x, path)
    dt = time.time() - t0
    dur = len(x) / 48000
    return prm, dict(dur=round(dur, 3), synth_sec=round(dt, 2), rtf=round(dt / dur, 2))


def render(name, cfg, keys, out, accent='hand'):
    params, timing = {}, {}
    for k in keys:
        params[k], timing[k] = synth_one(cfg, k, os.path.join(out, f'{k}.wav'), accent)
        print(f'{name} {k} {timing[k]["dur"]:5.2f}s synth {timing[k]["synth_sec"]:5.2f}s  '
              f'{params[k]["style"]} {"; ".join(params[k]["accent_changes"])}', flush=True)
    return params, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cand')
    ap.add_argument('--lines', default=','.join(TEXT))
    ap.add_argument('--out')
    ap.add_argument('--accent', default='hand', choices=['hand', 'auto'])
    ap.add_argument('--set', action='append', default=[], help='key=value で設定を一時上書き（speed=1.08 など）')
    a = ap.parse_args()
    cands = load_cands()
    cfg = dict(cands[a.cand])
    for s in a.set:
        k, v = s.split('=', 1)
        cfg[k] = json.loads(v) if v[:1] in '0123456789-[{"' else v
    out = a.out or os.path.join(ROOT, 'exports', 'v3', 'voice_cands', 'aivis_' + a.cand)
    keys = a.lines.split(',')
    params, timing = render(a.cand, cfg, keys, out, a.accent)
    pj = os.path.join(out, 'params.json')
    old = json.load(open(pj)) if os.path.exists(pj) else {}
    old.update(params)
    json.dump(old, open(pj, 'w'), ensure_ascii=False, indent=1)
    tj = os.path.join(out, 'timing.json')
    oldt = json.load(open(tj)) if os.path.exists(tj) else {}
    oldt.update(timing)
    json.dump(dict(sorted(oldt.items())), open(tj, 'w'), indent=1)
    if all(os.path.exists(os.path.join(out, f'{k}.wav')) for k in TEXT):
        tot = sum(v['dur'] for v in oldt.values())
        print(f'total {tot:.2f}s (VOICEVOX temp: 33.78s)')


if __name__ == '__main__':
    main()
