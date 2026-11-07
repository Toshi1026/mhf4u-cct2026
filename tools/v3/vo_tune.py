"""案B ナレーションの VOICEVOX 手調整版を作る（edl/v3/SCRIPT_B_ig.md §3 の12行）。

  python3 tools/v3/vo_tune.py <speaker_id> <out_dir> [--lines 01,04] [--plain] [--name 雀松朱司]

- 読みとアクセントは AquesTalk 風カナ（is_kana=true）で全部指定する（' ＝アクセント核、/ ＝区切り（ポーズなし）、、＝ポーズ）。
  平板は句の最後に ' を置く（VOICEVOX は accent=句のモーラ数 を平板として扱う）。
- 句ごとの間（pause_mora）、文末の下げ、数字の軽い強調、「キラーン」の前の一拍と遊び、を audio_query に直接書き込む。
- --plain は自動の間・抑揚の手調整をせず、アクセントだけ直した状態で出す（話者の聞き比べ用）。
- 出力：48 kHz モノラル NN.wav、params.json（実際に送った audio_query 一式）。
エンジン：localhost:50021（docker の voicevox/voicevox_engine:cpu-0.26.1、コンテナ名 vv）。
"""
import argparse, json, os, sys, urllib.parse, urllib.request

HOST = 'http://localhost:50021'

# ---------------------------------------------------------------- 原稿（カナ・アクセント）
# アクセントの根拠（東京式。NHK 日本語発音アクセント辞典の型に合わせた）
#  速さ・高さ＝1型（ハ'ヤサ / タ'カサ）、約＝1型、落ちた＝オ'チタ、建つ＝タ'ツ、二階・一階・屋上・河口・産直＝平板、
#  仁淀川＝〜川の複合（多摩川・隅田川型）で ニヨド'ガワ、桂浜＝カツラ'ハマ（〜浜の複合）、旅客機＝リョカ'クキ、
#  十キロ＝ジュッキ'ロ、千百＝センヒャ'ク、三十キロ＝サンジュッキ'ロ、百九十トン＝ヒャクキュウジュ'ットン、
#  三十五頭分＝サンジュウゴト'ウブン（〜分 は直前に核）、四十五秒後＝ヨンジュウゴビョ'オゴ（〜後 は直前に核）、
#  二千五百メートル＝ニセ'ン/ゴヒャクメ'エトル、Seaside MAZE＝シイサイド'マゼ（短い後部要素の複合名）
KANA = {
    '01': "ア'ニメノ/アクヤクワ'、パ'ンチ/イッパツデ'、ソ'ラノ/カ'ナタエ",
    '02': "アレ'、ホントオニ'/ヤッタラ'？",
    '03': "タト'エバ、カツラ'ハマカラ/ナグ'ッテ、ジュッキ'ロサキマデ/トバスナラ'",
    '04': "ハ'ヤサワ、ジ'ソク/センヒャ'ク/サンジュッキ'ロ、リョカ'クキヨリ/ハヤ'イ",
    '05': "パ'ンチワ、ヤ'ク、ヒャクキュウジュ'ットン、ゾ'オ、サンジュウゴト'ウブン",
    '06': "タ'カサワ、ニセ'ン/ゴヒャクメ'エトル、シタカラワ'、ホ'ボ/テン'、キ'ラアン",
    '07': "ソシテ'、ヨンジュウゴビョ'オゴ",
    '08': "オ'チタ/サキワ'、ニヨド'ガワノ、カコオノ'/マ'エノ/ウ'ミ",
    '09': "メノマ'エニ/タ'ツノガ、シイサイド'マゼ",
    '10': "イッカイワ'、ロ'オソント、サンチョク'",
    '11': "ニカイワ'、ウ'ミノ/ミエ'ル、カ'フェ",
    '12': "オクジョオト'、ヒナンタ'ワアカラ、タイヘ'エヨオ",
}

# ---------------------------------------------------------------- 手調整
# pauses: 各「、」の後の間（秒、speedScale をかける前）。 'S' 付きは文の切れ目（その前の句を文末として下げる）。
# emph: 軽く立てる句の番号（数字など）。 play: 遊ぶ句（キラーン）。
TUNE = {
    '01': dict(pauses=[0.20, 0.16]),
    '02': dict(pauses=[0.34]),
    '03': dict(pauses=[0.24, 0.16], emph=[4]),
    '04': dict(pauses=[0.16, ('S', 0.42)], emph=[2, 3]),
    '05': dict(pauses=[0.16, 0.10, ('S', 0.42), 0.14], emph=[2, 4]),
    '06': dict(pauses=[0.16, ('S', 0.40), 0.12, ('S', 0.58)], emph=[1, 2], play=[6]),
    '07': dict(pauses=[0.36], emph=[1]),
    '08': dict(pauses=[0.16, 0.12], emph=[2]),
    '09': dict(pauses=[0.22], emph=[2]),
    '10': dict(pauses=[0.14, 0.14]),
    '11': dict(pauses=[0.14, 0.18]),
    '12': dict(pauses=[0.14, 0.20]),
}

# 話者ごとの基本設定（試聴・測定で決めたもの）。ない話者は DEFAULT。
DEFAULT = dict(speed=1.08, pitch=0.0, inton=1.12, end_drop=0.045, end_last=0.035, emph_up=0.025,
               play_up=0.13, play_inton=1.6, play_len=1.35)
SPEAKER_BASE = {}


def post(path, params, body=None):
    url = f"{HOST}{path}?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method='POST',
                                 headers={'Content-Type': 'application/json'} if data else {})
    with urllib.request.urlopen(req) as r:
        return r.read()


def build_query(key, speaker, base, plain=False):
    aps = json.loads(post('/accent_phrases', {'text': KANA[key], 'speaker': speaker, 'is_kana': 'true'}))
    q = json.loads(post('/audio_query', {'text': 'あ', 'speaker': speaker}))
    q['accent_phrases'] = aps
    q.update(speedScale=base['speed'], pitchScale=base['pitch'], intonationScale=base['inton'],
             volumeScale=1.0, prePhonemeLength=0.08, postPhonemeLength=0.18, outputSamplingRate=48000,
             outputStereo=False)
    edits = []
    if plain:
        return q, edits
    t = TUNE[key]
    # ---- 間（pause_mora）と文の切れ目
    sent_end = {len(aps) - 1}
    pi = 0
    for i, ap in enumerate(aps):
        if ap['pause_mora'] is None:
            continue
        p = t['pauses'][pi]; pi += 1
        if isinstance(p, tuple):
            sent_end.add(i); p = p[1]
        old = ap['pause_mora']['vowel_length']
        ap['pause_mora']['vowel_length'] = p
        edits.append(f"pause after phrase {i}: {old:.2f}->{p:.2f}s")
    # ---- 数字などの軽い強調（句全体を少し上げ、母音を 4% 伸ばす）
    for i in t.get('emph', []):
        for m in aps[i]['moras']:
            if m['pitch'] > 0:
                m['pitch'] += base['emph_up']
            m['vowel_length'] *= 1.04
        edits.append(f"emph phrase {i}: pitch +{base['emph_up']}, vowel x1.04")
    # ---- 文末を下げる（句全体 -end_drop、最後の2モーラはさらに -end_last を段階的に）
    for i in sorted(sent_end):
        ap = aps[i]
        if ap['is_interrogative'] or i in t.get('play', []):
            continue
        ms = ap['moras']
        for m in ms:
            if m['pitch'] > 0:
                m['pitch'] -= base['end_drop']
        for j, m in enumerate(ms[-2:]):
            if m['pitch'] > 0:
                m['pitch'] -= base['end_last'] * (j + 1) / 2
        edits.append(f"sentence end phrase {i}: pitch -{base['end_drop']} (last 2 moras extra -{base['end_last']/2:.3f}/-{base['end_last']:.3f})")
    # ---- キラーン：一拍おいて（pause は TUNE 側）、高く・抑揚を大きく・ラーを伸ばす
    for i in t.get('play', []):
        ms = aps[i]['moras']
        vals = [m['pitch'] for m in ms if m['pitch'] > 0]
        mean = sum(vals) / len(vals)
        for m in ms:
            if m['pitch'] > 0:
                m['pitch'] = mean + (m['pitch'] - mean) * base['play_inton'] + base['play_up']
        # キ ラ ア ン：ラ・ア を伸ばし、ン を少し伸ばす
        for j, m in enumerate(ms):
            if j in (1, 2):
                m['vowel_length'] *= base['play_len']
            if j == len(ms) - 1:
                m['vowel_length'] *= 1.2
        edits.append(f"play phrase {i}: pitch +{base['play_up']}, local intonation x{base['play_inton']}, ラー x{base['play_len']}, ン x1.2")
    return q, edits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('speaker', type=int)
    ap.add_argument('out')
    ap.add_argument('--lines', default=','.join(KANA))
    ap.add_argument('--plain', action='store_true')
    ap.add_argument('--set', default='', help='base overrides, e.g. speed=1.1,inton=1.2')
    a = ap.parse_args()
    base = dict(DEFAULT, **SPEAKER_BASE.get(a.speaker, {}))
    for kv in filter(None, a.set.split(',')):
        k, v = kv.split('='); base[k] = float(v)
    os.makedirs(a.out, exist_ok=True)
    params = {'speaker': a.speaker, 'base': base, 'plain': a.plain, 'lines': {}}
    for key in a.lines.split(','):
        q, edits = build_query(key, a.speaker, base, a.plain)
        wav = post('/synthesis', {'speaker': a.speaker}, q)
        open(os.path.join(a.out, f'{key}.wav'), 'wb').write(wav)
        params['lines'][key] = {'kana': KANA[key], 'edits': edits, 'audio_query': q}
        print(key, 'ok', file=sys.stderr)
    json.dump(params, open(os.path.join(a.out, 'params.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
