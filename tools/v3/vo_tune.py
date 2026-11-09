"""案B ナレーションの VOICEVOX 手調整版を作る（edl/v3/SCRIPT_B_ig.md §3 の12行）。

  python3 tools/v3/vo_tune.py <speaker_id> <out_dir> [--lines 01,04] [--plain] [--name 雀松朱司]

- 読みとアクセントは AquesTalk 風カナ（is_kana=true）で全部指定する（' ＝アクセント核、/ ＝区切り（ポーズなし）、、＝ポーズ）。
  平板は句の最後に ' を置く（VOICEVOX は accent=句のモーラ数 を平板として扱う）。
- 句ごとの間（pause_mora）、文末の下げ、数字の軽い強調、「キラーン」の前の一拍と遊び、を audio_query に直接書き込む。
- --plain は間・強調・文末・キラーンの手調整をせず、カナのアクセント指定と核の形の補正だけで出す（話者の聞き比べ用）。
- 出力：48 kHz モノラル NN.wav、params.json（実際に送った audio_query 一式）。
エンジン：localhost:50021（docker の voicevox/voicevox_engine:cpu-0.26.1、コンテナ名 vv）。
"""
import argparse, json, os, sys, urllib.parse, urllib.request

HOST = 'http://localhost:50021'

# ---------------------------------------------------------------- 原稿（カナ・アクセント）
# アクセントの根拠（東京式。NHK 日本語発音アクセント辞典の型に合わせた）
#  速さ・高さ＝1型（ハ'ヤサ / タ'カサ）、約＝1型、落ちた＝オ'チタ、建つ＝タ'ツ、二階・一階・屋上・河口・産直＝平板、
#  仁淀川＝〜川の複合（多摩川・隅田川型）で ニヨド'ガワ、桂浜＝カツラハマ'（4モーラの〜浜地名は平板：横浜・白浜・長浜型）、旅客機＝リョカク'キ（〜機は直前に核：扇風機・戦闘機型）、
#  十キロ＝ジュッキ'ロ、千百＝センヒャ'ク、三十キロ＝サンジュッキ'ロ、百九十トン＝ヒャクキュウジュ'ットン、
#  三十五頭分＝サンジュウゴト'ウブン（〜分 は直前に核）、四十五秒後＝ヨンジュウゴビョ'オゴ（〜後 は直前に核）、
#  二千五百メートル＝ニセ'ン/ゴヒャクメ'エトル、Seaside MAZE＝シイサイド'マゼ（短い後部要素の複合名）
KANA = {
    '01': "ア'ニメノ/アクヤクワ'、パ'ンチ/イッパツデ'、ソ'ラノ/カ'ナタエ",
    '02': "アレ'、ホントオニ'/ヤッタラ'？",
    '03': "タト'エバ、カツラハマカラ'/ナグ'ッテ、ジュッキ'ロサキマデ/トバスナラ'",
    '04': "ハ'ヤサワ、ジ'ソク/センヒャ'ク/サンジュッキ'ロ、リョカク'キヨリ/ハヤ'イ",
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
               play_up=0.15, play_fall=0.24, play_len=1.35, acc_drop=0.20, acc_rise=0.10, pre=0.03, post=0.06)
# 53 麒ヶ島宗麟（渋い年配の男声）：素の話速が遅めなので speed 1.10 で仮ナレと同程度の尺に、抑揚は控えめ（素の音高レンジが広いので inton を下げる）。
# 21 剣崎雌雄（落ち着いた男声）：標準。
SPEAKER_BASE = {53: dict(speed=1.10, inton=1.05), 21: dict(speed=1.08, inton=1.10)}


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
             volumeScale=1.0, prePhonemeLength=base['pre'], postPhonemeLength=base['post'], outputSamplingRate=48000,
             outputStereo=False)
    edits = []
    # ---- アクセントの形をはっきりさせる（モデルの予測は核の後で下がりきらない・1型の頭が上がらないことがある）
    #  核のモーラを句頭からの最高に、核の次の有声モーラを核より acc_drop 以上低く、以降はそれ以下に。
    #  2型以上・平板は1モーラ目を2モーラ目より acc_rise 低く（東京式の句頭の上昇）。pitch は log F0。
    for i, ap in enumerate(aps):
        ms = ap['moras']; a = ap['accent']; n = len(ms); fixed = []
        voiced = [j for j, m in enumerate(ms) if m['pitch'] > 0]
        if a >= 2 and len(voiced) >= 2 and voiced[0] == 0 and voiced[1] == 1:
            lim = ms[1]['pitch'] - base['acc_rise']
            if ms[0]['pitch'] > lim:
                fixed.append(f"rise m0 {ms[0]['pitch']:.2f}->{lim:.2f}"); ms[0]['pitch'] = lim
        if a == 1 and len(voiced) >= 2 and voiced[0] == 0 and ms[voiced[1]]['pitch'] > ms[0]['pitch']:
            mid = (ms[0]['pitch'] + ms[voiced[1]]['pitch']) / 2   # 1型の頭：上げすぎないよう次との中間まで
            fixed.append(f"head m0 {ms[0]['pitch']:.2f}->{mid:.2f}"); ms[0]['pitch'] = mid
        if a < n and ms[a - 1]['pitch'] > 0:
            k = a - 1
            hi = max(m['pitch'] for m in ms[:k + 1] if m['pitch'] > 0)
            if ms[k]['pitch'] < hi:
                fixed.append(f"nucleus m{k} {ms[k]['pitch']:.2f}->{hi:.2f}"); ms[k]['pitch'] = hi
            after = [j for j in range(k + 1, n) if ms[j]['pitch'] > 0]
            if after:
                j0 = after[0]; lim = ms[k]['pitch'] - base['acc_drop']
                if ms[j0]['pitch'] > lim:
                    fixed.append(f"fall m{j0} {ms[j0]['pitch']:.2f}->{lim:.2f}"); ms[j0]['pitch'] = lim
                for j in after[1:]:
                    if ms[j]['pitch'] > ms[j0]['pitch']:
                        fixed.append(f"tail m{j} {ms[j]['pitch']:.2f}->{ms[j0]['pitch']:.2f}"); ms[j]['pitch'] = ms[j0]['pitch']
        if fixed:
            edits.append(f"accent phrase {i} {''.join(m['text'] for m in ms)} (accent {a}): " + ', '.join(fixed))
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
    # ---- キラーン：一拍おいて（pause は TUNE 側）、高い所で「キ・ラー」を平らに張って「ン」で落とす（光って消える形）。
    #  モデル任せの 1型の下降（キ'ラーン）だと溜め息のように聞こえやすいので、音高を直接書く。
    #  top＝その行の有声モーラの平均と最高の中間 + play_up。キ top-0.04 / ラ top+0.03 / ア(ー) top / ン top-play_fall。
    all_v = [m['pitch'] for ap in aps for m in ap['moras'] if m['pitch'] > 0]
    for i in t.get('play', []):
        ms = aps[i]['moras']
        top = (sum(all_v) / len(all_v) + max(all_v)) / 2 + base['play_up']
        shape = [-0.04, 0.03, 0.0, -base['play_fall']]
        for j, m in enumerate(ms):
            if m['pitch'] > 0:
                m['pitch'] = top + shape[min(j, 3)]
            if j in (1, 2):
                m['vowel_length'] *= base['play_len']
            if j == len(ms) - 1:
                m['vowel_length'] *= 1.25
        edits.append(f"play phrase {i} キラーン: pitch set to キ{top-0.04:.2f} ラ{top+0.03:.2f} ー{top:.2f} ン{top-base['play_fall']:.2f} "
                     f"(line mean/max midpoint +{base['play_up']}), ラー x{base['play_len']}, ン x1.25")
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
