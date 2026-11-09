"""AivisSpeech の行ごとの候補（スタイル × アクセント × 表記 × 話速）を作り、Whisper で読みを確かめて表にする。

  python3 tools/v3/vo_aivis_pick.py <candidate> --lines 03,06 --styles Calm,Mid --accents hand,auto \
      [--speeds 1.12] [--texts '06=高さは、2500メートル。下からは、ほぼ点。キラーン！'] [--out DIR]
  → DIR/<key>/<variant>.wav と DIR/pick.json（各候補の 読み一致率・必須語・Whisper の平均確信度・長さ）
  python3 tools/v3/vo_aivis_pick.py collect <candidate> DIR meta.json [06=部分文字列 ...]
  → 行ごとに一番よいテイクを exports/v3/voice_cands/aivis_<candidate>/NN.wav に写し、candidate.json を書く

読みの基準：exports/v3/vo_B/moras.json（VOICEVOX 仮ナレーションのモーラ列 ＝ 想定の読み）。
必須語：EXPECT（固有名詞・数字）。Whisper の同音の書き分け（産直→3直、ゾウ→増）は読みで比べるので減点しない。
AivisSpeech の合成は毎回少しずつ違う（同じ設定でも別テイクになる）ので、--takes で複数作って選ぶ。
採点は「必須語を全部満たす → 読み一致率 → 平均確信度」の順。最後に聞いて決めるのは人。ここは読み違いのふるい。
"""
import argparse, itertools, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import vo_aivis as va                      # noqa: E402
import listen                              # noqa: E402

EXPECT = {
    '03': [['桂浜', 'かつらはま', 'カツラハマ', 'かつら浜'], ['10キロ', '十キロ', '10km']],
    '01': [['パンチ']],
    '04': [['速さ', '早さ', 'はやさ'], ['1130', '1,130', '千百三十'], ['旅客機']],
    '05': [['パンチ'], ['190', '百九十'], ['35', '三十五']],
    '06': [['高さ', 'たかさ'], ['2500', '2,500', '二千五百'], ['キラーン', 'きらーん', 'キラン']],
    '07': [['45', '四十五']],
    '08': [['仁淀川', 'によどがわ', 'ニヨドガワ'], ['河口']],
    '09': [['シーサイド'], ['まぜ', 'マゼ', 'MAZE', '混ぜ']],
    '10': [['ローソン']],
    '12': [['屋上'], ['避難タワー'], ['太平洋']],
}


def expect_kana(k):
    m = json.load(open(os.path.join(ROOT, 'exports', 'v3', 'vo_B', 'moras.json')))
    return ''.join(x[0] for x in m[k])


def score(path, k, threads):
    tr = listen.whisper(path, 'large-v3-turbo', threads)
    r = listen.check_reading(tr, expect_kana(k), EXPECT.get(k))
    must_ok = all(ok for _, ok in r.get('must', []))
    p = sum(w['p'] for w in tr['words']) / max(1, len(tr['words']))
    return dict(text=tr['text'], ratio=r.get('ratio'), must_ok=must_ok,
                must_ng=[a[0] for a, ok in r.get('must', []) if not ok], p=round(p, 3), diffs=r.get('diffs'))


def rank_key(v):
    return (v['must_ok'], v['ratio'] or 0, v['p'])


def collect(cand, pick_dir, out, prefer, meta):
    """pick.json から行ごとに一番よいテイクを選んで out/NN.wav に。prefer={key: 部分文字列}でその名前の候補に限る"""
    import shutil
    res = json.load(open(os.path.join(pick_dir, 'pick.json')))
    os.makedirs(out, exist_ok=True)
    chosen = {}
    for k in sorted(va.TEXT):
        for v in res.get(k, {}).values():       # 必須語は今の EXPECT で付け直す（Whisper の文字は保存済み）
            r = listen.check_reading(dict(text=v['text']), None, EXPECT.get(k))
            v['must_ok'] = all(ok for _, ok in r.get('must', []))
            v['must_ng'] = [al[0] for al, ok in r.get('must', []) if not ok]
        vs = res.get(k, {})
        if k in prefer:
            vs = {n: v for n, v in vs.items() if prefer[k] in n} or vs
        if not vs:
            raise SystemExit(f'no take for {k}')
        name, v = max(vs.items(), key=lambda kv: rank_key(kv[1]))
        shutil.copy(os.path.join(pick_dir, k, name + '.wav'), os.path.join(out, f'{k}.wav'))
        chosen[k] = dict(take=name, dur=v['dur'], whisper=v['text'], reading_ratio=v['ratio'], must_ok=v['must_ok'],
                         must_ng=v['must_ng'], whisper_p=v['p'], settings=v['settings'], n_takes=len(res.get(k, {})))
        print(f'{k} {name:30s} {v["dur"]:5.2f}s ratio {v["ratio"]} must {"OK" if v["must_ok"] else "NG"} {v["text"]}')
    tot = sum(c['dur'] for c in chosen.values())
    print(f'total {tot:.2f}s (VOICEVOX temp 33.78s)')
    meta = dict(meta, lines=chosen, total_sec=round(tot, 2))
    json.dump(meta, open(os.path.join(out, 'candidate.json'), 'w'), ensure_ascii=False, indent=1)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'collect':
        # python3 vo_aivis_pick.py collect <cand> <pick_dir> <meta.json> [key=部分文字列 ...]
        cand, pick_dir, metaf = sys.argv[2:5]
        prefer = dict(x.split('=', 1) for x in sys.argv[5:])
        collect(cand, pick_dir, os.path.join(ROOT, 'exports', 'v3', 'voice_cands', 'aivis_' + cand), prefer,
                json.load(open(metaf)))
        return
    ap = argparse.ArgumentParser()
    ap.add_argument('cand')
    ap.add_argument('--lines', required=True)
    ap.add_argument('--styles')
    ap.add_argument('--accents', default='hand,auto')
    ap.add_argument('--speeds')
    ap.add_argument('--intons', default='')
    ap.add_argument('--texts', action='append', default=[], help="key=別表記（複数可）")
    ap.add_argument('--out')
    ap.add_argument('--skip-base', action='store_true', help='既定の表記は作らず --texts の表記だけ（追いテイク用）')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--takes', type=int, default=1, help='同じ設定で何回合成するか（AivisSpeech は毎回少し違う＝テイク選び）')
    a = ap.parse_args()
    cands = va.load_cands()
    base = cands[a.cand]
    out = a.out or os.path.join('/tmp', 'vo_aivis_pick', a.cand)
    styles = a.styles.split(',') if a.styles else [base['style']]
    speeds = [float(x) for x in a.speeds.split(',')] if a.speeds else [base['speed']]
    intons = [float(x) for x in a.intons.split(',')] if a.intons else [None]
    alt_texts = {}
    for t in a.texts:
        kk, v = t.split('=', 1)
        alt_texts.setdefault(kk, []).append(v)
    res = {}
    pj = os.path.join(out, 'pick.json')
    if os.path.exists(pj):
        res = json.load(open(pj))
    for k in a.lines.split(','):
        texts = ([] if a.skip_base else [None]) + alt_texts.get(k, [])
        for st, acc, sp, it, (ti, tx), tk in itertools.product(styles, a.accents.split(','), speeds, intons,
                                                               [(i + (1 if a.skip_base else 0), t) for i, t in enumerate(texts)], range(a.takes)):
            pl = dict(base.get('per_line', {}).get(k, {}))
            pl.update(style=st, accent=acc, speed=sp)
            if it is not None:
                pl['inton'] = it
            if tx:
                pl['text'] = tx
            cfg = {**base, 'per_line': {k: pl}}
            name = f'{st}_{acc}_s{sp:g}' + (f'_i{it:g}' if it is not None else '') + (f'_t{ti}' if ti else '') + (f'_k{tk}' if a.takes > 1 else '')
            path = os.path.join(out, k, name + '.wav')
            try:
                prm, tim = va.synth_one(cfg, k, path)
            except Exception as e:
                print(k, name, 'FAILED', e, flush=True)
                continue
            sc = score(path, k, a.threads)
            res.setdefault(k, {})[name] = dict(settings=pl, dur=tim['dur'], **sc)
            print(f'{k} {name:28s} {tim["dur"]:5.2f}s ratio {sc["ratio"]} must {"OK" if sc["must_ok"] else "NG " + ",".join(sc["must_ng"])}'
                  f' p {sc["p"]}  {sc["text"]}', flush=True)
            json.dump(res, open(pj, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
