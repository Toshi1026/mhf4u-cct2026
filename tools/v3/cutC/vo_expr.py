"""案C ナレーションの「抑揚」版（2026-11-08 クライアント「抑揚が足りない。ただ棒読みで読んでるだけみたい」への修正）。

声は同じ AivisSpeech 阿井田 茂（ACML 1.0）のまま、行ごとにスタイル（ノーマル/Calm/Far/Heavy/Mid/Shout/Surprise）、
intonationScale（＝スタイルの強さ）、tempoDynamicsScale、話速、表記（「！」「？」「、」「……」）を変えて
テイクをたくさん作り、読み（Whisper）・長さ（今の版の −15〜+15 %）・抑揚の数値（vo_metrics.py）で選ぶ。

  python3 tools/v3/cutC/vo_expr.py takes [--lines 04,05] [--seeds 1]
      → exports/v3/voice_cands/C_expr/<NN>/<name>.wav と expr.json（読み・長さ・抑揚の数値）
  python3 tools/v3/cutC/vo_expr.py verify [--top 6]                  # 抑揚の点数の上位から Whisper で読みを確認
  python3 tools/v3/cutC/vo_expr.py more 04=<name> ...  [--seeds 4]   # 選んだ設定でテイクを増やす
  python3 tools/v3/cutC/vo_expr.py rank [--lines 04]                  # 行ごとの上位
  <venv>/bin/python tools/v3/cutC/vo_expr.py collect 01=<name> ...    # 全13行を選んで exports/v3/vo_C/ に書く（SHAPE の
                                                                      # PSOLA を最後にかけるので parselmouth の入った venv で）
  python3 tools/v3/cutC/vo_expr.py ab                                 # 行ごとに 旧 → 新 の試聴ファイル
  その後 <venv>/bin/python -I tools/v3/cutC/vo_align.py run exports/v3/vo_C で moras.json。

旧（平らな）版は exports/v3/vo_C/_flat_backup/ に置いてある（vo_c.py で作った Calm・話速 1.12 一律の版）。

2026-11-08 修正5（クライアント：「アニメなら」をゆっくり・間を取って説明を足す／キラーンはナレーションでなく効果音／
1,400トンに「威力」を入れる）で 01・03・04 の文が変わった。この3行は --fix5 を付けて別の置き場で作り直す：
  python3 tools/v3/cutC/vo_expr.py takes --fix5 [--lines 01]        → exports/v3/voice_cands/C_fix5/<NN>/
  python3 tools/v3/cutC/vo_expr.py verify --fix5 / rank --fix5
  <venv>/bin/python tools/v3/cutC/vo_expr.py collect --fix5 --partial 01=<name> 03=<name> 04=<name>
      → 選んだ行だけ vo_C/ を差し替え（前の版は vo_C/_prev_fix5/ に退避）、json は他の行を残してマージ
長さの範囲は旧版の比（文が違うので使えない）ではなく、TARGET の秒数で見る。
"""
import argparse
import itertools
import json
import os
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(V3, '..', '..'))
sys.path.insert(0, V3)
sys.path.insert(0, HERE)
import vo_aivis as va      # noqa: E402
import vo_c                # noqa: E402
import vo_metrics as vm    # noqa: E402

OUT = vo_c.OUT
FLAT = os.path.join(OUT, '_flat_backup')
TAKES = os.path.join(ROOT, 'exports', 'v3', 'voice_cands', 'C_expr')
SR = vo_c.SR
LINES = vo_c.LINES
OLD_DUR = json.load(open(os.path.join(FLAT, 'durations.json')))
DUR_LO, DUR_HI = 0.85, 1.15
# 修正5で文が変わった行は、旧版との比でなく秒数（トリム後）の範囲で見る
TARGET = {'01': (2.6, 4.4), '03': (1.6, 3.0), '04': (1.6, 2.8), '08': (1.3, 2.5)}
FIX5 = ('01', '03', '04')
PREV5 = os.path.join(OUT, '_prev_fix5')
if '--fix5' in sys.argv:                   # 修正5のテイクは別の置き場（旧テイクの expr.json と名前がぶつからないように）
    sys.argv.remove('--fix5')
    TAKES = os.path.join(ROOT, 'exports', 'v3', 'voice_cands', 'C_fix5')

S_N, S_C, S_F, S_H, S_M, S_SH, S_SU = 'ノーマル', 'Calm', 'Far', 'Heavy', 'Mid', 'Shout', 'Surprise'

# 行ごとの狙い（台本の演出）と、試す表記・スタイル・強さ・テンポ・話速。
#   texts：表記（parts 形式も可。parts の各要素に style を書けば、その部分だけ別スタイル）
#   styles × intons × tempos × speeds の全組み合わせを作る（ノーマルは intonationScale が効かないので 1.0 だけ）
PLAN = {
    # 2026-11-08 修正5：01・03・04 は文が変わった（旧 01「アニメなら、キラーン」03「現実だと、こうなる」04「約1,400トン」）。
    # キラーンは効果音（効果音ラボ kira2）だけにして、ナレーションでは説明を足す。01 はゆっくり（話速 0.95〜1.0）・おどけて、
    # 読点で自然な間。03 は「……」で本当に間をあけてから「こうなる」。04 は「威力」を必ず入れて数字を言い切る
    '01': dict(aim='ゆっくり・おどけて説明（アニメなら、悪役は、パンチ一発で、空のかなたへ）',
               texts=[dict(text='アニメなら、悪役は、パンチ一発で、空のかなたへ。'),
                      dict(text='アニメなら？悪役は、パンチ一発で、空のかなたへ！'),
                      dict(text='アニメなら、悪役は、パンチ一発で、空のかなたへ。', pause_scale=1.5),
                      dict(parts=[dict(text='アニメなら、'), dict(text='悪役は、パンチ一発で、'),
                                  dict(text='空のかなたへ！', style=S_SU)], gaps=[0.24, 0.10]),
                      # 「アニメなら」だけさらにゆっくり（部分ごとの話速）＋そのあとの間を長めに
                      dict(parts=[dict(text='アニメなら、', speed=0.85), dict(text='悪役は、パンチ一発で、'),
                                  dict(text='空のかなたへ！', style=S_SU)], gaps=[0.30, 0.10]),
                      dict(parts=[dict(text='アニメなら、', speed=0.9), dict(text='悪役は、パンチ一発で、'),
                                  dict(text='空のかなたへ！', style=S_SU)], gaps=[0.28, 0.10]),
                      # 旧 01 の「アニメなら？」（話速 1.12）は 0.64 秒。t4/t5 でも 0.65 秒と変わらなかったので、さらに落とす
                      dict(parts=[dict(text='アニメなら、', speed=0.75), dict(text='悪役は、パンチ一発で、'),
                                  dict(text='空のかなたへ！', style=S_SU)], gaps=[0.32, 0.10]),
                      dict(parts=[dict(text='アニメなら？', speed=0.72), dict(text='悪役は、パンチ一発で、'),
                                  dict(text='空のかなたへ！', style=S_SU)], gaps=[0.32, 0.10]),
                      dict(parts=[dict(text='アニメなら〜、', speed=0.85), dict(text='悪役は、パンチ一発で、'),
                                  dict(text='空のかなたへ！', style=S_SU)], gaps=[0.30, 0.10])],
               styles=[S_M, S_SU, S_N], intons=[1.0, 1.5], tempos=[1.0, 1.4], speeds=[0.95, 1.0]),
    '03': dict(aim='不穏に溜める（でも、現実だと……こうなる）',
               texts=[dict(text='でも、げんじつだと……こうなる。'),
                      dict(text='でも、げんじつだと……こうなる。', pause_scale=1.6),
                      dict(parts=[dict(text='でも、げんじつだと……'), dict(text='こうなる。')], gaps=[0.32]),
                      dict(parts=[dict(text='でも、げんじつだと、'), dict(text='こうなる。')], gaps=[0.42])],
               styles=[S_H, S_F, S_C], intons=[1.0, 1.5], tempos=[1.0, 1.4], speeds=[1.0, 1.06]),
    '04': dict(aim='「威力」を入れて数字を言い切る（パンチの威力は、約1,400トン）',
               texts=[dict(text='パンチのいりょくは、やく、せんよんひゃく、トン！'),
                      dict(parts=[dict(text='パンチのいりょくは、'), dict(text='やく、せんよんひゃく、トン！', style=S_SU)],
                           gaps=[0.10]),
                      dict(parts=[dict(text='パンチのいりょくは、'), dict(text='やく、'),
                                  dict(text='せんよんひゃく、トン！', style=S_SU)], gaps=[0.08, 0.06]),
                      dict(parts=[dict(text='パンチのいりょくは、', style=S_M), dict(text='やく！せんよんひゃくトン！')],
                           gaps=[0.10])],
               styles=[S_SU, S_SH, S_M, S_H], intons=[1.0, 1.5], tempos=[1.0, 1.4], speeds=[1.12]),
    '05': dict(aim='数字を言い切る（ゾウ250頭分）',
               texts=[dict(text='ゾウ、にひゃくごじゅっとうぶん！'), dict(text='象、にひゃくごじゅっとうぶん！'),
                      dict(text='ゾウ！にひゃくごじゅっとうぶん！')],
               styles=[S_SU, S_SH, S_M, S_H], intons=[1.0, 1.5], tempos=[1.0, 1.4], speeds=[1.06]),
    '06': dict(aim='数字を言い切る（マッハ6.7）。「マハ」で合成して h を伸ばす（vo_c.hstretch と同じ）',
               texts=[dict(text='マハ、ろくてんなな！', hstretch=0.08), dict(text='マハ！ろくてんなな！', hstretch=0.08),
                      dict(text='マハ、ろくてん、なな！', hstretch=0.08), dict(text='マハ、6.7！', hstretch=0.08)],
               styles=[S_SU, S_SH, S_M, S_H], intons=[1.0, 1.5], tempos=[1.0, 1.4], speeds=[1.12]),
    '07': dict(aim='数字を言い切る（高さ、132キロ）',
               texts=[dict(text='たかさ、ひゃくさんじゅうにキロ！'), dict(text='たかさ！ひゃくさんじゅうにキロ！'),
                      dict(parts=[dict(text='たかさ、'), dict(text='ひゃくさんじゅうにキロ！', style=S_SU)], gaps=[0.08]),
                      dict(parts=[dict(text='たかさ、'), dict(text='ひゃく、さんじゅうにキロ！', style=S_SU)], gaps=[0.08]),
                      dict(text='たかさ、ひゃく、さんじゅうに、キロ！')],
               styles=[S_SU, S_SH, S_M, S_H], intons=[1.0, 1.5], tempos=[1.0, 1.4], speeds=[1.12]),
    # 2026-11-09 クライアント：「宇宙です」→「宇宙まで来ちゃいました」（ちょっと楽しげに・やっちゃった感）
    '08': dict(aim='ひと呼吸おいて、楽しげに（……宇宙まで、来ちゃいました）',
               texts=[dict(text='……うちゅうまで、きちゃいました！'), dict(text='……うちゅうまで、きちゃいました。'),
                      dict(text='うちゅうまで、きちゃいました！'),
                      dict(parts=[dict(text='うちゅうまで、', style=S_SU), dict(text='きちゃいました！')], gaps=[0.06])],
               styles=[S_SU, S_M, S_SH, S_N], intons=[1.0, 1.5], tempos=[1.0, 1.4], speeds=[1.08]),
    '09': dict(aim='サスペンス（そして、5分半後）',
               texts=[dict(text='そして、ごふんはんご。'), dict(text='そして……ごふんはんご。'),
                      dict(text='そして、ごふんはんご…。')],
               styles=[S_H, S_F, S_M, S_C], intons=[1.0, 1.5], tempos=[1.0, 1.4], speeds=[1.06, 1.12]),
    '11': dict(aim='あたたかく（落ちたのは、太平洋と仁淀川が出会う海）',
               texts=[dict(text='落ちたのは、太平洋と、仁淀川が、出会う海。'),
                      dict(text='落ちたのは、太平洋と、によどがわが出会う海。')],
               styles=[S_M, S_N, S_F, S_SU], intons=[1.0, 1.5], tempos=[1.0, 1.4], speeds=[1.12]),
    '12': dict(aim='あたたかく誘う（目の前に、Seaside MAZE）',
               texts=[dict(text='目の前に、シーサイド・まぜ！'), dict(text='目の前に、シーサイド・まぜ。'),
                      dict(text='目の前に！シーサイド・まぜ。'),
                      dict(parts=[dict(text='目の前に、'), dict(text='シーサイド・まぜ！', style=S_SU)], gaps=[0.06]),
                      dict(parts=[dict(text='目の前に、'), dict(text='シーサイド・まぜ。', style=S_SU)], gaps=[0.06])],
               styles=[S_M, S_N, S_SU, S_F], intons=[1.0, 1.5], tempos=[1.0, 1.4], speeds=[1.12]),
    '13': dict(aim='あたたかく誘う（海を見ながら、パスタとスイーツ）',
               texts=[dict(text='海を見ながら、パスタとスイーツ！'), dict(text='海を見ながら、パスタと、スイーツ。'),
                      dict(text='海を見ながら、パスタと、スイーツ！')],
               styles=[S_M, S_N, S_SU, S_F], intons=[1.0, 1.5], tempos=[1.0, 1.4], speeds=[1.12]),
    '14': dict(aim='ひらけた感じで（屋上から、太平洋）',
               texts=[dict(text='屋上から、太平洋！'), dict(text='屋上から……太平洋！'), dict(text='屋上から、太平洋。')],
               styles=[S_M, S_SU, S_F, S_N], intons=[1.0, 1.5], tempos=[1.0, 1.4], speeds=[1.12]),
    '15': dict(aim='笑みを含んで軽く（悪役も、ここなら悪くないかも）',
               texts=[dict(text='悪役も、ここなら悪くないかも。'), dict(text='悪役も、ここなら、悪くないかも？'),
                      dict(text='悪役も、ここなら悪くないかもっ。')],
               styles=[S_M, S_N, S_SU, S_F], intons=[1.0, 1.5], tempos=[1.0, 1.4], speeds=[1.12]),
}

BASE = dict(vo_c.BASE)

# 仕上げの抑揚（vo_prosody.expand：Praat PSOLA）。alpha＝中央値より上の高さの動きの倍率、end_st＝語尾 0.4 秒の上げ下げ（半音）。
# 2026-11-08 選定：α は 1.0 / 1.25 / 1.4 を Whisper の読みと抑揚の数値で比べて決めた（読みが崩れた α は使わない）
SHAPE = {
    '01': dict(alpha=1.35), '03': dict(alpha=1.4, end_st=-1.5), '04': dict(alpha=1.4), '05': dict(alpha=1.35),
    '06': dict(alpha=1.4), '07': dict(alpha=1.4), '08': dict(alpha=1.4), '09': dict(alpha=1.4, end_st=-1.5),
    '11': dict(alpha=1.4), '12': dict(alpha=1.4), '13': dict(alpha=1.4), '14': dict(alpha=1.35),
    '15': dict(alpha=1.4, end_st=1.5),
}
# 行の音量の手直し（dB、ピークはリミッターで抑える）。06「マッハ6.7」だけ他の数字の行より 3〜4 dB 小さかった（-17.1 LUFS）
LEVEL = {'06': 2.0}


def dur_ok(k, e):
    if k in TARGET:
        return TARGET[k][0] <= e['tdur'] <= TARGET[k][1]
    return DUR_LO <= e['rel'] <= DUR_HI


def cfg_for(k, t, st, it, tp, sp):
    pl = {kk: v for kk, v in t.items() if kk != 'hstretch'}
    pl.update(style=st, inton=it, tempo=tp, speed=sp)
    if pl.get('parts'):
        pl['parts'] = [dict(p) for p in pl['parts']]
        for p in pl['parts']:          # 部分ごとの style が無ければ行の style。強さ・テンポ・話速は行と同じ
            p.setdefault('style', st)
    return pl


def text_of(pl):
    return pl.get('text') or ''.join(p['text'] for p in pl['parts'])


def hstretch_file(src, dst, add=0.08, gl=0.024, seed=0):
    """vo_c.hstretch と同じ処理（「マハ」の h の摩擦を add 秒のばして「マッハ」に）。戻り値 (dip0, dip1) 秒"""
    x = vo_c.read(src)
    h = int(SR * 0.005)
    e = 20 * np.log10(np.sqrt(np.mean(x[:len(x) // h * h].reshape(-1, h) ** 2, 1)) + 1e-9)
    on = int(np.argmax(e > e.max() - 30))
    lo, hi = on + 10, on + 70
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
    for kk in range(0, n_add, G // 2):
        o = rng.integers(0, max(1, len(seg) - G))
        buf[kk:kk + G] += seg[o:o + G] * win
    y = np.concatenate([x[:mid], buf[:n_add], x[mid:]])
    write(dst, y)
    return a * 0.005, b * 0.005


def write(path, y):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype('<i2').tobytes())


def tdur(path):
    x = vo_c.read(path)
    t0, t1 = vo_c.speech_span(x)
    return round(min(len(x) / SR, t1 + vo_c.PAD1) - max(0, t0 - vo_c.PAD0), 3)


def load_res():
    """expr.json（合成と抑揚の数値）に verify.json（Whisper の読み確認）を重ねて返す。
    takes と verify を同時に走らせても上書きし合わないよう、ファイルを分けてある"""
    pj = os.path.join(TAKES, 'expr.json')
    res = json.load(open(pj)) if os.path.exists(pj) else {}
    for k, d in load_ver().items():
        for n, v in d.items():
            if n in res.get(k, {}):
                res[k][n].update(v)
    return res


WKEYS = ('kana', 'text', 'ratio', 'must_ok', 'must_ng', 'p', 'diffs')


def load_ver():
    pv = os.path.join(TAKES, 'verify.json')
    return json.load(open(pv)) if os.path.exists(pv) else {}


def save_ver(k, name, ent):
    ver = load_ver()
    ver.setdefault(k, {})[name] = {kk: ent[kk] for kk in WKEYS if kk in ent}
    json.dump(ver, open(os.path.join(TAKES, 'verify.json'), 'w'), ensure_ascii=False, indent=1)


def save_res(res):
    for d in res.values():                 # 読み確認は verify.json の方に置く
        for n, e in d.items():
            for kk in WKEYS:
                e.pop(kk, None)
    json.dump(res, open(os.path.join(TAKES, 'expr.json'), 'w'), ensure_ascii=False, indent=1)


def make(k, name, pl, hst, res, threads, whisper_all=False, whisper=True):
    path = os.path.join(TAKES, k, name + '.wav')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    va.TEXT.setdefault(k, text_of(pl))
    try:
        va.synth_one({**BASE, 'per_line': {k: pl}}, k, path, accent='auto')
    except Exception as e:      # noqa: BLE001
        print(k, name, 'FAILED', e, flush=True)
        return
    rpl = dict(pl)
    if hst:
        raw = path[:-4] + '_raw.wav'
        os.replace(path, raw)
        dip = hstretch_file(raw, path, hst)
        os.remove(raw)
        rpl['hstretch'] = dict(add=hst, dip=[round(dip[0], 3), round(dip[1], 3)])
    d = tdur(path)
    rel = d / OLD_DUR[k]
    m = vm.liveliness(path)
    ent = dict(settings=rpl, tdur=d, rel=round(rel, 3), **{'m_' + kk: v for kk, v in m.items()})
    if whisper and (whisper_all or dur_ok(k, ent)):
        verify_one(k, name, ent, threads)
    res.setdefault(k, {})[name] = ent
    print(f'{k} {name:34s} {d:5.2f}s ({rel:4.2f}) f0r {m.get("f0_range", 0):5.2f} f0sd {m.get("f0_sd", 0):4.2f} '
          f'ensd {m["en_sd"]:5.2f} score {lively(ent):4.2f} ' + (f'must {"OK" if ent["must_ok"] else "NG"} r {ent["ratio"]} {ent["text"]}'
                                        if 'must_ok' in ent else '(len out)'), flush=True)


def verify_one(k, name, ent, threads):
    kana = ''.join(vo_c.reading(ent['settings']))
    ent.update(kana=kana, **vo_c.score(os.path.join(TAKES, k, name + '.wav'), k, kana, threads))
    save_ver(k, name, ent)


def verify(lines, top, threads):
    """長さが範囲内のテイクを抑揚の点数順に、Whisper の読み確認が top 本そろうまで（読み NG は数えない）"""
    res = load_res()
    for k in lines or list(PLAN):
        vs = sorted(((n, e) for n, e in res.get(k, {}).items()
                     if dur_ok(k, e) and f0_ok(k, e)),
                    key=lambda kv: -lively(kv[1]))
        good = sum(1 for _, e in vs if ok(e, k))
        for n, e in vs:
            if good >= top:
                break
            if 'must_ok' in e:
                continue
            verify_one(k, n, e, threads)
            good += bool(ok(e, k))
            print(f'{k} {n:34s} score {lively(e):4.2f} must {"OK" if e["must_ok"] else "NG"} r {e["ratio"]} {e["text"]}',
                  flush=True)


def takes(a):
    os.makedirs(TAKES, exist_ok=True)
    res = load_res()
    keys = a.lines.split(',') if a.lines else (list(FIX5) if TAKES.endswith('C_fix5') else list(PLAN))
    for k in keys:
        P = PLAN[k]
        for (ti, t), st, it, tp, sp, sd in itertools.product(enumerate(P['texts']), P['styles'], P['intons'],
                                                              P['tempos'], P['speeds'], range(a.seeds)):
            if st == S_N and it != P['intons'][0]:
                continue
            name = f't{ti}_{st}_i{it:g}_d{tp:g}_s{sp:g}_k{sd}'
            if name in res.get(k, {}):
                continue
            make(k, name, cfg_for(k, t, st, it, tp, sp), t.get('hstretch'), res, a.threads, whisper=False)
            save_res(res)


def more(picks, seeds, threads):
    res = load_res()
    for k, name in picks:
        base = res[k][name]
        pl = {kk: v for kk, v in base['settings'].items() if kk != 'hstretch'}
        hst = base['settings'].get('hstretch', {}).get('add')
        stem = name.rsplit('_k', 1)[0]
        for sd in range(seeds):
            nm = f'{stem}_k{sd}'
            if nm not in res[k]:
                make(k, nm, pl, hst, res, threads)
                save_res(res)


def _measure(arg):
    k, n = arg
    return k, n, vm.liveliness(os.path.join(TAKES, k, n + '.wav'))


def remeasure(lines, procs=3):
    """抑揚の数値をすべて測り直す（vo_metrics を直したとき）"""
    from multiprocessing import Pool
    res = load_res()
    jobs = [(k, n) for k in (lines or list(res)) for n in res.get(k, {})]
    with Pool(procs) as pool:
        for k, n, m in pool.imap_unordered(_measure, jobs, chunksize=4):
            for kk in [kk for kk in res[k][n] if kk.startswith('m_')]:
                del res[k][n][kk]
            res[k][n].update({'m_' + kk: v for kk, v in m.items()})
    save_res(res)
    print('remeasured', len(jobs))


def lively(e):
    """抑揚の点数（大きいほど抑揚がある）：高さの幅と標準偏差（きしみ声を除く）、大きさの標準偏差"""
    return (e.get('m_f0_range', 0) / 6 + e.get('m_f0_sd', 0) / 2 + e.get('m_en_sd', 0) / 9
            - 4 * max(0.0, e.get('m_creak', 0) - 0.1))      # きしみ声が多いもの（張りがない）は下げる


# 声の高さの中央値の範囲（旧版は 99〜152 Hz）。「！」で張ると 190〜230 Hz まで上がる。数字の行とキラーンは張ってよいが、
# 250 Hz 前後（裏声ぎみ）は別人に聞こえるので外す。MAZE の行・不穏な行は落ち着いた高さに
F0_LO = 95
F0_HI = {'01': 220, '04': 220, '05': 220, '06': 220, '07': 220, '08': 200, '14': 190}
F0_HI_DEF = 175


def f0_ok(k, e):
    return F0_LO <= e.get('m_f0_med', 0) <= F0_HI.get(k, F0_HI_DEF)


def ok(e, k):
    return (e.get('must_ok') and (e.get('ratio') or 0) >= 0.75 and dur_ok(k, e) and f0_ok(k, e))


def rank(lines, top):
    res = load_res()
    flat = vm.table([FLAT])[FLAT]
    for k in lines or (list(FIX5) if TAKES.endswith('C_fix5') else list(PLAN)):
        vs = sorted(((n, e) for n, e in res.get(k, {}).items() if ok(e, k)), key=lambda kv: -lively(kv[1]))
        f = flat[k]
        print(f'{k} {PLAN[k]["aim"]}  flat: f0r {f["f0_range"]:5.2f} f0sd {f["f0_sd"]:4.2f} ensd {f["en_sd"]:5.2f} '
              f'score {lively({"m_" + kk: v for kk, v in f.items()}):4.2f}  ok {len(vs)}/{len(res.get(k, {}))}')
        for n, e in vs[:top]:
            print(f'   {n:34s} {e["tdur"]:5.2f}s ({e["rel"]:4.2f}) f0r {e["m_f0_range"]:5.2f} f0sd {e["m_f0_sd"]:4.2f} '
                  f'ensd {e["m_en_sd"]:5.2f} med {e["m_f0_med"]:5.1f} creak {e.get("m_creak", 0):4.2f} score {lively(e):4.2f} r {e["ratio"]} {e["text"]}')


def collect(picks, partial=False):
    """partial=True：picks の行だけ vo_C/ を差し替える（前の wav と json は _prev_fix5/ に退避）。他の行の json は残す"""
    import shutil
    res = load_res()
    missing = [k for k in LINES if k not in picks]
    if missing and not partial:
        raise SystemExit('pick every line: missing ' + ','.join(missing))
    keys = [k for k in LINES if k in picks]
    if partial:
        os.makedirs(PREV5, exist_ok=True)
        for f in [k + '.wav' for k in keys] + ['durations.json', 'lines.json', 'reading.json', 'candidate.json',
                                                'moras.json', 'moras_ctc_raw.json']:
            src, dst = os.path.join(OUT, f), os.path.join(PREV5, f)
            if os.path.exists(src) and not os.path.exists(dst):      # 2回目以降は最初の退避を残す
                shutil.copy2(src, dst)
    durs, chosen, rd = {}, {}, {}
    for k in keys:
        e = res[k][picks[k]]
        dst = os.path.join(OUT, k + '.wav')
        durs[k] = vo_c.trim(os.path.join(TAKES, k, picks[k] + '.wav'), dst)
        shp = None
        if k in SHAPE:
            import vo_prosody                      # parselmouth が要る（scratchpad の venv で実行）
            shp = vo_prosody.expand(dst, dst, SHAPE[k]['alpha'], SHAPE[k].get('end_st', 0.0), 0.4)
        if k in LEVEL:
            import vo_prosody
            shp = dict(shp or {}, level=vo_prosody.gain_limited(dst, LEVEL[k]))
        st = e['settings']
        rd[k] = dict(text=text_of(st), moras=vo_c.reading(st))
        chosen[k] = dict(take=picks[k], aim=PLAN[k]['aim'], dur_trimmed=durs[k], dur_flat=OLD_DUR[k],
                         takes_dir=os.path.relpath(TAKES, ROOT),
                         whisper=e['text'], reading_ratio=e['ratio'], must_ok=e['must_ok'], whisper_p=e['p'],
                         settings=st, prosody=shp, liveliness=vm.liveliness(dst),
                         n_takes=len(res[k]))
        print(f'{k} {picks[k]:34s} {durs[k]:5.2f}s (flat {OLD_DUR[k]:4.2f}) {SHAPE.get(k)}')
    if partial:
        return collect_merge(durs, rd, chosen)
    json.dump(durs, open(os.path.join(OUT, 'durations.json'), 'w'))
    json.dump(LINES, open(os.path.join(OUT, 'lines.json'), 'w'), ensure_ascii=False, indent=0)
    json.dump(rd, open(os.path.join(OUT, 'reading.json'), 'w'), ensure_ascii=False, indent=1)
    meta = dict(
        engine='AivisSpeech Engine (docker ghcr.io/aivis-project/aivisspeech-engine:cpu-latest, CPU), localhost:10101',
        render='tools/v3/cutC/vo_expr.py (takes → verify → collect, PSOLA by vo_prosody.py), moras by vo_align.py',
        model='阿井田 茂 v1.0.0 (47e53151-a378-46f3-abee-ce13aa07feb1), style per line, accent auto',
        license='ACML 1.0 (commercial OK, credit optional)', credit_text='音声合成：AivisSpeech（阿井田 茂 / CV：古山キリヲ）',
        note='2026-11-08 抑揚版（クライアント「棒読みみたい」への修正）。旧版は _flat_backup/',
        trim=f'speech span at max({vo_c.THR} dBFS, hum p10 + 8 dB), head {vo_c.PAD0} s / tail {vo_c.PAD1} s, 4 ms fades',
        lines=chosen, total_sec=round(sum(durs.values()), 2), total_flat=round(sum(OLD_DUR.values()), 2))
    json.dump(meta, open(os.path.join(OUT, 'candidate.json'), 'w'), ensure_ascii=False, indent=1)
    print('total %.2fs (flat %.2fs)' % (sum(durs.values()), sum(OLD_DUR.values())))


def collect_merge(durs, rd, chosen):
    """修正5：差し替えた行だけ json を更新（他の行と candidate.json の他の項目は残す）"""
    def upd(name, new, indent=None):
        p = os.path.join(OUT, name)
        d = json.load(open(p)) if os.path.exists(p) else {}
        d.update(new)
        d = {k: d[k] for k in sorted(d)}
        json.dump(d, open(p, 'w'), ensure_ascii=False, indent=indent)
        return d
    allr = upd('durations.json', durs)
    upd('lines.json', {k: LINES[k] for k in LINES}, indent=0)
    upd('reading.json', rd, indent=1)
    p = os.path.join(OUT, 'candidate.json')
    meta = json.load(open(p))
    old = {k: meta['lines'][k] for k in chosen if k in meta['lines']}
    meta['lines'].update(chosen)
    meta['lines'] = {k: meta['lines'][k] for k in sorted(meta['lines'])}
    meta['total_sec'] = round(sum(allr.values()), 2)
    meta['note_fix5'] = ('2026-11-08 修正5：01「アニメなら、悪役は、パンチ一発で、空のかなたへ。」（ゆっくり・キラーンは効果音のみ）、'
                         '03「でも、現実だと……こうなる。」、04「パンチの威力は、約1,400トン！」を作り直し。前の版は _prev_fix5/')
    pv = meta.setdefault('prev_fix5', {})          # 最初に退避した版だけを記録（作り直しの途中の版で上書きしない）
    for k, v in old.items():
        pv.setdefault(k, dict(take=v.get('take'), dur_trimmed=v.get('dur_trimmed'), text=LINES_PREV5.get(k)))
    if 'metrics_flat_vs_new' in meta:      # 抑揚の数値の表も新しい行に合わせる（old は平らな版なので文が違う：参考）
        for k in chosen:
            meta['metrics_flat_vs_new'].setdefault(k, {})['new'] = chosen[k]['liveliness']
            meta['metrics_flat_vs_new'][k].update(whisper=chosen[k]['whisper'], must_ok=chosen[k]['must_ok'],
                                                  ratio=chosen[k]['reading_ratio'], text_changed_fix5=True)
    json.dump(meta, open(p, 'w'), ensure_ascii=False, indent=1)
    print('total %.2fs' % meta['total_sec'])


LINES_PREV5 = {'01': 'アニメなら、キラーン。', '03': '現実だと、こうなる。', '04': '約1,400トン。', '08': '……宇宙です。'}


def ab():
    """行ごとに 旧（_flat_backup）→ 0.35 秒 → 新 → 0.9 秒。AB_intonation.wav と、行の時刻の AB_intonation.json"""
    out, marks, t = [np.zeros(int(0.3 * SR))], [], 0.3
    for k in LINES:
        o = vo_c.read(os.path.join(FLAT, k + '.wav'))
        n = vo_c.read(os.path.join(OUT, k + '.wav'))
        marks.append(dict(line=k, text=LINES[k], old=round(t, 2), new=round(t + len(o) / SR + 0.35, 2)))
        out += [o, np.zeros(int(0.35 * SR)), n, np.zeros(int(0.9 * SR))]
        t += (len(o) + len(n)) / SR + 1.25
    y = np.concatenate(out)
    write(os.path.join(OUT, 'AB_intonation.wav'), y)
    json.dump(marks, open(os.path.join(OUT, 'AB_intonation.json'), 'w'), ensure_ascii=False, indent=1)
    print('AB_intonation.wav %.1fs' % (len(y) / SR))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd in ('collect', 'more'):
        ap = argparse.ArgumentParser()
        ap.add_argument('cmd')
        ap.add_argument('picks', nargs='*')
        ap.add_argument('--seeds', type=int, default=4)
        ap.add_argument('--threads', type=int, default=2)
        ap.add_argument('--partial', action='store_true')
        a = ap.parse_intermixed_args()
        pairs = [tuple(x.split('=', 1)) for x in a.picks]
        collect(dict(pairs), a.partial) if cmd == 'collect' else more(pairs, a.seeds, a.threads)
        return
    if cmd == 'ab':
        ab()
        return
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['takes', 'rank', 'verify', 'remeasure'])
    ap.add_argument('--lines')
    ap.add_argument('--seeds', type=int, default=1)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--top', type=int, default=8)
    a = ap.parse_args()
    if a.cmd == 'takes':
        takes(a)
    elif a.cmd == 'remeasure':
        remeasure(a.lines.split(',') if a.lines else None)
    elif a.cmd == 'verify':
        verify(a.lines.split(',') if a.lines else None, a.top, a.threads)
    else:
        rank(a.lines.split(',') if a.lines else None, a.top)


if __name__ == '__main__':
    main()
