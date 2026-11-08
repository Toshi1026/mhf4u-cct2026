"""案C（アニメだと → 現実だと → 宇宙 → MAZE）のタイムライン。edl/v3/SCRIPT_C_ig.md の C01〜C16。
ナレーション exports/v3/vo_C/NN.wav の実測（durations.json）と、強制アラインメントのモーラ時刻（moras.json）から決める。
声を差し替えたら vo_C の wav と durations.json / moras.json を作り直すだけで全体が追従する（tools/v3/cutC/vo_c.py → vo_align.py）。

長さ＝ max(MIN, LEAD + PRE + 声の長さ + GAP + TAIL)。FIXED は声に関係なく固定。
  2026-11-08 クライアント修正（フックの間）：
  01：「アニメなら、悪役は、パンチ一発で、空のかなたへ。」（ゆっくり・楽しげ）→ 声が終わって GLINT_AFTER 秒あとに
      キラーン（効果音 kira2 と光のいちばん大きい所）→ BREATH 秒の息 → 02。長さ＝LEAD＋声＋GLINT_AFTER＋BREATH。
  02：「現実だと」で止まる（レコードを止めた音）。03 の声「でも、現実だと……」は 02 の頭＋LEAD02 から始まり（VO_EARLY）、
      止まった絵の上で言い切る。02 の長さ＝LEAD02＋「コ」までの時間−PRE_KO（「……」の間に現実の CG へ切り替わる）。
  03：頭（02 の終わり）から PRE_KO 秒のスローの拳 →「コ」の 2 フレーム前に当たる。「コ」のあとは POST_KO フレーム
      （修正前と同じ。03 の引き〜05 の時間の流れ（_cfx）を変えない）。
  10：宇宙から着水まで一気にズームイン。悪役の悲鳴（vo_C/scream.wav）を SCREAM_AT 秒から。約 3.4 秒。
  16：エンドカード（看板 → ロゴ）2.4 秒。
  08：「……宇宙です」の「……」ぶんの間を PRE で声の前に置く。
  14：「屋上から、太平洋。」のあとに声なしでローソン・産直・タワーを 0.4 秒ずつ（TAIL 1.2 秒）。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import FPS, OUT   # noqa: E402

VO = os.environ.get('MAZE_VO_C', os.path.join(OUT, 'vo_C'))
SCREAM = os.path.join(VO, 'scream_client.wav')   # 2026-11-08 クライアント本人が演じた録音（本人確認済み）。TANAKA 版は vo_C/scream.wav
LEAD, GAP = 0.08, 0.14
ORDER = ['%02d' % i for i in range(1, 17)]
FIXED = {'10': 3.40, '16': 2.40}
MIN = {'14': 2.40}
GLINT_AFTER, BREATH = 0.20, 0.45   # 01：声の終わり → キラーン（光のいちばん大きい所）→ 息 → 02
LEAD02 = 0.12                      # 02 の頭（止まる音）から「でも」まで
VO_EARLY = {'03': '02'}            # 03 の声は 02 の頭から
KO_INDEX = 8                       # 03 のモーラ「コ」（デモゲンジツダト|コオナル）
PRE_KO = 0.50                      # 現実の CG（03）は「コ」の 0.5 秒前から（「……」の間）
POST_KO = 50                       # 03 の「コ」のあとのフレーム数（修正前と同じ）
PRE = {'08': 0.10}            # 08 のテイクには「……」の息（約 0.26 秒）が頭に入っている
TAIL = {'14': 1.20}
SCREAM_AT = 0.10            # 10 の頭から悲鳴が始まるまで（秒）


def load():
    dur = json.load(open(os.path.join(VO, 'durations.json')))
    mp = os.path.join(VO, 'moras.json')
    mor = json.load(open(mp)) if os.path.exists(mp) else {}
    beats, t = [], 0.0
    early = {}                                   # 前のビートで始まる声：key → 声の頭（秒）
    for k in ORDER:
        v0 = t + LEAD + PRE.get(k, 0) if k in dur else None
        if k in early:
            v0 = early[k]
        if k == '01':
            d = LEAD + dur['01'] + GLINT_AFTER + BREATH
        elif k in VO_EARLY.values():             # 02：次の声（03）の「コ」の PRE_KO 秒前まで
            nk = next(a for a, b in VO_EARLY.items() if b == k)
            early[nk] = t + LEAD02
            d = LEAD02 + mor[nk][KO_INDEX][1] - PRE_KO
        elif k in VO_EARLY:                      # 03：「コ」のあと POST_KO フレーム
            assert mor[k][KO_INDEX][0] == 'コ', mor[k]
            d = (round((v0 + mor[k][KO_INDEX][1]) * FPS) + POST_KO) / FPS - t
        elif k in FIXED:
            d = FIXED[k]
        else:
            d = max(MIN.get(k, 0), LEAD + PRE.get(k, 0) + dur.get(k, 0) + GAP + TAIL.get(k, 0))
        f0, f1 = round(t * FPS), round((t + d) * FPS)
        b = dict(key=k, f0=f0, f1=f1, n=f1 - f0, vo=v0,
                 moras=[(m, v0 + s) for m, s in mor.get(k, [])] if v0 is not None else [])
        if k == '10':
            b['scream'] = t + SCREAM_AT
        beats.append(b)
        t += d
    return beats


def mora_frame(beat, index):
    return round(beat['moras'][index][1] * FPS) - beat['f0']


if __name__ == '__main__':
    bs = load()
    for b in bs:
        print(b['key'], b['f0'], b['f1'], b['n'], '%.2fs' % (b['n'] / FPS), '%5.2f-%5.2f' % (b['f0'] / FPS, b['f1'] / FPS),
              ''.join(m for m, _ in b['moras']) or ('(scream)' if 'scream' in b else ''))
    print('total', bs[-1]['f1'], '%.2fs' % (bs[-1]['f1'] / FPS))
