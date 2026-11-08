"""案C（アニメだと → 現実だと → 宇宙 → MAZE）のタイムライン。edl/v3/SCRIPT_C_ig.md の C01〜C16。
ナレーション exports/v3/vo_C/NN.wav の実測（durations.json）と、強制アラインメントのモーラ時刻（moras.json）から決める。
声を差し替えたら vo_C の wav と durations.json / moras.json を作り直すだけで全体が追従する（tools/v3/cutC/vo_c.py → vo_align.py）。

長さ＝ max(MIN, LEAD + PRE + 声の長さ + GAP + TAIL)。FIXED は声に関係なく固定。
  01：最短 1.6 秒（48f）。拳のはずみ（f30・f39、各9f）が終わったところで切る。2.0 秒だと後半が空の草原で間延びし、
      03 の一撃が 3.27 秒になる → 1.6 秒で一撃が 2.87 秒（最初の3秒に入る）。
  02：「現実だと」で一瞬止まる（レコードを止めた音）0.4 秒。
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
FIXED = {'02': 0.40, '10': 3.40, '16': 2.40}
MIN = {'01': 1.60, '03': 2.60, '14': 2.40}   # 01：キラーンのあと拳が2回はずんだら切る（空の草原で間延びさせない。パンチの一撃を3秒以内に）
PRE = {'08': 0.10}            # 08 のテイクには「……」の息（約 0.26 秒）が頭に入っている
TAIL = {'14': 1.20}
SCREAM_AT = 0.10            # 10 の頭から悲鳴が始まるまで（秒）


def load():
    dur = json.load(open(os.path.join(VO, 'durations.json')))
    mp = os.path.join(VO, 'moras.json')
    mor = json.load(open(mp)) if os.path.exists(mp) else {}
    beats, t = [], 0.0
    for k in ORDER:
        if k in FIXED:
            d = FIXED[k]
        else:
            d = max(MIN.get(k, 0), LEAD + PRE.get(k, 0) + dur.get(k, 0) + GAP + TAIL.get(k, 0))
        f0, f1 = round(t * FPS), round((t + d) * FPS)
        v0 = t + LEAD + PRE.get(k, 0) if k in dur else None
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
