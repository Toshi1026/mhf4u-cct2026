"""案B（アニメの悪役を桂浜から殴ったら？）のタイムライン。仮ナレーション exports/v3/vo_B/*.wav の実測から決める。
本番の声が届いたら vo_B/*.wav を差し替え、durations.json / moras.json を作り直すだけで全体が追従する。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import FPS, OUT   # noqa: E402

VO = os.path.join(OUT, 'vo_B')
LEAD, GAP = 0.08, 0.08
ORDER = ['01', '02', '03', '04', '05', '06', '07', 'T', '08', '09', '10', '11', '12', '13']
FIXED = {'T': 1.20, '13': 1.80}                    # ナレーションなし：T＝着水→CG から実写へ、13＝施設名のエンドカード
MIN = {'08': 3.10, '10': 2.60, '11': 3.00, '12': 3.00}   # 画を見せるのに要る最低の長さ


def load():
    dur = json.load(open(os.path.join(VO, 'durations.json')))
    mor = json.load(open(os.path.join(VO, 'moras.json')))
    beats, t = [], 0.0
    for k in ORDER:
        d = FIXED[k] if k in FIXED else max(MIN.get(k, 0), LEAD + dur[k] + GAP)
        f0, f1 = round(t * FPS), round((t + d) * FPS)
        beats.append(dict(key=k, f0=f0, f1=f1, n=f1 - f0, vo=(t + LEAD) if k in dur else None,
                          moras=[(m, t + LEAD + s) for m, s in mor.get(k, [])]))
        t += d
    return beats


def mora_frame(beat, index):
    return round(beat['moras'][index][1] * FPS) - beat['f0']


if __name__ == '__main__':
    bs = load()
    for b in bs:
        print(b['key'], b['f0'], b['f1'], b['n'], '%.2fs' % (b['n'] / FPS), ''.join(m for m, _ in b['moras']))
    print('total', bs[-1]['f1'], '%.2fs' % (bs[-1]['f1'] / FPS))
