"""ナレーション（exports/v3/vo/*.wav）の実測の長さから、ビートごとのフレーム範囲を決める。

ビートの頭から LEAD 秒後にナレーションを鳴らし、話し終わりから GAP 秒で次のビートへ。
11（CG→実写の間）はナレーションなしで SILENT11 秒、13 は話し終わりから TAIL 秒（施設名を見せる）。
本番の声が届いたら vo/*.wav を差し替えて durations.json / moras.json を作り直すだけで、全体の尺が追従する。
"""
import json
import os

from core import FPS, VO

LEAD, GAP, SILENT11, TAIL = 0.08, 0.06, 1.20, 0.45
FOOD14 = 2.70             # 14：2F カフェのパスタとケーキ（ナレーションなし・施設名で終わる）
ORDER = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14']


def load():
    dur = json.load(open(os.path.join(VO, 'durations.json')))
    mor = json.load(open(os.path.join(VO, 'moras.json')))
    beats, t = [], 0.0
    for k in ORDER:
        if k == '11':
            d = SILENT11
        elif k == '14':
            d = FOOD14
        elif k == '13':
            d = LEAD + dur[k] + TAIL
        else:
            d = LEAD + dur[k] + GAP
        f0, f1 = round(t * FPS), round((t + d) * FPS)
        beats.append(dict(key=k, f0=f0, f1=f1, n=f1 - f0, vo=(t + LEAD) if k in dur else None,
                          moras=[(m, t + LEAD + s) for m, s in mor.get(k, [])]))
        t += d
    return beats


def mora_frame(beat, index):
    """ビートの中で index 番目のモーラが鳴り始めるフレーム（ビート頭からの相対）。負の index は後ろから。"""
    m = beat['moras'][index]
    return round(m[1] * FPS) - beat['f0']


if __name__ == '__main__':
    bs = load()
    for b in bs:
        print(b['key'], b['f0'], b['f1'], b['n'], '%.2fs' % (b['n'] / FPS), ''.join(m for m, _ in b['moras']))
    print('total', bs[-1]['f1'], '%.2fs' % (bs[-1]['f1'] / FPS))
