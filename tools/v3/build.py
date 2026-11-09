"""v3・案A をつないで書き出す：exports/v3/MAZE_v3_ig_A.mp4（＋確認用のコンタクトシート）。
  python3 build.py            全部
  python3 build.py --sheet    書き出し後、1秒ごとの縮小を並べた exports/v3/sheet_A.png も作る
"""
import importlib
import os
import subprocess
import sys

import numpy as np

import audio
import core
import timeline

CG_LAST = 11          # 01〜11 は CG（左上に「CGシミュレーション」）


def frames():
    for b in timeline.load():
        mod = importlib.import_module('beats.b%s' % b['key'])
        k = 0
        for f in mod.render(b, {}):
            if k >= b['n']:
                break
            if int(b['key']) <= CG_LAST:
                # 11 の最後の 8 フレームでラベルを消す（実写に入る前に）
                a = 0.7
                if int(b['key']) == CG_LAST:
                    a *= float(np.clip((b['n'] - 1 - k) / 8, 0, 1))
                f = core.cg_label(f, a)
            yield b['key'], k, f
            k += 1
        assert k == b['n'], ('beat', b['key'], k, b['n'])


def main():
    bs = timeline.load()
    n = bs[-1]['f1']
    silent = os.path.join(core.OUT, '_video_A.mp4')
    enc = core.Encoder(silent, n)
    sheet = []
    for g, (key, k, f) in enumerate(frames()):
        enc.write(f)
        if g % 30 == 15:
            sheet.append(f)
        if g % 150 == 0:
            print('frame', g, '/', n, 'beat', key, flush=True)
    enc.close()
    audio.main()
    out = os.path.join(core.OUT, 'MAZE_v3_ig_A.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', silent, '-i', os.path.join(core.OUT, 'mix.wav'),
                    '-filter_complex', '[1:a]loudnorm=I=-14:TP=-1.5:LRA=11[a]', '-map', '0:v', '-map', '[a]',
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-ar', '48000', '-shortest',
                    '-movflags', '+faststart', out], check=True)
    if '--sheet' in sys.argv:
        core.contact(sheet, os.path.join(core.OUT, 'sheet_A.png'), cols=8, scale=0.2)
    print('wrote', out)


if __name__ == '__main__':
    main()
