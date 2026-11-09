"""案B を書き出す：exports/v3/MAZE_v3_ig_B.mp4（＋ sheet_B.png）。tools/v3 から python3 cutB/build_b.py"""
import importlib
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core            # noqa: E402
import timeline_b as TL   # noqa: E402
import audio_b         # noqa: E402

CG = ['01', '02', '03', '04', '05', '06', '07', 'T']


def main():
    bs = TL.load()
    n = bs[-1]['f1']
    silent = os.path.join(core.OUT, '_video_B.mp4')
    enc = core.Encoder(silent, n)
    for b in bs:
        mod = importlib.import_module('cutB.beats.b%s' % b['key'])
        k = 0
        for f in mod.render(b, {}):
            if k >= b['n']:
                break
            if b['key'] in CG:
                a = 0.7 * (float(np.clip((b['n'] - 1 - k) / 8, 0, 1)) if b['key'] == 'T' else 1.0)
                f = core.cg_label(f, a)
            enc.write(f)
            k += 1
        assert k == b['n'], (b['key'], k, b['n'])
        print('beat', b['key'], 'done', flush=True)
    enc.close()
    audio_b.main()
    out = os.path.join(core.OUT, 'MAZE_v3_ig_B.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', silent, '-i', os.path.join(core.OUT, 'mix_B.wav'),
                    '-filter_complex', '[1:a]afade=t=in:d=0.01,loudnorm=I=-14:TP=-1.5:LRA=11[a]', '-map', '0:v',
                    '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-ar', '48000', '-shortest',
                    '-movflags', '+faststart', out], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', out, '-vf', 'fps=1,scale=216:384,tile=10x4:padding=4',
                    '-frames:v', '1', os.path.join(core.OUT, 'sheet_B.png')], check=True)
    print('wrote', out)


if __name__ == '__main__':
    main()
