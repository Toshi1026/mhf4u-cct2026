"""案C を書き出す：exports/v3/MAZE_v3_ig_C.mp4（＋ sheet_C.png：2 fps のコンタクトシート）。tools/v3 から python3 cutC/build_c.py
音だけ直したときは python3 cutC/build_c.py --audio-only（画の _video_C.mp4 を使い回す）"""
import importlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cutB.beats import _useA  # noqa: E402,F401  最上位の `beats` を案A（tools/v3/beats）に固定（cutB の _motion などが使う）
import core            # noqa: E402
import timeline_c as TL   # noqa: E402
import audio_c         # noqa: E402
from preview import label   # noqa: E402


def main():
    bs = TL.load()
    n = bs[-1]['f1']
    silent = os.path.join(core.OUT, '_video_C.mp4')
    if '--audio-only' in sys.argv:                     # 画はそのまま（_video_C.mp4）、音だけ作り直して書き出す
        return finish(silent)
    enc = core.Encoder(silent, n)
    for b in bs:
        mod = importlib.import_module('cutC.beats.c%s' % b['key'])
        k = 0
        for f in mod.render(b, {}):
            if k >= b['n']:
                break
            enc.write(label(b['key'], f, k, b['n']))
            k += 1
        assert k == b['n'], (b['key'], k, b['n'])
        print('beat', b['key'], 'done', flush=True)
    enc.close()
    finish(silent)


def finish(silent):
    audio_c.main(write_cues=True)   # mix_C.wav は -14 LUFS・-1.5 dBTP にそろえ済み（1 パスの loudnorm は TP で縮めて -14.8 になるので通さない）
    out = os.path.join(core.OUT, 'MAZE_v3_ig_C.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', silent, '-i', os.path.join(core.OUT, 'mix_C.wav'),
                    '-filter_complex', '[1:a]afade=t=in:d=0.01[a]', '-map', '0:v',
                    '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-ar', '48000', '-shortest',
                    '-movflags', '+faststart', out], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', out, '-vf', 'fps=2,scale=216:384,tile=10x8:padding=4',
                    '-frames:v', '1', os.path.join(core.OUT, 'sheet_C.png')], check=True)
    print('wrote', out)


if __name__ == '__main__':
    main()
