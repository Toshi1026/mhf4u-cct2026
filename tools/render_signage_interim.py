#!/usr/bin/env python3
"""【暫定】サイネージ30秒版の料理の枠（カット4＋5、270コマ＝9.0秒）を、承認済みの「縦パスタの2画面」（案c）で埋める。

料理の2カット（料理＋水平線／フォークの持ち上げ）が撮れるまでの放映用。構成・カットの秒は承認済みの v5 のまま。
2画面は1本の連続した動き（IMG_9644 1.0→5.5秒、IMG_9633 4.5→9.0秒、どちらも0.5倍）で作り、
カット4（180コマ）とカット5（90コマ）の2ファイルに分ける。つなぎ目でコマは飛ばない。

15秒版の料理の枠（カット3、90コマ）は、ケーキ版で作った proposals/signage15_cake/03_… をそのまま使う。

  python3 tools/render_signage_interim.py      # → exports/capcut/proposals/signage30_interim/
差し替えは remotion/clip_overrides.json で行う。
"""
import json
import os

import render_signage_proposal as P

R = P.R
OUT = os.path.join(P.PROP, 'signage30_interim')
SPLIT = [('04_IMG_9644+IMG_9633_diptych_a.mp4', 180), ('05_IMG_9644+IMG_9633_diptych_b.mp4', 90)]


def main():
    os.makedirs(OUT, exist_ok=True)
    total = sum(n for _, n in SPLIT)
    panels = P.option_panels('c', total, 1.00, 4.50)
    its = [p.iter() for p in panels]
    log, min_margin = [], 1e9
    for name, count in SPLIT:
        path = os.path.join(OUT, name)
        enc = R.encoder(path, count)
        for _ in range(count):
            canvas = P.np.empty((P.H, P.W, 3), P.np.float32)
            canvas[:] = P.GAP_RGB
            for p, it in zip(panels, its):
                img, m = next(it)
                min_margin = min(min_margin, m)
                canvas[:, p.x:p.x + p.w] = img
            enc.stdin.write(R.to48(R.dim(P.np.clip(canvas, 0, 1), P.DIM_EV['IMG_9633'])))
        enc.stdin.close()
        enc.wait()
        log.append(dict(file=os.path.relpath(path, P.CAP), frames=count))
    info = dict(clips=log, min_corner_margin_src_px=round(min_margin, 1), panels=[p.info() for p in panels],
                dim_ev=P.DIM_EV['IMG_9633'])
    json.dump(info, open(os.path.join(OUT, 'render_log.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(info, ensure_ascii=False))


if __name__ == '__main__':
    main()
