#!/usr/bin/env python3
"""【暫定】サイネージ30秒版・15秒版の料理の枠を、承認済みの「縦パスタの2画面」（案c）で埋める。

料理のカット（料理＋水平線／フォークの持ち上げ）が撮れるまでの放映用。構成・カットの秒は承認済みの版のまま。

- 30秒版：カット4＋5（351〜620コマ、270コマ＝9.0秒）。1本の連続した動き（IMG_9644 1.0→5.5秒、IMG_9633 4.5→9.0秒、
  どちらも0.5倍）で作り、カット4（180コマ）とカット5（90コマ）の2ファイルに分ける。つなぎ目でコマは飛ばない
- 15秒版：カット3（180〜269コマ、90コマ）。ケーキ版のカット3と同じ区間（IMG_9644 4.0秒〜、IMG_9633 7.25秒〜）

右下の減光は、右下のマークがフェードアウトするコマでは、マークの不透明度に合わせて弱める（再QC：ケーキ版では
マークが最後まで乗っていて隠れていたが、暫定版ではマークが消える間に海の右下に暗いしみが見えていた）。
マークの出方は remotion/src/timeline.json から読む。

  python3 tools/render_signage_interim.py [30] [15]   # → exports/capcut/proposals/signage{30,15}_interim/
差し替えは remotion/clip_overrides.json で行う。
"""
import json
import os
import sys

import numpy as np

import render_signage_proposal as P

R = P.R
TIMELINE = os.path.join(P.ROOT, 'remotion', 'src', 'timeline.json')

PLAN = {
    '30': dict(comp='Signage30', start=351, pasta_t=1.00, sea_t=4.50, out='signage30_interim',
               split=[('04_IMG_9644+IMG_9633_diptych_a.mp4', 180), ('05_IMG_9644+IMG_9633_diptych_b.mp4', 90)]),
    '15': dict(comp='Signage15', start=180, pasta_t=4.00, sea_t=7.25, out='signage15_interim',
               split=[('03_IMG_9644+IMG_9633_diptych.mp4', 90)]),
}


def mark_opacity(comp, frame):
    """timeline.json の右下のマークの、全体のコマ番号 frame での不透明度（直線補間）。"""
    tl = json.load(open(TIMELINE))
    c = [x for x in tl['compositions'] if x['id'] == comp][0]
    ov = [o for o in c['overlays'] if o['label'].startswith('右下')][0]
    if not (ov['from'] <= frame < ov['to']):
        return 0.0
    k = np.array(ov['keyframes'], float)
    return float(np.interp(frame, k[:, 0], k[:, 1]))


def render(key):
    plan = PLAN[key]
    out = os.path.join(P.PROP, plan['out'])
    os.makedirs(out, exist_ok=True)
    total = sum(n for _, n in plan['split'])
    panels = P.option_panels('c', total, plan['pasta_t'], plan['sea_t'])
    its = [p.iter() for p in panels]
    ev = P.DIM_EV['IMG_9633']
    log, min_margin, i, alphas = [], 1e9, 0, []
    for name, count in plan['split']:
        path = os.path.join(out, name)
        enc = R.encoder(path, count)
        for _ in range(count):
            canvas = np.empty((P.H, P.W, 3), np.float32)
            canvas[:] = P.GAP_RGB
            for p, it in zip(panels, its):
                img, m = next(it)
                min_margin = min(min_margin, m)
                canvas[:, p.x:p.x + p.w] = img
            a = mark_opacity(plan['comp'], plan['start'] + i)
            alphas.append(round(a, 3))
            enc.stdin.write(R.to48(R.dim(np.clip(canvas, 0, 1), ev * a)))
            i += 1
        enc.stdin.close()
        enc.wait()
        log.append(dict(file=os.path.relpath(path, P.CAP), frames=count))
    info = dict(comp=plan['comp'], clips=log, min_corner_margin_src_px=round(min_margin, 1),
                panels=[p.info() for p in panels], dim_ev=ev, dim_scale_by_mark_opacity=alphas)
    json.dump(info, open(os.path.join(out, 'render_log.json'), 'w'), ensure_ascii=False, indent=1)
    print(key, json.dumps({k: v for k, v in info.items() if k not in ('panels', 'dim_scale_by_mark_opacity')},
                          ensure_ascii=False))


if __name__ == '__main__':
    for k in (sys.argv[1:] or ['30', '15']):
        render(k)
