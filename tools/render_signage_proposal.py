#!/usr/bin/env python3
"""【提案】サイネージ30秒版・15秒版に「ケーキ」と「縦で撮ったパスタ＋水平線」を入れる案（2026-09-24）。

承認済みの版（signage_30s_v5 / サイネージ15秒 v6）と、その書き出し（exports/capcut/signage30|signage15/、
tools/render_signage_capcut.py）には手を触れない。値は edl/edl_result.json の proposals（signage_v6_cake /
signage15_v7_cake）と、共通の数値表 edl/mockups/signage_spec.py から取る。

  python3 tools/render_signage_proposal.py options     # パスタの見せ方 (a)(b)(c) の静止画・短い動画・比較JPG・ケーキ候補の評価
  python3 tools/render_signage_proposal.py clips       # 提案版で新しく作るクリップ（パスタの2画面、ケーキの仮スレート）
  python3 tools/render_signage_proposal.py preview [30|15]

出力はすべて exports/capcut/proposals/ の下（Git管理外）。
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
sys.path.insert(0, os.path.join(ROOT, 'edl', 'mockups'))
import render_signage_capcut as R  # noqa: E402
import signage_spec as S  # noqa: E402

W, H, FPS = R.W, R.H, R.FPS
FF = R.FF
CAP = os.path.join(ROOT, 'exports', 'capcut')
PROP = os.path.join(CAP, 'proposals')
OPT = os.path.join(PROP, 'signage_pasta_options')

# 提案で新しく使う素材の色（承認済みの素材の値には触れない。+5程度まで・Teal&Orangeにしない）
R.GRADE.setdefault('IMG_9644', dict(contrast=0.05, sat=1.03, hue=0.0))   # ガラス越しで少し霞む。彩度は料理の赤を守る程度
R.GRADE.setdefault('IMG_9633', dict(contrast=0.05, sat=1.05, hue=0.0))
R.GRADE.setdefault('IMG_9651', dict(contrast=0.05, sat=1.00, hue=0.0))
DIM_EV = {'IMG_9633': -0.50, 'IMG_9644': -0.30}   # 右下の減光（FOODの仮の値 −0.30、海の右のパネルは河口と同じ −0.50）

GAP = 8                     # 2画面のすき間（px、1080p）。色は黒
GAP_RGB = (0.0, 0.0, 0.0)

CAKE_TEXT = 'ケーキ＋水平線（横・三脚・120fpsでフォークを入れる瞬間）'


# ---------------------------------------------------------------------------
# 素材：コマ番号と水平線の実測
# ---------------------------------------------------------------------------
def frame_index(src, t):
    """素材の時刻 t 秒に最も近いコマ番号。"""
    pts = R.pts_of(src)
    return int(np.argmin([abs(p - t) for p in pts]))


def frames(src, t_in, count, step=1):
    n0 = frame_index(src, t_in)
    idx = [n0 + step * i for i in range(count)]
    assert idx[-1] < len(R.pts_of(src)), (src, idx[-1])
    return idx


def measure_horizon(src, idx, dw, dh, yr):
    """各コマの水平線（中央の列での高さ[素材の高さ比]、傾き[度、+は右下がり]）。7コマの移動平均で均す。"""
    ys, ts = [], []
    y0, y1 = int(yr[0] * dh), int(yr[1] * dh)
    xs = np.linspace(0.08 * dw, 0.86 * dw, 10).astype(int)
    bw = int(0.06 * dw)
    for fr in R.decode(src, idx, dw, dh):
        L = fr.mean(axis=2)
        pts = []
        for x in xs:
            col = L[y0:y1, x:x + bw].mean(axis=1)
            j = int(np.argmin(np.diff(col)))       # 空（明）→ 海（暗）の境
            pts.append(y0 + j + 0.5)
        p = np.polyfit(xs + bw / 2, np.array(pts), 1)
        ys.append(np.polyval(p, dw / 2) / dh)
        ts.append(np.degrees(np.arctan(p[0])))
    k = 7
    sm = lambda a: np.convolve(np.pad(a, k // 2, mode='edge'), np.ones(k) / k, 'valid')
    return sm(np.array(ys)), sm(np.array(ts))


# ---------------------------------------------------------------------------
# パネル（水平線をまっすぐ、決めた高さに固定して切り出す）
# ---------------------------------------------------------------------------
def panel_warp(fr, out_w, out_h, zoom, hz_src_frac, tilt_deg, hz_out_frac, cx_frac=0.5):
    """fr：素材（dw×dh、float RGB）。出力 out_w×out_h の中で、水平線が高さ hz_out_frac・水平になるよう切り出す。
    zoom：素材の短い方の辺でぴったり覆う大きさ（=1.0）に対する拡大率。戻り値：(画, 四隅の余裕[素材px、負ならはみ出し])"""
    dh, dw = fr.shape[:2]
    k = min(dw / out_w, dh / out_h) / zoom          # 出力1pxあたりの素材px
    a = np.radians(tilt_deg)
    c, s = np.cos(a), np.sin(a)
    A = k * np.array([[c, -s], [s, c]])
    P = np.array([cx_frac * dw, hz_src_frac * dh])  # 素材の水平線上の点
    X = np.array([out_w / 2.0, hz_out_frac * out_h])
    t = P - A @ X
    M = np.hstack([A, t[:, None]])
    out = cv2.warpAffine(fr, M, (out_w, out_h), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    corners = np.array([[0, 0], [out_w, 0], [0, out_h], [out_w, out_h]], float)
    u = corners @ A.T + t
    margin = float(min(u[:, 0].min(), u[:, 1].min(), dw - u[:, 0].max(), dh - u[:, 1].max()))
    return out, margin


# 素材ごとの設定（デコードの大きさ、水平線を探す範囲）
SRC = {
    'IMG_9644': dict(dec=(2160, 3840), meas=(540, 960), yr=(0.33, 0.45)),   # 縦
    'IMG_9633': dict(dec=(3840, 2160), meas=(960, 540), yr=(0.36, 0.56)),   # 横
}


class Panel:
    def __init__(self, src, t_in, count, x, w, zoom, hz_out, dec=None):
        self.src, self.x, self.w, self.zoom = src, x, w, zoom
        self.hz_out = hz_out            # 数値か、各コマの値の配列
        self.idx = frames(src, t_in, count)
        cfg = SRC[src]
        self.dec = dec or cfg['dec']
        self.hz, self.tilt = measure_horizon(src, self.idx, *cfg['meas'], cfg['yr'])

    def info(self):
        pts = R.pts_of(self.src)
        return dict(src=self.src, n_first=self.idx[0], n_last=self.idx[-1], src_in=round(pts[self.idx[0]], 4),
                    src_out=round(pts[self.idx[-1]], 4), x=self.x, w=self.w, zoom=self.zoom,
                    tilt_deg_first=round(float(self.tilt[0]), 3), tilt_deg_last=round(float(self.tilt[-1]), 3),
                    horizon_src_first=round(float(self.hz[0]), 4), horizon_src_last=round(float(self.hz[-1]), 4))

    def iter(self):
        for i, fr in enumerate(R.decode(self.src, self.idx, *self.dec)):
            g = R.grade(fr, self.src)
            hz_out = self.hz_out[i] if np.ndim(self.hz_out) else self.hz_out
            out, m = panel_warp(g, self.w, H, self.zoom, self.hz[i], self.tilt[i], hz_out)
            yield out, m


def render_layout(path, panels, count, dim_ev, overlay_corner=False, stills=()):
    """panels を横に並べた 1920×1080・30fps のクリップを書き出す。stills：静止画を残すコマ番号 → JPGのパス。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    enc = R.encoder(path, count)
    its = [p.iter() for p in panels]
    min_margin = 1e9
    for i in range(count):
        canvas = np.empty((H, W, 3), np.float32)
        canvas[:] = GAP_RGB
        for p, it in zip(panels, its):
            img, m = next(it)
            min_margin = min(min_margin, m)
            canvas[:, p.x:p.x + p.w] = img
        canvas = R.dim(np.clip(canvas, 0, 1), dim_ev)
        if overlay_corner or i in dict(stills):
            im = Image.fromarray((np.clip(canvas, 0, 1) * 255 + 0.5).astype(np.uint8))
            comp = S.comp(im, 'corner', 1.0)
            if i in dict(stills):
                comp.save(dict(stills)[i], quality=92)
            if overlay_corner:
                canvas = np.asarray(comp).astype(np.float32) / 255.0
        enc.stdin.write(R.to48(canvas))
    enc.stdin.close()
    enc.wait()
    return dict(file=os.path.relpath(path, CAP), frames=count, min_corner_margin_src_px=round(min_margin, 1),
                panels=[p.info() for p in panels], dim_ev=dim_ev)


def slate(path, count, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    enc = R.encoder(path, count)
    buf = R.to48(R.placeholder_frame(text))
    for _ in range(count):
        enc.stdin.write(buf)
    enc.stdin.close()
    enc.wait()
    return dict(file=os.path.relpath(path, CAP), frames=count, placeholder=text)


# ---------------------------------------------------------------------------
# パスタの見せ方の案（すべて IMG_9644。海のパネルは IMG_9633）
# ---------------------------------------------------------------------------
HALF = (W - GAP) // 2                 # (b) 956px
THIRD = 636                           # (c) 左のパスタのパネル
BIG = W - GAP - THIRD                 # (c) 右の海のパネル 1276px


def option_panels(opt, count, pasta_t, sea_t):
    if opt == 'a':
        # 16:9の1枚。前半は皿（上に細い海の帯）→ 後半は水平線（上1/3）へ、ゆっくり上へ動く窓
        start, end = -0.12, 0.33      # 水平線の出力上の高さ（負＝画面より上＝まだ見えない）
        hz = np.linspace(start, end, count)
        return [Panel('IMG_9644', pasta_t, count, 0, W, 1.03, hz)]
    if opt == 'b':
        return [Panel('IMG_9644', pasta_t, count, 0, HALF, 1.05, 0.30, dec=(1080, 1920)),
                Panel('IMG_9633', sea_t, count, HALF + GAP, HALF, 1.70, 0.30)]
    if opt == 'c':
        # 水平線は41%（30秒版のカット1・ENDと同じ高さ）。パスタは台の手前の縁と脚が入らない大きさ
        return [Panel('IMG_9644', pasta_t, count, 0, THIRD, 1.20, 0.41, dec=(1080, 1920)),
                Panel('IMG_9633', sea_t, count, THIRD + GAP, BIG, 1.60, 0.41)]
    raise ValueError(opt)


OPT_LABEL = {
    'a': '(a) 16:9の1枚・皿から水平線へゆっくり上がる窓',
    'b': '(b) 2画面 1:1（パスタ＋水平線｜海）水平線をつなぐ',
    'c': '(c) 2画面 1:2（パスタ＋水平線｜海が主役）水平線をつなぐ',
}


def options(which='abc'):
    os.makedirs(OPT, exist_ok=True)
    lp = os.path.join(OPT, 'options_log.json')
    log = json.load(open(lp)) if os.path.exists(lp) else {}
    count = 135                       # 4.5秒（30秒版の尺）。素材 IMG_9644 3.25→5.5秒、IMG_9633 6.75→9.0秒（どちらも0.5倍）
    for opt in which:
        ev = DIM_EV['IMG_9644'] if opt == 'a' else DIM_EV['IMG_9633']
        stills = {i: os.path.join(OPT, 'option_%s_t%.1f.jpg' % (opt, i / FPS)) for i in (15, 67, 120)}
        info = render_layout(os.path.join(OPT, 'option_%s.mp4' % opt), option_panels(opt, count, 3.25, 6.75),
                             count, ev, overlay_corner=True, stills=stills)
        # 読みやすさ（中央のコマ）
        im = Image.open(stills[67]).convert('RGB')
        info['contrast_mid'] = {k: round(v, 2) for k, v in S.measure(im, 'corner').items()}
        info['stills'] = [os.path.relpath(p, CAP) for p in stills.values()]
        log[opt] = info
        print(opt, json.dumps({k: v for k, v in info.items() if k != 'panels'}, ensure_ascii=False))
    cake_eval()
    compare()
    json.dump(log, open(os.path.join(OPT, 'options_log.json'), 'w'), ensure_ascii=False, indent=1)


def cake_eval():
    """IMG_9651 の冒頭（窓の海＋ハンギングチェア＋ショーケース）を16:9に切ったときの評価用の静止画。"""
    src = 'IMG_9651'
    out = []
    for t, top in ((0.0, 0.07), (0.0, 0.16), (0.6, 0.10)):
        i = frame_index(src, t)
        fr = R.grade(next(R.decode(src, [i], 2160, 3840)), src)
        y0 = int(top * 3840)
        crop = fr[y0:y0 + 1215, :]
        img = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)
        im = Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
        p = os.path.join(OPT, 'cake_eval_IMG_9651_t%.1f_top%02d.jpg' % (t, round(top * 100)))
        S.comp(im, 'corner', 1.0).save(p, quality=90)
        out.append(p)
    return out


def _font(sz):
    return ImageFont.truetype(S.font_path('Regular'), sz)


def compare():
    """3案の比較（本編の見え方・1/8に縮めた遠目の見え方・0.5秒ごとの流れ）＋ケーキ候補の評価を1枚に。"""
    tw, th = 640, 360
    rows = []
    for opt in 'abc':
        ims = [Image.open(os.path.join(OPT, 'option_%s_t%.1f.jpg' % (opt, t))).convert('RGB') for t in (0.5, 2.2, 4.0)]
        rows.append((OPT_LABEL[opt], ims))
    cake = sorted(p for p in os.listdir(OPT) if p.startswith('cake_eval_'))
    rows.append(('参考：IMG_9651 冒頭の16:9切り出し（ケーキ候補の評価・不採用）',
                 [Image.open(os.path.join(OPT, p)).convert('RGB') for p in cake][:3]))
    pad, lab = 16, 44
    Wc = pad + 3 * (tw + pad) + (tw // 2 + pad)
    Hc = pad + len(rows) * (lab + th + pad) + 40
    sheet = Image.new('RGB', (Wc, Hc), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 8), 'サイネージ：縦のパスタ（IMG_9644）の見せ方の比較　左3列＝本編（0.5／2.2／4.0秒、右下のマーク込み）　右端＝1/8に縮めた遠目の見え方（2倍表示）',
           font=_font(20), fill=(230, 230, 230))
    y = pad + 32
    for label, ims in rows:
        d.text((pad, y + 8), label, font=_font(24), fill=(255, 255, 255))
        y += lab
        for j, im in enumerate(ims):
            sheet.paste(im.resize((tw, th), Image.LANCZOS), (pad + j * (tw + pad), y))
        far = ims[1].resize((W // 8, H // 8), Image.BOX).resize((W // 4, H // 4), Image.NEAREST)
        sheet.paste(far, (pad + 3 * (tw + pad), y + (th - H // 4) // 2))
        y += th + pad
    sheet.save(os.path.join(OPT, 'compare_pasta_options.jpg'), quality=90)


# ---------------------------------------------------------------------------
# 提案版のクリップとプレビュー
# ---------------------------------------------------------------------------
PLAN = {
    '30': dict(T=30.00, E=20.70, out='signage30_cake', clips=[
        dict(n=1, reuse='signage30/01_IMG_9631.mp4', count=162),
        dict(n=2, reuse='signage30/02_IMG_9665.mp4', count=99),
        dict(n=3, reuse='signage30/03_IMG_9674.mp4', count=90),
        dict(n=4, layout='c', count=135, pasta_t=3.25, sea_t=6.75, file='04_IMG_9644+IMG_9633_diptych.mp4'),
        dict(n=5, slate=CAKE_TEXT, count=135, file='05_placeholder_cake_horizon_120fps.mp4'),
        dict(n=6, reuse='signage30/06_IMG_9658.mp4', count=279),
    ]),
    '15': dict(T=18.00, E=12.00, out='signage15_cake', clips=[
        dict(n=1, reuse='signage15/01_IMG_9665.mp4', count=90),
        dict(n=2, reuse='signage15/02_IMG_9674.mp4', count=90),
        dict(n=3, layout='c', count=90, pasta_t=4.00, sea_t=7.25, file='03_IMG_9644+IMG_9633_diptych.mp4'),
        dict(n=4, slate=CAKE_TEXT, count=90, file='04_placeholder_cake_horizon_120fps.mp4'),
        dict(n=5, reuse='signage15/04_IMG_9631.mp4', count=180),
    ]),
}


def clip_path(key, c):
    return os.path.join(CAP, c['reuse']) if 'reuse' in c else os.path.join(PROP, PLAN[key]['out'], c['file'])


def clips():
    log = {}
    for key, plan in PLAN.items():
        log[key] = []
        for c in plan['clips']:
            p = clip_path(key, c)
            if 'layout' in c:
                info = render_layout(p, option_panels(c['layout'], c['count'], c['pasta_t'], c['sea_t']),
                                     c['count'], DIM_EV['IMG_9633'])
            elif 'slate' in c:
                info = slate(p, c['count'], c['slate'])
            else:
                info = dict(file=c['reuse'], frames=c['count'], reused_from_approved=True)
            info['cut'] = c['n']
            log[key].append(info)
            print(key, json.dumps({k: v for k, v in info.items() if k != 'panels'}, ensure_ascii=False))
    json.dump(log, open(os.path.join(PROP, 'render_log_proposal.json'), 'w'), ensure_ascii=False, indent=1)


def preview(key):
    plan = PLAN[key]
    T, E = plan['T'], plan['E']
    tm = S.timing(E, T)
    (c0, c1), (e0, e1) = (tm['corner'][0][0], tm['corner'][-1][0]), (tm['end'][0][0], tm['end'][-1][0])
    assert sum(c['count'] for c in plan['clips']) == round(T * FPS)
    cin = []
    for c in plan['clips']:
        cin += ['-i', clip_path(key, c)]
    n = len(plan['clips'])
    ovl = os.path.join(CAP, 'overlays')
    fc = ''.join('[%d:v]' % i for i in range(n)) + 'concat=n=%d:v=1:a=0,zscale=matrixin=709:rangein=limited:range=full,format=gbrp[base];' % n
    fc += ('[%d:v]format=rgba,fade=t=in:st=%.4f:d=%.4f:alpha=1,fade=t=out:st=%.4f:d=%.4f:alpha=1[co];'
           % (n, c0, S.FADE, c1 - S.FADE, S.FADE))
    fc += ('[%d:v]format=rgba,fade=t=in:st=%.4f:d=%.4f:alpha=1,fade=t=out:st=%.4f:d=%.4f:alpha=1[en];'
           % (n + 1, e0, S.FADE, e1 - S.FADE, S.FADE))
    fc += "[base][co]overlay=format=gbrp:enable='between(t,%.4f,%.4f)'[b1];" % (c0, c1)
    fc += ("[b1][en]overlay=format=gbrp:enable='between(t,%.4f,%.4f)',"
           "zscale=rangein=full:matrix=709:range=limited:dither=error_diffusion,format=yuv420p[v]" % (e0, e1))
    out = os.path.join(PROP, 'preview_signage%s_cake.mp4' % key)
    cmd = [FF, '-v', 'error', '-y'] + cin + [
        '-loop', '1', '-framerate', '30', '-t', '%.4f' % T, '-i', os.path.join(ovl, 'signage_corner_1920x1080.png'),
        '-loop', '1', '-framerate', '30', '-t', '%.4f' % T, '-i', os.path.join(ovl, 'signage_end_1920x1080.png'),
        '-filter_complex', fc, '-map', '[v]', '-frames:v', str(int(round(T * FPS))), '-r', '30',
        '-c:v', 'libx264', '-profile:v', 'high', '-preset', 'slow', '-crf', '16', '-g', '30', '-pix_fmt', 'yuv420p',
        '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-color_range', 'tv',
        '-an', '-movflags', '+faststart', out]
    subprocess.run(cmd, check=True)
    print(out, 'corner', tm['corner'], 'end', tm['end'])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'options':
        options(a[1] if len(a) > 1 else 'abc')
    elif a[0] == 'compare':
        compare()
    elif a[0] == 'clips':
        clips()
    elif a[0] == 'preview':
        for k in (a[1:] or ['30', '15']):
            preview(k)
