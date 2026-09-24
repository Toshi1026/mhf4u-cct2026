#!/usr/bin/env python3
"""サイネージ30秒版・15秒版の CapCut 用素材を書き出す（承認済みEDL：signage_30s_v5 / サイネージ15秒 v6）。

  python3 tools/render_signage_capcut.py stab        # IMG_9658 の横ずれ（素材7.2〜8.4秒）を測って stab_IMG_9658.json に保存
  python3 tools/render_signage_capcut.py overlays    # 透過PNG（signage_spec.py から。30秒版・15秒版で共通）
  python3 tools/render_signage_capcut.py clips [30|15] [カット番号...]
  python3 tools/render_signage_capcut.py preview [30|15]

各カットのクリップ：1920x1080・30fps（CFR）・コマ数ちょうど・H.264 High・CRF16・yuv420p・bt709・音声なし。
焼き込むもの：HDR(HLG)→SDR のトーンマップ、基本の色補正、速度（コマ番号で選ぶ）、拡大・回転（キーフレームは
素材の秒で直線補間）・位置、スタビライズ（IMG_9658 の素材7.2〜8.4秒、位置のみ）、右下の減光（素材ごとのEV）。
右下のマーク・END・減光の値は edl/mockups/signage_spec.py（唯一の正）から読む。
"""
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'edl', 'mockups'))
import signage_spec as S  # noqa: E402

FF = '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
FOOT = os.path.join(ROOT, 'footage', '01_撮影素材')
OUT = os.path.join(ROOT, 'exports', 'capcut')
W, H, FPS = 1920, 1080, 30
TONEMAP = ('zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,'
           'zscale=t=bt709:m=bt709:r=tv')

# ---------------------------------------------------------------------------
# カット表（EDLの値そのまま。rot は [(素材の秒, CapCutの回転欄の値＝+が時計回り)]、dx は右へ[w比]、dy は下へ[h比]）
# ---------------------------------------------------------------------------
PH_FOOD = '料理＋水平線（横・低い位置）'
PH_FORK = 'フォークの持ち上げ（横・120fps）'
ROT_9631 = [(3.20, 0.70), (8.60, 0.81)]
ROT_9665 = [(10.35, -1.2), (11.00, -1.65), (11.95, -2.15)]
ROT_9674 = [(6.70, -2.4)]
ROT_9658 = [(6.10, 0.3)]

EDITS = {
    '30': dict(T=30.00, E=20.70, cuts=[
        dict(n=1, src='IMG_9631', n0=192, step=2, count=162, scale=1.07, rot=ROT_9631, dx=0.0, dy=-0.010, dim=S.DIM_EV['IMG_9631']),
        dict(n=2, src='IMG_9665', n0=621, step=1, count=99, scale=1.15, rot=ROT_9665, dx=0.0, dy=0.036, dim=S.DIM_EV['IMG_9665']),
        dict(n=3, src='IMG_9674', n0=402, step=1, count=90, scale=1.12, rot=ROT_9674, dx=0.012, dy=-0.012, dim=S.DIM_EV['IMG_9674']),
        dict(n=4, placeholder=PH_FOOD, count=180, slug='placeholder_food_horizon'),
        dict(n=5, placeholder=PH_FORK, count=90, slug='placeholder_fork_lift'),
        dict(n=6, src='IMG_9658', n0=366, step=1, count=279, scale=1.15, rot=ROT_9658, dx=0.0, dy=0.015, dim=None,
             stab=(7.2, 8.4)),
    ]),
    '15': dict(T=15.00, E=9.00, cuts=[
        dict(n=1, src='IMG_9665', n0=621, step=1, count=90, scale=1.15, rot=ROT_9665, dx=0.0, dy=0.036, dim=S.DIM_EV['IMG_9665']),
        dict(n=2, src='IMG_9674', n0=402, step=1, count=90, scale=1.12, rot=ROT_9674, dx=0.012, dy=-0.012, dim=S.DIM_EV['IMG_9674']),
        dict(n=3, placeholder=PH_FOOD, count=90, slug='placeholder_food_horizon'),
        dict(n=4, src='IMG_9631', n0=198, step=1, count=180, scale=1.07, rot=ROT_9631, dx=0.0, dy=-0.010, dim=None),
    ]),
}


def clip_name(cut):
    return '%02d_%s.mp4' % (cut['n'], cut.get('slug') or cut['src'])


def edit_dir(key):
    return os.path.join(OUT, 'signage%s' % key)


# ---------------------------------------------------------------------------
# 素材
# ---------------------------------------------------------------------------
_PTS = {}


def pts_of(src):
    """表示順の各コマの時刻（秒）。時間の単位は1/600秒。"""
    if src not in _PTS:
        out = subprocess.run([FF, '-v', 'error', '-i', os.path.join(FOOT, src + '.MOV'), '-map', '0:v:0',
                              '-c', 'copy', '-f', 'framecrc', '-'], capture_output=True, text=True).stdout
        p = sorted(int(l.split(',')[2]) for l in out.splitlines() if l and not l.startswith('#'))
        _PTS[src] = [v / 600.0 for v in p]
    return _PTS[src]


def decode(src, idx, w, h):
    """コマ番号 idx（昇順）を、トーンマップ → w×h（lanczos）に縮めた float RGB で1枚ずつ返す。"""
    a, b = idx[0], idx[-1]
    step = idx[1] - idx[0] if len(idx) > 1 else 1
    sel = "select='between(n,%d,%d)*not(mod(n-%d,%d))'" % (a, b, a, step)
    vf = '%s,%s,zscale=w=%d:h=%d:filter=lanczos,format=gbrpf32le' % (sel, TONEMAP, w, h)
    cmd = [FF, '-v', 'error', '-threads', '4', '-i', os.path.join(FOOT, src + '.MOV'), '-map', '0:v:0', '-an',
           '-vf', vf, '-fps_mode', 'passthrough', '-f', 'rawvideo', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    nbytes = w * h * 3 * 4
    got = 0
    while True:
        buf = p.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        g = np.frombuffer(buf, np.float32).reshape(3, h, w)
        got += 1
        yield np.ascontiguousarray(np.stack([g[2], g[0], g[1]], -1))   # gbrp → RGB
    p.wait()
    if got != len(idx):
        raise RuntimeError('%s: %d コマを期待したが %d コマ' % (src, len(idx), got))


# ---------------------------------------------------------------------------
# 色（EDLの color_ja：彩度・コントラストは+5程度まで、Teal&Orangeにしない、白を飛ばさない）
# ---------------------------------------------------------------------------
GRADE = {
    # contrast：中間の傾き +5%（両端で寝かせるS字）。sat：彩度の倍率。hue：海の色相を青へ寄せる角度（度）。
    'IMG_9631': dict(contrast=0.05, sat=1.05, hue=0.0),
    'IMG_9665': dict(contrast=0.05, sat=1.00, hue=0.0),          # 彩度は上げない（砂州と外洋の差を残す）
    'IMG_9674': dict(contrast=0.05, sat=1.05, hue=2.0, hl_pull=0.06),   # 窓と白い壁のハイライトだけ下げる
    'IMG_9658': dict(contrast=0.05, sat=1.05, hue=3.0, hue_below=0.41),  # 海だけ青緑→わずかに青（空は触らない）
}
KNEE = 0.90   # これより上は肩で丸めて、1.0を超えた白波・砂州を飛ばさない


def _shoulder(x):
    k = KNEE
    return np.where(x > k, k + (1 - k) * np.tanh((x - k) / (1 - k)), x)


def grade(rgb, src, y_frac=None):
    """rgb：float（トーンマップ直後、表示のガンマのまま）。y_frac：各行の、素材の上からの割合（海の範囲のマスク用）。"""
    g = GRADE[src]
    x = np.maximum(rgb, 0.0)
    x = _shoulder(x)
    k = 2 * g['contrast']                     # f(x)=x+k·x(1−x)(2x−1)：中間の傾き 1+k/2、両端は寝かせる
    x = x + k * x * (1 - x) * (2 * x - 1)
    Y = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    cb = (x[..., 2] - Y) / 1.8556
    cr = (x[..., 0] - Y) / 1.5748
    if g.get('hl_pull'):
        t = np.clip((Y - 0.70) / 0.30, 0, 1)
        Y2 = Y - g['hl_pull'] * t * t * (3 - 2 * t)
        r = np.where(Y > 1e-4, Y2 / np.maximum(Y, 1e-4), 1.0)
        Y, cb, cr = Y2, cb * r, cr * r
    sat = g['sat']
    if sat != 1.0:                            # 明るいところほど上げ幅を減らす（ハイライトの色飛びを防ぐ）
        s = 1 + (sat - 1) * np.clip((0.95 - Y) / 0.25, 0, 1)
        cb, cr = cb * s, cr * s
    if g.get('hue'):
        ang = np.degrees(np.arctan2(cr, cb))
        chroma = np.hypot(cb, cr)
        wgt = np.exp(-0.5 * ((ang + 52.0) / 9.0) ** 2) * np.clip(chroma / 0.06, 0, 1)   # 青緑〜青の海の色相だけ
        if g.get('hue_below') is not None and y_frac is not None:
            m = np.clip((y_frac - (g['hue_below'] + 0.01)) / 0.03, 0, 1)[:, None]  # 水平線より下だけ
            wgt = wgt * m
        th = np.radians(g['hue']) * wgt
        c, s_ = np.cos(th), np.sin(th)
        cb, cr = cb * c - cr * s_, cb * s_ + cr * c
    r = Y + 1.5748 * cr
    b = Y + 1.8556 * cb
    gch = (Y - 0.2126 * r - 0.0722 * b) / 0.7152
    return np.clip(np.stack([r, gch, b], -1), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# 形（CapCutと同じ順：画面に合わせる → 中心で拡大 → 中心で回転（+が時計回り）→ 位置）
# ---------------------------------------------------------------------------
def lerp(keys, t):
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    return keys[-1][1]


def inv_matrix(pw, ph, scale, rot_deg, dx_px, dy_px):
    """出力の画素（中心が整数）→ 縮小済みの素材（pw×ph）の画素、の2x3行列（WARP_INVERSE_MAP用）。"""
    f = (pw / 3840.0, ph / 2160.0)             # 4Kの素材 → 縮小済み
    k = 2.0                                    # 1080pの画面に合わせたときの 4K/1080p
    th = math.radians(rot_deg)
    c, s = math.cos(th), math.sin(th)
    # 出力の連続座標 X（画素の中心が +0.5）→ 4Kの連続座標 U：U = C4 + (k/scale)·R(−θ)·(X − C − t)
    A = np.array([[c, s], [-s, c]]) * (k / scale)
    C = np.array([W / 2.0, H / 2.0]) + np.array([dx_px, dy_px])
    C4 = np.array([1920.0, 1080.0])
    # 縮小済みの整数座標 p = U·f − 0.5、出力 X = x + 0.5
    M = np.zeros((2, 3))
    Af = A * np.array(f)[:, None]
    M[:, :2] = Af
    M[:, 2] = (C4 - A @ C) * np.array(f) + Af @ np.array([0.5, 0.5]) - 0.5
    return M


def prescale_size(scale):
    return int(math.ceil(1920 * scale)) + 2, int(math.ceil(1080 * scale)) + 2


def warp(img, M):
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def coverage(pw, ph, M):
    """素材の内側に入っている割合（1未満の画素＝はみ出し）。"""
    ones = np.ones((ph, pw), np.float32)
    return cv2.warpAffine(ones, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ---------------------------------------------------------------------------
# 減光（signage_spec.apply_dim と同じ式を float で）
# ---------------------------------------------------------------------------
_DIM = None


def dim(rgb, ev):
    global _DIM
    if not ev:
        return rgb
    if _DIM is None:
        _DIM = S.dim_matte(W, H)[..., None]
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lin = lin * (1 - _DIM + _DIM * (2.0 ** ev))
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055).astype(np.float32)


# ---------------------------------------------------------------------------
# スタビライズ（IMG_9658、位置のみ）
# ---------------------------------------------------------------------------
STAB_JSON = os.path.join(OUT, '_work', 'stab_IMG_9658.json')


def analyze_stab(src='IMG_9658', n0=366, count=279, ref_t=7.2):
    """全コマの平行移動（4Kの画素）を、砂利浜と左の砂州（素材の縦62〜80%）の位相相関で測る。
    基準は素材 ref_t 秒のコマ。コマ間の積算はせず、どのコマも基準のコマと直接比べる（誤差がたまらない）。"""
    idx = list(range(n0, n0 + count))
    pts = pts_of(src)
    ref_i = int(np.argmin([abs(pts[i] - ref_t) for i in idx]))
    a, b = int(0.62 * 1080), int(0.80 * 1080)
    lum = np.array([0.2126, 0.7152, 0.0722], np.float32)
    bands = [np.ascontiguousarray((fr @ lum)[a:b]) for fr in decode(src, idx, 1920, 1080)]
    win = cv2.createHanningWindow((1920, b - a), cv2.CV_32F)
    res = []
    for i, g in enumerate(bands):
        (sx, sy), resp = cv2.phaseCorrelate(bands[ref_i], g, win)
        res.append(dict(n=idx[i], t=pts[idx[i]], dx4k=float(sx * 2), dy4k=float(sy * 2), response=float(resp)))
    os.makedirs(os.path.dirname(STAB_JSON), exist_ok=True)
    json.dump(res, open(STAB_JSON, 'w'), indent=1)
    return res


def stab_offsets(cut):
    """各コマの補正量（出力の画素、右・下が+）。素材 t0〜t1 秒の横ずれ（位置のみ）を止める：
    区間の中は t0 の位置に固定し、t1 より後は t1 時点の補正量をそのまま保つ（区間の出口で画が跳ばない）。
    区間より前は補正なし。測った値は3コマの移動平均で均してから使う。"""
    t0, t1 = cut['stab']
    data = json.load(open(STAB_JSON))
    t = np.array([d['t'] for d in data])
    k = np.ones(3) / 3
    x = np.convolve(np.pad([d['dx4k'] for d in data], 1, mode='edge'), k, 'valid')
    y = np.convolve(np.pad([d['dy4k'] for d in data], 1, mode='edge'), k, 'valid')
    x0, y0 = np.interp(t0, t, x), np.interp(t0, t, y)
    x1, y1 = np.interp(t1, t, x), np.interp(t1, t, y)
    f = cut['scale'] / 2                       # 4Kの画素 → 出力の画素
    out = []
    for ti, xi, yi in zip(t, x, y):
        if ti <= t0:
            out.append((0.0, 0.0))
        elif ti < t1:
            out.append((-(xi - x0) * f, -(yi - y0) * f))
        else:
            out.append((-(x1 - x0) * f, -(y1 - y0) * f))
    return out


# ---------------------------------------------------------------------------
# 書き出し
# ---------------------------------------------------------------------------
def encoder(path, nframes):
    cmd = [FF, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-s', '%dx%d' % (W, H), '-r', str(FPS),
           '-i', '-', '-frames:v', str(nframes),
           '-vf', 'zscale=rangein=full:primariesin=709:transferin=709:matrix=709:range=limited:primaries=709:'
                  'transfer=709:dither=error_diffusion,format=yuv420p',
           '-c:v', 'libx264', '-profile:v', 'high', '-preset', 'slow', '-crf', '16', '-g', '30',
           '-pix_fmt', 'yuv420p', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
           '-color_range', 'tv', '-an', '-movflags', '+faststart', '-video_track_timescale', '15360', path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def to48(rgb):
    return (np.clip(rgb, 0, 1) * 65535 + 0.5).astype('<u2').tobytes()


def placeholder_frame(text):
    im = Image.new('RGB', (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(S.font_path('Regular'), 30)
    s = '撮影待ち：' + text
    tw = d.textlength(s, font=f)
    d.text(((W - tw) / 2, H / 2 - 20), s, font=f, fill=(255, 255, 255))
    f2 = ImageFont.truetype(S.font_path('Regular'), 20)
    s2 = '（仮スレート・差し替え前提）'
    d.text(((W - d.textlength(s2, font=f2)) / 2, H / 2 + 26), s2, font=f2, fill=(160, 160, 160))
    return np.asarray(im).astype(np.float32) / 255.0


def render_cut(key, cut, log):
    path = os.path.join(edit_dir(key), clip_name(cut))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    enc = encoder(path, cut['count'])
    info = dict(file=os.path.relpath(path, OUT), frames=cut['count'])
    if 'placeholder' in cut:
        buf = to48(placeholder_frame(cut['placeholder']))
        for _ in range(cut['count']):
            enc.stdin.write(buf)
        enc.stdin.close()
        enc.wait()
        info.update(placeholder=cut['placeholder'])
        log.append(info)
        return info
    src = cut['src']
    idx = [cut['n0'] + cut['step'] * i for i in range(cut['count'])]
    pts = pts_of(src)
    pw, ph = prescale_size(cut['scale'])
    stab = stab_offsets(cut) if cut.get('stab') else None
    if stab is not None:
        data = json.load(open(STAB_JSON))
        assert data[0]['n'] == cut['n0'] and len(data) == cut['count']
    yfrac_cache = {}
    rots, min_cov, per = [], 1.0, []
    for i, fr in enumerate(decode(src, idx, pw, ph)):
        sec = pts[idx[i]]
        rot = lerp(cut['rot'], sec)
        dxp, dyp = cut['dx'] * W, cut['dy'] * H
        sx, sy = stab[i] if stab else (0.0, 0.0)
        M = inv_matrix(pw, ph, cut['scale'], rot, dxp + sx, dyp + sy)
        # 海のマスクは出力の座標で（水平線41%）決めたいので、素材の各行の出力上の高さを近似で求める
        if src not in yfrac_cache:
            rows = (np.arange(ph) + 0.5) / ph              # 素材の上からの割合
            yfrac_cache[src] = ((rows * 2160 - 1080) * cut['scale'] / 2 + 540 + dyp) / H
        g = grade(fr, src, yfrac_cache[src])
        out = warp(g, M)
        cov = coverage(pw, ph, M)
        mc = float(cov.min())
        min_cov = min(min_cov, mc)
        if i == 0 or i == cut['count'] - 1 or i % 10 == 0:
            per.append(dict(i=i, n=idx[i], src_t=round(sec, 4), rot=round(rot, 4), stab_px=[round(sx, 2), round(sy, 2)],
                            min_coverage=round(mc, 4)))
        out = dim(np.clip(out, 0, 1), cut['dim'])
        enc.stdin.write(to48(out))
        rots.append(rot)
    enc.stdin.close()
    enc.wait()
    info.update(src=src, n_first=idx[0], n_last=idx[-1], step=cut['step'], src_in=round(pts[idx[0]], 4),
                src_out=round(pts[idx[-1]], 4), rot_first=round(rots[0], 4), rot_last=round(rots[-1], 4),
                min_coverage=round(min_cov, 4), samples=per)
    log.append(info)
    print(json.dumps({k: v for k, v in info.items() if k != 'samples'}, ensure_ascii=False))
    return info


def render_clips(key, only=None):
    log = []
    for cut in EDITS[key]['cuts']:
        if only and cut['n'] not in only:
            continue
        render_cut(key, cut, log)
    p = os.path.join(OUT, '_work', 'render_log_signage%s.json' % key)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    old = json.load(open(p)) if os.path.exists(p) else []
    keep = {e['file']: e for e in old}
    for e in log:
        keep[e['file']] = e
    json.dump(list(keep.values()), open(p, 'w'), ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------
# 透過PNG と プレビュー
# ---------------------------------------------------------------------------
OVL = os.path.join(OUT, 'overlays')


def overlays():
    subprocess.run([sys.executable, os.path.join(ROOT, 'edl', 'mockups', 'signage_spec.py'), '--overlays', OVL], check=True)


def timeline(key):
    e = EDITS[key]
    tm = S.timing(e['E'], e['T'])
    t, clips = 0, []
    for cut in e['cuts']:
        clips.append(dict(cut=cut['n'], file='signage%s/%s' % (key, clip_name(cut)), frame_in=t, frame_out=t + cut['count'],
                          in_s=round(t / FPS, 4), out_s=round((t + cut['count']) / FPS, 4)))
        t += cut['count']
    assert t == round(e['T'] * FPS)
    return clips, tm


def preview(key):
    e = EDITS[key]
    clips, tm = timeline(key)
    (c0, c1), (e0, e1) = (tm['corner'][0][0], tm['corner'][-1][0]), (tm['end'][0][0], tm['end'][-1][0])
    cin = []
    for c in clips:
        cin += ['-i', os.path.join(OUT, c['file'])]
    n = len(clips)
    corner = os.path.join(OVL, 'signage_corner_1920x1080.png')
    end = os.path.join(OVL, 'signage_end_1920x1080.png')
    T = e['T']
    fc = ''.join('[%d:v]' % i for i in range(n)) + 'concat=n=%d:v=1:a=0,zscale=matrixin=709:rangein=limited:range=full,format=gbrp[base];' % n
    fc += ('[%d:v]format=rgba,fade=t=in:st=%.4f:d=%.4f:alpha=1,fade=t=out:st=%.4f:d=%.4f:alpha=1[co];'
           % (n, c0, S.FADE, c1 - S.FADE, S.FADE))
    fc += ('[%d:v]format=rgba,fade=t=in:st=%.4f:d=%.4f:alpha=1,fade=t=out:st=%.4f:d=%.4f:alpha=1[en];'
           % (n + 1, e0, S.FADE, e1 - S.FADE, S.FADE))
    fc += "[base][co]overlay=format=gbrp:enable='between(t,%.4f,%.4f)'[b1];" % (c0, c1)
    fc += ("[b1][en]overlay=format=gbrp:enable='between(t,%.4f,%.4f)',"
           "zscale=rangein=full:matrix=709:range=limited:dither=error_diffusion,format=yuv420p[v]" % (e0, e1))
    cmd = [FF, '-v', 'error', '-y'] + cin + [
        '-loop', '1', '-framerate', '30', '-t', '%.4f' % T, '-i', corner,
        '-loop', '1', '-framerate', '30', '-t', '%.4f' % T, '-i', end,
        '-filter_complex', fc, '-map', '[v]', '-frames:v', str(int(round(T * FPS))), '-r', '30',
        '-c:v', 'libx264', '-profile:v', 'high', '-preset', 'slow', '-crf', '16', '-g', '30', '-pix_fmt', 'yuv420p',
        '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709', '-color_range', 'tv',
        '-an', '-movflags', '+faststart', os.path.join(OUT, 'preview_signage%s.mp4' % key)]
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'stab':
        r = analyze_stab()
        for d in r[::6]:
            print('%.3f  dx4k %.1f  dy4k %.1f' % (d['t'], d['dx4k'], d['dy4k']))
    elif a[0] == 'overlays':
        overlays()
    elif a[0] == 'clips':
        render_clips(a[1], [int(x) for x in a[2:]] or None)
    elif a[0] == 'preview':
        preview(a[1])
