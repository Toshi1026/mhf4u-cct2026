#!/usr/bin/env python3
"""サイネージ2本（30秒版・15秒版）で共通にする「右下のマーク」「ENDの中央」「右下の減光」の数値表と描画。

この表が唯一の正（2026-09-23、サイネージ30秒版 v4 で作成）。30秒版・15秒版のEDLはここを参照し、
モックアップも、CapCutに置く透過PNGも、このファイルから作る。値を変えるときは2本同時に変わる。

  python3 edl/mockups/signage_spec.py --overlays DIR
      CapCutに置く透過PNGを DIR に書き出す（1920x1080 と 3840x2160 の両方）。
        signage_corner_<W>x<H>.png   右下のマーク（正式ロゴ＋「MAZE WIND~RETREAT」＋影）
        signage_end_<W>x<H>.png      ENDの中央（既定：「高知県土佐市」の1行＋影）
        signage_endlogo_<W>x<H>.png  ENDの中央（代案A：ロゴ＋「高知県土佐市」＋影）
      CapCutでは位置0・拡大100%・回転0で置くだけ。文字機能・影機能は使わない。
  python3 edl/mockups/signage_spec.py --report
      1080pでの位置（px・%）を表示する。

減光（右下の楕円）はPNGにせず、Claudeが書き出す各カットのクリップに焼き込む（素材ごとの強さは DIM_EV）。
大きさ・位置はすべて画面の高さ h・幅 w に対する割合なので、1080pでも4Kでも同じ見え方になる。
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOGO_PATH = os.path.join(ROOT, 'assets', 'logo_white.png')
FONT_DIRS = ['/tmp/claude-0/fonts', os.path.expanduser('~/.fonts'), '/usr/share/fonts/truetype/noto',
             '/usr/share/fonts/opentype/noto']

# ---------------------------------------------------------------------------
# 共通の数値表（h＝画面の高さ、w＝画面の幅）
# ---------------------------------------------------------------------------
SHADOW = [(0.003, 0.65), (0.012, 0.30)]   # 影：黒・ずらしなし。(ガウスのσ[h比], 不透明度) の2層。縁取り・帯・グラデーションは使わない

CORNER = dict(                 # 右下のマーク
    logo_h=0.090,              # ロゴの見えている円の高さ 9.0%h（1080pで97px）
    right=0.050,               # 円の右端：画面の右端から 5.0%w 内側
    bottom=0.050,              # 円の下端：画面の下端から 5.0%h 内側
    line='MAZE WIND~RETREAT',  # 半角チルダ U+007E、前後に空白なし
    weight='Regular',
    size=0.022,                # 文字サイズ（1em）2.2%h（大文字の高さは約1.6%h）
    track=0.15,                # 字間 0.15em
    gap=0.010,                 # 行の右端（インク）〜円の左端 1.0%w
    opacity=1.0,
)

END = dict(                    # ENDの中央・既定（右下を出したまま、地名だけを海に浮かべる）
    line='高知県土佐市',
    weight='Regular',
    size=0.036,                # 文字サイズ 3.6%h（1080pで39px）
    track=0.30,                # 字間 0.30em
    center_y=0.500,            # インクの縦の中心 50%h（水平線41%の下の海の上）
    opacity=1.0,
)

END_LOGO = dict(               # ENDの中央・代案A（右下を消し、ロゴと地名の1組を中央に出す）
    logo_h=0.160,              # ロゴの見えている円の高さ 16%h
    gap=0.018,                 # 円の右端〜文字の左端 1.8%w。1組全体を左右中央に置く
    center_y=0.520,            # 円と文字の縦の中心 52%h
    line='高知県土佐市', weight='Regular', size=0.036, track=0.30,
    opacity=1.0,
)

DIM = dict(                    # 右下の減光（楕円、リニア光で露出を下げる。各クリップに焼き込む）
    w=0.30, h=0.22,            # 楕円の幅 30%w × 高さ 22%h。中心は右下のマーク全体（行の左端〜円の右端、円の中心の高さ）
    feather=0.040,             # 境目のぼかし σ 4.0%h
)
DIM_EV = {                     # 素材ごとの強さ（同じ素材は2本とも同じ値）
    'IMG_9631': -0.70,         # 白波の帯（いちばん明るい背景）
    'IMG_9665': -0.50,         # 河口の砂州・水路
    'IMG_9674': -0.30,         # 店内（暗い）
    'IMG_9658': -0.25,         # 砂利浜（一様な面なので、強いと灰色の楕円が見える）
    'FOOD': -0.30,             # 料理＋水平線（撮影後に実測して決める仮の値）
}

# 出方の決まり（E＝ENDのカットの頭、T＝ファイルの長さ。30秒版は E=20.70・T=30.00、15秒版は E=9.00・T=15.00）
TIMING_RULES = dict(
    default=dict(              # 既定：右下は全編に100%で出したまま（承認済みの「常時表示」）。減光も全カット
        corner='0〜T 常に100%（フェードなし）',
        dim='0〜T すべてのクリップに焼き込む（強さは素材ごと）',
        end='E+1.0→E+1.6 でフェードイン、T−1.4→T−0.8 でフェードアウト',
    ),
    alt_a=dict(                # 代案A：ENDで右下から中央のロゴへ受け渡す（ユーザーが選んだ場合だけ）
        corner='0→0.6 でフェードイン、E−0.6→E でフェードアウト（ENDのカットに入る前に消しきる）',
        dim='0〜E のクリップだけに焼き込む（ENDのクリップにはかけない）',
        end='E+0.3→E+0.9 でフェードイン、T−1.4→T−0.8 でフェードアウト',
    ),
)


def timing(mode, E, T):
    """不透明度のキーフレーム [(秒, 0〜1), ...] を返す。"""
    if mode == 'default':
        return dict(corner=[(0.0, 1.0), (T, 1.0)],
                    end=[(E + 1.0, 0.0), (E + 1.6, 1.0), (T - 1.4, 1.0), (T - 0.8, 0.0)],
                    dim_until=T)
    return dict(corner=[(0.0, 0.0), (0.6, 1.0), (E - 0.6, 1.0), (E, 0.0)],
                end=[(E + 0.3, 0.0), (E + 0.9, 1.0), (T - 1.4, 1.0), (T - 0.8, 0.0)],
                dim_until=E)


def envelope(t, pts):
    if t < pts[0][0] or t > pts[-1][0]:
        return 0.0
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0) if t1 > t0 else v1
    return pts[-1][1]


# ---------------------------------------------------------------------------
# 描画（4Kで描き、1080pは4Kを1/2に縮める。どちらも同じ形になる）
# ---------------------------------------------------------------------------
def font_path(weight):
    for d in FONT_DIRS:
        p = os.path.join(d, 'NotoSansJP-%s.ttf' % weight)
        if os.path.exists(p):
            return p
    raise FileNotFoundError('NotoSansJP-%s.ttf' % weight)


_LOGO = None


def logo_trimmed():
    """ロゴPNGを、見えている円（不透明度>8/255）の外接矩形で切り出したもの。縦横比は変えない。"""
    global _LOGO
    if _LOGO is None:
        lg = Image.open(LOGO_PATH).convert('RGBA')
        _LOGO = lg.crop(lg.getchannel('A').point(lambda v: 255 if v > 8 else 0).getbbox())
    return _LOGO


def logo_alpha(h_px):
    lg = logo_trimmed()
    w_px = lg.width * h_px / lg.height
    return lg.resize((int(round(w_px)), int(round(h_px))), Image.LANCZOS).getchannel('A')


def text_alpha(s, weight, size_px, track_em, ss=4):
    """1文字ずつ字間を足して描いた文字のアルファ（インクの外接矩形で切り出し）。ss倍で描いて縮小。"""
    f = ImageFont.truetype(font_path(weight), int(round(size_px * ss)))
    adv = [f.getlength(ch) for ch in s]
    tr = track_em * size_px * ss
    asc, desc = f.getmetrics()
    m = Image.new('L', (int(sum(adv) + tr * len(s) + size_px * ss) + 80, asc + desc + 80), 0)
    d = ImageDraw.Draw(m)
    x = 40.0
    for ch, a in zip(s, adv):
        d.text((x, 40), ch, font=f, fill=255)
        x += a + tr
    capbb = f.getbbox('H')
    cap_top, cap_bot = 40 + capbb[1], 40 + capbb[3]
    bb = m.getbbox()
    # 縮小後に位置がずれないよう、ss の倍数に広げて切る
    x0 = bb[0] // ss * ss
    y0 = bb[1] // ss * ss
    x1 = -(-bb[2] // ss) * ss
    y1 = -(-bb[3] // ss) * ss
    m = m.crop((x0, y0, x1, y1))
    small = m.resize((m.width // ss, m.height // ss), Image.BOX)
    cap_mid = ((cap_top + cap_bot) / 2 - y0) / ss       # 縮小後の座標での大文字の中心
    ink = ((bb[0] - x0) / ss, (bb[1] - y0) / ss, (bb[2] - x0) / ss, (bb[3] - y0) / ss)
    return small, cap_mid, ink


def _paste_max(canvas, a, x, y):
    x, y = int(round(x)), int(round(y))
    arr = np.asarray(a).astype(np.float32) / 255.0
    H, W = canvas.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + arr.shape[1], W), min(y + arr.shape[0], H)
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], arr[y0 - y:y1 - y, x0 - x:x1 - x])


def layout(kind, W, H):
    """kind: 'corner' | 'end' | 'endlogo'。ink（白の不透明度、0〜1のfloat配列）と、各要素の箱（px）を返す。"""
    ink = np.zeros((H, W), np.float32)
    boxes = {}
    if kind == 'corner':
        c = CORNER
        la = logo_alpha(c['logo_h'] * H)
        lx = W * (1 - c['right']) - la.width
        ly = H * (1 - c['bottom']) - la.height
        _paste_max(ink, la, lx, ly)
        boxes['logo'] = (lx, ly, lx + la.width, ly + la.height)
        ta, cap_mid, tb = text_alpha(c['line'], c['weight'], c['size'] * H, c['track'])
        tx = lx - c['gap'] * W - tb[2]
        ty = ly + la.height / 2 - cap_mid
        _paste_max(ink, ta, tx, ty)
        boxes['line'] = (tx + tb[0], ty + tb[1], tx + tb[2], ty + tb[3])
    elif kind == 'end':
        e = END
        ta, cap_mid, tb = text_alpha(e['line'], e['weight'], e['size'] * H, e['track'])
        tx = W / 2 - (tb[0] + tb[2]) / 2
        ty = e['center_y'] * H - (tb[1] + tb[3]) / 2
        _paste_max(ink, ta, tx, ty)
        boxes['line'] = (tx + tb[0], ty + tb[1], tx + tb[2], ty + tb[3])
    elif kind == 'endlogo':
        e = END_LOGO
        la = logo_alpha(e['logo_h'] * H)
        ta, cap_mid, tb = text_alpha(e['line'], e['weight'], e['size'] * H, e['track'])
        total = la.width + e['gap'] * W + (tb[2] - tb[0])
        lx = (W - total) / 2
        ly = e['center_y'] * H - la.height / 2
        _paste_max(ink, la, lx, ly)
        boxes['logo'] = (lx, ly, lx + la.width, ly + la.height)
        tx = lx + la.width + e['gap'] * W - tb[0]
        ty = e['center_y'] * H - (tb[1] + tb[3]) / 2
        _paste_max(ink, ta, tx, ty)
        boxes['line'] = (tx + tb[0], ty + tb[1], tx + tb[2], ty + tb[3])
    else:
        raise ValueError(kind)
    return ink, boxes


def shadow_of(ink, H):
    """2層の柔らかい影の不透明度（0〜1）。"""
    img = Image.fromarray((ink * 255).round().astype(np.uint8))
    sh = np.zeros_like(ink)
    for sigma, op in SHADOW:
        b = np.asarray(img.filter(ImageFilter.GaussianBlur(sigma * H))).astype(np.float32) / 255.0
        sh = 1 - (1 - sh) * (1 - np.clip(b * op, 0, 1))
    return sh


_CACHE = {}


def planes(kind, W, H):
    """(ink, shadow, boxes)。4Kで描いてから必要なら縮小する。"""
    key = (kind, W, H)
    if key in _CACHE:
        return _CACHE[key]
    if (W, H) == (3840, 2160) or W > 1920:
        ink, boxes = layout(kind, W, H)
        sh = shadow_of(ink, H)
    else:
        k = 3840 / W
        ink4, boxes4 = planes(kind, 3840, 2160)[0], planes(kind, 3840, 2160)[2]
        sh4 = planes(kind, 3840, 2160)[1]
        ink = np.asarray(Image.fromarray(ink4).resize((W, H), Image.BOX)).astype(np.float32)
        sh = np.asarray(Image.fromarray(sh4).resize((W, H), Image.BOX)).astype(np.float32)
        boxes = {n: tuple(v / k for v in b) for n, b in boxes4.items()}
    _CACHE[key] = (ink, sh, boxes)
    return _CACHE[key]


def opacity_of(kind):
    return {'corner': CORNER, 'end': END, 'endlogo': END_LOGO}[kind]['opacity']


def rgba(kind, W, H):
    """CapCutに置く透過PNG（白の文字・ロゴ＋黒の影、ストレートアルファ）。"""
    ink, sh, _ = planes(kind, W, H)
    op = opacity_of(kind)
    a_ink = ink * op
    a = a_ink + sh * op * (1 - a_ink)
    rgb = np.where(a > 1e-6, a_ink / np.maximum(a, 1e-6), 0) * 255
    return Image.fromarray(np.dstack([rgb, rgb, rgb, a * 255]).round().clip(0, 255).astype(np.uint8), 'RGBA')


def comp(frame, kind, opacity=1.0):
    """frame（RGB、sRGB）に重ねる。CapCutの通常合成と同じ（sRGBの値で線形補間）。"""
    if opacity <= 0:
        return frame
    W, H = frame.size
    ink, sh, _ = planes(kind, W, H)
    op = opacity_of(kind) * opacity
    rgb = np.asarray(frame).astype(np.float32)
    rgb = rgb * (1 - (sh * op)[..., None])
    rgb = rgb * (1 - (ink * op)[..., None]) + 255 * (ink * op)[..., None]
    return Image.fromarray(rgb.round().clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# 減光（クリップに焼き込む）
# ---------------------------------------------------------------------------
def corner_group_center(W, H):
    _, _, b = planes('corner', W, H)
    return (b['line'][0] + b['logo'][2]) / 2, (b['logo'][1] + b['logo'][3]) / 2


def dim_matte(W, H):
    """楕円のマスク k（0〜1）。"""
    cx, cy = corner_group_center(W, H)
    m = Image.new('L', (W, H), 0)
    ImageDraw.Draw(m).ellipse([cx - DIM['w'] * W / 2, cy - DIM['h'] * H / 2,
                               cx + DIM['w'] * W / 2, cy + DIM['h'] * H / 2], fill=255)
    m = m.filter(ImageFilter.GaussianBlur(DIM['feather'] * H))
    return np.asarray(m).astype(np.float32) / 255.0


def _to_lin(a):
    a = a / 255.0
    return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def _to_srgb(l):
    l = np.clip(l, 0, 1)
    return np.where(l <= 0.0031308, l * 12.92, 1.055 * l ** (1 / 2.4) - 0.055) * 255


def apply_dim(frame, ev):
    """リニア光で露出を ev だけ下げる（楕円の中心で最大）。"""
    if not ev:
        return frame
    W, H = frame.size
    k = dim_matte(W, H)[..., None]
    lin = _to_lin(np.asarray(frame).astype(np.float32))
    lin = lin * (1 - k + k * (2.0 ** ev))
    return Image.fromarray(_to_srgb(lin).round().clip(0, 255).astype(np.uint8))


def report(W=1920, H=1080):
    for kind in ('corner', 'end', 'endlogo'):
        _, _, b = planes(kind, W, H)
        for n, (x0, y0, x1, y1) in b.items():
            print('%-8s %-5s x %7.1f-%7.1f (%.1f-%.1f%%w)  y %6.1f-%6.1f (%.1f-%.1f%%h)  %dx%d px  center (%.0f, %.0f)' % (
                kind, n, x0, x1, x0 / W * 100, x1 / W * 100, y0, y1, y0 / H * 100, y1 / H * 100,
                round(x1 - x0), round(y1 - y0), (x0 + x1) / 2, (y0 + y1) / 2))
    cx, cy = corner_group_center(W, H)
    print('dim ellipse center (%.1f%%w, %.1f%%h)' % (cx / W * 100, cy / H * 100))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == '--overlays':
        d = sys.argv[2]
        os.makedirs(d, exist_ok=True)
        for (W, H) in ((3840, 2160), (1920, 1080)):
            for kind in ('corner', 'end', 'endlogo'):
                p = os.path.join(d, 'signage_%s_%dx%d.png' % (kind, W, H))
                rgba(kind, W, H).save(p)
                print(p)
        return
    report()


if __name__ == '__main__':
    main()
