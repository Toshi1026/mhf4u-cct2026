#!/usr/bin/env python3
"""サイネージ2本（30秒版・15秒版）で共通にする「右下のマーク」「END」「右下の減光」「出方」の数値表と描画。

この表が唯一の正（2026-09-23 確定版）。30秒版・15秒版のEDLはここを参照し、
モックアップも、CapCutに置く透過PNGも、このファイルから作る。値を変えるときは2本同時に変わる。

2026-09-23 のユーザーの決定：ENDは③（「代案A＋店名の行」）を既定にする。
  右下のマークはENDのカットに入る直前（E−0.6→E秒）に消え、ENDでは海の上の中央に
  正式ロゴ＋「MAZE WIND~RETREAT」＋「高知県土佐市」の1組を出す。既定案（右下を常時表示し、
  中央は「高知県土佐市」だけ）と代案A（ロゴ＋「高知県土佐市」だけ）は検討のうえ不採用にしたので、
  このファイルには持たない。

  python3 edl/mockups/signage_spec.py --overlays DIR
      CapCutに置く透過PNGを DIR に書き出す（1920x1080 と 3840x2160 の両方）。
        signage_corner_<W>x<H>.png   右下のマーク（正式ロゴ＋「MAZE WIND~RETREAT」＋影）
        signage_end_<W>x<H>.png      END（正式ロゴ＋「MAZE WIND~RETREAT」＋「高知県土佐市」＋影）
      CapCutでは位置0・拡大100%・回転0で置くだけ。文字機能・影機能は使わない。
  python3 edl/mockups/signage_spec.py --report
      1080pでの位置（px・%）と、出方の時刻（30秒版・15秒版）を表示する。

減光（右下の楕円）はPNGにせず、Claudeが書き出す各カットのクリップに焼き込む（素材ごとの強さは DIM_EV）。
大きさ・位置はすべて画面の高さ h・幅 w に対する割合なので、1080pでも4Kでも同じ見え方になる。
ロゴは assets/logo_white.png だけを使う（縦横比はファイルのまま。加工した版は使わない）。
"""
import hashlib
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOGO_PATH = os.path.join(ROOT, 'assets', 'logo_white.png')
FONT_DIRS = [os.path.join(ROOT, 'assets', 'fonts'), '/tmp/claude-0/fonts', os.path.expanduser('~/.fonts'),
             '/usr/share/fonts/truetype/noto', '/usr/share/fonts/opentype/noto']
FONT_SHA256 = {   # assets/fonts/ の Noto Sans JP（fontsource 5.3.0、Version 2.004。IGの2本と同じファイル）
    'Light': '10387105d854fd95d8a7e9e78e42a17947aad2c9057f958a88ceceb597f85d2b',
    'Regular': '25d059a3d0fac5de6c5b0d1fa7c32c7ce5d85befca1752ef2eec8be9a2345158',
    'Medium': '29f6b29939a779e33601cbbe05f158e4b1568809eaf46b5e8d43cb7092e82412',
}

# ---------------------------------------------------------------------------
# 共通の数値表（h＝画面の高さ、w＝画面の幅）
# ---------------------------------------------------------------------------
# 影：黒・ずらしなし。(ガウスのσ[h比], 不透明度) の2層。縁取り・帯・グラデーションは使わない
SHADOW = [(0.003, 0.65), (0.012, 0.30)]           # 文字（右下の店名の行・ENDの2行）
SHADOW_LOGO = [(0.0012, 0.40), (0.0060, 0.22)]    # ロゴ（IGの2本と同じ値。4本で共通）

CORNER = dict(                 # 右下のマーク
    logo_h=0.090,              # ロゴの見えている部分の高さ 9.0%h（1080pで97px、幅101px）
    right=0.050,               # ロゴの右端：画面の右端から 5.0%w 内側
    bottom=0.050,              # ロゴの下端：画面の下端から 5.0%h 内側
    line='MAZE WIND~RETREAT',  # 半角チルダ U+007E、前後に空白なし
    weight='Regular',
    size=0.022,                # 文字サイズ（1em）2.2%h（大文字の高さは約1.6%h＝1080pで約17px）
    track=0.15,                # 字間 0.15em
    gap=0.010,                 # 行の右端（インク）〜ロゴの左端 1.0%w
    opacity=1.0,
)

END = dict(                    # END（③：正式ロゴの右に、店名と所在地の2行を左そろえで積んだ1組を中央に置く）
    logo_h=0.160,              # ロゴの見えている部分の高さ 16%h（1080pで173px）
    gap=0.018,                 # ロゴの右端〜2行の左端 1.8%w
    center_y=0.530,            # ロゴと2行のブロックの縦の中心 53%h（水平線41%の下の海の上）
    line1=dict(text='MAZE WIND~RETREAT', weight='Regular', size=0.036, track=0.15),  # 大文字の高さ約2.6%h
    line2=dict(text='高知県土佐市', weight='Regular', size=0.027, track=0.30),
    leading=0.020,             # 1行目の下端〜2行目の上端（インク）2.0%h
    opacity=1.0,
    # 予備案（運営側の回答で、歩行者が2〜3mで立ち止まって見る場所だと分かった場合だけ）：
    # 「高知県土佐市」の下に同じ行間で3行目を足す。既定では描かない（enabled=False）。
    handle=dict(enabled=False, text='@mazewind2026', weight='Regular', size=0.022, track=0.15),
)

DIM = dict(                    # 右下の減光（楕円、リニア光で露出を下げる。各クリップに焼き込む）
    w=0.30, h=0.22,            # 楕円の幅 30%w × 高さ 22%h。中心は右下のマーク全体（行の左端〜ロゴの右端、ロゴの中心の高さ）
    feather=0.040,             # 境目のぼかし σ 4.0%h
)
DIM_EV = {                     # 素材ごとの強さ（同じ素材は2本とも同じ値）。ENDのクリップには減光をかけない
    'IMG_9631': -0.70,         # 白波の帯（いちばん明るい背景。30秒版のカット1）
    'IMG_9665': -0.50,         # 河口の砂州・水路
    'IMG_9674': -0.30,         # 店内（暗い）
    'FOOD': -0.30,             # 料理＋水平線（撮影後に実測して決める仮の値）
}

# 右下の空き：料理の撮影で、皿・グラス・手を置かない範囲（右下のマークと、その影・減光がかかる範囲）
RESERVED_FOOD = dict(x_from=0.68, y_from=0.79)   # 横68%より右・縦79%より下

# 出方の決まり（E＝ENDのカットの頭、T＝ファイルの長さ。30秒版は E=20.70・T=30.00、15秒版は E=9.00・T=15.00）
FADE = 0.6                     # フェードの長さ（秒）。どれも直線
TIMING_RULE = dict(
    corner='0→0.6 でフェードイン、E−0.6→E でフェードアウト（ENDのカットに入る前に消しきる）',
    dim='0〜E のクリップだけに焼き込む（ENDのクリップにはかけない）',
    end='E+0.3→E+0.9 でフェードイン、T−1.4→T−0.8 でフェードアウト（最後の0.8秒は海だけでループの頭へ）',
)


def timing(E, T):
    """不透明度のキーフレーム [(秒, 0〜1), ...] を返す。30秒版と15秒版で同じ式。"""
    return dict(corner=[(0.0, 0.0), (FADE, 1.0), (E - FADE, 1.0), (E, 0.0)],
                end=[(E + 0.3, 0.0), (E + 0.3 + FADE, 1.0), (T - 0.8 - FADE, 1.0), (T - 0.8, 0.0)],
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
_FONT_CHECKED = set()


def font_path(weight):
    for d in FONT_DIRS:
        p = os.path.join(d, 'NotoSansJP-%s.ttf' % weight)
        if os.path.exists(p):
            if p not in _FONT_CHECKED and weight in FONT_SHA256:
                _FONT_CHECKED.add(p)
                with open(p, 'rb') as fh:
                    if hashlib.sha256(fh.read()).hexdigest() != FONT_SHA256[weight]:
                        print('警告：%s は承認時と別の版の書体です（字幅が変わるおそれ）' % p, file=sys.stderr)
            return p
    raise FileNotFoundError('NotoSansJP-%s.ttf' % weight)


_LOGO = None


def logo_trimmed():
    """ロゴPNGを、見えている部分（不透明度>8/255）の外接矩形で切り出したもの。縦横比は変えない。"""
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
    """kind: 'corner' | 'end'。ロゴのインクと文字のインク（白の不透明度、0〜1のfloat配列）を別々に返す。
    戻り値：(ink_logo, ink_text, boxes)。boxes は各要素のインクの外接枠（px）。"""
    ink_l = np.zeros((H, W), np.float32)
    ink_t = np.zeros((H, W), np.float32)
    boxes = {}
    if kind == 'corner':
        c = CORNER
        la = logo_alpha(c['logo_h'] * H)
        lx = W * (1 - c['right']) - la.width
        ly = H * (1 - c['bottom']) - la.height
        _paste_max(ink_l, la, lx, ly)
        boxes['logo'] = (lx, ly, lx + la.width, ly + la.height)
        ta, cap_mid, tb = text_alpha(c['line'], c['weight'], c['size'] * H, c['track'])
        tx = lx - c['gap'] * W - tb[2]
        ty = ly + la.height / 2 - cap_mid
        _paste_max(ink_t, ta, tx, ty)
        boxes['line'] = (tx + tb[0], ty + tb[1], tx + tb[2], ty + tb[3])
    elif kind == 'end':
        p = END
        la = logo_alpha(p['logo_h'] * H)
        rows = [p['line1'], p['line2']] + ([p['handle']] if p['handle'].get('enabled') else [])
        drawn = [text_alpha(r['text'], r['weight'], r['size'] * H, r['track']) for r in rows]
        hs = [tb[3] - tb[1] for _, _, tb in drawn]
        block_h = sum(hs) + p['leading'] * H * (len(rows) - 1)
        block_w = max(tb[2] - tb[0] for _, _, tb in drawn)
        total = la.width + p['gap'] * W + block_w
        lx = (W - total) / 2
        ly = p['center_y'] * H - la.height / 2
        _paste_max(ink_l, la, lx, ly)
        boxes['logo'] = (lx, ly, lx + la.width, ly + la.height)
        x_text = lx + la.width + p['gap'] * W
        y = p['center_y'] * H - block_h / 2
        for i, ((ta, _, tb), h) in enumerate(zip(drawn, hs)):
            _paste_max(ink_t, ta, x_text - tb[0], y - tb[1])
            boxes['line%d' % (i + 1)] = (x_text, y, x_text + tb[2] - tb[0], y + h)
            y += h + p['leading'] * H
    else:
        raise ValueError(kind)
    return ink_l, ink_t, boxes


def _shadow(ink, H, layers):
    img = Image.fromarray((ink * 255).round().astype(np.uint8))
    sh = np.zeros_like(ink)
    for sigma, op in layers:
        b = np.asarray(img.filter(ImageFilter.GaussianBlur(sigma * H))).astype(np.float32) / 255.0
        sh = 1 - (1 - sh) * (1 - np.clip(b * op, 0, 1))
    return sh


def shadow_of(ink_logo, ink_text, H):
    """ロゴには SHADOW_LOGO、文字には SHADOW を別々にかけ、重ねた影の不透明度（0〜1）。"""
    a = _shadow(ink_logo, H, SHADOW_LOGO)
    b = _shadow(ink_text, H, SHADOW)
    return 1 - (1 - a) * (1 - b)


_CACHE = {}


def planes(kind, W, H):
    """(ink, shadow, boxes, ink_logo, ink_text)。4Kで描いてから必要なら縮小する。"""
    key = (kind, W, H)
    if key in _CACHE:
        return _CACHE[key]
    if W >= 3840:
        il, it, boxes = layout(kind, W, H)
        sh = shadow_of(il, it, H)
    else:
        k = 3840 / W
        _, sh4, boxes4, il4, it4 = planes(kind, 3840, 2160)

        def down(a):
            return np.asarray(Image.fromarray(a).resize((W, H), Image.BOX)).astype(np.float32)
        il, it, sh = down(il4), down(it4), down(sh4)
        boxes = {n: tuple(v / k for v in b) for n, b in boxes4.items()}
    ink = np.maximum(il, it)
    _CACHE[key] = (ink, sh, boxes, il, it)
    return _CACHE[key]


def opacity_of(kind):
    return {'corner': CORNER, 'end': END}[kind]['opacity']


def rgba(kind, W, H):
    """CapCutに置く透過PNG（白の文字・ロゴ＋黒の影、ストレートアルファ）。"""
    ink, sh = planes(kind, W, H)[:2]
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
    ink, sh = planes(kind, W, H)[:2]
    op = opacity_of(kind) * opacity
    rgb = np.asarray(frame).astype(np.float32)
    rgb = rgb * (1 - (sh * op)[..., None])
    rgb = rgb * (1 - (ink * op)[..., None]) + 255 * (ink * op)[..., None]
    return Image.fromarray(rgb.round().clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# 読みやすさの測定（4本で同じ測り方。IGの2本の contrast() と同じ）
# WCAGの相対輝度で、字の芯（アルファ≥0.9）と、字のすぐ外1〜3px（アルファ<0.02）の平均の比。影込み。
# ---------------------------------------------------------------------------
def rel_lum(img):
    a = np.asarray(img.convert('RGB')).astype(np.float32) / 255.0
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def contrast(composited, alpha, box, pad=6):
    """composited：重ねたあとの画。alpha：その要素だけのインク（0〜1のfloat、画面と同じ大きさ）。box：外接枠。"""
    Hh, Ww = alpha.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, Ww), min(y1 + pad, Hh)
    Y = rel_lum(composited)[y0:y1, x0:x1]
    a = alpha[y0:y1, x0:x1]
    core = a >= 0.9
    near = np.asarray(Image.fromarray(((a > 0.5) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))) > 0
    ring = near & (a < 0.02)
    if core.sum() == 0 or ring.sum() == 0:
        return float('nan')
    return (float(Y[core].mean()) + 0.05) / (float(Y[ring].mean()) + 0.05)


def outer_ring_masks(il, core_min=0.6):
    """ロゴの外周の線（外縁から内側4px以内のインク）と、そのすぐ外1〜3px の画素のマスク。"""
    m = Image.fromarray(((il > 0.5) * 255).astype(np.uint8)).copy()
    ImageDraw.floodfill(m, (0, 0), 128)
    disk = np.asarray(m) != 128
    d = Image.fromarray((disk * 255).astype(np.uint8))
    inner = np.asarray(d.filter(ImageFilter.MinFilter(9))) > 0
    grown = np.asarray(d.filter(ImageFilter.MaxFilter(7))) > 0
    return (il >= core_min) & ~inner, grown & ~disk & (il < 0.02)


def contrast_masks(composited, core, ring):
    Y = rel_lum(composited)
    return (float(Y[core].mean()) + 0.05) / (float(Y[ring].mean()) + 0.05)


def measure(composited, kind):
    """kind の各要素のコントラスト比を dict で返す（重ねたあとの画で測る）。
    logo_outer：ロゴの外周の線とそのすぐ外（「円のまわり」）。logo：ロゴの線全体とそのすぐ外。line*：各行。"""
    W, H = composited.size
    _, _, b, il, it = planes(kind, W, H)
    out = {'logo_outer': contrast_masks(composited, *outer_ring_masks(il)),
           'logo': contrast(composited, il, b['logo'])}
    for n in b:
        if n.startswith('line'):
            m = np.zeros_like(it)
            x0, y0, x1, y1 = [int(round(v)) for v in b[n]]
            sl = (slice(max(y0 - 2, 0), y1 + 2), slice(max(x0 - 2, 0), x1 + 2))
            m[sl] = it[sl]
            out[n] = contrast(composited, m, b[n])
    return out


# ---------------------------------------------------------------------------
# 減光（クリップに焼き込む）
# ---------------------------------------------------------------------------
def corner_group_center(W, H):
    b = planes('corner', W, H)[2]
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
    for kind in ('corner', 'end'):
        b = planes(kind, W, H)[2]
        for n, (x0, y0, x1, y1) in b.items():
            print('%-7s %-6s x %7.1f-%7.1f (%.1f-%.1f%%w)  y %6.1f-%6.1f (%.1f-%.1f%%h)  %dx%d px' % (
                kind, n, x0, x1, x0 / W * 100, x1 / W * 100, y0, y1, y0 / H * 100, y1 / H * 100,
                round(x1 - x0), round(y1 - y0)))
    cx, cy = corner_group_center(W, H)
    print('dim ellipse center (%.1f%%w, %.1f%%h)' % (cx / W * 100, cy / H * 100))
    for name, E, T in (('30秒版', 20.70, 30.00), ('15秒版', 9.00, 15.00)):
        tm = timing(E, T)
        print(name, 'corner', [(round(a, 2), b) for a, b in tm['corner']],
              'end', [(round(a, 2), b) for a, b in tm['end']], 'dim 0〜%.2f' % E)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == '--overlays':
        d = sys.argv[2]
        os.makedirs(d, exist_ok=True)
        for (W, H) in ((3840, 2160), (1920, 1080)):
            for kind in ('corner', 'end'):
                p = os.path.join(d, 'signage_%s_%dx%d.png' % (kind, W, H))
                rgba(kind, W, H).save(p)
                print(p)
        return
    report()


if __name__ == '__main__':
    main()
