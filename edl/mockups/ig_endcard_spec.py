#!/usr/bin/env python3
"""Instagramの2本（IG30・IG15）で共通にする End Card の数値表と描画（2026-09-23 確定版）。

この表が唯一の正。ig30_render.py と ig15_render.py はここを読み込み、自分の画に合わせた「配置」だけを持つ。
値を変えるときは2本同時に変わる（サイネージ2本の signage_spec.py と同じ考え方）。

2026-09-23 のユーザーの決定：
  ・End CardにMAP（高知県の輪郭・土佐市の点）は入れない。正式ロゴ＋「高知県土佐市」＋「@mazewind2026」だけ。
  ・ロゴのファイル・大きさ（見えている部分の高さ14.5%h）、文字の大きさ・太さ・字間・不透明度、影、
    出る順番とフェードの長さ、セーフゾーンを2本で同じにする。
ロゴは assets/logo_white.png だけを使う（縦横比はファイルのまま。加工した版は使わない）。

書体：assets/fonts/ の Noto Sans JP（Google / SIL Open Font License 1.1、ライセンス文は assets/fonts/OFL.txt）。
  fontsource @fontsource/noto-sans-jp 5.3.0 の静的ウェイト（Version 2.004-H2）。読み込むたびにハッシュを確かめる。
"""
import hashlib
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LOGO = os.path.join(ROOT, 'assets', 'logo_white.png')
FONT_DIRS = [os.path.join(ROOT, 'assets', 'fonts'), '/tmp/claude-0/fonts', os.path.expanduser('~/.fonts'),
             '/usr/share/fonts/truetype/noto']
FONT_SHA256 = {
    'Light': '10387105d854fd95d8a7e9e78e42a17947aad2c9057f958a88ceceb597f85d2b',
    'Regular': '25d059a3d0fac5de6c5b0d1fa7c32c7ce5d85befca1752ef2eec8be9a2345158',
    'Medium': '29f6b29939a779e33601cbbe05f158e4b1568809eaf46b5e8d43cb7092e82412',
}
W, H = 1080, 1920

# ---------------------------------------------------------------------------
# 共通の数値表（h＝画面の高さ1920、w＝画面の幅1080 に対する割合）
# ---------------------------------------------------------------------------
PLACE = '高知県土佐市'
HANDLE = '@mazewind2026'
TEXT = dict(
    weight='Regular',
    size=0.0200,                                  # 1em＝2.0%h（1920pで38px。int(round(38.4))=38）
    opacity=0.95,                                 # 白95%（影も含めて）
    track_ja=0.20,                                # 「高知県土佐市」の字間 0.20em
    track_en=0.06,                                # 「@mazewind2026」の字間 0.06em
    row_pitch=0.034,                              # 2行の中心の間隔 3.4%h（65px）
    shadow=[(0.0016, 0.45), (0.0065, 0.22)],      # 黒・ずらしなし：σ3.1px 45% ＋ σ12.5px 22%
)
LOGO_SPEC = dict(
    h=0.145,                                      # 見えている部分（透明な余白を除いた外接枠）の高さ 14.5%h（278px、幅289px）
    opacity=1.00,                                 # 白100%
    shadow=[(0.0012, 0.40), (0.0060, 0.22)],      # 黒・ずらしなし：σ2.3px 40% ＋ σ11.5px 22%（サイネージのロゴも同じ値）
)

# 出方：2段。①「高知県土佐市」0.6秒のフェードイン → ② ロゴと「@mazewind2026」を同時に0.5秒のフェードイン。
# ②は①が出きった瞬間に始まる。直線、イージングなし。最後まで出したまま（フェードアウトしない）。
STEP1_FADE = 0.6
STEP2_FADE = 0.5
STAGE_PARTS = dict(s1=('place',), s2=('logo', 'handle'))
STAGE_FILES = dict(s1='1_place', s2='2_logo_handle')


def timing(start, end):
    """start：①の始まり（タイムラインの秒）、end：動画の終わり。"""
    s2 = start + STEP1_FADE
    return dict(s1=[(start, 0.0), (s2, 1.0), (end, 1.0)],
                s2=[(s2, 0.0), (s2 + STEP2_FADE, 1.0), (end, 1.0)])


# セーフゾーン（2本共通）：上14%（y≥269）、下35%（y≤1248。広告として配信する場合）、左右6%（x 65〜1015）。
# 参考として、Reelsの下20%（キャプション）と右のボタン列（x≥950、縦50〜80%）も描く。
SAFE = dict(top=0.14, bottom=0.65, side=0.06, caption=0.80, buttons_x=0.88, buttons_y=(0.50, 0.80))


def safe_box():
    return (int(round(SAFE["side"] * W)), int(np.ceil(SAFE["top"] * H - 1e-6)),
            int(round((1 - SAFE['side']) * W)), int(round(SAFE['bottom'] * H)))


def envelope(t, pts):
    if t < pts[0][0]:
        return 0.0
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0) if t1 > t0 else v1
    return pts[-1][1]


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


# ---------------------------------------------------------------------------
# 部品
# ---------------------------------------------------------------------------
def text_mask(s, size_px, track_em, weight=None):
    """1文字ずつ字間を足して描いた文字のマスク（インクの外接枠で切り出し）。"""
    f = ImageFont.truetype(font_path(weight or TEXT['weight']), int(round(size_px)))
    tr = track_em * size_px
    asc, desc = f.getmetrics()
    wtot = sum(f.getlength(c) for c in s) + tr * (len(s) - 1)
    m = Image.new('L', (int(wtot) + 80, asc + desc + 80), 0)
    d = ImageDraw.Draw(m)
    x = 40.0
    for c in s:
        d.text((x, 40), c, font=f, fill=255)
        x += f.getlength(c) + tr
    return m.crop(m.getbbox())


def place_mask():
    return text_mask(PLACE, TEXT['size'] * H, TEXT['track_ja'])


def handle_mask():
    return text_mask(HANDLE, TEXT['size'] * H, TEXT['track_en'])


def handle_top(row_center):
    """「@mazewind2026」は、x-heightの中心を行の中心に合わせる（「@」の下へのはみ出しで上にずれないように）。"""
    f = ImageFont.truetype(font_path(TEXT['weight']), int(round(TEXT['size'] * H)))
    bx = f.getbbox('x')
    bb = f.getbbox(HANDLE)
    return row_center - ((bx[1] + bx[3]) / 2 - bb[1])


def logo_alpha(h_px=None):
    """見えている部分（不透明度>8/255の外接枠）の高さを h_px にした白のアルファ。縦横比はファイルのまま。"""
    h_px = h_px or LOGO_SPEC['h'] * H
    a = Image.open(LOGO).convert('RGBA').getchannel('A')
    a = a.crop(a.point(lambda v: 255 if v > 8 else 0).getbbox())
    return a.resize((int(round(h_px * a.width / a.height)), int(round(h_px))), Image.LANCZOS)


def full(mask, xy):
    a = Image.new('L', (W, H), 0)
    x, y = int(round(xy[0])), int(round(xy[1]))
    a.paste(mask, (x, y))
    return a, (x, y, x + mask.width, y + mask.height)


def shadowed(alpha, shadows, opacity):
    """白（alpha）＋黒・ずらしなしの柔らかい影。影も含めて opacity を掛ける。ストレートアルファのRGBA。"""
    a = np.asarray(alpha).astype(np.float32) / 255.0
    sh = np.zeros_like(a)
    for sigma, op in shadows:
        b = np.asarray(alpha.filter(ImageFilter.GaussianBlur(sigma * H))).astype(np.float32) / 255.0
        sh = 1 - (1 - sh) * (1 - np.clip(b * op, 0, 1))
    ink = a * opacity
    out_a = ink + sh * opacity * (1 - ink)
    rgb = (ink / np.maximum(out_a, 1e-6)) * 255.0
    return Image.fromarray(np.dstack([rgb, rgb, rgb, out_a * 255.0]).clip(0, 255).round().astype(np.uint8), 'RGBA')


def stage_layers(parts):
    """parts：{'place': alpha, 'logo': alpha, 'handle': alpha}（どれも W x H の 'L'）。
    段ごとの透過PNG（RGBA）。ロゴにはロゴの影、文字には文字の影を別々に付けて重ねる。"""
    s1 = shadowed(parts['place'], TEXT['shadow'], TEXT['opacity'])
    s2 = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    s2.alpha_composite(shadowed(parts['logo'], LOGO_SPEC['shadow'], LOGO_SPEC['opacity']))
    s2.alpha_composite(shadowed(parts['handle'], TEXT['shadow'], TEXT['opacity']))
    return dict(s1=s1, s2=s2)


def comp(frame, ov, opacity):
    if opacity <= 0:
        return frame
    if opacity < 1:
        ov = ov.copy()
        ov.putalpha(ov.getchannel('A').point(lambda v: int(round(v * opacity))))
    out = frame.convert('RGBA')
    out.alpha_composite(ov)
    return out.convert('RGB')


def safezones(im):
    """確認用：上14%・下35%（広告配信時）・左右6%の外を半透明で塗り、Reelsの下20%と右のボタン列を点線で示す。"""
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    col = (255, 70, 40, 70)
    x0, y0, x1, y1 = safe_box()
    d.rectangle([0, 0, W, y0 - 1], fill=col)
    d.rectangle([0, y1 + 1, W, H], fill=col)
    d.rectangle([0, y0, x0 - 1, y1], fill=col)
    d.rectangle([x1 + 1, y0, W, y1], fill=col)
    d.rectangle([x0, y0, x1, y1], outline=(255, 255, 255, 190), width=2)
    yc = int(SAFE['caption'] * H)
    for x in range(0, W, 24):
        d.line([x, yc, x + 12, yc], fill=(255, 220, 0, 200), width=2)
    bx = int(SAFE['buttons_x'] * W)
    for y in range(int(SAFE['buttons_y'][0] * H), int(SAFE['buttons_y'][1] * H), 24):
        d.line([bx, y, bx, y + 12], fill=(255, 220, 0, 200), width=2)
    f = ImageFont.truetype(font_path('Medium'), 26)
    lab = (255, 255, 255, 235)
    d.text((24, y0 - 40), '上14%（y 269）：Reelsの上部UI', font=f, fill=lab)
    d.text((24, y1 + 10), '下35%（y 1248）：広告として配信するときの下のUI', font=f, fill=lab)
    d.text((24, yc + 10), '参考：下20%（キャプション・音源）', font=f, fill=(255, 220, 0, 235))
    d.text((bx - 10, int(SAFE['buttons_y'][0] * H) + 10), '参考：\nボタン列', font=f, fill=(255, 220, 0, 235),
           anchor='ra', align='right')
    d.text((x0 + 8, y0 + 10), '左右6%（x 65〜1015）', font=f, fill=lab)
    out = im.convert('RGBA')
    out.alpha_composite(ov)
    return out.convert('RGB')


# ---------------------------------------------------------------------------
# 測定（4本で同じ測り方：WCAGの相対輝度で、字の芯（alpha≥0.9）と、字のすぐ外1〜3px（alpha<0.02）の平均の比。影込み）
# ---------------------------------------------------------------------------
def rel_lum(img):
    a = np.asarray(img.convert('RGB')).astype(np.float32) / 255.0
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def contrast(composited, alpha, box, pad=6):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
    Y = rel_lum(composited)[y0:y1, x0:x1]
    a = np.asarray(alpha).astype(np.float32)[y0:y1, x0:x1] / 255.0
    core = a >= 0.9
    near = np.asarray(Image.fromarray(((a > 0.5) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))) > 0
    ring = near & (a < 0.02)
    return (float(Y[core].mean()) + 0.05) / (float(Y[ring].mean()) + 0.05)


def logo_regions(logo_alpha_full, box):
    """ロゴの中の「MAZE WIND」（上の弧の中央部）・「RETREAT」の箱（画面のpx）と、ロゴの外周のマスク。
    元PNG（970x933、見えている部分は x 11..959・y 12..923＝948x911）の座標から換算する。"""
    lx, ly, lx1, ly1 = box
    sx, sy = (lx1 - lx) / 948.0, (ly1 - ly) / 911.0

    def cv(x0, y0, x1, y1):
        return (lx + (x0 - 11) * sx, ly + (y0 - 12) * sy, lx + (x1 - 11) * sx, ly + (y1 - 12) * sy)
    il = np.asarray(logo_alpha_full).astype(np.float32) / 255.0
    m = Image.fromarray(((il > 0.5) * 255).astype(np.uint8)).copy()
    ImageDraw.floodfill(m, (0, 0), 128)
    disk = np.asarray(m) != 128
    d = Image.fromarray((disk * 255).astype(np.uint8))
    inner = np.asarray(d.filter(ImageFilter.MinFilter(9))) > 0
    grown = np.asarray(d.filter(ImageFilter.MaxFilter(7))) > 0
    return dict(maze_wind=cv(300, 55, 680, 175), retreat=cv(205, 668, 752, 760),
                outer_core=(il >= 0.6) & ~inner, outer_ring=grown & ~disk & (il < 0.02))


def outer_contrast(composited, reg):
    Y = rel_lum(composited)
    return (float(Y[reg['outer_core']].mean()) + 0.05) / (float(Y[reg['outer_ring']].mean()) + 0.05)
