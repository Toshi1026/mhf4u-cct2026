#!/usr/bin/env python3
"""Instagram 15秒版（EDL v6）End Card のモックアップと、CapCutに重ねる透過PNGを作るスクリプト。

実素材 IMG_9633（HDR/HLG）から指定の瞬間をトーンマップして1枚取り出し、
EDLの8番と同じ切り出し（900x1600、キーフレームで水平線に追従）→ 1080x1920 をかけ、
ロゴ・簡易MAP・文字を、EDLの数値（画面の高さ・幅に対する割合）どおりに重ねる。

  python3 edl/mockups/ig15_render.py                    # モックアップを edl/mockups/ に書き出す（位置とコントラストも表示）
  python3 edl/mockups/ig15_render.py --logo B           # ロゴの版を変えて書き出す（A／B／C。既定は C）
  python3 edl/mockups/ig15_render.py --overlays DIR --logo C
      # 承認後：CapCutに100%で重ねる透過PNG（1080x1920、出る段ごとに3枚）を DIR に書き出す。
      # ロゴの版はお店が選んだものを必ず指定する（お店の了承前には本番用に焼き込まない）。
  python3 edl/mockups/ig15_render.py --overlays DIR --logo C --no-text
      # 文字をCapCutのテキストで打ち直す場合だけ：輪郭アイコン・点・ロゴだけの3枚（*_notext）を書き出す。

ロゴの版（assets/。作り方は edl/mockups/logo_variants.py）：
  A  logo_white.png              今の白版。スキャンのまま（外側の輪が横に約4.0%長い楕円）、犬はネガ（明暗が逆）
  B  logo_white_round.png        Aを正円に戻した版（犬はネガのまま）
  C  logo_white_round_dogpos.png Bの犬の輪郭の内側だけ、明暗を元のロゴどおりにした版（推奨。モックアップの既定）

書体：assets/fonts/ の Noto Sans JP Regular（IG30 と同じファイル。sha256 25d059a3…5158）。
"""
import hashlib
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FOOT = os.path.join(ROOT, 'footage', '01_撮影素材')
LOGO_RAW = os.path.join(ROOT, 'footage', '_store', 'logo_raw.jpeg')
LOGOS = dict(
    A=(os.path.join(ROOT, 'assets', 'logo_white.png'), 'A：今の白版（スキャンのまま・犬はネガ）'),
    B=(os.path.join(ROOT, 'assets', 'logo_white_round.png'), 'B：正円に戻した版（犬はネガ）'),
    C=(os.path.join(ROOT, 'assets', 'logo_white_round_dogpos.png'), 'C：正円＋犬の明暗を元のロゴどおり'),
)
LOGO_DEFAULT = 'C'
OUT = os.path.join(ROOT, 'edl', 'mockups')
FONT_DIRS = [os.path.join(ROOT, 'assets', 'fonts'), '/tmp/claude-0/fonts', os.path.expanduser('~/.fonts'),
             '/usr/share/fonts/truetype/noto']
FONT_SHA256 = {
    'Regular': '25d059a3d0fac5de6c5b0d1fa7c32c7ce5d85befca1752ef2eec8be9a2345158',
    'Medium': '29f6b29939a779e33601cbbe05f158e4b1568809eaf46b5e8d43cb7092e82412',
}
FFMPEG = os.environ.get(
    'FFMPEG', '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
TONEMAP = ('zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,'
           'zscale=t=bt709:m=bt709:r=tv,format=yuv420p')
W, H = 1080, 1920

# ---------------------------------------------------------------------------
# End Card の共通仕様（IG15・IG30 で同じ値。h = 画面の高さ1920、w = 画面の幅1080 に対する割合）
# IG30 の ig30_render.py（v6作業版、2026-09-23 16:26時点）の TEXT_SIZE・TEXT_OPACITY・SHADOW_TEXT・SHADOW_LOGO・map と
# 同じ値にしてある。どちらかで値を変えたら、もう一方も同じ値に直す（2本同時に承認するため）。
# ---------------------------------------------------------------------------
COMMON = dict(
    weight='Regular',
    text_size=0.0200,                                  # 1em = 2.0%h（1920pで38px。int(round(38.4))=38）
    text_opacity=0.95,                                 # 文字・MAP（影も含めて）の不透明度
    track_ja=0.20,                                     # 字間：日本語 0.20em
    track_en=0.06,                                     # 字間：欧文 0.06em
    text_shadow=[(0.0016, 0.45), (0.0065, 0.22)],      # 黒・ずらしなし：σ3.1px 45% ＋ σ12.5px 22%
    logo_opacity=1.00,                                 # ロゴは白100%
    logo_shadow=[(0.0012, 0.40), (0.0060, 0.22)],      # 黒・ずらしなし：σ2.3px 40% ＋ σ11.5px 22%
    icon_h=0.023,                                      # 高知県の輪郭アイコンの高さ 2.3%h（44px）
    icon_stroke=0.0013,                                # 線の太さ 0.13%h（2.5px）
    dot_d=0.0055,                                      # 土佐市の点の直径 0.55%h（約10.5px）
    dot_clear=0.0013,                                  # 点のまわりの輪郭線を 0.13%h 切り欠く
    icon_gap=0.013,                                    # アイコン〜「高知県土佐市」 1.3%w（14px）
)

# IG15 だけの値（配置）
SPEC = dict(
    center_y=0.570,          # ロゴと文字の1組の縦の中心（上から57.0%h = 1094px。水平線は約45.4%）
    logo_h=0.145,            # 見えているエンブレム（透明な余白を除いた外接枠）の高さ 14.5%h（278px）。B・Cは正円なので幅も278px
    gap=0.045,               # エンブレムの右端〜文字の列の左端 4.5%w（49px）
    rows=(-0.017, +0.017),   # 2行の縦の中心：1組の中心から ∓1.7%h（2行の間隔 3.4%h = 65px。IG30 と同じ）
)

# 出る段（タイムラインの秒, 不透明度）。直線、イージングなし。15.00まで出したまま（フェードアウトしない）
STAGES = ('s1', 's2', 's3')
STAGE_PARTS = dict(s1=('icon', 'pref'), s2=('dot', 'city'), s3=('logo', 'handle'))
STAGE_NAMES = dict(s1='1_kochi', s2='2_tosa', s3='3_logo_handle')
TIMING = dict(
    s1=[(12.20, 0.0), (12.50, 1.0), (15.00, 1.0)],     # 高知県の輪郭 ＋「高知県」
    s2=[(12.50, 0.0), (12.80, 1.0), (15.00, 1.0)],     # 土佐市の点 ＋「土佐市」
    s3=[(12.80, 0.0), (13.30, 1.0), (15.00, 1.0)],     # ロゴ ＋「@mazewind2026」（同時に0.5秒）
)

# 高知県の輪郭（国土数値情報ベースの dataofjapan/land japan.geojson から本土部分を簡略化し、
# 浦戸湾・浦ノ内湾・須崎・宿毛の細かい出入りを落としたもの。経度, 緯度）。IG30 と同じ点列
KOCHI = [[133.583, 33.868], [133.646, 33.883], [133.747, 33.837], [133.839, 33.845], [133.913, 33.791],
         [133.96, 33.835], [134.032, 33.828], [134.057, 33.691], [134.174, 33.684], [134.184, 33.649],
         [134.149, 33.629], [134.193, 33.562], [134.306, 33.553], [134.208, 33.395], [134.177, 33.243],
         [133.934, 33.487], [133.729, 33.538], [133.574, 33.496], [133.35, 33.418], [133.343, 33.399],
         [133.231, 33.324], [133.263, 33.258], [133.17, 33.15], [133.094, 33.023], [133.04, 33.038],
         [133.011, 33.015], [133.009, 32.882], [133.02, 32.724], [132.936, 32.787], [132.88, 32.79],
         [132.794, 32.747], [132.711, 32.798], [132.631, 32.762], [132.691, 32.974], [132.619, 33.181],
         [132.695, 33.138], [132.775, 33.206], [132.793, 33.275], [132.9, 33.321], [132.806, 33.46],
         [133.015, 33.481], [133.063, 33.543], [133.077, 33.653], [133.116, 33.66], [133.193, 33.794],
         [133.236, 33.783], [133.277, 33.831], [133.497, 33.829], [133.539, 33.877]]
TOSA = (133.48, 33.458)      # 土佐市の海沿い（仁淀川河口の西）


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
# 素材（EDLの8番：IMG_9633 素材1.5〜4.5秒、等速、900x1600をキーフレームで動かして1.2倍）
# ---------------------------------------------------------------------------
def grab(sec, cache):
    out = os.path.join(cache, 'IMG_9633_%.4f.png' % sec)
    if not os.path.exists(out):
        subprocess.run([FFMPEG, '-v', 'error', '-ss', '%.4f' % sec, '-i', os.path.join(FOOT, 'IMG_9633.MOV'),
                        '-frames:v', '1', '-vf', TONEMAP, '-y', out], check=True)
    return Image.open(out).convert('RGB')


def cut8(t, cache):
    sec = 1.5 + (t - 12.0)
    lt = sec - 1.5
    x = 1950 - 38 * lt
    y = 90 + 50.667 * lt if lt < 1.5 else 166 + 25.333 * (lt - 1.5)
    src = grab(sec, cache)
    return src.transform((W, H), Image.AFFINE, (900 / W, 0, x, 0, 1600 / H, y), resample=Image.BICUBIC), sec, (x, y)


# ---------------------------------------------------------------------------
# 重ねるもの（すべて白。影は黒・ずらしなしのぼかしだけ。縁取り・グラデーション・座布団なし）
# ---------------------------------------------------------------------------
def text_mask(s, size_px, track_em, weight=None):
    """1文字ずつ字間を足して描いた文字のマスク（インクの外接枠で切り出し）と、各文字の送りの左端（切り出し後の座標）、字間(px)。"""
    f = ImageFont.truetype(font_path(weight or COMMON['weight']), int(round(size_px)))
    tr = track_em * size_px
    asc, desc = f.getmetrics()
    wtot = sum(f.getlength(c) for c in s) + tr * (len(s) - 1)
    m = Image.new('L', (int(wtot) + 80, asc + desc + 80), 0)
    d = ImageDraw.Draw(m)
    x = 40.0
    starts = []
    for c in s:
        starts.append(x)
        d.text((x, 40), c, font=f, fill=255)
        x += f.getlength(c) + tr
    bb = m.getbbox()
    return m.crop(bb), [v - bb[0] for v in starts], tr


def chaikin(P, n=2):
    for _ in range(n):
        R = []
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            R += [(0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]),
                  (0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1])]
        P = R
    return P


def icon_masks(h_px, stroke_px, dot_px, clear_px=0.0, ss=8):
    """高知県の輪郭（線）と土佐市の点を、8倍で描いて縮小（線の端をなめらかにする）。"""
    k = math.cos(math.radians(33.3))
    P = chaikin([(x * k, y) for x, y in KOCHI], 2)
    xs = [p[0] for p in P]
    ys = [p[1] for p in P]
    s = (h_px - stroke_px) / (max(ys) - min(ys))
    pad = stroke_px / 2 + 2
    wm = int(math.ceil((max(xs) - min(xs)) * s + 2 * pad))
    hm = int(math.ceil(h_px + 4))

    def tr(x, y):
        return ((x - min(xs)) * s + pad) * ss, ((max(ys) - y) * s + pad) * ss
    big = Image.new('L', (wm * ss, hm * ss), 0)
    pp = [tr(x, y) for x, y in P]
    d = ImageDraw.Draw(big)
    d.line(pp + pp[:2], fill=255, width=int(round(stroke_px * ss)), joint='curve')
    cx, cy = tr(TOSA[0] * k, TOSA[1])
    r = dot_px / 2 * ss
    rc = r + clear_px * ss
    d.ellipse([cx - rc, cy - rc, cx + rc, cy + rc], fill=0)
    outline = big.resize((wm, hm), Image.BOX)
    big = Image.new('L', (wm * ss, hm * ss), 0)
    ImageDraw.Draw(big).ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    return outline.crop(outline.getbbox()), big.resize((wm, hm), Image.BOX).crop(outline.getbbox())


def logo_mask(h_px, variant):
    """見えているエンブレム（不透明度>8/255の外接枠）の高さを h_px にした白のアルファ。縦横比はファイルのまま。"""
    a = Image.open(LOGOS[variant][0]).convert('RGBA').getchannel('A')   # 白一色なのでアルファだけ使う
    a = a.crop(a.point(lambda v: 255 if v > 8 else 0).getbbox())
    return a.resize((int(round(h_px * a.width / a.height)), int(round(h_px))), Image.LANCZOS)


def shadowed(alpha, shadows, opacity):
    """白（alpha）＋黒・ずらしなしの柔らかい影。影も含めて opacity を掛ける（IG30 と同じ式）。ストレートアルファのRGBA。"""
    a = np.asarray(alpha).astype(np.float32) / 255.0
    sh = np.zeros_like(a)
    for sigma, op in shadows:
        b = np.asarray(alpha.filter(ImageFilter.GaussianBlur(sigma * H))).astype(np.float32) / 255.0
        sh = 1 - (1 - sh) * (1 - np.clip(b * op, 0, 1))
    ink = a * opacity
    out_a = ink + sh * opacity * (1 - ink)
    rgb = (ink / np.maximum(out_a, 1e-6)) * 255.0
    return Image.fromarray(np.dstack([rgb, rgb, rgb, out_a * 255.0]).clip(0, 255).round().astype(np.uint8), 'RGBA')


def layout(variant=LOGO_DEFAULT, S=SPEC, C=COMMON):
    """各パーツの全画面アルファ（1080x1920の 'L'）と外接枠（px）を返す。"""
    ts = C['text_size'] * H
    lg = logo_mask(S['logo_h'] * H, variant)
    ol, dot = icon_masks(C['icon_h'] * H, C['icon_stroke'] * H, C['dot_d'] * H, C['dot_clear'] * H)
    loc, starts, tr = text_mask('高知県土佐市', ts, C['track_ja'])
    hdl, _, _ = text_mask('@mazewind2026', ts, C['track_en'])
    gap, ig = S['gap'] * W, C['icon_gap'] * W
    colw = max(ol.width + ig + loc.width, hdl.width)
    x0 = (W - (lg.width + gap + colw)) / 2          # 1組全体を左右中央に置く
    cy = S['center_y'] * H
    tx = x0 + lg.width + gap
    r1, r2 = cy + S['rows'][0] * H, cy + S['rows'][1] * H
    # 欧文の行は、x-heightの中心を行の中心に合わせる（「@」の下へのはみ出しで上にずれないように）
    f = ImageFont.truetype(font_path(C['weight']), int(round(ts)))
    bx = f.getbbox('x')
    bb = f.getbbox('@mazewind2026')
    hdl_top = r2 - ((bx[1] + bx[3]) / 2 - bb[1])
    P = dict(
        logo=(lg, (x0, cy - lg.height / 2)),
        icon=(ol, (tx, r1 - ol.height / 2)),
        dot=(dot, (tx, r1 - ol.height / 2)),
        loc=(loc, (tx + ol.width + ig, r1 - loc.height / 2)),
        handle=(hdl, (tx, hdl_top)),
    )
    full, boxes = {}, {}
    for k, (m, (x, y)) in P.items():
        x, y = int(round(x)), int(round(y))
        a = Image.new('L', (W, H), 0)
        a.paste(m, (x, y))
        full[k] = a
        boxes[k] = (x, y, x + m.width, y + m.height)
    # 「高知県土佐市」を1行で描いてから、「県」と「土」の間（字間の中央）で左右に切り分ける（位置・字間は1行のときと同じ）
    split = boxes['loc'][0] + int(round(starts[3] - tr / 2))
    la = np.asarray(full['loc'])
    pref, city = la.copy(), la.copy()
    pref[:, split:] = 0
    city[:, :split] = 0
    full['pref'], full['city'] = Image.fromarray(pref), Image.fromarray(city)
    boxes['pref'] = Image.fromarray(pref).getbbox()
    boxes['city'] = Image.fromarray(city).getbbox()
    boxes['split_x'] = split
    return full, boxes


def union(alphas):
    return Image.fromarray(np.max([np.asarray(a) for a in alphas], axis=0))


def stage_layers(variant=LOGO_DEFAULT, C=COMMON, no_text=False):
    """出る段ごとの透過PNG（1080x1920のRGBA）。s1・s2 は文字の影、s3 はロゴの影と文字の影を別々に付けて重ねる。
    no_text=True のときは文字（「高知県」「土佐市」「@mazewind2026」）を入れない（文字をCapCutで打ち直す場合用）。"""
    full, _ = layout(variant)
    L = {}
    for s in ('s1', 's2'):
        parts = [p for p in STAGE_PARTS[s] if not (no_text and p in ('pref', 'city'))]
        L[s] = shadowed(union([full[p] for p in parts]), C['text_shadow'], C['text_opacity'])
    s3 = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    s3.alpha_composite(shadowed(full['logo'], C['logo_shadow'], C['logo_opacity']))
    if not no_text:
        s3.alpha_composite(shadowed(full['handle'], C['text_shadow'], C['text_opacity']))
    L['s3'] = s3
    return L


def comp(frame, ov, opacity):
    if opacity <= 0:
        return frame
    if opacity < 1:
        ov = ov.copy()
        ov.putalpha(ov.getchannel('A').point(lambda v: int(round(v * opacity))))
    out = frame.convert('RGBA')
    out.alpha_composite(ov)
    return out.convert('RGB')


def render(t, L, cache):
    fr, sec, xy = cut8(t, cache)
    for s in STAGES:
        fr = comp(fr, L[s], envelope(t, TIMING[s]))
    return fr, sec, xy


# ---------------------------------------------------------------------------
# Reelsのセーフゾーン（確認用の半透明の帯）
# ---------------------------------------------------------------------------
def safezones(im):
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    col = (255, 70, 40, 80)
    d.rectangle([0, 0, W, int(0.14 * H) - 1], fill=col)                      # 上14%：プロフィール・タブ
    d.rectangle([0, int(0.80 * H), W, H], fill=col)                          # 下20%：ユーザー名・キャプション・音源
    d.rectangle([int(0.88 * W), int(0.50 * H), W, int(0.80 * H) - 1], fill=col)   # 右12%：いいね・コメント・シェア
    y35 = int(0.65 * H)
    for x in range(0, W, 24):                                                # 参考：広告として配信する場合の下35%ライン
        d.line([x, y35, x + 12, y35], fill=(255, 220, 0, 200), width=2)
    f = ImageFont.truetype(font_path('Medium'), 26)
    lab = (255, 255, 255, 235)
    d.text((24, int(0.14 * H) - 40), '上14%：Reelsの上部UI', font=f, fill=lab)
    d.text((24, int(0.80 * H) + 14), '下20%：ユーザー名・キャプション・音源', font=f, fill=lab)
    d.text((W - 24, int(0.50 * H) + 14), '右12%\nボタン列', font=f, fill=lab, anchor='ra', align='right')
    d.text((24, y35 + 8), '参考：広告配信時の下35%ライン（y=1248）', font=f, fill=(255, 220, 0, 235))
    out = im.convert('RGBA')
    out.alpha_composite(ov)
    return out.convert('RGB')


# ---------------------------------------------------------------------------
# 測定（WCAGの相対輝度。字の芯（alpha>=0.9）と、字のすぐ外1〜3px（alpha<0.02）の平均の比。IG30 と同じ方法）
# ---------------------------------------------------------------------------
def rel_lum(img):
    a = np.asarray(img.convert('RGB')).astype(np.float32) / 255.0
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def contrast(composited, alpha, box):
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = x0 - 6, y0 - 6, x1 + 6, y1 + 6
    Y = rel_lum(composited)[y0:y1, x0:x1]
    a = np.asarray(alpha).astype(np.float32)[y0:y1, x0:x1] / 255.0
    core = a >= 0.9
    near = np.asarray(Image.fromarray(((a > 0.5) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))) > 0
    ring = near & (a < 0.02)
    return (float(Y[core].mean()) + 0.05) / (float(Y[ring].mean()) + 0.05)


def report(cache, variant, L):
    full, boxes = layout(variant)
    for k in ('logo', 'icon', 'pref', 'city', 'handle'):
        x0, y0, x1, y1 = boxes[k]
        print('%-6s x %4d-%4d (%.1f-%.1f%%w)  y %4d-%4d (%.1f-%.1f%%h)  %dx%d' % (
            k, x0, x1, x0 / W * 100, x1 / W * 100, y0, y1, y0 / H * 100, y1 / H * 100, x1 - x0, y1 - y0))
    xs = [boxes[k][0] for k in ('logo', 'icon', 'handle')] + [boxes[k][2] for k in ('logo', 'city', 'handle')]
    ys = [boxes[k][1] for k in ('logo', 'icon', 'pref', 'handle')] + [boxes[k][3] for k in ('logo', 'icon', 'city', 'handle')]
    print('1組の外接枠 x %d-%d  y %d-%d   「県」と「土」の切れ目 x=%d' % (min(xs), max(xs), min(ys), max(ys), boxes['split_x']))
    txt = union([full['pref'], full['city'], full['handle']])
    for t in (12.50, 12.80, 13.30, 13.70, 14.10, 14.50, 14.90, 14.967):
        fr, _, _ = cut8(t, cache)
        allon = fr
        for s in STAGES:
            allon = comp(allon, L[s], 1.0)
        c_text = [contrast(allon, full[k], boxes[k]) for k in ('pref', 'city', 'handle')]
        c_icon = contrast(allon, full['icon'], boxes['icon'])
        # 影なし（白95%を海にそのまま置いた場合）の文字
        plain = np.asarray(fr).astype(np.float32)
        ta = (np.asarray(txt).astype(np.float32) / 255.0 * COMMON['text_opacity'])[..., None]
        plain = Image.fromarray((plain * (1 - ta) + 255 * ta).round().clip(0, 255).astype(np.uint8))
        c_plain = [contrast(plain, full[k], boxes[k]) for k in ('pref', 'city', 'handle')]
        print('t=%.3f  文字（影なし）%.2f〜%.2f:1  文字（影込み）%.2f〜%.2f:1  輪郭アイコン（影込み）%.2f:1' % (
            t, min(c_plain), max(c_plain), min(c_text), max(c_text), c_icon))


# ---------------------------------------------------------------------------
# 書き出し
# ---------------------------------------------------------------------------
def export_overlays(d, variant, no_text=False):
    os.makedirs(d, exist_ok=True)
    L = stage_layers(variant, no_text=no_text)
    for s in STAGES:
        p = os.path.join(d, 'ig15_end_%s%s_1080x1920.png' % (STAGE_NAMES[s], '_notext' if no_text else ''))
        L[s].save(p)
        print(p, ('（ロゴ：%s）' % LOGOS[variant][1]) if s == 's3' else '')


def label(im, text, xy=(16, 12), size=24):
    d = ImageDraw.Draw(im)
    d.text(xy, text, font=ImageFont.truetype(font_path('Medium'), size), fill=(255, 230, 120))


def logo_variants_sheet(cache):
    """End Cardの大きさ（等倍）で、ロゴの版A・B・Cを並べ、元のロゴ（スキャン）を添える。お店に版を選んでもらう資料。"""
    box = (190, 930, 890, 1260)                       # 1組のまわり（1080x1920の座標）
    bw, bh = box[2] - box[0], box[3] - box[1]
    pad, top = 16, 44
    tiles = []
    raw = Image.open(LOGO_RAW).convert('L')
    g = np.asarray(raw)
    ys, xs = np.nonzero(g[200:1400, 500:1800] < 110)
    rb = (xs.min() + 500 - 6, ys.min() + 200 - 6, xs.max() + 500 + 6, ys.max() + 200 + 6)
    rc = raw.crop(rb).convert('RGB')
    rc = rc.resize((int(round(rc.width * (bh - 40) / rc.height)), bh - 40), Image.LANCZOS)
    t0 = Image.new('RGB', (bw, bh), (245, 245, 242))
    t0.paste(rc, ((bw - rc.width) // 2, 20))
    tiles.append((t0, '元のロゴ（お店のPDF＝紙のスキャン。横に約4%長い）'))
    for v in ('A', 'B', 'C'):
        L = stage_layers(v)
        fr, _, _ = render(14.50, L, cache)
        tiles.append((fr.crop(box), LOGOS[v][1] + ('（推奨）' if v == LOGO_DEFAULT else '')))
    sheet = Image.new('RGB', (bw * 2 + pad * 3, (bh + top) * 2 + pad * 3), (18, 18, 18))
    for i, (tile, cap) in enumerate(tiles):
        x = pad + (i % 2) * (bw + pad)
        y = pad + (i // 2) * (bh + top + pad)
        sheet.paste(tile, (x, y + top))
        label(sheet, cap, (x, y + 8), 24)
    return sheet


def main():
    variant = LOGO_DEFAULT
    if '--logo' in sys.argv:
        variant = sys.argv[sys.argv.index('--logo') + 1].upper()
        assert variant in LOGOS, variant
    if len(sys.argv) > 2 and sys.argv[1] == '--overlays':
        export_overlays(sys.argv[2], variant, '--no-text' in sys.argv)
        return
    cache = os.path.join(tempfile.gettempdir(), 'maze_ig15_frames')
    os.makedirs(cache, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    L = stage_layers(variant)
    print('ロゴ：%s' % LOGOS[variant][1])
    report(cache, variant, L)
    suffix = '' if variant == LOGO_DEFAULT else '_logo%s' % variant
    shots = [('a0_kochi', 12.50, False), ('a_map', 12.80, True), ('b_full', 14.50, True)]
    full = None
    for name, t, zones in shots:
        im, sec, xy = render(t, L, cache)
        print(name, 't=%.2f' % t, 'IMG_9633 %.3fs' % sec, 'crop x=%.1f y=%.1f' % xy)
        base = os.path.join(OUT, 'ig15_end_%s_t%05.2f%s' % (name, t, suffix))
        im.save(base + '.png')
        if zones:
            safezones(im).save(base + '_safezones.png')
        im.resize((W // 3, H // 3), Image.LANCZOS).save(base + '_third.png')
        if name == 'b_full':
            full = im
    # ロゴと文字の拡大（2倍・画素そのまま）：ロゴの縁・犬の明暗・文字のにじみの確認用
    box = (190, 930, 890, 1260)
    full.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.NEAREST).save(
        os.path.join(OUT, 'ig15_end_c_detail_2x%s.png' % suffix))
    if suffix:
        return
    # 出方の確認：1/3に縮小したコマを時間順に並べる（「高知県」→「土佐市」→ ロゴとアカウント名 の3段が見えるか）
    seq = [12.00, 12.35, 12.50, 12.65, 12.80, 13.05, 13.30, 14.97]
    tw, th = W // 3, H // 3
    strip = Image.new('RGB', (tw * 4 + 5 * 12, (th + 40) * 2 + 12), (18, 18, 18))
    for i, t in enumerate(seq):
        im, _, _ = render(t, L, cache)
        x = 12 + (i % 4) * (tw + 12)
        y = (i // 4) * (th + 40 + 6)
        strip.paste(im.resize((tw, th), Image.LANCZOS), (x, y + 40))
        label(strip, '%.2f秒' % t, (x + 4, y + 8), 24)
    strip.save(os.path.join(OUT, 'ig15_end_e_sequence_third.png'))
    logo_variants_sheet(cache).save(os.path.join(OUT, 'ig15_end_d_logo_variants.png'))


if __name__ == '__main__':
    main()
