#!/usr/bin/env python3
"""Instagram 30秒版（EDL v6）のモックアップと、CapCutに重ねる透過PNGを作るスクリプト。

実素材（HDR/HLG）から指定の瞬間をトーンマップして1枚取り出し、EDLと同じ
切り出し・回転・空の補正をかけ、ロゴと文字をEDLの数値どおりに重ねる。
数値はすべて「画面の高さ（または幅）に対する割合」なので、1080x1920でもそれ以外でも同じ見た目になる。

  python3 edl/mockups/ig30_render.py                   # モックアップを edl/mockups/ に書き出す（数値と測定値も表示）
  python3 edl/mockups/ig30_render.py --overlays DIR    # 承認後：CapCutに重ねる透過PNG（1080x1920）を DIR に書き出す
      ig30_end_logo / ig30_end_lines（A案）/ ig30_c10_copy
  python3 edl/mockups/ig30_render.py --overlays DIR --map
      B案（MAPあり）に決まった場合：上に加えて ig30_end_lines_map（1行目が右へ約34pxずれた版。lines の代わりに使う）、
      ig30_end_mapicon（高知県の輪郭）、ig30_end_mapdot（土佐市の点）

書体：assets/fonts/ の Noto Sans JP（Google / SIL Open Font License 1.1、ライセンス文は assets/fonts/OFL.txt）。
  fontsource @fontsource/noto-sans-jp 5.3.0 の静的ウェイト（name表の版：Version 2.004-H2）。
  NotoSansJP-Light.ttf    sha256 10387105d854fd95d8a7e9e78e42a17947aad2c9057f958a88ceceb597f85d2b（300）
  NotoSansJP-Regular.ttf  sha256 25d059a3d0fac5de6c5b0d1fa7c32c7ce5d85befca1752ef2eec8be9a2345158（400）
  NotoSansJP-Medium.ttf   sha256 29f6b29939a779e33601cbbe05f158e4b1568809eaf46b5e8d43cb7092e82412（500）
  別の版の書体を使うと字幅が変わり、承認したモックアップとずれる。読み込むたびにハッシュを確かめ、違えば警告する。

注意（2026-09-23 時点）：assets/logo_white.png は、お店のPDF（白地に黒の版画風エンブレム）のインクを白に置き換えたもの。
  犬の目・マズル・首輪の明暗が元のロゴと逆になる。お店が白版で使ってよいと了承するか、
  白抜き（反転）版のデータが届くまでは、--overlays で本番用のPNGを焼き込まない（比較画像：ig30_g_logo_face_compare.png）。
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
LOGO = os.path.join(ROOT, 'assets', 'logo_white.png')
LOGO_RAW = os.path.join(ROOT, 'footage', '_store', 'logo_raw.jpeg')
OUT = os.path.join(ROOT, 'edl', 'mockups')
FONT_DIRS = [os.path.join(ROOT, 'assets', 'fonts'), '/tmp/claude-0/fonts', os.path.expanduser('~/.fonts'),
             '/usr/share/fonts/truetype/noto']
FONT_SHA256 = {
    'Light': '10387105d854fd95d8a7e9e78e42a17947aad2c9057f958a88ceceb597f85d2b',
    'Regular': '25d059a3d0fac5de6c5b0d1fa7c32c7ce5d85befca1752ef2eec8be9a2345158',
    'Medium': '29f6b29939a779e33601cbbe05f158e4b1568809eaf46b5e8d43cb7092e82412',
}
FFMPEG = os.environ.get(
    'FFMPEG', '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
TONEMAP = ('zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,'
           'zscale=t=bt709:m=bt709:r=tv,format=yuv420p')

W, H = 1080, 1920

# ---------------------------------------------------------------------------
# EDL v6 の数値（h = 画面の高さ、w = 画面の幅）
# ---------------------------------------------------------------------------
# 影：(ぼかしσ[h比], 不透明度)、黒・ずらしなし。縁取り・グラデーション・帯は使わない
SHADOW_LOGO = [(0.0012, 0.40), (0.0060, 0.22)]   # 1920pで σ2.3px 40% ＋ σ11.5px 22%（v5の60%＋28%から下げた）
SHADOW_TEXT = [(0.0016, 0.45), (0.0065, 0.22)]   # 1920pで σ3.1px 45% ＋ σ12.5px 22%（2行・カット10のコピー・MAP共通）
TEXT_SIZE = 0.0200                               # 文字サイズ 2.0%h（1920pで38px）。2行とカット10のコピーで共通
TEXT_OPACITY = 0.95

SPEC = dict(
    logo=dict(                       # End Card：お店のロゴ（assets/logo_white.png、白・透過、縦横比はそのまま）
        h=0.165,                     # 見えているエンブレムの高さ 16.5%h（1920pで317px）
        cx=0.500,                    # 横の中心 50%w
        bottom=0.510,                # 下端 上から51.0%h
        opacity=1.0,
        shadow=SHADOW_LOGO,
    ),
    lines=dict(                      # End Card：水平線のすぐ下の海に2行（インクの外接枠の縦の中心 cy）
        items=[
            dict(text='高知県土佐市', weight='Regular', size=TEXT_SIZE, track=0.20, cy=0.6030),
            dict(text='@mazewind2026', weight='Regular', size=TEXT_SIZE, track=0.06, cy=0.6370),
        ],
        cx=0.500,
        opacity=TEXT_OPACITY,
        shadow=SHADOW_TEXT,
    ),
    copy=dict(                       # カット10：コピー1行（空の上）
        items=[dict(text='海のそばで、少しだけ。', weight='Regular', size=TEXT_SIZE, track=0.10, cy=0.2400)],
        cx=0.500,
        opacity=TEXT_OPACITY,
        shadow=SHADOW_TEXT,
    ),
    map=dict(                        # B案（MAPあり）だけ：「高知県土佐市」の行の左に、高知県の輪郭と土佐市の点
        icon_h=0.023,                # 輪郭の高さ 2.3%h（44px）。IG15の部品と同じ値
        stroke=0.0013,               # 線の太さ 0.13%h（2.5px）
        dot_d=0.0055,                # 土佐市の点の直径 0.55%h（約10.5px）
        dot_clear=0.0013,            # 点のまわりの輪郭線を 0.13%h 切り欠く
        gap=0.013,                   # 輪郭の右端〜「高知県土佐市」の左端 1.3%w（14px）
        opacity=TEXT_OPACITY,
        shadow=SHADOW_TEXT,
    ),
    # カット12（9661）の空の補正：上端〜上から42%で−0.6段、そこから下へなめらかに（smoothstep）弱めて上から53.5%で0。
    # 水平線は上から約54.8〜55.7%
    sky=dict(stops=-0.6, plateau=0.42, zero=0.535),
    # カット10（9636）の淡い空の補正：上端〜上から30%で−0.4段、上から60%で0（水平線は約66%）
    sky_c10=dict(stops=-0.4, plateau=0.30, zero=0.60),
)

# カット（rec_in, rec_out, source, src_in, speed, 切り出し(x, y, w, h) or None, 時計回りの回転°, 空の補正のキー）
CUTS = [
    (4.00, 5.60, 'IMG_9673', 2.70, 0.5, None, 0.0, None),                            # カット3（のれんの波柄と黒い裾だけ）
    (5.60, 7.90, 'IMG_9673', 5.00, 1.0, (180, 640, 1800, 3200), 0.0, None),         # カット4
    (18.80, 22.30, 'IMG_9636', 1.70, 1.0, (415, 538, 1200, 2133), 0.95, 'sky_c10'),  # カット10（コピー）
    (24.70, 30.00, 'IMG_9661', 3.20, 1.0, (180, 18, 1800, 3200), 0.70, 'sky'),      # カット12（End Card）
]

# 重ねるものの出方（秒, 不透明度）。直線、イージングなし
TIMING = dict(
    copy=[(19.20, 0.0), (19.60, 1.0), (21.50, 1.0), (21.90, 0.0)],
    logo=[(25.30, 0.0), (26.10, 1.0), (30.00, 1.0)],
    lines=[(26.50, 0.0), (27.10, 1.0), (30.00, 1.0)],
    mapicon=[(26.50, 0.0), (26.80, 1.0), (30.00, 1.0)],    # B案：高知県（輪郭）
    mapdot=[(26.80, 0.0), (27.10, 1.0), (30.00, 1.0)],     # B案：土佐市（点）
)

# 高知県の輪郭（IG15の ig15_render.py と同じ点列。国土数値情報ベースの本土部分を簡略化。経度, 緯度）
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
TOSA = (133.48, 33.458)


def envelope(t, pts):
    if t < pts[0][0] or t > pts[-1][0]:
        return 0.0
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            return v0 + (v1 - v0) * (t - t0) / (t1 - t0) if t1 > t0 else v1
    return 0.0


_FONT_CHECKED = set()


def font_path(weight):
    for d in FONT_DIRS:
        p = os.path.join(d, 'NotoSansJP-%s.ttf' % weight)
        if os.path.exists(p):
            if p not in _FONT_CHECKED:
                _FONT_CHECKED.add(p)
                with open(p, 'rb') as fh:
                    if hashlib.sha256(fh.read()).hexdigest() != FONT_SHA256.get(weight):
                        print('警告：%s は承認時と別の版の書体です（字幅が変わるおそれ）' % p, file=sys.stderr)
            return p
    raise FileNotFoundError('NotoSansJP-%s.ttf' % weight)


# ---------------------------------------------------------------------------
# 素材
# ---------------------------------------------------------------------------
def grab(source, sec, cache):
    out = os.path.join(cache, '%s_%.4f.png' % (source, sec))
    if not os.path.exists(out):
        subprocess.run([FFMPEG, '-v', 'error', '-ss', '%.4f' % sec, '-i', os.path.join(FOOT, source + '.MOV'),
                        '-frames:v', '1', '-vf', TONEMAP, '-y', out], check=True)
    return Image.open(out).convert('RGB')


def place(src, crop, rot_cw):
    """切り出し枠の中心まわりに時計回りに回して（右上がりの水平線を戻す）、枠で切り出し、W x H に縮小する。
    回転で素材の外が入ったら赤く塗るので、赤が出ないことで余白を確かめられる。"""
    if crop is None:
        crop = (0, 0, src.width, src.height)
    x, y, w, h = crop
    if rot_cw:
        src = src.rotate(-rot_cw, resample=Image.BICUBIC, center=(x + w / 2, y + h / 2), fillcolor=(255, 0, 0))
    return src.crop((x, y, x + w, y + h)).resize((W, H), Image.LANCZOS)


def sky_grade(img, g):
    """空の補正：上端〜g['plateau']で g['stops'] 段、そこから g['zero'] へなめらかに0（線形光で掛ける）。"""
    a = np.asarray(img).astype(np.float32) / 255.0
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    y = (np.arange(H, dtype=np.float32) / H)[:, None, None]
    wgt = np.clip((g['zero'] - y) / (g['zero'] - g['plateau']), 0, 1)
    wgt = wgt * wgt * (3 - 2 * wgt)
    lin = lin * (2.0 ** (g['stops'] * wgt))
    out = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(np.clip(lin, 0, 1), 1 / 2.4) - 0.055)
    return Image.fromarray((out * 255).clip(0, 255).round().astype(np.uint8), 'RGB')


def frame_at(t, cache, grade=True, S=SPEC):
    for rec_in, rec_out, source, src_in, speed, crop, rot, gk in CUTS:
        if rec_in <= t < rec_out or (t == rec_out == 30.0):
            sec = src_in + (t - rec_in) * speed
            fr = place(grab(source, sec, cache), crop, rot)
            if grade and gk:
                fr = sky_grade(fr, S[gk])
            return fr, (source, round(sec, 3))
    raise ValueError(t)


# ---------------------------------------------------------------------------
# 重ねるもの
# ---------------------------------------------------------------------------
def text_mask(s, weight, size_px, track_em):
    f = ImageFont.truetype(font_path(weight), int(round(size_px)))
    adv = [f.getlength(ch) for ch in s]
    tr = track_em * size_px
    asc, desc = f.getmetrics()
    m = Image.new('L', (int(sum(adv) + tr * len(s) + size_px) + 40, asc + desc + 40), 0)
    d = ImageDraw.Draw(m)
    x = 20.0
    for ch, a in zip(s, adv):
        d.text((x, 20), ch, font=f, fill=255)
        x += a + tr
    return m.crop(m.getbbox())


def logo_alpha(h_px):
    """見えているエンブレム（不透明度>8/255の外接枠）の高さを h_px にした白のアルファ。縦横比はそのまま。"""
    a = Image.open(LOGO).convert('RGBA').getchannel('A')
    a = a.crop(a.point(lambda v: 255 if v > 8 else 0).getbbox())
    return a.resize((int(round(h_px * a.width / a.height)), int(round(h_px))), Image.LANCZOS)


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
    """高知県の輪郭（線）と土佐市の点（IG15と同じ描き方）。8倍で描いて縮小。"""
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
    bb = outline.getbbox()
    return outline.crop(bb), big.resize((wm, hm), Image.BOX).crop(bb)


def shadowed(alpha, shadows, opacity):
    """白い字・ロゴ（alpha）に、黒・ずらしなしの柔らかい影を付けたRGBA。縁取りやグラデーションは使わない。"""
    a = np.asarray(alpha).astype(np.float32) / 255.0
    sh = np.zeros_like(a)
    for sigma, op in shadows:
        b = np.asarray(alpha.filter(ImageFilter.GaussianBlur(sigma * H))).astype(np.float32) / 255.0
        sh = 1 - (1 - sh) * (1 - np.clip(b * op, 0, 1))
    ink = a * opacity
    out_a = ink + sh * opacity * (1 - ink)
    rgb = (ink / np.maximum(out_a, 1e-6)) * 255.0
    rgba = np.dstack([rgb, rgb, rgb, out_a * 255.0]).clip(0, 255).round().astype(np.uint8)
    return Image.fromarray(rgba, 'RGBA')


def map_masks(S=SPEC):
    m = S['map']
    return icon_masks(m['icon_h'] * H, m['stroke'] * H, m['dot_d'] * H, m['dot_clear'] * H)


def layout(kind, S=SPEC, with_map=False):
    """重ねるものの alpha（W x H）と、各要素の箱（px）を返す。kind: logo / lines / copy / mapicon / mapdot
    with_map=True（B案）のときは、「輪郭＋すき間＋高知県土佐市」の1組を左右中央に置く（1行目だけ右へずれる）。"""
    al = Image.new('L', (W, H), 0)
    boxes = []
    if kind == 'logo':
        c = S['logo']
        lg = logo_alpha(c['h'] * H)
        lx = int(round(c['cx'] * W - lg.width / 2))
        ly = int(round(c['bottom'] * H - lg.height))
        al.paste(lg, (lx, ly))
        boxes.append(('logo', (lx, ly, lx + lg.width, ly + lg.height)))
    elif kind in ('mapicon', 'mapdot'):
        m = S['map']
        ol, dot = map_masks(S)
        _, lb = layout('lines', S, with_map=True)
        x1, y0, _, y1 = lb[0][1]                       # B案の「高知県土佐市」の外接枠
        ix = int(round(x1 - m['gap'] * W - ol.width))
        iy = int(round((y0 + y1) / 2 - ol.height / 2))
        mk = ol if kind == 'mapicon' else dot
        al.paste(mk, (ix, iy), mk)
        boxes.append((kind, (ix, iy, ix + ol.width, iy + ol.height)))
    else:
        c = S[kind]
        for i, it in enumerate(c['items']):
            tm = text_mask(it['text'], it['weight'], it['size'] * H, it['track'])
            tx = c['cx'] * W - tm.width / 2
            if kind == 'lines' and with_map and i == 0:
                ol, _ = map_masks(S)
                tx += (ol.width + S['map']['gap'] * W) / 2
            tx = int(round(tx))
            ty = int(round(it['cy'] * H - tm.height / 2))
            al.paste(tm, (tx, ty), tm)
            boxes.append((it['text'], (tx, ty, tx + tm.width, ty + tm.height)))
    return al, boxes


def overlay(kind, S=SPEC, with_map=False):
    al, _ = layout(kind, S, with_map)
    c = S['map'] if kind.startswith('map') else S[kind]
    return shadowed(al, c['shadow'], c['opacity'])


def comp(frame, ov, opacity):
    if opacity <= 0:
        return frame
    if opacity < 1:
        ov = ov.copy()
        ov.putalpha(ov.getchannel('A').point(lambda v: int(round(v * opacity))))
    out = frame.convert('RGBA')
    out.alpha_composite(ov)
    return out.convert('RGB')


def render(t, cache, layers, with_map=False):
    fr, info = frame_at(t, cache)
    kinds = ['copy', 'logo'] + (['lines_map', 'mapicon', 'mapdot'] if with_map else ['lines'])
    for k in kinds:
        fr = comp(fr, layers[k], envelope(t, TIMING['lines' if k == 'lines_map' else k]))
    return fr, info


def safe_zones(img):
    """Reelsの画面UIの目安：上14%、下20%（キャプション）、右12%（ボタンの列）。半透明で塗る。"""
    o = Image.new('RGBA', img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(o)
    w, h = img.size
    d.rectangle([0, 0, w, int(0.14 * h)], fill=(255, 40, 40, 90))
    d.rectangle([0, int(0.80 * h), w, h], fill=(255, 40, 40, 90))
    d.rectangle([int(0.88 * w), int(0.14 * h), w, int(0.80 * h)], fill=(255, 160, 0, 90))
    for y in (0.14, 0.80):
        d.line([0, int(y * h), w, int(y * h)], fill=(255, 255, 255, 160), width=2)
    d.line([int(0.88 * w), int(0.14 * h), int(0.88 * w), int(0.80 * h)], fill=(255, 255, 255, 160), width=2)
    f = ImageFont.truetype(font_path('Regular'), 26)
    d.text((16, int(0.14 * h) - 40), '上14%：アカウント・音声などのUI', font=f, fill=(255, 255, 255, 230))
    d.text((16, int(0.80 * h) + 12), '下20%：キャプション・音源名', font=f, fill=(255, 255, 255, 230))
    d.text((int(0.88 * w) - 8, int(0.47 * h)), '右12%', font=f, fill=(255, 255, 255, 230), anchor='rs')
    out = img.convert('RGBA')
    out.alpha_composite(o)
    return out.convert('RGB')


# ---------------------------------------------------------------------------
# 測定（コントラスト比は WCAG の相対輝度で、字の芯と、字のすぐ外1〜3pxの平均の比。影込み）
# ---------------------------------------------------------------------------
def rel_lum(img):
    a = np.asarray(img.convert('RGB')).astype(np.float32) / 255.0
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def contrast(composited, alpha, box):
    """box の中の字（alpha>=0.9）と、字のすぐ外1〜3px（alpha<0.02）の相対輝度の比。"""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    Y = rel_lum(composited)[y0:y1, x0:x1]
    a = np.asarray(alpha).astype(np.float32)[y0:y1, x0:x1] / 255.0
    core = a >= 0.9
    near = np.asarray(Image.fromarray(((a > 0.5) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(7))) > 0
    ring = near & (a < 0.02)
    lt, lb = float(Y[core].mean()), float(Y[ring].mean())
    return (lt + 0.05) / (lb + 0.05)


def logo_regions(S=SPEC):
    """ロゴの中の「MAZE WIND」（上の弧の中央部）・「RETREAT」の箱（画面のpx）と、内側の輪の中の線のない所のマスク。
    元PNG（970x933、見えているエンブレムは x 11..963・y 12..923）の座標から換算する。"""
    al, boxes = layout('logo', S)
    lx, ly, lx1, ly1 = boxes[0][1]
    sx, sy = (lx1 - lx) / 952.0, (ly1 - ly) / 911.0

    def cv(x0, y0, x1, y1):
        return (lx + (x0 - 11) * sx, ly + (y0 - 12) * sy, lx + (x1 - 11) * sx, ly + (y1 - 12) * sy)
    yy, xx = np.mgrid[0:H, 0:W]
    rn = np.sqrt(((xx - (lx + lx1) / 2) / ((lx1 - lx) / 2)) ** 2 + ((yy - (ly + ly1) / 2) / ((ly1 - ly) / 2)) ** 2)
    interior = (rn < 0.60) & (np.asarray(al) < 5)       # 内側の細い輪（半径の約0.7）より内側で、線のない画素
    return dict(maze_wind=cv(300, 55, 680, 175), retreat=cv(205, 668, 752, 760), interior=interior)


def report(cache, S=SPEC):
    for kind, wm in (('logo', False), ('lines', False), ('copy', False), ('lines', True), ('mapicon', True)):
        _, boxes = layout(kind, S, wm)
        for name, (x0, y0, x1, y1) in boxes:
            print('%-8s %-22s x %5.1f-%5.1f%%w  y %5.1f-%5.1f%%h  (%dx%d px)' % (
                kind + ('(B)' if wm else ''), name, x0 / W * 100, x1 / W * 100, y0 / H * 100, y1 / H * 100,
                x1 - x0, y1 - y0))
    L = {k: overlay(k, S) for k in ('logo', 'lines', 'copy')}
    noshadow = shadowed(layout('logo', S)[0], [], S['logo']['opacity'])
    al = {k: layout(k, S)[0] for k in ('logo', 'lines', 'copy')}
    reg = logo_regions(S)
    lb = layout('lines', S)[1]
    for t in (27.20, 29.50, 29.90):
        fr, _ = frame_at(t, cache, S=S)
        c = comp(comp(fr, L['logo'], 1.0), L['lines'], 1.0)
        g1 = np.asarray(c.convert('L')).astype(np.float32)[reg['interior']]
        g0 = np.asarray(comp(fr, noshadow, 1.0).convert('L')).astype(np.float32)[reg['interior']]
        print('t=%.2f  MAZE WIND %.2f:1  RETREAT %.2f:1  高知県土佐市 %.2f:1  @mazewind2026 %.2f:1  '
              '輪の内側の線のない所：影なし%.0f→影あり%.0f' % (
                  t, contrast(c, al['logo'], reg['maze_wind']), contrast(c, al['logo'], reg['retreat']),
                  contrast(c, al['lines'], lb[0][1]), contrast(c, al['lines'], lb[1][1]), g0.mean(), g1.mean()))
    for t in (19.60, 21.40):
        fr, _ = frame_at(t, cache, S=S)
        c = comp(fr, L['copy'], 1.0)
        box = layout('copy', S)[1][0][1]
        print('t=%.2f  コピー %.2f:1  背景の平均 %.0f/255' % (
            t, contrast(c, al['copy'], box), np.asarray(fr.crop(box).convert('L')).mean()))


# ---------------------------------------------------------------------------
# 書き出し
# ---------------------------------------------------------------------------
def export_overlays(d, with_map):
    os.makedirs(d, exist_ok=True)
    overlay('logo').save(os.path.join(d, 'ig30_end_logo_1080x1920.png'))
    overlay('lines').save(os.path.join(d, 'ig30_end_lines_1080x1920.png'))
    overlay('copy').save(os.path.join(d, 'ig30_c10_copy_1080x1920.png'))
    if with_map:   # B案：lines の代わりに lines_map を使い、輪郭と点を足す
        overlay('lines', with_map=True).save(os.path.join(d, 'ig30_end_lines_map_1080x1920.png'))
        overlay('mapicon').save(os.path.join(d, 'ig30_end_mapicon_1080x1920.png'))
        overlay('mapdot').save(os.path.join(d, 'ig30_end_mapdot_1080x1920.png'))


def logo_face_compare():
    """元のロゴ（スキャン）と白版の顔の比較（等倍と2倍）。お店に白版の了承をもらうための資料。"""
    raw = Image.open(LOGO_RAW).convert('RGB')
    wl = Image.open(LOGO).convert('RGBA')
    ab = wl.getchannel('A').point(lambda v: 255 if v > 8 else 0).getbbox()
    gray = np.asarray(raw.convert('L'))
    ys, xs = np.where(gray[200:1400, 500:1800] < 110)              # スキャンの中のエンブレム（外側の輪）の外接枠
    rb = (xs.min() + 500, ys.min() + 200, xs.max() + 500, ys.max() + 200)
    sx = (rb[2] - rb[0]) / (ab[2] - ab[0])
    sy = (rb[3] - rb[1]) / (ab[3] - ab[1])

    def to_raw(b):
        return (int(round(rb[0] + (b[0] - ab[0]) * sx)), int(round(rb[1] + (b[1] - ab[1]) * sy)),
                int(round(rb[0] + (b[2] - ab[0]) * sx)), int(round(rb[1] + (b[3] - ab[1]) * sy)))
    bg = Image.new('RGB', wl.size, (70, 110, 150))
    bg.paste(wl.convert('RGB'), mask=wl.getchannel('A'))
    f = ImageFont.truetype(font_path('Medium'), 26)
    pad, top = 20, 60
    rows = []
    for box, scale, cap in (((270, 230, 750, 670), 1, '等倍'), ((380, 340, 620, 500), 2, '目・2倍')):
        a = raw.crop(to_raw(box))
        a = a.resize((a.width * scale, a.height * scale), Image.NEAREST)
        b = bg.crop(box).resize(a.size, Image.LANCZOS if scale == 1 else Image.NEAREST)
        c = Image.new('RGB', (a.width * 2 + pad * 3, a.height + top + pad), (18, 18, 18))
        c.paste(a, (pad, top))
        c.paste(b, (a.width + pad * 2, top))
        d = ImageDraw.Draw(c)
        d.text((pad, 16), '元のロゴ（お店のPDF）%s' % cap, font=f, fill=(255, 230, 120))
        d.text((a.width + pad * 2, 16), '白版（logo_white.png）%s' % cap, font=f, fill=(255, 230, 120))
        rows.append(c)
    out = Image.new('RGB', (max(r.width for r in rows), sum(r.height for r in rows)), (18, 18, 18))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    return out


def main():
    cache = os.path.join(tempfile.gettempdir(), 'maze_ig30_frames')
    os.makedirs(cache, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[1] == '--overlays':
        export_overlays(sys.argv[2], '--map' in sys.argv)
        return
    os.makedirs(OUT, exist_ok=True)
    L = {k: overlay(k) for k in ('copy', 'logo', 'lines', 'mapicon', 'mapdot')}
    L['lines_map'] = overlay('lines', with_map=True)
    report(cache)

    # End Card A案（MAPなし）
    for name, t in (('ig30_a_end_t27.20', 27.20), ('ig30_b_end_t29.90', 29.90)):
        fr, info = render(t, cache, L)
        print(name, info)
        fr.save(os.path.join(OUT, name + '.png'))
        safe_zones(fr).save(os.path.join(OUT, name + '_safezones.png'))
        fr.resize((W // 3, H // 3), Image.LANCZOS).save(os.path.join(OUT, name + '_third.png'))
    a_full, _ = render(27.20, cache, L)
    box = (330, 640, 750, 1260)
    a_full.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.NEAREST).save(
        os.path.join(OUT, 'ig30_a_end_t27.20_detail_2x.png'))

    # End Card B案（MAPあり）と、A・Bの並べ比べ
    b_full, _ = render(27.20, cache, L, with_map=True)
    b_full.save(os.path.join(OUT, 'ig30_d_end_map_t27.20.png'))
    b_full.resize((W // 3, H // 3), Image.LANCZOS).save(os.path.join(OUT, 'ig30_d_end_map_t27.20_third.png'))
    b_full.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.NEAREST).save(
        os.path.join(OUT, 'ig30_d_end_map_t27.20_detail_2x.png'))
    hw, hh = W // 2, H // 2
    cmp = Image.new('RGB', (hw * 2 + 30, hh + 60), (18, 18, 18))
    cmp.paste(a_full.resize((hw, hh), Image.LANCZOS), (0, 60))
    cmp.paste(b_full.resize((hw, hh), Image.LANCZOS), (hw + 30, 60))
    d = ImageDraw.Draw(cmp)
    f = ImageFont.truetype(font_path('Medium'), 24)
    d.text((12, 16), 'A案：MAPなし（ロゴ＋2行）', font=f, fill=(255, 230, 120))
    d.text((hw + 42, 16), 'B案：MAPあり（行の左に輪郭と点）', font=f, fill=(255, 230, 120))
    cmp.save(os.path.join(OUT, 'ig30_e_end_AB_compare.png'))

    # カット10のコピー
    for name, t in (('ig30_f_c10_copy_t19.60', 19.60), ('ig30_f_c10_copy_t21.40', 21.40)):
        fr, info = render(t, cache, L)
        print(name, info)
        fr.save(os.path.join(OUT, name + '.png'))
        fr.resize((W // 3, H // 3), Image.LANCZOS).save(os.path.join(OUT, name + '_third.png'))
        if t == 19.60:
            safe_zones(fr).save(os.path.join(OUT, name + '_safezones.png'))

    # カット3：のれんの波柄と黒い裾だけ。入り・中・終わりと、カット4の入り
    strip = [(4.00, 'c3 4.00（素材2.70）'), (4.60, 'c3 4.60（3.00）'), (5.20, 'c3 5.20（3.30）'),
             (5.58, 'c3 5.58（3.49）'), (5.60, 'c4 5.60（5.00）')]
    tw, th = 270, 480
    s = Image.new('RGB', (tw * len(strip), th + 36), (18, 18, 18))
    d = ImageDraw.Draw(s)
    f = ImageFont.truetype(font_path('Regular'), 20)
    for i, (t, lab) in enumerate(strip):
        fr, _ = frame_at(t, cache)
        s.paste(fr.resize((tw, th), Image.LANCZOS), (i * tw, 36))
        d.text((i * tw + 8, 8), lab, font=f, fill=(255, 230, 120))
    s.save(os.path.join(OUT, 'ig30_c_cut03_noren_strip.png'))

    # ロゴの顔の比較（お店への確認用）
    logo_face_compare().save(os.path.join(OUT, 'ig30_g_logo_face_compare.png'))


if __name__ == '__main__':
    main()
