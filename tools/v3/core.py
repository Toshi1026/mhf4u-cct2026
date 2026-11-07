"""v3（IG・案A「世界記録の波で、高知まで行けるか」）の共通部品。

フレームはすべて sRGB の float32（0..1）、形は (H, W, 3)。1080×1920・30fps。
各ビート（tools/v3/beats/bNN.py）は render(n, ctx) という generator を持ち、n 枚のフレームを順に yield する。
文字は text_layer() で RGBA を作り、put_text() で重ねる（白・やわらかい2層の影・縁取りなし）。
"""
import json
import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'exports', 'v3')
VO = os.path.join(OUT, 'vo')
FOOT = os.path.join(ROOT, 'footage')
MODELS = os.path.join(ROOT, 'tools', 'models')
FONT_VF = os.path.join(ROOT, 'assets', 'fonts', 'NotoSansJP-VF.ttf')
W, H, FPS = 1080, 1920, 30
SAFE = (65, 269, 1015, 1248)           # IG リールで UI に隠れない範囲 (x0, y0, x1, y1)
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- 色（sRGB 0..1）
def c(hexs):
    hexs = hexs.lstrip('#')
    return np.array([int(hexs[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255


NAVY = c('06182B')      # いちばん深い色（答えの背景など）
DEEP = c('0B4F8A')      # 深い海
SEA = c('3A8FD0')       # 明るい海（基調）
SKY = c('A9D6F0')       # 朝の空
FOAM = c('F2F8FC')      # 泡・線・文字の白
ACC = c('9EE6FF')       # 強調色（1画面に1語だけ）。オレンジ・金・赤・黄は使わない
SAND = c('8FA6B8')      # 砂（ベージュではなく青灰色）
LAND = c('C9D6E0')      # 地図の陸（明るい灰青）


def vgrad(top, bottom, h=H, w=W, gamma=1.0):
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None] ** gamma
    return np.broadcast_to(top * (1 - t) + bottom * t, (h, w, 3)).copy()


def ease(t, kind='inout'):
    t = float(np.clip(t, 0, 1))
    if kind == 'in':
        return t * t * t
    if kind == 'out':
        return 1 - (1 - t) ** 3
    if kind == 'linear':
        return t
    return t * t * (3 - 2 * t) if kind == 'smooth' else (4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2)


def lerp(a, b, t):
    return a + (b - a) * t


def over(frame, rgba):
    """frame（H,W,3）に RGBA（H,W,4、ストレートアルファ）を重ねる。"""
    a = rgba[..., 3:4]
    return frame * (1 - a) + rgba[..., :3] * a


# ---------------------------------------------------------------- 4倍で描いて縮めるレイヤー
class SS:
    """PIL で s 倍の大きさに描き、縮めて RGBA float にする（線・図形の縁をなめらかにする）。
    使い方： L = SS(); d = L.draw; d.line(L.p([(x0,y0),(x1,y1)]), fill=L.col(FOAM, 0.9), width=L.w(2)); rgba = L.done()"""

    def __init__(self, s=3, w=W, h=H):
        self.s, self.W, self.H = s, w, h
        self.im = Image.new('RGBA', (w * s, h * s), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.im)

    def p(self, pts):
        s = self.s
        return [(x * s, y * s) for x, y in pts]

    def box(self, x0, y0, x1, y1):
        s = self.s
        return (x0 * s, y0 * s, x1 * s, y1 * s)

    def w(self, px):
        return max(1, int(round(px * self.s)))

    @staticmethod
    def col(rgb, a=1.0):
        return tuple(int(round(v * 255)) for v in rgb) + (int(round(a * 255)),)

    def done(self):
        im = self.im.resize((self.W, self.H), Image.LANCZOS)
        a = np.asarray(im).astype(np.float32) / 255
        return a


def surfer(L, x, y, scale=1.0, ang=0.0, rgb=None, alpha=1.0, pose='ride'):
    """顔のない白いピクトグラムのサーファー。(x, y) はボードの中心、scale=1 で全長約 34px。ang は度（時計回り）。
    pose='ride'：ボードに立って両腕を広げる／'hold'：ボードを脇に抱えて立つ（砂浜）。L は SS。"""
    rgb = FOAM if rgb is None else rgb
    col = L.col(rgb, alpha)
    r = np.deg2rad(ang)
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])

    def T(pts):
        return L.p([tuple(R @ np.array(q) * scale + (x, y)) for q in pts])

    d = L.draw
    if pose == 'ride':
        d.polygon(T([(-17, 0), (-12, -2.4), (12, -2.4), (17, 0), (12, 2.4), (-12, 2.4)]), fill=col)     # ボード
        d.line(T([(-5, -2), (-1, -12), (5, -2)]), fill=col, width=L.w(2.6 * scale), joint='curve')    # 脚
        d.line(T([(-1, -12), (1, -21)]), fill=col, width=L.w(3.2 * scale))                           # 胴
        d.line(T([(-9, -16), (1, -19), (10, -15)]), fill=col, width=L.w(2.2 * scale), joint='curve')  # 腕
        hx, hy = R @ np.array([2, -25.5]) * scale + (x, y)
    else:
        d.polygon(T([(6, -2), (9, -1), (9, -33), (6, -34), (5, -18)]), fill=col)                    # 立てたボード
        d.line(T([(-3, 0), (-1, -11), (2, 0)]), fill=col, width=L.w(2.6 * scale), joint='curve')
        d.line(T([(-1, -11), (-1, -21)]), fill=col, width=L.w(3.2 * scale))
        d.line(T([(-1, -19), (6, -16)]), fill=col, width=L.w(2.2 * scale))
        hx, hy = R @ np.array([-1, -25.5]) * scale + (x, y)
    rr = 3.4 * scale
    d.ellipse(L.box(hx - rr, hy - rr, hx + rr, hy + rr), fill=col)


# ---------------------------------------------------------------- 文字
_FONTS = {}


def font(size, weight='Regular'):
    k = (int(size), weight)
    if k not in _FONTS:
        f = ImageFont.truetype(FONT_VF, int(size))
        f.set_variation_by_name(weight)
        _FONTS[k] = f
    return _FONTS[k]


def text_width(text, size, weight='Regular', track=0.04):
    f = font(size, weight)
    return sum(f.getlength(ch) for ch in text) + track * size * (len(text) - 1)


def text_layer(items, w=W, h=H):
    """items：dict のリスト。キー：text, x, y, size, weight（Regular/Medium/Bold…）, anchor（'ms' など PIL 形式）,
    track（字間 em）, alpha, hl（強調する部分文字列。ACC 色になる）, color（既定 FOAM）, shadow（影の強さ 0..1）。
    戻り値：(rgba, shadow) — put_text に渡す。"""
    im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    sh = Image.new('L', (w, h), 0)
    d, ds = ImageDraw.Draw(im), ImageDraw.Draw(sh)
    for it in items:
        text, size = it['text'], it['size']
        if not text:
            continue
        wt = it.get('weight', 'Regular')
        f = font(size, wt)
        tr = it.get('track', 0.04) * size
        anchor = it.get('anchor', 'ms')
        widths = [f.getlength(ch) for ch in text]
        total = sum(widths) + tr * (len(text) - 1)
        x = it['x'] - (total / 2 if anchor[0] == 'm' else total if anchor[0] == 'r' else 0)
        a = it.get('alpha', 1.0)
        base = it.get('color', FOAM)
        hl = it.get('hl')
        hs = text.find(hl) if hl else -1
        for i, (ch, wd) in enumerate(zip(text, widths)):
            col = ACC if hs >= 0 and hs <= i < hs + len(hl) else base
            d.text((x, it['y']), ch, font=f, fill=tuple(int(v * 255) for v in col) + (int(255 * a),), anchor='l' + anchor[1])
            ds.text((x, it['y']), ch, font=f, fill=int(255 * a * it.get('shadow', 0.6)), anchor='l' + anchor[1])
            x += wd + tr
    s1 = np.asarray(sh.filter(ImageFilter.GaussianBlur(5))).astype(np.float32) / 255
    s2 = np.asarray(sh.filter(ImageFilter.GaussianBlur(18))).astype(np.float32) / 255
    return np.asarray(im).astype(np.float32) / 255, np.clip(s1 * 0.6 + s2 * 0.4, 0, 1)


def put_text(frame, layer, alpha=1.0):
    t, sh = layer
    if alpha <= 0:
        return frame
    f = frame * (1 - sh[..., None] * 0.55 * alpha) + c('051526') * (sh[..., None] * 0.0)
    a = t[..., 3:4] * alpha
    return f * (1 - a) + t[..., :3] * a


def fade(i, n, fin=6, fout=6):
    """i フレーム目（0..n-1）の不透明度。最初 fin、最後 fout フレームでフェード。"""
    a = 1.0
    if fin:
        a = min(a, (i + 1) / fin)
    if fout:
        a = min(a, (n - i) / fout)
    return float(np.clip(a, 0, 1))


_CG = None


def cg_label(frame, alpha=0.7):
    """CG 区間に常時出す小さなラベル「CGシミュレーション」（左上・セーフゾーン内）。"""
    global _CG
    if _CG is None:
        _CG = text_layer([dict(text='CGシミュレーション', x=SAFE[0] + 8, y=300, size=26, weight='Medium', anchor='ls',
                               track=0.06, shadow=0.5)])
    return put_text(frame, _CG, alpha)


# ---------------------------------------------------------------- 実写（iPhone HLG → SDR）
TONEMAP = ('zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,'
           'zscale=t=bt709:m=bt709:r=full')


def footage(name, t0, t1, rot=0.0, cx=0.5, cy=0.5, zoom=1.0, w=W, h=H):
    """footage/<name> の t0..t1 秒を、60fps のまま全フレーム読む（list ではなく generator）。
    縦の 2160×3840 に対して rot 度回し、中心 (cx, cy)（0..1）・倍率 zoom の 9:16 の窓を切り出して w×h に。"""
    src = os.path.join(FOOT, name)
    cw, ch = 2160 / zoom, 3840 / zoom
    x0 = min(max(cx * 2160 - cw / 2, 0), 2160 - cw)
    y0 = min(max(cy * 3840 - ch / 2, 0), 3840 - ch)
    vf = TONEMAP
    if abs(rot) > 1e-4:
        vf += ',rotate=%f:ow=iw:oh=ih:c=black' % np.deg2rad(rot)
    vf += ',crop=%d:%d:%d:%d,zscale=w=%d:h=%d:f=lanczos,format=rgb48le' % (cw, ch, x0, y0, w, h)
    cmd = ['ffmpeg', '-v', 'error', '-ss', '%.4f' % t0, '-t', '%.4f' % (t1 - t0), '-i', src, '-vf', vf,
           '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    sz = w * h * 6
    while True:
        b = p.stdout.read(sz)
        if len(b) < sz:
            break
        yield np.frombuffer(b, '<u2').reshape(h, w, 3).astype(np.float32) / 65535
    p.wait()


def grade(a, con=0.08, sat=1.06, lift=0.0, gain=1.0, cool=0.0):
    a = np.clip(a * gain + lift, 0, 1)
    a = a + con * (a - a * a) * (2 * a - 1) * -2
    g = a.mean(axis=2, keepdims=True)
    a = g + (a - g) * sat
    if cool:
        a = a + np.array([-0.5, 0.0, 0.6], np.float32) * cool * (1 - a)
    return np.clip(a, 0, 1)


# ---------------------------------------------------------------- 書き出し
class Encoder:
    def __init__(self, path, n, w=W, h=H, crf=16):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.n, self.i = n, 0
        self.p = subprocess.Popen(
            ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-s', '%dx%d' % (w, h), '-r', str(FPS),
             '-i', '-', '-vf', 'scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int',
             '-frames:v', str(n), '-c:v', 'libx264', '-profile:v', 'high', '-level:v', '4.1', '-preset', 'medium',
             '-crf', str(crf), '-pix_fmt', 'yuv420p', '-color_primaries', 'bt709', '-color_trc', 'bt709',
             '-colorspace', 'bt709', '-color_range', 'tv', '-an', '-movflags', '+faststart', path],
            stdin=subprocess.PIPE)

    def write(self, f):
        # 量子化の前にごく弱いディザ（空や海のグラデーションの縞を防ぐ）
        f = np.clip(f, 0, 1)
        self.p.stdin.write((f * 65535 + 0.5).astype('<u2').tobytes())
        self.i += 1

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        assert self.i == self.n, (self.i, self.n)


def save_png(frame, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray((np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)).save(path)


def contact(frames, path, cols=4, scale=0.25):
    """フレームのリストを縮小して並べた確認用の1枚。"""
    sm = [cv2.resize(np.clip(f, 0, 1), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for f in frames]
    h, w = sm[0].shape[:2]
    rows = (len(sm) + cols - 1) // cols
    sheet = np.zeros((rows * (h + 6), cols * (w + 6), 3), np.float32)
    for i, f in enumerate(sm):
        r, q = divmod(i, cols)
        sheet[r * (h + 6):r * (h + 6) + h, q * (w + 6):q * (w + 6) + w] = f
    save_png(sheet, path)


# ---------------------------------------------------------------- 断面図（横から見た海）の共通の見た目
WL = 1150                       # 断面図の水面の高さ（px）
XS_SKY_TOP = c('1F5E96')        # 空の上端
XS_SKY_WL = c('5AA2D8')         # 水平線近くの空
XS_WATER_TOP = c('2F7FC0')      # 水面直下
XS_WATER_BOT = c('0A3F70')      # 画面下端の水


def xs_background(wl=WL):
    """断面図の背景：空のグラデーション＋水のグラデーション＋水面の白線は描かない（各ビートで描く）。"""
    f = np.empty((H, W, 3), np.float32)
    f[:wl] = vgrad(XS_SKY_TOP, XS_SKY_WL, h=wl)
    f[wl:] = vgrad(XS_WATER_TOP, XS_WATER_BOT, h=H - wl)
    return f


# ---------------------------------------------------------------- ビートのつなぎ（動きのぼかし）
def whip(frame, px, axis=1):
    """横（axis=1）または縦（axis=0）の動きのぼかし。px はぶれの長さ。ビートの最後／最初の数フレームで使い、
    前後のビートを同じ向きのぼかしでつなぐ（例：前のビートの最後を右へ whip、次のビートの最初を同じ強さから 0 へ）。"""
    k = int(round(abs(px)))
    if k < 2:
        return frame
    ker = np.ones((1, k), np.float32) / k if axis == 1 else np.ones((k, 1), np.float32) / k
    return cv2.filter2D(frame, -1, ker, borderType=cv2.BORDER_REFLECT)


def zoom_blur(frame, amount, cx=W / 2, cy=H / 2, steps=8):
    """中心 (cx, cy) から放射状のズームのぼかし。amount は最大の拡大率の差（例 0.15）。"""
    if amount < 0.005:
        return frame
    acc = np.zeros_like(frame)
    for i in range(steps):
        s = 1 + amount * i / (steps - 1)
        M = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]], np.float32)
        acc += cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    return acc / steps
