"""v2（施設PR版）の共通部品：写真の読み込み・奥行き・仮想カメラ・文字・書き出し。"""
import json
import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()

ROOT = '/home/user/maze'
PHOTO_DIR = os.path.join(ROOT, 'footage', '02_写真')
LP_DIR = os.path.join(ROOT, 'edl', 'lp', 'img')
CACHE = os.path.join(ROOT, 'exports', 'v2', '_cache')
FONT_DIR = os.path.join(ROOT, 'assets', 'fonts')
FF = '/usr/local/lib/python3.11/dist-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
FPS = 30
os.makedirs(CACHE, exist_ok=True)
PHOTOS = json.load(open(os.path.join(ROOT, 'edl', 'v2', 'photos.json')))


def photo_path(key):
    """key：photos.json の番号（int）か 'lp:about_s02.jpg' か、ファイル名。"""
    if isinstance(key, int):
        return os.path.join(PHOTO_DIR, PHOTOS[key]['file'])
    if key.startswith('lp:'):
        return os.path.join(LP_DIR, key[3:])
    return os.path.join(PHOTO_DIR, key)


def load(key, long_side=3200):
    """sRGB の float32（0..1）。長辺 long_side に縮小してキャッシュ。"""
    tag = str(key).replace(':', '_').replace('/', '_')
    cp = os.path.join(CACHE, 'img_%s_%d.npy' % (tag, long_side))
    if os.path.exists(cp):
        return np.load(cp)
    im = ImageOps.exif_transpose(Image.open(photo_path(key))).convert('RGB')
    s = long_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255.0
    np.save(cp, a)
    return a


_DA = None


def depth(key, img=None):
    """Depth Anything V2 Small（ONNX）の相対的な近さ（大きいほど手前）を 0..1 に正規化。画像と同じ大きさ。"""
    global _DA
    tag = str(key).replace(':', '_').replace('/', '_')
    if img is None:
        img = load(key)
    cp = os.path.join(CACHE, 'depth_%s_%dx%d.npy' % (tag, img.shape[1], img.shape[0]))
    if os.path.exists(cp):
        return np.load(cp)
    import onnxruntime as ort
    if _DA is None:
        _DA = ort.InferenceSession(os.path.join(ROOT, 'tools', 'models', 'depth_anything_v2_small.onnx'))
    h, w = img.shape[:2]
    s = 518 / min(h, w)
    nh, nw = int(round(h * s / 14)) * 14, int(round(w * s / 14)) * 14
    x = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    d = _DA.run(None, {'pixel_values': x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99)
    d = np.clip((d - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
    d = cv2.GaussianBlur(d, (0, 0), max(1.0, w / 800))
    np.save(cp, d)
    return d


def ease(t, kind='inout'):
    t = np.clip(t, 0, 1)
    if kind == 'in':
        return t * t * t
    if kind == 'out':
        return 1 - (1 - t) ** 3
    if kind == 'lin':
        return t
    return t * t * (3 - 2 * t) if kind == 'smooth' else (4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2)


def lerp(a, b, t):
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


def view(img, W, H, cx, cy, vw, dep=None, par=0.0, d0=0.5, rot=0.0):
    """仮想カメラ。画像の (cx,cy)（画像の比 0..1）を中心に、幅 vw（画像の幅の比）を W×H に写す。
    par：奥行きの視差（ドリー）。手前（dep大）ほど大きく写る。戻り値：(画, 出力→画像の写像 2x3)。"""
    h, w = img.shape[:2]
    # 画像の外（鏡に映ったような縁）が出ないよう、画角と中心を画像の内側に収める。視差の分だけ余白を足す
    m = 1.0 + (0.5 * abs(par) if dep is not None and par else 0.0) + 0.01
    vh = vw * w / W * H / h                            # 縦の画角（画像の高さの比）
    if vh * m > 1.0:
        vw = vw / (vh * m)
        vh = 1.0 / m
    if vw * m > 1.0:
        vh = vh / (vw * m)
        vw = 1.0 / m
    cx = float(np.clip(cx, vw * m / 2, 1 - vw * m / 2))
    cy = float(np.clip(cy, vh * m / 2, 1 - vh * m / 2))
    k = vw * w / W                                     # 出力1pxあたりの画像px
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    A = np.array([[k * c, -k * s, cx * w - k * (c * W / 2 - s * H / 2)],
                  [k * s, k * c, cy * h - k * (s * W / 2 + c * H / 2)]], np.float32)
    if dep is None or par == 0:
        out = cv2.warpAffine(img, A, (W, H), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        return out, A
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    mx = A[0, 0] * xs + A[0, 1] * ys + A[0, 2]
    my = A[1, 0] * xs + A[1, 1] * ys + A[1, 2]
    dd = cv2.remap(dep, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    f = 1.0 / (1.0 + par * (dd - d0))
    pcx, pcy = cx * w, cy * h
    mx = (pcx + (mx - pcx) * f).astype(np.float32)
    my = (pcy + (my - pcy) * f).astype(np.float32)
    out = cv2.remap(img, mx, my, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    return out, A


def to_out(A, x, y):
    """画像の座標（px）→ 出力の座標。"""
    M = np.vstack([A, [0, 0, 1]])
    p = np.linalg.inv(M) @ np.array([x, y, 1.0])
    return p[0], p[1]


def grade(a, con=0.06, sat=1.04, lift=0.0, gain=1.0):
    a = np.clip(a * gain + lift, 0, 1)
    a = a + con * (a - a * a) * (2 * a - 1) * -2          # ゆるいSカーブ
    g = a.mean(axis=2, keepdims=True)
    return np.clip(g + (a - g) * sat, 0, 1)


def font(size, weight='Regular'):
    for f in ('NotoSansJP-%s.ttf' % weight, 'NotoSansJP-Regular.ttf'):
        p = os.path.join(FONT_DIR, f)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    raise FileNotFoundError(FONT_DIR)


def text_layer(W, H, items):
    """items：[(text, x, y, size, opts)]。白・やわらかい影の RGBA（float 0..1）。
    opts：anchor（PILのアンカー）、weight、track（字間 em）、alpha、shadow（強さ）。"""
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    sh = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(im)
    ds = ImageDraw.Draw(sh)
    for text, x, y, size, o in items:
        f = font(int(size), o.get('weight', 'Regular'))
        tr = o.get('track', 0.08) * size
        a = int(255 * o.get('alpha', 1.0))
        anchor = o.get('anchor', 'ls')
        widths = [f.getlength(ch) for ch in text]
        total = sum(widths) + tr * (len(text) - 1)
        x0 = x - (total / 2 if anchor[0] == 'm' else total if anchor[0] == 'r' else 0)
        cx = x0
        for ch, wd in zip(text, widths):
            d.text((cx, y), ch, font=f, fill=(255, 255, 255, a), anchor='l' + anchor[1])
            ds.text((cx, y), ch, font=f, fill=int(a * o.get('shadow', 0.55)), anchor='l' + anchor[1])
            cx += wd + tr
    sh1 = np.asarray(sh.filter(ImageFilter.GaussianBlur(max(2, H * 0.004)))).astype(np.float32) / 255
    sh2 = np.asarray(sh.filter(ImageFilter.GaussianBlur(max(6, H * 0.014)))).astype(np.float32) / 255
    t = np.asarray(im).astype(np.float32) / 255
    return t, np.clip(sh1 * 0.65 + sh2 * 0.35, 0, 1)


def put_text(frame, layer, alpha=1.0):
    t, sh = layer
    if alpha <= 0:
        return frame
    f = frame * (1 - sh[..., None] * 0.6 * alpha)
    a = t[..., 3:4] * alpha
    return f * (1 - a) + t[..., :3] * a


class Encoder:
    def __init__(self, path, W, H, n):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.n, self.i = n, 0
        self.p = subprocess.Popen(
            [FF, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-s', '%dx%d' % (W, H), '-r', str(FPS),
             '-i', '-', '-vf', 'scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int',
             '-frames:v', str(n), '-c:v', 'libx264', '-profile:v', 'high', '-level:v', '4.1', '-preset', 'slow',
             '-crf', '16', '-pix_fmt', 'yuv420p', '-color_primaries', 'bt709', '-color_trc', 'bt709',
             '-colorspace', 'bt709', '-color_range', 'tv', '-an', '-movflags', '+faststart', path],
            stdin=subprocess.PIPE)

    def write(self, f):
        self.p.stdin.write((np.clip(f, 0, 1) * 65535 + 0.5).astype('<u2').tobytes())
        self.i += 1

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        assert self.i == self.n, (self.i, self.n)


def read_video(path, W, H, n=None, start=0):
    """焼き込み済みクリップ（BT.709 limited）を sRGB 相当の float で読む。"""
    cmd = [FF, '-v', 'error', '-ss', '%.4f' % (start / FPS), '-i', path, '-vf',
           'scale=%d:%d:in_color_matrix=bt709:in_range=tv:flags=lanczos+accurate_rnd,format=rgb48le' % (W, H)]
    if n:
        cmd += ['-frames:v', str(n)]
    cmd += ['-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    sz = W * H * 6
    while True:
        b = p.stdout.read(sz)
        if len(b) < sz:
            break
        yield np.frombuffer(b, '<u2').reshape(H, W, 3).astype(np.float32) / 65535
    p.wait()
