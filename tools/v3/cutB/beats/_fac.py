"""案B 後半（08〜13：実写と建物の写真）の共通部品。

- 写真の読み込み（HEIC 可）と縮小キャッシュ（exports/v3/cache/*.npy、uint8）
- 写真の色：案A の実写（_real.GRADE：すっきり寒色寄り）に合わせる。料理は b14 の _grade（暖かさを残す）
- 写真の切り出し：view(img, cx, cy, vw) — 画像の中心 (cx, cy)（比）・幅 vw（画像の幅に対する比）の 9:16 の窓
- テロップ：フロアの見出し（大 Bold＋小 Medium の2行、セーフゾーンの上寄り）と、文字の下だけ空を落とす影
- 屋上・タワーの動画（IMG_9660）の読み出し条件
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageOps

from cutB.beats import _useA  # noqa: F401
import core
from core import W, H
from beats import _real

FAC = os.path.join(core.FOOT, 'facility')
CACHE = os.path.join(core.OUT, 'cache')
os.makedirs(CACHE, exist_ok=True)

EXTERIOR = '76C67856-9213-47E0-ACDD-8C5ECD799F4C.JPEG'     # 外観（昼、切妻のロゴ）
LAWSON = 'CFF7DD45-7510-4E88-A096-8F90AC7CBF86.JPEG'       # 1F ローソンの入口

# 屋上・タワーから見た水平線と河口（縦 1x、ほぼ固定）。上寄りに 1.72 倍で切り出して下の柵・遊歩道・車を外す。
# 水平線の傾きは実測（3.0s で右上がり約1.13°、5.4s で約0.87°）→ +1.05° 回す。右端の小さなサーファーは窓の外。
DECK = dict(name='facility/IMG_9660.MOV', t0=3.0, rot=1.05, cx=0.5, cy=0.33, zoom=1.72)



def fast_clip(spec, n):
    """_real.clip と同じ素材の n 枚（素材の1フレーム＝出力1フレーム＝0.5倍スロー）を _real.grade して返す。
    core.footage と同じ処理だが、回転と切り出しを HLG→SDR の変換の前にする（変換する画素が 1/3 になり速い）。
    回転・切り出しの補間を変換の前にするだけなので、差は細部の補間だけ（平均 0.004 程度、目では見分けられない）。"""
    src = os.path.join(core.FOOT, spec['name'])
    cw, ch = 2160 / spec['zoom'], 3840 / spec['zoom']
    x0 = min(max(spec['cx'] * 2160 - cw / 2, 0), 2160 - cw)
    y0 = min(max(spec['cy'] * 3840 - ch / 2, 0), 3840 - ch)
    vf = 'rotate=%f:ow=iw:oh=ih:c=black,crop=%d:%d:%d:%d,' % (np.deg2rad(spec['rot']), cw, ch, x0, y0)
    vf += core.TONEMAP + ',zscale=w=%d:h=%d:f=lanczos,format=rgb48le' % (W, H)
    t0 = spec['t0']
    cmd = ['ffmpeg', '-v', 'error', '-ss', '%.4f' % t0, '-t', '%.4f' % ((n + 2) / 60.0), '-i', src, '-vf', vf,
           '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-']
    import subprocess
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    sz = W * H * 6
    k, last = 0, None
    while k < n:
        b = p.stdout.read(sz)
        if len(b) < sz:
            break
        last = _real.grade(np.frombuffer(b, '<u2').reshape(H, W, 3).astype(np.float32) / 65535)
        yield last
        k += 1
    p.stdout.close()
    p.wait()
    while k < n:
        yield last
        k += 1


Y_BIG, Y_SUB = 430, 512          # 見出しの2行のベースライン（08 の b12 と同じ高さの帯：空の部分）
SZ_BIG, SZ_SUB = 96, 46


def load(name, long_side):
    """facility/ の写真を、長辺 long_side px に縮めた float32 RGB（0..1）。uint8 でキャッシュ。"""
    path = os.path.join(CACHE, 'fac_%s_%d.npy' % (os.path.splitext(name)[0][:12], long_side))
    if os.path.exists(path):
        return np.load(path).astype(np.float32) / 255
    if name.lower().endswith('.heic'):
        import pillow_heif
        pillow_heif.register_heif_opener()
    im = ImageOps.exif_transpose(Image.open(os.path.join(FAC, name))).convert('RGB')
    r = long_side / max(im.size)
    im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)
    a = np.asarray(im)
    np.save(path, a)
    return a.astype(np.float32) / 255


def grade_photo(a, gain=1.0):
    """建物・産直の写真：案A の実写と同じ grade（寒色寄り・すっきり）。写真は SDR なのでハイライトだけ少し抑える。"""
    return _real.grade(core.grade(np.clip(a * gain, 0, 1), con=0.0, sat=1.0, gain=0.97))


def view(img, cx, cy, vw, interp=cv2.INTER_LANCZOS4):
    """画像の (cx, cy)（比）を中心に、画像の幅の vw 倍の幅の 9:16 の窓を W×H に。窓は画像の内側に収める。"""
    h, w = img.shape[:2]
    cw = vw * w
    chh = cw * H / W
    if chh > h:                                   # 縦が足りないときは縦いっぱい
        chh = h
        cw = chh * W / H
    x0 = float(np.clip(cx * w - cw / 2, 0, w - cw))
    y0 = float(np.clip(cy * h - chh / 2, 0, h - chh))
    s = W / cw
    M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=interp, borderMode=cv2.BORDER_REFLECT)


def max_vw(img):
    """縦いっぱいにしたときの vw（これより広い窓は取れない）。"""
    h, w = img.shape[:2]
    return h * W / H / w


def luma(f):
    return float((f * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(2).mean())


# ---------------------------------------------------------------- テロップ
class Text:
    """core.text_layer の結果を、文字と影のある矩形だけに切り詰めて持つ（put は core.put_text と同じ式で、矩形の中だけ計算）。"""

    def __init__(self, layer, dx=0):
        t, sh = layer
        m = (t[..., 3] > 0) | (sh > 0)
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            self.box = None
            return
        self.box = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
        y0, y1, x0, x1 = self.box
        self.t = t[y0:y1, x0:x1].copy()
        self.sh = sh[y0:y1, x0:x1, None].copy()

    def put(self, frame, alpha=1.0, dx=0):
        """frame に重ねた新しい配列を返す（frame は変えない）。dx：横のずらし（px、整数に丸める）。"""
        if alpha <= 0 or self.box is None:
            return frame
        y0, y1, x0, x1 = self.box
        k = int(round(dx))
        x0, x1 = x0 + k, x1 + k
        out = frame.copy()
        r = out[y0:y1, x0:x1]
        r = r * (1 - self.sh * 0.55 * alpha)
        a = self.t[..., 3:4] * alpha
        out[y0:y1, x0:x1] = r * (1 - a) + self.t[..., :3] * a
        return out


def shade(y0=Y_BIG - 150, y1=Y_SUB + 90, depth=0.22):
    """文字の帯の後ろだけをやわらかく暗くする縦のプロファイル（H,1,1）。端は滑らかに 0 へ。"""
    y = np.arange(H, dtype=np.float32)
    m = np.clip(np.minimum((y - (y0 - 120)) / 120.0, ((y1 + 140) - y) / 140.0), 0, 1)
    m = m * m * (3 - 2 * m)
    return (1 - depth * m)[:, None, None]


def frost(f, y0=Y_BIG - 120, y1=Y_SUB + 50, sigma=9.0, depth=0.18):
    """文字の帯の後ろを、すりガラスのように少しぼかして暗くする（値札などの細かい文字を読めなくする）。"""
    a, b = max(0, y0 - 140), min(H, y1 + 140)
    band = f[a:b]
    bl = cv2.GaussianBlur(band, (0, 0), sigma)
    y = np.arange(a, b, dtype=np.float32)
    m = np.clip(np.minimum((y - (y0 - 110)) / 110.0, ((y1 + 110) - y) / 110.0), 0, 1)
    m = (m * m * (3 - 2 * m))[:, None, None]
    out = f.copy()
    out[a:b] = band * (1 - m) + bl * (1 - depth) * m
    return out


def heading(big, sub, hl=None, y_big=Y_BIG, y_sub=Y_SUB, sz_big=SZ_BIG, sz_sub=SZ_SUB, shadow=0.9):
    """見出し（大：Bold、小：Medium）を1つの Text に。"""
    items = [dict(text=big, x=W / 2, y=y_big, size=sz_big, weight='Bold', hl=hl, track=0.03, shadow=shadow)]
    if sub:
        items.append(dict(text=sub, x=W / 2, y=y_sub, size=sz_sub, weight='Medium', track=0.06, shadow=shadow))
    return Text(core.text_layer(items))


def heading2(big, sub, hl=None, y_big=Y_BIG, y_sub=Y_SUB, sz_big=SZ_BIG, sz_sub=SZ_SUB, shadow=0.9):
    """見出しの2行を別々に出したいとき：(大の Text, 小の Text)。"""
    lb = Text(core.text_layer([dict(text=big, x=W / 2, y=y_big, size=sz_big, weight='Bold', hl=hl, track=0.03,
                                    shadow=shadow)]))
    ls = Text(core.text_layer([dict(text=sub, x=W / 2, y=y_sub, size=sz_sub, weight='Medium', track=0.06,
                                    shadow=shadow)]))
    return lb, ls


def split_line(parts, y, size, weight='Medium', track=0.06, shadow=0.9, hl=None):
    """1行を中央ぞろえのまま、部分ごとに別の text_layer にする（ナレーションに合わせて部分ごとに出すため）。"""
    tr = track * size
    widths = [core.text_width(p, size, weight, track) for p in parts]
    total = sum(widths) + tr * (len(parts) - 1)
    x = W / 2 - total / 2
    out = []
    for p, wd in zip(parts, widths):
        out.append(Text(core.text_layer([dict(text=p, x=x, y=y, size=size, weight=weight, anchor='ls', track=track,
                                              shadow=shadow, hl=hl)])))
        x += wd + tr
    return out


def appear(i, f_on, dur=5, n=None, fout=0):
    """f_on から dur フレームで出る不透明度（n と fout を渡すと最後の fout フレームで消える）。"""
    if i < f_on:
        return 0.0
    a = core.ease(min(1.0, (i - f_on + 1) / dur), 'out')
    if n is not None and fout:
        a *= float(np.clip((n - i) / fout, 0, 1))
    return a


def check_text_safe(layer):
    """テロップの不透明な画素がセーフゾーンの中にあるか（開発用）。"""
    if layer.box is None:
        return True
    by0, _, bx0, _ = layer.box
    ys, xs = np.nonzero(layer.t[..., 3] > 0.05)
    if len(xs) == 0:
        return True
    ys, xs = ys + by0, xs + bx0
    x0, y0, x1, y1 = core.SAFE
    return xs.min() >= x0 and xs.max() <= x1 and ys.min() >= y0 and ys.max() <= y1
