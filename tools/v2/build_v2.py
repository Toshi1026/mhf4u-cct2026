#!/usr/bin/env python3
"""v2（施設PR版・FIXED RULES v2）の4本を書き出す。

  python3 tools/v2/build_v2.py ig30 ig15 sg30 sg15     # → exports/v2/MAZE_v2_*.mp4
  python3 tools/v2/build_v2.py ig30 --stills 0,90,300   # 確認用の静止画だけ

構成は公式LPの順：場所 → 建物（昼→夜） → 1F → 2F → フロアガイド（3F・屋上・タワー） → END。
写真は Depth Anything V2 の奥行きで 2.5D に動かす。昼→夜は relight.py。
"""
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/user/maze/exports/capcut/_scripts')
import common as C  # noqa: E402
import relight as RL  # noqa: E402
import mapzoom as MZ  # noqa: E402

OUT = os.path.join(C.ROOT, 'exports', 'v2')
CAP = os.path.join(C.ROOT, 'exports', 'capcut')
FOOT = os.path.join(C.ROOT, 'footage', '01_撮影素材')


# ---------------------------------------------------------------------------
# 素材
# ---------------------------------------------------------------------------
def raw_video(src, t0, n, speed, W, H, crop=None, vf_extra='', rot_cw=0.0):
    """原素材（iPhone HLG）→ SDR、速度、切り出し。crop：(x,y,w,h) を素材の比で。"""
    import render_clips as RC
    vf = [RC.TONEMAP]
    if rot_cw:
        vf.append('rotate=%f*PI/180:fillcolor=black' % rot_cw)
    if crop:
        vf.append("crop=iw*%f:ih*%f:iw*%f:ih*%f" % (crop[2], crop[3], crop[0], crop[1]))
    vf.append('scale=%d:%d:flags=lanczos' % (W, H))
    if vf_extra:
        vf.append(vf_extra)
    vf.append('setpts=%g*(PTS-STARTPTS),fps=30,format=rgb48le' % (1 / speed))
    p = subprocess.Popen([C.FF, '-v', 'error', '-ss', '%.4f' % t0, '-i', src, '-an', '-vf', ','.join(vf),
                          '-frames:v', str(n), '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-'], stdout=subprocess.PIPE)
    sz = W * H * 6
    k = 0
    last = None
    while k < n:
        b = p.stdout.read(sz)
        if len(b) < sz:
            break
        last = np.frombuffer(b, '<u2').reshape(H, W, 3).astype(np.float32) / 65535
        k += 1
        yield last
    p.wait()
    while k < n:                 # 足りなければ最後のコマで埋める（起きないはず）
        k += 1
        yield last


def baked(path, n, W, H, start=0):
    """焼き込み済みクリップを1コマずつ（メモリに全部は載せない）。足りなければ最後のコマで埋める。"""
    k, last = 0, None
    for last in C.read_video(path, W, H, n=n, start=start):
        k += 1
        yield last
    while k < n:
        k += 1
        yield last


# ---------------------------------------------------------------------------
# 文字
# ---------------------------------------------------------------------------
class Txt:
    """出る・消えるをコマで決める文字。layout(W,H) → items。"""

    def __init__(self, f0, f1, items_fn, fade=12):
        self.f0, self.f1, self.items_fn, self.fade = f0, f1, items_fn, fade
        self.cache = {}

    def alpha(self, f):
        if f < self.f0 or f >= self.f1:
            return 0.0
        a = min(1.0, (f - self.f0 + 1) / self.fade, (self.f1 - f) / self.fade)
        return float(C.ease(a, 'smooth'))

    def layer(self, W, H):
        if (W, H) not in self.cache:
            self.cache[(W, H)] = C.text_layer(W, H, self.items_fn(W, H))
        return self.cache[(W, H)]


def floor_label(W, H, floor, name, sub=None, base=None):
    """フロア名（左下）。1F は細めの大きい数字、横に施設名。後ろに、左端から文字の右まで、縁をぼかした
    すりガラスの帯を敷く（産直の値札・白く飛んだ窓・砂浜の上でも読めるように）。
    IG は Reels の下の UI（y>0.65H）にかからない高さ。base：ベースライン（H の比）。"""
    ig = W < H
    m = 0.06 * W if ig else 0.05 * W
    base = H * (base or (0.612 if ig else 0.86))
    big = H * (0.045 if ig else 0.075)
    small = H * (0.019 if ig else 0.032)
    it = [(floor, m, base, big, dict(weight='Light' if os.path.exists(os.path.join(C.FONT_DIR, 'NotoSansJP-Light.ttf')) else 'Regular', track=0.02))]
    it.append((name, m + big * 1.25, base, small * 1.15, dict(track=0.12)))
    if sub:
        it.append((sub, m + big * 1.25, base + small * (1.5 if ig else 1.7), small * 0.8, dict(track=0.14, alpha=0.9)))
    boxes = [C.text_box(*t) for t in it]
    x1 = max(b[2] for b in boxes)
    y0, y1 = min(b[1] for b in boxes), max(b[3] for b in boxes)
    band = C.plate((-W * 0.2, y0 - big * 0.35, x1 + big * 0.9, y1 + big * 0.35), feather=big * 0.30,
                   sigma=big * 0.15, target=0.34, kmin=0.22, kmax=0.62)
    return [band] + it


# ---------------------------------------------------------------------------
# ショット（それぞれ n コマのリストかジェネレータを返す）
# ---------------------------------------------------------------------------
def photo_move(key, n, W, H, path, par=0.0, grade=None, long_side=3200):
    """path(u) → (cx, cy, vw, par_scale)。u は 0..1。"""
    img = C.load(key, long_side)
    img = C.grade(img, **(grade or {}))
    dep = C.depth(key, C.load(key, long_side)) if par else None
    for i in range(n):
        u = i / max(1, n - 1)
        cx, cy, vw, ps = path(u)
        f, _ = C.view(img, W, H, cx, cy, vw, dep, par * ps)
        yield f


def shot_place(n, W, H, fmt):
    """1：太平洋と仁淀川の交差点（タワーの高さから見た河口と砂州）。"""
    if fmt == 'ig':
        return baked(os.path.join(CAP, 'proposals', 'ig_v8_ig30', '02_IMG_9665.mp4'), n, W, H)
    return baked(os.path.join(CAP, 'signage30', '02_IMG_9665.mp4'), n, W, H)


def shot_road(n, W, H, fmt):
    """2：サイクリングロードの先に建物と海（54）。奥行きでゆっくりドリー。"""
    if fmt == 'ig':
        path = lambda u: (C.lerp(0.22, 0.30, u), 0.62, C.lerp(0.33, 0.20, C.ease(u, 'inout')), 1)
        par = 0.3                                          # 縦は画角の余白が少ないので、視差は控えめにして寄りを大きく
    else:
        path = lambda u: (C.lerp(0.36, 0.42, u), 0.68, C.lerp(0.86, 0.62, C.ease(u, 'inout')), 1)
        par = 0.6
    return photo_move(54, n, W, H, path, par=par, grade=dict(con=0.05, sat=1.04))


def shot_aerial(n, W, H, fmt):
    """0：冒頭。LP の空撮（仁淀川河口大橋と河口・太平洋）を奥行きでゆっくり押し込む。"""
    if fmt == 'ig':
        path = lambda u: (C.lerp(0.40, 0.44, u), 0.50, C.lerp(0.42, 0.30, C.ease(u, 'out')), 1)
    else:
        path = lambda u: (0.50, 0.52, C.lerp(0.98, 0.78, C.ease(u, 'out')), 1)
    return photo_move('lp:about_s02.jpg', n, W, H, path, par=0.5, grade=dict(con=0.04, sat=1.03), long_side=1500)


# 地名：(経度, 緯度)、名前、出る・消える（ショットの中の比）、大きさ。位置は、明るい田畑・白い河原・道筋を避けた暗い所
MAP_LABELS = [((133.56, 33.66), '高知県', 0.15, 0.40, 1.25),        # 県の色がついてから、県の内側に
              ((133.425, 33.47), '土佐市', 0.40, 0.62, 1.0),        # 市街地の南の山林（仁淀川の河原にかからない）
              (MZ.KATSURA, '桂浜', 0.50, 0.79, 0.9),                # 点の右下の海（道筋は左下へ伸びる）
              ((133.48, 33.485), '仁淀川', 0.64, 0.86, 0.9)]        # 下流のすぐ東の山林。土佐市と同時には出ない
MAP_OFFSET = {'桂浜': (0.020, 0.015)}                               # 点からのずらし（W, H の比）。anchor は左上


def safe_rect(W, H):
    """文字を置いてよい範囲。IG は Reels の UI（上14%・下35%・左右6%）を除く。サイネージは画面の内側。"""
    if W < H:
        return W * 0.06, H * 0.14, W * 0.94, H * 0.65
    return W * 0.03, H * 0.04, W * 0.97, H * 0.96


def edge_fade(box, W, H, ramp=0.04):
    """文字の枠が安全な範囲の端に近づいたら薄くし、はみ出す前に消す。"""
    L, T, R, B = safe_rect(W, H)
    x0, y0, x1, y1 = box
    d = min(x0 - L, R - x1, y0 - T, B - y1)
    return float(C.ease(np.clip(d / (W * ramp), 0, 1), 'smooth'))


def shot_map(n, W, H, fmt):
    """1：日本 → 高知県 → 土佐市 → 桂浜からの道筋 → 南風の真上。地名は地図の上の位置に貼りつく。"""
    ig = W < H
    small = H * (0.024 if ig else 0.036)
    # 出典：IG は Reels の下の UI より上に2行で。どちらも細い暗い帯の上に置く（白地図でも航空写真でも読める）
    if ig:
        lines = [('経路：© OpenStreetMap contributors', W * 0.93, H * 0.611),
                 ('地図：国土地理院（シームレス空中写真）・Natural Earth', W * 0.93, H * 0.629)]
        cs = H * 0.0105
    else:
        lines = [('地図：国土地理院（シームレス空中写真）・Natural Earth　経路：© OpenStreetMap contributors', W * 0.97, H * 0.975)]
        cs = H * 0.016
    citems = []
    for t, x, y in lines:
        o = dict(anchor='rs', alpha=0.9, track=0.02, shadow=0.3)
        bx = C.text_box(t, x, y, cs, o)
        citems.append(C.plate((bx[0] - cs * 0.5, bx[1] - cs * 0.3, bx[2] + cs * 0.5, bx[3] + cs * 0.3), kind='dim', k=0.42,
                              feather=cs * 0.12, radius=cs * 0.3))
        citems.append((t, x, y, cs, o))
    credit = C.text_layer(W, H, citems)
    for i, (f, z, cx, cy) in enumerate(MZ.frames(n, W, H, fmt)):
        u = i / (n - 1)
        items = []
        for ll, name, a0, a1, sz in MAP_LABELS:
            a = min(C.ease(np.clip((u - a0) / 0.06, 0, 1), 'smooth'), C.ease(np.clip((a1 - u) / 0.06, 0, 1), 'smooth'))
            if a <= 0:
                continue
            x, y = MZ.to_screen(ll, z, cx, cy, W, H)
            o = dict(anchor='mm', track=0.2, shadow=0.85)
            if name in MAP_OFFSET:
                dx, dy = MAP_OFFSET[name]
                x, y, o['anchor'] = x + W * dx, y + H * dy, 'lt'
            a *= edge_fade(C.text_box(name, x, y, small * sz, o), W, H)
            if a <= 0.01:
                continue
            o['alpha'] = float(a)
            items.append((name, x, y, small * sz, o))
        a = float(C.ease(np.clip((u - 0.88) / 0.06, 0, 1), 'smooth'))
        if a > 0:
            x, y = MZ.to_screen(MZ.MAZE, z, cx, cy, W, H)
            cv2.circle(f, (int(x), int(y)), int(small * 0.45), (1, 1, 1), -1, cv2.LINE_AA)
            cv2.circle(f, (int(x), int(y)), int(small * (0.9 + 0.6 * (1 - a))), (1, 1, 1), max(2, int(small * 0.08)), cv2.LINE_AA)
            # 名前はピンの輪（いちばん大きいとき半径 1.5×small）より上。橋・砂浜の上なので小さなすりガラスを敷く
            o = dict(anchor='mm', alpha=a, track=0.12, shadow=0.8)
            ty = y - small * 2.6
            bx = C.text_box('Seaside MAZE ｜ 南風', x, ty, small * 1.05, o)
            pd = small * 0.45
            items.append(C.plate((bx[0] - pd * 1.4, bx[1] - pd, bx[2] + pd * 1.4, bx[3] + pd), feather=pd * 0.5, radius=pd,
                                 alpha=a, sigma=small * 0.25, target=0.30, kmin=0.25, kmax=0.6))
            items.append(('Seaside MAZE ｜ 南風', x, ty, small * 1.05, o))
        if items:
            f = C.put_text(f, C.text_layer(W, H, items))
        f = C.put_text(f, credit, 1.0)
        yield f


def shot_driveby(n, W, H, fmt):
    """2：車窓。仁淀川を渡ると左手に南風。建物の写真（44）を横に流し、手前ほど速く動く視差と、
    コマの間を3枚重ねた本物のモーションブラーで、車の窓から見ているように。"""
    img = C.grade(C.load(44), con=0.05, sat=1.04)
    dep = C.depth(44, C.load(44))
    vw = 0.36 if fmt == 'ig' else 0.55
    cy = 0.50 if fmt == 'ig' else 0.42                     # サイネージは少し上を向けて、文字を置く空の帯を作る
    for i in range(n):
        acc = 0
        for sub in (-0.33, 0.0, 0.33):
            u = (i + sub) / (n - 1)
            e = float(np.clip(u, 0, 1))
            cx = C.lerp(0.20, 0.80, e) if fmt == 'ig' else C.lerp(0.28, 0.72, e)
            f, _ = C.view(img, W, H, cx, cy, vw, dep, 0.0, lat=C.lerp(-0.05, 0.05, e))
            acc = acc + f
        yield acc / 3


def shot_relight(n, W, H, fmt):
    """3：昼→夜。押し込みながら暮れて、明かりが左から灯り、最後に看板が光る。終わりは入口の光へ。"""
    P = RL.prepare()
    dep = C.depth(28, C.load(28))
    h, w = P['day'].shape[:2]
    door = RL.night_door(P)
    for i in range(n):
        u = i / (n - 1)
        t = np.clip((u - 0.08) / 0.62, 0, 1)
        img = RL.frame(P, float(t))
        if fmt == 'ig':
            vw0, vw1 = 0.36, 0.26
            cy0 = 0.45
        else:
            vw0, vw1 = 0.92, 0.62
            cy0 = 0.47
        e = C.ease(u, 'inout')
        cx = C.lerp(0.47, door[0] / w, C.ease(np.clip((u - 0.75) / 0.25, 0, 1), 'in'))
        cy = C.lerp(cy0, door[1] / h, C.ease(np.clip((u - 0.75) / 0.25, 0, 1), 'in'))
        vw = C.lerp(vw0, vw1, e) * C.lerp(1, 0.25, C.ease(np.clip((u - 0.78) / 0.22, 0, 1), 'in'))
        f, A = C.view(img, W, H, cx, cy, vw, dep, 0.30)
        wh = C.ease(np.clip((u - 0.9) / 0.1, 0, 1), 'in')       # 入口の光へ白く抜ける
        yield np.clip(f * (1 - wh) + wh, 0, 1)


def blob_center(img, hue, tol=14, smin=0.35):
    """色（OpenCV の Hue 0..180）のかたまりの重心（画像の比）。"""
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hh = hsv[..., 0].astype(np.int16)
    d = np.minimum(abs(hh - hue), 180 - abs(hh - hue))
    m = ((d < tol) & (hsv[..., 1] > smin * 255) & (hsv[..., 2] > 60)).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), img.shape[1] / 40)
    y, x = np.unravel_index(np.argmax(m), m.shape)
    return x / img.shape[1], y / img.shape[0]


# 1F の色のリズム：黄 → 橙 → 赤 → 紫 → 緑。色のかたまりがいつも画面の同じ位置に来るように切り出す
PRODUCE = [(7, 25), (53, 25), (72, 25), (46, 12), (65, 12), (42, 2), (108, 2), (55, 2), (102, 160), (29, 150),
           (31, 150), (109, 45), (26, 55)]
# 切り出しの横のずらし（画像の幅の比）。IG の直七（109）は、右の値札がフロア名のすぐ右にくっきり残るので右へ逃がす
PRODUCE_DX = {'ig': {109: -0.10}, 'sg': {}}


def tail_1f(n):
    """1F の最後の地酒（仁淀川）のカットのコマ数。文字はこのカットの手前で消す。"""
    return 20 if n >= 100 else 12


def shot_1f(n, W, H, fmt, cuts=None):
    """4：1F。ローソンの入口 → 色でつなぐ産直の連打 → 地場の酒（仁淀川）。"""
    cuts = cuts or PRODUCE
    head = 20 if n >= 100 else 10
    tail = tail_1f(n)
    per = max(5, (n - head - tail) // len(cuts))
    rest = n - head - tail - per * len(cuts)
    head += rest
    vw57 = (0.30, 0.22) if fmt == 'ig' else (0.62, 0.52)
    yield from photo_move(57, head, W, H, lambda u: (0.40, 0.66, C.lerp(*vw57, C.ease(u, 'out')), 1), par=0.3)
    for key, hue in cuts:
        img = C.grade(C.load(key, 2400), con=0.05, sat=1.06)
        cx, cy = blob_center(img, hue)
        ar = img.shape[1] / img.shape[0]
        vw = (0.42 if fmt == 'ig' else 0.62) * (1.0 if ar > 1 else 1.5)
        cx = np.clip(cx + PRODUCE_DX[fmt].get(key, 0.0), vw / 2, 1 - vw / 2)
        vh = vw * img.shape[1] / img.shape[0] * H / W
        cy = np.clip(cy, vh / 2, 1 - vh / 2)
        for i in range(per):
            u = i / max(1, per - 1)
            f, _ = C.view(img, W, H, cx, cy, vw * C.lerp(1.0, 0.94, u))
            yield f
    yield from photo_move(101, tail, W, H, lambda u: (0.42, 0.55, C.lerp(0.62, 0.55, u) if fmt == 'ig' else C.lerp(0.95, 0.88, u), 1), par=0.3)


def cuts_2f(n, fmt):
    """2F の3カット（階段・カウンター・窓の外の海）のコマ数。"""
    c = min(n - int(n * 0.55), 69 if fmt == 'ig' else 90)
    a = int((n - c) * 0.55)
    return a, n - c - a, c


def shot_2f(n, W, H, fmt):
    """5：階段（OCEANVIEW）を上がる → 2F のカウンターと海 → 窓の外の海（動画）。"""
    a, b, c = cuts_2f(n, fmt)
    # 階段：下から見上げて、上がっていく（奥行きで視差）
    path = (lambda u: (0.55, C.lerp(0.62, 0.40, C.ease(u, 'inout')), C.lerp(0.62, 0.48, u), 1)) if fmt == 'ig' else \
        (lambda u: (0.55, C.lerp(0.60, 0.45, C.ease(u, 'inout')), C.lerp(1.0, 0.85, u), 1))
    yield from photo_move(85, a, W, H, path, par=0.45, long_side=3024)
    path = (lambda u: (0.42, 0.50, C.lerp(0.55, 0.46, C.ease(u, 'out')), 1)) if fmt == 'ig' else \
        (lambda u: (0.50, 0.55, C.lerp(1.0, 0.88, C.ease(u, 'out')), 1))
    yield from photo_move(87, b, W, H, path, par=0.4, long_side=3024)
    if fmt == 'ig':
        yield from baked(os.path.join(CAP, 'proposals', 'ig_qc_fix', 'ig30', '05_IMG_9649.mp4'), c, W, H, start=max(0, 69 - c))
    else:
        yield from baked(os.path.join(CAP, 'signage30', '03_IMG_9674.mp4'), c, W, H, start=max(0, 90 - c))


# フロアガイド（56）：各フロアの点の位置（900×600 に縮めた座標）。IG と サイネージで画角が違うので分ける
#   IG：カメラは横にだけ動き、1F はいつも Reels の下の UI の中にある。点は1Fの柱、名前は UI より上に出して線でつなぐ
#   サイネージ：1F の点はローソンの看板の上のひさし（看板の文字を避ける）
FLOORS = [('1F', 'ローソン ＋ 産直マーケット', {'ig': (560, 450), 'sg': (345, 412)}),
          ('2F', 'オーシャンビューカフェ', {'ig': (640, 340), 'sg': (640, 355)}),
          ('3F', '展望スペース', {'ig': (600, 300), 'sg': (600, 300)}),
          ('', '津波避難タワー', {'ig': (700, 255), 'sg': (700, 255)})]
# 名前の置き方：IG は点の真上に中央揃え（横は安全な範囲に収める）。サイネージは点の横（左右は1本のあいだ固定）
FLOOR_SIDE = {'1F': 'right_mid', '2F': 'left', '3F': 'right', '': 'right_high'}
FLOOR_T0 = {'ig': (0.10, 0.36, 0.52, 0.66), 'sg': (0.12, 0.29, 0.46, 0.63)}


def floors_cam(u, fmt):
    e = C.ease(u, 'inout')
    if fmt == 'ig':
        return C.lerp(0.48, 0.72, e), C.lerp(0.70, 0.46, e), 0.36
    return C.lerp(0.58, 0.64, e), C.lerp(0.60, 0.50, e), C.lerp(0.66, 0.58, u)


def floor_tag(W, H, fmt, label, ox, oy, a):
    """1フロア分の items（すりガラスの小さな板＋名前＋点＋必要なら引き出し線）。"""
    ig = fmt == 'ig'
    size = H * (0.025 if ig else 0.034)
    o = dict(track=0.10, alpha=a, shadow=0.35)
    lw = C.text_width(label, size, track=0.10)
    L, T, R, B = safe_rect(W, H)
    side = FLOOR_SIDE[label[:2] if label[:1].isdigit() else '']
    if ig:
        # 点の真上（文字の下端と点のあいだを 0.02H）。下の UI より上に止め、離れたら線でつなぐ
        x = float(np.clip(ox, L + lw / 2 + size * 0.6, R - lw / 2 - size * 0.6))
        y = min(oy - H * 0.022, B - size * 0.55)
        o['anchor'] = 'ms'
    else:
        if side == 'right_mid':
            x, y, o['anchor'] = ox + W * 0.03, oy + size * 0.36, 'ls'
        elif side == 'right_high':
            x, y, o['anchor'] = ox + W * 0.06, oy - H * 0.05, 'ls'
        elif side == 'left':
            x, y, o['anchor'] = ox - W * 0.03, oy - H * 0.022, 'rs'
        else:
            x, y, o['anchor'] = ox + W * 0.03, oy - H * 0.022, 'ls'
        if o['anchor'] == 'ls':
            x = float(np.clip(x, L, R - lw))
        else:
            x = float(np.clip(x, L + lw, R))
    bx = C.text_box(label, x, y, size, o)
    pd = size * 0.38
    pb = (bx[0] - pd * 1.5, bx[1] - pd, bx[2] + pd * 1.5, bx[3] + pd)
    it = [C.plate(pb, feather=pd * 0.4, radius=pd * 1.2, alpha=a, sigma=size * 0.25, target=0.30, kmin=0.30, kmax=0.62)]
    # 点：画面の中にあるときだけ（端で半分に切れないように）
    r = H * (0.006 if ig else 0.007)
    if W * 0.02 < ox < W * 0.98 and H * 0.02 < oy < H * 0.98:
        it.append(('', ox, oy, r * 2, dict(shape='dot', alpha=a, shadow=0.6)))
        # 板の縁のいちばん近い点まで線（離れているときだけ）
        qx, qy = float(np.clip(ox, pb[0] + pd, pb[2] - pd)), float(np.clip(oy, pb[1], pb[3]))
        if np.hypot(qx - ox, qy - oy) > H * 0.03:
            it.append(('', ox, oy, 0, dict(shape='line', to=(qx, qy), width=max(2, H * 0.0016), alpha=a, line_alpha=0.95, shadow=0.7)))
    it.append((label, x, y, size, o))
    return it


def shot_floors(n, W, H, fmt):
    """6：フロアガイド。建物の横から見上げながら、各フロアの名前が建物の面に貼りついて下から順に出る。"""
    img = C.grade(C.load(56), con=0.05, sat=1.04)
    dep = C.depth(56, C.load(56))
    h, w = img.shape[:2]
    sc = w / 900.0
    for i in range(n):
        u = i / (n - 1)
        cx, cy, vw = floors_cam(u, fmt)
        f, A = C.view(img, W, H, cx, cy, vw, dep, 0.35)
        items = []
        for k, (fl, name, pts) in enumerate(FLOORS):
            t0 = FLOOR_T0[fmt][k]
            a = float(C.ease(np.clip((u - t0) / 0.10, 0, 1), 'smooth'))
            if a <= 0:
                continue
            px, py = pts[fmt]
            ox, oy = C.to_out(A, px * sc, py * sc)
            if fmt == 'ig':                                  # 点が画面の右から入ってくるまでは出さない
                a *= float(C.ease(np.clip((W * 0.94 - ox) / (W * 0.08), 0, 1), 'smooth'))
                if a <= 0.01:
                    continue
            items += floor_tag(W, H, fmt, (fl + '  ' if fl else '') + name, ox, oy, a)
        if items:
            f = C.put_text(f, C.text_layer(W, H, items))
        yield f


def shot_deck(n, W, H, fmt):
    """7：屋上・タワーから見た水平線（9660、0.5倍）。右上がり1.14°を補正（v8 と同じ）。"""
    src = os.path.join(FOOT, 'IMG_9660.MOV')
    if fmt == 'ig':
        return raw_video(src, 3.0, n, 0.5, W, H, crop=(0.12, 0.02, 0.76, 0.76), rot_cw=1.14)
    return raw_video(src, 3.0, n, 0.5, W, H, crop=(0.03, 0.27, 0.94, 0.94 * 2160 / 3840 * 1080 / 1920 * 3840 / 2160 * 2160 / 3840), rot_cw=1.14)


END_EMBLEM = (1065, 555, 810)          # 写真41（長辺2048）の看板：中心 x, y と、上の文字の弧から下の光までの高さ（px）


def shot_end(n, W, H, fmt):
    """8：END。夜に光る看板（41）をゆっくり押し込む。施設名・住所・営業時間。
    IG：看板を画面の上 0.08〜0.52H に収まる大きさで置き、下の暗い壁に文字（Reels の下の UI より上）。
    看板の下の窓の灯りは、壁の暗さへ溶かして見せない。"""
    if fmt != 'ig':
        path = lambda u: (0.50, 0.40, C.lerp(0.95, 0.86, C.ease(u, 'out')), 1)
        yield from photo_move(41, n, W, H, path, par=0.25, long_side=2048, grade=dict(con=0.04, sat=1.0))
        return
    img = C.grade(C.load(41, 2048), con=0.04, sat=1.0)
    ex, ey, eh = END_EMBLEM
    wall = np.median(img[-200:, :200].reshape(-1, 3), axis=0) * 0.6
    ys = np.arange(H, dtype=np.float32)[:, None]
    for i in range(n):
        u = i / max(1, n - 1)
        s = H * 0.44 / eh * C.lerp(0.97, 1.03, C.ease(u, 'out'))   # 出力1pxあたり、画像の 1/s px
        cy_out = H * 0.30
        A = np.array([[1 / s, 0, ex - W / 2 / s], [0, 1 / s, ey - cy_out / s]], np.float32)
        f = cv2.warpAffine(img, A, (W, H), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(v) for v in wall))
        top = cy_out - ey * s                                        # 写真の上端（出力の y）
        k = np.clip((ys - top) / (H * 0.03), 0, 1) * (1 - np.clip((ys - H * 0.56) / (H * 0.04), 0, 1))
        k = (k * k * (3 - 2 * k))[..., None]
        yield f * k + wall * (1 - k)


# ---------------------------------------------------------------------------
# 4本の構成（コマ数）と文字
# ---------------------------------------------------------------------------
PLANS = {
    # 2026-10-04 改訂：テンポをゆるめて 40秒／20秒。冒頭に LP の空撮、地図（日本→南風）、車窓を追加
    'ig30': dict(W=1080, H=1920, fmt='ig', shots=[('aerial', 105), ('map', 210), ('driveby', 90), ('relight', 165), ('1f', 150),
                                                   ('2f', 165), ('floors', 120), ('deck', 75), ('end', 120)]),
    'ig15': dict(W=1080, H=1920, fmt='ig', shots=[('aerial', 45), ('map', 135), ('driveby', 45), ('relight', 90), ('1f', 75),
                                                   ('2f', 60), ('floors', 60), ('end', 90)]),
    'sg30': dict(W=1920, H=1080, fmt='sg', shots=[('aerial', 105), ('map', 210), ('driveby', 90), ('relight', 165), ('1f', 150),
                                                   ('2f', 165), ('floors', 120), ('deck', 75), ('end', 120)]),
    'sg15': dict(W=1920, H=1080, fmt='sg', shots=[('aerial', 45), ('map', 135), ('driveby', 45), ('relight', 90), ('1f', 75),
                                                   ('2f', 60), ('floors', 60), ('end', 90)]),
}
SHOTS = dict(aerial=shot_aerial, map=shot_map, driveby=shot_driveby, place=shot_place, road=shot_road, relight=shot_relight, **{'1f': shot_1f}, **{'2f': shot_2f},
             floors=shot_floors, deck=shot_deck, end=shot_end)


def texts(plan):
    W, H = plan['W'], plan['H']
    ig = plan['fmt'] == 'ig'
    st = {}
    f = 0
    for name, n in plan['shots']:
        st[name] = (f, f + n)
        f += n
    T = []
    # 冒頭：IG は水平線より下の沖の海（島・砂州・白波を避ける）。サイネージは橋のない右下の川面
    s, e = st['aerial']
    T.append(Txt(s + 8, e - 2, lambda W, H: [
        ('太平洋と仁淀川の交差点', W * (0.5 if ig else 0.70), H * (0.384 if ig else 0.68), H * (0.026 if ig else 0.044),
         dict(anchor='mm', track=0.16, shadow=0.9 if ig else 0.8)),
        ('高知県土佐市 新居', W * (0.5 if ig else 0.70), H * (0.384 if ig else 0.68) + H * (0.029 if ig else 0.06), H * (0.015 if ig else 0.026),
         dict(anchor='mm', track=0.3, alpha=0.9, shadow=0.9 if ig else 0.8))]))
    # 車窓：建物の上の空。空の上の方を文字と一緒に少し暗くして読みやすく。タワーの避雷針が来る前に消す
    s, e = st['driveby']
    n = e - s
    ty, gy = (0.175, 0.235) if ig else (0.09, 0.16)
    T.append(Txt(s + 3 if n >= 80 else s + 6, s + 56 if n >= 80 else e - 2, lambda W, H: [
        C.plate((-W * 0.1, -H * 0.1, W * 1.1, H * gy), kind='dim', k=0.2, feather=H * (0.0175 if ig else 0.015)),
        ('仁淀川を渡ると、左手に。', W / 2, H * ty, H * (0.024 if ig else 0.042), dict(anchor='mm', track=0.16, shadow=0.8))], fade=10))
    # 建物（昼→夜）：屋根より上の暗い空。IG は1行、サイネージは左上の夜空。看板の光がせり上がる前に消す
    s, e = st['relight']
    n = e - s
    if ig:
        T.append(Txt(s + int(n * (0.45 if n >= 120 else 0.38)), s + int(n * 0.78), lambda W, H: [
            ('Seaside MAZE ｜ 南風', W / 2, H * 0.155, H * 0.024, dict(anchor='mm', track=0.12, shadow=0.8))], fade=10))
    else:
        T.append(Txt(s + int(n * 0.45), e - int(n * 0.12), lambda W, H: [
            ('Seaside MAZE', W * 0.05, H * 0.085, H * 0.052, dict(anchor='lm', track=0.12, shadow=0.8)),
            ('南風', W * 0.05, H * 0.147, H * 0.032, dict(anchor='lm', track=0.6, alpha=0.9, shadow=0.8))], fade=10))
    # 1F：地酒（仁淀川）のカットには文字を乗せない（箱の筆文字と重なるため、その手前で消す）
    s, e = st['1f']
    T.append(Txt(s + 4, e - tail_1f(e - s), lambda W, H: floor_label(W, H, '1F', 'ローソン ＋ 産直マーケット', '地元の野菜と、土佐市の特産品'), fade=8))
    # 2F：階段（1Fのロビー）では出さず、2Fのカウンターのカットから
    s, e = st['2f']
    T.append(Txt(s + cuts_2f(e - s, plan['fmt'])[0], e - 2, lambda W, H: floor_label(W, H, '2F', 'オーシャンビューカフェ', 'MAZE WIND RETREAT ｜ @mazewind2026' if ig else 'MAZE WIND RETREAT'), fade=8))
    if 'deck' in st:
        # 屋上：水平線と砂州のあいだの海の帯
        s, e = st['deck']
        T.append(Txt(s + 4, e, lambda W, H: floor_label(W, H, '3F', '屋上・展望スペース', '津波避難タワーは、どなたでも上まで',
                                                         base=0.594 if ig else 0.68), fade=8))
    # END：IG は看板の下の暗い壁（Reels の下の UI より上）。サイネージは左の暗い壁に左揃え
    s, e = st['end']
    if ig:
        ey, ex, an = (0.555, 0.034, 0.062), 0.5, 'mm'
    else:
        ey, ex, an = (0.78, 0.066, 0.118), 0.05, 'lm'
    T.append(Txt(s + 10, e + 1, lambda W, H: [
        ('Seaside MAZE', W * ex, H * ey[0], H * (0.036 if ig else 0.056), dict(anchor=an, track=0.12)),
        ('南風 ｜ 高知県土佐市新居・国道56号沿い', W * ex, H * (ey[0] + ey[1]), H * (0.019 if ig else 0.029),
         dict(anchor=an, track=0.12, alpha=0.95)),
        ('1F LAWSON 7:00–23:00　／　2F CAFE 11:00–18:00', W * ex, H * (ey[0] + ey[2]), H * (0.0155 if ig else 0.024),
         dict(anchor=an, track=0.08, alpha=0.85))], fade=18))
    return T, st


def render(key, stills=None):
    plan = PLANS[key]
    W, H = plan['W'], plan['H']
    T, st = texts(plan)
    total = sum(n for _, n in plan['shots'])
    path = os.path.join(OUT, 'MAZE_v2_%s_%ds.mp4' % (plan['fmt'], round(total / C.FPS)))
    enc = None if stills else C.Encoder(path, W, H, total)
    f = 0
    for name, n in plan['shots']:
        need = [s - f for s in (stills or []) if f <= s < f + n]
        if stills and not need:
            f += n
            continue
        frames = SHOTS[name](n, W, H, plan['fmt'])
        for i, fr in enumerate(frames):
            if i >= n:
                break
            if stills and i not in need:
                continue
            fr = C.grade(fr, con=0.0, sat=1.0) if False else fr
            for t in T:
                a = t.alpha(f + i)
                if a > 0:
                    fr = C.put_text(fr, t.layer(W, H), a)
            if stills:
                os.makedirs(os.path.join(OUT, 'stills'), exist_ok=True)
                cv2.imwrite(os.path.join(OUT, 'stills', '%s_%04d.jpg' % (key, f + i)),
                            (np.clip(fr, 0, 1)[..., ::-1] * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])
            else:
                enc.write(fr)
        f += n
    if enc:
        enc.close()
        if plan['fmt'] == 'ig':
            add_audio(path, total)
        print(path, total, 'frames', st)


def add_audio(path, total):
    """IG：波の音（IMG_5676 の現地音）をうすく敷く。頭0.3秒・終わり1.0秒でフェード。"""
    dur = total / C.FPS
    tmp = path.replace('.mp4', '.tmp.mp4')
    os.replace(path, tmp)
    subprocess.run([C.FF, '-v', 'error', '-y', '-i', tmp, '-ss', '2.0', '-t', '%.3f' % dur,
                    '-i', os.path.join(C.ROOT, 'footage', '02_写真', 'IMG_5676.MOV'),
                    '-filter_complex', '[1:a]volume=-6dB,afade=t=in:d=0.3,afade=t=out:st=%.3f:d=1.0,aresample=48000[a]' % (dur - 1.0),
                    '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-t', '%.3f' % dur,
                    '-movflags', '+faststart', path], check=True)
    os.remove(tmp)


if __name__ == '__main__':
    args = sys.argv[1:]
    st = None
    if '--stills' in args:
        i = args.index('--stills')
        st = [int(x) for x in args[i + 1].split(',')]
        args = args[:i] + args[i + 2:]
    for k in args:
        render(k, st)
