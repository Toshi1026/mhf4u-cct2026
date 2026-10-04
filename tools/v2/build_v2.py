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
    fr = list(C.read_video(path, W, H, n=n, start=start))
    while len(fr) < n:
        fr.append(fr[-1])
    return fr


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


def floor_label(W, H, floor, name, sub=None, pos='bl'):
    """フロア名（左下）。1F は細めの大きい数字、横に施設名。"""
    m = 0.06 * W if W < H else 0.05 * W
    base = H * (0.62 if W < H else 0.86)
    big = H * (0.045 if W < H else 0.075)
    small = H * (0.019 if W < H else 0.032)
    it = [(floor, m, base, big, dict(weight='Light' if os.path.exists(os.path.join(C.FONT_DIR, 'NotoSansJP-Light.ttf')) else 'Regular', track=0.02))]
    it.append((name, m + big * 1.25, base, small * 1.15, dict(track=0.12)))
    if sub:
        it.append((sub, m + big * 1.25, base + small * 1.7, small * 0.8, dict(track=0.14, alpha=0.85)))
    return it


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


def shot_1f(n, W, H, fmt, cuts=None):
    """4：1F。ローソンの入口 → 色でつなぐ産直の連打 → 地場の酒（仁淀川）。"""
    cuts = cuts or PRODUCE
    head = 20 if n >= 100 else 10
    tail = 20 if n >= 100 else 12
    per = max(5, (n - head - tail) // len(cuts))
    rest = n - head - tail - per * len(cuts)
    head += rest
    frames = []
    vw57 = (0.30, 0.22) if fmt == 'ig' else (0.62, 0.52)
    frames += list(photo_move(57, head, W, H, lambda u: (0.40, 0.66, C.lerp(*vw57, C.ease(u, 'out')), 1), par=0.3))
    for key, hue in cuts:
        img = C.grade(C.load(key, 2400), con=0.05, sat=1.06)
        cx, cy = blob_center(img, hue)
        ar = img.shape[1] / img.shape[0]
        vw = (0.42 if fmt == 'ig' else 0.62) * (1.0 if ar > 1 else 1.5)
        cx = np.clip(cx, vw / 2, 1 - vw / 2)
        vh = vw * img.shape[1] / img.shape[0] * H / W
        cy = np.clip(cy, vh / 2, 1 - vh / 2)
        for i in range(per):
            u = i / max(1, per - 1)
            f, _ = C.view(img, W, H, cx, cy, vw * C.lerp(1.0, 0.94, u))
            frames.append(f)
    frames += list(photo_move(101, tail, W, H, lambda u: (0.42, 0.55, C.lerp(0.62, 0.55, u) if fmt == 'ig' else C.lerp(0.95, 0.88, u), 1), par=0.3))
    return frames


def shot_2f(n, W, H, fmt):
    """5：階段（OCEANVIEW）を上がる → 2F のカウンターと海 → 窓の外の海（動画）。"""
    a = int(n * 0.30)
    b = int(n * 0.25)
    c = n - a - b
    fr = []
    # 階段：下から見上げて、上がっていく（奥行きで視差）
    path = (lambda u: (0.55, C.lerp(0.62, 0.40, C.ease(u, 'inout')), C.lerp(0.62, 0.48, u), 1)) if fmt == 'ig' else \
        (lambda u: (0.55, C.lerp(0.60, 0.45, C.ease(u, 'inout')), C.lerp(1.0, 0.85, u), 1))
    fr += list(photo_move(85, a, W, H, path, par=0.45, long_side=3024))
    path = (lambda u: (0.42, 0.50, C.lerp(0.55, 0.46, C.ease(u, 'out')), 1)) if fmt == 'ig' else \
        (lambda u: (0.50, 0.55, C.lerp(1.0, 0.88, C.ease(u, 'out')), 1))
    fr += list(photo_move(87, b, W, H, path, par=0.4, long_side=3024))
    if fmt == 'ig':
        fr += baked(os.path.join(CAP, 'proposals', 'ig_qc_fix', 'ig30', '05_IMG_9649.mp4'), c, W, H, start=max(0, 69 - c))
    else:
        fr += baked(os.path.join(CAP, 'signage30', '03_IMG_9674.mp4'), c, W, H, start=max(0, 90 - c))
    return fr


# フロアガイド（56）：各フロアの位置（900×600 に縮めた座標）
FLOORS = [('1F', 'ローソン ＋ 産直マーケット', (390, 440)),
          ('2F', 'オーシャンビューカフェ', (640, 355)),
          ('3F', '展望スペース', (600, 300)),
          ('', '津波避難タワー', (700, 255))]


def shot_floors(n, W, H, fmt):
    """6：フロアガイド。建物の横から見上げながら、各フロアの名前が建物の面に貼りついて下から順に出る。"""
    img = C.grade(C.load(56), con=0.05, sat=1.04)
    dep = C.depth(56, C.load(56))
    h, w = img.shape[:2]
    sc = w / 900.0
    small = H * (0.028 if W < H else 0.034)
    for i in range(n):
        u = i / (n - 1)
        e = C.ease(u, 'inout')
        if fmt == 'ig':
            cx, cy, vw = C.lerp(0.48, 0.72, e), C.lerp(0.70, 0.46, e), 0.36
        else:
            cx, cy, vw = C.lerp(0.58, 0.64, e), C.lerp(0.60, 0.50, e), C.lerp(0.66, 0.58, u)
        f, A = C.view(img, W, H, cx, cy, vw, dep, 0.35)
        items = []
        for k, (fl, name, (px, py)) in enumerate(FLOORS):
            t0 = 0.12 + k * 0.17
            a = float(C.ease(np.clip((u - t0) / 0.12, 0, 1), 'smooth'))
            if a <= 0:
                continue
            ox, oy = C.to_out(A, px * sc, py * sc)
            # 引き出し線の先（建物の面の点）に小さな点、文字はその右上
            x0, y0 = ox, oy
            label = (fl + '  ' if fl else '') + name
            lw = C.font(int(small)).getlength(label) * 1.1
            right = x0 + W * 0.03 + lw < W * 0.95            # 右に収まらなければ左に出す
            x1 = x0 + W * 0.03 if right else x0 - W * 0.03
            x1 = float(np.clip(x1, W * 0.05 + (0 if right else lw), W * 0.95 - (lw if right else 0)))
            y1 = float(np.clip(oy - H * 0.022, H * 0.08, H * 0.92))
            items.append(('●', x0, y0, small * 0.5, dict(anchor='mm', alpha=a, track=0)))
            items.append((label, x1, y1, small, dict(anchor='ls' if right else 'rs', alpha=a, track=0.10)))
        if items:
            f = C.put_text(f, C.text_layer(W, H, items))
        yield f


def shot_deck(n, W, H, fmt):
    """7：屋上・タワーから見た水平線（9660、0.5倍）。右上がり1.14°を補正（v8 と同じ）。"""
    src = os.path.join(FOOT, 'IMG_9660.MOV')
    if fmt == 'ig':
        return raw_video(src, 3.0, n, 0.5, W, H, crop=(0.12, 0.02, 0.76, 0.76), rot_cw=1.14)
    return raw_video(src, 3.0, n, 0.5, W, H, crop=(0.03, 0.27, 0.94, 0.94 * 2160 / 3840 * 1080 / 1920 * 3840 / 2160 * 2160 / 3840), rot_cw=1.14)


def shot_end(n, W, H, fmt):
    """8：END。夜に光る看板（41）を奥行きでゆっくり。施設名・住所・営業時間。"""
    path = (lambda u: (0.50, 0.40, C.lerp(0.36, 0.33, C.ease(u, 'out')), 1)) if fmt == 'ig' else \
        (lambda u: (0.50, 0.40, C.lerp(0.95, 0.86, C.ease(u, 'out')), 1))
    return photo_move(41, n, W, H, path, par=0.25, long_side=2048, grade=dict(con=0.04, sat=1.0))


# ---------------------------------------------------------------------------
# 4本の構成（コマ数）と文字
# ---------------------------------------------------------------------------
PLANS = {
    'ig30': dict(W=1080, H=1920, fmt='ig', shots=[('place', 75), ('road', 75), ('relight', 150), ('1f', 120), ('2f', 150),
                                                   ('floors', 105), ('deck', 60), ('end', 165)]),
    'ig15': dict(W=1080, H=1920, fmt='ig', shots=[('place', 45), ('relight', 90), ('1f', 75), ('2f', 75), ('floors', 75),
                                                   ('end', 90)]),
    'sg30': dict(W=1920, H=1080, fmt='sg', shots=[('place', 75), ('road', 75), ('relight', 150), ('1f', 120), ('2f', 150),
                                                   ('floors', 105), ('deck', 60), ('end', 165)]),
    'sg15': dict(W=1920, H=1080, fmt='sg', shots=[('place', 45), ('relight', 90), ('1f', 75), ('2f', 75), ('floors', 75),
                                                   ('end', 90)]),
}
SHOTS = dict(place=shot_place, road=shot_road, relight=shot_relight, **{'1f': shot_1f}, **{'2f': shot_2f},
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
    s, e = st['place']
    T.append(Txt(s + 8, e - 2, lambda W, H: [
        ('太平洋と仁淀川の交差点', W / 2, H * (0.30 if ig else 0.47), H * (0.026 if ig else 0.044), dict(anchor='mm', track=0.16)),
        ('高知県土佐市 新居', W / 2, H * (0.30 if ig else 0.47) + H * (0.034 if ig else 0.06), H * (0.015 if ig else 0.026),
         dict(anchor='mm', track=0.3, alpha=0.9))]))
    s, e = st['relight']
    T.append(Txt(s + int((e - s) * 0.45), e - int((e - s) * 0.12), lambda W, H: [
        ('Seaside MAZE', W / 2, H * (0.20 if ig else 0.16), H * (0.034 if ig else 0.058), dict(anchor='mm', track=0.12)),
        ('南風', W / 2, H * (0.20 if ig else 0.16) + H * (0.038 if ig else 0.07), H * (0.020 if ig else 0.034),
         dict(anchor='mm', track=0.6, alpha=0.9))], fade=10))
    s, e = st['1f']
    T.append(Txt(s + 4, e - 2, lambda W, H: floor_label(W, H, '1F', 'ローソン ＋ 産直マーケット', '地元の野菜と、土佐市の特産品'), fade=8))
    s, e = st['2f']
    T.append(Txt(s + 6, e - 2, lambda W, H: floor_label(W, H, '2F', 'オーシャンビューカフェ', 'MAZE WIND RETREAT'), fade=8))
    if 'deck' in st:
        s, e = st['deck']
        T.append(Txt(s + 4, e, lambda W, H: floor_label(W, H, '3F', '屋上・展望スペース', '津波避難タワーは、どなたでも上まで'), fade=8))
    s, e = st['end']
    T.append(Txt(s + 10, e + 1, lambda W, H: [
        ('Seaside MAZE', W / 2, H * (0.80 if ig else 0.78), H * (0.036 if ig else 0.056), dict(anchor='mm', track=0.12)),
        ('南風 ｜ 高知県土佐市新居・国道56号沿い', W / 2, H * (0.80 if ig else 0.78) + H * (0.040 if ig else 0.066),
         H * (0.019 if ig else 0.029), dict(anchor='mm', track=0.12, alpha=0.95)),
        ('1F LAWSON 7:00–23:00　／　2F CAFE 11:00–18:00', W / 2, H * (0.80 if ig else 0.78) + H * (0.072 if ig else 0.118),
         H * (0.0155 if ig else 0.024), dict(anchor='mm', track=0.08, alpha=0.85))], fade=18))
    return T, st


def render(key, stills=None):
    plan = PLANS[key]
    W, H = plan['W'], plan['H']
    T, st = texts(plan)
    total = sum(n for _, n in plan['shots'])
    path = os.path.join(OUT, 'MAZE_v2_%s.mp4' % key)
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
