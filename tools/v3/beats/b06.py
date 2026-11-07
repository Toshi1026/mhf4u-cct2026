"""b06（17.0–22.3秒）：そこで、水じゃなく、エネルギーに乗る。沖の水は、ほぼその場で回るだけ。
0–11：b05 の終わりを巻き戻す（泡が波に戻り、サーファーが波の上に戻る。テープを戻すような横のぶれ）。
12–21：水の中へ潜る（下へカメラを振る）→ 水の粒の断面図。
粒とブイはその場で小さな円を描くだけ。明るい青の帯（エネルギー）が左から右へ通り抜ける。
「ノ」ル で、サーファーが水面から光の帯の上へ「ぴょん」と乗り移り、帯に乗って右へ。カメラはゆっくり帯を追う。
最後の 8 フレーム：ズームアウトのぼかし（b07 の地図へ）。"""
import cv2
import numpy as np

import core
import timeline
from core import W, H
from beats import _wave

REW = 12            # 巻き戻しのフレーム数
DIVE0, DIVE1 = 12, 22
WLP = 700           # 粒の断面図での水面の高さ
FI_BACK = 24.0      # 巻き戻しの行き先（b05 のこのフレーム：沖の波にサーファーが乗っている）
SURF_X = 410.0      # 乗り移る前にサーファーが浮かんでいる位置（world x）
OUTB = 8


def _tape(f, speed, i):
    """巻き戻しの質感：速さに比例した横のぶれ＋数本の帯だけ横にずらす（明るさは変えない）。"""
    f = core.whip(f, 5 * speed)
    rng = np.random.default_rng(100 + i)
    for _ in range(3):
        y = int(rng.uniform(300, H - 200))
        hgt = int(rng.uniform(6, 22))
        f[y:y + hgt] = np.roll(f[y:y + hgt], int(rng.uniform(-1, 1) * 4 * speed), axis=1)
    return f


def render(beat, ctx):
    n = beat['n']
    b05 = next(b for b in timeline.load() if b['key'] == '05')
    xs = _wave.XSScene(b05['n'])
    fi_end = b05['n'] - 1
    f_e = timeline.mora_frame(beat, 8)          # 「エ」ネルギー
    f_no = timeline.mora_frame(beat, 14)        # 「ノ」ル
    f_o = timeline.mora_frame(beat, 16)         # 「オ」キノ
    HOP = 10
    land = f_no + HOP
    # 帯がちょうど着地点に来るように速さを決める
    x_land = SURF_X + 70
    vg = (x_land - (-260.0)) / (land - DIVE0)
    sea = _wave.ParticleSea(wl=WLP, vg=vg, band_x0=-260.0, f_band0=DIVE0)
    txtA = core.text_layer([dict(text='乗るのは エネルギー', x=W / 2, y=560, size=96, weight='Bold', hl='エネルギー')])
    txtB = core.text_layer([dict(text='沖の水は、ほぼその場で回るだけ', x=W / 2, y=1200, size=44, weight='Medium', shadow=0.9)])
    # 文字 B の後ろの粒を少し暗くするための、やわらかい帯
    yy = np.arange(H, dtype=np.float32)[:, None]
    dimB = (np.exp(-((yy - 1186) / 75.0) ** 2) * 0.22).astype(np.float32)[..., None]

    def cam_at(i):
        # 着地してから、ゆっくり帯を追う（帯の速さの 85% まで上げる）
        if i <= land:
            return 0.0
        t = i - land
        ramp = 18.0
        v = 0.85 * vg
        return v * (t * t / (2 * ramp) if t < ramp else ramp / 2 + (t - ramp))

    prev = fi_end
    for i in range(n):
        if i < REW:
            q = core.ease(i / (REW - 1), 'inout')
            fi = fi_end - (fi_end - FI_BACK) * q
            speed = abs(prev - fi)
            prev = fi
            f = xs.frame(fi)
            xs.hold(f, fi)
            f = _tape(f, speed, i)
            yield np.clip(f, 0, 1).astype(np.float32)
            continue
        # ---- 粒の断面図
        cam = cam_at(i)
        g = sea.render(i, cam=cam)
        # ブイ（その場で回るだけ）
        bx0 = 760.0
        bx, by = sea.surface(np.array([bx0], np.float32), i)
        _wave.draw_buoy(g, float(bx[0]) - cam, float(by[0]) + 4, scale=1.0)
        # サーファー
        if i < f_no:
            sx, sy = sea.surface(np.array([SURF_X], np.float32), i)
            sx, sy = float(sx[0]), float(sy[0])
            ang = 0.0
        else:
            ex, ey = sea.surface(np.array([sea.band_x(i) + 70], np.float32), i)
            ex, ey = float(ex[0]), float(ey[0])
            if i < land:
                # ぴょん：浮かんでいた所から帯の上の水面へ放物線
                p = (i - f_no) / HOP
                ox, oy = sea.surface(np.array([SURF_X], np.float32), f_no)
                sx = float(ox[0]) + (ex - float(ox[0])) * p
                sy = float(oy[0]) + (ey - float(oy[0])) * p - 110 * 4 * p * (1 - p)
                ang = -12 * np.sin(np.pi * p)
            else:
                sx, sy = ex, ey
                ang = 6.0
        g = _wave.draw_surfer(g, sx - cam, sy - 3, scale=2.2, ang=ang)
        # 潜るつなぎ：b05 の断面図から下へカメラを振る
        if i < DIVE1:
            p = (i - DIVE0) / (DIVE1 - DIVE0 - 1)
            e = core.ease(p, 'inout')
            o = (core.WL - WLP) * e
            a = float(_wave.smooth(p, 0.25, 0.75))
            M1 = np.array([[1, 0, 0], [0, 1, -o]], np.float32)
            top = cv2.warpAffine(xs.frame(FI_BACK), M1, (W, H), borderMode=cv2.BORDER_REPLICATE)
            M2 = np.array([[1, 0, 0], [0, 1, (core.WL - WLP) - o]], np.float32)
            bot = cv2.warpAffine(g, M2, (W, H), borderMode=cv2.BORDER_REPLICATE)
            g = top * (1 - a) + bot * a
            g = core.whip(g, 70 * np.sin(np.pi * p), axis=0)
        # 文字
        if i >= f_o:
            g = g * (1 - dimB * min(1.0, (i - f_o + 1) / 4))
        if i >= f_e:
            g = core.put_text(g, txtA, min(1.0, (i - f_e + 1) / 4))
        if i >= f_o:
            g = core.put_text(g, txtB, min(1.0, (i - f_o + 1) / 4))
        j = i - (n - OUTB)
        if j >= 0:
            g = core.zoom_blur(g, 0.2 * ((j + 1) / OUTB) ** 1.5)
        yield np.clip(g, 0, 1).astype(np.float32)
