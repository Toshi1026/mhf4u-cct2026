"""01・02 の舞台：朝の空（断面図と同じ XS_SKY_TOP → XS_SKY_WL の色）と、低い位置の穏やかな水平線、手前の浜。
01 で悪役が飛んでいき、点になって光る位置 DOT は 02 でもそのまま（押し込みの中心も DOT）。"""
import cv2
import numpy as np

import core
from core import W, H

HZ = 1150                 # 水平線
DOT = (868.0, 338.0)      # 悪役が点になる所（キラーン）
FEET = (430.0, 1430.0)    # 悪役の足もと（浜の上）
VSCALE = 5.2              # 立っている悪役の大きさ（全長 約150px）
PUSH01 = (1.0, 1.035)     # 01 の押し込み（DOT を中心に）
PUSH02 = (1.035, 1.10)    # 02 はその続き


def shore_y(x):
    x = np.asarray(x, np.float32)
    return 1372 + 0.05 * (x - 540) + 7 * np.sin(x / 210.0 + 0.6)


_BG = {}


def background():
    """動かない部分（空・海・浜）。雲は別（cloud_layer）。"""
    if 'bg' in _BG:
        return _BG['bg']
    f = np.empty((H, W, 3), np.float32)
    sky = core.vgrad(core.XS_SKY_TOP, core.XS_SKY_WL, h=HZ, gamma=0.85)
    # 水平線の近くだけ、ごく弱い朝のもや（白）
    yy = np.arange(HZ, dtype=np.float32)
    haze = (np.exp(-((HZ - yy) / 150.0) ** 2) * 0.16)[:, None, None]
    sky = sky + (core.FOAM - sky) * haze
    f[:HZ] = sky
    sea = core.vgrad(core.c('4D97D2'), core.c('2A73B2'), h=H - HZ, gamma=0.6)
    f[HZ:] = sea
    # 海の上の細い光の筋（横長・うすい）
    rng = np.random.default_rng(7)
    lay = np.zeros((H, W), np.float32)
    for _ in range(36):
        y = HZ + 6 + (rng.random() ** 1.6) * 120
        x = rng.uniform(-100, W)
        ln = rng.uniform(40, 220) * (0.4 + (y - HZ) / 120)
        cv2.line(lay, (int(x), int(y)), (int(x + ln), int(y)), float(rng.uniform(0.25, 0.6)), 1, cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 0.7)
    f += (core.FOAM - f) * (lay * 0.22)[..., None]
    # 水平線の白い細線
    f[HZ - 1:HZ + 1] = f[HZ - 1:HZ + 1] * 0.55 + core.FOAM * 0.45
    # 浜（青灰色）。波打ち際から下へ
    xs = np.arange(W, dtype=np.float32)
    sy = shore_y(xs)
    Y = np.arange(H, dtype=np.float32)[:, None]
    m = np.clip(Y - sy[None, :] + 0.5, 0, 1)[..., None]
    sand = core.vgrad(core.c('B4C5D3'), core.c('6A8397'), h=H, gamma=0.7)
    # 波打ち際のすぐ下：ぬれた砂（少し暗く、空を映して青い）
    dy = (Y - sy[None, :])
    wet = (np.exp(-np.clip(dy, 0, None) / 38.0) * 0.55)[..., None]
    sand = sand * (1 - wet) + core.c('7C9BB4') * wet
    # 遠近：下へ行くほどごく弱く暗く
    f = f * (1 - m) + sand * m
    _BG['bg'] = f.astype(np.float32)
    return _BG['bg']


def cloud_layer():
    """ごくうすい横長の雲（白）。(alpha マスク, y0)。毎フレーム少しずつ横へ流す。"""
    if 'cl' in _BG:
        return _BG['cl']
    y0, y1 = 560, 1080
    m = np.zeros((y1 - y0, W + 600), np.float32)
    for (cx, cy, rx, ry, a) in [(220, 820, 300, 22, 0.55), (420, 850, 200, 14, 0.35), (900, 760, 340, 26, 0.45),
                                (1180, 905, 260, 16, 0.35), (640, 980, 380, 18, 0.30), (60, 1010, 220, 12, 0.25),
                                (1450, 700, 300, 20, 0.40)]:
        cv2.ellipse(m, (cx, cy - y0), (rx, ry), 0, 0, 360, a, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 16)
    _BG['cl'] = (m * 0.32, y0)
    return _BG['cl']


def frame(gi):
    """gi：01 の頭からの通しのフレーム番号（02 でも続けて数える）。空・雲・海・波打ち際。"""
    f = background().copy()
    m, y0 = cloud_layer()
    sh = 300 - 0.35 * gi                              # 雲は左へゆっくり流れる
    M = np.float32([[1, 0, -sh], [0, 1, 0]])
    cm = cv2.warpAffine(m, M, (W, m.shape[0]), flags=cv2.INTER_LINEAR)
    sub = f[y0:y0 + m.shape[0]]
    sub += (core.FOAM - sub) * cm[..., None]
    # 波打ち際の白い泡の線（ゆっくり寄せて返す）
    t = gi / core.FPS
    xs = np.arange(0, W + 1, 6, dtype=np.float32)
    sy = shore_y(xs) - 3 - 2.5 * (1 + np.sin(t * 1.6)) - 1.2 * np.sin(xs / 37.0 + t * 2.2)
    L = np.zeros((90, W), np.float32)
    top = int(shore_y(xs).min()) - 40
    pts = np.stack([xs, sy - top], 1).astype(np.int32)
    cv2.polylines(L, [pts], False, 0.75, 2, cv2.LINE_AA)
    L = cv2.GaussianBlur(L, (0, 0), 0.8)
    sub = f[top:top + 90]
    sub += (core.FOAM - sub) * L[..., None]
    return f


def push(f, k, cx=DOT[0], cy=DOT[1]):
    if abs(k - 1) < 1e-5:
        return f
    M = np.float32([[k, 0, cx * (1 - k)], [0, k, cy * (1 - k)]])
    return cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
