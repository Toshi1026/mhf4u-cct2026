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


def background(extra=0):
    """動かない部分（空・海・浜）。雲は別（cloud_layer）。extra：下へ足す行（01 のカメラが下を見るとき。浜が続く）。"""
    if ('bg', extra) in _BG:
        return _BG[('bg', extra)]
    HH = H + extra
    f = np.empty((HH, W, 3), np.float32)
    sky = core.vgrad(core.XS_SKY_TOP, core.XS_SKY_WL, h=HZ, gamma=0.85)
    # 水平線の近くだけ、ごく弱い朝のもや（白）
    yy = np.arange(HZ, dtype=np.float32)
    haze = (np.exp(-((HZ - yy) / 150.0) ** 2) * 0.16)[:, None, None]
    sky = sky + (core.FOAM - sky) * haze
    f[:HZ] = sky
    sea = core.vgrad(core.c('4D97D2'), core.c('2A73B2'), h=HH - HZ, gamma=0.6)
    if extra:                                     # 足した行の分も、元の画面の範囲の色は変えない
        sea[:H - HZ] = core.vgrad(core.c('4D97D2'), core.c('2A73B2'), h=H - HZ, gamma=0.6)
    f[HZ:] = sea
    # 海の上の細い光の筋（横長・うすい）
    # 2026-11：光の筋は frame() で毎フレーム少しずつ流してきらめかせる（streak_layer）。ここでは描かない
    # 水平線の白い細線
    f[HZ - 1:HZ + 1] = f[HZ - 1:HZ + 1] * 0.55 + core.FOAM * 0.45
    # 浜（青灰色）。波打ち際から下へ
    xs = np.arange(W, dtype=np.float32)
    sy = shore_y(xs)
    Y = np.arange(HH, dtype=np.float32)[:, None]
    m = np.clip(Y - sy[None, :] + 0.5, 0, 1)[..., None]
    tt = np.clip(np.arange(HH, dtype=np.float32) / (H - 1), 0, None)[:, None, None] ** 0.7
    sand = np.clip(core.c('B4C5D3') * (1 - tt) + core.c('6A8397') * tt, 0, 1) * np.ones((1, W, 1), np.float32)
    # 波打ち際のすぐ下：ぬれた砂（少し暗く、空を映して青い）
    dy = (Y - sy[None, :])
    wet = (np.exp(-np.clip(dy, 0, None) / 38.0) * 0.55)[..., None]
    sand = sand * (1 - wet) + core.c('7C9BB4') * wet
    # 遠近：下へ行くほどごく弱く暗く
    f = f * (1 - m) + sand * m
    _BG[('bg', extra)] = f.astype(np.float32)
    return _BG[('bg', extra)]


SM = 360                  # 光の筋の板の左右の余白（流す分）


def streak_layer():
    """海の上の細い光の筋（横長・うすい）の板（行 HZ..HZ+140、幅 W+2·SM）。もとの 36 本＋余白の分。"""
    if 'st' in _BG:
        return _BG['st']
    rng = np.random.default_rng(7)
    lay = np.zeros((150, W + 2 * SM), np.float32)
    for _ in range(36):
        y = HZ + 6 + (rng.random() ** 1.6) * 120
        x = rng.uniform(-100, W)
        ln = rng.uniform(40, 220) * (0.4 + (y - HZ) / 120)
        cv2.line(lay, (int(x) + SM, int(y) - HZ), (int(x + ln) + SM, int(y) - HZ), float(rng.uniform(0.25, 0.6)), 1,
                 cv2.LINE_AA)
    r2 = np.random.default_rng(17)
    for _ in range(26):
        y = HZ + 6 + (r2.random() ** 1.6) * 120
        x = r2.choice([r2.uniform(-SM, -100), r2.uniform(W, W + SM)])
        ln = r2.uniform(40, 220) * (0.4 + (y - HZ) / 120)
        cv2.line(lay, (int(x) + SM, int(y) - HZ), (int(x + ln) + SM, int(y) - HZ), float(r2.uniform(0.25, 0.6)), 1,
                 cv2.LINE_AA)
    lay = cv2.GaussianBlur(lay, (0, 0), 0.7)
    _BG['st'] = lay
    return lay


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


def frame(gi, extra=0):
    """gi：01 の頭からの通しのフレーム番号（02 でも続けて数える）。空・雲・海・波打ち際。extra は background と同じ。"""
    f = background(extra).copy()
    m, y0 = cloud_layer()
    sh = 300 - 1.0 * gi                               # 雲はゆっくり流れる（2026-11：止まって見えないよう 0.35→1.0px/フレーム）
    M = np.float32([[1, 0, -sh], [0, 1, 0]])
    cm = cv2.warpAffine(m, M, (W, m.shape[0]), flags=cv2.INTER_LINEAR)
    sub = f[y0:y0 + m.shape[0]]
    sub += (core.FOAM - sub) * cm[..., None]
    # 海の光の筋：手前ほど速く左へ流れ、ゆっくりきらめく（止まった画にしない）
    st = streak_layer()
    yy = np.arange(st.shape[0], dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]
    spd = 0.25 + 0.75 * np.clip(yy / 120.0, 0, 1)
    mx = (xx + SM + spd * gi * 0.9).astype(np.float32)
    my = np.broadcast_to(yy, mx.shape).astype(np.float32)
    band = cv2.remap(st, mx, my, cv2.INTER_LINEAR)
    tw = 0.78 + 0.22 * np.sin(xx / 63.0 + yy / 7.0 - gi * 0.23) * np.sin(xx / 151.0 - gi * 0.07 + 1.3)
    sub = f[HZ:HZ + st.shape[0]]
    sub += (core.FOAM - sub) * (band * tw * 0.22 / 0.89)[..., None]
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


PUSH_MAX = 1.13           # 01 の頭から 02 のズームスルーの入りまでの、DOT を中心にした1本の押し込み


def push_k(gi, gi_end):
    """01・02 を通した押し込みの倍率（sine_in：だんだん速く、02 のズームスルーへ速さを渡す）。gi_end：02 の f46 の通し番号。"""
    t = min(max(gi / float(gi_end), 0.0), 1.0)
    return 1.0 + (PUSH_MAX - 1.0) * (0.4 * t + 0.6 * (1 - np.cos(t * np.pi / 2)))   # 頭から少し動いている


def push(f, k, cx=DOT[0], cy=DOT[1]):
    if abs(k - 1) < 1e-5:
        return f
    M = np.float32([[k, 0, cx * (1 - k)], [0, k, cy * (1 - k)]])
    return cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
