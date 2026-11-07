"""実写パート（b11 の終わり〜b12〜b13）の共通部品。

- 3本の iPhone HLG 素材の読み出し条件（回転・切り出し・速さ）を1か所にまとめる
- 色は1つの grade（すっきりした寒色寄り。CG の青に合わせる。ティール＆オレンジにはしない）
- b11 が線画に起こす「b12 の1フレーム目」をキャッシュする（b11 と b12 で同じ処理を通すので、画素単位で一致する）

傾きは素材のフレームから水平線を当てはめて実測した値（ffmpeg の rotate は時計回りが正）：
  IMG_9662 2.10–2.77s：右上がり 1.31〜1.38° → +1.34°
  IMG_9671 6.30–7.40s：右下がり 0.43〜0.67° → −0.55°
  IMG_9636 1.60s     ：右上がり 0.94°      → +0.95°
"""
import os

import cv2
import numpy as np

import core
from core import W, H

CACHE = os.path.join(core.OUT, 'cache')
os.makedirs(CACHE, exist_ok=True)

# 寒色寄り・すっきり。彩度はほぼそのまま、わずかにコントラスト
GRADE = dict(con=0.07, sat=1.0, lift=0.0, gain=1.0, cool=0.035)

# 読み出し条件
WAVE = dict(name='IMG_9662.MOV', t0=2.10, rot=1.34, cx=0.5, cy=0.5, zoom=1.06)     # 5倍・波が崩れる（0.5倍スロー）
RIVER = dict(name='IMG_9671.MOV', t0=6.30, rot=-0.55, cx=0.5, cy=0.41, zoom=1.25)  # 河口（0.5倍スロー・下の消波ブロックを外す）
# 2F の窓（1倍）。窓の外の柵の上端（回転後の素材の y≈0.5745）より上だけを使う：下端を y=0.572 に置いて 4 倍。
# 天井照明の映り込み（x 0.79–0.88, y 0.19–0.25）と左端の映り込みは切り出しの外。水平線は画面の y≈1420。
WINDOW = dict(name='IMG_9636.MOV', t0=1.60, rot=0.95, cx=0.70, cy=0.572 - 0.5 / 4.0, zoom=4.0)


def grade(f):
    f = core.grade(f, **GRADE)
    # 海の青緑を少しだけ青へ（緑を数%抑える）。オレンジ方向には振らない
    g = f[..., 1:2]
    b = f[..., 2:3]
    teal = np.clip((g - f[..., 0:1]) * 2.0, 0, 1) * np.clip(1 - np.abs(g - b) * 4, 0, 1)
    f[..., 1:2] = g - 0.025 * teal
    return np.clip(f, 0, 1)


def clip(spec, n, step=1):
    """spec の素材を n 枚。step=1 は素材の1フレーム＝出力1フレーム（60fps→30fps で 0.5倍スロー）、step=2 は等速。"""
    t1 = spec['t0'] + (n * step + 2) / 60.0
    gen = core.footage(spec['name'], spec['t0'], t1, rot=spec['rot'], cx=spec['cx'], cy=spec['cy'], zoom=spec['zoom'])
    k = 0
    last = None
    for i, f in enumerate(gen):
        if i % step:
            continue
        last = grade(f)
        yield last
        k += 1
        if k >= n:
            break
    while k < n:            # 念のため（素材が足りないときは最後の1枚で埋める）
        yield last
        k += 1


def wave_first_frame():
    """b12 の1フレーム目（IMG_9662 を WAVE の条件で読み、grade したもの）。キャッシュする。"""
    key = 'real_wave0_%s' % '_'.join('%s' % WAVE[k] for k in ('t0', 'rot', 'cx', 'cy', 'zoom'))
    key += '_' + '_'.join('%s%s' % kv for kv in sorted(GRADE.items())) + '_v1'
    path = os.path.join(CACHE, key + '.npy')
    if os.path.exists(path):
        return np.load(path)
    f = next(clip(WAVE, 1))
    np.save(path, f.astype(np.float32))
    return f


# ---------------------------------------------------------------- 実写の1フレームを線画に起こす（b11 の終わり／b12 の頭）
def horizon_y(f, lo=500, hi=700):
    """水平線の高さ（px、小数）と傾き（度）。空（明）→海（暗）の縦の段差を列ごとに探し、外れ値を除いて直線を当てる。"""
    L = cv2.GaussianBlur(f.mean(2), (0, 0), 1.0)
    d = L[lo + 1:hi] - L[lo:hi - 1]
    ys = np.argmin(d, 0).astype(np.float64)
    # 段差の位置を小数で（前後の差分で放物線補間）
    xs = np.arange(f.shape[1])
    i = ys.astype(int).clip(1, d.shape[0] - 2)
    a, b, c = d[i - 1, xs], d[i, xs], d[i + 1, xs]
    den = a - 2 * b + c
    ys = ys + np.where(np.abs(den) > 1e-6, 0.5 * (a - c) / np.where(np.abs(den) > 1e-6, den, 1), 0) + lo + 1.0
    m = np.ones(len(xs), bool)
    for _ in range(6):
        p = np.polyfit(xs[m], ys[m], 1)
        r = ys - np.polyval(p, xs)
        m = np.abs(r) < max(0.8, 2.5 * np.std(r[m]))
    return float(np.polyval(p, f.shape[1] / 2)), float(np.degrees(np.arctan(p[0])))


def _runs(y, min_len=40):
    """列ごとの境界 y（NaN あり）を、連続した部分ごとになめらかにした (N,2) の折れ線のリストに。"""
    out = []
    ok = np.isfinite(y)
    x = 0
    n = len(y)
    while x < n:
        if not ok[x]:
            x += 1
            continue
        s = x
        while x < n and ok[x] and (x == s or abs(y[x] - y[x - 1]) < 6):
            x += 1
        if x - s >= min_len:
            seg = y[s:x].astype(np.float64)
            k = np.ones(5) / 5
            pad = np.pad(seg, 2, mode='edge')
            seg = np.convolve(pad, k, 'valid')                 # ぎざぎざを均す（1px 未満のずれ）
            out.append(np.stack([np.arange(s, x, dtype=np.float64), seg], 1))
    return out


def _cross(V, y0, y1, thr, down=True, hold=4):
    """列ごとに、y0..y1 の範囲で V が thr を（down=True なら上から下へ下回る／False なら上回る）最初の位置（小数）。
    hold px 先まで同じ側にあることを条件にする（さざ波で拾わないように）。"""
    seg = V[y0:y1]
    s = (seg < thr) if down else (seg > thr)
    ok = s.copy()
    for k in range(1, hold + 1):
        ok[:-k] &= s[k:]
    ok[-hold:] = False
    first = np.argmax(ok, 0)
    has = ok.any(0) & (first > 0)
    xs = np.arange(V.shape[1])
    f1 = first.clip(1, None)
    va, vb = seg[f1 - 1, xs], seg[f1, xs]
    fr = np.where(np.abs(vb - va) > 1e-6, (thr - va) / np.where(np.abs(vb - va) > 1e-6, vb - va, 1), 0.5).clip(0, 1)
    y = y0 + f1 - 1 + fr + 0.5
    return np.where(has, y, np.nan)


def lines_of(f):
    """実写のフレーム f（b12 の頭の IMG_9662）から、線画の要素を取り出す。
    戻り値 dict：horizon（y）、main（白い線：波頭・泡の上下・波打ち際）、swell（うすい線：沖のうねり）。座標は画面 px。"""
    L = f.mean(2)
    mx, mn = f.max(2), f.min(2)
    sat = cv2.GaussianBlur((mx - mn) / (mx + 1e-3), (0, 0), 2.0)
    Lb = cv2.GaussianBlur(L, (0, 0), 1.4)
    Lf = cv2.GaussianBlur(L, (0, 0), 3.0)
    yh, _ = horizon_y(f)
    main = []
    crest = _cross(Lb, 1150, 1265, 0.46, down=True, hold=6)          # 立ち上がるうねりの暗い面の上の縁
    main += _runs(crest, 60)
    ftop = _cross(Lf, 1240, 1330, 0.80, down=False, hold=6)          # 泡の上の縁
    main += _runs(ftop, 60)
    # 泡の下の縁：泡の上の縁より下で、明るさが落ちるところ
    fb = np.full(f.shape[1], np.nan)
    ok = np.isfinite(ftop)
    if ok.any():
        y0 = int(np.nanmin(ftop)) + 8
        fb = _cross(Lf, y0, 1420, 0.74, down=True, hold=8)
        fb[~ok] = np.nan
    main += _runs(fb, 60)
    shore = _cross(sat, 1370, 1480, 0.16, down=True, hold=8)          # 波打ち際（海の青→砂の灰）
    main += _runs(shore, 60)
    # 沖のうねり：横に強くぼかしたときの縦の段差。長いものだけ、遠近に合わせた間隔で数本
    Ls = cv2.GaussianBlur(L, (0, 0), sigmaX=25, sigmaY=1.8)
    u8 = (np.clip((Ls - 0.25) / 0.6, 0, 1) * 255).astype(np.uint8)
    e = cv2.Canny(u8, 8, 20)
    e[:int(yh) + 30] = 0
    lim = int(np.nanmin(crest)) - 25 if np.isfinite(crest).any() else 1170
    e[lim:] = 0
    cs, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cand = []
    for c in cs:
        p = c[:, 0, :].astype(np.float64)
        # 輪郭は行きと帰りで2重なので、列ごとに平均した片道にする
        xs = np.unique(p[:, 0].astype(int))
        if len(xs) < 280:
            continue
        ys = np.array([p[p[:, 0] == x, 1].mean() for x in xs])
        cand.append((len(xs), xs, ys))
    cand.sort(key=lambda c: -c[0])
    swell, used = [], []
    for ln, xs, ys in cand:
        ym = float(np.median(ys))
        gap = 0.16 * (ym - yh)                     # 遠い（水平線に近い）ほど詰めてよい
        if any(abs(ym - u) < gap for u in used):
            continue
        used.append(ym)
        k = np.ones(9) / 9
        ys = np.convolve(np.pad(ys, 4, mode='edge'), k, 'valid')
        swell.append(np.stack([xs.astype(np.float64), ys], 1))
        if len(swell) >= 7:
            break
    return dict(horizon=yh, main=main, swell=swell)


def row_means(f, yh=None, shore=None):
    """行ごとの平均色（H,3）。b11 の背景のグラデーション（実写の色そのものを縦に並べたもの）。
    空・海・砂の3つの帯の中だけでなめらかにして、水平線と波打ち際の色の段差はくっきり残す。"""
    r = f.mean(1).astype(np.float32)
    if yh is None:
        return r
    out = r.copy()
    a, b = int(round(yh)), int(round(shore)) if shore is not None else f.shape[0]
    for (y0, y1, sg) in ((0, a, 6.0), (a, b, 4.0), (b, f.shape[0], 14.0)):
        if y1 - y0 < 3:
            continue
        seg = r[y0:y1]
        pad = int(3 * sg) + 1
        p = np.pad(seg, ((pad, pad), (0, 0)), mode='edge')
        p = cv2.GaussianBlur(p[:, None, :], (1, 0), sigmaX=0.1, sigmaY=sg)[:, 0, :]
        out[y0:y1] = p[pad:pad + (y1 - y0)]
    return out
