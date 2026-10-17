"""地図：日本 → 高知県 → 土佐市 → 桂浜からのドライブの道筋 → 南風（真上）。途切れのないズーム。

ベクトルの日本地図（Natural Earth、都道府県）で始め、寄るにつれて国土地理院の航空写真（シームレス空中写真）へ
切り替える。道筋は OSRM（OpenStreetMap）で求めた 桂浜 → 仁淀川河口大橋 → 南風（edl/v2/route_katsurahama_maze.json）。
"""
import json
import math
import os
import urllib.request

import cv2
import numpy as np
import shapefile

import common as C

MAZE = (133.4774570, 33.4589539)        # 新居地区観光交流施設【南風-まぜ-】（OpenStreetMap）
KATSURA = (133.5744564, 33.4971075)     # 桂浜
BRIDGE = (133.4823193, 33.4618544)      # 仁淀川河口大橋
TILES = os.path.join(C.ROOT, 'exports', 'v2', '_tiles')
NE = os.path.join(C.ROOT, 'tools', 'models', 'ne', 'ne_10m_admin_1_states_provinces')
ROUTE = os.path.join(C.ROOT, 'edl', 'v2', 'route_katsurahama_maze.json')

SEA = np.array([0.80, 0.88, 0.93], np.float32)
LAND = np.array([0.965, 0.955, 0.93], np.float32)
BORDER = np.array([0.80, 0.80, 0.78], np.float32)
KOCHI = np.array([0.16, 0.56, 0.66], np.float32)


def merc(lon, lat):
    """経度・緯度 → ズーム0の世界座標（0..256）。"""
    x = (lon + 180.0) / 360.0 * 256.0
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * 256.0
    return x, y


def merc_arr(pts):
    p = np.asarray(pts, np.float64)
    x = (p[:, 0] + 180.0) / 360.0 * 256.0
    s = np.sin(np.radians(p[:, 1]))
    y = (0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * 256.0
    return np.stack([x, y], 1)


_POLY = None


def polys():
    """日本と周辺の行政区画（世界座標）。[(name, is_kochi, [rings])]"""
    global _POLY
    if _POLY is not None:
        return _POLY
    r = shapefile.Reader(NE)
    f = [x[0] for x in r.fields[1:]]
    ia, inm = f.index('admin'), f.index('name')
    out = []
    for sr in r.iterShapeRecords():
        rec = sr.record
        if rec[ia] not in ('Japan', 'South Korea', 'North Korea', 'China', 'Russia'):
            continue
        bb = sr.shape.bbox
        if bb[2] < 118 or bb[0] > 150 or bb[3] < 22 or bb[1] > 48:
            continue
        pts = sr.shape.points
        parts = list(sr.shape.parts) + [len(pts)]
        rings = [merc_arr(pts[parts[i]:parts[i + 1]]) for i in range(len(parts) - 1)]
        out.append((rec[inm], rec[ia] == 'Japan' and rec[inm].startswith('K') and 'chi' in rec[inm], rec[ia] == 'Japan', rings))
    _POLY = out
    return out


def _load(z, x, y):
    """タイル1枚（無ければ None）。控えになければ国土地理院から取ってくる。"""
    os.makedirs(TILES, exist_ok=True)
    p = os.path.join(TILES, '%d_%d_%d.jpg' % (z, x, y))
    if not os.path.exists(p):
        url = 'https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/%d/%d/%d.jpg' % (z, x, y)
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'maze-video/1.0'}), timeout=30).read()
        except Exception:
            data = b''
        open(p, 'wb').write(data)
    a = cv2.imread(p) if os.path.getsize(p) >= 100 else None
    return None if a is None else a[..., ::-1].astype(np.float32) / 255.0


def tile_raw(z, x, y):
    """埋めないタイルと、写っている所のマスク（黒い所・無いタイルは 0）。"""
    a = _load(z, x, y)
    if a is None:
        return None, None
    v = cv2.erode((a.max(axis=2) >= 0.04).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
    return a, v


def tile(z, x, y):
    a = _load(z, x, y)
    black = None if a is None else (a.max(axis=2) < 0.04).astype(np.float32)
    if a is not None and black.mean() < 0.002:
        return a
    # 無い・一部が黒い（撮影範囲の外）→ 上のズームのタイルの一部を拡大して、黒い所だけ埋める
    par = tile(z - 1, x // 2, y // 2) if z > 6 else None
    if par is None:
        return a
    q = cv2.resize(par[(y % 2) * 128:(y % 2) * 128 + 128, (x % 2) * 128:(x % 2) * 128 + 128], (256, 256),
                   interpolation=cv2.INTER_CUBIC)
    q = np.clip(q, 0, 1)
    if a is None:
        return q
    m = cv2.GaussianBlur(cv2.dilate(black, np.ones((5, 5), np.uint8)), (0, 0), 2)[..., None]
    return a * (1 - m) + q * m


def _mosaic(W, H, z, cx, cy, zt, raw=False):
    """ズーム段 zt のタイルを並べて W×H に写す。raw=True：埋めないタイルと、写っている所のマスクも返す。"""
    s = 2.0 ** (z - zt)                                 # タイルの1pxが出力の何pxか
    ox, oy = cx * 2 ** zt - (W / 2) / s, cy * 2 ** zt - (H / 2) / s   # タイル座標系での左上
    x0, y0 = int(math.floor(ox / 256)), int(math.floor(oy / 256))
    x1, y1 = int(math.floor((ox + W / s) / 256)), int(math.floor((oy + H / s) / 256))
    mw, mh = (x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256
    mos = np.zeros((mh, mw, 3), np.float32)
    mos[:] = SEA
    val = np.zeros((mh, mw), np.float32) if raw else None
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            sl = (slice((ty - y0) * 256, (ty - y0 + 1) * 256), slice((tx - x0) * 256, (tx - x0 + 1) * 256))
            if raw:
                t, v = tile_raw(zt, tx, ty)
                if t is not None:
                    mos[sl], val[sl] = t, v
            else:
                t = tile(zt, tx, ty)
                if t is not None:
                    mos[sl] = t
    M = np.float32([[s, 0, -(ox - x0 * 256) * s], [0, s, -(oy - y0 * 256) * s]])
    img = np.clip(cv2.warpAffine(mos, M, (W, H), flags=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE), 0, 1)
    if not raw:
        return img
    return img, cv2.warpAffine(val, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


BASE_Z = 13                             # ここまではどの段もタイルがそろっている（海も写っている）
SEA_Z0, SEA_ZW = 16.4, 0.7              # z がここまでは、海は BASE_Z の写真を使う（上の段の沖は取得元ごとに色が違い、四角く見える）
SEA_CALM = (12.2, 13.0)                 # z9〜12 の段は沖に色の違う多角形がある。この範囲を越えるまで、沖はまわりの海の色をぼかして使う


def photo_layer(W, H, z, cx, cy):
    """航空写真（世界座標の中心 cx,cy、ズーム z）を W×H に。
    ズームの段は1コマで切り替えず、となりの段と z の端数でなめらかに重ねる。z14 から上は沖のタイルが無いので、
    z13 を土台にして、その上に写っている所だけを縁をぼかして重ねる（タイルの四角いつぎはぎを出さない）。"""
    zf = int(min(17, max(2, math.floor(z))))
    w = float(C.ease(np.clip(z - zf, 0, 1), 'smooth'))
    if zf + 1 <= BASE_Z:
        a = _mosaic(W, H, z, cx, cy, zf)
        return a if w <= 0 else a * (1 - w) + _mosaic(W, H, z, cx, cy, zf + 1) * w
    lv = base = _mosaic(W, H, z, cx, cy, BASE_Z)        # 段ごとの絵（BASE_Z はそろっている）
    c = 1 - float(C.ease(np.clip((z - SEA_Z0) / SEA_ZW, 0, 1), 'smooth'))
    S = (_sea_mask(base, 2.0 ** (z - BASE_Z)) * c)[..., None] if c > 0 else 0.0
    prev = None
    for L in range(BASE_Z + 1, zf + 2):
        img, v = _mosaic(W, H, z, cx, cy, L, raw=True)
        prev, lv = lv, _fill(img, v, lv) * (1 - S) + base * S
    if prev is None:                                    # zf == BASE_Z：BASE_Z と次の段を重ねる
        return lv
    return prev * (1 - w) + lv * w


def _fill(img, v, base):
    """写っていない所（沖のタイルが無い所）を、まわりの同じ段の写真の色を大きくぼかして埋める。
    下の段の絵で埋めると、色の違う四角いつぎはぎが出るため。まわりに写真が無い所だけ base を使う。"""
    H, W = v.shape
    k = 8                                               # ぼかしは 1/8 の大きさで（速く、なめらかに）
    sw, sh = max(1, W // k), max(1, H // k)
    vs = cv2.resize(v, (sw, sh), interpolation=cv2.INTER_AREA)
    ims = cv2.resize(img * v[..., None], (sw, sh), interpolation=cv2.INTER_AREA)
    F = cv2.resize(base, (sw, sh), interpolation=cv2.INTER_AREA)
    for f in (0.4, 0.15, 0.05):                         # 大きい範囲から小さい範囲へ、写っている所の色に寄せる
        sg = f * max(sw, sh)
        den = cv2.GaussianBlur(vs, (0, 0), sg)
        est = cv2.GaussianBlur(ims, (0, 0), sg) / np.maximum(den, 1e-4)[..., None]
        g = np.clip(den / 0.15, 0, 1)[..., None]
        F = F * (1 - g) + est * g
    F = cv2.resize(F, (W, H), interpolation=cv2.INTER_CUBIC)
    m = np.clip((cv2.GaussianBlur(v, (0, 0), 0.01 * max(W, H)) - 0.5) / 0.5, 0, 1)   # 写っている所の内側だけ
    m = (m * m * (3 - 2 * m) * np.clip(v, 0, 1))[..., None]                         # タイルの中の黒い行も使わない
    return np.clip(F * (1 - m) + img * m, 0, 1)


def _sea_mask(base, s):
    """BASE_Z の写真の海（青みの強い所）。s：BASE_Z の1pxが出力の何pxか。"""
    b = cv2.GaussianBlur(base, (0, 0), max(0.5, s))
    sea = (((b[..., 2] - b[..., 1]) > 0.09) & ((b[..., 2] - b[..., 0]) > 0.22)).astype(np.uint8)
    k = max(3, int(2 * s) | 1)
    sea = cv2.morphologyEx(sea, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    sea = cv2.morphologyEx(sea, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return np.clip(cv2.GaussianBlur(sea.astype(np.float32), (0, 0), 1.5 * s), 0, 1)


def _calm_sea(img, z, cx, cy):
    """低いズームのとき、沖の海を「まわりの海の色を大きくぼかしたもの」にする（段の写真の継ぎ目を見せない）。
    陸は Natural Earth の海岸線を 1.5km 広げて守る。z が SEA_CALM を越えるにつれて元の写真へ戻す。"""
    H, W = img.shape[:2]
    k = 1 - float(C.ease(np.clip((z - SEA_CALM[0]) / (SEA_CALM[1] - SEA_CALM[0]), 0, 1), 'smooth'))
    if k <= 0:
        return img
    q = 4                                               # 陸の形は 1/4 の大きさで作る
    w4, h4 = W // q, H // q
    land = np.zeros((h4, w4), np.uint8)
    sc = 2 ** z / q
    off = np.array([cx * 2 ** z - W / 2, cy * 2 ** z - H / 2]) / q
    for _, _, _, rings in polys():
        for r in rings:
            p = (r * sc - off).astype(np.int32)
            if p[:, 0].max() < 0 or p[:, 1].max() < 0 or p[:, 0].min() > w4 or p[:, 1].min() > h4:
                continue
            cv2.fillPoly(land, [p], 255)
    mpp = 40075016.7 * math.cos(math.radians(MAZE[1])) / (256 * 2 ** z) * q
    d = max(2, int(1500 / mpp))
    land = cv2.dilate(land, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1)))
    sea = np.clip((1 - cv2.GaussianBlur(land.astype(np.float32) / 255, (0, 0), d * 0.5) - 0.02) / 0.98, 0, 1)
    sea = cv2.resize(sea, (W, H), interpolation=cv2.INTER_LINEAR)
    sw, sh = max(1, W // 8), max(1, H // 8)
    sg = 0.06 * max(sw, sh)
    vs = cv2.resize(sea, (sw, sh), interpolation=cv2.INTER_AREA)
    ims = cv2.resize(img * sea[..., None], (sw, sh), interpolation=cv2.INTER_AREA)
    est = cv2.GaussianBlur(ims, (0, 0), sg) / np.maximum(cv2.GaussianBlur(vs, (0, 0), sg), 1e-6)[..., None]
    m = (sea * k)[..., None]
    return np.clip(img * (1 - m) + cv2.resize(est, (W, H), interpolation=cv2.INTER_CUBIC) * m, 0, 1)


def vector_layer(W, H, z, cx, cy, kochi_a):
    """ベクトルの地図（海・陸・県境・高知県の強調）。"""
    ss = 2                                              # 2倍で描いて縮める（なめらかに）
    img = np.empty((H * ss, W * ss, 3), np.float32)
    img[:] = SEA
    sc = 2 ** z * ss
    off = np.array([cx * 2 ** z - W / 2, cy * 2 ** z - H / 2]) * ss
    land = np.zeros((H * ss, W * ss), np.uint8)
    koch = np.zeros((H * ss, W * ss), np.uint8)
    edges = np.zeros((H * ss, W * ss), np.uint8)
    for name, is_kochi, is_jp, rings in polys():
        for r in rings:
            p = (r * sc - off).astype(np.int32)
            if p[:, 0].max() < 0 or p[:, 1].max() < 0 or p[:, 0].min() > W * ss or p[:, 1].min() > H * ss:
                continue
            cv2.fillPoly(land, [p], 255, cv2.LINE_AA)
            if is_kochi:
                cv2.fillPoly(koch, [p], 255, cv2.LINE_AA)
            if is_jp:
                cv2.polylines(edges, [p], True, 255, max(1, ss), cv2.LINE_AA)
    L = land.astype(np.float32)[..., None] / 255
    K = koch.astype(np.float32)[..., None] / 255 * kochi_a
    E = edges.astype(np.float32)[..., None] / 255
    img = img * (1 - L) + LAND * L
    img = img * (1 - E * 0.8) + BORDER * E * 0.8
    glow = cv2.GaussianBlur(K, (0, 0), 6 * ss)[..., None] if K.ndim == 2 else cv2.GaussianBlur(K[..., 0], (0, 0), 6 * ss)[..., None]
    img = img * (1 - K * 0.85) + KOCHI * K * 0.85
    img = img + glow * 0.10 * kochi_a
    return cv2.resize(np.clip(img, 0, 1), (W, H), interpolation=cv2.INTER_AREA)


def _interp(keys, u):
    """keys：[(u, span_deg, lon, lat)]。ズームは log(span) をなめらかに。中心は、次の目的地の画面上の位置が
    だんだん画面の中央へ寄っていくように決める（ズームしても目的地が画面から外れない）。"""
    us = [k[0] for k in keys]
    i = max(0, min(len(keys) - 2, np.searchsorted(us, u, side='right') - 1))
    a, b = keys[i], keys[i + 1]
    t = (u - a[0]) / (b[0] - a[0])
    t = t * t * (3 - 2 * t)
    span = math.exp(math.log(a[1]) * (1 - t) + math.log(b[1]) * t)
    ax, ay = merc(a[2], a[3])
    bx, by = merc(b[2], b[3])
    # 目的地 b の、画面上のずれ（span 単位）を始点のずれから 0 へ
    ox = (bx - ax) / a[1] * (1 - t)
    oy = (by - ay) / a[1] * (1 - t)
    cx, cy = bx - ox * span, by - oy * span
    return span, cx, cy


def keys_for(fmt):
    tall = fmt == 'ig'
    return [
        (0.00, 22.0 if tall else 30.0, 136.8, 35.6),       # 日本
        (0.26, 2.6 if tall else 3.6, 133.35, 33.45),       # 高知県
        (0.44, 0.40 if tall else 0.62, 133.50, 33.47),     # 土佐市〜高知市の海岸
        (0.78, 0.16 if tall else 0.22, 133.525, 33.475),   # 桂浜 → 南風の道筋
        (1.00, 0.0055 if tall else 0.0095, MAZE[0], MAZE[1]),  # 南風の真上
    ]


def frames(n, W, H, fmt):
    keys = keys_for(fmt)
    route = merc_arr(json.load(open(ROUTE))['routes'][0]['geometry']['coordinates'])
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(route, axis=0).T))]
    seg /= seg[-1]
    for i in range(n):
        u = i / (n - 1)
        span, cx, cyw = _interp(keys, u)
        z = math.log2(W / span * 360.0 / 256.0)
        kochi_a = float(C.ease(np.clip((u - 0.06) / 0.14, 0, 1), 'smooth')) * float(1 - C.ease(np.clip((u - 0.40) / 0.12, 0, 1), 'smooth'))
        pa = float(C.ease(np.clip((z - 8.7) / 2.3, 0, 1), 'smooth'))   # 航空写真へ切り替え
        if pa < 1:
            f = vector_layer(W, H, z, cx / 256 * 256 / 256 * 1.0 * 1, cyw / 256 * 1.0, kochi_a) if False else vector_layer(W, H, z, cx, cyw, kochi_a)
        if pa > 0:
            p = _calm_sea(photo_layer(W, H, z, cx, cyw), z, cx, cyw)
            p = C.grade(p, con=0.06, sat=1.08)
            f = p if pa >= 1 else f * (1 - pa) + p * pa
        # 道筋：0.47→0.77 で桂浜から伸びて南風へ。0.84→0.90 で消える
        prog = float(C.ease(np.clip((u - 0.47) / 0.30, 0, 1), 'inout'))
        if prog > 0:
            k = int(np.searchsorted(seg, prog))
            pts = route[: max(2, k + 1)].copy()
            pts = (pts * 2 ** z - np.array([cx * 2 ** z - W / 2, cyw * 2 ** z - H / 2]))
            fade = 1 - float(C.ease(np.clip((u - 0.84) / 0.06, 0, 1), 'smooth'))   # 南風の名前が出る前に消し切る
            lay = np.zeros((H * 2, W * 2), np.uint8)
            cv2.polylines(lay, [(pts * 2).astype(np.int32)], False, 255, int(H * 0.006 * 2), cv2.LINE_AA)
            L = cv2.resize(lay, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            G = cv2.GaussianBlur(L, (0, 0), H * 0.006)
            col = np.array([1.0, 0.86, 0.45], np.float32)
            f = f * (1 - G[..., None] * 0.5 * fade) + col * G[..., None] * 0.35 * fade
            f = f * (1 - L[..., None] * fade) + col * L[..., None] * fade
            hx, hy = pts[-1]
            cv2.circle(f, (int(hx), int(hy)), int(H * 0.007), (1.0, 1.0, 1.0), -1, cv2.LINE_AA) if fade > 0.1 else None  # ピンが出るまで残す
        yield f, z, cx, cyw                              # 1コマずつ（210コマを全部メモリに載せない）


def to_screen(lonlat, z, cx, cy, W, H):
    """cx, cy：世界座標の中心（frames の戻り値）。"""
    x, y = merc(*lonlat)
    return x * 2 ** z - (cx * 2 ** z - W / 2), y * 2 ** z - (cy * 2 ** z - H / 2)
