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


def tile(z, x, y):
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
    a = None if a is None else a[..., ::-1].astype(np.float32) / 255.0
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


def photo_layer(W, H, z, cx, cy):
    """航空写真（世界座標の中心 cx,cy、ズーム z）を W×H に。"""
    zt = int(min(18, max(2, math.floor(z + 0.5))))
    s = 2.0 ** (z - zt)                                 # タイルの1pxが出力の何pxか
    tw = 256 * s
    ox, oy = cx * 2 ** zt - (W / 2) / s, cy * 2 ** zt - (H / 2) / s   # タイル座標系での左上
    x0, y0 = int(math.floor(ox / 256)), int(math.floor(oy / 256))
    x1, y1 = int(math.floor((ox + W / s) / 256)), int(math.floor((oy + H / s) / 256))
    mw, mh = (x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256
    mos = np.zeros((mh, mw, 3), np.float32)
    mos[:] = SEA
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            t = tile(zt, tx, ty)
            if t is not None:
                mos[(ty - y0) * 256:(ty - y0 + 1) * 256, (tx - x0) * 256:(tx - x0 + 1) * 256] = t
    M = np.float32([[s, 0, -(ox - x0 * 256) * s], [0, s, -(oy - y0 * 256) * s]])
    return np.clip(cv2.warpAffine(mos, M, (W, H), flags=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE), 0, 1)


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
    out = []
    for i in range(n):
        u = i / (n - 1)
        span, cx, cyw = _interp(keys, u)
        z = math.log2(W / span * 360.0 / 256.0)
        kochi_a = float(C.ease(np.clip((u - 0.06) / 0.14, 0, 1), 'smooth')) * float(1 - C.ease(np.clip((u - 0.40) / 0.12, 0, 1), 'smooth'))
        pa = float(C.ease(np.clip((z - 8.7) / 2.3, 0, 1), 'smooth'))   # 航空写真へ切り替え
        if pa < 1:
            f = vector_layer(W, H, z, cx / 256 * 256 / 256 * 1.0 * 1, cyw / 256 * 1.0, kochi_a) if False else vector_layer(W, H, z, cx, cyw, kochi_a)
        if pa > 0:
            p = photo_layer(W, H, z, cx, cyw)
            p = C.grade(p, con=0.06, sat=1.08)
            f = p if pa >= 1 else f * (1 - pa) + p * pa
        # 道筋：0.48→0.80 で桂浜から伸び、最後は南風へ
        prog = float(C.ease(np.clip((u - 0.47) / 0.30, 0, 1), 'inout'))
        if prog > 0:
            k = int(np.searchsorted(seg, prog))
            pts = route[: max(2, k + 1)].copy()
            pts = (pts * 2 ** z - np.array([cx * 2 ** z - W / 2, cyw * 2 ** z - H / 2]))
            fade = 1 - float(C.ease(np.clip((u - 0.90) / 0.08, 0, 1), 'smooth'))
            lay = np.zeros((H * 2, W * 2), np.uint8)
            cv2.polylines(lay, [(pts * 2).astype(np.int32)], False, 255, int(H * 0.006 * 2), cv2.LINE_AA)
            L = cv2.resize(lay, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            G = cv2.GaussianBlur(L, (0, 0), H * 0.006)
            col = np.array([1.0, 0.86, 0.45], np.float32)
            f = f * (1 - G[..., None] * 0.5 * fade) + col * G[..., None] * 0.35 * fade
            f = f * (1 - L[..., None] * fade) + col * L[..., None] * fade
            hx, hy = pts[-1]
            cv2.circle(f, (int(hx), int(hy)), int(H * 0.007), (1.0, 1.0, 1.0), -1, cv2.LINE_AA) if fade > 0.5 else None
        out.append((f, z, cx, cyw))
    return out


def to_screen(lonlat, z, cx, cy, W, H):
    """cx, cy：世界座標の中心（frames の戻り値）。"""
    x, y = merc(*lonlat)
    return x * 2 ** z - (cx * 2 ** z - W / 2), y * 2 ** z - (cy * 2 ** z - H / 2)
