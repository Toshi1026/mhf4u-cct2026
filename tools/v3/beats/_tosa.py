"""土佐湾の細い線画（海岸線＋沖へ向かう等間隔の線）。b10 の最後にごくうすく出し始め、b11 が同じ絵から始められるように。
contours() は (H, W) の float マスク（0..1、線の濃さ）を返す。陸の塗りや光（グロー）はなし。"""
import os

import cv2
import numpy as np

import core
from core import W, H
from beats import _geo as G

CENTER = (33.30, 133.55)        # 画面の中心に置く緯度・経度（土佐湾のまんなか）
PX_KM = 4.6                     # 1km あたりの px
CX, CY = W / 2, 880.0
SHINKAI = G.SHINKAI             # 土佐市新居
STEPS = (10, 24, 42, 64, 90, 120, 155, 195)   # 海岸から沖への線の間隔（px）

_CACHE = {}


def to_screen(lat, lon):
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    kx = 111.32 * np.cos(np.radians(CENTER[0]))
    return CX + (lon - CENTER[1]) * kx * PX_KM, CY - (lat - CENTER[0]) * 111.2 * PX_KM


def contours():
    if 'm' in _CACHE:
        return _CACHE['m']
    path = os.path.join(G.CACHE, 'tosa_contours_%d_%d.png' % (int(PX_KM * 10), int(CY)))
    if os.path.exists(path):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
        _CACHE['m'] = m
        return m
    pad = 400
    land = np.zeros((H + 2 * pad, W + 2 * pad), np.uint8)
    polys = []
    for r in G.rings():
        if r[:, 0].max() < 128 or r[:, 0].min() > 139 or r[:, 1].max() < 29 or r[:, 1].min() > 37:
            continue
        x, y = to_screen(r[:, 1], r[:, 0])
        polys.append(np.round(np.stack([x + pad, y + pad], 1) * 16).astype(np.int32))
    cv2.fillPoly(land, polys, 255, lineType=cv2.LINE_AA, shift=4)
    # 海の側の距離（海岸からの px）。少しなめらかにして、等間隔の線に
    dist = cv2.distanceTransform((land < 128).astype(np.uint8), cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (0, 0), 6)
    m = np.zeros_like(dist)
    for k, d in enumerate(STEPS):
        a = 0.85 - 0.06 * k
        m = np.maximum(m, np.clip(1 - np.abs(dist - d) / 0.9, 0, 1) * a)
    # 海岸線
    coast = np.zeros_like(land)
    cv2.polylines(coast, polys, True, 255, 2, cv2.LINE_AA, 4)
    m = np.maximum(m, coast.astype(np.float32) / 255 * 0.9)
    m = m[pad:pad + H, pad:pad + W] * (land[pad:pad + H, pad:pad + W] < 128)
    m = np.maximum(m, (coast[pad:pad + H, pad:pad + W].astype(np.float32) / 255) * 0.9)
    os.makedirs(G.CACHE, exist_ok=True)
    cv2.imwrite(path, np.clip(m * 255 + 0.5, 0, 255).astype(np.uint8))
    _CACHE['m'] = m
    return m
