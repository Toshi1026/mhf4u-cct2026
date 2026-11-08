"""look-dev 06・07 の舞台：横から見た断面を「ジオラマの切り口」として描く（_render の道具だけで）。
- 空：散乱の空（左上の画面の外に太陽、そのにじみ）＋地平線の近くに遠い積雲の列（空気遠近でかすむ）
- 水（水面から下）：深さで光が吸われる色、左上から差しこむ光の筋（ゆっくり流れる・弱い）、水面のすぐ下の明るい帯（全反射）、
  海底の上のコースティクス（ゆらめく網目）、ただよう細かい粒
- 陸（桂浜）と海底：切り口の地層（砂 → 小石まじりの層 → 岩）。上の縁に細い光、下ほど濃紺へ
- 富士山：同じ縮尺の「ガラスの山」（ふくらませた形の法線で、縁のフレネル・左上の光のつや・山頂の雪のうすい白）。
  輪郭の線の描かれ方（stroke）と塗りの入り方（fill）は 06 のまま。
タイミング・形（_xsec の線・弧・数字）は変えない。"""
import math

from _common import np, cv2, core, tick, W, H  # noqa: F401
from cutB.beats import _render as R, _xsec as X
from core import WL

CAM = R.HorizonCam(WL, f=1500.0, cam_h=40.0)
SUN = R.sun_vec(-38.0, 33.0)                # 左上の画面の外
_C = {}


def _land_mask(ss=3):
    P = X._land_poly()
    m = np.zeros((H * ss, W * ss), np.uint8)
    pts = np.round(P * ss * 16).astype(np.int32)
    cv2.fillPoly(m, [pts], 255, cv2.LINE_AA, shift=4)
    return cv2.resize(m.astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_AREA)


def _surface_y():
    """各列の地面（陸の上・海底）の y。"""
    P = X._land_poly()[1:-1]
    xs = np.arange(W, dtype=np.float32) + 0.5
    return np.interp(xs, P[:, 0], P[:, 1]).astype(np.float32)


def static():
    if _C:
        return _C
    with tick('xsec static'):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        # ---- 空（線形）
        sky = R.sky(CAM, WL, W, SUN, zen=R.hexl('15528F'), hor=R.hexl('5C9FD6'), haze=R.hexl('B4D6EE'), mie=0.20)
        deck = R.cloud_deck(CAM, W, SUN, seed=23, n=14, alt=(500.0, 1100.0), dist=(30000.0, 110000.0),
                            size=(1400.0, 3200.0), bands=2)
        _C['sky'], _C['deck'] = sky, deck
        # ---- 水（線形）：深さで吸われる色
        d = np.clip(yy[WL:] - WL, 0, None)
        top, mid, deep = R.hexl('3C8CC8'), R.hexl('1A5E9A'), R.hexl('082C52')
        k1 = 1 - np.exp(-d / 260.0)
        k2 = 1 - np.exp(-d / 900.0)
        water = top * (1 - k1)[..., None] + mid * (k1 * (1 - k2))[..., None] + deep * k2[..., None]
        # 水面のすぐ下：空を下から映す明るい帯（全反射）
        water = water + R.hexl('9CCBEA') * (0.22 * np.exp(-d / 14.0))[..., None]
        _C['water'] = water.astype(np.float32)
        # ---- 陸・海底の切り口（線形）
        sy = _surface_y()
        ds = yy - sy[None]                                   # 地面の下へ（px）
        land = _land_mask()
        wob = R.fbm(xx / 260.0, yy / 90.0, 4, seed=31) * 10 + R.fbm(xx / 60.0, yy / 25.0, 3, seed=32) * 3
        q = ds + wob
        c_sand, c_peb, c_rock, c_deep = R.hexl('9DB2C2'), R.hexl('6E8599'), R.hexl('3E5368'), R.hexl('0D2338')
        b1 = R.smoothstep(16, 30, q)
        b2 = R.smoothstep(70, 120, q)
        b3 = R.smoothstep(150, 520, q)
        col = c_sand * (1 - b1)[..., None] + c_peb * (b1 * (1 - b2))[..., None] + c_rock * (b2 * (1 - b3))[..., None] + \
            c_deep * b3[..., None]
        # 層の細い線（地層）と小石
        stra = np.sin(q / 9.0 + R.fbm(xx / 400.0, yy / 400.0, 2, seed=33) * 3) * 0.5 + 0.5
        col = col * (1 - 0.10 * R.smoothstep(0.85, 1.0, stra) * b1)[..., None]
        peb = R.fbm(xx / 7.0, yy / 7.0, 2, seed=34)
        col = col * (1 + (R.smoothstep(0.45, 0.75, peb) * 0.25 - R.smoothstep(-0.45, -0.75, peb) * 0.2) *
                     (b1 * (1 - b3)))[..., None]
        gr = R.sample(R.fbm_tile(512, 128, 2, 0.6, 41), xx * 1.3, yy * 1.3)
        col = col * (1 + gr * 0.05)[..., None]
        # 上の縁：細い光（切り口の角）と、そのすぐ下のやわらかい影
        edge = np.exp(-((ds - 1.0) / 1.3) ** 2)
        under = R.hexl('8FB4CF')
        wk = R.smoothstep(WL - 2, WL + 30, sy)[None] * np.ones((H, 1), np.float32)   # 水の中の海底ほど 1（列ごとになめらかに）
        uw = (wk > 0.5) & (yy > sy[None])                     # 水の中の海底
        col = col + R.hexl('E2EFF8') * (edge * (0.55 - 0.30 * wk))[..., None]
        col = col * (1 - 0.25 * np.exp(-np.clip(ds - 2, 0, None) / 10.0) * (ds > 2))[..., None]
        # 水の中の海底の上の面：水の色が少しかかる
        kk = (wk * np.exp(-np.clip(ds, 0, None) / 60.0))[..., None]      # 海底の上の面の近くだけ、水の色が少しかかる
        col = col * (1 - 0.18 * kk) + under * (0.04 * kk)
        _C['land'], _C['land_a'], _C['sy'], _C['uw'] = col.astype(np.float32), land, sy, uw
        # 目盛り線（_xsec.background と同じ位置・同じうすさ）
        L = core.SS(s=3)
        for m in (1000, 2000):
            y = X.y_of(m)
            L.draw.line(L.p([(225, y), (1010, y)]), fill=L.col(core.FOAM, 0.11), width=L.w(1.2))
        L.draw.line(L.p([(205, WL), (W, WL)]), fill=L.col(core.FOAM, 0.55), width=L.w(1.6))
        _C['lines'] = L.done()
        # ただよう粒（固定の位置。時間でゆっくり流す）
        rng = np.random.default_rng(5)
        n = 420
        _C['motes'] = np.stack([rng.uniform(0, W, n), rng.uniform(WL + 8, H, n), rng.uniform(0.5, 1.6, n),
                                rng.uniform(0.15, 0.6, n), rng.uniform(0, 6.28, n)], 1)
        # 光の筋の 1 次元のノイズ
        u = np.arange(4096, dtype=np.float32)
        _C['shaft'] = (R.fbm(u / 90.0, np.zeros_like(u) + 3.0, 4, seed=36) * 0.5 + 0.5).astype(np.float32)
    return _C


def background(t):
    """断面の背景（sRGB、(H, W, 3)）。t：秒（光の筋・コースティクス・粒・雲がゆっくり動く）。"""
    C = static()
    out = np.empty((H, W, 3), np.float32)
    with tick('xsec sky'):
        out[:WL] = R.deck_put(C['sky'], C['deck'], drift_px=-t * 4.0)
    with tick('xsec water'):
        wat = C['water'].copy()
        yy = np.arange(WL, H, dtype=np.float32)[:, None]
        xx = np.arange(W, dtype=np.float32)[None]
        d = yy - WL
        # 光の筋（左上から右下へ斜め）：1 次元のノイズを斜めの座標で読む。深いほど・下ほど弱い。ゆっくり流れる
        uu = (xx - d * 0.55 + t * 9.0) % 4096
        sh = np.interp(uu.ravel(), np.arange(4096), C['shaft']).reshape(uu.shape)
        sh = R.smoothstep(0.52, 0.85, sh) * np.exp(-d / 420.0) * R.smoothstep(0, 40, d)
        wat = wat + R.hexl('A8D6F2') * (sh * 0.16)[..., None]
        out[WL:] = wat
    with tick('xsec land'):
        sy = C['sy']
        la = C['land_a'][..., None]
        land = C['land']
        # 海底のコースティクス：ゆらめく網目（ゆがめた尾根のノイズ）。海底の上の面の近くだけ
        ys, ye = WL, H
        band = C['uw'][ys:ye]
        if band.any():
            yy2, xx2 = np.mgrid[ys:ye, 0:W].astype(np.float32)
            near = np.exp(-np.clip(yy2 - sy[None], 0, None) / 6.0) * band
            rows = np.nonzero(near.max(1) > 0.01)[0]
            if len(rows):
                r0, r1 = rows[0], rows[-1] + 1
                xq, yq = xx2[r0:r1], yy2[r0:r1]
                w1 = R.fbm(xq / 70.0 + t * 0.12, yq / 70.0, 1, seed=37) * 0.6
                cst = 1 - np.abs(R.fbm(xq / 38.0 + w1, yq / 22.0 - t * 0.18, 2, seed=38))
                cst = R.smoothstep(0.72, 0.95, cst) * near[r0:r1] * np.exp(-(sy[None] - WL) / 260.0)
                land = land.copy()
                land[ys + r0:ys + r1] += R.hexl('BFE3F7')[None, None] * (cst * 0.22)[..., None]
        L = R.lin(np.clip(R.srgb(out), 0, 1))
        L = L * (1 - la) + land * la
        out = L
    with tick('xsec motes'):
        f = R.srgb(out)
        m = C['motes']
        x = (m[:, 0] + t * 6.0 * m[:, 3] + 4 * np.sin(t * 0.7 + m[:, 4])) % W
        y = m[:, 1] - t * 3.0 * m[:, 3]
        y = WL + 8 + (y - WL - 8) % (H - WL - 8)
        lay = np.zeros((H, W), np.float32)
        ok = C['land_a'][y.astype(int).clip(0, H - 1), x.astype(int).clip(0, W - 1)] < 0.5
        # 粒は小さなガウスの点：双一次で 4 画素に配る（円を 1 つずつ描かない＝速い）
        xi, yi = x[ok], y[ok]
        wv = m[ok, 3] * np.exp(-(yi - WL) / 700.0) * m[ok, 2] ** 2 * 0.8
        x0i, y0i = np.floor(xi).astype(int), np.floor(yi).astype(int)
        fx_, fy_ = xi - x0i, yi - y0i
        for ddx, ddy, ww in ((0, 0, (1 - fx_) * (1 - fy_)), (1, 0, fx_ * (1 - fy_)), (0, 1, (1 - fx_) * fy_),
                             (1, 1, fx_ * fy_)):
            np.add.at(lay, (np.clip(y0i + ddy, 0, H - 1), np.clip(x0i + ddx, 0, W - 1)), wv * ww)
        lay = cv2.GaussianBlur(lay, (0, 0), 1.1) * 6.0
        f = f + (core.FOAM - f) * (lay * 0.35)[..., None]
        f = core.over(f, C['lines'])
    return np.clip(f, 0, 1).astype(np.float32)


# ---------------------------------------------------------------- ガラスの富士山
def fuji_glass(fill, bg):
    """富士山のガラスの体（輪郭の線は X.fuji_layer2 が描く）。fill：0..1。bg：その下の画（sRGB）。(H, W, 3) を返す。"""
    if fill <= 0.003:
        return bg
    if 'fuji' not in _C:
        with tick('fuji static'):
            P = X.fuji_outline()
            ss = 3
            m = np.zeros((H * ss, W * ss), np.uint8)
            cv2.fillPoly(m, [np.round(np.vstack([P, [[P[-1, 0], WL], [P[0, 0], WL]]]) * ss * 16).astype(np.int32)],
                         255, cv2.LINE_AA, shift=4)
            mask = cv2.resize(m.astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_AREA)
            # 丸い火山の体：各行の半分の幅 half(y) の円柱の断面（中心の稜線の折れ目が出ない）＋すその広がりの傾き
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            hp = 3776 * X.PXM
            hw = np.clip((WL - yy) / hp, 0, 1)
            half = 30 + 172 * (1 - hw) ** 1.5
            u = np.clip((xx - X.FUJI_X) / half, -1, 1)
            nzc = np.sqrt(np.clip(1 - u * u, 0, 1))
            dhalf = 172 * 1.5 * (1 - hw) ** 0.5 / hp               # 上へ 1px で half が縮む量
            nx, ny, nz = u, -dhalf * nzc * 0.9, nzc + 0.25
            nn = np.sqrt(nx * nx + ny * ny + nz * nz)
            nx, ny, nz = nx / nn, ny / nn, nz / nn
            Lw = np.array([-0.85, -0.30, 0.42], np.float32)
            Lw /= np.linalg.norm(Lw)
            ndl = np.clip(nx * Lw[0] + ny * Lw[1] + nz * Lw[2], 0, 1)
            Hh = Lw + np.array([0, 0, 1], np.float32)
            Hh /= np.linalg.norm(Hh)
            spec = np.clip(nx * Hh[0] + ny * Hh[1] + nz * Hh[2], 0, 1) ** 24
            fres = (1 - nz) ** 2.5
            # 山頂の雪：上の 22% を白く（下の縁は谷すじでぎざぎざ）
            top = X.y_of(3776)
            hh = (WL - yy) / (WL - top)
            streak = (R.fbm(xx / 16.0, yy / 110.0, 3, seed=41) * 0.07 + R.fbm(xx / 40.0, yy / 40.0, 2, seed=42) * 0.04) * \
                (0.4 + 0.6 * np.abs(u))
            snow = R.smoothstep(0.69, 0.79, hh + streak) * mask
            # 下ほどうすい（ふもとは霞に溶ける）
            body = mask * (0.10 + 0.20 * R.smoothstep(0.0, 1.0, hh))
            ys, xs = np.nonzero(mask > 0.002)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            cut = lambda a: np.ascontiguousarray(a[y0:y1, x0:x1])
            _C['fuji'] = dict(mask=cut(mask), ndl=cut(ndl), spec=cut(spec), fres=cut(fres), snow=cut(snow),
                              body=cut(body), box=(y0, y1, x0, x1))
    F = _C['fuji']
    y0, y1, x0, x1 = F['box']
    out = bg.copy()
    L = R.lin(bg[y0:y1, x0:x1])
    tint = R.hexl('D6EAF8')
    a = F['body'] * fill
    L = L * (1 - a[..., None] * 0.45) + tint * (a * (0.10 + 0.75 * F['ndl'] ** 1.5))[..., None]
    L = L + R.hexl('EAF5FD') * (F['fres'] * F['mask'] * 0.28 * fill)[..., None]
    L = L + R.SUN_RGB * (F['spec'] * F['mask'] * 0.16 * fill)[..., None]
    L = L * (1 - (F['snow'] * 0.55 * fill)[..., None]) + R.hexl('F4F9FD') * (F['snow'] * 0.55 * fill)[..., None] * \
        (0.75 + 0.25 * F['ndl'])[..., None]
    out[y0:y1, x0:x1] = R.srgb(L)
    return out
