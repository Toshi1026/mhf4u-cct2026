"""look-dev 03：本当の地形（国土地理院の標高タイル）の 3D の地図。斜めに見下ろすカメラ（遠近）、陰影起伏＋地形の AO、
海は岸からの距離で深さの色（等深線ふうの細い線）、くっきりした海岸線（光らせない）、空気遠近と浅い被写界深度、
桂浜 → MAZE の前の海の放物線は 3D の弧（高さ R/4）として描き、海面にうすい影の線。文字・タイミングは b03 のまま。
python3 cutB/lookdev/ld03.py [frames...]
出典：国土地理院（標高タイル）を加工して作成。"""
import math
import sys

from _common import np, cv2, core, tick, report, save_pair, W, H
from beats import b11, _tosa, _geo as G
from beats import _mapui as U
from cutB.beats import _render as R, _bmap as B, b03, _motion as M
import timeline_b as timeline

C0, C1 = _tosa.CENTER
KX = 111.32 * math.cos(math.radians(C0)) * _tosa.PX_KM
KY = 111.2 * _tosa.PX_KM
CELL = 0.12                                     # 地図の px（1 画素 ≈ 26m）
EXAG = 1.8                                      # 高さの強調（2.5 はとげとげに見えた）
MPX = _tosa.PX_KM / 1000.0                      # 1m → 地図の px
LL = (33.15, 33.95, 133.15, 133.95)


def world_xy(lat, lon):
    return _tosa.CX + (np.asarray(lon) - C1) * KX, _tosa.CY - (np.asarray(lat) - C0) * KY


X0, Z0 = world_xy(LL[1], LL[2])                 # 左上（北西）
X1, Z1 = world_xy(LL[0], LL[3])
NX, NZ = int((X1 - X0) / CELL), int((Z1 - Z0) / CELL)
KAT = np.array(world_xy(*B.KAT_LL), float)      # 本当の桂浜（地形の海岸線の上）
DEST = np.array(world_xy(*B.DEST_LL), float)    # MAZE の前の海
_C = {}


def elev():
    def mk():
        E, info = R.gsi_dem(*LL, z=12)
        xs = X0 + (np.arange(NX) + 0.5) * CELL
        zs = Z0 + (np.arange(NZ) + 0.5) * CELL
        lon = C1 + (xs - _tosa.CX) / KX
        lat = C0 - (zs - _tosa.CY) / KY
        LON, LAT = np.meshgrid(lon, lat)
        out = np.empty((NZ, NX), np.float32)
        for r0 in range(0, NZ, 512):
            out[r0:r0 + 512] = R.gsi_sample_ll(E, info, LAT[r0:r0 + 512], LON[r0:r0 + 512])
        return out
    return R.cached('ld03_elev|%d|%d|%.3f' % (NX, NZ, CELL), mk)


def sea_land(E):
    """標高の NaN を「海」と「川・湖・欠測」に分ける：細い NaN（川）を開いて切り、南の端につながる大きな NaN だけを海に。"""
    nanm = np.isnan(E).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    op = cv2.morphologyEx(nanm, cv2.MORPH_OPEN, k)
    n, lab = cv2.connectedComponents(op, connectivity=4)
    keep = np.unique(lab[-1][op[-1] > 0])
    sea0 = np.isin(lab, keep[keep > 0]).astype(np.uint8)
    sea = (cv2.dilate(sea0, k, iterations=2) > 0) & (nanm > 0)
    river = (nanm > 0) & ~sea
    return sea, river


def textures():
    if _C:
        return _C
    with tick('terrain textures (once)'):
        E = elev()
        sea, river = sea_land(E)
        land = ~sea
        Ef = np.where(np.isnan(E), 0.0, np.maximum(E, 0)).astype(np.float32)
        Ef = cv2.GaussianBlur(Ef, (0, 0), 0.9)
        cm = CELL / MPX                                         # 1 画素の m
        hs = 0.62 * R.hillshade(Ef, cm, az=315, el=36, exag=EXAG) + 0.38 * R.hillshade(Ef, cm, az=250, el=55, exag=EXAG)
        ao = R.terrain_ao(Ef, cm, radii=(90, 360, 1400), exag=EXAG)
        hgt = (Ef * MPX * EXAG).astype(np.float32)
        with tick('terrain cast shadow (once)'):
            shd = R.cached('ld03_shadow|%d|%d|%.2f' % (NX, NZ, EXAG),
                           lambda: R.terrain_shadow(hgt, CELL, az=315, el=24, maxd=500))
        shd = cv2.GaussianBlur(shd, (0, 0), 1.0)
        mb = cv2.GaussianBlur(land.astype(np.float32), (0, 0), 1.2)
        gy, gx = np.gradient(mb)
        sdf = (mb - 0.5) / np.maximum(np.hypot(gx, gy), 1e-3)  # 海岸からの符号つき距離（画素）
        # 陸：標高で色（平野は明るい青灰・田畑のごく弱いまだら、山は深い青、尾根は光を受ける）
        e = Ef
        lowc, midc, highc = R.hexl('5A82A3'), R.hexl('3A5F7E'), R.hexl('5F84A3')
        k1 = R.smoothstep(15, 160, e)[..., None]
        k2 = R.smoothstep(300, 1300, e)[..., None]
        base = lowc * (1 - k1) + midc * k1 * (1 - k2) + highc * k2
        xs = np.arange(NX, dtype=np.float32)[None] * CELL
        zs = np.arange(NZ, dtype=np.float32)[:, None] * CELL
        fields = R.fbm(xs * 3.0, zs * 3.0, 3, seed=71)
        base = base * (1 + 0.05 * fields * (1 - k1[..., 0]))[..., None]
        light = (hs ** 1.1) * (0.35 + 0.65 * shd)
        landc = base * (0.30 + 1.05 * light)[..., None] + R.hexl('CFE2F1') * (0.12 * light ** 4)[..., None]
        landc = landc * (0.50 + 0.50 * ao)[..., None]
        # 川・湖：細い水の色
        rv = cv2.GaussianBlur(river.astype(np.float32), (0, 0), 0.7)[..., None]
        landc = landc * (1 - rv * 0.7) + R.hexl('2B6A9A') * (rv * 0.7)
        # 海：岸からの距離で深さ（浅い＝明るい青 → 大陸棚 → 深い紺）。うねりのような大きなむら
        seau = sea.astype(np.uint8)
        dist = cv2.distanceTransform(np.pad(seau, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1] * CELL / _tosa.PX_KM
        dn = dist * (1 + 0.25 * R.fbm(xs * 0.6, zs * 0.6, 3, seed=72))
        sh_, mid, deep = R.hexl('3C88BA'), R.hexl('124878'), R.hexl('06203D')
        d1 = 1 - np.exp(-dn / 1.2)
        d2 = 1 - np.exp(-dn / 9.0)
        seac = sh_ * (1 - d1)[..., None] + mid * (d1 * (1 - d2))[..., None] + deep * d2[..., None]
        for km, a in ((1.0, 0.10), (3.0, 0.09), (6.0, 0.08), (10.0, 0.07)):
            seac += R.hexl('9CC4E2') * (a * np.exp(-((dn - km) / 0.04) ** 2))[..., None]
        # 海底のごく弱い起伏（水を通して見える質感。地形の主張はしない）：ノイズの高さを陰影にして、深いほど消す
        sfl = R.fbm(xs * 0.25, zs * 0.25, 5, gain=0.55, seed=74, scale=1.0)
        shs = R.hillshade(sfl * 0.6, 0.12, az=315, el=40)
        seac = seac * (1 + (shs - 0.6)[..., None] * 0.22 * (1 - d2)[..., None])
        # 波打ち際の浅い明るい帯（光らせない：細く・うすく）
        seac += R.hexl('8DC2E4') * (0.10 * np.exp(-dist / 0.12))[..., None]
        kk = np.clip(sdf + 0.5, 0, 1)[..., None]
        col = seac * (1 - kk) + landc * kk
        line = np.exp(-(sdf / 0.8) ** 2)
        col = col + (R.hexl('D7E8F4') - col) * (line * 0.40)[..., None]
        _C['col'] = col.astype(np.float32)
        _C['col_blur'] = cv2.GaussianBlur(_C['col'], (0, 0), 1.6)
        _C['seam'] = cv2.GaussianBlur(sea.astype(np.float32), (0, 0), 0.8)
        _C['hgt'] = np.where(sea, np.nan, cv2.GaussianBlur(hgt, (0, 0), 0.7)).astype(np.float32)
    return _C


def camera(t):
    """03 の寄りの最後の構図に合わせた、斜めに見下ろすカメラ。弧の中心を画面の (540, 760) あたりに。"""
    mid = (KAT + DEST) / 2
    ch = DEST - KAT
    rl = float(np.hypot(*ch))
    tgt = np.array([mid[0], rl * 0.125, mid[1]])
    perp = np.array([ch[1], -ch[0]]) / rl                       # 弧に直交（陸の方＝北北西）
    if perp[1] > 0:
        perp = -perp
    pitch = math.radians(42.0)
    dist = 3140.0 * rl / (B.ARC_FRAC * W) * 1.02
    # 寄りの押し込み（drift）に合わせて少しだけ近づく
    dist /= math.exp(B.drift(t))
    eye = np.array([tgt[0] - perp[0] * dist * math.cos(pitch), tgt[1] + dist * math.sin(pitch),
                    tgt[2] - perp[1] * dist * math.cos(pitch)])
    return R.Cam3D(eye, tgt, fov=34.0, pp=(540.0 + B.pan(t)[0] * 0.3, 760.0))


def render_map(cam, t=0.0):
    T = textures()
    with tick('heightfield raymarch'):
        hit = R.heightfield_render_fast(cam, T['hgt'], (X0, Z0), CELL, scale=0.5)
    with tick('shade'):
        u = ((hit['x'] - X0) / CELL - 0.5).astype(np.float32)
        v = ((hit['z'] - Z0) / CELL - 0.5).astype(np.float32)
        c0 = cv2.remap(T['col'], u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        c1 = cv2.remap(T['col_blur'], u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fp = hit['t'] / cam.fpx / CELL                         # 1 画面 px が何画素か
        kb = np.clip((fp - 0.8) / 1.5, 0, 1)[..., None]
        col = c0 * (1 - kb) + c1 * kb
        # 海面のフレネル（遠く＝浅い角度ほど空を映して明るい）
        yy, xx = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32)
        D = cv2.resize(cam.rays(yy * 2 + 1.0, xx * 2 + 1.0).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        sea = cv2.remap(T['seam'], u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # 海面のうねり：世界の座標の fBm の傾きを法線に（遠くは平均に溶ける）。ゆっくり流れる
        tt = t / core.FPS
        hx2 = cv2.resize(hit['x'], (W // 2, H // 2), interpolation=cv2.INTER_AREA) / 1.6 + tt * 0.05
        hz2 = cv2.resize(hit['z'], (W // 2, H // 2), interpolation=cv2.INTER_AREA) / 0.7 + tt * 0.08
        e_ = 0.08                                                # 世界の座標の差分（解析的な傾き：画素の並びに依らない）
        h0 = R.fbm(hx2, hz2, 3, seed=73)
        gx_ = (R.fbm(hx2 + e_, hz2, 3, seed=73) - h0) / e_ * 0.06
        gz_ = (R.fbm(hx2, hz2 + e_, 3, seed=73) - h0) / e_ * 0.06
        fp2 = cv2.resize(hit['t'], (W // 2, H // 2), interpolation=cv2.INTER_AREA) / cam.fpx * 2 / 1.6
        att = np.clip(1.5 - fp2 * 2.0, 0, 1)                     # 1px（半分の解像度）が波の大きさに近づくと消す
        gx_ = cv2.resize(gx_ * att, (W, H), interpolation=cv2.INTER_LINEAR)
        gz_ = cv2.resize(gz_ * att, (W, H), interpolation=cv2.INTER_LINEAR)
        nx_, nz_ = -gx_, -gz_
        nn_ = np.sqrt(nx_ ** 2 + nz_ ** 2 + 1)
        dn = (D[..., 0] * nx_ + D[..., 1] + D[..., 2] * nz_) / nn_
        ry = D[..., 1] - 2 * dn / nn_
        cosv = np.clip(-dn, 0, 1)
        F = (0.02 + 0.98 * (1 - cosv) ** 5) * sea
        env = R.hexl('1E4D7B')[None, None] * (1 - np.clip(ry, 0, 1))[..., None] + R.hexl('6FA3CF')[None, None] * \
            np.clip(ry, 0, 1)[..., None] ** 1.5
        col = col * (1 - F * 1.6)[..., None] + env * (F * 1.6)[..., None]
        # 空気遠近（遠いほど霞の青へ）
        t0 = float(np.median(hit['t'][H // 2 - 50:H // 2 + 50, W // 2 - 50:W // 2 + 50]))
        col = R.aerial(col, np.clip(hit['t'] - t0 * 0.7, 0, None) / t0, R.hexl('1B456C'), 0.55)
        f = R.srgb(col)
    with tick('dof'):
        bl = cv2.GaussianBlur(f, (0, 0), 2.2)
        k = np.clip(np.abs(hit['t'] / t0 - 1.0) * 1.6 - 0.12, 0, 1) ** 1.2
        f = f * (1 - k[..., None]) + bl * k[..., None]
    return f, hit, t0


def arc3d(s):
    s = np.asarray(s, float)
    g = KAT[None] + (DEST - KAT)[None] * s[:, None]
    rl = float(np.hypot(*(DEST - KAT)))
    h = 4 * s * (1 - s) * rl * 0.25
    return np.stack([g[:, 0], h, g[:, 1]], 1), np.stack([g[:, 0], np.zeros_like(h), g[:, 1]], 1)


def overlay(f, cam, t, u):
    """3D の弧（白い破線）、海面の影の線、先頭の点、桂浜の点とラベル。"""
    A3, G3 = arc3d(np.linspace(0, u, max(3, int(240 * u))))
    a2, _ = cam.project(A3)
    g2, _ = cam.project(G3)
    L = G.Lines()
    # 海面の上の影（弧の真下の線）：うすく暗く
    Ls = G.Lines()
    for seg in G.dashed(g2, dash=9, gap=9, phase=b03.dash_phase(t) * 0.6):
        Ls.poly(seg, 2.2, 255 * 0.55)
    m = Ls.done()
    m = cv2.GaussianBlur(m, (0, 0), 1.2)[..., None]
    f = f * (1 - m * 0.55) + core.c('03101E') * (m * 0.55)
    # 弧：白い破線（根元は太く、遠い先は少し細い＝遠近）
    segs = G.dashed(a2, dash=14, gap=8, phase=b03.dash_phase(t))
    for j, seg in enumerate(segs):
        q = j / max(1, len(segs) - 1)
        L.poly(seg, 3.4 - 0.8 * q * u, 255 * (0.97 - 0.2 * q * u))
    if u >= 0.999:
        hr = b03.land_r(t)
        L.dot(a2[-1, 0], a2[-1, 1], hr, 255)
    pk = cam.project(np.array([[KAT[0], 0.0, KAT[1]]]))[0][0]
    L.dot(pk[0], pk[1], 7.5, 255)
    m = L.done()
    # 線の下にごく弱いにじみの影（地図から浮かせる）
    sh = cv2.GaussianBlur(m, (0, 0), 3.0)[..., None] / 255 * 0.35
    f = f * (1 - sh) + core.c('020A14') * sh
    f = b03._blend(f, m, core.FOAM)
    # 足もとから弧の頂点までの細い垂線（高さの手がかり）
    ap, _ = cam.project(arc3d(np.array([0.5]))[0])
    ag, _ = cam.project(arc3d(np.array([0.5]))[1])
    if u > 0.5:
        Lv = G.Lines()
        for seg in G.dashed(np.array([ag[0], ap[0]]), dash=5, gap=6):
            Lv.poly(seg, 1.4, 255 * 0.45)
        f = b03._blend(f, Lv.done(), core.FOAM)
    return f, pk


def frame_at(i, beat):
    T = b03.timing(beat)
    cam = camera(i)
    with tick('map'):
        f, hit, t0 = render_map(cam, i)
    with tick('overlay+text'):
        u = b03.arc_u(i)
        f, pk = overlay(f, cam, i, u)
    with tick('finish'):
        f = R.finish(f, i, bloom_k=0.08, bloom_thr=0.78, vignette=0.22, grain=0.010, ca_px=0.45, distort=0.0)
    with tick('overlay+text'):
        lab = b03._klabel()
        f = lab.reveal(f, i, T['ka'] - 2, 'track', track_from=0.20, dur=12, dx=pk[0] + 22 - 540, dy=pk[1] + 40 - 1000)
        sub, cnt = b03.texts()
        sa = 0.42 * M.ease((i - T['ka'] + 2) / 6.0, 'sine_inout')
        if sa > 0:
            f = U.scrim(f, b03.Y_SUB - 60, b03.Y_BIG + 10, strength=sa, feather=150)
        f = sub.reveal(f, i, T['ka'] - 2, 'char_rise')
        if u > 0.06:
            v = b03.counter_v(i)
            a = M.ease((u - 0.06) / 0.08, 'sine_inout')
            f = cnt.put(f, v, a, i=i, f_lock=T['g1'], dv=0.6 * (v - b03.counter_v(i - 1)))
    return np.clip(f, 0, 1).astype(np.float32)


def main(frames):
    beat = next(b for b in timeline.load() if b['key'] == '03')
    textures()
    report('03 static')
    for i in frames:
        out = frame_at(i, beat)
        report('03 f%d' % i)
        save_pair('03', i, out)


if __name__ == '__main__':
    main([int(v) for v in sys.argv[1:]] or [76])
