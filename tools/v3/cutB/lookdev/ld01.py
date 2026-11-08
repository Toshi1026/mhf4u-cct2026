"""look-dev 01：朝の浜（空の散乱＋立体の雲＋遠近の波の海＋泡＋砂）と、3D の悪役・拳、仕上げ。
タイミング・動き・文字は b01 のまま（b01 の関数を呼ぶ）。描く部品だけ _render / _figures に差し替える。
python3 cutB/lookdev/ld01.py [frames...]   → exports/v3/lookdev_B/b01_fNNNN_{after,before,sbs}.png"""
import math
import sys

from _common import np, cv2, core, tick, report, save_pair, W, H  # noqa: F401
from cutB.beats import _render as R, _figures as F, _art as A, _sky as S, b01, _motion as M
import timeline_b as timeline

HZ = S.HZ
EXTRA = b01.EXTRA
HH = H + EXTRA
CAM = R.HorizonCam(HZ, f=1660.0, cam_h=3.2)
SUN = R.sun_vec(-24.0, 14.0)              # 左の画面の外、低い朝の太陽（白）
LIGHT = (-0.62, -0.42, 0.30)              # 人形のキーの光（画面の向き）
ENV = dict(sky=R.hexl('9CCBEA') * 0.62, ground=R.hexl('B9C9D6') * 0.42, sun=np.array([1.0, 0.99, 0.975]) * 1.3,
           rim=R.hexl('E4F2FC') * 1.0)


def skyf(dx, dy, dz):
    return R.sky_radiance(dx, dy, dz, SUN)


XS = np.arange(W, dtype=np.float32) + 0.5
SHORE = S.shore_y(XS).astype(np.float32)
Y_SEA1 = int(SHORE.max()) + 14            # 寄せた水の膜の下端まで海を描く
Y_SAND0 = int(SHORE.min()) - 2
_C = {}


def static_layers():
    if not _C:
        with tick('sky(static)'):
            _C['sky'] = R.sky(CAM, HZ + 2, W, SUN)
        with tick('sand(static)'):
            _C['sand'], _C['sand_info'] = R.sand(CAM, Y_SAND0, HH, W, 0.0, SUN, SHORE, skyf)
    return _C


def world_bg(gi):
    """S.frame(gi, EXTRA) の代わり（sRGB、(HH, W, 3)）。"""
    C = static_layers()
    t = gi / core.FPS
    out = np.zeros((HH, W, 3), np.float32)
    with tick('sky'):
        out[:HZ + 2] = C['sky']
    with tick('clouds'):
        ccol, ca = R.clouds(CAM, HZ, W, SUN, t=t + 40.0, coverage=0.40, scale=2400.0, wind=(26.0, 0.0), seed=4,
                            cirrus=0.45)
        out[:HZ] = out[:HZ] * (1 - ca[..., None]) + ccol * ca[..., None]
    with tick('sea'):
        yy = np.arange(HZ, Y_SEA1, dtype=np.float32)[:, None] + 0.5
        Zr = CAM.ground(yy, XS[None])[1]
        Zs = CAM.h * CAM.f / np.maximum(SHORE[None] - HZ, 1)
        shallow = np.exp(-np.clip(Zr - Zs, 0, None) / 3.5) * 0.75
        sea, info = R.sea(CAM, HZ, Y_SEA1, W, t, SUN, skyf, amp=0.9, shallow=shallow)
    with tick('foam'):
        foam = R.shore_foam(CAM, HZ, Y_SEA1, W, SHORE, t)
        sea = sea * (1 - foam[..., None] * 0.85) + R.hexl('F2F8FC') * 1.05 * (foam[..., None] * 0.85)
    with tick('compose'):
        out[HZ:Y_SEA1] = sea
        # 砂（波打ち際より手前）。波打ち際の泡は砂の上にも少しかかる（寄せた水の膜）
        sand = C['sand']
        ys = np.arange(Y_SAND0, HH, dtype=np.float32)[:, None] + 0.5
        ph = 0.5 + 0.5 * math.sin(t * 1.6)
        film_edge = SHORE[None] + 2.0 + 5.0 * ph + 1.2 * np.sin(XS[None] / 37.0 + t * 2.2)
        m = np.clip(ys - film_edge + 0.5, 0, 1)[..., None]
        reg = out[Y_SAND0:HH]
        # 寄せた水の膜（波打ち際のすぐ下）：空を映して明るい
        film = np.clip(1 - (ys - SHORE[None]) / 14.0, 0, 1) * (1 - m[..., 0]) * (ys >= SHORE[None])
        reg_sand = sand * m + reg * (1 - m)
        top = np.minimum(Y_SEA1, HH) - Y_SAND0
        below_shore = (ys >= SHORE[None])[..., None].astype(np.float32)
        reg_new = reg_sand * below_shore + reg * (1 - below_shore)
        reg_new = reg_new + (R.hexl('CFE5F5') - reg_new) * (film[..., None] * 0.45)
        out[Y_SAND0:HH] = reg_new
        if top < 0:
            pass
        f = R.srgb(out)
    return f


# ---------------------------------------------------------------- 人形の差し替え
def villain_rgba(x, y, scale, ang=0.0, pose='fly', t=0.0, squash=1.0, M2=None, cape_ang=None, s=4):
    return F.villain_relief(x, y, scale, ang, pose, t, squash, M2, cape_ang, light=LIGHT, env=ENV)


_FI = {}


def fist_sprite():
    if 'f' not in _FI:
        with tick('fist(once)'):
            rgba, ax, ay = F.fist_relief(b01.DIR, b01.FS, light=LIGHT, env=ENV)
            # 空中の拳：ごく弱いやわらかい影（にじみ）
            sa = np.clip(cv2.GaussianBlur(rgba[..., 3], (0, 0), 7) * 0.22, 0, 1)
            sa = np.roll(np.roll(sa, 6, 0), 4, 1)
            a = rgba[..., 3]
            oa = a + sa * (1 - a)
            rgb = (rgba[..., :3] * a[..., None] + A.SHADOW * (sa * (1 - a))[..., None]) / np.maximum(oa, 1e-5)[..., None]
            _FI['f'] = (np.concatenate([rgb, oa[..., None]], 2).astype(np.float32), ax, ay)
    return _FI['f']


A.villain_rgba = villain_rgba
b01.fist_sprite = fist_sprite


def ground_shadow(f, rgba, x0, y0, feet, opacity=0.42):
    """立っている悪役の砂への影（太陽は左奥：影は右手前へ伸びる）＋足もとの接地の影。"""
    a = rgba[..., 3]
    h, w = a.shape
    fx, fy = feet[0] - x0, feet[1] - y0
    pad = 160
    big = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
    big[pad:pad + h, pad:pad + w] = a
    # 上へ伸びた形 → 地面に倒す（下・右へ）：y' = fy + (fy - y)·0.30、x' = x + (fy - y)·0.75
    sq, sh = 0.30, 0.75
    Mx = np.float32([[1, -sh, sh * (fy + pad)], [0, -sq, (fy + pad) * (1 + sq)]])
    sm = cv2.warpAffine(big, Mx, (big.shape[1], big.shape[0]), flags=cv2.INTER_LINEAR)
    # 遠いほどぼける（足もとはくっきり）
    yy = np.arange(big.shape[0], dtype=np.float32)[:, None] - (fy + pad)
    b1 = cv2.GaussianBlur(sm, (0, 0), 1.2)
    b2 = cv2.GaussianBlur(sm, (0, 0), 6.0)
    k = np.clip(yy / 40.0, 0, 1)
    sm = (b1 * (1 - k) + b2 * k) * opacity * (1 - 0.5 * k)
    X0, Y0 = x0 - pad, y0 - pad
    xa, ya, xb, yb = max(X0, 0), max(Y0, 0), min(X0 + big.shape[1], f.shape[1]), min(Y0 + big.shape[0], f.shape[0])
    sub = sm[ya - Y0:yb - Y0, xa - X0:xb - X0, None]
    col = core.c('3F5E7C')
    f[ya:yb, xa:xb] = f[ya:yb, xa:xb] * (1 - sub) + col * sub
    return R.contact_shadow(f, feet[0] + 2, feet[1] + 1, 16, 3.5, opacity=0.45, rgb=core.c('2E4A66'))


def put_villain(f, i, T):
    h = T['hit']
    ta = M.hitstop(i, [(h, b01.HOLD)])
    if i >= h + b01.HOLD:
        x, y, sc, ang, tp = b01.flight(ta, T)
        if sc * 34 < 5:
            return A.put_dot(f, x, y, max(1.6, sc * 8))
        sp = float(np.hypot(*b01.flight_vel(ta, T)))
        n = int(np.clip(math.ceil(sp * 0.35 / 3.0), 1, 10)) if sp > 40 else 1
        r = M.subframe(lambda t: b01.villain_draw(t, T), i, shutter=0.35, n=n)
        return M.put_rgba(f, *r, shadow=0.10, shadow_px=max(1.0, 1.1 * sc)) if r else f
    r = b01.villain_draw(float(i), T)
    f = ground_shadow(f, r[0], r[1], r[2], S.FEET)
    return M.put_rgba(f, *r)


def frame_at(i, T, gi_end, sparks, dust):
    h = T['hit']
    with tick('world'):
        f = world_bg(i)
    with tick('fx+figures'):
        if i >= h + b01.HOLD:
            ta = M.hitstop(i, [(h, b01.HOLD)])
            tau = b01.flight_tau(ta, T)
            tp = 1 - (1 - tau) ** 2.2
            f = b01.trail(f, tp, tau, 1 - core.ease((i - T['glint'] - 2) / 14.0))
        for k in range(dust.f0, i + 1):
            pass
        f = dust.draw(f, i)
        if i < T['glint']:
            f = put_villain(f, i, T)
        else:
            tw = 0.85 + 0.15 * np.cos((i - T['glint']) * 0.35)
            f = A.put_dot(f, S.DOT[0], S.DOT[1], 2.3, alpha=tw)
        f = b01.impact(f, i - h)
        f = sparks.draw(f, i)
        f = b01.put_fist(f, i, T)
        f = A.put_glint_fx(f, S.DOT[0], S.DOT[1], i - T['glint'])
    with tick('camera'):
        Mc = b01.cam_matrix(i, T, gi_end)
        f = cv2.warpAffine(f, Mc.astype(np.float32), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    with tick('finish'):
        f = R.finish(f, i, bloom_k=0.12, bloom_thr=0.70, vignette=0.14, grain=0.009, ca_px=0.5, distort=0.6)
    with tick('text'):
        f = b01.put_texts(f, i, T)
    return np.clip(f, 0, 1).astype(np.float32)


def main(frames):
    beat = next(b for b in timeline.load() if b['key'] == '01')
    T = b01.timing(beat)
    gi_end = b01.push_end()
    sparks, dust = b01.particles(T)
    static_layers()
    fist_sprite()
    report('01 static')
    for i in frames:
        out = frame_at(i, T, gi_end, sparks, dust)
        report('01 f%d' % i)
        save_pair('01', i, out)


if __name__ == '__main__':
    main([int(v) for v in sys.argv[1:]] or [46, 62])
