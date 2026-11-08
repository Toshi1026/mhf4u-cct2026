"""look-dev 05：35 頭のゾウを 3D（_figures3d の SDF）で、スタジオの床（遠近の床・上からのやわらかいスポット・
床へのうすい映り込みと接地の影・空気中の細かい粒）に並べる。並び・落ちる動き・文字は b05 のまま。
python3 cutB/lookdev/ld05.py [frames...]"""
import sys

from _common import np, cv2, core, tick, report, save_pair, W, H  # noqa: F401
from cutB.beats import _render as R, _figures3d as F3, b05, _motion as M
import timeline_b as timeline

LIGHT = (-0.55, -0.70, 0.45)
ENV = dict(sky=R.hexl('7FB4DE') * 0.42, ground=R.hexl('123E66') * 0.45, sun=np.array([1.0, 0.99, 0.98]) * 1.45,
           rim=R.hexl('BFE4FA') * 1.1)
FLOOR_Y = 742.0                     # 床の奥の端（いちばん上の段のうしろ）
_C = {}


def background():
    """スタジオ：上は濃紺の奥行き、床は手前へ明るく（上からのスポットの光だまり）。線形で作って sRGB に。"""
    if 'bg' in _C:
        return _C['bg']
    with tick('bg static'):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        top, mid = R.hexl('04121F'), R.hexl('0B3459')
        # 奥の壁：上ほど暗い、中央が少し明るい（スポットのこぼれ）
        r_w = ((xx - W / 2) / (W * 0.65)) ** 2 + ((yy - 640) / 520.0) ** 2
        wall = top + (mid - top) * (np.exp(-r_w * 1.4) * 0.9)[..., None]
        # 床：遠近（奥の端 FLOOR_Y → 手前）。光だまりは楕円（遠近でつぶれる）
        fy = np.clip(yy - FLOOR_Y, 0, None)
        Z = 900.0 / (fy + 60.0)                          # 床の奥行き（任意の単位）
        X = (xx - W / 2) * Z / 900.0
        pool = np.exp(-((X / 2.6) ** 2 + ((Z - 4.2) / 3.2) ** 2))
        floor = R.hexl('07213B') * (0.75 + 0.25 * np.clip(fy / 700.0, 0, 1))[..., None] + R.hexl('2D6EA6') * (pool * 0.50)[..., None]
        # 床のごく細かいつやの変化（ブラシの跡のような横の筋）
        br = R.fbm(X * 6.0, Z * 40.0, 3, seed=61) * 0.04
        floor = floor * (1 + br)[..., None]
        k = R.smoothstep(FLOOR_Y - 30, FLOOR_Y + 40, yy)[..., None]       # 壁と床の境はやわらかく（無限の背景）
        bg = wall * (1 - k) + floor * k
        # 上からの光の筋（ごく弱い、動かない）
        u = (xx - W / 2) / np.maximum(yy + 300, 1)
        beam = np.exp(-(u / 0.22) ** 2) * np.clip(1 - yy / 1300.0, 0, 1)
        bg = bg + R.hexl('5E9ACC') * (beam * 0.05)[..., None]
        _C['bg'] = R.srgb(bg).astype(np.float32)
    return _C['bg']


def elephant():
    if 'el' not in _C:
        with tick('elephant sdf (once)'):
            rgba, ax, ay = F3.elephant_rgba(b05.ESC * 1.06, light=LIGHT, env=ENV)
            # 床への映り込み（上下反転、足もとから下へ消える）
            h = rgba.shape[0]
            ref = rgba[::-1].copy()
            g = np.clip(1 - (np.arange(h, dtype=np.float32) - (h - ay)) / 40.0, 0, 1)[:, None]
            ref[..., 3] *= g * 0.16
            ref = cv2.GaussianBlur(ref, (0, 0), 1.6)
            _C['el'] = (rgba, ax, ay)
            _C['ref'] = (ref, ax, h - ay)
    return _C['el'], _C['ref']


def put_elephant(f, x, y, sx, sy, a):
    (el, ax, ay), (ref, rx, ry) = elephant()
    land = float(np.clip((sy - 0.84) / 0.21, 0, 1)) if a > 0.99 else 0.0
    # 接地の影（足もと、着いているときだけ）
    if a > 0.5:
        f = R.contact_shadow(f, x + 2, y + 1, 40 * sx, 4.5, opacity=0.55 * a, rgb=core.c('020A14'), blur=3.2)
        f = b05.put_sprite(f, ref, rx, ry, x, y, sx, a * 0.9, sy=sy)
    return b05.put_sprite(f, el, ax, ay, x, y, sx, a, sy=sy)


def frame_at(i, beat):
    n = beat['n']
    mf = lambda k: timeline.mora_frame(beat, k)
    f_pa, f_ya, f_n, f_zo = mf(0), mf(4), mf(13), mf(14)
    f_sa, f_go = mf(16), mf(20)
    head = M.KText([dict(text='パンチの力', x=W / 2, y=410, size=48, weight='Medium', track=0.12)])
    ctr = M.Odometer('190', W / 2, 588, digit_size=150, side_size=96, prefix='約', suffix='トン', lock_scale=1.12,
                     lock_drop=6.0, lock_dur=8)
    sub = M.Odometer('35', W / 2, 686, digit_size=48, side_size=48, prefix='ゾウ 約', suffix='頭分', weight='Medium',
                     side_weight='Medium', gap=0.08, digit_color=core.FOAM, lock_dur=6, lock_scale=1.08, lock_drop=3.0,
                     hide_zero=True, blur_k=0.3)
    note1 = M.KText([dict(text='アフリカゾウ（大人のオス）約5.5t として', x=W / 2, y=1162, size=28, weight='Medium',
                          alpha=0.9, track=0.04)])
    note2 = M.KText([dict(text='体重60kg・パンチが当たる時間0.01秒・空気抵抗なし（試算）', x=W / 2, y=1206, size=28,
                          weight='Medium', alpha=0.8, track=0.02)])
    dusts = b05.dust(f_zo)

    def val(t):
        return 190 * M.ease((t - f_ya) / float(f_n - f_ya), 'expo_out')
    with tick('bg'):
        f = background().copy()
    with tick('elephants'):
        gy = b05.grid_nudge(i, f_zo)
        for x, y, sx, sy, a in b05.elephants_at(i, f_zo):
            f = put_elephant(f, x, y + gy, sx, sy, a * 0.97)
        for b in dusts:
            if 0 <= i - b.f0 < 9:
                f = b.draw(f, i)
    with tick('camera'):
        tq = i / float(n - 1)
        k = 1 + (b05.PUSH - 1) * (0.5 * tq + 0.5 * M.ease(tq, 'sine_inout'))
        sh = M.shake(i, f_n, amp=4.0, tau=2.0, rot_deg=0.0, seed=5)
        cx, cy = W / 2, 800.0
        Mw = np.float32([[k, 0, cx * (1 - k) + sh[0]], [0, k, cy * (1 - k) + sh[1]]])
        f = cv2.warpAffine(f, Mw, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    with tick('finish'):
        f = R.finish(f, i, bloom_k=0.12, bloom_thr=0.62, vignette=0.20, grain=0.010, ca_px=0.45, distort=0.0)
    with tick('text'):
        # 文字は b05 では押し込みの前に重ねているが、ここでは仕上げ（粒・にじみ）を文字にかけないため後で重ね、同じ押し込みを掛ける
        txt = np.zeros((H, W, 3), np.float32)
        lay = [txt.copy(), np.ones((H, W, 3), np.float32)]
        out = []
        for base in lay:
            g = base.copy()
            g = head.reveal(g, i, f_pa - 2, 'char_rise', stagger=1.3, dur=10)
            v = val(i)
            if v >= 8 or i >= f_n:
                a = float(np.clip((v - 8) / 22.0, 0, 1)) if i < f_n else 1.0
                g = ctr.put(g, v, a, i=i, f_lock=f_n, dv=val(i + 0.5) - val(i - 0.5))
                g = note2.reveal(g, i, f_ya + 1, 'line_mask')
            s0 = f_sa - 3
            if i >= s0:
                cnt = lambda t: 35 * M.ease(min(1.0, max(0.0, (t - s0 + 1) / float(f_go - s0 + 1))), 'quad_out')
                ac = M.ease((i - s0 + 1) / 4.0, 'sine_inout')
                g = sub.put(g, cnt(i), ac, i=i, f_lock=f_go, dv=cnt(i + 0.5) - cnt(i - 0.5))
            if i >= f_zo + 6:
                g = note1.reveal(g, i, f_zo + 6, 'line_mask')
            out.append(cv2.warpAffine(g, Mw, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        # 黒の上と白の上の 2 枚から、文字の色とアルファを取り出して重ねる
        a = np.clip(1 - (out[1] - out[0]), 0, 1)
        col = out[0] / np.maximum(a, 1e-4)
        f = f * (1 - a) + col * a
    return np.clip(f, 0, 1).astype(np.float32)


def main(frames):
    beat = next(b for b in timeline.load() if b['key'] == '05')
    background()
    elephant()
    report('05 static')
    for i in frames:
        out = frame_at(i, beat)
        report('05 f%d' % i)
        save_pair('05', i, out)


if __name__ == '__main__':
    main([int(v) for v in sys.argv[1:]] or [84])
