"""look-dev 04：ガラスと金属のスピードメーター（_render.GlassDial）を、03 の新しい 3D の地図（ld03）を暗くぼかした上に。
針の動き・目盛りの数字・オドメーター・注は b04 のまま。python3 cutB/lookdev/ld04.py [frames...]"""
import sys

from _common import np, cv2, core, tick, report, save_pair, W, H  # noqa: F401
from cutB.beats import _render as R, _bmap as B, b04, b03, _motion as M
from cutB.beats._m34 import Board
import timeline_b as timeline
import ld03

_C = {}


def dial():
    if 'd' not in _C:
        with tick('dial static (once)'):
            _C['d'] = R.GlassDial(b04.MC, b04.R, vmax=b04.VMAX)
    return _C['d']


def frame_at(i, beat):
    n = beat['n']
    T = b04.timing(beat)
    f_ji, f_ki, f_o = T['ji'], T['ki'], T['o']
    t0 = B.n03()
    cnt = M.Odometer('1,130', W / 2, b04.Y_MAIN, digit_size=104, side_size=84, prefix='時速 約', suffix='km',
                     hl_suffix='km')
    note = M.KText([dict(text='旅客機（約900km/h）より速い', x=W / 2, y=b04.Y_NOTE, size=42, weight='Medium', track=0.04)])
    out = 1 - M.ease((i - (n - 6) + 1) / 6.0, 'cubic_in')
    push = b04.meter_push(i, n)
    dim = b04.DIM * core.ease((i + 1) / 8.0) + 0.32 * core.ease((i + 1) / 10.0) * out
    with tick('map'):
        cam = ld03.camera(t0 + i)
        f, hit, tz = ld03.render_map(cam, t0 + i)
        f, pk = ld03.overlay(f, cam, t0 + i, 1.0)
        # メーターの後ろは被写界深度でぼける（奥行き）
        f = cv2.GaussianBlur(f, (0, 0), 2.6)
    with tick('darken'):
        sh = 0.55 * core.ease((i + 1) / 8.0) * out
        appear = core.ease((i + 1) / 8.0) * out
        A_ = 1 - (1 - dim) * (1 - b04._VIG * sh) * (1 - b04._DISC * (0.82 * appear))
        f += (core.NAVY - f) * A_[..., None]
    with tick('dial'):
        val = b04.needle(i, T)
        ghosts = []
        if abs(val - b04.needle(i - 1, T)) * 180.0 / b04.VMAX > 4.0:
            ghosts = [(b04.needle(i - k / 3.0, T), a) for k, a in ((3, 0.1), (2, 0.2), (1, 0.35))]
        snd = M.ease((i - f_o + 1) / 6.0, 'sine_inout')
        excess = M.ease((i - f_o + 1) / 8.0, 'expo_out')
        D = dial()
        rgb, a, x0, y0 = D.frame(val, ghosts, excess, jet=b04.JET, jet_len=24 + 10 * M.ease((i - f_o + 1) / 7.0, 'back_out'),
                                 jet_a=0.55 + 0.45 * snd)
        grp = Board(rgb, a, x0, y0)
        LB = b04._label_boards()
        for k, lb in LB['labels']:
            grp.over(lb, alpha=1.0)
        grp.over(LB['jet'], alpha=0.6 + 0.4 * snd)
        if push > 1.0005:
            grp = grp.affine(push, b04.MC)
        # 計器の接地の影（文字盤の下に、やわらかく）
        grp.put(f, out)
    with tick('finish'):
        f = R.finish(f, i, bloom_k=0.14, bloom_thr=0.70, vignette=0.18, grain=0.009, ca_px=0.45)
    with tick('text'):
        if i >= f_ji:
            a = M.ease((i - f_ji + 2) / 4.0, 'sine_inout')
            v = min(max(val, 0.0), float(b04.V))
            dv = v - min(max(b04.needle(i - 1, T), 0.0), float(b04.V))
            f = cnt.put(f, v, a, i=i, f_lock=f_ki, dv=0.6 * dv)
        if i >= f_o - 1:
            f = note.reveal(f, i, f_o - 1, 'line_mask')
    return np.clip(f, 0, 1).astype(np.float32)


def main(frames):
    beat = next(b for b in timeline.load() if b['key'] == '04')
    ld03.textures()
    dial()
    report('04 static')
    for i in frames:
        out = frame_at(i, beat)
        report('04 f%d' % i)
        save_pair('04', i, out)


if __name__ == '__main__':
    main([int(v) for v in sys.argv[1:]] or [72])
