"""look-dev の 1 フレームの時間（静的な層を作ったあと、連続 4 フレームの平均の実時間）と、
となりのフレームの平均の明るさの差（点滅の規則：≤0.1/フレーム）を測る。保存はしない。
python3 cutB/lookdev/bench.py [01 03 04 05 06 07]"""
import sys
import time

from _common import np, core, TIMES
import timeline_b as timeline


def luma(f):
    return float((f[..., 0] * 0.2126 + f[..., 1] * 0.7152 + f[..., 2] * 0.0722).mean())


def run(key, frames, fn, warm):
    warm()
    fn(frames[0] - 1)                 # 1 回目（numba の読み込み・地図の作成）は数えない
    TIMES.clear()
    ls, ts = [], []
    for i in frames:
        t = time.perf_counter()
        f = fn(i)
        ts.append(time.perf_counter() - t)
        ls.append(luma(f))
    TIMES.clear()
    d = np.abs(np.diff(ls))
    print('%s frames %s  mean %.2f s/frame (min %.2f)  max |Δluma| %.4f' % (key, frames, np.mean(ts), np.min(ts),
                                                                          d.max() if len(d) else 0))


keys = sys.argv[1:] or ['01', '03', '04', '05', '06', '07']
bs = {b['key']: b for b in timeline.load()}
for k in keys:
    if k == '01':
        import ld01
        from cutB.beats import b01
        T = b01.timing(bs['01'])
        ge = b01.push_end()
        sp, du = b01.particles(T)
        run(k, [44, 45, 46, 47, 48, 62, 63], lambda i: ld01.frame_at(i, T, ge, sp, du), lambda: (ld01.static_layers(),
                                                                                          ld01.fist_sprite()))
    elif k == '03':
        import ld03
        run(k, [74, 75, 76, 77], lambda i: ld03.frame_at(i, bs['03']), ld03.textures)
    elif k == '04':
        import ld04
        run(k, [70, 71, 72, 73], lambda i: ld04.frame_at(i, bs['04']), ld04.dial)
    elif k == '05':
        import ld05
        run(k, [82, 83, 84, 85], lambda i: ld05.frame_at(i, bs['05']), lambda: (ld05.background(), ld05.elephant()))
    elif k == '06':
        import ld06
        from cutB.beats import b06
        T6 = b06.timing(bs['06'])
        import ldx
        run(k, [72, 73, 74, 75], lambda i: ld06.frame_at(i, T6), ldx.static)
    elif k == '07':
        import ld07
        import ldx
        run(k, [33, 34, 35, 36, 37], lambda i: ld07.frame_at(i, bs['07']), ldx.static)
