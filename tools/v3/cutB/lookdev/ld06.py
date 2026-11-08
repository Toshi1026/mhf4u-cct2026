"""look-dev 06：断面のジオラマ（ldx）＋ガラスの富士山＋ 3D の悪役（_figures3d）＋仕上げ（_render.finish）。
タイミング・カメラ・線・文字は b06 のまま（b06 の関数を呼ぶ）。
python3 cutB/lookdev/ld06.py [frames...]   → exports/v3/lookdev_B/b06_fNNNN_{after,before,sbs}.png"""
import sys

from _common import np, cv2, core, tick, report, save_pair, W, H  # noqa: F401
from cutB.beats import _render as R, _figures3d as F3, _art as A, _xsec as X, b06, _motion as M
from beats import _mapui as U
import timeline_b as timeline
import ldx

LIGHT = (-0.62, -0.55, 0.45)
ENV = dict(sky=R.hexl('9CCBEA') * 0.60, ground=R.hexl('2F6FA6') * 0.40, sun=np.array([1.0, 0.99, 0.975]) * 1.3,
           rim=R.hexl('E4F2FC') * 1.0)


def villain_rgba(x, y, scale, ang=0.0, pose='fly', t=0.0, squash=1.0, M2=None, cape_ang=None, s=4):
    return F3.villain_rgba(x, y, scale, ang, pose, t, squash, M2, cape_ang, light=LIGHT, env=ENV, ss=2)


A.villain_rgba = villain_rgba


def scene(i, T, fuji_a=None, fuji=None, t=None):
    """b06.scene の代わり：ジオラマの背景＋ガラスの富士山＋輪郭の線（stroke は 06 のまま）。"""
    f = ldx.background(i / core.FPS if t is None else t)
    st, fl = b06.fuji_state(i, T) if fuji is None else fuji
    if fuji_a is not None:
        fl *= fuji_a
    if st > 0.003 and (fuji_a is None or fuji_a > 0.003):
        with tick('fuji'):
            f = ldx.fuji_glass(fl, f)
            rgba, x0, y0 = X.fuji_layer2(st, 0.0)
            f = U.paste(f, rgba, x0, y0, 1.0 if fuji_a is None else fuji_a)
    return f


def frame_at(i, T):
    with tick('scene'):
        f = scene(i, T)
    with tick('lines+figure'):
        u = b06.villain_u(i, T)
        rgba, x0, y0 = X.trail_layer(u, ghost=M.ease((i + 1) / 10.0, 'sine_inout'))
        f = U.paste(f, rgba, x0, y0)
        a_ap = float(np.clip((i - T['ru'] + 6) / 10.0, 0, 1))
        f = b06.apex_lines(f, a_ap)
        f = b06.sight_line(f, float(np.clip((i - T['shi'] + 1) / 12.0, 0, 1)) * (1 - 0.5 * M.ease((i - T['ki']) / 12.0)),
                           phase=-0.8 * i)
        f = b06.put_villain(f, i, T)
        f = A.put_glint_fx(f, X.APEX[0], X.APEX[1] - 2, b06.glint_k(i, T), r=42)
    with tick('camera'):
        Mc = b06.cam_M(i, T)
        f = b06.warp(f, Mc)
    with tick('finish'):
        f = R.finish(f, i, bloom_k=0.10, bloom_thr=0.74, vignette=0.16, grain=0.009, ca_px=0.45, distort=0.4)
    with tick('text'):
        st, _fl = b06.fuji_state(i, T)
        f = b06.put_labels(f, i, T, Mc, fuji_a=M.ease((i - T['ru'] - 9) / 6.0, 'sine_inout') if st > 0 else 0.0,
                           apex_a=M.ease((i - T['ru'] + 1) / 6.0, 'sine_inout'))
        f = b06.put_counter(f, i, T)
        _ctr, t_sub, _ = b06.texts()
        f = t_sub.reveal(f, i, T['ru'] + 4, 'char_rise', stagger=0.8, dur=10)
    return np.clip(f, 0, 1).astype(np.float32)


def main(frames):
    beat = next(b for b in timeline.load() if b['key'] == '06')
    T = b06.timing(beat)
    ldx.static()
    report('06 static')
    for i in frames:
        out = frame_at(i, T)
        report('06 f%d' % i)
        save_pair('06', i, out)


if __name__ == '__main__':
    main([int(v) for v in sys.argv[1:]] or [74])
