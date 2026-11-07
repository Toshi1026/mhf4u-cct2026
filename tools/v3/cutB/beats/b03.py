"""案B 03（約3.7秒）「たとえば、桂浜から殴って、じゅっキロ先まで飛ばすなら。」

土佐湾の線画（真上から・沖の線はうすく少なく）。土佐湾全体が入る引きから、桂浜のあたりへゆっくり寄る。
 カ（かつらはま）：桂浜に白い点と「桂浜」、下に「桂浜から」。
 ナ（なぐって）→キ（じゅっキロ）：桂浜から西へ、海岸に沿って放物線がのびる（起こした白い破線の弧）。
   カウンター「0km先へ」→「10km先へ」（10km を強調）が弧の伸びと同じ速さで数える。行き先には点も名前も出さない。
寄りきると弧の横幅は画面の幅の約 62%（_bmap.ARC_FRAC）。
右上に小さな縮尺（寄りに合わせて長さが変わり、途中で 5km → 2km に溶ける）。最後まで線は残し、04 は同じ地図を暗くして始まる。
2026-11-07 訂正：行き先を MAZE の前の海に（13km → 10km）。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W
from beats import _mapui as U
from beats import _geo as G
from cutB.beats import _bmap as B

Y_SUB, Y_BIG = 1046, 1178
KM = '10'


def overlay_static(f, t, alpha=1.0, track_u=1.0, head=False, dot_a=1.0, label_a=1.0):
    """地図の上の桂浜の点・ラベル・軌跡（03 の最後の状態を 04 の背景でも使う）。"""
    ctr, z = B.cam(t)
    L = G.Lines()
    B.draw_track(L, ctr, z, track_u, alpha=alpha, head=head)
    p = B.kat_scr(ctr, z)
    if dot_a > 0:
        L.dot(p[0], p[1], 7.5 * (0.6 + 0.4 * dot_a), 255 * dot_a * alpha)
    m = L.done()[..., None]
    f = f * (1 - m) + core.FOAM * m
    if label_a > 0:
        f = _label().put(f, label_a * alpha, dx=p[0] + 22 - 540, dy=p[1] + 40 - 1000)
    return f


_LAB = []


def _label():
    if not _LAB:
        _LAB.append(U.Patch([dict(text='桂浜', x=540, y=1000, size=42, weight='Medium', anchor='ls', track=0.06)]))
    return _LAB[0]


def map_frame(t, a_map=1.0):
    S = B.setup()
    f = S['bg'].copy()
    la = B.land_mask(t)[..., None] * (0.55 * a_map)
    f = f * (1 - la) + B.LAND_TINT * la
    m = B.map_mask(t)[..., None] * a_map
    return f * (1 - m) + core.FOAM * m


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[4][0] == 'カ' and mo[11][0] == 'ナ' and mo[15][0] == 'ジュ' and mo[17][0] == 'キ', mo   # ジュッキロ の キ
    mf = lambda k: timeline.mora_frame(beat, k)
    f_ka, f_na, f_ki = mf(4), mf(11), mf(17)
    sub = U.Patch([dict(text='桂浜から', x=W / 2, y=Y_SUB, size=46, weight='Medium', track=0.08)])
    cnt = B.HCounter(KM, W / 2, Y_BIG, hl_suffix='km', digit_size=108, side_size=96, prefix='', suffix='km先へ')
    g0, g1 = f_na - 2, f_ki                        # 弧が伸びる区間
    for i in range(n):
        a_map = core.ease((i + 1) / 9.0)
        f = map_frame(i, a_map)
        # 点は「カ」の少し前に、ラベルと「桂浜から」は「カ」で
        dot_a = core.ease((i - (f_ka - 6) + 1) / 6.0, 'out')
        lab_a = core.ease((i - f_ka + 1) / 6.0)
        u = core.ease((i - g0) / float(g1 - g0), 'inout') if i > g0 else 0.0
        f = overlay_static(f, i, 1.0, track_u=u, head=True, dot_a=dot_a, label_a=lab_a)
        ctr, z = B.cam(i)
        f = B.scale_bar(f, z, alpha=a_map * 0.9)
        # 文字の後ろをわずかに暗く（読みやすさ）
        sa = 0.42 * lab_a
        if sa > 0:
            f = U.scrim(f, Y_SUB - 60, Y_BIG + 10, strength=sa, feather=150)
        f = sub.put(f, lab_a)
        if i >= g0 + 1:
            val = min(int(KM), int(np.floor(10.1 * u + 1e-6)))       # 試算の R＝10.10km
            f = cnt.put(f, str(val), core.ease((i - g0) / 4.0))
        yield np.clip(f, 0, 1).astype(np.float32)
