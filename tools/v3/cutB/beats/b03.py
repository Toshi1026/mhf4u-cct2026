"""案B 03（約3.7秒）「たとえば、桂浜から殴って、じゅっキロ先まで飛ばすなら。」

土佐湾の線画（真上から・沖の線はうすく少なく）。土佐湾全体が入る引きから、桂浜のあたりへ寄る。
 0–7 ：02 からのズームの続き（02 の最後と同じ画面の点 DOT (868, 338) まわりに 1.6→1.0、expo_out、ズームのぼかし 0.22→0）。
 0–84：寄り。log z と「弧の中心が画面のどこにあるか」を同じ cubic-bezier(0.65,0,0.35,1) で動かす（目標点がまっすぐ寄ってくる）。
 カ（かつらはま）：桂浜の白い点が back_out でぽんと出て、「桂浜」が字間 0.20→0.06em で寄りながら出る。下に「桂浜から」（1文字ずつ下から）。
 ナ（なぐって）→キ（じゅっキロ）：桂浜から西へ、海岸に沿って放物線がのびる（起こした白い破線の弧）。
   進みは cubic-bezier(0.3,0.6,0.5,1)（速く打ち出してやわらかく落ちる）。先頭は点＋彗星の尾。弧が届いたところで先頭の点がぽんとふくらむ。
   カウンター「10km先へ」（10km を強調）はオドメーター：弧の進みといっしょに桁が回り、キで 10 に決まる（落ち＋拡大）。
   「0km」は見せない（弧が 6% 進むまで出さない）。
 キのあと（74–111）：弧の破線の模様が飛ぶ向きへ 1.2px/フレームで流れ、弧の中心まわりの押し込み 1.00→1.045 が続く（04 も同じ流れのまま）。
寄りきると弧の横幅は画面の幅の約 62%（_bmap.ARC_FRAC）。右上に小さな縮尺（寄りに合わせて 5km → 2km）。
2026-11-07 訂正：行き先を MAZE の前の海に（13km → 10km）。
2026-11-07 動きの仕様（P1）に合わせて作り直し。
2026-11-08 絵の作り直し（美術監督の look-dev v2）：線画の地図 → 国土地理院の標高タイルの陰影起伏＋プラン・オブリークの
  立体（_maplook）。カメラ・弧・点・文字・タイミングはそのまま。弧は 0.9 の白い破線に、地面へ落ちるやわらかい影（0.25）。
  沖の等距離線はなし（海は岸からの距離の連続した色だけ）。仕上げは R.finish_ad（文字・弧・海岸線の前）。出典を右下に。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W
from beats import _mapui as U
from beats import _geo as G
from cutB.beats import _bmap as B, _motion as M, _maplook as ML, _render as R

Y_SUB, Y_BIG = 1046, 1178
KM = '10'
DOT = (868.0, 338.0)          # 02 の押し込みの中心（ズームのつなぎの点）
ZIN = 8                       # 02 からのズームの続き（フレーム）
DASH_V = 1.2                  # 破線の模様の流れ（px/フレーム）
ARC_EASE = M.cubic_bezier(0.3, 0.6, 0.5, 1.0)
ARC_A = 0.92                  # 弧の破線の濃さ（白 0.9 前後）
GRAIN = 0.007


def timing(beat):
    mo = beat['moras']
    assert mo[4][0] == 'カ' and mo[11][0] == 'ナ' and mo[15][0] == 'ジュ' and mo[17][0] == 'キ', mo   # ジュッキロ の キ
    mf = lambda k: timeline.mora_frame(beat, k)
    f_ka, f_na, f_ki = mf(4), mf(11), mf(17)
    return dict(ka=f_ka, na=f_na, ki=f_ki, g0=f_na - 2, g1=f_ki)


_T = {}


def _timing03():
    if not _T:
        b = next(b for b in timeline.load() if b['key'] == '03')
        _T.update(timing(b))
    return _T


def dash_phase(t):
    """破線の流れ（px）。弧がのびはじめてから、03 → 04 と通しで流れ続ける。"""
    return DASH_V * max(0.0, t - _timing03()['g0'])


def arc_u(t):
    T = _timing03()
    if t <= T['g0']:
        return 0.0
    return ARC_EASE((t - T['g0']) / float(T['g1'] - T['g0']))


def land_r(t):
    """弧が届いたところの点の半径（届く瞬間に 5→10 へふくらみ、6 に落ち着く）。"""
    T = _timing03()
    k = t - T['g1']
    if k < 0:
        return 5.0
    return M.Track([(0, 5.0, 'expo_out'), (2, 10.0, 'sine_inout'), (6, 6.0)]).at(k)


def overlay_static(f, t, alpha=1.0, track_u=1.0, head=False, dot_a=1.0, label_a=1.0, dot_s=1.0, label=None,
                   head_r=None, tail=0.0):
    """地図の上の桂浜の点・ラベル・軌跡（03 の最後の状態を 04 の背景でも使う）。
    dot_s：点の大きさ（back_out の出）、label：ラベルを描く関数（None なら止まった「桂浜」）。"""
    ctr, z = B.cam(t)
    L = G.Lines()
    hr = land_r(t) if head_r is None else head_r
    B.draw_track(L, ctr, z, track_u, alpha=alpha * ARC_A, head=head or track_u >= 0.999, dash_phase=dash_phase(t),
                 head_r=hr, tail=tail)
    p = B.kat_scr(ctr, z)
    if dot_a > 0 and dot_s > 0.02:
        L.dot(p[0], p[1], 7.5 * dot_s, 255 * min(1.0, dot_a) * alpha)
    m = L.done()
    f = ML.drop_shadow(f, m, 2.0, 3.0, 2.5, 0.40 * alpha)
    f = _blend(f, m, core.FOAM)
    if label is not None:
        f = label(f, p)
    elif label_a > 0:
        f = _label().put(f, label_a * alpha, dx=p[0] + 22 - 540, dy=p[1] + 40 - 1000)
    return f


def _blend(f, m, col):
    from cutB.beats import _m34
    return _m34.blend_mask(f, m, col)


_LAB = []


def _label():
    if not _LAB:
        _LAB.append(U.Patch([dict(text='桂浜', x=540, y=1000, size=42, weight='Medium', anchor='ls', track=0.06)]))
    return _LAB[0]


_KL = []


def _klabel():
    if not _KL:
        _KL.append(M.KText([dict(text='桂浜', x=540, y=1000, size=42, weight='Medium', anchor='ls', track=0.06)]))
    return _KL[0]


def map_frame(t, a_map=1.0, haze=0.16, tilt=1.6, coast_a=0.40):
    """地図（陰影起伏の立体の地図＋海岸線、文字・弧なし。sRGB float32）。a_map は互換のため（1 のみ）。"""
    ctr, z = B.cam(t)
    f, _ = ML.map_layer(t, ctr, z, haze=haze, tilt=tilt, coast_a=coast_a)
    return f


def cg_frame(t, u, fin_i=None, scale_a=0.0):
    """地図＋弧の地面の影＋文字の後ろの暗がり → 仕上げ（finish_ad）。弧・点・文字はこの後に重ねる。"""
    ctr, z = B.cam(t)
    f = map_frame(t)
    f = ML.track_ground_shadow(f, ctr, z, u)
    f = ML.label_scrims(f, 1.0, cg=True, scale_bar=scale_a > 0, credit_a=1.0)
    return R.finish_ad(f, 157 + int(t) if fin_i is None else fin_i, grain=GRAIN, vignette=0.10, bloom_k=0.05)


def texts():
    sub = M.KText([dict(text='桂浜から', x=W / 2, y=Y_SUB, size=46, weight='Medium', track=0.08)])
    cnt = M.Odometer(KM, W / 2, Y_BIG, digit_size=108, side_size=96, prefix='', suffix='km先へ', hl_suffix='km')
    return sub, cnt


def counter_v(t):
    return min(float(KM), 10.1 * arc_u(t))          # 試算の R＝10.10km


def render(beat, ctx):
    n = beat['n']
    T = timing(beat)
    f_ka, g0, g1 = T['ka'], T['g0'], T['g1']
    sub, cnt = texts()
    lab = _klabel()
    for i in range(n):
        u = arc_u(i)
        f = cg_frame(i, u, scale_a=1.0)
        if i < ZIN:                                    # 02 の最後（暗い紺）から明るくなる（明るさの差 ≤0.05/フレーム）
            k = 0.40 * (1 - core.ease((i + 1) / float(ZIN)))
            f = f * (1 - k) + np.asarray(core.c('0A1D33'), np.float32) * k
        # 桂浜の点：「カ」の 6 フレーム前に back_out でぽんと出る。ラベルは「カ」の 2 フレーム前から字間を寄せながら
        ds = M.ease((i - (f_ka - 6) + 1) / 6.0, 'back_out')

        def label(fr, p, i=i):
            return lab.reveal(fr, i, f_ka - 2, 'track', track_from=0.20, dur=12, dx=p[0] + 22 - 540, dy=p[1] + 40 - 1000)
        f = overlay_static(f, i, 1.0, track_u=u, head=True, dot_a=1.0 if i >= f_ka - 6 else 0.0, dot_s=ds,
                           label=label, tail=0.08 if i < g1 else 0.0)
        ctr, z = B.cam(i)
        if i >= ZIN:
            f = B.scale_bar(f, z, alpha=0.9)
        # 文字の後ろをわずかに暗く（読みやすさ）
        lab_a = M.ease((i - f_ka + 2) / 6.0, 'sine_inout')
        sa = 0.42 * lab_a
        if sa > 0:
            f = U.scrim(f, Y_SUB - 60, Y_BIG + 10, strength=sa, feather=150)
        f = sub.reveal(f, i, f_ka - 2, 'char_rise')
        if u > 0.06:
            v = counter_v(i)
            a = M.ease((u - 0.06) / 0.08, 'sine_inout')
            f = cnt.put(f, v, a, i=i, f_lock=g1, dv=0.6 * (v - counter_v(i - 1)))   # ぶれは少し控えめに（読める）
        # 02 からのズームの続き
        if i < ZIN:
            f = M.zoom_through(f, i / float(ZIN), DOT, 'in', s_in=(1.6, 1.0), blur=0.22, steps=20 if i < 3 else 12)
            f = B.scale_bar(f, z, alpha=0.9 * M.ease((i + 1) / float(ZIN), 'sine_inout'))   # 縮尺は画面に固定（ズームに乗せない）
        f = ML.credit(f)
        yield np.clip(f, 0, 1).astype(np.float32)
