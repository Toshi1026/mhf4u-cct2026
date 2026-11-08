"""案C の音 → exports/v3/mix_C.wav（案B の audio_b.py の作り方）。tools/v3 から python3 cutC/audio_c.py

ナレーション（vo_C）＋悪役の悲鳴（C10）＋ BGM（cutC/music_c.py：コードで作る）＋効果音（効果音ラボ：商用可・表記任意）
＋実写パートの波の音（IMG_9662 の現地の音 → 効果音ラボ 海岸4）。

効果音の時刻はビートのタイミング関数（c01.timing・c03.timing・c04/c05.timing・c06.timing・c07.alt・bT.SPL …）から決める。
置き方：hit／tick は音の立ち上がり（ピークの -30 dB を越える所の 5ms 前）を、whoosh はいちばん大きい所を、その画のフレームへ。
効果音の候補は earshot rank / find で選んだ（cue 表の z、2026-11-08）。cue 表は python3 cutC/audio_c.py --cues で出る
（exports/v3/cues_C.json：earshot timeline --cues 用）。

悲鳴：既定は vo_C/scream_client.wav（クライアント本人が演じた録音、2026-11-08 確認）。予備は vo_C/scream.wav（AivisSpeech TANAKA）。環境変数 MAZE_SCREAM に別の wav を渡すと差し替え
（例：vo_C/scream_client.wav。権利の確認が済んでから）。10 の頭＋0.1 秒から、着水のフレームで切れる（しぶきに飲まれる）。
遠い → 近い：-13 dB → 0 dB、こもり（ローパス 2.2k → 14k Hz）、ドップラーで音程がわずかに上がる（+1.6 半音）、風のゆらぎ。

音量：ミックスを -14 LUFS（pyloudnorm）にそろえ、トゥルーピーク -1.5 dBTP 以下（4倍オーバーサンプルのピークで制限）。
build_c.py の loudnorm（-14 LUFS）はほとんど動かさない。
"""
import json
import math
import os
import subprocess
import sys
import wave

import numpy as np
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio as A          # noqa: E402  案A の部品（read_wav, decode, add, db, sub_hit）
import core                # noqa: E402
import timeline_c as TL    # noqa: E402
import music_c as MU       # noqa: E402

SR = A.SR
FPS = core.FPS
LAB = os.path.join(core.OUT, 'sfx', 'lab')
SCREAM = os.environ.get('MAZE_SCREAM') or TL.SCREAM
RNG = np.random.default_rng(1108)
_CACHE = {}


# ------------------------------------------------------------------ 素材
def dur_of(p):
    return float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p],
                                capture_output=True, text=True).stdout)


def lab(rel, t0=0.0, t1=None, fi=0.003, fo=0.02):
    """効果音ラボの1本（または区間）。DC を抜き、ピーク 1 にそろえ、頭と尻に短いフェード（プチ音を防ぐ）。"""
    key = (rel, t0, t1)
    if key not in _CACHE:
        p = os.path.join(LAB, rel)
        x = A.decode(p, t0, dur_of(p) if t1 is None else t1)
        x = x - x.mean()
        x = x / max(1e-6, np.abs(x).max())
        _CACHE[key] = MU.fade(x, fi, fo)
    return _CACHE[key].copy()


def env(x, ms=5):
    k = max(1, int(ms * SR / 1000))
    return np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(k) / k, 'same'))


def onset(x):
    e = env(x, 2)
    i = int(np.argmax(e > e.max() * 10 ** (-30 / 20)))
    return max(0, i - int(0.005 * SR)) / SR


def peak(x):
    return int(np.argmax(env(x, 20))) / SR


def place(buf, x, t, g_db=0.0, anchor='onset'):
    """x の anchor（onset／peak／秒）を t に合わせて buf に足す。"""
    a = onset(x) if anchor == 'onset' else peak(x) if anchor == 'peak' else float(anchor)
    MU.add(buf, x, t - a, A.db(g_db))
    return t - a


def pitch(x, semis):
    r = 2 ** (semis / 12.0)
    n = int(len(x) / r)
    return np.interp(np.arange(n) * r, np.arange(len(x)), x).astype(np.float32)


# ------------------------------------------------------------------ 画のイベント（秒）
def events(bs):
    from cutB.beats import _motion as M, bT
    from cutC.beats import c01, c03, c04, c05, c06, c07, c10, c14, c16
    B = {b['key']: b for b in bs}
    ft = lambda k, i: (B[k]['f0'] + i) / FPS
    E = {}
    # C01（2026-11-08 修正）：拳がためる →「一発」の「パ」で当たり（2 フレームのヒットストップ）→ 3 回転して空へ
    # → 声が終わって 0.2 秒あとにキラーン（kira2：効果音だけ。ナレーションには入っていない）→ 拳のはずみ 2 回（飛んでいる間）
    T1 = c01.timing(B['01'])
    E['c01_hit'] = ft('01', T1['hit'])
    E['c01_ki'] = ft('01', T1['ki'])                    # 光がいちばん大きいフレーム（kira2 の立ち上がりをここへ）
    E['c01_wind'] = ft('01', T1['wind'])                # 「パンチ」で拳を引く
    f0, f1 = T1['hit'] + c01.HOLD, T1['glint']
    spins = []
    for j in range(1, 4):                                  # 回転の角度が 360° の倍数を越えるフレーム（サブフレーム）
        target = -14 + 360 * j - 180                      # 半回転の所（体が横を向いて風を切る）
        for i10 in range(f0 * 10, f1 * 10 + 1):
            i = i10 / 10.0
            ang = -14 + 1080 * M.ease(min(1.0, max(0.0, (i - f0) / float(f1 - f0))), 'cubic_out')
            if ang >= target:
                spins.append(ft('01', i))
                break
    E['c01_spins'] = spins
    E['c01_hops'] = [ft('01', k) for k in T1['hops']]
    E['c01_end'] = B['02']['f0'] / FPS
    # C02：止まる（レコードを止めた音）。03 の声「でも、現実だと……」は 02 の頭＋LEAD02 から（B['03']['vo']）
    E['c02'] = B['02']['f0'] / FPS
    E['c03_vo'] = B['03']['vo']
    # C03：当たり（コ−2）→ 殻がカメラに届く（wu0−2）→ 上へ振る（wu0..wb0）→ 雲の穴（wb0+3）→ 下へ振る（wb1）→ クレーター
    T3 = c03.timing(B['03'])
    E['c03_0'] = B['03']['f0'] / FPS
    E['c03_hit'] = ft('03', T3['hit'])
    E['c03_launch'] = ft('03', T3['launch'])            # 地面が裂けて土のカーテンが噴き上がる（2 フレーム実時間 → スロー）
    E['c03_slow'] = (ft('03', T3['launch'] + 2), ft('03', T3['launch'] + 9))   # 約 0.3 倍のスロー（土が空中に止まる）→ 加速
    # ショット C（03 の引き〜05）：大きなかたまり（> 3 m）が落ちる時刻（_cfx の揺れと同じ chunks_C）
    from cutC.beats import _cfx as CF
    a3 = CF.anchors()
    ch = CF.chunks_C()
    E['rock_lands'] = sorted(float((a3[1] + t) / FPS) for t, s in zip(ch.t_land, ch.size)
                             if s > 3.0 and t >= CF.g_c0())
    E['rain0'] = (a3[1] + CF.g_c0() - 2) / FPS          # 引きの画の 2 フレーム前から岩が降る
    E['c03_shell'] = ft('03', T3['wu0'] - 2)
    E['c03_whipup'] = ft('03', (T3['wu0'] + T3['wb0']) / 2.0)
    E['c03_hole'] = ft('03', T3['wb0'] + 3)
    E['c03_whipdn'] = ft('03', T3['wb1'])
    E['c03_crater'] = ft('03', T3['wc0'])
    # 地割れ（2026-11-08 修正 4）：1 段目はショット C の頭＋3 で縁から一気に走る、2 段目は 04 の「1,400」が止まるフレーム
    G1, G2 = CF.crack_times()
    E['crack1'] = (a3[1] + G1) / FPS
    E['crack2'] = (a3[1] + G2) / FPS
    # C04：数字（約1,400トン）が回る → 「ト」で止まる
    T4 = c04.timing(B['04'])
    E['c04_ticks'] = _ticks('04', B, lambda i: c04.value(i, T4), range(T4['ya'] - 1, T4['to']), lambda v: v >= 8, 100.0)
    E['c04_lock'] = ft('04', T4['to'])
    # C05：ゾウ 250 頭が降り立つ（手前のまんなかから外へ）→ 「ト」で止まる
    T5 = c05.timing(B['05'])
    a, b = T5['zo'] + c05.DROP - 1, T5['to']
    E['c05_land'] = (ft('05', a), ft('05', b))
    cnt = lambda i: 250.0 * float(np.clip((i - a) / float(max(1, b - a)), 0, 1))
    E['c05_ticks'] = _ticks('05', B, cnt, range(T5['zo'], T5['to']), lambda v: v >= 1, 10.0)
    E['c05_lock'] = ft('05', T5['to'])
    # C06：振りで入る → 「マ」で針が走る → ピンに当たる → 折れる（ガラスが割れる）→ 「ナ」で 6.7 に止まる
    T6 = c06.timing(B['06'])
    E['c06_0'] = B['06']['f0'] / FPS
    E['c06_go'] = ft('06', T6['go'])
    E['c06_peg'] = ft('06', T6['peg'])
    E['c06_snap'] = ft('06', T6['snap'])
    E['c06_lock'] = ft('06', T6['lock'])
    E['c06_ticks'] = _ticks('06', B, lambda i: c06.counter_v(i, T6), range(T6['go'], T6['lock']), lambda v: v >= 1, 5.0)
    # C07：高さ 0.1→132km（雲の層・10km の旅客機・100km の線）→ 「キ」で止まる → 下へ振って C08 へ
    b7 = B['07']
    alt = lambda i: c07.alt(i, b7)
    cross = lambda h: next((ft('07', i) for i in range(b7['n']) if alt(i) >= h), None)
    E['c07_cloud'] = cross(1.2)
    E['c07_jet'] = cross(10.0)
    E['c07_space'] = cross(100.0)
    E['c07_lock'] = ft('07', c07.f_lock(b7))
    E['c07_alt'] = [(ft('07', i), alt(i)) for i in range(b7['n'])]
    E['c07_whip'] = ft('07', b7['n'] - c07.N_X // 2)
    # C08：地球が見える（下へ 16 フレーム振る）、「ウ」で宇宙
    E['c08_0'] = B['08']['f0'] / FPS
    E['c08_reveal'] = ft('08', 16)
    E['c08_u'] = ft('08', TL.mora_frame(B['08'], 0))
    # C09・C10：k は C10 の頭から。−40 で頭が下へ（ばね）、−30 から落ちはじめる。着水は T の SPL[0]
    E['c09_flip'] = ft('10', -40)
    E['c09_fall'] = ft('10', -30)
    E['c10_0'] = B['10']['f0'] / FPS
    E['c10_scream'] = B['10']['scream']
    E['c10_splash'] = ft('10', c10.NZ_SPEC + bT.SPL[0])
    E['c10_end'] = B['10']['f1'] / FPS
    # C11〜C16
    E['c11_0'] = B['11']['f0'] / FPS
    E['c11_cut'] = ft('11', _mf(B['11'], 12, 'ニ'))
    E['c12_cut'] = ft('12', _mf(B['12'], 5, 'シ') - 2)
    E['c13_0'] = B['13']['f0'] / FPS                       # 12 → 13：上への振り（12 の最後 4・13 の最初 4 フレーム）
    E['c14_0'] = B['14']['f0'] / FPS
    R = B['14']['n'] - c14.PER * c14.NCUT
    E['c14_cuts'] = [ft('14', R + c14.PER * j) for j in range(c14.NCUT)]
    E['c15_0'] = B['15']['f0'] / FPS
    E['c16_0'] = B['16']['f0'] / FPS
    E['c16_logo'] = ft('16', c16.LOGO0)
    E['end'] = bs[-1]['f1'] / FPS
    return E


def _mf(beat, k, m):
    assert beat['moras'][k][0] == m, (beat['key'], beat['moras'][k], m)
    return TL.mora_frame(beat, k)


def _ticks(key, B, val, frames, show, step=1.0):
    """表示の値が step の倍数を越えたフレームごとに 1 つ（1 フレームに 1 つまで。回りが遅くなるとカチカチも遅くなる）。"""
    out, last = [], None
    for i in frames:
        v = val(i)
        if not show(v):
            continue
        iv = int(math.floor(v / step + 1e-6))
        if last is not None and iv != last:
            out.append((B[key]['f0'] + i) / FPS)
        last = iv
    return out


# ------------------------------------------------------------------ 効果音
SFX = {   # 役 → (ファイル, 区間, earshot の読み)。区間は earshot find のトリム
    'bonk': ('cartoon/punch-middle2.mp3', (0.0, 0.37), 'rank bonk z+3.63 / find z+2.71'),
    'bonk2': ('cartoon/pico-pico-hammer1.mp3', (0.0, 0.24), 'find z+2.07（重ねる軽い層）'),
    'swish': ('whoosh/sword-gesture2.mp3', (0.08, 0.40), 'rank soft swish z+4.01 / find z+4.27'),
    'boing': ('cartoon/boyon1.mp3', (0.0, 0.53), 'find boing z+5.47'),
    'kira': ('chimes/kira2.mp3', (0.0, 1.0), '案B と同じ キラッ2（kira2）'),
    'scratch': ('cartoon/dj-scratch1.mp3', (0.0, 0.54), 'find record scratch z+4.68'),
    'slowmo': ('whoosh/punch-slow2.mp3', (0.0, 1.41), 'rank slowmo z+2.61'),
    'punch': ('impact/punch-heavy3.mp3', (0.0, 0.87), 'rank heavy punch z+3.23 / find z+4.18'),
    'boom': ('impact/big-explosion2.mp3', (0.18, 2.18), 'rank cinematic boom z+5.04'),
    'shock': ('whoosh/magic-wind2.mp3', (0.30, 2.12), 'rank shockwave z+5.03 / find z+4.43'),
    'debris': ('debris/cliff-stone-falls1.mp3', (1.95, 3.96), 'rank debris z+4.14 / find z+3.81'),
    'debris2': ('debris/rock-break1.mp3', (2.5, 3.5), 'rank debris z+4.52 / find z+4.13'),
    'erupt': ('debris/rock-break1.mp3', (0.011, 1.832), 'find ground impact, earth and rocks bursting z+4.91（地面が裂けて噴き上がる）'),
    'quake': ('impact/bomb4.mp3', (0.776, 2.239), 'find deep low ground explosion z+4.36（低い地鳴りの層）'),
    'soil': ('debris/dash-soil1.mp3', (2.697, 3.225), 'find soil spraying, a burst of earth z+3.64（土が飛び散る）'),
    'rock': ('debris/cliff-stone-falls1.mp3', (1.972, 2.157), 'find single heavy rock thudding onto dirt z+3.39'),
    'rock2': ('debris/cliff-stone-falls1.mp3', (0.381, 0.602), 'find single heavy rock thudding onto dirt z+3.03'),
    'whip': ('whoosh/whip-gesture1.mp3', (0.04, 0.29), 'rank soft swish z+3.24 / find z+4.17'),
    'tick': ('ticks/clock-hand1.mp3', (9.975, 10.22), 'find single tick z+4.49'),
    'clank': ('glass/metal1.mp3', (0.0, 0.59), 'rank metal clank z+3.89'),
    'glass': ('glass/glass-break3.mp3', (0.0, 0.59), 'rank glass z+5.86 / find z+6.67'),
    'jet': ('ambience/airliner-pass2.mp3', (0.0, 2.86), 'rank airliner z+3.82'),
    'wind': ('ambience/wind2.mp3', (0.0, 5.0), 'rank wind roar z+4.41 / find z+4.69'),
    'roar': ('whoosh/wizard-flight1.mp3', (12.6, 16.9), 'rank wind roar z+4.30 / find z+4.77'),
    'splash': ('water/water-throw-stone2.mp3', (0.03, 1.09), 'rank splash z+4.38 / find z+3.70（頭の小さなプチッを外す）'),
    'spray': ('water/sprinkle-water1.mp3', None, 'rank splash z+2.73（しぶきの層）'),
    'drop': ('water/water-drop2.mp3', None, '水滴2'),
    'tin': ('chimes/tin1.mp3', None, 'ロゴの小さなチン'),
    'tremor': ('debris/earth-tremor1.mp3', (2.7, 4.5), 'rank deep earth rumble, fissure opening z+5.37 / find z+3.61（地響き）'),
    'crackle': ('debris/rock-crack1.mp3', (0.011, 1.983), 'rank ground cracking z+4.51 / find z+3.56（岩にヒビが入る）'),
}


def S(role):
    rel, rng, _ = SFX[role]
    return lab(rel, *(rng or (0.0, None)))


def crack_run(d, tre=(2.7, 4.5), crk=(0.011, 1.983), g_crk=1.0, hp_crk=800, swell=0.08):
    """地割れが走る音：地響き（ローパス 3.5k）＋ヒビ（ハイパス 800 Hz）。swell 秒で立ち上がり、d の 35% から落ちる。"""
    n = int(d * SR)
    r = lab(SFX['tremor'][0], *tre)[:n]
    c = MU.hp(lab(SFX['crackle'][0], *crk), hp_crk)[:n]
    y = np.zeros(n, np.float32)
    y[:len(r)] += MU.lp(r, 3500)
    y[:len(c)] += c * g_crk
    t = np.arange(n) / SR
    e = np.minimum(1, t / swell) * np.clip(1 - (t - 0.35 * d) / (0.65 * d), 0, 1) ** 1.5
    y = y * e
    return MU.fade(y / max(1e-6, np.abs(y).max()), 0.005, 0.05).astype(np.float32)


def thump(vel=1.0, f0=70, f1=38, dur=0.35):
    t = np.arange(int(dur * SR)) / SR
    f = f1 + (f0 - f1) * np.exp(-t * 22)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9)
    x += MU.lp(RNG.standard_normal(len(t)), 900) * np.exp(-t * 40) * 0.25
    return MU.fade(x * vel, 0.002, 0.04)


def zip_up(dur, f0=280, f1=2400):
    """針が走る音：上がるサイン＋薄いノイズ（短い）。"""
    t = np.arange(int(dur * SR)) / SR
    q = t / t[-1]
    f = f0 * (f1 / f0) ** (q ** 1.6)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.6 + MU.bp(RNG.standard_normal(len(t)), 1500, 7000) * 0.25
    return MU.fade(x * (0.2 + 0.8 * q ** 1.2), 0.005, 0.01)


def sfx_track(n, E):
    sfx = np.zeros(n, np.float32)
    cues = []

    def cue(name, role, t, g, anchor='onset', x=None):
        x = S(role) if x is None else x
        s = place(sfx, x, t, g, anchor)
        cues.append(dict(name=name, t=round(t, 3), start=round(s, 3), role=role, gain_db=g,
                         src=SFX.get(role, ('code', None, ''))[0]))

    # C01 アニメ：ポコッ（2 層）・3 回転のヒュッ（遠ざかるほど小さく高く）・キラーン・拳のボヨン
    cue('c01 bonk', 'bonk', E['c01_hit'], -2)
    cue('c01 bonk (toy layer)', 'bonk2', E['c01_hit'], -9)
    for j, t in enumerate(E['c01_spins']):
        cue('c01 spin %d' % (j + 1), 'swish', t, -6 - 4 * j, 'peak', pitch(S('swish'), 2 + 2 * j))
    cue('c01 glint (kira2)', 'kira', E['c01_ki'], -3)
    for j, t in enumerate(E['c01_hops']):
        cue('c01 fist hop %d' % (j + 1), 'boing', t, -12 - 2 * j, 'onset', pitch(S('boing'), 3 + 2 * j))
    # C02 レコードを止めた音（音楽は music_c.anime の tape_stop で止まる）
    cue('c02 record scratch', 'scratch', E['c02'] - 0.03, -3)
    # C03 現実：スローの風切り（いちばん大きい所を当たりに）→ 当たり（パンチ＋爆発の頭）→ 衝撃の殻 → 土と石 → 振り・雲の穴
    cue('c03 slow-mo fist whoosh', 'slowmo', E['c03_hit'] - 0.03, -6, 'peak')
    cue('c03 impact punch', 'punch', E['c03_hit'], -1)
    boom = S('boom')
    boom = MU.lp(boom, 2600, 2)                          # 火のパチパチ（高い所）を落とす
    k = int(1.2 * SR)
    boom = boom[:k] * np.concatenate([np.ones(int(0.35 * SR)), np.linspace(1, 0, k - int(0.35 * SR)) ** 2])
    cue('c03 impact boom (short, low-passed)', 'boom', E['c03_hit'], -2, 'onset', boom.astype(np.float32))
    # 2026-11-08 クレーターの迫力：地鳴りの低い層（当たり）→ 地面が裂けて土と岩が噴き上がる（launch）→ スローの間は
    # 低く引き伸ばした崩れの音（土が空中に止まる）→ 衝撃の殻 → 引きの画で大きな岩が 1 つずつ落ちる（_cfx.thump と同じ時刻）
    # 地鳴りと噴き上がりは引きの画（c03_crater）の前に消す：尾が残ると岩の雨が「こもった水っぽい音」に埋もれる（earshot）
    tail = lambda x, d: MU.fade(x[:int(d * SR)], 0.003, 0.35)
    cue('c03 ground heave (low layer)', 'quake', E['c03_hit'], -4, 'onset',
        tail(MU.lp(S('quake'), 420, 2), E['c03_crater'] - E['c03_hit'] - 0.15))
    cue('c03 earth bursts up', 'erupt', E['c03_launch'], -5, 'onset',
        tail(MU.lp(S('erupt'), 6000), E['c03_crater'] - E['c03_launch'] - 0.1))
    cue('c03 soil spray', 'soil', E['c03_launch'] + 0.02, -12)
    s0, s1 = E['c03_slow']
    hang = MU.lp(pitch(S('debris2'), -7), 2400)          # 約 1.5 倍に伸びる：スローで土が宙に止まる間の重い崩れ
    hang = MU.fade(hang[:int((s1 - s0 + 0.35) * SR)], 0.12, 0.25)
    cue('c03 slow-mo debris hang (pitched -7)', 'debris2', s0, -8, 0.0, hang)
    sh = S('shock')                                     # 殻の風は引きの画（c03_crater）で消す（岩の雨を風で隠さない）
    sh = MU.fade(sh[:int((peak(sh) + E['c03_crater'] + 0.1 - E['c03_shell']) * SR)], 0.003, 0.3)
    cue('c03 shockwave air blast', 'shock', E['c03_shell'], -3, 'peak', sh)
    deb = MU.hp(S('debris'), 250)
    cue('c03 debris patter', 'debris', E['c03_hit'] + 0.12, -12, 0.0, deb)
    cue('c03 whip up', 'whip', E['c03_whipup'], -5, 'peak')
    hole = MU.lp(S('boom'), 900, 2)[:int(1.0 * SR)] * np.linspace(1, 0, int(1.0 * SR)) ** 2
    cue('c03 cloud hole (distant thump)', 'boom', E['c03_hole'], -5, 'onset', hole.astype(np.float32))
    ha = pitch(S('shock'), 3)
    ha = MU.fade(ha[:int((peak(ha) + E['c03_crater'] + 0.05 - E['c03_hole'] - 0.1) * SR)], 0.003, 0.25)
    cue('c03 cloud hole air', 'shock', E['c03_hole'] + 0.1, -10, 'peak', ha)
    cue('c03 whip down', 'whip', E['c03_whipdn'], -5, 'peak', pitch(S('whip'), -3))
    cue('c03 crater dust settle', 'debris2', E['c03_crater'] + 0.05, -11, 0.0, MU.lp(S('debris2'), 3000))
    # 地割れが走る（2026-11-08 修正 4）：地響き（earth-tremor1）の低いうなり＋岩にヒビが入る音（rock-crack1、800 Hz より上の
    # パチパチだけ）を、割れ目が縁から走り出すフレーム（crack1・crack2）に立ち上がり 80ms で重ねる。走りが遅くなるのに合わせて減衰。
    # earshot match「earth splitting with a deep rumble and cracking rock」z+4.54（1 段目）/ z+3.67（2 段目）、p_real 0.985 / 0.967
    cue('c03 cracks race out (stage 1: tremor + rock crack)', 'tremor', E['crack1'], -9, 0.0, crack_run(1.4))
    cue('c04 cracks extend on 1,400 lock (stage 2)', 'tremor', E['crack2'], -7, 0.0,
        crack_run(0.9, tre=(9.0, 10.0), crk=(0.4, 1.4)))
    # 引きの画：岩の雨（崖の石が落ちる音そのまま・2 秒）＋ 大きな岩（> 3 m）の着地 1 つずつ（後ほど小さく、こもらせる）。
    # earshot（単体）：「heavy rocks and dirt clods falling and thudding onto the ground」z+5.24 / 「a rockfall」z+5.34。
    # 雨を thump の強さで揺らす版は声の下で「squishy wet squelch」と聞こえた（z+3.0）ので、揺らさない
    rain = MU.fade(MU.hp(S('debris'), 150), 0.05, 0.4)
    cue('c03-05 debris rain (cliff stones falling)', 'debris', E['rain0'], -11, 0.0, rain)
    t_last = -1.0
    for j, t in enumerate(E['rock_lands']):
        if t - t_last < 0.09:
            continue
        t_last = t
        q = (t - E['rock_lands'][0]) / max(0.1, E['rock_lands'][-1] - E['rock_lands'][0])
        role = 'rock' if j % 2 == 0 else 'rock2'
        x = MU.lp(pitch(S(role), -2 * (j % 3)), 5000 - 2000 * q)
        cue('c03-05 big rock lands %d' % (j + 1), role, t, round(-7 - 6 * q, 1), 'onset', x)
    # C04・C05 数字のカチカチ（小さく）と止まる音
    for t in E['c04_ticks']:
        cue('c04 tick', 'tick', t, -21)
    cue('c04 lock', 'tick', E['c04_lock'], -12, 'onset', pitch(S('tick'), -5))
    a, b = E['c05_land']
    m = 28
    for j in range(m):                                    # 250 頭のドスン：手前から奥へ（だんだん小さく・低く）
        q = j / float(m - 1)
        t = a + (b - a) * q + RNG.uniform(-0.012, 0.012)
        x = thump(1.0, 75 - 20 * q, 40 - 6 * q, 0.3)
        MU.add(sfx, x, t, A.db(-11 - 9 * q))
    cues.append(dict(name='c05 elephant thuds x%d (code)' % m, t=round(a, 3), start=round(a, 3), role='thump',
                     gain_db=-11, src='code'))
    for t in E['c05_ticks']:
        cue('c05 tick', 'tick', t, -22)
    cue('c05 lock', 'tick', E['c05_lock'], -12, 'onset', pitch(S('tick'), -5))
    # C06 メーター：振りで入る・針が走る・ピンに当たる・折れる（ガラス）・止まる
    cue('c06 whip in', 'whip', E['c06_0'] + 4 / FPS, -9, 'peak')
    z = zip_up(E['c06_peg'] - E['c06_go'])
    MU.add(sfx, z, E['c06_go'], A.db(-17))
    cues.append(dict(name='c06 needle run (code)', t=round(E['c06_go'], 3), start=round(E['c06_go'], 3), role='zip',
                     gain_db=-17, src='code'))
    cue('c06 needle hits stop pin', 'clank', E['c06_peg'], -8)
    cue('c06 needle snaps (glass)', 'glass', E['c06_snap'], -4)
    cue('c06 snap metal', 'clank', E['c06_snap'], -12, 'onset', pitch(S('clank'), 5))
    for t in E['c06_ticks']:
        cue('c06 tick', 'tick', t, -23)
    cue('c06 lock', 'tick', E['c06_lock'], -12, 'onset', pitch(S('tick'), -5))
    # C07 上昇：雲の層を抜ける・旅客機・止まる・下へ振る
    cue('c07 through cloud layer', 'shock', E['c07_cloud'], -15, 'peak', pitch(S('shock'), 2))
    jet = S('jet')
    pk = int(np.argmax(env(jet, 50)))
    jet = pitch(jet[max(0, pk - int(0.9 * SR)):pk + int(0.9 * SR)], 4)
    jet = MU.fade(jet, 0.25, 0.4)
    cue('c07 airliner pass', 'jet', E['c07_jet'], -16, 'peak', jet)
    cue('c07 lock', 'tick', E['c07_lock'], -13, 'onset', pitch(S('tick'), -5))
    cue('c07 whip to space', 'whip', E['c07_whip'], -11, 'peak', pitch(S('whip'), -2))
    # C09 頭が下へ（ばね）
    cue('c09 flip', 'swish', E['c09_flip'] + 4 / FPS, -14, 'peak', pitch(S('swish'), -2))
    # C10 着水（しぶきに悲鳴が飲まれる）
    cue('c10 splash', 'splash', E['c10_splash'], 3, 0.04)          # 本体の立ち上がり（トリムの 40ms）を着水のフレームへ
    cue('c10 splash spray', 'spray', E['c10_splash'] + 0.03, -4)
    MU.add(sfx, thump(1.0, 90, 45, 0.45), E['c10_splash'], A.db(-2))
    # C12→C13、C13→C14 の上への振り。C14 の 3 カット（ローソン・産直・タワー）はやわらかい風
    soft = MU.lp(S('swish'), 5000)
    cue('c13 whip up (from c12)', 'swish', E['c13_0'], -14, 'peak', soft)
    cue('c14 whip up (from c13)', 'swish', E['c14_0'], -14, 'peak', soft)
    for j, t in enumerate(E['c14_cuts']):
        cue('c14 cut %d' % (j + 1), 'swish', t, -19, 'peak', MU.lp(pitch(S('swish'), -3 + 2 * j), 3500))
    # C15 しずく 2 つ（小さく）
    for j, t in enumerate((E['c15_0'] + 0.55, E['c15_0'] + 1.45)):
        cue('c15 drip %d' % (j + 1), 'drop', t, -15 - 3 * j)
    return sfx, cues


# ------------------------------------------------------------------ 風・悲鳴・波
def wind_track(n, E):
    """C07：上がるほど強く（空気が薄い 60km より上で消える）。C09 の落ちはじめ → C10 の着水へ大きく、着水で切る。"""
    out = np.zeros(n, np.float32)
    t = np.arange(n) / SR
    w = np.concatenate([S('wind'), S('wind')[::-1]])
    w = np.tile(w, int(np.ceil(n / len(w))) + 1)[:n]
    r = S('roar')
    r = np.tile(np.concatenate([r, r[::-1]]), int(np.ceil(n / (2 * len(r)))) + 1)[:n]
    # C07
    ts = np.array([a for a, _ in E['c07_alt']])
    hs = np.array([h for _, h in E['c07_alt']])
    h = np.interp(t, ts, hs, left=0, right=200)
    g7 = np.clip(np.log10(np.maximum(h, 0.1) / 0.1) / 2.0, 0, 1) * np.clip(1 - (h - 25) / 40.0, 0, 1)
    g7 *= (t >= ts[0]) * (t <= ts[-1] + 0.1)
    out += MU.lp(w, 2500) * MU.lp(g7.astype(np.float32), 6) * A.db(-14)
    # C09–C10
    t0, t1 = E['c09_fall'], E['c10_splash']
    q = np.clip((t - t0) / (t1 - t0), 0, 1)
    gq = q ** 1.8
    lo = MU.lp(r, 1500) * (1 - gq) + r * gq                # 近づくほど明るい
    roar = (lo * 0.8 + w * 0.4) * (A.db(-22) ** (1 - gq)) * A.db(-2)
    cut = int(t1 * SR)
    k = int(0.05 * SR)
    roar[cut - k:cut] *= np.linspace(1, 0, k)
    roar[cut:] = 0
    roar[:int(t0 * SR)] = 0
    return out + roar.astype(np.float32)


def scream_track(n, E):
    out = np.zeros(n, np.float32)
    if not os.path.exists(SCREAM):
        return out
    x = A.read_wav(SCREAM).astype(np.float32)
    t0, t1 = E['c10_scream'], E['c10_splash']
    L = int((t1 - t0) * SR)
    # ドップラー：再生速度 1.0 → 1.097（+1.6 半音）。素材が足りなければ最後の 0.5 秒を逆向きにつないで伸ばす
    rate = 1.0 + 0.097 * (np.arange(L) / L) ** 1.5
    pos = np.cumsum(rate) - rate[0]
    if pos[-1] > len(x) - 2:
        x = np.concatenate([x, x[-int(0.5 * SR):][::-1]])
    y = np.interp(pos, np.arange(len(x)), x).astype(np.float32)
    q = np.arange(L) / L
    # 遠い → 近い：ローパスを 3 本の帯で重ねて開く
    near = q ** 1.2
    y = MU.lp(y, 3200) * (1 - near) + y * near
    # 素材の「うわ」の頭がいちばん大きいので、まず 80ms の包絡で平らにし（+12 dB まで）、そのうえで -7 → 0 dB へ上げる
    ev = env(y, 80)
    ref = np.percentile(ev[ev > ev.max() * 0.05], 75)
    flat = np.clip(ref / np.maximum(ev, 1e-6), A.db(-6), A.db(12))
    flat = MU.lp(flat.astype(np.float32), 8)
    y = y * flat
    # -7 → 0 dB（-12 からだと頭の「う」が埋もれて Whisper が「わぁぁ」と聞く。「うわー」と分かる大きさから始める）
    g = A.db(-7 + 7 * q ** 0.9) * (1 + 0.12 * np.sin(2 * np.pi * 5.5 * np.arange(L) / SR) * (1 - q))
    y = y * g
    k = int(0.04 * SR)
    y[-k:] *= np.linspace(1, 0, k)
    MU.add(out, MU.fade(y, 0.02, 0.0), t0)
    return out


def ambience_track(n, E, total):
    """着水の後：IMG_9662 の現地の波 → 海岸4 につなぐ。屋内（13）は下げ、15 の波打ち際は近く、16 の夜は小さく。"""
    t = np.arange(n) / SR
    a1 = A.decode(os.path.join(core.FOOT, 'IMG_9662.MOV'), 1.6, 4.6)
    a2 = A.decode(os.path.join(LAB, 'ambience', 'wave4.mp3'), 2.0, 2.0 + total)
    a3 = A.decode(os.path.join(core.FOOT, 'IMG_9662.MOV'), 4.6, 7.4)
    for a in (a1, a2, a3):
        a -= a.mean()
        a /= max(1e-6, np.sqrt(np.mean(a ** 2)))
    s1, s2, s3 = (np.zeros(n, np.float32) for _ in range(3))
    tA = E['c10_splash'] + 0.3                             # 着水の少し後（地図から実写の海へ降りていく間）から
    MU.add(s1, MU.fade(a1, 0.3, 0.6), tA)
    MU.add(s2, a2, 0.0)
    MU.add(s3, MU.fade(a3, 0.4, 0.6), E['c15_0'] - 0.2)
    w = np.clip((t - (E['c11_0'] + 1.0)) / 1.0, 0, 1)
    bed = s1 * np.cos(w * np.pi / 2) + s2 * np.sin(w * np.pi / 2) * 0.8
    # 11 の声の後半（「仁淀川が出会う海」）で波が「海」を消す（Whisper が「国／軍」と聞く）→ 着水直後だけ大きく、0.9 秒で半分へ
    lvl = np.interp(t, [0, tA, E['c11_0'], E['c11_0'] + 0.9, E['c13_0'], E['c13_0'] + 0.2, E['c14_0'], E['c14_0'] + 0.2,
                        E['c15_0'], E['c15_0'] + 0.3, E['c16_0'], E['c16_0'] + 0.5, total - 0.6, total],
                    [0, 0.15, 1, 0.5, 0.5, 0.45, 0.45, 0.9, 0.9, 0.7, 0.7, 0.45, 0.45, 0])
    lvl *= (t >= tA)
    return (bed * lvl + s3 * 0.7 * (t < E['c16_0'] + 0.5)).astype(np.float32) * 0.06


def music_track(n, E, total):
    m = np.zeros(n, np.float32)
    m += MU.anime(total, E['c01_end'], E['c01_hit'], E['c01_ki'], E['c01_hops']) * A.db(0)
    m += MU.real_hit(total, E['c03_vo'], E['c03_hit']) * A.db(1)   # 低いうなりは「でも、現実だと……」から（長いため）
    m += MU.pulse(total, E['c03_crater'] - 0.1, E['c06_0'], [E['c04_lock'], E['c05_lock']]) * A.db(-3)
    m += MU.riser(total, E['c06_0'], E['c08_0'] + 0.1) * A.db(-2)
    m += MU.space(total, E['c08_0'] + 0.05, E['c10_0'] + 1.0) * A.db(-1)
    m += MU.coast(total, E['c11_0'], E['c16_logo'], total) * A.db(-1)
    return m


def duck(vo, depth_db=-7.0, pre=0.125, post=0.2, smooth=0.06):
    """ナレーション（と悲鳴）がある所で下げるゲイン。声の pre 秒前から post 秒後まで depth_db、smooth 秒でなめらかに。"""
    e = np.abs(vo)
    e = signal.lfilter([1 - math.exp(-1 / (0.015 * SR))], [1, -math.exp(-1 / (0.015 * SR))], e)
    c = np.concatenate([[0], np.cumsum(e > 0.03)])
    i = np.arange(len(e))
    hold = (c[np.minimum(len(e), i + int(pre * SR) + 1)] - c[np.maximum(0, i - int(post * SR))]) > 0
    g = np.where(hold, A.db(depth_db), 1.0).astype(np.float32)
    a = math.exp(-1 / (smooth * SR))
    return signal.lfilter([1 - a], [1, -a], g).astype(np.float32)


def carve(x, vo, mid_db, low_db, pre=0.125, smooth=0.06, split=250, keep=()):
    """声の帯域だけ深く下げる：split Hz より下は low_db、上は mid_db（声と重なる効果音・音楽の中高域をあける）。
    keep の区間（秒）は下げない（声と一緒に聞かせる見せ場の音：ポコッ・キラーン・当たりの頭）。"""
    low = MU.lp(x, split, 4)
    m = np.zeros(len(x), np.float32)
    for a, b in keep:
        m[max(0, int(a * SR)):int(b * SR)] = 1
    k = math.exp(-1 / (0.01 * SR))
    m = np.maximum(m, signal.lfilter([1 - k], [1, -k], m[::-1])[::-1].astype(np.float32))
    gl = np.maximum(duck(vo, low_db, pre, 0.2, smooth), m)
    gm = np.maximum(duck(vo, mid_db, pre, 0.2, smooth), m)
    return low * gl + (x - low) * gm


def limit(x, ceiling_db=-1.5, win=0.010):
    """4 倍オーバーサンプルのピークで見た簡単なリミッター：必要なゲインの ±win 秒の最小 → 幅 win の平均（どの点でも必要な分以上に下げる）。"""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    up = signal.resample_poly(x.astype(np.float64), 4, 1)
    pk = np.abs(up[:4 * len(x)]).reshape(-1, 4).max(axis=1)
    need = np.minimum(1.0, A.db(ceiling_db) / np.maximum(pk, 1e-9))
    la = int(win * SR)
    g = minimum_filter1d(need, 2 * la + 1)
    g = uniform_filter1d(g, la + 1)
    return (x * g).astype(np.float32)


def main(write_cues=False):
    import pyloudnorm as pyln
    bs = TL.load()
    total = bs[-1]['f1'] / FPS
    n = int(total * SR)
    E = events(bs)
    vo = np.zeros(n, np.float32)
    for b in bs:
        p = os.path.join(TL.VO, b['key'] + '.wav')
        if b['vo'] is not None and os.path.exists(p):
            A.add(vo, A.read_wav(p), b['vo'])
    vo /= max(1e-6, np.abs(vo).max())
    scr = scream_track(n, E)
    sfx, cues = sfx_track(n, E)
    keep = [(E['c01_hit'] - 0.02, E['c01_hit'] + 0.25), (E['c01_ki'] - 0.03, E['c01_ki'] + 0.5),
            (E['c03_hit'] - 0.02, E['c03_hit'] + 0.06)]   # 「コ」（launch）からは声のために削る
    sfx = carve(sfx, vo, -12.0, -2.0, pre=0.02, smooth=0.025, keep=keep)   # 先読み 20ms
    music = carve(music_track(n, E, total), vo + scr * 0.5, -12.0, -4.0)
    wind = wind_track(n, E) * duck(scr, -4.0)
    amb = carve(ambience_track(n, E, total), vo, -10.0, -3.0)
    mix = (vo * A.db(-3) + scr * A.db(1) + music * A.db(-15) + sfx * A.db(-7) + wind * A.db(-6) + amb * A.db(-4))
    fi = int(0.005 * SR)
    mix[:fi] *= np.linspace(0, 1, fi)
    meter = pyln.Meter(SR)
    lufs = meter.integrated_loudness(np.stack([mix, mix], 1).astype(np.float64))
    mix = mix * A.db(-14.0 - lufs)
    mix = limit(mix, -1.5)
    lufs2 = meter.integrated_loudness(np.stack([mix, mix], 1).astype(np.float64))
    st = np.stack([mix, mix], 1)
    out = os.path.join(core.OUT, 'mix_C.wav')
    with wave.open(out, 'wb') as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((np.clip(st, -1, 1) * 32767).astype('<i2').tobytes())
    stems = os.environ.get('MAZE_STEMS')
    if stems:
        os.makedirs(stems, exist_ok=True)
        for name, x in (('vo', vo * A.db(-3)), ('scream', scr * A.db(1)), ('music', music * A.db(-15)),
                        ('sfx', sfx * A.db(-7)), ('wind', wind * A.db(-6)), ('amb', amb * A.db(-4))):
            with wave.open(os.path.join(stems, name + '.wav'), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(SR)
                f.writeframes((np.clip(x * A.db(-14.0 - lufs), -1, 1) * 32767).astype('<i2').tobytes())
    if write_cues:
        pic = [dict(t=c['t'], name=c['name']) for c in cues]
        pic.append(dict(t=round(E['c10_scream'], 3), name='c10 scream start'))
        json.dump(dict(cues=sorted(pic, key=lambda c: c['t']), detail=sorted(cues, key=lambda c: c['t'])),
                  open(os.path.join(core.OUT, 'cues_C.json'), 'w'), ensure_ascii=False, indent=1)
        for c in sorted(cues, key=lambda c: c['t']):
            print('%6.3f  %-38s %-36s %+5.1f dB' % (c['t'], c['name'], c['src'], c['gain_db']))
    print('wrote', out, '%.2fs' % total, 'LUFS %.1f -> %.1f' % (lufs, lufs2), 'scream', os.path.relpath(SCREAM, core.OUT))


if __name__ == '__main__':
    main('--cues' in sys.argv)
