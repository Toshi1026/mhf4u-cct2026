"""案C の BGM（コードで作る。録音は使わない）。audio_c.py から呼ぶ。区間ごとに別の曲：

  anime(t0, t1, cue)    C01：はずむピチカート（F メジャー・120BPM）。拳が当たる→悪役が飛ぶ間は上がっていく16分の駆け上がり、
                        「キ」で高い F（グロッケン）、そのあと拳のはずみに合わせて 2 音。C02 の頭でテープを止めたように音程が落ちて切れる。
  real_hit(t_hit, t0)   C03：スローの間の低いドローン（D）と逆再生のふくらみが当たりへ向かってふくらむ → 当たりの直前 70ms 無音
                        → 低い一撃（ピッチが落ちる太鼓＋サブ）＋短い地鳴りでなく「空気の」ゴーッ（1.4 秒で消える。地震を思わせない）。
  pulse(t0, t1, locks)  C04–C05：D マイナーの低い弦の刻み（8分・120BPM）。数字が止まる所（ト）に低いティンパニ風の1打。
  riser(t0, t1)         C06–C08：シェパード・トーンで上がり続ける音＋ノイズのライザー。刻みは 8分 → 16分。最後は C08 の頭ですっと抜く。
  space(t0, t1)         C08–C09：明るい高い和音（A add9）のゆっくりしたパッド＋空気のノイズ（残響つき）。
  coast(t0, t_logo, t1) C11–C16：あたたかく明るい海辺の曲（D メジャー、D–A–Bm–G、約109BPM：C11 の頭から 6 小節でロゴの出に着く）。
                        ウクレレ風のストローク・マリンバ・ベース・柔らかいキック・シェイカー・パッド。ロゴで D の和音を鳴らして伸ばす。
すべて 48 kHz モノラル float32。音量は audio_c.py 側で決める（ここでは曲の中のバランスだけ）。
"""
import numpy as np
from scipy import signal

SR = 48000
RNG = np.random.default_rng(2611)


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def tt(n):
    return np.arange(int(n)) / SR


def fade(x, a=0.003, r=0.01):
    x = x.astype(np.float32)
    na, nr = min(len(x), int(a * SR)), min(len(x), int(r * SR))
    if na:
        x[:na] *= np.linspace(0, 1, na)
    if nr:
        x[-nr:] *= np.linspace(1, 0, nr)
    return x


def add(buf, x, t, g=1.0):
    s = int(round(t * SR))
    if s < 0:
        x, s = x[-s:], 0
    if s >= len(buf) or len(x) == 0:
        return
    e = min(len(buf), s + len(x))
    buf[s:e] += x[:e - s] * g


def lp(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, min(fc, SR * 0.45), 'low', fs=SR, output='sos'), x).astype(np.float32)


def hp(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, fc, 'high', fs=SR, output='sos'), x).astype(np.float32)


def bp(x, f0, f1, order=2):
    return signal.sosfilt(signal.butter(order, [f0, f1], 'band', fs=SR, output='sos'), x).astype(np.float32)


def sweep_lp(x, fc):
    """時間で動くローパス：固定の遮断周波数の版を、対数の位置で隣どうし混ぜる（切れ目なし）。fc は配列。"""
    cs = np.array([150, 300, 600, 1200, 2400, 4800, 9600, 19000], float)
    fc = np.clip(np.asarray(fc, float), cs[0], cs[-1])
    pos = np.interp(np.log2(fc), np.log2(cs), np.arange(len(cs)))
    out = np.zeros(len(x), np.float32)
    for j, c in enumerate(cs):
        w = np.clip(1 - np.abs(pos - j), 0, 1)
        if w.max() > 0:
            out += lp(x, c) * w
    return out


def reverb(x, rt=1.4, wet=0.25, pre=0.012, bright=5000):
    """生成したノイズのインパルス応答（指数減衰・高域ほど早く減る）で畳み込む。"""
    n = int(rt * 1.2 * SR)
    t = tt(n)
    ir = RNG.standard_normal(n) * np.exp(-6.9 * t / rt)
    ir = lp(ir, bright) * 0.6 + lp(ir, bright * 0.35) * 0.4
    ir = np.concatenate([np.zeros(int(pre * SR)), ir])
    ir /= np.sqrt(np.sum(ir ** 2))
    y = signal.fftconvolve(x, ir)[:len(x)].astype(np.float32)
    return x * (1 - wet) + y * wet


# ------------------------------------------------------------------ 楽器
def pizz(f, dur=0.32, vel=1.0, bright=1.0):
    """弦のピチカート：倍音ごとに早く減衰する加算合成＋指で弾く短いノイズ。"""
    t = tt(dur * SR)
    v = np.zeros_like(t)
    for k in range(1, 14):
        fk = f * k * (1 + 0.0007 * k * k)
        if fk > 16000:
            break
        a = (1.0 / k ** 1.25) * (bright if k > 3 else 1.0)
        v += a * np.sin(2 * np.pi * fk * t + RNG.uniform(0, 0.3)) * np.exp(-t * (9 + 5.5 * k))
    click = bp(RNG.standard_normal(len(t)), 900, 4000) * np.exp(-t * 350) * 0.25
    return fade((v + click) * vel * np.minimum(1, t / 0.0025), 0.0, 0.02)


def glock(f, dur=1.2, vel=1.0):
    t = tt(dur * SR)
    v = (np.sin(2 * np.pi * f * t) * np.exp(-t * 3.2) + 0.35 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 9)
         + 0.12 * np.sin(2 * np.pi * f * 5.40 * t) * np.exp(-t * 16))
    return fade(v * vel * np.minimum(1, t / 0.0015), 0.0, 0.05)


def marimba(f, dur=0.9, vel=1.0):
    t = tt(dur * SR)
    v = (np.sin(2 * np.pi * f * t) * np.exp(-t * 4.5) + 0.28 * np.sin(2 * np.pi * f * 3.93 * t) * np.exp(-t * 16)
         + 0.08 * np.sin(2 * np.pi * f * 9.2 * t) * np.exp(-t * 40))
    mallet = lp(RNG.standard_normal(len(t)), 2500) * np.exp(-t * 400) * 0.15
    return fade((v + mallet) * vel * np.minimum(1, t / 0.002), 0.0, 0.05)


def uke(f, dur=0.7, vel=1.0):
    """ナイロン弦の明るいはじき（ピチカートより長く、倍音の減り方がゆるい）。"""
    t = tt(dur * SR)
    v = np.zeros_like(t)
    for k in range(1, 11):
        fk = f * k * (1 + 0.0004 * k * k)
        v += (1.0 / k ** 1.1) * np.sin(2 * np.pi * fk * t) * np.exp(-t * (3.5 + 2.2 * k))
    pick = hp(RNG.standard_normal(len(t)), 2500) * np.exp(-t * 500) * 0.12
    return fade((v + pick) * vel * np.minimum(1, t / 0.002), 0.0, 0.04)


def strum(notes, dur=0.7, vel=1.0, up=False, spread=0.011):
    ns = list(reversed(notes)) if up else list(notes)
    n = int((dur + spread * len(ns)) * SR)
    out = np.zeros(n, np.float32)
    for j, m in enumerate(ns):
        add(out, uke(hz(m), dur, vel * (0.8 if up else 1.0) * (0.92 ** j)), j * spread)
    return out


def bass(f, dur=0.5, vel=1.0):
    t = tt(dur * SR)
    v = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(4 * np.pi * f * t) + 0.08 * np.sin(6 * np.pi * f * t)
    return fade(v * vel * np.exp(-t * 2.5) * np.minimum(1, t / 0.006), 0.0, 0.04)


def kick(vel=1.0):
    t = tt(0.32 * SR)
    f = 48 + 60 * np.exp(-t * 32)
    v = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 10)
    v += lp(RNG.standard_normal(len(t)), 3000) * np.exp(-t * 300) * 0.08
    return fade(v * vel, 0.0, 0.03)


def shaker(vel=1.0):
    t = tt(0.09 * SR)
    v = bp(RNG.standard_normal(len(t)), 5500, 12000) * (t / 0.012) * np.exp(-t / 0.012) * 1.6
    return fade(v * vel, 0.0, 0.01)


def saw_pad(notes, dur, cutoff=1400, a=0.4, r=0.8, det=0.18):
    t = tt(dur * SR)
    v = np.zeros_like(t)
    for m in notes:
        for d in (-det, det):
            ph = (hz(m) * (1 + d / 100.0) * t + RNG.uniform()) % 1.0
            v += 2 * ph - 1
    v = lp(v / (2 * len(notes)), cutoff, 2)
    e = np.ones_like(t)
    na, nr = int(a * SR), int(r * SR)
    e[:na] = np.linspace(0, 1, na) ** 1.5
    e[-nr:] *= np.linspace(1, 0, nr) ** 1.5
    return (v * e).astype(np.float32)


def sine_pad(notes, dur, a=0.8, r=1.2, lfo=0.15):
    t = tt(dur * SR)
    v = np.zeros_like(t)
    for j, m in enumerate(notes):
        f = hz(m)
        v += np.sin(2 * np.pi * f * t + 0.003 * np.sin(2 * np.pi * (lfo + 0.04 * j) * t)) * (0.7 + 0.3 * np.sin(
            2 * np.pi * (0.11 + 0.03 * j) * t + j))
    e = np.ones_like(t)
    na, nr = int(a * SR), int(r * SR)
    e[:na] = np.linspace(0, 1, na) ** 2
    e[-nr:] *= np.linspace(1, 0, nr) ** 2
    return (v * e / len(notes)).astype(np.float32)


def tape_stop(x, t_stop, dur=0.14):
    """t_stop から dur 秒で再生速度が 0 へ落ちる（レコード・テープを止めた音）。そのあとは無音。"""
    s0 = int(t_stop * SR)
    n = int(dur * SR)
    if s0 >= len(x):
        return x
    seg_rate = np.linspace(1.0, 0.0, n) ** 1.6
    pos = s0 + np.cumsum(seg_rate)
    seg = np.interp(pos, np.arange(len(x)), x) * np.linspace(1, 0, n) ** 0.7
    y = x.copy()
    y[s0:s0 + n] = seg[:len(y) - s0]
    y[s0 + n:] = 0
    return y


# ------------------------------------------------------------------ 区間
def anime(total, t_end, hit, ki, hops):
    """C01。hit＝当たり、ki＝キラーン（光の一番大きい時刻）、hops＝拳がはずむ時刻 2 つ。t_end で止まる。"""
    out = np.zeros(int(total * SR), np.float32)
    b = 0.25                                              # 8分（120BPM）
    for k, m in enumerate([41, 48, 41, 48, 41]):         # ベース：F2 C3（4分）
        if k * 2 * b < t_end:
            add(out, pizz(hz(m), 0.45, 0.9, 0.7), k * 2 * b)
    for k in range(4):                                    # 裏拍：A3+C4 / G3+C4 の重音
        t = (2 * k + 1) * b
        for m in ([57, 60] if k % 2 == 0 else [55, 60]):
            add(out, pizz(hz(m), 0.25, 0.42), t)
    add(out, pizz(hz(72), 0.3, 0.75), 0.0)                # 頭：C5
    # 当たり → キラーン：16分で F4 → F6 へ駆け上がる（悪役が空へ飛ぶ）
    run = [65, 69, 72, 77, 81, 84, 89]
    for j, m in enumerate(run):
        t = hit + 0.03 + j * (ki - hit - 0.06) / (len(run) - 1)
        add(out, pizz(hz(m), 0.22, 0.42 + 0.06 * j, 1.2), t - 0.004)
    add(out, glock(hz(89), 1.4, 0.55), ki - 0.002)        # キラーン（F6）
    add(out, glock(hz(96), 1.0, 0.22), ki + 0.06)
    for t, m in zip(hops, [77, 81]):                      # 拳のはずみ（F5・A5）
        add(out, pizz(hz(m), 0.22, 0.55, 1.1), t)
    add(out, pizz(hz(84), 0.25, 0.5, 1.1), hops[-1] + 0.25)
    out = reverb(out, 0.9, 0.18)
    return tape_stop(out, t_end - 0.05, 0.13)


def real_hit(total, t0, t_hit):
    """C03。t0：現実パートの頭、t_hit：当たり。"""
    out = np.zeros(int(total * SR), np.float32)
    n = int((t_hit - t0) * SR)
    t = tt(n)
    q = t / t[-1]
    drone = (np.sin(2 * np.pi * 36.71 * t) * 0.8 + np.sin(2 * np.pi * 73.42 * t) * 0.35
             + np.sin(2 * np.pi * 110.0 * t) * 0.12 * q)
    air = sweep_lp(RNG.standard_normal(n), 300 + 3200 * q ** 3) * 0.22 * q ** 2.2       # 逆再生のふくらみ
    seg = (drone * (0.25 + 0.75 * q ** 1.5) + air) * np.minimum(1, t / 0.4)
    gap = int(0.07 * SR)                                   # 当たりの直前 70ms は無音（吸い込み）
    seg[-gap:] *= np.linspace(1, 0, gap) ** 3
    add(out, seg.astype(np.float32), t0)
    # 一撃：ピッチが落ちる太鼓＋サブ＋空気のゴーッ（1.4 秒で消える）
    t = tt(1.8 * SR)
    f = 38 + 85 * np.exp(-t * 18)
    drum = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.6)
    skin = lp(RNG.standard_normal(len(t)), 1800) * np.exp(-t * 45) * 0.5
    roar = lp(RNG.standard_normal(len(t)), 220, 4) * np.exp(-t * 3.6) * (1 - np.exp(-t * 40)) * 1.1
    hitx = fade((drum + skin + roar) * np.minimum(1, t / 0.002), 0.0, 0.3)
    add(out, hitx, t_hit)
    return out


def pulse(total, t0, t1, locks):
    """C04–C05：D マイナーの低い弦の刻み（8分）＋ロックの低い1打。"""
    out = np.zeros(int(total * SR), np.float32)
    e8 = 0.25
    k = 0
    while t0 + k * e8 < t1 - 0.05:
        m = [38, 38, 38, 38, 38, 38, 41, 40][k % 8]
        acc = 1.0 if k % 4 == 0 else 0.62
        x = saw_pad([m, m + 12], 0.22, cutoff=650, a=0.012, r=0.12, det=0.25) * acc
        add(out, x, t0 + k * e8)
        k += 1
    pad = saw_pad([50, 53, 57], t1 - t0 + 0.4, cutoff=900, a=0.8, r=0.5) * 0.35
    add(out, pad, t0)
    for t in locks:
        tm = tt(1.1 * SR)
        f = 55 + 30 * np.exp(-tm * 25)
        add(out, fade(np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tm * 3.5), 0.002, 0.2) * 0.9, t)
    return out


def riser(total, t0, t1):
    """C06–C08：シェパード・トーン（上がり続ける）＋ノイズ。t1 ですっと抜く（15ms）。"""
    out = np.zeros(int(total * SR), np.float32)
    n = int((t1 - t0) * SR)
    t = tt(n)
    q = t / t[-1]
    v = np.zeros(n)
    octs = 6
    rate = 0.55                                           # 1 秒に 0.55 オクターブ上がる
    for j in range(octs):
        pos = (j + rate * t) % octs                      # 0..octs
        f = 55.0 * 2 ** pos
        amp = np.sin(np.pi * pos / octs) ** 2
        v += amp * np.sin(2 * np.pi * np.cumsum(f) / SR)
    v = lp(v / octs, 2500) * (0.25 + 0.75 * q ** 1.4)
    nz = RNG.standard_normal(n)
    nz = signal.sosfilt(signal.butter(2, 400, 'high', fs=SR, output='sos'), nz)
    nzf = np.zeros(n)
    fc = 600 + 6000 * q ** 2                             # 帯域を上へ動かす：固定の帯を中心の近さで重ねる（切れ目なし）
    for c in (700, 1400, 2800, 5600, 9000):
        w = np.exp(-0.5 * (np.log2(fc / c) / 0.6) ** 2)
        nzf += bp(nz, c / 1.5, min(c * 1.5, 20000), 2) * w
    nzf = lp(nzf, 9000) * 0.35 * q ** 2.5
    # 刻み：8分 → 16分（D の低音）
    tk = 0.0
    while tk < t1 - t0 - 0.06:
        qq = tk / (t1 - t0)
        g = 0.35 + 0.65 * qq
        x = saw_pad([38, 50], 0.12, cutoff=500 + 1500 * qq, a=0.006, r=0.07, det=0.3) * g * 0.55
        add(v, x, tk)
        tk += 0.25 if qq < 0.45 else 0.125
    y = (v + nzf).astype(np.float32)
    k = int(0.015 * SR)
    y[-k:] *= np.linspace(1, 0, k)
    add(out, y * np.minimum(1, t / 0.15).astype(np.float32), t0)
    return out


def space(total, t0, t1):
    """C08–C09：高い明るい和音（A add9：A5 E6 B6 C#7（声の帯域より上））＋低い A の持続＋空気のノイズ。残響つき。"""
    out = np.zeros(int(total * SR), np.float32)
    d = t1 - t0
    v = sine_pad([81, 88, 95, 97], d, a=0.35, r=1.0) * 0.45 + sine_pad([45, 57], d, a=0.6, r=1.0) * 0.6
    air = lp(hp(RNG.standard_normal(len(v)), 2500), 9000) * 0.06
    e = np.ones(len(v))
    e[:int(0.4 * SR)] = np.linspace(0, 1, int(0.4 * SR))
    e[-int(1.0 * SR):] *= np.linspace(1, 0, int(1.0 * SR))
    y = reverb((v + air * e).astype(np.float32), 2.6, 0.45, bright=7000)
    add(out, y, t0)
    return out


def coast(total, t0, t_logo, t1):
    """C11–C16：D–A–Bm–G の明るい海辺の曲。t0 から 6 小節でロゴ（t_logo）に D の和音が着く。"""
    out = np.zeros(int(total * SR), np.float32)
    bar = (t_logo - t0) / 6.0
    bt = bar / 4.0
    CH = {'D': [62, 66, 69, 74], 'A': [61, 64, 69, 73], 'Bm': [62, 66, 71, 74], 'G': [62, 67, 71, 74]}
    ROOT = {'D': 38, 'A': 45, 'Bm': 47, 'G': 43}
    FIFTH = {'D': 45, 'A': 40, 'Bm': 42, 'G': 38}
    prog = ['D', 'A', 'Bm', 'G', 'D', 'A']
    # ストローク：D・DU・UDU（拍：0, 1, 1.5, 2.5, 3, 3.5）
    pat = [(0.0, False, 1.0), (1.0, False, 0.75), (1.5, True, 0.55), (2.5, True, 0.6), (3.0, False, 0.75),
           (3.5, True, 0.55)]
    mel = {0: [(0, 78), (1.5, 76), (2, 74), (3, 76)], 1: [(0, 76), (1.5, 73), (2, 69)],
           2: [(0, 74), (1.5, 76), (2, 78), (3, 81)], 3: [(0, 79), (2, 78), (3, 76)],
           4: [(0, 78), (1.5, 76), (2, 74), (3, 76)], 5: [(0, 76), (1, 78), (2, 81), (3, 85)]}
    for b, c in enumerate(prog):
        tb = t0 + b * bar
        for beat, up, vel in pat:
            add(out, strum(CH[c], 0.55, vel * 0.5, up), tb + beat * bt)
        add(out, bass(hz(ROOT[c]), bt * 1.8, 0.75), tb)
        add(out, bass(hz(FIFTH[c]), bt * 1.6, 0.55), tb + 2 * bt)
        if b >= 1:
            add(out, kick(0.55), tb)
            add(out, kick(0.4), tb + 2 * bt)
            for s in range(8):
                add(out, shaker(0.5 if s % 2 else 0.3), tb + s * bt / 2 + 0.006)
        for beat, m in mel.get(b, []):
            add(out, marimba(hz(m), 0.8, 0.38), tb + beat * bt)
        pad = saw_pad(CH[c][:3], bar + 0.3, cutoff=1100, a=0.25, r=0.3) * 0.18
        add(out, pad, tb)
    # ロゴ：D の和音（ストローク＋マリンバ＋グロッケン）を伸ばす
    d_end = t1 - t_logo
    add(out, strum([50, 57, 62, 66, 69, 74], d_end + 0.2, 0.75, False, 0.016), t_logo)
    add(out, bass(hz(38), d_end, 0.9), t_logo)
    for j, m in enumerate([74, 78, 81, 86]):
        add(out, marimba(hz(m), 1.4, 0.32), t_logo + j * 0.05)
    add(out, glock(hz(93), 1.6, 0.25), t_logo + 0.22)
    add(out, saw_pad([62, 66, 69], d_end, cutoff=1300, a=0.05, r=d_end * 0.9) * 0.22, t_logo)
    return reverb(out, 1.3, 0.2)
