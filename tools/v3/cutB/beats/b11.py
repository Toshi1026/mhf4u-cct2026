"""案B 11（約3.0秒）「2階は、海の見える、カフェ。」 写真：窓辺の明太子クリームパスタ → ケーキ（案A b14 の写真と動き）。

 0–(カ)：パスタ（窓の外に海。footage/new_1107/C176E666.jpg）にゆっくり寄る（b14 と同じ切り出し・同じ grade）。
 (カ)–：「カフェ」の「カ」でハードカットしてケーキの皿（IMG_0946.JPG）。
料理名は出さない（b14 と同じ理由）。料理の暖かさは b14 の grade のまま残す。
テロップ：「2F」（ACC）が「ニ」で、「オーシャンビューが広がるカフェ」（LP の見出しどおり）が「ウ（海の）」で出る。窓の外（空）の帯に置き、
その帯だけを b14 と同じようにやわらかく落としてコントラストを取る。
2026-11-07 動きの改訂：
 - 最初の 4 フレームは 10 からの縦の振り（カメラが上へ）を受け、最後の 4 フレームは 12 へ同じ向きに振る。
 - パスタ：奥行きで 2.5D のドリー（皿が窓に対して大きくなる。par 0→0.07、横の視差 −0.006→+0.006、寄り 1.07 倍）。
   最後のフレームでもピークの 40% の速さが残る（hermite）。窓の外（遠い所）はほんの少しぼかす（被写界深度、σ≤1.2px）。
 - ケーキ：パスタの最後の寄りの速さを受け継いで始まる（止まった所から始めない）。par 0→0.06、横の視差 0→+0.012。
   視差のぶんの余白がいるので、寄りは b14 より少し寄った所（パスタ 1.02、ケーキ 1.03）から。
 - 文字：「2F」は 1 文字ずつ上がって FOAM→ACC の強調、「オーシャンビュー…」は行ごと下のマスクから上がる。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from beats import b14
from cutB.beats import _fac as F
from cutB.beats import _motion as Mo
from cutB.beats import _p25 as P

# 寄り z（b14._view と同じ：窓の幅＝写真の幅の 1/z）・奥行きの視差 par・横の視差 lat（始め→終わり）・hermite の速さ
PASTA = dict(z=(1.02, 1.09), par=(0.0, 0.07), lat=(-0.006, 0.006), v0=0.5, v1=0.5, dof=1.2)
CAKE = dict(z=(1.03, 1.09), par=(0.0, 0.06), lat=(0.0, 0.012), v1=0.8)
CAKE_GAIN = 0.90          # ケーキの写真はパスタより明るいので、カットの明るさの段差を 0.1 未満に
_y = np.arange(core.H, dtype=np.float32)
TOP = (0.28 * np.exp(-((_y - 470) / 170.0) ** 2))[:, None, None]


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[0][0] == 'ニ' and mo[4][0] == 'ウ' and mo[10][0] == 'カ', mo
    f_ni, f_u, f_ka = [round(mo[k][1] * core.FPS) - beat['f0'] for k in (0, 4, 10)]
    imgs = [b14._grade(b14._load(s)) for s in b14.SHOTS]
    imgs[1] = imgs[1] * CAKE_GAIN
    deps = [P.depth(imgs[0], 'b11pasta'), P.depth(imgs[1], 'b11cake')]
    m0, m1 = f_ka, n - f_ka
    # ケーキの最初の速さ＝パスタの最後の寄りの速さ（z／フレーム）を、ケーキの平均の速さで割った比
    vz = (PASTA['z'][1] - PASTA['z'][0]) * PASTA['v1'] / (m0 - 1)
    v0c = vz / ((CAKE['z'][1] - CAKE['z'][0]) / (m1 - 1))
    tb = Mo.KText([dict(text='2F', x=core.W / 2, y=F.Y_BIG, size=F.SZ_BIG, weight='Bold', hl='2F', track=0.03, shadow=0.9)])
    ts = Mo.KText([dict(text='オーシャンビューが広がるカフェ', x=core.W / 2, y=F.Y_SUB, size=F.SZ_SUB, weight='Medium',
                        track=0.06, shadow=0.9)])
    for i in range(n):
        if i < f_ka:
            k, sp = 0, PASTA
            u = P.hermite(i / (m0 - 1), sp['v0'], sp['v1'])
        else:
            k, sp = 1, CAKE
            u = P.hermite((i - f_ka) / (m1 - 1), v0c, sp['v1'])
        s = b14.SHOTS[k]
        z = core.lerp(*sp['z'], u)
        f, A = Mo.parallax_view(imgs[k], deps[k], s['cx'], s['cy'], 1.0 / z, par=core.lerp(*sp['par'], u),
                                lat=core.lerp(*sp['lat'], u))
        if k == 0 and sp.get('dof'):
            f = Mo.dof(f, Mo.view_depth(deps[0], A), far_thresh=0.35, sigma=sp['dof'] * u)
        f = f * (1 - TOP)
        f = P.whip_in(f, i, +1)                               # 10 から：カメラが上へ振ってきた
        wa = 1 - P.whip_q(i, n)                                 # 振りの最後のフレームでは文字は消えている
        f = tb.reveal(f, i, f_ni - 2, 'char_rise', stagger=1.5, dur=10, accent=f_ni, alpha=wa)
        f = ts.reveal(f, i, f_u - 1, 'line_mask', alpha=wa)
        f = P.whip_out(f, i, n, +1)                            # 12 へ：カメラが上へ振る
        yield np.clip(f, 0, 1).astype(np.float32)
