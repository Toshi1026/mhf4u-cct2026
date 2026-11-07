"""案B 11（約3.0秒）「2階は、海の見える、カフェ。」 写真：窓辺の明太子クリームパスタ → ケーキ（案A b14 の写真と動き）。

 0–(カ)：パスタ（窓の外に海。footage/new_1107/C176E666.jpg）にゆっくり寄る（b14 と同じ切り出し・同じ grade）。
 (カ)–：「カフェ」の「カ」でハードカットしてケーキの皿（IMG_0946.JPG）。
料理名は出さない（b14 と同じ理由）。料理の暖かさは b14 の grade のまま残す。
テロップ：「2F」（ACC）が「ニ」で、「オーシャンビューが広がるカフェ」（LP の見出しどおり）が「ウ（海の）」で出る。窓の外（空）の帯に置き、
その帯だけを b14 と同じようにやわらかく落としてコントラストを取る。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
from beats import b14
from cutB.beats import _fac as F

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
    tb, ts = F.heading2('2F', 'オーシャンビューが広がるカフェ', hl='2F')
    for i in range(n):
        k, j, m = (0, i, f_ka) if i < f_ka else (1, i - f_ka, n - f_ka)
        s = b14.SHOTS[k]
        u = core.ease(j / max(1, m - 1), 'out' if k == 0 else 'smooth')
        f = b14._view(imgs[k], core.lerp(s['z0'], s['z1'], u), s['cx'], s['cy'])
        f = f * (1 - TOP)
        f = tb.put(f, F.appear(i, f_ni, 5))
        f = ts.put(f, F.appear(i, f_u, 5))
        yield np.clip(f, 0, 1).astype(np.float32)
