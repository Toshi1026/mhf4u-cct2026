"""案C 09（約 1.6 秒）「そして、5分半後。」 テロップなし。

C08 と同じ宇宙の構図（_space の 1 本のカメラ）。悪役は頂点でくるくる回るのがゆっくり止まり、「あれ？」と一瞬止まって
（ばねで頭が下へ）、ジタバタしながら真っ逆さまに落ちはじめる（_fall.state：C10 と同じ式で、つなぎ目の位置・速さ・形がそろう）。
最後の 14 フレームはカメラが悪役を追って下を向きはじめる（C10 の頭で真下まで振り切る）。
注：「5分半」は打ち上げから着水までの時間（ファクトチェックの注意）。この画は「落ちはじめ」なので、着水（C10 の最後）を
「5分半後」に見せるのは C10 の終わり。ナレーションの位置は timeline_c のまま（このビートでは動かさない）。
"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
from cutB.beats import _motion as M, _maplook as ML
from cutC.beats import _space as SPC, _vill as V, _fall as FA
from cutC.beats.c10 import dir_blur, cam_flow
from cutC.beats.c08 import telop


def render(beat, ctx):
    n = beat['n']
    for i in range(n):
        k = SPC.k_of(beat, i)
        rgb, info = SPC.earth(k)
        (vx, vy), dz = cam_flow(k)
        rgb = dir_blur(rgb, vx, vy, 0.5)
        f = SPC.map_post(rgb, k, info, 540 + i)
        st, st1 = FA.state(k), FA.state(k - 1)
        r, x0, y0 = V.rgba(st['x'], st['y'], st['scale'], st['ang'], st['pc'], st['t'])
        r = M.smear(r, (st['x'] - st1['x']) * 0.5, (st['y'] - st1['y']) * 0.5)
        f = M.put_rgba(f, r, x0, y0)
        pass  # 出典（国土地理院）は画面に出さず、キャプションに書く（2026-11-08 クライアントの指示）
        yield np.clip(f, 0, 1).astype(np.float32)
