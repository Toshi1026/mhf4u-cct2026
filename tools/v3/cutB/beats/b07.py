"""B07（20.9–22.6秒）「そして、52秒後。」
06 の続き（同じ断面）。06 の文字と富士山が消え、点だった悪役が元の大きさに戻りながら、放物線の右半分を回りながら落ちていく
（だんだん速く）。秒のカウンター（全体を強調）はソで「26秒後」（頂点＝飛んでいる時間の半分）で出て、悪役の横の進みと同じに数え、
「ゴジュウニ」のニの 3 フレーム後、水面に着くところで「52秒後」に止まる。悪役は水面で消え、着水（小さなしぶき）は T で土佐湾の線画の上に見せる。
着水の次のフレームから最後まで：着水する所へのズームのぼかし、最後の 11 フレームで濃紺（案A b10 と同じ背景）へ暗くし、土佐湾の線画をごくうすく出す（T は線画から）。"""
import numpy as np

from cutB.beats import _useA  # noqa: F401
import core
import timeline_b as timeline
from core import W, H, WL
from beats import _mapui as U
from beats import _tosa, b10
from cutB.beats import _art as A, _xsec as X, b06

LAND_AFTER_NI = 3                             # 着水（＝52秒）はナレーションの「ニ」の 3 フレーム後
DARKEN = 11                                   # 濃紺へ暗くするのは最後の 11 フレーム（1フレームの明るさの差を小さく）


class SecCounter:
    """「N秒後」。数字は同じ幅のマス、全体を ACC。右端の「秒後」を最終の文字列の中心で固定。"""

    def __init__(self, cx, y, size=110):
        self.y, self.size = y, size
        f = core.font(size, 'Bold')
        self.cell = max(f.getlength(str(d)) for d in range(10)) * 0.98
        self.sw = core.text_width('秒後', size, 'Bold', 0.02)
        tot = 2 * self.cell + 0.04 * size + self.sw
        self.left2 = cx - tot / 2
        self.cache = {}

    def put(self, f, v, alpha):
        s = str(int(v))
        p = self.cache.get(s)
        if p is None:
            x = self.left2 + (2 - len(s)) * self.cell / 2          # 1桁の間は少し右へ寄せて中心を保つ
            items = []
            for ch in s:
                items.append(dict(text=ch, x=x + self.cell / 2, y=self.y, size=self.size, weight='Bold', track=0,
                                  color=core.ACC))
                x += self.cell
            items.append(dict(text='秒後', x=x + 0.04 * self.size, y=self.y, size=self.size, weight='Bold',
                              anchor='ls', track=0.02, color=core.ACC))
            p = self.cache[s] = U.Patch(items)
        return p.put(f, alpha)


def render(beat, ctx):
    n = beat['n']
    mo = beat['moras']
    assert mo[0][0] == 'ソ' and mo[6][0] == 'ニ', mo
    f_so, f_ni = timeline.mora_frame(beat, 0), timeline.mora_frame(beat, 6)
    b6 = next(b for b in timeline.load() if b['key'] == '06')
    T6 = b06.timing(b6)
    n6 = b6['n']
    t_main, t_sub = b06.texts()
    lab = X.labels()
    ctr = SecCounter(W / 2, 450)
    nav = b10._background()
    lines = _tosa.contours()
    f_land = min(f_ni + LAND_AFTER_NI, n - DARKEN - 2)   # 水面に着くフレーム（52 秒＝着水。ニのすぐ後）
    out = n - (f_land + 1)                       # 着水の次のフレームから最後まで：着水点へのズームのぼかし
    for i in range(n):
        gi = n6 + i                              # 06 の頭からの通し番号
        t = gi / core.FPS
        f = b06.scene(gi, T6, fuji_a=1.0 - core.ease((i + 1) / 9.0))   # 富士山は落ちる弧と重なるので、06 の文字と一緒に消す
        # 落ちる位置
        tau = np.clip((i - 2) / float(f_land - 2), 0, 1)
        u = 0.5 + 0.5 * tau ** 1.7               # 横の進み＝時間（空気抵抗なしなら水平の速さは一定）なので、秒＝52·u
        rgba, x0, y0 = X.trail_layer(u, ghost=1.0 - core.ease(i / 30.0) * 0.4)
        f = U.paste(f, rgba, x0, y0)
        # 頂点の印（少しうすく）・見上げる点線（消す）
        f = b06.apex_lines(f, 1.0 - 0.4 * core.ease(i / 12.0))
        f = lab['apex'].put(f, 1.0 - 0.4 * core.ease(i / 12.0))
        f = b06.sight_line(f, 0.5 * (1 - core.ease(i / 8.0)))
        # 悪役：点から元の大きさへ戻りながら落ちる
        g = core.ease((i - 1) / 10.0, 'inout')
        sc = 0.12 + (X.VS - 0.12) * g
        x, y = X.arc(u)
        ang = -20 + 560 + 25 * (gi - T6['ru']) / core.FPS + 700 * tau ** 1.5
        va = 1 - core.ease((i - f_land + 1) / 3.0)   # 水面に着いたら 3 フレームで消える（しぶきは T の地図の上で見せる）
        if va > 0.003:
            if sc * 34 >= 5:
                f = A.put_villain(f, float(x), float(y) - 14 * sc * va, sc, ang=ang, pose='fly', t=t, alpha=va)
            else:
                f = A.put_dot(f, float(x), float(y) - 2, 2.4, alpha=va)
        # 06 のキラーンの名残
        gg = A.glint_env(gi - T6['ki'])
        if gg > 0.01:
            f = A.put_glint(f, X.APEX[0], X.APEX[1] - 2, 42 * gg, ang=10 * (1 - gg), alpha=min(1.0, gg * 1.4))
        # 文字：06 の文字が消え、カウンター
        a_old = 1 - core.ease((i + 1) / 5.0)
        f = t_main.put(f, a_old)
        f = t_sub.put(f, a_old)
        j = i - (f_land + 1)
        jd = i - (n - DARKEN)
        tout = 1 - core.ease((jd + 1) / 6.0) if jd >= 0 else 1.0
        if i >= f_so:
            a = core.ease((i - f_so + 1) / 5.0)
            f = ctr.put(f, round(52 * u), a * tout)      # 頂点＝26秒後 から、着水＝52秒後 まで
        # 最後：着水点へのズームのぼかし＋濃紺へ＋線画をうすく
        if j >= 0:
            q = (j + 1) / out
            f = core.zoom_blur(f, 0.24 * q ** 1.2, cx=X.X1, cy=WL, steps=18)
        if jd >= 0:
            qd = (jd + 1) / DARKEN
            d = 0.6 * qd + 0.4 * core.ease(qd, 'smooth')
            f = f * (1 - d) + nav * d
        if j >= 0:
            la = (0.14 * core.ease(q, 'in') * lines)[..., None]
            f = f * (1 - la) + core.FOAM * la
        yield np.clip(f, 0, 1).astype(np.float32)
