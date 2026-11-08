"""案C 15 の悪役：ずぶ濡れで砂浜に座り、海を眺める（顔のない白いピクトグラム＋マント。_vill / _figures3d と同じ材質の系統・大きさの単位）。

座り方（体のローカル：x 右、y 下、z 手前。体は −z を向く＝カメラに背を向けて海を見る）：
  ひざを立てて座り、両手をひざの上に置く。頭は少し前へ（海を見る）。マントは背中に張りついて砂の上まで垂れる（濡れて重い）。
  yaw（度）で体を縦の軸まわりに回し、斜め後ろから見る（ひざとすねの形が横に見えて、座っているのが読める）。
  マントは押し出した板（slab）なので回せない → 体の回転に合わせて、横を cos(yaw) で縮めた板を画面に向けて置く。
濡れ：体の白はつやを強く（coat・spec↑、rough↓）、マントは乾いた時より暗く・つやを足す（濡れた布）。
光：素材（IMG_9662）は海（南）を向いていて逆光ぎみ → 光は右の上から少し奥（体の右が明るく、背中の左は影の側で空の青い光）。
"""
import math

import numpy as np

import core
from cutB.beats import _sdf as S
from cutB.beats import _figures3d as F3

BODY_WET = S.Mat(core.c('E6EEF5'), spec=0.85, rough=0.16, sss=0.30, rim=1.0, coat=0.75)
CAPE_WET = S.Mat(core.c('86AACB'), spec=0.45, rough=0.32, sss=0.25, rim=0.7, coat=0.35, sheen=0.3)
LIGHT_BACK = (0.78, -0.58, -0.10)          # 右の上から少し奥
ENV_BEACH = dict(sun=np.array([1.0, 0.985, 0.96]) * 1.65, sky=F3._lin('A8C8E6') * 0.55, ground=F3._lin('A49C90') * 0.30,
                 rim=F3._lin('EAF6FF') * 1.25, rim_dir=(0.55, -0.83))
GROUND_Y = 6.2


def _yaw(p, yaw):
    r = math.radians(yaw)
    x, y, z = p
    return (x * math.cos(r) + z * math.sin(r), y, -x * math.sin(r) + z * math.cos(r))


def scene(yaw=-50.0, breath=0.0, head=0.0, slump=0.0, t=0.0):
    """breath：0..1 の息（肩が上がる）。head：頭の上げ下げ（+1＝水平線を見る、0＝少しうつむく）。slump：肩を落とす（ため息）。"""
    sc = S.Scene()
    Y = lambda p: _yaw(p, yaw)
    sh_y = -5.6 - 0.25 * breath + 0.45 * slump
    nk = (0.0, -6.6 - 0.2 * breath + 0.4 * slump, -0.9)
    # マント（背中側＝カメラ側）：濡れて体に張りつき、肩から砂の上まで重く垂れる。板（slab）は回せないので、
    # 肩から裾へ広がる円すい（中身の詰まった形）を背中に置き、体の前の半分（ローカル z < −0.6）を削る（前は開いている）。
    # 濡れたしわ：背中に沿った細い縦の筋を数本。
    top = (0.0, -6.0 + 0.4 * slump - 0.2 * breath, 0.2)
    sc.cone(Y(top), Y((0.0, 6.3, 1.2)), 2.45, 3.7, k=0.0, mat=1)
    sc.ellipsoid(Y((0.0, 6.0, 1.6)), (4.4, 0.45, 3.4), k=0.6, mat=1)          # 裾が砂の上に広がる
    def surf(y, dx):                                   # 円すいの表面（背中側）の z
        q = (y - top[1]) / (6.3 - top[1])
        r = 2.45 + 1.25 * q
        return top[2] + (1.2 - top[2]) * q + math.sqrt(max(r * r - dx * dx, 0.0)) - 0.10
    for dx, ph in ((-1.7, 0.0), (-0.5, 1.3), (0.8, 2.1), (1.9, 0.7)):
        w = 0.04 * math.sin(t * 1.3 + ph)
        x0, x1 = dx * 0.9, dx * 1.35
        sc.cone(Y((x0, -3.6, surf(-3.6, x0) + w)), Y((x1, 5.4, surf(5.4, x1) + w)), 0.13, 0.24, k=0.9, mat=1)
    rr = math.radians(yaw)
    Rm = np.array([[math.cos(rr), 0, math.sin(rr)], [0, 1, 0], [-math.sin(rr), 0, math.cos(rr)]])
    sc.obox(Y((0.0, 0.0, -6.6)), (9.0, 12.0, 6.0), 0.0, Rm=Rm, op='sub')
    sc.rbox((0.0, GROUND_Y + 6.0, 0.0), (20.0, 6.0, 20.0), 0.0, op='sub')        # 地面より下は切る
    # 胴（少し前かがみ）・胸・腰
    sc.cone(Y((0.0, 4.0, 0.3)), Y(nk), 1.85, 2.1, k=1.2, mat=0)
    sc.ellipsoid(Y((0.0, -3.9 + 0.3 * slump, -0.35)), (2.6, 3.0, 2.15), k=1.4, mat=0)
    sc.ellipsoid(Y((0.0, 4.4, 0.3)), (2.3, 1.7, 2.1), k=1.2, mat=0)
    # 首・頭（顔なし）。head で前へうつむく⇔水平線を見る
    hd = (0.0, -11.2 + 0.5 * head + 0.3 * slump, -2.7 + 1.8 * head)
    sc.cone(Y(nk), Y((0.0, -8.8 + 0.25 * head, -1.7 + 0.9 * head)), 1.25, 1.15, k=0.8, mat=0)
    sc.ellipsoid(Y(hd), (3.48, 3.85, 3.4), k=1.0, mat=0)
    sc.ellipsoid(Y((0.0, -7.3 + 0.4 * slump, -0.8)), (2.5, 1.15, 2.35), k=0.7, mat=1)     # えり（マントの留め）
    # 脚：ひざを立てる
    for s in (-1, 1):
        hip = (1.2 * s, 4.2, -0.4)
        knee = (1.55 * s, -0.6, -5.2)
        foot = (1.45 * s, 5.4, -7.7)
        sc.cone(Y(knee), Y(hip), 1.35, 1.55, k=0.9, mat=0)
        sc.cone(Y(foot), Y(knee), 1.2, 1.35, k=0.9, mat=0)
        sc.ellipsoid(Y((1.45 * s, 5.75, -8.5)), (1.0, 0.85, 1.8), k=0.8, mat=0)
        # 腕：肩 → ひじ → 手はひざの上
        sho = (2.3 * s, sh_y, -0.8)
        elb = (2.95 * s, -1.6 + 0.3 * slump, -3.7)
        hand = (1.0 * s, -1.4, -6.4)
        sc.cone(Y(elb), Y(sho), 1.05, 1.15, k=0.8, mat=0)
        sc.cone(Y(hand), Y(elb), 0.95, 1.05, k=0.8, mat=0)
        sc.sphere(Y(hand), 1.15, k=0.6, mat=0)
    return sc


def elbows(yaw=-50.0, slump=0.0):
    """しずくが落ちる点（ローカル）：両ひじの下側と、マントの左右の裾の少し上。"""
    pts = []
    for s in (-1, 1):
        x, y, z = _yaw((2.95 * s, -1.6 + 0.3 * slump + 1.1, -3.7), yaw)
        pts.append((x, y, z))
    return pts


def rgba(x, y, scale, yaw=-50.0, ss=2, **kw):
    """(x, y)：地面の上の体の中心（画面 px）＝お尻の下の砂の点。scale：px／ローカル単位。
    戻り値 (rgba, x0, y0, A, (cx, cy))：A と原点の板の中の位置は、しずくの点を画面へ写すため。"""
    Rx = int(math.ceil(11 * scale + 8))                         # 横：ひざ・マントの裾まで
    Up, Dn = int(math.ceil(23 * scale + 8)), int(math.ceil(4 * scale + 8))   # 上：頭のてっぺんまで、下：裾の広がり
    fx, fy = x - math.floor(x), y - math.floor(y)
    cx, cy = Rx + fx, Up + fy - GROUND_Y * scale               # ローカルの原点（胴のまんなか）は地面より上
    A = np.eye(2) * scale
    sc = scene(yaw=yaw, **kw)
    out, _z = S.render(sc, [BODY_WET, CAPE_WET], 2 * Rx, Up + Dn, A, (cx, cy), zs=scale, ss=ss, light=LIGHT_BACK,
                       env=ENV_BEACH, shadow_k=7.0, ao_d=0.9)
    return F3.cap_white(out), int(math.floor(x)) - Rx, int(math.floor(y)) - Up, A, (cx, cy)
