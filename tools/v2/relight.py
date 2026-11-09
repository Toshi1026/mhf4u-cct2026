"""昼（28）→ 夜（12）。夜の写真を昼の画角に合わせ、空が暮れて、明かりが左から順に灯り、最後に看板が光る。"""
import cv2
import numpy as np

import common as C

# 対応点（どちらも 900×600 に縮めた座標）：屋根の頂点、窓3・窓4、看板の「南風」、帯の「B」、帯の最後の点
DAY_PTS = [(425, 135), (457, 413), (510, 413), (423, 345), (152, 480), (748, 488)]
NIGHT_PTS = [(430, 80), (457, 322), (505, 322), (427, 258), (155, 385), (693, 389)]


def prepare(day_key=28, night_key=12):
    day = C.grade(C.load(day_key), con=0.05, sat=1.03)
    night = C.load(night_key, long_side=2048)
    h, w = day.shape[:2]
    nh, nw = night.shape[:2]
    sd = np.float32(DAY_PTS) * (w / 900.0)
    sn = np.float32(NIGHT_PTS) * (nw / 900.0)
    H, _ = cv2.findHomography(sn, sd, 0)
    nw_ = cv2.warpPerspective(night, H, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpPerspective(np.ones((nh, nw), np.float32), H, (w, h))
    valid = cv2.GaussianBlur(cv2.erode(valid, np.ones((31, 31), np.uint8)), (0, 0), w / 40)
    nw_ = np.clip(nw_ * 1.45, 0, 1)                    # 夜の写真を少し明るく（見やすさ）
    # 夜の明かり（昼より明るい・暖色の光源）
    ld = day.mean(axis=2)
    ln = nw_.mean(axis=2)
    light = np.clip((ln - ld * 0.6 - 0.18) / 0.35, 0, 1) * valid
    light = cv2.GaussianBlur(light, (0, 0), w / 900)
    glow = cv2.GaussianBlur(nw_ * light[..., None], (0, 0), w / 120)
    # 点灯の順番：帯の電球は左→右、入口、窓、最後に看板（「南風」のまわり）
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    order = 0.15 + 0.45 * xs / w
    ex, ey = sd[3]
    sign = np.exp(-(((xs - ex) / (w * 0.09)) ** 2 + ((ys - ey) / (h * 0.16)) ** 2))
    order = order * (1 - sign) + 0.78 * sign
    # 夜の空の色（明かりのない所の平均）。画角の外は、昼の画をこの色へ暮れさせる
    sky = np.median(nw_[(light < 0.05) & (valid > 0.9)][::50], axis=0).astype(np.float32)
    return dict(day=day, night=nw_, valid=valid, light=light, glow=glow, order=order, H=H, pts=sd, sky=sky)


def frame(P, t):
    """t：0（昼）→ 1（夜）。"""
    day, night, light, order, valid = P['day'], P['night'], P['light'], P['order'], P['valid']
    # 空と建物が暮れていく（青みを帯びて暗く）
    dusk = C.ease(np.clip(t / 0.7, 0, 1), 'smooth')
    v = valid[..., None]
    # 昼の画を、夜の空の色へ暮れさせる（明るさの比をそろえる）
    lum = day.mean(axis=2, keepdims=True)
    dark = day * (1 - dusk) + (P['sky'] * (0.6 + 0.8 * lum)) * dusk
    # 明かりの消えた夜（光源の所は、暮れた昼の画で埋める）
    Lm = light[..., None]
    unlit = night * (1 - Lm) + dark * Lm
    base = dark * (1 - dusk * v) + unlit * (dusk * v)
    # 明かり：それぞれの順番の時刻に、0.08 の間でふっと灯る（少しだけちらつく）
    on = np.clip((t - order) / 0.08, 0, 1)
    on = on * on * (3 - 2 * on)
    flick = 1 + 0.25 * np.sin((t - order) * 90) * np.exp(-np.clip(t - order, 0, None) * 30)
    L = (light * on * np.clip(flick, 0, 1.3))[..., None]
    out = base * (1 - L) + night * L
    out = out + P['glow'] * (on[..., None] * 0.35) * C.ease(np.clip((t - 0.5) / 0.5, 0, 1), 'smooth')
    return np.clip(out, 0, 1)


def night_door(P):
    """夜の入口（いちばん明るい所）の位置を、昼の画の座標で返す。"""
    L = P['light'].copy()
    h, w = L.shape
    L[: int(h * 0.55)] = 0                              # 下半分（入口）だけ
    L = cv2.GaussianBlur(L, (0, 0), w / 60)
    y, x = np.unravel_index(np.argmax(L), L.shape)
    return float(x), float(y)
