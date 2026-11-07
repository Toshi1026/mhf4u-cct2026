"""ナレーション候補の聞き比べ動画 → exports/v3/voice_cands/compare.mp4（1080x1920）と compare_index.md。

  nice -n 10 python3 tools/v3/vo_compare_pack.py [--order temp_suzumatsu,aivis_aida,...]

各候補：タイトルカード（候補名・エンジン／声・ライセンス）→ 01, 03, 04, 06, 08, 09 を 0.4 秒の間で続けて読む。
声は「声のある区間」で切り出し、1行ずつ -20 LUFS にそろえる（音量の差で良し悪しが決まらないように）。ピークは -1 dBFS まで。
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import vo_cands_eval as E  # noqa: E402

VC = E.VC
SR = 48000
W, H, FPS = 1080, 1920, 30
LINES = ['01', '03', '04', '06', '08', '09']
GAP, CARD, TAILPAD = 0.4, 2.4, 0.8
TARGET_LUFS = -20.0
FONT = os.path.join(ROOT, 'assets', 'fonts', 'NotoSansJP-VF.ttf')
TEXT = {
    '01': 'アニメの悪役は、パンチ一発で、空のかなたへ。', '03': 'たとえば、桂浜から殴って、10キロ先まで飛ばすなら。',
    '04': '速さは、時速1130キロ。旅客機より速い。', '06': '高さは、2500メートル。下からは、ほぼ点。キラーン。',
    '08': '落ちた先は、仁淀川の、河口の前の海。', '09': '目の前に建つのが、Seaside MAZE。',
}
INFO = {   # 表示用（エンジン／声、ライセンス・クレジット）
    'temp_suzumatsu': ('VOICEVOX', '雀松朱司（現在の仮ナレーション）', '商用可・クレジット必須「VOICEVOX:雀松朱司」'),
    'vv_kigashima_sourin': ('VOICEVOX', '麒ヶ島宗麟（アクセント・間を手で調整）', '商用可・クレジット必須「VOICEVOX:麒ヶ島宗麟」'),
    'vv_kenzaki_mesuo': ('VOICEVOX', '剣崎雌雄（アクセント・間を手で調整）', '商用可・クレジット必須「VOICEVOX:剣崎雌雄」'),
    'vv_kenzaki_mesuo_s114': ('VOICEVOX', '剣崎雌雄（調整済み・話速1.14）', '商用可・クレジット必須「VOICEVOX:剣崎雌雄」'),
    'aivis_aida': ('AivisSpeech', '阿井田 茂（Calm）', 'ACML 1.0：商用可・クレジット任意'),
    'aivis_honoka': ('AivisSpeech', 'ほのか', 'ACML 1.0：商用可・クレジット任意（作者は表記を依頼）'),
    'aivis_lux': ('AivisSpeech', 'Lux', 'ACML 1.0：商用可・クレジット任意'),
    'aivis_morioki': ('AivisSpeech', 'morioki', 'ACML 1.0：商用可・クレジット任意'),
}


def font(size, weight=500):
    f = ImageFont.truetype(FONT, size)
    try:
        f.set_variation_by_axes([weight])
    except Exception:
        pass
    return f


def wrap(draw, text, f, maxw):
    """読点・句点のあとで折る（語の途中で折らない）。1つの句が長すぎるときだけ字で折る"""
    import re
    toks = [t for t in re.split(r'(?<=[、。）])', text) if t]
    out, cur = [], ''
    for tk in toks:
        if draw.textlength(cur + tk, font=f) <= maxw:
            cur += tk
            continue
        if cur:
            out.append(cur)
            cur = ''
        for ch in tk:
            if draw.textlength(cur + ch, font=f) > maxw:
                out.append(cur)
                cur = ch
            else:
                cur += ch
    if cur:
        out.append(cur)
    return out


def card(path, head, sub, lic, line_key=None, line_text=None, big=True):
    im = Image.new('RGB', (W, H), (16, 22, 30))
    d = ImageDraw.Draw(im)
    if big:
        y = 640
        for ln in wrap(d, head, font(84, 800), W - 160):
            d.text((W // 2, y), ln, font=font(84, 800), fill=(255, 255, 255), anchor='mm')
            y += 110
        y += 30
        for ln in wrap(d, sub, font(52, 500), W - 160):
            d.text((W // 2, y), ln, font=font(52, 500), fill=(170, 210, 235), anchor='mm')
            y += 72
        y += 40
        for ln in wrap(d, lic, font(36, 400), W - 160):
            d.text((W // 2, y), ln, font=font(36, 400), fill=(150, 150, 150), anchor='mm')
            y += 52
        d.text((W // 2, 1500), 'この後：01・03・04・06・08・09 を続けて読みます', font=font(36, 400), fill=(120, 120, 120),
               anchor='mm')
    else:
        d.text((80, 200), head, font=font(60, 800), fill=(255, 255, 255))
        d.text((80, 290), sub, font=font(38, 500), fill=(170, 210, 235))
        d.text((W // 2, 820), line_key, font=font(64, 700), fill=(120, 170, 200), anchor='mm')
        y = 960
        for ln in wrap(d, line_text, font(68, 600), W - 180):
            d.text((W // 2, y), ln, font=font(68, 600), fill=(255, 255, 255), anchor='mm')
            y += 100
        d.text((80, H - 200), lic, font=font(32, 400), fill=(130, 130, 130))
    im.save(path)


def prep(path):
    import pyloudnorm as pyln
    x = E.read(path).astype(np.float64)
    t0, t1 = E.speech_span(x)
    a, b = max(0, int((t0 - 0.02) * SR)), min(len(x), int((t1 + 0.06) * SR))
    y = x[a:b].copy()
    f = int(0.005 * SR)
    y[:f] *= np.linspace(0, 1, f)
    y[-f:] *= np.linspace(1, 0, f)
    lu = pyln.Meter(SR).integrated_loudness(y)
    y *= 10 ** ((TARGET_LUFS - lu) / 20)
    pk = np.abs(y).max()
    if pk > 10 ** (-1 / 20):
        y *= 10 ** (-1 / 20) / pk
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--order', default=','.join(n for n, _, _ in E.SETS))
    ap.add_argument('--out', default=os.path.join(VC, 'compare.mp4'))
    a = ap.parse_args()
    dirs = {n: d for n, d, _ in E.SETS}
    order = a.order.split(',')
    letters = 'ABCDEFGHIJ'
    td = tempfile.mkdtemp(dir=os.environ.get('TMPDIR'))
    segs, audio, idx = [], [], []
    t = 0.0
    li = 0
    for name in order:
        eng, voice, lic = INFO[name]
        if name.startswith('temp'):
            head = '現状：' + voice.split('（')[0]
        else:
            head = '候補%s：%s' % (letters[li], voice.split('（')[0])
            li += 1
        sub = '（%s／%s）' % (eng, voice)
        p = os.path.join(td, f'{name}_card.png')
        card(p, head, sub, lic)
        segs.append((p, CARD))
        audio.append(np.zeros(int(CARD * SR)))
        idx.append((t, head, sub, lic))
        t += CARD
        for k in LINES:
            y = prep(os.path.join(dirs[name], k + '.wav'))
            seg = np.concatenate([y, np.zeros(int(GAP * SR))])
            p = os.path.join(td, f'{name}_{k}.png')
            card(p, head, sub, lic, line_key=k, line_text=TEXT[k], big=False)
            dur = len(seg) / SR
            segs.append((p, dur))
            audio.append(seg)
            t += dur
        audio.append(np.zeros(int(TAILPAD * SR)))
        segs[-1] = (segs[-1][0], segs[-1][1] + TAILPAD)
        t += TAILPAD
    wav = np.concatenate(audio)
    # 映像の長さをフレームにそろえる（音とずれないよう、各区間の境目を累積でフレームに丸める）
    lst = os.path.join(td, 'list.txt')
    with open(lst, 'w') as f:
        acc, prev = 0.0, 0
        for p, d in segs:
            acc += d
            fr = round(acc * FPS)
            f.write(f"file '{p}'\nduration {(fr - prev) / FPS:.6f}\n")
            prev = fr
        f.write(f"file '{segs[-1][0]}'\n")
    import wave
    wp = os.path.join(td, 'a.wav')
    with wave.open(wp, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(wav, -1, 1) * 32767).astype('<i2').tobytes())
    subprocess.run(['nice', '-n', '10', 'ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', lst, '-i', wp,
                    '-vf', f'fps={FPS},format=yuv420p', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
                    '-threads', '2', '-c:a', 'aac', '-b:a', '192k', '-ac', '2', '-ar', '48000', '-shortest',
                    '-movflags', '+faststart', a.out], check=True)
    md = ['# 聞き比べ（compare.mp4）の目次', '',
          '各候補：01・03・04・06・08・09 を 0.4 秒の間で続けて読む。1行ずつ -20 LUFS にそろえてある（音量差で選ばないように）。', '',
          '| 開始 | 候補 | エンジン／声 | ライセンス |', '|---|---|---|---|']
    for t0, h, s, l in idx:
        md.append('| %d:%05.2f | %s | %s | %s |' % (t0 // 60, t0 % 60, h, s[1:-1], l))
    md.append('')
    md.append('全体 %.1f 秒' % (len(wav) / SR))
    open(os.path.join(os.path.dirname(a.out), 'compare_index.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
