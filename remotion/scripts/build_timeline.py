#!/usr/bin/env python3
"""書き出し済みのマニフェスト・EDLから、Remotion用のタイムライン（src/timeline.json）を作る。

  python3 scripts/build_timeline.py          # src/timeline.json を作り、public/ にリンクを張る
  python3 scripts/build_timeline.py --check  # 作らずに、クリップのコマ数だけ確かめる

読むもの（どれも書き換えない）：
  exports/capcut/signage_manifest.json            … サイネージ30秒・15秒（承認済み）
  exports/capcut/proposals/render_log_proposal.json … サイネージのケーキ版のクリップ一覧
  edl/edl_result.json（edls.signage_cake / signage15_cake / ig30 / ig15） … カットの秒
  edl/mockups/signage_spec.py（timing()）          … サイネージの右下マーク・ENDの出方の式
  exports/capcut/ig_manifest.json                 … IGの透過PNGの秒・フェード（v8でも同じ）
  clip_overrides.json（任意）                       … 撮り直したクリップへの差し替え

public/ には素材フォルダへのシンボリックリンクだけを置く（動画をコピーしない）。
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RM = os.path.dirname(HERE)                       # remotion/
ROOT = os.path.dirname(RM)                       # maze/
CAP = os.path.join(ROOT, 'exports', 'capcut')
PROP = os.path.join(CAP, 'proposals')
PUBLIC = os.path.join(RM, 'public')
OUT_JSON = os.path.join(RM, 'src', 'timeline.json')
OVERRIDES = os.path.join(RM, 'clip_overrides.json')
FPS = 30

sys.path.insert(0, os.path.join(ROOT, 'edl', 'mockups'))
import signage_spec as S  # noqa: E402  timing(E, T) と FADE（サイネージ4本で共通の式）


def ffprobe_bin():
    base = os.path.join(RM, 'node_modules', '@remotion')
    for d in sorted(os.listdir(base)):
        if d.startswith('compositor-'):
            p = os.path.join(base, d, 'ffprobe')
            if os.path.exists(p):
                return p, os.path.join(base, d)
    return None, None


FFPROBE, FFLIB = ffprobe_bin()


def probe(path):
    """(幅, 高さ, コマ数, fps) を返す。"""
    if not FFPROBE:
        return None
    env = dict(os.environ, LD_LIBRARY_PATH=FFLIB)
    r = subprocess.run([FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-count_packets',
                        '-show_entries', 'stream=width,height,nb_read_packets,r_frame_rate',
                        '-of', 'json', path], capture_output=True, text=True, env=env, check=True)
    s = json.loads(r.stdout)['streams'][0]
    n, d = s['r_frame_rate'].split('/')
    return int(s['width']), int(s['height']), int(s['nb_read_packets']), float(n) / float(d)


def fr(sec):
    return int(round(sec * FPS))


# ---------------------------------------------------------------------------
# 公開用のパス（public/ からの相対）。public/<名前> → 実フォルダ のリンクで解決する
# ---------------------------------------------------------------------------
LINKS = {
    'capcut': CAP,   # exports/capcut 全体を1本のリンクで見せる（コピーしない）
}


def pub(abs_path):
    abs_path = os.path.abspath(abs_path)
    rel = os.path.relpath(abs_path, CAP)
    if rel.startswith('..'):
        # capcut の外の素材（撮り直しのクリップなど）は、ファイルごとにリンクを張る
        name = 'extra/' + re.sub(r'[^A-Za-z0-9._+-]', '_', os.path.relpath(abs_path, ROOT))
        LINKS[name] = abs_path
        return name
    return 'capcut/' + rel


def make_links():
    os.makedirs(PUBLIC, exist_ok=True)
    for name, target in LINKS.items():
        p = os.path.join(PUBLIC, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if os.path.islink(p):
            if os.readlink(p) == target:
                continue
            os.unlink(p)
        elif os.path.exists(p):
            raise SystemExit('public/%s がリンクではありません。消してからやり直してください' % name)
        os.symlink(target, p)


# ---------------------------------------------------------------------------
# 差し替え（clip_overrides.json）
#   { "Signage30Cake": { "5": "exports/capcut/proposals/signage30_cake/05_IMG_9700_cake.mp4" } }
#   パスは maze/ からの相対か絶対パス。カット番号は EDL の n
# ---------------------------------------------------------------------------
def load_overrides():
    if not os.path.exists(OVERRIDES):
        return {}
    d = json.load(open(OVERRIDES, encoding='utf-8'))
    return {k: v for k, v in d.items() if not k.startswith('_')}


OVR = load_overrides()


def resolve(comp, cut_n, default_abs):
    o = OVR.get(comp, {}).get(str(cut_n))
    if o:
        p = o if os.path.isabs(o) else os.path.join(ROOT, o)
        return os.path.abspath(p), True
    return default_abs, False


def clip_entry(comp, n, abs_path, start, count, placeholder, label, W, H, problems):
    abs_path, overridden = resolve(comp, n, abs_path)
    if overridden:
        placeholder = False
    if not os.path.exists(abs_path):
        problems.append('%s カット%d：ファイルがありません %s' % (comp, n, abs_path))
    else:
        p = probe(abs_path)
        if p:
            w, h, nf, fps = p
            if (w, h) != (W, H):
                problems.append('%s カット%d：%dx%d（%dx%dが必要）%s' % (comp, n, w, h, W, H, abs_path))
            if abs(fps - FPS) > 0.01:
                problems.append('%s カット%d：%.3ffps（30fpsが必要）%s' % (comp, n, fps, abs_path))
            if nf < count:
                problems.append('%s カット%d：%dコマしかありません（%dコマ必要）%s' % (comp, n, nf, count, abs_path))
            elif nf > count:
                problems.append('（注意）%s カット%d：%dコマあります。頭から%dコマだけ使います %s'
                                % (comp, n, nf, count, abs_path))
    return dict(cut=n, src=pub(abs_path), file=os.path.relpath(abs_path, ROOT), from_=start,
                durationInFrames=count, placeholder=bool(placeholder), overridden=overridden, label=label)


def kf_overlay(file_abs, label, keyframes_s, in_s, out_s):
    """keyframes_s: [(秒, 0〜1)]。コマ番号に直して持つ（どれも整数コマになる）。"""
    kfs = []
    for t, v in keyframes_s:
        f = t * FPS
        if abs(f - round(f)) > 1e-6:
            raise SystemExit('キーフレームが整数コマにのっていません：%s %.4f秒' % (label, t))
        kfs.append([int(round(f)), round(float(v), 6)])
    return dict(label=label, src=pub(file_abs), file=os.path.relpath(file_abs, ROOT),
                from_=fr(in_s), to=fr(out_s), keyframes=kfs)


# ---------------------------------------------------------------------------
# サイネージ
# ---------------------------------------------------------------------------
def signage_overlays(E, T):
    tm = S.timing(E, T)
    ov = os.path.join(CAP, 'overlays')
    c, e = tm['corner'], tm['end']
    return [
        kf_overlay(os.path.join(ov, 'signage_corner_1920x1080.png'), '右下のマーク', c, c[0][0], c[-1][0]),
        kf_overlay(os.path.join(ov, 'signage_end_1920x1080.png'), 'END', e, e[0][0], e[-1][0]),
    ]


def signage_approved(comp, key, problems):
    m = json.load(open(os.path.join(CAP, 'signage_manifest.json'), encoding='utf-8'))
    e = m['edits'][key]
    T = e['duration_s']
    E = e['end_cut_start_s']
    clips = [clip_entry(comp, c['cut'], os.path.join(CAP, c['file']), c['timeline_in_frame'], c['frames'],
                        c['placeholder'], c.get('source', ''), 1920, 1080, problems) for c in e['clips']]
    ovs = signage_overlays(E, T)
    # マニフェストに書かれたキーフレームと、式から出した値が同じかを確かめる
    for o, mo in zip(ovs, e['overlays']):
        want = [[k['frame'], k['opacity_pct'] / 100.0] for k in mo['opacity_keyframes']]
        if want != o['keyframes']:
            problems.append('%s %s：マニフェストのキーフレーム %s と式 %s が違います' % (comp, o['label'], want, o['keyframes']))
    return dict(width=1920, height=1080, durationInFrames=fr(T), clips=clips, overlays=ovs, audio=None,
                version=m['edl_versions']['30' if key == 'signage30' else '15'])


def signage_cake(comp, edl_key, log_key, problems):
    d = json.load(open(os.path.join(ROOT, 'edl', 'edl_result.json'), encoding='utf-8'))
    edl = d['edls'][edl_key]['edl']
    log = json.load(open(os.path.join(PROP, 'render_log_proposal.json'), encoding='utf-8'))[log_key]
    files = {c['cut']: c for c in log}
    T = float(edl['target_s'])
    clips = []
    for c in edl['cuts']:
        start, count = fr(c['rec_in']), fr(c['rec_out']) - fr(c['rec_in'])
        lc = files[c['n']]
        if lc['frames'] != count:
            problems.append('%s カット%d：EDLは%dコマ、書き出しログは%dコマ' % (comp, c['n'], count, lc['frames']))
        clips.append(clip_entry(comp, c['n'], os.path.join(CAP, lc['file']), start, count, c.get('placeholder'),
                                c.get('source', ''), 1920, 1080, problems))
    E = float(edl['cuts'][-1]['rec_in'])   # ENDのカットの頭
    return dict(width=1920, height=1080, durationInFrames=fr(T), clips=clips, overlays=signage_overlays(E, T),
                audio=None, version=edl['version'].split('（')[0])


# ---------------------------------------------------------------------------
# Instagram（v8）
# ---------------------------------------------------------------------------
def ig(comp, key, problems):
    d = json.load(open(os.path.join(ROOT, 'edl', 'edl_result.json'), encoding='utf-8'))
    edl = d['edls'][key]['edl']
    man = json.load(open(os.path.join(CAP, 'ig_manifest.json'), encoding='utf-8'))[key]
    cdir = os.path.join(PROP, 'ig_v8_%s' % key)
    listing = sorted(os.listdir(cdir))
    T = float(edl['target_s'])
    clips = []
    for c in edl['cuts']:
        cands = [f for f in listing if f.startswith('%02d_' % c['n']) and f.endswith('.mp4')]
        if len(cands) != 1:
            problems.append('%s カット%d：%s に %02d_*.mp4 が%d本あります' % (comp, c['n'], cdir, c['n'], len(cands)))
            continue
        start, count = fr(c['rec_in']), fr(c['rec_out']) - fr(c['rec_in'])
        clips.append(clip_entry(comp, c['n'], os.path.join(cdir, cands[0]), start, count, c.get('placeholder'),
                                c.get('source', ''), 1080, 1920, problems))
    ovs = []
    for o in man['overlays']:
        kfs = [(t, v / 100.0) for t, v in o['opacity_keyframes']]
        ovs.append(kf_overlay(os.path.join(CAP, o['file']), os.path.basename(o['file']).split('_1080')[0],
                              kfs, o['in_s'], o['out_s']))
    wav = os.path.join(PROP, 'ig_ambience_%s_v8.wav' % key)
    audio = dict(src=pub(wav), file=os.path.relpath(wav, ROOT), volume=1.0)
    return dict(width=1080, height=1920, durationInFrames=fr(T), clips=clips, overlays=ovs, audio=audio,
                version=edl['version'].split('「')[0])


# ---------------------------------------------------------------------------
DELIVERABLES = [
    # (Composition id, 書き出しファイル名, 見本, 作り方)
    ('Signage30', 'MAZE_signage30.mp4', 'exports/capcut/preview_signage30.mp4',
     lambda p: signage_approved('Signage30', 'signage30', p)),
    ('Signage15', 'MAZE_signage15.mp4', 'exports/capcut/preview_signage15.mp4',
     lambda p: signage_approved('Signage15', 'signage15', p)),
    ('Signage30Cake', 'MAZE_signage30_cake.mp4', 'exports/capcut/proposals/preview_signage30_cake.mp4',
     lambda p: signage_cake('Signage30Cake', 'signage_cake', '30', p)),
    ('Signage18Cake', 'MAZE_signage18_cake.mp4', 'exports/capcut/proposals/preview_signage15_cake.mp4',
     lambda p: signage_cake('Signage18Cake', 'signage15_cake', '15', p)),
    ('IG30', 'MAZE_ig30.mp4', 'exports/capcut/proposals/ig_preview_ig30.mp4', lambda p: ig('IG30', 'ig30', p)),
    ('IG15', 'MAZE_ig15.mp4', 'exports/capcut/proposals/ig_preview_ig15.mp4', lambda p: ig('IG15', 'ig15', p)),
]


def main():
    problems = []
    comps = []
    for cid, out, ref, build in DELIVERABLES:
        c = build(problems)
        # すき間・重なりがないか
        t = 0
        for k in c['clips']:
            if k['from_'] != t:
                problems.append('%s カット%d：%dコマ目から始まるはずが%dコマ目' % (cid, k['cut'], t, k['from_']))
            t = k['from_'] + k['durationInFrames']
        if t != c['durationInFrames']:
            problems.append('%s：クリップの合計%dコマ、長さ%dコマ' % (cid, t, c['durationInFrames']))
        c.update(id=cid, fps=FPS, output=out, reference=ref, hasAudio=c['audio'] is not None,
                 placeholders=[k['cut'] for k in c['clips'] if k['placeholder']])
        comps.append(c)
    # JSONのキーは from_ → from に直す
    for c in comps:
        for k in c['clips'] + c['overlays']:
            k['from'] = k.pop('from_')
    fatal = [p for p in problems if not p.startswith('（注意）')]
    for p in problems:
        print(p, file=sys.stderr)
    if fatal:
        raise SystemExit('問題があるので timeline.json は作りませんでした（%d件）' % len(fatal))
    if '--check' in sys.argv:
        print('OK（%d本）' % len(comps))
        return
    make_links()
    doc = dict(_generated_by='scripts/build_timeline.py（手で直さないこと）', fps=FPS, compositions=comps)
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)
    for c in comps:
        ph = '・仮スレート：カット%s' % ','.join(map(str, c['placeholders'])) if c['placeholders'] else ''
        print('%-14s %dx%d %4dコマ クリップ%2d本 PNG%d枚 音声%s%s' % (
            c['id'], c['width'], c['height'], c['durationInFrames'], len(c['clips']), len(c['overlays']),
            'あり' if c['hasAudio'] else 'なし', ph))
    print('→', os.path.relpath(OUT_JSON, ROOT))


if __name__ == '__main__':
    main()
