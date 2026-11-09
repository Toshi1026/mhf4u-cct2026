#!/usr/bin/env python3
"""フリー効果音の検索・取得（Openverse / Real SFX / earshot）→ exports/v3/sfx/<source>/

  python3 tools/v3/sfx_search.py "ocean waves" --source openverse --commercial --limit 10
  python3 tools/v3/sfx_search.py "ocean waves" --commercial --downloadable      # Freesound 以外（このコンテナで落とせる所）だけ
  python3 tools/v3/sfx_search.py --get 3                                         # 直前の検索結果の 3 番を取得（API を再度呼ばない）
  python3 tools/v3/sfx_search.py "soft whoosh" --source earshot                  # earshot のコード製効果音バンク（252 音）を検索
  python3 tools/v3/sfx_search.py "ocean" --source realsound                      # Real SFX 用の手元ライブラリ（取得済み素材）を検索

3 つの「ソース」は性質がまったく違う（詳しくは edl/v3/research/sfx_sources.md）：
  openverse  … https://openverse.org  CC ライセンス音源の横断検索。公開 API（キー不要、匿名は 20 回/分・200 回/日）。
  realsound  … Clipwing「Real SFX」（https://clipwing.pro/skills/real-sfx）。Claude Code 用スキルで、検索サービスではない。
               自分のライブラリの音を映像のカットに合わせて置く道具。ここでは「取得済みの手元ライブラリ」を検索する。
  earshot    … https://github.com/Leonxlnx/earshot  音を「聞く」・コードで「作る」MIT のツール（CLI / Python / MCP）。
               検索サービスではないが、同梱のコード製効果音バンクのカタログ（公開 JSON）を検索できる。
               音にするには earshot 本体（torch 必須）のインストールが要るので、ここでは render コマンドを出すだけ。

--get N は音声ファイル＋同名 .json（クレジット文つき）を保存し、exports/v3/sfx/CREDITS.md に追記する。
audio.py やキャプションのクレジットはこの .json の "credit_ja" / "attribution" を使う。
標準ライブラリのみ（ffmpeg は --wav のときだけ）。
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SFX = os.path.join(ROOT, 'exports', 'v3', 'sfx')
LAST = os.path.join(SFX, '.last_search.json')
CACHE = os.path.join(SFX, '.cache')
CREDITS = os.path.join(SFX, 'CREDITS.md')

# Wikimedia と Openverse はどちらも「連絡先の分かる User-Agent」を求めている（ブラウザのふりはしない）
UA = 'MAZE-v3-sfx_search/1.0 (CC-licensed SFX lookup for a small video project; python-urllib)'

OPENVERSE_API = 'https://api.openverse.org/v1/audio/'
OPENVERSE_WEB = 'https://openverse.org/search/audio?q='
EARSHOT_RAW = 'https://raw.githubusercontent.com/Leonxlnx/earshot/main/earshot/forge/bank/%s.json'
EARSHOT_BANKS = ['forge_v2', 'forge04', 'fit', 'pluck']
EARSHOT_SITE = 'https://leonxlnx.github.io/earshot-site/'
REALSFX_URL = 'https://clipwing.pro/skills/real-sfx'
REALSFX_REPO = 'https://github.com/Clipwing/skills/tree/main/skills/real-sfx'

# 商用（SNS 広告・屋外サイネージ）で使ってよいライセンス。SA は動画全体に継承義務が及ぶ恐れがあるので既定では外す
COMMERCIAL_OK = {'cc0', 'pdm', 'by', 'by-sa'}
DEFAULT_COMMERCIAL_LICENSES = 'cc0,by'
LICENSE_NAMES = {'cc0': 'CC0', 'pdm': 'Public Domain Mark', 'by': 'CC BY', 'by-sa': 'CC BY-SA', 'by-nc': 'CC BY-NC',
                 'by-nd': 'CC BY-ND', 'by-nc-sa': 'CC BY-NC-SA', 'by-nc-nd': 'CC BY-NC-ND', 'sampling+': 'Sampling+',
                 'nc-sampling+': 'NC Sampling+', 'mit': 'MIT'}
SOURCE_NAMES = {'freesound': 'Freesound', 'wikimedia_audio': 'Wikimedia Commons', 'jamendo': 'Jamendo'}
# このコンテナから音声ファイル本体を落とせなかった所（2026-11-07 確認：cdn.freesound.org は 403）
BLOCKED_HERE = {'freesound'}


# ---------------------------------------------------------------- 共通
def http_get(url, timeout=30, tries=4):
    """GET。429/503 は Retry-After（無ければ 5,10,20 秒）待って再試行。403 などはそのまま例外。"""
    for i in range(tries):
        req = urllib.request.Request(url, headers={'User-Agent': UA, 'Accept': '*/*'})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and i < tries - 1:
                wait = e.headers.get('Retry-After')
                wait = int(wait) if wait and wait.isdigit() else 5 * 2 ** i
                if wait > 60:                           # 長い待ちを指示されたら粘らずに諦める（相手の負荷を増やさない）
                    print('  … %d：Retry-After %d 秒。今回は取得をやめます（%d 秒以上あけて --get をやり直す）' % (
                        e.code, wait, wait), file=sys.stderr)
                    raise
                print('  … %d が返ったので %d 秒待って再試行（%d/%d）' % (e.code, wait, i + 1, tries - 1), file=sys.stderr)
                time.sleep(min(wait, 60))
                continue
            raise


def lic_name(lic, ver=None):
    n = LICENSE_NAMES.get(lic, (lic or '?').upper())
    return '%s %s' % (n, ver) if ver and lic not in ('pdm',) else n


def slug(s, n=48):
    s = re.sub(r'\.(wav|mp3|ogg|oga|flac|opus|aiff?)$', '', s or '', flags=re.I)
    s = re.sub(r'[^0-9A-Za-z]+', '_', s).strip('_').lower()
    return (s[:n] or 'sound').rstrip('_')


def fmt_dur(sec):
    if sec is None:
        return '?'
    sec = float(sec)
    return '%d:%02d' % divmod(round(sec), 60) if sec >= 60 else '%.1fs' % sec


def tokens(q):
    return [t for t in re.split(r'[^0-9a-zA-Z]+', q.lower()) if len(t) > 1]


# ---------------------------------------------------------------- Openverse
def search_openverse(q, a):
    p = {'q': q, 'page_size': min(a.limit, 20)}       # 匿名の page_size 上限は 20
    if a.commercial:
        p['license_type'] = 'commercial'
        p['license'] = a.license or DEFAULT_COMMERCIAL_LICENSES
    elif a.license:
        p['license'] = a.license
    if a.category:
        p['category'] = a.category
    if a.length:
        p['length'] = a.length
    if a.only:
        p['source'] = a.only
    elif a.downloadable:
        p['excluded_source'] = ','.join(sorted(BLOCKED_HERE))
    url = OPENVERSE_API + '?' + urllib.parse.urlencode(p)
    try:
        body, hdr = http_get(url)
    except urllib.error.HTTPError as e:
        print('Openverse API エラー %s（%s）。手動検索：%s' % (e.code, url, OPENVERSE_WEB + urllib.parse.quote(q)))
        return []
    d = json.loads(body)
    left = hdr.get('x-ratelimit-available-anon_sustained') or hdr.get('X-Ratelimit-Available-Anon_sustained')
    print('Openverse: %s 件ヒット（表示 %d）  API 残り %s/日   %s' % (
        d.get('result_count'), len(d.get('results', [])), left or '?', OPENVERSE_WEB + urllib.parse.quote(q)))
    out = []
    for r in d.get('results', [])[:a.limit]:
        lic = r.get('license')
        out.append({
            'source': 'openverse', 'id': r['id'], 'title': r.get('title') or '',
            'creator': r.get('creator') or '', 'creator_url': r.get('creator_url'),
            'license': lic, 'license_version': r.get('license_version'), 'license_url': r.get('license_url'),
            'provider': r.get('source') or r.get('provider'),
            'duration_s': (r['duration'] / 1000) if r.get('duration') else None,
            'filetype': r.get('filetype'), 'category': r.get('category'),
            'page': r.get('foreign_landing_url'), 'download_url': r.get('url'),
            'detail_url': r.get('detail_url'), 'attribution': r.get('attribution'),
            'commercial_ok': lic in COMMERCIAL_OK,
            'downloadable_here': (r.get('source') not in BLOCKED_HERE),
        })
    return out


# ---------------------------------------------------------------- earshot（コード製効果音バンクのカタログ）
def earshot_catalog():
    os.makedirs(os.path.join(CACHE, 'earshot'), exist_ok=True)
    items = []
    for b in EARSHOT_BANKS:
        path = os.path.join(CACHE, 'earshot', b + '.json')
        if not os.path.exists(path) or time.time() - os.path.getmtime(path) > 7 * 86400:
            try:
                body, _ = http_get(EARSHOT_RAW % b)
                with open(path, 'wb') as f:
                    f.write(body)
            except Exception as e:                      # noqa: BLE001
                print('earshot のカタログ %s を取れませんでした：%s' % (b, e), file=sys.stderr)
                if not os.path.exists(path):
                    continue
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        cats = d.get('categories', {})
        for eid, e in d.get('entries', {}).items():
            c = cats.get(e.get('category'), {})
            prof = e.get('profile') or {}
            meta = e.get('meta') or {}
            src = e.get('source') or {}
            text = ' '.join(str(x) for x in [eid, e.get('category'), e.get('member'), c.get('target'), c.get('use'),
                                             ' '.join(e.get('kinds') or c.get('kinds') or []), e.get('notes'),
                                             src.get('title') if isinstance(src, dict) else src])
            items.append({'id': eid, 'bank': b, 'text': text.lower(),
                          'desc': (c.get('target') or (src.get('title') if isinstance(src, dict) else '') or ''),
                          'use': c.get('use') or e.get('notes') or '',
                          'dur': prof.get('dur_s') or meta.get('dur_s') or meta.get('dur')})
    return items


def search_earshot(q, a):
    toks = tokens(q)
    alias = {'chime': ['chime', 'bell', 'notify', 'ding', 'mallet', 'glock'], 'waves': ['wave', 'water', 'drip'],
             'ocean': ['water', 'sea'], 'soft': ['soft', 'gentle', 'airy'], 'whoosh': ['whoosh', 'swish', 'swoosh']}
    out = []
    for it in earshot_catalog():
        score = 0
        for t in toks:
            alts = alias.get(t, [t])
            if any(x in it['id'].lower() for x in alts):          # id（カテゴリ/メンバー名）に当たるものを優先
                score += 3
            elif any(x in it['text'] for x in alts):
                score += 2 if t in it['text'] else 1
        if score:
            out.append((score, it))
    out.sort(key=lambda x: -x[0])
    print('earshot bank: %d 件（コードで作る音。録音ではない）  試聴：%s' % (len(out), EARSHOT_SITE))
    res = []
    for _, it in out[:a.limit]:
        res.append({'source': 'earshot', 'id': it['id'], 'title': it['id'], 'creator': 'earshot (Leon Lin)',
                    'creator_url': 'https://github.com/Leonxlnx/earshot', 'license': 'mit', 'license_version': None,
                    'license_url': 'https://github.com/Leonxlnx/earshot/blob/main/LICENSE', 'provider': 'earshot bank',
                    'duration_s': it['dur'], 'filetype': 'wav (render)', 'category': it['bank'],
                    'page': EARSHOT_SITE, 'download_url': None, 'desc': it['desc'], 'use': it['use'],
                    'commercial_ok': True, 'downloadable_here': False,
                    'attribution': 'Sound rendered with earshot (MIT), bank id %s' % it['id']})
    if not res:
        print('  バンクに該当なし（earshot は環境音・波のような長い録音系は持っていない）')
    return res


# ---------------------------------------------------------------- Real SFX（手元ライブラリの検索）
def search_realsound(q, a):
    print('Real SFX は Claude Code 用スキル（Clipwing、スクリプト MIT・同梱 UI 音 78 個は CC0）で、オンラインの検索サービスではありません。')
    print('  自分の音ライブラリから映像の「瞬間」に音を置く道具です：%s  /  %s' % (REALSFX_URL, REALSFX_REPO))
    print('  ここでは Real SFX に渡す「手元ライブラリ」= exports/v3/sfx/ の取得済み素材を検索します。')
    print('  Real SFX に登録する例：python scripts/setup.py init --library main="%s" --uisfx --index' % SFX)
    toks = tokens(q)
    res = []
    for dp, _, fs in os.walk(SFX):
        if '.cache' in dp:
            continue
        for fn in fs:
            if not fn.endswith('.json') or fn.startswith('.'):
                continue
            try:
                with open(os.path.join(dp, fn), encoding='utf-8') as f:
                    m = json.load(f)
            except Exception:                           # noqa: BLE001
                continue
            text = ' '.join(str(m.get(k, '')) for k in ('title', 'creator', 'query', 'provider')).lower()
            if not toks or any(t in text for t in toks):
                m = dict(m)
                m['source'] = 'realsound'
                m['downloadable_here'] = False
                m['page'] = m.get('page') or m.get('file')
                res.append(m)
    print('  手元ライブラリ：%d 件' % len(res))
    return res[:a.limit]


# ---------------------------------------------------------------- 表示・取得
def show(results):
    for i, r in enumerate(results, 1):
        lic = lic_name(r.get('license'), r.get('license_version'))
        flags = []
        if not r.get('commercial_ok', True):
            flags.append('商用不可')
        elif (r.get('license') or '').endswith('-sa'):
            flags.append('SA：動画全体に継承の恐れ→広告では使わない')
        if r.get('source') == 'openverse' and not r.get('downloadable_here'):
            flags.append('この環境からDL不可→手動')
        print('\n[%d] %s' % (i, r.get('title')))
        print('    作者: %s   ライセンス: %s %s   長さ: %s   形式: %s   提供元: %s' % (
            r.get('creator') or '-', lic, ('（%s）' % '・'.join(flags)) if flags else '', fmt_dur(r.get('duration_s')),
            r.get('filetype') or '-', SOURCE_NAMES.get(r.get('provider'), r.get('provider') or '-')))
        if r.get('desc'):
            print('    説明: %s / %s' % (r['desc'], r.get('use', '')))
        print('    ライセンスURL: %s' % (r.get('license_url') or '-'))
        print('    ページ: %s' % (r.get('page') or '-'))
        if r.get('source') == 'earshot':
            print('    作り方: earshot sfx render %s -o exports/v3/sfx/earshot/%s.wav' % (r['id'], slug(r['id'])))
        elif r.get('file'):
            print('    ファイル: %s' % r['file'])
        else:
            print('    DL: %s' % (r.get('download_url') or '-'))


def credit_ja(r):
    lic = lic_name(r.get('license'), r.get('license_version'))
    prov = SOURCE_NAMES.get(r.get('provider'), r.get('provider'))
    need = r.get('license') not in ('cc0', 'pdm', 'mit')
    s = '効果音：“%s” by %s（%s）%s' % (r.get('title'), r.get('creator') or '作者不明', lic,
                                    ' / %s' % prov if prov else '')
    return s, need


def ext_for(r):
    path = urllib.parse.urlparse(r.get('download_url') or '').path
    e = os.path.splitext(path)[1].lower().lstrip('.')
    if not e:
        e = {'mp32': 'mp3', 'ogx': 'ogg', 'oga': 'ogg'}.get(r.get('filetype') or '', r.get('filetype') or 'bin')
    return e


def get_item(r, query, want_wav):
    src = r['source']
    sub = SOURCE_NAMES.get(r.get('provider'), src).lower().replace(' ', '_') if src == 'openverse' else src
    outdir = os.path.join(SFX, src if src != 'openverse' else 'openverse', sub if src == 'openverse' else '')
    os.makedirs(outdir, exist_ok=True)
    base = '%s_%s' % (slug(r.get('title')), str(r.get('id'))[:8].replace('/', '_'))
    meta = {k: r.get(k) for k in ('source', 'id', 'title', 'creator', 'creator_url', 'license', 'license_version',
                                  'license_url', 'provider', 'duration_s', 'filetype', 'page', 'download_url',
                                  'detail_url', 'attribution', 'commercial_ok')}
    cj, need = credit_ja(r)
    meta.update({'query': query, 'credit_ja': cj, 'credit_required': need,
                 'license_name': lic_name(r.get('license'), r.get('license_version')),
                 'retrieved_at': datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                 'note': 'Openverse はライセンスを保証しない。使う前に page（元サイト）でライセンスを確認すること。'})
    if need:
        meta['credit_requirements'] = ('CC BY：タイトル・作者・ライセンス名とその URL（license_url）を表記し、'
                                       '切り出し・音量調整などの改変をしたら「改変あり」と書く。')
    if src == 'earshot':
        cmd = 'earshot sfx render %s -o %s' % (r['id'], os.path.join(outdir, slug(r['id']) + '.wav'))
        meta.update({'status': 'render_required', 'render_cmd': cmd, 'credit_required': False,
                     'credit_ja': '（不要）earshot のコード製効果音（MIT）'})
        print('earshot は torch＋モデル（約1.1GB）が要るため、この環境では音にしていません。インストール後に：\n  ' + cmd)
        _write_meta(os.path.join(outdir, slug(r['id']) + '.json'), meta)
        return
    if src == 'realsound':
        print('これは手元ライブラリの素材です：%s' % r.get('file'))
        return
    path = os.path.join(outdir, base + '.' + ext_for(r))
    # 手で置いた原音（Freesound の元 WAV など）があればそれを使う。拡張子はプレビューと違ってよい
    manual = [os.path.join(outdir, fn) for fn in sorted(os.listdir(outdir))
              if fn.startswith(base + '.') and not fn.endswith(('.json', '.48k.wav'))]
    try:
        if manual:
            path = manual[0]
            with open(path, 'rb') as f:
                body = f.read()
            meta['manually_placed'] = True
        else:
            if not r.get('downloadable_here'):
                raise urllib.error.HTTPError(r.get('download_url'), 403, 'blocked here (checked 2026-11-07)', {}, None)
            body, hdr = http_get(r['download_url'], timeout=120)
            ctype = (hdr.get('Content-Type') or hdr.get('content-type') or '')
            if ctype.startswith('text/html'):
                raise RuntimeError('音声でなく HTML が返った（%s）' % ctype)
            with open(path, 'wb') as f:
                f.write(body)
        meta.update({'status': 'downloaded', 'file': os.path.relpath(path, ROOT), 'bytes': len(body),
                     'sha256': hashlib.sha256(body).hexdigest()})
        print('保存：%s（%.1f MB）' % (os.path.relpath(path, ROOT), len(body) / 1e6))
        if want_wav and shutil.which('ffmpeg'):
            wav = os.path.splitext(path)[0] + '.48k.wav'
            subprocess.run(['nice', '-n', '15', 'ffmpeg', '-v', 'error', '-y', '-threads', '1', '-i', path,
                            '-ar', '48000', '-ac', '2', wav], check=True)
            meta['wav'] = os.path.relpath(wav, ROOT)
            print('WAV：%s' % meta['wav'])
            st = subprocess.run(['nice', '-n', '15', 'ffmpeg', '-hide_banner', '-threads', '1', '-i', wav,
                                 '-af', 'astats', '-f', 'null', '-'], capture_output=True, text=True).stderr
            pk = re.findall(r'Peak level dB: (-?[\d.]+|-inf)', st)
            rm = re.findall(r'RMS level dB: (-?[\d.]+|-inf)', st)
            if pk and rm:
                meta['peak_db'], meta['rms_db'] = float(pk[-1]), float(rm[-1])
                print('レベル：peak %.1f dBFS / RMS %.1f dBFS%s' % (
                    meta['peak_db'], meta['rms_db'],
                    '  ← ほぼ無音。持ち上げるとノイズが目立つので使わない方がよい' if meta['peak_db'] < -30 else ''))
    except Exception as e:                              # noqa: BLE001
        meta.update({'status': 'manual_download_required', 'error': str(e),
                     'manual': '元ページ %s を開いて%sダウンロードし、%s に置いてから --get をやり直す（拡張子は .wav なども可）' % (
                         r.get('page'), '（Freesound はログインが必要。元の WAV を推奨）' if r.get('provider') == 'freesound' else '',
                         os.path.relpath(path, ROOT))})
        print('取得できませんでした（%s）。手動で：%s' % (e, meta['manual']))
    _write_meta(os.path.join(outdir, base + '.json'), meta)


def _write_meta(path, meta):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print('メタ：%s' % os.path.relpath(path, ROOT))
    print('クレジット：%s%s' % (meta['credit_ja'], '' if meta.get('credit_required') else '（表記は任意）'))
    if meta.get('status') == 'downloaded':
        new = not os.path.exists(CREDITS)
        if not new and ('`%s`' % meta.get('file')) in open(CREDITS, encoding='utf-8').read():
            return                                      # 同じファイルの再取得では二重に書かない
        with open(CREDITS, 'a', encoding='utf-8') as f:
            if new:
                f.write('# v3 効果音クレジット（sfx_search.py が追記）\n\n')
            f.write('- %s%s — %s — %s — `%s`\n' % (meta['credit_ja'], '' if meta['credit_required'] else '（表記任意）',
                                                   meta.get('license_url'), meta.get('page'), meta.get('file')))


def main():
    ap = argparse.ArgumentParser(description='フリー効果音の検索・取得（Openverse / Real SFX / earshot）')
    ap.add_argument('query', nargs='?', help='検索語（英語推奨）。--get だけなら省略で直前の結果を使う')
    ap.add_argument('--source', default='openverse',
                    choices=['openverse', 'realsound', 'realsfx', 'earshot', 'all'])
    ap.add_argument('--commercial', action='store_true', help='商用OKだけ（既定で CC0・CC BY に限定）')
    ap.add_argument('--license', help='Openverse の license（例 cc0 / cc0,by / by-sa）')
    ap.add_argument('--category', choices=['sound_effect', 'music', 'podcast'], help='Openverse のカテゴリ')
    ap.add_argument('--length', choices=['shortest', 'short', 'medium', 'long'], help='Openverse の長さ')
    ap.add_argument('--only', help='Openverse の source を限定（freesound / wikimedia_audio / jamendo）')
    ap.add_argument('--downloadable', action='store_true', help='この環境から落とせない Freesound を除外')
    ap.add_argument('--limit', type=int, default=10)
    ap.add_argument('--get', type=int, metavar='N', help='結果 N 番を exports/v3/sfx/ に保存')
    ap.add_argument('--wav', action='store_true', help='取得後に 48kHz WAV も作る（ffmpeg、低優先度）')
    ap.add_argument('--json', action='store_true', help='結果を JSON で出す')
    a = ap.parse_args()
    src = 'realsound' if a.source == 'realsfx' else a.source
    os.makedirs(SFX, exist_ok=True)

    if a.query is None:
        if a.get is None:
            ap.error('query か --get が必要です')
        with open(LAST, encoding='utf-8') as f:
            last = json.load(f)
        results, query = last['results'], last['query']
    else:
        query = a.query
        srcs = ['openverse', 'earshot', 'realsound'] if src == 'all' else [src]
        results = []
        for s in srcs:
            results += {'openverse': search_openverse, 'earshot': search_earshot,
                        'realsound': search_realsound}[s](query, a)
        with open(LAST, 'w', encoding='utf-8') as f:
            json.dump({'query': query, 'results': results}, f, ensure_ascii=False, indent=1)

    if a.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
    elif a.query is not None:
        show(results)
    if a.get is not None:
        if not 1 <= a.get <= len(results):
            sys.exit('--get %d：結果は %d 件です' % (a.get, len(results)))
        print('\n--- 取得 [%d] %s' % (a.get, results[a.get - 1].get('title')))
        get_item(results[a.get - 1], query, a.wav)


if __name__ == '__main__':
    main()
