# フリー効果音の3ツール：オープンバース／リアルサウンド／イヤーショット（2026-11-07 調査）

ユーザーの依頼：「フリーで使える音を検索してくれるオープンバース、リアルサウンド、イヤーショット、これらの三つを入れてください」

## 結論（先に）

3つはセットで紹介されている（出典は下記）。ただし**「音を検索するサービス」はOpenverseだけ**で、残りの2つは性質がまったく違う。

| 呼び名 | 正体 | 種類 | 音のライセンス | API / MCP | このコンテナから |
|---|---|---|---|---|---|
| オープンバース | **Openverse**（WordPress Foundation）https://openverse.org | CCライセンス音源の横断**検索エンジン**（約522万件：Wikimedia Commons 399万／Jamendo 64万／Freesound 59万） | 1件ごとに違う（CC0、CC BY、CC BY-SA、CC BY-NC など） | 公開API https://api.openverse.org/v1/ （キー不要。匿名は20回/分・200回/日）。MCPはなし | API OK。**音声ファイル本体はFreesound分が403、Wikimedia分が429（レート制限）で落とせない**。Jamendoは落とせる（ただし中身はほぼ音楽） |
| リアルサウンド | **Real SFX**（Clipwing）https://clipwing.pro/skills/real-sfx ／ https://github.com/Clipwing/skills/tree/main/skills/real-sfx | **Claude Code用スキル**。自分の音ライブラリから音を選び、映像のカットや動きに合わせてフレーム単位で置く。新しいMP4、効果音だけのWAV、キューのレポートを出す。**音の検索サービスではないし、音はほとんど同梱されていない** | スクリプトはMIT。同梱のUI音78個（uisfx）はCC0。ほかは「自分で権利を持っているライブラリ」を使う | APIもMCPもない（スキルとして動く） | GitHub raw とサイトは読める（github.com 本体はこのセッションでは403） |
| イヤーショット | **earshot**（Leon Lin）https://github.com/Leonxlnx/earshot | AIに音を「聞かせる」（説明、合致度、波形のどこで鳴るか、批評）＋**コードで効果音を作る**（forge／252音のバンク／kit）道具。**検索サービスではない**（`rank` は手元のファイルを説明文で並べ替える機能） | コードはMIT。バンクの音はコードで合成（うち15音はFreesoundのCC0録音から特徴を取ったもの。サンプル自体は含まない） | CLI、Python API、**MCPサーバーあり**（`earshot mcp`） | GitHub raw は読める。本体の動作には torch と約1.1GBのモデルが必要なので、**今回はインストールしていない**（CPU負荷と容量のため） |

> ※ 同じ名前の別物に注意：「Earshot SFX」（Paul Urmson の1996年のCDライブラリで、itch.io で配布）は今回のearshotとは別物。「リアルサウンド」という名前の音楽メディアや効果音サイトもあるが、それも別物。

### 出典（3つをまとめて紹介している記事）
- Vivian Lin「Give Claude-made videos real sound effects: Openverse, Real SFX and earshot, three free tools」（数字とコマンドは2026-10-27時点で確認したもの）
  https://vivian.hobbyland-group.com/en/article/free-sound-effects-claude-video-openverse-real-sfx-earshot
  （中国語版：https://vivian.hobbyland-group.com/article/free-sound-effects-claude-video-openverse-real-sfx-earshot 。日本語版のURL（/ja/…）は404。著者のInstagramは @vivianlin.tech）
- 同じ3つを扱った記事：「3 Free Tools That Add Pro Sound Effects to Your Claude Edits」https://www.freshstack.ai/resources/claude-code-sound-effects
- 記事内の注意書き：著者はReal SFXとearshotを自分の環境では動かしておらず、コマンドは公式READMEから引用している

---

## 1. Openverse（オープンバース）

- サイト：https://openverse.org （このコンテナからは403。ブラウザ用のページ）
- API：https://api.openverse.org/v1/audio/ （このコンテナから200）。統計：https://api.openverse.org/v1/audio/stats/
- コード：https://github.com/WordPress/openverse （MIT）
- 主なパラメータ（2026-11-07 に実際に動かして確認）：`q`、`license=cc0,by`、`license_type=commercial`（`modification` も指定できる）、`category=sound_effect|music|podcast`、`length=shortest|short|medium|long`、`source=freesound|wikimedia_audio|jamendo`、`excluded_source=…`、`page_size`（匿名は最大20）
- レート制限（レスポンスヘッダーで確認）：`x-ratelimit-limit-anon_burst: 20/min`、`anon_sustained: 200/day`。もっと使いたいときは `/v1/auth_tokens/register/` でOAuthのクライアントを登録する（任意）
- 1件ごとに返る情報：`title`、`creator`、`license`、`license_version`、`license_url`、`foreign_landing_url`（元ページ）、`url`（音声ファイル）、`duration`（ミリ秒）、`attribution`（そのまま使えるクレジット文）
- **注意**：Freesound由来の `url` は **128kbps のMP3プレビュー**（`cdn.freesound.org/previews/…-hq.mp3`）。元のWAVはFreesoundにログインしないと落とせない
- **注意**：Openverseはライセンスが正しいことを保証しない。使う前に元ページでライセンスを確認する
- `category=sound_effect` で絞ると、Freesoundの多くは `category` が空なので結果がかなり減る（「soft whoosh」は0件になった）。既定では絞らない

### ライセンス別：SNS広告・屋外サイネージでの使い方

| ライセンス | 商用（IG広告・サイネージ） | クレジット | このプロジェクトでの扱い |
|---|---|---|---|
| CC0 / Public Domain Mark | ○ | 不要（書いてもよい） | **最優先** |
| CC BY 3.0 / 4.0 | ○ | **必要**：題名・作者・ライセンス名とリンク・元ページ。4.0は「媒体に合った妥当な方法」で表記すればよい | IGはキャプションに入れる。サイネージは画面内に入れるのが難しいので、LPや店頭のクレジット欄にまとめる。**クライアントの承認を取ってから使う** |
| CC BY-SA | ○（ただし作品全体に継承義務がかかる可能性がある） | 必要 | 広告動画では**使わない**（`--commercial` の既定から外してある） |
| CC BY-NC / NC系 / Sampling+ | ×（広告は商用にあたる） | — | 使わない |

---

## 2. Real SFX（リアルサウンド）

- https://clipwing.pro/skills/real-sfx ／ https://github.com/Clipwing/skills/tree/main/skills/real-sfx （SKILL.md、README.md は raw.githubusercontent.com から読めた）
- 作者：Alexander Korbanov（Clipwing）。無料
- できること：映像のどこに音が要るか（キューのリスト、合成音のスクリプト、HyperFrames、画面録画、映像そのもの）を見つけ、**自分のライブラリにある本物の録音**をフレーム単位で置く。音楽とナレーションは残したまま、ナレーションの邪魔にならないモード（`--voice-safe`）、BGMを下げるモード（`--duck`）、ラウドネスの指定、出来の自己採点（critique）ができる。出力は新しいMP4、効果音だけのWAV、キューのレポート
- 方針は「音は合成しない」。ちょうどいい音がライブラリになければ、その場所は無音にしてレポートに書く
- 同梱の音：uisfx の78個（UI用。CC0。READMEでは「これ自体は合成音で、例外扱い」）。波などの環境音は入っていない。READMEがすすめる入手先：Freesound（CC0で絞る）、Sonniss GDC bundles、Mixkit、Pixabay
- 必要なもの：Python 3.9以上、FFmpeg、Claude Code
- インストール（公式どおり。**今回は未実施**：`~/.claude/skills` を変えることになるのでユーザーの判断に任せる）：
  ```
  git clone https://github.com/Clipwing/skills.git clipwing-skills
  mkdir -p ~/.claude/skills && cp -R clipwing-skills/skills/real-sfx ~/.claude/skills/
  # real-sfx フォルダの中で
  python scripts/setup.py init --library main="<このリポジトリ>/exports/v3/sfx" --uisfx --index
  python scripts/setup.py check
  ```
- このプロジェクトとの関係：v3 は `tools/v3/audio.py` でBGMと効果音をコードで作り、ミックスしている（Real SFXの「合成しない」方針とは逆）。Real SFXを使うなら、ライブラリ（`exports/v3/sfx/`）に録音の素材が揃ったあとで「どこに置くかの提案と自己採点」に使うのが現実的。**FIXED RULES とは衝突しない**（音の作り方の話なので）

## 3. earshot（イヤーショット）

- https://github.com/Leonxlnx/earshot （MIT、v0.4.1、2026-10作成）。試聴ページ：https://leonxlnx.github.io/earshot-site/
- できること：`describe`（どんな音かを言葉で説明）、`match` / `rank`（説明文にどれだけ合うか、候補のファイルを並べ替える）、`find`（長い録音の中から合う瞬間を探す）、`timeline` / `sheet`（映像全体の「聞き取り記録」と、それを画像にしたもの）、`critique`（ローンチ映画7本と比べた採点）、`forge`（コードで音を作り、自分で聞いて直す）、`sfx render`（252音のバンクからIDで書き出す。モデル不要）、`kit`（作品1本分の効果音セット）、`mcp`
- 必要なもの：Python 3.10以上、FFmpeg、torch（CPU版）。最初の実行でモデルを約1.1GBダウンロードする（CLAP `laion/larger_clap_general`、AST）。CLIは呼ぶたびにモデルを読み込むので20〜40秒かかる（MCPなら読み込みは1回だけ）
- インストール（公式どおり。**今回は未実施**）：
  ```
  git clone https://github.com/Leonxlnx/earshot && cd earshot
  python -m venv .venv
  .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  .venv/bin/python -m pip install -e ".[mcp]"
  .venv/bin/python -m earshot doctor --load
  claude mcp add earshot --scope user -e EARSHOT_THREADS=2 -- <path>/.venv/bin/python -m earshot mcp
  ```
- 音の権利：バンクの音はコード（レシピ）から書き出すものなので、MITの範囲で商用利用できると考えてよい（クレジットは不要。念のため控えには残す）。15音はFreesoundのCC0録音から特徴を取ったもの（クレジット：https://github.com/Leonxlnx/earshot/blob/main/docs/CREDITS_FIT.md ）
- このプロジェクトとの関係：**v3 のやり方（コードで音を作る）にいちばん近い**。`describe` / `timeline` / `critique` は、既存の `tools/v3/listen.py`（音を数値で確かめる仕組み）を補える。バンクにある `notify/*`（打った鐘の通知音）、`mallet/*`、`swipe/*`、`whoosh` 系、`logo_sting/soft_bloom` は、10の「26」のベルや場面転換の柔らかい音の候補になる。**波や海の環境音は持っていない**

---

## 4. 作ったもの：`tools/v3/sfx_search.py`

```
python3 tools/v3/sfx_search.py "ocean waves" --source openverse|realsound|earshot|all --commercial --limit 10
python3 tools/v3/sfx_search.py "ocean waves" --commercial --downloadable   # Freesound を除外（この環境で落とせる所だけ）
python3 tools/v3/sfx_search.py --get 3 [--wav]                             # 直前の結果の3番を取得（APIを呼び直さない）
```
- `openverse`：Openverse APIで検索する。`--commercial` を付けると `license_type=commercial&license=cc0,by`。ほかに `--license`、`--category`、`--length`、`--only`、`--downloadable`
- `earshot`：バンクのカタログ（公開JSON 4つ。`exports/v3/sfx/.cache/earshot/` に7日間キャッシュ）をキーワードで検索し、`earshot sfx render <id>` のコマンドを出す（音にするには本体のインストールが必要）
- `realsound`：Real SFXは検索サービスではないので、その説明とインストール方法を出したうえで、「手元のライブラリ」（`exports/v3/sfx/` の取得済み素材の .json）を検索する
- `--get N`：`exports/v3/sfx/openverse/<提供元>/` に音声と同じ名前の `.json`（`credit_ja`、`attribution`、`license_url`、`page`、`sha256` など）を保存し、`exports/v3/sfx/CREDITS.md` に追記する。`--wav` を付けると48kHzのWAVも作る（ffmpegを `nice -n 15`、1スレッドで実行）
- 取得に失敗したとき（403、429、HTMLが返った）は、`status: manual_download_required` と「手動でここに置く」という指示を .json に書く。サーバーから長い待ち時間（Retry-After が60秒を超える）を指示されたら、粘らずにやめる
- User-Agent はツール名がわかるものを付けている（Wikimedia／Openverseの方針どおり）。ログインが必要な所、ボット対策のある所には入らない

## 5. 実際に検索した結果（2026-11-07、すべて `--commercial`）

### "ocean waves"（240件）— 上位はすべてFreesound（この環境からは落とせない）
| # | 題名 | 作者 | ライセンス | 長さ | 元ページ |
|---|---|---|---|---|---|
| 1 | Zen Ocean Waves, Ocean Waves Ambience | INNORECORDS | CC0 1.0 | 1:36 | https://freesound.org/people/INNORECORDS/sounds/456899 |
| 2 | ocean waves 01 | rucisko | CC0 1.0 | 2:34 | https://freesound.org/people/rucisko/sounds/164767 |
| 3 | Ocean waves crushing on the chilean shore. | felix.blume | CC0 1.0 | 3:00 | https://freesound.org/people/felix.blume/sounds/155641 |
| 4 | ocean waves between rocks | Yoyodaman234 | CC0 1.0 | 1:57 | https://freesound.org/people/Yoyodaman234/sounds/331224 |
| 5 | Gentle ocean waves birdsong and gull | jackmichaelking | CC0 1.0 | 1:55 | https://freesound.org/people/jackmichaelking/sounds/518637 |
| 6 | Indian ocean wave sound | yurishkoda | CC BY 3.0 | 11:43 | https://freesound.org/people/yurishkoda/sounds/125935 |
| 7 | Ocean Waves 1 | gnrja | CC BY 4.0 | 0:47 | https://freesound.org/people/gnrja/sounds/151706 |
| 8 | Gentle Ocean Waves Mix (2018) | esh9419 | CC0 1.0 | 12:00 | https://freesound.org/people/esh9419/sounds/417797 |

Freesound以外（`--downloadable`）は、波の環境音がほとんどない（Jamendoは音楽、Wikimediaは発音の録音ばかり）。見つかった候補：
- **Water on Rocks**（Dsw4、CC BY 3.0、2:35、Lake Ontario の岸で波が岩に当たる音）https://commons.wikimedia.org/w/index.php?curid=5171057
- **Small sea waves at rocky beach**（mnalis、CC BY 3.0、10秒）https://commons.wikimedia.org/w/index.php?curid=132731782
- Bubbling Waterfall and Ocean Waves（CC0、約60分。滝の音が混ざっている）https://commons.wikimedia.org/w/index.php?curid=167502104

### "soft whoosh"（Openverse 63件／earshot 93件）
- Openverse：digital_whoosh_soft（ludvique、CC0、1.0秒）、whoosh.soft（junggle、CC BY 4.0、0.4秒）、Whoosh_Simple_ER_SFX_1／_4（Alex_hears_things、CC0）、Whoosh Simpler（original_sound、CC BY 3.0）。すべてFreesound
- earshot：`logo_sting/soft_bloom`（1.9秒、whooshからヒットとシマーするチャイムへ）、`swipe/cloth|paper|velvet|flick|glide`（0.6〜1.1秒の柔らかいスワイプ）など

### "bell chime"（Openverse 240件／earshot 67件）
- Openverse：School Bell Chime（melokacool、CC0）、FDP - Tibetan Bell - Chime 02（FirediesProductions、CC0、10.6秒）、Ceramic bell chime（Destructo20、CC0）、Chillingham Castle chimes（carthny、CC0）。すべてFreesound
- earshot：`fit/bell/small_bell`（CC0録音 steffcaffrey「Small Bell #2」から特徴を取ったもの、1.7秒）、`notify/m1`〜`m5`（打った鐘の通知音、A4〜E6、1.7〜2.5秒）。**10の「26」で鳴らす澄んだベル1回の候補になる**

## 6. 取得の状況と、ユーザーへのお願い

（取得結果は下の「取得ログ」を参照）

この環境から波の環境音を落とせなかった場合、方向性は変えずに、次のどちらかで補う：
1. **Freesoundから手で落とす（おすすめ）**：上の表の CC0 #3（felix.blume、チリの海岸、3:00）か #5（gentle waves＋鳥とカモメ）。Freesoundに無料登録してログインし、元ページの「Download」から元のWAVを取得する。置き場所は `exports/v3/sfx/openverse/freesound/`。そのあと `python3 tools/v3/sfx_search.py "ocean waves" --commercial --get 3` を実行すると、クレジットの .json が作られる（音声は手で置いたものを使う）
2. **現地で録る**：土佐の海の音をiPhoneで録る（不足素材のテンプレート）
   - 不足している素材：実写パート（12以降）の下に敷く、途切れない波の環境音（30〜60秒）
   - なぜ必要か：撮影素材の音には風や人の声が入っていて、ループにすると継ぎ目が目立つ。BGMから現地の音へのつなぎを自然にしたい
   - 追加で撮るべきショット：撮影素材と同じ浜（または2階・屋上から見える海）で、静かな寄せ波を60秒以上×2テイク。人の声や車の音が入らない時間帯に録る
   - iPhoneでの撮り方：ボイスメモ（設定で「ロスレス」とステレオをオン）か、カメラで動画を撮って音だけ使う。風よけ（ふわふわのウインドジャマー、なければ体で風をさえぎる）を使い、iPhoneの下側のマイクを海に向けて地面や手すりに固定する。手持ちのノイズが入らないよう、録っている間は触らない

## 取得ログ

| 素材 | ライセンス | 結果 | ファイル / メタ |
|---|---|---|---|
| Small sea waves at rocky beach（mnalis） | CC BY 3.0（クレジット必須） | **取得できた**。ただし 10秒、モノラル、15kbps の opus で、**peak −49 dBFS／RMS −70 dBFS とほぼ無音**。持ち上げるとノイズが目立つので本番には不向き（参考用） | `exports/v3/sfx/openverse/wikimedia_commons/small_sea_waves_at_rocky_beach_f1c1a537.opus`（`.48k.wav`、`.json`） |
| Water on Rocks（Dsw4） | CC BY 3.0 | **取得できず**：upload.wikimedia.org が 429（Retry-After 600秒）。10分以上あけて3回試したが、通ったのは上の1回だけ。このコンテナの出口IPが共有されていて、レート制限にかかっていると考えられる | `exports/v3/sfx/openverse/wikimedia_commons/water_on_rocks_b92f58c8.json`（status: manual_download_required） |
| Freesound の CC0 波（上の表 #1〜#8） | CC0 / CC BY | **取得できず**：cdn.freesound.org が 403。ボット対策を回避するような手段（Referer を偽るなど）は使っていない | — |

- `exports/v3/sfx/CREDITS.md` に、取得できた1件のクレジットを追記済み
- まとめ：**この環境から、広告に使える品質の波の環境音は手に入らなかった**。上の §6 の 1（Freesound のCC0をユーザーが手で取得）か 2（現地で録る）で補う。今の `audio.py` は、実写パートで撮影素材の音（現地の波の音）を使っているので、それで足りるなら追加の素材は不要
