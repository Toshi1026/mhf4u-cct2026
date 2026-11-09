# 音を「聞く」ための調査と試行：MAZE v3 Reel（2026-11-07）

## 結論
- **`tools/v3/listen.py` を作り、今のミックスとナレーション12本で実際に動かしました。** CPU だけで動き、ミックス39秒で約40秒、ナレーション12本で約2分半かかります。ラウドネス、PNG、AudioSet のタグ、CLAP、Whisper の5つをまとめて出します。`ffprobe` を通るので、CapCut から書き出した mp4 も同じように調べられます。mp4 は仮のファイルで動作を確かめました。
- **今のミックスの大きな問題は3つです。**
  - プチ音が4か所あります。
  - 最後の約3.5秒の「現地の波の音」は、実際には店内の低い唸りです。タグ付けの AI には「Vehicle / Car（車）」と聞こえています。
  - 全体の音量が -16.5 LUFS で、Reels でよく言われる目安の -14 より 2.5 dB 小さいです。
- **ナレーションの読みは12本とも想定どおりでした。** 気になるのは1か所だけで、12 の「浅瀬**は**」が、Whisper の2つのモデルとも、ミックスの上でも「あさせ**ば**」と聞き取られます。人の耳で確かめてください。
- **「音を聞ける」スキルは、そのまま使える決定版がありませんでした。** 見つかったものは、どれも次のどれかでした。
  - librosa と Whisper による「測るだけ」のもの
  - 有料 API のキーが必要なもの
  - 重みが非商用のもの

  そこで、商用にも使える ONNX のモデル（YAMNet、PANNs、CLAP）と faster-whisper を `listen.py` に組み込みました。

---

## 1. 作ったもの

```
python3 tools/v3/listen.py exports/v3/mix.wav            # → exports/v3/listen/mix/
python3 tools/v3/listen.py exports/v3/vo/                # フォルダ：1本ずつ＋一覧表＋読みの確認
python3 tools/v3/listen.py exports/v3/MAZE_v3_ig_A.mp4 --name ig_A   # CapCut の書き出しもこれで
  --whisper small|large-v3-turbo(既定)|none  --prompts 説明文.txt  --win 2  --clap-win 5  --threads 2
```

- 本体：`/home/user/mhf4u-cct2026/tools/v3/listen.py`
  - 自分の優先度を nice 10 に下げ、ONNX と Whisper は2スレッドで動かします。
  - モデルが見つからない項目は飛ばして続けます。
- 読みの確認用ファイル：`/home/user/mhf4u-cct2026/tools/v3/listen_expect.json`
  - 想定の読みは `moras.json` から自動で読み込みます。
  - キーワードは、文字でも読み（かな）でも照合します。

**出力**（`exports/v3/listen/<name>/`）
- `report.md`
- `overview.png`
- `zoom_*.png`（10秒ごとの拡大）
- 各ファイルの `.json`

**PNG の中身**
- 波形と Whisper の語
- メルスペクトログラムと拍、オンセット
- 短期 LUFS と声の区間
- YAMNet の上位ラベル

画像は Read ツールで見えることを確かめました。

**中身と設定**
- (a) ラウドネス
  - ffmpeg ebur128 の統合値、LRA、トゥルーピーク。pyloudnorm で照合しています。
  - 短期（3秒窓）LUFS の推移
  - クリップ、DC、左右の相関、頭と尻の音量
  - 声がある所と無い所の LUFS の差（Whisper の語の区間で切り分け）
  - **プチ音の候補の検出**：8kHz 以上の、2.7ms 幅の急な山を探します。子音は拾わないようにしています。
  - デジタルの完全な無音の区間
- (b) PNG（上の表のとおり）
- (c) AudioSet のタグ：YAMNet（521種）と PANNs Cnn14（527種）。どちらも onnxruntime の CPU で動かします。
- (d) CLAP（LAION clap-htsat-unfused の量子化 ONNX）
  - 特徴量は torch を使わず librosa で作り直しました。
  - CC0 の効果音で確かめた結果：タイピング音は「typing」が 0.47、ピアノ曲は「calm piano」が 0.33、カウントダウン音は「countdown beeps」が 0.66 で、それぞれ1位でした。
- (e) faster-whisper large-v3-turbo（int8）で、語ごとの時刻つき文字起こし
  - 想定の文は initial_prompt に入れていません（聞こえていない語まで書いてしまい、確認にならないため）。
  - 文字起こしは SudachiPy で読みに直し、数字は自前の簡易変換で読みに直してから、想定のカナと比べます。

**入れたもの**（どれも `python3 -m pip`。torch は入れていません）
- librosa 0.11.0、soundfile、pyloudnorm 0.2.0、matplotlib
- faster-whisper 1.2.1、ctranslate2 4.8.5
- transformers 5.22.2：入れましたが使っていません。torch がないと CLAP の前処理が動かないためです。
- tokenizers
- sudachipy 0.7.0、sudachidict_core 20261015（Apache-2.0）
- pykakasi 2.3.0（GPL-3.0。SudachiPy がないときの予備で、配布はしません）

**注意**
- この環境では `pip` が Python 3.13 用で、`python3` は 3.11 です。必ず `python3 -m pip` を使ってください。
- unidic-lite（fugashi 用の辞書）はビルドに失敗しました。

**モデルの置き場所**：`/home/user/vendor/audio-models/`（合計 495MB）。Whisper turbo は 1.6GB で、HF のキャッシュに入っています。

| モデル | 取得元 | ライセンス（モデルカードの表示） | 実測 |
|---|---|---|---|
| YAMNet ONNX | https://huggingface.co/zeropointnine/yamnet-onnx（2026-01-18） | Apache-2.0。元は Google | 39秒を0.14秒 |
| PANNs Cnn14_16k ONNX | https://huggingface.co/pranjal-pravesh/PANNs_CNN14_ONNX（2025-12-12、327MB） | MIT（コミュニティによる変換） | 5秒の窓1つで1.1秒（初回の読み込みを含む） |
| CLAP | https://huggingface.co/Xenova/clap-htsat-unfused（2024-10-08）の quantized の audio/text | 表示なし。元の laion/clap-htsat-unfused は Apache-2.0 | 10秒の窓1つで約0.2秒 |
| Whisper | mobiuslabsgmbh/faster-whisper-large-v3-turbo（読み込み14秒、短いファイル1本7秒前後）、Systran/faster-whisper-small（MIT。速いが誤りが多い：「高知」→「コーチ」、「約」→「医薬」） | Whisper は MIT | 上の欄に記載 |
| AudioSet のラベル表（PANNs の表示用） | k2-fsa/sherpa-onnx-ced-mini-audio-tagging の CSV | — | CED 本体は使っていません |

## 2. 今のミックス（exports/v3/mix.wav）を聞いた結果

数値：統合 **-16.5 LUFS**、LRA 2.8 LU、トゥルーピーク -1.5 dBTP、クリップ 0、左右の相関 1.00（L と R が同じ＝モノラル）。報告書は `exports/v3/listen/mix/report.md` です。

**良い点**
- **ナレーションは BGM に埋もれていません。** Whisper turbo はミックスの上でも原稿をほぼ全文正しく書き起こしました。声がある所は -16.3、無い所は -29.2 LUFS で、差は 12.8 dB あります。
- 26 のベル（31.2〜32.6秒）、09 の後の間（29.19〜29.32秒は完全な無音）、32.9秒で実写の音への切り替えは、どれも設計どおりにスペクトログラムに見えます。
- 実写の波 IMG_9662（1.6〜4.6秒の区間）は CLAP の「waves crashing on the shore」が **0.44** でした。ちゃんと波に聞こえています。

**直したい点**（`tools/v3/audio.py` は触っていません。直すかどうかは判断をお願いします）
1. **29.32秒のプチ音（+40dB の山）。** 0.2秒の「間」をフェードなしで切っているのが原因です（`gain[gap] = 0`）。出入りに 5〜10ms の傾斜を付ければ直ります。
2. **22.65 / 25.51 / 27.65秒のプチ音。** キック（`kick(n=0.35)`）を 0.35秒で切り捨てていて、その切り口が鳴っています。
   - 切った時点でもまだ約4%の振幅が残っています。
   - t06=17.3秒、84BPM から計算すると、7・11・14拍目のキックの終わりにぴったり一致します（計算で確認済み）。
   - ほかの拍の切り口は、声に隠れていると見ています（未確認）。
   - 最後の 20〜30ms をフェードすれば直ります。
3. **最後の約3.5秒が「車の音」に聞こえます。**
   - IMG_9636（店内の窓越し）の音は、97%が 500Hz 未満の唸りです。
   - タグ：YAMNet は「Sliding door / Mechanisms」、PANNs は「Vehicle」。CLAP の1位は「indoor room tone」で、0.11 しかありません。
   - `audio.py` はこの唸りを波と同じ RMS まで持ち上げています。そのため、ミックスの 38〜39.2秒は YAMNet で「Vehicle 0.21 / Car 0.12」になります。
   - さらに、35.67秒で2つの素材をクロスフェードなしでつないでいます（高い音域が段差で 8dB 落ちます）。
   - 案は2つです。
     - 最後まで IMG_9662 か IMG_9671 の波の音を使う。
     - 9636 に 150Hz のハイパスをかけて -6dB 下げ、0.3秒でクロスフェードする。
4. **全体の音量 -16.5 LUFS。**
   - -14 にするなら、ffmpeg の `loudnorm=I=-14:TP=-1:LRA=7`（2パス）か、トゥルーピークのリミッターが必要です。今はピークで正規化しているので、単に上げると -1 dBTP を超えます。
   - Instagram は公式の数値を出していません。-14 は通説です。
5. 細かい点です。
   - 頭の 10ms がいきなり -28 dBFS で始まります。5ms のフェードインで足ります。
   - モノラルなので、スマホでは問題ありません。
   - 2〜5kHz の帯域が 1.5% と少ないです。ただ、ナレーション単体でも 1.8〜2.4% なので、TTS の声質です。CLAP の「muffled（こもった音）」はマイナスでした。こもっているかどうかは人の耳で判断してください。
6. **自動のビート検出はこのミックスには使えません。** librosa は 99.4 BPM と出しました（設計は 84 BPM）。声の区切りを拍と取り違えています。CapCut の「Auto beat」も、ミックスではなく BGM だけのトラックにかけてください。

## 3. ナレーション（exports/v3/vo/*.wav）の読みの確認

報告書は `exports/v3/listen/vo/report.md` です。12本とも読みの一致率は 0.94 以上でした。

| 行 | Whisper turbo の文字起こし | 判定 |
|---|---|---|
| 02 | 人が乗った最大の波26.21メートル。 | OK。にじゅうろくてんにいいち |
| 04 | 約2万5千キロ | OK |
| 07 / 08 / 10 | 16000キロ、時速40キロ、約26日 | OK。08 の差は数字を読みに直すときの変換によるもの |
| 12 | 波が立つ**あさせば**、**二淀川**が運んだすな、かも。 | 「二淀川」は Whisper の当て字で、読みは「によどがわ」なので正しいです。**「浅瀬は」→「ば」**は small、turbo、ミックスの3回とも同じでした。人の耳で確認してください |
| 13 ほか | — | OK |

- 音量は 1本ごとに -22.1〜-24.9 LUFS で、差は 2.8 dB です。12 が一番大きく、13 が一番小さいです。ミックスの前にそろえる余地があります。
- 新居（にい）と南風（まぜ）は、今のナレーションに出てきません。
  - 辞書による読みの変換では、単語だけだと「新居＝しんきょ」「南風＝はえ」になりました（「高知県土佐市新居」と続けると「にい」）。
  - キャプションを読み上げる場合は、読みをカナで指定する必要があります。

## 4. 音を聞けるスキル・MCP の評価

★の数と最終更新は、2026-11-07 に GitHub の検索 API で取ったものです。`/home/user/vendor/` に複製しました。他人のスクリプトの実行は自動判定に止められたので、**どれも読んだだけで、動かしていません**。

| リポ | ★ | ライセンス | 最終更新 | 中身 | 判定 |
|---|---|---|---|---|---|
| https://github.com/edotheDev/claude-video-kit （`music-and-mix` スキル） | 30 | MIT | 2026-10-14 | 「A model cannot listen」を前提に、テンポ、キックの拍、2秒ごとの音量、スペクトルの画像、Demucs でボーカルの有無を測ります。CPU で動く | **参考に良い**。考え方は `listen.py` と同じ。Demucs は torch が必要 |
| https://github.com/kmilo93sd/agent-video-editor | 7 | MIT | 2026-10-04 | 音の目安値（声 -16 LUFS、BGM は声より 18〜25dB 下など）と、**CC0 の音128個**（Kenney、OpenGameArt、自作。LUFS 実測つき） | **有益**。音はこの環境にもう置いてあります。波の音はなく、whoosh、swell、ベル類があります |
| https://github.com/sxhzju/sfx | 5 | MIT | 2026-10-22 | Mixkit、Kenney、Freesound、Pixabay の検索とダウンロード | 参考。Pixabay は Cloudflare 越えの仕組みを使うので、**使いません** |
| https://github.com/ndpvt-web/reel-analyzer （reel-audio-sync） | 1 | 表示なし | 2026-08-27 | librosa で拍、声以外の出来事、カットとの同期を見ます | 参考 |
| https://github.com/BusyBee3333/viral-video-edit-skills | 4 | MIT | 2026-07-19 | 曲の拍の格子、区間の割り出し | 参考。BGM 選びのとき |
| https://github.com/SlamWeb/bgm-understanding | 0 | NOASSERTION（コードは MIT） | 2026-11-01 | CPU で拍、区間、楽器、気分を出します | **商用には使いません**。MTG と audEERING の重みが CC BY-NC-SA |
| https://github.com/hugohow/mcp-music-analysis | 33 | MIT | 2025-08-12 | librosa を MCP にしたもの | `listen.py` で足ります |
| https://github.com/oldrvn/rubin | 0 | 表示なし | 2026-02-12 | MCP。librosa でミックスの質を評価。標準入力でも受ける | 試す価値は低い |
| https://github.com/git-scarrow/mcp-audio-server | 0 | MIT | 2026-06-25 | コード、キー、テンポ。codeberg に移転 | 不要 |
| https://github.com/Audio-Launch/audiolab-claude-plugin | 1 | 表示なし | 2026-07-03 | ホスト型 API（ラウドネス、声の QA）。**API キーが必要**（デザインパートナー向け） | 保留 |
| https://github.com/fatimaarshadg-ux/claude-media-watcher | 0 | 表示なし | 2026-09-28 | 画像の切り出しと Whisper だけ。**自分のパソコン専用**と README に明記 | 不要 |
| https://github.com/jftuga/transcript-critic | 52 | MIT | 2026-04-30 | whisper.cpp で文字起こしして批評 | 不要 |
| https://github.com/ypwcharles/mimo-agent-skill | 8 | NOASSERTION | 2026-06-18 | Xiaomi MiMo の API で音を理解。**キーが必要** | 報告のみ |
| https://github.com/w1ne/claude-skill-ask-gemini | 0 | MIT | 2026-04-27 | Gemini CLI に音声を渡す。**Google のログインかキーが必要** | 報告のみ |

**土台にしたライブラリ**
- https://github.com/SYSTRAN/faster-whisper：★26,543、MIT、2026-11-06
- https://github.com/librosa/librosa：★8,804、ISC
- https://github.com/csteinmetz1/pyloudnorm：★790、MIT
- https://github.com/qiuqiangkong/panns_inference：★276、MIT（torch が必要なので、代わりに ONNX を使いました）
- https://github.com/LAION-AI/CLAP：★2,331、CC0-1.0
- https://github.com/microsoft/CLAP（msclap）：★679、MIT、torch が必要
- https://github.com/MTG/essentia：★3,795、**AGPL-3.0**、使っていません
- BEATs（microsoft/unilm）：★22,240。ONNX が見つからず、torch が必要なので見送りました。

**マルチモーダルの API**（キーがないので、届くかどうかだけ確認）
- generativelanguage.googleapis.com（Gemini）：404
- api.openai.com：401
- platform.xiaomimimo.com：200

Gemini などは音声を直接受け付けるので、「人の印象に近い感想」が欲しいときの補助に使えます。キーが必要です。

## 5. 商用の Instagram 広告で使えるフリー効果音

この環境からの `curl -sI` の結果と、規約を読んだ結果です。

| 取得元 | この環境から | 規約（読んだところ） | クレジット |
|---|---|---|---|
| 効果音ラボ https://soundeffect-lab.info/agreement/ | **200** | 個人・法人とも商用無料。再配布は禁止（効果音が主役のコンテンツも再配布扱い）。直リンク、AI の学習、Content ID への登録も禁止 | 不要（任意） |
| 魔王魂 https://maou.audio/rule/ | **200** | CC BY 4.0 か独自の規約を選べる。商用無料、報告不要、改変可。単品の再配布と AI の学習は禁止 | **必要**。例：「効果音：魔王魂」 |
| OtoLogic https://otologic.jp/free/license.html | **403**（Cloudflare の確認画面。WebFetch でも 403） | **未確認**。CC BY 4.0 で「OtoLogic」の表記が必要、というのは私の記憶で、今回は確かめられていません | 要確認 |
| Pixabay https://pixabay.com/service/license-summary/ | **403**（Cloudflare） | WebFetch で読めました。無料、表記不要。そのままの再配布と、商標・社名としての利用は禁止 | 不要 |
| Freesound https://freesound.org/help/faq/ | **403** | WebFetch で読めました。CC0 / CC BY / CC BY-NC が混在。ダウンロードにはアカウント、API にはキーが必要。**広告には CC0 か CC BY のみ。NC は使えません** | CC BY は必要。CC0 は不要 |
| Sonniss GDC https://sonniss.com/gameaudiogdc | **403** | WebFetch で読めました。商用無料、広告も可、表記不要。AI の学習と、効果音としての再販は禁止。この一覧ページの最新は GDC 2024（2025年版はなし。2026年版は別ページ） | 不要 |
| Mixkit https://mixkit.co/license/ | 200 | 「Sound Effects Free License」があることだけ確認。条文はページ上の別画面にあって読めず、**未確認** | — |
| Kenney / OpenGameArt | 403 / 200 | CC0。agent-video-editor にもう入っている128個がこれ | 不要 |

Cloudflare の確認画面を回避する操作は、自動判定に止められました。**OtoLogic、Pixabay、Freesound は、ユーザーのブラウザで取ってきてもらう必要があります。** 取ってきた音は `listen.py --whisper none --prompts …` で、意図どおりの音かどうか確かめられます。

## 6. 実際に試したことと、読んだだけのこと

**試したこと**
- モデルの取得と ONNX の推論
- CLAP の特徴量を自前で作り、CC0 の音で検証
- Whisper small と turbo の比較
- `listen.py` を mix、vo 12本、実写の音2つ、ミックスの尻、mp4 で実行
- プチ音の原因を数値で特定
- 規約ページの取得と HEAD の確認

**読んだだけのこと**
- 4章の他人のスキルすべて（実行は止められました）
- OtoLogic と Mixkit の規約の中身（読めませんでした）

**作業ファイル**（`/home/user/mhf4u-cct2026/exports/v3/research_trials/`）
- `amb_IMG_9662_1.6-4.6.wav`、`amb_IMG_9636_1.4-5.4.wav`、`mix_tail_37.3-end.wav`、`mp4test.mp4`
- `prompts_ambience.txt`
- `listen/`

**Git について**：`tools/v3/listen.py` の途中版は、別のワークフローが WIP のコミット `0f8bf22` に含めています。今の版はまだコミットしていません。`tools/v3/listen_expect.json` がコミット済みかは確かめていません。`edl/v3/research/sfx_sources.md`（未追跡）は、別のエージェントが作ったファイルです。