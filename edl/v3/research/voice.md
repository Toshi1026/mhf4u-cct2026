# 日本語ナレーション用TTSの比較調査（Instagram雑学ナレーション 30〜40秒・広告用）

調査日は 2026-11-06 です。Higgsfield では音声を生成していません（使ったのは `list_voices`、`models_explore`、`balance` の読み取りだけです）。VOICEVOX だけは、この環境で実際に動かして試しました。

## 結論

| 位置づけ | 選択肢 | 理由 |
|---|---|---|
| **第一候補** | **Fish Audio S2.1 Pro（有料APIまたは Plus プラン）** | 日本語の自然さと、感情タグで演出できる点が最も期待できます。商用利用は有料なら可能です。費用はAPIの従量課金で1本あたり約1円です。 |
| **代替** | **Google Cloud TTS（Chirp 3 HD の ja-JP 音声）** | 落ち着いた大人のナレーション向きです。商用利用ができ、クレジット表記は不要です。月100万字まで無料なので実質0円です。gcloud はこの環境に入っています。 |
| **すぐ使える保険** | **VOICEVOX（この環境で動作確認済み）** | 無料で、APIキーも不要です。商用利用は「VOICEVOX:キャラ名」のクレジットを入れれば可能です。ただし声は合成っぽさが残ります。 |

**ユーザーに用意してもらうもの：**
- Fish Audio を使う場合
  - 有料アカウントの API キー（`FISH_API_KEY`）。チャットには貼らず、安全な入力フォームで登録してもらいます。
  - 使う声の決定。他人が作ったコミュニティ音声は広告に使えるかがはっきりしません。Voice Design で作る声か、本人の同意を取った自前の声のクローンを推奨します。
- Google を使う場合
  - 課金を有効にした GCP プロジェクト（無料枠を使うにも課金の有効化が必要です）
  - 認証。ユーザーが `gcloud auth login` するか、サービスアカウントで認証します。
  - Text-to-Speech API の有効化
- VOICEVOX を使う場合
  - 用意するものはありません。キャラクターを決めて、投稿キャプションに「VOICEVOX:○○」と書くだけです。

## 1. 比較表

| サービス | 日本語ナレーションの質（アニメ調でない自然さ） | 広告での商用利用 | クレジット表記 | 費用（35秒＝約200字） | MCP | この環境で使えるか |
|---|---|---|---|---|---|---|
| Fish Audio | ◎ S2.1 Pro は公式が「日本語品質No.1」と宣伝。80以上の言語に対応し、[whisper] などのインライン感情タグが使える | **無料枠は不可**。規約では無料は「personal, non-commercial」、有料は「licensed … for commercial uses」 | 不要（AIで作ったと開示することは推奨） | API は $15 / 100万UTF-8バイト（日本語1字≒3バイトなので1本 $0.01 未満）。Plus は $11/月 | 公式は**ドキュメント用のみ**（`https://docs.fish.audio/mcp`）。音声生成はコミュニティ製の https://github.com/da-okazaki/mcp-fish-audio-server を使う | ◎ API 到達可（`api.fish.audio/model` が 200）。PyPI の `fish-audio-sdk` 1.4.0 も使える。必要なのはキーだけ |
| ElevenLabs | ○〜◎ 多言語モデルが日本語に対応。固有名詞の読み間違いや、わずかな外国語なまりが出ることがある | **無料枠は不可**（非商用で帰属表示が必要）。Starter 以上で商用ライセンス | 無料なら必要、有料なら不要 | 無料は10kクレジット/月。Starter は $6/月（初月 $1） | 公式の `elevenlabs/elevenlabs-mcp` はローカル版で、**非推奨・保守終了**。後継はホスト型の `https://api.elevenlabs.io/v1/mcp`（OAuth認証）。PyPI 版は 0.12.2 | ○ API 到達可（`/v1/voices` が 200）。キーが必要 |
| VOICEVOX | △〜○ 抑揚はやや機械的。落ち着いた声は 雀松朱司・九州そら・小夜/SAYO・剣崎雌雄・麒ヶ島宗麟 | **可**。キャラクターごとの規約に従う（§3参照） | **必須**（「VOICEVOX:キャラ名」） | 無料 | 公式の音声生成MCPなし。ローカルの HTTP API を直接呼べばよい | **◎ 動作確認済み**（§3） |
| AivisSpeech / Style-Bert-VITS2 | ○〜◎ VOICEVOX より自然。モデル次第 | ソフト自体は商用可。音声の扱いは**モデルのライセンス次第**（ACML 系は商用可、ACML-NC は非商用） | ソフトは不要。モデルの規約による | 無料 | なし | ○ 公式 Docker イメージ `ghcr.io/aivis-project/aivisspeech-engine:cpu-latest` があり、Linux の CPU で動く。初回にモデル約250MB と BERT 約650MB を取得し、RAM は1.5GB以上必要。ghcr.io は到達可だが、pull は未実施 |
| COEIROINK | △〜○ キャラクター寄りの声が多い | ソフトは商用可。声ごとの規約に従う | **必須**（「COEIROINK:声名」。音声のみの作品では読み上げで明示する必要あり） | 無料 | なし | △ Linux 版の有無は未確認。採用はおすすめしない |
| Google Cloud TTS | ○〜◎ Chirp 3 HD は自然。Gemini TTS は文章で話し方を指示できる | 可（GCP の規約に従う） | 不要 | Chirp 3 HD は月100万字まで無料、超過分は $30/100万字。Neural2 は $16、Standard は $4（月400万字まで無料）。Gemini 3.8 Flash TTS は無料枠なしで、入力 $0.50・音声出力 $9（いずれも100万トークンあたり、2026年末まで） | なし（gcloud か REST API で十分） | ○ gcloud はインストール済みだが**ログインしている認証情報がない**。API は到達可（認証なしで 403 が返る） |
| Azure Speech | ○ ja-JP の Nanami・Keita など。ニュース調で無難 | 可 | 不要 | F0（無料）は月50万字まで。S1 Neural は $15/100万字（東日本リージョンの小売価格 API で確認） | なし | ○ 到達可（401）。キーとリージョンが必要 |
| OpenAI TTS | △ 日本語は話せるが、英語ネイティブ風の抑揚が残りやすい | 可。ただし**聞き手にAI音声だと開示する義務**がある（利用ポリシー。記憶ベースのため要再確認） | 開示が必要 | tts-1 は $15/100万字、tts-1-hd は $30、gpt-4o-mini-tts は入力 $0.60・音声出力 $12（100万トークンあたり） | なし | ○ 到達可（401）。キーが必要 |
| Higgsfield（接続済みMCP） | ？ プリセット音声は約120種あるが、英語名ばかりで日本語専用のものはない（Hana、Ken などは名前だけ）。日本語は多言語エンジン任せになり、なまりが出るおそれ | 公式ヘルプは「出力の商用利用は制限しない、プランにも依存しない」。ただし第三者の記事に「無料プランは個人・評価用のみ」とするものがあり、**はっきりしない** | 不明 | 現在は **free プランで残り 8.12 クレジット** | ◎ 接続済み。音声モデルは `elevenlabs_v4` / `elevenlabs_v4_turbo` / `qwen_audio_tts`（言語に `ja` を指定可）/ `seed_audio` / `text2speech_v2`（elevenlabs・minimax など） | ◎。ただし今回は生成していない |

## 2. 各サービスの補足

**Fish Audio**
- 公式の料金ページでは、無料枠（8,000クレジット/月、1回500字まで）の欄に「Commercial use」と書かれています。一方、同じページのFAQと規約は「無料は非商用のみ」としていて矛盾しています。**規約の方を優先し、無料枠は広告には使わない**のが安全です。
- 料金ページには「Premium subscribers can use verified voices (that you own) for commercial purposes」とあります。広告には、自分が権利を持つ声（Voice Design で作る声、またはクローン）を使うべきです。
- `s2.1-pro-free`（$0）という API モデルがありますが、規約上は無料利用と同じ扱いになると考えるのが安全です。
- `s1` は 2026-12-31 に提供終了します。コミュニティ製MCP（npm の `@alanse/fish-audio-mcp-server` 0.6.1、MIT、最終公開 2025-07-04、スター16）は既定モデルが `s1` です。使うなら `FISH_MODEL_ID` を指定する必要がありますが、新しいモデルIDに対応しているかは未確認です。**MCPを入れずに、SDK か curl で直接APIを呼ぶ方が確実**です。

**ElevenLabs**
- 無料プランは商用不可です（公式ヘルプ「Can I publish the content I generate」）。
- Higgsfield 経由でも `elevenlabs_v4` が使えますが、その場合の権利は Higgsfield の規約の下での扱いになります。

**Google**
- 35秒のナレーションを20回撮り直しても5,000字程度なので、無料枠の中に収まります。
- 声は ja-JP の Chirp 3 HD 音声からサンプルを試聴して選びます。

## 3. VOICEVOX の実機確認（この環境）

- **Docker：** Docker デーモンは最初は起動していませんでした。`dockerd` を手動で起動すると動きました。
  - `voicevox/voicevox_engine:cpu-0.26.1` の pull に成功しました（圧縮時 2.08GB、展開後 4.5GB）。
  - 空きディスクは pull 後で約27GB です。
  - Docker Hub の配布元は到達可能でした。
- **Linux CPU版のバイナリ：** GitHub Releases は、このセッションでは GitHub へのアクセスが制限されているため 403 でした。ファイルサイズも確認できませんでした。**Docker を使う方法が現実的**です。
- **起動と速度：** 起動は数秒でした。収録キャラクターは44です。4 CPU で、音声長に対する処理時間の比は約0.3〜0.5でした（例：27.8秒の音声を13.0秒で合成）。
- **試聴用サンプル：** 施設紹介の短い一文を合成しました。読みの確認はしていません。
  - `/tmp/claude-0/-home-user-mhf4u-cct2026/d6ba61d0-2d98-537d-84ff-35fa9fa6bdb7/scratchpad/vv_52.wav`（雀松朱司）
  - `/tmp/claude-0/-home-user-mhf4u-cct2026/d6ba61d0-2d98-537d-84ff-35fa9fa6bdb7/scratchpad/vv_16.wav`（九州そら）
  - `/tmp/claude-0/-home-user-mhf4u-cct2026/d6ba61d0-2d98-537d-84ff-35fa9fa6bdb7/scratchpad/vv_46.wav`（小夜/SAYO）
  - `/tmp/claude-0/-home-user-mhf4u-cct2026/d6ba61d0-2d98-537d-84ff-35fa9fa6bdb7/scratchpad/vv_20.wav`（もち子さん）
  - `/tmp/claude-0/-home-user-mhf4u-cct2026/d6ba61d0-2d98-537d-84ff-35fa9fa6bdb7/scratchpad/vv_13.wav`（青山龍星）
- **キャラクターごとの規約（エンジン内の要約）：**

| キャラクター | 規約の要点 | クライアント案件での扱い |
|---|---|---|
| 雀松朱司、九州そら、小夜/SAYO、剣崎雌雄、麒ヶ島宗麟 | 「VOICEVOX:名前」と書けば商用・非商用で利用可 | **候補**（詳しい規約はリンク先で最終確認が必要） |
| 青山龍星 | **企業が関わる利用は事前確認が必要** | 今回のような店舗案件では避ける |
| もち子さん | **企業が関わる利用は事前確認が必要**。音声素材などの用途は対象外 | 避ける |

- 店名などの固有名詞は、「シーサイドまぜ」のようにカナで入力するのが無難です。
- 屋外サイネージに流用する場合は、クレジットを画面内に入れる必要があります。

## 4. この環境からの通信確認

| 接続先 | 結果 |
|---|---|
| fish.audio / api.fish.audio | 200（OK） |
| elevenlabs.io / api.elevenlabs.io | 200（OK） |
| voicevox.hiroshiba.jp / hub.docker.com / registry-1.docker.io | OK（pull 成功） |
| github.com（Releases） | 403（このセッションでは制限あり） |
| aivis-project.com | 200 |
| ghcr.io | 401（認証前の通常の応答で、到達は可能） |
| coeiroink.com | 200 |
| texttospeech.googleapis.com | 403（認証なし。到達は可能） |
| japaneast.tts.speech.microsoft.com | 401（到達可能） |
| api.openai.com | 401（到達可能） |
| pypi.org / registry.npmjs.org | OK |

## 5. 後片付けの状況

- この環境では `dockerd` が起動したままです。
- VOICEVOX のイメージ（4.5GB）は残してあります。
- コンテナ `vv` は停止済みです。再開は `docker start vv` で、API は `localhost:50021` です。

## 出典
- Fish Audio：[規約](https://fish.audio/terms) / [料金](https://fish.audio/plan) / [API料金](https://docs.fish.audio/developer-platform/models-pricing/pricing-and-rate-limits) / [公式ドキュメントMCP](https://docs.fish.audio/developer-platform/sdks-tools/coding-agents) / [コミュニティ製MCP](https://github.com/da-okazaki/mcp-fish-audio-server) / [S2の日本語記事](https://www.techno-edge.net/article/2026/03/16/4926.html) / [ja TTSページ](https://fish.audio/ja/tts/)
- ElevenLabs：[料金](https://elevenlabs.io/pricing) / [elevenlabs-mcp](https://github.com/elevenlabs/elevenlabs-mcp) / [公開・商用利用のヘルプ](https://elevenlabs.io/docs/help-center/legal/can-i-publish-the-content-i-generate-on-the-platform)
- VOICEVOX：[トップ](https://voicevox.hiroshiba.jp/) / [利用規約](https://voicevox.hiroshiba.jp/term/) / [雀松朱司](https://voicevox.hiroshiba.jp/product/wakamatsu_akashi/)
- AivisSpeech：[公式サイト](https://aivis-project.com/) / [AivisSpeech-Engine](https://github.com/Aivis-Project/AivisSpeech-Engine)
- COEIROINK：[利用規約](https://coeiroink.com/terms)
- Google：[Cloud TTS 料金](https://cloud.google.com/text-to-speech/pricing)
- Azure：[Speech 料金](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services/)（あわせて prices.azure.com の小売価格 API で確認）
- OpenAI：[料金](https://developers.openai.com/api/docs/pricing)
- Higgsfield：[出力の所有権・商用利用（ヘルプ）](https://higgsfield.ai/creator-hub/help-center/account-and-privacy/who-owns-my-generations-and-can-i-use-them-commercially) / [無料プランは個人・評価用とする第三者記事](https://flowith.io/blog/higgsfield-faq-video-length-skin-rendering-commercial-rights)