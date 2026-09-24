# CapCut PC（デスクトップ版）調査レポート — MAZE WIND RETREAT 用

調査日：2026-09-24
対象：CapCut PC（Windows / Mac）デスクトップ版
目的：4本（サイネージ 30s/15s 1920x1080 無音ループ、Instagram Reels 30s/15s 1080x1920 音あり）を CapCut PC で組み立てるときに、どの機能を使い、どの機能を使わないかを決める。

---

## 0. 調査方法と限界（先に読んでください）

- この環境では **capcut.com などのページ本文を直接開けませんでした**（ネットワークの制限で WebFetch がブロックされた）。内容は **Web検索結果の要約**（検索エンジンがページから抜き出した文）から組み立てています。
- そのため、**メニュー名（特に日本語UIの表記）は一部が推定**です。推定のものには「（推定）」と付けています。CapCut はUIの表記を頻繁に変えるので、**最終的には実機の画面で確認してください**。
- 公式ページには日付が書かれていないものが多く、下の「日付」は「検索で確認した日」または記事内の日付です。
- CapCut の公式 "resource" / "create" 系の記事は SEO 用の解説記事で、機能の説明が一般論になっていることがあります。ヘルプセンターや規約（clause）ほど正確ではありません。

---

## 1. 情報源一覧

| # | URL | 種類 | 日付 | メモ |
|---|---|---|---|---|
| S1 | https://www.capcut.com/resource/new-release | 公式・新機能一覧 | 2026-09 検索時点（本文未取得） | 本文を取得できず |
| S2 | https://www.capcut.com/clause/material-license-agreement | 公式・素材ライセンス規約 | 2026-09 検索時点（改定日未確認） | 商用/非商用素材の区分 |
| S3 | https://www.capcut.com/clause/terms-of-service | 公式・利用規約 | 同上 | |
| S4 | https://www.capcut.com/ja-jp/help/monthly-and-yearly-plans | 公式ヘルプ（日本語）・Proプラン | 同上 | |
| S5 | https://www.capcut.com/help/export-videos-in-capcut | 公式ヘルプ・書き出し | 同上 | |
| S6 | https://www.capcut.com/help/video-without-watermark | 公式ヘルプ・ウォーターマーク | 同上 | |
| S7 | https://www.capcut.com/resource/seedance-2-5-is-now-on-capcut-pc | 公式・Seedance 2.5 が CapCut PC に | 2026-07-31 以降（Seedance 2.5 の世界公開日が 7/31） | Pro/Ultra でクレジット消費 |
| S8 | https://www.capcut.com/tools/desktop-video-editor | 公式・デスクトップ版ページ | 2026-09 | タイトルに "Dreamina Seedance 2.5" |
| S9 | https://www.capcut.com/tools/color-grading , /tools/match-color-with-ai , /resource/luts-for-color-grading , /resource/capcut-color-grading | 公式・色補正/カラーマッチ/LUT | 日付なし | .cube / .3dl の LUT 読み込み |
| S10 | https://www.capcut.com/create/video-color-space-srgb-rec-709-rec-2020 | 公式・色空間 | 日付なし | SDR は Rec.709 推奨 |
| S11 | https://www.capcut.com/tools/relight-videos-with-ai | 公式・AIリライト | 日付なし | |
| S12 | https://www.capcut.com/resource/optical-flow-in-capcut | 公式・オプティカルフロー | 2025 | デスクトップのみ |
| S13 | https://www.capcut.com/resource/ai-video-stabilizer , /tools/auto-reframe , /resource/ai-enhance-video | 公式・手ぶれ補正/自動リフレーム/高画質化 | 2025 | |
| S14 | https://www.capcut.com/tools/ai-music-generator , /resource/top-5-ai-background-music-generators | 公式・AI音楽 | 2026 | 「商用も可」と書いてあるが規約で要確認 |
| S15 | https://9to5google.com/2026/05/21/capcut-announces-partnership-with-gemini-app/ | 報道 | 2026-05-21 | Gemini アプリと提携（リリース時期は未定） |
| S16 | https://bigvu.tv/blog/capcut-free-vs-pro-what-2026s-restructure-actually-gives-you/ | レビュー | 2026 | Standard $9.99 / Pro $19.99 |
| S17 | https://fluxnote.io/guides/capcut-ai-features-free-vs-paid-2026 | レビュー | 2026 | どのAI機能が有料か |
| S18 | https://note.com/witty_ixora1236/n/nf25b57093ba6 ほか（aipicks.jp/mag/capcut-ai-guide-2026） | 日本語レビュー | 2026-09-23 時点の記載 | PC版 Pro 参考価格 月額2,180円 / 年額19,800円 |
| S19 | https://detail.chiebukuro.yahoo.co.jp/qa/question_detail/q13303916088 | Q&A（ユーザー報告） | 日付未確認 | PC版で書き出すと自動で Rec.2100 HLG になり白飛びする報告 |
| S20 | https://capcutguide.com/capcut-hdr-sdr-washed-out/ , https://www.miracamp.com/learn/capcut/whats-the-best-color-space | 非公式ガイド | 2026 | HDR→SDRの白っぽさ対策。H.264 では HDR メタデータが落ちるとの記載 |
| S21 | https://kunver.dev/blog/capcut-export-settings-explained/ | 非公式ガイド | 日付未確認 | 書き出し項目（Color space に Rec.2100 HLG/PQ） |
| S22 | https://studio.monoist.work/entry/export-settings-in-capcut | 日本語ガイド | 日付未確認 | 書き出し画面の日本語項目 |
| S23 | https://blazemedia.co.uk/2025/06/26/capcut-2025-copyright-changes , https://omniweb.jp/m49/ , https://do-gaku.com/column/capcut-business-rules | レビュー（規約改定） | 2025-06-26 / 2026 | 2025年6月の規約改定（アップロード内容へのライセンス条項） |
| S24 | https://www.moviehowto.jp/capcut-commercial-use/ , https://yokotashurin.com/etc/capcut-business.html | 日本語ガイド | 日付未確認 | 「商用」フィルターの場所。商用音源は TikTok 内のみ |
| S25 | https://www.conbu.jp/tool/font-license/noto-sans-jp.html | フォントライセンス | 日付未確認 | Noto Sans JP は SIL OFL 1.1 |

---

## 2. 最近の変化（2025〜2026）と料金

### 2-1. 大きな変化
| 時期 | 内容 | 情報源 | 確度 |
|---|---|---|---|
| 2025 | 有料プランが **Standard（約$9.99/月）と Pro（約$19.99/月）** の2段に分かれた | S16, S17 | 中（レビュー記事） |
| 2025-06 | 利用規約の改定。ユーザーがアップロードした内容に ByteDance が広いライセンス（全世界・無償・取消不能・サブライセンス可など）を持つ条項が話題になった | S23 | 中（報道・解説。原文未確認） |
| 2026前半 | 画像生成（Seedream 5.0、Nano Banana Pro など）が CapCut 内で使えるように | 検索結果（BIGVU 等） | 低〜中 |
| 2026-05-21 | **Google Gemini アプリと提携を発表**。Gemini のチャットから CapCut の編集機能を呼び出す構想。**配信開始時期は未定**と報道 | S15 | 高（複数報道） |
| 2026-07-31〜 | **Seedance 2.5（AI動画生成）が CapCut PC に搭載**。4K・最長30秒の生成、Pro/Ultra のクレジット消費 | S7, S8 | 高（公式） |
| 2026 | 「Ultra」という上位プランの名前が公式ページに出てくる（S7）。料金は未確認 | S7 | 低 |

### 2-2. 料金（2026年9月時点）
- **無料**：カット、分割、マルチトラック、キーフレーム、クロマキー、速度変更、フィルター、無料の音楽・効果音、**1080p 書き出し**（S16）。
- **Standard（約$9.99/月）**：ウォーターマークを外す、追加のテンプレ・トランジション・文字スタイル・エフェクト。4K と AI 機能の全部は入らない（S16）。
- **Pro（約$19.99/月、年$179.99）**：4K 書き出し、カメラトラッキング、ボーカル分離、ちらつき除去（Remove flickers）、高画質化などの AI 機能、クラウド容量（S16, S17）。
- **日本の PC 版 Pro の参考価格**：月額 2,180円 / 年額 19,800円（2026-09-23 時点の記事による。S18）。**地域・購入経路・キャンペーンで変わる**。
- **注意**：どの機能が無料でどれが Pro かは、**時期と地域でよく変わります**。CapCut PC では、Pro の機能を使うとタイムライン上に「Pro」マークが付き、書き出し時に「Pro 機能を外すか購読するか」を聞かれます。最終判断は実機のマークで行ってください。

**今回の結論**：今回の作業（ハードカット、PNG 重ね、BGM、軽い色の調整、1080p 書き出し）は **無料版で足りる見込み** です。Pro が必要になりそうなのは「高画質化」「ちらつき除去」「オプティカルフロー」など、**今回は使わない機能** だけです。

---

## 3. 機能表

凡例：「使う」＝今回使う、「使わない」＝今回使わない、「部分的」＝条件付きで使う

| 機能 | どこにあるか（英語UI / 日本語UI） | 無料か | 今回 | 理由 |
|---|---|---|---|---|
| カット・分割・並べる（ハードカット） | タイムライン / Split（分割）ツール、ショートカット Ctrl/Cmd+B | 無料 | **使う** | 素材は切り出し済み。並べてハードカットでつなぐだけ |
| トランジション | 左上 Transitions / トランジション | 一部 Pro | **使わない** | ルール：ハードカットのみ |
| エフェクト・フィルター | Effects / エフェクト、Filters / フィルター | 一部 Pro | **使わない** | 広告っぽい・テンプレっぽい見た目になる。色は素材側で仕上げ済み |
| テンプレート | Templates / テンプレート | 一部 Pro | **使わない** | 「テンプレっぽく見せない」ルール。商用利用の制限もある |
| 透過 PNG の重ね（文字・ロゴ） | Media / メディア → Import / インポートし、メイン動画の上のトラックへ置く | 無料 | **使う** | PNG の透過はそのまま使える（CapCut は PNG のアルファに対応、一般的な挙動）。**大きさは Position & Size の Scale を 100% のまま** にし、1920x1080 / 1080x1920 で作った PNG をそのまま置く |
| 不透明度・フェード（PNG） | 右パネル Video → Basic → Blend / Opacity（動画 → 基本 → ブレンド → 不透明度、推定） ＋ キーフレーム（◇）。または Animation → In / Out → Fade In / Fade Out（アニメーション → イン/アウト → フェードイン/フェードアウト） | 無料 | **使う**（文字・ロゴの出入りだけ） | 映像はハードカット。文字の出入りだけ静かにフェード |
| キーフレーム | 各パラメータ横の ◇ | 無料 | **部分的** | 文字の不透明度と、音声の最後のフェード程度 |
| 手動の色調整 | 右パネル Adjust → Basic（調整 → 基本）：Temperature, Tint, Saturation, Exposure, Contrast, Highlights, Shadows など。HSL / Curves / Color wheels（HSL / カーブ / カラーホイール） | 無料（一部項目が Pro の場合あり） | **部分的** | 素材はグレーディング済み。**クリップ間の差を小さく直すだけ**。Adjustment layer（調整レイヤー）で全体に同じ値を当てるのがよい |
| LUT の読み込み | Adjust → LUT → Import（調整 → LUT → インポート）。.cube / .3dl | 無料（と記載。実機で確認） | **使わない（原則）** | 素材側で色変換と仕上げが終わっている。二重にかけると色がずれる。例外：グレーディング担当者が「最後に当てる LUT」を渡した場合だけ使う |
| AI 色補正（AI Color correction / AIカラー補正） | Adjust → Basic の中の「Color correction」「Auto adjust」（推定） | 不明（Pro の可能性） | **使わない** | 自動で彩度とコントラストを上げがちで「AIっぽい」見た目になる。海の青がずれるおそれ |
| カラーマッチ（Color match） | Adjust 内（推定） | 不明 | **部分的（試すだけ）** | 2つのクリップの色の差が目立つときだけ試し、強さ（Intensity）を 30% 以下にする。仕上がりが不自然なら使わない |
| AI リライト（Relight） | Video → Basic → Relight（動画 → 基本 → リライト、推定） | 不明（Pro の可能性） | **使わない** | 光を作る機能。自然さが崩れやすく「AI感」が出る |
| 自動リフレーム（Auto reframe） | Video → Basic 付近、または Ratio 変更時（推定） | 不明 | **使わない** | 素材は縦・横それぞれクロップ済み |
| 手ぶれ補正（Stabilize） | Video → Basic → Stabilize（動画 → 基本 → 手ぶれ補正）。Recommended / Minimum cut / Most stable | 無料（と説明されることが多い） | **使わない** | 補正済み。二重にかけると画面が拡大され、揺れ方が不自然になる |
| 映像のノイズ除去・ちらつき除去（Reduce noise / Remove flickers） | Video → Basic | Pro（ちらつき除去は Pro と明記、S16） | **使わない** | 素材側で処理済み。質感がつるっとして「AI感」が出る |
| 高画質化（Enhance quality / 超解像） | Video → Basic → Enhance quality（HD / UHD / 4K） | Pro | **使わない** | 1080p で書き出すので不要。「AI感」のもと |
| オプティカルフロー（Smooth slow-mo / Optical flow） | Speed → Smooth slow-mo（速度 → スムーズスローモーション、推定） | Pro | **使わない** | 速度は素材側で調整済み。**速度タブを触らない**こと |
| 音声のノイズ除去・声の補正（Reduce noise / Enhance voice / ノイズを低減・音声を強調） | Audio → Basic（音声 → 基本、推定） | ノイズ除去は無料、声の補正は Pro の場合あり | **使わない** | 現場の環境音は調整済みのものが来る。波や風の音を「ノイズ」として消してしまう |
| 音量の正規化（Normalize loudness / ラウドネスを正規化） | Audio → Basic | 無料（推定） | **部分的** | BGM と環境音の音量をそろえる補助として。最後は耳で確認 |
| 音声のフェード | 音声クリップを選ぶ → Audio → Basic → Fade in / Fade out（フェードイン / フェードアウト） | 無料 | **使う**（Instagram 版のみ） | BGM の頭とお尻。映像のハードカットとは別 |
| ビート検出（Beat / Auto beat） | 音声クリップを選ぶ → タイムライン上の Beat ボタン（ビート / 自動ビート、推定） | 無料 | **部分的** | 印（マーカー）を付けるだけ。**カット位置の目安**にする。自動でカットする機能（Auto cut / オートカット）は使わない |
| オートカット（Auto cut） | Templates 付近の AI 機能（推定） | 不明 | **使わない** | リズムに合わせて自動で切る機能で、テンプレっぽさが出る |
| CapCut 内の音楽ライブラリ | Audio → Music（オーディオ → 音楽） | 無料 / Pro 素材あり | **使わない** | 規約上、ライブラリ曲は個人・非商用のみ。「商用」の音源も使える場所が CapCut / TikTok / TikTok for Business 内に限られる（S2, S24）。**屋外サイネージと Instagram 広告には使えない** |
| AI 音楽生成（AI music） | Audio → AI music（オーディオ → AI音楽、推定）でプロンプトを入れて Generate | 不明（クレジット消費の可能性） | **使わない（原則）** | 公式ツールページは「商用も可」と書くが、**規約上の扱いを確認できていない**。店舗の広告として外に出す以上、ライセンスがはっきりした外部の BGM を使う方が安全。ためしに使う場合のプロンプトは 4章 |
| AI キャプション（Auto captions / 自動キャプション） | Text → Auto captions（テキスト → 自動キャプション） | 一部無料 | **使わない** | 大きな文字は禁止。文字は PNG で入れる |
| テキスト機能・フォント | Text / テキスト | 一部 Pro | **使わない** | 文字は PNG で来る。CapCut 内蔵フォントは商用の条件が素材ごとに違う |
| スマートカットアウト（背景除去） | Video → Remove BG / Cutout（動画 → 背景を削除、推定） | 一部 Pro | **使わない** | 必要な場面がない。「AI感」が出る |
| AI 動画生成（Seedance 2.5）・テキストから動画・AI スクリプト | 左上 AI 系メニュー（AI video / AI media など、推定） | Pro/Ultra、クレジット | **使わない** | 実写だけで作る方針。生成映像は「AI感」の原因 |
| AI 編集アシスタント（チャットで編集） | CapCut PC 本体の中に、**会話で編集する機能はまだ確認できていない**。2026-05 に Gemini アプリとの提携を発表したが、配信時期は未定（S15） | — | **使わない** | 確認できない。EDL どおり手で組む方が正確 |
| 書き出し（Export） | 右上 Export / エクスポート | 1080p は無料 | **使う** | 5章の設定を使う |
| 音声なしの書き出し | 書き出し画面で **Audio（音声）のチェックを外しても映像から音は消えない**（Audio にチェックを入れると、別に MP3 などの音声ファイルが出るだけ、S21/S22）。**無音にするには、タイムラインで全クリップをミュートするか音声トラックを消す** | 無料 | **使う**（サイネージ版） | サイネージ版は無音。音声トラックが無音で残る可能性あり（下の注意） |

---

## 4. 使う機能の「そのまま使える」設定・プロンプト

### 4-1. プロジェクトの準備
1. 新しいプロジェクトを作る。
2. 右上の **Settings / 設定**（または メニュー → **Project settings / プロジェクト設定**、推定）で：
   - **Ratio / 比率**：サイネージ版 `16:9`（1920x1080）、Instagram 版 `9:16`（1080x1920）
   - **Frame rate / フレームレート**：`30fps`（素材と同じ値にする。素材が 29.97 の場合も 30 を選ぶ）
   - **Color space / カラースペース**：`Rec.709 SDR`（HDR 系が選ばれていたら直す）
3. 素材（H.264 1080p SDR）と PNG を **Import / インポート** する。
4. サイネージ版と Instagram 版は別のプロジェクトにする（比率が違うため）。

### 4-2. PNG の重ね（文字・ロゴ）
- 映像トラックの上のトラックに PNG を置く。
- **Position & Size**（位置とサイズ）：Scale `100%`、X `0`、Y `0`。PNG は書き出しサイズと同じ大きさで作ってあるので動かさない。
- 出入り：**Animation → In → Fade In**（長さ `0.5s`）、**Out → Fade Out**（長さ `0.5s`）。
  - キーフレームで作る場合：Opacity `0% → 100%`（0.5秒）、終わりに `100% → 0%`（0.5秒）。
- **Blend mode（ブレンドモード）は Normal / 標準のまま**。

### 4-3. 色の微調整（全体）
- **Adjustment layer / 調整レイヤー**（Adjust → Custom adjust / カスタム調整 をトラックに置く、推定）を全体にかける。クリップ1本だけずれている場合はそのクリップだけ直す。
- 目安の範囲（**これ以上は動かさない**）：
  - Temperature（色温度）：`-5 〜 +5`
  - Tint（色合い）：`-3 〜 +3`
  - Saturation（彩度）：`-5 〜 +5`
  - Contrast（コントラスト）：`-5 〜 +5`
  - Highlights（ハイライト）：`-10 〜 0`（白飛び気味の空・海面だけ）
  - Sharpen（シャープ）、Vignette（ビネット）、Grain（粒子）：**`0` のまま**
- 海の青：**HSL** で Blue / Cyan の Hue を大きく動かさない（±3 以内）。**ティールに寄せない**。
- オレンジ（肌・木）とティール（影）を分けるような調整（カラーホイールで影を青緑、ハイライトをオレンジ）は **しない**。

### 4-4. カラーマッチ（差が目立つときだけ）
- 基準にするクリップ（一番きれいな海と水平線のクリップ）を選び、ずれているクリップに適用。
- **Intensity / 強さ：`20〜30%`**。見比べて不自然なら削除する。

### 4-5. ビート検出（Instagram 版で、目安として）
- BGM を置く → 選択 → タイムラインの **Beat / ビート** → Auto beat（自動）→ 印が付く。
- 印は **カット位置の候補** として見るだけ。全部の印に合わせない。EDL のカット位置を優先。

### 4-6. BGM と環境音（Instagram 版）
- BGM は **外部の、商用ライセンスがはっきりした曲** を使う（例：Artlist、Epidemic Sound、日本なら Audiostock など。**購入時のライセンスが「広告・店舗・SNS」を含むか確認**）。
- 音量の目安：環境音 `0 dB`（基準）、BGM は `-12 〜 -18 dB` から始めて耳で決める。
- **Fade in / フェードイン** `0.5s`、**Fade out / フェードアウト** `1.0〜1.5s`。
- Reduce noise / ノイズ低減、Enhance voice / 音声強調、Voice changer（ボイスチェンジャー）：**オフ**。

### 4-7. AI 音楽（使う場合のみ。原則は使わない）
使う前に、書き出し時・規約上で「商用利用可」の表示があるか必ず確認。
- 日本語：
  `静かでシネマティックなアンビエント。ピアノとやわらかいパッド、ゆったりしたテンポ（BPM 70前後）。海辺のカフェの朝の空気。ドラムなし、ボーカルなし、盛り上がりなし。広告っぽくしない。30秒。`
- English:
  `Calm, cinematic ambient track. Soft felt piano and warm pads, slow tempo around 70 BPM. Morning air at a seaside café. No drums, no vocals, no build-up or drop. Understated, not like a commercial. 30 seconds.`

### 4-8. 書き出し設定（Export / エクスポート）
| 項目（英 / 日） | サイネージ 30s・15s | Instagram 30s・15s |
|---|---|---|
| Resolution / 解像度 | 1080p（1920x1080） | 1080p（1080x1920） |
| Bit rate / ビットレート | Custom（カスタム）`20〜25 Mbps`。無い場合は Higher（高） | Custom `12〜16 Mbps`（Instagram 側で再圧縮される）。無い場合は Recommended（推奨）か Higher |
| Codec / コーデック | **H.264**（サイネージ機の対応を先に確認。HEVC は機器によって再生できない） | **H.264** |
| Format / 形式 | MP4 | MP4 |
| Frame rate / フレームレート | 30fps | 30fps |
| Color space / カラースペース（項目がある場合） | **Rec.709（SDR）**。Rec.2100 HLG / PQ にしない | **Rec.709（SDR）** |
| Audio / 音声（別ファイルの書き出し） | チェックなし | チェックなし |
| Add ending（エンディングを追加、ある場合） | オフ | オフ |
| Copyright check（著作権チェック、ある場合） | 任意 | ON で確認 |
| 無音にする方法 | **全クリップをミュート**（トラック左のスピーカーアイコン）＋ 音声クリップを削除 | — |

- **サイネージ版の無音**：CapCut の書き出しは、ミュートしても **無音の音声トラックが入る可能性** があります（未確認）。サイネージ機が「音声トラックなし」を必要とする場合は、書き出し後に `ffmpeg -i in.mp4 -c:v copy -an out.mp4` で音声トラックを取り除いてください（再圧縮なし）。
- **ループ**：CapCut にはループ再生用の書き出し設定はありません。最初と最後のカットがつながるように EDL で決め、ループはサイネージ機側で設定します。

---

## 5. 注意点（既知の問題）

### 5-1. HDR / iPhone 素材の色ずれ
- ユーザー報告：**PC 版で書き出すと、カラースペースが自動で Rec.2100 HLG になり、白飛び・白っぽくなる**（S19）。iPhone の HDR（Dolby Vision / HLG）素材がタイムラインにあると、プロジェクトが HDR 扱いになるケースがあると考えられます。
- 非公式ガイド：「素材が SDR でプロジェクトが HDR だとずれる。プロジェクトを Rec.709 SDR にして書き出し直す」「H.264 では HDR のメタデータが落ちる」（S20）。
- **今回の対策**：素材はすでに SDR（Rec.709）に変換済みなので、原因の大部分はなくなります。それでも：
  1. プロジェクト設定と書き出しの **Color space を必ず Rec.709 にする**。
  2. **iPhone の元素材（HDR の .mov）を1本でも混ぜない**。
  3. 書き出した MP4 を `ffprobe` で確認：`color_primaries=bt709`、`color_transfer=bt709`、`color_space=bt709` であること。`arib-std-b67`（HLG）や `smpte2084`（PQ）が出たら書き出し直す。

### 5-2. 書き出し後のガンマずれ（明るさ・コントラストの変化）
- Mac では、QuickTime 再生と CapCut のプレビューで明るさが違って見える問題（Rec.709 のガンマの扱いの違い）が、一般的に知られています。**CapCut 特有の不具合かどうかは確認できませんでした**。
- 対策：色の判断は **書き出したファイルを、実際に流す環境（サイネージ機・スマホの Instagram）で見て** 行う。CapCut のプレビューだけで決めない。

### 5-3. Pro 制限とウォーターマーク
- Pro 素材・Pro 機能を1つでも使うと、書き出し時に購読を求められます（S6）。**無料で書き出すには、Pro マークの付いたものを全部外す**。
- エンディングの CapCut ロゴ（Add ending）はオフにできる（S6）。
- 今回の手順（ハードカット、PNG、手動の色調整、外部 BGM）だけなら、Pro 表示は出ない見込みです。出たら、どの機能か確認して外す。

### 5-4. 商用利用（とても大事）
- CapCut の素材ライセンス規約では、素材は **「非商用のみ」** と **「商用も可（Dual Use）」** に分かれます。**1つでも非商用の素材が入ると、動画全体が非商用扱い** になります（S2）。
- 音楽ライブラリの曲は、原則 **個人・非商用のみ**（S2）。「商用」フィルター（Audio → フィルターのアイコン → Commercial / 商用）で出る音源も、**CapCut・TikTok・TikTok for Business の中でしか使えない**と説明されています（S2, S24）。→ **屋外サイネージと Instagram には使えない** と考えるべきです。
- テンプレート、エフェクト、ステッカー、CapCut 内蔵フォントも素材ごとに条件が違います（S2）。今回は **全部使わない** ので問題になりません。
- 2025年6月の規約改定で、アップロードしたコンテンツへの ByteDance のライセンス条項が話題になりました（S23）。お店の素材（ロゴ、空撮）をクラウド同期（Cloud / Space）に上げない、**ローカル保存のプロジェクト** で作業するのが安心です（原文未確認）。

### 5-5. フォント（Noto Sans JP）
- 文字は PNG で来るので、CapCut の中でフォントを使う必要はありません。
- Noto Sans JP は **SIL Open Font License 1.1** で、商用利用 OK・表記不要（S25）。CapCut で使いたい場合は PC にインストールすれば、CapCut の「System fonts / システムフォント」から選べます（一般的な挙動、未確認）。CapCut 内蔵の「Noto Sans JP」があるかは未確認。

---

## 6. 確認できなかったこと（実機で確かめてください）

1. 日本語 UI の正しいメニュー名（本文中の「推定」の箇所）。
2. 書き出し画面に Color space（カラースペース）の項目があるか、デフォルトが何か。
3. ミュートして書き出したとき、音声トラックが残るか消えるか。
4. AI 色補正・カラーマッチ・LUT 読み込み・リライトが無料か Pro か（地域・時期で変わる）。
5. AI 音楽で作った曲を、屋外サイネージや Instagram で使ってよいか（規約の原文）。
6. CapCut PC 本体に、チャットで編集する AI アシスタントが入っているか（2026年9月時点で確認できず。Gemini 連携は発表のみ）。
7. 「Ultra」プランの内容と料金。

---

## 7. まとめ

- **CapCut PC は「素材を並べて、PNG を重ねて、BGM を入れて、書き出す」だけに使う。** AI 機能はほぼ全部使わない。
- 無料版で足りる見込み。Pro 表示が出たら、その機能を外す。
- **色**：プロジェクトと書き出しを必ず Rec.709 SDR にする。調整は小さく。LUT・AI 色補正・リライト・高画質化は使わない。
- **音**：CapCut の音楽ライブラリと「商用」音源は使えない（TikTok 内限定）。BGM は外部の商用ライセンス曲。
- **書き出し**：1080p / H.264 / MP4 / 30fps / Rec.709。サイネージ版は必要なら ffmpeg で音声トラックを取る。
