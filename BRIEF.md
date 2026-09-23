# MAZE WIND RETREAT / Seaside Cafe — SNS Short Movie

Master Promptの要約です。制作中の判断はここを基準にします。
**FIXED RULES（§1）は変更禁止。** 変えてよいのは実制作パラメータ（カット、秒数、編集点、Speed、BGM、Sound、Color、文字の配置、細かなTransition）だけ。

---

## 1. 固定したCreative Rules（変更禁止）

| # | ルール | 実務上の意味 |
|---|---|---|
| 1 | 広告として緻密に設計する。ただし広告っぽく見せない | 「高知のすごく良い場所を見つけた」という投稿の温度感。セール感、企業PV感、説明動画感、テンプレ感はNG |
| 2 | 映像美を最優先。主役は **海・水平線・オーシャンビュー** | **場所 → 空気 → 料理** の順に見せる。グルメ動画にはしない |
| 3 | 視聴後に残したい感情は「え、こんな場所が高知に？」「行ってみたい」 | 「美味しそう」で終わらせない |
| 4 | Premium / Cinematic / Minimal / Natural / Coastal / Modern Japanese | Apple的な余白と上質さ。冒頭だけはSNSの強いフックにする（**SNS Hook × Cinematic Film**） |
| 5 | 冒頭に広告コピーを並べない | 海の映像そのものの強さで始める。「MAZE WIND」は海や空間と重ねて自然に浮かび上がらせる。巨大文字は使わない |

**付随ルール（実制作の制約）**
- 編集：Hard Cut / Match Cut / 動きで繋ぐ。派手なトランジション、CapCutテンプレ、過剰なZoom・Speed Ramp、SE連打は禁止
- ペース：冒頭FAST → 中盤MEDIUM → 料理 FAST→SLOW→FAST → ラストSLOW。海の重要カットは少し長めに残す
- 文字：少なく、白、必要なら薄いShadow。縁取り・グラデーションは禁止
- 音：BGMだけで仕上げない。波・風・食器などの現地音で「その場にいる感覚」を作る
- 色：海の青が主役。Natural / Clean。Teal & Orangeは禁止。料理と肌の暖かさは残す。HDR→SDR変換を管理する
- End Card：海の映像に情報が自然に浮かぶデザイン。MAPは 高知県 → 土佐市 → MAZE WIND のMinimal Motion

## 2. 15秒版の目的：Discovery / Impact

指を止めさせ、「何ここ？」→「高知なの？」→「行きたい」→「保存しよう」の流れを15秒で作る。説明は最小限にする。

| 秒 | パート | 内容 |
|---|---|---|
| 0.0–1.5 | HOOK | いちばん強い海・水平線。ロゴや説明から始めない |
| 1.5–3.5 | PLACE REVEAL | 店内・窓・テーブル → 海 |
| 3.5–7.0 | FOOD #1 | パスタ（持ち上げ、質感、湯気）をテンポよく |
| 7.0–10.0 | FOOD #2 | ケーキ。フォークを入れる瞬間で少しスローにする |
| 10.0–12.5 | EXPERIENCE | 料理＋水平線で「この時間を過ごしたい」と思わせる |
| 12.5–15.0 | END | 店名 / SNS / Location（＋簡易MAP） |

## 3. 30秒版の目的：Experience / Story

15秒版を伸ばしたものにはしない。**実際に訪れたような小さな体験**を作る。
海 → 発見 → 入店 → 窓の向こうの海 → 席 → パスタ → ケーキ → 海を見ながら過ごす → 夕方・波・空間 → 店名 → Location/SNS。
「入口 → 店内 → 海が見える」視線移動の素材があれば、軸として使う。

## 4. 素材を受け取った後の作業手順

1. 全素材を解析（解像度、fps、HDR/SDR、長さ、音声の有無、手ブレ、露出）
2. SHOT CATEGORY別に分類：A HERO OCEAN / B ARRIVAL / C INTERIOR / D PASTA / E CAKE / F EXPERIENCE / G DETAIL / H LOCATION
3. Best Takeを選定
4. 15秒版EDLを作成
5. 30秒版EDLを作成
6. 各カットに Source / In / Out / Duration / Speed / Crop / Transition / Text / Audio / Purpose を記入
7. 編集案を提示 → **承認を待つ**
8. 承認後に編集。FFmpegとCapCutを組み合わせたHybrid Workflowで、品質が上がる方を選ぶ

素材が足りずFIXED RULESを守れない場合は、方向性を変えずに「不足素材 / 必要な理由 / 追加ショット / iPhone 16 Pro Maxでの撮り方」を提示する。

## フォルダ

- `footage/` — iPhone原素材（Git管理外）
- `edl/` — Edit Decision List、CapCut指示書
- `exports/` — 書き出し（Git管理外）
