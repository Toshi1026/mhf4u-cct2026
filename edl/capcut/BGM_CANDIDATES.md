# BGM候補：Instagram 30秒版・15秒版（OpenTracks／旧DOVA-SYNDROME）

作成：2026-09-24 ／ 対象：IG30（Experience／Story）・IG15（Discovery／Impact）で**同じ曲を1曲**使う前提

---

## 0. 先に読むこと（今回できなかったこと）

**音源ファイルを、この作業環境へダウンロードできませんでした。** OpenTracks のページ（検索・曲の詳細・ライセンス）は取得できますが、音声ファイルの配信元 `dova-worker.tracks-cid.workers.dev` への接続が、この環境のネットワーク制限（組織のアクセス制御ポリシー）で拒否されます（HTTP 403）。試聴プレーヤーと、公式の「DOWNLOAD FILE」ボタン（`/bgm/detail/<ID>/download` にPOSTすると、この配信元へ302で転送される）のどちらも同じ配信元です。YouTube（公式の試聴動画、使用例の確認）もこの環境からは開けません。回避策は使っていません。

そのため次の作業は**まだ済んでいません**。

| 手順 | 状態 |
|---|---|
| 1. ライセンスの確認 | 済み（下の §1） |
| 2. カタログ検索（候補64曲の詳細を取得。検索結果の全257曲はJSONで保存） | 済み（§4・付録） |
| 3. 音源の解析（BPM・音量の推移・展開・終わり方・声の有無） | **未実施**。解析スクリプトは用意済みで、テスト信号では最後まで動くことを確認した |
| 4. 使用例の調査 | Web検索で実施。YouTubeを直接開けないため、確認できた範囲は限られる（§3） |
| 5. 試聴用プレビュー（bgm_<n>_*.mp4 / .mp3） | **未作成**。mp3を置けばコマンド1つで作れる |

以下の3曲は、**ページの情報（作曲者の説明文、タグ、BPMの記載、長さ、DL数）だけで選んだ暫定の候補**です。耳で確かめるまで確定にはできません。

### 仕上げの手順（mp3を置けば、あとは自動）
1. 下の3曲のページ →「音楽素材ダウンロードページへ」→「DOWNLOAD FILE」（トラック1）でmp3を保存する
2. 保存したファイルを曲IDの名前に変えて置く：`exports/bgm/candidates/17543.mp3`、`10647.mp3`、`9307.mp3`
3. 実行する
   ```
   python3 tools/bgm_preview.py analyze   # → exports/bgm/analysis.md（BPM・セクションの変わり目・終わり方・声の目安）
   python3 tools/bgm_preview.py build     # → exports/bgm/bgm_<n>_<short>_ig30.mp4 / _ig15.mp4 / .mp3、bgm_cutpoints.json
   ```
   開始位置が音楽的におかしいときは、耳で決めた秒数を渡して作り直す：`python3 tools/bgm_preview.py build --n 1 --ig30-start 12.3 --ig15-start 40.1`

---

## 1. ライセンスの要点（OpenTracks 音源利用ライセンス）

出典：[音源利用ライセンス](https://opentracks.com/help/articles/license/)／[利用できる人・利用できる用途](https://opentracks.com/help/articles/license-usage/)／[YouTubeでの利用](https://opentracks.com/help/articles/youtube/)（2026-09-24 に確認。2026-09-15 に DOVA-SYNDROME から OpenTracks に名称変更）

| 項目 | 内容（引用） | このプロジェクトでの判断 |
|---|---|---|
| 誰が使えるか | 「個人・法人を問わず利用できます。」「営利・非営利を問わず利用できます。」 | お店のPR動画に使える |
| 料金・クレジット | 「著作権使用料その他の利用料を請求することはありません。」「著作権表示・提供等の表示は不要です。」FAQ：「必須ではありませんが、可能な限り記載をお願いします。」 | 無料。クレジットは任意。キャプションに入れるならお店と相談（入れると広告っぽさが出るので、入れないのが自然） |
| 広告・SNS | 背景音楽の例に「テレビ番組、CMにおける利用」「YouTube、TikTok…その他のオンライン映像・コンテンツ配信での利用」。FAQ：「BGMを使用して制作した動画をYouTube・TikTok・Instagram等で配信しても問題ありませんか？ → ご利用ください。」「屋外広告（ビジョンや街頭ディスプレイ等）のBGMとして…問題ありません。」 | IG広告・通常投稿・屋外サイネージのどれも可 |
| 条件：背景音楽であること | 「音源を何か別の主となるものの背景（従）として利用すること」。音楽が主役の「作業用BGM」「ヒーリングミュージック」などは禁止 | 映像が主役なので問題ない |
| 編集 | 「音源の加工（編集・エフェクト・フェードイン/アウト等）を行うことができます。」FAQ：「イントロをカットする等の編集…フェードイン（アウト）・ブツ切り等、お好みの編集をしてください。」ただし「元の音源から乖離、あるいは評価を著しく損なうと判断される加工は不可」 | 切り詰め・つなぎ・フェード・ダッキングは可 |
| 作曲者ごとの条件 | 「作曲・制作者が別途利用条件を設けていることがあります。その場合は、作曲・制作者の利用条件が優先され」る（料金・クレジット不要と禁止事項は、サイトのライセンスが優先） | 候補ごとに作曲者ページの「素材利用条件」を確認した（§2） |
| 主な禁止事項 | 政治・宗教の制作物、音源そのものの配布・販売、「音源をAIのトレーニングに使用すること」、JASRAC等への登録、「音楽としてUGCサービスその他のプラットフォームにおけるフィンガープリント（YouTubeにおいてはContent ID…）に登録すること」 | 該当しない。**完成動画をYouTubeやMetaの権利管理に登録しない**ことだけ注意 |
| YouTube | 「ライセンスの範囲内であれば問題なく無料で利用いただけます。」誤った権利の申し立てが来ることがあり、その場合は異議申し立てで対応（申し立てをする団体の一覧あり） | IGが主。YouTubeにも上げるなら、申し立てが来たときの手順だけ知っておく |
| 公開が終わったあと | 「当サイトでの音源の公開終了後も、当サイトでダウンロードした音源に限り…利用することができます。」 | 使う曲は公式のダウンロードボタンで保存し、保存日を記録しておく |
| AIで作った曲 | 作曲者参加規約で禁止（「AI…によって⽣成されたものでないこと」） | 候補はすべて人が作った曲という前提 |
| 保証・責任 | 「個別の許可・判断・お問合せには対応いたしかねます。…自己責任でご判断ください。」 | 最終判断はお店側で |

---

## 2. 候補3曲（暫定。音を聴いての確認が必要）

3曲は、ギター＋パッド／ピアノ＋パッド／軽いビートの3系統に分けた。作曲者も3人とも別。

### 候補1（推し）：高原の小さなカフェにて — のる
| 項目 | 内容 |
|---|---|
| URL | https://opentracks.com/bgm/detail/17543 |
| 長さ | 3:34（トラック1。トラック2はループ版で、入りが違う） |
| BPM | 記載なし・**未測定**。タグは「遅い」「四拍子」 |
| 楽器 | アコースティックギター、シンセパッド（タグにピアノもある） |
| 声 | なし（ボーカルなしで検索。タグにボーカル・コーラスなし） |
| 作曲者の個別条件 | なし（「OpenTracks…の音源利用ライセンスに準拠します」）。説明文：クレジットは「強制ではありません」。個別の使用許可の問い合わせには返信しない、とのこと |
| DL数 | 5,336（公開 2022-10） |

**なぜ選んだか**：作曲者の説明が「やわらかくあたたかいギターのフレーズと、優しいシンセパッドの響き」「風景映像や動画のオープニング…幅広く合わせやすい」で、EDL（IG30）の「アコースティックギターかピアノが主体の、ミニマルなインスト」「明るすぎない長調」にいちばん近い。のるさんの曲はメロディーを控えめにした作りが多く、現地音（波・店内の音）の邪魔をしにくい。
- IG30：1.50のカット頭でギターの最初の音 → 4.00（入店）の拍 → 13.40〜16.60（パスタのスロー）はパッドの持続音で薄く → 16.60でギターのリズムが戻る → 24.70のEnd Cardで最後のフレーズ、の流れを1曲の中で作りやすい編成。
- IG15：1.20〜1.50のフェードインで入り、8.50の開放的なコードから12.00のENDまで、パッドで静かに保てる。
**懸念**：「遅い」なので、IG15の「Discovery／Impact」には推進力が足りないかもしれない（聴いて確認）。曲名の「高原」「カフェ」は視聴者には見えないので問題ない。

### 候補2：穏やかな世界 — こばっと
| 項目 | 内容 |
|---|---|
| URL | https://opentracks.com/bgm/detail/10647 |
| 長さ | 2:19 |
| BPM | 記載なし・**未測定**。タグは「遅い」「四拍子」 |
| 楽器 | ピアノ、シンセパッド |
| 声 | なし |
| 作曲者の個別条件 | なし（ライセンスに準拠）。作曲者の公式サイトに「ショート版・ループ版・ジングル版」などのバリエーションがあると説明文にある |
| DL数 | 3,819（公開 2019-02） |

**なぜ選んだか**：「早朝の澄んだ空気感、夕暮れの淡い景色…映画・ドラマ・舞台のBGMに」という説明とタグ（シネマ、ドラマ、ヒーリング）が、Premium／Cinematic／Minimal に合う。ピアノ＋パッドなので、13.40〜16.60の「持続音中心」と、24.70〜29.70の静かな終わりが作りやすい。**ショート版・ジングル版があれば、IG15の最後（12.00〜14.70）をきれいな終止で終われる可能性がある**（作曲者のサイトはこの環境から開けず未確認）。
**懸念**：遅くて「しっとり」寄りなので、冒頭のフック（0〜4秒）とIG15の勢いが弱くなるおそれ。切なさが強すぎると「行ってみたい」より感傷が勝つ。

### 候補3：Chill Ocean — MFP【Marron Fields Production】
| 項目 | 内容 |
|---|---|
| URL | https://opentracks.com/bgm/detail/9307 |
| 長さ | 3:01 |
| BPM | **90**（作曲者の記載。未測定） |
| 楽器 | ピアノ、アコースティックギター、シンセドラム、シンセパッド、ベース、オルガン |
| 声 | なし |
| 作曲者の個別条件 | 「利用条件有」の表示だが、中身は「アダルトコンテンツでの使用も大丈夫です。」だけで、制限の追加はない。説明文に「オフィシャル／商用コンテンツで音源を使用される方は…有償のライセンス込みのオリジナルの音源をご使用されることをおすすめします」とある（推奨で、禁止ではない） |
| DL数 | 1,138（公開 2018-06） |

**なぜ選んだか**：頼まれた条件（80〜100BPM、ピアノ＋アコギ＋軽い打ち込み＋パッド、チル）に、ページの記載がいちばんそのまま合う。拍がはっきりしているはずなので、IG15の編集点（1.5／3.0／4.0／5.5／6.5／8.5／12.0）に合わせやすく、「Discovery／Impact」の推進力を出しやすい。
**懸念**：①シンセドラムの質感しだいで「Vlog／チルBGM」っぽくなり、テンプレ感が出るおそれ。②MFPは曲をストリーミングにも配信している（YouTubeの自動生成「Topic」チャンネル、Instagramのリール音源・TikTokの楽曲ページが検索で見つかった。ただし「Chill Ocean」自体が配信されているかは未確認）。配信されている曲は、Instagramの自動照合で「音源あり」と表示されたり、ミュートされたりする可能性がゼロではない。③作曲者本人が商用には有償の曲をすすめている。

---

## 3. どういう動画で使われているか

調べ方：Web検索（曲名＋作曲者、曲名＋「BGM」「使用」、YouTube／TikTok／Instagramに絞った検索）。この環境ではYouTubeのページ自体と、OpenTracksの「動画内BGM検索」（動画URLから曲を探す機能で、曲から動画は探せない）が使えないため、**見つかったのは検索エンジンの結果の一覧まで**で、動画の中身とクレジットは確認できていない。

| 曲 | 見つかったもの | 使われ方の見立て |
|---|---|---|
| 高原の小さなカフェにて | 公式試聴動画（[#1](https://www.youtube.com/watch?v=mZlvN1FUSb0)、[#2](https://www.youtube.com/watch?v=2lixsMDxXQ8)）。曲名での検索に、駅・都市計画の解説動画が3本出た（[水沢江刺駅](https://www.youtube.com/watch?v=xkYaK84fvnc)、[萩](https://www.youtube.com/watch?v=DE3WM4S_jgE)、[鹿児島中央駅](https://www.youtube.com/watch?v=cYx1y9I4kzg)。説明欄にBGMクレジットがあると思われるが未確認）。ほかに、曲名だけのタイトルの動画（[Mj0Io3dhYdU](https://www.youtube.com/watch?v=Mj0Io3dhYdU)）と同じ名前のチャンネルもあり、配信サービスの自動生成ページの可能性がある | 解説系YouTubeの背景に少し使われている程度。カフェや旅の映像での使用例は見つからなかった |
| 穏やかな世界 | [公式試聴動画](https://www.youtube.com/watch?v=U0vQed_kyFM)だけ | 具体的な使用例は**見つからない** |
| Chill Ocean | [公式試聴動画](https://www.youtube.com/watch?v=V_hPrlQJxHI)だけ | 具体的な使用例は**見つからない**。MFPのほかの曲はInstagramのリール音源・TikTokの楽曲ページに登録がある |

**聞き覚えのリスク（使われすぎ）**：DL数は5,336／3,819／1,138。サイトの定番曲（HIRAHIRA 110,333、South Wind 67,688、Good Morning Sunshine 56,758）と比べて1〜2桁少ないので、「どこかで聞いた曲」と気づかれる可能性は低いと見ている。定番曲は、この理由で候補から外した（§4）。

---

## 4. 次点（候補から外した曲と理由）

| 曲 | 外した理由 |
|---|---|
| [こもれびの道](https://opentracks.com/bgm/detail/19895)（のる、AG、3,419DL） | 1位と同じ作曲者・同じ系統。アコギソロでパッドがないので、13.40〜16.60の持続音が作りにくい。**候補1が合わなかったときの最初の代わり** |
| [朝の風に](https://opentracks.com/bgm/detail/16335)（のる、ピアノソロ、3,729DL） | 1位と同じ作曲者。ピアノ系の代わりとしては有力（ループ版あり） |
| [Good Morning Sunshine](https://opentracks.com/bgm/detail/12626)（modus、AG＋ピアノ、BPM100、56,758DL） | 「ミニマルなフレーズの上に楽器が少しずつ足されていく」構成は理想的だが、DL数が多く使われすぎ。タグが「明るい・楽しい・ポップ」でCMっぽく明るすぎるおそれ |
| [morining on the sea](https://opentracks.com/bgm/detail/15065)（伊藤ケイスケ、AG） | 「海、砂浜…ドライブ」向けの爽やかさで、少しカントリー寄り（タグ） |
| [ナミノネ](https://opentracks.com/bgm/detail/12213)（MATSU、AG＋パッド） | 「ちょっとノスタルジック」で切なさが強め。雰囲気は近いので予備 |
| [海のほとりにて](https://opentracks.com/bgm/detail/11993)（corico、ギター＋エレピ＋パッド） | 曲の頭に「さざ波のSE」が入っていて、現地の波音とぶつかる |
| 蒲鉾さちこの海の曲（[優しい波打ち際](https://opentracks.com/bgm/detail/19255)、[海辺のCafeにて](https://opentracks.com/bgm/detail/16603)ほか） | 雰囲気は合うが、タグが「ﾌﾘｰ（自由なリズム）」の即興演奏で、4.00・13.40・24.70の拍合わせがしにくい |
| [海風に乗って](https://opentracks.com/bgm/detail/11516)（こばっと、ピアノ＋弦＋笛＋パーカッション） | 軽いパーカッションは合うが、「夏の爽やかさ」で笛と弦が明るく、CMっぽくなりそう |
| [Heuristics](https://opentracks.com/bgm/detail/15179)（Anonyment、BPM91.5）、[Connectedness](https://opentracks.com/bgm/detail/17212)（BPM98） | テンポは合うが、エレキ・ベース・ドラムのバンド編成でR&B／ラウンジ寄り。企業VP感が出る |
| [海風のアルペジオ](https://opentracks.com/bgm/detail/17177)（ハモおた） | 「作業用・ループ」前提の作りで、展開が少ない |
| Lo-fi系（Khaim「Sometimes」、のる「Secret Talk」、Addpico「Old Tape Memory」ほか） | ノイズ・ヒップホップのビートで「Vlog／作業用BGM」の典型になり、テンプレ感が出る。Khaimはフル版をSpotify等でも配信している |
| [Serene](https://opentracks.com/bgm/detail/13199)（SHUNTA）、[Island Travel](https://opentracks.com/bgm/detail/15579)（BPM128）、のるの「潮風」系など | トロピカルハウス／南国・リゾート寄りで「トロピカルすぎ・明るすぎ」 |
| [HIRAHIRA](https://opentracks.com/bgm/detail/2307)、[South Wind](https://opentracks.com/bgm/detail/9297) | 定番すぎ（11万・6.7万DL）で聞き覚えのリスクが高い |

---

## 5. プレビューの切り方（スクリプトの設計。数値は mp3 を置いて実行すると `exports/bgm/bgm_cutpoints.json` に記録される）

- **開始位置**：解析で拍とセクションの変わり目を出し、「曲のアタックが 1.50（IG15は 1.50／3.00／4.00／5.50／6.50／8.50／12.00、IG30は 4.00／13.40／16.60／24.70 も）に重なる」「セクションの頭から始まる」位置を自動で選ぶ。ほぼ同点なら曲の前の方を選ぶ。
- **IG30**：0.00〜1.50はBGMなし → 1.50に曲の開始点を置く（フェードインなし）→ 22.30〜23.00（波が開く瞬間）は −3.5dB（前後0.2秒でなめらかに）→ 28.00〜29.70で等パワーのフェードアウト → 29.70〜30.00は波音だけ。曲が自然に終わる形なら、曲の終わりを29.70に合わせる置き方（開始は近い拍に寄せる）を優先する。
- **IG15**：0.00〜1.20はBGMなし → 1.20〜1.50でフェードイン → 11.30〜12.60（波音のJカット）は −3.5dB → 13.00〜14.70でフェードアウト → 最後の0.3秒は波音だけ。
- **音量**：BGMだけの積分ラウドネス（鳴っている区間）を −21 LUFS にそろえ、現地音（`exports/capcut/<edit>_ambience.wav`。`preview_<edit>.mp4` の音と同じもの）と足して、ピークが −1 dBFS を超える所だけ下げる。映像は `preview_<edit>.mp4` をそのまま使う（再エンコードなし）。
- **注意**：IG15のEDLは「120BPM（体感60）」を前提に編集点を0.5秒刻みにしている。候補はどれもそれより遅いので、プレビューでは「拍の格子」ではなく「その位置にアタックがあるか」で合わせる。本番でCapCutの拍マーカーに合わせて編集点を±2フレームより大きく動かす場合は、EDLの秒数も直す（EDL_ig15 の手順10）。

---

## 6. おすすめ

**暫定のおすすめは候補1「高原の小さなカフェにて」（のる）。** アコギ＋パッドの編成、控えめなメロディー、作曲者の個別条件なし、使われすぎていない、の4点で、FIXED RULES（広告っぽく見せない／主役は海と水平線）にいちばん素直に合う。IG15で推進力が足りなければ、候補3（Chill Ocean、BPM90）を比べる。

ただし、**3曲とも音を聴いていない選定**です。§0の手順でプレビューを作って聴き、「現地音が聞こえるか」「広告・テンプレっぽくないか」「最後が静かに終わるか」を確かめてから確定してください。

---

## 付録：詳細を確認した候補（64曲。DL数の多い順）

検索に使った言葉：海＋（アコースティック／ピアノ／カフェ／お洒落／爽やか／穏やか／ギター／アンビエント／チル）、海辺、水平線、潮風、海岸、渚、seaside、ocean、カフェ＋アコースティック、旅＋アコースティック、シネマ＋海、Lo-fi、Vlog＋海、Vlog＋アコースティック、アコースティック＋シネマ＋穏やか＋爽やか ほか（すべて「ボーカルなし」）。ダウンロード順と新着順の両方で見た。検索でヒットした257曲の一覧：`exports/bgm/opentracks_search_results.json`、下の64曲の詳細（説明文・全タグ）：`exports/bgm/opentracks_candidates_detail.json`。

「作曲者の個別条件」は曲ページの「利用条件有」の表示。中身は、蒲鉾さちこ＝使用許可・クレジット不要（任意）、MFP＝アダルト用途も可、ハモおた＝AI学習の禁止、で、どれもこのプロジェクトの使い方を制限しない（ほかの作曲者の条件は未確認）。

| ID | 曲名 / 作曲者 | 長さ | DL数 | 楽器（タグ） | 速さ・BPM | 作曲者の個別条件 |
|---|---|---|---|---|---|---|
| [2307](https://opentracks.com/bgm/detail/2307) | HIRAHIRA / もっぴーさうんど | 3:13 | 110,333 | Pf AG Bass Str 旋律打 Dr | 遅い  | なし（ライセンスどおり） |
| [9297](https://opentracks.com/bgm/detail/9297) | South Wind / FLASH☆BEAT | 2:30 | 67,688 | Pf AG Dr Perc SynDr Str | ﾌﾘｰ 速い 普通の速さ BPM120 | なし（ライセンスどおり） |
| [12626](https://opentracks.com/bgm/detail/12626) | Good Morning Sunshine / modus | 5:33 | 56,758 | Pf AG | 普通の速さ BPM100 | なし（ライセンスどおり） |
| [17805](https://opentracks.com/bgm/detail/17805) | 新しい朝、風に乗って / 蒲鉾さちこ | 1:38 | 39,188 | Pf Key | ﾌﾘｰ 普通の速さ  | あり |
| [12795](https://opentracks.com/bgm/detail/12795) | Sunlight/陽射し / 田中芳典 | 3:35 | 30,183 | AG |   | なし（ライセンスどおり） |
| [16331](https://opentracks.com/bgm/detail/16331) | 新緑芽吹く頃 / 蒲鉾さちこ | 1:43 | 27,424 | Pf Key Str | ﾌﾘｰ 普通の速さ  | あり |
| [16085](https://opentracks.com/bgm/detail/16085) | Daily Accumulation / Anonyment | 3:14 | 19,726 | Pf EG Bass AG Dr Str |  BPM130 | なし（ライセンスどおり） |
| [17212](https://opentracks.com/bgm/detail/17212) | Connectedness / Anonyment | 3:19 | 19,283 | EG Bass Dr Perc |  BPM98 | なし（ライセンスどおり） |
| [13199](https://opentracks.com/bgm/detail/13199) | Serene / SHUNTA | 4:26 | 19,238 | Pf EG SynDr Str | 普通の速さ  | なし（ライセンスどおり） |
| [16707](https://opentracks.com/bgm/detail/16707) | 優しい呼吸で(Breath,tenderly) / 蒲鉾さちこ | 3:02 | 19,015 | Pf Key | ﾌﾘｰ 普通の速さ  | あり |
| [13228](https://opentracks.com/bgm/detail/13228) | Cecilia/セシリア / 田中芳典 | 5:13 | 17,033 | AG Bass Perc Pad Str | 一部遅い 普通の速さ  | なし（ライセンスどおり） |
| [3336](https://opentracks.com/bgm/detail/3336) | ブルーボトル / かずち | 1:10 | 15,241 | Dr SynLead Pad | 遅い  | あり |
| [15579](https://opentracks.com/bgm/detail/15579) | Island Travel / Anonyment | 2:40 | 14,473 | Pf Org Bass AG |  BPM128 | なし（ライセンスどおり） |
| [13997](https://opentracks.com/bgm/detail/13997) | 新しい季節 / 田中芳典 | 2:21 | 13,611 | AG Bass Perc | 遅い  | なし（ライセンスどおり） |
| [13132](https://opentracks.com/bgm/detail/13132) | 海を見に行こうよ！ / こばっと | 2:10 | 12,538 | Pf Org EG Bass AG Dr Perc | 普通の速さ  | なし（ライセンスどおり） |
| [19829](https://opentracks.com/bgm/detail/19829) | 海辺のステップ / 蒲鉾さちこ | 2:42 | 11,457 | Pf Key | ﾌﾘｰ  | あり |
| [20951](https://opentracks.com/bgm/detail/20951) | あの夏の日 / 蒲鉾さちこ | 3:44 | 11,391 | Pf Key | ﾌﾘｰ 普通の速さ  | あり |
| [15034](https://opentracks.com/bgm/detail/15034) | 光輝く朝に(In radiant morning) / 蒲鉾さちこ | 2:07 | 10,777 | Pf Key | ﾌﾘｰ 普通の速さ  | あり |
| [20733](https://opentracks.com/bgm/detail/20733) | 優しい海辺 / 蒲鉾さちこ | 3:00 | 9,948 | AG Str | ﾌﾘｰ 遅い 一部遅い  | あり |
| [15758](https://opentracks.com/bgm/detail/15758) | 始まりの朝、優しい海辺 / 蒲鉾さちこ | 1:24 | 9,766 | Pf Key | ﾌﾘｰ  | あり |
| [21833](https://opentracks.com/bgm/detail/21833) | Secret Talk / のる | 2:59 | 8,982 | Pf Bass AG Dr | 遅い  | なし（ライセンスどおり） |
| [16941](https://opentracks.com/bgm/detail/16941) | 海沿いロードウェイ / のる | 2:45 | 8,355 | Pf Perc |   | なし（ライセンスどおり） |
| [15179](https://opentracks.com/bgm/detail/15179) | Heuristics / Anonyment | 3:47 | 8,266 | EG Bass AG Dr Perc Str |  BPM91 | なし（ライセンスどおり） |
| [20172](https://opentracks.com/bgm/detail/20172) | 潮風と渚(Coast,sea breeze) / 蒲鉾さちこ | 1:29 | 8,201 | Pf Key AG Dr |  BPM150 | あり |
| [16603](https://opentracks.com/bgm/detail/16603) | 海辺のCafeにて(At seaside Cafe) / 蒲鉾さちこ | 3:16 | 7,700 | Pf Key | ﾌﾘｰ 普通の速さ  | あり |
| [19255](https://opentracks.com/bgm/detail/19255) | 優しい波打ち際(Gentle surf) / 蒲鉾さちこ | 3:48 | 6,920 | Pf Key | ﾌﾘｰ 普通の速さ  | あり |
| [21848](https://opentracks.com/bgm/detail/21848) | Luminous time / 蒲鉾さちこ | 1:56 | 6,748 | Pf Key | ﾌﾘｰ 普通の速さ  | あり |
| [15065](https://opentracks.com/bgm/detail/15065) | morining on the sea / 伊藤ケイスケ | 2:31 | 5,865 | AG | 普通の速さ  | なし（ライセンスどおり） |
| [14726](https://opentracks.com/bgm/detail/14726) | 爽やかな空の下で / 蒲鉾さちこ | 2:48 | 5,672 | Pf Key | ﾌﾘｰ 普通の速さ  | あり |
| [17543](https://opentracks.com/bgm/detail/17543) | 高原の小さなカフェにて / のる | 3:34 | 5,336 | Pf AG Pad | 遅い  | なし（ライセンスどおり） |
| [12697](https://opentracks.com/bgm/detail/12697) | 海辺の夕暮れ / Motoyuki | 1:12 | 5,248 | AG | 一部遅い 普通の速さ  | なし（ライセンスどおり） |
| [13172](https://opentracks.com/bgm/detail/13172) | Old Tape Memory / Addpico | 2:53 | 4,883 | Pf Key Bass Str SynDr Pad | 遅い BPM90 | なし（ライセンスどおり） |
| [17177](https://opentracks.com/bgm/detail/17177) | 海風のアルペジオ / ハモおた | 5:21 | 4,585 | Pf EG Bass AG Dr Pad | 遅い 普通の速さ  | あり |
| [22167](https://opentracks.com/bgm/detail/22167) | もしもの話。 / のる | 2:52 | 4,527 | Key AG Perc | 普通の速さ  | なし（ライセンスどおり） |
| [12213](https://opentracks.com/bgm/detail/12213) | ナミノネ / MATSU | 2:28 | 4,477 | AG Pad | 遅い  | なし（ライセンスどおり） |
| [18594](https://opentracks.com/bgm/detail/18594) | Artificial Meaning / Anonyment | 2:37 | 3,912 | SynDr SynLead | 遅い BPM70 | なし（ライセンスどおり） |
| [22400](https://opentracks.com/bgm/detail/22400) | 妖精の庭 / のる | 4:59 | 3,886 | Pf | 普通の速さ  | なし（ライセンスどおり） |
| [10647](https://opentracks.com/bgm/detail/10647) | 穏やかな世界 / こばっと | 2:19 | 3,819 | Pf Pad | 遅い  | なし（ライセンスどおり） |
| [16335](https://opentracks.com/bgm/detail/16335) | 朝の風に / のる | 2:15 | 3,729 | Pf | 普通の速さ  | なし（ライセンスどおり） |
| [18365](https://opentracks.com/bgm/detail/18365) | special flight / のる | 2:08 | 3,490 | Pf | 普通の速さ  | なし（ライセンスどおり） |
| [19895](https://opentracks.com/bgm/detail/19895) | こもれびの道 / のる | 2:25 | 3,419 | AG | 普通の速さ  | なし（ライセンスどおり） |
| [14192](https://opentracks.com/bgm/detail/14192) | Lo-Fi Sunset / だんご工房 | 3:14 | 3,402 | Pf Key Str | 普通の速さ  | なし（ライセンスどおり） |
| [23532](https://opentracks.com/bgm/detail/23532) | Sometimes / Khaim | 2:15 | 3,182 | Pf Bass Fl等 Dr Perc SynDr | 普通の速さ BPM85 | あり |
| [11993](https://opentracks.com/bgm/detail/11993) | 海のほとりにて / corico | 2:39 | 3,058 | Key AG Pad | 普通の速さ  | なし（ライセンスどおり） |
| [20925](https://opentracks.com/bgm/detail/20925) | 水面のきらめき / こばっと | 2:31 | 3,027 | Pf AG Str | 普通の速さ  | なし（ライセンスどおり） |
| [9875](https://opentracks.com/bgm/detail/9875) | 海と水平線 / MAKOOTO | 2:24 | 2,967 | Key 旋律打 Pad | 遅い  | なし（ライセンスどおり） |
| [21674](https://opentracks.com/bgm/detail/21674) | Shining waves / 蒲鉾さちこ | 2:22 | 2,846 | Pf Key AG | ﾌﾘｰ 速い  | あり |
| [21667](https://opentracks.com/bgm/detail/21667) | あの日の海へ / のる | 3:36 | 2,455 | Pf | 遅い  | なし（ライセンスどおり） |
| [23366](https://opentracks.com/bgm/detail/23366) | Treatise Seven / Anonyment | 3:11 | 2,443 | EG Bass SynDr SynLead Pad |  BPM124 | なし（ライセンスどおり） |
| [23692](https://opentracks.com/bgm/detail/23692) | 夏の海風 / のる | 3:07 | 2,393 | Pf AG Fl等 | 普通の速さ  | なし（ライセンスどおり） |
| [23350](https://opentracks.com/bgm/detail/23350) | draw in the night / のる | 3:04 | 2,387 | Pf EG Dr | 遅い  | なし（ライセンスどおり） |
| [22478](https://opentracks.com/bgm/detail/22478) | Tiny House / のる | 2:43 | 1,982 | AG | 遅い  | なし（ライセンスどおり） |
| [23191](https://opentracks.com/bgm/detail/23191) | In Pursuit / Anonyment | 3:15 | 1,846 | Pf EG Bass AG Dr Perc |  BPM96 | なし（ライセンスどおり） |
| [23024](https://opentracks.com/bgm/detail/23024) | Coastal Road / Heitaro Ashibe | 2:16 | 1,814 | EG Bass AG Perc | 普通の速さ  | なし（ライセンスどおり） |
| [11516](https://opentracks.com/bgm/detail/11516) | 海風に乗って / こばっと | 1:52 | 1,758 | Pf Str Fl等 Perc | 普通の速さ  | なし（ライセンスどおり） |
| [14334](https://opentracks.com/bgm/detail/14334) | 貝殻の中の思い出 / 秦暁 | 3:00 | 1,534 | Pf AG Pad | 遅い  | あり |
| [23614](https://opentracks.com/bgm/detail/23614) | Blue Ocean / FLASH☆BEAT | 2:21 | 1,505 | Pf Key SynDr Str | 遅い 普通の速さ BPM90 | なし（ライセンスどおり） |
| [14955](https://opentracks.com/bgm/detail/14955) | 海岸線を散歩 / alaki paca | 2:29 | 1,255 | Pf | 普通の速さ  | あり |
| [9307](https://opentracks.com/bgm/detail/9307) | Chill  Ocean / MFP【Marron Fields Production】 | 3:01 | 1,138 | Pf Org Bass AG SynDr Pad | 遅い BPM90 | あり |
| [18601](https://opentracks.com/bgm/detail/18601) | In The Ocean / えすにっく・かわひろ | 2:19 | 1,055 | Pf Key Str SynLead Pad | ﾌﾘｰ  | なし（ライセンスどおり） |
| [22523](https://opentracks.com/bgm/detail/22523) | Twilight Session / MFP【Marron Fields Production】 | 1:51 | 912 | Key AG 旋律打 Perc | 普通の速さ BPM110 | あり |
| [19239](https://opentracks.com/bgm/detail/19239) | Sea / Heitaro Ashibe | 2:30 | 547 | 旋律打 Pad | 遅い  | なし（ライセンスどおり） |
| [23538](https://opentracks.com/bgm/detail/23538) | BGM - 156 - Coffee Break / Sound Of Incense | 3:22 | 457 | Key EG Fl等 | 遅い  | なし（ライセンスどおり） |
| [23782](https://opentracks.com/bgm/detail/23782) | Fading Heat Days / MFP【Marron Fields Production】 | 2:37 | 331 | Pf Org Bass AG Fl等 Dr Str | 普通の速さ BPM125 | あり |
