# CapCut PC の草稿を、クラウドのエージェントから作る・直す方法の調査

調査日：2026-11-07。テスト用の草稿を1つ作り、構造は点検しました。ただし CapCut で開けるかは確かめられていません。この環境には CapCut がないためです。

## 0. 結論

1. **本命は、こちらで草稿フォルダを丸ごと作って渡す方法です。** ツールは **pyCapCut** を使います。素材を草稿フォルダの中に入れて zip で渡し、ユーザーのPCで素材パスを1回だけ書き換えます。この形のテスト草稿は作りました（§5）。
2. **CapCut 公式の MCP サーバーがあり、このコンテナから届きました。** アドレスは `https://www.capcut.com/api/external_mcp` で、ツールの一覧も取れました。ただしログインに OAuth（ブラウザでの承認）が必要で、承認の戻り先はユーザーのPCの localhost です。そのため、ここから使うには**ユーザーのPC側の Claude（デスクトップ版か、ローカルの Claude Code）から接続する**必要があります。しかも編集できるのは**クラウド上の草稿（Web 版エディタ）**で、PC のローカル草稿ではありません。
3. **古い CapCut が書いた目印のまま渡すと、新しい CapCut に開いてもらえない例があります。** pyCapCut の雛形には「CapCut 6.7.0（Windows）が書いた」という目印が入っています。CapCut 8.7 以降では、古い目印のままの草稿が「unusual path」として拒否されたと報告されています。いちばん確実なのは、**ユーザーの CapCut で空のプロジェクトを1つ作ってもらい、そのフォルダを雛形にすること**です。
4. **CapCut 国際版は、草稿の暗号化の対象外と報告されています。** 暗号化されるのは中国版の剪映 6.0 以降です。ただし、CapCut 10.x ではツールで書いた草稿が「破損」と出るという報告があり、まだ確認されていません。**ユーザーの CapCut のバージョン番号がいちばん大事な情報です。**

## 1. 候補一覧

★と最終更新は 2026-11-07 に GitHub 検索 API で取った値です。複製はすべて `/home/user/vendor/` の下にあります。

| ツール | URL | ★ | ライセンス | 最終更新 | 確かめた範囲 | 判定 |
|---|---|---|---|---|---|---|
| pyCapCut（CapCut 国際版の草稿を作る Python ライブラリ） | https://github.com/GuanYixuan/pyCapCut | 737 | **LICENSE ファイルなし**（setup.py にも書いていない） | 2026-10-16（ec61408） | **実際に動かした**。PyPI の 0.0.4 を venv に入れ、改行コード以外はリポジトリ先頭と同じ中身だと確認した | **本命（生成）** |
| capcut-cli（Node 製のコマンド。依存なし、Claude 用スキル付き） | https://github.com/renezander030/capcut-cli | 1,125 | MIT | 2026-11-06（v0.33.1） | **読んだだけ**（npm での導入は権限で止められた。§7） | **本命（ユーザーPCでの仕上げ・点検）** |
| CapCut 公式 MCP（Codex 用プラグイン 0.2.7） | `https://www.capcut.com/api/external_mcp`（同梱物 `https://sf16-sg.tiktokcdn.com/obj/eden-sg/962370eh7nupenuhbo/capcut_codex_plugin/plugins_v27.zip`） | — | **UNLICENSED**（CapCut の専有物） | 2026-11 時点で 0.2.7 | **一部試した**。initialize と tools/list は通り、ツール29個を確認した。アカウント系の呼び出しは 401 で、OAuth は authorization_code＋PKCE だけ（デバイスコード方式なし） | 有望（ただしユーザーPC側から使う） |
| capcut-claude-skill（公式 MCP を Claude 向けに直したスキル） | https://github.com/Altergom/capcut-claude-skill | 2 | MIT（CapCut の同梱物は別扱い） | 2026-10-27 | 読んだだけ | 公式 MCP を使うときの手順書 |
| VectCutAPI（旧名 CapCutAPI。HTTP の API と MCP サーバー） | https://github.com/sun-guannan/VectCutAPI | 2,365 | Apache-2.0 | 2026-10-24 | 読んだだけ | 次点。素材を URL から取ってくる作りで、CapCut 用の雛形は古いものしかない |
| pyJianYingDraft（剪映向け） | https://github.com/GuanYixuan/pyJianYingDraft | 4,830 | Apache-2.0 | 2026-10-14 | 読んだだけ | 剪映向け。CapCut 向けは上の pyCapCut に分かれている |
| aoguai/pyJianYingDraft（上の派生。暗号化された草稿も扱える） | https://github.com/aoguai/pyJianYingDraft | 27 | Apache-2.0 | 2026-10-14 | 読んだだけ | 剪映専用なので今回は不要 |
| capcut-mcp（実在する草稿を雛形にして部品を複製する MCP） | https://github.com/JmsLdrn/capcut-mcp | 17 | MIT | 2026-08-07 | 読んだだけ | 参考（ユーザーのPCで動かすもの） |
| capcut-mate（剪映の補助ツール群） | https://github.com/Hommy-master/capcut-mate | 2,112 | Apache-2.0 | 2026-11-01 | 草稿内パスの書き方だけ読んだ | 参考 |
| その他 | AISEO-VirtualabDigital/mcp-video-editor（★78、ライセンスなし）、hey-jian-wei/jianying-mcp（★323、ライセンスなし、2025-08 で止まっている）、fancyboi999/capcut-mcp（アーカイブ済み） | | | | 一覧を見ただけ | 不要 |

awesome-claude-video-skills の一覧に載っていた CapCut 関係（rea-capcut、video-recap-skills、SeeCut ほか）は、どれも剪映向けか、自分で描画する別のエディタでした。

GitHub への接続：`git clone` と検索 API は通りました。リポジトリごとの API（`/repos/...`）は「このセッションでは許可されていない」と返されました。gh CLI のトークンは無効です。

## 2. 問1：今の CapCut PC は外から作った草稿を開くか

- **剪映 6.0 以降**は草稿を暗号化します。それでも、暗号化していない（平文の）草稿は開いて、自分で暗号化し直すと報告されています（剪映 11.4 の Mac、9.0.0 の Windows）。
- **CapCut 国際版**は暗号化の対象外と書かれています（capcut-cli の `docs/jianying-encryption.md`）。報告されているバージョンごとの状況は次のとおりです（capcut-cli の `docs/version-support.md`）。

| バージョン | 報告されている状況 |
|---|---|
| 6.2.8 | テスト用の実データがある（確認済み） |
| 8.4（Mac） | 実在の草稿をコピーして作ったものは開けた |
| 8.7（Windows） | 古い雛形から作ったものは「unusual path」で**拒否**。その PC の CapCut で作った草稿を雛形にすれば開けた |
| 9.2.8 / 9.3.0（Mac） | `Timelines/<id>/draft_info.json` のほうが優先される |
| 9.1.0（Mac） | `draft_meta_info.json` の `draft_materials` が空だと「素材にアクセスできない」と出る（pyCapCut#13） |
| **10.x** | ツールで書いた草稿が「内容が破損」とされた報告あり。**未確認** |

- 公式の最新版番号は、Web 検索では確定できませんでした。Windows 版の履歴には 8.7.0.3685 までが載っていました。

## 3. 問2：どの要素を書けるか

**pyCapCut（今回実際に書けたもの）**
- 映像トラック（素材の切り出し、速度、音量）
- トランジション（叠化＝ディゾルブ）
- 拡大のキーフレーム
- 音声トラックを3本に分ける（ナレーション、BGM、効果音）。フェードと音量のキーフレームも
- テキストと、透明度・縦位置のキーフレーム、縁取り
- SRT 字幕の取り込み（専用のテキストトラック）

README には次のものも「対応」とあります：マスク、クロマキー、合成モード、フィルター、エフェクト、テキストのアニメ、花字、フォント。

**公式 MCP のクラウド草稿**（Altergom の実地テストによる）
- 残るもの：映像、トランジション、音楽・効果音、フィルター、テキスト、語ごとのタイミング付き字幕、速度1の区間のキーフレーム
- **消えるもの**：映像エフェクト。速度を変えた区間のキーフレームもほぼ消える
- 書き出しの機能はない

## 4. 問3：草稿の置き場所と、素材パスの扱い

**置き場所**
- Windows：`%LOCALAPPDATA%\CapCut\User Data\Projects\com.lveditor.draft`
- Mac：`~/Movies/CapCut/User Data/Projects/com.lveditor.draft`
- CapCut の「設定 → 草稿の場所」で変えている場合は、そちらが優先されます。

**一覧に表示されるかどうか**
- capcut-cli によると、CapCut は `root_meta_info.json` を見て一覧を作ります。ここに登録がないと表示されません。
- pyCapCut は「CapCut を再起動すれば出る」としていて、二つの説明が食い違っています。

**素材パス**
- pyCapCut は素材パスを**絶対パス**で書き込みます。素材ファイルが実在しないと作れません。
- 対策として、草稿フォルダの中に `media/` を作り、パスを `__MAZE_DRAFT_DIR__/media/...` という目印に置き換えました。ユーザーのPCでスクリプトを1回実行すると、実際の場所に書き換わります。
- 剪映が使う草稿内の相対パス `##_draftpath_placeholder_0E685133-18CE-45ED-8CB8-2904A212EC80_##/…`（capcut-mate の実装にある書き方）も使える可能性があります。CapCut 国際版で通るかは未確認です。

## 5. 実際に作ったもの

**中身**
- 1080x1920、30fps、5秒
- 映像：テストパターンの mp4 を2つに分け、0〜3秒は等速で 1.0→1.15 倍に寄るキーフレーム付き、3〜5秒は1.5倍速。間にディゾルブ（0.5秒）
- 音：ナレーション（`exports/v3/vo/01.wav` を 0.08 秒から）、BGM（テスト用の音、音量0.3、フェード付き、音量のキーフレーム付き）、効果音（テスト用の音、2.75 秒から）
- タイトル文字（透明度0→1、縦位置のキーフレーム）
- SRT 字幕を2行。字幕の文は、ナレーションの読み（カナ）から私が推定して起こした**仮の文**です。

**点検（`validate_draft.py` を自作して実行）**
- 結果は **OK**（構造上の誤りなし）。
- 調べたこと：素材の参照切れ、同じトラック内での重なり、素材の切り出しが長さに収まっているか（速度込み）、トラックと素材の種類の対応、パスの置き換え、このコンテナのパスが残っていないか。
- **pyCapCut 0.0.4 の不具合を1つ見つけて直しました。** テキストは速度の素材を参照しているのに、その素材が素材一覧に入っていません。生成スクリプトの中で補っています。
- `draft_meta_info.json` の `draft_materials` を埋めました（pyCapCut#13 で報告された回避策）。

**パス書き換えスクリプトと確認用の動画**
- Mac 用（`relink_mac.command`）は、「CapCut Drafts」という空白入りの名前のフォルダにコピーして実行しました。8か所のパスが全部実在するファイルを指し、目印が残っていないことを確認しました。
- **Windows 用（.ps1 / .bat）は動かせていません。** この環境に PowerShell がないためです。
- 草稿の JSON を ffmpeg で再生しなおす簡易版も作りました（`proxy_render.py`）。キーフレームとトランジションは無視しますが、カット・音・字幕の時刻は目で確認できました（`proxy_contact.png`）。

**CapCut がないので確かめられないこと**
- CapCut で開けるか、「unusual path」や「破損」と出ないか
- 一覧に表示されるか
- 素材の再リンクを求められないか（Windows はスラッシュ区切りのパス）
- キーフレーム・トランジション・SRT の見た目
- 日本語の字幕がどのフォントで表示されるか

## 6. 問4：ライセンス

- **pyCapCut は LICENSE がありません。** 作者に権利がすべて残っている扱いになるので、コードを配ったりリポジトリに入れたりは避けてください。このコンテナで道具として動かすだけにとどめます。
- 同じ機能が欲しい場合、Apache-2.0 の選択肢があります。pyJianYingDraft、または VectCutAPI に入っている CapCut 対応の派生版です。
- capcut-cli は MIT です。
- 公式 MCP の同梱物は UNLICENSED（CapCut の専有物）で、CapCut の規約に従います。
- capcut-cli の `caption --cloud` は、音声を ByteDance に送ります。

## 7. 権限で止められたこと（必要なら許可をお願いします）

- **pyCapCut を、複製したリポジトリから直接インストールすること**が止められました。代わりに PyPI の版を固定（`pycapcut==0.0.4`）して入れました。
- **`npm install capcut-cli@0.33.1` が止められました。** そのため capcut-cli の `lint`・`migrate`・`relink`・`render` は一度も動かしていません。ここで capcut-cli の点検まで回したい場合は、npm での導入を許可してください。

## 8. おすすめの進め方

**A. 今すぐできること（ユーザーの作業）**
1. zip を解凍し、`MAZE_v3_capcut_test` フォルダを草稿の場所に置く。
2. `relink_windows.bat`（Mac は `sh relink_mac.command`）を1回だけ実行する。
3. CapCut を起動して開く（一覧に出なければ CapCut を再起動）。
4. 結果と **CapCut のバージョン番号** を教えてもらう。手順は zip の中の `はじめに_README.txt` に書きました。

**B. いちばん確実な方法（A で開けなかったとき、または本番用）**
1. ユーザーの CapCut で、空の 1080x1920 のプロジェクトを1つ作り、CapCut を終了してもらう。
2. そのフォルダを zip にして共有してもらう（素材は入っていないので小さい）。
3. こちらでそれを雛形にし、バージョンの目印・ID・`Timelines/` を引き継いだまま、本番の内容を書き込む。
4. ユーザーのPCに Node があれば、`npx capcut-cli@0.33.1 migrate <草稿> --from-store`、`relink <草稿> --dir <草稿>/media`、`register <草稿> --materials --apply`、`lint <草稿>` で仕上げと点検もできる（未検証）。

**C. 公式 MCP を使う場合**
- ユーザーのPCの Claude デスクトップ版の設定（`claude_desktop_config.json`）に `npx mcp-remote https://www.capcut.com/api/external_mcp` を登録し、ブラウザで承認してもらう。
- その後、ユーザーのPC側の Claude からクラウド草稿を作り、CapCut Web で書き出す。
- 効果音・音楽の検索（`search_capcut_material`）も使えます。ただし CapCut の音源の規約は、`CAPCUT_RESEARCH.md` に書いた懸念のとおり確認が必要です。

**本番での分担（案）**
- 色・合成・見せ場の「人間には不可能な精度」の加工は、これまでどおり `tools/v3` で行い、区間ごとの mp4 に書き出す。
- CapCut の草稿は、それらを並べる映像トラック、ナレーション・BGM・効果音の別トラック、テキスト、字幕だけにする。
- エフェクトは CapCut 側では使わない。

## 9. ファイル（すべて `/home/user/mhf4u-cct2026/exports/v3/research_trials/capcut_draft/`）

- `MAZE_v3_capcut_test.zip`：ユーザーに渡すもの（6.4MB）
- `drafts/MAZE_v3_capcut_test/`：`draft_content.json`、`draft_meta_info.json`、`media/`、`relink_windows.ps1`、`relink_windows.bat`、`relink_mac.command`、`はじめに_README.txt`
- `make_test_draft.py`：生成スクリプト（`/home/user/vendor/venv-capcut/bin/python -I make_test_draft.py` で実行）
- `validate_draft.py`：構造の点検
- `proxy_render.py`、`proxy_preview.mp4`、`proxy_contact.png`：確認用の代用プレビュー
- 複製したリポジトリ：`/home/user/vendor/GuanYixuan_pyCapCut`、`GuanYixuan_pyJianYingDraft`、`aoguai_pyJianYingDraft`、`sun-guannan_VectCutAPI`、`renezander030_capcut-cli`、`Altergom_capcut-claude-skill`、`JmsLdrn_capcut-mcp`、`Hommy-master_capcut-mate`
- 公式プラグインの同梱物（読んだだけ、実行していない）：`/home/user/vendor/_capcut_official_dl/`

Web 検索の情報源：
- [jmsldrn/capcut-mcp](https://github.com/jmsldrn/capcut-mcp)
- [pycapcut (PyPI)](https://pypi.org/project/pycapcut/)
- [JyDraft README](https://gh.r-mj.com/HTWMedia/JyDraft/blob/main/README_EN.md)
- [capcut-cli-david draft schema](https://cdn.jsdelivr.net/npm/capcut-cli-david@1.14.1/docs/draft-schema/00-overview.md)
- [CapCut PC 版の更新（公式ヘルプ）](https://www.capcut.com/de-de/help/update-capcut-pc-for-latest-version)
- [geekchamp の Windows 版履歴](https://geekchamp.com/?p=1089856)
- [codegenes の CapCut Windows 版ガイド](https://www.codegenes.net/blog/capcut-download-windows/)