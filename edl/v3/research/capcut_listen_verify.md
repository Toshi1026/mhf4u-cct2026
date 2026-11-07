# 検証レビュー：CapCut 草稿の生成と「音を聞く」仕組み（2026-11-07）

## 結論
- **listen.py は最後まで動きます。** `exports/v3/vo/01.wav` で1回実行し、正常に終わりました（24秒）。
- **CapCut の草稿をこちらで作る方法は、ユーザーの CapCut で開けるかがまだ分かりません。** 草稿の中の目印が古い版（6.7.0）で、CapCut 10.x では開けない見込みが強いです。先にユーザーの CapCut のバージョン番号を聞き、空のプロジェクトを1つもらってから進めるのが前提になります。
- **商用の Instagram 広告で、表記なしで使えると条文で確かめられた効果音の取得元は、効果音ラボと Pixabay です。** 魔王魂も使えますが、クレジット表記が必要です。CapCut の音源は使わない前提にします。

## 確認できたこと（自分で動かした・取り直したもの）

**listen.py の実行**
- 実行したコマンド：`nice -n 10 python3 tools/v3/listen.py exports/v3/vo/01.wav --name review_vo01 --threads 2`
- 結果：正常終了（exit 0）、実時間24秒。
- 出力：`/home/user/mhf4u-cct2026/exports/v3/listen/review_vo01/`（`report.md`、`overview.png`、`01.json`）。実行ログは `/home/user/mhf4u-cct2026/exports/v3/research_trials/review/listen_vo01.log` です。
- 5つの解析（ラウドネス、YAMNet、PANNs、CLAP、Whisper large-v3-turbo）が全部動きました。
- 01 の数値：-24.1 LUFS、トゥルーピーク -5.1 dBTP。
- Whisper の聞き取り：「人類初挑戦、世界記録の波で高知まで行けるか。」想定の読みと完全に一致しました。
- 警告は SudachiPy の DeprecationWarning が1つ出ただけです。
- listen.py は git 上で変更ありのまま、まだコミットされていません（報告どおり）。

**今のミックスの数値**
- ffmpeg の ebur128 で測り直しました。統合 -16.5 LUFS、LRA 2.8 LU、ピーク -1.5 dBFS で、報告と一致します。

**CapCut の草稿について**
- テスト草稿 `capcut_draft/drafts/MAZE_v3_capcut_test/draft_content.json` の目印は次のとおりで、報告と一致します。
  - `platform` は `cc / 6.7.0 / windows`
  - `version` は 360000
- pyCapCut は `draft_content.json` だけを書き出します（`pycapcut/draft_folder.py:91`）。
- CapCut 国際版は暗号化の対象外で、暗号化されるのは中国版の剪映 6.0 以降です。capcut-cli の `docs/jianying-encryption.md` にそう書いてあることを確認しました。
- CapCut 8.7（Windows）で「unusual path」として拒否された件は、capcut-cli の `docs/version-support.md`（issue #111）にあります。同じ文書に、その PC の CapCut で作った草稿を雛形にすれば 9.3.0 で開いて保存し直せたとあります。
- 10.x で「破損」とされた報告があることも、同じ文書で確認しました（根拠は pyJianYingDraft#177、#194、Jianying-CapCut2XML#4）。
- CapCut 公式の MCP（`https://www.capcut.com/api/external_mcp`）は、このコンテナから initialize が通りました。

**ツールのライセンス**

★と最終更新は GitHub 検索 API で取り直しました。

| ツール | ★ | ライセンス | 最終更新 |
|---|---|---|---|
| pyCapCut https://github.com/GuanYixuan/pyCapCut | 737 | LICENSE ファイルなし。PyPI 0.0.4 の表示も `License: UNKNOWN` | 2026-10-16 |
| capcut-cli https://github.com/renezander030/capcut-cli | 1,125 | MIT（v0.33.1） | 2026-11-06 |
| VectCutAPI https://github.com/sun-guannan/VectCutAPI | 2,365 | Apache-2.0 | 2026-10-24 |
| pyJianYingDraft https://github.com/GuanYixuan/pyJianYingDraft | 4,830 | Apache-2.0 | 2026-10-14 |

- agent-video-editor の LICENSE も MIT でした。
- 音のモデルのライセンス（Hugging Face API で取得）：
  - YAMNet ONNX（zeropointnine）：Apache-2.0
  - PANNs ONNX（pranjal-pravesh）：MIT
  - laion/clap-htsat-unfused：Apache-2.0
- GPL・AGPL のもの：
  - pykakasi（GPL-3.0）は、listen.py の中で SudachiPy が無いときだけ読み込まれる予備です（338行目）。社内で動かすだけなら問題ありません。
  - Essentia（AGPL）は使っていません。

**効果音の規約（条文を取得して確認）**
- 効果音ラボ：商用無料で、CM での利用も可、表記は任意です。禁止は次のとおりです。
  - 効果音が主役のコンテンツ（再配布と同じ扱い）
  - Content ID への登録
  - AI の学習
  - 直リンク
- 魔王魂：商用無料ですが、**著作表記が必要**です（例：「効果音：魔王魂」）。動画内か概要欄に書けば足ります。ほかに、単品の再配布、生成 AI の学習、著作権管理団体への登録が禁止です。
- Pixabay：無料で、表記は不要です。そのままの単品の再配布と、商標・社名としての利用が禁止です。

## 直すべきこと

1. **「pyCapCut の 6.7.0 の目印は 8.7 で拒否される」は、直接の証拠がありません。** 8.7 で拒否されたのは capcut-cli に同梱の **6.5.0** の雛形です。同じく古い目印なので危ないという推定としては妥当ですが、6.7.0 で試した人はいません。
2. **CapCut 10.x では、さらに別の問題が見込まれます。** 報告では、10.x の Mac は `draft_info.json` を主ファイルとして読みます。剪映 11.4 では、`draft_content.json` を置くと「破損」になるとされています。pyCapCut は `draft_content.json` しか書かないので、10.x ではほぼ開けない見込みです。また capcut-cli は、9.x より新しい草稿には `--force-write` を付けない限り書き込みを拒否します。
3. **PANNs の重みの元のライセンスは「MIT」ではなく CC BY 4.0 です。** Zenodo の記録 3987831 と 3960586 で確認しました。ONNX に変換した人が MIT と表示しているだけです。社内の解析に使うだけなら問題はありませんが、表記は正しておくべきです。
4. **テスト草稿のディゾルブは、本番では使えません。** `CAPCUT_RESEARCH.md` の機能表に「トランジション：使わない（ルール：ハードカットのみ）」とあります。本番の草稿はハードカットだけにします。
5. **公式 MCP の効果音検索（`search_capcut_material`）は、Instagram 広告には使わない前提にします。** `CAPCUT_RESEARCH.md`（S24）に「商用音源は TikTok 内のみ」とあるためです。
6. **pyCapCut は、次の 0.1.0 で使い方が大きく変わると README で予告しています**（`add_track` を廃止）。`pycapcut==0.0.4` に固定したのは正しい判断です。

## 確かめられなかったこと
- **今の CapCut PC の最新版番号**：Web 検索では公式の情報が見つからず、ネットの情報も「8.5 以上」から「14.x」まで食い違っています。
- **テスト草稿が実際の CapCut で開けるか、一覧に出るか、再リンクを求められないか**：この環境に CapCut がないので確かめられません。
- **Windows 用のパス書き換えスクリプト（.ps1 / .bat）**：PowerShell がないので動かせていません。
- **9.1.0 の Mac で `draft_materials` が空だとエラーになる件（pyCapCut#13）**：issue 本文を読めていません。
- **OtoLogic の規約**：WebFetch でも 403 で読めませんでした。「CC BY 4.0、表記が必要」という説明は記憶によるもので、確認できていません。
- **Mixkit の効果音ライセンスの条文**：ページに条文が出てきませんでした。二次情報では「商用可、表記不要」とされています。
- **Freesound・Sonniss**：私は取り直していません。CC BY-NC は使えない、という判断はそのままで正しいです。
- **公式 MCP の OAuth の情報**：取りに行ったら 429（回数制限）で返されました。tools/list の 29 個も、今回は数え直していません。
- **capcut-cli の各コマンド**：npm でのインストールが止められたので、一度も動かしていません。

## おすすめの進め方

1. **最初にユーザーに頼むこと（必須）：**
   - CapCut PC のバージョン番号と、Windows か Mac か。
   - 空の 1080x1920 のプロジェクトを1つ作り、CapCut を終了してから、そのフォルダを zip で送ってもらう。
2. **草稿を作る方法：** ユーザーの CapCut のバージョンで決めます。

   | ユーザーの CapCut | 進め方 |
   |---|---|
   | 9.x 以下 | もらった空のプロジェクトを雛形にして、こちらの Python で中身を書き込む。版の目印・ID・`Timelines/` は引き継ぐ。9.x では `Timelines/<id>/draft_info.json` にも書く |
   | 10.x | 草稿を書き込む方法はやめて、下の「書き込みがだめなとき」の形にする |

   - pyCapCut はライセンスがないので、コンテナの中で道具として使うだけにして、リポジトリには入れません。代わりに Apache-2.0 の pyJianYingDraft のコードを使う手もあります。
   - ユーザーの PC に Node があれば、`npx capcut-cli@0.33.1` の `lint`・`relink`・`register` で仕上げと点検をします。これは未検証です。
   - **書き込みがだめなとき（10.x、または開けなかったとき）：** `tools/v3` で区間ごとの mp4 と、ナレーション・BGM・効果音を別々の wav に書き出し、時刻の表と SRT を付けて渡します。ユーザーが CapCut に手で並べます。これがいちばん確実です。
3. **公式 MCP は予備にとどめます。** ユーザーの PC の Claude から OAuth で接続する必要があり、編集できるのはクラウドの草稿だけです。エフェクトは消え、書き出しの機能もありません。
4. **音の確認：** `tools/v3/listen.py` をそのまま使います。CapCut から書き出した mp4 も同じように調べられます。
   - 他人のスキルは参考にとどめます。bgm-understanding は重みが非商用（CC BY-NC-SA）なので使いません。
   - 音量は 2パスの loudnorm で -14 LUFS、トゥルーピーク -1 dBTP に合わせます。
   - プチ音4か所と、最後の約3.5秒の音の差し替えは、`audio.py` を直すかどうか判断をお願いします。
5. **効果音の取得元：** 次の順で使います。
   - Kenney・OpenGameArt の CC0（agent-video-editor に入っている128個）
   - 効果音ラボ（表記不要）
   - Pixabay（表記不要。Cloudflare に止められるので、ユーザーのブラウザで取ってきてもらう）
   - 魔王魂（「効果音：魔王魂」の表記が必要）
   - Freesound は CC0 か CC BY だけ使い、NC は使いません。OtoLogic と Mixkit は、条文を確かめてから使います。

参照した Web の情報源：
- [capcut-cli COMPATIBILITY](https://cdn.jsdelivr.net/npm/capcut-cli-david@1.14.1/COMPATIBILITY.md)
- [pycapcut (PyPI)](https://pypi.org/project/pycapcut/)
- [emosheeep/capcut-export](https://github.com/emosheeep/capcut-export)
- [JyDraft README](https://gh.r-mj.com/HTWMedia/JyDraft/blob/main/README_EN.md)
- [video-autopilot-kit v0.8.0](https://github.com/Hao0321/video-autopilot-kit/releases/tag/v0.8.0)
- [jy-draftc](https://wingetgui.com/apps/wenshui330-jy-draftc)
- [効果音ラボ 規約](https://soundeffect-lab.info/agreement/)
- [魔王魂 ルール](https://maou.audio/rule/)
- [Pixabay ライセンス概要](https://pixabay.com/service/license-summary/)
- [Mixkit ライセンス](https://mixkit.co/license/)
- [Newcastle 大学の効果音サイト一覧（PDF）](https://www.newcastle.edu.au/__data/assets/pdf_file/0007/875995/Suggested-sound-effects-sites.pdf)
- [codegenes の CapCut Windows 版ガイド](https://www.codegenes.net/blog/capcut-download-windows/)
- [windowsmode の CapCut Windows 版ガイド](https://www.windowsmode.com/?p=22872)
- [Zenodo PANNs の学習済みモデル](https://zenodo.org/records/3987831)