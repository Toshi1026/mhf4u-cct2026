# MAZE WIND RETREAT — SNS Short Movie

高知県土佐市の海沿いにあるカフェ「MAZE WIND RETREAT」の縦型ショート動画（15秒版と30秒版）を制作するプロジェクトです。

## 最初に読むもの
- `BRIEF.md` — Creative Rules（**FIXED RULESは変更禁止**）、15秒版と30秒版の目的、作業手順
- `SHOOTING_GUIDE.md` — iPhone 16 Pro Maxでの撮影ガイド
- `STORE_ASSETS_REQUEST.md` — お店にお願いする素材のリスト

## 守ること
- FIXED RULES（広告っぽく見せない／主役は海と水平線／場所 → 空気 → 料理の順）を、Claudeの判断で変えたり別の案に置き換えたりしない
- 素材が足りないときは方向性を変えず、次の4点をユーザーに提示する：不足している素材、なぜ必要か、追加で撮るべきショット、iPhoneでの撮り方
- EDL（Edit Decision List）を提示して**承認を得るまで、実際の編集は始めない**
- 原素材と書き出した動画はGitに入れない（`footage/` と `exports/` は `.gitignore` で除外済み）

## 素材の置き場所
- Google Drive「MAZE 広告動画用素材」
  - `01_撮影素材`（リンク共有）：iPhoneで撮った素材
  - `02_お店提供`（非公開）：ロゴ、空撮など、お店からもらった素材
- ダウンロード：`tools/fetch.sh <フォルダのURL>` を実行すると `footage/` に保存される
- 解析：`python3 tools/analyze.py footage/ edl/analysis/` を実行すると、`report.md`、`report.csv`、各クリップのサムネイル一覧が出力される
