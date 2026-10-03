# MAZE WIND~RETREAT 動画6本の最終書き出し（Remotion）

これまで CapCut で手作業で組んでいた最終の書き出しを、Remotion で組み立てます。
**色・拡大・回転・速度・減光などは、クリップ（`exports/capcut/…/*.mp4`）に焼き込み済み**です。
Remotion がするのは、組み立てだけです。

- クリップを決められたコマに並べる
- 透過PNG（右下のマーク・END・コピー・End Card）をフェード付きで重ねる
- IG の現地音（ambience）を入れる
- （あとで）BGM を入れる

カットの秒・PNGの出方・音は、すべて `src/timeline.json` のデータです。
コードにはありません。`timeline.json` は `scripts/build_timeline.py` が、マニフェストと EDL から作ります。

| Composition | 書き出しファイル | サイズ | 長さ | 音声 | 元になるデータ |
|---|---|---|---|---|---|
| `Signage30` | `MAZE_signage30.mp4` | 1920×1080 | 30.00秒（900コマ） | なし | `exports/capcut/signage_manifest.json`（v5）。料理の枠（カット4・5）は暫定で「縦パスタの2画面」 |
| `Signage15` | `MAZE_signage15.mp4` | 1920×1080 | 15.00秒（450コマ） | なし | 同上（v6）。料理の枠（カット3）は暫定で「縦パスタの2画面」 |
| `Signage30Cake` | `MAZE_signage30_cake.mp4` | 1920×1080 | 30.00秒（900コマ） | なし | `edl_result.json` の `edls.signage_cake`、`proposals/render_log_proposal.json`、`signage_spec.py` の `timing()` |
| `Signage18Cake` | `MAZE_signage18_cake.mp4` | 1920×1080 | 18.00秒（540コマ） | なし | `edls.signage15_cake`（同上） |
| `IG30` | `MAZE_ig30.mp4` | 1080×1920 | 30.00秒（900コマ） | AAC 48kHz | `edls.ig30`（v8）、`proposals/ig_v8_ig30/`、`ig_ambience_ig30_v8.wav`、PNGの秒は `ig_manifest.json` |
| `IG15` | `MAZE_ig15.mp4` | 1080×1920 | 15.00秒（450コマ） | AAC 48kHz | `edls.ig15`（v8）（同上） |

ケーキ版の2本と IG の2本には、撮影待ちのカットがあります。そこは黒の仮スレートのままです（`timeline.json` の `placeholders`）。
**仮スレートが残っている間は、公開用に使わないでください**（EDL の条件）。

**2026-10 最終QCのあとの修正（`clip_overrides.json` で差し替え中）**
- サイネージ30秒・15秒：料理の撮影待ちの枠（黒の仮スレート）を、ケーキ版で承認済みの「縦パスタの2画面」で埋めた**暫定の放映版**です。
  - 30秒版：カット4＋5（270コマ）を1本の連続した2画面で埋めます（`tools/render_signage_interim.py`、IMG_9644 1.0→5.5秒・IMG_9633 4.5→9.0秒、0.5倍）
  - 15秒版：ケーキ版のカット3と同じ区間の2画面（`proposals/signage15_interim/03_…`）
  - 右下の減光は、マークが消えるコマではマークの不透明度に合わせて弱めます（マークが消えたあとに海の右下へ暗いしみが残らないように）
  - 料理の2カットを撮ったら、`clip_overrides.json` の該当行を新しいクリップに書き換えます
- IG：QCの指摘を直したクリップ（`exports/capcut/proposals/ig_v8_scripts/ig_qc_fixes.py` → `proposals/ig_qc_fix/`）
  - IG30-4：ピントが合いきる4.93秒から、誘導灯が入る前の6.60秒までを0.726倍で元の尺に伸ばす
  - IG30-5：振り戻しを切り出し枠の動きで打ち消し、右へ流れて止まる動きにする。傾き1°を補正し、約2.1倍に寄ってレジ台を外す
  - IG30-7：窓の横桟より下のガラス全体（窓枠の内側の縁と窓台まで）に、下のガラスが曇ったような浅いぼかし。看板・スクーター・人物が読めなくなる
  - IG15-2：v8 のまま。枠を上げてレジ台を外すと水平線が約69%に下がり、カット1からのマッチカットが崩れたため戻した。販促ののぼりは A3 の撮り直しで解決する
  - IG15-3：わずかに寄って、下端の消波ブロックと手すりを外す
  - IG15-8（END）：上下の揺れを測って枠で追い、文字が出る間の水平線の上下を止める
  - End Card（IG30・IG15）と IG30 の c10 のコピー：文字の不透明度を95%→100%（`edl/mockups/ig_endcard_spec.py`。IG15 の「@mazewind2026」が4.0:1をわずかに下回るコマがあったため）。元のPNGは `exports/capcut/overlays/_before_2026-10/`

---

## 1. 書き出し（作り直し）

`remotion/` フォルダで実行します。

```bash
cd remotion
npm install                        # 初回だけ
python3 scripts/build_timeline.py  # マニフェスト・EDL → src/timeline.json、public/ にリンクを張る
node scripts/render.mjs            # 6本すべて → ../exports/final/
python3 scripts/verify.py          # 見本と比べる → ../exports/final/verify_report.json
python3 scripts/check_fades.py     # PNGのフェードが見本と同じコマで始まり・終わるか
```

書き出しの時間の目安（4コア）：30秒版は1本10〜11分、15秒版は1本5〜6分、18秒版は約6分です。6本すべてで約50分かかります。

- 1本だけ書き出す場合：`node scripts/render.mjs IG30`（複数も可：`node scripts/render.mjs Signage30 Signage15`）
- 試しに一部だけ書き出す場合：`node scripts/render.mjs IG30 --out out --frames 560-680`（出力先は `remotion/out/`）
- 画面で確かめる場合：`npx remotion studio`（ブラウザでタイムラインを見られる）
- 出力：`exports/final/MAZE_*.mp4`。書き出しの時間とサイズは `exports/final/render_log.json` に残ります
- 仕様：H.264 High・**レベル4.1**、CRF 17、yuv420p、BT.709（タグ付き・limited）、30fps
  - レベルは、屋外サイネージの再生機の多くが「High@4.1/4.2まで」としているため 4.1 に固定しています
  - サイネージの4本：音声トラックなし（`muted`）
  - IG の2本：AAC 48kHz 256kbps
- **`npx remotion render` で直接 IG を書き出さないでください。**
  Remotion が mp4 に直接 AAC を書くと、音が映像より約43ms（2048サンプル）遅れます（実測）。
  `render.mjs` は、IG をいったん h264-mkv（映像と無圧縮の音）で書き出し、そのあと音だけ AAC にして mp4 に詰め替えます。この方法では、ずれは0です。
  サイネージは音声がないので、どちらの方法でも同じ結果になります
- Chromium：作業環境にある `/opt/pw-browsers/…/headless_shell` を使います。別の場所にある場合は、環境変数 `REMOTION_CHROME=/path/to/chrome` で指定します

### 色について（自動で補正しています）
Remotion は、動画のコマを同梱の ffmpeg で YUV から RGB に変換してから、ブラウザに渡します。
この変換は切り捨て寄りのため、そのままでは RGB が約1〜1.5段（0〜255のうち）暗くなります。
これを打ち消すため、クリップの層にだけ 8bit の補正表（`src/colorFix.json`、SVG の feComponentTransfer）をかけています。
黒は黒のまま変わりません。PNG には補正をかけません。

- 補正表を作り直す場合：`python3 scripts/calibrate_color.py`。ふだんは不要です。Remotion を上げたときに `verify.py` の差が大きくなったら実行します
- 補正を切る場合：props で `{"colorFix": false}` を渡します

---

## 2. 撮り直したクリップへの差し替え（ケーキ・料理の撮影のあと）

クリップは、これまでと同じように**色・拡大・速度まで焼き込んだ完成クリップ**として作ります。
形式は、サイネージが 1920×1080、IG が 1080×1920、どちらも 30fps・H.264・BT.709 です。
Remotion は、クリップの色や動きには手を加えません。

**方法A（いちばん簡単）：仮スレートと同じ名前で上書きする**
1. 新しいクリップを、仮スレートと同じパス・同じ名前で置きます
   - 例：`exports/capcut/proposals/signage30_cake/05_placeholder_cake_horizon_120fps.mp4`
2. 作り直します
   ```bash
   python3 scripts/build_timeline.py
   node scripts/render.mjs Signage30Cake
   ```
   この方法では、`timeline.json` の上では「仮スレート」の印が残ります。気になる場合は方法Bを使います

**方法B：新しい名前で置き、差し替え表に書く**
1. 例：`exports/capcut/proposals/signage30_cake/05_IMG_9800_cake.mp4` のように置きます
2. `clip_overrides.example.json` を `clip_overrides.json` という名前でコピーし、Composition 名とカット番号（EDL の `n`）で指定します
   ```json
   {"Signage30Cake": {"5": "exports/capcut/proposals/signage30_cake/05_IMG_9800_cake.mp4"}}
   ```
3. `python3 scripts/build_timeline.py` を実行してから、書き出します

**IG の場合**：`proposals/ig_v8_ig30/` と `ig_v8_ig15/` の中の、カット番号で始まるファイル（`08_…mp4` など）が使われます。
仮スレートのファイルを消して、同じ番号で始まる新しいクリップ（例：`08_IMG_9800.mp4`）を置けば、そのまま差し替わります。
方法Bの差し替え表も使えます。
現地音（`ig_ambience_*_v8.wav`）を撮り直しに合わせて作り直した場合も、同じファイル名で置き換えるだけで反映されます。

`build_timeline.py` は、次の点を確かめます。問題があると `timeline.json` を作りません。
- クリップのサイズと fps
- コマ数が足りているか（多い場合は、頭から必要な分だけ使い、注意を表示します）
- クリップの間にすき間や重なりがないか
- サイネージの PNG の出方が、マニフェストの値と式（`timing()`）で一致するか

**カットの秒を変える場合**は、`timeline.json` を手で直さないでください。
元の EDL（`edl/edl_result.json`）かマニフェストを直してから、`build_timeline.py` をもう一度実行します。
FIXED RULES と EDL の承認の流れは、これまでと同じです。

---

## 3. BGM を入れる（初期設定は「なし」）

BGM の候補（OpenTracks の3曲）は `edl/capcut/BGM_CANDIDATES.md` にあります。曲が決まったら、次の手順で入れます。

1. 曲のファイルを `remotion/public/bgm/` に置きます
   - 例：`remotion/public/bgm/17543.mp3`
   - `public/` は Git に入りません。元のファイルは別に保管しておきます
2. props の JSON を作ります
   - 例：`bgm_ig30.json`
   - 秒はタイムライン上の秒です。`startFromS` だけは曲の中の秒です
   ```json
   {
     "bgm": {
       "src": "bgm/17543.mp3",
       "startFromS": 12.3,
       "startAtS": 1.5,
       "gainDb": -14,
       "fadeInS": 0,
       "fadeOutS": 0.7,
       "endAtS": 29.7,
       "ducking": [
         {"fromS": 22.3, "toS": 23.0, "gainDb": -2.5, "rampS": 0.15}
       ]
     }
   }
   ```
   | 項目 | 意味 | 既定値 |
   |---|---|---|
   | `src` | `public/` から見たパス（または http の URL） | （必須） |
   | `startFromS` | 曲の何秒目から使うか | 0 |
   | `startAtS` | タイムラインの何秒目で鳴り始めるか | 0 |
   | `endAtS` | 何秒目で止めるか | 最後まで |
   | `gainDb` | 全体の音量（dB） | 0 |
   | `fadeInS` | 鳴り始めのフェードの長さ（秒） | 0 |
   | `fadeOutS` | 止める位置の手前でのフェードの長さ（秒） | 0 |
   | `ducking` | 下げる区間の一覧（下の説明を参照） | なし |

   `ducking` の各区間は、`fromS` から `rampS` 秒かけて `gainDb` まで下げ、`toS` の `rampS` 秒前から戻し始めます（`rampS` の既定は 0.3秒）。
   下げる位置の目安は、`ig_manifest.json` の `bgm_ducking_points` にあります。
3. 書き出します
   ```bash
   node scripts/render.mjs IG30 --props bgm_ig30.json
   ```

BGM は、音声のある IG の2本にだけ入ります。サイネージは音声なしのままです（props を渡しても入りません）。

---

## 4. ファイルの構成

```
remotion/
  scripts/build_timeline.py    マニフェスト・EDL → src/timeline.json、public/ のリンク
  scripts/render.mjs           書き出し（6本まとめて／指定の本だけ）
  scripts/verify.py            見本との比較（コマ数・長さ・解像度・色のタグ・音声・画素の差）
  scripts/check_fades.py       PNGのフェードの始まり・終わりのコマを見本と比べる
  scripts/calibrate_color.py   色の補正表（src/colorFix.json）を作る
  src/Root.tsx                 Composition 6本（timeline.json から）
  src/Edit.tsx                 クリップ（OffthreadVideo・無音）、PNG（Img・直線のフェード）、現地音（Audio）、BGM
  src/timeline.json            自動生成。手で直さない
  clip_overrides.example.json  差し替え表の見本
  public/                      素材へのシンボリックリンクだけ（capcut → exports/capcut）。Git に入れない
  out/                         試し書き。Git に入れない
```

動画はコピーしません。
`public/capcut` は `exports/capcut` へのリンクで、書き出しのときもリンクのまま使います（`symlinkPublicDir`）。

---

## 5. Remotion のライセンス（必ず確認してください）

Remotion は、使う人や会社によっては有料です。
- 無料で使えるのは、個人、または従業員3人以下の会社などです
- それより大きい会社や組織では、会社向けのライセンス（Company License）が必要です

条件は変わることがあります。お店やお店の運営会社、制作を頼む側の規模に当てはめて、**使う前に必ず [remotion.dev/license](https://www.remotion.dev/license) で確かめてください**。
（この README の内容は、ライセンスの判断そのものではありません。）
