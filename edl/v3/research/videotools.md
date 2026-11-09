# 動画制作ツール調査（2026-11-06）

12本のリポジトリ（以下「リポ」）を `/home/user/vendor/` に `--depth 1` で複製しました。pip・npm のインストールはしていません。Higgsfield は読み取りの3つ（balance / models_explore / show_plans_and_credits）だけを呼び、生成系は一度も呼んでいません。

先に結論です。
- **Open-Higgsfield 系は不要です。** どれも有料 API の画面だけで、手元で動く生成はありません。
- **Higgsfield の残高は 8.12 クレジットです。** 動画は1本も作れません。
- **本命は Remotion 公式スキルと Depth Anything V2 Small（ONNX）です。** 後者は今の `tools/v2` がもう前提にしていますが、モデルファイルと Python の依存が入っていません（§3）。

## 1. 判定一覧

★の数と最終更新は、GitHub 検索 API で 2026-11-06 に取得した値です。

| ツール | GitHub | ★ | ライセンス | 最終更新 | 4CPU・GPUなし | 判定 | 理由と使い道 |
|---|---|---|---|---|---|---|---|
| Remotion 公式エージェントスキル（`/remotion-best-practices` ほか12個） | github.com/remotion-dev/skills | 6,471 | 表示なし（中身は Remotion 公式ドキュメントへの案内） | 2026-11-05 | ○ | **必須** | 最終の組み立ては `remotion/`（4.0.529）で行っている。字幕・書き出し・フォント・タイミングの作法を参照する。下の注意あり |
| Depth Anything V2 Small（ONNX） | github.com/fabio-sim/Depth-Anything-ONNX（変換と推論の例）、本家は github.com/DepthAnything/Depth-Anything-V2 | 447 / 9,111 | Apache-2.0（Small の重み。Base・Large は CC-BY-NC） | 2024-10-03 / 2026-03-24 | ○（onnxruntime の CPU で動く） | **必須** | `tools/v2/common.py` の `depth()` が `tools/models/depth_anything_v2_small.onnx` を読むが、ファイルがない。学習済みの ONNX は HuggingFace `onnx-community/depth-anything-v2-small` から取れることを確認済み。写真から作る擬似3Dの仮想カメラに使う |
| faster-whisper | github.com/SYSTRAN/faster-whisper | 26,531 | MIT | 2026-11-06 | ○（CPU の int8。README では small モデルで約1分42秒、`batch_size=8` で51秒） | 有益 | 豆知識動画にナレーションを付けるなら、`word_timestamps=True` で語ごとの秒を取り、字幕（カラオケ表示）をコマ単位で合わせる |
| Practical-RIFE（コマ補間） | github.com/hzwer/Practical-RIFE | 1,037 | MIT | 2026-08-27 | △（CUDA がなければ自動で CPU。1080p で1コマ数秒と見込む） | 有益（条件付き） | 速度の上げ下げや超スロー（「人間には不可能な精度」の見せ場）用。iPhone は 4K120 で撮れるので、出番は限られる。重みは Google Drive から別に取る必要がある |
| claude-code-video-toolkit | github.com/digitalsamba/claude-code-video-toolkit | 2,463 | MIT | 2026-11-06 | △（Remotion の部分は動く。AI 生成は Modal などのクラウド GPU が前提） | 有益（参照） | 縦9:16の解説ショートの型と、コンタクトシートで見直すやり方を参考にする |
| claude-faceless-shorts-creator | github.com/hassancs91/claude-faceless-shorts-creator | 310 | MIT | 2026-08-18 | ○（TSX 版は Remotion だけで動く） | 有益（参照） | 30〜40秒のショートの「6拍の構成」、語ぴったりの字幕、スマホの大きさでの画質チェック。豆知識動画の構成にそのまま使える。声は ElevenLabs が前提 |
| video-use | github.com/browser-use/video-use | 30,640 | MIT | 2026-10-08 | ○（ffmpeg）。ただし文字起こしは ElevenLabs Scribe（有料 API） | 有益（参照） | 人が話す素材向けで、本件とは方向が違う。切れ目ごとに書き出しを自分で確認する手順と、フィルムストリップ＋波形＋単語を並べた確認画像は、QC に借りられる |
| Spectral Film LUT | github.com/JanLohse/spectral_film_lut | 127 | MIT | 2026-11-04 | △（PyQt6 の画面アプリ。画面なしで動くかは未確認） | 有益 | フィルムのデータシートから `.cube` LUT を作る。作った LUT を ffmpeg の `lut3d` に渡し、サイネージに映画的な色を付ける。粒子の合成機能もある |
| awesome-claude-video-skills | github.com/zhuyansen/awesome-claude-video-skills | 1,665 | NOASSERTION（不明） | 2026-11-05 | — | 有益（索引） | エージェント向け動画スキル180本の一覧と、そのうち16本を同じ条件で試した結果。結果では Remotion 公式スキルが「確実に書き出せる（4.5分）」、hyperframes が「最も遅い（23.5分）」。何も入れない Claude Code も全チェックに合格していた |
| awesome-opus-5.5-video | github.com/zhuyansen/awesome-opus-5.5-video | 307 | なし | 2026-11-05 | — | 有益（参考） | ご指名の名前と完全一致。Opus 5.5 で作った動画とプロンプトの事例集 |
| awesome-opus5-5-videos | github.com/yihui-dev/awesome-opus5-5-videos | 13,861 | MIT | 2026-10-13 | — | 有益（参考） | 同じ種類の事例集で最大。ほかに opusvideo/awesome-claude-video（★238）と athemeroy/awesome-claude-5-5-videos（★962）がある |
| HyperFrames | github.com/heygen-com/hyperframes | 79,993 | Apache-2.0 | 2026-11-06 | ○（Node22＋ffmpeg） | 不要 | HTML を MP4 にする仕組みで、`remotion/` と役割が重なる。描画の仕組みを2つ持つと事故が増える。約450個の部品集は参考になるが、リポが 520MB あるため複製しなかった |
| Open-Higgsfield（ご指名の Anil-matcha/Open-Higgsfield-AI） | github.com/Anil-matcha/Open-Higgsfield-AI（git の参照は通るが API と検索では見えない。改名先は Anil-matcha/Open-Generative-AI と推定） | 32,798 | MIT | 2026-11-06 | ×（MuAPI という有料 API の画面にすぎない） | 不要 | 手元では何も生成しない。README の中身はほぼ宣伝 |
| Open-Higgsfield（別系統） | github.com/wide-trace/open-higgsfield ほか（ClabstreamTeam/Open-Higgsfield-AI など） | 5,416 ほか | なし | 2026-09-17 | × | 不要 | Higgsfield の鍵を入れて使う Next.js の画面。今つながっている MCP と同じことしかできない |
| rife-ncnn-vulkan | github.com/nihui/rife-ncnn-vulkan | 1,127 | MIT | 2024-01-02 | ×（Vulkan GPU が要る。この環境にはない） | 不要 | CPU で動く Practical-RIFE か、ffmpeg の `minterpolate` で代わりになる |
| FILM | github.com/google-research/frame-interpolation | 3,155 | Apache-2.0 | 2024-08-10（**アーカイブ済み**） | △（TensorFlow で重い） | 不要 | 更新が止まっている |
| Real-ESRGAN / ncnn 版 | github.com/xinntao/Real-ESRGAN / github.com/xinntao/Real-ESRGAN-ncnn-vulkan | 37,388 / 2,290 | BSD-3 / NOASSERTION | 2024-08 / 2024-05 | ×（ncnn 版は Vulkan が要る。PyTorch の CPU 版はとても遅い） | 不要 | 素材は iPhone の 4K で、書き出しは 1080 幅。お店の素材の解像度が低い場合だけ検討する |
| whisperX / stable-ts / whisper-timestamped | m-bain/whisperX（BSD-2）、jianfch/stable-ts（**アーカイブ済み**）、linto-ai/whisper-timestamped（AGPL） | 25k / 2.3k / 2.9k | — | — | △ | 不要 | faster-whisper で足りる。whisperX は torch と wav2vec2 で重く、AGPL は避ける |
| Motion Canvas | github.com/motion-canvas/motion-canvas | 19,711 | MIT | 2026-07-02 | ○ | 不要 | Remotion と役割が重なる |
| OpenMontage / video-shotcraft / ffmpeg-skill | calesthio/OpenMontage（AGPL、★77k）、Vincentwei1021/video-shotcraft（Apache、★17k、製品画面の宣伝向け）、kajisho5/ffmpeg-skill（MIT、★4.6k） | — | — | — | — | 不要 | AGPL、目的が違う、今の `tools/v2` と重なる、のいずれか |
| filmgrainer | github.com/larspontoppidan/filmgrainer | 56 | MIT | 2022-11-21 | ○ | 不要 | 粒子は ffmpeg の `noise` か numpy で足りる |
| ffmpeg の内蔵フィルター | （ffmpeg 6.1.1、この環境に入っている） | — | LGPL/GPL | — | ○ | **必須** | `lut3d` `haldclut` `noise`（粒子） `vignette` `curves` `colortemperature` `minterpolate` `framerate` `tmix` `vidstab*` `deflicker` `zscale` `ass` `drawtext` が使えることを確認した。`libplacebo` は Vulkan が要るので使えない |

**フォント（4つとも OFL-1.1 で商用利用できます）**

google/fonts の `ofl/<名前>/` に METADATA.pb があることを確認しました（HTTP 200）。フォントのリポは大きいので複製せず、必要なときに `raw.githubusercontent.com/google/fonts/main/ofl/…` から TTF を取る方法を勧めます。

| フォント | 置き場所 | 判定 | 使い道 |
|---|---|---|---|
| Noto Serif JP | `ofl/notoserifjp` | 必須 | サイネージの明朝。字の種類が多く、太さもそろっている |
| Zen Old Mincho | `ofl/zenoldmincho`（googlefonts/zen-oldmincho） | 有益 | 映画のような見出し |
| Shippori Mincho | `ofl/shipporimincho`（fontdasu/ShipporiMincho） | 有益 | 同上（候補として） |
| BIZ UDPGothic / BIZ UDPMincho | `ofl/bizudpgothic` `ofl/bizudpmincho` | 有益 | 豆知識動画の本文や字幕。小さい画面でも読みやすい |

今の `assets/fonts` には Noto Sans JP（Light・Regular・Medium）しかありません。

## 2. Higgsfield MCP（読み取りだけで確認）

**残高**
- 8.12 クレジット、プランは `free`。
- 料金表の目安は「100クレジットで Nano Banana Pro が約50枚、Seedance 2.0 が約4本」です。今の残高では**動画は0本、画像はおよそ4枚**しか作れません。
- 有料プランは PLUS（月49ドル、1,000クレジット）と ULTRA（月129ドル、3,000クレジット）です。
- 3日間の無料お試しも表示されました。100クレジットで MCP の中でしか使えず、カードの登録が必要で、解約しないと月49ドルで自動更新されます。

**動画のモデル**
- 主なもの：Seedance 2.0 / 2.0 Mini / 2.5（4〜30秒）、Kling 3.0 / 4.0 / 4.0 Flash、Veo 3.1 / Lite、Wan 2.6 / 2.7 / 3.0 / 3.0 Prime、MiniMax Hailuo / H3、Grok Video 1.5、FLUX 3 Video、Gemini Omni Flash 1.1 / 1.2、Cinema Studio 3.0 / 3.5 / 4.0、Marketing Studio。
- 編集・後処理：Topaz、Bytedance Video Upscale、FPS Boost、Video Deflicker、Reframe、Depth Anything Video、背景の除去、Kling や FLUX の動画編集。

**画像のモデル**
- GPT Image 2 / 2.5 / 2.6、Nano Banana / 2 / 2.1 / Pro、Seedream 4.5 / 5.0、FLUX.2 / FLUX 3、Soul 2.0 / Soul Cinema、Kling O1、Ideogram 4.5 / Vector、Recraft V4.1、Grok Image。
- 補助：Topaz の画像拡大、外側への描き足し、レイヤー分解。

**音声のモデル**
- Eleven v4 / v4 Turbo（読み上げ）、ElevenLabs Music v2 / v2.5、Seed Audio 1.0、Qwen Audio 3.0 TTS（日本語に対応）、Text to Speech V2、Inworld TTS。

**使うときの注意**
- 実在の施設を AI で描くと、LP の事実とずれるおそれがあります。
- 使うとしても、実写素材の拡大・コマ数の追加・BGM や読み上げなど、事実を変えない処理に限るのが安全です。

## 3. 今のパイプラインに組み込むときの注意
- **`tools/v2` は今の環境では動きません。** `cv2` `numpy` `onnxruntime` `pillow_heif` `shapefile`（pyshp）が入っていません。動かすには `pip install opencv-python-headless numpy onnxruntime pillow-heif pyshp` が必要です（今回は入れていません）。
- **深度のモデルがありません。** `tools/models/depth_anything_v2_small.onnx` を HuggingFace の `onnx-community/depth-anything-v2-small`（`onnx/model.onnx`、Apache-2.0）から置く必要があります。先に `.gitignore` に入れるかどうか決めてください。
- **Remotion のライセンス：** 無料で使えるのは「個人、または社員3人までの営利団体」です。それより大きい組織は Company License が要ります。
- **Remotion スキルの版：** 一部の内容は v4.0.540 以降を前提にしています（今の `remotion/` は 4.0.529）。

**全体の流れ（案）**
1. 写真や素材の加工は `tools/v2`（深度で仮想カメラ、relight、mapzoom）で行う。
2. 色は Spectral Film LUT で作った `.cube` を ffmpeg `lut3d` に通し、`noise` で粒子、`vignette` で周辺減光を足す。
3. 必要な区間だけ RIFE でコマを補間する。
4. 連番 PNG にしてから Remotion で文字・字幕を重ねる。字幕の秒は faster-whisper の語ごとの秒を使う。
5. 最後に ffmpeg で書き出す。

## 4. 複製したもの（すべて `--depth 1`、`/home/user/vendor/` 以下）
| フォルダ | 元のリポ | 大きさ |
|---|---|---|
| `/home/user/vendor/remotion-skills` | remotion-dev/skills | 4.1M |
| `/home/user/vendor/Depth-Anything-ONNX` | fabio-sim/Depth-Anything-ONNX | 15M |
| `/home/user/vendor/faster-whisper` | SYSTRAN/faster-whisper | 43M |
| `/home/user/vendor/Practical-RIFE` | hzwer/Practical-RIFE | 6.1M（重みは別に取る必要あり） |
| `/home/user/vendor/claude-code-video-toolkit` | digitalsamba/claude-code-video-toolkit | 66M |
| `/home/user/vendor/claude-faceless-shorts-creator` | hassancs91/claude-faceless-shorts-creator | 264M |
| `/home/user/vendor/video-use` | browser-use/video-use | 1.7M |
| `/home/user/vendor/spectral_film_lut` | JanLohse/spectral_film_lut | 3.8M |
| `/home/user/vendor/awesome-claude-video-skills` | zhuyansen/awesome-claude-video-skills | 56M |
| `/home/user/vendor/awesome-opus-5.5-video` | zhuyansen/awesome-opus-5.5-video | 3.9M |
| `/home/user/vendor/awesome-opus5-5-videos` | yihui-dev/awesome-opus5-5-videos | 4.8M |

複製しなかったもの：hyperframes（520MB で、Remotion と重なる）、フォントのリポ（大きすぎるので TTF を個別に取る）、表で「不要」にしたリポすべて。