#!/usr/bin/env python3
"""Render the footage-to-EDL workflow result into review documents.

Usage: python3 render_edl.py <result.json> <edl_dir>

Writes CATALOG.md, GAPS.md and one EDL_<version>.md per deliverable.
"""
import json
import sys
from pathlib import Path

CATEGORY = {
    "A_HERO_OCEAN": "A 海（HERO OCEAN）",
    "B_ARRIVAL": "B 到着・発見",
    "C_INTERIOR": "C 店内",
    "D_PASTA": "D パスタ",
    "E_CAKE": "E ケーキ・デザート",
    "F_EXPERIENCE": "F 体験",
    "G_DETAIL": "G ディテール",
    "H_LOCATION": "H 周辺・場所",
    "X_UNUSABLE": "X 使用不可",
}
VERSION = {
    "signage": "屋外サイネージ 30秒（横 1920×1080・無音・ループ）",
    "ig30": "Instagram 30秒（縦 1080×1920）",
    "ig15": "Instagram 15秒（縦 1080×1920）",
}
ORIENT = {"horizontal": "横", "vertical": "縦"}
PRIORITY = {"must": "必須", "should": "推奨", "nice": "あれば"}


def cell(s):
    return str(s).replace("|", "／").replace("\n", " ")


def tc(sec):
    return f"{int(sec // 60)}:{sec % 60:05.2f}"


def catalog_md(res, facts):
    clips = sorted(res["catalog"], key=lambda c: c["file"])
    out = ["# 素材カタログ", ""]
    out.append(f"全{len(clips)}ファイル（動画{sum(c['kind'] == 'video' for c in clips)}本・写真{sum(c['kind'] == 'photo' for c in clips)}枚）。"
               "各カットの見た目は `sheets/<ファイル名>.jpg`（タイムコード付きの一覧）で確認できます。")
    out.append("")
    out.append("## カテゴリ別の本数")
    out.append("")
    out.append("| カテゴリ | 本数 | ファイル |")
    out.append("|---|---|---|")
    for key, name in CATEGORY.items():
        fs = [c["file"] for c in clips if c["category"] == key]
        if fs:
            out.append(f"| {name} | {len(fs)} | {', '.join(f.split('.')[0] for f in fs)} |")
    out.append("")
    out.append("## ファイル別の詳細")
    for c in clips:
        f = facts.get(c["file"], {})
        head = f"### {c['file']} — {CATEGORY.get(c['category'], c['category'])}（{c['shot_id']}）"
        out += ["", head, ""]
        meta = [ORIENT.get(c["orientation"], c["orientation"])]
        if c["kind"] == "video":
            meta += [f"{f.get('duration_s', '?')}秒", f"{f.get('fps', '?')}fps", c["camera_move"]]
            if f.get("shake_px"):
                meta.append(f"揺れ {f['shake_px']}")
            if f.get("audio_wind_ratio"):
                meta.append(f"風音比 {f['audio_wind_ratio']}")
        else:
            meta.append("写真")
        meta.append(f"強さ {'★' * c['hero_score']}{'☆' * (5 - c['hero_score'])}")
        out.append("- " + " / ".join(str(m) for m in meta))
        out.append(f"- 内容：{c['description_ja']}")
        if c["segments"]:
            out.append("")
            out.append("| # | 区間 | 内容 | 品質 | メモ |")
            out.append("|---|---|---|---|---|")
            for i, s in enumerate(c["segments"]):
                best = "◎ " if i == c["best_segment"] else ""
                out.append(f"| {best}{i} | {s['in_s']:.1f}–{s['out_s']:.1f}s | {cell(s['content_ja'])} | {s['quality']} | {cell(s['note_ja'])} |")
            out.append("")
        if c["issues_ja"]:
            out.append("- 問題点：" + "／".join(c["issues_ja"]))
        if c["other_people_faces"]:
            out.append("- ⚠ 他のお客さんの顔が映っている")
        out.append(f"- 使いどころ：サイネージ＝{c['fit_signage']}／IG30＝{c['fit_ig30']}／IG15＝{c['fit_ig15']}")
    if res.get("corrections"):
        out += ["", "## 検証で修正した点", ""]
        out += [f"- {x}" for x in res["corrections"]]
    return "\n".join(out) + "\n"


def gaps_md(res):
    g = res.get("gaps") or {}
    out = ["# 不足素材と追加撮影リスト", ""]
    if g.get("rule_risks_ja"):
        out += ["## FIXED RULESに関わるリスク", ""]
        out += [f"- {x}" for x in g["rule_risks_ja"]]
        out.append("")
    missing = sorted(g.get("missing", []), key=lambda m: ["must", "should", "nice"].index(m["priority"]))
    if missing:
        out += ["## 追加で撮るべきショット", ""]
        out.append("| 優先度 | 不足している素材 | 使う版 |")
        out.append("|---|---|---|")
        for m in missing:
            out.append(f"| {PRIORITY[m['priority']]} | {cell(m['shot'])} | {', '.join(m['versions'])} |")
        for m in missing:
            out += ["", f"### {PRIORITY[m['priority']]}：{m['shot']}", "",
                    f"- **なぜ必要か**：{m['why_ja']}",
                    f"- **追加で撮るべきショット**：{m['what_ja']}",
                    f"- **iPhone 16 Pro Maxでの撮り方**：{m['how_ja']}"]
        out.append("")
    if g.get("shooting_notes_ja"):
        out += ["## 次の撮影で直すこと", ""]
        out += [f"- {x}" for x in g["shooting_notes_ja"]]
        out.append("")
    if g.get("have_ja"):
        out += ["## 撮れているもの", ""]
        out += [f"- {x}" for x in g["have_ja"]]
    return "\n".join(out) + "\n"


def edl_md(key, r):
    e = r["edl"]
    out = [f"# EDL：{VERSION.get(key, key)}", ""]
    out.append(f"**コンセプト**：{e['concept_ja']}")
    out.append("")
    out.append(f"**書き出し**：{e['format_ja']}")
    out.append("")
    if r.get("remaining"):
        out.append("> ⚠ 自動チェックで未解決の問題：" + "／".join(r["remaining"]))
        out.append("")
    out.append("| # | 時間 | 長さ | 素材 | 使う区間 | 速度 | クロップ | つなぎ | 文字 | 音 | 狙い |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in e["cuts"]:
        src = "（追加撮影待ち）" if c["placeholder"] else c["source"].split(".")[0]
        rng = "—" if c["placeholder"] else f"{c['src_in']:.2f}–{c['src_out']:.2f}s"
        speed = f"{c['speed']:g}x"
        out.append(
            f"| {c['n']} | {tc(c['rec_in'])}–{tc(c['rec_out'])} | {c['duration']:.2f}s | {src} | {rng} | {speed} | "
            f"{cell(c['crop_ja'])} | {cell(c['transition_ja'])} | {cell(c['text_ja'])} | {cell(c['audio_ja'])} | {cell(c['purpose_ja'])} |"
        )
    total = sum(c["duration"] for c in e["cuts"])
    out += ["", f"合計 {total:.2f}秒（{len(e['cuts'])}カット）", ""]
    out += ["## 音楽", "", e["music_ja"], "", "## 色（HDR→SDR）", "", e["color_ja"], ""]
    if e["capcut_tasks_ja"]:
        out += ["## CapCutで行う作業", ""]
        out += [f"{i + 1}. {x}" for i, x in enumerate(e["capcut_tasks_ja"])]
        out.append("")
    if e["notes_ja"]:
        out += ["## メモ・リスク", ""]
        out += [f"- {x}" for x in e["notes_ja"]]
        out.append("")
    crit = r.get("critiques") or []
    if crit:
        out += ["## レビューの結論（修正前の案に対して）", ""]
        out += [f"- {c['verdict_ja']}" for c in crit]
    return "\n".join(out) + "\n"


def main():
    res = json.load(open(sys.argv[1]))
    d = Path(sys.argv[2])
    facts = json.load(open("/tmp/claude-0/wf/facts.json")) if Path("/tmp/claude-0/wf/facts.json").exists() else {}
    (d / "CATALOG.md").write_text(catalog_md(res, facts))
    (d / "GAPS.md").write_text(gaps_md(res))
    names = {"signage": "EDL_signage30.md", "ig30": "EDL_ig30.md", "ig15": "EDL_ig15.md"}
    for key, r in (res.get("edls") or {}).items():
        if r and r.get("edl"):
            (d / names.get(key, f"EDL_{key}.md")).write_text(edl_md(key, r))
    print("written to", d)


if __name__ == "__main__":
    main()
