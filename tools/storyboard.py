#!/usr/bin/env python3
"""Storyboard images for reviewing an EDL: one row per cut, frames at the
cut's source in / middle / out, tone-mapped to SDR.

Usage: python3 storyboard.py <result.json> <footage_dir> <out_dir>

Labels: cut number, record timecode, source, source range, speed and the start
of the cut's purpose (Japanese, IPAGothic). Placeholder cuts show a grey row.
"""
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
TONEMAP = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
    "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)
ROW_H = 300
GAP = 6
LABEL_W = 420
JP_FONT = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"


def font(size):
    try:
        return ImageFont.truetype(JP_FONT, size)
    except OSError:
        return ImageFont.load_default()


def wrap(text, n, lines):
    text = text.replace("\n", " ")
    return [text[i:i + n] for i in range(0, min(len(text), n * lines), n)]


def frame(path, t):
    if path.suffix.lower() == ".heic":
        import pillow_heif
        from PIL import ImageOps
        pillow_heif.register_heif_opener()
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    else:
        out = subprocess.run(
            [FFMPEG, "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(path), "-frames:v", "1",
             "-vf", f"scale=-2:{ROW_H * 2},{TONEMAP}", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True)
        if not out.stdout:
            return None
        im = Image.open(BytesIO(out.stdout)).convert("RGB")
    w = round(im.width * ROW_H / im.height)
    return im.resize((w, ROW_H))


def board(edl, footage, out):
    rows = []
    for c in edl["cuts"]:
        label = [f"#{c['n']}  {c['rec_in']:.2f}-{c['rec_out']:.2f}s",
                 f"dur {c['duration']:.2f}s  x{c['speed']:g}"]
        if c["placeholder"]:
            label.append("撮影待ち")
            label += wrap(c["purpose_ja"], 16, 3)
            rows.append((label, []))
            continue
        label.append(f"{c['source'].split('.')[0]}  {c['src_in']:.2f}-{c['src_out']:.2f}s")
        label += wrap(c["purpose_ja"], 16, 3)
        p = footage / c["source"]
        span = c["src_out"] - c["src_in"]
        ts = [c["src_in"] + 0.03, c["src_in"] + span / 2, max(c["src_out"] - 0.05, c["src_in"])]
        frames = [f for f in (frame(p, t) for t in ts) if f]
        rows.append((label, frames))
    width = LABEL_W + max((sum(f.width for f in fr) + GAP * len(fr) for _, fr in rows), default=900)
    width = max(width, LABEL_W + 900)
    canvas = Image.new("RGB", (width, len(rows) * (ROW_H + GAP) + 50), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    d.text((10, 10), f"{edl['version']}  {edl['target_s']:g}s  ({len(edl['cuts'])} cuts)", fill=(255, 255, 255), font=font(28))
    y = 50
    for label, frames in rows:
        for i, line in enumerate(label):
            d.text((10, y + 8 + i * 30), line, fill=(255, 220, 0) if i == 0 else (230, 230, 230), font=font(26 if i == 0 else 22))
        x = LABEL_W
        if not frames:
            d.rectangle([x, y, width - 10, y + ROW_H], fill=(70, 70, 70))
            d.text((x + 20, y + ROW_H // 2 - 16), "撮影待ち（追加撮影が必要）", fill=(255, 255, 255), font=font(32))
        for f in frames:
            canvas.paste(f, (x, y))
            x += f.width + GAP
        y += ROW_H + GAP
    canvas.save(out, quality=85)


def main():
    res = json.load(open(sys.argv[1]))
    footage, out = Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    for key, r in res["edls"].items():
        if r and r.get("edl"):
            board(r["edl"], footage, out / f"storyboard_{key}.jpg")
            print(key, flush=True)


if __name__ == "__main__":
    main()
