#!/usr/bin/env python3
"""Timestamped contact sheets for picking In/Out points by eye.

Usage: python3 sheets.py <footage_dir> <out_dir> [frames_per_clip]

For every video: <out_dir>/<name>.jpg, a grid of evenly spaced frames, each
labelled with its timestamp. HDR (HLG/PQ) clips are tone-mapped to SDR so the
previews show roughly what the final SDR grade will look like.
Photos (HEIC/JPG) get a JPEG preview: <out_dir>/<name>.jpg.
"""
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from pymediainfo import MediaInfo

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VIDEO_EXT = {".mov", ".mp4", ".m4v"}
IMAGE_EXT = {".heic", ".jpg", ".jpeg", ".png"}
TILE = 360  # long side of each tile in px
TONEMAP = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
    "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)


def probe(path):
    info = MediaInfo.parse(str(path))
    g = next(t for t in info.tracks if t.track_type == "General")
    v = next((t for t in info.tracks if t.track_type == "Video"), None)
    transfer = (v.transfer_characteristics or "").upper() if v else ""
    hdr = "HLG" in transfer or "PQ" in transfer or bool(v and v.hdr_format)
    return float(g.duration or 0) / 1000, hdr


def grab(path, t, hdr):
    # Downscale first: tone-mapping a full 4K frame in float is ~10x slower.
    vf = f"scale='if(gt(iw,ih),{TILE},-2)':'if(gt(iw,ih),-2,{TILE})'" + ("," + TONEMAP if hdr else "")
    out = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
         "-vf", vf, "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True,
    )
    if out.returncode or not out.stdout:
        # zscale can fail on unusual inputs; fall back to an un-tonemapped frame
        if hdr:
            return grab(path, t, False)
        return None
    from io import BytesIO
    return Image.open(BytesIO(out.stdout)).convert("RGB")


def label(img, text):
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    x, y = 6, 4
    d.rectangle([x - 4, y - 2, x + 11 * len(text) + 8, y + 26], fill=(0, 0, 0))
    d.text((x, y), text, fill=(255, 255, 0), font=font)
    return img


def sheet(path, out, n):
    dur, hdr = probe(path)
    if dur <= 0:
        return
    n = max(4, min(n, int(dur * 2)))  # at most 2 frames per second
    ts = [dur * (i + 0.5) / n for i in range(n)]
    tiles = [(t, grab(path, t, hdr)) for t in ts]
    tiles = [(t, im) for t, im in tiles if im]
    if not tiles:
        return
    w, h = tiles[0][1].size
    cols = 6 if w > h else 8
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w + (cols - 1) * 4, rows * h + (rows - 1) * 4 + 40), (20, 20, 20))
    head = ImageDraw.Draw(canvas)
    try:
        hfont = ImageFont.load_default(size=26)
    except TypeError:
        hfont = ImageFont.load_default()
    head.text((8, 6), f"{path.name}  {dur:.1f}s  {'HDR→SDR preview' if hdr else 'SDR'}  {w}x{h} tiles",
              fill=(255, 255, 255), font=hfont)
    for i, (t, im) in enumerate(tiles):
        im = im.resize((w, h))
        canvas.paste(label(im, f"{t:5.1f}s"), ((i % cols) * (w + 4), 40 + (i // cols) * (h + 4)))
    canvas.save(out, quality=82)


def photo(path, out):
    if path.suffix.lower() == ".heic":
        import pillow_heif
        pillow_heif.register_heif_opener()
    im = Image.open(path).convert("RGB")
    im.thumbnail((1200, 1200))
    im.save(out, quality=85)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.rglob("*")):
        ext = p.suffix.lower()
        if ext in VIDEO_EXT:
            sheet(p, out / f"{p.stem}.jpg", n)
        elif ext in IMAGE_EXT:
            photo(p, out / f"{p.stem}.jpg")
        else:
            continue
        print(p.name, flush=True)


if __name__ == "__main__":
    main()
