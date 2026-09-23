#!/usr/bin/env python3
"""Analyze iPhone footage for the MAZE WIND edit.

Usage: python3 analyze.py <footage_dir> <out_dir>

Writes <out_dir>/report.md, <out_dir>/report.csv and a contact sheet per clip
(<out_dir>/sheets/<name>.jpg) for visual review.
"""
import csv
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from pymediainfo import MediaInfo

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VIDEO_EXT = {".mov", ".mp4", ".m4v", ".hevc"}
IMAGE_EXT = {".jpg", ".jpeg", ".heic", ".png", ".tif", ".tiff", ".dng"}

# Analysis frames: small grayscale frames are enough for exposure and shake.
AW, AH = 160, 284  # 9:16 thumbnail; landscape clips get letterboxed by scale+pad
SAMPLE_FPS = 10


def media_info(path):
    info = MediaInfo.parse(str(path))
    g = next((t for t in info.tracks if t.track_type == "General"), None)
    v = next((t for t in info.tracks if t.track_type == "Video"), None)
    a = next((t for t in info.tracks if t.track_type == "Audio"), None)
    d = {"file": path.name, "size_mb": round(path.stat().st_size / 1e6, 1)}
    if g:
        d["duration_s"] = round(float(g.duration or 0) / 1000, 2)
        d["recorded"] = g.encoded_date or g.recorded_date or g.tagged_date or ""
        d["device"] = (g.comapplequicktimemodel or g.performer or "") if hasattr(g, "comapplequicktimemodel") else ""
    if v:
        w, h = int(v.width or 0), int(v.height or 0)
        rot = float(v.rotation or 0)
        if int(rot) % 180 == 90:
            w, h = h, w
        d.update(
            codec=v.format or "",
            width=w,
            height=h,
            orientation="vertical" if h > w else "horizontal",
            fps=round(float(v.frame_rate or v.original_frame_rate or 0), 2),
            fps_mode=v.frame_rate_mode or "",
            bit_depth=v.bit_depth or "",
            transfer=v.transfer_characteristics or "",
            primaries=v.color_primaries or "",
            hdr_format=v.hdr_format or "",
            bitrate_mbps=round(float(v.bit_rate or 0) / 1e6, 1),
        )
        t = (d["transfer"] or "").upper()
        d["dynamic_range"] = "HDR" if ("HLG" in t or "PQ" in t or d["hdr_format"]) else "SDR"
    if a:
        d["audio"] = f"{a.format} {a.channel_s}ch {a.sampling_rate}Hz"
    else:
        d["audio"] = "none"
    return d


def gray_frames(path, fps=SAMPLE_FPS):
    vf = (
        f"fps={fps},scale={AW}:{AH}:force_original_aspect_ratio=decrease,"
        f"pad={AW}:{AH}:(ow-iw)/2:(oh-ih)/2,format=gray"
    )
    cmd = [FFMPEG, "-v", "error", "-i", str(path), "-vf", vf, "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (AW * AH)
    return np.frombuffer(raw[: n * AW * AH], np.uint8).reshape(n, AH, AW).astype(np.float32)


def shift(a, b):
    """Global translation between two frames via phase correlation."""
    fa, fb = np.fft.fft2(a - a.mean()), np.fft.fft2(b - b.mean())
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-6
    c = np.abs(np.fft.ifft2(r))
    iy, ix = np.unravel_index(np.argmax(c), c.shape)

    def sub(m, p, z):  # parabolic sub-pixel refinement around the peak
        d = m - 2 * p + z
        return 0.0 if d == 0 else 0.5 * (m - z) / d

    y = iy + sub(c[(iy - 1) % AH, ix], c[iy, ix], c[(iy + 1) % AH, ix])
    x = ix + sub(c[iy, (ix - 1) % AW], c[iy, ix], c[iy, (ix + 1) % AW])
    if y > AH // 2:
        y -= AH
    if x > AW // 2:
        x -= AW
    return x, y


def quality(path):
    f = gray_frames(path)
    if len(f) < 2:
        return {}
    win = np.hanning(AH)[:, None] * np.hanning(AW)[None, :]
    moves = np.array([shift(f[i] * win, f[i + 1] * win) for i in range(len(f) - 1)], float)
    path_xy = np.cumsum(moves, axis=0)
    k = max(3, SAMPLE_FPS)  # 1 s moving average = intended camera move
    kern = np.ones(k) / k
    pad = np.pad(path_xy, ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(pad[:, i], kern, mode="valid") for i in range(2)], 1)
    jitter = float(np.sqrt(((path_xy - smooth) ** 2).sum(1)).mean())
    travel = float(np.abs(moves).sum(0).sum())
    return {
        "luma_mean": round(float(f.mean()), 1),
        "clip_hi_pct": round(float((f >= 250).mean() * 100), 2),
        "crush_lo_pct": round(float((f <= 5).mean() * 100), 2),
        "shake_px": round(jitter, 2),  # at 160px width; ~0.6 smooth move, >0.9 handheld jitter
        "camera_travel_px": round(travel, 1),
    }


def contact_sheet(path, out, duration):
    n = 6
    step = max(duration / (n + 1), 0.1)
    vf = f"fps=1/{step:.3f},scale=360:-2,tile={n}x1:padding=4"
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-ss", f"{step/2:.2f}", "-i", str(path),
         "-vf", vf, "-frames:v", "1", "-q:v", "4", str(out)],
        check=False,
    )


def verdict(d):
    notes = []
    if d.get("fps", 0) >= 100:
        notes.append("slow-mo capable")
    elif d.get("fps", 0) and d["fps"] < 50:
        notes.append("≤30fps: no slow-mo")
    if d.get("dynamic_range") == "HDR":
        notes.append("HDR→SDR needed")
    if d.get("clip_hi_pct", 0) > 3:
        notes.append("highlight clipping")
    if d.get("crush_lo_pct", 0) > 10:
        notes.append("crushed shadows")
    if d.get("shake_px", 0) > 0.9:
        notes.append("shaky")
    if d.get("orientation") == "horizontal":
        notes.append("horizontal: 9:16 crop")
    return "; ".join(notes)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    rows = []
    files = sorted(p for p in src.rglob("*") if p.is_file())
    for p in files:
        ext = p.suffix.lower()
        if ext in VIDEO_EXT:
            d = media_info(p)
            try:
                d.update(quality(p))
            except subprocess.CalledProcessError as e:
                d["error"] = e.stderr.decode(errors="ignore")[-200:]
            contact_sheet(p, out / "sheets" / f"{p.stem}.jpg", d.get("duration_s", 1))
            d["type"] = "video"
        elif ext in IMAGE_EXT:
            d = {"file": p.name, "type": "image", "size_mb": round(p.stat().st_size / 1e6, 1)}
        else:
            continue
        d["path"] = str(p.relative_to(src))
        d["flags"] = verdict(d)
        rows.append(d)
        print(p.name, d.get("flags", ""), flush=True)

    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out / "report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    cols = ["path", "duration_s", "width", "height", "fps", "dynamic_range", "codec",
            "audio", "luma_mean", "clip_hi_pct", "shake_px", "flags"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} files -> {out}")


if __name__ == "__main__":
    main()
