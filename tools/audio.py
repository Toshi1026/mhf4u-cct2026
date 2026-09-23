#!/usr/bin/env python3
"""Per-clip audio levels for sound design: loudness, peak, wind rumble.

Usage: python3 audio.py <footage_dir> <out_csv>

wind_ratio = energy below 150 Hz / total energy. Handheld iPhone audio on a
windy shore is dominated by rumble (> ~0.5); clean wave ambience sits lower.
"""
import csv
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
SR = 16000


def pcm(path):
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", str(SR),
         "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() not in {".mov", ".mp4"}:
            continue
        x = pcm(p)
        if not len(x):
            rows.append({"file": p.name})
            continue
        rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), 1 / SR)
        low = float(spec[freqs < 150].sum() / (spec.sum() + 1e-12))
        # loudness over 0.5 s windows: how much the level moves (speech/clatter vs steady ambience)
        win = SR // 2
        seg = x[: len(x) // win * win].reshape(-1, win) if len(x) >= win else x[None, :]
        seg_db = 20 * np.log10(np.sqrt((seg ** 2).mean(1)) + 1e-9)
        rows.append({
            "file": p.name,
            "rms_db": round(20 * np.log10(rms), 1),
            "peak_db": round(20 * np.log10(float(np.abs(x).max()) + 1e-12), 1),
            "wind_ratio": round(low, 2),
            "level_range_db": round(float(seg_db.max() - seg_db.min()), 1),
        })
        print(rows[-1], flush=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "rms_db", "peak_db", "wind_ratio", "level_range_db"])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
