#!/usr/bin/env bash
# Download a shared Google Drive folder ("anyone with the link") into footage/.
# Usage: tools/fetch.sh <folder-url-or-id>
set -euo pipefail
cd "$(dirname "$0")/.."
pip install -q gdown imageio-ffmpeg pymediainfo numpy pillow 2>/dev/null
gdown --folder "$1" -O footage/
