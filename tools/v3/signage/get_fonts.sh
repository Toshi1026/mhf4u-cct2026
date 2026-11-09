#!/usr/bin/env bash
# しっぽり明朝 B1（SIL Open Font License 1.1、Google Fonts）を assets/fonts/ に取ってくる。
# 1 本 14MB と大きいので Git には入れない（.gitignore）。サイネージ v3（build_sg.py）で使う。
set -euo pipefail
cd "$(dirname "$0")/../../../assets/fonts"
css=$(curl -sS -A "Mozilla/4.0" "https://fonts.googleapis.com/css2?family=Shippori+Mincho+B1:wght@500;600")
m=$(echo "$css" | grep -A3 "font-weight: 500" | grep -o 'https://[^)]*\.ttf')
s=$(echo "$css" | grep -A3 "font-weight: 600" | grep -o 'https://[^)]*\.ttf')
[ -f ShipporiMinchoB1-Medium.ttf ] || curl -sSf -o ShipporiMinchoB1-Medium.ttf "$m"
[ -f ShipporiMinchoB1-SemiBold.ttf ] || curl -sSf -o ShipporiMinchoB1-SemiBold.ttf "$s"
ls -la ShipporiMinchoB1-*.ttf
