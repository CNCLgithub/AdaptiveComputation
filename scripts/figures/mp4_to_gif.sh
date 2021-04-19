#!/usr/bin/env bash
set -euo pipefail

VIDEO="$1"
echo "$VIDEO"
ffmpeg -v warning -i "$VIDEO" -vf "scale=450:-1:flags=lanczos,palettegen" -y "${VIDEO%.mp4}.png"
ffmpeg -v warning -i "$VIDEO" -i "${VIDEO%.mp4}.png" -lavfi "scale=450:-1:flags=lanczos [x]; [x][1:v] paletteuse" -y "${VIDEO%.mp4}.gif"
