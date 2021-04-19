#!/usr/bin/env bash
set -euo pipefail

SRCF="$1"
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ffmpeg -r 24 -f image2  -i "$SRCF"/%001d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p "${SRCF}.mp4"
echo  "${SCRIPTPATH}/mp4_to_gif.sh"
exec "${SCRIPTPATH}/mp4_to_gif.sh" "${SRCF}.mp4"
