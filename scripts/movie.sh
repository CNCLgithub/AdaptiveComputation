#!/usr/bin/env bash
set -euo pipefail
########################################
# Goes through every folder in `WRKDIR`
# and uses `ffmpeg` to generate moveies
########################################
WRKDIR="${1:-$PWD}"

find $WRKDIR -type d -exec \
    ffmpeg -y -framerate 24 -i '{}/%03d.png' \
    -hide_banner -crf 5 -preset slow \
    -c:v libx264  -pix_fmt yuv420p '{}'.mp4 \;
