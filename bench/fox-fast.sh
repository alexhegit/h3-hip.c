#!/usr/bin/env bash
# fox-fast benchmark (512², 22 frames, 20 steps, 45 layers, reuse 2)
# Upstream tutorial "first fast video" knobs. ~15s on MI300X.
# Usage: ./bench/fox-fast.sh [/path/to/MiniMax-H3] [--profile]
set -euo pipefail

MODEL="${1:-${H3_MODEL:-/mnt/doscratch/MiniMax-H3}}"
shift 2>/dev/null || true

exec ./h3 --profile -d "$MODEL" \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 --frames 22 \
  --steps 20 --layers 45 --reuse 2 \
  -o outputs/fox-fast.mp4 \
  "$@"
