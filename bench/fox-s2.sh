#!/usr/bin/env bash
# fox-s2 benchmark (512², 22 frames, 2 steps, 35 layers, reuse 1)
# Short smoke test for HIP A/B. ~15s on MI300X, ~85-90s on Strix Halo.
# Usage: ./bench/fox-s2.sh [/path/to/MiniMax-H3] [--profile]
set -euo pipefail

MODEL="${1:-${H3_MODEL:-/mnt/doscratch/MiniMax-H3}}"
shift 2>/dev/null || true

exec ./h3 --profile -d "$MODEL" \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 \
  --steps 2 --layers 35 --reuse 1 \
  -o outputs/fox-s2.mp4 \
  "$@"
