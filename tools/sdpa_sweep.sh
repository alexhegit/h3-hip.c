#!/bin/bash
# SDPA config sweep for MI300X
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

MODEL="${H3_MODEL:-/mnt/doscratch/MiniMax-H3}"
PROMPT="A red fox walks through fresh snow."
OUT="/tmp/h3-sdpa-sweep.mp4"

echo "=== SDPA sweep on MI300X (fox-s2, 2 steps) ==="
printf "%-18s %8s %8s %8s\n" "Config" "Denoise" "E2E" "SDPA"
printf "%-18s %8s %8s %8s\n" "------" "-------" "---" "----"

sweep_one() {
  local label="$1" w="$2" b="$3"
  rm -f "$OUT"
  local out
  out=$(H3_SDPA_CDNA_WAVES="$w" H3_SDPA_CDNA_BK="$b" \
    ./h3 --profile -d "$MODEL" \
    -p "$PROMPT" \
    --width 512 --height 512 --frames 22 --steps 2 --layers 35 --reuse 1 \
    -o "$OUT" 2>&1)

  local denoise e2e sdpa
  denoise=$(echo "$out" | grep "Euler denoise" | sed 's/.*wall= *//' | awk '{print $1}')
  e2e=$(echo "$out" | grep "H3 DiT.*total" | head -1 | sed 's/.*wall= *//' | awk '{print $1}')
  sdpa=$(echo "$out" | grep "Euler denoise" | sed 's/.*sdpa= *//' | awk '{print $1}')
  printf "%-18s %7ss %7ss %7ss\n" "$label" "$denoise" "$e2e" "$sdpa"
  rm -f "$OUT"
}

sweep_one "waves=8,bk=32"  8 32
sweep_one "waves=4,bk=32"  4 32
sweep_one "waves=4,bk=64"  4 64
sweep_one "waves=8,bk=64"  8 64
sweep_one "waves=16,bk=32" 16 32
sweep_one "waves=16,bk=64" 16 64
