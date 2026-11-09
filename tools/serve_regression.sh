#!/usr/bin/env bash
# KI-005 gate: `h3 --serve` must reproduce the CLI fox-s2 bit for bit.
#
# Renders fox-s2 with the CLI, starts a daemon from the same build on a
# private port and output root, and runs tools/h3d_smoke.py --generate
# against it with the CLI md5. The daemon renders fox-s2 twice (cache miss,
# then prepared-DiT cache hit with release_after), so both paths are gated.
# Works on any ISA: the reference is this build's own CLI output.
#
# Usage:
#   HIP_ARCH=gfx1201 ./tools/serve_regression.sh
#   HIP_ARCH=gfx1151 H3_MODEL=/path/to/MiniMax-H3 ./tools/serve_regression.sh
#
# Equivalent: make HIP_ARCH=gfx1201 serve-regression
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

: "${HIP_ARCH:?set HIP_ARCH (gfx1151, gfx1201, gfx90a, gfx942)}"
MODEL="${H3_MODEL:-/home/amd/HF-MODELS/MiniMax-H3}"
PORT="${H3D_REGRESSION_PORT:-8597}"
WORK="${H3_SERVE_REGRESSION_DIR:-/tmp/h3-serve-regression}"
JOBS="${JOBS:-$(nproc)}"
URL="http://127.0.0.1:$PORT"

rm -rf "$WORK"
mkdir -p "$WORK/out" "$WORK/media" "$WORK/log"

echo "=== h3-hip serve regression ==="
echo "root=$ROOT arch=$HIP_ARCH model=$MODEL port=$PORT work=$WORK"

make -j"$JOBS" HIP_ARCH="$HIP_ARCH" h3

echo "=== CLI fox-s2 ==="
./h3 -d "$MODEL" \
    -p "A red fox walks through fresh snow." \
    --width 512 --height 512 --frames 22 \
    --steps 2 --layers 35 --reuse 1 --seed 42 \
    -o "$WORK/cli-fox-s2.mp4" > "$WORK/cli.log" 2>&1
cli_md5=$(md5sum "$WORK/cli-fox-s2.mp4" | awk '{print $1}')
echo "CLI fox-s2 md5 $cli_md5"

echo "=== daemon ==="
H3D_PORT="$PORT" H3D_OUTPUT_ROOT="$WORK/out" H3D_MEDIA_ROOT="$WORK/media" \
    H3D_LOG_DIR="$WORK/log" ./h3 -d "$MODEL" --serve > "$WORK/daemon.log" 2>&1 &
daemon=$!
trap 'kill "$daemon" 2>/dev/null || true; wait "$daemon" 2>/dev/null || true' EXIT

for _ in $(seq 1 300); do
    if ! kill -0 "$daemon" 2>/dev/null; then
        echo "FAIL daemon exited; see $WORK/daemon.log" >&2
        exit 1
    fi
    if curl -sf -H 'X-H3-Protocol: v1alpha' "$URL/v1/info" | grep -q '"ready":true'; then
        break
    fi
    sleep 1
done

./tools/h3d_smoke.py --url "$URL" --generate --fox-md5 "$cli_md5"
echo "PASS serve regression (daemon fox-s2 = CLI $cli_md5)"
