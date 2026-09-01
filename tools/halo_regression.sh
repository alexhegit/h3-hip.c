#!/usr/bin/env bash
# gfx1151 (Strix Halo) regression gate for the mi210 dual-arch branch.
#
# Runs the default unit suite, real-weight functional smokes, and the fox-s2
# bit-identical E2E check against the v0.9.0 md5. Use on Halo before merging
# mi210 changes that must not regress RDNA.
#
# Usage:
#   ./tools/halo_regression.sh
#   HIP_ARCH=gfx1151 H3_MODEL=/path/to/MiniMax-H3 ./tools/halo_regression.sh
#   ./tools/halo_regression.sh --strict    # also run hip-test-strict (known red)
#   ./tools/halo_regression.sh --clean     # make clean before build
#   ./tools/halo_regression.sh --skip-e2e  # unit + functional only
#
# Equivalent: make HIP_ARCH=gfx1151 halo-regression
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

MODEL="${H3_MODEL:-/home/amd/HF-MODELS/MiniMax-H3}"
ARCH="${HIP_ARCH:-gfx1151}"
FOX_MD5="${H3_FOX_S2_MD5:-1731f95c4aa582597cf83d57f46b8f9e}"
OUT="${H3_HALO_REGRESSION_OUT:-/tmp/h3-halo-regression-fox-s2.mp4}"
JOBS="${JOBS:-$(nproc)}"

DO_CLEAN=0
DO_STRICT=0
SKIP_E2E=0
for arg in "$@"; do
    case "$arg" in
        --clean) DO_CLEAN=1 ;;
        --strict) DO_STRICT=1 ;;
        --skip-e2e) SKIP_E2E=1 ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *)
            echo "unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

export H3_MODEL="$MODEL"
export HIP_ARCH="$ARCH"

echo "=== h3-hip halo regression ==="
echo "root=$ROOT arch=$ARCH model=$MODEL"

if [ "$DO_CLEAN" -eq 1 ]; then
    echo "=== make clean ==="
    make clean
fi

echo "=== build + unit + functional ==="
make -j"$JOBS" HIP_ARCH="$ARCH" h3 hip-test hip-functional

if [ "$DO_STRICT" -eq 1 ]; then
    echo "=== hip-test-strict (informational; attn.out harness is known red) ==="
    if ! make HIP_ARCH="$ARCH" hip-test-strict; then
        echo "note: hip-test-strict failed (expected until real_dit harness is fixed)" >&2
    fi
fi

if [ "$SKIP_E2E" -eq 1 ]; then
    echo "skip: fox-s2 E2E (--skip-e2e)"
    echo "PASS halo regression (unit + functional)"
    exit 0
fi

echo "=== fox-s2 E2E (md5 gate) ==="
/usr/bin/time -f 'E2E %e s' ./h3 --profile \
    -d "$MODEL" \
    -p "A red fox walks through fresh snow." \
    --width 512 --height 512 --frames 22 \
    --steps 2 --layers 35 --reuse 1 \
    -o "$OUT"

got=$(md5sum "$OUT" | awk '{print $1}')
if [ "$got" = "$FOX_MD5" ]; then
    echo "PASS fox-s2 md5 $got"
else
    echo "FAIL fox-s2 md5 got=$got want=$FOX_MD5" >&2
    exit 1
fi

echo "PASS halo regression"
