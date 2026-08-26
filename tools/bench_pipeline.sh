#!/usr/bin/env bash
# Runs the fox benchmark and reports per-phase wall time plus the output hash.
# Disk and thermal state on this box move end-to-end wall by roughly ten
# percent, so pass REPEATS=2 or more before trusting a comparison.
set -u
MODEL=${MODEL:-$HOME/HF-MODELS/MiniMax-H3}
REPEATS=${REPEATS:-1}
OUT=${OUT:-/tmp/h3-profile/bench-pipeline.mp4}
LABEL=${LABEL:-run}

for pass in $(seq 1 "$REPEATS"); do
    log=$(mktemp)
    start=$(date +%s.%N)
    H3_PROFILE=1 ./h3 -d "$MODEL" \
        -p "A red fox walks through fresh snow." \
        --width 512 --height 512 --frames 22 --steps 2 --layers 35 \
        --reuse 1 -o "$OUT" >"$log" 2>&1
    status=$?
    end=$(date +%s.%N)
    if [ "$status" -ne 0 ]; then
        echo "$LABEL pass=$pass FAILED (see $log)"
        continue
    fi
    pick() { sed -n "s/.*$1.*wall= *\([0-9.]*\)s.*/\1/p" "$log" | head -1; }
    stage() { sed -n "s/.*load stage $1 *\([0-9.]*\)s.*/\1/p" "$log" | head -1; }
    printf '%s pass=%s total=%.1f text=%s adaln=%s core=%s dit=%s audio=%s video=%s md5=%s\n' \
        "$LABEL" "$pass" "$(echo "$end - $start" | bc)" \
        "$(pick 'Qwen text encoder *total')" \
        "$(stage 'AdaLN precompute')" "$(stage 'transformer core')" \
        "$(pick 'H3 DiT *total')" \
        "$(pick 'audio VAE decoder *total')" \
        "$(pick 'video VAE decoder *total')" \
        "$(md5sum "$OUT" | cut -c1-8)"
    rm -f "$log"
done
