#!/usr/bin/env bash
# Isolates the transformer-core load stage (~31 GiB of block weights) and its
# split between waiting on block I/O and running the INT8 quantize batches.
# Arms are interleaved because page-cache drift dominates single samples.
set -u
MODEL=${MODEL:-$HOME/HF-MODELS/MiniMax-H3}
PAIRS=${PAIRS:-2}
# An arm is "preadN" to sweep per-tensor read streams, or "pinned" to put the
# weights back in page-locked host memory. Read concurrency and the allocator
# interact: extra streams only pay off once the loader threads have stopped
# spending their wall inside hipHostMalloc.
ARMS=${ARMS:-"pread8 pread16 pread32"}

for pair in $(seq 1 "$PAIRS"); do
    for arm in $ARMS; do
        extra=()
        case "$arm" in
            pread*) extra=(H3_PREAD_THREADS="${arm#pread}");;
            pinned) extra=(H3_DEVICE_WEIGHTS=0);;
        esac
        out=$(env "${extra[@]}" H3_PROFILE=1 \
            ./h3 -d "$MODEL" -p "A red fox walks through fresh snow." \
            --width 512 --height 512 --frames 22 --steps 2 --layers 35 \
            --reuse 1 -o /tmp/h3-profile/bench-core.mp4 2>&1 |
            stdbuf -oL grep -m2 -E 'core split|load stage transformer core')
        printf 'pair=%s arm=%-9s core=%ss io=%ss quantize=%ss\n' \
            "$pair" "$arm" \
            "$(printf '%s\n' "$out" | sed -n 's/.*transformer core *\([0-9.]*\)s.*/\1/p')" \
            "$(printf '%s\n' "$out" | sed -n 's/.*block-io= *\([0-9.]*\)s.*/\1/p')" \
            "$(printf '%s\n' "$out" | sed -n 's/.*quantize= *\([0-9.]*\)s.*/\1/p')"
    done
done
