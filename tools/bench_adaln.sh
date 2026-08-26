#!/usr/bin/env bash
# Isolates the AdaLN precompute stage, which reads 26 GiB of adaln_proj weights.
# Interleaves prefetch depths so page-cache drift hits every arm equally, and
# stops the run as soon as the stage reports.
set -u
MODEL=${MODEL:-$HOME/HF-MODELS/MiniMax-H3}
PAIRS=${PAIRS:-3}
DEPTHS=${DEPTHS:-"0 2"}
PREAD=${PREAD:-16}
# Each arm is "depth:preadThreads". Defaults expand DEPTHS at a single PREAD so
# older invocations keep working; pass ARMS to sweep both at once.
ARMS=${ARMS:-}
if [ -z "$ARMS" ]; then
    for depth in $DEPTHS; do ARMS="$ARMS $depth:$PREAD"; done
fi

for pair in $(seq 1 "$PAIRS"); do
    for arm in $ARMS; do
        depth=${arm%%:*}
        pread=${arm##*:}
        seconds=$(H3_PROFILE=1 H3_ADALN_PREFETCH="$depth" \
            H3_PREAD_THREADS="$pread" \
            ./h3 -d "$MODEL" -p "A red fox walks through fresh snow." \
            --width 512 --height 512 --frames 22 --steps 2 --layers 35 \
            --reuse 1 -o /tmp/h3-profile/bench-adaln.mp4 2>&1 |
            stdbuf -oL grep -m1 'load stage AdaLN' |
            sed -n 's/.* \([0-9.]*\)s.*/\1/p')
        printf 'pair=%s depth=%s pread=%-2s adaln=%ss\n' "$pair" "$depth" \
            "$pread" "${seconds:-?}"
    done
done
