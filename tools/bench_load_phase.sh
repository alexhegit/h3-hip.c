#!/usr/bin/env bash
# Times the text-encoder load phase, which is pure weight I/O plus pinning, and
# stops as soon as that phase reports. Repeats each configuration so the disk
# and thermal noise on this box is visible rather than hidden in one sample.
set -u
MODEL=${MODEL:-$HOME/HF-MODELS/MiniMax-H3}
REPEATS=${REPEATS:-2}
# Each configuration is "pinCacheGiB:preadThreads". Configurations are
# interleaved across passes so drift hits every arm equally.
CONFIGS=${CONFIGS:-"0:8 0:16 6:8 6:16"}

for pass in $(seq 1 "$REPEATS"); do
    for config in $CONFIGS; do
        gib=${config%%:*}
        threads=${config##*:}
        line=$(H3_PROFILE=1 H3_PIN_CACHE_GIB="$gib" H3_PREAD_THREADS="$threads" \
            ./h3 -d "$MODEL" -p "A red fox walks through fresh snow." \
            --width 512 --height 512 --frames 22 --steps 2 --layers 35 \
            --reuse 1 -o /tmp/h3-profile/bench-load.mp4 2>&1 |
            stdbuf -oL grep -m2 -E 'Qwen text encoder +(total|weight-load)')
        wall=$(printf '%s\n' "$line" | sed -n 's/.*wall= *\([0-9.]*\)s.*/\1/p')
        pin=$(printf '%s\n' "$line" | sed -n 's/.*pin= *\([0-9.]*\)s.*/\1/p')
        recycled=$(printf '%s\n' "$line" | sed -n 's/.*recycled= *\([0-9.]*\)%.*/\1/p')
        printf 'cache=%-2sGiB pread=%-2s pass=%s wall=%ss pin=%ss recycled=%s%%\n' \
            "$gib" "$threads" "$pass" "${wall:-?}" "${pin:-?}" \
            "${recycled:-?}"
    done
done
