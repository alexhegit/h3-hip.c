#!/usr/bin/env bash
# A/B for device-resident weights (H3_DEVICE_WEIGHTS): does allocating weight
# buffers in the VRAM carveout beat page-locking them in the 31 GiB host pool?
# Reports every phase wall plus the load stages, because the win is spread over
# all four phases rather than concentrated in one. Arms are interleaved so
# page-cache drift, which moves a single end-to-end sample by 15%, hits both.
set -u
MODEL=${MODEL:-$HOME/HF-MODELS/MiniMax-H3}
PAIRS=${PAIRS:-3}
ARMS=${ARMS:-"device pinned"}
OUT=${OUT:-/tmp/h3-profile/bench-device.mp4}

field() { printf '%s\n' "$2" | sed -n "s/.*$1 *\([0-9.]*\)s.*/\1/p" | head -1; }

for pair in $(seq 1 "$PAIRS"); do
    for arm in $ARMS; do
        extra=(H3_PROFILE=1)
        [ "$arm" = pinned ] && extra+=(H3_DEVICE_WEIGHTS=0)
        log=$(env "${extra[@]}" \
            ./h3 -d "$MODEL" -p "A red fox walks through fresh snow." \
            --width 512 --height 512 --frames 22 --steps 2 --layers 35 \
            --reuse 1 -o "$OUT" 2>&1)
        text=$(printf '%s\n' "$log" | grep -m1 'Qwen text encoder .*total')
        dit=$(printf '%s\n' "$log" | grep -m1 'H3 DiT  *total')
        vae=$(printf '%s\n' "$log" | grep -m1 'video VAE decoder .*total')
        ditpin=$(printf '%s\n' "$log" | grep -m1 'H3 DiT .*weight-load')
        adaln=$(printf '%s\n' "$log" | grep -m1 'stage AdaLN precompute')
        core=$(printf '%s\n' "$log" | grep -m1 'stage transformer core')
        printf 'pair=%s arm=%-7s text=%-7s dit=%-7s vae=%-7s adaln=%-7s core=%-7s ditpin=%-7s sum=%s md5=%s\n' \
            "$pair" "$arm" \
            "$(field 'wall=' "$text")" "$(field 'wall=' "$dit")" \
            "$(field 'wall=' "$vae")" \
            "$(printf '%s\n' "$adaln" | sed -n 's/.*precompute *\([0-9.]*\)s.*/\1/p')" \
            "$(printf '%s\n' "$core" | sed -n 's/.*transformer core *\([0-9.]*\)s.*/\1/p')" \
            "$(field 'pin=' "$ditpin")" \
            "$(printf '%s\n' "$log" | sed -n 's/.*wall= *\([0-9.]*\)s.*/\1/p' |
               awk '{t+=$1} END {printf "%.1f", t}')" \
            "$(md5sum "$OUT" | cut -c1-8)"
    done
done
