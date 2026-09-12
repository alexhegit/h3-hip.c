#!/usr/bin/env bash
# TR schedule sweep — tests various schedule configs on 15s pipeline
set -euo pipefail

PROMPT="15 seconds, 16:9 landscape cinematic. A lone software engineer works late in a dim home office lit only by monitor glow and a desk lamp. Photoreal live-action feel with subtle handheld camera breathing.

[0-3 seconds] Medium shot from behind the desk. Code scrolls on dual monitors; warm red accent light reflects on glass. Ambient: quiet keyboard clicks, soft fan hum, distant city rain.

[3-6 seconds] Slow push-in over the shoulder. On screen, glowing matrix tiles and magenta wavefronts visualize a neural network training. The engineer pauses, sips coffee. Sound: gentle electronic pulse, a single soft notification chime.

[6-9 seconds] Cut to close-up of hands typing, then rack focus to a small window showing a red fox walking through digital snow inside the monitor reflection. Sound: rising synthesized tone, subtle wind.

[9-12 seconds] Smooth lateral move across the desk: terminal windows, GPU metrics, and a grid of video frames assembling on screen. Warm amber grade, volumetric dust in the lamp beam.

[12-15 seconds] Controlled pullback reveals the full workspace at rest. The engineer leans back, satisfied. Sound: clean final impact, room tone fades.

No readable text, no logos, no subtitles. Premium technology documentary aesthetic."

SWEEP_DIR="/tmp/h3-sweep"
mkdir -p "$SWEEP_DIR"

run_sweep() {
    local name="$1"
    local sched="$2"
    local log="$SWEEP_DIR/${name}.log"
    echo -n "  $name: "
    if [ -n "$sched" ]; then
        H3_TOKEN_REDUCTION_SCHEDULE="$sched" \
        H3_PROFILE=1 ./h3 -d /mnt/doscratch/MiniMax-H3 \
          -p "$PROMPT" \
          --width 864 --height 480 --seconds 15 \
          --steps 20 --layers 45 --reuse 3 --seed 42 --token-reduction \
          -o "$SWEEP_DIR/${name}.mp4" \
          > "$log" 2>&1 || true
    else
        H3_PROFILE=1 ./h3 -d /mnt/doscratch/MiniMax-H3 \
          -p "$PROMPT" \
          --width 864 --height 480 --seconds 15 \
          --steps 20 --layers 45 --reuse 3 --seed 42 --token-reduction \
          -o "$SWEEP_DIR/${name}.mp4" \
          > "$log" 2>&1 || true
    fi
    local dit=$(grep "Euler denoise" "$log" | grep -o 'wall=[[:space:]]*[0-9.]*s' | head -1 | sed 's/wall=[[:space:]]*//' | tr -d 's')
    local sdpa=$(grep "op-classes" "$log" | tail -1 | grep -o 'sdpa=[[:space:]]*[0-9.]*s' | sed 's/sdpa=[[:space:]]*//' | tr -d 's')
    local linear=$(grep "op-classes" "$log" | tail -1 | grep -o 'linear=[[:space:]]*[0-9.]*s' | sed 's/linear=[[:space:]]*//' | tr -d 's')
    local other=$(grep "op-classes" "$log" | tail -1 | grep -o 'other=[[:space:]]*[0-9.]*s' | sed 's/other=[[:space:]]*//' | tr -d 's')
    if [ -z "$dit" ]; then
        echo "FAILED (schedule rejected?)"
    else
        echo "DiT=${dit}s SDPA=${sdpa}s Linear=${linear}s Other=${other}s"
    fi
}

echo "=== TR Schedule Sweep (15s, reuse=3) ==="
echo ""
echo "--- Baseline (no schedule) ---"
run_sweep "baseline" ""

echo ""
echo "--- Previous best ---"
run_sweep "prev_50_45_30" "0:4:50,10:4:45,20:4:30"

echo ""
echo "--- Aggressive early, within 50 blocks ---"
run_sweep "agg_50_45_35" "0:4:50,10:4:45,20:4:35"
run_sweep "agg_50_40_30" "0:4:50,10:4:40,20:4:30"
run_sweep "agg_48_42_30" "0:4:48,10:4:42,20:4:30"

echo ""
echo "--- Finer granularity (all valid <=50) ---"
run_sweep "fine_50_48_45_30" "0:4:50,5:4:48,10:4:45,20:4:30"
run_sweep "fine_50_47_45_40_30" "0:4:50,5:4:47,10:4:45,15:4:40,20:4:30"

echo ""
echo "--- Early aggression ---"
run_sweep "early_50_50_45_30" "0:4:50,3:4:50,10:4:45,20:4:30"
run_sweep "early_48_45_40_30" "0:4:48,5:4:45,10:4:40,20:4:30"

echo ""
echo "--- Smooth gradient ---"
run_sweep "smooth_50_45_40_35_30" "0:4:50,5:4:45,10:4:40,15:4:35,20:4:30"

echo ""
echo "=== Done ==="
