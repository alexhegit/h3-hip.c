#!/usr/bin/env bash
# On-demand 15s TR *schedule* bench (not part of the default scoreboard).
# Same knobs as fox-15s.sh (reuse=2) plus H3_TOKEN_REDUCTION_SCHEDULE.
# Default string matches docs/BEST_PRACTICE.md: aggressive early, conservative late.
# Override: H3_TOKEN_REDUCTION_SCHEDULE="0:4:50,10:0:0" ./bench/fox-15s-sched.sh
# Usage: ./bench/fox-15s-sched.sh [/path/to/MiniMax-H3]
set -euo pipefail

MODEL="${1:-${H3_MODEL:-/mnt/doscratch/MiniMax-H3}}"
shift 2>/dev/null || true

export H3_TOKEN_REDUCTION_SCHEDULE="${H3_TOKEN_REDUCTION_SCHEDULE:-0:4:50,10:4:45,20:4:30}"

exec ./h3 --profile -d "$MODEL" \
  -p "15 seconds, 16:9 landscape cinematic. A lone software engineer works late in a dim home office lit only by monitor glow and a desk lamp. Photoreal live-action feel with subtle handheld camera breathing.

[0–3 seconds] Medium shot from behind the desk. Code scrolls on dual monitors; warm red accent light reflects on glass. Ambient: quiet keyboard clicks, soft fan hum, distant city rain.

[3–6 seconds] Slow push-in over the shoulder. On screen, glowing matrix tiles and magenta wavefronts visualize a neural network training. The engineer pauses, sips coffee. Sound: gentle electronic pulse, a single soft notification chime.

[6–9 seconds] Cut to close-up of hands typing, then rack focus to a small window showing a red fox walking through digital snow inside the monitor reflection. Sound: rising synthesized tone, subtle wind.

[9–12 seconds] Smooth lateral move across the desk: terminal windows, GPU metrics, and a grid of video frames assembling on screen. Warm amber grade, volumetric dust in the lamp beam.

[12–15 seconds] Controlled pullback reveals the full workspace at rest. The engineer leans back, satisfied. Sound: clean final impact, room tone fades.

No readable text, no logos, no subtitles. Premium technology documentary aesthetic." \
  --width 864 --height 480 --seconds 15 \
  --steps 20 --layers 45 --reuse 2 --seed 42 \
  -o outputs/fox-15s-sched.mp4 \
  "$@"
