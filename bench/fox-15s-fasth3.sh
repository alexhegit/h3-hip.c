#!/usr/bin/env bash
# Opt-in 15s fox canvas with the dense FastH3 adapter and TAEH3.
# Not a scoreboard preset. FastH3 forces 4 steps, 50 layers, and reuse 1,
# and rejects token reduction, Sol-Attn, and --fbc.
# gfx1151 (2026-10-10): process E2E 1435.79 s, TAEH3 decode 19.475 s,
# peak 29.522 GiB. The same command without --taeh3 was 1706.16 s.
# Usage: ./bench/fox-15s-fasth3.sh [/path/to/MiniMax-H3]
set -euo pipefail

MODEL="${1:-${H3_MODEL:-/mnt/doscratch/MiniMax-H3}}"
shift 2>/dev/null || true

LORA="${H3_FASTH3_LORA:-./FastH3-4-step-LoRA/dense-datafree/adapter_model.safetensors}"
TAE="${H3_TAEH3:-./FastH3-4-step-LoRA/taeh3.safetensors}"

exec ./h3 --profile --fasth3-lora "$LORA" --taeh3 "$TAE" -d "$MODEL" \
  -p "15 seconds, 16:9 landscape cinematic. A lone software engineer works late in a dim home office lit only by monitor glow and a desk lamp. Photoreal live-action feel with subtle handheld camera breathing.

[0–3 seconds] Medium shot from behind the desk. Code scrolls on dual monitors; warm red accent light reflects on glass. Ambient: quiet keyboard clicks, soft fan hum, distant city rain.

[3–6 seconds] Slow push-in over the shoulder. On screen, glowing matrix tiles and magenta wavefronts visualize a neural network training. The engineer pauses, sips coffee. Sound: gentle electronic pulse, a single soft notification chime.

[6–9 seconds] Cut to close-up of hands typing, then rack focus to a small window showing a red fox walking through digital snow inside the monitor reflection. Sound: rising synthesized tone, subtle wind.

[9–12 seconds] Smooth lateral move across the desk: terminal windows, GPU metrics, and a grid of video frames assembling on screen. Warm amber grade, volumetric dust in the lamp beam.

[12–15 seconds] Controlled pullback reveals the full workspace at rest. The engineer leans back, satisfied. Sound: clean final impact, room tone fades.

No readable text, no logos, no subtitles. Premium technology documentary aesthetic." \
  --width 864 --height 480 --seconds 15 \
  --layers 50 --reuse 1 --seed 42 \
  -o outputs/fox-15s-fasth3-taeh3.mp4 \
  "$@"
