# Long T2VA (10 s and 15 s)

MiniMax-H3 on h3-hip.c supports up to **362 aligned frames** (~15 s at 24 fps).
On gfx1151 (Strix Halo), long clips are dominated by **DiT denoise wall time**, not
the short fox presets.

## Showcase clips

| Clip | Duration | E2E wall | Denoise wall | File |
|------|----------|----------|--------------|------|
| 10 s cinematic office | 10.1 s (243 f) | **24.7 min** | 21.2 min | [`long-10s-cinematic.mp4`](../../assets/showcase/long-10s-cinematic.mp4) |
| **15 s cinematic office** | **15.1 s (362 f)** | **45.0 min** | **40.4 min** | [`long-15s-cinematic.mp4`](../../assets/showcase/long-15s-cinematic.mp4) |

Posters: `assets/showcase/long-*-cinematic.jpg`  
Phase splits and logs: [`docs/perf-runs/LONG_VIDEO.md`](../perf-runs/LONG_VIDEO.md)

Shared knobs (same as fox-fast quality): `--steps 20 --layers 45 --reuse 2`
(11 DiT evaluations), 864×480, T2VA, no references.

## Reproduce the 15 s clip

```bash
MODEL=/path/to/MiniMax-H3

./h3 --profile -d "$MODEL" \
  -p "15 seconds, 16:9 landscape cinematic. A lone software engineer works late in a dim home office lit only by monitor glow and a desk lamp. Photoreal live-action feel with subtle handheld camera breathing.

[0–3 seconds] Medium shot from behind the desk. Code scrolls on dual monitors; warm red accent light reflects on glass. Ambient: quiet keyboard clicks, soft fan hum, distant city rain.

[3–6 seconds] Slow push-in over the shoulder. On screen, glowing matrix tiles and magenta wavefronts visualize a neural network training. The engineer pauses, sips coffee. Sound: gentle electronic pulse, a single soft notification chime.

[6–9 seconds] Cut to close-up of hands typing, then rack focus to a small window showing a red fox walking through digital snow inside the monitor reflection. Sound: rising synthesized tone, subtle wind.

[9–12 seconds] Smooth lateral move across the desk: terminal windows, GPU metrics, and a grid of video frames assembling on screen. Warm amber grade, volumetric dust in the lamp beam.

[12–15 seconds] Controlled pullback reveals the full workspace at rest. The engineer leans back, satisfied. Sound: clean final impact, room tone fades.

No readable text, no logos, no subtitles. Premium technology documentary aesthetic." \
  --width 864 --height 480 --seconds 15 \
  --steps 20 --layers 45 --reuse 2 --seed 42 \
  -o outputs/long-15s-cinematic.mp4
```

## Reproduce the 10 s clip

Same prompt structure with `[0–3]` … `[8–10]` blocks and `--seconds 10`.
See [`docs/perf-runs/LONG_VIDEO.md`](../perf-runs/LONG_VIDEO.md) for the full
command and `--profile` phase table.

## Frame alignment

Output frame count must align to `5 + 17×N` (5 … 362). `--seconds N` picks the
nearest valid count at 24 fps (`--seconds 15` → 362 frames, 15.08 s).

## Project page

Live clips and the reproduce block:
[alexhegit.github.io/h3-hip.c/#long](https://alexhegit.github.io/h3-hip.c/#long)
