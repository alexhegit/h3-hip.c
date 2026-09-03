# Long T2VA (10 s and 15 s)

MiniMax-H3 on h3-hip.c supports up to **362 aligned frames** (~15 s at 24 fps).
Long clips are dominated by **DiT denoise wall time**, not the short fox
presets. Same knobs on both ISAs: `--steps 20 --layers 45 --reuse 2`
(11 DiT evaluations), 864×480, T2VA, no references.

Build with `HIP_ARCH=gfx1151`, `HIP_ARCH=gfx90a`, or `HIP_ARCH=gfx942`
([Getting started](Getting-started.md)).

## Showcase clips

| Clip | GPU | Duration | E2E wall | Denoise wall | File |
|------|-----|----------|----------|--------------|------|
| 10 s cinematic office | gfx1151 | 10.1 s (243 f) | **24.7 min** | 21.2 min | [`long-10s-cinematic.mp4`](../../assets/showcase/long-10s-cinematic.mp4) |
| **15 s cinematic office** | gfx1151 | **15.1 s (362 f)** | **40 min 46 s** | **36 min 38 s** | [`long-15s-cinematic.mp4`](../../assets/showcase/long-15s-cinematic.mp4) |
| **15 s all-opts** | gfx1151 | **15.1 s (362 f)** | **27 min 3 s** | **23 min 10 s** | sampler+TR+INT8 VAE; [`long-15s-all-opts-2026-09-03.log`](../perf-runs/long-15s-all-opts-2026-09-03.log) |
| **15 s cinematic office** | gfx90a | **15.1 s (362 f)** | **12 min 11 s** | **10 min 47 s** | `main` 2026-09-02; 480 px VAE tiles |
| **15 s + `--token-reduction`** | gfx90a | **15.1 s (362 f)** | **8 min 21 s** | **6 min 50 s** | opt-in; [`TOKEN_REDUCTION.md`](../perf-mi210/TOKEN_REDUCTION.md) |

Posters: `assets/showcase/long-*-cinematic.jpg`  
Halo phase splits: [`docs/perf-runs/LONG_VIDEO.md`](../perf-runs/LONG_VIDEO.md)  
MI210 session: [`docs/perf-mi210/STATUS.md`](../perf-mi210/STATUS.md)

10 s was not re-timed on MI210 or on `main` gfx1151. The gallery MP4 is the
**quality path** (no TR). gfx1151 quality path on `main` (2026-09-03): E2E
**40 min 46 s**, denoise **36 min 38 s**, VAE **174 s** (2×1 @ 480 px), DiT
peak **27.9 GiB**. Log: [`long-15s-default-2026-09-03.log`](../perf-runs/long-15s-default-2026-09-03.log).

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

Add `--token-reduction` and `H3_INT8_VAE=1` when wall clock matters more than
the quality path. GPU sampler is not required on gfx1151.

| | quality path | **`--token-reduction`** |
|--|--:|--:|
| gfx1151 15 s E2E | **40 min 46 s** (`main` 2026-09-03) | **27 min 3 s** (all-opts) |
| gfx90a 15 s E2E | 12 min 11 s | **8 min 21 s** (denoise 6 min 56 s all-opts) |

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
