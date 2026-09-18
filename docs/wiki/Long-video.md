# Long T2VA (10 s and 15 s)

MiniMax-H3 on h3-hip.c supports up to **362 aligned frames** (~15 s at 24 fps).
Long clips are dominated by **DiT denoise wall time**, not the short fox
presets. Short fox-fast on MI300X is warm **process E2E 9.8 s**. 15 s
no-TR process E2E is **199 s** on v0.13.0 (v0.12.0 was **213 s**; denoise
**176 s**). Same knobs on Strix Halo (gfx1151), MI210 (gfx90a), and MI300X
(gfx942): `--steps 20 --layers 45 --reuse 2`
(11 DiT evaluations), 864×480, T2VA, no references.

Build with `HIP_ARCH=gfx1151`, `HIP_ARCH=gfx90a`, or `HIP_ARCH=gfx942`
([Getting started](Getting-started.md)). Why 15 s is SDPA-bound, and what
TR / Sol-Attn / INT8 actually change: [Optimizations](Optimizations.md).

## Showcase clips

| Clip | Product | Duration | E2E wall | Denoise wall | File |
|------|---------|----------|----------|--------------|------|
| 10 s cinematic office | Strix Halo (gfx1151) | 10.1 s (243 f) | **24.7 min** | 21.2 min | [`long-10s-cinematic.mp4`](../../assets/showcase/long-10s-cinematic.mp4) |
| **15 s cinematic office** | Strix Halo (gfx1151) | **15.1 s (362 f)** | **40 min 4 s** | **36 min 7 s** | no TR, 2026-09-12; [`gfx1151-2026-09-12-fox-15s-notr.log`](../perf-runs/gfx1151-2026-09-12-fox-15s-notr.log) |
| **15 s all-opts** | Strix Halo (gfx1151) | **15.1 s (362 f)** | **27 min 3 s** | **23 min 10 s** | sampler+TR+INT8 VAE; v0.11.0 |
| **15 s `fox-15s.sh`** | Strix Halo (gfx1151) | **15.1 s (362 f)** | **26 min 56 s** | **22 min 53 s** | reuse=2+TR; [`gfx1151-2026-09-12-fox-15s.log`](../perf-runs/gfx1151-2026-09-12-fox-15s.log) |
| **15 s-fast** | Strix Halo (gfx1151) | **15.1 s (362 f)** | **20 min 44 s** | **16 min 28 s** | reuse=3+TR; [`gfx1151-2026-09-12-fox-15s-fast.log`](../perf-runs/gfx1151-2026-09-12-fox-15s-fast.log) |
| **15 s three-way** | Strix Halo (gfx1151) | **15.1 s (362 f)** | dense 40:57 / TR 27:46 / Sol 29:47 | — | labelled triptych; [`fox-15s-3way-compare-gfx1151.mp4`](../../assets/showcase/fox-15s-3way-compare-gfx1151.mp4) |
| **15 s cinematic office** | MI210 (gfx90a) | **15.1 s (362 f)** | **12 min 12 s** | **10 min 47 s** | no TR, 2026-09-12; [`gfx90a-2026-09-12-fox-15s-notr.log`](../perf-runs/gfx90a-2026-09-12-fox-15s-notr.log) |
| **15 s `fox-15s.sh`** | MI210 (gfx90a) | **15.1 s (362 f)** | **8 min 13 s** | **6 min 48 s** | reuse=2+TR; [`gfx90a-2026-09-12-fox-15s.log`](../perf-runs/gfx90a-2026-09-12-fox-15s.log) |
| **15 s-fast** | MI210 (gfx90a) | **15.1 s (362 f)** | **6 min 18 s** | **4 min 54 s** | reuse=3+TR; [`gfx90a-2026-09-12-fox-15s-fast.log`](../perf-runs/gfx90a-2026-09-12-fox-15s-fast.log) |
| **15 s cinematic office** | MI300X (gfx942) | **15.1 s (362 f)** | **199 s** | **176 s** | no TR, v0.13 quality SDPA; v0.12.0 process E2E was 213 s |
| **15 s `fox-15s.sh`** | MI300X (gfx942) | **15.1 s (362 f)** | **149 s** | **111.0 s** | reuse=2+TR; process E2E 2026-09-18 |
| **15 s-fast** | MI300X (gfx942) | **15.1 s (362 f)** | **118 s** | **80.1 s** | reuse=3+TR; process E2E 2026-09-18 |
| **15 s 1344×768 dense A/B** | MI300X (gfx942) | **15.1 s (362 f)** | **1027 → 947 s (−7.8%)** | **912 → 833 s** | max canvas, no TR; [`MI300X_2026-09-21_1344x768-15s-sdpa.md`](../perf-runs/MI300X_2026-09-21_1344x768-15s-sdpa.md) |
| **15 s 1344×768 TR / Sol-Attn** | MI300X (gfx942) | **15.1 s (362 f)** | dense **947 s** / TR **612 s** / Sol **719 s** | 833 / 497 / 605 s | not stacked; [`MI300X_2026-09-21_1344x768-15s-tr-sol.md`](../perf-runs/MI300X_2026-09-21_1344x768-15s-tr-sol.md) |

Posters: `assets/showcase/long-*-cinematic.jpg`  
Halo phase splits: [`docs/perf-runs/LONG_VIDEO.md`](../perf-runs/LONG_VIDEO.md)  
MI210 2026-09-12: [`docs/perf-runs/MI210_2026-09-12.md`](../perf-runs/MI210_2026-09-12.md)
MI300X 2026-09-12: [`docs/perf-runs/MI300X_2026-09-12.md`](../perf-runs/MI300X_2026-09-12.md)
MI300X 2026-09-18 process E2E: [`docs/perf-runs/MI300X_2026-09-18_full.md`](../perf-runs/MI300X_2026-09-18_full.md)
MI300X 2026-09-21 1344×768 15 s SDPA A/B: [`docs/perf-runs/MI300X_2026-09-21_1344x768-15s-sdpa.md`](../perf-runs/MI300X_2026-09-21_1344x768-15s-sdpa.md)
MI300X 2026-09-21 1344×768 15 s TR / Sol-Attn: [`docs/perf-runs/MI300X_2026-09-21_1344x768-15s-tr-sol.md`](../perf-runs/MI300X_2026-09-21_1344x768-15s-tr-sol.md)

10 s was not re-timed on MI210. The gallery MP4 is the **quality path** (no
TR). Strix Halo (gfx1151) quality path on **`main` 2026-09-12**: E2E
**40 min 4 s**, denoise **36 min 7 s**, VAE **171 s** (2×1 @ 480 px), DiT
peak **27.9 GiB**. Log: [`gfx1151-2026-09-12-fox-15s-notr.log`](../perf-runs/gfx1151-2026-09-12-fox-15s-notr.log).
v0.11.0 on the same knobs was 40 min 46 s / 36 min 38 s / 174 s.

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
the quality path. GPU sampler is not required on Strix Halo (gfx1151).

`--sol-attn` is a further **lossy** long-SDPA option (off by default; dense
is the quality path). Strix Halo 15 s no-TR vs dense: E2E **−27%**, SDPA
**−42%**, video **19.55 dB / 0.721**, audio SNR **10.99 dB**. MI210:
E2E **−18%**, SDPA **−29%**; MI300X: E2E **−11%**, SDPA **−33%**. Same-seed
Halo three-way (dense / TR / Sol-Attn, not stacked): TR is faster
(**1666 s**, 18.23 dB / 0.667, audio **3.11 dB**); Sol-Attn is closer to
dense (**1787 s**, 19.55 dB / 0.721, audio **10.99 dB**). Play
[fox-15s-3way-compare-gfx1151.mp4](../../assets/showcase/fox-15s-3way-compare-gfx1151.mp4).
See [`docs/SOL_ATTN.md`](../SOL_ATTN.md).
MI300X **1344×768 · 15 s**: dense **947 s**, TR **612 s** (16.9 dB),
Sol-Attn **719 s** (17.7 dB); Sol-Attn needs the >64k-token path.

| | quality path | **`--token-reduction`** |
|--|--:|--:|
| Strix Halo (gfx1151) 15 s E2E | **40 min 4 s** (no TR, 2026-09-12) | **26 min 56 s** (`fox-15s.sh`) |
| MI210 (gfx90a) 15 s E2E | **12 min 12 s** (no TR, 2026-09-12) | **8 min 13 s** (`fox-15s.sh`) |
| MI300X (gfx942) 15 s E2E | **199 s** (no TR, v0.13) | **149 s** (`fox-15s.sh`) |

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
