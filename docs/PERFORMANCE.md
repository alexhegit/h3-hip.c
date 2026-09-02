# Performance

Headline numbers. Three HIP ISAs share this tree; build with an explicit
`HIP_ARCH` ([Getting started](wiki/Getting-started.md)). Wall time moves with
page-cache state; treat E2E as a band, denoise GPU time as the stable GPU
figure.

Engineering logs (phase tables, rejected experiments) live under
[`perf/`](perf/README.md) and [`perf-mi210/`](perf-mi210/SUMMARY.md) and are
**not** part of the GitHub release body.

## Current release — v0.10.1 (2026-09-01)

One tree, three ISAs. Build with `make HIP_ARCH=gfx1151`, `make HIP_ARCH=gfx90a`,
or `make HIP_ARCH=gfx942`. `h3 --info` prints `h3-hip 0.10.1`.

| Preset | gfx1151 | gfx90a | gfx942 |
|--------|--------:|-------:|-------:|
| fox-s2 E2E | 83–87 s | ~10.5 s | **~16 s** |
| fox-fast E2E | ~95 s | ~18 s | **~12 s** |
| 15 s cinematic E2E | 45.0 min | 12 min 33 s | — |

gfx1151 fox-s2 md5 is unchanged from v0.9.0: `1731f95c4aa582597cf83d57f46b8f9e`.

## gfx1151 — same numbers as v0.9.0 (2026-08-26)

AMD Ryzen AI MAX+ 395 / Radeon 8060S. Build: `make HIP_ARCH=gfx1151`.

| Preset | Command knobs | E2E | Denoise GPU |
|--------|---------------|----:|------------:|
| **fox-s2** | 512² · 22 frames · `--steps 2 --layers 35 --reuse 1` | **83–87 s** | **6.3–6.5 s** |
| **fox-fast** | 512² · 22 frames · `--steps 20 --layers 45 --reuse 2` | **95 s** | **28 s** (11 evals) |
| **15 s cinematic** | 864×480 · 362 f · `--steps 20 --layers 45 --reuse 2` | **45.0 min** | **40.4 min** denoise wall |

fox-s2 and fox-fast are complete T2VA MP4s (~0.9 s of picture+sound at 24 fps),
not truncated previews. fox-fast matches upstream’s “first fast video”
(`--reuse 2` → 11 DiT evals). The README fox showcase clip is `--layers 50
--reuse 1` and is **not** the fox-fast scoreboard. 15 s cinematic uses the
fox-fast quality knobs at long duration.

fox-s2 output md5 `1731f95c4aa582597cf83d57f46b8f9e` on this tree.

```bash
# fox-s2 (short smoke used for HIP A/B)
./h3 --profile -d /path/to/MiniMax-H3 \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 --steps 2 --layers 35 --reuse 1 \
  -o outputs/fox-s2.mp4

# fox-fast (upstream tutorial “first fast video”, without --show)
./h3 --profile -d /path/to/MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 --frames 22 --steps 20 --layers 45 --reuse 2 \
  -o outputs/fox-fast.mp4
```

First invocation pays weight load from disk (~107 GiB on the T2VA path). Repeat
runs still miss the page cache: host RAM on this box is ~31 GiB.

## What changed (high level)

| When | fox-s2 E2E | fox-fast E2E | What landed |
|------|-----------:|-------------:|-------------|
| 2026-08-18 | ~257 s | — | First HIP profile; denoise and VAE were GPU-bound |
| 2026-08-19 | ~117 s | — | INT8 DiT, faster load I/O |
| 2026-08-22 | — | ~213 s | fox-fast measured; denoise still ~105 s |
| **v0.9.0** | **83–87 s** | **95 s** | WMMA attention/linear/conv; weights in the VRAM carveout; AdaLN and VAE load overlap |

The remaining E2E on Halo is mostly NVMe weight I/O. Denoise is a small slice of
fox-s2 and about a third of fox-fast.

Upstream [antirez/h3.c](https://github.com/antirez/h3.c) publishes **denoise
wall** on an M5 Max, not T2VA end-to-end. On the same fox-fast knobs that
figure is 16.7 s; HIP denoise wall here is 35 s. That ratio mixes two GPUs and
two memory systems and is not a port-quality score.

## gfx90a / MI210 — v0.10.0

Same MiniMax-H3 checkpoint. Build: `make HIP_ARCH=gfx90a`. Default DiT is
**BF16 hipBLAS** (not INT8). Measured on a four-GPU MI210 box (`H3_HIP_DEVICE=1`
for the fox gates below).

| Preset | Command knobs | E2E | Notes |
|--------|---------------|----:|-------|
| **fox-s2** | 512² · 22 f · `--steps 2 --layers 35 --reuse 1` | **~10.5 s** | 2026-09-01 at `6fe5c0d` |
| **fox-fast** | 512² · 22 f · `--steps 20 --layers 45 --reuse 2` | **~18 s** | denoise ~8.2 s (linear 6.6 · sdpa 1.08) |
| **15 s cinematic** | 864×480 · 362 f · `--steps 20 --layers 45 --reuse 2` | **12 min 33 s** | denoise 10 min 48 s; vs 45 min on gfx1151 |

Commands are the same as the gfx1151 block above. 15 s reproduce:
[`wiki/Long-video.md`](wiki/Long-video.md). Session log:
[`perf-mi210/SUMMARY.md`](perf-mi210/SUMMARY.md).

Long T2VA on MI210 is still DiT-SDPA bound (~63% of the 15 s E2E). fox-s2 is
mostly weight I/O.

## gfx942 / MI300X — v0.10.1 (2026-09-02)

Same MiniMax-H3 checkpoint. Build: `make HIP_ARCH=gfx942`. Default DiT is
**BF16 hipBLAS** (same as gfx90a). 192 GiB VRAM; weight I/O dominates E2E on
short presets.

| Preset | Command knobs | E2E | Denoise GPU | Peak VRAM | Notes |
|--------|---------------|----:|------------:|----------:|-------|
| **fox-s2** | 512² · 22 f · `--steps 2 --layers 35 --reuse 1` | **~16 s** | **0.92 s** | **25.7 GiB** | weight I/O ~10.5 s |
| **fox-fast** | 512² · 22 f · `--steps 20 --layers 45 --reuse 2` | **~12 s** | **2.89 s** | **25.4 GiB** | 11 DiT evals |
| **15 s cinematic** | 864×480 · 362 f · `--steps 20 --layers 45 --reuse 2` | **3 min 46 s** | **3 min 39 s** | **25.7 GiB** | denoise 77% of E2E |
| fox-s2 INT8 | same knobs + `H3_INT8_MLP=1` | **~14 s** | **0.34 s** | **~79 GiB** | INT8 opt-in |
| fox-fast INT8 | same knobs + `H3_INT8_MLP=1` | **~12 s** | **1.86 s** | **19.7 GiB** | 33% faster than BF16 denoise |

Commands are the same as the gfx1151 block above. INT8 is opt-in on CDNA
(`H3_INT8_MLP=1`); default is BF16 GEMM. MI300X denoise is ~3x faster than
MI210 on the same BF16 path. fox-fast E2E is I/O-bound on both CDNA cards.

MI300X profile breakdown (15 s cinematic, BF16):
- **SDPA (flash MFMA)**: 144.5 s — **77%** of denoise
- **Linear GEMM (hipBLAS)**: 34.4 s — 18%
- **Other (norms, activations,)**: 7 s — 4%

INT8 encode overhead eliminated: pre-allocated workspaces removed 2497 GiB
of per-block allocation churn. Denoise encode time dropped from 179 s to
0.06 s. INT8 denoise GPU time (1.86 s) is now 33% faster than BF16 (2.86 s).

**GPU Euler sampler** (`H3_GPU_SAMPLER=1`): keeps latents on GPU across
Euler steps, eliminating CPU-GPU round-trips. fox-s2 denoise: 1.95 s → 0.84 s
(57% faster). fox-fast denoise: 2.89 s → 2.76 s (4.5% faster). Enable with
`H3_GPU_SAMPLER=1`.

## How a GitHub Release should quote this

Paste the **current** summary table (all three ISAs). Do not paste phase splits,
KEEP/REJECT lists, or Metal ratio tables into the release body.
