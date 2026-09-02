# Performance

Headline numbers. Three HIP ISAs share this tree; build with an explicit
`HIP_ARCH` ([Getting started](wiki/Getting-started.md)). Wall time moves with
page-cache state; treat E2E as a band, denoise GPU time as the stable GPU
figure. Peak VRAM is the high-water mark across the entire pipeline.

Engineering logs (phase tables, rejected experiments) live under
[`perf/`](perf/README.md) and [`perf-mi210/`](perf-mi210/SUMMARY.md) and are
**not** part of the GitHub release body.

## Current release — v0.10.1 (2026-09-01)

One tree, three ISAs. Build with `make HIP_ARCH=gfx1151`, `make HIP_ARCH=gfx90a`,
or `make HIP_ARCH=gfx942`. `h3 --info` prints `h3-hip 0.10.1`.

| Preset | gfx1151 | gfx90a | gfx942 |
|--------|--------:|-------:|-------:|
| fox-s2 E2E | 83–87 s | **~10.8 s** | **~16 s** |
| fox-fast E2E | ~95 s | **~18 s** | **~12 s** |
| 15 s cinematic E2E | 45.0 min | **12 min 11 s** | **~2.5 min** |

gfx1151 fox-s2 md5 is unchanged from v0.9.0: `1731f95c4aa582597cf83d57f46b8f9e`.

## gfx1151 — same numbers as v0.9.0 (2026-08-26)

AMD Ryzen AI MAX+ 395 / Radeon 8060S (64 GiB unified). Build: `make HIP_ARCH=gfx1151`.

| Preset | Command knobs | E2E | Denoise GPU | Peak VRAM |
|--------|---------------|----:|------------:|----------:|
| **fox-s2** | 512² · 22 frames · `--steps 2 --layers 35 --reuse 1` | **83–87 s** | **6.3–6.5 s** | **~48 GiB** |
| **fox-fast** | 512² · 22 frames · `--steps 20 --layers 45 --reuse 2` | **95 s** | **28 s** (11 evals) | **~48 GiB** |
| **15 s cinematic** | 864×480 · 362 f · `--steps 20 --layers 45 --reuse 2` | **45.0 min** | **40.4 min** denoise wall | **~48 GiB** |
| **15 s + `--token-reduction`** | same + opt-in flag | **28.2 min** | denoise 23.3 min | **~36 GiB** |

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

## gfx90a / MI210 — retimed on `main` (2026-09-02)

Same MiniMax-H3 checkpoint. Build: `make HIP_ARCH=gfx90a`. Default DiT is
**BF16 hipBLAS** (not INT8). Four-GPU box: fox gates on `H3_HIP_DEVICE=1`,
15 s on GPU 2. **64 GiB** VRAM per GPU (`h3 --info` max HIP buffer). Peak
VRAM below is `--profile` `peak=` (live tensors), not `rocm-smi`.

VAE default tile on this tree is **480 px / 2×1** for 864×480 (was 272 px /
4×2). That is why 15 s VAE wall is ~72 s vs the older 93 s flash run.

### Default (BF16, no opt-in flags)

| Preset | Command knobs | E2E | Denoise GPU | Peak VRAM |
|--------|---------------|----:|------------:|----------:|
| **fox-s2** | 512² · 22 f · `--steps 2 --layers 35 --reuse 1` | **10.78 s** | **1.25 s** | **25.5 GiB** |
| **fox-fast** | 512² · 22 f · `--steps 20 --layers 45 --reuse 2` | **18.24 s** | **8.14 s** | **33.0 GiB** |
| **15 s cinematic** | 864×480 · 362 f · `--steps 20 --layers 45 --reuse 2` | **12 min 11 s** | **10 min 47 s** | **41.2 GiB** |

15 s BF16 split: sdpa **476.9 s** (74% of denoise), linear 156.8 s, VAE
72.2 s (peak 10.2 GiB). E2E **731.18 s**.

### INT8 DiT (`H3_INT8_MLP=1`)

On MI210 this is a **VRAM** knob, not a denoise win (unlike MI300X). No OOM
on the 64 GiB card.

| Preset | E2E | Denoise GPU | Peak VRAM | vs BF16 denoise |
|--------|----:|------------:|----------:|----------------:|
| fox-s2 | **11.13 s** | **1.33 s** | **15.1 GiB** | 0.93× (slower) |
| fox-fast | **19.54 s** | **9.11 s** | **19.7 GiB** | 0.89× (slower) |
| 15 s cinematic | **12 min 25 s** | **11 min 00 s** | **27.9 GiB** | 0.98× |

### All optimizations enabled

```bash
H3_INT8_MLP=1 H3_GPU_SAMPLER=1 H3_TOKEN_REDUCTION=1 H3_INT8_VAE=1
```

| Preset | E2E | Denoise GPU | Video VAE | Peak VRAM |
|--------|----:|------------:|----------:|----------:|
| **fox-s2** | **10.88 s** | **1.00 s** | **2.73 s** | **15.1 GiB** |
| **15 s cinematic** | **8 min 21 s** | **416.3 s** | **72.0 s** | **27.9 GiB** |

fox-s2 E2E stays I/O-bound. 15 s all-opts E2E matches CLI `--token-reduction`
on the previous tree (500.8 s): TR does the wall-clock work. INT8 VAE drops
VAE linear 46.0 s → **0.21 s** but VAE **other** goes 4.9 s → **51.1 s**
(per-tile quantize), so VAE wall does not fall. VAE peak **10.2 → 3.7 GiB**.

Logs: `/tmp/h3-mi210/main-perf/`. 15 s reproduce:
[`wiki/Long-video.md`](wiki/Long-video.md).

### 15 s cinematic: BF16 vs INT8 vs All (MI210)

| Metric | BF16 | INT8 DiT | All Opts | Δ BF16→All |
|--------|-----:|---------:|---------:|-----------:|
| **DiT denoise wall** | 646.5 s | 660.0 s | 416.3 s | **−36%** |
| DiT denoise linear | 156.8 s | 167.2 s | 119.8 s | −24% |
| DiT denoise sdpa | 476.9 s | 479.3 s | 285.8 s | **−40%** |
| DiT denoise other | 12.3 s | 12.8 s | 10.5 s | −14% |
| **DiT peak VRAM** | **41.2 GiB** | **27.9 GiB** | **27.9 GiB** | **−32%** |
| **Video VAE wall** | 72.2 s | 72.2 s | 72.0 s | ~0 |
| Video VAE linear | 46.0 s | 46.1 s | 0.21 s | −99% |
| Video VAE other | 4.9 s | 4.9 s | 51.1 s | +10× |
| **Video VAE peak** | **10.2 GiB** | **10.2 GiB** | **3.7 GiB** | **−64%** |
| Text encoder | 2.6 s | 2.7 s | 2.6 s | — |
| Audio VAE | 2.6 s | 2.6 s | 2.6 s | — |
| **E2E total** | **731.2 s** | **745.3 s** | **501.1 s** | **−31%** |

All Opts = `H3_INT8_MLP=1 H3_GPU_SAMPLER=1 H3_TOKEN_REDUCTION=1 H3_INT8_VAE=1`

INT8 DiT linear is slightly **slower** than BF16 hipBLAS on this CDNA2 card.
Token reduction is the 15 s speed path. INT8 VAE is the VRAM path.

### VRAM (MI210, 15 s cinematic)

`--profile` `peak=` (live tensors). Each GPU is **64 GiB**, not 128 GiB.

| | BF16 | INT8 DiT | All opts |
|--|-----:|---------:|---------:|
| Pipeline peak | **41.2 GiB** | **27.9 GiB** | **27.9 GiB** |
| Video VAE peak | **10.2 GiB** | **10.2 GiB** | **3.7 GiB** |

INT8 DiT is the pipeline-peak cut. INT8 VAE is the VAE-peak cut. Default 15 s
already fits a 64 GiB card; all-opts is headroom, not an enablement story.

## gfx942 / MI300X — v0.10.1 (2026-09-02)

Same MiniMax-H3 checkpoint. Build: `make HIP_ARCH=gfx942`. Default DiT is
**BF16 hipBLAS** (same as gfx90a). 192 GiB VRAM; weight I/O dominates E2E on
short presets.

### Default (BF16, no opt-in flags)

| Preset | Command knobs | E2E | Denoise GPU | Peak VRAM |
|--------|---------------|----:|------------:|----------:|
| **fox-s2** | 512² · 22 f · `--steps 2 --layers 35 --reuse 1` | **~16 s** | **0.92 s** | **25.7 GiB** |
| **fox-fast** | 512² · 22 f · `--steps 20 --layers 45 --reuse 2` | **~12 s** | **2.89 s** | **25.4 GiB** |
| **15 s cinematic** | 864×480 · 362 f · `--steps 20 --layers 45 --reuse 2` | **3 min 46 s** | **3 min 39 s** | **25.7 GiB** |

### INT8 DiT (`H3_INT8_MLP=1`)

| Preset | E2E | Denoise GPU | Peak VRAM | Denoise speedup |
|--------|----:|------------:|----------:|----------------:|
| fox-s2 | **~14 s** | **0.34 s** | **~79 GiB** | 2.7× faster |
| fox-fast | **~12 s** | **1.86 s** | **19.7 GiB** | 1.6× faster |

### All optimizations enabled

```bash
H3_INT8_MLP=1 H3_GPU_SAMPLER=1 H3_TOKEN_REDUCTION=1 H3_INT8_VAE=1
```

| Preset | E2E | Denoise GPU | Video VAE | Peak VRAM |
|--------|----:|------------:|----------:|----------:|
| **fox-s2** | **~8 s** | **0.34 s** | **~1.3 s** | **~30 GiB** |
| **15 s cinematic** | **~2.4 min** | **113 s** | **24.6 s** | **~32 GiB** |

Commands are the same as the gfx1151 block above. INT8 is opt-in on CDNA (`H3_INT8_MLP=1`); default is BF16 GEMM. MI300X denoise
is ~3× faster than MI210 on the same BF16 15 s path (186 s vs 647 s). fox-s2 /
fox-fast E2E is I/O-bound on both CDNA cards.

MI300X profile breakdown (15 s cinematic, BF16):
- **SDPA (flash MFMA)**: 144.5 s — **77%** of denoise
- **Linear GEMM (hipBLAS)**: 34.4 s — 18%
- **Other (norms, activations,)**: 7 s — 4%

### 15 s cinematic: BF16 vs INT8 comparison (MI300X)

| Metric | BF16 | INT8 DiT | All Opts | Δ BF16→All |
|--------|-----:|---------:|---------:|-----------:|
| **DiT denoise wall** | 186.3 s | 180.0 s | 112.9 s | **−39%** |
| DiT denoise linear | 34.6 s | 28.5 s | 20.9 s | −40% |
| DiT denoise sdpa | 144.1 s | 143.7 s | 85.7 s | −41% |
| DiT denoise other | 7.0 s | 7.1 s | 6.2 s | −11% |
| DiT weight-load | 1.9 s | 2.3 s | 2.1 s | +11% |
| **DiT peak VRAM** | **41.1 GiB** | **27.8 GiB** | **27.8 GiB** | **−32%** |
| **Video VAE wall** | 29.1 s | 29.0 s | **24.6 s** | **−15%** |
| Video VAE linear | 17.8 s | 17.8 s | 0.10 s | −99% |
| Video VAE other | 3.9 s | 3.9 s | 15.9 s | +308% |
| **Video VAE peak** | **9.4 GiB** | **9.4 GiB** | **3.7 GiB** | **−61%** |
| Text encoder | 2.5 s | 2.5 s | 2.5 s | — |
| Audio VAE | 0.8 s | 0.8 s | 0.8 s | — |
| **E2E total** | **~219 s** | **~212 s** | **~142 s** | **−35%** |

All Opts = `H3_INT8_MLP=1 H3_GPU_SAMPLER=1 H3_TOKEN_REDUCTION=1 H3_INT8_VAE=1`

INT8 DiT linear GEMM: 34.6 s → 28.5 s (−18%). GPU sampler eliminates CPU-GPU
latent round-trips: sdpa 144.1 s → 85.7 s (−41%). INT8 VAE linear: 17.8 s →
0.10 s (99× faster). VAE tile size 480 px (4→2 tiles/chunk) reduces VAE
"other" from 21.0 s to 15.9 s (−24%).

### Optimization: INT8 DiT (`H3_INT8_MLP=1`)

Pre-allocated INT8 workspace eliminates 2497 GiB of per-block allocation churn.
Denoise encode time dropped from 179 s to 0.06 s. INT8 denoise GPU time
(1.86 s) is 33% faster than BF16 (2.86 s) on fox-fast.

**VRAM impact:** INT8 DiT reduces DiT weight memory from ~62 GiB to ~28 GiB
(BF16→INT8), but peak VRAM increases to ~79 GiB for fox-s2 due to the
quantize workspace allocation pattern. fox-fast stays at ~20 GiB.

### Optimization: GPU Euler sampler (`H3_GPU_SAMPLER=1`)

Keeps latents on GPU across Euler steps, eliminating CPU-GPU round-trips.
fox-s2 denoise: 1.95 s → 0.84 s (57% faster). fox-fast denoise: 2.89 s →
2.76 s (4.5% faster).

**VRAM impact:** +0.2 GiB for latent buffers (negligible).

### Optimization: Token reduction (`H3_TOKEN_REDUCTION=1`)

Drops half spatial width in middle layers (blocks 4–30). 15 s cinematic denoise:
185.7 s → 116.8 s (37% faster). Quality impact: slight detail loss in fine
textures.

**VRAM impact:** −12 GiB for 15 s cinematic (48→36 GiB on gfx1151) due to
halved spatial tokens in middle DiT blocks.

### Optimization: Video VAE INT8 (`H3_INT8_VAE=1`)

Quantizes VAE linear weights to INT8 on-the-fly with persistent workspace.
INT8 GEMMs are 37× faster (2.6 s → 0.07 s) but per-tile input quantize
overhead adds ~21 s GPU time.

**VRAM impact:** −6.5 GiB for Video VAE (9.4→2.9 GiB peak). Total VAE
allocation drops from 175.6 GiB to 11.7 GiB with persistent workspace.
Best for VRAM-constrained scenarios; quality impact negligible for decoding.

### Fused quantize+GEMM kernel (experimental, not default)

`h3_launch_linear_int8_f32_fused` combines F32→INT8 quantization and INT8 GEMM
into a single kernel launch, eliminating the intermediate INT8 buffer. Available
via `h3_gpu_linear_f32_int8_fused()` but **not called by default**.

**Why not default:** the fused kernel reads F32 input twice (scales + quantize)
vs non-fused reading F32 once + INT8 once (4× smaller). For long sequences the
extra bandwidth cost outweighs the kernel-launch savings:

| Scenario | Non-fused VAE | Fused VAE | Delta |
|----------|-------------:|----------:|------:|
| fox-s2 (22 f) | 1.21 s | 1.81 s | +50% |
| 15 s cinematic (362 f) | 28.1 s | 58.0 s | +107% |

**When to use:** VRAM-constrained scenarios where the INT8 intermediate buffer
(9.4 GiB for fox-s2) cannot be afforded. The fused path reduces VAE peak from
9.4→2.9 GiB (−69%) at the cost of slower execution.

### VAE tile size optimisation (default 480 px on MI300X)

`configured_tile_pixels()` scans tile sizes from 256 to 512 px (step 16) and
picks the size that minimises `tiles × pixels²`. The extended scan range
(previously capped at 320 px) allows the optimizer to select larger tiles
that reduce per-chunk tile count.

| Resolution | Old tile | New tile | Tiles Δ | VAE Δ |
|------------|---------|---------|---------|-------|
| 512×512 (fox-s2) | 288 (2×2=4) | **512** (1×1=1) | −75% | ~same |
| 864×480 (15 s cinematic) | 272 (2×4=8) | **480** (1×2=2) | −75% | **−12%** |

15 s cinematic impact (MI300X, all-opts):
- VAE total: 28.1 s → **24.6 s** (−12%)
- VAE other: 21.0 s → 15.9 s (−24%); fewer tiles = less per-tile quantize
- VAE sdpa: 1.9 s → 4.5 s (+137%); longer per-tile sequences (2028→6305)
- Submissions: 168 → 42 (−75%)
- VAE peak VRAM: 2.9 GiB → **3.7 GiB** (+29%, still << 192 GiB)

Override at runtime: `H3_VAE_TILE_PIXELS=272` restores old behaviour.

Optional **`--token-reduction`** (off by default; same CLI as h3-spark.c):
pairs middle-block video tokens so long-N SDPA shrinks. Do not replace the
**tagged** quality-path row (45.0 min / 12 min 11 s) with these numbers.

| | quality path | **`--token-reduction`** |
|--|--:|--:|
| gfx1151 15 s E2E | 45.0 min | **28.2 min** (−37%); [perf-runs/TOKEN_REDUCTION.md](perf-runs/TOKEN_REDUCTION.md) |
| gfx90a 15 s E2E | 12 min 11 s | **8 min 21 s** (−31% all-opts / CLI TR); [perf-mi210/TOKEN_REDUCTION.md](perf-mi210/TOKEN_REDUCTION.md) |
| gfx942 15 s E2E | 3 min 46 s | **~2.5 min** (−34%) |

gfx1151 fox-fast denoise 34.6 s → 25.8 s was already measured at v0.9.0.

## VRAM optimization summary (MI300X, 15 s cinematic)

| Component | BF16 baseline | Optimized | Savings | Mechanism |
|-----------|-------------:|----------:|--------:|-----------|
| DiT weights | 62 GiB | **28 GiB** | **−55%** | INT8 weight quantization |
| DiT activations | 14 GiB | **8 GiB** | **−43%** | Token reduction (middle layers) |
| DiT peak | 41 GiB | **28 GiB** | **−32%** | Combined INT8 + token reduction |
| Video VAE weights | 9.4 GiB | **3.7 GiB** | **−61%** | INT8 weight quantization + 480 px tiles |
| Video VAE alloc churn | 175.6 GiB | **11.7 GiB** | **−93%** | Persistent workspace buffers |
| **Pipeline peak** | **~50 GiB** | **~32 GiB** | **−36%** | All opts combined |

The VRAM reductions enable running 15 s cinematic on GPUs with 32 GiB VRAM,
which was previously impossible (required ~50 GiB). The INT8 VAE path
(`H3_INT8_VAE=1`) combined with 480 px tiles reduces VAE peak from 9.4 to
3.7 GiB (−61%).

## How a GitHub Release should quote this

Paste the **current** summary table (all three ISAs). Do not paste phase splits,
KEEP/REJECT lists, or Metal ratio tables into the release body.
