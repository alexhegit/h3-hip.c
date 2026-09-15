# Sol-Attn (lossy long-sequence SDPA)

**Default: quality.** Dense flash SDPA stays on unless you pass `--sol-attn`.
Do not enable this for publication, audio-sensitive, or fox-s2 identity work.

This PR **closes MI210**. Other SKUs are follow-ups, not merge blockers.

| SKU | ISA | This PR | Notes |
|---|---|---|---|
| **MI210** | `gfx90a` | **measured** | 15 s no-TR A/B below; KEEP as opt-in only |
| MI300X | `gfx942` | same MFMA source, **untimed** | should run; no 15 s PSNR/speed table yet |
| Strix Halo | `gfx1151` | **not ported** | `--sol-attn` stays dense (safe no-op) |

`--sol-attn` is an **opt-in quality/speed trade**: skipped 64-token KV tiles
are pooled into the online softmax instead of computed exactly. Keep-all
(`H3_SOL_ATTN_TAU=-100`) matches dense bit-for-bit; the default τ=0.5 path
does not. Sequences shorter than `H3_SOL_ATTN_MIN_SEQ` (default 4096) stay
dense on every ISA.

```bash
./h3 -d MODEL --sol-attn -p '...' --seconds 15
# or: H3_SOL_ATTN=1 H3_SOL_ATTN_TAU=0.5
# keep-all (debug): H3_SOL_ATTN=1 H3_SOL_ATTN_TAU=-100
# microbench: H3_BENCH_SDPA=1 H3_BENCH_SDPA_SOL=1 H3_BENCH_SDPA_SEQ=44800 ./h3_hip_bf16_tests
```

## Measured vs dense baseline (MI210, gfx90a)

Fixed seed, 15 s cinematic, **no token reduction**, same prompt/checkpoint as
the dense quality path. Sol-Attn τ=0.5, blocks 4–40, prefix tiles exact.
Quality is versus that dense MP4 (not versus BF16 gold).

| metric | dense (quality path) | `--sol-attn` τ=0.5 | vs dense |
|---|---:|---:|---|
| E2E | 725.31 s | 593.04 s | **−18.2%** (1.22×) |
| denoise | 634.95 s | 502.89 s | **−20.8%** |
| denoise SDPA | 456.21 s | 324.38 s | **−28.9%** |
| peak VRAM | 27.91 GiB | 27.91 GiB | unchanged |
| exact KV tiles | 100% | 33.43% | 66.57% pooled |
| video PSNR / SSIM | reference | **18.73 dB / 0.712** | preview-grade vs dense |
| decoded audio SNR | reference | **6.69 dB** | large waveform error |

SDPA kernel microbench (56 heads, d128, τ=0.5): 8,192 tokens **2.05×**,
44,800 tokens **2.56×**. fox-s2 (~1.9k tokens) stays on dense by default.

Do not stack with `--token-reduction` unless you accept compounded error.

Packed H3 layout is text/cond/audio then video. Query CTAs below
`video_target_start` already ran exact SDPA via the prefix rule; making that
explicit did not change outputs. Last-20% denoise dense
(`H3_SOL_ATTN_DENSE_TAIL=4` on 20 steps) also did not help versus dense:
video stayed ~18.72 dB / 0.713 SSIM and audio SNR ~6.67 dB, while denoise
went 503 s → 547 s and SDPA 324 s → 366 s. Default tail is therefore 0.

Full design notes follow.

This document evaluates porting the training-free, Sol-Attn-style sparse SDPA
implemented in
[`h3-spark.c#1`](https://github.com/alexhegit/h3-spark.c/pull/1) to the HIP
backend. The feature must remain **opt-in** and must not change the default
quality path.

## Executive summary

Sol-Attn is a good match for h3-hip.c's remaining long-video bottleneck. It
routes 64-token KV tiles using a cheap proxy: selected tiles use exact matrix
attention, while skipped tiles are represented by pooled K/V statistics in the
online softmax. It needs no new model weights or training.

The CUDA code cannot be reused directly. The routing algorithm should be
ported into the existing HIP flash kernels:

- one shared host/API layer and one architecture-neutral KV-summary kernel
- one wave64 MFMA implementation shared by `gfx90a` and `gfx942`
- one separate wave32 rocWMMA implementation for `gfx1151`

Therefore this is **one algorithm with two kernel families**, not three
independent implementations. Start on MI300X (`gfx942`), validate the same
MFMA path on MI210 (`gfx90a`), then port it to Strix Halo (`gfx1151`).

## Evidence from h3-spark.c

The merged Spark implementation reports the following on GB10:

- 15 s cinematic: 1076 s to 694.7 s E2E (**-35%**)
- denoise: 988 s to 608.9 s
- SDPA: 845 s to 464 s
- output quality: PSNR 19.2 dB / SSIM 0.72 versus the dense quality path
- 44,800-token kernel: about **3x** faster
- keep-all mode (`H3_SOL_ATTN_TAU=-100`) is bit-identical to dense MMA

On the short fox-fast workload, the speedup was small and quality failed the
project's KEEP threshold. Sol-Attn is therefore a long-sequence speed option,
not a short-clip optimization.

These figures are reference evidence, not a performance promise for AMD GPUs.
HIP KEEP/REJECT decisions require measurements on each timed SKU.

## Why it fits h3-hip.c

Long T2VA is still dominated by DiT SDPA:

- Strix Halo no-TR 15 s: SDPA 1581 s of 2167 s denoise (about 73%)
- MI210 no-TR 15 s: SDPA about 466-477 s (about 72-74%)
- MI300X no-TR 15 s: SDPA about 144 s of 186 s denoise (about 77%)

The current DiT shape is also a direct fit: 56 heads and `HEAD_DIM=128`.
Sol-Attn reduces work in the long KV-tile loop without changing checkpoint
weights.

It will not materially improve cold fox-s2 E2E on Halo because weight I/O, not
SDPA, dominates that workload.

## Existing HIP kernel split

`h3_launch_sdpa_bf16` already dispatches by ISA:

| Timed product | ISA | Current long-sequence SDPA | Sol-Attn work |
|---|---|---|---|
| MI300X | `gfx942` | wave64 MFMA flash, d128 | add routing to shared MFMA kernel; tune independently |
| MI210 | `gfx90a` | wave64 MFMA flash, d128 | validate/tune the same MFMA implementation |
| Strix Halo | `gfx1151` | wave32 rocWMMA flash, d128 | separate WMMA port due to fragment and wave layout |

`gfx90a` and `gfx942` may use different optimal `WAVES` / `BK` values, but
they should share source templates. The wave32 implementation cannot safely
share matrix-fragment code with CDNA.

## Proposed behavior

Add an opt-in CLI flag and matching environment controls:

| Control | Proposed default | Meaning |
|---|---:|---|
| `--sol-attn` / `H3_SOL_ATTN=1` | off | enable sparse SDPA |
| `H3_SOL_ATTN_TAU` | `0.5` | keep tiles scoring at least mean + tau * stddev |
| `H3_SOL_ATTN_BAND` | `1` | always keep nearby KV tiles |
| `H3_SOL_ATTN_PREFIX` | derived | always keep text/condition/audio prefix tiles |
| `H3_SOL_ATTN_BLOCKS` | `4:40` | DiT blocks allowed to use sparse SDPA |
| `H3_SOL_ATTN_MIN_SEQ` | `4096` | shorter sequences stay dense on HIP |
| `H3_SOL_ATTN_DENSE_TAIL` | `0` | last N denoise steps stay dense (off; no quality win on MI210) |
| `H3_SOL_ATTN_STATS` | off | print aggregate exact/skipped tile counts |
| `H3_SOL_ATTN_DROP` | off | A/B-only mode that drops skipped tiles entirely |

Sequences shorter than 4096 tokens remain on dense SDPA after MI210
measurements showed that routing overhead exceeds the saved work below this
point. Tests can override the gate down to 512. The default generation path
must remain unchanged.

Do not enable Sol-Attn and token reduction together by default. Both reduce
attention work by different mechanisms, but their errors can compound. Measure
the combination only after each feature passes independently.

SageAttention and Sol-Attn should also be separate experimental dispatches:
Sage reduces arithmetic precision; Sol-Attn reduces the number of exact KV
tiles.

## Kernel design

### 1. Shared KV summaries

Create an architecture-neutral HIP kernel that computes, per head and per
64-token KV tile:

- K proxy/mean used by the router
- pooled V statistic used when a tile is not selected

Allocate and reuse summary/routing workspace from the HIP context. Do not
allocate per DiT block or per denoising step.

### 2. Routing

For each query block:

1. score all KV summaries
2. calculate the score mean and standard deviation
3. keep tiles above `mean + tau * stddev`
4. always keep the local band and dense prefix
5. process kept tiles through the existing exact MMA loop
6. incorporate pooled K/V for skipped tiles into the online-softmax state

`tau=-100` must take every tile through the same exact path and match dense
output bit-for-bit where the existing kernel is deterministic.

### 3. CDNA MFMA path

Modify `h3_sdpa_bf16_mfma_d128_kernel<WAVES, BK, FP16_PV>` without changing
its dense behavior:

- keep the current BF16 QK, FP16/BF16 PV, and FP32 accumulator choices
- branch at the KV-tile loop
- preserve head-major K/V handling and online-softmax rescaling
- tune MI300X and MI210 launch parameters separately, using the same source

MI300X is first because its 15 s test cycle is shortest and SDPA is the largest
fraction of denoise. MI210 is the compatibility and CDNA2 tuning pass.

### 4. RDNA rocWMMA path

Port the same routing semantics to
`h3_sdpa_bf16_wmma_d128_kernel<WAVES, BK>`. Keep it separate from MFMA
fragment handling because `gfx1151` uses wave32 and different accumulator
layouts.

Halo is last because a 15 s A/B takes roughly 40 minutes on the no-TR path.
Its potential absolute saving is large, but short clips remain I/O-bound.

## Host integration

1. Add `sol_attn` to `h3_params`, `--sol-attn` to the CLI, and an interactive
   toggle if useful.
2. At each DiT block, configure whether sparse SDPA is active and pass the
   dense-prefix tile count derived from `video_target_start`.
3. Extend the HIP context with reusable summary/routing buffers and current
   layer configuration.
4. Dispatch Sol-Attn only for supported d128 DiT shapes and sufficiently long
   sequences; otherwise use dense SDPA.
5. Print a warning that output is approximate and not bit-identical.

Text, condition, and audio KV tiles must stay exact through the prefix rule.
Audio query rows should also remain dense initially. This follows the Spark
implementation and limits cross-modal quality loss.

## Implementation phases

### Phase 0: baseline and test harness

- add a long-sequence SDPA microbenchmark for sequence lengths 1,874, 8,192,
  16,384, and 44,800
- capture dense output and timing on all three GPUs
- add a keep-all comparison test

### Phase 1: MI300X (`gfx942`)

- implement summary, routing, and the MFMA sparse loop
- verify keep-all output
- sweep tau and local band at sequence 44,800
- run fixed-seed 15 s no-TR A/B with `--profile`
- measure E2E, denoise, `sdpa=`, PSNR, SSIM, audio SNR, and peak VRAM

### Phase 2: MI210 (`gfx90a`)

- compile the shared MFMA implementation for `gfx90a`
- retune `WAVES` / `BK` only where measurements justify it
- repeat microbench, keep-all, and fixed-seed 15 s gates

### Phase 3: Strix Halo (`gfx1151`)

- implement the wave32 rocWMMA sparse loop
- run microbench first to avoid expensive failed 15 s experiments
- repeat keep-all and 15 s gates
- confirm fox-s2/default path remains unchanged

### Phase 4: combinations

Only after independent KEEP:

- Sol-Attn + token reduction
- Sol-Attn + reuse 3
- SKU-specific tau/band defaults, if quality results support them

No combination becomes the default quality path.

## KEEP / REJECT gates

### Correctness

- feature off: existing tests and fox-s2 regression remain unchanged
- keep-all: zero output differences versus the corresponding dense MMA kernel
  (gfx90a: seq 512–44800, 56 heads, `H3_SOL_ATTN_TAU=-100`)
- unsupported/short shapes: verified dense fallback

### Performance

For 44,800 tokens, target at least:

- **2x** SDPA kernel speedup in sparse mode
- **20%** reduction in 15 s `sdpa=`
- no material regression outside SDPA

MI210 (`gfx90a`) microbench, 56 heads, d128, 3 iters:

| seq | dense | keep-all | tau=0.5 | vs dense |
|---:|---:|---:|---:|---:|
| 1,874 | 30.1 ms | 32.3 ms | 23.9 ms | 1.26x |
| 8,192 | 534.8 ms | 539.8 ms | 261.4 ms | 2.05x |
| 16,384 | 2.11 s | 2.12 s | 0.91 s | 2.33x |
| 44,800 | 15.69 s | 15.75 s | 6.12 s | **2.56x** |

MI210 fixed-seed 15 s no-TR A/B after batching 16 pooled pseudo-KV tiles
through MFMA for both QK and PV:

| metric | dense | Sol-Attn τ=0.5 | change |
|---|---:|---:|---:|
| E2E | 725.31 s | 593.04 s | **−18.2%** |
| denoise | 634.95 s | 502.89 s | **−20.8%** |
| denoise SDPA | 456.21 s | 324.38 s | **−28.9%** |
| peak VRAM | 27.91 GiB | 27.91 GiB | unchanged |
| exact KV tiles | — | 33.43% | 66.57% skipped |
| video PSNR / SSIM | reference | 18.73 dB / 0.712 | approximate |
| decoded audio SNR | reference | 6.69 dB | approximate |

Same 15 s numbers as the table at the top of this document. Speed meets the
20% SDPA gate; quality stays **REJECT for default on** (preview video, weak
audio). Duplicate table kept only for the KEEP checklist.

Reject or retune if routing/summary overhead erases the gain, particularly on
short sequences.

### Quality

Use the same prompt, seed, geometry, layers, reuse, and checkpoint as the dense
15 s baseline. Record:

- video PSNR and SSIM
- audio waveform SNR
- representative MP4s for visual/temporal inspection

Spark's 19.2 dB / 0.72 is a reference point, not an automatic HIP default.
HIP KEEP for **default on** would need audio closer to the dense waveform.
Until then the dense path is the quality default; `--sol-attn` is documented
lossy acceleration only.

## Risks

- Divergent tile decisions can lower wave occupancy or serialize work.
- Summary and pooled-value traffic may consume much of the theoretical saving.
- Optimal tile geometry may differ between CDNA2, CDNA3, and RDNA3.5.
- Approximation error compounds across DiT blocks and can affect audio through
  cross-modal feedback even when audio tiles remain dense.
- A kernel-only 3x gain will not produce a 3x E2E gain because linear layers,
  VAE, weight loading, and dense DiT blocks remain.

## Expected result

The project treats Sol-Attn as optional long-video **lossy** acceleration.
MI210 already meets the 20% SDPA speed gate; quality does not meet a default-on
bar. Preserve the dense path exactly when the flag is off.
