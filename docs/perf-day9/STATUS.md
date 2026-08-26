# Day-9 perf loop — STATUS

Started: 2026-08-24 22:16 CST  
Budget: ~10 hours  
Baseline fox s2: denoise GPU **7.61s** (lin **3.44** · sdpa **3.54**)  
Baseline fox-fast: denoise GPU **54.27s** (lin **24.18** · sdpa **25.44**)  
Priority: **P0 — replace the d128 SDPA architecture** (Q6 variants are exhausted);
then P1 DiT linear utilization, P2 leftover.  
Do not retry: everything on the day-8 list, plus Q6 variant tweaks of any kind.

## P0 result: tiled WMMA flash attention (d128)

Q6's ceiling was structural, not tunable. At `seq=1920 heads=56 d=128` the op is
**105.7 GFLOP**; Q6 ran it in 45 ms = **2.4 TFLOP/s**, about 4% of the ~59
TFLOP/s BF16 WMMA peak. Two causes, both fixed by the same kernel:

1. Every `(query, key)` score was a cross-lane `h3_warp_reduce_sum` — 5 shuffles
   per score, 960 shuffles per 32 keys for 6 queries. WMMA keeps the dot product
   inside the matrix unit, so scores cost no cross-lane traffic at all.
2. Only 6 queries shared each K/V load, so global traffic was ~17.6 GB per call
   (68 ms of pure bandwidth at 256 GB/s, partly hidden by L2). A 128-row query
   block sharing a 32-key LDS tile cuts that to ~0.83 GB.

Result: **45.0 → 6.7 ms** micro (**6.7×**, 15.8 TFLOP/s, 27% of peak).

### Fragment layouts (probed on gfx1151, rocWMMA 2.2, wave32)

| Fragment | lane l holds | element i holds |
|---|---|---|
| `matrix_a` row_major | row `l&15` | k `8*(l>>4)+i` |
| `matrix_b` col_major | col `l&15` | k `8*(l>>4)+i` |
| `accumulator` f32 | col `l&15` | row `2*i+(l>>4)` |

The accumulator row mapping is what makes this cheap: element `i` of *every* O
fragment belongs to row `2*i+(l>>4)`, so the online-softmax rescale is one
`alpha[i]` multiply per element index, entirely in registers — no LDS round-trip
for O. Row stats reduce across the 16 lanes sharing a row; XOR masks 1/2/4/8
never flip lane bit 4, so the two row groups stay independent.

K is staged row-major `[key][dim]` in LDS and read back as `col_major` with the
same leading dimension, which yields `K^T` for free.

### Tile shape sweep (micro, seq=1920 heads=56 d=128)

| waves × BK | ms | | waves × BK | ms |
|---|---:|---|---|---:|
| 2 × 32 | 13.9 | | 8 × 16 | 7.7 |
| 4 × 16 | 8.3 | | **8 × 32** | **6.7** |
| 4 × 32 | 8.0 | | 8 × 64 | 7.7 |
| 4 × 64 | 12.3 | | 16 × 16 | 7.9 |
| | | | 16 × 32 | 6.9 |

## P1 finding: the linear ceiling is much closer than day-8 assumed

Day-8 concluded that 27 TOPS was a hardware ceiling because "RDNA 3.5 has no
INT8 matrix unit". Both halves of that are wrong, in opposite directions.

A register-resident WMMA throughput probe (`/tmp/wmma_peak.hip`, no memory
traffic in the timed loop) on gfx1151 (20 CU @ 2.9 GHz):

| dtype | T(FL)OP/s |
|---|---:|
| bf16 → f32 | 56.9 |
| f16 → f32 | 55.7 |
| int8 → i32 | 55.7 |

So there *is* an INT8 WMMA path, but unlike CDNA/Tensor Core it gives **no**
throughput bonus over bf16 — the ceiling is ~57 for everything. INT8 is
therefore a memory-capacity choice (17.3 GB of DiT weights vs 34.6 GB in bf16,
against a 19.7 GiB peak), not a throughput one.

Per-shape hipBLAS INT8 at the DiT geometry (M=1920):

| GEMM | K | N | ms | TOPS | % of 57 |
|---|---:|---:|---:|---:|---:|
| QKV | 5376 | 21504 | 10.7 | 41.4 | 73% |
| out | 7168 | 5376 | 4.6 | 32.1 | 56% |
| FC1 | 5376 | 28672 | 15.7 | 37.6 | 66% |
| FC2 | 14336 | 5376 | 10.9 | 27.2 | 48% |

41.9 ms/layer × 45 layers × 11 evals = 20.7 s, which accounts for the measured
23.3 s. The average is already **62% of peak**, not 47% — day-8's 27 TOPS was
just the worst of the four shapes. A custom GEMM would have to beat Tensile on
all four to win ~1.3×, so this is parked as poor effort/reward.

The grouped FC2 path does have a real structural cost: `H3_INT8_SPLIT=1` splits
it into GEMM 8.0 ms + **grouped-reduce 2.67 ms**, because the batched GEMM
materializes 14 i32 planes (578 MB) that the reduce then reads back. Fusing the
group reduction into a custom GEMM epilogue is the one linear idea with real
headroom left, and it only applies where `K % group_size == 0`.

## P2: per-kernel profile found the actual leftover hot spot

`rocprofv3 --kernel-trace` on fox s2 showed `h3_qkv_rope_bf16_kernel` at
**5.98 ms per call**, as expensive as the entire (already optimized) SDPA, for
an op that only moves 165 MB — 9× off bandwidth. Cause: every one of the 128
threads in a block re-summed the whole q/k row for the RMS norm, so each row was
read head_dim times.

Two steps: a cooperative LDS pass with a block reduction got 5.98 → 3.22 ms, but
the 9 barriers on a 128-thread block then dominated. The vectorized wave version
(one wave per (row, head), 4 dims per lane, wave-reduced RMS, RoPE partner via
lane shuffle, no LDS and no barrier) reached **0.72 ms** — 90% of the 0.65 ms
bandwidth bound, **8.3×** over the original.

`rope_half` is 48, not head_dim/2, but 48 % 4 == 0 so each lane's four dims fall
entirely in one rotation region and the partner is a fixed `rope_half/4` lane
shuffle. The shuffles must be issued outside the rotation branch: a source lane
has to be active, which it would not be inside the divergent region.

## P3: the whole-process profile moved the target to the video VAE

Once denoise came down, a `rocprofv3` pass over all of fox s2 (28.06s of GPU)
showed the DiT is no longer where the GPU time is:

| kernel | calls | total | phase |
|---|---:|---:|---|
| `h3_linear_f32_r128_kernel<32>` | 582 | **10.48s (37%)** | video VAE linear |
| `h3_sdpa_f32_wave_d64_qn_kernel<4>` | 144 | 4.75s | video VAE sdpa |
| `h3_conv1d_f32_kernel` | 204 | 3.50s | audio VAE |
| Tensile INT8 GEMM (2 variants) | 280 | 2.87s | **denoise** linear |
| `h3_linear_bf16_kernel` | 410 | 2.56s | text encoder 1.76 + DiT load 0.81 |
| `h3_quantize_bf16_int8_rows_kernel` | 280 | 2.13s | DiT load weight quantize |
| `h3_sdpa_bf16_wmma_d128_kernel<8,32>` | 70 | 0.45s | denoise sdpa |
| `h3_qkv_rope_bf16_d128_kernel` | 72 | 0.05s | denoise other |

Two conclusions changed the plan:

- **`h3_linear_bf16_kernel` is not worth optimizing.** Its 2.56s is exactly the
  text-encoder (1.76) plus DiT-load (0.81) linear totals, and both phases are
  host-I/O bound: text encoder wall 27.28s against 1.79s of GPU and 0.02s of
  encode, DiT load 44.13s against 2.91s. Taking its GEMM to zero would not move
  the wall. It is a naive 16x16 tile doing one output element per thread, so the
  kernel is genuinely bad — it just does not matter.
- Per-phase wall vs GPU says only three phases are GPU-bound at all: denoise
  (4.74 / 4.07), audio VAE (4.13 / 3.64), video VAE (21.07 / 15.72).

## P3 result: f32 linear on the bf16 matrix cores

`h3_linear_f32_r128_kernel` is a competent SGEMM — 4.6-4.8 TFLOP/s measured, and
the FP32 FMA ceiling is only 7.42 (20 WGP x 128 FLOP/clk x 2.9 GHz), so it is at
~62% of what plain FMA can do. The only way up is the matrix cores at 57.

Tensile is not the answer here. hipBLAS on the same shapes:

| path | geglu | qkv | ff-down | out |
|---|---:|---:|---:|---:|
| f32 | 2.74 | 2.72 | 2.61 | 2.73 |
| bf16 in / **f32** out | 4.77 | 4.92 | 4.66 | 5.00 |
| bf16 in / **bf16** out | 21.65 | 26.22 | 18.26 | 40.31 |

(TFLOP/s.) Tensile only has well-tuned kernels when the output is also bf16,
which would add a 2^-9 rounding on every output element. A hand-written kernel
keeping the f32 accumulator and f32 output stays at ~1.1e-4 relative error —
20x better — and still beats Tensile's mixed-output path by 2.4x.

Tile sweep at the VAE shapes, ms (all bf16 in / f32 out, K tile 32, prefetched):

| cfg | tile | per-wave | VGPR | spill | occ | geglu | qkv | ff-down | out |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| — | r128 (current) | — | — | — | — | 32.5 | 12.3 | 16.9 | 5.2 |
| 1 | 256x128 | 64x64 | 256 | 266 | 5 | 31.7 | 13.5 | 18.1 | 4.8 |
| 2 | 256x256 | 64x64 | 237 | 0 | 6 | 14.8 | 6.5 | 9.3 | 2.2 |
| **3** | **256x128** | **32x64** | **164** | **0** | **9** | **13.0** | **5.3** | **7.2** | **2.1** |
| 4 | 128x128 | 32x64 | 171 | 0 | 8 | 19.5 | 7.6 | 8.4 | 4.2 |
| 6 | 256x128 | 64x32 | 183 | 0 | 8 | 13.3 | 5.5 | 7.5 | 2.1 |
| 7 | 256x256 | 32x64 | 213 | 0 | 7 | 14.5 | 6.2 | 8.2 | 2.1 |
| 8 | 512x128 | 32x64 | 164 | 0 | 9 | 12.8 | 5.7 | 9.3 | 2.5 |

A 64x64 per-wave tile needs 16 accumulator fragments = 128 VGPRs and spills to
scratch, which costs exactly as much as the matrix cores gain: cfg 1 ties r128.
A 32x64 tile needs 8, fits in 164 VGPRs, and wins everywhere at **~2.4x**.

Two measurement traps worth recording. A standalone probe picked cfg 1 as the
best at "12.4 TFLOP/s" — it was spilling there too, and its timing was simply
unreliable. And `bench_linear_f32` ran 2 iterations around one begin/submit, so
its ~23ms of fixed host cost swamped the kernel: it reported 15.2ms where
rocprof showed 3.6ms. It now runs 50 (`H3_BENCH_ITERS`).

### The size gate is what makes this safe

The first fox s2 run with the WMMA path on produced a *different video* — same
prompt, same seed, an equally plausible fox facing the other way, 19.9 dB PSNR
against the baseline. The VAE decoder is terminal and cannot change the latents,
so something upstream was going through `h3_launch_linear_f32` too: of the 582
r128 dispatches, 576 are the video VAE's three shapes and **6 are small f32
projections** (one group at N≈5376, the DiT hidden size, with M≤128) that feed
the DiT conditioning. A 1e-4 perturbation there is amplified by the diffusion
trajectory into a different sample.

Gating on `rows >= 256 && input_dim >= 512 && output_dim >= 512` keeps those on
the exact path — which costs nothing, since a GEMM under 256 rows cannot fill
even one 256x128 tile. PSNR against the baseline went **19.9 → 46.9 dB** (y;
48.1 average), i.e. the same sample with decode-level rounding, and the extracted
frames are indistinguishable.

Runs are bit-reproducible per configuration (seed defaults to 42), so mp4 md5 is
a valid same-sample check for anything that does *not* touch the decoder, and
PSNR is the check for anything that does. **That held again only after the tail
bug below was fixed** -- for part of this session it did not, which cost the
PSNR numbers above their meaning until they were re-measured.

## The tiled query-block tail made the whole pipeline irreproducible

Two runs of the same binary with the same environment stopped producing the same
video once the tiled f32 d64 SDPA was on. That invalidates md5 as a check, and
worse, it put a floor of **46.96 dB** under every PSNR comparison -- the exact
number that had been read as "same sample, decode-level rounding" for the gated
f32 linear. The conclusion was right, but the evidence had been measuring
nothing but the noise.

Localizing it: audio was bit-identical across all runs, so the DiT and both
latents were fine and only the video decoder had drifted. Rerunning the kernel
on its own inputs inside the process and counting differing elements put the
disagreement at ~0.1% of the output, and printing the row indices pinned it to
rows **2257..2271** of 2273 -- exactly the overlap of the last query block.

Both tiled kernels clamp a short tail back onto the previous query block
(`qbase = sequence - QROWS`) so the tail needs no masked path, on the assumption
that the duplicate stores write the same value. They do not. Two waves
computing the same row disagree in the last bits, and which store lands last is
a race, so the decoder output changed run to run and the mp4 changed with it.

The isolated repeatability probe missed this twice: with smooth synthetic
inputs the duplicate writers agreed bit-for-bit, and the CPU-reference tests
cannot see it at all because both candidate values are well inside tolerance.
`test_sdpa_tiled_repeatable` compares repeated runs against *each other* at
sequences that are not whole query blocks (257/300/511/1000, both d64 and d128);
reverting either fix makes it fail.

Fixes, both zero-cost:

- d64 masks the tail instead of clamping (query rows past the end read as zero
  and are never stored), since it already stages Q through LDS.
- d128 keeps the clamp, because its query fragments load straight from global,
  but stores only rows at or after the wave's *unclamped* start. The union over
  waves is then exactly the sequence, with no row written twice.

Neither shows up in the microbenchmarks (d64 4.12 -> 4.15 ms). The d128 path had
carried this latent since it was written: the DiT's 1920 is a whole number of
128-row blocks, so it never had a tail to get wrong.

## Scoreboard (fox s2)

| Version | denoise GPU | linear | sdpa | other |
|---------|------------:|-------:|-----:|------:|
| day-8 final | 7.61 | 3.44 | 3.54 | 0.63 |
| + WMMA flash SDPA | 4.49 | 3.36 | 0.51 | 0.62 |
| + qkv-rope LDS | 4.24 | 3.39 | 0.43 | 0.42 |
| + qkv-rope vectorized | **4.09** | 3.38 | 0.46 | **0.25** |

Denoise is untouched by the P3 work (4.02 with the WMMA f32 linear on).

## Scoreboard (fox s2, video VAE decoder)

| Version | GPU | linear | sdpa | other | wall |
|---------|----:|-------:|-----:|------:|-----:|
| r128 f32 (same-binary A/B) | 15.82 | 10.56 | 4.76 | 0.50 | 21.07 |
| + gated WMMA f32 linear | 9.51 | **4.32** | 4.73 | 0.47 | 14.85 |
| + tiled WMMA f32 d64 SDPA | **5.27** | 4.24 | **0.56** | 0.47 | **10.65** |

Wall no longer tracks GPU: 5.27s of GPU under a 10.65s phase, so the video
decoder is now roughly half host time (tile stitching and RGB conversion are
scalar single-threaded loops), and the audio decoder is host-bound outright.

## Scoreboard (fox s2, audio VAE decoder)

| Version | GPU | conv | other | wall |
|---------|----:|-----:|------:|-----:|
| scalar conv1d | 3.77 | 3.67 | 0.10 | 4.25 |
| + time-blocked conv1d | **0.45** | **0.35** | 0.10 | **0.92** |

## The f32 d64 SDPA had the same shape of problem as Q6

The video VAE's attention (seq 2304, 32 heads, d64) ran the per-key wave
kernel at **1.31 TFLOP/s** — 2% of the bf16 ceiling, for the same reason the
DiT's Q6 kernel did: one query row per wave means K and V are re-read from
global for every row. `h3_sdpa_f32_wmma_d64_kernel` is the d128 kernel with 4
dim-slices instead of 8, staging f32 K/V into LDS as bf16 pairs the way the f32
linear kernel does. Micro at the decoder shape **33.15 → 4.12 ms (8.0x,
10.56 TFLOP/s)**; in-app sdpa **4.74 → 0.57s**.

| waves | BK=32 | 64 | 128 |
|------:|------:|---:|----:|
| 4 | 4.80 | 5.71 | 7.37 |
| 8 | 4.28 | 4.92 | LDS |
| **16** | **4.12** | LDS | LDS |

Wider key tiles lose here, unlike d128: at d64 a BK=128 tile is 4 waves' worth
of LDS traffic per WMMA step and the staging stops overlapping. 16 waves (a
256-row query block) wins because seq 2304 still gives 9 blocks x 32 heads =
288 workgroups, enough to fill 20 WGPs.

Gated to `sequence >= 256`, which keeps every short-sequence f32 attention
bit-exact so the bf16 operand rounding can only ever reach the decoder. PSNR
against the exact path is **46.96 dB** (y; 48.1 average) — the same sample. This
was re-measured against reproducible runs after the tail fix; so was the gated
f32 linear's **46.95 dB**.

## Audio VAE conv1d: 16-thread workgroups and a weight re-read per time step

The audio decoder's 136 conv1d calls were 3.67s at shapes that total only ~60
GFLOP. Two things were wrong with `h3_conv1d_f32_kernel`'s launch: the block is
`dim3(16,1,1)`, half a wave, and at the decoder's narrowest layers (C=8) only 8
of those 16 lanes have an output channel to compute; and one thread owns one
output element, so every weight is re-fetched once per time step. At C=512 that
is 2.1 GB of weight traffic for one call.

The shapes are all channel-preserving with `C * L` pinned at 236800 (C from 8 to
512, taps 3/7/11, dilation 1/3/5), so one kernel covers them: a block owns BC
output channels over `BY * TT` time steps and stages the weight slice in LDS, so
each weight crosses the bus once per 32 time steps instead of once per one.
Threads are (channel, time) with the channel fastest and each thread's TT steps
strided by BY, so every step writes one contiguous run of the time-major output.
Input stays in global on purpose: the whole activation is under a megabyte, so
it is L2-resident, and a wave spanning one time step reads it as a broadcast.

BC follows the channel count (8/16/32/64 with BY 32/16/8/4, always 256 threads),
which keeps the narrow layers off the 8-of-16-lanes cliff. Conv 3.67 -> **0.35s
(10.5x)**, and the phase stops being GPU-bound entirely: wall 4.25 -> **0.92s**.

The accumulation order is transposed to match the weight layout ((in, k) instead
of (k, in)), so results can differ at f32 rounding; on the test shapes they come
out bit-identical, and the audio decoder is terminal, so the video cannot move.

## Scoreboard (fox-fast, 11 DiT evals)

| Version | denoise GPU | linear | sdpa | other | Euler wall |
|---------|------------:|-------:|-----:|------:|-----------:|
| day-8 final (same-binary A/B) | 54.27 | 24.18 | 25.44 | 4.66 | 63.31 |
| WMMA flash, waves=4 | 31.80 | 23.45 | 3.78 | 4.55 | 40.36 |
| WMMA flash, waves=8 | 31.12 | 23.41 | 3.16 | 4.55 | 39.75 |
| + qkv-rope vectorized | **28.45** | **23.34** | **3.26** | **1.85** | **37.00** |
| day-9 final (all fixes) | 28.44 | 23.36 | 3.23 | 1.85 | 36.63 |

Day-9 final, whole-run phases (fox-fast): text encoder 31.7s wall / 1.8 GPU,
DiT load 71.4 / 6.2, Euler 36.6 / 28.4, audio VAE 0.94 / 0.46, video VAE
9.87 / 5.12. Total GPU 42.0s, of which denoise is 68%.

M5 Max published denoise wall (same knobs): **16.69s**. HIP denoise GPU now
**1.70×** that, was 3.25×. SDPA alone is **3.9× faster than Spark's 12.85s**;
the whole remaining gap to Spark is linear (23.34 vs 7.89).

Denoise is now **82% linear**, 11% sdpa, 7% everything else.

## P4: with the GPU work cut in half, the run is a weight-loading benchmark

The day-9 kernel work took total GPU time to 42s, but an end-to-end fox render
still takes 80–120s. Instrumenting `hipHostMalloc` and the file reads separately
(`H3_PROFILE`, new `weight-load` line) showed where the rest goes:

| Phase | wall | GPU | bytes read | pinned |
|-------|-----:|----:|-----------:|-------:|
| Qwen text encoder | 27.4 | 1.8 | 46.9 GiB | 46.9 GiB |
| DiT load | 60.9 | 3.2 | 26.0 + 30.8 GiB | 78.7 GiB |
| audio VAE | 0.9 | 0.5 | 0.24 GiB | 0.54 GiB |
| video VAE | 10.4 | 5.1 | 9.0 GiB | 9.5 GiB |

The pipeline reads ~98 GiB per run against 30 GiB of RAM, so nothing stays in
page cache and every run pays full disk cost. Two probes set the ceiling
(`tests/probe_read.hip`, `tests/probe_alloc.hip`):

- Cold reads scale to about **2.2 GiB/s** and flatten there. Pinned and ordinary
  heap destinations read at the same speed, so pinned memory is not the problem.
- `hipHostMalloc` runs at 12 GiB/s on an idle machine but only **2 GiB/s** once
  the box is under memory pressure, and it gets slower the more you hold.

### The AdaLN precompute was 50 blocking 520 MiB reads

`h3_dit_schedule_precompute` reads `blocks.N.adaln_proj.linear.weight` — 96768 ×
2688 bf16, **520 MiB per block** — runs a skinny projection whose GPU cost is a
rounding error, submits (which synchronizes the stream), frees, and only then
starts the next read. 26 GiB moved at single-stream speed with the drive idle
during every submit.

Reading ahead on worker threads (`adaln_slot_*`, depth `H3_ADALN_PREFETCH`)
turns it into a pipeline. Interleaved A/B, three pairs, cold cache:

| depth | AdaLN precompute |
|------:|-----------------:|
| 0 (serial) | 24.4s |
| 2 | 19.0s |
| 4 | **17.4s** |
| 6 | 17.5s — no further gain |

Depth 4 is the default; each slot holds a 520 MiB weight, so the ring costs
about 2 GiB.

### Read concurrency is per tensor, and the isolated probe was misleading

The probe says one tensor wants 16 streams. That is the wrong unit: the layer
loaders already run four tensors at once, so 8 per tensor is 32 outstanding
reads. Measured on the core load stage, interleaved:

| streams per tensor | core load |
|-------------------:|----------:|
| 4 | 33.9s |
| 8 | **30.5s** |
| 16 | 35.0s |
| 32 | 32.9s |

8 stays the default (`H3_PREAD_THREADS`). Capping streams *globally* across
loaders was far worse — 155s versus 80s end to end — because the first caller
reserved the whole budget and left the others at single-stream speed.

### Recycling pinned buffers, and why hipHostFree was load-bearing

A phase streams several times its peak live footprint through identically sized
tensors (the DiT allocates 78.7 GiB but never holds more than 15 GiB), so freed
pinned blocks are recycled by exact byte size instead of being unmapped.

The first version changed the rendered video. `hipHostFree` implicitly
synchronizes, so freeing a buffer that queued kernels were still reading was
safe; a free list is not, and two runs disagreed. Freed blocks now wait in a
`pending` list and only become reusable after a stream synchronization, with a
fast path when `hipStreamQuery` says the stream is already idle — which is the
case that actually hits, since layer loops free right after a submit. Fresh
pinned pages read as zero, so a recycled block is cleared unless the caller
overwrites all of it (weight loads do, and say so).

This removes most of the pinning work — text encoder 10.0s → 0.7s at 87%
recycled — but **does not move wall time**, because pinning already overlapped
the reads. It is kept for the CPU-time reduction, capped at 3 GiB (12% of RAM),
which is where recycling saturates; 6 GiB and 10 GiB were slower, since held
pinned pages are unreclaimable and compete with page cache.

### Video VAE: load the next block while the GPU runs this one

The decoder loaded all 36 blocks (9.5 GiB) before submitting any work, then ran
5.1s of GPU. A loader thread now publishes blocks as they arrive and
`run_resident_tile` waits per block. Phase wall 10.4–11.5s → **8.4–9.3s**. Opt
out `H3_VAE_LOAD_BLOCKING=1`.

### End to end

Interleaved A/B of the whole load pipeline against the serial arm
(`tools/bench_ab_io.sh`), best and worst pairs: 122.1 → 80.3s and 133.8 → 78.4s,
against a 93.4 → 107.2s pair that went the other way. Wall time on this box
drifts by 25% or more between sessions as page-cache state changes, so only the
interleaved per-stage numbers above are trustworthy. Every configuration
produced a **bit-identical** MP4 (`md5 1731f95c…`).

Remaining: the run is disk-bound at ~1 GiB/s against a 2.2 GiB/s drive, and the
only way to go substantially faster is to read fewer bytes — caching the INT8
quantized core (30.8 → 15.4 GiB) or the AdaLN projection outputs (26 GiB → 97
MiB, keyed by the sigma schedule). Both write derived artifacts to disk, so they
are left for a decision rather than taken unilaterally.

*(The "30 GiB of RAM" this section reasons from is only half the machine. See
the next section, which invalidates the pin-cache sizing above.)*

## P5: the 30 GiB this box was tuned for is only a quarter of its memory

Every allocator decision in P4 came from `free -g` reporting `total 30`, and the
pin cache was deliberately kept small on that basis. The machine actually has
128 GB of unified memory. The BIOS splits it and `free` only ever sees one side:

| pool | size | evidence |
|---|---:|---|
| host system RAM | 30.98 GiB | `/proc/meminfo` `MemTotal` 32481828 kB |
| GPU VRAM carveout | 96.00 GiB | `mem_info_vram_total` 103079215104 |
| …of it CPU-visible | 96.00 GiB | `mem_info_vis_vram_total`, identical |
| GTT (host pages the GPU may map) | 15.49 GiB | `mem_info_gtt_total` |

`rocm-smi --showmeminfo all` reports the same three numbers, and `radeontop -d -
-l 1` agrees on utilization (`vram 0.89% 875.89mb, gtt 0.51% 80.47mb` at idle).
96.00 + 30.98 = 126.98 GiB, i.e. the 128 GB on the box's spec sheet. So the
"30 GiB" was right about host RAM and wrong about the machine: **three quarters
of memory is a carveout only the GPU can allocate from, and it was empty.**

One P4 conclusion survives and one does not.

- The page cache really is confined to ~31 GiB minus the process, so 98 GiB of
  weights per run can never be cached and the cold-read floor stands. Nothing
  about the read path changes.
- But `hipHostMalloc` was never losing to "memory pressure" in the abstract. It
  page-locks out of the *small* pool, and every pinned byte is a byte the page
  cache cannot have. `hipMalloc` draws on the *large* pool that nothing else can
  touch. That is the whole explanation for 12 GiB/s idle versus 2 GiB/s loaded,
  and for why a bigger pin cache measured worse.

`tests/probe_stage.hip` allocates the DiT layer shapes both ways and fills each:

| footprint | pinned destination | device + staged upload |
|---|---|---|
| 5.74 GiB | alloc 22.48 GiB/s, fill 23.30 | alloc **293.50**, fill 24.18, upload 49.64 |
| 22.97 GiB | alloc **1.61 GiB/s**, fill 21.90 | alloc **90.98**, fill 24.62, upload 52.08 |

Pinning falls 14× between 5.7 and 23 GiB, which is exactly where it becomes a
large fraction of the 31 GiB pool. Device allocation does not notice, because
23 GiB is a quarter of its pool.

### Weights belong in the carveout

`hipMalloc` memory has no host alias here: `hipPointerGetAttributes` reports
`type=2` with a null `hostPointer`, so it cannot be pread into. (`hipMallocManaged`
*is* CPU-addressable at 25 GiB/s, and allocates 9× faster than pinned, but
`mem_info_vram_used` never moves when you fill 23 GiB of it — it is still host
memory, so it solves the rate problem and not the capacity one. Rejected on that
basis.) A device weight therefore needs a bounce: pread into a pinned staging
buffer, `hipMemcpyAsync` to the device, release the staging buffer.

The trade is lopsided. The staging buffer is page-locked once and then recycled,
so the per-tensor cost becomes one 30–50 GiB/s DMA instead of one 1–2 GiB/s
page-lock — and the pinned pool's job shrinks from "recycle the whole streamed
working set" to "hold the few reads in flight", which it does at 80–98%.

The refactor stays in one file because `struct h3_gpu_tensor` is private to
`backends/h3_gpu_hip.c`; `h3_gpu.h` only forward-declares it. Of 313
dereferences of the tensor data pointer, all but twelve are kernel arguments,
which do not care whether the pointer is device or pinned. The twelve that care
are the allocator, the two file-read entry points, the free path and the eight
accessors, and each now tests a `device` flag. Two invariants replace the ones
the pinned pool needs:

- `hipFree` synchronizes the device exactly as `hipHostFree` does, so a device
  weight needs no deferred-retirement queue — the hazard that made the first
  pinned free list produce wrong video does not exist here.
- A staging buffer is safe to reuse the moment the upload stream drains, because
  the DMA has already read it. It carries no dependency on queued compute, so it
  goes straight to the ready list rather than through `pending`.

Uploads use a dedicated `hipStreamNonBlocking` stream, so synchronizing to
release staging never stalls compute and never orders a loader thread against
the GPU.

Coverage came in two steps, and the profile line grew `pinned=`/`device=` byte
counters to show it. Marking only the file-loaded weights left the DiT still
page-locking 27.7 GiB, which turned out to be the INT8 quantized weights — the
phase's actual resident footprint, produced by the quantize kernel and read by
the GEMMs, never touched by host code. Those moved too. What is left pinned in
the DiT is 15.1 GiB of activations, which the host reads and writes every
denoise step and which cost 2.4–6.4s to allocate; leave them alone.

One call site is converted but unmeasured: `allocate_stream_slot`, the SSD
streaming mode's per-layer buffers. That mode is only reachable from the REPL
`ssd-streaming on` command, so the fox benchmark never touches it. It is safe
either way — the accessors bounce whatever they are given — but its benefit is
inferred from the resident path, not measured.

### What it bought

`tools/bench_device_weights.sh`, three interleaved pairs (phase wall, seconds):

| pair | arm | text | DiT | video VAE | AdaLN | core | DiT pin |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | device | 31.1 | **40.4** | 9.95 | 20.3 | 12.3 | **3.9** |
| 1 | pinned | 31.9 | 69.1 | 9.64 | 20.1 | 42.8 | 71.8 |
| 2 | device | 30.1 | **48.4** | 9.81 | 19.7 | 21.6 | **2.6** |
| 2 | pinned | 30.4 | 65.5 | 8.92 | 19.2 | 39.6 | 65.4 |
| 3 | device | 30.1 | **39.1** | 14.33 | 11.3 | 21.4 | **2.5** |
| 3 | pinned | 36.0 | 67.4 | 8.75 | 20.0 | 41.0 | 66.3 |

Medians: DiT wall **67.4 → 40.4s**, DiT pinning **66.3 → 2.6s**, core load stage
**41.0 → 21.4s**. End to end that is roughly **112 → 85s**. All six runs are
bit-identical (`md5 1731f95c…`).

The win is entirely the DiT, and the reason is instructive: every other phase was
*already* fine. The text encoder recycled 87% of its pinned bytes and spent 0.7s
pinning, the video VAE 92.6% and 1.7s. Only the DiT streamed 78.7 GiB through a
3 GiB cache at a 45% hit rate, so only the DiT was actually page-locking at
scale. Text encoder (30.1 vs 31.9) and AdaLN (19.7 vs 20.0) are a wash, and the
video VAE is ~0.5s *worse* — its loader thread now synchronizes an upload it did
not previously need, against pinning it had already recycled away. That is inside
the noise and not worth a second code path.

A side effect worth naming: the INT8 quantize batch stopped costing anything.
It was billed 5.6s of the core stage, essentially all of it page-locking the
13.4 GiB of INT8 destinations; it is now **0.12–0.18s**.

Re-run after the rest of P5 landed (32 pread streams, no cross-block read-ahead,
bucketed staging), two more interleaved pairs — both arms carry the new tuning,
so this isolates device residency alone:

| pair | arm | text | DiT | video VAE | AdaLN | core | DiT pin |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | device | 21.7 | **37.2** | 8.54 | 11.8 | **16.6** | **6.4** |
| 1 | pinned | 31.2 | 70.9 | 11.01 | 17.6 | 46.4 | 63.6 |
| 2 | device | 29.8 | **43.1** | 10.96 | 18.9 | **17.7** | **2.4** |
| 2 | pinned | 31.1 | 58.2 | 8.87 | 17.1 | 34.9 | 50.9 |

Core load **40.7 → 17.2s** median, DiT wall **64.5 → 40.2s**, and summing the
four phase walls gives **100–115s pinned against 69–86s device**. The spread
inside each arm is the page-cache drift this box always has; the gap between
arms is larger than the spread in every pair. Ten renders across all of P5, in
both arms, produced `md5 1731f95c…`.

Worth stating plainly: reading 98 GiB off the drive is still the floor, and
nothing here changed a single byte of I/O. The whole gain came from stopping the
loaders competing with their own destination memory.

### Read concurrency reverses once page-locking is gone

P4 measured 8 pread streams per tensor beating 16 (30.5 vs 35.0s) and made 8 the
default. With weights in device memory that inverts, on the same harness:

| streams per tensor | core load stage (2 interleaved pairs) |
|---:|---|
| 4 | 31.4 / 31.9 |
| 8 | 23.0 / 22.9 |
| 16 | **19.9 / 19.6** |

Both measurements were right about their own build. When a loader thread spent
most of its wall page-locking, extra read streams only added contention for the
same host memory; with that gone the threads are genuinely waiting on the drive
and concurrency pays. This is the second time the *unit* of this measurement has
been the trap — first per-tensor versus per-process, now with-pinning versus
without.

AdaLN precompute agrees, at depth 4 (`tools/bench_adaln.sh`, arms now
`depth:pread`): 18.96/18.25s at 8, 17.81/19.49 at 16, **17.27/18.13 at 32**. So
32 is best on both stages in every pair and becomes the default. It is also the
cap, which is comfortable rather than awkward: the isolated read probe already
flattens between 16 and 32 (2.24 then 2.19 GiB/s), and four concurrent tensors
make this 128 outstanding reads.

### The cross-block read-ahead is now provably pointless — removed

Day-9 kept `core_prefetch` weakly, on the grounds that it overlapped block reads
with the INT8 quantize batch and "the quantize is only 6s of a 40s stage, so
there is little to hide behind". Device-resident INT8 destinations took the
quantize batch to **0.12–0.18s**, so there is now nothing at all to hide behind,
and the arms are indistinguishable:

| arm (pread 32→16, 2 interleaved pairs) | core load |
|---|---|
| read-ahead on | 19.219 / 19.249 |
| read-ahead off | 19.296 / 19.101 |

0.2% apart, in opposite directions. `core_prefetch`, its worker thread, its
join-on-failure path and `H3_DIT_NO_PREFETCH` are deleted. The read concurrency
that does matter is per tensor, inside `h3_hip_pread_all`.

### Staging blocks are bucketed, not exact-sized

The pinned pool matches by exact byte size, which is correct for a tensor whose
extent is fixed but needlessly strict for a staging buffer that only has to be
big enough. Qwen's layer shapes are all different, so exact matching missed and
page-locked on 7% of 46.9 GiB. Rounding staging requests to a 32 MiB grain
collapses the model's shapes onto a couple of dozen block sizes: recycling goes
audio VAE 53.3 → **98.4%**, video VAE 92.6 → **96.4%**, DiT 77.8 → **80.2%**, and
the text encoder's staging page-locking 20.3 → **14.5s** of thread time. Phase
wall does not move measurably — this is a CPU-time and jitter argument, like the
original pinned free list, not a wall-time one.

## The one pre-existing test failure

`make test` fails exactly one case, and it is not from this work:

```
FAIL attn.out exceeds abs 4 or rel 0.05
attn.out  max_abs=281.9  ref_abs=286  rel_l2=1.099
```

`tests/test_hip_real_dit.c` has not been touched in a week, the failure is
identical with every new feature disabled and with `H3_SDPA_WMMA=0`, and
rendering with `H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT=1` gives PSNR=inf
(bit-identical) against the default path. The production path is fine; the test
harness's comparison is miswired. `rel_l2` is still 1.099 after everything
above, i.e. unchanged, which is itself evidence the two are unrelated. Left
alone deliberately.

## Decisions

- **KEEP** tiled WMMA flash attention as the default d128 SDPA path
  (`h3_sdpa_bf16_wmma_d128_kernel<8,32>`). Opt out `H3_SDPA_WMMA=0`; forcing any
  wave Q-variant (`H3_SDPA_D128_Q1..Q5`, `H3_SDPA_Q6_SCALAR`) also falls back, so
  the old A/B knobs still work. Tile knobs `H3_SDPA_WMMA_WAVES` /
  `H3_SDPA_WMMA_BK`. Micro 45.0→**6.7 ms**. Fox s2 sdpa 3.54→**0.51**, denoise
  GPU 7.61→**4.49**. Fox-fast sdpa 25.44→**3.16**, denoise GPU 54.27→**31.12**.
  Numerics match the Q6 path at bf16 rounding level (max abs 1.5e-05); new test
  `test_sdpa_head128_tiled` covers seq 192/100/17 for the query-block clamp.
- **REJECT** WMMA tile shapes other than 8×32 (see sweep above).
- **KEEP** vectorized wave `h3_qkv_rope_bf16_d128_kernel` for head_dim=128 with
  `rope_half % 4 == 0`. Opt out `H3_QKV_ROPE_SCALAR=1`. Per call
  5.98→**0.72 ms**. Fox-fast denoise other 4.55→**1.85**, denoise GPU
  31.12→**28.45**, Euler wall 39.75→**37.00**. New test `test_qkv_rope_d128`
  covers rope_half 48 and 64 against the CPU reference (the pre-existing
  qkv-rope tests use head_dim=4 and only reach the scalar path).
- **REJECT** custom INT8 GEMM for the DiT linear shapes, for now: hipBLAS
  already averages 62% of the 57 TOPS ceiling across the four shapes.
- **KEEP** `h3_linear_f32_wmma_kernel<256,128,32,32,64,8>` for f32 linear with
  `rows >= 256 && input_dim >= 512 && output_dim >= 512 && input_dim % 32 == 0`.
  Opt out `H3_F32_WMMA=0`, tile select `H3_F32_WMMA_CFG` (default 3). Video VAE
  linear 10.56→**4.26s**, VAE GPU 15.82→**9.51**, VAE wall 21.07→**15.05**.
  Same sample as the exact path (PSNR 46.9 dB); denoise unaffected.
- **KEEP** `h3_sdpa_f32_wmma_d64_kernel<16,32>` for f32 SDPA with
  `head_dim == 64 && sequence >= 256`. Opt out `H3_SDPA_F32_WMMA=0`, tile knobs
  `H3_SDPA_F32_WMMA_WAVES` / `H3_SDPA_F32_WMMA_BK`. Micro 33.15→**4.12 ms**,
  video VAE sdpa 4.74→**0.57s**, VAE GPU 9.51→**5.15**. Same sample (PSNR
  46.96 dB). New test `test_sdpa_f32_d64_tiled` covers seq 256 and 300 (the
  latter for the query-block clamp) at max abs 5.6e-06.
- **KEEP** `h3_conv1d_f32_tiled_kernel` for f32 conv1d with `kernel <= 64`
  (BC/BY chosen from the channel count, TT=8). Opt out `H3_CONV1D_TILED=0`.
  Audio VAE conv 3.67→**0.35s**, phase wall 4.25→**0.92s**. New test
  `test_conv1d_f32_audio_shapes` covers multi-slice input channels, a channel
  count that is not a whole number of tiles, and dilation 3/5.
- **FIX** the tiled-SDPA tail so each output row has exactly one writer
  (d64 masks, d128 restricts stores to its unclamped range). Without it the
  whole pipeline was irreproducible run to run and every PSNR comparison had a
  46.96 dB floor. Covered by `test_sdpa_tiled_repeatable`.
- **REJECT** f16 instead of bf16 as the WMMA operand type. Its 1.3e-5 error is
  8x better than bf16's 1.1e-4, but `v_cvt_f16_f32` is dearer than the bf16 bit
  shift and it measured 3.3-4.7 TFLOP/s against bf16's 10-12 on the large tiles.
- **REJECT** a 3-pass hi/lo bf16 split for near-f32 accuracy. It reaches 2^-16
  relative but runs at 1.9-3.4 TFLOP/s, i.e. slower than the r128 f32 kernel it
  would replace.
- **REJECT** optimizing `h3_linear_bf16_kernel`: I/O-bound phases only.
- **KEEP** AdaLN precompute read-ahead at depth 4 (`H3_ADALN_PREFETCH`, 0
  disables). 24.4→**17.4s** on interleaved A/B; depth 6 adds nothing.
- **KEEP** the video VAE loader thread (`H3_VAE_LOAD_BLOCKING=1` disables).
  Phase wall 10.4–11.5→**8.4–9.3s**.
- ~~**KEEP** the DiT core cross-block read-ahead~~ — **REMOVED** in P5. It was
  kept weakly (42.6→38.9s, favourable in two of three pairs) because it hid
  reads behind the INT8 quantize batch. Device-resident INT8 destinations took
  that batch from 5.6s to 0.12s, and the arms then measure 19.30 vs 19.22s. Gone
  along with `H3_DIT_NO_PREFETCH`.
- ~~**KEEP** 8 pread streams per tensor~~ — superseded by P5: **32** is the
  default. 8 was correct while loader threads were busy page-locking; with that
  gone, core load is 22.9s at 8, 19.5 at 16 and **18.4 at 32**, and AdaLN agrees.
- **REJECT** a global cap on pread streams across loaders: 155s versus 80s end to
  end, because the first caller took the whole budget.
- **KEEP** the pinned-buffer free list (`H3_PIN_CACHE_GIB`, 0 disables). In P4 it
  held streamed weights at a 3 GiB cap for CPU time, not wall time —
  text-encoder pinning 10.0→**0.7s** at 87% recycled, wall unchanged because
  pinning already overlapped reads, and larger caps measured slower. Since P5 it
  holds only staging buffers, so the default is **6 GiB** and the RAM clamp 25%
  rather than 12%: the 15 GiB of live weights it used to keep pinned are out of
  host RAM entirely, which is what the old clamp was protecting page cache from.
- **KEEP** device-resident weights (`H3_DEVICE_WEIGHTS=0` disables), i.e.
  `hipMalloc` from the 96 GiB VRAM carveout plus a recycled pinned staging buffer
  per read, for every buffer only the GPU touches: file-loaded weights
  (`h3_gpu_tensor_new_bf16_device`) and the INT8 quantized weights the quantize
  kernel produces (`..._i8_device` / `..._f32_device`). Interleaved over three
  pairs, DiT wall **67.4→40.4s**, DiT pinning **66.3→2.6s**, core load
  **41.0→21.4s**, INT8 quantize batch **5.6→0.12s**; roughly **112→85s** end to
  end. Bit-identical (`md5 1731f95c…`) in every arm.
- **REJECT** `hipMallocManaged` as the allocator instead. It is CPU-addressable
  at 25 GiB/s and allocates 9x faster than pinned (14.05 vs 1.61 GiB/s at
  23 GiB), which would have needed no refactor at all, but `mem_info_vram_used`
  does not move when 23 GiB of it is filled: it is host memory, so it fixes the
  allocation rate and leaves the 31 GiB capacity problem untouched.
- **REJECT** device residency for activations. The accessors handle it, but the
  gain is nil (activations are allocated once and reused) and the DiT reads and
  writes them from the host every denoise step.
- **KEEP** 32 MiB bucketing for staging block sizes, on CPU time rather than
  wall: recycling 53.3→98.4% (audio VAE), 92.6→96.4% (video VAE), 77.8→80.2%
  (DiT), text-encoder staging page-locks 20.3→14.5s of thread time. Phase wall
  does not move measurably.

## Log

Elapsed is approximate minutes from the 22:16 start.

| +min | Action | Result |
|-----:|--------|--------|
| 0 | Start day-9 10h; P0 = SDPA architecture | rocWMMA 2.2 + CK present, gfx1151 supported |
| 6 | Probe WMMA fragment layouts | acc row = `2*i+(l>>4)` — register-only rescale viable |
| 16 | Tiled WMMA flash kernel (waves=4, BK=32) | tests ok; micro 45.0→**8.0 ms** |
| 25 | fox s2 same-binary A/B | sdpa 3.54→**0.51** / GPU 7.61→**4.49** — KEEP |
| 32 | fox-fast WMMA vs Q6 (same binary) | GPU 54.27→**31.80** / sdpa 25.44→**3.78** |
| 38 | Make WMMA default; tile sweep | 8×32 best: 8.0→**6.7 ms** |
| 44 | fox-fast waves=8 | GPU **31.12** / sdpa **3.16** / Euler wall **39.75** |
| 49 | WMMA peak probe (bf16/f16/int8) | all ~57 T(FL)OP/s — no INT8 bonus on RDNA 3.5 |
| 53 | Per-shape hipBLAS INT8 sweep | avg 62% of peak — park custom GEMM |
| 56 | rocprofv3 per-kernel profile | `qkv_rope` 5.98 ms/call, 9× off bandwidth |
| 60 | qkv-rope cooperative LDS | 5.98→3.22 ms; barriers now dominate |
| 64 | qkv-rope vectorized wave | **0.72 ms**; tests ok — KEEP |
| 72 | fox-fast both fixes | GPU **28.45** / other **1.85** / Euler wall **37.00** |
| 78 | Whole-process rocprofv3 | video VAE linear is now 37% of all GPU time |
| 84 | Phase wall vs GPU | text encoder + DiT load are I/O bound — drop bf16 linear |
| 92 | hipBLAS f32/bf16/f16 probe | only bf16→bf16 is tuned (21.7); bf16→f32 is 4.8 |
| 105 | WMMA f32 linear + tile sweep | cfg 3 (32x64/wave, no spill) **2.4x** over r128 |
| 118 | fox s2, ungated | VAE linear 10.56→4.26 but **different sample** (19.9 dB) |
| 128 | Size gate on rows/dims | PSNR **46.9 dB**, same sample — KEEP |
| 141 | Tiled WMMA f32 d64 SDPA + sweep | micro 33.15→**4.12 ms**; 16×32 best |
| 150 | fox s2 same-binary A/B | VAE sdpa 4.74→**0.57** / VAE GPU 9.51→**5.15** |
| 165 | Time-blocked conv1d | audio conv 3.67→**0.35s**, phase wall 4.25→**0.92** |
| 175 | Two same-config runs differ | md5 unstable — PSNR 46.96 was a **noise floor** |
| 190 | Audio identical, video not | divergence is inside the video decoder only |
| 215 | In-process self-check | ~0.1% of elements differ on identical inputs |
| 225 | Differing rows = 2257..2271 | exactly the clamped query-block overlap |
| 240 | Unique-writer fix (d64 + d128) | two runs bit-identical; no perf cost |
| 250 | Re-measure with real baselines | d64+conv **46.96 dB**, f32 linear **46.95 dB** |
| 265 | fox-fast day-9 final | denoise GPU **28.44** / Euler wall **36.63** / VAE 9.87 |
| 272 | Phase wall vs GPU, whole run | 98 GiB read per run at ~1 GiB/s — GPU is no longer the job |
| 280 | Split pin vs read timing | pinning 10.0s of the text encoder's 27.4s wall |
| 288 | `probe_read` / `probe_alloc` | disk tops out 2.2 GiB/s; hipHostMalloc 2 GiB/s under pressure |
| 300 | Pinned free list, first cut | **md5 unstable** — hipHostFree was implicitly syncing |
| 312 | Deferred retire + idle fast path | bit-identical; pinning 10.0→**0.7s**, wall unchanged |
| 325 | AdaLN is 50 blocking 520 MiB reads | read-ahead depth 4: 24.4→**17.4s** |
| 340 | Video VAE loader thread | phase wall 10.4→**8.4–9.3s** |
| 352 | Core read-ahead A/B | 42.6→38.9s — real but small; quantize is only 6s of 40 |
| 365 | pread stream sweep in-app | 8 beats 16 (30.5 vs 35.0) — probe measured the wrong unit |
| 372 | Global stream budget | 155s vs 80s — reverted |
| 380 | Final fox-fast | wall **87.1s**, md5 `1731f95c` unchanged since day-8 |
| 395 | `radeontop` / `mem_info_*` / `rocm-smi` | 96 GiB VRAM carveout + 31 GiB host — the "30 GiB" was one half |
| 402 | `probe_stage` pinned vs device at 23 GiB | pinning **1.61 GiB/s**, hipMalloc **90.98** — the collapse is the small pool |
| 408 | `probe_devhost` / `probe_managed` | hipMalloc has no host alias; managed is host-backed — staging it is |
| 415 | Device weights for file-loaded tensors | DiT still pinned 27.7 GiB — the INT8 weights |
| 421 | Device weights for INT8 destinations too | DiT pin 40.6→**2.8s**, quantize 5.6→**0.16s** |
| 430 | Interleaved A/B, 3 pairs | DiT **67.4→40.4s**, core **41.0→21.4s**, md5 unchanged |
| 445 | pread sweep, post-device | 4:31.6 8:22.9 16:19.5 **32:18.4** — day-9's 8 was measuring pinning |
| 452 | AdaLN pread sweep at depth 4 | 8:18.6 16:18.7 **32:17.7** — 32 is best on both stages |
| 458 | Core read-ahead re-A/B | 19.30 off vs 19.22 on — **removed** |
| 464 | 32 MiB staging buckets | recycling 53→98% / 93→96%; pin thread time 20.3→14.5s |
| 470 | `make test` | only the pre-existing `real_dit_smoke` (rel_l2 1.099, unchanged) |
