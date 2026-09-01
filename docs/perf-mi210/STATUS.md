# MI210 12h loop (device activations and long-sequence SDPA)

No MI300. This box has four MI210s; bind with `H3_HIP_DEVICE`. Do not run two
weight-streaming T2VA jobs on the same NVMe at once.

| GPU | Role |
|-----|------|
| 0 | Compile, kernel microbench, interactive debug |
| 1 | Quality gates (encoder, fox-s2, fox-fast) |
| 2 | Long T2VA / A/B |
| 3 | Spare A/B |

Baseline after previous commit: fox-fast **45 s**, fox-s2 **17 s**.

## KEEP

Text / AdaLN / audio / HIP temps onto HBM (same bug as VAE/DiT). fox-fast vs `fox-fast-on-devact.mp4`: **PSNR inf**.

| | Before this loop | After |
|--|------------------|-------|
| fox-fast E2E | 45 s | **23 s** |
| fox-s2 E2E | 17 s | **12 s** |
| Text encoder GPU | 6.0 s | **0.7 s** |
| AdaLN precompute | 4.7 s | **1.4 s** |
| Denoise wall / GPU | 24.7 / 17.5 s | **12.4 / 12.2 s** |
| Audio VAE | 3.5 s | **0.6 s** |
| Video VAE | 2.7 s | 2.7 s (already device) |

AdaLN submit batching (4 GEMMs per sync) did not move wall; load-bound. Left in as fewer round-trips.

## Long 15-second website example

Exact prompt and knobs from `https://alexhegit.github.io/h3-hip.c/`:
864x480, 362 frames, 20 steps, 45 layers, reuse 2, seed 42.

| Hardware | E2E | DiT denoise | Video VAE |
|----------|-----|-------------|-----------|
| Strix Halo gfx1151, v0.9.0 | 45.0 min | 40.4 min | not reported |
| MI210 gfx90a, pre-flash | 50 min 14.58 s | 48 min 22.17 s | 1 min 37.28 s |
| MI210 gfx90a, CDNA MFMA flash | **12 min 33.29 s** | **10 min 48.29 s** | **1 min 32.75 s** |

The pre-flash MI210 result was 1.116x the Halo wall time. The final result is
**4.00x faster than the MI210 baseline and 3.58x faster than Halo**. Output
verification: 864x480, 362 frames, 24 fps, 15.084 s. Sampled early/middle/late
frames preserve the requested office, typing close-up, monitor fox, warm
lighting, and pullback composition.

Worker threads used to allocate on GPU 0 regardless of `H3_HIP_DEVICE`, so
encoder roundtrip illegal-accessed on GPU 1/3. Allocation, upload, and submit
now rebind the calling thread. GPU 1 and GPU 3 encoder roundtrips pass in
parallel with GPU 2 kernel tests.

The enabling change is a gfx90a hipBLAS path that computes QK on BF16 matrix
cores, keeps softmax and PV in F32 for the fox quality gate, and chunks sequences
above 8192 into 128-query blocks. At `seq=12000, heads=56, d=128`, this path
runs in 317.5 ms instead of falling back to the wave kernel.

fox-fast after this change is **22.33 s** E2E, with 11.80 s denoise and
24.48 dB PSNR versus the prior 22.88 s clip.

## 4h loop (page-cache warmup + SDPA cleanup)

GPU 1 fox-s2 baseline at loop start: **11.84 s**. After this loop: **11.69 s**,
PSNR **inf** versus `/tmp/h3-mi210/fox-s2-base-4h.mp4`.

GPU 1 fox-fast: **22.51 s** (denoise 11.88 s, linear 7.29 s, sdpa **3.67 s**).
PSNR **24.48 dB** versus `fox-fast-p1.mp4` (same KEEP as the prior matrix-core
clip) and **inf** versus `fox-fast-final-sdpa.mp4`.

KEEP:

- **Page-cache warmup** of audio/video VAE `*.safetensors` on a host thread
  during DiT denoise (`h3_weight_directory_warmup`, opt out `H3_WEIGHT_WARMUP=0`).
  No GPU alloc. Warm fox-s2 does not move VAE wall (read is already ~10 GiB/s
  from cache); this is for cold NVMe.
- **Direct hipBLAS SDPA**: drop the unused K copy, scatter F32→BF16 straight
  into SHD (skip HSD→SHD transpose), vectorize softmax `float4`. seq=1920
  **6.2 ms**; seq=12000 **294 ms** (was 317.5 ms).
- **Online K-tile** for seq>8192 with default `H3_SDPA_BF16_KTILE=4096`.
  seq=12000: **274 ms** vs 294 ms full-K. seq=40000: **3146 ms** vs 3162 ms
  (tie). Override with `H3_SDPA_BF16_KTILE`.

## Rejected

Loading the resident video VAE concurrently with DiT denoise was **12.12 s**
versus **11.92 s** serial on fox-s2. GPU/upload contention erased the I/O
overlap, so the change was fully reverted. Outputs were pixel-identical.

Online K-tile **1024** (and 512/2048) is slower than full-K at seq=12000
(353/481/380 ms vs 294 ms) and much slower at seq=40000 (3870 vs 3162 ms).
Do not use those sizes as defaults.

## Still open

fox-s2 wall is serial-ish: text I/O ~2.6 s + DiT load ~3 s + denoise 1.9 s + video VAE 2.7 s. Stretch 10 s needs I/O overlap, not kernels. fox-fast leftover is denoise linear **7.3 s** + sdpa **3.7 s**. 15 s remains GPU-bound on long-seq SDPA (~3.1 s/head-batch at seq=40k); real CDNA MFMA flash that does not materialize QK is still the gap to Halo 45 min.

## 4h loop 2 (BF16 hipBLAS linear)

HBM microbench (device tensors, DiT fox shapes):

| GEMM | INT8 hipBLAS | BF16 hipBLAS |
|------|-------------:|-------------:|
| FC2 1920×14336×5376 | 2.3 ms (131 TFLOP/s) | 2.6 ms (112 TFLOP/s) |
| FC1 1920×5376×28672 | 4.3 ms (138 TFLOP/s) | 4.3 ms (138 TFLOP/s) |
| QKV 1920×5376×16128 | — | 2.3 ms (145 TFLOP/s) |

INT8 GEMM is already near peak. The remaining fox-fast linear time is mostly those FLOPs plus activation quantize. Skipping INT8 (keep BF16 weights, hipBLAS BF16 GEMM) drops quantize:

| | INT8 default (prior) | CDNA BF16 default |
|--|---------------------:|------------------:|
| fox-s2 E2E | 11.69 s | **11.28 s** |
| fox-fast E2E | 22.51 s | **21.01 s** |
| fox-fast denoise linear / sdpa | 7.29 / 3.67 s | **6.58 / 3.57 s** |
| fox-fast peak HBM | 19.7 GiB | **33.0 GiB** |

Quality: fox-s2 **30.7 dB** vs INT8 `fox-s2-base-4h.mp4`; fox-fast **22.0 dB** vs `fox-fast-4h.mp4` and **21.8 dB** vs `fox-fast-p1.mp4`. Mid-frame still a recognizable fox (KEEP vs the 24.5 dB SDPA bar). Pixel-exact INT8 is `H3_INT8_MLP=1`.

KEEP:

- **hipBLAS BF16 linear** on CDNA (`H3_BF16_HIPBLAS=0` falls back to the 16×16 kernel, 90 ms vs 2.6 ms on FC2).
- **NAX FC1 scratch reuse** (stop hipMalloc of the SwiGLU temp every layer).
- **INT8 epilogue `float4` scale load**.
- **Default `h3_gpu_has_int8_mlp()` off on CDNA.** RDNA still defaults INT8. Force INT8 anywhere with `H3_INT8_MLP=1`.

Encoder roundtrip on GPU 1 still left-red / right-blue.

## 10h loop (gfx90a MFMA flash attention)

KEEP: a native wave64 online-softmax kernel for BF16 QK and FP16 PV, with F32
accumulators. Eight waves share a 32-key K/V LDS tile across 128 query rows.
No score matrix is written to HBM. `H3_SDPA_CDNA_FLASH=0` restores the
hipBLAS/F32 fallback; `H3_SDPA_CDNA_FP16_PV=0` selects BF16 PV.

| seq, heads, d | hipBLAS + F32 scores/PV | CDNA flash | Speedup |
|---------------|------------------------:|-----------:|--------:|
| 1,920, 56, 128 | 6.1 ms | **2.0 ms** | 3.05x |
| 12,000, 56, 128 | 274.4 ms | **69.4 ms** | 3.95x |
| 44,700, 56, 128 | 3,942.3 ms | **949.3 ms** | 4.15x |

The full BF16 test suite passes. d128 reference checks have max absolute error
`1.53e-5` (seq 192) and all long-tail repeatability checks have zero differing
elements. FP16 PV is effectively free (+0.3% at seq 44,700) and improves the
fox-fast PSNR versus the BF16-linear/F32-PV reference from 25.53 dB (BF16 PV)
to **29.99 dB**. It is therefore the default.

End-to-end gates:

| Gate | Before flash | CDNA flash |
|------|-------------:|-----------:|
| fox-s2 E2E | 11.28 s | **10.87 s** |
| fox-fast E2E | 21.01 s | **18.56 s** |
| fox-fast denoise SDPA | 3.57 s | **1.08 s** |
| 15 s E2E | 3,014.58 s | **753.29 s** |
| 15 s denoise | 2,902.17 s | **648.29 s** |
| 15 s denoise SDPA | 2,721.21 s | **474.18 s** |

Rejected tiles at seq 44,700: waves/BK 4/32 1473 ms, 8/16 1275 ms,
8/64 1322 ms, 10/32 1421 ms, 12/32 1304 ms, and 16/32 1021 ms. Keep 8/32.

## Still open (after loop 3)

Long-video SDPA fell from 93.8% to 73.2% of denoise GPU time. BF16 linears are
now 24.9% (161.23 s) and SDPA is still 474.18 s, so further work should tune
MFMA occupancy/exp throughput rather than return to score materialization.
fox-s2 remains dominated by serial model I/O and VAE decode.

## Repro

```bash
H3_HIP_DEVICE=1 ./h3 --profile -d /home/alex/data/HF-MODELS/MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 --frames 22 --steps 20 --layers 45 --reuse 2 \
  -o /tmp/h3-mi210/fox-fast-p1.mp4

H3_HIP_DEVICE=1 ./h3 --profile -d /home/alex/data/HF-MODELS/MiniMax-H3 \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 --steps 2 --layers 35 --reuse 1 \
  -o /tmp/h3-mi210/fox-s2-p1.mp4
```
