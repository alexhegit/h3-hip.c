# Day-8 perf loop — STATUS

Started: 2026-08-24 08:02 CST  
Budget: ~11 hours  
Stop after: 2026-08-24 19:00 CST  
Git start: `852dbd0` (SDPA Q6)  
Baseline fox s2: denoise GPU **11.71s** (lin **7.20** · sdpa **3.87**)  
Baseline fox-fast: denoise GPU **85.26s** (lin **52.3** · sdpa **28.2**)  
Priority: INT8 via hipBLAS (not naive packing); then SDPA Q7  
Loop: 45m one-shot (`AGENT_LOOP_WAKE_perf_day8`) until 19:00; **unilab / gfx1151 only**  
Do not retry: fc1 t128, fused SwiGLU dual-B, flash, F32 256×64/BK16/LDS double-buffer/float4 LDS BK+4, mmap memcpy default, t128 skip-last-sync, launch_bounds(256,3), INT8 LDS_K BK+8, grouped k+=16, naive INT8 WMMA, SDPA Q7, Q6 pipe, Q6 K-unroll-2, Q6×2 L2, hipBLAS/hipBLASLt F32, hipBLASLt i8 fused scale, VAE d64 Q8 default, INT8 LDS quantize default, grouped hipBLAS unpacked lda=K, grouped hipBLAS per-group loop

## Scoreboard (fox s2)

| Version | denoise GPU | denoise linear | denoise sdpa |
|---------|------------:|---------------:|-------------:|
| night-5 `852dbd0` | 11.71 | 7.20 | 3.87 |
| hipBLAS INT8 (same-binary t128 A/B) | 8.99 | **4.50** | 3.85 |
| Q6 vec loads (same-binary scalar A/B) | 8.54 | 4.45 | **3.47** |
| grouped hipBLAS packed (same-binary r64 A/B) | **7.49** | **3.39** | 3.48 |
| t128 opt-out (`H3_INT8_T128=1`) | 11.61 | 7.15 | 3.81 |

Microbench M=1920 K=14336 N=5376: hipBLAS **11.0 ms / 26.9 TFLOP/s** vs t128 **18.8 ms / 15.7 TFLOP/s**.

## Scoreboard (fox-fast, 11 DiT evals)

| Version | denoise GPU | denoise linear | denoise sdpa |
|---------|------------:|---------------:|-------------:|
| night-5 `852dbd0` | 85.26 | 52.3 | 28.2 |
| hipBLAS INT8 | 65.46 | **32.72** | 28.14 |
| Q6 vec loads | 62.33 | 32.24 | **25.48** |
| grouped hipBLAS packed | **55.51** | **25.41** | 25.43 |

M5 Max published denoise wall (same knobs): **16.69s**. HIP denoise GPU now **3.33×** that (was 5.7×).

## Decisions

- **KEEP** hipBLAS INT8 GEMM (`hipblasGemmEx` i8→i32 + vector scale/SwiGLU epilogue) as default for DiT linear and fc1 SwiGLU. Opt out `H3_INT8_T128=1` (tiled t128) or `H3_INT8_HIPBLAS=0`; fc1 only `H3_INT8_FC1_HIPBLAS=0`. Fox s2 denoise GPU 11.61→**8.99s** (linear 7.15→**4.50**). Fox-fast denoise GPU 85.26→**65.46s** (linear 52.3→**32.72** · sdpa 28.14). vs M5 16.69s denoise wall: **3.92×** (was 5.7×). Tests pass.
- **REJECT** SDPA Q7 (micro 55.6 ms vs Q6 53.9).
- **REJECT** hipBLAS F32 linear (VAE shapes slower than tiled r128: 23.6 vs 13.8 ms on 1797×8192×2048).
- **REJECT** hipBLASLt fused outer-vec scale / i8→bf16 (0 algs on gfx1151).
- **REJECT** Q6 K/V software pipeline (micro 56.3 vs 53.9).
- **REJECT** Q6 K-unroll-2 (micro 53.3–54.1 vs 53.9; noise).
- **REJECT** hipBLASLt F32 (78 ms vs tiled 13.8 ms on VAE shape).
- **KEEP** vectorized BF16 add (uint2×4) and RMS-norm loads/stores. Tests pass; fox s2 other 0.632→0.627 (noise). No denoise claim.

- **REJECT** Q6×2 (two K sweeps per block for L2 reuse): micro 55–57 ms vs Q6 53.9.
- **KEEP** SDPA Q6 consecutive-4 vector loads (`uint2` K/V/Q/out). Opt out `H3_SDPA_Q6_SCALAR=1`. Micro 53.2→**43.5 ms**. Fox s2 same-binary: sdpa 3.80→**3.47**, denoise GPU 8.87→**8.54**. Fox-fast denoise GPU 65.46→**62.33** (sdpa 28.14→**25.48**). vs M5 16.69s: **3.73×**. Tests pass.
- **KEEP** VAE d64 SDPA consecutive `float2` loads on Q2 (default). Fox s2 video VAE sdpa 6.39→**6.07**, VAE GPU 17.44→**16.99**. Tests pass.
- **KEEP** VAE d64 Q4 (4 queries share K/V; opt out `H3_SDPA_D64_Q2=1`). Fox s2 video VAE sdpa 6.07→**4.74**, VAE GPU 16.99→**15.74**. Tests pass.
- **REJECT** VAE d64 Q8 as default (fox s2 video VAE sdpa 4.97 vs Q4 4.74). Opt-in `H3_SDPA_D64_Q8=1`.
- **REJECT** LDS-staged row quantize as default (`H3_INT8_QUANT_LDS=1`). Split: quantize is ~0.4 ms vs gemm ~10.5 ms; LDS made full linear 10.6→10.9 ms.
- **KEEP** grouped FC2 hipBLAS with pack-to-lda=G then `hipblasGemmStridedBatchedEx` + group-scale reduce. Opt out `H3_INT8_GROUPED_HIPBLAS=0` (custom r64). Micro M=1920 K=14336 N=5376 g=1024: 22.2→**10.9 ms**. Fox s2 same-binary: linear 4.45→**3.39**, denoise GPU 8.56→**7.49**. Fox-fast denoise GPU 62.33→**55.51** (linear 32.24→**25.41**). vs M5 16.69s: **3.33×**. Tests pass.
- **REJECT** grouped hipBLAS without packing (lda=K_full, ~21.8–22.1 ms) and per-group gemm loop (~32.4 ms).

## Log

| Time | Action | Result |
|------|--------|--------|
| 08:02 | Start day-8 11h | Next: rocWMMA/hipBLAS INT8 |
| 08:13 | hipBLAS i8 GEMM + scale | tests ok; 11.5 ms vs t128 18.9 |
| 08:18 | vec epilogue + fc1 SwiGLU gemm | tests ok; 11.0 ms |
| 08:21 | fox s2 hipBLAS | denoise GPU **8.99** / lin **4.50** / sdpa 3.85 |
| 08:24 | fox s2 t128 A/B | denoise GPU 11.61 / lin 7.15 — KEEP hipBLAS |
| 08:34 | fox-fast hipBLAS | denoise GPU **65.46** / lin **32.72** / sdpa 28.14 |
| 08:40 | SDPA Q7 / F32 hipBLAS / Q6 pipe / Lt fuse | all REJECT |
| 08:45 | Q6 K-unroll-2 / hipBLASLt F32 | REJECT |
| 08:50 | vec add+rms | tests ok; fox s2 GPU 8.95 (noise vs 8.99) |
| 08:55 | Q6×2 L2 reuse | 55–57 ms — REJECT |
| 17:47 | Q6 consecutive uint2 loads | micro 53.2→43.5; tests ok |
| 17:53 | fox s2 Q6 vec vs scalar | sdpa 3.80→**3.47** / GPU 8.87→**8.54** — KEEP |
| 18:03 | fox-fast Q6 vec | denoise GPU **62.33** / sdpa **25.48** |
| 18:19 | VAE d64 float2 loads | video VAE sdpa 6.39→**6.07** — KEEP |
| 18:31 | VAE d64 Q4 | video VAE sdpa 6.07→**4.74** — KEEP |
| 18:46 | VAE d64 Q8 default | sdpa 4.97 vs Q4 4.74 — REJECT |
| 19:30 | split INT8 linear | quantize ~0.4 · gemm ~10.5 · epi ~0.24 |
| 19:40 | LDS row quantize | 10.9 vs 10.6 ms — REJECT default |
| 20:05 | grouped hipBLAS unpacked/loop | 21.8 / 32.4 vs r64 22.2 — REJECT |
| 20:12 | grouped hipBLAS pack lda=G | micro 22.2→**10.9**; tests ok |
| 20:16 | fox s2 packed vs r64 | lin 4.45→**3.39** / GPU 8.56→**7.49** — KEEP |
| 20:20 | fox-fast packed | denoise GPU **55.51** / lin **25.41** |
