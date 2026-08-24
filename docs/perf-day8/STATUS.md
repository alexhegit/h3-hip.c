# Day-8 perf loop — STATUS

Started: 2026-08-24 08:02 CST  
Budget: ~11 hours  
Stop after: 2026-08-24 19:00 CST  
Git start: `852dbd0` (SDPA Q6)  
Baseline fox s2: denoise GPU **11.71s** (lin **7.20** · sdpa **3.87**)  
Baseline fox-fast: denoise GPU **85.26s** (lin **52.3** · sdpa **28.2**)  
Priority: INT8 via hipBLAS (not naive packing); then SDPA Q7  
Loop: 45m one-shot (`AGENT_LOOP_WAKE_perf_day8`) until 19:00; **unilab / gfx1151 only**  
Do not retry: fc1 t128, fused SwiGLU dual-B, flash, F32 256×64/BK16/LDS double-buffer/float4 LDS BK+4, mmap memcpy default, t128 skip-last-sync, launch_bounds(256,3), INT8 LDS_K BK+8, grouped k+=16, naive INT8 WMMA

## Scoreboard (fox s2)

| Version | denoise GPU | denoise linear | denoise sdpa |
|---------|------------:|---------------:|-------------:|
| night-5 `852dbd0` | 11.71 | 7.20 | 3.87 |
| hipBLAS INT8 (same-binary t128 A/B) | **8.99** | **4.50** | 3.85 |
| t128 opt-out (`H3_INT8_T128=1`) | 11.61 | 7.15 | 3.81 |

Microbench M=1920 K=14336 N=5376: hipBLAS **11.0 ms / 26.9 TFLOP/s** vs t128 **18.8 ms / 15.7 TFLOP/s**.

## Decisions

- **KEEP** hipBLAS INT8 GEMM (`hipblasGemmEx` i8→i32 + vector scale/SwiGLU epilogue) as default for DiT linear and fc1 SwiGLU. Opt out `H3_INT8_T128=1` (tiled t128) or `H3_INT8_HIPBLAS=0`; fc1 only `H3_INT8_FC1_HIPBLAS=0`. Fox s2 denoise GPU 11.61→**8.99s** (linear 7.15→**4.50**). Tests pass.
- (pending) SDPA Q7 opt-in A/B vs Q6.

## Log

| Time | Action | Result |
|------|--------|--------|
| 08:02 | Start day-8 11h | Next: rocWMMA/hipBLAS INT8 |
| 08:13 | hipBLAS i8 GEMM + scale | tests ok; 11.5 ms vs t128 18.9 |
| 08:18 | vec epilogue + fc1 SwiGLU gemm | tests ok; 11.0 ms |
| 08:21 | fox s2 hipBLAS | denoise GPU **8.99** / lin **4.50** / sdpa 3.85 |
| 08:24 | fox s2 t128 A/B | denoise GPU 11.61 / lin 7.15 — KEEP hipBLAS |
