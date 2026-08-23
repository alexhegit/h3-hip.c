# Day-6 perf loop — STATUS

Started: 2026-08-23 08:24 CST  
Budget: ~10 hours  
Stop after: 2026-08-23 18:24 CST  
Git start: `e60c3de` (night-4 INT8 k+=8 + SDPA Q4)  
Baseline fox s2: denoise GPU **12.15s** (lin **7.28** · sdpa **4.23**)  
Baseline fox-fast: denoise GPU **88.48s** (lin **53.0** · sdpa **30.8**)  
Priority: DiT INT8 linear, then SDPA Q5; VAE/I/O later  
Loop: 45m one-shot (`AGENT_LOOP_WAKE_perf_day6`) until 18:24  
Do not retry: fc1 t128, fused SwiGLU dual-B, flash, F32 256×64/BK16/LDS double-buffer/float4 LDS BK+4, mmap memcpy default, t128 skip-last-sync, launch_bounds(256,3)

## Scoreboard (fox s2)

| Version | denoise GPU | denoise linear | denoise sdpa |
|---------|------------:|---------------:|-------------:|
| night-4 `e60c3de` | 12.15 | 7.28 | 4.23 |
| SDPA Q5 default | **11.89** | 7.24 | **4.00** |
| INT8 k+=16 | **11.81** | **7.20** | 3.97 |

## Decisions

- **KEEP** d128 SDPA **Q5** as default. Opt out `H3_SDPA_D128_Q4=1`. Fox s2 sdpa 4.23→4.00; fox-fast GPU 88.48→86.98 (sdpa 30.8→29.2).
- **KEEP** INT8 t128 + fc1 128×64 inner `k+=16` (four sdot4 per LDS pair). Fox s2 lin 7.24→7.20; microbench 19.5→18.7 ms.
- **REJECT** grouped INT8 `k+=16` (22.2 → 23.3 ms).

## Log

| Time | Action | Result |
|------|--------|--------|
| 08:24 | Start day-6 10h | Next: INT8 LDS align / SDPA Q5 |
| 08:28 | INT8 LDS_K BK+8 | 19.6 ms, reject |
| 08:32 | SDPA Q5 | fox s2 GPU **11.89** / sdpa **4.00** |
| 08:40 | fox-fast Q5 | denoise GPU **86.98** (lin 53.0 · sdpa **29.24**) vs 88.48 |
| 08:48 | t128 `k+=16` | micro 18.8 ms; fox lin 7.28 (noise vs 7.24) |
| 08:50 | fc1 `k+=16` | fox s2 GPU **11.81** / lin **7.20** |
| 08:52 | grouped `k+=16` | 23.3 ms, reject |
