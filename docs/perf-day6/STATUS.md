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

## Decisions

- **KEEP** d128 SDPA **Q5** as default. Opt out `H3_SDPA_D128_Q4=1`. Fox s2 sdpa 4.23→4.00.
- **REJECT** INT8 t128/fc1 `LDS_K=BK+8` (microbench 19.6 vs 19.5, no win).

## Log

| Time | Action | Result |
|------|--------|--------|
| 08:24 | Start day-6 10h | Next: INT8 LDS align / SDPA Q5 |
| 08:28 | INT8 LDS_K BK+8 | 19.6 ms, reject |
| 08:32 | SDPA Q5 | fox s2 GPU **11.89** / sdpa **4.00** |
