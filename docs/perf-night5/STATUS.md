# Night-5 perf loop — STATUS

Started: 2026-08-23 19:47 CST  
Budget: ~12 hours  
Stop after: 2026-08-24 07:44 CST  
Git start: `734206b` (Q5 + INT8 k+=16)  
Baseline fox s2: denoise GPU **11.81s** (lin **7.20** · sdpa **3.97**)  
Baseline fox-fast: denoise GPU **86.56s** (lin **52.4** · sdpa **29.5**)  
Current fox-fast: **85.26s** (lin **52.3** · sdpa **28.2**) after Q6  
Do not retry: fc1 t128, fused SwiGLU dual-B, flash, F32 256×64/BK16/LDS double-buffer/float4 LDS BK+4, mmap memcpy default, t128 skip-last-sync, launch_bounds(256,3), INT8 LDS_K BK+8, grouped k+=16, **naive INT8 WMMA**

## Scoreboard (fox s2)

| Version | denoise GPU | denoise linear | denoise sdpa |
|---------|------------:|---------------:|-------------:|
| day-6 `734206b` | 11.81 | 7.20 | 3.97 |
| SDPA Q6 default | **11.71** | 7.20 | **3.87** |

## Decisions

- **KEEP** d128 SDPA **Q6** as default. Opt out `H3_SDPA_D128_Q5=1`. Fox s2 sdpa 3.97→3.87; fox-fast GPU 86.56→85.26 (sdpa 29.5→28.2).
- **REJECT** hand-rolled INT8 WMMA 16×16×16 (wrong C/K layout vs CPU; naive global loads 199 ms vs t128 18.7 ms). Needs rocWMMA/CK fragment loads, not another packing guess.

## Log

| Time | Action | Result |
|------|--------|--------|
| 19:47 | Start night-5 12h | Next: INT8 WMMA opt-in |
| 20:05 | naive WMMA | tests fail; 199 ms. Revert |
| 20:15 | SDPA Q6 | fox s2 GPU **11.71** / sdpa **3.87** |
| 20:20 | fox-fast Q6 | denoise GPU **85.26** (lin 52.3 · sdpa **28.2**) |
