# Night-5 perf loop — STATUS

Started: 2026-08-23 19:47 CST  
Budget: ~12 hours  
Stop after: 2026-08-24 07:44 CST  
Git start: `734206b` (Q5 + INT8 k+=16)  
Baseline fox s2: denoise GPU **11.81s** (lin **7.20** · sdpa **3.97**)  
Baseline fox-fast: denoise GPU **86.56s** (lin **52.4** · sdpa **29.5**)  
Priority: INT8 WMMA 16×16×16 iu8, then SDPA Q6  
Loop: 45m one-shot (`AGENT_LOOP_WAKE_perf_night5`) until 07:44; **unilab / gfx1151 only**  
Do not retry: fc1 t128, fused SwiGLU dual-B, flash, F32 256×64/BK16/LDS double-buffer/float4 LDS BK+4, mmap memcpy default, t128 skip-last-sync, launch_bounds(256,3), INT8 LDS_K BK+8, grouped k+=16

## Scoreboard (fox s2)

| Version | denoise GPU | denoise linear | denoise sdpa |
|---------|------------:|---------------:|-------------:|
| day-6 `734206b` | 11.81 | 7.20 | 3.97 |
| SDPA Q6 default | **11.71** | 7.20 | **3.87** |

## Decisions

- (pending)

## Log

| Time | Action | Result |
|------|--------|--------|
| 19:47 | Start night-5 12h | Next: INT8 WMMA opt-in |
