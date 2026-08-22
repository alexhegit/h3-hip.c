# Evening perf loop — STATUS

Started: 2026-08-22 17:09 CST  
Budget: ~4 hours  
Stop after: 2026-08-22 21:09 CST  
Git: `091b249` (d128 Q3 default)  
Baseline fox-fast: denoise GPU **95.6s** (lin 58.0 · sdpa 32.9) · VAE **17.7s**  
Baseline fox s2: denoise GPU **13.1s** (lin ~8.0 · sdpa 4.5) · VAE linear **10.7s**  
Order: INT8 fc1 (not t128) → t128 epilogue if time → load I/O A/B  
Loop: 45m one-shot (`AGENT_LOOP_WAKE_perf_eve`) until 21:09

## Scoreboard (fox s2)

| Version | denoise GPU | denoise linear | denoise sdpa |
|---------|------------:|---------------:|-------------:|
| Q3 `091b249` | 13.17 | 8.00 | 4.53 |
| **fc1 128×64** | **12.67** | **7.48** | 4.55 |

## Decisions

- **KEEP** INT8 fc1 SwiGLU 128×64 (`h3_fc1_swiglu_int8_r128x64_kernel`) + `h3_silu`/`h3_fast_exp`. Same-binary A/B vs `H3_INT8_FC1_R64=1`: linear **8.00→7.48s**. Not 128×128 t128. Opt out `H3_INT8_FC1_R64=1`.

## Log

| Time | Action | Result |
|------|--------|--------|
| 17:09 | Commit Q3 `091b249`; start 4h | Next: INT8 fc1 128×64 |
| ~17:20 | fc1 128×64 + fast SiLU | KEEP linear 8.00→7.48s |
