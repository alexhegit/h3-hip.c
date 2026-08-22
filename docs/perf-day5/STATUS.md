# Day-5 perf loop — STATUS

Started: 2026-08-22 09:28 CST  
Budget: ~6 hours  
Stop after: 2026-08-22 15:28 CST  
Git: uncommitted Q3 keep + pread WILLNEED  
Baseline GPU: denoise **13.8s** (lin 8.0 · sdpa 5.3) · VAE **17.5s** (lin 10.5 · sdpa 6.5)  
Loop: 45m one-shot wakes (`AGENT_LOOP_WAKE_perf_day5`) until 15:28

## Scoreboard (fox `steps=2 layers=35`)

| Version | E2E | load | denoise GPU | denoise sdpa | VAE GPU | VAE linear |
|---------|----:|-----:|------------:|-------------:|--------:|-----------:|
| day-4 Q2 (`404f4b3`) | 115.5† | 51.4† | 13.8 | 5.3 | 17.5 | 10.5 |
| + r128 dual-B SwiGLU | 115.6† | 49.8† | 14.0 | 5.4 | 18.0 | **11.2** |
| **d128 Q3** | 96.1† | 34.4† | **13.1** | **4.5** | 17.6 | 10.7 |

† load/E2E I/O noisy. Judge GPU `op-classes`.

## Decisions

- **KEEP** DiT d128 3-query wave SDPA (share K/V). Microbench 77.0→66.3 ms; fox denoise sdpa **5.3→4.5s**. Opt out `H3_SDPA_D128_Q2=1` or `H3_SDPA_D128_Q1=1`.
- **REJECT** VAE fused F32 fc1+SwiGLU r128 dual-B (linear 10.5→11.2s). Same lesson as r64 fuse: extra B tile loses the unfused r128 GEMM.
- **KEEP (provisional)** `posix_fadvise(WILLNEED)` on pread ranges ≥16MiB. Load still noisy; no proven GPU-irrelevant I/O win yet.

## Still open

1. VAE F32 linear ~10.7s (K=8192 leftover; not new tiles / not fuse)
2. Denoise INT8 linear ~8.0s (not fc1 t128)
3. Load I/O A/B (not mmap memcpy, not more pread threads)

## Log

| Time | Action | Result |
|------|--------|--------|
| 09:28 | Start day-5 6h | F32 r128 occupancy min-blocks=2 (no isolated win) |
| ~09:45 | VAE r128 dual-B SwiGLU fuse | REJECT (11.2 vs 10.5s linear) |
| ~10:00 | DiT d128 Q3 wave | KEEP denoise sdpa 5.3→4.5s |
