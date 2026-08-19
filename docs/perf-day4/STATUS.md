# Day-4 perf loop — STATUS

Started: 2026-08-20 07:40 CST  
Budget: ~10 hours  
Stop after: 2026-08-20 17:40 CST  
Git: `04652e6`  
Baseline: n3 warm E2E **104.3s** (VAE GPU 18.5 · sdpa 6.5 · linear 11.5; denoise GPU 17.2)  
Order: VAE F32 linear → DiT INT8 linear → remaining load I/O  
Loop: every 45m (`AGENT_LOOP_TICK_perf_day4`)

## Scoreboard (fox `steps=2 layers=35`)

| Version | E2E | load | denoise GPU | VAE GPU | VAE sdpa | VAE linear |
|---------|----:|-----:|------------:|--------:|---------:|-----------:|
| n3 warm (`04652e6`) | 104.3 | 33.7 | 17.2 | 18.5 | 6.5 | 11.5 |
| F32 r128 (this morning) | 112.2† | 36.6 | 17.1 | **17.4** | 6.4 | **10.5** |

† E2E/load noisy (text+DiT I/O). GPU: VAE linear **11.5→10.5s**. Opt-out `H3_F32_R64=1`.

## Decisions

- **KEEP** F32 GEMM 128×128 / 8×8 acc (default when K%32==0). Microbench: 7.3→4.3 / 14.5→13.8 / 10.0→4.7 ms vs r64.
- **REJECT** F32 LDS double-buffer (K=8192 14.4→16.5 ms). Reverted before r128.
- Next: DiT INT8 linear (~8.5s), then leftover F32 K=8192.

## Log

| Time | Action | Result |
|------|--------|--------|
| 07:40 | Push `04652e6`; start day-4 | Next: VAE F32 linear |
| ~08:00 | F32 LDS double-buffer | REJECT (mixed microbench) |
| ~08:20 | F32 r128 tile | KEEP VAE linear 11.5→10.5s |
