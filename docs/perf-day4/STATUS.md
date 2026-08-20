# Day-4 perf loop — STATUS

Started: 2026-08-20 07:40 CST  
Budget: ~10 hours  
Stop after: 2026-08-20 17:40 CST  
Git: INT8 t128 pending (`662e268` + kernel)  
Baseline: n3 warm E2E **104.3s** (VAE GPU 18.5 · sdpa 6.5 · linear 11.5; denoise GPU 17.2)  
Order: VAE F32 linear → DiT INT8 linear → remaining load I/O  
Loop: 45m one-shot wakes (`AGENT_LOOP_WAKE_perf_day4`) until 17:40

## Scoreboard (fox `steps=2 layers=35`)

| Version | E2E | load | denoise GPU | VAE GPU | VAE sdpa | VAE linear |
|---------|----:|-----:|------------:|--------:|---------:|-----------:|
| n3 warm (`04652e6`) | 104.3 | 33.7 | 17.2 | 18.5 | 6.5 | 11.5 |
| F32 r128 (`662e268`) | 112.2† | 36.6 | 17.1 | **17.4** | 6.4 | **10.5** |
| + INT8 t128 | **104.6** | 41.2† | **16.7** | 17.5 | 6.5 | 10.5 |

† load/E2E I/O noisy. GPU: VAE linear **11.5→10.5s**; denoise linear **8.5→8.0s**.

## Decisions

- **KEEP** F32 GEMM 128×128 / 8×8 acc (default when K%32==0). Opt-out `H3_F32_R64=1`.
- **KEEP** INT8 linear 128×128 / 8×8 sudot4 (`h3_linear_int8_t128_kernel`). Opt-out `H3_INT8_R64=1`.
- **REJECT** INT8 fc1 SwiGLU t128 (denoise linear 8.01 vs 7.97; extra LDS, no win).
- **REJECT** F32 LDS double-buffer (K=8192 14.4→16.5 ms). Reverted before r128.
- Next: grouped INT8 t128, leftover VAE F32 K=8192.

## Log

| Time | Action | Result |
|------|--------|--------|
| 07:40 | Push `04652e6`; start day-4 | Next: VAE F32 linear |
| ~08:00 | F32 LDS double-buffer | REJECT (mixed microbench) |
| ~08:20 | F32 r128 tile | KEEP VAE linear 11.5→10.5s (`662e268`) |
| ~08:40 | INT8 linear t128 | KEEP denoise linear 8.5→8.0s |
| ~08:55 | INT8 fc1 t128 | REJECT (no denoise win) |
