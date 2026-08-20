# Day-4 perf loop — STATUS

Started: 2026-08-20 07:40 CST  
Budget: ~10 hours  
Stop after: 2026-08-20 17:40 CST  
Git: `404f4b3`  
Baseline: n3 warm E2E **104.3s** (VAE GPU 18.5 · sdpa 6.5 · linear 11.5; denoise GPU 17.2)  
Order: VAE F32 linear → DiT INT8 linear → remaining load I/O  
Loop: 45m one-shot wakes (`AGENT_LOOP_WAKE_perf_day4`) until 17:40

## Scoreboard (fox `steps=2 layers=35`)

| Version | E2E | load | denoise GPU | VAE GPU | VAE sdpa | VAE linear |
|---------|----:|-----:|------------:|--------:|---------:|-----------:|
| n3 warm (`04652e6`) | 104.3 | 33.7 | 17.2 | 18.5 | 6.5 | 11.5 |
| F32 r128 (`662e268`) | 112.2† | 36.6 | 17.1 | **17.4** | 6.4 | **10.5** |
| + INT8 t128 (`530480d`) | 104.6† | 41.2† | **16.7** | 17.5 | 6.5 | 10.5 |
| + d128 Q2 (`404f4b3`) | 115.5† | 51.4† | **13.8** | 17.5 | 6.5 | 10.5 |

† load/E2E I/O noisy. GPU: VAE linear **11.5→10.5s**; denoise linear **8.5→8.0s**; denoise sdpa **8.1→5.3s**.

## Decisions

- **KEEP** F32 GEMM 128×128 / 8×8 acc (default when K%32==0). Opt-out `H3_F32_R64=1`.
- **KEEP** INT8 linear 128×128 / 8×8 sudot4 (`h3_linear_int8_t128_kernel`). Opt-out `H3_INT8_R64=1`.
- **KEEP** DiT d128 2-query wave SDPA (share K/V). Opt-out `H3_SDPA_D128_Q1=1`.
- **REJECT** INT8 fc1 SwiGLU t128 (denoise linear 8.01 vs 7.97; extra LDS, no win).
- **REJECT** F32 LDS double-buffer (K=8192 14.4→16.5 ms). Reverted before r128.
- **REJECT** DiT d128 wave 4-key (SDPA bench 109→120 ms).
- **REJECT** F32 256×64 tile for K≥8192 (13.8/14.2→16.3 ms).
- **REJECT** VAE fused F32 fc1+SwiGLU r64 (VAE linear 10.5→11.2s; dual-B loses r128).
- Next: load I/O (not mmap memcpy), denoise linear ~7.9s, VAE linear ~10.5s.

## Log

| Time | Action | Result |
|------|--------|--------|
| 07:40 | Push `04652e6`; start day-4 | Next: VAE F32 linear |
| ~08:00 | F32 LDS double-buffer | REJECT (mixed microbench) |
| ~08:20 | F32 r128 tile | KEEP VAE linear 11.5→10.5s (`662e268`) |
| ~08:40 | INT8 linear t128 | KEEP denoise linear 8.5→8.0s (`530480d`) |
| ~08:55 | INT8 fc1 t128 | REJECT (no denoise win) |
| ~09:05 | F32 r128 BK=16 for K≥8192 | REJECT (16.3 vs 13.8 ms) |
| ~08:50 | d128 SDPA 4-key | REJECT (109→120 ms) |
| ~09:00 | F32 256×64 K≥8192 | REJECT (14.2→16.3 ms) |
| ~09:50 | VAE fused SwiGLU r64 | REJECT (linear 10.5→11.2s) |
| ~10:45 | DiT d128 Q2 SDPA | KEEP denoise sdpa 8.1→5.3s (`404f4b3`) |
