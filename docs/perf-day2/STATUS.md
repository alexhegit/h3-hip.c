# Day-2 perf loop — STATUS

Started: 2026-08-19 12:05 CST  
Budget: ~10 hours  
Stop after: 2026-08-19 22:05 CST  
Baseline: FOX Run B — E2E **117.3s**  
**Best so far: E2E 105.9s** (12:31; load 32.9 · denoise 18.5/17.3 · VAE 29.8/25.0)  
Δ vs Run B: **−11.4s (−9.7%)**  
Order: DiT load I/O → Video VAE → DiT denoise → text load  
Loop: every 45m (`AGENT_LOOP_TICK_perf_day2`)

## Scoreboard (fox `steps=2 layers=35`)

| Version | E2E | load | denoise GPU | linear | sdpa | VAE GPU |
|---------|----:|-----:|------------:|-------:|-----:|--------:|
| Run B | 117.3 | 40.6 | 17.4 | 8.6 | 8.1 | 25.1 |
| **day2 best** | **105.9** | **32.9** | **17.3** | **8.6** | **8.1** | **25.0** |

## Decisions

- Parallel DiT / VAE block loads + pread8 + fd mutex + WILLNEED → **keep**
- Vectorized INT8 quantize → **keep**
- F32 float4 store + INT8 packed BF16 store → **keep** (small)
- VAE d64 8-key, d128 2-key, quant batch=8 → **reject**

## Log

| Time | Action | Result |
|------|--------|--------|
| 12:05 | Start | Run B = 117.3 |
| 12:09–12:24 | parallel load / quantize / VAE load | E2E ~108 |
| 12:27 | quant batch 8 / d128 2-key | reject |
| 12:31 | INT8 packed stores | **E2E 105.9** |
| 12:34 | cold-cache E2E | 135s (I/O variance; GPU flat) |
| 12:37 | Qwen prefetch depth 4 | text peak↑; wall noisy — keep trial |
| | Next | VAE sdpa / denoise linear |
