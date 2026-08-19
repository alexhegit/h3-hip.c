# Night-3 perf loop — STATUS

Started: 2026-08-19 22:50 CST  
Budget: ~7 hours  
Stop after: 2026-08-20 05:50 CST  
Baseline: FOX Run B E2E **117.3s**; day-2 best **105.9s**

## Scoreboard (fox `steps=2 layers=35`)

| Version | E2E | load | denoise GPU | VAE wall/GPU | VAE sdpa |
|---------|----:|-----:|------------:|-------------:|---------:|
| Run B | 117.3 | 40.6 | 17.4 | 32.9 / 25.1 | 13.3 |
| day-2 best | 105.9 | 32.9 | 17.3 | 29.8 / 25.0 | ~13.4 |
| n3 delayed-flush (no Q2) | 102.7 | 32.1 | 17.2 | 29.8 / 25.0 | 13.4 |
| **n3 warm + VAE Q2+4key** | **104.3** | 33.7 | 17.2 | **23.3 / 18.5** | **6.5** |

vs Run B: E2E **−13s (−11%)** on warm n3-t5; VAE GPU **25.1→18.5 (−26%)**.

## Decisions

- Weight **mmap+memcpy default** → **reject** (`H3_WEIGHT_MMAP=1` opt-in only)
- **Delayed quant flush** (I/O of next block overlaps GPU quantize) → **keep**
- **VAE d64 2-query + 4-key** → **keep** (sdpa 13.4→6.5s). Opt-out `H3_SDPA_D64_Q1=1`

## Log

| Time | Action | Result |
|------|--------|--------|
| 22:59 | mmap memcpy | reject, load 129s |
| 23:01 | delayed flush | E2E 102.7, load 32.1 |
| 23:09 | VAE d64 Q2+4key | sdpa 6.5s, VAE GPU 18.4s |
| 23:12 | warm combined | **E2E 104.3** |
| | Next | VAE F32 linear ~11.5s now VAE bottleneck |

Do not retry: flash default, d128 2-key, Q-HM, VAE 8-key, mmap default, quant batch=8.
