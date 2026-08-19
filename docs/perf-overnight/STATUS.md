# Overnight perf loop — STATUS

Started: 2026-08-18 22:18 CST  
Budget: ~10 hours  
Stop after: 2026-08-19 08:18 CST  
**STOP** — overnight budget complete (2026-08-19 ~08:05 CST)

Order: DiT SDPA → DiT INT8/linear → Video VAE → I/O  
Loop: every 45m (`AGENT_LOOP_TICK_perf_overnight`)

## Final scoreboard (fox `steps=2 layers=35`)

| Version | denoise wall | denoise GPU | sdpa | linear | VAE GPU | DiT load |
|---------|-------------:|------------:|-----:|-------:|--------:|---------:|
| Baseline | 41.5s | 40.3s | 18.7s | 17.2s | 53.6s | ~114s |
| +SDPA grid / INT8 / VAE | ~18.4s | ~17.3s | **~8.0s** | ~8.5s | **~25s** | ~71s (FD) |
| **+parallel pread** | **~18.5s** | **~17.3s** | **~8.0s** | ~8.6s | **~25s** | **~40–54s** |

**Net vs baseline:** denoise **41.5→~18.5s (−55%)**; VAE GPU **53.6→~25s (−53%)**; DiT load **~114→~40–54s (−53–65%)**.

## Kept wins

1. SDPA `h3_fast_exp` / wave paths  
2. F32 GEMM LDS bank pad (VAE linear)  
3. INT8 LDS pad + BK=128  
4. KV head-major + QKV-RoPE fuse  
5. VAE d64 specialized SDPA + 4-key online softmax  
6. Wave SDPA launch grid `(seq, heads)` — largest denoise win  
7. Weight-shard FD cache + looped pread  
8. Per-block batched INT8 quantize (free BF16 after submit); `H3_DIT_QUANT_BATCH` (default 4)  
9. **4-way parallel pread** for ≥64MiB weight reads (`H3_PREAD_SERIAL=1` opt-out)

## Rejected

Flash multi-Q default; d128 2-key; Q head-major fuse; F32 BK=64; INT8 LDS double-buffer; BF16/F32 float4 remaps that didn't move E2E.

## Remaining (post-overnight)

1. Denoise linear ≈ sdpa (~8.5s / ~8.0s)  
2. VAE sdpa ~13s / linear ~11s  
3. DiT load still I/O-heavy (~40s+); text encoder wall  
4. 20-step showcase path extrapolation  

## Log (late ticks)

| Time | Action | Result |
|------|--------|--------|
| 05:30 | SDPA grid=(seq,heads) | denoise sdpa 16.5→8.0s |
| 07:05 | FD cache + pread loop | DiT load 118→71s |
| 07:50 | Quantize batch (free-after-submit) | submits↓; load wall flat ~71s |
| 07:58 | Parallel pread (4 threads, ≥64MiB) | DiT load **71→~40s** (confirm ~54s warm variance) |
| 08:05 | **STOP** | Overnight complete |
