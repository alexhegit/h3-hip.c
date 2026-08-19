# HIP performance baseline (gfx1151)

Machine: AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`), ROCm local build.
Preset: T2VA fox, `512×512`, `frames=22`, `reuse=1`.

## Profile tooling

`--profile` / `H3_PROFILE=1` prints Metal-compatible phase lines. On HIP:

| Field | Meaning |
|-------|---------|
| `wall` | Phase wall time (mark deltas match Metal) |
| `encode` | Host time inside `begin`→`submit` (includes overlapped GPU work when launches do not block) |
| `wait` / `root-gpu` | `hipStreamSynchronize` only — often near-zero if the GPU finished during encode |
| `op-classes` | HIP-event exclusive GPU time for linear / sdpa / conv / other (authoritative GPU split) |

Kernel-level: `rocprofv3 --hip-trace --kernel-trace --stats` (see `KNOWN_ISSUES.md`).

## Baseline A — short smoke (2026-08-18)

```bash
./h3 --profile -d "$MODEL" \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 \
  --steps 2 --layers 35 --reuse 1 \
  -o /tmp/h3-profile/fox-s2.mp4
```

| Phase | wall | GPU (`op-classes`) | Notes |
|-------|-----:|-------------------:|-------|
| Qwen text encoder | 23.5s | 2.1s (≈all linear) | Wall dominated by weight I/O |
| H3 DiT **load** | 114.0s | 1.1s | Weight load / quantize; almost no GPU |
| H3 DiT **Euler denoise** (2 steps) | 41.5s | 40.3s | linear 17.2s · **sdpa 18.7s** · other 4.4s |
| audio VAE decoder | 4.5s | 3.8s | ≈all conv |
| video VAE decoder | 73.8s | 53.6s | **linear 28.2s · sdpa 24.9s** |

## After overnight (2026-08-19 morning, same preset)

| Phase | wall / GPU | Delta vs A |
|-------|-----------:|------------|
| DiT denoise | wall ~18.5s · GPU ~17.3s (sdpa ~8.0 · linear ~8.6) | **−55%** denoise wall |
| video VAE | GPU ~25s (linear ~11 · sdpa ~13) | **−53%** VAE GPU |
| DiT load | wall ~40–54s | **−53–65%** (FD cache + parallel pread) |

Live log: [`perf-overnight/STATUS.md`](perf-overnight/STATUS.md) (**STOP**).

## Hotspot ranking (post-overnight)

1. **DiT linear ≈ SDPA** — ~8.5s / ~8.0s of denoise GPU.
2. **Video VAE** — ~25s GPU (sdpa ~13 · linear ~11); still hurts short runs.
3. **Weight load I/O** — DiT load ~40–54s; text encoder wall still large.
4. **Audio VAE conv** — ~3.5s; low priority.

## Next optimization targets

1. DiT INT8 linear further (still ~half of denoise)  
2. Video VAE F32 SDPA / GEMM  
3. Remaining load I/O (mmap / streaming / warmer cache)  
4. Host Euler overhead once GPU drops enough  

## Log

Raw stderr from baseline A: keep under `/tmp/h3-profile/fox-s2.log` on the build machine (not committed).
