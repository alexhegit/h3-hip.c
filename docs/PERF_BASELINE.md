# HIP performance baseline (gfx1151)

Machine: AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`), ROCm local build.
Preset: T2VA fox, `512×512`, `frames=22`, `reuse=1`.

**Latest tagged baseline:** [`perf-runs/V0.9.0.md`](perf-runs/V0.9.0.md)
(2026-08-26, git `e0d8558` + baseline commit). Fox s2 E2E **82.9–87.3 s**;
fox-fast E2E **94.8 s**. vs M5 Max: [`perf-runs/VS_UPSTREAM.md`](perf-runs/VS_UPSTREAM.md).
Dated ledger: [`perf-runs/FOX_S2.md`](perf-runs/FOX_S2.md).

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

Phase-wall sum ≈ **257 s** (no process `time` recorded).

## Baseline B — post-overnight (2026-08-19 09:11 CST)

Git: `dcb4858`. Full table: [`perf-runs/FOX_S2.md`](perf-runs/FOX_S2.md#run-b--post-overnight-2026-08-19-0911-cst).

| Phase | wall | GPU (`op-classes`) | GPU split |
|-------|-----:|-------------------:|-----------|
| Qwen text encoder | 20.0s | 1.9s | linear 1.9 |
| H3 DiT **load** | **40.6s** | 2.4s | quantize-dominated GPU |
| H3 DiT **Euler denoise** (2 steps) | **18.6s** | **17.4s** | linear 8.6 · sdpa 8.1 · other 0.7 |
| audio VAE decoder | 4.4s | 3.9s | conv 3.8 |
| video VAE decoder | **32.9s** | **25.1s** | linear 11.3 · **sdpa 13.3** |
| **E2E process wall** | **117.3s** | | `/usr/bin/time` |

## Run C — day-2 uncommitted (2026-08-19 21:13 CST)

Same preset. Ledger: [`perf-runs/FOX_S2.md`](perf-runs/FOX_S2.md#run-c--day-2-tree-2026-08-19-2113-cst).

| Phase | wall | GPU |
|-------|-----:|----:|
| Qwen text encoder | 23.8s | 1.8s |
| H3 DiT **load** | 44.7s | 3.5s |
| H3 DiT **Euler denoise** | 18.4s | 17.2s (linear 8.6 · sdpa 8.1) |
| audio VAE | 4.3s | 3.8s |
| video VAE | 29.8s | 25.0s (linear 11.2 · sdpa 13.4) |
| **E2E** | **121.8s** | vs B **+4.5s** (I/O; GPU flat) |

Day-2 best earlier same tree: E2E **105.9s** (load 32.9). Use that, not C, as optimistic I/O.

| vs A | Δ wall |
|------|-------:|
| DiT load | −73.4s (−64%) |
| DiT denoise | −22.9s (−55%) |
| video VAE | −40.9s (−55%) |
| E2E (phase sum) | ≈257→117s (**−54%**) |

## Hotspot ranking (Baseline B)

1. **DiT weight load** — 40.6s wall (I/O + INT8 quantize).  
2. **Video VAE** — 25.1s GPU (sdpa 13.3 · linear 11.3).  
3. **Text encoder** — 20.0s wall (GPU 1.9s).  
4. **DiT denoise** — linear ≈ sdpa (~8.6 / ~8.1s).  
5. **Audio VAE conv** — ~3.9s GPU.

## Next optimization targets

1. DiT load I/O / quantize overlap  
2. Video VAE F32 SDPA + GEMM  
3. DiT INT8 linear (denoise)  
4. Text-encoder load wall  
5. Host Euler overhead once GPU drops enough  

Overnight notes: [`perf-overnight/STATUS.md`](perf-overnight/STATUS.md).

## Log

- Baseline A: `/tmp/h3-profile/fox-s2.log` (machine-local).  
- Baseline B: `/tmp/h3-profile/fox-s2-post-overnight.log`.
