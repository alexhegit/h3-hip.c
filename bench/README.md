# Benchmarks

Standardized E2E benchmarks for cross-GPU and cross-version comparison.
All scripts use `--profile` and print per-phase GPU timing.

## Presets

| Script | Resolution | Frames | Steps | Layers | Reuse | Quality | Use case |
|--------|-----------|--------|-------|--------|-------|---------|----------|
| `fox-s2.sh` | 512×512 | 22 | 2 | 35 | 1 | Lossless | Quick smoke / A/B (< 2 min) |
| `fox-fast.sh` | 512×512 | 22 | 20 | 45 | 2 | Lossless | Standard short clip (~15s MI300X) |
| `fox-15s.sh` | 864×480 | 362 | 20 | 45 | 2 | Lossless | Long cinematic (~180s MI300X) |
| `fox-15s-fast.sh` | 864×480 | 362 | 20 | 45 | 3 | **~15 dB** | Fast long cinematic (~90s MI300X) |

## Usage

```bash
# Default model path
./bench/fox-fast.sh

# Custom model path
./bench/fox-fast.sh /path/to/MiniMax-H3

# Fast 15s (reuse=3, TR 4:45, quality trade)
./bench/fox-15s-fast.sh

# Lossless 15s (reuse=2, TR 4:30, identical to no TR)
./bench/fox-15s.sh

# Root-level wrappers still work
./fox-fast.sh
```

## Environment knobs

| Variable | Default | Effect |
|----------|---------|--------|
| `H3_MODEL` | `/mnt/doscratch/MiniMax-H3` | Model checkpoint path |
| `H3_TOKEN_REDUCTION` | 0 | Enable token reduction (TR) |
| `H3_TOKEN_REDUCTION_BLOCKS` | 4:30 | TR block range BEGIN:END |
| `H3_TOKEN_REDUCTION_EARLY` | 10:40 | Early steps wider range (STEPS:END) |
| `H3_INT8_VAE` | 0 | INT8 Video VAE weights (−64% VRAM, −9% wall) |
| `H3_GPU_SAMPLER` | 0 | GPU Euler sampler |
| `H3_PROFILE` | 0 | Enable profiling (always on via --profile) |

## Strix Halo (gfx1151) — `main` 2026-09-12

AMD Ryzen AI MAX+ 395 / Radeon 8060S. Logs under
[`docs/perf-runs/HALO_2026-09-12.md`](../docs/perf-runs/HALO_2026-09-12.md).

| Script | DiT denoise | E2E (phase sum) | vs v0.11.0 Halo denoise |
|--------|----------:|----------------:|------------------------|
| `fox-s2.sh` | **3.81 s** | I/O | 3.36 s → +13% |
| `fox-fast.sh` | **26.4 s** | I/O | 24.5 s → +8% |
| no TR 15 s | **2167 s** | **40 min 4 s** | 2198 s → −1.4% |
| `fox-15s.sh` | **1373 s** | **26 min 56 s** | no-TR 2167 s → **−37%** |
| `fox-15s-fast.sh` | **988 s** | **20 min 44 s** | reuse=3 |

Do not use `--token-reduction` on fox-s2 / fox-fast (same as MI300X). GPU
sampler is not a Halo win.

## MI210 (gfx90a) — `main` 2026-09-12

Four-GPU MI210 box, 64 GiB/card. Logs under
[`docs/perf-runs/MI210_2026-09-12.md`](../docs/perf-runs/MI210_2026-09-12.md).
Default DiT is INT8.

| Script | DiT denoise | E2E | vs 2026-09-02 INT8 denoise |
|--------|----------:|----:|---------------------------:|
| `fox-s2.sh` | **1.38 s** | **11.16 s** | 1.33 s → +4% |
| `fox-fast.sh` | **9.04 s** | **19.50 s** | 9.11 s → −1% |
| no TR 15 s | **647 s** | **12 min 12 s** | 660 s → −2% |
| `fox-15s.sh` | **408 s** | **8 min 13 s** | no-TR 647 s → **−37%** |
| `fox-15s-fast.sh` | **294 s** | **6 min 18 s** | reuse=3 |

Do not use `--token-reduction` on fox-s2 / fox-fast.

## MI300X (gfx942) — `main` 2026-09-12

MI300X VF, 192 GiB. Logs under
[`docs/perf-runs/MI300X_2026-09-12.md`](../docs/perf-runs/MI300X_2026-09-12.md).
Default DiT is INT8.

| Script | DiT denoise | E2E | sdpa / linear | vs no-TR denoise |
|--------|----------:|----:|---------------|-----------------:|
| `fox-s2.sh` | **0.29 s** | **3.27 s** | 0.05 / 0.18 s | — |
| `fox-fast.sh` | **1.94 s** | **5.21 s** | 0.39 / 1.17 s | — |
| no TR 15 s | **176.2 s** | **179.5 s** | 139.8 / 28.7 s | — |
| `fox-15s.sh` | **111.3 s** | **114.7 s** | 83.4 / 21.1 s | **−37%** |
| `fox-15s-fast.sh` | **79.9 s** | **83.1 s** | 59.8 / 15.1 s | **−55%** |

Do not use `--token-reduction` on fox-s2 / fox-fast.
