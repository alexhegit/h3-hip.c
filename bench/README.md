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

## MI300X (gfx942) — perf-v2 branch

### fox-fast (512×512, 22f, 20 steps, reuse=2)

| Metric | No TR | With TR (4:30) | Δ |
|--------|------:|---------------:|--:|
| DiT denoise wall | **1.94 s** | 9.72 s | +401% |
| DiT SDPA | 0.378 s | 0.228 s | −39% |
| DiT Linear | 1.16 s | 6.07 s | +423% |
| Video VAE | 1.31 s | 2.21 s | +69% |

TR hurts short clips. **Do not use `--token-reduction` for fox-fast.**

### fox-15s — TR × Reuse parameter sweep (864×480, 362f, 20 steps)

**Key finding:** TR 4:30 + Reuse 2 produces **identical output** to no TR
(PSNR=inf) thanks to bilinear expansion. This is the lossless sweet spot.

| Config | DiT denoise | SDPA | Linear | Total DiT | vs Gold |
|--------|----------:|-----:|-------:|----------:|--------:|
| Reuse 2, no TR | 175.8 s | 139.4 s | 28.6 s | 179.1 s | — (gold) |
| **Reuse 2, TR 4:30** | **130.9 s** | **103.0 s** | **21.0 s** | **130.9 s** | **PSNR=inf (lossless)** |
| Reuse 3, no TR | 128.8 s | — | — | 132.1 s | 15.1 dB |
| Reuse 3, TR 4:30 | 80.5 s | 60.2 s | 15.3 s | 83.9 s | ~15 dB |
| Reuse 3, TR 4:40 | 71.4 s | 52.3 s | 14.2 s | 74.8 s | ~14.5 dB |
| **Reuse 3, TR 4:45** | **62.1 s** | **44.2 s** | **13.1 s** | **65.5 s** | **~15 dB** |
| Reuse 3, TR 4:45 + early 10:48 | 59.4 s | 41.9 s | 12.8 s | 62.7 s | ~13.5 dB |
| Reuse 3, TR 4:50 | 52.9 s | 36.2 s | 12.0 s | 56.3 s | ~13 dB |

**Quality notes:**
- Reuse 3 introduces ~15 dB PSNR vs reuse=2 gold (skipped denoising steps).
- TR block range beyond 4:45 degrades fine detail (blocks 45-49 = detail refinement).
- `early 10:48` extends aggressive TR to first10 steps; marginal additional gain.
- Max reuse is 3 (hard cap in code).

### Recommended configurations

| Profile | Script | DiT denoise | Quality |
|---------|--------|----------:|---------|
| **Lossless** | `./bench/fox-15s.sh` | 130.9 s | Identical to no TR |
| **Fast** | `./bench/fox-15s-fast.sh` | 62.1 s | ~15 dB vs gold |
| **Aggressive** | Manual (add `H3_TOKEN_REDUCTION_EARLY=10:48`) | 59.4 s | ~13.5 dB vs gold |

### INT8 VAE impact

| Config | Video VAE | VRAM peak | Δ vs BF16 VAE |
|--------|----------:|----------:|---------------:|
| BF16 VAE (default) | 27.0 s | 10.2 GiB | — |
| INT8 VAE | 24.9 s | 3.7 GiB | −8% wall, −64% VRAM |

### Cross-branch comparison (MI300X, no TR)

| Metric | main (v0.11.0) | perf-v2 | Δ |
|--------|---------------:|--------:|--:|
| fox-fast DiT denoise | 8.81 s | **1.94 s** | **−78%** |
| fox-fast SDPA | 0.533 s | 0.378 s | −29% |
| fox-fast Linear | 7.40 s | **1.16 s** | **−84%** |
| fox-fast Video VAE | 4.85 s | **1.31 s** | **−73%** |
| 15s DiT denoise | 177.3 s | 175.8 s | −1% |
| 15s SDPA | 139.9 s | 139.4 s | ~0% |
| 15s Linear | 29.5 s | 28.6 s | −3% |
| 15s Video VAE | 26.8 s | 26.8 s | ~0% |

perf-v2 gains come from `__ocml_native_exp_f32` (SDPA softmax) and fused
VAE ops (scale_add+rms_norm). The fox-fast speedup is dramatic because
short-clip VAE is the dominant cost; the15s pipeline is SDPA-bound and
shows minimal improvement without TR.
