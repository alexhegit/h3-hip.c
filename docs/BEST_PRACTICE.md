# Best Practice Guide

Quick reference for optimal H3 settings. All settings are environment variables.

## Quick Start

```bash
# MI300X (gfx942) — best performance + quality
H3_INT8_VAE=1 H3_TOKEN_REDUCTION=1 H3_GPU_SAMPLER=1 ./h3 ...

# MI210 (gfx90a) — same settings
H3_INT8_VAE=1 H3_TOKEN_REDUCTION=1 H3_GPU_SAMPLER=1 ./h3 ...

# Strix Halo (gfx1151) — same settings
H3_INT8_VAE=1 H3_TOKEN_REDUCTION=1 H3_GPU_SAMPLER=1 ./h3 ...
```

**INT8 DiT is enabled by default on all ISAs.** No need to set `H3_INT8_MLP=1`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `H3_INT8_MLP` | `1` | INT8 DiT weights (faster, lower VRAM) |
| `H3_FP8_MLP` | `0` | FP8 DiT weights (gfx942 only, experimental) |
| `H3_INT8_VAE` | `0` | INT8 Video VAE weights (−61% VAE VRAM) |
| `H3_TOKEN_REDUCTION` | `0` | Halve spatial width in middle layers (−37% denoise) |
| `H3_GPU_SAMPLER` | `0` | GPU Euler sampler (reduces latency) |
| `H3_VAE_TILE_PIXELS` | `480` | VAE tile size (256–512) |

## Recommended Settings by Use Case

### 1. Short Video (fox-s2, ≤5s)

**Goal**: Lowest latency

```bash
H3_INT8_VAE=1 H3_GPU_SAMPLER=1 ./h3 \
  -p "prompt" --width 512 --height 512 --frames 22 \
  --steps 2 --layers 35 --reuse 1 -o output.mp4
```

| ISA | DiT | E2E | VRAM |
|-----|----:|----:|-----:|
| MI300X | 3.1s | ~8s | ~15 GiB |
| MI210 | 3.1s | ~11s | ~15 GiB |
| Strix Halo | 3.4s | ~89s | ~15 GiB |

### 2. Long Video (15s cinematic)

**Goal**: Best throughput

```bash
H3_INT8_VAE=1 H3_TOKEN_REDUCTION=1 H3_GPU_SAMPLER=1 ./h3 \
  -p "prompt" --width 864 --height 480 --frames 362 \
  --steps 20 --layers 45 --reuse 2 -o output.mp4
```

| ISA | DiT | E2E | VRAM |
|-----|----:|----:|-----:|
| MI300X | 113s | ~142s | ~32 GiB |
| MI210 | 540s | ~8min | ~28 GiB |
| Strix Halo | 2198s | ~41min | ~28 GiB |

### 3. Maximum Quality

**Goal**: Highest PSNR/SSIM

```bash
H3_INT8_MLP=0 ./h3 \
  -p "prompt" --width 864 --height 480 --frames 362 \
  --steps 20 --layers 45 --reuse 2 -o output.mp4
```

BF16 has ~4 dB higher PSNR than INT8 (29.5 vs 25.9 dB). Use when quality
matters more than speed/VRAM.

### 4. Minimum VRAM

**Goal**: Run on 16 GiB GPU

```bash
H3_INT8_VAE=1 H3_TOKEN_REDUCTION=1 ./h3 ...
```

Peak VRAM: ~15 GiB (fox-s2), ~28 GiB (15s cinematic).

## ISA-Specific Notes

### MI300X (gfx942)

- **FP8 available** (`H3_FP8_MLP=1`): 6% faster than INT8 for long video, but lower quality (PSNR 25.9 vs 29.5 dB)
- **INT8 recommended** for most use cases: best balance of speed, quality, and VRAM
- **GPU sampler recommended**: reduces latency for short video

### MI210 (gfx90a)

- **INT8 recommended** (default): same speed as BF16, 32% less VRAM
- **FP8 not available**: falls back to INT8 with warning
- **GPU sampler recommended**: reduces latency

### Strix Halo (gfx1151)

- **INT8 recommended** (default): 64% faster than BF16 for short video
- **FP8 not available**: falls back to INT8 with warning
- **GPU sampler NOT recommended**: no benefit on RDNA
- **I/O bound**: E2E dominated by weight loading (~107 GiB)

## Trade-off Summary

| Setting | Speed | Quality | VRAM |
|---------|-------|---------|------|
| INT8 (default) | Fast | Good (29.5 dB) | Low |
| BF16 | Slow | Best (33.5 dB) | High |
| FP8 | Fastest (long video) | Lower (25.9 dB) | Low |
| Token Reduction | Faster (−37%) | Slightly lower | Lower |
| INT8 VAE | Same | Same | −61% VAE VRAM |
| GPU Sampler | Lower latency | Same | Same |

## Performance Tuning

### Monitor Performance

```bash
# Print per-phase GPU timing
H3_PROFILE=1 ./h3 ...

# Check VRAM usage
rocm-smi -d vim
```

### Common Issues

1. **OOM on 16 GiB GPU**: Enable `H3_INT8_VAE=1` and `H3_TOKEN_REDUCTION=1`
2. **Slow short video**: Enable `H3_GPU_SAMPLER=1`
3. **Low quality**: Disable INT8 (`H3_INT8_MLP=0`) for BF16
4. **FP8 not working**: Only available on MI300X (gfx942)

## Examples

### Fox-s2 (Short Video)

```bash
# Default (INT8, fast)
./h3 -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 \
  --steps 2 --layers 35 --reuse 1 -o fox.mp4

# BF16 (high quality)
H3_INT8_MLP=0 ./h3 -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 \
  --steps 2 --layers 35 --reuse 1 -o fox-hq.mp4
```

### 15s Cinematic (Long Video)

```bash
# Optimized (INT8 + all opts)
H3_INT8_VAE=1 H3_TOKEN_REDUCTION=1 H3_GPU_SAMPLER=1 \
  ./h3 -p "A cinematic shot of a fox in the forest." \
  --width 864 --height 480 --frames 362 \
  --steps 20 --layers 45 --reuse 2 -o fox-15s.mp4

# Maximum quality (BF16)
H3_INT8_MLP=0 ./h3 -p "A cinematic shot of a fox in the forest." \
  --width 864 --height 480 --frames 362 \
  --steps 20 --layers 45 --reuse 2 -o fox-15s-hq.mp4
```
