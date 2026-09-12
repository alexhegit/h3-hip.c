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
| `H3_TOKEN_REDUCTION` | `0` | Halve spatial width in middle layers (−33% denoise) |
| `H3_TOKEN_REDUCTION_SCHEDULE` | (none) | Per-step TR range schedule (see below) |
| `H3_GPU_SAMPLER` | `0` | GPU Euler sampler (reduces latency) |
| `H3_VAE_TILE_PIXELS` | `480` | VAE tile size (256–512) |

## Quality Scale

All benchmarks report PSNR (peak signal-to-noise ratio) against the gold
reference (no TR, BF16 DiT, reuse=1). Use this scale to interpret values:

| PSNR | Label | What you see |
|-----:|-------|-------------|
| ≥30 dB | **gold** | Indistinguishable from reference |
| 25–30 dB | **high** | Slight quantization noise, textures intact |
| 20–25 dB | **good** | Spatial detail softens, structure intact |
| 15–20 dB | **preview** | Noticeable softness, suitable for quick review |
| <15 dB | **draft** | Composition reference only |

## Recommended Settings by Use Case

### 1. Short Video (fox-s2, ≤5 s)

**Goal**: Lowest latency

```bash
H3_INT8_VAE=1 H3_GPU_SAMPLER=1 ./h3 \
  -p "prompt" --width 512 --height 512 --frames 22 \
  --steps 2 --layers 35 --reuse 1 -o output.mp4
```

| Setting | DiT | E2E | VRAM | PSNR |
|---------|----:|----:|-----:|-----:|
| INT8 default | 3.1 s | ~8 s | ~15 GiB | 29.5 dB |
| BF16 (`H3_INT8_MLP=0`) | 6.5 s | ~16 s | ~26 GiB | 33.5 dB (gold) |

### 2. Long Video — Balanced (15 s cinematic)

**Goal**: Best speed/quality balance

```bash
H3_INT8_VAE=1 H3_TOKEN_REDUCTION=1 H3_GPU_SAMPLER=1 ./h3 \
  -p "prompt" --width 864 --height 480 --frames 362 \
  --steps 20 --layers 45 --reuse 2 -o output.mp4
```

| Setting | DiT | E2E | VRAM | PSNR | vs gold |
|---------|----:|----:|-----:|-----:|--------:|
| no TR (baseline) | 176 s | ~210 s | ~28 GiB | 29.5 dB | −4 dB |
| TR 4:30 (`--token-reduction`) | 112 s | ~145 s | ~32 GiB | 20.4 dB | −13 dB |
| TR schedule `50→45→30` | 79 s | ~111 s | ~32 GiB | 19.3 dB | −14 dB |

> TR 4:30 trades 13 dB for −37% denoise. Spatial detail (fur, textures) softens
> in middle blocks; temporal consistency and scene structure preserved.

### 3. Long Video — Fast (15 s cinematic)

**Goal**: Maximum speed, accept quality trade

```bash
H3_TOKEN_REDUCTION_SCHEDULE="0:4:50,10:4:45,20:4:30" \
  H3_INT8_VAE=1 H3_GPU_SAMPLER=1 ./h3 \
  -p "prompt" --width 864 --height 480 --frames 362 \
  --steps 20 --layers 45 --reuse 2 --token-reduction -o output.mp4
```

Or use the preset script: `./bench/fox-15s-fast.sh`

| Setting | DiT | E2E | VRAM | PSNR | vs gold |
|---------|----:|----:|-----:|-----:|--------:|
| reuse=3 + TR 4:45 | 62 s | ~90 s | ~32 GiB | ~15 dB | −18 dB |
| reuse=3 + TR schedule | 57 s | ~85 s | ~32 GiB | ~13 dB | −20 dB |

> ~15 dB: noticeable softness across the frame, some temporal drift in fine
> details. Still recognisable; suitable for previews and rapid iteration.

### 4. Maximum Quality

**Goal**: Highest PSNR/SSIM

```bash
H3_INT8_MLP=0 ./h3 \
  -p "prompt" --width 864 --height 480 --frames 362 \
  --steps 20 --layers 45 --reuse 2 -o output.mp4
```

| Setting | DiT | E2E | VRAM | PSNR | vs gold |
|---------|----:|----:|-----:|-----:|--------:|
| BF16 DiT | 186 s | ~219 s | ~41 GiB | 33.5 dB | gold |
| INT8 DiT (default) | 176 s | ~210 s | ~28 GiB | 29.5 dB | −4 dB |
| FP8 DiT (gfx942 only) | 179 s | ~221 s | ~31 GiB | 25.9 dB | −8 dB |

### 5. Minimum VRAM

**Goal**: Run on 16 GiB GPU

```bash
H3_INT8_VAE=1 H3_TOKEN_REDUCTION=1 ./h3 ...
```

| Setting | VRAM (fox-s2) | VRAM (15 s) | PSNR |
|---------|-------------:|------------:|-----:|
| INT8 + TR + INT8 VAE | ~15 GiB | ~28 GiB | 20.4 dB |
| INT8 + TR (no INT8 VAE) | ~15 GiB | ~32 GiB | 20.4 dB |

## ISA-Specific Notes

### MI300X (gfx942)

- **FP8 available** (`H3_FP8_MLP=1`): 6% faster than INT8 for long video, but −8 dB PSNR (25.9 vs 29.5 dB)
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

## Token Reduction Schedule

`H3_TOKEN_REDUCTION_SCHEDULE` provides per-step control over which DiT blocks
use token reduction. Works independently of `--token-reduction` — setting the
env var auto-enables TR.

**Format:** `STEP:BEGIN:END,STEP:BEGIN:END,...`
- `STEP` — denoising step to apply from (0-indexed)
- `BEGIN:END` — TR block range (e.g., `4:30` = blocks 4–29)
- Later entries override earlier ones
- `STEP:0:0` disables TR from that step onward

**Examples:**

```bash
# Fixed TR blocks 4–29 for all steps (auto-enables TR)
H3_TOKEN_REDUCTION_SCHEDULE="0:4:30" ./h3 ...

# Aggressive early, conservative late
H3_TOKEN_REDUCTION_SCHEDULE="0:4:50,10:4:45,20:4:30" ./h3 ...

# TR for first 10 steps, then full quality
H3_TOKEN_REDUCTION_SCHEDULE="0:4:50,10:0:0" ./h3 ...

# Combine with --token-reduction for maximum speed
H3_TOKEN_REDUCTION_SCHEDULE="0:4:50,10:4:45,20:4:30" \
  ./h3 --token-reduction ...
```

### Benchmark data (MI300X / gfx942)

**5 s video** (864×480, 20 steps, 45 layers, reuse=2):

| Config | DiT | PSNR | vs gold | Label |
|--------|----:|-----:|--------:|-------|
| no TR | 30.5 s | 29.5 dB | −4 dB | high |
| TR 4:30 | 20.4 s | 20.4 dB | −13 dB | good |
| Schedule `50→45→30` | 15.4 s | 19.3 dB | −14 dB | preview |
| Schedule `4:50` full | 14.3 s | lower | — | draft |
| Schedule `4:50→disable` | 23.2 s | 19.0 dB | −14 dB | preview |

**15 s cinematic** (864×480, 20 steps, 45 layers, reuse=2):

| Config | DiT | PSNR | vs gold | Label |
|--------|----:|-----:|--------:|-------|
| no TR | 176.2 s | 29.5 dB | −4 dB | high |
| TR 4:30 | 111.5 s | ~20 dB | −13 dB | good |
| Schedule `50→45→30` | 79.3 s | ~19 dB | −14 dB | preview |

## Trade-off Summary

| Setting | Speed | Quality (PSNR) | Label | VRAM |
|---------|-------|---------------:|-------|------|
| BF16 DiT | Slowest | 33.5 dB | gold | High |
| INT8 DiT (default) | Fast | 29.5 dB | high | Low |
| FP8 DiT (gfx942) | Fastest | 25.9 dB | high | Low |
| TR 4:30 | −33% | 20.4 dB | good | Lower |
| TR schedule 50→45→30 | −50% | 19.3 dB | preview | Lower |
| reuse=3 + TR 4:45 | −65% | ~15 dB | preview | Lower |
| INT8 VAE | Same | Same | — | −61% VAE |
| GPU Sampler | Lower latency | Same | — | Same |

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
3. **Low quality**: Disable INT8 (`H3_INT8_MLP=0`) for BF16 (+4 dB)
4. **FP8 not working**: Only available on MI300X (gfx942)

## Examples

### Fox-s2 (Short Video)

```bash
# Default (INT8, fast, 29.5 dB)
./h3 -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 \
  --steps 2 --layers 35 --reuse 1 -o fox.mp4

# BF16 (high quality, 33.5 dB)
H3_INT8_MLP=0 ./h3 -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 \
  --steps 2 --layers 35 --reuse 1 -o fox-hq.mp4
```

### 15 s Cinematic (Long Video)

```bash
# Balanced (INT8 + TR, 20.4 dB, ~145s)
H3_INT8_VAE=1 H3_TOKEN_REDUCTION=1 H3_GPU_SAMPLER=1 \
  ./h3 -p "A cinematic shot of a fox in the forest." \
  --width 864 --height 480 --frames 362 \
  --steps 20 --layers 45 --reuse 2 -o fox-15s.mp4

# Fast (TR schedule, ~19 dB, ~111s)
H3_TOKEN_REDUCTION_SCHEDULE="0:4:50,10:4:45,20:4:30" \
  H3_INT8_VAE=1 H3_GPU_SAMPLER=1 \
  ./h3 -p "A cinematic shot of a fox in the forest." \
  --width 864 --height 480 --frames 362 \
  --steps 20 --layers 45 --reuse 2 --token-reduction -o fox-15s-fast.mp4

# Maximum quality (BF16, 33.5 dB, ~219s)
H3_INT8_MLP=0 ./h3 -p "A cinematic shot of a fox in the forest." \
  --width 864 --height 480 --frames 362 \
  --steps 20 --layers 45 --reuse 2 -o fox-15s-hq.mp4
```
