# MI210 overnight — read this first

Work stayed on **one MI210 (`gfx90a`, GPU 0)**. No MI300.

## Headline

| Preset | Halo v0.9.0 | MI210 start | After hipBLAS | Device VAE/DiT | **This loop** |
|--------|-------------|-------------|---------------|----------------|---------------|
| fox-s2 E2E | 83–87 s | 241 s | 92 s | 17 s | **12 s** |
| fox-fast E2E | 95 s | — | 232 s | 45 s | **23 s** |

fox-fast clip `/tmp/h3-mi210/fox-fast-p1.mp4` is pixel-identical to the 45 s device-act clip (PSNR inf).

## What landed

1. Portability + hipBLAS SGEMM/SDPA.
2. HBM activations for video VAE and DiT hot buffers.
3. **Same for Qwen, AdaLN, audio VAE, and HIP MLP/QKV temps.** Pinned A/C was still hiding ~20 s on fox-fast.

## Quality

fox-fast 20-step KEEP vs wave SDPA (fur/face). Memory-placement changes have been PSNR inf.

## Knobs

```bash
H3_HIP_DEVICE=0
H3_F32_HIPBLAS=0
H3_SDPA_HIPBLAS=0
```
