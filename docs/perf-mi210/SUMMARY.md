# MI210 overnight — read this first

Work stayed on **one MI210 (`gfx90a`, GPU 0)**. No MI300.

## Headline

fox-s2 (512², 22 frames, `--steps 2 --layers 35 --reuse 1`):

| | gfx1151 Halo v0.9.0 | MI210 start (wave32) | After hipBLAS | **Device activations** |
|--|---------------------|----------------------|---------------|------------------------|
| E2E | 83–87 s | 241 s | 92 s | **~17 s** |
| VAE GPU | ~5 s | 170 s | ~60 s | **~2.4 s** |
| Denoise GPU | ~6.3 s | 59 s | ~21 s | **~2.7 s** |

Output: `/tmp/h3-mi210/fox-s2-devact.mp4` (PSNR inf vs the 92 s hipBLAS clip).

## What landed

1. **Portability** — `HIP_ARCH=gfx90a`, `H3_HIP_DEVICE`, 32-lane shuffles, no RDNA WMMA by default, device weights + 20 GiB HBM reserve.
2. **hipBLAS** SGEMM + SDPA (KEEP after fox-fast ON/OFF: both look like a fox).
3. **Device activations** for video VAE f32 workspace and DiT bf16/int8 hot buffers. Pinned-host A/C made hipBLAS ~40× slower on discrete GPU.

## Quality

fox-fast 20-step ON vs wave SDPA: PSNR ~24 dB but both have fur/face. After device activations the same ON clip is **45 s** E2E. fox-s2 2-step is not a quality proxy.

## Knobs

```bash
H3_HIP_DEVICE=0
H3_F32_HIPBLAS=0      # back to hand r128 f32 GEMM
H3_SDPA_HIPBLAS=0     # back to wave32 SDPA
```

Branch `mi210`, uncommitted unless you asked for a commit.
