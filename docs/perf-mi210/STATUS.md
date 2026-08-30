# MI210 8h quality gate + device activations

GPU 0 only. No MI300. No commit unless asked.

## Quality gate (fox-fast, 20 steps)

Prompt: `A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind.`

`--width 512 --height 512 --frames 22 --steps 20 --layers 45 --reuse 2`

Times below are the quality-gate pair **before** device activations. After HBM activations the same ON clip is **45 s** E2E (`/tmp/h3-mi210/fox-fast-on-devact.mp4`, PSNR inf vs the 232 s file).

| | hipBLAS SDPA ON | `H3_SDPA_HIPBLAS=0` |
|--|-----------------|---------------------|
| E2E | **232 s** (now **45 s**) | **590 s** |
| Denoise | 154 s (sdpa 5.7 / lin 132) | 431 s (sdpa 282 / lin 132) |
| Video VAE | 60 s (sdpa 15 / lin 43) | 141 s (sdpa 95 / lin 43) |
| Clip | `/tmp/h3-mi210/fox-fast-on.mp4` | `/tmp/h3-mi210/fox-fast-off.mp4` |

ON vs OFF PSNR **~23.8 dB** (20-step trajectories diverge). Mid/first/late frames: both show a red fox with fur, face, and a walk cycle — **KEEP hipBLAS SDPA**.

## KEEP: HBM activations (not more GEMM math)

On gfx90a, `h3_gpu_tensor_new_f32` / `_bf16` / `_i8` are **pinned host**. hipBLAS SGEMM with host A/C was ~40× slower than device (64 ms vs 1.9 ms on VAE QKV). Halo unified memory hid this.

fox-s2 vs previous hipBLAS-SDPA clip: **PSNR inf** (same pixels).

| Stage | Before (pinned acts) | After (device acts) |
|-------|----------------------|---------------------|
| DiT denoise GPU | 20.7 s (lin 18.5 / sdpa 0.84) | **2.7 s** (lin 1.3 / sdpa 0.66) |
| Video VAE GPU | 59.7 s (lin 42.5 / sdpa 14.7) | **2.4 s** (lin 1.5 / sdpa 0.71) |
| **fox-s2 E2E** | **92 s** | **17 s** |

Halo v0.9.0 fox-s2 was 83–87 s.

## Repro

```bash
H3_HIP_DEVICE=0 ./h3 --profile -d /home/alex/data/HF-MODELS/MiniMax-H3 \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 --steps 2 --layers 35 --reuse 1 \
  -o /tmp/h3-mi210/fox-s2.mp4

H3_HIP_DEVICE=0 H3_SDPA_HIPBLAS=0 ./h3 --profile -d /home/alex/data/HF-MODELS/MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 --frames 22 --steps 20 --layers 45 --reuse 2 \
  -o /tmp/h3-mi210/fox-fast-off.mp4
```

Opt out: `H3_SDPA_HIPBLAS=0`, `H3_F32_HIPBLAS=0`.
