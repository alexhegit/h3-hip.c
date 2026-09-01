# MI210 overnight — read this first

Work is on **MI210 (`gfx90a`)**. No MI300. This box has four cards:
GPU 0 compile/debug, GPU 1 quality gates, GPU 2 long T2VA, GPU 3 spare.

## Headline

| Preset | Halo v0.9.0 | MI210 start | After hipBLAS | Device VAE/DiT | BF16 linear | **CDNA flash** |
|--------|-------------|-------------|---------------|----------------|------------:|---------------:|
| fox-s2 E2E | 83–87 s | 241 s | 92 s | 17 s | 11.28 s | **10.87 s** |
| fox-fast E2E | 95 s | — | 232 s | 45 s | 21.01 s | **18.56 s** |
| 15 s / 362f E2E | 45.0 min | — | — | — | 50 min 14.58 s | **12 min 33.29 s** |

fox-fast clip `/tmp/h3-mi210/fox-fast-p1.mp4` is pixel-identical to the 45 s device-act clip (PSNR inf).
The final scalable-SDPA clip is 24.48 dB versus `fox-fast-p1.mp4` and passes
the recognizable fur/face/motion gate.

## What landed

1. Portability + hipBLAS SGEMM/SDPA.
2. HBM activations for video VAE and DiT hot buffers.
3. **Same for Qwen, AdaLN, audio VAE, and HIP MLP/QKV temps.** Pinned A/C was still hiding ~20 s on fox-fast.
4. **gfx90a BF16-QK/F32-PV hipBLAS SDPA with bounded query chunks.** This
   removes the 8192-token ceiling and makes the public 15-second example
   runnable without a quadratic score allocation.
5. **SDPA cleanup + optional 4096-key online tile** (seq=12000 317→274 ms).
   VAE page-cache warmup during denoise (cold I/O only).
6. **CDNA default DiT path is BF16 hipBLAS**, not INT8. fox-fast **21.01 s**,
   fox-s2 **11.28 s**. Restore INT8 with `H3_INT8_MLP=1`.
7. **gfx90a wave64 MFMA flash SDPA**: online softmax, shared 32-key K/V tiles,
   no HBM score matrix, BF16 QK + FP16 PV + F32 accumulation. The 44.7k kernel
   is **4.15x** faster; full 15-second E2E is **4.00x** faster and beats Halo
   by **3.58x**.

## Quality

The CDNA flash path passes the full BF16 suite (d128 max absolute error
`1.53e-5`, zero repeatability differences). fox-fast is **29.99 dB** versus
the same BF16-linear/F32-PV reference and **21.86 dB** versus the older
INT8/p1 clip, matching the pre-flash BF16-linear quality level. The 15-second
early/middle/late frames preserve the requested composition and detail.

## Knobs

```bash
H3_HIP_DEVICE=0   # compile / microbench
H3_HIP_DEVICE=1   # fox / encoder gates
H3_HIP_DEVICE=2   # long T2VA
H3_INT8_MLP=1     # restore INT8 DiT GEMMs (more VRAM-tight, pixel-closer to older clips)
H3_BF16_HIPBLAS=0
H3_F32_HIPBLAS=0
H3_SDPA_HIPBLAS=0
H3_SDPA_CDNA_FLASH=0    # restore gfx90a hipBLAS score-materializing fallback
H3_SDPA_CDNA_FP16_PV=0 # use BF16 rather than default FP16 for flash PV
```
