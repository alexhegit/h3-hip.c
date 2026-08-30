# MI210 overnight — read this first

Cutoff was **2026-08-30 07:14 +08**. Work stayed on **one MI210 (`gfx90a`, GPU 0)**. No MI300.

## Headline

fox-s2 (512², 22 frames, `--steps 2 --layers 35 --reuse 1`):

| | gfx1151 Halo v0.9.0 | MI210 start (wave32 fallback) | MI210 now |
|--|---------------------|-------------------------------|-----------|
| E2E | 83–87 s | 241 s | **~92 s** |
| VAE GPU | ~5 s | 170 s | **~60 s** |
| Denoise GPU | ~6.3 s | 59 s | **~21 s** |

Output: `/tmp/h3-mi210/fox-s2-sdpa-bf16.mp4`

## What landed (KEEP)

1. **Portability** — `HIP_ARCH=gfx90a`, `H3_HIP_DEVICE`, 32-lane shuffles, no RDNA WMMA by default, `hipMalloc` fallback + 20 GiB HBM reserve.
2. **hipBLAS SGEMM** for large VAE f32 linear — VAE linear 73→43 s.
3. **hipBLAS attention** (QK^T + softmax + PV) for long f32 d64 (VAE) and bf16 d128 (DiT) — VAE sdpa 95→15 s, DiT sdpa 39→0.8 s.

Functional: `make hip-functional` was green before the BLAS SDPA; `h3_hip_bf16_tests` + VAE encoder roundtrip green after. Known harness miss `h3_hip_real_dit_smoke` attn.out rel_l2=1.099 (same as Halo).

## Still open (not the 12h goal, still the next levers)

- VAE f32 linear **~43 s** (already hipBLAS; needs tile/shape or fused bias, not wave kernels).
- DiT INT8 linear **~19 s**.
- Flash / MFMA attention not written; BLAS materializes `heads×seq×seq` scores (capped at 2 GiB scratch).

## Knobs

```bash
H3_HIP_DEVICE=0
H3_F32_HIPBLAS=0      # back to hand r128 f32 GEMM
H3_SDPA_HIPBLAS=0     # back to wave32 SDPA
```

Branch `mi210`, uncommitted unless you asked for a commit.
