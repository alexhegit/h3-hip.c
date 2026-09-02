# Getting started (HIP)

One source tree. Two supported HIP offload ISAs. You pass `HIP_ARCH` yourself;
the Makefile does **not** probe the GPU.

| `HIP_ARCH` | GPU | Runtime DiT | Runtime SDPA |
|---|---|---|---|
| `gfx1151` | Strix Halo (RDNA, wave32) | INT8 weights (hipBLAS), BF16 activations | rocWMMA |
| `gfx90a` | MI210 (CDNA, wave64) | BF16 weights (hipBLAS BF16 GEMM) | MFMA flash (BF16 QK, FP16 PV, FP32 accum) |

Other AMD targets are experimental. Tagged **v0.10.1** on `main` covers both
ISAs (same kernels as v0.10.0). gfx1151 fox-s2 still matches the v0.9.0 md5 gate.

## Build

```bash
git clone https://github.com/alexhegit/h3-hip.c.git
cd h3-hip.c
git checkout v0.10.1

# Strix Halo
make HIP_ARCH=gfx1151 -j$(nproc) h3

# MI210
make HIP_ARCH=gfx90a -j$(nproc) h3

./h3 --info -d /path/to/MiniMax-H3
```

`make clean` does not need `HIP_ARCH`. After switching ISA, `make clean` then
rebuild. A missing `HIP_ARCH` on any non-clean target is a hard error.

Halo fox-s2 md5 gate (same hash as v0.9.0):

```bash
make HIP_ARCH=gfx1151 halo-regression
```

MI210 unit + smokes:

```bash
make HIP_ARCH=gfx90a hip-test
```

Set `H3_MODEL` if weights are not at the Makefile default
(`/home/amd/HF-MODELS/MiniMax-H3`).

## Weights

Official [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). T2VA uses
`text_encoder/`, `transformer/`, `vae/`, and the audio decoder. See
[T2VA pipeline](T2VA-pipeline.md) for dtypes (checkpoint BF16/F32; INT8 is
runtime-only on gfx1151).

## First T2VA

```bash
MODEL=/path/to/MiniMax-H3

./h3 --profile -d "$MODEL" \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 \
  --steps 20 --layers 50 --reuse 1 \
  -o outputs/fox.mp4
```

Three fox presets (all complete MP4s, not stubs):

| Name | Knobs | What you get |
|------|-------|----------------|
| **fox-s2** | `--steps 2 --layers 35 --reuse 1` | HIP A/B; gfx1151 md5 gate |
| **fox-fast** | `--steps 20 --layers 45 --reuse 2` | ~0.9 s complete clip; 11 DiT evals |
| **fox showcase** | `--steps 20 --layers 50 --reuse 1` | README / wiki gallery fox |

For a ~15 s clip, keep fox-fast knobs and raise `--seconds 15` at 864×480
([Long video](Long-video.md)). Timings: [`docs/PERFORMANCE.md`](../PERFORMANCE.md).

Optional **`--token-reduction`** (off by default, same flag as h3-spark.c):
pairs video tokens in middle DiT blocks. Faster long T2VA; **visible quality
loss**; fox-s2 md5 gate does not apply. Generate prints a warning when on.

On a multi-GPU box bind one card: `H3_HIP_DEVICE=N` (default 0). Do not run two
weight-streaming T2VA jobs against the same NVMe at once.

## Optional knobs

```text
H3_INT8_MLP=1          # gfx90a: restore INT8 DiT (gfx1151 default)
H3_INT8_MLP=0          # gfx1151: keep BF16 DiT weights
H3_SDPA_CDNA_FLASH=0   # gfx90a: hipBLAS score-matrix SDPA fallback
H3_SDPA_CDNA_FP16_PV=0 # gfx90a flash: BF16 PV instead of FP16
```
