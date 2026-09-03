# Getting started (HIP)

One source tree. Three supported HIP offload ISAs. You pass `HIP_ARCH` yourself;
the Makefile does **not** probe the GPU.

| Product | `HIP_ARCH` | Runtime DiT | Runtime SDPA |
|---|---|---|---|
| **Strix Halo** (RDNA, wave32) | `gfx1151` | INT8 weights (hipBLAS), BF16 activations | rocWMMA |
| **MI210** (CDNA2, wave64) | `gfx90a` | BF16 weights (hipBLAS BF16 GEMM) | MFMA flash (BF16 QK, FP16 PV, FP32 accum) |
| **MI300X** (CDNA3, wave64) | `gfx942` | same as MI210 | same as MI210 |

Timed scoreboard SKUs are Strix Halo, MI210, and MI300X. MI250 / MI250X share
`gfx90a` with MI210 but were not timed.

Tagged **v0.11.0** is the three-product line plus VAE/INT8/sampler knobs. Strix Halo
(gfx1151) fox-s2 **v0.9.0** md5 `1731f95c4aa582597cf83d57f46b8f9e` is the historical
gate. On v0.11.0 the default 512 px VAE tile changes fox-s2 to
`34507f072c5cabbde6592b3f70b8fa35`.

## Build

```bash
git clone https://github.com/alexhegit/h3-hip.c.git
cd h3-hip.c
git checkout v0.11.0

# Strix Halo
make HIP_ARCH=gfx1151 -j$(nproc) h3

# MI210
make HIP_ARCH=gfx90a -j$(nproc) h3

# MI300X
make HIP_ARCH=gfx942 -j$(nproc) h3

./h3 --info -d /path/to/MiniMax-H3
```

`make clean` does not need `HIP_ARCH`. After switching ISA, `make clean` then
rebuild. A missing `HIP_ARCH` on any non-clean target is a hard error.

Halo fox-s2 md5 gate (v0.9.0 hash; override `H3_FOX_S2_MD5` on v0.11.0):

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
H3_INT8_MLP=1          # gfx90a/gfx942: INT8 DiT (gfx1151 default)
H3_INT8_MLP=0          # gfx1151: keep BF16 DiT weights
H3_GPU_SAMPLER=1       # GPU Euler sampler (opt-in on HIP)
H3_TOKEN_REDUCTION=1   # same as --token-reduction
H3_INT8_VAE=1          # INT8 Video VAE weights (VRAM; fox VAE wall may not fall)
H3_VAE_TILE_PIXELS=272 # restore v0.9.0-sized VAE tiles
H3_SDPA_CDNA_FLASH=0   # gfx90a: hipBLAS score-matrix SDPA fallback
H3_SDPA_CDNA_FP16_PV=0 # gfx90a flash: BF16 PV instead of FP16
```
