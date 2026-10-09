# Getting started (HIP)

One source tree. Three supported HIP offload ISAs. You pass `HIP_ARCH` yourself;
the Makefile does **not** probe the GPU.

| Product | `HIP_ARCH` | Runtime DiT | Runtime SDPA |
|---|---|---|---|
| **Strix Halo** (RDNA, wave32) | `gfx1151` | INT8 weights (hipBLAS), BF16 activations | rocWMMA |
| **MI210** (CDNA2, wave64) | `gfx90a` | INT8 weights (hipBLAS), BF16 activations | MFMA flash (BF16 QK, FP16 PV, FP32 accum) |
| **MI300X** (CDNA3, wave64) | `gfx942` | INT8 weights (hipBLAS), BF16 activations | MFMA flash (BF16 QK, FP16 PV, FP32 accum) |

Timed scoreboard SKUs are Strix Halo, MI210, and MI300X. MI250 / MI250X share
`gfx90a` with MI210 but were not timed.

Tagged **v0.16.0** runs `--vsa` on Strix Halo through a wave32 rocWMMA
fine kernel, and TAEH3 conv runs as a tiled FP32 kernel. On gfx1151 the
832×480 · 5 s INT8 fixture is dense
**3477.57 s** and VSA+TAEH3 **226.69 s**
([ledger](../perf-runs/HALO_FASTH3_2026-10-09.md)). v0.15.0 added opt-in
`--fasth3-lora`, `--taeh3`, and `--vsa`. All three stay off unless set.
MI300X 832×480 · 5 s INT8 process E2E is dense **166.87 s**, FastH3
**30.12 s**, FastH3+TAEH3 **27.72 s**, VSA **32.61 s**, VSA+TAEH3
**26.02 s**. The 15 s fox-15s canvas is in
[the comparison ledger](../perf-runs/MI300X_COMPARE_2026-10-07.md).
The 4-step picture follows the prompt and is a different composition
from dense; TAEH3 is softer. v0.14.0 remains `--fbc`. v0.13.0 added
quality-path CDNA flash SDPA and Sol-Attn past 64k tokens. INT8 DiT is
default on all ISAs. Strix Halo
(gfx1151) fox-s2 **v0.9.0** md5 `1731f95c4aa582597cf83d57f46b8f9e` is the historical
gate. From v0.12.0 the default 512 px VAE tile changes fox-s2 to
`34507f072c5cabbde6592b3f70b8fa35`.

## Build

```bash
git clone https://github.com/alexhegit/h3-hip.c.git
cd h3-hip.c
git checkout v0.16.0

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

Halo fox-s2 md5 gate (v0.9.0 hash; override `H3_FOX_S2_MD5` on v0.12+):

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

Official [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). The full
repo is about **464 GiB** and contains three copies of the same components.
`h3` opens **`FL2VA/`** for T2VA and first/last-frame (~134 GiB: text encoder,
DiT, video VAE, audio VAE, tokenizer). Ordered references switch those five
paths to **`Ref2VA/`** (~134 GiB more). The root `transformer/`,
`transformer_ref/`, `text_encoder/`, and `vae/` trees are not opened.

```bash
pip install -U "huggingface_hub[cli]"
./tools/download_weights.sh --dir ./MiniMax-H3          # T2VA / FL2VA
./tools/download_weights.sh --dir ./MiniMax-H3 --ref2va # add references later
./tools/download_weights.sh --dry-run                   # print patterns only
export H3_MODEL=$PWD/MiniMax-H3
./tools/download_fasth3_lora.sh --dir ./FastH3-4-step-LoRA   # optional 4-step LoRA, ~1.4 GiB
./tools/download_fasth3_lora.sh --vsa --dir ./FastH3-4-step-LoRA   # adds the VSA adapter
./tools/download_taeh3.sh --dir ./FastH3-4-step-LoRA          # optional tiny video decoder
```

Dtypes: [T2VA pipeline](T2VA-pipeline.md). Checkpoint stays BF16/F32; INT8 is
runtime-only.

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
| **fox-s2** | `--steps 2 --layers 35 --reuse 1` | HIP A/B; gfx1151 md5 gate; MI300X warm process E2E **10.6 s** |
| **fox-fast** | `--steps 20 --layers 45 --reuse 2` | ~0.9 s complete clip; 11 DiT evals; MI300X warm process E2E **9.8 s** |
| **fox showcase** | `--steps 20 --layers 50 --reuse 1` | README / wiki gallery fox |

For a ~15 s clip, keep fox-fast knobs and raise `--seconds 15` at 864×480
([Long video](Long-video.md)). Timings: [`docs/PERFORMANCE.md`](../PERFORMANCE.md).
On MI300X (tag v0.13.0): `./bench/fox-fast.sh` warm **process
E2E 9.8 s** (DiT total **5.0 s**); `fox-s2` **10.6 s**; 15 s no-TR
**199 s**; `fox-15s.sh` **149 s**. v0.12.0 15 s no-TR was **213 s**. Ledger:
[`docs/perf-runs/MI300X_2026-09-18_full.md`](../perf-runs/MI300X_2026-09-18_full.md).

Optional **`--token-reduction`** (off by default, same flag as h3-spark.c):
pairs video tokens in middle DiT blocks. Faster long T2VA; **visible quality
loss**; fox-s2 md5 gate does not apply. Generate prints a warning when on.

Optional **`--sol-attn`** (off by default): lossy sparse SDPA for long
sequences on gfx1151, gfx90a, and gfx942. Quality path stays dense. Fixed-seed
15 s no-TR E2E/SDPA reductions: Halo −27%/−42%, MI210 −18%/−29%, MI300X
−11%/−33%. 1344×768 · 15 s on MI300X: Sol-Attn E2E **719 s** vs dense
**947 s** (−24%); sequences past 64k tokens use dynamic-shared routing.
See [`docs/SOL_ATTN.md`](../SOL_ATTN.md).

On a multi-GPU box bind one card: `H3_HIP_DEVICE=N` (default 0). Do not run two
weight-streaming T2VA jobs against the same NVMe at once.

## Daemon (experimental)

Loopback HTTP for the DSH plugin (protocol v1alpha, not stable):

```bash
./h3 -d "$MODEL" --serve
# http://127.0.0.1:8571/v1/info  header X-H3-Protocol: v1alpha
```

Env: `H3D_BIND`, `H3D_PORT`, `H3D_MODEL_PATH`, `H3D_OUTPUT_ROOT`,
`H3D_MEDIA_ROOT`, `H3D_GPUS`. Spec: [`docs/DESIGN_DHS.md`](../DESIGN_DHS.md).

## Optional knobs

```text
H3_INT8_MLP=1          # default on all ISAs; H3_INT8_MLP=0 for BF16
H3_INT8_MLP=0          # disable INT8, use BF16 DiT weights
H3_GPU_SAMPLER=1       # GPU Euler sampler (opt-in on HIP)
H3_TOKEN_REDUCTION=1   # same as --token-reduction
H3_TOKEN_REDUCTION_SCHEDULE=0:4:50,10:4:45,20:4:30  # per-step TR; or ./bench/fox-15s-sched.sh
H3_SOL_ATTN=1          # same as --sol-attn; lossy long SDPA (off = quality)
H3_INT8_VAE=1          # INT8 Video VAE weights (VRAM; fox VAE wall may not fall)
H3_VAE_TILE_PIXELS=272 # restore v0.9.0-sized VAE tiles
H3_SDPA_CDNA_FLASH=0   # gfx90a: hipBLAS score-matrix SDPA fallback
H3_SDPA_CDNA_FP16_PV=0 # gfx90a flash: BF16 PV instead of FP16
```
