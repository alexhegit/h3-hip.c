# AGENTS.md — h3-hip.c

HIP port of antirez/h3.c for AMD GPUs. One tree, three ISAs.

## Build

`HIP_ARCH` is **required** on every build (except `make clean`). The Makefile will error without it.

```bash
make HIP_ARCH=gfx1151 -j$(nproc) h3   # Strix Halo (RDNA)
make HIP_ARCH=gfx90a  -j$(nproc) h3   # MI210 / MI250X (CDNA2)
make HIP_ARCH=gfx942  -j$(nproc) h3   # MI300X (CDNA3)
make clean                              # no HIP_ARCH needed
```

After changing arch, always `make clean` before rebuilding.

## Test commands

```bash
# Full unit + smoke suite (requires model weights for some sub-tests)
make HIP_ARCH=gfx1151 -j$(nproc) test    # Strix Halo
make HIP_ARCH=gfx90a  -j$(nproc) test    # MI210
make HIP_ARCH=gfx942  -j$(nproc) test    # MI300X

# Functional encoders only (needs $H3_MODEL)
make HIP_ARCH=gfx942 hip-functional

# Fox-s2 md5 regression gate (Halo only)
make HIP_ARCH=gfx1151 halo-regression
# or: ./tools/halo_regression.sh [--clean] [--skip-e2e] [--strict]
```

Set `H3_MODEL=/path/to/MiniMax-H3` if weights are not at the Makefile default (`/home/amd/HF-MODELS/MiniMax-H3`).

`hip-test` skips tokenizer and av-mux sub-tests if weights/FFmpeg are missing (prints "skip:" lines). `hip-functional` requires `H3_MODEL` for all sub-tests.

## Architecture

- **Entry point:** `main.c` → `h3_cli.c` → `h3.c` (public API in `h3.h`)
- **HIP backend:** `backends/h3_gpu_hip.c` (host-side GPU calls), `kernels/h3_kernels.hip` + `kernels/h3_kernels_extra.hip` (device kernels)
- **Metal sources** (`h3_gpu.m`, `h3_shaders.metal`, `h3_metal.m`) are reference-only for upstream parity — not compiled on Linux
- **Major pipelines:** DiT (`h3_dit.c`, `h3_dit_schedule.c`), Video VAE (`h3_video_vae.c`), Audio VAE (`h3_audio_vae.c`), Text encoder (`h3_text_encoder.c`), Vision encoder (`h3_vision_encoder.c`), FFmpeg mux (`h3_ffmpeg.c`)
- **Wave mode:** gfx1151 uses wave32 (rocWMMA); gfx90a/gfx942 use wave64 (MFMA). Kernel selection is compile-time via `HIP_ARCH`.

## Key gotchas

- `H3_HIP_DEVICE=N` or `HIP_VISIBLE_DEVICES=N` selects GPU on multi-GPU boxes. Do not run two weight-streaming T2VA jobs simultaneously.
- Multi-task parallelism is **not supported** on a single GPU (shared kernel state). Use `HIP_VISIBLE_DEVICES` for multi-process isolation. See `docs/DESIGN_MULTI_TASK.md`.
- `H3_INT8_MLP=1` enables INT8 DiT on gfx90a/gfx942 (default is BF16 GEMM on CDNA).
- `H3_INT8_VAE=1` enables INT8 Video VAE weights (69% VRAM reduction, ~9% slower).
- `H3_GPU_SAMPLER=1` keeps latents on GPU during Euler denoise (large fox-s2
  win on MI300X; **not** a gfx1151 short-clip win in the 2026-09-03 retune).
- `H3_TOKEN_REDUCTION=1` halves spatial width in middle DiT blocks (~37% faster
  long video). Same as `--token-reduction`.
- `--profile` sets `H3_PROFILE=1` and prints per-phase GPU timing with op-class breakdown.
- `--show` live preview requires Kitty/Ghostty/WezTerm/Konsole; override with `H3_TERMINAL=kitty`.
- Weight loading is ~107 GiB on the T2VA path. First run is slow; page-cache miss is expected on low-RAM boxes.
- `linenoise.o` is vendored and compiled with relaxed warnings (`-Wno-conversion`).

## Model weights

Official checkpoint: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). Expected at `./MiniMax-H3/` or set `H3_MODEL`.

## Documentation

- `docs/PERFORMANCE.md` — scoreboard numbers for all ISAs
- `docs/KNOWN_ISSUES.md` — tracked gaps (CPU Euler sampler, nearest-neighbor host scale, etc.)
- `docs/DESIGN_MULTI_TASK.md` — multi-task parallelism design options
- `docs/wiki/` — Getting started, T2VA pipeline, Long video
