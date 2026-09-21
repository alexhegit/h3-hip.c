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
- **HTTP daemon (experimental):** `./h3 -d MODEL --serve` → `h3d.c` (loopback JSON+SSE, protocol **v1alpha**). Bind `127.0.0.1:8571` unless `H3D_BIND`/`H3D_PORT` are set. Do not expose the daemon on a public interface. Contract: [`docs/DESIGN_DHS.md`](docs/DESIGN_DHS.md). Agent plugin lives in a separate repo (`dsh-plugin-h3-hip`).
- **Wave mode:** gfx1151 uses wave32 (rocWMMA); gfx90a/gfx942 use wave64 (MFMA). Kernel selection is compile-time via `HIP_ARCH`.

## Key gotchas

- `H3_HIP_DEVICE=N` or `HIP_VISIBLE_DEVICES=N` selects GPU on multi-GPU boxes. Do not run two weight-streaming T2VA jobs simultaneously.
- Multi-task parallelism is **not supported** on a single GPU (shared kernel state). Use `HIP_VISIBLE_DEVICES` for multi-process isolation. See `docs/DESIGN_MULTI_TASK.md`.
- `H3_INT8_MLP=1` enables INT8 DiT on all ISAs (default is INT8 on all platforms). Set `H3_INT8_MLP=0` to use BF16.
- `H3_FP8_MLP=1` enables FP8 DiT on gfx942 only (experimental, uses hipBLASLt). FP8 supersedes INT8 when set.
- `H3_FP8_CLIP=0.95` (default) weight quantization clip factor. Lower values improve PSNR (+1.2 dB at 0.95 vs 1.0).
- `H3_INT8_VAE=1` enables INT8 Video VAE weights (69% VRAM reduction, ~9% slower).
- `H3_GPU_SAMPLER=1` keeps latents on GPU during Euler denoise (large fox-s2
  win on MI300X; **not** a gfx1151 short-clip win in the 2026-09-03 retune).
- `H3_TOKEN_REDUCTION=1` halves spatial width in middle DiT blocks (~37% faster
  long video). Same as `--token-reduction`.
- `--sol-attn` / `H3_SOL_ATTN=1` is **lossy** long SDPA (off by default).
  Measured on gfx1151, gfx90a, and gfx942. gfx1151 uses a separate wave32
  rocWMMA sparse kernel. Do not stack with `--token-reduction` by default.
  ≤1024 KV tiles (64k tokens) keep the static LDS route tables; longer
  sequences (1344×768 · 15 s is 109k tokens) use dynamic-shared 8/32.
  Halo 15 s three-way (dense / TR / Sol-Attn): `docs/SOL_ATTN.md` and
  `assets/showcase/fox-15s-3way-compare-gfx1151.mp4`.
- `H3_TOKEN_REDUCTION_SCHEDULE` is per-step TR (`STEP:BEGIN:END,...`); setting
  it enables TR. On-demand bench only: `./bench/fox-15s-sched.sh` — not part of
  the default `bench/` scoreboard. See [issue #4](https://github.com/alexhegit/h3-hip.c/issues/4).
- `--profile` sets `H3_PROFILE=1` and prints per-phase GPU timing with op-class breakdown.
- `--show` live preview requires Kitty/Ghostty/WezTerm/Konsole; override with `H3_TERMINAL=kitty`.
- `--serve` is experimental. Clients must send `X-H3-Protocol: v1alpha`. `-d` / `H3D_MODEL_PATH` select weights; `H3D_MODEL_PATH_FL2VA` and `H3D_MODEL_PATH_REF2VA` override per mode. Outputs go to `H3D_OUTPUT_ROOT` (default `~/.h3d/outputs`).
- Weight loading is ~107 GiB on the T2VA path. First run is slow; page-cache miss is expected on low-RAM boxes.
- `linenoise.o` is vendored and compiled with relaxed warnings (`-Wno-conversion`).

## Model weights

Official checkpoint: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3). Expected at `./MiniMax-H3/` or set `H3_MODEL`.

## Documentation

- `docs/BEST_PRACTICE.md` — quick reference for optimal settings
- `docs/PERFORMANCE.md` — scoreboard; MI300X process E2E vs DiT total
- `docs/perf-runs/MI300X_2026-09-18_full.md` — 2026-09-18 MI300X retake
- `docs/SOL_ATTN.md` — opt-in lossy Sol-Attn (speed vs quality vs dense)
- `docs/KNOWN_ISSUES.md` — tracked gaps (CPU Euler sampler, nearest-neighbor host scale, etc.)
- `docs/DESIGN_MULTI_TASK.md` — multi-task parallelism design options
- `docs/DESIGN_DHS.md` — `--serve` daemon protocol v1alpha and DSH plugin phases
- `docs/wiki/` — Getting started, T2VA pipeline, Long video
