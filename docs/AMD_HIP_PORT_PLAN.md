# AMD HIP Port Plan

This document records the agreed strategy for porting **h3-metal** to AMD GPUs on
Linux while preserving the project's two core advantages:

1. **Performance** — fused kernels, int8 fast paths, SSD streaming, activation
   aliasing, and other h3-specific optimizations (not a generic BLAS/DNN stack).
2. **Simplicity** — a single C binary with minimal runtime dependencies.

## Decision

**Primary backend: pure HIP.**

- **Runtime:** `h3` + `libamdhip64` + FFmpeg (same media dependency as today).
- **Build:** `hipcc` + `gcc`/`clang` + `make`, targeting `gfx1151` (Strix Halo /
  Radeon 8060S) initially.
- **Kernels:** translate `h3_shaders.metal` to `kernels/h3_kernels.hip`; implement
  MPSGraph-equivalent paths as HIP kernels, not rocBLAS/MIOpen.
- **Do not use at runtime:** rocBLAS, hipBLASLt, MIOpen, PyTorch, FlyDSL,
  TileLang, or Vulkan.

FlyDSL / TileLang may be used **temporarily during development** to prototype an
individual kernel; any accepted result must be checked in as `.hip` and must not
remain a build-time dependency.

## Target Architecture

```
h3/
├── h3_gpu.h                    # unchanged public GPU API
├── backends/
│   ├── h3_gpu_metal.m          # macOS (existing code)
│   ├── h3_gpu_hip.c            # HIP dispatch, buffers, plan cache
│   └── h3_hip_probe.c          # Linux device probe
├── kernels/
│   ├── h3_kernels.h            # shared argument structs
│   └── h3_kernels.hip          # HIP compute kernels (from Metal)
├── h3_host.c                   # replace Accelerate/vImage with swscale
├── Makefile                    # H3_BACKEND=metal|hip
└── h3_shaders.metal            # kept as reference for porting
```

## Scale

| Item | Count |
|------|------:|
| `h3_gpu.h` API functions | ~88 |
| Metal kernels total | 92 |
| Kernels for BF16 reference path | ~40 |
| Additional int8 fast-path kernels | ~25 |
| Apple-only NAX/TensorOps kernels | ~35 (defer) |

## Phase 0 — Infrastructure (1–2 weeks)

**Status (local `hip` branch):** Phase 1A core BF16 kernels landed (elementwise,
norms, linear, AdaLN/gate, QKV+RoPE, SDPA, MLP composition, Euler). Token-reduction
and patch-projection kernels remain stubbed.

- `backends/h3_gpu_hip.c` — HIP context, tensors, `h3_gpu_add_bf16`, cast/copy
- `backends/h3_hip_probe.c` — gfx1151 device probe via `hipGetDeviceProperties`
- `kernels/h3_kernels.hip` — smoke kernels (`add`, `cast`)
- `Makefile` — `H3_BACKEND=hip` (default on Linux), `HIP_ARCH=gfx1151`
- `tests/test_hip_smoke.c` — `./h3_hip_smoke` GPU smoke test
- Unimplemented `h3_gpu_*` ops stubbed in `backends/h3_gpu_hip_stubs.c`

**Goal:** build on Linux, probe GPU, load HIP module, run one smoke kernel.

| Task | Deliverable |
|------|-------------|
| 0.1 | `h3_hip_probe()` — device name, VRAM, unified memory |
| 0.2 | `h3_gpu_hip.c` skeleton — create/free, tensors, begin/submit/stats |
| 0.3 | HIP buffer management — `hipMalloc`, pinned host, BF16 I/O |
| 0.4 | Makefile — `H3_BACKEND=hip`, `HIP_ARCH=gfx1151`, kernel build rule |
| 0.5 | Platform `#ifdef` — probe branch in `h3.c` |
| 0.6 | Smoke kernel — `h3_add_bf16` or `h3_cast_f32_to_bf16` |

**Gate:**

```sh
make H3_BACKEND=hip -j8
./h3 --info -d ./MiniMax-H3
make test   # host tests pass; GPU fixtures may skip
```

## Phase 1 — BF16 DiT Block (3–4 weeks)

**Goal:** one DiT block passes parity tests; no end-to-end video yet.

### 1A — Basic ops

Port from Metal: `h3_cast_*`, `h3_add/sub_bf16`, `h3_copy_*`, `h3_silu/swiglu/gelu_bf16`,
`h3_rms/layer_norm_bf16`, `h3_linear_bf16` (16×16 tile).

### 1B — DiT core

`h3_adaln_bf16`, `h3_gate_bf16`, `h3_gate_adaln_bf16`, `h3_qkv_rope_bf16_coop`,
`h3_sdpa_bf16`, `h3_mlp_bf16`, `h3_embedding_bf16`.

### 1C — Dispatch layer

Pipeline cache, command batching, dispatch logic ported from `h3_gpu.m`.

**Gate:**

```sh
./h3_bf16_tests misc/fixtures/h3_dit_bf16.safetensors
./h3_real_dit_block_test MiniMax-H3 misc/fixtures/h3_real_dit_block0_bf16.safetensors
```

**Success:** single-block relative L2 vs MLX fixture < 1e-3 (BF16; not bit-identical).

## Phase 2 — Full DiT + Sampler (2–3 weeks)

**Goal:** 512×512, 22 frames, BF16, 50-layer denoise completes (no VAE/text yet).

| Task | Notes |
|------|-------|
| Patch projections | `h3_patch_linear_bf16`, `_map`, `_offset` |
| Fused heads | `h3_adaln_linear_bf16`, gate+slice fusions |
| Euler sampler | `h3_euler_bf16`, GPU sampler state |
| Activation alias | mirror `H3_DISABLE_DIT_ACTIVATION_ALIAS` path |

Skip for now: token reduction, int8, NAX/TensorOps.

**Gate:** `h3_real_dit_test`, `h3_dit_bench` — record gfx1151 baseline.

## Phase 3 — Text + VAE + End-to-End Video (3–4 weeks)

**Goal:** `./h3 -p "..." -o out.mp4` works on AMD.

### 3A — Qwen text encoder

`h3_text_qk_rope_bf16`, `h3_head_rms_norm_bf16`, `h3_rope_text_bf16`,
`h3_gqa_causal_bf16`, `h3_gelu_bf16`.

### 3B — Video / audio VAE

F32 conv/activation kernels (~15); conv3d via im2col+GEMM first, specialize later.

### 3C — Host replacements

| macOS | Linux |
|-------|-------|
| Accelerate/vImage | FFmpeg swscale |
| `h3_tokenizer.m` | pure C tokenizer |
| Foundation | standard C |

### 3D — Integration

Full CLI, `--ssd-streaming`, interactive session.

**Gate:**

```sh
./h3 -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 50 --reuse 1 \
  -o outputs/fox-hip.mp4
```

**Success:** playable MP4; subject and motion coherent (not pixel-identical to Metal).

## Phase 4 — Memory Optimization (2 weeks)

**Goal:** stable on 32 GB unified memory (gfx1151).

- `--ssd-streaming` DiT weights
- Phase-separated loading (text → dit → vae)
- Activation buffer aliasing
- Pinned weight buffers + async prefetch
- CLI thinning: `--layers`, `--reuse`, etc.

**Gate:** `--profile --ssd-streaming` peak < 20 GB at 512².

## Phase 5 — int8 Fast Path (3–4 weeks)

**Goal:** materially faster denoise; preserve h3 fast-path semantics.

Port order:

1. `h3_quantize_bf16_int8_*`
2. `h3_linear_int8_*` (core tile, not Morton/NAX names)
3. `h3_fc1_swiglu_int8_*`
4. `h3_qkv_project_split_int8_rope_*`
5. `h3_gate_adaln_quantize_int8`
6. `h3_linear_int8_head_major_bf16`

Keep `--use-slower-bf16-*` flags as oracle fallbacks.

**Gate:** int8 denoise ≥30% faster than BF16; video semantics match.

## Phase 6 — gfx1151 Tuning + Advanced Features (ongoing)

- Tune `h3_linear_bf16` tiles, LDS, wave size
- Flash-style SDPA
- RDNA 3.5 pitfalls (constant-cache broadcast, LDS banking — see llama.cpp #24438)
- Token reduction kernels
- `--core-reuse`, `--token-reduction`, internal canvas
- Ref2VA / FL2VA multimodal references
- Terminal `--show` preview

## Testing Matrix

| Layer | Tests |
|-------|-------|
| Host | `make test` → `h3_tests` |
| GPU unit | new `h3_hip_tests` per kernel |
| Block parity | `h3_bf16_tests`, `h3_real_dit_block_test` |
| Module | `h3_text_tests`, `h3_real_*_vae_test`, etc. |
| E2E | fox 512² prompt → MP4 |
| Performance | `h3_dit_bench`, `--profile` regression |

## Timeline (single developer, full-time)

| Phase | Duration | Milestone |
|-------|----------|-----------|
| 0 | 1–2 weeks | build + GPU probe |
| 1 | 3–4 weeks | DiT block parity |
| 2 | 2–3 weeks | full DiT denoise |
| 3 | 3–4 weeks | **first playable video** |
| 4 | 2 weeks | 32 GB stable |
| 5 | 3–4 weeks | int8 fast path |
| 6 | ongoing | tuning + advanced |

**First video:** ~10–13 weeks. **Performance-competitive build:** ~16–20 weeks.

## Explicit Non-Goals

- No rocBLAS / hipBLASLt / MIOpen runtime dependency
- No PyTorch / Python at inference time
- No permanent FlyDSL / TileLang / TVM build chain
- No Vulkan backend in v1
- No requirement to match M5 int8 speed on first release

## Reference Hardware

Initial target: **AMD Ryzen AI MAX+ PRO 395 w/ Radeon 8060S (gfx1151)**,

- 40 CU, 32 GB VRAM (unified), ROCm 7.2
- Validate on this platform before broadening `GPU_TARGETS`

## Related Reading

- Upstream: [h3-metal README](../README.md) — performance and memory notes
- llama.cpp multi-backend pattern — backend vtable and buffer semantics only
- ROCm/FlyDSL, tile-ai/tilelang — optional dev-time kernel prototypes only
