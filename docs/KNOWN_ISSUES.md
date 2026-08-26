# Known issues (HIP / gfx1151)

Tracked gaps that do **not** block the default `./h3` T2VA generate path on
Strix Halo (`gfx1151`). Production DiT uses BF16 activations and INT8 weights
(hipBLAS) on the default path.

## KI-001: CPU Euler sampler

**Status:** intentional on HIP until a GPU Euler path is proven

`h3_gpu_is_m5()` returns 0, so DiT uses the host Euler sampler. Metal on M5
defaults to the GPU-state sampler (`H3_GPU_SAMPLER=1`).

**Impact**

- Default CLI / `h3_generate()`: **works**. Timing and exact step numerics
  differ from M5 Metal.

## KI-002: Nearest-neighbor host scale

**Status:** HIP host fallback (no Accelerate / vImage)

Reference and preview resizes on HIP use nearest-neighbor in `h3_host.c`.
Metal uses vImage. Same-size canvases (no scale) are unaffected.

**Impact**

- Default 512² T2VA from text: **none**.
- `--ref-image` / first-last when the still is scaled to the canvas: slightly
  sharper/blockier than Metal.

## KI-003: MLX fixture numerical parity

**Status:** pending (no `misc/fixtures/` on this machine)

`make parity` / `make real-parity` skip without the MLX safetensor dumps.
HIP correctness is gated by CPU-oracle kernel tests, `make test`,
`make hip-functional`, and end-to-end smokes — not bit-exact Metal dumps.

## KI-004: Apple TensorOps kernels

**Status:** deferred (same product decision as the CUDA sibling)

M5 NAX / TensorOps shaders are not the HIP path. HIP provides its own INT8
tiles (`sdot4`) and fused MLP kernels instead of Apple hardware ops.

## Related backlog

| ID | Topic | Status |
|----|-------|--------|
| P | Remaining E2E vs Metal | GPU kernels are no longer the fox-s2 bottleneck; wall is mostly NVMe weight I/O. Headlines: [`PERFORMANCE.md`](PERFORMANCE.md) |
| V | `--ref-video-audio VIDEO AUDIO` E2E clip | Same kernels as `--ref-video`; not separately showcased |

### Profiling notes (HIP)

`--profile` sets `H3_PROFILE=1` and prints Metal-compatible phase lines plus an
`op-classes` breakdown (linear / sdpa / conv / other) from HIP events flushed
on each submit. DiT emits `load` then `Euler denoise` (or GPU/RES variants) as
per-phase deltas. On HIP, `wait`/`root-gpu` is stream-sync only; use
`op-classes` for GPU time. Tagged headlines: [`PERFORMANCE.md`](PERFORMANCE.md).

```bash
rocprofv3 --hip-trace --kernel-trace --stats -o /tmp/h3-prof -- \
  ./h3 --profile -d "$MODEL" -p "..." --width 512 --height 512 \
  --frames 22 --steps 2 --layers 35 --reuse 1 -o /tmp/prof.mp4
```

Port plan (historical): [`deprecated/AMD_HIP_PORT_PLAN.md`](deprecated/AMD_HIP_PORT_PLAN.md).
