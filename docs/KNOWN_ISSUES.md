# Known issues (HIP)

Tracked gaps that do **not** block the default `./h3` T2VA generate path.

Build with an explicit `HIP_ARCH` (`gfx1151`, `gfx90a`, or `gfx942`); the
Makefile does not probe the GPU. Runtime then selects kernels for that ISA:

- **gfx1151:** BF16 activations, INT8 DiT weights (hipBLAS), wave32 rocWMMA SDPA.
- **gfx90a:** BF16 activations and DiT weights (hipBLAS BF16 GEMM), wave64 MFMA
  flash SDPA. Restore INT8 with `H3_INT8_MLP=1`.
- **gfx942:** Same kernel paths as gfx90a (CDNA wave64 MFMA). INT8 opt-in via
  `H3_INT8_MLP=1`; peak VRAM ~181 GiB on fox-fast (192 GiB card, tight).
  FP8 opt-in via `H3_FP8_MLP=1` (experimental, uses hipBLASLt).
- **gfx1201 (branch `r9700`):** wave32 rocWMMA like gfx1151 with the gfx12
  accumulator row map, INT8 DiT via hipBLAS. 1344×768 needs `H3_INT8_HOST=1`
  on the 32 GiB R9700.

## KI-001: CPU Euler sampler

**Status:** resolved — GPU Euler sampler works on HIP

`H3_GPU_SAMPLER=1` enables the GPU Euler path on HIP. `h3_gpu_is_m5()`
returns 0, so it is opt-in rather than default. fox-s2 denoise: 57% faster;
fox-fast: 4.5% faster.

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

## KI-005: Daemon ↔ CLI fox-s2 md5 gate

**Status:** not implemented

[`docs/DESIGN_DHS.md`](DESIGN_DHS.md) §18.2 requires the `--serve` path to
reproduce the CLI fox-s2 md5 (extend `halo-regression` / optional
MI210/MI300X). `--serve` is experimental until that gate, plus crash and
quota drills, land. CLI `./h3 -o` is unchanged.

## KI-006: `--vsa` at 1344×768 · 15 s does not fit in 32 GiB (gfx1201)

**Status:** open, not being worked on. Measured on one Radeon AI PRO R9700
(29.86 GiB), `H3_INT8_HOST=1`, `bench/fox-15s.sh` prompt, 109334 tokens.

The DiT working set is 24.0 GiB live with the INT8 linears already parked in
host RAM. VSA setup (`h3_vsa_prepare`) then asks for another 11.63 GiB of raw
`hipMalloc` with 1.41 GiB free, and the run stops before the first step:

```
h3: DiT VSA: cannot allocate VSA buffers: 11.63 GiB for 109334 tokens, 1.41 GiB free
```

| VSA buffer | layout | size at 109334 tokens |
|---|---|---:|
| `vsa_q` / `vsa_k` / `vsa_v` / `vsa_g` / `vsa_o` | 56 heads × padded tokens × 128, BF16 | 5 × 1.56 GiB |
| `vsa_acc` | 56 × padded × 128, F32 | 3.13 GiB |
| `vsa_scores` / `vsa_mask` / pools | 56 × 1709² tiles, F32 / u8 | ~0.8 GiB |

`H3_INT8_HOST` cannot help further (the weights are already off the card), and
`H3_DEVICE_WEIGHT_RESERVE_GIB` does not cover these buffers. 864×480 · 15 s
and 1344×768 · 5 s VSA run, and FastH3 on this canvas is 1254 s.

Candidate fixes, none implemented:

1. **Reuse dead DiT activations.** During attention, the QKV projection output
   (`int8_qkv_ws`, 4.38 GiB) and the MLP intermediate (`activated`, 3.13 GiB)
   hold nothing live. Lending them to VSA covers most of the gap without
   changing numerics, but needs a lifetime audit of every block path
   (fused AdaLN, token reduction, fbc).
2. **Drop the tiled Q/K/V copies.** Have QKV/RoPE write directly in VSA tile
   order so `vsa_q/k/v` replace `query/key/value` instead of duplicating them
   (about 4.7 GiB). Touches the QKV/RoPE kernels.
3. **Run VSA in head groups.** Coarse routing and fine attention are
   independent per head, so 8 heads at a time cuts the tile buffers about
   7×, for more launches. Output stays bit-identical if the per-head order
   is kept.
4. **Smaller accumulator.** A BF16 `vsa_acc`, or accumulating in registers,
   saves up to 3.13 GiB but changes the numbers.

## Related backlog

| ID | Topic | Status |
|----|-------|--------|
| P | Remaining E2E vs Metal | gfx1151 fox-s2 wall is mostly NVMe I/O. gfx90a long T2VA is still DiT SDPA. Headlines: [`PERFORMANCE.md`](PERFORMANCE.md) |
| V | `--ref-video-audio VIDEO AUDIO` E2E clip | Same kernels as `--ref-video`; not separately showcased |
| W | gfx1201 `--vsa` at 1344×768 · 15 s | Does not fit in 32 GiB; analysis and options in KI-006 |
| R | gfx1201 (R9700) `hip-functional` Ref2VA smoke | Open. Every FL2VA sub-test passes; `test_hip_ref2va_smoke` needs `Ref2VA/` (`./tools/download_weights.sh --both`), not downloaded on the R9700 box |

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
