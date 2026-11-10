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

**Status:** gate in v0.19.0; passes on gfx1201 only so far.

[`docs/DESIGN_DHS.md`](DESIGN_DHS.md) §18.2 requires the `--serve` path to
reproduce the CLI fox-s2 md5. `make HIP_ARCH=… serve-regression` renders
fox-s2 with the CLI, starts a daemon from the same build, and runs
`tools/h3d_smoke.py --generate --fox-md5 <CLI md5>`: the daemon renders
fox-s2 on a cache miss and again on a prepared-DiT cache hit with
`release_after`, and both must match. The reference is the build's own CLI,
so the gate works on every ISA. gfx1201: both runs `6d17f81a…`, 53 checks,
about 100 s. Not yet run on gfx1151, gfx90a or gfx942. `--serve` stays
experimental until the crash and quota drills in §18.2 land too.

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

## KI-007: `--serve` cache overflows 32 GiB cards without `H3_INT8_HOST=1`

**Status:** fixed in v0.19.0: `--serve` now sets `H3_INT8_HOST=1` itself
on a discrete card below 48 GiB unless `H3_INT8_HOST` is already set or the
DiT is not INT8 (`H3_INT8_MLP=0`, `H3_FP8_MLP`). With no environment set the
same three jobs (same prompt, new seed, new prompt), with the VAE BF16
activations merged, take 106.5 s, **52.1 s** and 101.9 s, against 212.7,
179.4 and 208.6 s before. A new prompt re-prepares the DiT (the prepared
key includes the conditioning), so reuse pays off for reruns of one prompt.

Measured on one Radeon AI PRO R9700 (29.86 GiB), 864×480 · 5 s, FastH3
4-step, jobs on one daemon.

The daemon enables the generation cache, so the prepared DiT (21.2 GiB)
stays on the card after denoise. The video VAE decoder then sees about
4 GiB free and its tensors silently fall back to pinned host memory
(`H3_HIP_MEMLOG=1` prints 901 `pinned fallback` lines). The next job's
DiT hits the cache but denoise runs 5× slower:

| daemon | job 1 | job 2 | job 2 denoise |
|---|---:|---:|---:|
| default | 212.7 s | 179.4 s | 180.0 s (job 1: 34.2 s) |
| `H3_INT8_HOST=1` | 120.6 s | **56.6 s** | |

The CLI is not affected: it runs one job with the cache off and frees the
DiT before the VAE. `H3_INT8_HOST=0` restores the old behaviour.

## Related backlog

| ID | Topic | Status |
|----|-------|--------|
| P | Remaining E2E vs Metal | gfx1151 fox-s2 wall is mostly NVMe I/O. gfx90a long T2VA is still DiT SDPA. Headlines: [`PERFORMANCE.md`](PERFORMANCE.md) |
| V | `--ref-video-audio VIDEO AUDIO` E2E clip | Same kernels as `--ref-video`; not separately showcased |
| X | Re-verify gfx1151 / gfx90a / gfx942 after the R9700 merge | Open. Merged after runtime checks on gfx1201 only; the other ISAs were compile-checked. Changes that reach every GPU: row-chunked int32 INT8 scratch, scratch release between phases, lazy INT8 MLP workspace, SDPA K/V scratch in VRAM on discrete GPUs, and the row-chunked grouped INT8 path. Run `hip-test`, `halo-regression` (gfx1151 fox-s2 `1731f95c…`), and fox-s2 / fox-fast / 832×480 · 5 s dense md5 against each ISA's last ledger. Grouped INT8 matches the old loop bit for bit on gfx1201 only with fp contraction off; CDNA codegen of the old loop epilogue may differ. 1344×768 · 15 s outputs are expected to change on every ISA (grouped INT8 no longer falls back above 65535 rows) |
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
