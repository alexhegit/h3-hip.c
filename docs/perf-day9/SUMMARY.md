# Stage summary — 2026-08-26

Consolidated state at the `v0.9.0` tag. `STATUS.md` in this directory holds the
working notes and the full KEEP/REJECT record; this file is the summary a reader
should start from.

Benchmark throughout ("fox s2"):

```
H3_PROFILE=1 ./h3 -d ~/HF-MODELS/MiniMax-H3 \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 --steps 2 --layers 35 --reuse 1 \
  -o /tmp/h3-profile/out.mp4
```

Output is `md5 1731f95c4aa582597cf83d57f46b8f9e` and has been for every accepted
change since the tiled-SDPA tail fix.

## Where the run stands

End to end **82.9–87.3 s** on two `/usr/bin/time` repeats at tag time
([`../perf-runs/V0.9.0.md`](../perf-runs/V0.9.0.md)); an earlier same-tree
sample hit 73.8 s without `time`. From a 104 s I/O-work starting point and
from Run B at 117 s.

| phase | wall | GPU op | read | read floor @2.2 GiB/s | disk utilization |
|---|---:|---:|---:|---:|---:|
| Qwen text encoder | 22.48s | 3.09s | 46.86 GiB | 21.30s | **95%** |
| DiT load | 33.74s | 0.67s | 50.96 GiB | 23.16s | **69%** |
| DiT Euler denoise | 7.28s | 6.65s | — | — | 0% (idle) |
| audio VAE | 0.87s | 0.44s | 0.24 GiB | 0.11s | — |
| video VAE | 8.41s | 5.23s | 9.03 GiB | 4.10s | GPU-bound |
| **total** | **72.80s** | **16.08s** | **107.09 GiB** | **48.68s** | |

DiT load stages: refine text 0.79 · **AdaLN precompute 15.70** · rope/maps 0.01 ·
**transformer core 17.04** · activations 0.20. The core splits as
`block-io 16.876s / quantize 0.121s`, i.e. it is waiting on disk essentially all
of the time.

## What the profile says

**GPU is 22% of wall (16.08s of 72.80s).** The kernel work from days 5-9 is done;
every remaining second is weight I/O or a scheduling gap around it.

**The disk is the serial resource and it sets a hard floor.** 107.09 GiB at the
measured 2.2 GiB/s sequential ceiling is 48.68s. Concurrency does not move the
ceiling — 16 pread threads reach 2.24 GiB/s and 32 reach 2.19. Two I/O-bound
phases running concurrently therefore gain nothing, which rules out most of the
obvious phase-overlap ideas.

**The text encoder is finished.** 22.48s wall against a 21.30s read floor is 95%
of the drive. Nothing but reading fewer bytes will improve it.

**The DiT load is where the slack is.** 33.74s against a 23.16s floor leaves
**10.6s**, the single largest recoverable item in the run. AdaLN manages
1.66 GiB/s and the core 1.47 GiB/s against the 2.2 ceiling.

**Denoise leaves the disk completely idle for 7.28s.** It reads zero bytes. At
2.2 GiB/s that is 16 GiB of unused read capacity, and both VAE decoders together
need only 9.27 GiB.

## Hardware envelope

The box is 128 GB of unified memory that the BIOS splits, and `free` only ever
sees one side. Getting this wrong shaped several earlier decisions.

| pool | size | HIP allocator | notes |
|---|---:|---|---|
| host system RAM | 30.98 GiB | `malloc`, page cache | `MemTotal` 32481828 kB |
| GPU VRAM carveout | 96.00 GiB | `hipMalloc` | `mem_info_vram_total`; invisible to the kernel |
| GTT | 15.49 GiB | `hipHostMalloc` | a *ceiling* on host pages the GPU may map, not a pool |

Consequences that matter:

- `hipHostMalloc` page-locks out of the small pool and every pinned byte is one
  the page cache cannot have. That, not "memory pressure" in the abstract, is why
  it runs at 12 GiB/s idle and 1.6 GiB/s at a 23 GiB footprint, and why a larger
  pin cache measured worse. `hipMalloc` draws on the large pool and holds
  91 GiB/s at the same footprint.
- Only `hipHostMalloc` is zero-copy. Device-resident weights need an explicit
  upload, and the pipeline deliberately pays that (52 GiB/s staged DMA) to avoid
  page-locking. On UMA, adding a copy was faster than not copying.
- The page cache is confined to ~31 GiB minus the process, so 107 GiB of reads
  per run can never be cached. The cold-read floor is structural.

**Peak VRAM in a full run is 15.1 GiB** against the 96 GiB carveout, so 84% of
device memory is idle at the high-water mark. Capacity is not a constraint
anywhere in the current design.

## Model footprint

Only three of the checkpoint's directories are on the video path; `FL2VA`,
`Ref2VA` and `transformer_ref` are unused variants.

| component | on disk | dtype | tensors |
|---|---:|---|---:|
| `text_encoder` (Qwen3-VL, 64 layers) | 62.13 GiB | BF16 | 1058 |
| `transformer` (DiT, 50 + 2 refiner) | 61.73 GiB | BF16 61.66 + F32 0.06 | 638 |
| `vae` | 9.70 GiB | **all F32** | 703 |
| total | 133.56 GiB | | |

133.56 GiB does not fit the 96 GiB carveout, but the phases are sequential, so
the binding number is the largest single phase (~62 GiB) and even that is far
above what the streaming loader actually holds (15.1 GiB peak).

## Weight lifecycle: every byte is read exactly once

Worth stating explicitly because it closes off a whole class of ideas.

- **DiT core** — `load_core` reads each active block once into `dit->blocks[]`
  and keeps it; `h3.c` caches the whole `h3_dit` across steps (`dit_is_cached`).
  Blocks are quantized to INT8 on arrival and the BF16 original is freed after
  the stream drains, which is why peak residency is half the streamed bytes.
- **`--ssd-streaming`** is the opt-in mode that re-reads weights every forward
  pass, and it disables INT8. The benchmark does not use it.
- **text encoder** — streamed and progressively retired; 46.86 GiB flows through
  a 4.54 GiB peak.
- **AdaLN precompute** — each block's 520 MiB `adaln_proj` is discarded after its
  small GEMM; only the precomputed result is kept.

So there is no intra-run re-read to eliminate. Skipped DiT blocks are not read at
all, but AdaLN must read all 50 because the gate-ranked skip decision is derived
from its output.

## Backlog, ranked by measured headroom

Perfect overlap plus a saturated drive would put E2E at roughly **56s** (48.68s
of disk plus the 7.28s denoise tail that overlaps nothing). Current 73.8s.

1. **Raise DiT-load disk utilization from 69%** — up to **~9s**. Pipeline depth
   and thread configuration, not architecture. Largest single item.
2. **Prefetch both VAE decoders during denoise** — **~3.4s**. The disk is 100%
   idle for 7.28s and the two decoders need 9.27 GiB of the 16 GiB available.
   No dependency obstacle; the weights do not derive from denoise output.
3. **Pipeline the first denoise step into the core load** — **~3.6s**. Block 0
   only needs block 0's weights. Entangles with the INT8 quantize batching, so
   more invasive than its size suggests.
4. **Read fewer bytes.** The only lever below the 56s floor.
   - `vae` is stored F32; an offline BF16 conversion halves 9.70 GiB, **~2.2s**,
     and looks numerically cheap.
   - Caching AdaLN projection *outputs* rather than weights: 26 GiB → 97 MiB,
     keyed by the sigma schedule.
   - Caching the INT8 quantized core: 30.8 → 15.4 GiB.
   - Offline INT8 for the text encoder would halve its 46.86 GiB (**~10.6s**) but
     changes numerics and needs a quality gate.
   Items 2-4 write derived artifacts to disk and are a product decision, not a
   unilateral one.
5. **Faster storage.** Everything above is anchored to a single NVMe at
   2.2 GiB/s. A second drive in RAID0 halves the 48.68s floor — a larger lever
   than every software item combined, and unavoidable given that 96 GiB of the
   machine's memory cannot serve as page cache.

## Portability debt

The allocator currently assumes this machine's shape and will not degrade
gracefully elsewhere.

- `h3_hip_tensor_new_ex` treats a failed `hipMalloc` as fatal, so any host whose
  carveout is smaller than the working set refuses to run rather than falling
  back to the pinned path. This is the one item that is a correctness/portability
  bug rather than a tuning gap.
- `h3_hip_device_weights()` is a hardcoded boolean with an env opt-out. It should
  be a budget derived from `hipMemGetInfo`, with per-allocation fallback.
- Nothing consults `hipGetDeviceProperties`. The `integrated` flag is the right
  discriminator for APU versus discrete, where the staging trade-off inverts:
  here device residency avoids page-locking, on a dGPU it avoids per-access PCIe
  traffic, and there the upload bandwidth (~20 GiB/s) becomes the new bottleneck
  and wants multi-buffered streaming.
- `h3_hip_pin_cache_capacity()` sizes from `_SC_PHYS_PAGES`. With weights in the
  carveout the pool only holds in-flight staging buffers (AdaLN 4 × 520 MiB, core
  4 × 308 MiB ≈ 3.3 GiB), so it should be sized from concurrency and capped by
  the GTT ceiling.

## Validation state

- Output `md5 1731f95c4aa582597cf83d57f46b8f9e`, unchanged across every accepted
  change in the I/O work.
- `make test` passes except one **pre-existing** failure,
  `h3_hip_real_dit_smoke` → `attn.out rel_l2=1.099`. It is a test-harness
  problem, not a production one: `tests/test_hip_real_dit.c` predates this work,
  the failure is identical with all new features disabled and with
  `H3_SDPA_WMMA=0`, and rendering with
  `H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT=1` gives PSNR=inf against the default
  path. See `STATUS.md` for the full argument.
- Wall time on this box drifts 15-25% between sessions with page-cache state.
  Only interleaved A/B measurements at per-stage granularity are trustworthy;
  `tools/bench_*.sh` all interleave for this reason.
