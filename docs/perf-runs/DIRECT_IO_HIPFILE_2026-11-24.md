# O_DIRECT and hipFile weight loading: functional check and gain (2026-11-24)

Branch `exp/direct-io` (from `origin/main` a0e6ff3). Opt-in switches, default behaviour unchanged:

- `H3_WEIGHT_DIRECT=1`: O_DIRECT read of the 4 KiB-aligned superset of each tensor chunk into pinned staging, then H2D upload.
- `H3_WEIGHT_HIPFILE=1` (build with `HIPFILE=1`): `hipFileRead` of the same aligned superset into a 64 MiB device scratch, then D2D copy of the tensor slice.
- Any failure falls back to the existing buffered loader for that tensor. Neither switch changes numerics.

## Functional result

| Check | Result |
|---|---|
| fox-s2 md5, buffered (baseline) | `6d17f81ae47f48b46e6144ce92c2d167` |
| fox-s2 md5, `H3_WEIGHT_DIRECT=1` | `6d17f81ae47f48b46e6144ce92c2d167` (match) |
| fox-s2 md5, `H3_WEIGHT_HIPFILE=1` | `6d17f81ae47f48b46e6144ce92c2d167` (match in all 6 hipFile runs, including the profiled one) |
| Fallbacks to buffered in hipFile profiled run (`H3_PROFILE=1`) | 0 |
| `make HIP_ARCH=gfx1201 HIPFILE=1 hip-test` | completed, no FAIL or skip lines |

## Performance: fox-s2 (512x512, 22 frames, 2 steps), cache dropped before each run

Three interleaved cold repetitions per mode (`posix_fadvise(DONTNEED)` on the FL2VA shards first). Times in seconds.

| Mode | Wall (mean of 3) | DiT total (mean) | Qwen total (mean) | Wall vs buffered |
|---|---|---|---|---|
| Buffered | 36.51 | 18.44 | 11.02 | baseline |
| O_DIRECT | 33.57 | 16.23 | 9.85 | -8.1% |
| hipFile | 24.20 | 10.57 | 8.51 | -33.7% |

Spread across the three runs was under 1.5 s for each mode. The hipFile DiT weight read ran at 5.84 GiB/s (50.96 GiB in 8.72 s). The O_DIRECT and buffered runs read at roughly 4 GB/s in microbenchmarks.

Single warm-cache runs (not repeated, cache not controlled; FL2VA is about 134 GiB and host RAM is 59 GiB, so "warm" is partial): buffered 35.1 s, O_DIRECT 32.7 s, hipFile 22.7 s. Same ordering as cold.

## End-to-end comparison: fox-fast and fox-15s

Same build, buffered vs `H3_WEIGHT_HIPFILE=1`, interleaved runs. The FL2VA page cache is dropped before every run, so weights always come from the SSD. Two disk-cache states:

- **Cold** (`H3_DISK_CACHE=0`): first run or a new prompt. Qwen, AdaLN, DiT and VAE weights are all read (114.3 GiB).
- **Disk-cache hit** (`H3_DISK_CACHE=1`, primed by one untimed run): the prompt embedding and AdaLN schedule come from the disk cache, so Qwen and the AdaLN weights are skipped (43.1 GiB read).

"IO wall" is the new `io-wall=` field on each `weight-load` profile line: the wall time during which at least one file load is in flight, summed over phases. The `read=`/`upload=` fields are summed across loader threads (buffered fox-fast cold shows 154 thread-seconds inside a 46 s run), so they cannot be compared with wall time.

| Case | Disk cache | Runs | Buffered E2E | hipFile E2E | E2E saved | Buffered IO wall | hipFile IO wall | IO saved |
|---|---|---|---|---|---|---|---|---|
| fox-fast (512x512, 22 frames, 20 steps) | cold | 3 | 45.94 s | 29.35 s | -16.59 s (-36.1%) | 30.87 s | 20.77 s | -10.10 s (-32.7%) |
| fox-fast | hit | 3 | 22.56 s | 16.58 s | -5.98 s (-26.5%) | 11.32 s | 7.74 s | -3.58 s (-31.6%) |
| fox-15s (864x480, 15 s, 20 steps, TR) | cold | 2 | 346.84 s | 334.38 s | -12.46 s (-3.6%) | 31.43 s | 23.52 s | -7.91 s (-25.2%) |
| fox-15s | hit | 2 | 324.99 s | 319.18 s | -5.81 s (-1.8%) | 11.30 s | 7.78 s | -3.52 s (-31.2%) |

- md5 is identical between modes in every case (fox-fast `c36b933c...`, fox-15s `58570166...`). There were no hipFile fallbacks.
- Effective weight IO bandwidth: buffered 3.7 to 3.8 GiB/s, hipFile 5.5 to 5.6 GiB/s.
- Run-to-run spread: fox-fast hipFile runs differ by under 0.1 s. Buffered fox-fast cold ranges from 44.30 to 48.96 s. fox-15s runs differ by under 1.9 s in both modes.

### Where the saving comes from

| Case | Disk cache | IO wall | Euler denoise | Everything else |
|---|---|---|---|---|
| fox-fast | cold | -10.10 s | -4.15 s | -2.33 s |
| fox-fast | hit | -3.58 s | -2.15 s | -0.25 s |
| fox-15s | cold | -7.91 s | -3.03 s | -1.51 s |
| fox-15s | hit | -3.52 s | -1.91 s | -0.38 s |

**IO.** The DiT load stages shrink the most. For fox-fast cold, AdaLN load goes from 7.27 to 4.53 s and the transformer core load from 9.12 to 5.93 s. The video VAE load goes from 1.90 to 1.53 s.

**Denoise (not IO, cause not identified).** In buffered mode the Euler denoise is 2 to 4 s slower, even though it submits the same kernels (fox-fast: direct=5115, attention=495 in both modes). The extra time is GPU time in the `linear` op class: 8.18 s buffered vs 4.85 s hipFile for fox-fast cold, and 6.32 vs 4.82 s for fox-fast with a disk-cache hit. It is roughly a fixed cost and does not grow with step count: on fox-15s it is 1.9 to 3.0 s out of 265 s.

A control run separates the candidate causes. `H3_WEIGHT_DIRECT=1` (O_DIRECT, which still uploads from pinned staging with H2D like buffered) denoises in 6.31 s with linear 4.84 s, the same as hipFile. So the slowdown follows the page-cache read path, not the upload path. Page-cache fill or reclaim on this 59 GiB host while the GPU runs is a plausible cause but is not confirmed. Until it is, count only the IO-wall column as the measured hipFile gain. The denoise part is a side effect of leaving the page cache, which O_DIRECT also gets.

**Qwen on a long prompt.** On fox-fast the Qwen phase drops from 11.2 to 8.6 s. On fox-15s (a long multi-section prompt, 4.9 s of Qwen GPU time) it does not improve: 11.2 s buffered vs 11.5 s hipFile. The hipFile loader holds one process-wide mutex, so loads overlapped with heavier Qwen compute may not gain. Not investigated.

**Memory.** fox-15s puts one 0.45 GiB DiT tensor in pinned host memory in both modes (`H3_HIP_MEMLOG`). hipFile only applies to device tensors, so that tensor still loads through the buffered path.

### Reading the result

The IO saving is a fixed cost per run: about 3.5 s with a disk-cache hit and 8 to 10 s cold. On short clips that is a large share of E2E (26 to 36%). On the 15 s clip GPU time dominates and the E2E gain is 2 to 4%.

## Caveats

- **Not P2P.** hipFile reported `p2p=0` for every read. The gain comes from the hipFile driver path with aligned superset reads, not from a verified PCIe P2P DMA from NVMe into VRAM. The P2P/ACS question is still open (`hipFileDriverGetProperties` returns 5030; no sudo on this box).
- **Alignment is mandatory.** Every DiT tensor in the FL2VA shards starts unaligned. An unaligned hipFile read falls back to POSIX inside the library and was slower than buffered (2.1 vs 3.2 GB/s in microbenchmarks). The loader always reads the aligned superset.
- **Serialized.** The hipFile loader holds one process-wide mutex and one 64 MiB scratch per tensor chunk. A per-thread scratch might raise throughput further; not tested.
- **Not covered:** `--ssd-streaming`, `H3_INT8_HOST`, Ref2VA, `--serve`, multi-GPU, gfx1151.

## Reproduce

```bash
make HIP_ARCH=gfx1201 HIPFILE=1 HIPFILE_INC=<rocm devel>/include HIPFILE_LIB=<rocm devel>/lib -j$(nproc) h3
H3_WEIGHT_HIPFILE=1 ./bench/fox-s2.sh $H3_MODEL   # md5 must be 6d17f81a...
```
