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

**Denoise: KFD queue eviction under host memory pressure.** In buffered mode the Euler denoise is 2 to 4 s slower, even though it submits the same kernels (fox-fast: direct=5115, attention=495 in both modes). The cause is that KFD evicts the process's GPU queues:

- `/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms` reads 2286 ms and 2048 ms after two buffered fox-fast runs with a disk-cache hit, and 0 ms after a hipFile run. Each buffered run jumps by about 1 s at a time during the denoise.
- The extra denoise time is 2.2 s and 2.0 s (8.49 and 8.30 s vs 6.28 s), which matches the eviction time.
- In a `rocprofv3 --kernel-trace` run, one single `h3_qkv_rope_bf16_d128_kernel` launch takes 879 ms in buffered mode (median 0.301 ms, 497 launches). No hipFile launch exceeds 0.4 ms. The profiler's `linear` class counts it because the stall lands between that op's GPU events. Host `encode` time does not change (0.49 s in both modes), so the submitting thread is not what stalls.
- Host memory explains why only buffered mode is hit. Buffered loading fills the page cache to 48.7 GiB on this 59 GiB host and leaves 0.7 GiB free when the denoise starts. The video VAE prefetch reads another 9.5 GiB during the denoise. The kernel then reclaims: kswapd scans 3.8 M pages, with 1615 allocation stalls and 2392 direct compaction stalls, all inside the denoise window. hipFile and O_DIRECT bypass the page cache, keep 28 GiB or more free, and show zero reclaim and zero eviction.
- `H3_WEIGHT_DIRECT=1` (O_DIRECT, still H2D from pinned staging) denoises in 6.31 s, the same as hipFile. So the upload path is not the cause.

Which driver path turns memory pressure into a queue eviction (for example a TTM shrinker evicting system-memory buffers, or an MMU-notifier invalidation) was not determined. This is a property of the buffered loader on a host whose RAM is smaller than the weights. Leaving the page cache avoids it, so O_DIRECT gets the same benefit as hipFile.

**Qwen on a long prompt: the hipFile loader serializes read and copy.** On fox-fast the Qwen phase drops from 11.2 to 8.6 s. On fox-15s (a long multi-section prompt, 4.9 s of Qwen GPU time vs 0.6 s) it does not improve: 11.2 s buffered vs 11.5 s hipFile. `hipFileRead` itself takes the same 8.07 s (5.81 GiB/s) in both cases. The D2D copy from scratch into the tensor takes 3.21 s on fox-15s vs 0.43 s on fox-fast, because it now competes with Qwen compute on the GPU. The loader holds one mutex across read, copy, and stream sync, so 8.07 + 3.21 = 11.28 s matches the 11.3 s IO wall. The next read waits behind each slowed copy, and loads stop overlapping compute. Buffered mode uploads H2D on a DMA engine and overlaps through the Qwen prefetch lanes. Likely fixes: release the mutex during the D2D copy, use double-buffered or per-lane scratch, or read straight into the destination when its offsets allow.

**Host memory.** fox-fast with a disk-cache hit, sampled every 0.25 s:

| Mode | Peak page cache | Minimum free RAM | Peak h3 RSS | Reclaim / compaction stalls | KFD eviction |
|---|---|---|---|---|---|
| Buffered | 48.7 GiB | 0.7 GiB | 1.6 GiB | 1615 / 2392 | 2.0 to 2.3 s |
| hipFile | grows only by the 9.5 GiB VAE prefetch | 9.9 GiB | 0.8 GiB | 0 / 0 | 0 |
| O_DIRECT | 20.5 GiB (unchanged) | 28.6 GiB | 1.5 GiB | 0 / 0 | 0 |

Page cache is reclaimable, so hipFile does not lower the minimum RAM a run needs, except for the pinned staging it skips (0.8 GiB lower RSS here). What it removes is the page-cache churn: about 28 GiB of other cached data is no longer evicted, and the reclaim that led to the eviction above disappears. The video VAE prefetch still goes through the page cache in hipFile mode, because those tensors are staged in host memory first.

**Memory.** fox-15s puts one 0.45 GiB DiT tensor in pinned host memory in both modes (`H3_HIP_MEMLOG`). hipFile only applies to device tensors, so that tensor still loads through the buffered path.

### Reading the result

The IO saving is a fixed cost per run: about 3.5 s with a disk-cache hit and 8 to 10 s cold. On short clips that is a large share of E2E (26 to 36%). On the 15 s clip GPU time dominates and the E2E gain is 2 to 4%.

## Caveats

- **Not P2P.** hipFile reported `p2p=0` for every read. The gain comes from the hipFile driver path with aligned superset reads, not from a verified PCIe P2P DMA from NVMe into VRAM. The P2P/ACS question is still open (`hipFileDriverGetProperties` returns 5030; no sudo on this box).
- **Alignment is mandatory.** Every DiT tensor in the FL2VA shards starts unaligned. An unaligned hipFile read falls back to POSIX inside the library and was slower than buffered (2.1 vs 3.2 GB/s in microbenchmarks). The loader always reads the aligned superset.
- **Serialized.** The hipFile loader holds one process-wide mutex across read, D2D copy, and sync. This costs about 3 s on the long-prompt Qwen phase (see above).
- **Big-RAM hosts.** hipFile and O_DIRECT always read from the SSD. On a host with enough RAM to keep the weights in the page cache, a buffered repeat run can be faster. Keep both switches opt-in until that case is measured.
- **Not covered:** `--ssd-streaming`, `H3_INT8_HOST`, Ref2VA, `--serve`, multi-GPU, gfx1151.

## Reproduce

```bash
make HIP_ARCH=gfx1201 HIPFILE=1 HIPFILE_INC=<rocm devel>/include HIPFILE_LIB=<rocm devel>/lib -j$(nproc) h3
H3_WEIGHT_HIPFILE=1 ./bench/fox-s2.sh $H3_MODEL   # md5 must be 6d17f81a...
```
