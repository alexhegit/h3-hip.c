# O_DIRECT and hipFile weight loading: functional check and gain (2026-11-24)

Branch `exp/direct-io` (from `origin/main` a0e6ff3). Opt-in switches, default behaviour unchanged:

- `H3_WEIGHT_DIRECT=1`: O_DIRECT read of the 4 KiB-aligned superset of each tensor chunk into pinned staging, then H2D upload.
- `H3_WEIGHT_HIPFILE=1` (build with `HIPFILE=1`): `hipFileRead` of the same aligned superset into VRAM scratch, then an SDMA copy of the tensor slice. The first builds used one 64 MiB scratch under a global lock; the final build uses a pool of slots (see the slot pool section). `auto` picks hipFile only when RAM is smaller than the weights. User guide: [`../HIPFILE.md`](../HIPFILE.md).
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

**Qwen on a long prompt: the hipFile loader serializes read and copy.** On fox-fast the Qwen phase drops from 11.2 to 8.6 s. On fox-15s (a long multi-section prompt, 4.9 s of Qwen GPU time vs 0.6 s) it does not improve: 11.2 s buffered vs 11.5 s hipFile. `hipFileRead` itself takes the same 8.07 s (5.81 GiB/s) in both cases. The D2D copy from scratch into the tensor takes 3.21 s on fox-15s vs 0.43 s on fox-fast, because it now competes with Qwen compute on the GPU. The loader holds one mutex across read, copy, and stream sync, so 8.07 + 3.21 = 11.28 s matches the 11.3 s IO wall. The next read waits behind each slowed copy, and loads stop overlapping compute. Buffered mode uploads H2D on a DMA engine and overlaps through the Qwen prefetch lanes.

**Fixed by moving the copy to SDMA** (see the `--ssd-streaming` section). On a single cold fox-15s hipFile run, the Qwen phase took 9.15 s (buffered 11.2 s, compute-unit copy 11.5 s), and the D2D copy took 0.77 s instead of 3.21 s. E2E was 333.44 s, against 334.38 s with the compute-unit copy and 346.84 s buffered. IO wall was 21.58 s, against 23.52 s and 31.43 s. The md5 was unchanged. That run still showed 248 ms of KFD eviction, even though free RAM stayed at 26.7 GiB or more. The cause is not known.

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

## `--ssd-streaming`, and the D2D copy moved to SDMA

`--ssd-streaming` re-reads the original BF16 DiT layers on every forward into two VRAM slots. A background thread reads the next layer while the current one computes. Those reads go through the same loader, so `H3_WEIGHT_HIPFILE=1` applies to them.

**The first hipFile version made streaming slower.** On fox-s2 with `--ssd-streaming`, `hipFileRead` took 11.0 s in total, but the D2D copies from scratch into the slot took 33.3 s. A compute-unit copy queues behind the DiT kernels. The stream fell to 1.21 GiB/s and left 6.2 s of unhidden wait, and the denoise rose from 37.8 s (buffered) to 42.1 s. This is the same contention as the long-prompt Qwen result above.

**Fix:** the hipFile loader now copies with `hipMemcpyDeviceToDeviceNoCU`, which uses SDMA and stays off the compute units. `H3_HIPFILE_COPY_CU=1` restores the compute-unit copy for A/B runs. With SDMA, fox-s2 streaming reads at 5.45 GiB/s with no unhidden wait, the copies take 2.0 s, and the denoise takes 35.3 s. The md5 is unchanged (`2dd269fd...`).

The SDMA copy costs a little when the GPU is otherwise idle. Resident fox-fast with a disk-cache hit went from 16.58 s to 17.16 s, with IO wall from 7.74 to 8.05 s (one run).

**fox-fast with `--ssd-streaming`** (20 steps, 45 layers, disk-cache hit, page cache dropped, 356 GiB streamed per run):

| Mode | Runs | E2E | Euler denoise | Stream read | Unhidden wait | KFD eviction | Minimum free RAM |
|---|---|---|---|---|---|---|---|
| Buffered | 2 | 262.41 s | 255.83 s | 4.41 to 4.46 GiB/s | 0.03 s | 0.2 to 3.9 s | 0.4 GiB |
| hipFile (SDMA copy) | 2 | 259.62 s | 255.01 s | 5.26 GiB/s | 0 | 0 | 29.1 GiB |
| O_DIRECT | 1 | 262.05 s | 255.44 s | 3.18 GiB/s | 0.03 s | 0 | 28.4 GiB |

- md5 is identical in all modes (`eadd3d0c...`). The buffered eviction ranged from 0.2 to 3.9 s across three runs, counting the priming run.
- **The E2E gain is 2.8 s (-1.1%).** On this card the BF16 forward is compute-bound. 356 GiB in 255 s needs only 1.4 GiB/s, and the buffered loader already streams at 4.4 GiB/s and hides it. The faster hipFile stream has nothing to hide. The saving comes from the video VAE load (3.75 to 2.70 s) and from avoiding evictions.
- **The memory difference is large.** Buffered streaming keeps the page cache at 49 GiB and free RAM at 0.4 GiB for the whole denoise. hipFile keeps 29 GiB free and has no evictions.
- Streaming would become IO-bound only if a 45-layer forward took less than about 12 s (51 GiB at 4.4 GiB/s), for example on a faster GPU. That was not measured.

## Follow-up cases (overnight 2026-11-25, SDMA copy)

Same build (SDMA copy), buffered vs `H3_WEIGHT_HIPFILE=1`, interleaved. Unless noted, the model page cache (FL2VA and Ref2VA) is dropped before each run and the disk cache is primed. "KFD eviction" is the largest `evicted_ms` seen for the h3 process. Every case below has one md5 across all its runs and modes.

| Case | Runs per mode | Buffered E2E | hipFile E2E | E2E saved | IO wall (buffered → hipFile) | KFD eviction (buffered / hipFile) |
|---|---|---|---|---|---|---|
| fox-fast cold (rerun) | 3 | 42.40 s | 30.27 s | -12.13 s (-28.6%) | 29.99 → 21.76 s | 1.1 to 6.4 s / 0 |
| fox-fast disk-cache hit (rerun) | 3 | 21.47 s | 16.98 s | -4.49 s (-20.9%) | 11.52 → 8.17 s | 0.8 to 1.2 s / 0 |
| fox-15s cold (rerun) | 2 | 343.71 s | 331.08 s | -12.63 s (-3.7%) | 30.50 → 21.60 s | 2.5 to 4.8 s / 0.3 s |
| fox-15s disk-cache hit (rerun) | 2 | 321.71 s | 317.91 s | -3.80 s (-1.2%) | 11.49 → 8.20 s | 0.7 to 1.2 s / 0.3 s |
| Back-to-back fox-fast, page cache kept, runs 2 and 3 | 4 | 21.58 s | 17.21 s | -4.37 s (-20.2%) | 11.61 → 8.42 s | 1.3 to 2.3 s / 0 |
| FastH3 4-step, 864x480, 5 s, disk-cache hit | 2 | 61.64 s | 56.91 s | -4.73 s (-7.7%) | 12.42 → 8.87 s | 0.6 to 1.5 s / 0.1 s |
| FastH3 4-step, 864x480, 5 s, cold | 2 | 82.92 s | 70.38 s | -12.54 s (-15.1%) | 31.12 → 22.44 s | 2.7 to 5.7 s / 0.1 s |
| Low RAM (27 GiB held), fox-fast hit | 2 | 23.42 s | 17.02 s | -6.40 s (-27.3%) | 12.35 → 8.15 s | 1.3 s / 0.05 s |
| Low RAM (27 GiB held), FastH3 hit | 1 | 62.85 s | 57.20 s | -5.65 s (-9.0%) | 12.77 → 9.00 s | 2.2 s / 0.8 s |
| Ref2VA, 1 image, 512x512, 22 frames, 2 steps | 2 | 30.13 s | 23.25 s | -6.88 s (-22.8%) | 21.83 → 15.96 s | 1.2 to 2.4 s / 0 |
| `--serve` fox-s2, first job of a session | 2 | 24.61 s | 19.53 s | -5.08 s (-20.6%) | n/a | see below |
| `--serve` fox-s2, job after `/v1/cache/release` | 6 | 24.15 s (22.21 to 26.73) | 19.34 s (19.33 to 19.34) | -4.81 s (-19.9%) | n/a | 12.8 to 20.4 s per session / 0 |

**Back-to-back runs** answer the fairness question for the earlier tables, which always dropped the page cache. Keeping the page cache helps buffered mode by only 0.8 s (22.37 s on the first run, 21.58 s on runs 2 and 3), because the 134 GiB FL2VA tree does not fit in 59 GiB. hipFile is unchanged (17.05 s and 17.21 s). The earlier comparisons did not overstate the gain on this host.

**Low RAM.** A 27 GiB balloon (anonymous memory, touched every second; swap stayed at 0.1 GiB) leaves about 32 GiB, close to a 32 GiB desktop or the Strix Halo host. Buffered fox-fast gets 1.95 s slower (21.47 to 23.42 s), and hipFile does not change (16.98 to 17.02 s). O_DIRECT in the same setup took 22.21 s, so on this case it does not beat buffered. hipFile was not completely free of evictions here: 0.8 s on the FastH3 run, with 3.9 GiB free at the minimum.

**FastH3.** The 4-step dense adapter makes the denoise short (31 s), but the 864x480 video VAE decode takes about 15 s in both modes. The disk-cache hit gain is therefore 7.7%; cold is 15.1%.

**Ref2VA** loads the Qwen vision encoder and the video VAE encoder on top of the T2VA set. All of those read faster. For example, the vision encoder phase goes from 1.58 to 1.26 s, with IO wall 0.54 to 0.24 s.

**`--serve`.** On this card the daemon sets `H3_INT8_HOST=1` itself, so the INT8 DiT linears live in pinned host memory and hipFile does not touch them. The gain comes from the other device tensors and from avoiding evictions:

- Buffered daemon sessions accumulate 12.8 s and 20.4 s of KFD eviction over four jobs. Single jobs vary from 22.2 to 26.7 s.
- hipFile jobs take 19.33 to 19.34 s every time, with zero eviction.
- `/v1/cache/release` takes about 2 s in both modes.

**The SDMA reruns** agree with the earlier compute-unit tables. hipFile fox-fast is 0.4 to 0.9 s slower than with the compute-unit copy (cold 30.27 vs 29.35 s, hit 16.98 vs 16.58 s). fox-15s cold is 3.3 s faster (331.08 vs 334.38 s), mostly from the Qwen fix: Qwen 9.01 s vs 11.17 s buffered. Buffered runs vary more than hipFile runs: fox-fast cold buffered ranges from 40.24 to 45.60 s, hipFile from 30.20 to 30.38 s.

## hipFile fastpath vs compat mode, and the residual evictions (2026-11-25)

**The fastpath is a different data path from the bounce buffer.** Every aligned hipFile read is counted as `nvfs` (the library's "fastpath"). The profile prints these counts with `H3_PROFILE=1`, on a `hipfile-reads` line per phase. `HIPFILE_FORCE_COMPAT_MODE=true` forces the POSIX bounce-buffer path. fox-fast with a disk-cache hit, 2 runs each:

| hipFile mode | E2E | DiT weight read | Read counts | Peak page cache | KFD eviction |
|---|---|---|---|---|---|
| default (fastpath) | 16.96 to 17.15 s | 5.78 GiB/s | nvfs=1998, posix=0 | 21.5 GiB | 0 |
| `HIPFILE_FORCE_COMPAT_MODE=true` | 32.23 to 32.65 s | 1.79 GiB/s | nvfs=0, posix=1998 | 47.1 GiB | 0.4 to 1.0 s |

hipFile 0.5.0 zero-fills `n_p2p_reads`, so the earlier `p2p=0` says nothing about P2P. The fastpath needs O_DIRECT, leaves the page cache untouched, and is faster than both buffered reads (3.7 GiB/s) and O_DIRECT through pinned staging (3.2 GiB/s).

**The fastpath is P2P: the data does not pass through DRAM.** Counting DRAM CAS commands (`amd_uncore`, `amd_umc_0/1` event 0x0a, 64 B each, `perf stat -a`) while the 500 DiT tensors (60.12 GiB) are read into VRAM shows the difference. Idle background is about 3.1 GiB/s read and 1.2 GiB/s write (display scanout from the iGPU). Excess over that background, scaled to each read's duration:

| Read path | Time | Throughput | Excess DRAM read | Excess DRAM write |
|---|---|---|---|---|
| hipFile fastpath (`nvfs=500`, `posix=0`) | 10.13 s | 6.37 GB/s | about 10 GiB | about 4 GiB |
| O_DIRECT into pinned host memory + H2D | 16.04 s | 4.02 GB/s | about 35 GiB | about 33 GiB |

The bounce path has to write every byte to DRAM and read it back. Its counted excess (about 33 to 35 GiB each way) is lower than the 60 GiB it moved, so the absolute counts run low (the background estimate also includes process startup). The comparison between the two paths still holds: for the same 60 GiB, the fastpath causes about one eighth of the bounce path's DRAM writes, within the background noise of this desktop.

`sudo tools/hipfile_p2p_check.sh` shows why this works without a shared switch. GPU (PCIe 5.0 x16, root port 00:01.1, behind the card's own switch) and NVMe (PCIe 4.0 x4, root ports 00:02.4/00:02.5) sit on different root ports of the Phoenix root complex. ACS `ReqRedir+ CmpltRedir+` sends peer traffic up to the root complex and IOMMU, which forwards it to the GPU's 32 GiB `p2pmem` BAR (P2PDMA through the host bridge, which Linux allows on AMD Zen). The PCIe 4.0 x4 NVMe link (about 7 GB/s) is the limit.

**Residual hipFile evictions came from pageable readback.** After the SDMA fix, hipFile runs still showed 50 to 300 ms of KFD eviction with 27 GiB free. A 0.1 s timeline put each of them at a video VAE chunk boundary, about 11 ms per chunk (7 chunks on FastH3 5 s), with no disk reads and no VRAM pressure. `hipMemcpy` into pageable host memory makes ROCm pin it in place behind an MMU notifier. Freeing or reusing that memory later evicts all queues of the process.

Fix (affects every mode):

- `h3_hip_download_bytes` stages pageable destinations through pinned memory. `H3_DOWNLOAD_PAGEABLE=1` restores the old copy.
- The video VAE reads its output patches into one reused pinned buffer, so the large per-tile readback has no extra `memcpy`.

Staging everything without the pinned VAE buffer removed the evictions but made the VAE decode 0.25 s slower, which is why the VAE buffer is pinned instead.

| Case (disk-cache hit) | Mode | Runs | E2E | Video VAE decode | KFD eviction |
|---|---|---|---|---|---|
| FastH3 864x480 5 s | hipFile, new readback | 2 | 56.38 s | 14.51 s | 0, 0 |
| FastH3 864x480 5 s | buffered, new readback | 2 | 61.24 s | 14.78 s | 1.38, 1.41 s |
| FastH3 864x480 5 s | buffered, old readback | 2 | 61.66 s | 15.40 s | 1.68, 2.15 s |
| fox-fast | hipFile, new readback | 3 | 16.99 s | 2.53 s | 0, 0, 0 |
| fox-fast | buffered, new readback | 3 | 21.53 s | 2.90 s | 0.8 to 1.3 s |
| fox-fast | buffered, old readback | 3 | 21.59 s | 2.90 s | 1.0 to 1.2 s |

md5 is unchanged (FastH3 `9ec19d9e...`, fox-fast `c36b933c...`). hipFile runs now show no eviction at all. Buffered runs keep most of theirs. Those come from page-cache pressure through a driver path that was not identified (it needs root to trace), and they are recorded as KI-008 in `docs/KNOWN_ISSUES.md`.

**Default path.** No default changes. hipFile and O_DIRECT both avoid the buffered evictions. O_DIRECT reads more slowly than buffered here, so its E2E is about the same (fox-fast hit 22.1 to 24.1 s vs 21.5 s). hipFile is faster in every case measured. An `auto` mode (hipFile when it is built in and RAM is smaller than the weights) is the natural next step. It waits for the big-RAM measurement on MI300X.

## Slot pool and `auto` (2026-11-25, final build)

The hipFile loader used to hold one process-wide mutex across read, copy, and sync. It now uses a pool of registered VRAM scratch slots, 4 × 64 MiB by default (`H3_HIPFILE_SLOTS`). A loader thread reads into a slot without a lock, queues the SDMA copy, and records the slot's event. The next user of that slot waits on the event first. Consecutive chunks of one tensor overlap read and copy, and Qwen's prefetch lanes read in parallel.

fox-fast cold, hipFile, 2 runs each:

| Slots | E2E | IO wall | Qwen phase |
|---|---|---|---|
| global lock (previous build) | 30.27 s | 21.76 s | 8.92 s |
| 1 | 31.31 s | 21.48 s | 8.95 s |
| 2 | 28.24 s | 19.60 s | 8.08 s |
| 4 (default) | 27.78 s | 19.18 s | 7.90 s |
| 8 | 27.73 s | 19.14 s | 7.86 s |

fox-15s cold: 330.35 s with 4 slots vs 331.75 s with 1, Qwen 8.04 vs 9.01 s, with no extra pinned fallbacks from the larger scratch. `--ssd-streaming` is compute-bound and does not change (39.74 vs 39.96 s on fox-s2). md5 is unchanged throughout.

`H3_WEIGHT_HIPFILE=auto` finds the weight tree as the nearest parent of the first weight file that has a `transformer/` directory (FL2VA, 134.1 GiB). It sums the `.safetensors` files under it, three levels deep, and picks hipFile when that exceeds physical RAM (59.7 GiB here), buffered otherwise.

## TAEH3, VSA, Pika, and `hip-functional` (2026-11-26)

832x480, 2 s, seed 42, disk-cache hit, page cache dropped, 2 runs per mode:

| Path | md5 (both modes) | Buffered | hipFile | E2E | KFD eviction (buffered / hipFile) |
|---|---|---|---|---|---|
| FastH3 dense + `--taeh3` | `aae31480...` | 26.08 s | 21.47 s | -17.7% | 0.4 to 0.8 s / 0 |
| VSA (`--fasth3-lora` vsa-datafree + `--vsa`) | `4043313c...` | 36.61 s | 30.19 s | -17.5% | 2.3 to 2.8 s / 0 |
| Pika H3 Flex (`--pika-lora` fl2va) | `c694fefc...` | 37.84 s | 32.14 s | -15.1% | 1.3 to 1.4 s / 0 |

The LoRA and TAEH3 files load through the same path, and the output matches in every case. The Pika Ref2VA adapter is not downloaded on this box, so `--pika-ref-lora` was not run.

`make HIP_ARCH=gfx1201 HIPFILE=1 hip-functional` passes all six sub-tests, with and without `H3_WEIGHT_HIPFILE=1`: encoder roundtrip, AudioVAE, Qwen vision, Qwen text, FL2VA multimodal, and Ref2VA. Every `ok`/metric line is identical between the two modes. This is also the first Ref2VA smoke pass on gfx1201 (the Ref2VA weights are now on the box).

Both weight trees fit the loader's file cache: FL2VA and Ref2VA have 29 `.safetensors` files each, 58 in total against a limit of 64 open O_DIRECT handles. Files beyond the limit would fall back to buffered reads. Switching FL2VA ↔ Ref2VA inside one `--serve` daemon was not run.

Final A/B with `tools/hipfile_ab.py` (page cache dropped, 2 runs each, md5 identical, `posix=0`):

| fox-fast | Buffered | hipFile | E2E | IO wall | KFD eviction |
|---|---|---|---|---|---|
| cold | 42.40 s | 27.77 s | -34.5% | 29.90 → 19.18 s | 1.9 to 2.0 s / 0 |
| disk-cache hit | 21.69 s | 16.15 s | -25.5% | 11.53 → 7.33 s | 0.9 to 1.4 s / 0 |

## Caveats

- **P2P evidence is indirect.** It rests on DRAM traffic counts and throughput, because the `p2p` counter is not implemented in hipFile 0.5.0. The absolute DRAM counts run low (see the fastpath section). The conclusion rests on the ratio between the two paths.
- **Alignment is mandatory.** Every DiT tensor in the FL2VA shards starts unaligned. An unaligned hipFile read falls back to POSIX inside the library and was slower than buffered (2.1 vs 3.2 GB/s in microbenchmarks). The loader always reads the aligned superset.
- **Serialized.** The hipFile loader still holds one process-wide mutex across read, copy, and sync. With the SDMA copy this no longer showed up as a cost, but it caps the loader at one read at a time.
- **Compute-unit vs SDMA numbers.** The fox-fast/fox-15s tables before the SDMA section used the compute-unit copy. The overnight section reruns them with SDMA.
- **Big-RAM hosts.** hipFile and O_DIRECT always read from the SSD. On a host with enough RAM to keep the weights in the page cache, a buffered repeat run can be faster. Keep both switches opt-in until that case is measured (MI300X).
- **Low-RAM simulation** uses a balloon without mlock (RLIMIT_MEMLOCK is 8 MiB), so the kernel could have swapped it. It did not: swap stayed at 0.1 GiB.
- **Not covered:** the `H3_INT8_HOST` long-video path (hipFile does not apply to host tensors), FL2VA/Ref2VA switching inside one daemon, multi-GPU, gfx942, gfx1151.

## Reproduce

```bash
make HIP_ARCH=gfx1201 HIPFILE=1 HIPFILE_INC=<rocm devel>/include HIPFILE_LIB=<rocm devel>/lib -j$(nproc) h3
H3_WEIGHT_HIPFILE=1 ./bench/fox-s2.sh $H3_MODEL   # md5 must be 6d17f81a...

tools/hipfile_ab.py --case fox-fast --runs 3                 # buffered vs hipFile, disk-cache hit
tools/hipfile_ab.py --case fox-fast --disk-cache 0           # cold
tools/hipfile_ab.py --case fox-fast --keep-page-cache        # back-to-back
sudo tools/hipfile_p2p_check.sh                              # P2P prerequisites
```

Open items for other machines (MI300X, Strix Halo, P2P with root) are tracked in [issue #23](https://github.com/alexhegit/h3-hip.c/issues/23).
