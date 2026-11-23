# Design: O_DIRECT weight loading (remove the page-cache copy)

Status: **proposed, unclaimed.** Nothing here is implemented. Measure
step 1 before anyone writes the loader change.

Owner: _none yet_. To claim it, comment on the PR or open an issue that
links here.

Backlog: [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md), item D.

## Problem

Every resident weight takes three hops on its way from disk to VRAM. The
path is in `backends/h3_gpu_hip.c`:

| # | Hop | Done by | Code |
|---|---|---|---|
| 1 | SSD → kernel page cache | NVMe DMA | (kernel readahead, `POSIX_FADV_WILLNEED`) |
| 2 | page cache → pinned staging buffer | CPU, inside `pread()` (`copy_to_user`) | `h3_hip_pread_all` → `h3_hip_pread_serial` |
| 3 | staging → device allocation | GPU DMA engine | `hipMemcpyAsync` in `h3_hip_load_tensor_from_file` |

Hop 2 is the copy this design removes. Hop 3 stays. On an APU the device
allocation is the carveout in the same DDR, so hop 3 is DDR → DDR and does not
cross PCIe. The design applies to both discrete and APU.

Two facts constrain the design:

- The pinned staging pool is recycled (`h3_hip_stage_take`, `h3_hip_pin_take`),
  so the buffers are already allocated when a read starts. That is what makes
  direct reads into them cheap to set up.
- A device allocation has no host alias (comment at `h3_gpu_hip.c`, "a weight
  is a `hipMalloc` device allocation with no host alias"). `O_DIRECT` cannot
  read straight into it. The target has to be the pinned staging buffer.

## Goal

Read the file into pinned staging with `O_DIRECT`, so the NVMe DMA lands in
the staging buffer and the kernel does not copy it through the page cache.
Output bytes must not change.

Not in scope: the video VAE and text encoder loaders, `H3_INT8_HOST`
parking, and the `--serve` path. They can follow once the DiT path is proven.

## Non-goals and trade-offs

- **Warm-cache runs get slower.** With the file already in page cache,
  `pread` copies from RAM. `O_DIRECT` always goes to the SSD. The gain is
  therefore conditional on the file not being cached. Pick the default with
  measurements, not by assumption.
- **The copy might not be the bottleneck.** In one R9700 `--profile` run of
  the video VAE load (log not committed), read ran at 4.67 GiB/s and upload at
  3.98 GiB/s. Those figures look bounded by SSD and PCIe, not by the CPU copy.
  Hop 2 is only worth removing if measurement shows it on the critical path.
- **O_DIRECT needs aligned I/O.** The file offset, length and buffer address
  must each be aligned to the device's logical block size. Safetensors does
  not guarantee alignment of tensor offsets, so an aligned read must cover
  more than the tensor (see "Alignment").

## Proposed read path

```
hop 1+2 now:   pread(page cache) ──copy_to_user──► staging ──DMA──► VRAM
O_DIRECT:      pread_direct(SSD) ──────────────────► staging ──DMA──► VRAM
```

`hipMemcpyAsync` and the staging ring do not change.

### Opening the file

- Keep the existing fd cache (`h3_hip_open_weight_fd`). Add a second fd per
  path, opened with `O_RDONLY | O_DIRECT`. `O_DIRECT` is set per open file
  description, so it cannot share the buffered fd.
- Query alignment with `statx(STATX_DIOALIGN)` (Linux 6.1 and later). Fall
  back to 4096 if the query is missing or fails.
- If the direct open fails (tmpfs, some overlayfs and NFS setups), or the
  file system reports no direct I/O support, use the buffered path. The
  fallback must be silent in output and counted in the profile.

### Alignment

For a tensor at file offset `off` with length `len` and alignment `A`:

```
start = off & ~(A - 1)
end   = (off + len + A - 1) & ~(A - 1)
read  [start, end) into the staging buffer
hipMemcpyAsync(device_tensor, staging + (off - start), len)
```

Overhead is at most `2A` bytes per tensor (8 KiB for 4 KiB alignment).
The staging size must include that slack. `h3_hip_stage_bucket` already
rounds up to a 32 MiB grain, so the slack fits without changing the bucket
sizes.

A read that ends at EOF returns fewer bytes than requested. Today
`h3_hip_pread_serial` treats `count <= 0` as failure. It must accept a short
read at EOF and fail only on a short read before EOF.

### Parallelism

Keep the existing 32-thread split in `h3_hip_pread_all`. Its 1 MiB chunk
boundaries are already aligned, so each thread reads an aligned range. Only
the fd changes.

An `io_uring` path with registered (fixed) buffers is an optional second step.
It is limited by `RLIMIT_MEMLOCK` and adds a dependency. It is not needed for
the first version.

### Streaming (`--ssd-streaming`)

The structure is unchanged. Only the read call differs:

```
for block b:
  wait for slot[b & 1]
  prefetch block b+1 into slot[(b + 1) & 1] on the I/O thread:
    for each of the 4 matrices (qkv, out, fc1, fc2), in file order:
      O_DIRECT read → staging → hipMemcpyAsync → slot tensor
  compute block b from slot[b & 1]
```

`read_stream_layer` calls `h3_gpu_tensor_stream_file_bf16` for each matrix.
That reaches `h3_hip_load_tensor_from_file`, so changing that one function
covers streaming and resident loading.

Streaming reads most of the BF16 DiT (about 61.7 GiB on disk) every forward.
That is far larger than RAM, so the page cache rarely keeps it. Bypassing
the cache costs little here, though it is still a measurement, not a
certainty.

### Independent improvement: double-buffer staging

Today each chunk is read, then uploaded, then synchronized, in sequence.
Two staging buffers would let chunk k+1 read while chunk k uploads. This helps
with or without `O_DIRECT`, and should be measured as its own change first.

## Switch

```
H3_WEIGHT_DIRECT=0      buffered (current behaviour)
H3_WEIGHT_DIRECT=1      O_DIRECT, fall back on unsupported file systems
H3_WEIGHT_DIRECT=auto   see "Default" below
```

Default: `0` until the measurements below show a win on the target
machines. Changing the default is a separate PR.

`H3_WEIGHT_DIRECT` changes how bytes are read, not which bytes. Add it to
`ignored_environment` in `h3_disk_cache.c`, next to `H3_PREAD_SERIAL` and
`H3_PREAD_THREADS`, which are the same kind of switch. Otherwise every
switch flip becomes a cache miss for no reason.

## Verification

1. **Byte equality (unit test).** For each of: unaligned offset, unaligned
   length, last tensor in a shard, and a tmpfs file (fallback), compare the
   direct read with a buffered read. The bytes must match exactly.
2. **Output md5.** Run fox-s2 with `H3_WEIGHT_DIRECT=0` and `=1`. The md5 must
   match. Then run `make halo-regression` (gfx1151) or `serve-regression`
   (any ISA).
3. **Profile counters.** Add `direct=` and `buffered=` byte counts and a
   `fallback=<reason>` line to the `h3 profile:` weight-load output. The
   existing `read=` and `pin=` fields stay as they are.
4. **Performance, cold and warm.**
   - Cold: `echo 3 > /proc/sys/vm/drop_caches` before each run.
   - Warm: run twice and take the second.
   - Report the DiT load time, the streamed forward time, and the peak RSS.
   - Machines: the R9700 (59 GiB RAM, 32 GiB VRAM) and Strix Halo (31 GiB
     RAM, 96 GiB carveout), both already measured in `docs/perf-runs/`.
5. **Raw device check before coding.** With `dd iflag=direct` and `fio`,
   measure buffered versus `O_DIRECT` read into the same pinned buffer size.
   Do this cold and warm. If `O_DIRECT` does not beat buffered reads on the
   target drive, stop here and record the result.

## Go / no-go

Proceed to the loader change only if step 5 shows `O_DIRECT` at least as fast
as buffered reads on a cold cache, and a measured end-to-end gain on at least
one target machine. Record the numbers in `docs/perf-runs/` either way.

## Open questions

- Does `hipHostMalloc` memory work as an `O_DIRECT` target on the target
  drivers? The alignment of those pages needs checking. The probe in
  `tests/probe_read.hip` already does pinned `pread` and is the natural
  starting point.
- Does the APU carveout change the trade-off? Hop 3 is DDR → DDR there, so
  the saving may be smaller.
- Should the fallback reason be visible outside `--profile`?
