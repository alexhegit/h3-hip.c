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

## Caveats

- **Not P2P.** hipFile reported `p2p=0` for every read. The gain comes from the hipFile driver path with aligned superset reads, not from a verified PCIe P2P DMA from NVMe into VRAM. The P2P/ACS question is still open (`hipFileDriverGetProperties` returns 5030; no sudo on this box).
- **Alignment is mandatory.** Every DiT tensor in the FL2VA shards starts unaligned. An unaligned hipFile read falls back to POSIX inside the library and was slower than buffered (2.1 vs 3.2 GB/s in microbenchmarks). The loader always reads the aligned superset.
- **Serialized.** The hipFile loader holds one process-wide mutex and one 64 MiB scratch per tensor chunk. A per-thread scratch might raise throughput further; not tested.
- **Clip length.** This is a 2-step, 22-frame clip. The saving is a fixed cost per cold load, so the relative gain on a 50-step job is smaller. Not measured.
- **Not covered:** `--ssd-streaming` (uses a different helper), `H3_INT8_HOST`, Ref2VA, `--serve`, multi-GPU, gfx1151.

## Reproduce

```bash
make HIP_ARCH=gfx1201 HIPFILE=1 HIPFILE_INC=<rocm devel>/include HIPFILE_LIB=<rocm devel>/lib -j$(nproc) h3
H3_WEIGHT_HIPFILE=1 ./bench/fox-s2.sh $H3_MODEL   # md5 must be 6d17f81a...
```
