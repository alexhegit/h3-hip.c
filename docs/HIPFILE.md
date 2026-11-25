# hipFile weight loading (experimental)

`H3_WEIGHT_HIPFILE=1` loads device weights with hipFile, the ROCm counterpart of GPUDirect Storage. Each tensor chunk goes from NVMe straight into VRAM. There is no page cache and no pinned host staging, and the output is bit-identical. It is off by default and needs a build with `HIPFILE=1`.

Measurements, method, and root causes are in [`perf-runs/DIRECT_IO_HIPFILE_2026-11-24.md`](perf-runs/DIRECT_IO_HIPFILE_2026-11-24.md). Open items for other machines are tracked in [issue #23](https://github.com/alexhegit/h3-hip.c/issues/23).

## Build

hipFile ships with ROCm (`libhipfile.so`, `hipfile.h`). The TheRock wheel keeps the header and library under `_rocm_sdk_devel`:

```bash
DEVEL=<wheel>/lib/python3.14/site-packages/_rocm_sdk_devel
make HIP_ARCH=gfx1201 HIPFILE=1 HIPFILE_INC=$DEVEL/include HIPFILE_LIB=$DEVEL/lib -j$(nproc) h3
```

With a regular `/opt/rocm`, `make HIP_ARCH=... HIPFILE=1` is enough. Switching between a `HIPFILE=1` build and a default build needs `make clean`, because make does not track the flag. A build without `HIPFILE=1` ignores `H3_WEIGHT_HIPFILE=1` and prints one notice.

## Use

| Setting | What it does |
|---|---|
| `H3_WEIGHT_HIPFILE=1` | hipFile for every device tensor read from a file |
| `H3_WEIGHT_HIPFILE=auto` | hipFile only when host RAM is smaller than the weight tree (FL2VA is about 134 GiB), otherwise buffered |
| `H3_WEIGHT_DIRECT=1` | O_DIRECT into pinned staging, then H2D. Avoids page-cache pressure but is not faster than buffered |

Works for the CLI, `--serve`, FastH3, Ref2VA, and `--ssd-streaming`. Any failed read falls back to the buffered loader for that tensor.

To check that hipFile is active, run with `H3_PROFILE=1` (or `--profile`). Each phase then prints a line like this:

```
h3 profile: H3 DiT                   hipfile-reads  total=2259 p2p=0 nvfs=2259 posix=0 unaligned=0 errors=0 (cumulative)
```

`nvfs` is hipFile's fastpath, the direct read. `posix` counts bounce-buffer fallbacks and should be 0. `p2p` is always 0 in hipFile 0.5.0 because the field is not implemented. `auto` also prints which way it decided.

## Measured gain (Radeon AI PRO R9700, 59 GiB host, PCIe 4.0 x4 NVMe)

Page cache dropped before each run, md5 identical between modes.

| Case | Buffered | hipFile | E2E |
|---|---|---|---|
| fox-fast, cold (no disk cache) | 42.40 s | 27.77 s | -34.5% |
| fox-fast, disk-cache hit | 21.69 s | 16.15 s | -25.5% |
| FastH3 4-step 864x480 5 s, cold / hit | 82.92 / 61.64 s | 70.38 / 56.91 s | -15.1% / -7.7% |
| Ref2VA, one image | 30.13 s | 23.25 s | -22.8% |
| `--serve` job after `/v1/cache/release` | 24.15 s | 19.34 s | -19.9% |
| fox-15s (864x480, 15 s), cold / hit | 343.71 / 321.71 s | 331.08 / 317.91 s | -3.7% / -1.2% |
| fox-fast `--ssd-streaming` | 262.41 s | 259.62 s | -1.1% |

Weight reads run at about 6 GiB/s, against 3.7 GiB/s buffered. On this box the fastpath is a true P2P transfer: reading 60 GiB into VRAM adds about 4 GiB of DRAM writes over background, while O_DIRECT through pinned memory adds about 33 GiB. The traffic goes NVMe → root complex → GPU BAR. The saving is a fixed cost per load, so it matters most for short clips, first runs, and daemon reloads. On a 15 s clip GPU time dominates.

hipFile also removes the KFD queue evictions that buffered loading triggers on hosts with less RAM than the weights (KI-008 in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)). Buffered runs lose 1 to 6 s per run to them, and up to 20 s per `--serve` session. hipFile runs lose none, so their times repeat to within 0.1 s.

## Limits

- **Host tensors are not covered.** `H3_INT8_HOST` (which `--serve` turns on below 48 GiB), and VRAM-overflow pinned fallbacks, still load buffered.
- **Big-RAM hosts are not measured yet.** If the weights fit in the page cache, a buffered repeat run reads from RAM and may beat hipFile. `auto` stays buffered there.
- **The file system must accept O_DIRECT.** If it does not, the tensor falls back to buffered.
- **VRAM:** 4 scratch slots of 64 MiB (256 MiB).

## Knobs for diagnosis

| Variable | Effect |
|---|---|
| `H3_HIPFILE_SLOTS=1..8` | Number of VRAM scratch slots (default 4). 1 serializes read and copy |
| `H3_HIPFILE_COPY_CU=1` | Copy scratch → tensor with a compute-unit blit instead of SDMA (slower when kernels run) |
| `H3_DOWNLOAD_PAGEABLE=1` | Old device → host readback into pageable memory (causes small evictions) |
| `HIPFILE_FORCE_COMPAT_MODE=true` | hipFile library: force the bounce-buffer path (about 1.8 GiB/s) |

## Tools

- `tools/hipfile_ab.py`: interleaved buffered / hipFile / O_DIRECT runs, with E2E, load wall time, KFD eviction, host memory, and read counts.
- `tools/hipfile_p2p_check.sh`: topology, ACS, IOMMU, and `p2pmem` (run with sudo for the full report).
- `tools/kfd_eviction_trace.sh`: kernel stacks for KFD queue evictions during a command (sudo).
