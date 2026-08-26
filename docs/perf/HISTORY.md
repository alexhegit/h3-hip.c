# HIP optimization history (internal)

Single chronological index for gfx1151 T2VA. Session KEEP/REJECT detail stays in
the dated `STATUS.md` files; this page only stitches them.

Public scoreboard: [`../PERFORMANCE.md`](../PERFORMANCE.md).
v0.9.0 measured tables: [`../perf-runs/V0.9.0.md`](../perf-runs/V0.9.0.md).
Day-9 I/O snapshot: [`../perf-day9/SUMMARY.md`](../perf-day9/SUMMARY.md).

fox-s2 is `512² · 22 frames · --steps 2 --layers 35 --reuse 1`.
fox-fast is the upstream tutorial knobs (`--steps 20 --layers 45 --reuse 2`).
E2E on this box is cache-noisy; GPU `op-classes` is the line to trust.

## Arc

Three jobs, in order:

1. **Make the GPU cheap** (18–24 Aug). Denoise GPU 40.3 s → ~7.5 s (day-8), then
   WMMA took it to ~6.3–6.5 s at v0.9.0. Video VAE GPU 53.6 s → ~5.2 s.
2. **Stop page-locking the 31 GiB host pool** (24–26 Aug). Weights moved to the
   96 GiB VRAM carveout; AdaLN and VAE load overlap the disk.
3. **Measure a tag** (26 Aug). Official fox-s2 **83–87 s** E2E, fox-fast **95 s**.
   Remaining wall is NVMe, not kernels.

fox-s2 E2E ≈ 257 s (18 Aug) → ~117 s (19 Aug) → 83–87 s (v0.9.0).
fox-fast E2E ≈ 213 s (22 Aug) → 95 s (v0.9.0).

## Era 1 — GPU kernels (18–24 Aug)

| When | Log | fox-s2 denoise GPU | What kept |
|------|-----|-------------------:|-----------|
| 18 Aug | [`perf-runs/FOX_S2.md`](../perf-runs/FOX_S2.md) A | 40.3 s | First HIP `--profile` |
| 18–19 overnight | [`perf-overnight/STATUS.md`](../perf-overnight/STATUS.md) | ~17.3 s | Wave SDPA grid, INT8 BK=128, VAE d64 4-key, 4-way pread, quant batch 4 |
| 19 Aug | [`perf-runs/FOX_S2.md`](../perf-runs/FOX_S2.md) B | 17.4 s | E2E 117.3 s dated ledger |
| day-2 | [`perf-day2/STATUS.md`](../perf-day2/STATUS.md) | 17.3 s | pread8, packed stores; best E2E 105.9 s |
| night-3 | [`perf-night3/STATUS.md`](../perf-night3/STATUS.md) | 17.2 s | Delayed quant flush; VAE d64 Q2+4-key; VAE GPU 18.5 s |
| day-4 | [`perf-day4/STATUS.md`](../perf-day4/STATUS.md) | 13.8 s | F32 r128; INT8 t128; d128 Q2 (sdpa 8.1→5.3) |
| day-5 Q3 | [`perf-day5/STATUS.md`](../perf-day5/STATUS.md) | **13.1 s** | d128 Q3. Frozen: [`COMPARISON.md`](../perf-day5/COMPARISON.md) |
| 22 Aug eve | [`perf-eve/STATUS.md`](../perf-eve/STATUS.md) | 12.7 s | fc1 128×64 |
| 22 Aug | [`perf-runs/FOX_FAST.md`](../perf-runs/FOX_FAST.md) | (fox-fast 95.6 s GPU) | First fox-fast: E2E 213 s, denoise wall 105 s |
| night-4 | [`perf-night4/STATUS.md`](../perf-night4/STATUS.md) | 12.2 s | INT8 k+=8, Q4 |
| day-6 | [`perf-day6/STATUS.md`](../perf-day6/STATUS.md) | 11.8 s | Q5, INT8 k+=16 |
| night-5 | [`perf-night5/STATUS.md`](../perf-night5/STATUS.md) | 11.7 s | Q6; naive INT8 WMMA rejected |
| day-8 | [`perf-day8/STATUS.md`](../perf-day8/STATUS.md) | **7.49 s** | hipBLAS INT8, Q6 vec loads, grouped packed FC2; VAE d64 Q4. fox-fast denoise GPU **55.5 s** |

Q6 was exhausted at ~4% of BF16 WMMA peak. Day-9 replaced the architecture.

## Era 2 — WMMA then I/O (24–26 Aug)

Full KEEP/REJECT: [`perf-day9/STATUS.md`](../perf-day9/STATUS.md).

**Kernels (P0–P3)** — tiled WMMA d128 SDPA (45→6.7 ms micro); f32 linear and
f32 d64 SDPA on the bf16 matrix cores; tiled f32 conv1d; vectorized qkv-rope.
Tiled-SDPA tail fix: one writer per output row (reproducibility). After this,
fox-s2 denoise GPU is ~6.3–6.5 s and the run is no longer GPU-bound.

**I/O (P4–P5)** — `free` shows 31 GiB because the BIOS carved 96 GiB VRAM.
`hipHostMalloc` page-locks the small pool; `hipMalloc` uses the carveout.
Weights are device-resident with pinned staging. AdaLN prefetch depth 4.
Video VAE loader thread. Core cross-block prefetch removed (zero once quantize
fell to 0.12 s). Global pread cap rejected.

## Era 3 — v0.9.0 baseline (26 Aug)

Timed with `/usr/bin/time` (mixed page cache), md5 `1731f95c…` on fox-s2:

| Preset | E2E | Denoise GPU |
|--------|----:|------------:|
| fox-s2 | 82.9–87.3 s | 6.3–6.5 s |
| fox-fast | 94.8 s | 28.4 s (11 evals) |

An earlier same-tree fox-s2 at **73.8 s** (25 Aug, no `time`) is the profile
used for I/O *analysis* in SUMMARY.md (disk floor, AdaLN/core split). It is
not the tagged E2E. Use 83–87 s when quoting the release.

Upstream README denoise walls (M5 Max, different GPU): fox-fast 16.69 s,
token-reduction 12.60 s, four-step ~3.5 s. HIP reruns:
[`perf-runs/VS_UPSTREAM.md`](../perf-runs/VS_UPSTREAM.md). Not a port score.

## Still open (not shipped)

Ranked I/O leftover: [`perf-day9/SUMMARY.md`](../perf-day9/SUMMARY.md) backlog.
Portability: `hipMalloc` hard-fail, no `integrated` split, pin cache sized from
`_SC_PHYS_PAGES`. Discussed, not coded: denoise-window VAE prefetch, first
Euler step pipelined into core load, MI300X staging.

`v0.9.0` may still point at the pre-split squash commit; move it to the commit
that contains `perf-runs/V0.9.0.md` if the tag is published.

## Do not retry (defaults)

See each STATUS “REJECT” list. Recurring: flash multi-Q as default, mmap+memcpy
as default, global pread budget, oversized pin cache, Q6/Q7 variants after
WMMA, hipBLAS F32 on VAE shapes, `hipMallocManaged` as the weight allocator.
