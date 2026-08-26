# Internal performance notes

Working notes for people changing the HIP backend. **Not** user documentation.
The public scoreboard is [`../PERFORMANCE.md`](../PERFORMANCE.md).

Do not link this directory from `README.md` except through that scoreboard.
Do not put KEEP/REJECT tables, `hipMalloc` vs GTT essays, or interleaved A/B
spreadsheets on the GitHub release.

## Three trees (do not rename)

| Tree | What belongs there |
|------|-------------------|
| `docs/perf/` | Indexes only: this file, [`HISTORY.md`](HISTORY.md) |
| `docs/perf-day*` / `perf-night*` / `perf-eve` / `perf-overnight` | Frozen session `STATUS.md` (KEEP/REJECT). Leave in place so old links work. |
| `docs/perf-runs/` | Dated numbers, tagged baselines, grepped `--profile` logs. README shorthand “runs” means this directory. |

## Layout

| Path | Role |
|------|------|
| [`HISTORY.md`](HISTORY.md) | Chronology 18–26 Aug (start here) |
| [`../PERFORMANCE.md`](../PERFORMANCE.md) | User-facing release scoreboard |
| [`../perf-runs/FOX_S2.md`](../perf-runs/FOX_S2.md) | fox-s2 ledger (A/B/C; v0.9.0 is a pointer) |
| [`../perf-runs/FOX_FAST.md`](../perf-runs/FOX_FAST.md) | fox-fast ledger (day-5 run; v0.9.0 is a pointer) |
| [`../perf-runs/V0.9.0.md`](../perf-runs/V0.9.0.md) | Full v0.9.0 timed tables |
| [`../perf-runs/VS_UPSTREAM.md`](../perf-runs/VS_UPSTREAM.md) | What antirez/h3.c publishes vs HIP reruns |
| `docs/perf-day*/STATUS.md` | Session logs |

## When tagging a release

1. Time fox-s2 and fox-fast with `/usr/bin/time` and `--profile` (two fox-s2
   repeats). Save grepped `h3 profile:` lines under `docs/perf-runs/`.
2. Write `docs/perf-runs/vX.Y.Z.md` with phase tables and md5.
3. Update **only** the headline table in [`../PERFORMANCE.md`](../PERFORMANCE.md)
   and the journey row. Leave STATUS.md out of that file.
4. Point the git tag at the commit that contains those docs.
5. GitHub release body = the PERFORMANCE.md “Current release” section.

## Measurement rules

- Interleave A/B; do not declare an I/O win from one E2E sample.
- Denoise `gpu-op` is the GPU baseline; `linear`/`sdpa` splits can jitter.
- fox-s2 md5 `1731f95c…` is the bit-identical check for that preset.
- Load-pipeline A/B: `tools/bench_pipeline.sh`, `bench_adaln.sh`, `bench_core.sh`,
  `bench_load_phase.sh`, `bench_device_weights.sh`, `bench_ab_io.sh`.
- Allocator / NVMe probes: `tests/probe_alloc.hip`, `probe_read.hip`,
  `probe_stage.hip`, `probe_devhost.hip`, `probe_managed.hip`.
