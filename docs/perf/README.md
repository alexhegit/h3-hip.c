# Internal performance notes

Working notes for people changing the HIP backend. **Not** user documentation.
The public scoreboard is [`../PERFORMANCE.md`](../PERFORMANCE.md).

Do not link this directory from `README.md` except through that scoreboard.
Do not put KEEP/REJECT tables, `hipMalloc` vs GTT essays, or interleaved A/B
spreadsheets on the GitHub release.

## Layout

| Path | Role |
|------|------|
| [`HISTORY.md`](HISTORY.md) | Chronology 18–26 Aug (start here) |
| [`../PERFORMANCE.md`](../PERFORMANCE.md) | User-facing release scoreboard |
| [`PERF_BASELINE.md`](../PERF_BASELINE.md) | Internal pointer + early HIP baselines |
| [`runs/`](../perf-runs/) | Dated ledgers, tagged baselines, raw `--profile` extracts |
| [`runs/V0.9.0.md`](../perf-runs/V0.9.0.md) | Full v0.9.0 fox-s2 / fox-fast tables |
| [`runs/VS_UPSTREAM.md`](../perf-runs/VS_UPSTREAM.md) | What antirez/h3.c actually publishes vs what we ran |
| `docs/perf-day*/`, `docs/perf-night*/` | Session STATUS logs (KEEP/REJECT) |

The dated folders stay where they are (`docs/perf-day9/`, …) so old links keep
working. Add new session logs next to them; add tagged numbers under `runs/`.

## When tagging a release

1. Time fox-s2 and fox-fast with `/usr/bin/time` and `--profile` (two fox-s2
   repeats). Save grepped `h3 profile:` lines under `runs/`.
2. Write `runs/vX.Y.Z.md` with phase tables and md5.
3. Update **only** the headline table in [`../PERFORMANCE.md`](../PERFORMANCE.md)
   and the journey row. Leave STATUS.md out of that file.
4. Point the git tag at the commit that contains those docs.
5. GitHub release body = the PERFORMANCE.md “Current release” section.

## Measurement rules

- Interleave A/B; do not declare an I/O win from one E2E sample.
- Denoise `gpu-op` is the GPU baseline; `linear`/`sdpa` splits can jitter.
- fox-s2 md5 `1731f95c…` is the bit-identical check for that preset.
