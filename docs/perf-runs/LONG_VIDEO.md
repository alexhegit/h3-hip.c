# Long T2VA timings (864×480, gfx1151)

Halo ledger for the 10 s / 15 s cinematic clips. Same knobs on gfx90a
(v0.10.0) run the 15 s clip in **12 min 33 s** — see
[`../PERFORMANCE.md`](../PERFORMANCE.md) and
[`../perf-mi210/STATUS.md`](../perf-mi210/STATUS.md). 10 s was not re-timed
on MI210.

Measured on AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`), v0.9.0 binary,
official MiniMax-H3 checkpoint at `/home/amd/HF-MODELS/MiniMax-H3`.

Shared knobs (same as fox-fast quality path):

```bash
--steps 20 --layers 45 --reuse 2   # 11 DiT evaluations
--width 864 --height 480
```

Prompt style: fal.ai timed shot list (T2VA, no references). Logs and MP4 under
`outputs/long-*-cinematic.*`.

## Summary

| Duration | Aligned frames | E2E wall | DiT denoise wall | Video VAE wall | Output |
|---------:|---------------:|---------:|-----------------:|---------------:|--------|
| 10.125 s | 243 | **24.7 min** (1479.8 s) | **21.2 min** (1273.2 s) | **2.3 min** (139.5 s) | [`assets/showcase/long-10s-cinematic.mp4`](../../assets/showcase/long-10s-cinematic.mp4) |
| 15.083 s | 362 | **45.0 min** (2701.4 s) | **40.4 min** (2423.0 s) | **3.5 min** (207.1 s) | [`assets/showcase/long-15s-cinematic.mp4`](../../assets/showcase/long-15s-cinematic.mp4) |
| 15.083 s + `--token-reduction` | 362 | **28.2 min** (1694.2 s) | **23.3 min** (1400.9 s) | **3.5 min** (209.3 s) | opt-in; [`TOKEN_REDUCTION.md`](TOKEN_REDUCTION.md) |

For reference, fox-fast (512² · 22 frames) is **95 s** E2E with **28 s** denoise
GPU on the same machine ([`PERFORMANCE.md`](../PERFORMANCE.md)).

Long clips are dominated by DiT denoise. Scaling 22 → 362 frames is ~16× in
frame count; denoise wall grows ~86× (28 s → 2423 s), reflecting super-linear
cost in temporal sequence length (attention over video+audio tokens).

## 10 s run — 2026-08-26

Command:

```bash
./h3 --profile -d /home/amd/HF-MODELS/MiniMax-H3 \
  -p "<timed shot list prompt>" \
  --width 864 --height 480 --seconds 10 \
  --steps 20 --layers 45 --reuse 2 --seed 42 \
  -o outputs/long-10s-cinematic.mp4
```

Grepped `--profile` phases (`outputs/long-10s-cinematic.log`):

| Phase | Wall (s) | GPU wait (s) | Notes |
|-------|--------:|-------------:|-------|
| Qwen text encoder | 22.0 | 12.8 | |
| H3 DiT load | 39.1 | 1.8 | |
| H3 DiT Euler denoise | **1273.2** | 0.5 | sdpa 761 s · linear 391 s |
| H3 DiT total (incl. load) | 1312.5 | 2.3 | |
| audio VAE decoder | 3.9 | 3.4 | |
| video VAE decoder | **139.5** | 131.2 | 4×2 tiles @ 272 px |
| **E2E** | **1479.8** | — | 864×480 · 243 frames · 2.3 MiB |

## 15 s run — 2026-08-26

Same command with `--seconds 15` → 362 aligned frames.
Log: `outputs/long-15s-cinematic.log`.

| Phase | Wall (s) | GPU wait (s) | Notes |
|-------|--------:|-------------:|-------|
| Qwen text encoder | 21.6 | 12.2 | |
| H3 DiT load | 41.6 | 1.5 | |
| H3 DiT Euler denoise | **2423.0** | 0.7 | sdpa 1672 s · linear 574 s |
| H3 DiT total (incl. load) | 2464.7 | 2.2 | |
| audio VAE decoder | 5.6 | 5.1 | |
| video VAE decoder | **207.1** | 197.0 | 4×2 tiles @ 272 px |
| **E2E** | **2701.4** | — | 864×480 · 362 frames · 3.5 MiB |

10 s → 15 s: denoise **1.90×**, video VAE **1.48×**, E2E **1.83×**.

## 15 s + `--token-reduction` — 2026-09-02

Same prompt / knobs / seed as the quality-path 15 s run, plus
`--token-reduction`. Binary `h3-hip 0.10.1` (`HIP_ARCH=gfx1151`). Not the
tagged scoreboard. Full table: [`TOKEN_REDUCTION.md`](TOKEN_REDUCTION.md).

| Phase | Wall (s) | Notes |
|-------|--------:|-------|
| Qwen text encoder | 32.5 | I/O band vs 21.6 s on 26 Aug |
| H3 DiT load | 45.4 | peak 23.3 GiB |
| H3 DiT Euler denoise | **1400.9** | gpu-op 1399.9; sdpa 965.5 · linear 405.5; peak 25.7 GiB |
| audio VAE decoder | 3.8 | |
| video VAE decoder | **209.3** | 4×2 tiles @ 272 px; same as quality path |
| **E2E** | **1694.2** | 864×480 · 362 frames · 3.3 MiB |

Quality path 2701 s → 1694 s (**−37%** E2E, **−42%** denoise). VAE did not move.
