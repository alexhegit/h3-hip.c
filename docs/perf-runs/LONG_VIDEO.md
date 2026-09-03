# Long T2VA timings (864×480, Strix Halo)

Halo ledger for the 10 s / 15 s cinematic clips. Same knobs on MI210 (gfx90a)
(`main` 2026-09-02) run the 15 s clip in **12 min 11 s** — see
[`../PERFORMANCE.md`](../PERFORMANCE.md) and
[`../perf-mi210/STATUS.md`](../perf-mi210/STATUS.md). 10 s was not re-timed
on MI210.

Measured on **Strix Halo (gfx1151)** (AMD Ryzen AI MAX+ 395 / Radeon 8060S), official
MiniMax-H3 at `/home/amd/HF-MODELS/MiniMax-H3`. v0.9.0 numbers (26 Aug) stay
below; **`main` 2026-09-03** retimes the 15 s quality path.

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
| 15.083 s | 362 | **45.0 min** (2701.4 s) | **40.4 min** (2423.0 s) | **3.5 min** (207.1 s) | v0.9.0; [`long-15s-cinematic.mp4`](../../assets/showcase/long-15s-cinematic.mp4) |
| 15.083 s (`main` 2026-09-03) | 362 | **40 min 46 s** (2446 s) | **36 min 38 s** (2198 s) | **2.9 min** (174 s) | 2×1 @ 480 px; peak 27.9 GiB; [`long-15s-default-2026-09-03.log`](long-15s-default-2026-09-03.log) |
| 15.083 s all-opts (`main` 2026-09-03) | 362 | **27 min 3 s** (1623 s) | **23 min 10 s** (1390 s) | **2.5 min** (150 s) | sampler+TR+INT8 VAE; VAE peak 3.7 GiB; [`long-15s-all-opts-2026-09-03.log`](long-15s-all-opts-2026-09-03.log) |

For reference, fox-fast (512² · 22 frames) on this machine is I/O-bound E2E
with **24.5 s** denoise GPU on `main` ([`PERFORMANCE.md`](../PERFORMANCE.md)).

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

## 15 s quality path — 2026-09-03 (`main`)

Same prompt / knobs / seed as 26 Aug. Binary `h3-hip 0.10.1` rebuilt
`HIP_ARCH=gfx1151` at `e5b0499`. Default INT8 DiT, CPU Euler sampler, BF16 VAE,
480 px tiles. Log: [`long-15s-default-2026-09-03.log`](long-15s-default-2026-09-03.log).

| Phase | Wall (s) | Notes |
|-------|--------:|-------|
| Qwen text encoder | 26.4 | peak 4.6 GiB |
| H3 DiT load | 41.7 | peak 27.9 GiB |
| H3 DiT Euler denoise | **2197.8** | gpu-op 2197; sdpa 1613 · linear 548 · other 36; peak **27.9 GiB** |
| audio VAE decoder | 3.9 | |
| video VAE decoder | **173.6** | **2×1 tiles @ 480 px**; peak 10.2 GiB |
| **E2E** (wall clock) | **2446** | 08:00:30–08:41:16; 864×480 · 362 frames |

vs v0.9.0 2701 / 2423 / 207 s: **−9.4% E2E**, **−9.3% denoise**, **−16% VAE**.
md5 `e54e2cfa1b6d87c13d8401003a5575a7` (not the gallery file).

## 15 s TR + INT8 VAE — 2026-09-03 (`main`)

Same prompt / knobs / seed, plus `--token-reduction` and `H3_INT8_VAE=1`.
CPU Euler sampler (no `H3_GPU_SAMPLER`). Detached `nohup`. Log:
[`long-15s-tr-int8vae-2026-09-03.log`](long-15s-tr-int8vae-2026-09-03.log).

| Phase | Wall (s) | Notes |
|-------|--------:|-------|
| Qwen text encoder | 31.0 | |
| H3 DiT load | 45.7 | peak 27.9 GiB |
| H3 DiT Euler denoise | **1390.6** | gpu-op 1390; sdpa 959 · linear 404 · other 27 |
| audio VAE decoder | 3.9 | |
| video VAE decoder | **149.7** | 2×1 @ 480 px; linear 0.33 · other 104; peak **3.7 GiB** |
| **E2E** (wall clock) | **1623** | 13:01:07–13:28:10; **27 min 3 s** |

vs quality path 2446 / 2198 / 174: **−34% E2E**, **−37% denoise**, **−14% VAE**.
md5 `54e8806b28cba9a4c8e529180e2b39dc`.

## 15 s all-opts — 2026-09-03 (`main`)

`H3_GPU_SAMPLER=1 H3_TOKEN_REDUCTION=1 H3_INT8_VAE=1` plus CLI
`--token-reduction`. Detached `nohup`. Log:
[`long-15s-all-opts-2026-09-03.log`](long-15s-all-opts-2026-09-03.log).

| Phase | Wall (s) | Notes |
|-------|--------:|-------|
| Qwen text encoder | 32.3 | |
| H3 DiT load | 45.0 | peak 27.9 GiB |
| H3 DiT GPU Euler denoise | **1390.2** | sdpa 960 · linear 405 · other 26 |
| audio VAE decoder | 3.9 | |
| video VAE decoder | **149.6** | peak **3.7 GiB** |
| **E2E** (wall clock) | **1623** | 13:32:44–13:59:47; **27 min 3 s** |

Matches the no-sampler TR+INT8 VAE run to the second. md5
`0da4bff28eb6ee736bf435804ef7c682`.
