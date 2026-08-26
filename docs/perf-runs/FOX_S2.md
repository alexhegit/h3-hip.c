# Fox video latency ledger (HIP / gfx1151)

Machine: AMD Ryzen AI MAX+ 395 / Radeon 8060S (`gfx1151`).  
Purpose: dated E2E + phase timings for ongoing optimization and A/B comparison.

## Canonical preset (fox short)

```bash
MODEL=/home/amd/HF-MODELS/MiniMax-H3   # or $MODEL
./h3 --profile -d "$MODEL" \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 \
  --steps 2 --layers 35 --reuse 1 \
  -o /tmp/h3-profile/fox-s2.mp4
```

| Knob | Value | Notes |
|------|------:|-------|
| resolution | 512×512 | |
| frames | 22 | |
| denoise steps | 2 | short smoke; showcase uses 20 |
| layers | 35 | must stay in `[35, 50]` |
| reuse | 1 | |

### How to read numbers

| Field | Meaning |
|-------|---------|
| **E2E wall** | `/usr/bin/time` process wall (includes mux + gaps between phases) |
| **phase wall** | `--profile` mark delta for that phase |
| **encode** | Host time inside `begin`→`submit` (can overlap GPU) |
| **wait / root-gpu** | `hipStreamSynchronize` only — often near-zero |
| **GPU (`op-classes`)** | HIP-event exclusive kernel time (authoritative GPU split) |
| **linear / sdpa / conv / other** | Split of `gpu-op` |

Raw stderr for a run can live under `/tmp/h3-profile/` (not always committed).

**Current tagged baseline:** [`V0.9.0.md`](V0.9.0.md) (2026-08-26, git `e0d8558`).

---

## Run v0.9.0 — day-9 tree (2026-08-26 13:32 CST)

Full write-up including fox-fast: [`V0.9.0.md`](V0.9.0.md). Two timed fox-s2 repeats, mixed page cache.

| Meta | |
|------|--|
| Git | `e0d8558` (intended `v0.9.0` tree) |
| Output | `/tmp/h3-profile/fox-s2-v0.9.0.mp4` · md5 `1731f95c4aa582597cf83d57f46b8f9e` |
| Logs | `docs/perf-runs/fox-s2-v0.9.0-r1.log`, `…-r2.log` |
| E2E wall | **87.30 s** then **82.88 s** (`time` r1: user 15.49 · sys 41.91) |
| Phase-wall sum | **86.23 s** / **81.76 s** |
| DiT layer policy | same 15 gate-ranked skips as Run B |

### Phase table (r1)

| Phase | wall (s) | GPU op (s) | linear | sdpa | conv | other | Notes |
|-------|---------:|-----------:|-------:|-----:|-----:|------:|-------|
| Qwen text encoder | 29.228 | 2.056 | 1.782 | 0.001 | 0.000 | 0.273 | I/O; r2 was 23.95 |
| H3 DiT **load** | 38.580 | 0.667 | 0.432 | 0.000 | 0.000 | 0.235 | AdaLN 18.76 · core 18.39 |
| H3 DiT **Euler denoise** (2 steps) | 6.992 | 6.496 | 4.403 | 1.719 | 0.000 | 0.374 | r2 gpu-op 6.29 |
| audio VAE decoder | 1.666 | 0.436 | 0.000 | 0.000 | 0.350 | 0.086 | tiled conv1d |
| video VAE decoder | 9.761 | 5.249 | 4.220 | 0.558 | 0.000 | 0.470 | WMMA f32 linear + d64 SDPA |
| **Sum / E2E** | **86.23 / 87.30** | **14.90** | | | | | |

### vs Run B

| Phase | B wall / GPU | v0.9.0 r1 wall / GPU | Δ wall |
|-------|-------------:|---------------------:|-------:|
| Text encoder | 20.0 / 1.9 | 29.2 / 2.1 | +9.2 s (I/O) |
| DiT load | 40.6 / 2.4 | 38.6 / 0.7 | −2.0 s |
| DiT denoise | 18.6 / 17.4 | **7.0 / 6.5** | **−11.6 s** |
| audio VAE | 4.4 / 3.9 | 1.7 / 0.4 | −2.7 s |
| video VAE | 32.9 / 25.1 | **9.8 / 5.2** | **−23.1 s** |
| **E2E** | **117.3** | **87.3** | **−30.0 s** |

---

## Run B — post-overnight (2026-08-19 09:11 CST)

| Meta | |
|------|--|
| Git | `dcb4858` (Speed up HIP DiT/VAE paths and weight load I/O) |
| Output | `/tmp/h3-profile/fox-s2-post-overnight.mp4` (136 KiB) |
| Log | `/tmp/h3-profile/fox-s2-post-overnight.log` |
| E2E wall | **117.28 s** (`time`: user 48.71 · sys 84.84) |
| Phase-wall sum | **116.49 s** (E2E − sum ≈ 0.8 s mux/overhead) |
| DiT layer policy | gate-ranked skips 15 inactive of 50 (active ≈ 35) |

### Phase table

| Phase | wall (s) | GPU op (s) | linear | sdpa | conv | other | Notes |
|-------|---------:|-----------:|-------:|-----:|-----:|------:|-------|
| Qwen text encoder | 20.007 | 1.917 | 1.895 | 0.001 | 0.000 | 0.021 | Wall ≈ I/O + setup |
| H3 DiT **load** | 40.561 | 2.438 | 1.042 | 0.000 | 0.000 | 1.396 | Weight pread + INT8 quantize |
| H3 DiT **Euler denoise** (2 steps) | 18.563 | 17.377 | 8.599 | 8.127 | 0.000 | 0.651 | ~8.7 s/step GPU |
| audio VAE decoder | 4.444 | 3.867 | 0.000 | 0.000 | 3.762 | 0.106 | ≈all conv |
| video VAE decoder | 32.912 | 25.113 | 11.335 | 13.298 | 0.000 | 0.480 | Wall includes load |
| **Sum / E2E** | **116.49 / 117.28** | | | | | | |

### DiT detail (from profile lines)

| Line | wall | encode | wait | submissions | peak |
|------|-----:|-------:|-----:|------------:|-----:|
| load | 40.561 | 21.053 | 1.181 | 64 | 14.74 GiB |
| Euler denoise | 18.563 | 18.549 | 0.007 | 2 | 14.74 GiB |
| DiT total | 59.402 | 39.602 | 1.188 | 66 | 14.74 GiB |

### Hotspots (this run)

1. **DiT load** 40.6 s wall — largest single phase (I/O + quantize).  
2. **Video VAE** 32.9 s wall / 25.1 s GPU — sdpa 13.3 · linear 11.3.  
3. **Text encoder** 20.0 s wall (GPU only 1.9 s).  
4. **Denoise** 18.6 s wall / 17.4 s GPU — linear 8.6 ≈ sdpa 8.1.  
5. **Audio VAE** 4.4 s — small.

### vs Baseline A (2026-08-18, same preset)

| Phase | A wall / GPU | B wall / GPU | Δ wall |
|-------|-------------:|-------------:|-------:|
| Text encoder | 23.5 / 2.1 | 20.0 / 1.9 | −3.5 s |
| DiT load | 114.0 / 1.1 | 40.6 / 2.4 | **−73.4 s** |
| DiT denoise | 41.5 / 40.3 | 18.6 / 17.4 | **−22.9 s** |
| audio VAE | 4.5 / 3.8 | 4.4 / 3.9 | ~0 |
| video VAE | 73.8 / 53.6 | 32.9 / 25.1 | **−40.9 s** |

Approx E2E then (phase sum) ≈ 257 s → now **117 s** (**≈ −54%**).

### Extrapolation (denoise only, same shape)

GPU denoise ≈ 17.4 s / 2 steps → **≈ 8.7 s/step**.  
20-step showcase denoise GPU ≈ **174 s**, plus fixed load+VAE+text ≈ **98 s** → rough E2E ≈ **4.5+ min** before further opts (order-of-magnitude only).

---

## Run C — day-2 tree (2026-08-19 21:13 CST)

| Meta | |
|------|--|
| Git | `dcb4858` + **uncommitted day-2** (parallel DiT/VAE load, pread8, vectorized quantize, packed stores, Qwen prefetch depth 4) |
| Output | `/tmp/h3-profile/fox-s2-C.mp4` (136 KiB) |
| Log | `docs/perf-runs/fox-s2-C-2026-08-19.log` |
| E2E wall | **121.81 s** (`time`: user 47.35 · sys 86.01) |
| Phase-wall sum | **121.00 s** |
| DiT layer policy | same gate-ranked skips as Run B |

### Phase table

| Phase | wall (s) | GPU op (s) | linear | sdpa | conv | other | Notes |
|-------|---------:|-----------:|-------:|-----:|-----:|------:|-------|
| Qwen text encoder | 23.797 | 1.808 | 1.787 | 0.001 | 0.000 | 0.021 | Prefetch depth 4; peak 4.54 GiB |
| H3 DiT **load** | 44.680 | 3.474 | 1.026 | 0.000 | 0.000 | 2.447 | I/O-noisy vs B 40.6 / day2-best 32.9 |
| H3 DiT **Euler denoise** (2 steps) | 18.428 | 17.236 | 8.552 | 8.051 | 0.000 | 0.633 | GPU stable vs B |
| audio VAE decoder | 4.302 | 3.777 | 0.000 | 0.000 | 3.670 | 0.107 | |
| video VAE decoder | 29.779 | 25.028 | 11.164 | 13.387 | 0.000 | 0.478 | Wall −3.1s vs B; GPU flat |
| **Sum / E2E** | **121.00 / 121.81** | | | | | | |

### DiT detail

| Line | wall | encode | wait | submissions | peak |
|------|-----:|-------:|-----:|------------:|-----:|
| load | 44.680 | 17.418 | 1.137 | 64 | 14.74 GiB |
| Euler denoise | 18.428 | 18.414 | 0.006 | 2 | 14.74 GiB |
| DiT total | 63.378 | 35.832 | 1.144 | 66 | 14.74 GiB |

### vs Run B

| Phase | B wall / GPU | C wall / GPU | Δ wall |
|-------|-------------:|-------------:|-------:|
| Text encoder | 20.0 / 1.9 | 23.8 / 1.8 | **+3.8 s** (I/O) |
| DiT load | 40.6 / 2.4 | 44.7 / 3.5 | **+4.1 s** (I/O) |
| DiT denoise | 18.6 / 17.4 | 18.4 / 17.2 | −0.2 s |
| audio VAE | 4.4 / 3.9 | 4.3 / 3.8 | ~0 |
| video VAE | 32.9 / 25.1 | 29.8 / 25.0 | **−3.1 s** wall |
| **E2E** | **117.3** | **121.8** | **+4.5 s** |

GPU compute (denoise + VAE) is unchanged. E2E is worse than B because text+DiT load I/O was slower this evening; day-2's best E2E remains **105.9 s** (12:31, load 32.9).

---

## Run A — first HIP profile (2026-08-18)

No git SHA recorded. No `/usr/bin/time`. Phase-wall sum ≈ **257 s**.

| Phase | wall | GPU (`op-classes`) | Notes |
|-------|-----:|-------------------:|-------|
| Qwen text encoder | 23.5s | 2.1s | Wall ≈ I/O |
| H3 DiT **load** | 114.0s | 1.1s | Almost no GPU |
| H3 DiT **Euler denoise** (2 steps) | 41.5s | 40.3s | linear 17.2 · sdpa 18.7 · other 4.4 |
| audio VAE decoder | 4.5s | 3.8s | ≈all conv |
| video VAE decoder | 73.8s | 53.6s | linear 28.2 · sdpa 24.9 |

vs B: DiT load −73.4 s, denoise −22.9 s, video VAE −40.9 s, phase sum ≈257→117 s.

---

## How to append the next run

1. Re-run the canonical command with `/usr/bin/time -f 'TIME_E2E wall_sec=%e ...'`.  
2. Copy the `h3 profile:` lines into a new `## Run C — …` section above (newest first or append below).  
3. Fill the phase table; note git SHA and date.  
4. Update the headline table in [`../PERFORMANCE.md`](../PERFORMANCE.md) only when tagging a release.

Kernel-level traces: `rocprofv3` notes in [`../KNOWN_ISSUES.md`](../KNOWN_ISSUES.md).
