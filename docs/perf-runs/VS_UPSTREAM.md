# Apples-to-apples vs antirez/h3.c (v0.9.0)

Internal. User-facing scoreboard: [`../PERFORMANCE.md`](../PERFORMANCE.md).

Upstream ([antirez/h3.c](https://github.com/antirez/h3.c) README, retrieved
2026-08-26) does **not** ship a performance harness. `make test` / `make parity`
are host and Metal/MLX correctness checks. All published speed numbers are
`--profile` **denoise wall** on an IT M5 Max (128 GB), written into the README
next to the tutorial commands. E2E process wall is unpublished for the T2VA fox
presets; the only full-run figures are Ref2VA image+audio / video+audio at
74.58 / 76.99 s, which is a different pipeline.

This file matches the tutorial CLIs on gfx1151 at git `5d25f4a` (plus the two
runs below). The first HIP fox-fast (2026-08-22, denoise wall 105 s) is
[`FOX_FAST.md`](FOX_FAST.md), not this table.

`--show` is omitted: the README says generation without it is
unchanged, and the preview VAE would add ~10 GiB and extra decode.

HIP numbers: denoise **wall** is the fair match to the README; `gpu-op` is
stricter (HIP events). Load/text/VAE walls are I/O-noisy on this box and are
not ratio'd against Metal.

## What can be compared

| Upstream command | M5 Max (README) | What it is |
|---|---|---|
| `--steps 20 --layers 45 --reuse 2` | denoise **16.69 s** | tutorial “first fast video”, 11 DiT evals |
| same + `--token-reduction` | denoise **12.60 s** | same shape, pairs middle-block tokens |
| `--steps 4 --layers 50 --reuse 1` | denoise **~3.5 s** | four-pass fox (vs 26.4 s 29-pass reference) |
| `--steps 20 --layers 50 --reuse 1` | (no denoise wall) | README default; int8 path quotes a **19-transition** 19.18–19.32 s that is **not** this CLI |
| `--ssd-streaming` warm 50-block forward | resident **1.35 s** vs streaming **2.49 s** (84% slower) | per-forward, not a full run |
| Ref2VA image+audio / video+audio | E2E 74.58 / 76.99 s | needs reference media; not run here |

The 19-transition int8 ladder (BF16 36.30 → MLP int8 25.80 → QKV int8 19.32 →
attn-out 19.18) is an internal A/B, not a documented `./h3` invocation. Do not
map it onto `--steps 20 --layers 50 --reuse 1` (that is 20 evals).

## Head-to-head (denoise)

| Preset | Evals | M5 denoise wall | HIP denoise wall | HIP denoise GPU | HIP / M5 (wall) | HIP E2E |
|--------|------:|----------------:|-----------------:|----------------:|----------------:|--------:|
| fox-fast (`20/45/reuse 2`) | 11 | **16.69 s** | **34.63 s** | 28.37 s | **2.08×** | 94.81 s |
| fox-fast + `--token-reduction` | 11 | **12.60 s** | **25.84 s** | 22.09 s | **2.05×** | 109.57 s† |
| four-step (`4/50/reuse 1`) | 4 | **~3.5 s** | **14.38 s** | 12.50 s | **4.1×** | 94.43 s |

† Same-session I/O was slower (text 27.3 s, DiT load 42.6 s vs fox-fast 22.0 / 28.0). Token reduction still cut denoise wall 34.63 → 25.84 s (**−25%**), matching M5's 16.69 → 12.60 s (**−24.5%**).

Per DiT forward:

| Preset | M5 wall / eval | HIP GPU / eval | HIP wall / eval |
|--------|---------------:|---------------:|----------------:|
| fox-fast 45L | 1.52 s | 2.58 s | 3.15 s |
| fox-fast + TR | 1.15 s | 2.01 s | 2.35 s |
| four-step 50L | 0.88 s | 3.13 s | 3.59 s |

Four-step looks worse on the ratio because it is only four evals of the **full
50-block** transformer; fox-fast skips 5 blocks and amortizes load across 11
evals. The M5 3.5 s figure is also a single published point (“about 3.5
seconds”), not a thermally-balanced A/B.

## HIP phase split (this session)

### Fox-fast (already in [`V0.9.0.md`](V0.9.0.md))

E2E 94.81 s. Denoise 34.63 / GPU 28.37 (linear 22.50 · sdpa 3.85 · other 2.03).
DiT load 27.99 s, text 22.02 s, video VAE 8.37 s. md5 `aa2dd874…`.

### Fox-fast + token-reduction

Log: [`fox-fast-tr-v0.9.0.log`](fox-fast-tr-v0.9.0.log). md5 `31c90685…`.

| Phase | wall (s) | GPU op (s) | linear | sdpa | other |
|-------|---------:|-----------:|-------:|-----:|------:|
| text encoder | 27.281 | 2.620 | 1.886 | 0.003 | 0.730 |
| DiT load | 42.576 | 1.549 | 1.200 | 0.000 | 0.349 |
| Euler denoise | 25.837 | 22.087 | 17.514 | 2.560 | 2.012 |
| audio VAE | 1.629 | 0.437 | | conv 0.351 | 0.086 |
| video VAE | 10.989 | 5.208 | 4.178 | 0.558 | 0.472 |
| **E2E** | **109.57** | | | | |

### Four-step fox

Log: [`fox-four-step-v0.9.0.log`](fox-four-step-v0.9.0.log). md5 `84e33516…`.

| Phase | wall (s) | GPU op (s) | linear | sdpa | other |
|-------|---------:|-----------:|-------:|-----:|------:|
| text encoder | 22.404 | 4.311 | 3.615 | 0.002 | 0.693 |
| DiT load (50 blocks) | 47.063 | 0.828 | 0.492 | 0.000 | 0.335 |
| Euler denoise (4 steps) | 14.377 | 12.504 | 10.008 | 1.647 | 0.849 |
| audio VAE | 0.869 | 0.440 | | conv 0.351 | 0.089 |
| video VAE | 8.515 | 5.236 | 4.213 | 0.552 | 0.471 |
| **E2E** | **94.43** | | | | |

Four-step E2E is almost the same as fox-fast (~94 s) because **load + text
(~69 s) dominate**; denoise is only 14 s of the run. On M5 the 3.5 s denoise is
small next to whatever they pay for load, but they never published that load.

## How to read the gap

1. **Fair number is denoise wall on the fox-fast CLI.** HIP is **2.1×** the IT
   M5 Max (34.6 vs 16.7 s). Token reduction scales the same way on both (~25%).
2. **HIP E2E is not comparable to M5 denoise.** 95 s fox-fast is ~35 s denoise
   plus ~60 s of NVMe weight I/O. M5 unified memory holds the working set;
   this box's page cache cannot (31 GiB host RAM, 107 GiB read per T2VA run).
3. **Linear is still the HIP denoise leftover.** Fox-fast denoise GPU is 79%
   linear (22.5 / 28.4 s). SDPA is already small (3.85 s).
4. **Hardware is not matched.** M5 Max vs Strix Halo iGPU; ratios mix silicon
   and remaining kernel gap. Do not treat 2.1× as “HIP is 2× slower than Metal
   on the same GPU.”

## Upstream tests we are not using as benches

```sh
make test
make parity
```

Host unit tests plus optional Metal-vs-MLX toy-block fixtures under
`misc/fixtures/`. No timing, no fox CLI, no published golden times. HIP
`make test` is the analogue (one known `h3_hip_real_dit_smoke` harness miss).
