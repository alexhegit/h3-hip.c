# Fox-fast apples-to-apples vs antirez/h3.c

Upstream tutorial §2 “Make a first fast video”
<https://github.com/antirez/h3.c> (README, balanced preset).

## Command

Identical generation knobs. `--show` omitted: it loads a resident preview VAE
(~10 GiB) and extra decode after every Euler step; the README says generation
without `--show` is unchanged. Timed on a non-graphical Linux session.

```bash
./h3 --profile \
  -d /home/amd/HF-MODELS/MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 \
  --frames 22 --steps 20 \
  --layers 45 --reuse 2 \
  -o outputs/fox-fast.mp4
```

| Knob | Value |
|------|------:|
| frames | 22 (0.92 s @ 24 fps) |
| steps | 20 |
| layers | 45 (5 DiT blocks skipped) |
| reuse | 2 → **11** fresh DiT evaluations |
| this machine | Ryzen AI MAX+ 395 / Radeon 8060S `gfx1151`, Q3 working tree |

## This run (2026-08-22 ~16:25 CST)

Log: `/tmp/h3-profile/fox-fast.log` · output: `outputs/fox-fast.mp4`

| Phase | wall | GPU (`op-classes`) | split |
|-------|-----:|-------------------:|-------|
| Qwen text encoder | 20.05 | 2.65 | linear 2.62 |
| H3 DiT **load** | 60.43 | 5.71 | I/O + quantize |
| H3 DiT **Euler denoise** (11 evals) | **104.99** | **95.58** | linear 58.03 · sdpa 32.86 · other 4.68 |
| audio VAE | 4.26 | 3.75 | conv 3.64 |
| video VAE | 21.94 | 17.66 | linear 10.65 · sdpa 6.52 |
| **E2E** (`/usr/bin/time`) | **212.68** | | user 84.7 · sys 194.2 |
| DiT peak live | 19.7 GiB | | |

Per DiT forward: denoise GPU **95.58 / 11 ≈ 8.69 s**.

## Published M5 Max (same 512², 45 layers, reuse 2)

From antirez/h3.c README (IT M5 Max, no `--token-reduction`):

> token reduction cut the `45 layers + reuse 2` denoise profile from **16.69** to 12.60 seconds

So the matching Metal denoise profile is **16.69 s**. README does not publish
E2E wall for this exact fox-fast command. Nearby M5 numbers (different knobs):

| M5 Max figure | What it is | Not the same as fox-fast |
|---------------|------------|--------------------------|
| denoise **16.69 s** | 512², 45 layers, reuse 2 | **this comparison** |
| denoise 12.60 s | same + `--token-reduction` | extra quality knob |
| denoise 19.32 s | 50-layer, 19-transition, int8 QKV | more evals, full 50 blocks |
| denoise 3.5 s | `--steps 4 --reuse 1` | 4 forwards only |
| E2E 74.6 / 77.0 s | Ref2VA image+audio / video+audio | different pipeline |

## Head-to-head (denoise is the published match)

| | M5 Max (Metal) | gfx1151 HIP (this run) | HIP / M5 |
|--|---------------:|----------------------:|---------:|
| Denoise wall (45L, reuse 2, 512²) | **16.69 s** | **105.0 s** | **6.3×** |
| Denoise GPU (`op-classes`) | (not published) | 95.6 s | |
| Per DiT forward (÷11) | ~1.52 s | ~8.69 s GPU | **~5.7×** |
| Video VAE | (not published for this preset) | 17.7 s GPU / 21.9 s wall | |
| Process E2E | (not published) | **212.7 s** | |
| DiT peak tensors | ~25.9 GiB (50L int8 path) | 19.7 GiB (45L) | |

E2E on HIP is dominated by denoise (105 s) + DiT load (60 s) + text I/O (20 s)
+ VAE (26 s). Load is filesystem-noisy and should not be ratio’d against Metal.

M5 Max is a much larger GPU than Strix Halo iGPU; the 6× denoise gap is
hardware + remaining HIP kernel gap (INT8 linear still ~61% of denoise GPU).
