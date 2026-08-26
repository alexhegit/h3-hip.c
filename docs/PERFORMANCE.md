# Performance (gfx1151)

Headline numbers for tagged releases. Reproduced on an AMD Ryzen AI MAX+ 395 /
Radeon 8060S with the official MiniMax-H3 checkpoint. Wall time on this
machine moves with page-cache state; treat E2E as a band, denoise GPU time as
the stable GPU figure.

Engineering logs (phase tables, rejected experiments, Metal A/B detail) live
under [`perf/`](perf/README.md) and are **not** part of the user-facing docs.

## Current release — v0.9.0 (2026-08-26)

| Preset | Command knobs | E2E | Denoise GPU |
|--------|---------------|----:|------------:|
| **fox-s2** | 512² · 22 frames · `--steps 2 --layers 35 --reuse 1` | **83–87 s** | **6.3–6.5 s** |
| **fox-fast** | 512² · 22 frames · `--steps 20 --layers 45 --reuse 2` | **95 s** | **28 s** (11 evals) |

fox-s2 output md5 `1731f95c4aa582597cf83d57f46b8f9e` on this tree.

```bash
# fox-s2 (short smoke used for HIP A/B)
./h3 --profile -d /path/to/MiniMax-H3 \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 --steps 2 --layers 35 --reuse 1 \
  -o outputs/fox-s2.mp4

# fox-fast (upstream tutorial “first fast video”, without --show)
./h3 --profile -d /path/to/MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 --frames 22 --steps 20 --layers 45 --reuse 2 \
  -o outputs/fox-fast.mp4
```

First invocation pays weight load from disk (~107 GiB on the T2VA path). Repeat
runs still miss the page cache: host RAM on this box is ~31 GiB.

## What changed (high level)

| When | fox-s2 E2E | fox-fast E2E | What landed |
|------|-----------:|-------------:|-------------|
| 2026-08-18 | ~257 s | — | First HIP profile; denoise and VAE were GPU-bound |
| 2026-08-19 | ~117 s | — | INT8 DiT, faster load I/O |
| 2026-08-22 | — | ~213 s | fox-fast measured; denoise still ~105 s |
| **v0.9.0** | **83–87 s** | **95 s** | WMMA attention/linear/conv; weights in the VRAM carveout; AdaLN and VAE load overlap |

The remaining E2E is mostly NVMe weight I/O. Denoise is a small slice of fox-s2
and about a third of fox-fast.

Upstream [antirez/h3.c](https://github.com/antirez/h3.c) publishes **denoise
wall** on an M5 Max, not T2VA end-to-end. On the same fox-fast knobs that
figure is 16.7 s; HIP denoise wall here is 35 s. That ratio mixes two GPUs and
two memory systems and is not a port-quality score.

## How a GitHub Release should quote this

Paste the **Current release** table and the two commands. Do not paste phase
splits, KEEP/REJECT lists, or Metal ratio tables into the release body.
