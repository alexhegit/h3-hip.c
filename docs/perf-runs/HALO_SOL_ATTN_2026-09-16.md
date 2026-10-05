# Strix Halo (gfx1151) — 15 s three-way (dense / TR / Sol-Attn)

2026-09-16. Same box as
[`HALO_SOL_ATTN_2026-09-15.md`](HALO_SOL_ATTN_2026-09-15.md): Ryzen AI MAX+
395 / Radeon 8060S, `gfx1151`, `H3_MODEL=/home/amd/HF-MODELS/MiniMax-H3`,
seed **42**, 864×480, 362 frames, `--steps 20 --layers 45 --reuse 2`.
Tree: `sol-attn` @ `b5a3d22` plus this comparison. Do **not** stack
`--sol-attn` with `--token-reduction` here.

Side-by-side reel (left dense, centre TR, right Sol-Attn; audio from dense):
[`assets/showcase/fox-15s-3way-compare-gfx1151.mp4`](../../assets/showcase/fox-15s-3way-compare-gfx1151.mp4).

## Runs

| Path | Command | Output |
|---|---|---|
| dense (quality) | 15 s cinematic knobs, **no** `--token-reduction`, **no** `--sol-attn` | `outputs/fox-15s-sol-attn-dense-gfx1151.mp4` |
| TR 4:30 | `./bench/fox-15s.sh` (`--token-reduction`) | `outputs/fox-15s-tr-gfx1151.mp4` |
| Sol-Attn τ=0.5 | same as dense plus `--sol-attn` | `outputs/fox-15s-sol-attn-tau05-gfx1151.mp4` |

Logs:

- dense: [`gfx1151-2026-09-15-fox-15s-sol-dense.log`](gfx1151-2026-09-15-fox-15s-sol-dense.log)
- Sol-Attn: [`gfx1151-2026-09-15-fox-15s-sol-tau05.log`](gfx1151-2026-09-15-fox-15s-sol-tau05.log)
- TR: [`gfx1151-2026-09-16-fox-15s-tr.log`](gfx1151-2026-09-16-fox-15s-tr.log)
- Sol vs dense ffmpeg: [`gfx1151-2026-09-15-fox-15s-sol-quality.log`](gfx1151-2026-09-15-fox-15s-sol-quality.log)

`fox-15s.sh` is **not** bit-identical to no-TR. The script comment that
called TR 4:30 “lossless / PSNR=inf” was wrong on this Halo A/B.

## Speed

| path | E2E | denoise | vs dense E2E |
|---|---:|---:|---|
| dense (no TR, no Sol) | **2457.04 s** (40 min 57 s) | 2170.15 s | reference |
| `--token-reduction` | **1666.33 s** (27 min 46 s) | 1371.25 s | **−32.2%** |
| `--sol-attn` τ=0.5, no TR | **1786.72 s** (29 min 47 s) | 1506.12 s | **−27.3%** |

TR is the faster of the two lossy knobs (~2 min E2E over Sol-Attn). Dense
without TR is the slowest. v0.12.0 Halo `fox-15s.sh` was **26 min 56 s**
(2026-09-12); this TR rerun is in the same band (VAE tile path on this
tree is 4×2 @ 272 px, so E2E is not a byte-for-byte retake of the tagged
scoreboard).

## Quality versus dense MP4

FFmpeg `psnr` / `ssim` on the muxed H.264 streams. Audio SNR is
float32 PCM vs the dense soundtrack.

| vs dense | video PSNR / SSIM | decoded audio SNR |
|---|---|---:|
| `--token-reduction` | **18.23 dB / 0.667** | **3.11 dB** |
| `--sol-attn` τ=0.5 | **19.55 dB / 0.721** | **10.99 dB** |
| Sol-Attn vs TR (not vs dense) | 18.60 dB / 0.700 | — |

Both lossy paths sit in the **preview** band versus dense. Sol-Attn is
closer on **video and especially audio**. TR’s 3.11 dB soundtrack is a
large waveform error. Sol-Attn vs TR is the same order as each vs dense:
the three MP4s are related scenes, not the same clip with extra blur.

## What to look at in the triptych

Blocking, props, and screen contents diverge. Typical:

- **t=2 s** — desk layout and lamp/plant placement differ on all three.
- **t=5 s** — dense keeps a tiled “matrix” on the centre monitor; TR
  warps that grid; Sol-Attn turns it into large square tiles and adds a
  plant on the right.
- **t=8 s** — fox-in-monitor shot: TR is softer; Sol-Attn keeps a
  sharper fox than TR but still not the dense pose.
- **t=11 s** — TR collapses to a single curved display; dense and
  Sol-Attn stay dual-monitor, with different lamp geometry.

Stills:
[`t2`](../../assets/showcase/fox-15s-3way-t2s.jpg) ·
[`t5`](../../assets/showcase/fox-15s-3way-t5s.jpg) ·
[`t8`](../../assets/showcase/fox-15s-3way-t8s.jpg) ·
[`t11`](../../assets/showcase/fox-15s-3way-t11s.jpg) ·
[`t14`](../../assets/showcase/fox-15s-3way-t14s.jpg).

## 2026-10-05 — same knobs plus `--fbc`

Same prompt, 864×480, 362 frames, `--steps 20 --layers 45 --reuse 2`,
seed 42, no `--token-reduction`, no `--sol-attn`. `--fbc` defaults
(accumulated relative L2 0.15, head 2, tail 2). Tree `main` @ `fadfcda`,
VRAM carveout 32 GiB. The September dense / TR / Sol-Attn rows above are
an older tree. The September dense MP4 is not on this disk, so this row
has no PSNR.

| path | E2E | denoise | sdpa / linear | attention calls |
|---|---:|---:|---|---:|
| dense (2026-09-15) | **2457.04 s** | 2170.15 s | 1583.504 / 553.875 s | 495 |
| `--fbc` (2026-10-05) | **2206.19 s** (−10.2%) | 1916.435 s | 1331.428 / 561.583 s | 275 |
| `--token-reduction` (2026-09-16) | **1666.33 s** | 1371.25 s | — | — |
| `--sol-attn` τ=0.5 (2026-09-15) | **1786.72 s** | 1506.12 s | — | — |

`--reuse 2` only evaluates 11 of 20 steps (1, 3, …, 19, 20). Those steps
are two scheduler steps apart, so block-0 relative L2 sat around
0.05–0.09, not the ~0.02 of consecutive 50-step evaluations. **6 full /
5 skip** (full steps 1, 5, 11, 17, 19, 20). Attention calls 495 → 275
(`6×45 + 5×1`). Denoise wall only fell 12%: SDPA 1584 → 1331 s, linear
stayed ~560 s. Video VAE 209.9 → 247.3 s. Peak **29.699 GiB**.

On this fixture `--fbc` is the smaller speedup. TR and Sol-Attn τ=0.5
from September remain faster. Output
`outputs/fox-15s-fbc-gfx1151.mp4` (864×480, 362 frames, 15.083 s), md5
`cb127873d7fc818ecd9d33bd559ac9e1`. Log:
[`gfx1151-2026-10-05-fox-15s-fbc.log`](gfx1151-2026-10-05-fox-15s-fbc.log).
