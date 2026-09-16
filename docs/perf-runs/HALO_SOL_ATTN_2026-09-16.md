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
