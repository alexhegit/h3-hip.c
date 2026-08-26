# h3-hip.c

HIP port of [antirez/h3.c](https://github.com/antirez/h3.c) for **AMD GPUs**
(`gfx1151` / Strix Halo). The original project is a native MiniMax-H3 inference
engine (Apple Metal / macOS); this repository reimplements the GPU backend in
pure HIP so the same CLI and model stack run on ROCm.

**Original project:** [antirez/h3.c](https://github.com/antirez/h3.c)
**CUDA sibling (DGX Spark):** [alexhegit/h3-spark.c](https://github.com/alexhegit/h3-spark.c)
**Official weights:** [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)

On gfx1151 (v0.9.0): fox-fast **~95 s** end-to-end (512², 22 frames, 20 steps);
short fox-s2 smoke **83–87 s**. Release scoreboard:
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

## Showcase (AMD gfx1151)

Clip below was generated on an AMD Strix Halo iGPU (`gfx1151`) with this HIP
port (512×512, 22 frames, `--steps 20 --layers 50 --reuse 1`). Click the poster
for the MP4.

| Mode | Sample |
|------|--------|
| **T2VA** — text → video+audio | [![T2VA fox](assets/showcase/t2va-fox.jpg)](assets/showcase/t2va-fox.mp4) [mp4](assets/showcase/t2va-fox.mp4) |

FL2VA and Ref2VA (image, silent video, embedded soundtrack, image+audio) are
wired and smoke-tested on the same binary; see [Status](#status-2026-08-17).

### Reproduce the showcase clip

```bash
MODEL=/path/to/MiniMax-H3

./h3 -d "$MODEL" \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 \
  --steps 20 --layers 50 --reuse 1 \
  -o assets/showcase/t2va-fox.mp4
```

## Status (2026-08-17)

| Capability | Status |
|------------|--------|
| T2VA (text → video+audio) | ✅ |
| FL2VA (`--first-frame` / `--last-frame`) | ✅ |
| Ref2VA (`--ref-image`, `--ref-silent-video`, `--ref-video`, `--ref-audio`) | ✅ |
| Runtime INT8 MLP | ✅ |
| `--frames-dir` / `--ssd-streaming` / `--token-reduction` | ✅ |
| gfx1151 fox-fast / fox-s2 timings | [docs/PERFORMANCE.md](docs/PERFORMANCE.md) |

Known gaps: [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)

## Requirements

- Linux + ROCm (`hipcc`, `libamdhip64`) and a `gfx1151` GPU (Strix Halo /
  Radeon 8060S). Other AMD targets are untested.
- Official BF16 checkpoint at `MiniMax-H3/` (`FL2VA/*`, optional `Ref2VA/*`)
- FFmpeg / FFprobe on `PATH`
- ICU (`libicu-dev`)

## Build

```bash
git clone https://github.com/alexhegit/h3-hip.c.git
cd h3-hip.c
make -j$(nproc) h3
./h3 --info -d /path/to/MiniMax-H3
```

On Linux the Makefile defaults to `H3_BACKEND=hip` and `--offload-arch=gfx1151`.
Apple Metal sources remain in-tree for reference against [antirez/h3.c](https://github.com/antirez/h3.c)
and are not the Linux build path.

## Quick generate (T2VA)

Same fox preset as the T2VA showcase clip:

```bash
./h3 --profile \
  -d /path/to/MiniMax-H3 \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 \
  --frames 22 --steps 20 \
  --layers 50 --reuse 1 \
  -o outputs/fox.mp4
```

First run pays model load + filesystem cache; repeat runs for timing.

## Conditional paths

Ordered references (`--ref-image`, `--ref-video`, `--ref-audio`, …) and
first/last-frame anchors (`--first-frame`, `--last-frame`) follow the same CLI
as [antirez/h3.c](https://github.com/antirez/h3.c).

`--ref-video` with an embedded soundtrack needs at least 2 seconds of audio
(request `--frames 56` or more). `--ref-video-audio VIDEO AUDIO` is the same
generation path with the soundtrack supplied as a separate file.

## Preview while generating

- **`--frames-dir DIR`** — write each decoded frame as PPM (works everywhere)
- **`--show`** — live denoise preview in Kitty/Ghostty/iTerm2/WezTerm/Konsole.
  Override detection: `H3_TERMINAL=kitty` (useful over SSH).
  Default terminal zoom: **1× on Linux**, 2× on macOS.

## Tests

```bash
make -j$(nproc) test
make hip-functional
```

Set `H3_MODEL=/path/to/MiniMax-H3` if weights are not at the Makefile default.
MLX toy-block fixtures are optional; without `misc/fixtures/` those parity
targets are skipped.

## Repository layout

Host/model/CLI code follows [antirez/h3.c](https://github.com/antirez/h3.c).
The Linux build uses the HIP backend (`backends/h3_gpu_hip.c`,
`kernels/h3_kernels.hip`, `kernels/h3_kernels_extra.hip`). Apple Metal sources
(`h3_gpu.m`, `h3_shaders.metal`) remain for reference against the original
project and are not the HIP build path.

## License

See [`LICENSE`](LICENSE). Model weights are subject to the MiniMax-H3 license on
Hugging Face.
