# h3-hip.c

HIP port of [antirez/h3.c](https://github.com/antirez/h3.c) for **AMD GPUs**
(`gfx1151` / Strix Halo). The original project is a native MiniMax-H3 inference
engine (Apple Metal / macOS); this repository reimplements the GPU backend in
pure HIP so the same CLI and model stack run on ROCm.

[![h3-hip.c ident](assets/showcase/h3-hip-ident.jpg)](assets/showcase/h3-hip-ident.mp4)

Project ident, generated on gfx1151 (864×480, 56 frames, `--steps 20 --layers 50
--reuse 1`). Click the poster for the MP4:
[h3-hip-ident.mp4](assets/showcase/h3-hip-ident.mp4).

**Project page:** [alexhegit.github.io/h3-hip.c](https://alexhegit.github.io/h3-hip.c/)
**Wiki:** [github.com/alexhegit/h3-hip.c/wiki](https://github.com/alexhegit/h3-hip.c/wiki)
**Original project:** [antirez/h3.c](https://github.com/antirez/h3.c)
**CUDA sibling (DGX Spark):** [alexhegit/h3-spark.c](https://github.com/alexhegit/h3-spark.c)
**Official weights:** [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)

On gfx1151 (v0.9.0): fox-fast **~95 s** end-to-end (512², 22 frames, 20 steps);
short fox-s2 smoke **83–87 s**. Release scoreboard:
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

## Showcase (AMD gfx1151)

Clips below were generated on an AMD Strix Halo iGPU (`gfx1151`) with this HIP
port. Click a poster for the MP4. The last three are **untitled** model output
(no ffmpeg captions).

| Mode | Sample |
|------|--------|
| **T2VA** — fox tutorial clip | [![T2VA fox](assets/showcase/t2va-fox.jpg)](assets/showcase/t2va-fox.mp4) [mp4](assets/showcase/t2va-fox.mp4) |
| **T2VA** — Linux 35 tribute (untitled) | [![Linux 35](assets/showcase/linux35.jpg)](assets/showcase/linux35.mp4) [mp4](assets/showcase/linux35.mp4) |
| **T2VA** — project ident draft (untitled) | [![ident draft](assets/showcase/h3-hip-ident-draft-raw.jpg)](assets/showcase/h3-hip-ident-draft-raw.mp4) [mp4](assets/showcase/h3-hip-ident-draft-raw.mp4) |
| **Ref2VA** — AMD developer community (untitled) | [![AMD community](assets/showcase/amd-developer-community-raw.jpg)](assets/showcase/amd-developer-community-raw.mp4) [mp4](assets/showcase/amd-developer-community-raw.mp4) |

FL2VA and Ref2VA (image, silent video, embedded soundtrack, image+audio) are
wired and smoke-tested on the same binary; see [Status](#status).

The titled ident at the top of this README is a later overlay of a 864×480
run (`--layers 50 --reuse 1`). The ident row in the table is the untitled
512×288 draft.

### Reproduce the showcase clips

```bash
MODEL=/path/to/MiniMax-H3

# Fox (512², 22 frames, full layers)
./h3 -d "$MODEL" \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 \
  --steps 20 --layers 50 --reuse 1 \
  -o assets/showcase/t2va-fox.mp4

# Linux 35 tribute (untitled; 864×480, 56 frames)
./h3 -d "$MODEL" \
  -p "A single penguin stands on a snowy ridge at blue hour, facing a glowing amber terminal screen floating in the cold air, scrolling lines of green monospaced code reflected in its eyes. Slow dolly-in, shallow depth of field, volumetric mist, cinematic teal and amber grade. Ambient wind, soft mechanical keyboard clicks, a low warm synth pad." \
  --width 864 --height 480 --seconds 2.33 \
  --steps 20 --layers 50 --reuse 1 \
  --seed 1991 \
  -o assets/showcase/linux35.mp4

# Project ident draft (untitled; 512×288, 56 frames)
./h3 -d "$MODEL" \
  -p "Cinematic technology project ident. In a dark graphite silicon landscape, streams of glowing model weights flow from an SSD into a powerful red GPU compute array. Magenta and amber wavefronts race through precise matrix tiles, code particles transform into a grid of moving video frames, and a realistic red fox briefly emerges from a snowy digital scene while a luminous audio waveform pulses beneath it. Slow controlled camera pullback reveals the complete accelerator board, elegant engineering visualization, AMD-red and HIP-magenta color palette, volumetric light, crisp reflections, premium cinematic motion graphics, no readable text, no logos. Sound: deep electronic startup pulse, rapid soft data clicks, rising synthesized tone, clean final impact." \
  --width 512 --height 288 --frames 56 \
  --steps 20 --layers 45 --reuse 2 \
  --seed 1991 \
  -o assets/showcase/h3-hip-ident-draft-raw.mp4

# AMD developer community (untitled Ref2VA; 864×480, 56 frames)
# Needs Ref2VA weights. Image encode on this HIP path is slow (~5 min per still).
./h3 -d "$MODEL" \
  -p "Cinematic AMD developer community invitation. Preserve the two referenced black developer T-shirt designs: first, the gold Helios chariot artwork on the back; second, the futuristic runner and open-world artwork on the front. Begin with a close view of the golden Helios shirt worn by a developer in a modern hardware lab. The camera makes a smooth orbit to reveal another developer wearing the runner shirt. Together they walk through a luminous orange open doorway into a welcoming diverse group of software engineers. Every engineer in the group, including the people further away in the background, wears the same matching black AMD team T-shirt with the crisp geometric AMD arrow emblem clearly printed on the chest and on the back, sharp white and orange print, consistent uniform team branding across the whole room. They gather around glowing code displays and an AMD-powered workstation. Confident, inclusive, collaborative energy, black graphite with AMD orange and electric blue accents, premium technology campaign, realistic fabric, crisp apparel print detail, cinematic lighting, no generated captions. Sound: subtle server room ambience, keyboard clicks, warm rising electronic pulse, uplifting final impact." \
  --ref-image assets/showcase/refs/helios-shirt.png \
  --ref-image assets/showcase/refs/run-open-shirt.png \
  --ref-image-size max \
  --width 864 --height 480 --seconds 2.33 \
  --steps 20 --layers 50 --reuse 1 \
  --seed 2026 \
  -o assets/showcase/amd-developer-community-raw.mp4
```

## Status

Current tagged line is **v0.9.0**.

| Capability | Status |
|------------|--------|
| T2VA (text → video+audio) | ✅ |
| FL2VA (`--first-frame` / `--last-frame`) | ✅ |
| Ref2VA (`--ref-image`, `--ref-silent-video`, `--ref-video`, `--ref-audio`) | ✅ |
| Runtime INT8 DiT (hipBLAS) | ✅ |
| `--frames-dir` / `--ssd-streaming` / `--token-reduction` | ✅ |

## Documentation

- **Project page** (speedup ladder and clips): [alexhegit.github.io/h3-hip.c](https://alexhegit.github.io/h3-hip.c/)
- **Wiki:** [Getting started](https://github.com/alexhegit/h3-hip.c/wiki/Getting-started) and related pages
- **Tagged timings** (fox-s2 / fox-fast on gfx1151): [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)
- **Known gaps:** [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md)

The fox showcase uses `--steps 20 --layers 50 --reuse 1`. The published
fox-fast numbers use `--layers 45 --reuse 2`; commands are in PERFORMANCE.md.

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

First run pays model load from disk (~107 GiB on this T2VA path). Repeat runs
still miss most of the page cache on this box (~31 GiB host RAM). For the
commands used in the tagged scoreboard, see [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

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
project and are not the HIP build path. User-facing extra docs are under
`docs/PERFORMANCE.md` and `docs/KNOWN_ISSUES.md`.

## License

See [`LICENSE`](LICENSE). Model weights are subject to the MiniMax-H3 license on
Hugging Face.
