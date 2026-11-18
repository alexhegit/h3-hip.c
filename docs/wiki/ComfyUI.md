# ComfyUI (experimental)

Render MiniMax-H3 from ComfyUI on an AMD GPU through h3-hip. ComfyUI gets
one node, **MiniMax-H3 (h3-hip, AMD)**; the model runs in a separate
`h3 --serve` process on the same machine, and the node hands ComfyUI the
finished MP4 (video and audio).

```text
LoadImage (optional) ──► MiniMax-H3 (h3-hip, AMD) ──► SaveVideo
                               │  HTTP, 127.0.0.1:8571
                               ▼
                         h3 --serve  (GPU, weights, cache)
```

When to use it instead of ComfyUI's native MiniMax-H3 nodes: on a 32 GiB
card with ~64 GiB of RAM, native ComfyUI needs `--cache-none` and a restart
per job, while the daemon keeps its weights and renders job after job. On a
Radeon AI PRO R9700 h3-hip was also 1.34× faster end to end at 864×480 ·
20 steps ([comparison](../perf-runs/R9700_COMFYUI_COMPARE_2026-11-09.md)).
The pictures differ from native ComfyUI (sampler, INT8 scheme, RNG); this
page makes no quality claim.

Tested with ComfyUI 0.44.0 (`fca60fe`) on gfx1201 and gfx1100. The node uses ComfyUI's
V3 node API (`comfy_api.latest`), so older ComfyUI builds will not load it.

## 1. Weights (the big download)

h3-hip reads the **official** checkpoint, not the Comfy-Org repackaged
files ComfyUI uses. Even if ComfyUI already has MiniMax-H3, plan for this:

```bash
pip install -U "huggingface_hub[cli]"
./tools/download_weights.sh --dir ./MiniMax-H3             # FL2VA/, ~134 GiB
./tools/download_fasth3_lora.sh --dir ./FastH3-4-step-LoRA # optional 4-step LoRA, ~1.4 GiB
```

`FL2VA/` covers text-to-video and first/last-frame. Details:
[Getting started → Weights](Getting-started.md#weights).

## 2. Build h3 and start the daemon

Build for your GPU ([Getting started → Build](Getting-started.md#build)),
then start the daemon from the repository root and leave it running:

```bash
make HIP_ARCH=gfx1201 -j$(nproc) h3        # your ISA: gfx1151, gfx1201, gfx90a, gfx942
H3D_FASTH3_LORA=$PWD/FastH3-4-step-LoRA/dense-datafree/adapter_model.safetensors \
  ./h3 -d ./MiniMax-H3 --serve
```

It listens on `127.0.0.1:8571`. Check it:

```bash
curl -s -H 'X-H3-Protocol: v1alpha' http://127.0.0.1:8571/v1/info
# "ready": true, "adapters": {"fasth3": true}
```

`H3D_FASTH3_LORA` is only needed for the node's **fasth3** switch, and only
if the adapter is not at `./FastH3-4-step-LoRA/` relative to where you start
the daemon. Keep the daemon on loopback; it has no authentication.

## 3. Install the node

Link the node directory into ComfyUI and restart ComfyUI:

```bash
ln -s "$PWD/integrations/comfyui" /path/to/ComfyUI/custom_nodes/h3-hip
```

ComfyUI's console then lists `custom_nodes/h3-hip` under "Import times for
custom nodes". The node needs only what ComfyUI already ships (NumPy,
Pillow).

## 4. Use the workflow

ComfyUI opens on its own template. That canvas is not the h3-hip graph.
Load one of the two files in `integrations/comfyui/workflows/` (API format):
drop the file on the canvas, or use the workflow menu's *Load*.

| file | graph | what you get |
|---|---|---|
| `h3hip_t2va_api.json` | MiniMax-H3 → SaveVideo | text to video, 864×480 · 5 s, 20 dense steps |
| `h3hip_fl2va_api.json` | LoadImage → MiniMax-H3 → SaveVideo | that image as the first frame, FastH3, 4 steps |

### Text to video

1. Leave the daemon from [step 2](#2-build-h3-and-start-the-daemon) running,
   start ComfyUI, and open `http://127.0.0.1:8188`.
2. Load `h3hip_t2va_api.json`. Two nodes appear: **MiniMax-H3 (h3-hip, AMD)**
   and **SaveVideo** (filename prefix `h3hip/t2va`).
3. Edit `prompt`. A short one that has been run through this graph:

   > A red fox walks through fresh snow at dawn, its breath steaming in the
   > cold air. The camera tracks low beside it. Soft crunch of snow
   > underfoot, distant birdsong.

4. Leave the other widgets unless you want a different clip. `width` and
   `height` step by 32. `length` is frames at 24 fps: 124 is about 5 s,
   362 is about 15 s. `steps` 20 is the dense default.
5. Queue the prompt. The progress bar is one scale for the whole job
   (text encoder, DiT load, denoise, video VAE decode, MP4 write) and
   reaches 100% only when the file is written. ComfyUI's console logs each
   phase (`h3-hip <job>: denoise`, then `video VAE decode`, then `FFmpeg`).
6. **SaveVideo** writes
   `ComfyUI/output/h3hip/t2va_00001_.mp4` (the number goes up on each run).
   The daemon keeps its own copy under `~/.h3d/outputs`.

**Interrupt** stops the daemon job at its next progress report: the next
denoise step, or the next temporal chunk of the VAE decode. A cancel during
the DiT load still waits for that load to finish
([W7900 record](../perf-runs/W7900_COMFYUI_2026-11-18.md)).

Queueing the same prompt again is much faster: the daemon keeps the
generation cache. Turn on `release_vram` when the next node in the graph
needs the GPU; see [Sharing the GPU](#sharing-the-gpu-with-other-comfyui-nodes).

### First frame (FL2VA)

1. Put a picture in ComfyUI's `input/` directory and name it `example.png`,
   or load the graph and change **LoadImage** to the file you have.
   ComfyUI and the daemon must be on the same machine: the node copies the
   frame into `~/.h3d/media` and the daemon reads it from there.
2. Load `h3hip_fl2va_api.json`. **LoadImage** feeds `first_frame`. The prompt
   refers to that picture as `<Picture 1>`. `fasth3` is on, so the daemon
   runs 4 steps and the `steps` widget is ignored. The daemon needs the
   FastH3 adapter (`adapters.fasth3` from [step 2](#2-build-h3-and-start-the-daemon)).
3. Queue. **SaveVideo** writes `ComfyUI/output/h3hip/fl2va_00001_.mp4`.

To build the graph yourself instead of loading a file: add
**MiniMax-H3 (h3-hip, AMD)** from the node menu (category `video/h3-hip`),
connect its video output to **SaveVideo**, and, for a first frame, connect
**LoadImage** to `first_frame`.

### Headless

From the repository root, with ComfyUI already running:

```bash
./integrations/comfyui/run_workflow.py integrations/comfyui/workflows/h3hip_t2va_api.json \
  --set 1.fasth3=true --set 1.seed=7
```

`--set 1.<input>=<value>` overrides one widget on the MiniMax-H3 node
(`1` is that node's id in the T2VA file; in the FL2VA file the node id is
`2`).

## 5. Inputs

| input | default | notes |
|---|---|---|
| `prompt` | | MiniMax-H3 prompt; refer to the first frame as `<Picture 1>` |
| `width` × `height` | 864 × 480 | multiples of 32, at most 768×1344 pixels |
| `length` | 124 | frames at 24 fps (5 + 17·k; 124 is 5 s, 362 is 15 s) |
| `seed` | 42 | |
| `steps` | 20 | ignored when `fasth3` is on (always 4) |
| `fasth3` | off | 4-step FastH3 LoRA; needs `adapters.fasth3` on the daemon, and **`layers` = 50 and `reuse` = 1** |
| `layers` | 50 | advanced; 35–50. Below 50 is faster and lossy, but only with `fasth3` off. With `fasth3` on it must be 50 |
| `reuse` | 1 | advanced; 1–3. Above 1 is faster and lossy. With `fasth3` on it must be 1 |
| `free_comfy_vram` | on | advanced; unload ComfyUI's own models before the job |
| `release_vram` | off | advanced; daemon frees its cache after the job (see below) |
| `daemon_url` | `http://127.0.0.1:8571` | advanced; also `H3_HIP_URL` in ComfyUI's environment |
| `first_frame`, `last_frame` | | optional IMAGE inputs; either one switches to first/last-frame mode |

## Sharing the GPU with other ComfyUI nodes

After a job the daemon keeps a generation cache so a rerun of the same
prompt is fast. On the R9700, 864×480 · 5 s FastH3:

| | daemon VRAM | daemon RAM | job time |
|---|---:|---:|---:|
| first job | | | 104.5 s |
| cached, idle | 9.5 GiB | 19 GiB | |
| rerun, same prompt | | | 61.6 s |
| after release | 0.55 GiB | 0.89 GiB | |

If later nodes in your workflow need the GPU (upscalers, other models),
give the memory back:

- turn on the node's `release_vram` (freed before the node returns), or
- start the daemon with `H3D_IDLE_RELEASE_SEC=60` (freed 60 s after the
  last job), or
- `curl -X POST -H 'X-H3-Protocol: v1alpha' http://127.0.0.1:8571/v1/cache/release`.

The other direction is covered by `free_comfy_vram`: ComfyUI unloads its
own models from VRAM before handing the GPU to the daemon (into RAM; see
the native MiniMax-H3 pitfall below). Output does not change either way.

Checked in one ComfyUI session on the R9700: an h3 job with `release_vram`
left the daemon at 0.76 GiB VRAM with an empty cache, and a native ComfyUI
MiniMax-H3 job right after it ran normally.

## Pitfalls

- **Don't mix with native MiniMax-H3 nodes on a small-RAM box.** After a
  native ComfyUI MiniMax-H3 job, ComfyUI keeps about 35 GiB of those
  weights in RAM, even after *Unload models* (`/free`). The daemon needs
  about 20 GiB more for a job, so on a 59 GiB machine the next h3 job got
  ComfyUI OOM-killed in our test. Restart ComfyUI before switching to the
  h3 node; the node prints a warning when available RAM is below 24 GiB.
- **`fasth3` and `layers` don't mix.** The FastH3 LoRA is trained on all 50
  DiT layers, so with `fasth3` on the node rejects `layers` below 50 (and
  `reuse` above 1) before it sends the job. The error reads
  `fasth3 needs layers = 50`. Set `layers` back to 50, or turn `fasth3` off
  to use 35–49 layers. Without `fasth3`, any `layers` from 35 to 50 runs.
- **One daemon per GPU, one job at a time.** Jobs from several ComfyUI
  queues line up in the daemon. On a multi-GPU box pin the daemon with
  `HIP_VISIBLE_DEVICES=N`.
- **Cards below 48 GiB:** the daemon sets `H3_INT8_HOST=1` itself so its
  cache leaves room for the video VAE
  ([KI-007](../KNOWN_ISSUES.md)); 1344×768 needs it too. Same output.
- **Keyframes go through the daemon's media root** (`~/.h3d/media` unless
  `H3D_MEDIA_ROOT` is set). ComfyUI and the daemon must run on the same
  machine, as the same user or with that directory writable by ComfyUI.
- **Outputs** stay under `~/.h3d/outputs` (`H3D_OUTPUT_ROOT`, 20 GB quota);
  `SaveVideo` copies them into ComfyUI's output folder.
- **`SaveVideo` remuxes the MP4**, so its file md5 differs from the
  daemon's; the streams are identical
  (`ffmpeg -i X -map 0 -c copy -f streamhash -hash md5 -`).
- "h3 daemon not reachable": the daemon is not running, or on another
  port; set `daemon_url`.

## Checked

On gfx1201 through ComfyUI 0.44.0: fox-s2 (512×512, 22 frames, 2 steps,
35 layers) and FL2VA + FastH3 from a `LoadImage` first frame have the
same video and audio stream hashes as the CLI; Interrupt during an
864×480 dense denoise frees the daemon within 12 s. The daemon itself is
gated against the CLI by `make serve-regression`
([KI-005](../KNOWN_ISSUES.md)).

On gfx1100 (Radeon Pro W7900) through ComfyUI 0.44.0 with PyTorch
2.10.0+rocm7.0, the same fox-s2 and FL2VA + FastH3 checks match the CLI.
Interrupt during denoise stops the job within one step (~17 s), and a
1344×768 · 5 s job renders through the node in 745 s
([record](../perf-runs/W7900_COMFYUI_2026-11-18.md)). An Interrupt during
the DiT load waits for the load to finish, which is minutes on a spinning
disk.
