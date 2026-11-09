# R9700: ComfyUI native MiniMax-H3 vs h3-hip, 2026-11-09

Question: is a ComfyUI node that calls `h3 --serve` worth building, given
that ComfyUI now runs MiniMax-H3 natively on ROCm?

Box: Radeon AI PRO R9700 (gfx1201, 32 GiB, PCIe 4.0 x4), 59 GiB RAM.
Prompt `prompt_starship.txt`, seed 42, 5 s (124 frames), one job per
process start unless noted.

- **ComfyUI** `fca60fe` (2026-11-08), PyTorch 2.11.0+rocm7.13.0 from
  `repo.amd.com` (the TheRock torch 2.14 wheel ships a mismatched
  torchvision). comfy_kitchen reports its HIP backend. Graph = the official
  `video_minimax_h3_t2v` template: `res_multistep`, `simple`, BasicGuider.
  Weights from `Comfy-Org/minimax-h3`: `minimax_h3_fl2va_int8_convrot`
  (full, not pruned), `qwen3vl_32b_minimax_h3_int8_convrot`, fp16 video VAE.
  Turbo = `minimax_h3_fl2v_turbo_4step_v1.0_768p` LoRA.
- **h3-hip** branch `h3d-fixes` (r9700 + daemon fixes, without the VAE
  BF16-activation PR), CLI with `--profile`. Turbo = FastH3 dense 4-step.

## Settings per row

Every row is one process: text encoder, DiT, audio and video VAE, MP4.
None of the h3-hip rows use token reduction, Sol-Attn, FBC, VSA, TAEH3 or
`H3_GPU_SAMPLER`. Both sides run INT8 DiT weights.

| row | ComfyUI | h3-hip | dense? |
|---|---|---|---|
| A. 864×480, 20 steps | template graph, no LoRA, 20 steps | `--steps 20 --layers 50 --reuse 1` | both dense |
| B. 864×480, h3-hip fast | (not run) | `--steps 20 --layers 45 --reuse 2` (validated fast path: 5 gate-ranked blocks dropped, every other step evaluated) | h3-hip lossy |
| C. 864×480, 4-step turbo | `LoraLoaderModelOnly` turbo 4-step 768p, strength 1.0, 4 steps | `--fasth3-lora dense-datafree --steps 4 --layers 50 --reuse 1` | both dense |
| D. 1344×768, 4-step turbo | as C | as C, plus `H3_INT8_HOST=1` (same output as resident INT8) | both dense |

ComfyUI's turbo row uses the 4-step LoRA at 4 steps to match FastH3; the
template's own turbo switch is the 8-step LoRA at 6 steps. The two turbo
LoRAs are different adapters.

## Times

| row | ComfyUI s/step | ComfyUI denoise | ComfyUI E2E | h3-hip s/step | h3-hip denoise | h3-hip E2E | denoise speedup | E2E speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (run 1 / run 2) | 11.82 / 10.82 | 236.4 / 216.4 s | 294.5 / 273.7 s | 8.62 | **172.4 s** | **229.5 s** | 1.26–1.37× | 1.19–1.28× |
| B | | | | 4.38 ¹ | 87.6 s | 142.2 s | (lossy, not comparable) | |
| C | 15.69 | 62.8 s | 129.5 s | 9.61 | **38.5 s** | **96.0 s** | 1.63× | 1.35× |
| D | 55.71 | 222.8 s | 303.2 s | 37.18 | **148.7 s** | **258.1 s** | 1.50× | 1.17× |

¹ Average over 20 steps; with reuse 2 only 10 are full DiT evaluations.

ComfyUI s/step is its tqdm average; E2E is `Prompt executed in` (the
process was already up). h3-hip times are `--profile` phases and
`/usr/bin/time` for the whole process. The samplers (Euler vs
`res_multistep`) and the turbo LoRAs differ, so outputs are different
pictures; quality was not compared.

### Where the rest of the time goes

| row | ComfyUI outside denoise ² | h3-hip text encoder | h3-hip DiT load | h3-hip video VAE | h3-hip other ³ |
|---|---:|---:|---:|---:|---:|
| A | 58.1 / 57.3 s | 12.1 s | 19.8 s | 21.6 s | 3.6 s |
| B | | 11.8 s | 18.1 s | 21.4 s | 3.3 s |
| C | 66.7 s | 11.8 s | 20.3 s | 21.6 s | 3.8 s |
| D | 80.4 s | 13.0 s | 34.2 s | 54.6 s | 7.5 s |

² Text encoder load and encode, DiT load, both VAEs, mux; ComfyUI does not
log them separately. ³ Audio VAE (0.9 s), FFmpeg, process start.

At 1344×768 h3-hip spends more than ComfyUI outside the denoise loop:
`H3_INT8_HOST` parks the INT8 linears in pinned RAM at load (34 s), and
the video VAE takes 54.6 s. The VAE BF16-activation PR cuts the 864×480
VAE from 21.7 to 16.9 s and should help here too; not measured at 1344.

## Memory

| row | ComfyUI DiT on GPU / offloaded | ComfyUI LoRA patches | h3-hip DiT peak VRAM | h3-hip VAE peak |
|---|---|---:|---:|---:|
| A run 1 | 10.1 GB / 22.4 GB | 0 | 21.4 GiB | 5.1 GiB |
| A run 2 | 25.1 GB / 7.3 GB | 0 | | |
| B | | | 19.7 GiB | 5.1 GiB |
| C | 10.0 GB / 22.4 GB | 206 | 21.2 GiB | 5.1 GiB |
| D | 19.8 GB / 12.6 GB | 160 | 21.0 GiB | 5.3 GiB |

ComfyUI loads the 25.9 GB text encoder fully onto the GPU first, then the
DiT partially. Because the DiT is not fully resident, the turbo LoRA stays
as per-step patches (`lowvram patches`), which is why row C is slower per
step than row A. h3-hip merges FastH3 into the weights at load.

Memory decides more than speed on this box:

- With default caching ComfyUI was OOM-killed (47 GB RSS) while loading the
  DiT next to the cached text encoder. `--cache-none` fits, peaking at
  52 GB used RAM.
- A second job in the same ComfyUI process was OOM-killed again (the
  offloaded DiT stays in RAM). Every ComfyUI number above is a fresh
  process.
- ComfyUI kept 10–25 GB of the 32 GB DiT on the GPU and streamed the rest;
  the run with 7 GB offloaded was only 8% faster per step than the one with
  22 GB offloaded, so its per-step time is mostly compute.
- The h3-hip daemon keeps weights resident: two consecutive 864×480 FastH3
  jobs took 120.6 s and then **56.6 s** (same prompt, so the text encoder
  result was reused too).

## What this supports, and what it does not

Matched on both sides: MiniMax-H3 FL2VA, all 50 DiT layers, the same step
count, one DiT forward per step (no CFG on either side), INT8 QKV /
attention-out / FC1 / FC2, same canvas, frames, prompt and seed value.

Not matched:

- INT8 schemes differ (Comfy-Org's offline `int8_convrot` file vs h3-hip
  quantizing the official BF16 weights at load). That is part of what is
  being compared, but it also means the pictures differ.
- Text encoder (INT8 Qwen3-VL vs original weights) and video VAE (fp16 vs
  F32 with BF16 block linears) differ, so the non-denoise phases are not a
  like-for-like comparison. The denoise column is the clean one.
- Sampler (`res_multistep` vs Euler) and RNG differ. Both cost one DiT call
  per step, so speed is unaffected; outputs are not comparable.
- Turbo rows use different adapters (MiniMax turbo LoRA vs FastH3), and
  ComfyUI applies its LoRA as per-step patches because the DiT does not fit
  on the card. Rows C and D therefore mix kernel speed with memory effects.
- ROCm runtime differs: ComfyUI's torch wheel bundles ROCm 7.13; h3-hip
  runs on the TheRock 10.1 runtime (the only torch build for that runtime
  ships a mismatched torchvision).
- ComfyUI E2E starts at prompt execution in an already-running server;
  h3-hip E2E includes process start. This favours ComfyUI by a few seconds.
- Sample size: ComfyUI row A has two runs (8% apart), every other cell one
  run. h3-hip runs on a 300 W-capped card that sits at 94–95 °C.
- Quality was not compared.

Claim that holds on this data: on an R9700 (32 GiB, 59 GiB RAM), at the
same model, step count and layer count, with INT8 main linears on both
sides, h3-hip's DiT denoise is 1.26–1.37× faster than ComfyUI native
(`fca60fe`, PyTorch 2.11+rocm7.13) and the whole job is 1.19–1.28×
faster. Row B (fox-fast) is lossy and must not be quoted against ComfyUI.
Turbo-row ratios should be quoted with the adapter and offload caveats.

The ComfyUI node prototype (`H3HipGenerate` → `SaveVideo`) produced the
same streams as the CLI for fox-s2; the daemon bugs it exposed are fixed
on `h3d-fixes`.
