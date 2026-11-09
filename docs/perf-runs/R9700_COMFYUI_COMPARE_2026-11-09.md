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

| workload | ComfyUI denoise | ComfyUI E2E | h3-hip denoise | h3-hip E2E |
|---|---:|---:|---:|---:|
| 864×480, 20 steps, 50 layers | 216–236 s (10.8–11.8 s/step) | 274–295 s | **172.4 s** (8.6 s/step) | **229.5 s** |
| 864×480, h3-hip fox-fast (45 layers, reuse 2) | | | 87.6 s | 142.2 s |
| 864×480, 4-step turbo | 62.8 s (15.7 s/step) | 129.5 s | **38.5 s** | **96.0 s** |
| 1344×768, 4-step turbo | 222.8 s (55.7 s/step) | 303.2 s | **148.7 s** (`H3_INT8_HOST=1`) | **258.1 s** |

At the same step count h3-hip denoises 1.25–1.6× faster and finishes
1.2–1.35× faster end to end. The samplers (Euler vs `res_multistep`) and
the turbo LoRAs differ, so outputs are different pictures; quality was not
compared.

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

The ComfyUI node prototype (`H3HipGenerate` → `SaveVideo`) produced the
same streams as the CLI for fox-s2; the daemon bugs it exposed are fixed
on `h3d-fixes`.
