# R9700: ComfyUI native MiniMax-H3 vs h3-hip, 2026-11-09

Question: is a ComfyUI node that calls `h3 --serve` worth building, given
that ComfyUI now runs MiniMax-H3 natively on ROCm?

Box: Radeon AI PRO R9700 (gfx1201, 32 GiB, PCIe 4.0 x4, 300 W cap),
59 GiB RAM. Prompt `prompt_starship.txt`, seed 42, 5 s (124 frames).

- **ComfyUI** `fca60fe` (2026-11-08), PyTorch 2.11.0+rocm7.13.0 from
  `repo.amd.com` (the TheRock torch 2.14 wheel ships a mismatched
  torchvision). comfy_kitchen reports its HIP backend. Graph = the official
  `video_minimax_h3_t2v` template: `res_multistep`, `simple`, BasicGuider.
  Weights from `Comfy-Org/minimax-h3`: `minimax_h3_fl2va_int8_convrot`
  (full, not pruned), `qwen3vl_32b_minimax_h3_int8_convrot`, fp16 video VAE.
  Started with `--cache-none` (see Memory).
- **h3-hip** branch `h3d-fixes` (r9700 + daemon fixes, without the VAE
  BF16-activation PR), CLI with `--profile`.

## Method

Three rounds per row, ComfyUI and h3-hip alternating (ComfyUI first in
rounds 1 and 3, h3-hip first in round 2), 30 s idle between runs. Every
run is a fresh process on both sides, so both read their weights through
a page cache the other side just evicted. E2E is process launch to MP4 on
both sides (`/usr/bin/time` for h3-hip; ComfyUI server launch to the
history entry, about 5 s more than its `Prompt executed in`). ComfyUI
s/step is its tqdm average; denoise = steps × s/step. GPU clock, power and
junction temperature are 1 s samples of `hwmon` while `gpu_busy_percent`
≥ 80. Driver: `bench_comfy.py` + `rerun.py` in the prototype directory.

## Settings per row

None of the h3-hip rows use token reduction, Sol-Attn, FBC, VSA, TAEH3 or
`H3_GPU_SAMPLER`. Both sides run INT8 DiT weights.

| row | ComfyUI | h3-hip | dense? |
|---|---|---|---|
| A. 864×480, 20 steps | template graph, no LoRA, 20 steps | `--steps 20 --layers 50 --reuse 1` | both dense |
| C. 864×480, 4-step turbo | `LoraLoaderModelOnly` turbo 4-step 768p, strength 1.0, 4 steps | `--fasth3-lora dense-datafree --steps 4 --layers 50 --reuse 1` | both dense |
| D. 1344×768, 4-step turbo | as C | as C, plus `H3_INT8_HOST=1` (same output as resident INT8) | both dense |

ComfyUI's turbo row uses the 4-step LoRA at 4 steps to match FastH3; the
template's own turbo switch is the 8-step LoRA at 6 steps. The two turbo
LoRAs are different adapters.

## Results (median of 3, range in brackets)

| row | ComfyUI s/step | ComfyUI denoise | ComfyUI E2E | h3-hip s/step | h3-hip denoise | h3-hip E2E | denoise ratio | E2E ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 11.80 [10.75–11.85] | 236.0 s [215.0–237.0] | 298.7 s [279.5–302.3] | 9.27 [9.09–9.34] | **185.3 s** [181.8–186.7] | **258.4 s** [254.7–259.4] | **1.27×** [1.15–1.30] | **1.16×** [1.08–1.19] |
| C | 12.64 [12.61–15.65] | 50.6 s [50.4–62.6] | 115.1 s [111.6–133.0] | 11.46 [10.28–11.71] | **45.8 s** [41.1–46.8] | 118.7 s [115.8–119.6] | **1.10×** [1.08–1.52] | **0.97×** [0.93–1.15] |
| D | 56.11 [55.70–59.50] | 224.4 s [222.8–238.0] | 304.1 s [303.2–312.5] | 39.43 [38.21–39.97] | **157.7 s** [152.8–159.9] | **261.4 s** [256.6–267.5] | **1.42×** [1.39–1.56] | **1.16×** [1.13–1.22] |

Ratios are ComfyUI time / h3-hip time; the bracket is the worst and best
cross-pairing of the three runs. The samplers (Euler vs `res_multistep`)
and turbo LoRAs differ, so outputs are different pictures; quality was not
compared.

### Where the rest of the h3-hip time goes (median)

| row | text encoder | DiT load | video VAE | audio VAE, mux, start |
|---|---:|---:|---:|---:|
| A | 11.9 s | 32.3 s | 21.6 s | 7.3 s |
| C | 11.7 s | 33.9 s | 21.4 s | 5.9 s |
| D | 11.2 s | 31.7 s | 53.7 s | 7.1 s |

DiT load is 30–39 s here against 18–20 s when h3-hip runs back to back:
ComfyUI's 62 GB of weights evict h3-hip's from the page cache. In row C
h3-hip's fixed costs (load, VAE) outweigh its 5 s denoise lead. The VAE
BF16-activation PR cuts the 864×480 VAE from 21.7 to 16.9 s; not included.

### GPU telemetry (mean over busy samples, per run)

| row | ComfyUI sclk / power / Tj max | h3-hip sclk / power / Tj max |
|---|---|---|
| A | 2.55–2.64 GHz / 293–299 W / 94–95 °C | 2.59–2.61 GHz / 280–287 W / 94–95 °C |
| C | 2.56–2.62 GHz / 292–297 W / 92–94 °C | 2.82–2.86 GHz / 234–254 W / 90–92 °C |
| D | 2.50–2.54 GHz / 296–298 W / 94–96 °C | 2.69–2.71 GHz / 277–280 W / 97–98 °C |

ComfyUI sits on the 300 W cap in every row. h3-hip stays below it, most in
row C, where the 4-step FastH3 run is not fully compute-bound.

## Memory

| row | ComfyUI DiT on GPU / offloaded, per run | ComfyUI LoRA patches | h3-hip DiT peak VRAM |
|---|---|---|---:|
| A | 25.1/7.3, 10.1/22.4, 10.1/22.4 GB | 0 | 21.4 GiB |
| C | 25.1/7.5, 10.0/22.4, 25.1/7.5 GB | 117, 206, 117 | 21.2 GiB |
| D | 19.8/12.6, 19.8/12.6, 5.2/27.3 GB | 160, 160, 205 | 21.0 GiB |

ComfyUI loads the 25.9 GB text encoder fully onto the GPU first, then the
DiT partially; how much stays resident varies run to run. The runs with
more offloaded are the slow ones (C 15.65 vs 12.6 s/step, D 59.5 vs
55.7–56.1). With the DiT not fully resident the turbo LoRA stays as
per-step patches. h3-hip merges FastH3 into the weights at load.

- With default caching ComfyUI was OOM-killed (47 GB RSS) while loading the
  DiT next to the cached text encoder. `--cache-none` fits, peaking at
  52 GB used RAM.
- A second job in the same ComfyUI process was OOM-killed again (the
  offloaded DiT stays in RAM), so ComfyUI needs a restart per job here.
- The h3-hip daemon keeps weights resident: two consecutive 864×480 FastH3
  jobs took 120.6 s and then **56.6 s** (same prompt, so the text encoder
  result was reused too). This is the setting a ComfyUI node would use; the
  table above deliberately compares fresh processes on both sides.

## What this supports, and what it does not

Matched on both sides: MiniMax-H3 FL2VA, all 50 DiT layers, the same step
count, one DiT forward per step (no CFG on either side), INT8 QKV /
attention-out / FC1 / FC2, same canvas, frames, prompt and seed value,
fresh processes, alternating order.

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
  ComfyUI's per-step cost there depends on how much of the DiT it offloads.
- ROCm runtime differs: ComfyUI's torch wheel bundles ROCm 7.13; h3-hip
  runs on the TheRock 10.1 runtime.
- Quality was not compared.

Claim that holds on this data: on an R9700 (32 GiB, 59 GiB RAM), at the
same model, step count and layer count, with INT8 main linears on both
sides and fresh processes, h3-hip's DiT denoise is **1.27×** faster than
ComfyUI native (`fca60fe`, PyTorch 2.11+rocm7.13) at 864×480 · 20 steps
and **1.42×** at 1344×768 · 4 steps; whole jobs are **1.16×** faster in
both. At 864×480 · 4 steps the whole job is on par (0.97×). The larger
practical difference is memory: ComfyUI needs a restart per job on this
box, while the daemon reuses resident weights.

Superseded: an earlier single-run pass (h3-hip runs back to back with a
warm page cache) gave 172.4 / 229.5 s for row A, 38.5 / 96.0 s for C and
148.7 / 258.1 s for D (denoise / E2E), and 1.19–1.35× E2E. Those runs were
not alternated and are not used for the claim. The same pass measured
h3-hip fox-fast (`--layers 45 --reuse 2`, lossy) at 87.6 / 142.2 s for
row A's shape; it has no ComfyUI counterpart and must not be quoted
against it.

The ComfyUI node prototype (`H3HipGenerate` → `SaveVideo`) produced the
same streams as the CLI for fox-s2; the daemon bugs it exposed are fixed
on `h3d-fixes`.
