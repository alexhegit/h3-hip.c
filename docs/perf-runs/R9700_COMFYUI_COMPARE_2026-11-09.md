# R9700: ComfyUI native MiniMax-H3 vs h3-hip, 2026-11-09

Question: is a ComfyUI node that calls `h3 --serve` worth building, given
that ComfyUI now runs MiniMax-H3 natively on ROCm?

## Summary

h3-hip is faster than ComfyUI native on this card in every matched row,
including the worst pairing of individual runs:

| row | denoise | whole job | worst pairing (whole job) | how to quote it |
|---|---:|---:|---:|---|
| A. 864×480, 20 steps, dense | **1.39×** | **1.34×** | 1.27× | headline: same model, steps and layers, both dense |
| C. 864×480, 4-step turbo | 1.84× | 1.48× | 1.28× | different turbo adapters; ComfyUI patches its LoRA per step because the DiT does not fit |
| D. 1344×768, 4-step turbo | 1.50× | 1.27× | 1.24× | as C; h3-hip's lead shrinks outside denoise (VAE 41 s, `H3_INT8_HOST` load 32 s) |

The bigger practical difference is memory. On 59 GiB of RAM ComfyUI with
the full INT8 DiT needs `--cache-none` and a restart per job, or it is
OOM-killed. The h3-hip daemon reuses resident weights: a rerun of the same
prompt takes 52.1 s after a 106.5 s first job (KI-007 fix and VAE BF16
activations included, no environment needed).

Scope of the claim: this R9700 (32 GiB, 59 GiB RAM), ComfyUI `fca60fe`
with PyTorch 2.11+rocm7.13, fresh processes. Quality is not compared
(samplers, INT8 schemes and noise differ). The h3-hip fox-fast numbers
(≈2× whole job) are lossy and must not be quoted against ComfyUI.

Where h3-hip can gain more: the video VAE is still not power-bound (BF16
activations already cut 864×480 from ~21 to ~17 s and 1344×768 from ~54
to ~41 s, included above), and the DiT kernels themselves (denoise already
sits on the 300 W cap, so only efficiency helps: SDPA at about 40 % of
BF16 peak, grouped INT8 at about 112 of 391 TOPS).

For the node decision: the speed lead is real but moderate; the case for
the node is mostly that it lets ComfyUI users on smaller-RAM AMD boxes
render job after job.

Box: Radeon AI PRO R9700 (gfx1201, 32 GiB, PCIe 4.0 x4, 300 W cap),
59 GiB RAM. Prompt `prompt_starship.txt`, seed 42, 5 s (124 frames).

- **ComfyUI** `fca60fe` (2026-11-08), PyTorch 2.11.0+rocm7.13.0 from
  `repo.amd.com` (the TheRock torch 2.14 wheel ships a mismatched
  torchvision). comfy_kitchen reports its HIP backend. Graph = the official
  `video_minimax_h3_t2v` template: `res_multistep`, `simple`, BasicGuider.
  Weights from `Comfy-Org/minimax-h3`: `minimax_h3_fl2va_int8_convrot`
  (full, not pruned), `qwen3vl_32b_minimax_h3_int8_convrot`, fp16 video VAE.
  Started with `--cache-none` (see Memory).
- **h3-hip** branch `h3d-fixes` (r9700 after the VAE BF16-activation merge,
  plus daemon KI-007), CLI with `--profile`.

## Method

Three rounds per row, ComfyUI and h3-hip alternating (ComfyUI first in
rounds 1 and 3, h3-hip first in round 2), 30 s idle between runs; rows A
and C were run as a second full pass after the erratum below. Every
run is a fresh process on both sides, so both read their weights through
a page cache the other side just evicted. E2E is process launch to MP4 on
both sides (`/usr/bin/time` for h3-hip; ComfyUI server launch to the
history entry, about 5 s more than its `Prompt executed in`). ComfyUI
s/step is its tqdm average; denoise = steps × s/step. GPU clock, power and
junction temperature are 1 s samples of `hwmon` while `gpu_busy_percent`
≥ 80. Driver: `bench_comfy.py` + `rerun.py` in the prototype directory.

After the VAE BF16-activation PR merged, h3-hip alone was remeasured
(`rerun_h3.py`, three rounds A→C→D) with ComfyUI weight files dropped from
the page cache between runs; ComfyUI medians are unchanged from the
alternating passes.

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

## Results (median, range in brackets)

ComfyUI: 6 runs for A and C (two alternating passes), 3 for D. h3-hip:
3 runs per row after the VAE remasure. See the erratum below for why the
first pass's h3-hip A and C runs are excluded, and the superseded note for
the pre-VAE clean pass.

| row | ComfyUI s/step | ComfyUI denoise | ComfyUI E2E | h3-hip s/step | h3-hip denoise | h3-hip E2E | denoise ratio | E2E ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 11.82 [10.75–11.85] | 236.3 s [215.0–237.0] | 295.8 s [279.5–307.6] | 8.48 [8.45–8.48] | **169.6 s** [168.9–169.7] | **220.2 s** [220.0–220.3] | **1.39×** [1.27–1.40] | **1.34×** [1.27–1.40] |
| C | 15.65 [12.61–15.69] | 62.6 s [50.4–62.8] | 125.3 s [111.6–135.8] | 8.49 [8.47–8.50] | **34.0 s** [33.9–34.0] | **84.7 s** [84.6–87.4] | **1.84×** [1.48–1.85] | **1.48×** [1.28–1.60] |
| D | 56.11 [55.70–59.50] | 224.4 s [222.8–238.0] | 304.1 s [303.2–312.5] | 37.28 [37.25–37.91] | **149.1 s** [149.0–151.6] | **239.5 s** [239.5–243.9] | **1.50×** [1.47–1.60] | **1.27×** [1.24–1.30] |

Ratios are ComfyUI median / h3-hip median; the bracket is the worst and
best cross-pairing of individual runs. ComfyUI is bimodal: runs that keep
7.3–7.5 GB of the DiT offloaded take 10.8 (A) / 12.6 (C) s/step, runs that
offload 22.4 GB take 11.8 / 15.7 s/step, and which one happens varies per
start. The samplers (Euler vs `res_multistep`) and turbo LoRAs differ, so
outputs are different pictures; quality was not compared.

### Where the rest of the h3-hip time goes (median)

| row | text encoder | DiT load | video VAE | audio VAE, mux, start |
|---|---:|---:|---:|---:|
| A | 11.4 s | 18.9 s | 17.1 s | 3.2 s |
| C | 10.8 s | 19.9 s | 17.0 s | 3.2 s |
| D | 10.7 s | 31.5 s | 41.3 s | 6.9 s |

Row D's load includes parking 17.9 GiB of INT8 linears in pinned RAM
(`H3_INT8_HOST`), and its VAE is ~2.4× row A's for 2.5× the pixels. The
VAE BF16-activation merge is what moved 864×480 from ~21 s to ~17 s and
1344×768 from ~54 s to ~41 s.

### GPU telemetry (mean over busy samples, per run)

| row | ComfyUI sclk / power / Tj max | h3-hip sclk / power / Tj max |
|---|---|---|
| A | 2.53–2.64 GHz / 293–299 W / 94–95 °C | 2.49–2.52 GHz / 287–291 W / 93–96 °C |
| C | 2.56–2.65 GHz / 292–297 W / 92–94 °C | 2.62–2.68 GHz / 247–263 W / 91–94 °C |
| D | 2.50–2.54 GHz / 296–298 W / 94–96 °C | 2.63–2.66 GHz / 280–284 W / 97 °C |

These are whole-run means, so they mix phases. A 5 Hz trace of one row C
h3-hip run shows the denoise steps at 300 W mean, 2.47 GHz, 100 % busy,
8.4 s per step, i.e. on the power cap like ComfyUI. The lower run means
come from the text encoder (72 W), DiT load (122 W) and video VAE (254 W,
2.95 GHz, not power-bound).

### Erratum: the first alternating pass

The first three-round pass ran with `H3_INT8_HOST=1` leaked into the
environment from the row D setup, so its h3-hip A and C runs parked the
INT8 linears in host RAM and copied them every block: 9.27 s/step and
185.3 / 258.4 s (A), 11.46 s/step and 45.8 / 118.7 s (C). Those six runs
are replaced by the clean pass, and that clean pass is itself superseded by
the post-VAE remasure in the table (the driver now strips `H3_*` from the
environment and keeps each run's log). Its ComfyUI runs are unaffected and
are kept. Row D still needs `H3_INT8_HOST=1` on the CLI.

## Memory

| row | ComfyUI DiT on GPU / offloaded, per run | ComfyUI LoRA patches | h3-hip DiT peak VRAM |
|---|---|---|---:|
| A | 25.1/7.3, 10.1/22.4, 10.1/22.4, 10.1/22.4, 25.1/7.3, 10.1/22.4 GB | 0 | 21.4 GiB |
| C | 25.1/7.5, 10.0/22.4, 25.1/7.5, then 10.0/22.4 three times | 117, 206, 117, 206 ×3 | 21.2 GiB |
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
- The h3-hip daemon keeps weights resident. Three consecutive 864×480
  FastH3 jobs (same prompt, new seed, new prompt) take 106.5 s, **52.1 s**
  and 101.9 s with no environment set. Before the KI-007 fix the cached DiT
  pushed the video VAE and the next job into pinned host memory (212.7,
  179.4, 208.6 s); `--serve` now turns on `H3_INT8_HOST` itself on cards
  below 48 GiB. A new prompt re-prepares the DiT, so reuse pays off for
  reruns of one prompt. This is the setting a ComfyUI node would use; the
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
sides and fresh processes, h3-hip's DiT denoise is **1.39×** faster than
ComfyUI native (`fca60fe`, PyTorch 2.11+rocm7.13) at 864×480 · 20 steps,
**1.84×** at 864×480 · 4 turbo steps and **1.50×** at 1344×768 · 4 turbo
steps; whole jobs are **1.34×**, **1.48×** and **1.27×** faster. The
turbo ratios carry the adapter and offload caveats above. The larger
practical difference is memory: ComfyUI needs a restart per job on this
box, while the daemon reuses resident weights (with `H3_INT8_HOST=1`).

Superseded: the pre-VAE alternating clean pass gave h3-hip 172.7 /
228.8 s for row A, 36.8 / 95.5 s for C and 157.7 / 261.4 s for D (denoise
/ E2E; ratios 1.37× / 1.29×, 1.70× / 1.31×, 1.42× / 1.16×). An earlier
single-run pass (not alternated) gave 172.4 / 229.5, 38.5 / 96.0 and
148.7 / 258.1 s and is not used for the claim. The same pass measured
h3-hip fox-fast (`--layers 45 --reuse 2`, lossy) at 87.6 / 142.2 s for
row A's shape; it has no ComfyUI counterpart and must not be quoted
against it.

The ComfyUI node prototype (`H3HipGenerate` → `SaveVideo`) produced the
same streams as the CLI for fox-s2; the daemon bugs it exposed are fixed
in v0.19.0, which also ships the node as `integrations/comfyui/`
([guide](../wiki/ComfyUI.md)).
