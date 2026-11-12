# Optimizing the other ISAs after the R9700 work

> Status: plan, not started. Priority: **MI300X (gfx942) first**, then
> Strix Halo (gfx1151), then MI210 (gfx90a). Tracks
> [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md) item X. Needs hardware that the R9700
> box does not have.

v0.18.0 to v0.20.0 were tuned and timed on one Radeon AI PRO R9700
(gfx1201) only. The other ISAs were compile-checked. Most of the kernel
gains are gated to gfx12; a few shared changes reach every ISA and have
never run there.

## What reaches which ISA

| change | gfx1201 / gfx1200 | gfx1151 | gfx90a / gfx942 |
|---|---|---|---|
| Prompt / AdaLN disk cache, video-encoder tiled Conv3d and device activations, row-chunked grouped INT8, VRAM placement, `H3_INT8_HOST` double buffer | on | on | on |
| Video VAE BF16 weights and BF16 activations (`h3_f32_wmma_on`, RDNA only) | on | **on, never run on gfx11** | off: VAE linears go through FP32 hipBLAS SGEMM |
| Video VAE BK=64 tile (`H3_F32_WMMA_CFG=6`) | on | off (cfg 3, BK=32) | off |
| SDPA prefetch / S^T-O^T kernel, finite-only P rounding, last-tile mask | on | off (8-wave WMMA kernel) | off (`h3_sdpa_bf16_mfma_d128_kernel`) |

## P0 — MI300X (gfx942)

Why first: at 1344×768 · 15 s dense SDPA was 80% of process E2E (744.2 s
of 947 s, [`perf-runs/MI300X_2026-09-21_1344x768-15s-sdpa.md`](perf-runs/MI300X_2026-09-21_1344x768-15s-sdpa.md)).
One call at 109334 tokens is 4·S²·128·56 = 3.43e14 FLOP. On the same run
the R9700 spends 1783.5 s, about 91 TFLOP/s per call, 45% of its measured
202 TFLOP/s BF16 peak. Scaled by 1783.5 / 744.2, MI300X runs about
220 TFLOP/s, **about 17% of its 1307 TFLOP/s spec peak**. The MFMA SDPA
kernel is the largest known headroom on any ISA.

- [ ] **M0 Regression on v0.20.0.** `make HIP_ARCH=gfx942 test`, fox-s2 /
  fox-fast / 832×480 · 5 s dense md5 against the MI300X ledger. Grouped INT8
  was proven bit-identical on gfx1201 only (with fp contraction off), so a
  changed md5 here needs an A/B against the pre-R9700 binary before anything
  else. 1344×768 · 15 s is expected to change (no `grouped_r64` fallback
  above 65535 rows).
- [ ] **M1 Measure the ceiling.** Sustained BF16 / FP8 / INT8 GEMM rate on
  this MI300X (as done for the R9700: the first R9700 number was taken
  before the clock ramped and was wrong). Then per-call SDPA time at 32k and
  109k tokens: `H3_SDPA_DUMP` + `make h3_sdpa_bench` on captured inputs,
  plus `rocprofv3` counters (MFMA busy, LDS bank conflicts, occupancy).
- [ ] **M2 Bit-identical SDPA work.** Candidates from the gfx12 kernel: keep
  P out of LDS (S^T / O^T form or register-resident P), more K/V reuse per
  block, finite-only P rounding and masking only the last key tile, wave and
  key-tile sweep. Gate: same md5 on fox-s2, 832×480 · 5 s and 1344×768 · 5 s.
- [ ] **M3 Reference kernels.** Time CK / aiter flash attention on the same
  captures. They change accumulation order, so adopting one is a lossy
  switch: opt-in, PSNR / SSIM against dense like `SOL_ATTN.md`.
- [ ] **M4 Video VAE on CDNA (optional, lossy).** The block linears run FP32
  SGEMM. Rounding weights and activations to BF16 MFMA, as RDNA already does,
  changes CDNA output bytes; needs a quality check before it can be default.
- [ ] **M5 New MI300X scoreboard.** The ledger is 2026-10-07. Rerun the full
  matrix like [`perf-runs/R9700_2026-11-12.md`](perf-runs/R9700_2026-11-12.md),
  with md5 per cell.

## P1 — Strix Halo (gfx1151)

- [ ] **H0 Regression.** `make HIP_ARCH=gfx1151 test`, `halo-regression`
  (fox-s2 `1731f95c…`), fox-fast and 832×480 · 5 s dense md5. The VAE BF16
  weight / activation path is new on gfx11: check it bit for bit with
  `make h3_bench_vae_linear` and `H3_VAE_F32_ACT=1` / `H3_F32_BF16_WEIGHT=0`
  A/Bs.
- [ ] **H1 VAE tile sweep.** `h3_bench_vae_linear` over every config and
  BK on gfx1151; BK=64 was only measured on gfx12.
- [ ] **H2 SDPA.** Port the S^T / O^T idea to the gfx11 WMMA layout (inputs
  replicated across half-waves, accumulator row `2*i + (lane>>4)`), so it is
  a new kernel, not a flag. Halo's linears are bandwidth-bound on unified
  memory, so expect a smaller E2E share than on the R9700.

## P2 — MI210 (gfx90a)

- [ ] **A0 Regression** as M0 (`make HIP_ARCH=gfx90a test`, md5 against
  [`perf-runs/MI210_2026-11-05.md`](perf-runs/MI210_2026-11-05.md)).
- [ ] **A1** Pick up M2 if it lands in the shared MFMA kernel; no FP8 here.

## Later — Radeon Pro W7900 (gfx1100)

The W7900 port is public `h3-hip.c` PR #14, not in this tree. Bring it in
before H2 so one gfx11 SDPA kernel serves gfx1100 and gfx1151.

## Rules for every item

- A default-on change must keep the output bytes, checked by md5 on that
  ISA; otherwise it is opt-in with PSNR / SSIM against dense.
- Record measurements under `docs/perf-runs/` with the ISA and date, and
  update `PERFORMANCE.md` and `KNOWN_ISSUES.md` item X when an ISA is
  re-verified.
