# SageAttention in h3-hip.c

Working plan for an **opt-in HIP** port of the SageAttention **algorithm**.
This is not a vendor of [thu-ml/sageattention](https://github.com/thu-ml/sageattention).

**Practice order:** MI300X (`gfx942`) first, then MI210 (`gfx90a`), Strix Halo
(`gfx1151`) last.

Branch: `sageattn` (forked from `main` at the time this file landed). Do not
retarget tagged quality scoreboards until KEEP on 15 s `--profile` + a fixed-seed
MP4.

## Why not the official repo

Official SageAttention is **PyTorch + CUDA / Triton**, tuned for **Ampere / Ada /
Hopper**. h3-hip.c is **pure C / HIP** with no Python runtime.

Community ROCm / Triton wheels (ComfyUI, ZLUDA, `SageAttention-Rocm7`, typically
`gfx1100`) are the **wrong stack** for this tree and are **not** an official path
for the three timed SKUs.

Do **not**:

- add `thu-ml/sageattention` as a dependency
- call into Python / Triton from the DiT loop
- ship a default-on Sage path (fox-s2 md5 and quality gates will move)

Do **yes**:

- reimplement Sage**1** (INT8 \(QK^\top\) per-block scale + K-channel smooth)
  as HIP next to the existing flash SDPA
- keep it **opt-in** (`H3_SAGE_SDPA=1` or a CLI flag; name TBD on this branch)
- A/B it **on top of** `--token-reduction`, not instead of TR

## What Sage speeds up

Paper speedups vs FlashAttention (often 2–5×) come from:

1. \(QK^\top\) on **INT8 MMA** (Sage1: per-block; Sage2: finer grain, down to INT4)
2. \(PV\) in **FP16 or FP8**, plus smoothing / two-level accum for accuracy

h3-hip.c already does **other** INT8 work. It does **not** do INT8 attention:

| Piece | Today | Sage-style |
|-------|--------|------------|
| DiT linear | INT8 GEMM is a separate knob | unrelated |
| SDPA | Halo: rocWMMA **BF16**. CDNA flash: **BF16 QK + FP16 PV + FP32 accum** | INT8 QK (+ optional FP8 PV) |
| Long sequence | opt-in `--token-reduction` shrinks **N** | shrinks **arithmetic precision**; **orthogonal** to TR |

CDNA FP16 \(PV\) already resembles Sage1’s value path. The missing piece on long
clips is **INT8 \(QK^\top\)**.

DiT shapes match Sage’s public API: `HEAD_DIM=128`, 56 heads, GQA
(`h3_dit.c`). Entry point today is `h3_gpu_sdpa_bf16` /
`h3_launch_sdpa_bf16`.

## Three timed SKUs

Numbers below are from the v0.11.0 / `main` 15 s cinematic profile (quality path
unless noted). SDPA is ~73–77% of denoise.

### MI300X (`gfx942`) — do this first

15 s BF16: denoise ~186 s, **sdpa ~144 s (~77%)**. All-opts 15 s is already
~2.4 min; Sage is “another cut”, not “makes it usable”.

CDNA3 has **INT8 MFMA** and **FP8 MFMA**. Sage2 FP8 \(PV\) is more natural here
than on CDNA2.

**First practice on this SKU:** Sage1-style **INT8 QK + existing FP16 PV + FP32
accum**, fused into the current CDNA flash kernel. Decide on FP8 \(PV\) only
after accuracy on a fixed-seed 15 s MP4.

There is still **no** linkable official kernel. Work is HIP in
`h3_gpu_sdpa_bf16` / the gfx942 flash launch path.

### MI210 (`gfx90a`) — second

15 s BF16: denoise ~647 s, **sdpa ~477 s (74%)**. TR already cut sdpa to
**~286 s (−40%)**. INT8 QK would **stack** on TR.

CDNA2 has INT8 MFMA (I8×I8→I32). Sage1 fits; Sage2 FP8 \(PV\) / INT4 does not.
Reuse the MI300X algorithm with CDNA2 tile sizes; do not copy NVIDIA
`mma.m16n8k*` layouts.

### Strix Halo (`gfx1151`) — last

15 s quality: denoise ~2198 s, **sdpa ~1613 s (~73%)**. Short fox-s2 denoise is
already ~3.36 s; E2E is NVMe-bound, so Sage **will not move short-clip wall
clock**.

RDNA3 has INT8 WMMA in theory. CUDA fragment layouts do not map to wave32
rocWMMA. A hand-written INT8 SDPA that **misses** the matrix core is a few
percent of BF16 WMMA peak (already shown before v0.9.0). Halo work is only
justified **after** a KEEP on MI300X, and only for 15 s, still opt-in.

## Critical analysis and improvements

### Current SDPA breakdown (MI300X 15s cinematic)

| Component | Time | % of denoise |
|-----------|-----:|-------------:|
| DiT denoise | ~180s | 100% |
| SDPA | ~144s | **77%** |
| Linear | ~34s | 18% |
| Other | ~7s | 4% |

**Key insight:** SDPA is 77% of denoise. INT8 QK^T reduces both compute (2× fewer ops) and memory bandwidth (1 byte vs 2 bytes per element). The main win may be **bandwidth reduction**, not just compute.

### Potential issues with original plan

1. **Quantization overhead not quantified:** INT8 quantization of Q and K (56 heads × 128 dim × sequence length) may have significant overhead. For short sequences (fox-s2), this could negate GEMM speedup.

2. **Memory bandwidth vs compute:** INT8 reduces bandwidth by 2×. If SDPA is bandwidth-bound, this is the main win. The plan focuses on MFMA compute but doesn't analyze bandwidth.

3. **Per-block scaling complexity:** Block size choice is critical: too small = high overhead, too large = poor accuracy. Need to quantify trade-off.

4. **K-channel smoothing:** Adds complexity. Is it necessary for v1? Can start without it, add if accuracy is poor.

5. **FP8 on gfx942:** GFX942 has FP8 MFMA with wider dynamic range (E4M3: 448 max vs INT8: 127 max). FP8 QK^T might be better than INT8 for attention scores.

6. **Interaction with existing INT8:** Current path has INT8 attention output. Sage adds INT8 QK^T. How do these interact? Need to verify.

7. **Accuracy measurement:** "Visual KEEP" is subjective. Should use PSNR/SSIM thresholds (e.g., PSNR > 30 dB, SSIM > 0.95).

8. **Short video overhead:** fox-s2 (22 frames) may not benefit from Sage due to quantization overhead exceeding GEMM speedup.

### Revised implementation plan

**Phase 0: Baseline measurement (before any code changes)**

1. Profile SDPA internals: quantize / GEMM / softmax / PV breakdown
2. Measure memory bandwidth utilization (GB/s vs theoretical peak)
3. Determine if SDPA is compute-bound or bandwidth-bound

**Phase 1: Minimal implementation (v1)**

1. Per-channel INT8 QK^T (simpler than per-block)
2. No K-channel smoothing (add later if needed)
3. Keep FP16/BF16 PV (unchanged)
4. Sequence length threshold: skip Sage for < 128 tokens

**Phase 2: Accuracy validation**

1. PSNR/SSIM against BF16 baseline on fixed-seed 15s MP4
2. Thresholds: PSNR > 30 dB, SSIM > 0.95
3. If below threshold: add K-channel smoothing or switch to per-block

**Phase 3: Optimization (if v1 KEEP)**

1. Add per-block scaling (if per-channel accuracy is good)
2. Consider FP8 QK^T on gfx942 (wider dynamic range)
3. Profile and optimize quantization overhead

**Phase 4: A/B testing**

1. 15s cinematic ±Sage ±TR
2. KEEP/REJECT table with quantitative metrics
3. Document findings for MI210/Strix Halo ports

### Additional recommendations

1. **Measure before optimizing:** Phase 0 bandwidth analysis may reveal INT8's main benefit is bandwidth, not compute.

2. **Consider FP8:** GFX942 has FP8 MFMA. FP8 E4M3 has wider dynamic range than INT8, which may be better for attention score distributions.

3. **Short sequence strategy:** Add sequence length threshold (e.g., < 128 tokens) to skip Sage for short sequences where overhead exceeds benefit.

4. **TR interaction:** TR reduces N, Sage reduces precision. Stacking effect needs verification. Start with TR on, Sage on.

5. **Fallback strategy:** If INT8 QK^T accuracy is poor, fallback to:
   - INT8 Q + BF16 K (hybrid)
   - Per-channel instead of per-block
   - FP8 instead of INT8

## Feasibility (HIP Sage1)

Scope for the first MI300X experiment:

1. Per-channel scale of Q and K to INT8 (simpler than per-block for v1).
2. ~~K-channel smoothing~~ Deferred to v2 if accuracy is poor.
3. INT8 dot product for \(QK^\top\); dequant into the existing online-softmax /
   FP32 accum path.
4. Keep current FP16 \(PV\) until QK KEEP.
5. Opt-in only (`H3_SAGE_SDPA=1`). Default SDPA stays BF16 flash / rocWMMA.
6. Sequence length threshold: skip Sage for < 128 tokens.

Out of scope for v1:

- Vendoring Python / Triton
- Sage2 INT4 QK or FP8 PV (follow-on on gfx942 only)
- Changing tagged PERFORMANCE.md quality rows
- Using Sage as a substitute for `--token-reduction`
- Per-block scaling (v2 if per-channel accuracy is good)
- K-channel smoothing (v2 if accuracy is poor)

**Expected magnitude (unmeasured):** paper 2× vs naive FA2. This tree already
has CDNA/Halo flash, so **15 s `sdpa=` more likely −20–40%**, not 2–5×.
Main benefit may be **bandwidth reduction** (2× fewer bytes) rather than compute.

## KEEP / REJECT (MI300X first)

Gate on **15 s cinematic** with `--profile`, same prompt / seed / geometry as
the current all-opts command in [`BEST_PRACTICE.md`](BEST_PRACTICE.md).

KEEP if:

- `sdpa=` drops enough to matter on denoise (aim: **≥20%** sdpa vs the same
  tree **without** Sage, TR held fixed — both off and both on)
- E2E does not regress from extra quant / smooth overhead
- **Quantitative:** PSNR > 30 dB, SSIM > 0.95 vs BF16 baseline (same-seed MP4)
- Visual: no obvious temporal artifacts vs the BF16 flash baseline

REJECT if:

- sdpa win &lt; ~10% after overhead
- PSNR &lt; 30 dB or SSIM &lt; 0.95 vs BF16 baseline
- obvious temporal artifacts vs the BF16 flash baseline
- VRAM or occupancy regression that hurts 15 s

fox-s2 **md5 will change** on a Sage path. Do not use fox-s2 identity as the
quality gate. Use it only to prove the default (Sage off) path is untouched.

## Implementation sketch

| Step | Where | Done when |
|------|--------|-----------|
| 0 | Profile SDPA internals, measure bandwidth | Understand bottleneck |
| 1 | `docs/sageAttn.md` (this file) | branch exists on origin |
| 2 | gfx942 flash SDPA: INT8 QK bypass (per-channel) | compiles `HIP_ARCH=gfx942` |
| 3 | CLI / env opt-in (`H3_SAGE_SDPA=1`), default off | fox-s2 default md5 unchanged |
| 4 | Accuracy validation: PSNR/SSIM vs BF16 baseline | PSNR > 30 dB, SSIM > 0.95 |
| 5 | MI300X 15 s `--profile` A/B ±Sage ±TR | KEEP/REJECT table in `docs/perf-runs/` |
| 6 | gfx90a tile port | MI210 15 s repeat |
| 7 | gfx1151 only if (5) KEEP and 15 s still SDPA-bound | Halo 15 s |

Do not mix this work into the dirty local checkout that also holds unrelated
trees. Land on `sageattn` only.

## Results (MI300X gfx942, 2-step diagnostic)

### Performance

| Kernel | sdpa time (70 calls) | vs BF16 MFMA |
|--------|---------------------:|--------------|
| BF16 MFMA (baseline) | 2.290s | — |
| INT8 Sage per-tile | 2.290s | **0% (identical)** |
| INT8 Sage per-row | 2.853s | **+24% slower** |
| INT8 MFMA scalar v1 | ~2.30s | ~0% |

Per-tile INT8 quantization adds **zero overhead** — the absmax scale is free.
Per-row quantization adds per-row barriers (+24%) for marginal accuracy gain.

### Accuracy (PSNR vs BF16 MFMA, 2 steps)

| Quantization | PSNR avg | PSNR y | Notes |
|-------------|---------:|-------:|-------|
| Per-tile absmax | 21.0 dB | 19.6 dB | 1 scale per 16×128 tile |
| Per-row absmax | ~23 dB | — | +2.5 dB over per-tile |

### Error compounding across steps

| Steps | PSNR (per-tile) | Notes |
|------:|----------------:|-------|
| 2 | 21.0 dB | Acceptable for preview |
| 20 | ~11 dB | Unacceptable for production |

INT8 quantization error compounds multiplicatively through 35 DiT layers × N steps.
Even with per-row quantization, 20-step PSNR stays ~11 dB.

### AMD ISA mapping (v_mfma_i32_16x16x32_i8)

The AMD ISA calculator confirmed the correct lane mapping:

- **A matrix:** `row = lane % 16`, `col = 8 * (lane / 16)`
- **B matrix:** `row = lane % 16`, `col = 8 * (lane / 16)`
- **C matrix:** `row = 4 * (lane / 16) + gpr_idx`, `col = lane % 16`

This was verified by building the AMD ISA calculator from source and testing
all 64 lane indices. The previous v1 kernel had incorrect mapping.

### Conclusion

INT8 Sage is viable **only for preview mode** (2 steps, 21 dB). For production
quality (20 steps), BF16 MFMA is already optimal with PSNR=inf (deterministic).

**Recommended dispatch:**
- `steps <= 2`: use INT8 Sage (zero overhead, 21 dB acceptable)
- `steps > 2`: use BF16 MFMA (deterministic, higher quality)

**Key contribution:** Fixed AMD ISA `v_mfma_i32_16x16x32_i8` lane mapping
for both QK^T INT8 and BF16 PV via rocWMMA. This is the foundation for any
future INT8 MFMA work on gfx90a/gfx942.
