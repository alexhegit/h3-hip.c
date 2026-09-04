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

## Feasibility (HIP Sage1)

Scope for the first MI300X experiment:

1. Per-block (or per-tile) scale of Q and K to INT8.
2. K-channel smoothing as in Sage1 (reduce outlier damage before quant).
3. INT8 MFMA for \(QK^\top\); dequant into the existing online-softmax /
   FP32 accum path.
4. Keep current FP16 \(PV\) until QK KEEP.
5. Opt-in only. Default SDPA stays BF16 flash / rocWMMA.

Out of scope for v1:

- Vendoring Python / Triton
- Sage2 INT4 QK or FP8 PV (follow-on on gfx942 only)
- Changing tagged PERFORMANCE.md quality rows
- Using Sage as a substitute for `--token-reduction`

**Expected magnitude (unmeasured):** paper 2× vs naive FA2. This tree already
has CDNA/Halo flash, so **15 s `sdpa=` more likely −20–40%**, not 2–5×.

## KEEP / REJECT (MI300X first)

Gate on **15 s cinematic** with `--profile`, same prompt / seed / geometry as
the current all-opts command in [`BEST_PRACTICE.md`](BEST_PRACTICE.md).

KEEP if:

- `sdpa=` drops enough to matter on denoise (aim: **≥20%** sdpa vs the same
  tree **without** Sage, TR held fixed — both off and both on)
- E2E does not regress from extra quant / smooth overhead
- visual KEEP vs the same-seed MP4 (not fox-s2 bit identity)

REJECT if:

- sdpa win &lt; ~10% after overhead
- obvious temporal artifacts vs the BF16 flash baseline
- VRAM or occupancy regression that hurts 15 s

fox-s2 **md5 will change** on a Sage path. Do not use fox-s2 identity as the
quality gate. Use it only to prove the default (Sage off) path is untouched.

## Implementation sketch

| Step | Where | Done when |
|------|--------|-----------|
| 1 | `docs/sageAttn.md` (this file) | branch exists on origin |
| 2 | gfx942 flash SDPA: INT8 QK bypass | compiles `HIP_ARCH=gfx942` |
| 3 | CLI / env opt-in, default off | fox-s2 default md5 unchanged |
| 4 | MI300X 15 s `--profile` A/B ±Sage ±TR | KEEP/REJECT table in `docs/perf-runs/` |
| 5 | gfx90a tile port | MI210 15 s repeat |
| 6 | gfx1151 only if (4) KEEP and 15 s still SDPA-bound | Halo 15 s |

Do not mix this work into the dirty local checkout that also holds unrelated
trees. Land on `sageattn` only.
