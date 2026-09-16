# SageAttention INT8 on MI300X (gfx942) — experiment record

**Branch:** `sageattn` (PR [#2](https://github.com/alexhegit/h3-hip.c/pull/2)).  
**Decision:** **REJECT for merge to `main`.** Opt-in code stays on this branch for
reference only.

**One-line summary:** We implemented Sage1-style INT8 \(QK^\top\) (+ optional
INT8 \(PV\)) on MI300X, fixed several correctness bugs, and measured end-to-end.
**SDPA did not get faster** (+1% best case, +41% with full INT8 PV). **Quality
can be tuned** (up to ~38.8 dB PSNR on a 2-step diagnostic after bugfixes), but
that path is **slower**, and production 20-step runs still lose badly to dense
flash unless you accept large error. The bottleneck is **online softmax / scalar
work**, not MFMA throughput.

See also the design notes in [`sageAttn.md`](sageAttn.md).

---

## Test environment

| Item | Value |
|------|--------|
| GPU | AMD MI300X (`gfx942`), CDNA3, 192 GiB HBM3 |
| Model | MiniMax-H3, `head_dim=128`, 56 heads (DiT), 24 heads in some early benches |
| Primary diagnostic | 100 frames, **2 denoise steps**, seed 42, fox prompt |
| Baseline | BF16 flash SDPA (`h3_launch_sdpa_bf16`, MFMA D=128) |
| Opt-in | `H3_SAGE_SDPA=1` (default **off**); optional `H3_SAGE_MAX_STEPS=N` |

---

## Final numbers (after v7–v8 + QK+PV bugfixes)

### SDPA wall (`sdpa=` from `--profile`, 100f · 2 steps)

| Path | sdpa | vs BF16 | Video PSNR / SSIM vs BF16 | Notes |
|------|-----:|--------:|---------------------------|--------|
| **BF16 baseline** | **2.837 s** | — | reference (deterministic) | Production default on `main` |
| INT8 QK + BF16 PV | **2.866 s** | **+1.0%** | **32.9 dB / 0.939** | No speed win; mild quality loss |
| INT8 QK + INT8 PV | **4.007 s** | **+41.3%** | **38.8 dB / 0.962** | Slower; extra V-quant + traffic |

**Takeaway:** There is **no performance benefit**. The “high quality” INT8 PV
path is **much slower**. INT8 QK alone is a wash on speed with **~7 dB** PSNR
drop on this diagnostic.

### Long-sequence reference (what actually moves wall clock)

| Config | sdpa (100f · 20 steps) | vs no TR |
|--------|------------------------:|---------|
| BF16 baseline | 28.77 s | — |
| `--token-reduction` | **16.31 s** | **−43%** |

Token reduction (on `main`) remains the validated MI300X lever for long SDPA.
Sol-Attn (merged on `main` after this branch) is a **separate** opt-in sparse
path; Sage INT8 QK is **orthogonal** and did not beat dense flash here.

---

## Experiment timeline (branch commits)

| Phase | Commit | What we tried | Outcome |
|-------|--------|---------------|---------|
| Plan | `f433bf1` | HIP port plan, no CUDA/Triton vendor | — |
| v1 | `d2a4bf5` | Scalar INT8 QK | PSNR **8.69 dB** — broken |
| v5–v6 | `8ac757d` | MFMA INT8 QK + rocWMMA PV; AMD lane map fix | ~21 dB @ 2 steps; ISA map validated |
| v7 | `9b1f977` | Pre-quant K, register Q, LDS bank padding | **9% faster** sdpa @ 2 steps, PSNR 28.5 dB — superseded by later analysis |
| v8 | `de4aa46` | K-smoothing + per-row scales | PSNR **36.6 dB** @ 2 steps, **27.7 dB** @ 20 steps — still no stable E2E win vs BF16 at production steps |
| QK+PV | `71f4f4b` | Full INT8 QK+PV; **3 correctness bugs fixed** | PSNR **18 → 38.8 dB**; sdpa **+41%** vs BF16 |
| Close-out | `10c61bf` | Root-cause write-up | **REJECT** — bandwidth / softmax bound |

### Correctness bugs fixed in INT8 QK+PV kernel (`71f4f4b`)

1. **PV tile count:** used `D_TILES=4` (K tiling) instead of `PV_TILES=8` (D/16).
2. **Output indexing:** wrote with `j*32` instead of `j*16` for PV MFMA N=16.
3. **V load:** missing per-head offset in transposed V layout.

Until these were fixed, measured “quality” was meaningless (PSNR ~18 dB).

---

## Root cause (why INT8 MFMA did not help)

1. **Online softmax dominates.** Per KV tile, softmax + P quant ≈ **320 cycles**;
   PV MFMA ≈ **64 cycles** (~20% of tile time). Halving MFMA saves ~32 cycles
   that are invisible next to softmax.
2. **K bandwidth is tiny.** INT8 QK halves K bytes, but K is ~**5%** of SDPA
   traffic; savings are microsecond-scale and lost in scalar work.
3. **INT8 PV adds overhead.** Extra `h3_sage_quant_v_int8_kernel` launch +
   HBM read/write **increases** wall time despite higher PSNR.
4. **FP8 PV rejected.** gfx942 FP8 FNUZ (max 224) accumulated error → ~**16.7 dB**
   E2E PSNR in early trials; not pursued.

---

## Quality vs performance (plain language)

| Question | Answer |
|----------|--------|
| Did Sage make MI300X **faster**? | **No** (+1% … +41% sdpa on measured paths). |
| Did Sage **always** reduce quality? | **No** — after fixes, INT8 QK+PV reached **38.8 dB** on the 2-step diagnostic, but **slower** than BF16. |
| Best “free” quality path on `main`? | **Dense INT8 DiT + BF16 flash SDPA** (default). |
| Best **fast** long-video knobs on `main`? | **`--token-reduction`** and/or **`--sol-attn`** (each lossy; documented separately). |

**Do not merge this branch** expecting Sage to replace those knobs on MI300X.

---

## How to reproduce on this branch

```bash
git checkout sageattn
make HIP_ARCH=gfx942 clean h3

# Default (Sage off) — same as main dense path for SDPA
H3_MODEL=/path/to/MiniMax-H3 ./bench/fox-fast.sh

# Opt-in Sage (exploration only)
H3_SAGE_SDPA=1 H3_MODEL=/path/to/MiniMax-H3 ./h3 --profile ...
```

---

## 中文摘要

**结论：SageAttention INT8 在 MI300X 上无法获得可用的性能提升，不建议合入 main。**

- **性能：** INT8 QK + BF16 PV 的 sdpa **+1%**（无收益）；INT8 QK + INT8 PV **+41%**（更慢）。
- **质量：** 修 bug 后 2 步诊断可达 **38.8 dB**，但路径更慢；INT8 QK + BF16 PV 约 **32.9 dB** 且无加速。
- **根因：** SDPA 瓶颈在 online softmax 标量计算，不在 MFMA；INT8 省下的 K/V 带宽和 MFMA 周期被掩盖。
- **对比：** 同平台上 **token reduction**（−43% sdpa）和 **Sol-Attn**（main 上已测）才是长视频加速方向；Sage 本分支保留作实验记录。

---

## Recommended follow-ups (not on this branch)

- Use **`--token-reduction`** or **`--sol-attn`** on `main` when wall clock beats fidelity.
- Keep **`H3_SAGE_SDPA=0`** on any production / scoreboard path.
- Re-open Sage only if a new hypothesis changes the softmax-bound profile (unlikely on gfx942 flash as implemented today).
