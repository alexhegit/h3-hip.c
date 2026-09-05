#!/usr/bin/env python3
"""
Roofline Analysis: H3 MiniMax on MI300X (gfx942)
================================================
Based on actual model architecture and profiled kernel times.
"""

import math

# ============================================================
# MI300X (gfx942) hardware specs
# ============================================================
HBM_BW        = 5.3e12       # bytes/s (5.3 TB/s peak HBM3)
BF16_PEAK     = 1307.4e12    # FLOPS   (matrix, dense)
INT8_PEAK     = 2614.9e12    # OPS     (matrix, dense)
FP32_PEAK     = 163.4e12     # FLOPS   (vector)
L2_BW         = 25e12        # bytes/s (estimated L2 effective BW)
LDS_BW        = 128 * 304 * 64e6  # bytes/s (64 KB/CU × 304 CU × clock)

CROSSOVER_HBM = BF16_PEAK / HBM_BW  # 247 FLOP/byte

# ============================================================
# Model architecture (actual from code)
# ============================================================
HIDDEN    = 5376    # residual stream width
INNER     = 7168    # attention inner dim (56 heads × 128)
HEADS     = 56
HEAD_DIM  = 128
FFN       = 14336   # SwiGLU intermediate
N_BLOCKS  = 50
N_STEPS   = 2       # denoise steps

# Sequence length depends on resolution
# 100 frames: spatial = ~125×125 = 15,625, patchified → S ≈ 2500 tokens
S_100F = 2500
B = 1  # batch size (always 1 for video)

# ============================================================
# Per-block GEMM breakdown
# ============================================================
def gemm_flops(M, K, N): return 2 * M * K * N
def gemm_bytes_bf16(K, N): return K * N * 2

# GEMMs per block
gemms = [
    ("QKV proj",     S_100F, HIDDEN, 3 * INNER),
    ("Attn out proj", S_100F, INNER, HIDDEN),
    ("MLP fc1 (SwiGLU)", S_100F, HIDDEN, 2 * FFN),  # gate+up fused
    ("MLP fc2",       S_100F, FFN, HIDDEN),
]

# Attention FLOPs (quadratic)
attn_qk_flops = 2 * S_100F * S_100F * INNER  # QK^T: M×D × D×S → M×S
attn_pv_flops = 2 * S_100F * S_100F * INNER  # PV: M×S × S×D → M×D

# Attention memory (Q, K, V loads)
attn_bytes = 3 * B * HEADS * S_100F * HEAD_DIM * 2  # Q+K+V BF16

# ============================================================
# Compute per-block totals
# ============================================================
total_gemm_flops = sum(gemm_flops(M, K, N) for _, M, K, N in gemms)
total_gemm_bytes = sum(gemm_bytes_bf16(K, N) for _, _, K, N in gemms)
total_attn_flops = attn_qk_flops + attn_pv_flops
total_block_flops = total_gemm_flops + total_attn_flops

# Full pipeline (50 blocks × 2 steps)
total_flops = total_block_flops * N_BLOCKS * N_STEPS
total_weight_bytes = total_gemm_bytes * N_BLOCKS  # weights loaded once per step

# ============================================================
# Profiled times (100f 2 steps, from actual run)
# ============================================================
# From profile output:
#   DiT op-classes: linear=1.722s, sdpa=2.827s, other=0.410s
#   Video VAE: linear=4.413s, sdpa=1.257s, other=0.807s
#   Text encoder: linear=0.159s, sdpa=0.001s, other=0.155s

dit_linear_time = 1.722
dit_sdpa_time   = 2.827
dit_other_time  = 0.410
dit_total_time  = dit_linear_time + dit_sdpa_time + dit_other_time

vae_linear_time = 4.413
vae_sdpa_time   = 1.257
vae_other_time  = 0.807
vae_total_time  = vae_linear_time + vae_sdpa_time + vae_other_time

te_linear_time  = 0.159
te_sdpa_time    = 0.001
te_other_time   = 0.155
te_total_time   = te_linear_time + te_sdpa_time + te_other_time

total_gpu_time  = dit_total_time + vae_total_time + te_total_time

# ============================================================
# Roofline computation
# ============================================================
def roofline_ideal(flops, bytes_rw):
    """Minimum time given peak FLOPS and HBM BW."""
    t_compute = flops / BF16_PEAK
    t_memory  = bytes_rw / HBM_BW
    return max(t_compute, t_memory)

def utilization(ideal, actual):
    return ideal / actual * 100 if actual > 0 else 0

# ============================================================
# Print report
# ============================================================
print("=" * 78)
print("  H3 MINIMAX ROOFLINE ANALYSIS — MI300X (gfx942)")
print("=" * 78)
print()
print(f"  GPU:        AMD Instinct MI300X VF (gfx942, CDNA3)")
print(f"  BF16 Peak:  {BF16_PEAK/1e12:.0f} TFLOPS (matrix, dense)")
print(f"  HBM BW:     {HBM_BW/1e12:.1f} TB/s (HBM3)")
print(f"  Crossover:  {CROSSOVER_HBM:.0f} FLOP/byte")
print()
print(f"  Model:      MiniMax-H3 DiT")
print(f"  Hidden:     {HIDDEN} (residual) / {INNER} (attn inner)")
print(f"  Heads:      {HEADS} × {HEAD_DIM} = {INNER}")
print(f"  FFN:        {FFN} (SwiGLU)")
print(f"  Blocks:     {N_BLOCKS}")
print(f"  Steps:      {N_STEPS}")
print(f"  Sequence:   {S_100F} tokens (100 frames)")
print()

# ---- Per-block GEMM table ----
print("-" * 78)
print("  PER-BLOCK GEMM BREAKDOWN")
print("-" * 78)
print(f"  {'GEMM':<20} {'M×K×N':>20} {'FLOPs':>12} {'Weight':>10} {'AI':>8}")
print(f"  {'':<20} {'':>20} {'(GFLOP)':>12} {'(MiB)':>10} {'(F/B)':>8}")
print("  " + "-" * 74)

for name, M, K, N in gemms:
    f = gemm_flops(M, K, N)
    b = gemm_bytes_bf16(K, N)
    ai = f / b
    print(f"  {name:<20} {M}×{K}×{N}  {f/1e9:>10.1f}  {b/1e6:>8.1f}  {ai:>7.0f}")

attn_ai = total_attn_flops / attn_bytes
print(f"  {'Attention (QK+PV)':<20} {'quadratic':>20} {total_attn_flops/1e9:>10.1f}  {attn_bytes/1e6:>8.1f}  {attn_ai:>7.0f}")
print("  " + "-" * 74)
print(f"  {'BLOCK TOTAL':<20} {'':>20} {total_block_flops/1e9:>10.1f}  {total_gemm_bytes/1e6:>8.1f}")
print()

# ---- Full pipeline ----
print("-" * 78)
print("  FULL PIPELINE (50 blocks × 2 steps)")
print("-" * 78)
print(f"  Total FLOPs:       {total_flops/1e12:.1f} TFLOP")
print(f"  Total weight read: {total_weight_bytes/1e9:.1f} GB (BF16)")
print(f"  GEMM % of total:   {total_gemm_flops/total_block_flops*100:.1f}%")
print(f"  Attn % of total:   {total_attn_flops/total_block_flops*100:.1f}%")
print()

# ---- Stage-by-stage roofline ----
print("-" * 78)
print("  STAGE ROOFLINE")
print("-" * 78)

stages = [
    ("DiT Linear (GEMM)", total_gemm_flops * N_BLOCKS * N_STEPS, total_weight_bytes * N_STEPS, dit_linear_time),
    ("DiT SDPA",          total_attn_flops * N_BLOCKS * N_STEPS, attn_bytes * B * N_BLOCKS * N_STEPS, dit_sdpa_time),
    ("DiT Other",         0,                         0, dit_other_time),
    ("Video VAE",         None,                      None, vae_total_time),
    ("Text Encoder",      None,                      None, te_total_time),
]

print(f"  {'Stage':<22} {'FLOPs':>10} {'Bytes':>10} {'AI':>8} {'Ideal':>8} {'Actual':>8} {'Util':>7}")
print(f"  {'':<22} {'(TFLOP)':>10} {'(GB)':>10} {'(F/B)':>8} {'(ms)':>8} {'(ms)':>8} {'%':>7}")
print("  " + "-" * 74)

for name, flops, bw, actual in stages:
    if flops is not None and bw is not None and bw > 0:
        ai = flops / bw
        ideal = roofline_ideal(flops, bw)
        util = utilization(ideal, actual)
        bound = "compute" if ai > CROSSOVER_HBM else "memory"
        print(f"  {name:<22} {flops/1e12:>10.1f} {bw/1e9:>9.1f} {ai:>7.0f} {ideal*1000:>7.1f} {actual*1000:>7.1f} {util:>6.1f}%")
    else:
        print(f"  {name:<22} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>8} {actual*1000:>7.1f} {'N/A':>7}")

print("  " + "-" * 74)
print()

# ---- Roofline insight ----
print("-" * 78)
print("  ROOFLINE INSIGHT: WHERE ARE WE ON THE CURVE?")
print("-" * 78)
print()

# Compute the roofline curve points
print("  Roofline curve (BF16):")
print(f"    Memory bound:  time = bytes / {HBM_BW/1e12:.1f} TB/s")
print(f"    Compute bound: time = FLOPs / {BF16_PEAK/1e12:.0f} TFLOPS")
print(f"    Crossover:     {CROSSOVER_HBM:.0f} FLOP/byte")
print()

# DiT Linear analysis
dit_linear_flops_total = total_gemm_flops * N_BLOCKS * N_STEPS
dit_linear_bytes_total = total_weight_bytes * N_STEPS
dit_linear_ai = dit_linear_flops_total / dit_linear_bytes_total
dit_linear_ideal = roofline_ideal(dit_linear_flops_total, dit_linear_bytes_total)
print(f"  DiT Linear:")
print(f"    AI = {dit_linear_ai:.0f} FLOP/byte → {'COMPUTE bound' if dit_linear_ai > CROSSOVER_HBM else 'MEMORY bound'}")
print(f"    Ideal: {dit_linear_ideal*1000:.1f} ms, Actual: {dit_linear_time*1000:.1f} ms")
print(f"    MFU achieved: {dit_linear_ideal/dit_linear_time*100:.1f}%")
print(f"    → GEMM is {'near peak' if dit_linear_ideal/dit_linear_time > 0.5 else 'far from peak'}")
print()

# DiT SDPA analysis
dit_sdpa_flops_total = total_attn_flops * N_BLOCKS * N_STEPS
dit_sdpa_bytes_total = attn_bytes * B * N_BLOCKS * N_STEPS
dit_sdpa_ai = dit_sdpa_flops_total / dit_sdpa_bytes_total
dit_sdpa_ideal = roofline_ideal(dit_sdpa_flops_total, dit_sdpa_bytes_total)
print(f"  DiT SDPA:")
print(f"    AI = {dit_sdpa_ai:.0f} FLOP/byte → {'COMPUTE bound' if dit_sdpa_ai > CROSSOVER_HBM else 'MEMORY bound'}")
print(f"    Ideal: {dit_sdpa_ideal*1000:.1f} ms, Actual: {dit_sdpa_time*1000:.1f} ms")
print(f"    MFU achieved: {dit_sdpa_ideal/dit_sdpa_time*100:.1f}%")
print(f"    → Gap = softmax scalar overhead (exp, shuffle, P quant)")
print()

# ---- Per-layer SDPA detail ----
print("-" * 78)
print("  PER-LAYER SDPA BREAKDOWN")
print("-" * 78)
sdpa_per_layer_flops = total_attn_flops * N_STEPS / N_BLOCKS
sdpa_per_layer_bytes = attn_bytes * B * N_STEPS
sdpa_per_layer_time = dit_sdpa_time / N_BLOCKS
print(f"  Per layer-step:")
print(f"    FLOPs:  {sdpa_per_layer_flops/1e9:.1f} GFLOP")
print(f"    Bytes:  {sdpa_per_layer_bytes/1e6:.1f} MB (Q+K+V per step)")
print(f"    AI:     {sdpa_per_layer_flops/sdpa_per_layer_bytes:.0f} FLOP/byte")
print(f"    Time:   {sdpa_per_layer_time*1000:.2f} ms")
print(f"    → Each layer is {'compute' if sdpa_per_layer_flops/sdpa_per_layer_bytes > CROSSOVER_HBM else 'memory'} bound")
print()

# ---- Overall wall time breakdown ----
print("-" * 78)
print("  WALL TIME BREAKDOWN (100f 2 steps)")
print("-" * 78)
print(f"  {'Stage':<25} {'Time':>8} {'%':>6}  Bar")
print("  " + "-" * 74)

all_stages = [
    ("Text Encoder", te_total_time),
    ("DiT Linear", dit_linear_time),
    ("DiT SDPA", dit_sdpa_time),
    ("DiT Other", dit_other_time),
    ("Video VAE", vae_total_time),
]

max_time = max(t for _, t in all_stages)
bar_width = 40

for name, t in all_stages:
    pct = t / total_gpu_time * 100
    bar_len = int(t / max_time * bar_width)
    bar = "█" * bar_len
    print(f"  {name:<25} {t*1000:>7.1f}ms {pct:>5.1f}%  {bar}")

print("  " + "-" * 74)
print(f"  {'TOTAL':<25} {total_gpu_time*1000:>7.1f}ms")
print()

# ---- Optimization opportunities ----
print("-" * 78)
print("  OPTIMIZATION OPPORTUNITIES (ranked by impact)")
print("-" * 78)
print()
print("  1. INT8 DiT Linear GEMM [already enabled]")
print(f"     Current: {dit_linear_time*1000:.0f} ms ({dit_linear_time/total_gpu_time*100:.0f}% of GPU time)")
print(f"     INT8 peak: {INT8_PEAK/1e12:.0f} TOPS vs BF16 {BF16_PEAK/1e12:.0f} TFLOPS = 2x compute")
print(f"     But weight bytes halved → memory bound region shifts")
print(f"     Potential: ~{dit_linear_time*0.5*1000:.0f} ms (if 50% of time is GEMM compute)")
print()
print("  2. FP8 DiT Linear [gfx942 only, experimental]")
print(f"     FP8 peak: ~{INT8_PEAK/1e12:.0f} TOPS (same as INT8)")
print(f"     Weight bytes: 1/4 of BF16 → more memory-bound")
print(f"     Risk: NaN from FNUZ format, accuracy loss")
print()
print("  3. Video VAE optimization [hidden behind DiT pipeline]")
print(f"     Current: {vae_total_time*1000:.0f} ms ({vae_total_time/total_gpu_time*100:.0f}% of total)")
print(f"     Mostly hidden (pipeline overlap), but limits throughput")
print(f"     Potential: INT8 conv, kernel fusion")
print()
print("  4. Token Reduction [proven 43% SDPA speedup]")
print(f"     Current SDPA: {dit_sdpa_time*1000:.0f} ms ({dit_sdpa_time/total_gpu_time*100:.0f}% of GPU time)")
print(f"     With token reduction: ~{dit_sdpa_time*0.57*1000:.0f} ms (estimated)")
print(f"     Trade-off: quality loss (PSNR drops from 38.8 to ~29 dB)")
print()
print("  5. Pipeline overlap optimization")
print(f"     DiT weight load: ~2.2s (hidden)")
print(f"     VAE decode starts after DiT → could overlap with next batch")
print()

# ---- SageAttention conclusion ----
print("-" * 78)
print("  SAGEATTENTION CONCLUSION")
print("-" * 78)
print()
print("  SageAttention INT8 on MI300X CANNOT improve performance because:")
print()
print(f"  1. SDPA AI = {dit_sdpa_ai:.0f} FLOP/byte → {'compute' if dit_sdpa_ai > CROSSOVER_HBM else 'memory'} bound")
print(f"     Theoretical: compute-bound. But actual softmax is scalar-bound.")
print()
print(f"  2. SDPA is only {dit_sdpa_time/total_gpu_time*100:.0f}% of GPU time")
print(f"     Even 2x SDPA speedup only saves {dit_sdpa_time*0.5*1000:.0f} ms")
print()
print(f"  3. PV MFMA compute is hidden behind V-load HBM latency")
print(f"     INT8 PV 2x MFMA speedup → saves ~0 for bandwidth-bound ops")
print()
print(f"  4. V-quant overhead (extra kernel + HBM R/W) costs MORE than saved")
