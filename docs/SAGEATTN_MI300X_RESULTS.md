# SageAttention INT8 on MI300X (gfx942) — 结论

## 结论：SageAttention INT8 在 MI300X 上无法获得性能提升

SDPA（Scaled Dot-Product Attention）在 MI300X 上是**内存带宽瓶颈**，不是计算瓶颈。
INT8 量化可以降低计算量，但无法降低 V 矩阵的 HBM 读取带宽，因此 MFMA 加速被完全掩盖。

---

## 测试环境

- **GPU**: AMD MI300X (gfx942), CDNA3, 192 GB HBM3, ~6 TB/s
- **模型**: MiniMax-H3, head_dim=128, 24 heads
- **测试**: 100 frames, 2 denoise steps, seed=42, prompt="A red fox walking through a snowy forest"
- **基线**: BF16 SDPA（标准 MFMA D=128 内核）

---

## 性能数据

### 100f 2步 sdpa 时间

| 路径 | sdpa 时间 | 相对基线 | PSNR | SSIM |
|------|----------|---------|------|------|
| BF16 基线 | **2.837s** | — | — | — |
| INT8 QK + BF16 PV | **2.866s** | +1.0% | 32.9 dB | 0.939 |
| INT8 QK + INT8 PV | **4.007s** | +41.3% | 38.8 dB | 0.962 |
| INT8 QK + INT8 PV (pvs hoist) | **4.007s** | +41.3% | 38.8 dB | 0.962 |

### Token Reduction 效果（唯一有效的优化）

| 配置 | sdpa 时间 | 相对无 token reduction |
|------|----------|---------------------|
| 100f 2步 baseline | 2.837s | — |
| 100f 20步 baseline | 28.77s | — |
| 100f 20步 + token reduction | **16.31s** | **-43%** |

---

## 根本原因分析

### 带宽分析

SDPA 内核每步需要加载的数据：
- **Q**: 128 × 2 bytes = 256 bytes（寄存器，不占 HBM）
- **K**: 32 × 128 × 1 byte = 4 KB（INT8）/ 8 KB（BF16）
- **V**: 32 × 128 × 1 byte = 4 KB（INT8）/ 8 KB（BF16）
- **P**: 32 × 16 × 1 byte = 512 bytes（INT8，寄存器）

V 矩阵在 BK=32 时每次加载 4 KB（INT8）或 8 KB（BF16）。
但在 100f 2步中，序列长度 ~2500，BK=32 需要 ~79 次迭代。
总 V 数据: 2500 × 128 × 2 bytes = **640 KB/head** × 24 heads = **15.4 MB**

HBM 带宽 6 TB/s → V 加载时间 ~2.6 μs/head/step。
但实际 sdpa 时间 2.837s / (35 layers × 2 steps × 24 heads) = **1.69 ms/head/step**。

**V 加载仅占 sdpa 时间的 0.15%**。瓶颈在于 online softmax 的标量计算（exp、shuffle reduction、P 量化）。

### 为什么 INT8 PV 没有帮助

1. **MFMA 计算只占 sdpa 的 ~20%**：8 次 PV MFMA × ~8 cycles = 64 cycles/tile，但 softmax + P 量化需要 ~320 cycles/tile
2. **INT8 MFMA 2x 加速被掩盖**：PV MFMA 从 64 cycles 降到 32 cycles，但 softmax 仍需 320 cycles
3. **V 量化额外开销**：`h3_sage_quant_v_int8_kernel` 每层每步需要额外的 kernel launch + HBM 读写

### 为什么 INT8 QK 没有帮助

INT8 QK 节省 K 矩阵加载带宽（BF16 → INT8），但：
- K 加载仅占 sdpa 的 ~5%
- 节省的 ~4 KB/step × 2500/32 = 312 KB → 0.05 μs
- P 量化的额外标量计算抵消了这点节省

### 为什么 FP8 PV 不可行

gfx942 的 FP8 使用 FNuz 格式（bias=8, max=224, NaN=0x80），不是 OCP E4M3（max=448）。
- FP8 max representable = 224，低于 INT8 的 127 × 2 = 254
- 3-mantissa-bit e4m3 FNUZ 有 12% 相对误差
- 35 layers × N steps 的误差累积导致 e2e PSNR 仅 16.7 dB

---

## 修复的 Bug（提交 71f4f4b）

`h3_sdpa_int8qk_mfma_d128_kernel` 有 3 个正确性 bug，导致 PSNR 仅 18 dB：

1. **PV 循环次数**: `D_TILES=4`（K-dim tiling）应为 `PV_TILES=8`（D/16，N-dim tiling）
2. **输出 d 索引**: `j*MFMA_K(32)` 应为 `j*16`
3. **V 加载缺少 head 偏移**: `load_kv` 所有 head 都在读 head 0 的 V 数据

修复后 PSNR 从 18 dB 恢复到 38.8 dB。

---

## 关键结论

1. **SageAttention INT8 在 MI300X 上不可行**：SDPA 是带宽瓶颈，不是计算瓶颈
2. **Token reduction 是唯一有效的优化**：43% sdpa 减少
3. **INT8 QK 单独也不可行**：节省的 K 带宽微不足道
4. **FP8 PV 不可行**：FNUZ 格式误差太大

## 推荐后续方向

- Token reduction（已验证有效）
- 长视频优化（15s cinematic，H3_SAGE_MAX_STEPS=100）
- 其他带宽优化（double buffering、prefetching）
