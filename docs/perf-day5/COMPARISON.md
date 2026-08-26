# fox 性能演进对比

**Frozen at day-5 Q3 (2026-08-22).** Later WMMA, hipBLAS, and device-resident
weights are in [`../perf/HISTORY.md`](../perf/HISTORY.md) and
[`../perf-runs/V0.9.0.md`](../perf-runs/V0.9.0.md). Do not treat the “当前 Q3”
row as HEAD.

预设：`fox steps=2 layers=35`，512×512、22 帧，MiniMax-H3 on gfx1151。

E2E 与 DiT load 受文件系统缓存影响很大（标 †），判断优化效果请看 `--profile` 的
`op-classes` GPU 时间，不要只看单次 E2E。

## 各保留版本对比（秒）

| 版本 | 保留内容 | E2E | DiT load | denoise GPU | den. sdpa | den. linear | VAE GPU | VAE sdpa | VAE linear |
|------|----------|----:|---------:|------------:|----------:|------------:|--------:|---------:|-----------:|
| 08-18 baseline A | 起点 | ~257† | 114 | 40.3 | 18.7 | 17.2 | 53.6 | 24.9 | 28.2 |
| overnight | wave grid、INT8 BK=128、VAE d64 4-key、4 路 pread | — | 40–54† | ~17.3 | ~8.0 | ~8.6 | ~25 | ~13 | ~11 |
| Run B `dcb4858` | day-2 基线 | 117.3 | 40.6 | 17.4 | 8.1 | 8.6 | 25.1 | 13.3 | ~11.8 |
| day-2 best | 8 路 pread、packed stores | 105.9 | 32.9 | 17.3 | 8.1 | 8.6 | 25.0 | ~13.4 | ~11.5 |
| night-3 warm `04652e6` | 延迟量化 flush、VAE d64 Q2+4key | 104.3 | 33.7 | 17.2 | ~8.1 | ~8.5 | 18.5 | 6.5 | 11.5 |
| F32 r128 `662e268` | F32 GEMM 128×128 / 8×8 | 112.2† | 36.6† | 17.1 | ~8.1 | ~8.5 | 17.4 | 6.4 | 10.5 |
| INT8 t128 `530480d` | INT8 128×128 sudot4（不含 fc1） | 104.6† | 41.2† | 16.7 | ~8.1 | 8.0 | 17.5 | 6.5 | 10.5 |
| d128 Q2 `404f4b3` | DiT 2-query wave SDPA | 115.5† | 51.4† | 13.8 | 5.3 | ~8.0 | 17.5 | 6.5 | 10.5 |
| **d128 Q3（当前工作树，未 commit）** | DiT 3-query wave SDPA | 96.1† | 34.4† | **13.1** | **4.5** | ~8.0 | 17.6 | 6.5 | 10.7 |

## 累计收益（baseline A → 当前 Q3）

| 指标 | baseline A | 当前 | 变化 |
|------|-----------:|-----:|-----:|
| denoise GPU | 40.3 | 13.1 | −67% |
| denoise sdpa | 18.7 | 4.5 | −76% |
| denoise linear | 17.2 | ~8.0 | −53% |
| VAE GPU | 53.6 | 17.6 | −67% |
| VAE sdpa | 24.9 | 6.5 | −74% |
| VAE linear | 28.2 | 10.7 | −62% |

现在 denoise 和 VAE 两侧都是 **linear 比 sdpa 更热**。

## 上游 fox-fast（20 steps / 45 layers / reuse 2）

与 [antirez/h3.c](https://github.com/antirez/h3.c) 教程 §2 同一组 knobs。
完整记录：[`../perf-runs/FOX_FAST.md`](../perf-runs/FOX_FAST.md)（day-5 跑）。
当前对照：[`../perf-runs/VS_UPSTREAM.md`](../perf-runs/VS_UPSTREAM.md)。HIP 跑于 2026-08-22，Q3 工作树。

| | M5 Max Metal | gfx1151 HIP |
|--|-------------:|------------:|
| Denoise wall | **16.69 s**（README，无 token-reduction） | **105.0 s**（6.3×） |
| Denoise GPU | 未公布 | 95.6 s（linear 58.0 · sdpa 32.9） |
| 每 DiT forward（11 次） | ~1.52 s | ~8.69 s GPU |
| Process E2E | 未公布 | **212.7 s** |

## 当前剩余热点

1. VAE F32 linear ~10.7s（K=8192 尾巴）
2. denoise INT8 linear ~8.0s（fc1 仍走 r64）
3. DiT load I/O，仍缺可信 A/B

## 已否决，不要再作为默认重试

| 实验 | 数据 | 原因 |
|------|------|------|
| flash 多 Q SDPA 默认 | gfx1151 DiT 形状 | 慢于 wave |
| d128 2-key / 4-key | 4-key bench 109→120 ms | 不如 1-key wave / Q2 |
| Q head-major fuse | fox | 无收益 |
| F32 BK=64、K≥8192 用 BK=16、256×64 | 13.8–14.2 → 16.3 ms | r128 BK=32 更好 |
| F32 LDS 双缓冲 | K=8192 14.4→16.5 ms | LDS 额外开销 |
| VAE fc1+SwiGLU 融合（r64 与 r128 dual-B） | VAE linear 10.5→11.2s | 双 B tile 输给未融合 r128 |
| INT8 fc1 t128（BK=128 / BK=32） | BK=32 时 linear 7.9→10.2s | fc1 保持 r64 |
| INT8 LDS 双缓冲 | 旧 microbench | 否决 |
| VAE d64 8-key | day-2 | 不如 4-key |
| mmap+memcpy 默认 | load 129s | 仅 `H3_WEIGHT_MMAP=1` 可选 |
| quant batch=8 | day-2 | 不如默认 4 |

## 回退开关

| 开关 | 作用 |
|------|------|
| `H3_SDPA_D128_Q2=1` | DiT SDPA 退回 2-query |
| `H3_SDPA_D128_Q1=1` | DiT SDPA 退回 1-query |
| `H3_F32_R64=1` | F32 GEMM 退回 64 tile |
| `H3_INT8_R64=1` | INT8 linear 退回 64 tile |
| `H3_PREAD_SERIAL=1` | 关闭并行 pread |
| `H3_WEIGHT_MMAP=1` | 启用 mmap+memcpy 权重加载 |
