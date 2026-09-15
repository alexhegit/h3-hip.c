# Strix Halo (gfx1151) Sol-Attn — 2026-09-15

Machine: AMD Ryzen AI MAX+ 395 / Radeon 8060S, `gfx1151`. Build:
`make HIP_ARCH=gfx1151`. Model:
`/home/amd/HF-MODELS/MiniMax-H3`. Sol-Attn uses the dedicated wave32
rocWMMA kernel, 64-token routing tiles, pooled skipped K/V, tau 0.5, band 1,
and the default 4096-token minimum.

## Correctness and microbench

The targeted keep-all test at sequence 512 reports zero bitwise differences
from dense. Its tau 0.5 companion reports nonzero differences, so a dense
fallback cannot satisfy the test accidentally.

| seq | dense | keep-all | tau 0.5 | tau 0.5 vs dense | keep-all diff |
|---:|---:|---:|---:|---:|---:|
| 1,874 | 6.9 ms | 8.2 ms | 5.9 ms | 1.16x | 0 |
| 8,192 | 114.1 ms | 130.1 ms | 61.7 ms | 1.85x | 0 |
| 16,384 | 440.1 ms | 481.9 ms | 215.9 ms | 2.04x | 0 |
| 44,800 | 3286.2 ms | 3776.2 ms | 1445.6 ms | **2.27x** | 0 |

Log:
[`gfx1151-2026-09-15-sol-attn-correctness.log`](gfx1151-2026-09-15-sol-attn-correctness.log)
and
[`gfx1151-2026-09-15-sol-attn-microbench.log`](gfx1151-2026-09-15-sol-attn-microbench.log).

## Fixed-seed 15 s no-TR A/B

Both runs use the same tree and binary, prompt/checkpoint, seed 42,
864x480, 362 frames, 20 steps, 45 layers, reuse 2, and no token reduction.

| metric | dense | Sol-Attn tau 0.5 | change |
|---|---:|---:|---:|
| E2E | 2457.04 s | 1786.72 s | **-27.3%** |
| denoise | 2170.15 s | 1506.12 s | **-30.6%** |
| denoise SDPA | 1583.50 s | 917.19 s | **-42.1%** |
| denoise linear | 553.88 s | 555.59 s | +0.3% |
| peak VRAM | 27.91 GiB | 27.91 GiB | unchanged |
| exact KV tiles | 100% | 33.45% | 66.55% pooled |
| video PSNR / SSIM | reference | 19.55 dB / 0.721 | preview-grade |
| decoded audio SNR | reference | 10.99 dB | approximate |

Logs:

- [`gfx1151-2026-09-15-fox-15s-sol-dense.log`](gfx1151-2026-09-15-fox-15s-sol-dense.log)
- [`gfx1151-2026-09-15-fox-15s-sol-tau05.log`](gfx1151-2026-09-15-fox-15s-sol-tau05.log)
- [`gfx1151-2026-09-15-fox-15s-sol-quality.log`](gfx1151-2026-09-15-fox-15s-sol-quality.log)

## Default-path regression

`H3_SOL_ATTN_MIN_SEQ=4096` keeps fox-s2 (about 1.9k tokens) dense. A manual
feature-off fox-s2 run and a pristine `origin/sol-attn` parent build both
produced MP4 md5 `26ab2980c2fe10bd6d42483e7a902c36`, confirming this change
does not alter the default path.

The aggregate `halo-regression` target now passes `h3_hip_bf16_tests` after
the short-row fused INT8 AdaLN max reduce (`c351f23`). Feature-off fox-s2
md5 remains `26ab2980c2fe10bd6d42483e7a902c36` (script default is still the
v0.9.0 hash; override with `H3_FOX_S2_MD5`). Log:
[`gfx1151-2026-09-16-halo-regression.log`](gfx1151-2026-09-16-halo-regression.log).

## Decision

KEEP as opt-in lossy acceleration on gfx1151. The kernel clears the 2x
44,800-token and 20% denoise-SDPA gates. Quality remains below the default-on
bar, so dense stays the default and Sol-Attn should not be combined with token
reduction by default.
