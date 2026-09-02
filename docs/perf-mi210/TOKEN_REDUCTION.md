# gfx90a 15 s cinematic + `--token-reduction`

Measured 2026-09-02 on MI210 GPU 2 (`H3_HIP_DEVICE=2`), branch
`token-reduction`, binary `h3-hip 0.10.1` rebuilt `HIP_ARCH=gfx90a`.
Official MiniMax-H3 at `/home/alex/data/HF-MODELS/MiniMax-H3`.

Same prompt / knobs / seed as the then-tagged 12 min 33 s run, plus
`--token-reduction` (off by default; visible quality trade). Not a fox-s2
md5 path. On later `main` (2026-09-02, 480 px VAE tiles) the quality-path
15 s is **12 min 11 s**; all-opts / CLI TR is still **8 min 21 s**. This
ledger is the pre-tile A/B (753.29 → 500.81 s).

Log: [`long-15s-tr-2026-09-02.log`](long-15s-tr-2026-09-02.log)  
Output: `/tmp/h3-mi210/long-15s-tr.mp4` (864×480, 362 frames, 15.084 s,
md5 `ede8e1c58f051f92a0cc44fb3682a95d`).

```bash
H3_HIP_DEVICE=2 ./h3 --profile -d /home/alex/data/HF-MODELS/MiniMax-H3 \
  -p "<15 s cinematic prompt>" \
  --width 864 --height 480 --seconds 15 \
  --steps 20 --layers 45 --reuse 2 --seed 42 \
  --token-reduction \
  -o /tmp/h3-mi210/long-15s-tr.mp4
```

## Wall clock

| Stage | no TR (CDNA flash, tagged) | **`--token-reduction`** | Δ |
|-------|---------------------------:|------------------------:|--:|
| Qwen text encoder | — | 2.613 s | |
| DiT load | — | 3.451 s | |
| GPU Euler denoise | **648.29 s** (10 min 48 s) | **409.81 s** (6 min 50 s) | **−37%** |
| ├ gpu-op sdpa | 474.18 s | **284.57 s** | **−40%** |
| ├ gpu-op linear | 161.23 s | 114.45 s | −29% |
| audio VAE | — | 2.592 s | |
| video VAE | 92.75 s | 78.94 s | (I/O band) |
| **E2E** (`TIME_E2E`) | **753.29 s** (12 min 33 s) | **500.81 s** (8 min 21 s) | **−33.5%** |

11 DiT evaluations, 495 attention dispatches. Gate-ranked skips 4, 14, 16, 17, 13.
Denoise peak live 41.195 GiB.

VAE wall is not the TR win. Long-N SDPA is.

## vs Spark

Spark 15 s + TR: 1124 s → 695 s E2E (−38%). HIP gfx90a: 753 s → 501 s (−33.5%).
Same knob, different GPU; ratios are in the same band.

## gfx1151

Same clip on Strix Halo (2026-09-02): E2E **1694 s (28.2 min)** vs 2701 s
without the flag (−37%); denoise 1401 s vs 2423 s (−42%). Ledger:
[`../perf-runs/TOKEN_REDUCTION.md`](../perf-runs/TOKEN_REDUCTION.md).
