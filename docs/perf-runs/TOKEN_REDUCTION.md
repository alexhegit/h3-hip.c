# gfx1151 15 s cinematic + `--token-reduction`

Measured 2026-09-02 on Strix Halo (`gfx1151`), branch `token-reduction` @
`903a7f0`, binary `h3-hip 0.10.1` rebuilt `HIP_ARCH=gfx1151`. Official
MiniMax-H3 at `/home/amd/HF-MODELS/MiniMax-H3`.

Same prompt / knobs / seed as [`LONG_VIDEO.md`](LONG_VIDEO.md) (v0.9.0 quality
path, 45.0 min E2E), plus `--token-reduction` (off by default; visible quality
trade). Not a fox-s2 md5 path.

Log: [`long-15s-tr-2026-09-02.log`](long-15s-tr-2026-09-02.log)  
Output: `/tmp/halo-long/long-15s-tr.mp4` (864×480, 362 frames, 15.083 s,
md5 `dfe2188f426f1c7c925373508062b3d4`).

```bash
./h3 --profile -d /home/amd/HF-MODELS/MiniMax-H3 \
  -p "<15 s cinematic prompt>" \
  --width 864 --height 480 --seconds 15 \
  --steps 20 --layers 45 --reuse 2 --seed 42 \
  --token-reduction \
  -o /tmp/halo-long/long-15s-tr.mp4
```

## Wall clock

| Stage | no TR (v0.9.0 quality path) | **`--token-reduction`** | Δ |
|-------|----------------------------:|------------------------:|--:|
| Qwen text encoder | 21.6 s | 32.5 s | (I/O band) |
| DiT load | 41.6 s | 45.4 s | (I/O band) |
| GPU Euler denoise | **2423.0 s** (40.4 min) | **1400.9 s** (23.3 min) | **−42%** |
| ├ gpu-op sdpa | 1672 s | **965.5 s** | **−42%** |
| ├ gpu-op linear | 574 s | 405.5 s | −29% |
| audio VAE | 5.6 s | 3.8 s | |
| video VAE | 207.1 s | 209.3 s | (unchanged) |
| **E2E** (`TIME_E2E`) | **2701.4 s** (45.0 min) | **1694.2 s** (28.2 min) | **−37%** |

11 DiT evaluations, 495 attention dispatches. Gate-ranked skips 4, 14, 16, 17, 13.
Denoise peak live **25.658 GiB** (96 GiB VRAM carveout is not tight).

VAE wall is not the TR win. Long-N SDPA is. Text-encoder / DiT-load walls move
with page cache on this box; do not ratio them.

## vs gfx90a (same flag, same clip)

MI210 ledger: [`../perf-mi210/TOKEN_REDUCTION.md`](../perf-mi210/TOKEN_REDUCTION.md).

| | gfx1151 + TR | gfx90a + TR | Halo / MI210 |
|--|-------------:|------------:|-------------:|
| E2E | 1694 s | 501 s | **3.38×** |
| denoise wall | 1401 s | 410 s | **3.42×** |
| gpu-op sdpa | 965 s | 285 s | **3.39×** |
| gpu-op linear | 406 s | 114 s | **3.56×** |

Relative TR savings vs each GPU's quality path sit in the same band (Halo E2E
−37%, MI210 −33.5%, Spark −38%). Cross-GPU gap with TR on is still denoise, not
NVMe.

## vs Spark

Spark 15 s + TR: 1124 s → 695 s E2E (−38%). HIP gfx1151: 2701 s → 1694 s (−37%).
Same knob, different GPU; the *relative* cut matches. Absolute Halo wall remains
I/O-plus-iGPU.
