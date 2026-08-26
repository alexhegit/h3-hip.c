# Fox-fast ledger (HIP / gfx1151)

Canonical knobs: upstream tutorial §2, **without** `--show`.

```bash
./h3 --profile -d /path/to/MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 --frames 22 --steps 20 --layers 45 --reuse 2 \
  -o outputs/fox-fast.mp4
```

**Current (v0.9.0):** [`V0.9.0.md`](V0.9.0.md) and vs M5 Max [`VS_UPSTREAM.md`](VS_UPSTREAM.md).

---

## Run day-5 Q3 — 2026-08-22 ~16:25 CST

First HIP fox-fast. Working tree then used d128 Q3 wave SDPA (see
[`../perf-day5/COMPARISON.md`](../perf-day5/COMPARISON.md), frozen).

| Phase | wall | GPU (`op-classes`) | split |
|-------|-----:|-------------------:|-------|
| Qwen text encoder | 20.05 | 2.65 | linear 2.62 |
| H3 DiT **load** | 60.43 | 5.71 | I/O + quantize |
| H3 DiT **Euler denoise** (11 evals) | **104.99** | **95.58** | linear 58.03 · sdpa 32.86 · other 4.68 |
| audio VAE | 4.26 | 3.75 | conv 3.64 |
| video VAE | 21.94 | 17.66 | linear 10.65 · sdpa 6.52 |
| **E2E** (`/usr/bin/time`) | **212.68** | | user 84.7 · sys 194.2 |
| DiT peak live | 19.7 GiB | | |

Per DiT forward: denoise GPU **95.58 / 11 ≈ 8.69 s**. vs M5 Max denoise wall
16.69 s this run was **6.3×**; v0.9.0 is **2.1×** on denoise wall (34.63 s).
