# Strix Halo (gfx1151) — H3-OnDevice 480p / 5 s baseline

2026-10-05. Same machine as
[`HALO_2026-09-12.md`](HALO_2026-09-12.md) (Ryzen AI MAX+ 395 / Radeon 8060S,
`gfx1151`, `192.168.1.43`), after the BIOS VRAM carveout change.
`mem_info_vram_total` **32.00 GiB**, host `MemTotal` **94.06 GiB**,
`mem_info_gtt_total` **47.03 GiB**. Swap still on (`/swap.img` 8G, used 0,
`vm.swappiness=60`). NVMe readahead still `read_ahead_kb=128`. Scheduler
`none`. This is **not** the 2026-09-12 `bench/` scoreboard.

Tree: branch `gfx1151` @ `fb2e603`. Build: `make HIP_ARCH=gfx1151`.
Model: `/home/amd/HF-MODELS/MiniMax-H3`. No `H3_*` environment overrides, so
DiT stays the default INT8 path. Seed **42**.

Workload matches the DGX Spark column on
[H3-OnDevice](https://nvlabs.github.io/Sana/Sol-Engine/H3-OnDevice/):
**832×480**, 24 fps, **124 frames**, **5.167 s**, **50 steps**. Command:

```bash
./h3 --profile -d /home/amd/HF-MODELS/MiniMax-H3 \
  -p 'Cinematic push-in on a starship bridge: a captain in a high-collared navy tunic stands silhouetted at the observation window as a dreadnought armada charges its hyperdrives. A blinding flash, the bridge shudders, and the fleet is gone.' \
  --width 832 --height 480 --seconds 5 \
  --steps 50 --layers 50 --reuse 1 --seed 42 \
  -o outputs/halo-ondevice-p1-baseline.mp4
```

`--seconds 5` is 120 requested frames and aligns to **124**. The page prints
that sentence with a “[full prompt compacted]” note and does not publish the
rest. Spark’s published baseline is a **warm** unmodified BF16 reference
(**710.6 s**). This run is one cold-ish process on the current HIP defaults
(INT8, existing fusions, dense SDPA, reuse 1) and includes weight load.

## Dense baseline

| phase | wall |
|---|---:|
| Qwen text encoder | 10.640 s |
| DiT load | 17.458 s |
| Euler denoise | **3357.873 s** |
| denoise SDPA / linear / other | **1494.108 / 1794.197 / 68.920 s** |
| audio VAE | 2.428 s |
| video VAE | 87.577 s |
| process E2E (`/usr/bin/time`) | **3477.57 s** (57 min 58 s) |

Peak live tensors **22.657 GiB**. Output
`outputs/halo-ondevice-p1-baseline.mp4`: 832×480, 124 frames, 24 fps,
duration 5.167 s, stereo 32 kHz. md5 `eb8180f74420567031bb4ac64e4b5614`.
NVMe `nvme0n1` sectors-read delta **249353736** (**118.9 GiB**). Denoise is
**96.6%** of E2E. Load is about 28 s, so the gap versus Spark 710.6 s
(**4.9×**, 3477.57 / 710.6) is compute.

Log:
[`gfx1151-2026-10-05-ondevice-p1-baseline.log`](gfx1151-2026-10-05-ondevice-p1-baseline.log).

## Same-fixture comparisons

Every row is the command above on this machine. Quality is FFmpeg `psnr` /
`ssim` on the muxed H.264 streams and float32 PCM SNR versus
`outputs/halo-ondevice-p1-baseline.mp4`. Speed ratios are versus that dense
row. Lossy flags are not stacked: Sol-Attn, `--fbc`, and `--token-reduction`
are separate arms. `--fbc` still uses `--reuse 1`.

| path | E2E | denoise | sdpa / linear | video PSNR / SSIM | audio SNR |
|---|---:|---:|---|---|---:|
| dense | **3477.57 s** | 3357.873 s | 1494.108 / 1794.197 s | reference | reference |
| Sol-Attn τ=0.5, blocks 4:40 | **3252.45 s** (1.07×) | 3134.158 s | 976.775 / 2084.212 s | 13.92 dB / 0.615 | 6.57 dB |
| Sol-Attn τ=1, blocks 2:50 | **2335.83 s** (1.49×) | 2218.648 s | 420.180 / 1730.608 s | 13.85 dB / 0.602 | 5.18 dB |
| `--fbc` | **697.78 s** (4.98×) | 588.978 s | 256.156 / 315.306 s | 16.68 dB / 0.707 | 3.87 dB |
| `--token-reduction` | **2178.14 s** (1.60×) | 2065.080 s | 806.598 / 1207.618 s | 14.90 dB / 0.624 | −2.85 dB |

τ=1 blocks 2:50 keeps layers 0–1 dense (`index >= 2 && index < 50`). Exact KV
tiles: τ=0.5 kept 34.63%, τ=1 kept 21.06%. Both are well below the 15 s Halo
τ=0.5 result (19.55 dB / audio 10.99 dB). After τ=1, denoise time left in
linear is 1731 s. Denoise “other” was 72.360 s (τ=0.5) and 67.225 s (τ=1),
next to the dense 68.920 s.

`--fbc` always runs the first active block and skips the other 49 when the
accumulated relative L2 of that residual stays under `H3_FBC_REL` (default
**0.15**). Head and tail windows (`H3_FBC_HEAD` / `H3_FBC_TAIL`, default
**2**) always run the full DiT. This run was **9 full / 41 skip** (full
steps 1–2, 11, 22, 33, 41, 46, 49–50). Attention calls 2500 → 491
(`9×50 + 41×1`). Denoise other 16.976 s. Peak **23.254 GiB**. Per-step
relative L2 was about 0.014–0.026 through the middle and 0.08–0.11 at the
end. Audio err rms 0.0685 on 331776 samples (ref rms 0.10687939). Video
PSNR/SSIM sit above both Sol-Attn arms; the soundtrack sits below them.

Spark’s published 480p ladder reaches **181.3 s** only after kernels +
Sol-Attn τ=1 + first-block cache (3.92× versus their 710.6 s BF16 baseline).
The `--fbc` row is that cache alone. 697.78 s lands next to Spark’s dense
baseline, not next to their stacked 181.3 s.

`--token-reduction` is the speed knob inside the older Halo all-opts bundle
(`H3_TOKEN_REDUCTION=1` with `H3_GPU_SAMPLER=1` and `H3_INT8_VAE=1`). Sampler
and INT8 VAE are not in this row: on the 2026-09-03 15 s Halo run, sampler
matched the no-sampler TR time to the second, and INT8 VAE only changes the
video-VAE slice. Default range is blocks **4:30**, with the first **10**
steps deepened to **4:40**. Attention calls stay **2500**. Denoise other
50.157 s. Peak **22.660 GiB**. Video PSNR y/u/v 13.184 / 31.429 / 29.802
(average 14.905, min 11.637, max 17.250). Audio SNR **−2.85 dB** on the same
331776 samples (ref rms 0.10687939, err rms 0.14837007): the soundtrack
diverges by more energy than the dense reference itself. On this fixture TR
is slower than `--fbc` (2178 s versus 698 s) and farther from the dense clip
on both picture and audio.

Outputs:

- τ=1, blocks 2:50: `outputs/halo-ondevice-p1-sol-tau1-b2-50.mp4`, md5 `1165fec5abcfa371f7eab8ee56e48f81`
- τ=0.5, blocks 4:40: `outputs/halo-ondevice-p1-sol-tau05-b4-40.mp4`, md5 `7c5f1247b10f2f46de607e6033d2b8a6`
- `--fbc`: `outputs/halo-ondevice-p1-fbc.mp4`, md5 `177df2a1d01674e8706bd383a6fc2782`
- `--token-reduction`: `outputs/halo-ondevice-p1-tr.mp4`, md5 `e9a5e4c2544cffc6171250a72c679092`

Logs:

- [`gfx1151-2026-10-05-ondevice-p1-sol-tau1-b2-50.log`](gfx1151-2026-10-05-ondevice-p1-sol-tau1-b2-50.log)
- [`gfx1151-2026-10-05-ondevice-p1-sol-tau05-b4-40.log`](gfx1151-2026-10-05-ondevice-p1-sol-tau05-b4-40.log)
- [`gfx1151-2026-10-05-ondevice-p1-fbc.log`](gfx1151-2026-10-05-ondevice-p1-fbc.log)
- [`gfx1151-2026-10-05-ondevice-p1-tr.log`](gfx1151-2026-10-05-ondevice-p1-tr.log)
