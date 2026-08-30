# MI210 12h loop (device activations everywhere)

GPU 0 only. No MI300. Baseline after previous commit: fox-fast **45 s**, fox-s2 **17 s**.

## KEEP

Text / AdaLN / audio / HIP temps onto HBM (same bug as VAE/DiT). fox-fast vs `fox-fast-on-devact.mp4`: **PSNR inf**.

| | Before this loop | After |
|--|------------------|-------|
| fox-fast E2E | 45 s | **23 s** |
| fox-s2 E2E | 17 s | **12 s** |
| Text encoder GPU | 6.0 s | **0.7 s** |
| AdaLN precompute | 4.7 s | **1.4 s** |
| Denoise wall / GPU | 24.7 / 17.5 s | **12.4 / 12.2 s** |
| Audio VAE | 3.5 s | **0.6 s** |
| Video VAE | 2.7 s | 2.7 s (already device) |

AdaLN submit batching (4 GEMMs per sync) did not move wall; load-bound. Left in as fewer round-trips.

## Still open

fox-s2 wall is serial-ish: text I/O ~2.6 s + DiT load ~3 s + denoise 1.9 s + video VAE 2.7 s. Stretch 10 s needs I/O overlap, not kernels. fox-fast leftover is denoise linear 7.2 s + sdpa 4.4 s.

## Repro

```bash
H3_HIP_DEVICE=0 ./h3 --profile -d /home/alex/data/HF-MODELS/MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 --frames 22 --steps 20 --layers 45 --reuse 2 \
  -o /tmp/h3-mi210/fox-fast-p1.mp4

H3_HIP_DEVICE=0 ./h3 --profile -d /home/alex/data/HF-MODELS/MiniMax-H3 \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 --steps 2 --layers 35 --reuse 1 \
  -o /tmp/h3-mi210/fox-s2-p1.mp4
```
