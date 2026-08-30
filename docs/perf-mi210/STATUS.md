# MI210 12h loop

Deadline: **2026-08-30 07:14 +08**. GPU 0 only. No MI300 work.

Baseline fox-s2 (pre-P1): E2E **241 s**, VAE GPU **170 s** (sdpa 95 + linear 73), denoise **59 s** (sdpa 39 + linear 18).

Halo v0.9.0 fox-s2: E2E **83–87 s**.

| When | KEEP/REJECT | Notes |
|------|-------------|-------|
| start | — | P1.1 hipBLAS SGEMM for large f32 linear |
| tick0 | **KEEP** hipBLAS f32 | VAE linear 72.8→**42.6 s**. E2E 241→**211 s**. |
| tick1 | incomplete | `H3_SDPA_D64_Q8=1` run died after VAE load; no KEEP. |
| tick2 | **KEEP** hipBLAS f32 SDPA | VAE sdpa 95→**14.6 s**. E2E 211→**130 s**. Tests + roundtrip ok. `H3_SDPA_HIPBLAS=0` opts out. |
| tick3 | **KEEP** hipBLAS bf16 SDPA | DiT sdpa 39.1→**0.84 s**. Denoise 59→**21.7 s**. E2E **92 s**. |

## Latest fox-s2 (`/tmp/h3-mi210/fox-s2-sdpa-bf16.mp4`)

| Stage | GPU op | sdpa | linear |
|-------|--------|------|--------|
| DiT Euler denoise | 20.7 s | 0.84 s | 18.5 s |
| Video VAE | 59.7 s | 14.7 s | 42.5 s |
| **E2E** | **92 s** | | |

Remaining wall: VAE f32 GEMM (~43 s) then DiT INT8 linear (~19 s). Opt out both SDPA/GEMM hipBLAS with `H3_SDPA_HIPBLAS=0` / `H3_F32_HIPBLAS=0`.
