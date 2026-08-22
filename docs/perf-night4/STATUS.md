# Night-4 perf loop — STATUS

Started: 2026-08-22 22:47 CST  
Budget: ~8 hours  
Stop after: 2026-08-23 06:47 CST  
Git start: `1e0368c` (fc1 128×64 keep)  
Baseline fox s2: denoise GPU **12.67s** (lin **7.48** · sdpa 4.55)  
Baseline fox-fast: denoise GPU **95.6s** (lin 58.0 · sdpa 32.9) — before this night’s INT8/Q4 keeps  
Loop: 45m one-shot (`AGENT_LOOP_WAKE_perf_night4`) until 06:47  
Do not retry: fc1 t128, fused SwiGLU dual-B, flash, F32 256×64/BK16/LDS double-buffer, mmap memcpy default, t128 skip-last-sync / hoist `scale_b`

## Scoreboard (fox s2)

| Version | denoise GPU | denoise linear | denoise sdpa |
|---------|------------:|---------------:|-------------:|
| Q3+fc1 128×64 (`1e0368c`) | 12.67 | 7.48 | 4.55 |
| INT8 k+=8 + Q4 (this night) | **12.15** | **7.28** | **4.23** |

Isolated confirm: `/tmp/h3-profile/fox-s2-night4-q4b.log` (ignore q4.log: clocks were ~775 MHz, linear 12.5s garbage).

## Decisions

- **KEEP** INT8 t128 / fc1 128×64 / grouped r64: pair two `sdot4` from one `uint64` LDS load (`k += 8`).
- **KEEP** t128 `__launch_bounds__(256, 2)` (microbench 19.5 ms, no fox regression).
- **KEEP** d128 SDPA **Q4** as default. Opt out `H3_SDPA_D128_Q3=1` / `Q2` / `Q1`. Microbench KV-HM 62.7 vs Q3 72.5 ms.
- **KEEP** skip whole-file `POSIX_FADV_SEQUENTIAL` unless `H3_PREAD_SERIAL=1` (striped 8-way pread). Load wall still noisy; not judged as a GPU keep.
- **REJECT** t128 skip last `__syncthreads` + hoist `scale_b` to regs (INT8 bench 20.1 → 57.7 ms). Reverted before this keep set.
- **REJECT** F32 r128 LDS `BK+4` + float4 LDS stores (K=8192 13.6 → 21.8 ms).
- **REJECT** t128 `__launch_bounds__(256, 3)` (no win vs `, 2`).

## Log

| Time | Action | Result |
|------|--------|--------|
| 22:47 | Start night-4 8h | Next: INT8 t128 |
| 22:54 | Confirm skip-sync revert | INT8 bench 19.9 ms restored |
| 23:00 | t128 `k+=8` uint64 LDS | fox s2 GPU 12.50 / lin 7.36 |
| 23:05 | same on fc1 128×64 | GPU 12.43 / lin 7.30 |
| 23:10 | grouped `k+=8` + t128 `launch_bounds(256,2)` | GPU 12.51 / lin 7.29 (noise) |
| 23:16 | d128 SDPA Q4 default | isolated GPU **12.15** / lin 7.28 / sdpa **4.23** |
| 23:20 | fox-fast re-time | denoise GPU **88.48** (lin 53.0 · sdpa 30.8) vs 95.58 |
| 23:25 | F32 LDS float4/`BK+4`; t128 bounds 3 | both reject; revert kernels |
