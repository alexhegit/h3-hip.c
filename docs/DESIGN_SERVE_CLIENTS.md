# `--serve` clients — plan and progress

> Branch: `serve-clients`　Base: `r9700`　Status: open
>
> Decision input: [`perf-runs/R9700_COMFYUI_COMPARE_2026-11-09.md`](perf-runs/R9700_COMFYUI_COMPARE_2026-11-09.md).
> Daemon protocol: [`DESIGN_DHS.md`](DESIGN_DHS.md) Part II (v1alpha).
> Plugin stages: [`DESIGN_DHS.md`](DESIGN_DHS.md) §6.1 / §8.

## Why

On R9700, h3-hip is moderately faster than ComfyUI native MiniMax-H3
(denoise/E2E **1.39×/1.34×** at 864×480 · 20 steps after VAE BF16). The
stronger product reason for a ComfyUI node (and for DSH) is memory: native
ComfyUI needs a restart per job on 59 GiB RAM; `h3 --serve` reuses resident
weights (KI-007 fixed). Speed alone does not justify the node.

## Goal

Make `--serve` shippable to two clients that share one daemon contract:

1. **ComfyUI custom node** — installable package evolved from
   `comfyui-h3-hip-prototype` (fox-s2 stream already matched CLI).
2. **DSH plugin** — `dsh-plugin-h3-hip` M1 per `DESIGN_DHS.md`
   (`h3_generate` + `h3-prompt` + jobs text progress).

Neither client publishes until the daemon gate below is green.

## Non-goals (this branch)

- Kernel / VAE performance work (separate tracks).
- Replacing ComfyUI's native MiniMax-H3 graph with in-process DiT.
- DSH M2+ (Web progress card, profile SSE fields) until M1 lands.
- Cross-host `/v1/media` upload.

## Workstreams

Checkboxes are the live tracker; tick them in commits on this branch (or
note the landing PR / commit SHA in the table).

### A. Daemon readiness (this repo)

| # | Item | Status |
|---|---|---|
| A1 | **KI-005**: daemon ↔ CLI fox-s2 md5 gate (extend `halo-regression` and/or `tools/h3d_smoke.py --generate`) | [ ] |
| A2 | Document crash / quota / cancel drills expected before dropping "experimental" on `--serve` | [ ] |
| A3 | Expose client-needed options cleanly: FastH3 LoRA path, geometry presets used by the node, cancel + SSE progress already in v1alpha | [ ] |
| A4 | Mark KI-007 fixed on `r9700` in `KNOWN_ISSUES.md` (branch landed) | [ ] |

### B. ComfyUI node (separate repo; contract owned here)

| # | Item | Status |
|---|---|---|
| B1 | Promote prototype → installable custom node (`H3HipGenerate` → `SaveVideo`) | [ ] |
| B2 | Cover T2VA / FL2VA, steps/layers/reuse, FastH3, 864×480 and 1344×768 | [ ] |
| B3 | SSE + Comfy `ProgressBar`; interrupt → `/cancel` | [ ] |
| B4 | README: requires local `h3 --serve`, weight layout, vs native MiniMax-H3 (speed moderate, memory wins on small RAM) | [ ] |
| B5 | Drop stale prototype assumptions (compact-only JSON, single-threaded server) — fixed in h3d PR #2 | [ ] |

### C. DSH plugin M1 (separate repo `alexhegit/dsh-plugin-h3-hip`)

| # | Item | Status |
|---|---|---|
| C1 | Host bundle + `h3_generate` tool + `/v1/info` negotiation | [ ] |
| C2 | `h3-prompt` skill (three-part prompt norm) | [ ] |
| C3 | Map existing SSE → `ctx.jobs` text (phase / step / ETA / `cache_hint` / queue) | [ ] |
| C4 | Fake daemon skeleton for plugin CI without a GPU | [ ] |

## Sequence

1. **A1–A2** first — shared gate for both clients.
2. **B** and **C** in parallel after A1 is green; B can reuse the existing prototype.
3. Drop `--serve` "experimental" only when A1 + agreed drills in A2 pass on at least gfx1201 (and gfx1151 if halo-regression owns the gate).

## Exit criteria

- [ ] KI-005 gate fails the build/regression when daemon and CLI fox-s2 diverge.
- [ ] ComfyUI node repo installable; one documented workflow matches CLI fox-s2 md5 through the daemon.
- [ ] DSH plugin M1 can submit a job and show text progress against a live or fake daemon.
- [ ] This doc's workstream tables are fully checked or explicitly deferred with a link.

## References

- R9700 vs ComfyUI: [`perf-runs/R9700_COMFYUI_COMPARE_2026-11-09.md`](perf-runs/R9700_COMFYUI_COMPARE_2026-11-09.md)
- KI-005 / KI-007: [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)
- Protocol + DSH stages: [`DESIGN_DHS.md`](DESIGN_DHS.md)
- Smoke: `tools/h3d_smoke.py`
