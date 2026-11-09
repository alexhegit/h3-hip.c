# ComfyUI through `--serve` — plan and progress

> Branch: `serve-clients`　Base: `r9700`　Status: open
>
> Decision input: [`perf-runs/R9700_COMFYUI_COMPARE_2026-11-09.md`](perf-runs/R9700_COMFYUI_COMPARE_2026-11-09.md).
> Daemon protocol: [`DESIGN_DHS.md`](DESIGN_DHS.md) Part II (v1alpha).

## Why

On the R9700, h3-hip is moderately faster than ComfyUI native MiniMax-H3
(denoise/E2E **1.39×/1.34×** at 864×480 · 20 steps). The stronger reason
for a ComfyUI path is memory: native ComfyUI needs a restart per job on
59 GiB of RAM, while `h3 --serve` reuses resident weights. Speed alone
does not justify the work.

## Goal

A ComfyUI user on an AMD GPU can follow one tutorial and render MiniMax-H3
through a ComfyUI node backed by a local `h3 --serve`.

Everything lives in this repository: daemon changes, the node under
`integrations/comfyui/`, and the tutorial under `docs/wiki/`. Nothing is
submitted to ComfyUI itself. Development happens here (private); users get
it from the public [`alexhegit/h3-hip.c`](https://github.com/alexhegit/h3-hip.c)
after a release sync.

## Non-goals

- Kernel / VAE performance work.
- Running the DiT inside ComfyUI or replacing its native MiniMax-H3 graph.
- Loading Comfy-Org's repackaged `int8_convrot` weights (users download the
  official checkpoint; see Known limits).
- DSH plugin (`dsh-plugin-h3-hip`), deferred until the ComfyUI path ships.
- The drills needed to drop "experimental" from `--serve`; the node ships
  as experimental too.
- Cross-host `/v1/media` upload.

## Workstreams

Tick the boxes in commits on this branch, or note the landing PR / SHA.

### A. Daemon

| # | Item | Status |
|---|---|---|
| A1 | **KI-005 gate.** `tools/h3d_smoke.py --generate` checks fox-s2 against a hard-coded gfx1201 md5 (`6d17f81a…`; gfx1151 is `1731f95c…`). Compare the daemon output with a CLI run from the same build instead, and wire it into a make target | [x] `make serve-regression`; gfx1201 pass (miss + hit = CLI `6d17f81a…`). Other ISAs not run |
| A2 | **Release VRAM when idle.** The worker always enables the generation cache and has no unload, so the daemon keeps the prepared DiT on the card between jobs and competes with ComfyUI's own models on a 32 GiB card. Add an unload endpoint and/or an idle timeout that frees the cache, reported in `/v1/info` | [x] `POST /v1/cache/release`, `options.release_after`, `H3D_IDLE_RELEASE_SEC`, `workers[].cache` in `/v1/info` (`DESIGN_DHS.md` §13.7). 864×480 FastH3: 9.5 GiB VRAM / 19 GiB RSS → 0.55 / 0.89 GiB |
| A3 | **FastH3 without server paths.** Today a client must pass `--fasth3-lora PATH` through `extra_args`. Resolve the adapter on the daemon side (default download location or `H3D_*` setting) behind a named option, and fold this into the public `DESIGN_DHS_PATHS.md` plan ([h3-hip.c PR #12](https://github.com/alexhegit/h3-hip.c/pull/12)) instead of keeping two plans | [x] `options.fasth3` + `H3D_FASTH3_LORA` + `adapters.fasth3` in `/v1/info`; 512×512 FastH3 through the daemon = CLI `--fasth3-lora` (`388f2138…`). Folding into PR #12 waits for the public sync (D1) |
| A4 | KI-007 status in `KNOWN_ISSUES.md` points at `r9700` | [x] |

### B. ComfyUI node (`integrations/comfyui/`)

| # | Item | Status |
|---|---|---|
| B1 | Move the node from `comfyui-h3-hip-prototype` into the tree; the benchmark scripts stay out | [x] rewritten as `integrations/comfyui/__init__.py` |
| B2 | Remove assumptions PR #2 made obsolete (compact-only JSON, one connection at a time) | [x] also fixes keyframe conversion (rounds instead of truncating, so a `LoadImage` PNG reaches the daemon unchanged) |
| B3 | Cover T2VA / FL2VA, steps / layers / reuse, FastH3 (via A3), 864×480 and 1344×768 | [x] fox-s2 T2VA and FL2VA + FastH3 through ComfyUI have the CLI's stream hashes; 864×480 rendered through the node; 1344×768 accepted, not rendered through the node |
| B4 | Poll the job once a second into a ComfyUI `ProgressBar`; interrupt sends `/cancel`. No SSE client | [x] Interrupt during 864×480 dense denoise frees the daemon in 12 s, after the denoise-cancel fix (`c4b104b`); it used to run the remaining steps |
| B5 | Ship one example API workflow (`H3HipGenerate` → `SaveVideo`) | [x] `workflows/h3hip_t2va_api.json`, `h3hip_fl2va_api.json`; `run_workflow.py` runs them headless |
| B6 | State the minimum ComfyUI version (the node uses the V3 `comfy_api.latest` API; tested on `fca60fe`) | [x] ComfyUI 0.44.0 (`fca60fe`) in the tutorial and README |
| B7 | Node-side memory switches: `free_comfy_vram` (unload ComfyUI's models before the job), `release_vram` (→ `options.release_after`) | [x] |

### C. Tutorial (`docs/wiki/ComfyUI.md`)

| # | Item | Status |
|---|---|---|
| C1 | Weights first: the official FL2VA checkpoint (~134 GiB, `./tools/download_weights.sh`) is separate from ComfyUI's model files | [x] |
| C2 | Build h3, start `h3 --serve`, check `/v1/info` | [x] |
| C3 | Install the node (symlink into `custom_nodes/`), load the example workflow, render | [x] |
| C4 | Pitfalls: VRAM sharing with other nodes (A2), port and `H3_MODEL`, the automatic `H3_INT8_HOST` on cards below 48 GiB, one daemon per GPU | [x] plus media root, outputs/quota, `SaveVideo` remux, RAM when mixed with native MiniMax-H3 |
| C5 | Link from `Getting-started.md` (Daemon section) and the README | [x] |

### D. Release

| # | Item | Status |
|---|---|---|
| D1 | Sync to public `alexhegit/h3-hip.c` (it is at 0.17; this tree carries 0.18, the PR #2 daemon fixes and the ComfyUI comparison) | [ ] |
| D2 | Tag the release that first ships the node and tutorial | [ ] |

## Sequence

1. A1 and A2. A2 is the one missing basic feature for living next to ComfyUI.
2. B and A3 in parallel; B3's FastH3 input waits for A3.
3. C once B renders end to end.
4. D.

## Exit criteria

- [ ] The A1 gate fails when daemon and CLI fox-s2 differ, on the build's own ISA.
- [ ] Following `docs/wiki/ComfyUI.md` on a clean ComfyUI install renders fox-s2
      through the node. `SaveVideo` remuxes the MP4, so compare streams
      (`ffmpeg -f streamhash`) with the CLI, not file md5.
- [ ] ComfyUI can load another GPU model after an h3 job without restarting
      the daemon (A2).
- [ ] The node and tutorial are on public `h3-hip.c` in a tagged release.

## Known limits

- Users download the official checkpoint even if ComfyUI already has
  Comfy-Org's MiniMax-H3 files.
- Outputs are not comparable to ComfyUI native (different sampler, INT8
  scheme and RNG); the comparison doc covers speed only.

## References

- R9700 vs ComfyUI: [`perf-runs/R9700_COMFYUI_COMPARE_2026-11-09.md`](perf-runs/R9700_COMFYUI_COMPARE_2026-11-09.md)
- KI-005 / KI-007: [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md)
- Protocol: [`DESIGN_DHS.md`](DESIGN_DHS.md)
- Smoke: `tools/h3d_smoke.py`
