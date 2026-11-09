# ComfyUI node for h3-hip (experimental)

`MiniMax-H3 (h3-hip, AMD)` renders MiniMax-H3 through a local
`h3 --serve` daemon and returns the MP4 to ComfyUI.

```bash
ln -s "$PWD/integrations/comfyui" /path/to/ComfyUI/custom_nodes/h3-hip
```

Setup, inputs, VRAM sharing and pitfalls: [`docs/wiki/ComfyUI.md`](../../docs/wiki/ComfyUI.md).

- `__init__.py` — the node (ComfyUI V3 API; tested on ComfyUI 0.44.0)
- `workflows/` — API-format example graphs (T2VA; FL2VA with FastH3)
- `run_workflow.py` — queue a workflow headless and wait for it
