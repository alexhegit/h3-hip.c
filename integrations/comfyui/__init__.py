"""ComfyUI node: MiniMax-H3 on AMD GPUs through a local `h3 --serve` (h3-hip).

The node is a thin HTTP client. The daemon owns the GPU, the weights and the
generation cache; ComfyUI receives the finished MP4. Setup and pitfalls:
docs/wiki/ComfyUI.md in the h3-hip repository.
"""
import http.client
import json
import logging
import os
import socket
import tempfile
import time
import urllib.parse

import numpy as np
from PIL import Image

import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, InputImpl, io

PROTOCOL = "v1alpha"
DEFAULT_URL = os.environ.get("H3_HIP_URL", "http://127.0.0.1:8571")
TERMINAL = ("done", "failed", "cancelled", "timeout")

# The bar runs 0..PROGRESS_TOTAL over the whole job. Each daemon phase owns a
# span of it, sized by its share of the wall time on the W7900 at 768p
# (denoise and the video VAE decode dominate). A phase's own step/total moves
# the bar across its span; phases not listed here hold the bar where it is.
PROGRESS_TOTAL = 1000
PHASE_SPANS = {
    "loading_weights": (0, 20),
    "tokenizer": (20, 20),
    "text encoder": (20, 120),
    "refine text": (120, 140),
    "precompute AdaLN": (140, 180),
    "load transformer core": (180, 280),
    "denoise enqueue": (280, 700),
    "denoise": (280, 700),
    "audio VAE": (700, 720),
    "video VAE load": (720, 760),
    "preview VAE load": (720, 760),
    "video VAE decode": (760, 960),
    "video TAEH3": (760, 960),
    # The last 10% is held back until the job is done, so the bar reaches 100%
    # only when the MP4 is written.
    "FFmpeg": (960, 990),
}


class DaemonError(RuntimeError):
    pass


def _request(url, method, path, body=None, timeout=30):
    parts = urllib.parse.urlsplit(url)
    connection = http.client.HTTPConnection(parts.hostname, parts.port or 80,
                                            timeout=timeout)
    headers = {"X-H3-Protocol": PROTOCOL}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body)
    try:
        connection.request(method, path, body=data, headers=headers)
        response = connection.getresponse()
        text = response.read().decode("utf-8")
    except (ConnectionError, socket.timeout, OSError) as error:
        raise DaemonError(
            f"h3 daemon not reachable at {url} ({error}). Start it with "
            f"`./h3 -d /path/to/MiniMax-H3 --serve`.") from error
    finally:
        connection.close()
    payload = json.loads(text) if text else None
    if response.status == 426:
        raise DaemonError(f"h3 daemon speaks {payload.get('server_protocol')}, "
                          f"this node speaks {PROTOCOL}; update one of them")
    return response.status, payload


def _error_text(payload):
    error = (payload or {}).get("error") or {}
    return error.get("message") or json.dumps(payload)


def _warn_if_low_ram():
    try:
        import psutil
    except ImportError:
        return
    available = psutil.virtual_memory().available / 2**30
    if available >= 24:
        return
    held = psutil.Process().memory_info().rss / 2**30
    logging.warning(
        "h3-hip: %.0f GiB RAM available and ComfyUI holds %.0f GiB. The daemon needs "
        "about 20 GiB for a job on a card below 48 GiB. If this ComfyUI ran native "
        "MiniMax-H3 nodes, restart it first: their weights stay in RAM and the job "
        "can get ComfyUI OOM-killed.", available, held)


def _save_keyframe(tensor, media_root, name):
    """Write an IMAGE tensor as PNG under the daemon's media_root."""
    array = np.rint(np.clip(tensor[0].cpu().numpy(), 0.0, 1.0) * 255.0).astype(np.uint8)
    os.makedirs(media_root, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=f"comfy-{name}-", suffix=".png", dir=media_root)
    os.close(fd)
    Image.fromarray(array).save(path)
    return path


class H3HipGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3HipGenerate",
            display_name="MiniMax-H3 (h3-hip, AMD)",
            category="video/h3-hip",
            description="Text (and optional first/last frame) to video with audio, "
                        "rendered by a local h3-hip daemon.",
            inputs=[
                io.String.Input("prompt", multiline=True),
                io.Int.Input("width", default=864, min=32, max=1344, step=32),
                io.Int.Input("height", default=480, min=32, max=1344, step=32),
                io.Int.Input("length", default=124, min=5, max=362, step=17,
                             tooltip="Frames at 24 fps; 124 is 5 s."),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFF,
                             control_after_generate=True),
                io.Int.Input("steps", default=20, min=2, max=100,
                             tooltip="Ignored with fasth3, which always runs 4 steps."),
                io.Boolean.Input("fasth3", default=False,
                                 tooltip="4-step FastH3 LoRA configured on the daemon. "
                                         "Requires layers = 50 and reuse = 1."),
                io.Int.Input("layers", default=50, min=35, max=50, advanced=True,
                             tooltip="DiT layers (35-50). Below 50 is faster and lossy. "
                                     "Must be 50 when fasth3 is on."),
                io.Int.Input("reuse", default=1, min=1, max=3, advanced=True,
                             tooltip="Denoise reuse; above 1 is faster and lossy. "
                                     "Must be 1 when fasth3 is on."),
                io.Boolean.Input("free_comfy_vram", default=True, advanced=True,
                                 tooltip="Unload ComfyUI's own models before the job so "
                                         "the daemon has the GPU."),
                io.Boolean.Input("release_vram", default=False, advanced=True,
                                 tooltip="Have the daemon free its cache after the job. "
                                         "Turn on when later nodes need the GPU; reruns "
                                         "of the same prompt are then slower."),
                io.String.Input("daemon_url", default=DEFAULT_URL, advanced=True),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
            ],
            outputs=[io.Video.Output()],
        )

    @classmethod
    def execute(cls, prompt, width, height, length, seed, steps, fasth3, layers,
                reuse, free_comfy_vram, release_vram, daemon_url,
                first_frame=None, last_frame=None) -> io.NodeOutput:
        # The daemon enforces the same rule, but only after it loads weights.
        if fasth3 and layers != 50:
            raise DaemonError(
                f"fasth3 needs layers = 50 (layers is {layers}). Set layers to 50, "
                "or turn fasth3 off to use fewer layers.")
        if fasth3 and reuse != 1:
            raise DaemonError(
                f"fasth3 needs reuse = 1 (reuse is {reuse}). Set reuse to 1, "
                "or turn fasth3 off.")
        url = daemon_url.rstrip("/")
        status, info = _request(url, "GET", "/v1/info")
        if status != 200 or not info.get("ready"):
            raise DaemonError(f"h3 daemon not ready: {(info or {}).get('unready_reason')}")
        if fasth3 and not info.get("adapters", {}).get("fasth3"):
            raise DaemonError("this h3 daemon has no FastH3 adapter; run "
                              "tools/download_fasth3_lora.sh or set H3D_FASTH3_LORA")

        refs = {}
        keyframes = []
        try:
            for tensor, name in ((first_frame, "first_frame"), (last_frame, "last_frame")):
                if tensor is not None:
                    refs[name] = _save_keyframe(tensor, info["media_root"], name)
                    keyframes.append(refs[name])
            body = {
                "prompt": prompt,
                "mode": "FL2VA" if refs else "T2VA",
                "size": {"width": width, "height": height},
                "duration": {"frames": length},
                "quality": {"preset": "custom", "steps": 4 if fasth3 else steps,
                            "layers": layers, "reuse": reuse},
                "seed": seed,
                "options": {"fasth3": fasth3, "release_after": release_vram},
            }
            if refs:
                body["refs"] = refs
            if free_comfy_vram:
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache()
            _warn_if_low_ram()

            status, data = _request(url, "POST", "/v1/jobs", body)
            if status != 202:
                raise DaemonError(f"h3 daemon rejected the job ({status}): {_error_text(data)}")
            return io.NodeOutput(InputImpl.VideoFromFile(cls._wait(url, data["job_id"])))
        finally:
            for path in keyframes:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    @classmethod
    def _wait(cls, url, job_id):
        bar = comfy.utils.ProgressBar(PROGRESS_TOTAL)
        phase = None
        value = 0
        try:
            while True:
                comfy.model_management.throw_exception_if_processing_interrupted()
                _, job = _request(url, "GET", f"/v1/jobs/{job_id}")
                progress = job.get("progress") or {}
                if job.get("phase") and job["phase"] != phase:
                    phase = job["phase"]
                    logging.info("h3-hip %s: %s", job_id, phase)
                if phase in PHASE_SPANS:
                    low, high = PHASE_SPANS[phase]
                    total = progress.get("total_steps") or 0
                    share = min(1.0, progress.get("step", 0) / total) if total else 0.0
                    # Never move backwards if a phase reports out of order.
                    value = max(value, round(low + (high - low) * share))
                    bar.update_absolute(value, PROGRESS_TOTAL)
                if job["status"] == "done":
                    bar.update_absolute(PROGRESS_TOTAL, PROGRESS_TOTAL)
                    return job["result"]["mp4_path"]
                if job["status"] in TERMINAL:
                    raise DaemonError(f"h3 job {job['status']}: {_error_text(job)}")
                time.sleep(1.0)
        except comfy.model_management.InterruptProcessingException:
            _request(url, "POST", f"/v1/jobs/{job_id}/cancel")
            raise


class H3HipExtension(ComfyExtension):
    async def get_node_list(self):
        return [H3HipGenerate]


async def comfy_entrypoint() -> H3HipExtension:
    return H3HipExtension()
