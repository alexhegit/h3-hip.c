#!/usr/bin/env python3
"""Queue an API-format ComfyUI workflow and wait for it.

    ./run_workflow.py workflows/h3hip_t2va_api.json
    ./run_workflow.py workflows/h3hip_t2va_api.json --set 1.width=512 --set 1.fasth3=true
    ./run_workflow.py WORKFLOW --comfy http://127.0.0.1:8188 --interrupt-after 20

Prints one JSON line: status, wall time, saved files (absolute paths when
--output-dir is given) and the error message, if any. Values given with
--set are parsed as JSON when they can be (numbers, true/false), else kept
as strings.
"""
import argparse
import json
import os
import sys
import time
import urllib.request


def call(base, path, body=None):
    request = urllib.request.Request(
        base + path, data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read()
    return json.loads(text) if text else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workflow")
    parser.add_argument("--comfy", default="http://127.0.0.1:8188")
    parser.add_argument("--set", action="append", default=[], metavar="NODE.INPUT=VALUE")
    parser.add_argument("--output-dir", help="ComfyUI output directory, to print full paths")
    parser.add_argument("--interrupt-after", type=float,
                        help="POST /interrupt after this many seconds (cancel test)")
    args = parser.parse_args()

    graph = json.load(open(args.workflow))
    for item in args.set:
        key, _, value = item.partition("=")
        node, _, name = key.partition(".")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        graph[node]["inputs"][name] = value

    t0 = time.time()
    queued = call(args.comfy, "/prompt", {"prompt": graph, "client_id": "h3hip-run"})
    if queued.get("node_errors"):
        sys.exit(json.dumps(queued["node_errors"]))
    prompt_id = queued["prompt_id"]
    interrupted = False
    while True:
        if args.interrupt_after and not interrupted and time.time() - t0 > args.interrupt_after:
            call(args.comfy, "/interrupt", {})
            interrupted = True
        history = call(args.comfy, f"/history/{prompt_id}")
        status = history.get(prompt_id, {}).get("status", {})
        if status.get("completed") is not None and status.get("status_str") in ("success", "error"):
            break
        time.sleep(1)

    messages = {m[0]: m[1] for m in status.get("messages", [])}
    files = []
    for output in history[prompt_id]["outputs"].values():
        for item in output.get("images", []) + output.get("videos", []):
            relative = os.path.join(item.get("subfolder", ""), item["filename"])
            files.append(os.path.join(args.output_dir, relative) if args.output_dir else relative)
    error = messages.get("execution_error") or messages.get("execution_interrupted") or {}
    print(json.dumps({"status": status["status_str"],
                      "interrupted": "execution_interrupted" in messages,
                      "wall_s": round(time.time() - t0, 1), "files": files,
                      "error": error.get("exception_message")}))
    sys.exit(0 if status["status_str"] == "success" else 1)


if __name__ == "__main__":
    main()
