#!/usr/bin/env python3
"""Smoke test for a running `h3 --serve` daemon (protocol v1alpha).

Default checks need little GPU time: request parsing, validation,
concurrency, cancel, SSE framing, cache release. --generate also renders
fox-s2 twice (the second run hits the prepared-DiT cache and sets
release_after) and compares both with --fox-md5, the CLI's fox-s2 md5 from
the same build; with --first-frame/--fl2va-md5 it compares FL2VA too.
tools/serve_regression.sh runs the CLI and this script together.

    ./tools/h3d_smoke.py [--url http://127.0.0.1:8571] [--generate]
        [--fox-md5 MD5] [--first-frame PNG --fl2va-md5 MD5]

The first frame must live under the daemon's media_root.
"""
import argparse
import hashlib
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse

PROTO = {"X-H3-Protocol": "v1alpha"}
FOX = "A red fox walks through fresh snow."
failures = []


def connect(url, timeout=10):
    parts = urllib.parse.urlsplit(url)
    return http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)


def request(url, method, path, body=None, raw=None, timeout=10):
    conn = connect(url, timeout)
    try:
        headers = dict(PROTO)
        data = raw
        if body is not None:
            # Spaces after separators on purpose: the daemon must not depend on
            # compact JSON.
            data = json.dumps(body, separators=(", ", ": "))
        if data is not None:
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=data, headers=headers)
        response = conn.getresponse()
        text = response.read().decode("utf-8")
        return response.status, (json.loads(text) if text else None), response
    finally:
        conn.close()


def check(name, condition, detail=""):
    print(("ok   " if condition else "FAIL ") + name + (f"  ({detail})" if detail and not condition else ""))
    if not condition:
        failures.append(name)


def job_body(**overrides):
    body = {"prompt": FOX, "mode": "T2VA", "size": {"width": 512, "height": 512},
            "duration": {"frames": 22},
            "quality": {"preset": "custom", "steps": 2, "layers": 35, "reuse": 1},
            "seed": 42}
    body.update(overrides)
    return body


def submit(url, body):
    status, data, _ = request(url, "POST", "/v1/jobs", body)
    return status, data


def cancel(url, job_id):
    return request(url, "POST", f"/v1/jobs/{job_id}/cancel")


def wait(url, job_id, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, job, _ = request(url, "GET", f"/v1/jobs/{job_id}")
        if job["status"] in ("done", "failed", "cancelled", "timeout"):
            return job
        time.sleep(1)
    raise TimeoutError(job_id)


def sse_events(url, job_id, timeout=900):
    """Spec-style SSE parse: one event per blank-line-separated block."""
    conn = connect(url, timeout)
    conn.request("GET", f"/v1/jobs/{job_id}/events", headers=PROTO)
    response = conn.getresponse()
    events, block = [], []
    while True:
        line = response.readline()
        if not line:
            break
        line = line.decode("utf-8").rstrip("\r\n")
        if line:
            block.append(line)
            continue
        if block:
            fields = {}
            for item in block:
                key, _, value = item.partition(":")
                fields.setdefault(key, []).append(value[1:] if value.startswith(" ") else value)
            events.append((fields.get("event", ["message"])[0], "\n".join(fields.get("data", []))))
            block = []
    conn.close()
    return events


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def test_parsing(url):
    status, data = submit(url, job_body())
    check("spaced JSON is accepted", status == 202, f"{status} {data}")
    if status == 202:
        _, job, _ = request(url, "GET", f"/v1/jobs/{data['job_id']}")
        q = job["request"]["quality"]
        check("spaced JSON keeps quality", (q["steps"], q["layers"], q["reuse"]) == (2, 35, 1), q)
        cancel(url, data["job_id"])

    status, data, _ = request(url, "POST", "/v1/jobs", raw='{"prompt": "x", ')
    check("truncated JSON -> 400", status == 400, status)
    status, data = submit(url, job_body(mode="T3VA"))
    check("unknown mode -> 400", status == 400, status)
    status, data = submit(url, job_body(duration={"frames": 22, "seconds": 1}))
    check("frames + seconds -> 400", status == 400, status)
    status, data = submit(url, job_body(quality={"preset": "custom", "steps": "2"}))
    check("string steps -> 400", status == 400, status)

    status, data = submit(url, job_body(duration={"seconds": 0.9}))
    check("seconds is accepted", status == 202, status)
    if status == 202:
        _, job, _ = request(url, "GET", f"/v1/jobs/{data['job_id']}")
        check("0.9 s -> 22 frames", job["request"]["duration"]["frames"] == 22, job["request"]["duration"])
        cancel(url, data["job_id"])

    long_prompt = ("A red fox \"walks\" through snow.\n" * 240)[:7000]
    status, data = submit(url, job_body(prompt=long_prompt))
    check("7000-char prompt is accepted", status == 202, status)
    if status == 202:
        status, job, _ = request(url, "GET", f"/v1/jobs/{data['job_id']}")
        check("long-prompt job is valid JSON", status == 200 and job["job_id"] == data["job_id"])
        cancel(url, data["job_id"])


def test_geometry(url):
    for w, h in ((832, 480), (1344, 768), (480, 832)):
        status, data = submit(url, job_body(size={"width": w, "height": h}))
        check(f"{w}x{h} is accepted", status == 202, f"{status} {data}")
        if status == 202:
            cancel(url, data["job_id"])
    status, _ = submit(url, job_body(size={"width": 1536, "height": 1024}))
    check("over 768*1344 pixels -> 422", status == 422, status)
    status, _ = submit(url, job_body(size={"width": 500, "height": 512}))
    check("width not a multiple of 32 -> 422", status == 422, status)
    status, _ = submit(url, job_body(duration={"frames": 400}))
    check("frames over 362 -> 422", status == 422, status)
    status, _ = submit(url, job_body(quality={"preset": "custom", "steps": 1}))
    check("steps 1 -> 422", status == 422, status)


def test_refs(url, media_root):
    status, data = submit(url, job_body(mode="FL2VA", refs={"images": ["/etc/hostname"]}))
    check("ref outside media_root -> 400 PATH_OUT_OF_ROOT",
          status == 400 and data["error"]["code"] == "PATH_OUT_OF_ROOT", data)
    status, data = submit(url, job_body(
        mode="FL2VA", refs={"images": [os.path.join(media_root, "../../etc/hostname")]}))
    check("ref escaping with .. -> 400", status == 400, data)
    status, data = submit(url, job_body(
        mode="FL2VA", refs={"images": [os.path.join(media_root, "missing.png")]}))
    check("missing ref -> 400", status == 400, data)
    status, data = submit(url, job_body(mode="Ref2VA"))
    check("Ref2VA without refs -> 400", status == 400, data)

    probe = os.path.join(media_root, "h3d-smoke-probe.png")
    try:
        with open(probe, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")
    except OSError:
        print("skip: media_root not writable here, keyframe validation not checked")
        return
    try:
        status, data = submit(url, job_body(refs={"images": [probe]}))
        check("T2VA with keyframes -> 400", status == 400, data)
        status, data = submit(url, job_body(mode="FL2VA", refs={"images": [probe, probe, probe]}))
        check("FL2VA with 3 images -> 400", status == 400, data)
        status, data = submit(url, job_body(mode="FL2VA", refs={"last_frame": probe}))
        check("FL2VA last frame only is accepted", status == 202, data)
        if status == 202:
            _, job, _ = request(url, "GET", f"/v1/jobs/{data['job_id']}")
            refs = job["request"]["refs"]
            check("last_frame echoed, first_frame null",
                  refs["last_frame"] == os.path.realpath(probe) and refs["first_frame"] is None, refs)
            cancel(url, data["job_id"])
    finally:
        os.unlink(probe)


def test_fasth3(url, info):
    status, data = submit(url, job_body(options={"fasth3": "yes"}))
    check("string fasth3 -> 400", status == 400, data)
    available = info.get("adapters", {}).get("fasth3")
    check("info reports the FastH3 adapter", isinstance(available, bool), info.get("adapters"))
    status, data = submit(url, job_body(options={"fasth3": True}))
    if not available:
        check("fasth3 without an adapter -> 422", status == 422, data)
        return
    check("fasth3 is accepted", status == 202, data)
    if status == 202:
        _, job, _ = request(url, "GET", f"/v1/jobs/{data['job_id']}")
        lora = job["request"]["options"]["fasth3_lora"]
        check("fasth3 resolves the adapter", bool(lora) and os.path.isabs(lora), lora)
        cancel(url, data["job_id"])
        status, data = submit(url, job_body(options={"fasth3": True,
                                                     "extra_args": ["--fasth3-lora", lora]}))
        check("fasth3 plus --fasth3-lora -> 422", status == 422, data)


def test_options(url):
    status, data = submit(url, job_body(options={"extra_args": ["--bogus"]}))
    check("unknown extra arg -> 422", status == 422, data)
    status, data = submit(url, job_body(options={"extra_args": ["--fasth3-lora", "/nonexistent"]}))
    check("missing LoRA file -> 422", status == 422, data)
    status, data = submit(url, job_body(options={"token_reduction_schedule": "4:30"}))
    check("token_reduction_schedule -> 422", status == 422, data)
    status, data = submit(url, job_body(options={"release_after": "yes"}))
    check("string release_after -> 400", status == 400, data)
    status, data = submit(url, job_body(options={"release_after": True}))
    check("release_after is accepted", status == 202, data)
    if status == 202:
        _, job, _ = request(url, "GET", f"/v1/jobs/{data['job_id']}")
        check("release_after is echoed", job["request"]["options"]["release_after"] is True,
              job["request"]["options"])
        cancel(url, data["job_id"])
    status, data = submit(url, job_body(options={"extra_args": "--token-reduction --core-reuse 4"}))
    check("allowed extra args are accepted", status == 202, data)
    if status == 202:
        _, job, _ = request(url, "GET", f"/v1/jobs/{data['job_id']}")
        o = job["request"]["options"]
        check("extra args reach the job", o["token_reduction"] and o["core_reuse"] == 4, o)
        cancel(url, data["job_id"])


def test_concurrency_and_cancel(url):
    status, first = submit(url, job_body(quality={"preset": "fox-fast"}))
    status2, second = submit(url, job_body())
    if status != 202 or status2 != 202:
        check("submit two jobs", False, f"{status} {status2}")
        return
    # Hold a keep-alive connection open and an SSE stream on the running job.
    idle = connect(url)
    idle.connect()
    stream = threading.Thread(target=sse_events, args=(url, first["job_id"]), daemon=True)
    stream.start()
    time.sleep(1.0)

    t0 = time.time()
    status, info, response = request(url, "GET", "/v1/info")
    check("info answers while SSE and an idle socket are open",
          status == 200 and time.time() - t0 < 2, f"{time.time() - t0:.1f}s")
    check("responses close the connection",
          (response.getheader("Connection") or "").lower() == "close")

    status, job, _ = cancel(url, second["job_id"])
    check("cancel a queued job -> cancelled at once",
          status == 200 and job["status"] == "cancelled", job and job["status"])

    events = sse_events(url, second["job_id"], timeout=10)
    check("cancelled job stream ends with a cancelled event",
          events and events[-1][0] == "cancelled", [e[0] for e in events])

    cancel(url, first["job_id"])
    final = wait(url, first["job_id"], timeout=300)
    check("running job cancels", final["status"] == "cancelled", final["status"])
    stream.join(timeout=10)
    idle.close()

    status, job, _ = cancel(url, first["job_id"])
    check("cancel a finished job -> 409", status == 409, status)


def worker_cache(url):
    _, info, _ = request(url, "GET", "/v1/info")
    return info["workers"][0]["cache"]


def test_cache_release(url):
    status, info, _ = request(url, "GET", "/v1/info")
    check("info reports idle_release_sec", isinstance(info.get("idle_release_sec"), int), info)
    check("info reports the worker cache",
          {"prepared_dit", "video_decoder", "embedding_bytes"} <= set(info["workers"][0].get("cache", {})),
          info["workers"][0])
    status, data, _ = request(url, "POST", "/v1/cache/release", timeout=30)
    check("release on an idle daemon -> 200", status == 200 and data["released"] is True, f"{status} {data}")
    cache = worker_cache(url)
    check("release empties the cache", not cache["prepared_dit"] and not cache["video_decoder"], cache)


def check_fox(name, job, fox_md5):
    got = md5(job["result"]["mp4_path"])
    if fox_md5:
        check(f"{name} md5 matches the CLI", got == fox_md5, f"got {got} want {fox_md5}")
    else:
        print(f"skip: {name} md5 {got} not compared (pass --fox-md5 from a CLI run)")


def test_generate(url, fox_md5, first_frame, fl2va_md5):
    status, data = submit(url, job_body())
    check("fox-s2 submit", status == 202, data)
    events = sse_events(url, data["job_id"])
    names = [e[0] for e in events]
    check("SSE has progress events", "progress" in names, names[:5])
    check("SSE ends with done", names and names[-1] == "done", names[-3:])
    if names and names[-1] == "done":
        job = json.loads(events[-1][1])
        check("done event carries the job", job["status"] == "done" and job["result"]["mp4_path"])
        check_fox("fox-s2", job, fox_md5)
        cache = worker_cache(url)
        check("fox-s2 leaves the DiT cached", cache["prepared_dit"], cache)

    # Same request again: prepared-DiT cache hit, then release before done.
    status, data = submit(url, job_body(options={"release_after": True}))
    check("fox-s2 rerun submit", status == 202, data)
    if status == 202:
        job = wait(url, data["job_id"])
        check("fox-s2 rerun done", job["status"] == "done", job.get("error"))
        if job["status"] == "done":
            check_fox("fox-s2 cache-hit rerun", job, fox_md5)
            cache = worker_cache(url)
            check("release_after empties the cache by done",
                  not cache["prepared_dit"] and not cache["video_decoder"], cache)

    if first_frame:
        status, data = submit(url, job_body(mode="FL2VA", refs={"first_frame": first_frame}))
        check("FL2VA submit", status == 202, data)
        if status == 202:
            job = wait(url, data["job_id"])
            check("FL2VA done", job["status"] == "done", job.get("error"))
            if job["status"] == "done" and fl2va_md5:
                got = md5(job["result"]["mp4_path"])
                check("FL2VA md5 matches the CLI", got == fl2va_md5, got)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8571")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--fox-md5", help="CLI fox-s2 md5 from the same build")
    parser.add_argument("--first-frame")
    parser.add_argument("--fl2va-md5")
    args = parser.parse_args()

    status, info, _ = request(args.url, "GET", "/v1/info")
    if status != 200 or not info.get("ready"):
        sys.exit(f"daemon not ready at {args.url}: {status} {info}")
    _, idle_jobs, _ = request(args.url, "GET", "/v1/jobs?status=running")
    if idle_jobs["count"]:
        sys.exit("daemon is busy; run this against an idle daemon")

    test_parsing(args.url)
    test_geometry(args.url)
    test_refs(args.url, info["media_root"])
    test_options(args.url)
    test_fasth3(args.url, info)
    test_concurrency_and_cancel(args.url)
    test_cache_release(args.url)
    if args.generate:
        test_generate(args.url, args.fox_md5, args.first_frame, args.fl2va_md5)

    print(f"\n{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
