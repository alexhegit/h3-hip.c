#!/usr/bin/env python3
"""A/B the weight loader: buffered vs H3_WEIGHT_HIPFILE=1 vs H3_WEIGHT_DIRECT=1.

Runs a bench script once per (repetition, mode), interleaved, and prints one
CSV row per run. Each run records E2E wall, phase totals, load wall time
(io-wall), KFD queue eviction, host memory, and hipFile per-path read counts.

    tools/hipfile_ab.py --case fox-fast --runs 3 --modes buffered,hipfile
    tools/hipfile_ab.py --case fox-fast --disk-cache 0          # cold
    tools/hipfile_ab.py --case fox-fast --keep-page-cache       # back-to-back
    tools/hipfile_ab.py --case fox-s2 -- --ssd-streaming        # extra h3 args

Build with `make HIP_ARCH=... HIPFILE=1 HIPFILE_INC=... HIPFILE_LIB=...` first;
without HIPFILE=1 the hipfile mode silently falls back to buffered (check the
`hipfile` column). Run from the repository root. The page cache of the model
directory is dropped before every run with posix_fadvise(DONTNEED), which
needs no root but only drops clean pages.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time

MODES = {
    "buffered": {},
    "hipfile": {"H3_WEIGHT_HIPFILE": "1"},
    "direct": {"H3_WEIGHT_DIRECT": "1"},
}


def drop_page_cache(root):
    for base, _dirs, files in os.walk(root):
        for name in files:
            try:
                fd = os.open(os.path.join(base, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)


def meminfo():
    out = {}
    with open("/proc/meminfo") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            out[key] = int(value.split()[0]) / 1048576.0
    return out


class Sampler(threading.Thread):
    """Largest KFD evicted_ms of any GPU process, and host memory extremes."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        self.evicted_ms = 0
        self.cached_max = 0.0
        self.free_min = float("inf")
        self.swap_used_max = 0.0

    def run(self):
        while not self.stop.is_set():
            for path in glob.glob("/sys/class/kfd/kfd/proc/*/stats_*/evicted_ms"):
                try:
                    with open(path) as handle:
                        self.evicted_ms = max(self.evicted_ms, int(handle.read().strip() or 0))
                except (OSError, ValueError):
                    pass
            m = meminfo()
            self.cached_max = max(self.cached_max, m["Cached"])
            self.free_min = min(self.free_min, m["MemFree"])
            self.swap_used_max = max(self.swap_used_max, m["SwapTotal"] - m["SwapFree"])
            self.stop.wait(0.2)


def parse(text):
    phases = {}
    for name, value in re.findall(r"h3 profile: (.+?)\s+total\s+wall=\s*([0-9.]+)s", text):
        phases.setdefault(name.strip(), float(value))
    io_wall = sum(float(v) for v in re.findall(r"weight-load.*?io-wall=\s*([0-9.]+)s", text))
    read_gib = sum(float(v) for v in re.findall(r"weight-load.*?read=\s*[0-9.]+s \(\s*([0-9.]+)GiB", text))
    hipfile = re.findall(r"hipfile-reads\s+total=(\d+) p2p=(\d+) nvfs=(\d+) posix=(\d+) "
                         r"unaligned=(\d+) errors=(\d+)", text)
    denoise = re.search(r"H3 DiT\s+Euler denoise\s+wall=\s*([0-9.]+)s", text)
    stream = re.search(r"BF16 SSD stream ([0-9.]+) GiB read in ([0-9.]+)s \(([0-9.]+) GiB/s\), "
                       r"unhidden wait ([0-9.]+)s", text)
    return {
        "phases": phases, "io_wall": io_wall, "read_gib": read_gib,
        "hipfile_reads": dict(zip(("total", "p2p", "fastpath", "posix", "unaligned", "errors"),
                                  map(int, hipfile[-1]))) if hipfile else None,
        "denoise": float(denoise.group(1)) if denoise else None,
        "stream_gib_per_s": float(stream.group(3)) if stream else None,
        "stream_unhidden_s": float(stream.group(4)) if stream else None,
        "fallbacks": len(re.findall(r"direct read fell back", text)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", default="fox-fast", help="bench/CASE.sh (default fox-fast)")
    parser.add_argument("--script", help="explicit bench script path instead of --case")
    parser.add_argument("--output", help="mp4 the script writes (default outputs/CASE.mp4)")
    parser.add_argument("--model", default=os.environ.get("H3_MODEL", "/home/amd/HF-MODELS/MiniMax-H3"))
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--modes", default="buffered,hipfile")
    parser.add_argument("--disk-cache", choices=("0", "1"), default="1")
    parser.add_argument("--keep-page-cache", action="store_true",
                        help="do not drop the model page cache before each run")
    parser.add_argument("--prime", action="store_true",
                        help="one untimed buffered run first (fills the disk cache)")
    parser.add_argument("--out-dir", default="outputs/hipfile_ab")
    parser.add_argument("extra", nargs="*", help="arguments after -- go to the bench script")
    args = parser.parse_args()

    script = args.script or "./bench/%s.sh" % args.case
    output = args.output or "outputs/%s.mp4" % args.case
    modes = args.modes.split(",")
    for mode in modes:
        if mode not in MODES:
            sys.exit("unknown mode %s (choose from %s)" % (mode, ", ".join(MODES)))
    os.makedirs(args.out_dir, exist_ok=True)

    def run(tag, mode):
        if not args.keep_page_cache:
            drop_page_cache(args.model)
        env = dict(os.environ)
        env.update({"H3_DISK_CACHE": args.disk_cache, "H3_PROFILE": "1"})
        env.update(MODES[mode])
        if os.path.exists(output):
            os.remove(output)
        log_path = os.path.join(args.out_dir, tag + ".log")
        sampler = Sampler()
        sampler.start()
        start = time.monotonic()
        with open(log_path, "w") as log:
            rc = subprocess.run([script, args.model] + args.extra, env=env,
                                stdout=log, stderr=subprocess.STDOUT).returncode
        wall = time.monotonic() - start
        sampler.stop.set()
        sampler.join()
        with open(log_path, errors="replace") as handle:
            result = parse(handle.read().replace("\r", "\n"))
        md5 = "NA"
        if os.path.exists(output):
            with open(output, "rb") as handle:
                md5 = hashlib.md5(handle.read()).hexdigest()
        result.update({"tag": tag, "mode": mode, "rc": rc, "md5": md5, "wall": wall,
                       "evicted_ms": sampler.evicted_ms, "cached_max_gib": sampler.cached_max,
                       "free_min_gib": sampler.free_min, "swap_used_max_gib": sampler.swap_used_max})
        with open(os.path.join(args.out_dir, tag + ".json"), "w") as handle:
            json.dump(result, handle, indent=1)
        return result

    if args.prime:
        run("%s-prime" % args.case, "buffered")
    print("tag,mode,rc,md5,wall_s,denoise_s,io_wall_s,read_gib,evicted_ms,"
          "cached_max_gib,free_min_gib,swap_max_gib,hipfile_reads,fallbacks")
    for repetition in range(1, args.runs + 1):
        for mode in modes:
            r = run("%s-%s-r%d" % (args.case, mode, repetition), mode)
            hipfile = r["hipfile_reads"]
            print("%s,%s,%d,%s,%.2f,%s,%.2f,%.1f,%d,%.1f,%.1f,%.1f,%s,%d" % (
                r["tag"], mode, r["rc"], r["md5"], r["wall"],
                "%.2f" % r["denoise"] if r["denoise"] is not None else "NA",
                r["io_wall"], r["read_gib"], r["evicted_ms"], r["cached_max_gib"],
                r["free_min_gib"], r["swap_used_max_gib"],
                "fastpath=%d/posix=%d/p2p=%d" % (hipfile["fastpath"], hipfile["posix"], hipfile["p2p"])
                if hipfile else "none", r["fallbacks"]), flush=True)


if __name__ == "__main__":
    main()
