#!/usr/bin/env bash
# Find what makes KFD evict a process's GPU queues (docs/KNOWN_ISSUES.md KI-008).
#
# Puts kprobes with kernel stack traces on the two KFD eviction entry points
# and their callers, runs COMMAND as the invoking (sudo) user, then prints how
# often each probe fired and the most common stacks:
#   - MMU notifier path: amdgpu_hmm_invalidate_hsa -> amdgpu_amdkfd_evict_userptr
#     -> kgd2kfd_quiesce_mm (pinned-in-place user memory freed, reclaimed, or
#     migrated)
#   - TTM path: amdkfd_fence_enable_signaling -> kgd2kfd_schedule_evict_and_
#     restore_process -> evict_process_worker (a KFD buffer is moved or swapped)
#
#   sudo tools/kfd_eviction_trace.sh -- ./bench/fox-fast.sh /path/to/MiniMax-H3
#
# Needs root for tracefs. Removes its probes on exit. Full trace: $OUT
# (default /tmp/kfd_eviction_trace.txt).
set -u
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: sudo $0 -- COMMAND [ARGS...]"; exit 1; }
T=/sys/kernel/tracing
[ -d "$T/events" ] || T=/sys/kernel/debug/tracing
OUT=${OUT:-/tmp/kfd_eviction_trace.txt}
RUN_AS=${SUDO_USER:-root}
PROBES="amdgpu_hmm_invalidate_hsa amdgpu_amdkfd_evict_userptr kgd2kfd_quiesce_mm
        amdkfd_fence_enable_signaling kgd2kfd_schedule_evict_and_restore_process
        evict_process_worker restore_process_worker ttm_bo_swapout"

cleanup() {
    echo 0 > "$T/tracing_on"
    echo 0 > "$T/events/kprobes/enable" 2>/dev/null
    echo 0 > "$T/options/stacktrace"
    for p in $PROBES; do echo "-:kfdev_$p" >> "$T/kprobe_events" 2>/dev/null; done
}
trap cleanup EXIT

echo 0 > "$T/tracing_on"
echo > "$T/trace"
echo 65536 > "$T/buffer_size_kb"
for p in $PROBES; do
    echo "p:kfdev_$p $p" >> "$T/kprobe_events" 2>/dev/null || echo "probe $p unavailable"
done
echo 1 > "$T/options/stacktrace"
echo 1 > "$T/events/kprobes/enable"
echo 1 > "$T/tracing_on"

echo "running as $RUN_AS: $*"
start=$(date +%s)
sudo -u "$RUN_AS" -E -- "$@"
rc=$?
echo 0 > "$T/tracing_on"
cp "$T/trace" "$OUT"
chmod a+r "$OUT"
echo "command rc=$rc, $(( $(date +%s) - start )) s; trace: $OUT"

echo "== probe hits"
grep -o 'kfdev_[a-z0-9_]*' "$OUT" | sort | uniq -c | sort -rn

echo "== top stacks per probe (first 8 frames)"
awk '
    /kfdev_/ { if (probe != "") print probe "\t" stack; match($0, /kfdev_[a-z0-9_]+/);
               probe = substr($0, RSTART + 6, RLENGTH - 6); stack = ""; frames = 0; next }
    /^ *=> / { if (probe != "" && frames < 8) { sub(/^ *=> /, ""); sub(/\+0x.*/, "");
               stack = stack (frames ? " < " : "") $0; frames++ } next }
    END { if (probe != "") print probe "\t" stack }
' "$OUT" | sort | uniq -c | sort -rn | head -20
