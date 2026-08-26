#!/usr/bin/env bash
# Alternates the weight-load pipeline on and off within one session. Wall time
# on this box drifts with page-cache state, so interleaving the two arms is the
# only way to read a difference smaller than ten percent.
set -u
PAIRS=${PAIRS:-2}

# Every overlap and allocation feature off, i.e. one blocking read at a time
# into freshly page-locked host memory. H3_PREAD_THREADS=1 rather than 8 now
# that the cross-block read-ahead is gone: it was the only other source of
# concurrency, so leaving the per-tensor streams up would not be a serial arm.
off_env=(
    H3_ADALN_PREFETCH=0
    H3_DEVICE_WEIGHTS=0
    H3_PIN_CACHE_GIB=0
    H3_PREAD_THREADS=1
    H3_VAE_LOAD_BLOCKING=1
)

for pair in $(seq 1 "$PAIRS"); do
    env "${off_env[@]}" LABEL="pair$pair/serial  " ./tools/bench_pipeline.sh
    LABEL="pair$pair/pipelined" ./tools/bench_pipeline.sh
done
