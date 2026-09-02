# Multi-task Parallelism Design

Status: **not supported**. h3-hip.c is designed for single-task-at-a-time
execution. This document analyses the blocking state and evaluates three
options for concurrent inference on MI300X (192 GiB VRAM).

## Current state: single-task only

Concurrent use of multiple `h3` instances on the same GPU will corrupt
shared kernel-level state. There is no software lock preventing this.

### Shared kernel-level globals (`kernels/h3_kernels_extra.hip`)

| Variable | Type | Corruption risk |
|----------|------|----------------|
| `h3_hipblas` | `hipblasHandle_t` | Bound to last caller's stream; concurrent BLAS runs on wrong stream |
| `h3_sdpa_bf16_direct_hipblas` | `hipblasHandle_t` | Same as above |
| `h3_int8_i32_scratch` | `void*` | Resized on demand; concurrent use overwrites buffer mid-operation |
| `h3_int8_f32_scratch` | `void*` | Same |
| `h3_sdpa_scratch` | `void*` | Same |
| `h3_sdpa_bf16_scratch` | `void*` | Same |
| `h3_sdpa_bf16_direct_scratch` | `void*` | Same |
| `h3_int8_pack_scratch` | `void*` | Same |
| `h3_hip_upload_stream` | `hipStream_t` | Single global DMA stream; serialises all uploads |
| `static hipEvent_t` (x5 pairs) | `hipEvent_t` | Lazily created, no thread safety |

### Shared CPU-side state (mutex-protected but contended)

| Variable | Lock | Impact under concurrency |
|----------|------|--------------------------|
| `h3_hip_pin_ready/pending` | `h3_hip_pin_lock` | Global pinned pool (~6 GiB); contention under concurrent loads |
| `h3_hip_fd_cache` | `h3_hip_fd_cache_lock` | Weight-shard FD cache; serialises concurrent weight loads |
| `h3_hip_load_*` counters | `h3_hip_load_lock` | Timing stats would corrupt |

AGENTS.md already warns: *"Do not run two weight-streaming T2VA
jobs simultaneously."*

---

## Option A: Multi-process + HIP_VISIBLE_DEVICES

### Approach

Run each inference task in a separate OS process. Each process binds to a
different GPU via `HIP_VISIBLE_DEVICES=N` or `H3_HIP_DEVICE=N`. No code
changes required.

### Usage

```bash
# Terminal 1: GPU 0
HIP_VISIBLE_DEVICES=0 ./h3 -d /path/to/MiniMax-H3 -p "..." -o /tmp/out0.mp4

# Terminal 2: GPU 1
HIP_VISIBLE_DEVICES=1 ./h3 -d /path/to/MiniMax-H3 -p "..." -o /tmp/out1.mp4
```

Batch launcher (single node, 4x MI300X):

```bash
#!/bin/bash
PROMPTS=("prompt 1" "prompt 2" "prompt 3" "prompt 4")
for i in "${!PROMPTS[@]}"; do
  HIP_VISIBLE_DEVICES=$((i % 4)) \
    ./h3 -d "$H3_MODEL" -p "${PROMPTS[$i]}" -o "/tmp/out_$i.mp4" &
done
wait
```

### Pros

- Zero code changes.
- Full GPU isolation (no VRAM contention beyond OS/hypervisor limits).
- Works today on any multi-GPU box.
- Each process gets full 192 GiB on MI300X.

### Cons

- Weight loading duplicated per process (~107 GiB read times N).
- No shared page cache unless weights are pre-cached in RAM.
- Requires N GPUs for N tasks (no oversubscription on one GPU).
- Process startup overhead (~3 s weight load per task).

### Best for

- Datacentre deployments with multiple MI300X cards.
- Batch rendering pipelines.
- Tasks that can tolerate weight duplication.

---

## Option B: Per-task hipBLAS handle + scratch buffers

### Approach

Refactor kernel-level globals into per-stream resources. Each `h3_gpu`
instance owns its own hipBLAS handle and scratch buffers, eliminating
all shared mutable state in the kernel layer.

### Changes required

| File | Scope |
|------|-------|
| `kernels/h3_kernels_extra.hip` | Replace 8 static globals with a `h3_kernel_ctx` struct passed as argument |
| `kernels/h3_kernels.h` | Add `h3_kernel_ctx` creation / destruction API |
| `backends/h3_gpu_hip.c` | Create `h3_kernel_ctx` per `h3_gpu` instance |
| `h3_dit.c` | Thread `h3_kernel_ctx` through `h3_gpu_linear_*` calls |
| `h3_video_vae.c` | Same for VAE linear calls |

Estimated scope: ~200 lines across 4-5 files.

### Proposed `h3_kernel_ctx` struct

```c
typedef struct {
    hipblasHandle_t hipblas;
    hipblasHandle_t sdpa_bf16_hipblas;
    void *int8_i32_scratch;
    void *int8_f32_scratch;
    void *sdpa_scratch;
    void *sdpa_bf16_scratch;
    void *sdpa_bf16_direct_scratch;
    void *int8_pack_scratch;
    size_t scratch_sizes[8];  /* for lazy realloc */
} h3_kernel_ctx;
```

### Pros

- Single-process concurrency (no weight duplication).
- Shared page cache for weights across tasks.
- Clean architecture (no hidden global state).

### Cons

- ~200 lines of refactor across 4-5 files.
- Each task still needs its own VRAM (no sub-device partitioning).
- hipBLAS handle creation is ~50 ms per task (one-time cost).
- Scratch buffers compete for VRAM (total budget = 192 GiB shared).

### Best for

- Single-MI300X scenarios where weight sharing matters.
- Future GPU scheduler / job queue integration.

---

## Option C: CUDA MPS / HIP MIG (hardware partitioning)

### Approach

Use AMD's Multi-Process Service (MPS) or MIG-like partitioning to
hardware-isolate concurrent tasks on the same GPU.

### MI300X reality check

| Feature | MI300X support |
|---------|---------------|
| NVIDIA MPS | N/A (AMD) |
| AMD MPS (hipMPS) | **Not available on MI300X** (CDNA3) |
| MIG (Multi-Instance GPU) | **Not supported** on MI300X (only MI300A) |
| Virtualisation (SR-IOV) | Available on MI300X (requires hypervisor) |

MI300X does **not** support hardware-level GPU partitioning. The only
virtualisation path is hypervisor-level (AMD MxGPU / SR-IOV), which splits
the GPU into VMs -- not useful for single-host multi-task.

### Pros (if available)

- True hardware isolation.
- No software refactor needed.
- Guaranteed QoS per partition.

### Cons

- **Not available on MI300X.**
- Would require MI300A (APU) or future hardware.
- Even on MI300A, MIG partitions are fixed-size (not flexible).

### Best for

- Future hardware consideration only.
- Not actionable for current MI300X fleet.

---

## Recommendation

| Scenario | Recommended option |
|----------|-------------------|
| Multi-GPU datacentre | **A: Multi-process** (zero code change, works today) |
| Single MI300X, weight sharing matters | **B: Per-task kernel ctx** (refactor, ~200 lines) |
| Future hardware | **C: Wait for MI300A or AMD MPS** |

**Immediate next step:** Option A for production. Option B as a future
refactor when single-GPU concurrency becomes a requirement.

### VRAM budget analysis (single MI300X, all-opts)

| Component | Per-task | 2 tasks | 3 tasks |
|-----------|----------|---------|---------|
| DiT weights | 28 GiB | 56 GiB | 84 GiB |
| DiT activations | 8 GiB | 16 GiB | 24 GiB |
| Video VAE | 2.9 GiB | 5.8 GiB | 8.7 GiB |
| Overhead | ~2 GiB | ~4 GiB | ~6 GiB |
| **Total** | **~41 GiB** | **~82 GiB** | **~123 GiB** |
| Fits in 192 GiB? | Yes | Yes | Yes |

With all optimizations enabled, up to **3 concurrent 15 s cinematic tasks**
fit in a single MI300X's 192 GiB VRAM. Without optimisations (BF16 baseline),
only **2 tasks** fit (~100 GiB each).
