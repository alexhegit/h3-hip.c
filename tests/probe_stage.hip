/* Probe: is a device allocation plus a staged upload cheaper than pinning?
 *
 * This box has a 96 GiB BIOS VRAM carveout and only 31 GiB of host RAM, so
 * `hipMalloc` draws on memory the host cannot use for anything else while
 * `hipHostMalloc` competes with the page cache that the weight reads need.
 * The question is whether the extra copy through a reusable pinned staging
 * buffer costs less than page-locking the destination once per tensor. */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Mirror of the DiT per-layer weight sizes that dominate the load phase. */
static const size_t block_bytes[] = {
    (size_t)5376 * 21504 * 2, (size_t)7168 * 5376 * 2,
    (size_t)5376 * 28672 * 2, (size_t)14336 * 5376 * 2,
};

enum { PER_BLOCK = 4 };

int main(int argc, char **argv) {
    unsigned count = argc > 1 ? (unsigned)atoi(argv[1]) : 32;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    size_t total = 0, largest = 0;
    for (unsigned i = 0; i < count; i++) {
        size_t bytes = block_bytes[i % PER_BLOCK];
        total += bytes;
        if (bytes > largest) largest = bytes;
    }
    printf("probe_stage: %u tensors, %.2f GiB total, largest %.2f GiB\n", count,
           (double)total / gib, (double)largest / gib);

    void **ptrs = (void **)calloc(count, sizeof(*ptrs));
    unsigned char *source = (unsigned char *)malloc(largest);
    memset(source, 0x5a, largest);

    hipStream_t stream;
    hipStreamCreate(&stream);

    /* Arm A: page-lock every destination, then fill it on the CPU. This is what
     * the loaders do today, with `source` standing in for the file read. */
    double start = now(), alloc = 0.0, fill = 0.0;
    for (unsigned i = 0; i < count; i++) {
        size_t bytes = block_bytes[i % PER_BLOCK];
        double t0 = now();
        if (hipHostMalloc(&ptrs[i], bytes, hipHostMallocDefault) != hipSuccess) {
            printf("  hipHostMalloc failed at %u\n", i);
            count = i;
            break;
        }
        alloc += now() - t0;
        t0 = now();
        memcpy(ptrs[i], source, bytes);
        fill += now() - t0;
    }
    double wall = now() - start;
    printf("A pinned destination      total %6.3fs  alloc %6.3fs (%6.2f GiB/s)"
           "  fill %6.3fs (%6.2f GiB/s)\n",
           wall, alloc, (double)total / gib / alloc, fill,
           (double)total / gib / fill);
    start = now();
    for (unsigned i = 0; i < count; i++)
        if (ptrs[i]) { hipHostFree(ptrs[i]); ptrs[i] = NULL; }
    printf("  free %6.3fs\n", now() - start);

    /* Arm B: device destination plus one reusable pinned staging buffer. The
     * staging buffer is page-locked once for the whole phase, so the per-tensor
     * cost is the CPU fill plus a DMA upload. */
    void *staging = NULL;
    size_t stage_bytes = 256u << 20;
    if (hipHostMalloc(&staging, stage_bytes, hipHostMallocDefault) !=
        hipSuccess) {
        printf("  staging allocation failed\n");
        return 1;
    }
    start = now();
    alloc = 0.0; fill = 0.0;
    double upload = 0.0;
    for (unsigned i = 0; i < count; i++) {
        size_t bytes = block_bytes[i % PER_BLOCK];
        double t0 = now();
        if (hipMalloc(&ptrs[i], bytes) != hipSuccess) {
            printf("  hipMalloc failed at %u\n", i);
            count = i;
            break;
        }
        alloc += now() - t0;
        /* Chunked through the staging buffer the way a bounded pool would. */
        for (size_t done = 0; done < bytes; done += stage_bytes) {
            size_t span = bytes - done;
            if (span > stage_bytes) span = stage_bytes;
            t0 = now();
            memcpy(staging, source + done, span);
            fill += now() - t0;
            t0 = now();
            hipMemcpyAsync((unsigned char *)ptrs[i] + done, staging, span,
                           hipMemcpyHostToDevice, stream);
            hipStreamSynchronize(stream);
            upload += now() - t0;
        }
    }
    wall = now() - start;
    printf("B device + staged upload  total %6.3fs  alloc %6.3fs (%6.2f GiB/s)"
           "  fill %6.3fs (%6.2f GiB/s)  upload %6.3fs (%6.2f GiB/s)\n",
           wall, alloc, (double)total / gib / alloc, fill,
           (double)total / gib / fill, upload, (double)total / gib / upload);
    start = now();
    for (unsigned i = 0; i < count; i++)
        if (ptrs[i]) { hipFree(ptrs[i]); ptrs[i] = NULL; }
    printf("  free %6.3fs\n", now() - start);

    hipHostFree(staging);
    hipStreamDestroy(stream);
    free(source);
    free(ptrs);
    return 0;
}
