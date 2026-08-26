/* Probe: cost of the allocation strategies available for resident weights.
 * The video VAE pins ~9.5 GiB per phase, so the per-GiB cost of the allocator
 * is worth as much as a kernel optimization. */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

__global__ void touch(float *p, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] += 1.0f;
}

/* Mirror of the video VAE per-block weight sizes (f32 bytes). */
static const size_t block_bytes[] = {
    (size_t)6144 * 2048 * 4, (size_t)2048 * 2048 * 4,
    (size_t)16384 * 2048 * 4, (size_t)2048 * 8192 * 4,
};

enum { BLOCKS = 36, PER_BLOCK = 4 };

static double bandwidth_test(void **ptrs, unsigned count, const char *label) {
    double start = now();
    for (unsigned i = 0; i < count; i++) {
        size_t n = block_bytes[i % PER_BLOCK] / sizeof(float);
        touch<<<(unsigned)((n + 255) / 256), 256>>>((float *)ptrs[i], n);
    }
    hipDeviceSynchronize();
    double seconds = now() - start;
    size_t total = 0;
    for (unsigned i = 0; i < count; i++) total += block_bytes[i % PER_BLOCK];
    printf("  %-22s kernel touch %6.3fs (%6.2f GiB/s effective)\n", label,
           seconds, (double)total * 2.0 / (1024.0 * 1024.0 * 1024.0) / seconds);
    return seconds;
}

int main(int argc, char **argv) {
    unsigned count = BLOCKS * PER_BLOCK;
    if (argc > 1) count = (unsigned)atoi(argv[1]);
    void **ptrs = (void **)calloc(count, sizeof(*ptrs));
    size_t total = 0;
    for (unsigned i = 0; i < count; i++) total += block_bytes[i % PER_BLOCK];
    const double gib = 1024.0 * 1024.0 * 1024.0;
    printf("probe: %u tensors, %.3f GiB total\n", count, (double)total / gib);

    struct {
        const char *label;
        unsigned flags;
        int device;
    } cases[] = {
        {"hipHostMalloc default", hipHostMallocDefault, 0},
        {"hipHostMalloc numa", hipHostMallocNumaUser, 0},
        {"hipHostMalloc noncoherent", hipHostMallocNonCoherent, 0},
        {"hipMalloc", 0, 1},
    };
    for (unsigned c = 0; c < sizeof(cases) / sizeof(*cases); c++) {
        double start = now();
        int ok = 1;
        for (unsigned i = 0; i < count && ok; i++) {
            hipError_t status = cases[c].device
                ? hipMalloc(&ptrs[i], block_bytes[i % PER_BLOCK])
                : hipHostMalloc(&ptrs[i], block_bytes[i % PER_BLOCK],
                                cases[c].flags);
            ok = status == hipSuccess;
            if (!ok) printf("  %s failed at %u: %s\n", cases[c].label, i,
                            hipGetErrorString(status));
        }
        double alloc = now() - start;
        printf("%-26s alloc %6.3fs (%6.2f GiB/s)\n", cases[c].label, alloc,
               (double)total / gib / alloc);
        if (ok) bandwidth_test(ptrs, count, cases[c].label);
        start = now();
        for (unsigned i = 0; i < count; i++) {
            if (!ptrs[i]) continue;
            if (cases[c].device) hipFree(ptrs[i]);
            else hipHostFree(ptrs[i]);
            ptrs[i] = NULL;
        }
        printf("  %-22s free  %6.3fs\n", cases[c].label, now() - start);
    }
    free(ptrs);
    return 0;
}
