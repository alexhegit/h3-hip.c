/* Probe: where does hipMallocManaged memory come from, and how fast is it?
 *
 * `hipMallocManaged` is CPU-addressable like pinned memory but may be backed by
 * the 96 GiB VRAM carveout rather than the 31 GiB host pool. If so it is a
 * drop-in replacement for `hipHostMalloc` in the weight loaders: the loaders
 * keep preading straight into the buffer and the allocation stops competing
 * with the page cache. This reports the allocation rate at a footprint large
 * enough to trigger the collapse pinned memory shows (1.6 GiB/s at 23 GiB) and
 * prints the driver's VRAM/GTT accounting so the backing store is visible. */
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

static long long sysfs_value(const char *name) {
    char path[256];
    snprintf(path, sizeof(path), "/sys/class/drm/card1/device/%s", name);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    long long value = -1;
    if (fscanf(f, "%lld", &value) != 1) value = -1;
    fclose(f);
    return value;
}

static void report(const char *when) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    printf("  [%s] vram_used %.2f GiB  gtt_used %.2f GiB\n", when,
           (double)sysfs_value("mem_info_vram_used") / gib,
           (double)sysfs_value("mem_info_gtt_used") / gib);
}

__global__ void touch(float *p, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] += 1.0f;
}

/* DiT per-layer weight sizes, the shapes that dominate the load phase. */
static const size_t block_bytes[] = {
    (size_t)5376 * 21504 * 2, (size_t)7168 * 5376 * 2,
    (size_t)5376 * 28672 * 2, (size_t)14336 * 5376 * 2,
};
enum { PER_BLOCK = 4 };

int main(int argc, char **argv) {
    unsigned count = argc > 1 ? (unsigned)atoi(argv[1]) : 128;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    size_t total = 0, largest = 0;
    for (unsigned i = 0; i < count; i++) {
        size_t bytes = block_bytes[i % PER_BLOCK];
        total += bytes;
        if (bytes > largest) largest = bytes;
    }
    printf("probe_managed: %u tensors, %.2f GiB total\n", count,
           (double)total / gib);
    void **ptrs = (void **)calloc(count, sizeof(*ptrs));
    unsigned char *source = (unsigned char *)malloc(largest);
    memset(source, 0x5a, largest);
    report("start");

    double alloc = 0.0, fill = 0.0;
    double start = now();
    for (unsigned i = 0; i < count; i++) {
        size_t bytes = block_bytes[i % PER_BLOCK];
        double t0 = now();
        hipError_t status = hipMallocManaged(&ptrs[i], bytes);
        if (status != hipSuccess) {
            printf("  hipMallocManaged failed at %u (%.2f GiB in): %s\n", i,
                   (double)(i ? i : 1) * 0.2, hipGetErrorString(status));
            count = i;
            break;
        }
        alloc += now() - t0;
        t0 = now();
        memcpy(ptrs[i], source, bytes);
        fill += now() - t0;
    }
    double wall = now() - start;
    printf("managed  total %6.3fs  alloc %6.3fs (%6.2f GiB/s)  cpu fill %6.3fs"
           " (%6.2f GiB/s)\n", wall, alloc, (double)total / gib / alloc, fill,
           (double)total / gib / fill);
    report("allocated");

    /* Kernel-visible at full speed, or does the managed page migrate? */
    start = now();
    for (unsigned i = 0; i < count; i++) {
        size_t n = block_bytes[i % PER_BLOCK] / sizeof(float);
        touch<<<(unsigned)((n + 255) / 256), 256>>>((float *)ptrs[i], n);
    }
    (void)hipDeviceSynchronize();
    double seconds = now() - start;
    printf("  kernel touch %6.3fs (%6.2f GiB/s effective)\n", seconds,
           (double)total * 2.0 / gib / seconds);
    report("touched");

    start = now();
    for (unsigned i = 0; i < count; i++) if (ptrs[i]) (void)hipFree(ptrs[i]);
    printf("  free %6.3fs\n", now() - start);
    report("freed");
    free(source);
    free(ptrs);
    return 0;
}
