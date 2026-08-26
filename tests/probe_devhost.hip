/* Probe: can the CPU touch `hipMalloc` memory directly, and how fast?
 *
 * `mem_info_vis_vram_total` equals `mem_info_vram_total` on this APU, so the
 * whole 96 GiB carveout sits in the CPU-visible aperture. If a plain memcpy
 * into a device allocation works at a useful rate the loaders could pread
 * straight into it; if it faults or crawls, uploads must go through a pinned
 * staging buffer. */
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

int main(void) {
    const double gib = 1024.0 * 1024.0 * 1024.0;
    size_t bytes = 1u << 30;
    unsigned char *source = (unsigned char *)malloc(bytes);
    memset(source, 0x5a, bytes);

    struct {
        const char *label;
        int kind; /* 0 hipMalloc, 1 hipMallocManaged, 2 hipHostMalloc */
    } cases[] = {
        {"hipMalloc", 0}, {"hipMallocManaged", 1}, {"hipHostMalloc", 2},
    };

    for (unsigned c = 0; c < sizeof(cases) / sizeof(*cases); c++) {
        void *p = NULL;
        hipError_t status;
        if (cases[c].kind == 0) status = hipMalloc(&p, bytes);
        else if (cases[c].kind == 1) status = hipMallocManaged(&p, bytes);
        else status = hipHostMalloc(&p, bytes, hipHostMallocDefault);
        if (status != hipSuccess) {
            printf("%-18s alloc failed: %s\n", cases[c].label,
                   hipGetErrorString(status));
            continue;
        }
        /* Ask the runtime whether the pointer is CPU-addressable at all before
         * dereferencing it, so a hostile case reports instead of segfaulting. */
        hipPointerAttribute_t attr;
        memset(&attr, 0, sizeof(attr));
        if (hipPointerGetAttributes(&attr, p) == hipSuccess) {
            printf("%-18s type=%d hostPointer=%p devicePointer=%p\n",
                   cases[c].label, (int)attr.type, attr.hostPointer,
                   attr.devicePointer);
        }
        void *cpu = attr.hostPointer ? attr.hostPointer
                                     : (cases[c].kind == 0 ? NULL : p);
        if (!cpu) {
            printf("  no host-addressable alias; CPU access needs staging\n");
        } else {
            double start = now();
            memcpy(cpu, source, bytes);
            double write = now() - start;
            start = now();
            memcpy(source, cpu, bytes);
            double read = now() - start;
            printf("  cpu write %6.3fs (%6.2f GiB/s)  cpu read %6.3fs "
                   "(%6.2f GiB/s)\n", write, (double)bytes / gib / write, read,
                   (double)bytes / gib / read);
        }
        if (cases[c].kind == 2) hipHostFree(p); else hipFree(p);
    }
    free(source);
    return 0;
}
