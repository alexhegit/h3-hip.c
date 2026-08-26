/* Probe: does the destination buffer type change weight-read throughput?
 * The loaders pread straight into pinned host memory, so if pinned pages are
 * slower to fill than ordinary heap pages the whole load phase pays for it. */
#include <hip/hip_runtime.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

typedef struct {
    int fd;
    unsigned char *data;
    size_t bytes;
    off_t offset;
} job;

static void *worker(void *arg) {
    job *j = (job *)arg;
    size_t done = 0;
    while (done < j->bytes) {
        ssize_t n = pread(j->fd, j->data + done, j->bytes - done,
                          j->offset + (off_t)done);
        if (n <= 0) break;
        done += (size_t)n;
    }
    return NULL;
}

static double read_threads(int fd, unsigned char *dst, size_t bytes,
                           off_t offset, unsigned threads) {
    pthread_t t[64];
    job jobs[64];
    size_t chunk = (bytes + threads - 1) / threads;
    chunk = (chunk + ((1u << 20) - 1)) & ~((size_t)(1u << 20) - 1);
    double start = now();
    unsigned n = 0;
    for (unsigned i = 0; i < threads; i++) {
        size_t begin = (size_t)i * chunk;
        if (begin >= bytes) break;
        size_t len = bytes - begin;
        if (len > chunk) len = chunk;
        jobs[n] = (job){fd, dst + begin, len, offset + (off_t)begin};
        pthread_create(&t[n], NULL, worker, &jobs[n]);
        n++;
    }
    for (unsigned i = 0; i < n; i++) pthread_join(t[i], NULL);
    return now() - start;
}

/* Each measurement needs a range the page cache has never seen, so every
 * thread count gets its own file from the argument list. */
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: probe_read MiB FILE...\n");
        return 2;
    }
    size_t bytes = (size_t)atoi(argv[1]) << 20;
    void *pinned = NULL;
    if (hipHostMalloc(&pinned, bytes, hipHostMallocDefault) != hipSuccess) {
        fprintf(stderr, "pinned allocation failed\n");
        return 1;
    }
    void *heap = malloc(bytes);
    memset(heap, 0, bytes);
    memset(pinned, 0, bytes);
    const double gib = 1024.0 * 1024.0 * 1024.0;
    unsigned counts[] = {1, 2, 4, 8, 16, 32};
    unsigned cases = sizeof(counts) / sizeof(*counts);
    printf("cold reads of %.2f GiB per case\n", (double)bytes / gib);
    for (unsigned i = 0; i < cases; i++) {
        for (unsigned which = 0; which < 2; which++) {
            int index = 2 + (int)(i * 2 + which);
            if (index >= argc) break;
            int fd = open(argv[index], O_RDONLY);
            if (fd < 0) {
                perror("open");
                continue;
            }
            unsigned char *dst =
                (unsigned char *)(which ? heap : pinned);
            double seconds = read_threads(fd, dst, bytes, 0, counts[i]);
            printf("  threads=%2u %-6s %6.3fs (%5.2f GiB/s)  %s\n", counts[i],
                   which ? "heap" : "pinned", seconds,
                   (double)bytes / gib / seconds, argv[index]);
            close(fd);
        }
    }
    hipHostFree(pinned);
    free(heap);
    return 0;
}
