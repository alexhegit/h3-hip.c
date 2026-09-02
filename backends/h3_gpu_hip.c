#include "h3_gpu.h"
#include "h3_device.h"
#include "kernels/h3_kernels.h"

#include <hip/hip_runtime_api.h>

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* Reuse open weight-shard fds + mmap across thousands of tensor copies. */
enum { H3_HIP_FD_CACHE_MAX = 64 };
static struct {
    char path[768];
    int fd;
    void *map;
    size_t size;
} h3_hip_fd_cache[H3_HIP_FD_CACHE_MAX];
static int h3_hip_fd_cache_count;
static pthread_mutex_t h3_hip_fd_cache_lock = PTHREAD_MUTEX_INITIALIZER;

static int h3_hip_mmap_disabled(void) {
    /* mmap+memcpy page-faulted worse than pread on this APU (load 129s).
     * Opt in: H3_WEIGHT_MMAP=1. */
    const char *value = getenv("H3_WEIGHT_MMAP");
    return !value || strcmp(value, "1") != 0;
}

static int h3_hip_open_weight_fd(const char *path, int *from_cache) {
    *from_cache = 0;
    pthread_mutex_lock(&h3_hip_fd_cache_lock);
    for (int index = 0; index < h3_hip_fd_cache_count; index++) {
        if (!strcmp(h3_hip_fd_cache[index].path, path)) {
            *from_cache = 1;
            int fd = h3_hip_fd_cache[index].fd;
            pthread_mutex_unlock(&h3_hip_fd_cache_lock);
            return fd;
        }
    }
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        pthread_mutex_unlock(&h3_hip_fd_cache_lock);
        return -1;
    }
#if defined(POSIX_FADV_SEQUENTIAL)
    /* Sequential hint fights 8-way striped pread. Keep it only for serial I/O. */
    if (getenv("H3_PREAD_SERIAL"))
        posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
    void *map = MAP_FAILED;
    size_t size = 0;
    struct stat st;
    if (!h3_hip_mmap_disabled() && fstat(fd, &st) == 0 && st.st_size > 0) {
        size = (size_t)st.st_size;
        map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map != MAP_FAILED) {
#if defined(MADV_SEQUENTIAL)
            madvise(map, size, MADV_SEQUENTIAL);
#endif
        } else {
            map = MAP_FAILED;
            size = 0;
        }
    }
    if (h3_hip_fd_cache_count < H3_HIP_FD_CACHE_MAX &&
        strlen(path) < sizeof(h3_hip_fd_cache[0].path)) {
        memcpy(h3_hip_fd_cache[h3_hip_fd_cache_count].path, path,
               strlen(path) + 1);
        h3_hip_fd_cache[h3_hip_fd_cache_count].fd = fd;
        h3_hip_fd_cache[h3_hip_fd_cache_count].map =
            map == MAP_FAILED ? NULL : map;
        h3_hip_fd_cache[h3_hip_fd_cache_count].size = size;
        h3_hip_fd_cache_count++;
        *from_cache = 1;
    } else if (map != MAP_FAILED) {
        munmap(map, size);
    }
    pthread_mutex_unlock(&h3_hip_fd_cache_lock);
    return fd;
}

static void *h3_hip_weight_map(const char *path, size_t *size_out) {
    pthread_mutex_lock(&h3_hip_fd_cache_lock);
    for (int index = 0; index < h3_hip_fd_cache_count; index++) {
        if (!strcmp(h3_hip_fd_cache[index].path, path)) {
            if (size_out) *size_out = h3_hip_fd_cache[index].size;
            void *map = h3_hip_fd_cache[index].map;
            pthread_mutex_unlock(&h3_hip_fd_cache_lock);
            return map;
        }
    }
    pthread_mutex_unlock(&h3_hip_fd_cache_lock);
    if (size_out) *size_out = 0;
    return NULL;
}

static int h3_hip_pread_serial(int fd, void *data, size_t bytes, off_t offset) {
    size_t done = 0;
    while (done < bytes) {
        size_t remaining = bytes - done;
        size_t chunk = remaining > (size_t)(1u << 28) ? (size_t)(1u << 28)
                                                      : remaining;
        ssize_t count = pread(fd, (unsigned char *)data + done, chunk,
                              offset + (off_t)done);
        if (count <= 0) return 0;
        done += (size_t)count;
    }
    return 1;
}

typedef struct {
    int fd;
    unsigned char *data;
    size_t bytes;
    off_t offset;
    int ok;
} h3_hip_pread_job;

static void *h3_hip_pread_worker(void *arg) {
    h3_hip_pread_job *job = (h3_hip_pread_job *)arg;
    job->ok = h3_hip_pread_serial(job->fd, job->data, job->bytes, job->offset);
    return NULL;
}

/* Parallel pread for large weight tensors (NVMe can serve concurrent reads).
 *
 * This count has been measured twice with opposite answers, and both were right
 * about the build they ran on. While weight buffers were page-locked, a loader
 * thread spent most of its wall inside hipHostMalloc, so extra read streams only
 * added contention and 8 beat 16 (core load 30.5s versus 35.0s). Once weights
 * moved to device memory the threads are genuinely waiting on the drive, and
 * more streams pay:
 *
 *   streams:      4       8      16      32
 *   core        31.6    22.9    19.5    18.4     (medians, interleaved pairs)
 *   AdaLN         -     18.6    18.7    17.7
 *
 * 32 wins both stages in every pair, so it is the default. It is also the cap,
 * because the isolated read probe already flattens between 16 and 32 (2.24 then
 * 2.19 GiB/s) and the job array is fixed at that size; nothing suggests more
 * would help, and the loaders run four tensors at once, so this is already 128
 * outstanding reads.
 *
 * Capping streams globally across loaders instead was much worse (155s versus
 * 80s end to end): the first caller reserved the whole budget and left the rest
 * at single-stream speed. */
static int h3_hip_pread_threads(void) {
    static int cached;
    if (cached) return cached;
    const char *value = getenv("H3_PREAD_THREADS");
    int count = value ? atoi(value) : 32;
    if (count < 1) count = 1;
    if (count > 32) count = 32;
    cached = count;
    return cached;
}

static int h3_hip_pread_all(int fd, void *data, size_t bytes, off_t offset) {
    enum { H3_PREAD_THREADS = 32, H3_PREAD_MIN = 16u << 20 };
#if defined(POSIX_FADV_WILLNEED)
    if (bytes >= H3_PREAD_MIN)
        posix_fadvise(fd, offset, (off_t)bytes, POSIX_FADV_WILLNEED);
#endif
    if (bytes < H3_PREAD_MIN || getenv("H3_PREAD_SERIAL"))
        return h3_hip_pread_serial(fd, data, bytes, offset);

    int want = h3_hip_pread_threads();
    size_t chunk = (bytes + (size_t)want - 1) / (size_t)want;
    chunk = (chunk + ((1u << 20) - 1)) & ~((size_t)(1u << 20) - 1);
    pthread_t threads[H3_PREAD_THREADS];
    h3_hip_pread_job jobs[H3_PREAD_THREADS];
    int started[H3_PREAD_THREADS];
    unsigned n = 0;
    for (unsigned i = 0; i < (unsigned)want; i++) {
        size_t start = (size_t)i * chunk;
        if (start >= bytes) break;
        size_t len = bytes - start;
        if (len > chunk) len = chunk;
        jobs[n].fd = fd;
        jobs[n].data = (unsigned char *)data + start;
        jobs[n].bytes = len;
        jobs[n].offset = offset + (off_t)start;
        jobs[n].ok = 0;
        started[n] = pthread_create(&threads[n], NULL, h3_hip_pread_worker,
                                    &jobs[n]) == 0;
        if (!started[n]) {
            jobs[n].ok = h3_hip_pread_serial(fd, jobs[n].data, jobs[n].bytes,
                                             jobs[n].offset);
        }
        n++;
    }
    int ok = 1;
    for (unsigned i = 0; i < n; i++) {
        if (started[i]) pthread_join(threads[i], NULL);
        if (!jobs[i].ok) ok = 0;
    }
    return ok;
}

static double h3_hip_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* hipSetDevice is per-thread. Loader/prefetch workers start on device 0, so
 * every HIP allocation and DMA path rebinds to H3_HIP_DEVICE. */
static int h3_hip_bind_thread_device(void) {
    return hipSetDevice(h3_hip_device_index()) == hipSuccess;
}

/* Weight-load accounting: allocation, file-read and device-upload costs are
 * billed separately so a slow load can be attributed without a profiler. The
 * seconds are summed across loader threads, so they exceed phase wall time. */
static double h3_hip_load_alloc_seconds;
static double h3_hip_load_read_seconds;
static double h3_hip_load_upload_seconds;
static size_t h3_hip_load_alloc_bytes;
static size_t h3_hip_load_read_bytes;
static size_t h3_hip_load_upload_bytes;
/* Split of the allocated bytes by backing store, so the profile shows how much
 * of a phase still page-locks host memory rather than using the carveout. */
static size_t h3_hip_load_pinned_bytes;
static size_t h3_hip_load_device_bytes;
static pthread_mutex_t h3_hip_load_lock = PTHREAD_MUTEX_INITIALIZER;

static void h3_hip_load_account(double *slot, size_t *counter, double seconds,
                                size_t bytes) {
    pthread_mutex_lock(&h3_hip_load_lock);
    *slot += seconds;
    *counter += bytes;
    pthread_mutex_unlock(&h3_hip_load_lock);
}

static void h3_hip_load_count(size_t *counter, size_t bytes) {
    pthread_mutex_lock(&h3_hip_load_lock);
    *counter += bytes;
    pthread_mutex_unlock(&h3_hip_load_lock);
}

/* Pinning host memory costs about 2 GiB/s on this APU, and a phase streams
 * several times its peak live footprint through identically sized tensors as
 * it walks the layers. Recycling freed pinned blocks by exact byte size turns
 * most of that page-locking into a free-list pop.
 *
 * Two invariants keep recycling faithful to hipHostFree. A freed buffer may
 * still be read by kernels already queued on the stream -- hipHostFree stalls
 * for them, a free list would not -- so freed blocks wait in `pending` and only
 * become reusable after a stream synchronization. And fresh pinned pages read
 * as zero, so a reused block is cleared unless the caller overwrites all of it.
 */
enum { H3_HIP_PIN_CACHE_MAX = 96, H3_HIP_PIN_CACHE_MIN_BYTES = 1u << 20 };

typedef struct {
    void *host;
    size_t bytes;
} h3_hip_pin_block;

typedef struct {
    h3_hip_pin_block block[H3_HIP_PIN_CACHE_MAX];
    int count;
    size_t bytes;
} h3_hip_pin_list;

static h3_hip_pin_list h3_hip_pin_ready;
static h3_hip_pin_list h3_hip_pin_pending;
static size_t h3_hip_pin_cache_limit;
static size_t h3_hip_pin_hit_bytes;
static size_t h3_hip_pin_miss_bytes;
static pthread_mutex_t h3_hip_pin_lock = PTHREAD_MUTEX_INITIALIZER;

/* Prefer device-resident weights (VRAM / APU carveout). Loaders pread into a
 * recycled pinned staging buffer and DMA to the device.
 *
 * On Strix Halo the BIOS split (~96 GiB carveout + ~31 GiB host) made
 * hipHostMalloc the bottleneck; hipMalloc uses the carveout. On a discrete
 * GPU the same default still avoids repeated PCIe reads of hot weights, but
 * hipMalloc failure falls back to pinned host (64 GiB MI210 cannot hold the
 * full ~107 GiB checkpoint).
 *
 * Opt out with H3_DEVICE_WEIGHTS=0. Force on with H3_DEVICE_WEIGHTS=1. */
static int h3_hip_device_weights(void) {
    static int cached;
    if (!cached) {
        const char *value = getenv("H3_DEVICE_WEIGHTS");
        cached = (value && !strcmp(value, "0")) ? -1 : 1;
    }
    return cached > 0;
}

/* Keep enough HBM free for activations and temporary device buffers. MI210
 * cannot safely allocate weights until hipMalloc fails: by then kernels may
 * have no room left for their working set. */
static size_t h3_hip_device_weight_reserve(void) {
    static size_t cached;
    if (cached) return cached;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    hipDeviceProp_t props;
    int device = 0;
    int integrated = 0;
    if (hipGetDevice(&device) == hipSuccess &&
        hipGetDeviceProperties(&props, device) == hipSuccess) {
        integrated = props.integrated ? 1 : 0;
    }
    const char *value = getenv("H3_DEVICE_WEIGHT_RESERVE_GIB");
    double want = value ? atof(value) : (integrated ? 4.0 : 20.0);
    if (want < 0.0) want = 0.0;
    cached = (size_t)(want * gib);
    return cached ? cached : 1;
}

static int h3_hip_device_weight_fits(size_t request) {
    size_t free_bytes = 0, total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) return 0;
    size_t reserve = h3_hip_device_weight_reserve();
    if (reserve > total_bytes / 2) reserve = total_bytes / 2;
    return request <= free_bytes && free_bytes - request >= reserve;
}

/* With weights in device memory the pinned pool only holds transient staging
 * buffers, so it is sized by how many reads are in flight at once (the AdaLN
 * ring is 4 x 520 MiB, the core loader 4 x 308 MiB) rather than by the whole
 * streamed working set. That is a smaller job than the old cache had, and it no
 * longer has to be traded off against page cache, because the 15 GiB of live
 * weights it used to hold pinned are now out of host RAM entirely.
 *
 * In the all-pinned fallback the old sizing still applies: 3 GiB recycles ~87%
 * of the streamed layer weights and wider was measured slower, since held
 * pinned pages cannot be reclaimed for page cache. */
static size_t h3_hip_pin_cache_capacity(void) {
    if (h3_hip_pin_cache_limit) return h3_hip_pin_cache_limit;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    int staging = h3_hip_device_weights();
    const char *value = getenv("H3_PIN_CACHE_GIB");
    double want = value ? atof(value) : (staging ? 6.0 : 3.0);
    if (want < 0.0) want = 0.0;
    long pages = sysconf(_SC_PHYS_PAGES), page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0) {
        double share = (staging ? 0.25 : 0.12) * (double)pages *
                       (double)page_size / gib;
        if (want > share) want = share;
    }
    h3_hip_pin_cache_limit = (size_t)(want * gib);
    if (!h3_hip_pin_cache_limit) h3_hip_pin_cache_limit = 1;
    return h3_hip_pin_cache_limit;
}

static int h3_hip_pin_push(h3_hip_pin_list *list, void *host, size_t bytes) {
    if (list->count >= H3_HIP_PIN_CACHE_MAX) return 0;
    list->block[list->count].host = host;
    list->block[list->count].bytes = bytes;
    list->count++;
    list->bytes += bytes;
    return 1;
}

static void *h3_hip_pin_take(size_t bytes) {
    if (bytes < H3_HIP_PIN_CACHE_MIN_BYTES) return NULL;
    void *host = NULL;
    pthread_mutex_lock(&h3_hip_pin_lock);
    for (int index = h3_hip_pin_ready.count - 1; index >= 0; index--) {
        if (h3_hip_pin_ready.block[index].bytes != bytes) continue;
        host = h3_hip_pin_ready.block[index].host;
        h3_hip_pin_ready.bytes -= bytes;
        h3_hip_pin_ready.block[index] =
            h3_hip_pin_ready.block[--h3_hip_pin_ready.count];
        break;
    }
    if (host) h3_hip_pin_hit_bytes += bytes;
    else h3_hip_pin_miss_bytes += bytes;
    pthread_mutex_unlock(&h3_hip_pin_lock);
    return host;
}

/* `idle` means the owning stream has drained, so no queued kernel can still be
 * reading the block and it can skip the pending queue. Layer loops free their
 * weights right after a submit, so this is the path that actually hits. */
static int h3_hip_pin_give(void *host, size_t bytes, int idle) {
    if (!host || bytes < H3_HIP_PIN_CACHE_MIN_BYTES) return 0;
    size_t limit = h3_hip_pin_cache_capacity();
    int kept = 0;
    pthread_mutex_lock(&h3_hip_pin_lock);
    if (h3_hip_pin_ready.bytes + h3_hip_pin_pending.bytes + bytes <= limit) {
        kept = h3_hip_pin_push(idle ? &h3_hip_pin_ready : &h3_hip_pin_pending,
                               host, bytes);
    }
    pthread_mutex_unlock(&h3_hip_pin_lock);
    return kept;
}

/* Called after a stream synchronization: everything queued before these blocks
 * were released has retired, so they are safe to hand out again. */
static void h3_hip_pin_retire(void) {
    pthread_mutex_lock(&h3_hip_pin_lock);
    for (int index = 0; index < h3_hip_pin_pending.count; index++) {
        h3_hip_pin_block block = h3_hip_pin_pending.block[index];
        if (!h3_hip_pin_push(&h3_hip_pin_ready, block.host, block.bytes))
            hipHostFree(block.host);
    }
    h3_hip_pin_pending.count = 0;
    h3_hip_pin_pending.bytes = 0;
    pthread_mutex_unlock(&h3_hip_pin_lock);
}

static void h3_hip_pin_purge(void) {
    pthread_mutex_lock(&h3_hip_pin_lock);
    h3_hip_pin_list *lists[] = {&h3_hip_pin_ready, &h3_hip_pin_pending};
    for (unsigned which = 0; which < 2; which++) {
        for (int index = 0; index < lists[which]->count; index++)
            hipHostFree(lists[which]->block[index].host);
        lists[which]->count = 0;
        lists[which]->bytes = 0;
    }
    pthread_mutex_unlock(&h3_hip_pin_lock);
}

/* Uploads run on their own non-blocking stream so a weight arriving on a loader
 * thread never orders itself against queued compute, and so the synchronization
 * that releases the staging buffer cannot stall the GPU. It outlives any one
 * h3_gpu, because the tests create and destroy several and pthread_once would
 * not rebuild a destroyed stream. */
static hipStream_t h3_hip_upload_stream;
static pthread_once_t h3_hip_upload_once = PTHREAD_ONCE_INIT;

static void h3_hip_upload_stream_create(void) {
    if (!h3_hip_bind_thread_device() ||
        hipStreamCreateWithFlags(&h3_hip_upload_stream, hipStreamNonBlocking) !=
            hipSuccess) {
        h3_hip_upload_stream = NULL;
    }
}

static hipStream_t h3_hip_upload(void) {
    pthread_once(&h3_hip_upload_once, h3_hip_upload_stream_create);
    return h3_hip_upload_stream;
}

/* A staging buffer is reusable as soon as the upload stream drains: the DMA has
 * already read it, so unlike a weight buffer it carries no dependency on queued
 * compute and can go straight back to the ready list. */
enum {
    H3_HIP_STAGE_MAX_BYTES = (size_t)640u << 20,
    H3_HIP_STAGE_GRAIN = (size_t)32u << 20
};

/* The pinned pool matches blocks by exact size, which is right for a tensor but
 * wrong for staging: a staging buffer only has to be big enough. Qwen's layer
 * shapes are all different, so exact matching missed 7% of 46.9 GiB and each
 * miss page-locked. Rounding the request to a 32 MiB grain collapses the model's
 * shapes onto a couple of dozen block sizes, which then recycle. */
static size_t h3_hip_stage_bucket(size_t bytes) {
    size_t rounded = (bytes + H3_HIP_STAGE_GRAIN - 1) & ~(H3_HIP_STAGE_GRAIN - 1);
    return rounded < bytes ? bytes : rounded;
}

static void *h3_hip_stage_take(size_t bytes) {
    void *host = h3_hip_pin_take(bytes);
    if (host) return host;
    /* A staging miss page-locks, which is the cost this whole scheme exists to
     * avoid, so bill it to the same `pin=` counter the tensor allocator uses. */
    double start = h3_hip_now();
    if (hipHostMalloc(&host, bytes, hipHostMallocDefault) != hipSuccess)
        return NULL;
    h3_hip_load_account(&h3_hip_load_alloc_seconds, &h3_hip_load_alloc_bytes,
                        h3_hip_now() - start, 0);
    return host;
}

static void h3_hip_stage_give(void *host, size_t bytes) {
    if (!host) return;
    if (!h3_hip_pin_give(host, bytes, 1)) hipHostFree(host);
}

/* Split a transfer so each piece fits a staging buffer, keeping the pieces equal
 * so repeated tensors of the same shape recycle the same block sizes. */
static size_t h3_hip_stage_chunk(size_t bytes) {
    if (bytes <= H3_HIP_STAGE_MAX_BYTES) return bytes;
    size_t pieces = (bytes + H3_HIP_STAGE_MAX_BYTES - 1) /
                    H3_HIP_STAGE_MAX_BYTES;
    return (bytes + pieces - 1) / pieces;
}

/* Size of the staging block to reserve for a transfer of `bytes`: one bucketed
 * block per chunk. Callers copy only the bytes they need out of it. */
static size_t h3_hip_stage_block(size_t bytes) {
    return h3_hip_stage_bucket(h3_hip_stage_chunk(bytes));
}

/* Copy host memory into a device allocation through a pinned staging buffer.
 * `source` may be ordinary heap, which is the whole point: the caller never has
 * to own pinned memory. */
static int h3_hip_upload_bytes(void *device, const void *source, size_t bytes) {
    if (!h3_hip_bind_thread_device()) return 0;
    hipStream_t stream = h3_hip_upload();
    if (!stream || !bytes) return bytes == 0;
    size_t chunk = h3_hip_stage_chunk(bytes), block = h3_hip_stage_block(bytes);
    void *staging = h3_hip_stage_take(block);
    if (!staging) return 0;
    int ok = 1;
    for (size_t done = 0; done < bytes && ok; done += chunk) {
        size_t span = bytes - done;
        if (span > chunk) span = chunk;
        memcpy(staging, (const unsigned char *)source + done, span);
        ok = hipMemcpyAsync((unsigned char *)device + done, staging, span,
                            hipMemcpyHostToDevice, stream) == hipSuccess &&
             hipStreamSynchronize(stream) == hipSuccess;
    }
    h3_hip_stage_give(staging, block);
    return ok;
}

/* Read a device allocation back into host memory. Readback is rare (a few bulk
 * calls per phase), so it goes straight through the runtime rather than paying
 * for a staging buffer it would only use once. */
static int h3_hip_download_bytes(void *destination, const void *device,
                                 size_t bytes) {
    if (!h3_hip_bind_thread_device()) return 0;
    hipStream_t stream = h3_hip_upload();
    if (!stream || !bytes) return bytes == 0;
    return hipMemcpyAsync(destination, device, bytes, hipMemcpyDeviceToHost,
                          stream) == hipSuccess &&
           hipStreamSynchronize(stream) == hipSuccess;
}

static int h3_hip_copy_from_file(int fd, const char *path, void *data,
                                 size_t bytes, off_t offset) {
    size_t map_size = 0;
    const unsigned char *map = (const unsigned char *)h3_hip_weight_map(
        path, &map_size);
    double start = h3_hip_now();
    int ok;
    if (map && map_size && offset >= 0 &&
        (size_t)offset <= map_size && bytes <= map_size - (size_t)offset) {
        memcpy(data, map + (size_t)offset, bytes);
        ok = 1;
    } else {
        ok = h3_hip_pread_all(fd, data, bytes, offset);
    }
    h3_hip_load_account(&h3_hip_load_read_seconds, &h3_hip_load_read_bytes,
                        h3_hip_now() - start, bytes);
    return ok;
}

/* `data` is whatever pointer the kernels dereference. For an activation that is
 * pinned host memory, which host code can also touch directly; for a weight it
 * is a `hipMalloc` device allocation with no host alias, and `device` says so.
 * Every CPU-side path checks the flag and bounces through staging. */
struct h3_gpu_tensor {
    void *data;
    size_t elements;
    size_t bytes;
    h3_gpu_dtype dtype;
    unsigned device;
    struct h3_gpu *owner;
};

struct h3_gpu {
    int device_id;
    hipStream_t stream;
    char last_error[512];
    h3_gpu_stats stats;
    char profile_label[128];
    int in_command;
    double profile_start_wall;
    h3_gpu_stats profile_start_stats;
    double profile_mark_wall;
    h3_gpu_stats profile_mark_stats;
    double command_start_wall;
    /* Optional op-class GPU timing (H3_PROFILE). Events are recorded on the
     * stream and resolved after synchronize so kernels stay overlapped. */
    hipEvent_t *profile_events;
    uint8_t *profile_cats;
    int profile_pair_cap;
    int profile_pair_count;
    double profile_linear_ms;
    double profile_sdpa_ms;
    double profile_conv_ms;
    double profile_other_ms;
    /* Scratch for SDPA K/V [seq][head][dim] → [head][seq][dim] transpose. */
    void *kv_hm_scratch;
    size_t kv_hm_scratch_bytes;
    /* Set by QKV/RoPE when K/V were written head-major; consumed by SDPA. */
    int sdpa_kv_already_hm;
    void *nax_fc1_temp;
    size_t nax_fc1_temp_elems;
};

enum {
    H3_HIP_PROF_LINEAR = 0,
    H3_HIP_PROF_SDPA = 1,
    H3_HIP_PROF_CONV = 2,
    H3_HIP_PROF_OTHER = 3
};

static int h3_hip_profile_enabled(void) {
    const char *value = getenv("H3_PROFILE");
    return value && *value && strcmp(value, "0");
}

static int h3_hip_profile_grow(struct h3_gpu *gpu) {
    int want = gpu->profile_pair_cap ? gpu->profile_pair_cap * 2 : 512;
    hipEvent_t *events = realloc(gpu->profile_events,
                                 (size_t)want * 2u * sizeof(*events));
    uint8_t *cats = realloc(gpu->profile_cats, (size_t)want * sizeof(*cats));
    if (!events || !cats) {
        free(events);
        free(cats);
        return 0;
    }
    for (int index = gpu->profile_pair_cap; index < want; index++) {
        if (hipEventCreateWithFlags(&events[2 * index], hipEventDefault)
                != hipSuccess ||
            hipEventCreateWithFlags(&events[2 * index + 1], hipEventDefault)
                != hipSuccess) {
            return 0;
        }
    }
    gpu->profile_events = events;
    gpu->profile_cats = cats;
    gpu->profile_pair_cap = want;
    return 1;
}

static void h3_hip_profile_begin_op(struct h3_gpu *gpu) {
    if (!gpu || !h3_hip_profile_enabled()) return;
    if (gpu->profile_pair_count >= gpu->profile_pair_cap &&
        !h3_hip_profile_grow(gpu)) {
        return;
    }
    if (gpu->profile_pair_count >= gpu->profile_pair_cap) return;
    hipEventRecord(gpu->profile_events[2 * gpu->profile_pair_count],
                   gpu->stream);
}

static void h3_hip_profile_end_op(struct h3_gpu *gpu, int category) {
    if (!gpu || !h3_hip_profile_enabled()) return;
    if (gpu->profile_pair_count >= gpu->profile_pair_cap) return;
    hipEventRecord(gpu->profile_events[2 * gpu->profile_pair_count + 1],
                   gpu->stream);
    gpu->profile_cats[gpu->profile_pair_count] = (uint8_t)category;
    gpu->profile_pair_count++;
}

static void h3_hip_profile_flush_ops(struct h3_gpu *gpu) {
    if (!gpu || !gpu->profile_pair_count) return;
    for (int index = 0; index < gpu->profile_pair_count; index++) {
        float ms = 0.0f;
        if (hipEventElapsedTime(&ms, gpu->profile_events[2 * index],
                                gpu->profile_events[2 * index + 1])
            != hipSuccess) {
            continue;
        }
        switch (gpu->profile_cats[index]) {
        case H3_HIP_PROF_LINEAR: gpu->profile_linear_ms += ms; break;
        case H3_HIP_PROF_SDPA: gpu->profile_sdpa_ms += ms; break;
        case H3_HIP_PROF_CONV: gpu->profile_conv_ms += ms; break;
        default: gpu->profile_other_ms += ms; break;
        }
    }
    gpu->profile_pair_count = 0;
}

static void h3_hip_profile_destroy_ops(struct h3_gpu *gpu) {
    if (!gpu || !gpu->profile_events) return;
    for (int index = 0; index < gpu->profile_pair_cap; index++) {
        hipEventDestroy(gpu->profile_events[2 * index]);
        hipEventDestroy(gpu->profile_events[2 * index + 1]);
    }
    free(gpu->profile_events);
    free(gpu->profile_cats);
    gpu->profile_events = NULL;
    gpu->profile_cats = NULL;
    gpu->profile_pair_cap = 0;
    gpu->profile_pair_count = 0;
}

static uint64_t h3_hip_counter_delta(uint64_t value, uint64_t start) {
    return value >= start ? value - start : 0;
}

static void h3_hip_profile_emit_load(struct h3_gpu *gpu) {
    if (!gpu || !h3_hip_profile_enabled()) return;
    pthread_mutex_lock(&h3_hip_load_lock);
    double alloc = h3_hip_load_alloc_seconds, read = h3_hip_load_read_seconds;
    double upload = h3_hip_load_upload_seconds;
    size_t alloc_bytes = h3_hip_load_alloc_bytes;
    size_t read_bytes = h3_hip_load_read_bytes;
    size_t upload_bytes = h3_hip_load_upload_bytes;
    size_t pinned_bytes = h3_hip_load_pinned_bytes;
    size_t device_bytes = h3_hip_load_device_bytes;
    h3_hip_load_alloc_seconds = h3_hip_load_read_seconds = 0.0;
    h3_hip_load_upload_seconds = 0.0;
    h3_hip_load_alloc_bytes = h3_hip_load_read_bytes = 0;
    h3_hip_load_upload_bytes = 0;
    h3_hip_load_pinned_bytes = h3_hip_load_device_bytes = 0;
    pthread_mutex_unlock(&h3_hip_load_lock);
    if (alloc + read + upload <= 0.0) return;
    pthread_mutex_lock(&h3_hip_pin_lock);
    size_t hit = h3_hip_pin_hit_bytes, miss = h3_hip_pin_miss_bytes;
    h3_hip_pin_hit_bytes = h3_hip_pin_miss_bytes = 0;
    pthread_mutex_unlock(&h3_hip_pin_lock);
    const double gib = 1024.0 * 1024.0 * 1024.0;
    fprintf(stderr,
            "h3 profile: %-24s %-14s pin=%7.3fs (%6.3fGiB, %6.2fGiB/s) "
            "read=%7.3fs (%6.3fGiB, %6.2fGiB/s) "
            "upload=%7.3fs (%6.3fGiB, %6.2fGiB/s) recycled=%5.1f%% "
            "pinned=%6.3fGiB device=%6.3fGiB\n",
            gpu->profile_label[0] ? gpu->profile_label : "HIP context",
            "weight-load", alloc, (double)alloc_bytes / gib,
            alloc > 0.0 ? (double)alloc_bytes / gib / alloc : 0.0,
            read, (double)read_bytes / gib,
            read > 0.0 ? (double)read_bytes / gib / read : 0.0,
            upload, (double)upload_bytes / gib,
            upload > 0.0 ? (double)upload_bytes / gib / upload : 0.0,
            hit + miss ? 100.0 * (double)hit / (double)(hit + miss) : 0.0,
            (double)pinned_bytes / gib, (double)device_bytes / gib);
}

static void h3_hip_profile_emit_ops(struct h3_gpu *gpu) {
    if (!gpu || !h3_hip_profile_enabled()) return;
    double total = gpu->profile_linear_ms + gpu->profile_sdpa_ms +
                   gpu->profile_conv_ms + gpu->profile_other_ms;
    if (total <= 0.0) return;
    fprintf(stderr,
            "h3 profile: %-24s %-14s gpu-op=%7.3fs "
            "linear=%7.3fs sdpa=%7.3fs conv=%7.3fs other=%7.3fs\n",
            gpu->profile_label[0] ? gpu->profile_label : "HIP context",
            "op-classes", total / 1000.0,
            gpu->profile_linear_ms / 1000.0,
            gpu->profile_sdpa_ms / 1000.0,
            gpu->profile_conv_ms / 1000.0,
            gpu->profile_other_ms / 1000.0);
}

static void h3_hip_profile_emit(struct h3_gpu *gpu, const char *phase,
                                h3_gpu_stats start, double wall_start) {
    if (!gpu || !phase || !h3_hip_profile_enabled()) return;
    h3_gpu_stats value = gpu->stats;
    double wall = h3_hip_now() - wall_start;
    fprintf(stderr,
        "h3 profile: %-24s %-14s wall=%8.3fs encode=%7.3fs "
        "wait=%8.3fs root-gpu=%7.3fs "
        "peak=%7.3fGiB alloc=%7.3fGiB submissions=%llu "
        "direct=%llu linear=%llu conv=%llu attention=%llu\n",
        gpu->profile_label[0] ? gpu->profile_label : "HIP context",
        phase, wall,
        value.command_encode_seconds - start.command_encode_seconds,
        value.command_wait_seconds - start.command_wait_seconds,
        value.gpu_seconds - start.gpu_seconds,
        (double)value.peak_live_bytes / (1024.0 * 1024.0 * 1024.0),
        (double)h3_hip_counter_delta(value.allocated_bytes,
                                     start.allocated_bytes) /
            (1024.0 * 1024.0 * 1024.0),
        (unsigned long long)h3_hip_counter_delta(value.submissions,
                                                 start.submissions),
        (unsigned long long)h3_hip_counter_delta(value.direct_dispatches,
                                                 start.direct_dispatches),
        (unsigned long long)h3_hip_counter_delta(value.mps_linear_dispatches,
                                                 start.mps_linear_dispatches),
        (unsigned long long)h3_hip_counter_delta(value.mps_conv_dispatches,
                                                 start.mps_conv_dispatches),
        (unsigned long long)h3_hip_counter_delta(value.mps_sdpa_dispatches,
                                                 start.mps_sdpa_dispatches));
}

static struct h3_gpu_tensor *tensor_ptr(const h3_gpu_tensor *tensor) {
    return (struct h3_gpu_tensor *)(void *)tensor;
}

/* Fill a tensor from a file. A pinned tensor is read into directly; a device
 * tensor is read into a recycled pinned staging buffer and uploaded. Chunks are
 * equal so the staging sizes repeat across identically shaped tensors and the
 * pinned pool recycles instead of page-locking. */
static int h3_hip_load_tensor_from_file(struct h3_gpu_tensor *obj, int fd,
                                        const char *path, size_t bytes,
                                        off_t offset) {
    if (!obj->device) {
        return h3_hip_copy_from_file(fd, path, obj->data, bytes, offset);
    }
    if (!bytes) return 1;
    hipStream_t stream = h3_hip_upload();
    if (!stream) return 0;
    size_t chunk = h3_hip_stage_chunk(bytes), block = h3_hip_stage_block(bytes);
    void *staging = h3_hip_stage_take(block);
    if (!staging) return 0;
    int ok = 1;
    for (size_t done = 0; done < bytes && ok; done += chunk) {
        size_t span = bytes - done;
        if (span > chunk) span = chunk;
        ok = h3_hip_copy_from_file(fd, path, staging, span,
                                   offset + (off_t)done);
        if (!ok) break;
        double start = h3_hip_now();
        ok = hipMemcpyAsync((unsigned char *)obj->data + done, staging, span,
                            hipMemcpyHostToDevice, stream) == hipSuccess &&
             hipStreamSynchronize(stream) == hipSuccess;
        h3_hip_load_account(&h3_hip_load_upload_seconds,
                            &h3_hip_load_upload_bytes, h3_hip_now() - start,
                            span);
    }
    h3_hip_stage_give(staging, block);
    return ok;
}

static struct h3_gpu *gpu_ptr(const h3_gpu *gpu) {
    return (struct h3_gpu *)(void *)gpu;
}

static void h3_hip_set_error(struct h3_gpu *gpu, const char *format, ...) {
    if (!gpu) return;
    va_list args;
    va_start(args, format);
    vsnprintf(gpu->last_error, sizeof(gpu->last_error), format, args);
    va_end(args);
}

int h3_hip_unimplemented(struct h3_gpu *gpu, const char *name);

/* `overwritten` tells the allocator that the caller fills every byte before
 * anything reads the buffer, which lets a recycled block skip the clear.
 * `weight` asks for a device allocation from the VRAM carveout, which is right
 * for a buffer the GPU reads and host code only ever fills from a file. */
static struct h3_gpu_tensor *h3_hip_tensor_new_ex(struct h3_gpu *gpu,
                                                  const void *values,
                                                  size_t elements,
                                                  size_t item_size,
                                                  h3_gpu_dtype dtype,
                                                  int overwritten,
                                                  int weight) {
    if (!gpu || elements > SIZE_MAX / item_size) {
        return NULL;
    }
    size_t bytes = elements * item_size;
    struct h3_gpu_tensor *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) {
        h3_hip_set_error(gpu, "out of memory allocating tensor metadata");
        return NULL;
    }
    if (!h3_hip_bind_thread_device()) {
        free(tensor);
        h3_hip_set_error(gpu, "cannot bind HIP device %d",
                         h3_hip_device_index());
        return NULL;
    }
    double alloc_start = h3_hip_now();
    size_t request = bytes > 0 ? bytes : 1;
    if (weight && h3_hip_device_weights() &&
        h3_hip_device_weight_fits(request) &&
        hipMalloc(&tensor->data, request) == hipSuccess) {
        tensor->device = 1;
        /* hipHostMalloc hands back zeroed pages and callers depend on it;
         * hipMalloc does not, so match it explicitly when nobody is going to
         * overwrite the buffer first. */
        if (!overwritten && !values) {
            if (hipMemsetAsync(tensor->data, 0, request, h3_hip_upload()) !=
                    hipSuccess ||
                hipStreamSynchronize(h3_hip_upload()) != hipSuccess) {
                hipFree(tensor->data);
                free(tensor);
                h3_hip_set_error(gpu, "cannot clear %zu-byte device buffer",
                                 bytes);
                return NULL;
            }
        }
    } else {
        tensor->data = h3_hip_pin_take(request);
        if (tensor->data) {
            if (!overwritten && !values) memset(tensor->data, 0, request);
        } else if (hipHostMalloc(&tensor->data, request, hipHostMallocDefault)
                   != hipSuccess) {
            free(tensor);
            h3_hip_set_error(gpu, "cannot allocate %zu-byte HIP buffer", bytes);
            return NULL;
        }
    }
    h3_hip_load_account(&h3_hip_load_alloc_seconds, &h3_hip_load_alloc_bytes,
                        h3_hip_now() - alloc_start, bytes);
    h3_hip_load_count(tensor->device ? &h3_hip_load_device_bytes
                                     : &h3_hip_load_pinned_bytes, bytes);
    if (values && bytes) {
        if (tensor->device) {
            if (!h3_hip_upload_bytes(tensor->data, values, bytes)) {
                hipFree(tensor->data);
                free(tensor);
                h3_hip_set_error(gpu, "cannot upload %zu bytes to device",
                                 bytes);
                return NULL;
            }
        } else {
            memcpy(tensor->data, values, bytes);
        }
    }
    tensor->elements = elements;
    tensor->bytes = bytes;
    tensor->dtype = dtype;
    tensor->owner = gpu;
    gpu->stats.allocated_bytes += bytes;
    gpu->stats.live_bytes += bytes;
    if (gpu->stats.live_bytes > gpu->stats.peak_live_bytes) {
        gpu->stats.peak_live_bytes = gpu->stats.live_bytes;
    }
    gpu->stats.tensor_allocations++;
    return tensor;
}

static struct h3_gpu_tensor *h3_hip_tensor_new(struct h3_gpu *gpu,
                                               const void *values,
                                               size_t elements,
                                               size_t item_size,
                                               h3_gpu_dtype dtype) {
    return h3_hip_tensor_new_ex(gpu, values, elements, item_size, dtype, 0, 0);
}

static int h3_hip_require_bf16(struct h3_gpu *gpu, const h3_gpu_tensor *tensor,
                               size_t elements, const char *label) {
    if (!tensor || tensor_ptr(tensor)->dtype != H3_GPU_BF16) {
        h3_hip_set_error(gpu, "%s tensor dtype mismatch", label);
        return 0;
    }
    if (elements && h3_gpu_tensor_elements(tensor) < elements) {
        h3_hip_set_error(gpu, "%s tensor is too small", label);
        return 0;
    }
    return 1;
}

static int h3_hip_require_f32(struct h3_gpu *gpu, const h3_gpu_tensor *tensor,
                              size_t elements, const char *label) {
    if (!tensor || tensor_ptr(tensor)->dtype != H3_GPU_F32) {
        h3_hip_set_error(gpu, "%s tensor dtype mismatch", label);
        return 0;
    }
    if (elements && h3_gpu_tensor_elements(tensor) < elements) {
        h3_hip_set_error(gpu, "%s tensor is too small", label);
        return 0;
    }
    return 1;
}

h3_gpu *h3_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size) {
    (void)shader_source_path;
    struct h3_gpu *gpu = calloc(1, sizeof(*gpu));
    if (!gpu) {
        if (error && error_size) {
            snprintf(error, error_size, "out of memory creating HIP context");
        }
        return NULL;
    }
    int device = h3_hip_device_index();
    if (hipSetDevice(device) != hipSuccess ||
        hipStreamCreate(&gpu->stream) != hipSuccess) {
        if (error && error_size) {
            snprintf(error, error_size, "cannot initialize HIP device/stream");
        }
        free(gpu);
        return NULL;
    }
    gpu->device_id = device;
    snprintf(gpu->profile_label, sizeof(gpu->profile_label), "HIP context");
    gpu->profile_start_wall = h3_hip_now();
    gpu->profile_start_stats = gpu->stats;
    gpu->profile_mark_wall = gpu->profile_start_wall;
    gpu->profile_mark_stats = gpu->stats;
    return (h3_gpu *)gpu;
}

void h3_gpu_free(h3_gpu *gpu) {
    if (!gpu) return;
    struct h3_gpu *ctx = gpu_ptr(gpu);
    h3_hip_profile_flush_ops(ctx);
    h3_hip_profile_emit(ctx, "total", ctx->profile_start_stats,
                        ctx->profile_start_wall);
    h3_hip_profile_emit_ops(ctx);
    h3_hip_profile_emit_load(ctx);
    h3_hip_profile_destroy_ops(ctx);
    if (ctx->kv_hm_scratch) hipHostFree(ctx->kv_hm_scratch);
    if (ctx->nax_fc1_temp) h3_gpu_tensor_free((h3_gpu_tensor *)ctx->nax_fc1_temp);
    hipStreamDestroy(ctx->stream);
    free(ctx);
    h3_hip_pin_purge();
}

int h3_gpu_is_m5(const h3_gpu *gpu) {
    (void)gpu;
    return 0;
}

int h3_gpu_has_nax_mlp(const h3_gpu *gpu) {
    (void)gpu;
    return 1;
}

int h3_gpu_has_int8_mlp(const h3_gpu *gpu) {
    (void)gpu;
    const char *e = getenv("H3_INT8_MLP");
    if (e && strcmp(e, "0") == 0) return 0;
    if (e && strcmp(e, "1") == 0) return 1;
    /* gfx90a: hipBLAS BF16 GEMM is faster than INT8+epilogue. RDNA keeps INT8. */
    return h3_hip_rdna_wmma_default();
}

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements) {
    return (h3_gpu_tensor *)h3_hip_tensor_new(gpu_ptr(gpu), NULL, elements,
                                            sizeof(float), H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements) {
    return (h3_gpu_tensor *)h3_hip_tensor_new(gpu_ptr(gpu), NULL, elements,
                                            sizeof(uint16_t), H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements) {
    return (h3_gpu_tensor *)h3_hip_tensor_new(gpu_ptr(gpu), NULL, elements,
                                            sizeof(int8_t), H3_GPU_I8);
}

/* The loaders pre-allocate this and then fill it from a file on an I/O thread,
 * so it belongs in the carveout and needs no clear. */
h3_gpu_tensor *h3_gpu_tensor_new_bf16_device(h3_gpu *gpu, size_t elements) {
    return (h3_gpu_tensor *)h3_hip_tensor_new_ex(
        gpu_ptr(gpu), NULL, elements, sizeof(uint16_t), H3_GPU_BF16, 1, 1);
}

/* Kernel destinations that are later re-read by kernels. These keep the clear,
 * so they behave exactly like the pinned allocation they replace even if a
 * producing kernel turns out not to cover every element. */
h3_gpu_tensor *h3_gpu_tensor_new_i8_device(h3_gpu *gpu, size_t elements) {
    return (h3_gpu_tensor *)h3_hip_tensor_new_ex(
        gpu_ptr(gpu), NULL, elements, sizeof(int8_t), H3_GPU_I8, 0, 1);
}

h3_gpu_tensor *h3_gpu_tensor_new_f32_device(h3_gpu *gpu, size_t elements) {
    return (h3_gpu_tensor *)h3_hip_tensor_new_ex(
        gpu_ptr(gpu), NULL, elements, sizeof(float), H3_GPU_F32, 0, 1);
}

h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements) {
    return (h3_gpu_tensor *)h3_hip_tensor_new(gpu_ptr(gpu), values, elements,
                                            sizeof(float), H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements) {
    return (h3_gpu_tensor *)h3_hip_tensor_new(gpu_ptr(gpu), values, elements,
                                            sizeof(uint16_t), H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements) {
    return (h3_gpu_tensor *)h3_hip_tensor_new(gpu_ptr(gpu), values, elements,
                                            sizeof(uint32_t), H3_GPU_U32);
}

h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    h3_gpu_tensor *tensor = (h3_gpu_tensor *)h3_hip_tensor_new_ex(
        ctx, NULL, elements, sizeof(uint16_t), H3_GPU_BF16, 1, 1);
    if (!tensor) return NULL;
    char detail[256];
    if (!h3_gpu_tensor_read_file_bf16(tensor, path, file_offset, elements,
                                      detail, sizeof(detail))) {
        h3_hip_set_error(ctx, "%s", detail);
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    return tensor;
}

h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    h3_gpu_tensor *tensor = (h3_gpu_tensor *)h3_hip_tensor_new_ex(
        ctx, NULL, elements, sizeof(float), H3_GPU_F32, 1, 1);
    if (!tensor) return NULL;
    if (elements > SIZE_MAX / sizeof(float)) {
        h3_gpu_tensor_free(tensor);
        h3_hip_set_error(ctx, "F32 weight load size overflow");
        return NULL;
    }
    int from_cache = 0;
    int fd = h3_hip_open_weight_fd(path, &from_cache);
    if (fd < 0) {
        h3_gpu_tensor_free(tensor);
        h3_hip_set_error(ctx, "cannot open %s: %s", path, strerror(errno));
        return NULL;
    }
    int ok = h3_hip_load_tensor_from_file(tensor_ptr(tensor), fd, path,
                                          elements * sizeof(float),
                                          (off_t)file_offset);
    if (!from_cache) close(fd);
    if (!ok) {
        h3_gpu_tensor_free(tensor);
        h3_hip_set_error(ctx, "cannot read %zu F32 values from %s", elements,
                         path);
        return NULL;
    }
    return tensor;
}

int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || !path || !*path || elements > obj->elements) {
        if (error && error_size) {
            snprintf(error, error_size, "invalid BF16 file read request");
        }
        return 0;
    }
    int from_cache = 0;
    int fd = h3_hip_open_weight_fd(path, &from_cache);
    if (fd < 0) {
        if (error && error_size) {
            snprintf(error, error_size, "cannot open %s: %s", path,
                     strerror(errno));
        }
        return 0;
    }
    int ok = h3_hip_load_tensor_from_file(obj, fd, path,
                                          elements * sizeof(uint16_t),
                                          (off_t)file_offset);
    if (!from_cache) close(fd);
    if (!ok) {
        if (error && error_size) {
            snprintf(error, error_size, "cannot read %zu BF16 values from %s",
                     elements, path);
        }
        return 0;
    }
    return 1;
}

int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size) {
    return h3_gpu_tensor_read_file_bf16(tensor, path, file_offset, elements,
                                       error, error_size);
}

void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj) return;
    struct h3_gpu *gpu = obj->owner;
    if (obj->data && obj->device) {
        /* hipFree synchronizes the device, so like hipHostFree it already waits
         * for any queued kernel still reading the buffer. Device allocation runs
         * at ~91 GiB/s, so there is nothing to gain from a free list here and
         * none of the retirement bookkeeping the pinned pool needs. */
        hipFree(obj->data);
    } else if (obj->data) {
        size_t request = obj->bytes > 0 ? obj->bytes : 1;
        int idle = gpu && hipStreamQuery(gpu->stream) == hipSuccess;
        if (!h3_hip_pin_give(obj->data, request, idle))
            hipHostFree(obj->data);
    }
    if (gpu) {
        gpu->stats.live_bytes -= obj->bytes;
    }
    free(obj);
}

size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor) {
    return tensor ? tensor_ptr(tensor)->elements : 0;
}

h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor) {
    return tensor ? tensor_ptr(tensor)->dtype : H3_GPU_F32;
}

/* These accessors are the only places host code touches tensor contents, so
 * they are also the only places a device-resident tensor needs a bounce. */
static int h3_hip_tensor_out(const struct h3_gpu_tensor *obj, size_t offset,
                             void *values, size_t bytes) {
    if (!obj->device) {
        memcpy(values, (const unsigned char *)obj->data + offset, bytes);
        return 1;
    }
    /* Only the loaders write a device tensor, and they synchronize the upload
     * before returning, so the contents are already visible. Draining the
     * owning stream as well keeps this correct if a device buffer ever becomes
     * a kernel destination. */
    if (obj->owner) hipStreamSynchronize(obj->owner->stream);
    return h3_hip_download_bytes(
        values, (const unsigned char *)obj->data + offset, bytes);
}

static int h3_hip_tensor_in(struct h3_gpu_tensor *obj, size_t offset,
                            const void *values, size_t bytes) {
    if (!obj->device) {
        memcpy((unsigned char *)obj->data + offset, values, bytes);
        return 1;
    }
    return h3_hip_upload_bytes((unsigned char *)obj->data + offset, values,
                               bytes);
}

int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || obj->dtype != H3_GPU_F32 || !values ||
        elements > obj->elements) {
        return 0;
    }
    return h3_hip_tensor_out(obj, 0, values, elements * sizeof(float));
}

int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || obj->dtype != H3_GPU_F32 || !values ||
        source_offset > obj->elements ||
        elements > obj->elements - source_offset) {
        return 0;
    }
    return h3_hip_tensor_out(obj, source_offset * sizeof(float), values,
                             elements * sizeof(float));
}

int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || obj->dtype != H3_GPU_BF16 || !values ||
        elements > obj->elements) {
        return 0;
    }
    return h3_hip_tensor_out(obj, 0, values, elements * sizeof(uint16_t));
}

int h3_gpu_tensor_read_i8(const h3_gpu_tensor *tensor, int8_t *values,
                          size_t elements) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || obj->dtype != H3_GPU_I8 || !values ||
        elements > obj->elements) {
        return 0;
    }
    return h3_hip_tensor_out(obj, 0, values, elements * sizeof(int8_t));
}

int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || obj->dtype != H3_GPU_F32 || !values ||
        elements > obj->elements) {
        return 0;
    }
    return h3_hip_tensor_in(obj, 0, values, elements * sizeof(float));
}

int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || obj->dtype != H3_GPU_F32 || !values ||
        destination_offset > obj->elements ||
        elements > obj->elements - destination_offset) {
        return 0;
    }
    return h3_hip_tensor_in(obj, destination_offset * sizeof(float), values,
                            elements * sizeof(float));
}

int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || obj->dtype != H3_GPU_BF16 || !values ||
        elements > obj->elements) {
        return 0;
    }
    return h3_hip_tensor_in(obj, 0, values, elements * sizeof(uint16_t));
}

int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || obj->dtype != H3_GPU_BF16 || !values ||
        destination_offset > obj->elements ||
        elements > obj->elements - destination_offset) {
        return 0;
    }
    return h3_hip_tensor_in(obj, destination_offset * sizeof(uint16_t), values,
                            elements * sizeof(uint16_t));
}

int h3_gpu_begin(h3_gpu *gpu) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx) return 0;
    ctx->in_command = 1;
    ctx->command_start_wall = h3_hip_now();
    return 1;
}

int h3_gpu_continue(h3_gpu *gpu) {
    return h3_gpu_submit(gpu) && h3_gpu_begin(gpu);
}

int h3_gpu_submit(h3_gpu *gpu) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx) return 0;
    if (!h3_hip_bind_thread_device()) {
        h3_hip_set_error(ctx, "cannot bind HIP device %d",
                         h3_hip_device_index());
        return 0;
    }
    double now = h3_hip_now();
    if (ctx->in_command && ctx->command_start_wall > 0.0) {
        ctx->stats.command_encode_seconds += now - ctx->command_start_wall;
        ctx->command_start_wall = 0.0;
    }
    double wait_start = h3_hip_now();
    hipError_t sync = hipStreamSynchronize(ctx->stream);
    if (sync != hipSuccess) {
        h3_hip_set_error(ctx, "HIP stream synchronization failed: %s",
                         hipGetErrorString(sync));
        return 0;
    }
    ctx->stats.command_wait_seconds += h3_hip_now() - wait_start;
    ctx->stats.gpu_seconds = ctx->stats.command_wait_seconds;
    ctx->stats.submissions++;
    ctx->in_command = 0;
    h3_hip_pin_retire();
    h3_hip_profile_flush_ops(ctx);
    return 1;
}

const char *h3_gpu_error(const h3_gpu *gpu) {
    return gpu ? gpu_ptr(gpu)->last_error : "invalid HIP context";
}

int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats) {
    if (!gpu || !stats) return 0;
    *stats = gpu_ptr(gpu)->stats;
    return 1;
}

void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label) {
    if (!gpu || !label) return;
    snprintf(gpu_ptr(gpu)->profile_label, sizeof(gpu_ptr(gpu)->profile_label),
             "%s", label);
}

void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !phase || !*phase || !h3_hip_profile_enabled()) return;
    h3_hip_profile_flush_ops(ctx);
    /* Emit since the previous mark (Metal parity), not since context create. */
    h3_hip_profile_emit(ctx, phase, ctx->profile_mark_stats,
                        ctx->profile_mark_wall);
    h3_hip_profile_emit_ops(ctx);
    ctx->profile_linear_ms = 0.0;
    ctx->profile_sdpa_ms = 0.0;
    ctx->profile_conv_ms = 0.0;
    ctx->profile_other_ms = 0.0;
    ctx->profile_mark_stats = ctx->stats;
    ctx->profile_mark_wall = h3_hip_now();
}

static int h3_hip_finish_launch(struct h3_gpu *gpu, int ok, const char *name,
                                int category) {
    if (ok) {
        h3_hip_profile_end_op(gpu, category);
        gpu->stats.direct_dispatches++;
        return 1;
    }
    h3_hip_set_error(gpu, "%s launch failed", name);
    return 0;
}

static int h3_hip_finish_conv(struct h3_gpu *gpu, int ok, const char *name) {
    if (!h3_hip_finish_launch(gpu, ok, name, H3_HIP_PROF_CONV)) return 0;
    gpu->stats.mps_conv_dispatches++;
    return 1;
}

static int h3_hip_finish_sdpa(struct h3_gpu *gpu, int ok, const char *name) {
    if (!h3_hip_finish_launch(gpu, ok, name, H3_HIP_PROF_SDPA)) return 0;
    gpu->stats.mps_sdpa_dispatches++;
    return 1;
}

/* Match Metal: F32 GEMMs that would go through MPSGraph count as linear
 * dispatches. The DiT fused patch tile (in=32/96, out=5376) stays a direct
 * kernel on Metal and must not increment this counter. */
static int h3_hip_mps_linear_f32(uint32_t rows, uint32_t input_dim,
                                 uint32_t output_dim) {
    if (rows >= 16u && output_dim == 5376u &&
        (input_dim == 32u || input_dim == 96u)) {
        return 0;
    }
    return rows >= 32u && input_dim >= 256u && output_dim >= 256u;
}

static int h3_hip_finish_linear(struct h3_gpu *gpu, int ok, const char *name,
                                int count_mps) {
    if (!h3_hip_finish_launch(gpu, ok, name, H3_HIP_PROF_LINEAR)) return 0;
    if (count_mps) gpu->stats.mps_linear_dispatches++;
    return 1;
}

/* Comma-operator macros record the start event before the launch expression
 * runs (C evaluates the launch as an argument to the finish helper). */
#define h3_hip_launch_ok(gpu, expr, name) \
    (h3_hip_profile_begin_op(gpu), \
     h3_hip_finish_launch((gpu), (expr), (name), H3_HIP_PROF_OTHER))
#define h3_hip_launch_conv(gpu, expr, name) \
    (h3_hip_profile_begin_op(gpu), h3_hip_finish_conv((gpu), (expr), (name)))
#define h3_hip_launch_sdpa(gpu, expr, name) \
    (h3_hip_profile_begin_op(gpu), h3_hip_finish_sdpa((gpu), (expr), (name)))
#define h3_hip_launch_linear(gpu, expr, name, count_mps) \
    (h3_hip_profile_begin_op(gpu), \
     h3_hip_finish_linear((gpu), (expr), (name), (count_mps)))

int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_f32(ctx, input, elements, "cast input") ||
        !h3_hip_require_bf16(ctx, output, elements, "cast output")) {
        return 0;
    }
    return h3_hip_launch_ok(ctx, h3_launch_cast_f32_to_bf16(
            (const float *)tensor_ptr(input)->data,
            (uint16_t *)tensor_ptr(output)->data, elements, ctx->stream),
            "h3_cast_f32_to_bf16");
}

int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_bf16(ctx, input, elements, "cast input") ||
        !h3_hip_require_f32(ctx, output, elements, "cast output")) {
        return 0;
    }
    return h3_hip_launch_ok(ctx, h3_launch_cast_bf16_to_f32(
            (const uint16_t *)tensor_ptr(input)->data,
            (float *)tensor_ptr(output)->data, elements, ctx->stream),
            "h3_cast_bf16_to_f32");
}

int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_bf16(ctx, left, elements, "add left") ||
        !h3_hip_require_bf16(ctx, right, elements, "add right") ||
        !h3_hip_require_bf16(ctx, output, elements, "add output")) {
        return 0;
    }
    return h3_hip_launch_ok(ctx, h3_launch_add_bf16(
            (const uint16_t *)tensor_ptr(left)->data,
            (const uint16_t *)tensor_ptr(right)->data,
            (uint16_t *)tensor_ptr(output)->data, elements, ctx->stream),
            "h3_add_bf16");
}

int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination,
                     size_t destination_offset,
                     const h3_gpu_tensor *source, size_t source_offset,
                     size_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements || elements > UINT32_MAX ||
        !h3_hip_require_bf16(ctx, source, source_offset + elements,
                             "copy source") ||
        !h3_hip_require_bf16(ctx, destination, destination_offset + elements,
                             "copy destination")) {
        return 0;
    }
    /* Match Metal blit encoding: the copy must wait for prior GPU writes on
     * the same stream. A host memcpy races with in-flight kernels and is how
     * tiled VAE packing was reading stale embeddings. */
    if (!h3_hip_launch_ok(ctx, h3_launch_copy_u16(
            (const uint16_t *)tensor_ptr(source)->data + source_offset,
            (uint16_t *)tensor_ptr(destination)->data + destination_offset,
            (uint32_t)elements, ctx->stream), "h3_copy_u16")) {
        return 0;
    }
    ctx->stats.blit_copies++;
    return 1;
}

static int h3_hip_require_u32(struct h3_gpu *gpu, const h3_gpu_tensor *tensor,
                              size_t elements, const char *label) {
    if (!tensor || tensor_ptr(tensor)->dtype != H3_GPU_U32) {
        h3_hip_set_error(gpu, "%s tensor dtype mismatch", label);
        return 0;
    }
    if (elements && h3_gpu_tensor_elements(tensor) < elements) {
        h3_hip_set_error(gpu, "%s tensor is too small", label);
        return 0;
    }
    return 1;
}

static int h3_hip_require_i8(struct h3_gpu *gpu, const h3_gpu_tensor *tensor,
                             size_t elements, const char *label) {
    if (!tensor || tensor_ptr(tensor)->dtype != H3_GPU_I8) {
        h3_hip_set_error(gpu, "%s tensor dtype mismatch", label);
        return 0;
    }
    if (elements && h3_gpu_tensor_elements(tensor) < elements) {
        h3_hip_set_error(gpu, "%s tensor is too small", label);
        return 0;
    }
    return 1;
}

static int h3_hip_require_f32_tensor(struct h3_gpu *gpu,
                                     const h3_gpu_tensor *tensor,
                                     size_t elements, const char *label) {
    if (!tensor || tensor_ptr(tensor)->dtype != H3_GPU_F32) {
        h3_hip_set_error(gpu, "%s tensor dtype mismatch", label);
        return 0;
    }
    if (elements && h3_gpu_tensor_elements(tensor) < elements) {
        h3_hip_set_error(gpu, "%s tensor is too small", label);
        return 0;
    }
    return 1;
}

int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination,
                    size_t destination_offset, const h3_gpu_tensor *source,
                    size_t source_offset, size_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements || elements > UINT32_MAX ||
        !h3_hip_require_f32(ctx, source, source_offset + elements,
                            "copy source") ||
        !h3_hip_require_f32(ctx, destination, destination_offset + elements,
                             "copy destination")) {
        return 0;
    }
    if (!h3_hip_launch_ok(ctx, h3_launch_copy_f32(
            (const float *)tensor_ptr(source)->data + source_offset,
            (float *)tensor_ptr(destination)->data + destination_offset,
            (uint32_t)elements, ctx->stream), "h3_copy_f32")) {
        return 0;
    }
    ctx->stats.blit_copies++;
    return 1;
}

int h3_gpu_patch_linear_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                                    size_t output_offset,
                                    const h3_gpu_tensor *input,
                                    size_t input_offset,
                                    const h3_gpu_tensor *weight,
                                    const h3_gpu_tensor *bias, uint32_t rows,
                                    uint32_t input_dim, uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!ctx || output_dim != 5376u ||
        (input_dim != 32u && input_dim != 96u) ||
        input_offset + input_count > tensor_ptr(input)->elements ||
        output_offset + output_count > tensor_ptr(output)->elements ||
        !h3_hip_require_f32_tensor(ctx, input, input_offset + input_count,
                                   "patch input") ||
        !h3_hip_require_f32_tensor(ctx, weight, weight_count,
                                   "patch weight") ||
        !h3_hip_require_bf16(ctx, output, output_offset + output_count,
                            "patch output") ||
        (bias && !h3_hip_require_f32_tensor(ctx, bias, output_dim,
                                            "patch bias"))) {
        return 0;
    }
    const float *bias_ptr = bias ?
        (const float *)tensor_ptr(bias)->data :
        (const float *)tensor_ptr(input)->data;
    h3_linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    return h3_hip_launch_ok(ctx, h3_launch_linear_f32_tiled_bf16(
        (const float *)tensor_ptr(input)->data + input_offset,
        (const float *)tensor_ptr(weight)->data, bias_ptr,
        (uint16_t *)tensor_ptr(output)->data + output_offset, &args,
        ctx->stream), "h3_linear_f32_tiled_bf16");
}

int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!ctx || !rows || !input_dim || !output_dim ||
        !h3_hip_require_f32(ctx, input, input_count, "linear input") ||
        !h3_hip_require_f32(ctx, weight, weight_count, "linear weight") ||
        !h3_hip_require_f32(ctx, output, output_count, "linear output") ||
        (bias && !h3_hip_require_f32(ctx, bias, output_dim, "linear bias"))) {
        return 0;
    }
    const float *bias_ptr = bias ?
        (const float *)tensor_ptr(bias)->data :
        (const float *)tensor_ptr(input)->data;
    h3_linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    return h3_hip_launch_linear(ctx, h3_launch_linear_f32(
        (const float *)tensor_ptr(input)->data,
        (const float *)tensor_ptr(weight)->data, bias_ptr,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_linear_f32",
        h3_hip_mps_linear_f32(rows, input_dim, output_dim));
}

int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_patch_linear_bf16_offset(gpu, output, 0, input, 0, weight,
                                           bias, rows, input_dim, output_dim);
}

int h3_gpu_patch_linear_bf16_map(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *bias,
                                 const h3_gpu_tensor *row_map,
                                 uint32_t output_rows, uint32_t rows,
                                 uint32_t input_dim, uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)output_rows * output_dim;
    if (!ctx || output_dim != 5376u ||
        (input_dim != 32u && input_dim != 96u) ||
        !h3_hip_require_f32_tensor(ctx, input, input_count, "mapped patch input") ||
        !h3_hip_require_f32_tensor(ctx, weight, weight_count,
                                   "mapped patch weight") ||
        !h3_hip_require_bf16(ctx, output, output_count, "mapped patch output") ||
        !h3_hip_require_u32(ctx, row_map, rows, "mapped patch row map") ||
        (bias && !h3_hip_require_f32_tensor(ctx, bias, output_dim,
                                            "mapped patch bias"))) {
        return 0;
    }
    const float *bias_ptr = bias ?
        (const float *)tensor_ptr(bias)->data :
        (const float *)tensor_ptr(input)->data;
    h3_linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    return h3_hip_launch_ok(ctx, h3_launch_linear_f32_tiled_bf16_map(
        (const float *)tensor_ptr(input)->data,
        (const float *)tensor_ptr(weight)->data, bias_ptr,
        (uint16_t *)tensor_ptr(output)->data,
        (const uint32_t *)tensor_ptr(row_map)->data, &args, ctx->stream),
        "h3_linear_f32_tiled_bf16_map");
}

int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_bf16(ctx, left, elements, "sub left") ||
        !h3_hip_require_bf16(ctx, right, elements, "sub right") ||
        !h3_hip_require_bf16(ctx, output, elements, "sub output")) {
        return 0;
    }
    return h3_hip_launch_ok(ctx, h3_launch_sub_bf16(
        (const uint16_t *)tensor_ptr(left)->data,
        (const uint16_t *)tensor_ptr(right)->data,
        (uint16_t *)tensor_ptr(output)->data, elements, ctx->stream),
        "h3_sub_bf16");
}

int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_bf16(ctx, input, elements, "SiLU input") ||
        !h3_hip_require_bf16(ctx, output, elements, "SiLU output")) {
        return 0;
    }
    return h3_hip_launch_ok(ctx, h3_launch_silu_bf16(
        (const uint16_t *)tensor_ptr(input)->data,
        (uint16_t *)tensor_ptr(output)->data, elements, ctx->stream),
        "h3_silu_bf16");
}

int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_f32(ctx, input, elements, "SiLU input") ||
        !h3_hip_require_f32(ctx, output, elements, "SiLU output")) {
        return 0;
    }
    return h3_hip_launch_ok(ctx, h3_launch_silu_f32(
        (const float *)tensor_ptr(input)->data,
        (float *)tensor_ptr(output)->data, elements, ctx->stream),
        "h3_silu_f32");
}

int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_bf16(ctx, input, elements, "GELU input") ||
        !h3_hip_require_bf16(ctx, output, elements, "GELU output")) {
        return 0;
    }
    h3_gelu_bf16_args args = {elements, approximate ? 1u : 0u};
    return h3_hip_launch_ok(ctx, h3_launch_gelu_bf16(
        (const uint16_t *)tensor_ptr(input)->data,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_gelu_bf16");
}

int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t fused_count = (size_t)rows * width * 2;
    size_t out_count = (size_t)rows * width;
    if (!ctx || !rows || !width ||
        !h3_hip_require_bf16(ctx, fused, fused_count, "SwiGLU input") ||
        !h3_hip_require_bf16(ctx, output, out_count, "SwiGLU output")) {
        return 0;
    }
    h3_swiglu_args args = {rows, width};
    return h3_hip_launch_ok(ctx, h3_launch_swiglu_bf16(
        (const uint16_t *)tensor_ptr(fused)->data,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_swiglu_bf16");
}

int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t fused_count = (size_t)rows * width * 2;
    size_t out_count = (size_t)rows * width;
    if (!ctx || !rows || !width ||
        !h3_hip_require_f32(ctx, fused, fused_count, "SwiGLU input") ||
        !h3_hip_require_f32(ctx, output, out_count, "SwiGLU output")) {
        return 0;
    }
    h3_swiglu_args args = {rows, width};
    return h3_hip_launch_ok(ctx, h3_launch_swiglu_f32(
        (const float *)tensor_ptr(fused)->data,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_swiglu_f32");
}

int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width ||
        !h3_hip_require_f32(ctx, residual, count, "scale-add residual") ||
        !h3_hip_require_f32(ctx, branch, count, "scale-add branch") ||
        !h3_hip_require_f32(ctx, scale, width, "scale-add scale") ||
        !h3_hip_require_f32(ctx, output, count, "scale-add output")) {
        return 0;
    }
    h3_swiglu_args args = {rows, width};
    return h3_hip_launch_ok(ctx, h3_launch_scale_add_f32(
        (const float *)tensor_ptr(residual)->data,
        (const float *)tensor_ptr(branch)->data,
        (const float *)tensor_ptr(scale)->data,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_scale_add_f32");
}

int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_f32(ctx, left, elements, "add-scaled left") ||
        !h3_hip_require_f32(ctx, right, elements, "add-scaled right") ||
        !h3_hip_require_f32(ctx, output, elements, "add-scaled output")) {
        return 0;
    }
    h3_add_scaled_f32_args args = {elements, left_scale, right_scale};
    return h3_hip_launch_ok(ctx, h3_launch_add_scaled_f32(
        (const float *)tensor_ptr(left)->data,
        (const float *)tensor_ptr(right)->data,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_add_scaled_f32");
}

int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_f32(ctx, input, elements, "clip input") ||
        !h3_hip_require_f32(ctx, output, elements, "clip output")) {
        return 0;
    }
    h3_clip_f32_args args = {elements, minimum, maximum};
    return h3_hip_launch_ok(ctx, h3_launch_clip_f32(
        (const float *)tensor_ptr(input)->data,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_clip_f32");
}

int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)outer * inner;
    if (!ctx || !outer || !inner ||
        !h3_hip_require_f32(ctx, vector, count, "weight norm vector") ||
        !h3_hip_require_f32(ctx, magnitude, outer, "weight norm magnitude") ||
        !h3_hip_require_f32(ctx, output, count, "weight norm output")) {
        return 0;
    }
    h3_weight_norm_args args = {outer, inner};
    return h3_hip_launch_ok(ctx, h3_launch_weight_norm_f32(
        (const float *)tensor_ptr(vector)->data,
        (const float *)tensor_ptr(magnitude)->data,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_weight_norm_f32");
}

int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate,
                     const h3_gpu_tensor *linear, uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_f32(ctx, gate, elements, "GeGLU gate") ||
        !h3_hip_require_f32(ctx, linear, elements, "GeGLU linear") ||
        !h3_hip_require_f32(ctx, output, elements, "GeGLU output")) {
        return 0;
    }
    return h3_hip_launch_ok(ctx, h3_launch_geglu_f32(
        (const float *)tensor_ptr(gate)->data,
        (const float *)tensor_ptr(linear)->data,
        (float *)tensor_ptr(output)->data, elements, ctx->stream),
        "h3_geglu_f32");
}

int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width ||
        !h3_hip_require_bf16(ctx, input, count, "RMSNorm input") ||
        !h3_hip_require_bf16(ctx, weight, width, "RMSNorm weight") ||
        !h3_hip_require_bf16(ctx, output, count, "RMSNorm output")) {
        return 0;
    }
    h3_norm_args args = {rows, width, epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_rms_norm_bf16(
        (const uint16_t *)tensor_ptr(input)->data,
        (const uint16_t *)tensor_ptr(weight)->data,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_rms_norm_bf16");
}

int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width ||
        !h3_hip_require_bf16(ctx, input, count, "LayerNorm input") ||
        !h3_hip_require_bf16(ctx, weight, width, "LayerNorm weight") ||
        !h3_hip_require_bf16(ctx, bias, width, "LayerNorm bias") ||
        !h3_hip_require_bf16(ctx, output, count, "LayerNorm output")) {
        return 0;
    }
    h3_norm_args args = {rows, width, epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_layer_norm_bf16(
        (const uint16_t *)tensor_ptr(input)->data,
        (const uint16_t *)tensor_ptr(weight)->data,
        (const uint16_t *)tensor_ptr(bias)->data,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_layer_norm_bf16");
}

int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width ||
        !h3_hip_require_f32(ctx, input, count, "RMSNorm input") ||
        !h3_hip_require_f32(ctx, weight, width, "RMSNorm weight") ||
        !h3_hip_require_f32(ctx, output, count, "RMSNorm output")) {
        return 0;
    }
    h3_norm_args args = {rows, width, epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_rms_norm_f32(
        (const float *)tensor_ptr(input)->data,
        (const float *)tensor_ptr(weight)->data,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_rms_norm_f32");
}

int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width ||
        !h3_hip_require_f32(ctx, input, count, "LayerNorm input") ||
        !h3_hip_require_f32(ctx, weight, width, "LayerNorm weight") ||
        !h3_hip_require_f32(ctx, bias, width, "LayerNorm bias") ||
        !h3_hip_require_f32(ctx, output, count, "LayerNorm output")) {
        return 0;
    }
    h3_norm_args args = {rows, width, epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_layer_norm_f32(
        (const float *)tensor_ptr(input)->data,
        (const float *)tensor_ptr(weight)->data,
        (const float *)tensor_ptr(bias)->data,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_layer_norm_f32");
}

int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!ctx || !rows || !input_dim || !output_dim ||
        !h3_hip_require_bf16(ctx, input, input_count, "linear input") ||
        !h3_hip_require_bf16(ctx, weight, weight_count, "linear weight") ||
        !h3_hip_require_bf16(ctx, output, output_count, "linear output") ||
        (bias && !h3_hip_require_bf16(ctx, bias, output_dim, "linear bias"))) {
        return 0;
    }
    const h3_gpu_tensor *bias_tensor = bias ? bias : input;
    h3_linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    return h3_hip_launch_linear(ctx, h3_launch_linear_bf16(
        (const uint16_t *)tensor_ptr(input)->data,
        (const uint16_t *)tensor_ptr(weight)->data,
        (const uint16_t *)tensor_ptr(bias_tensor)->data,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_linear_bf16", 1);
}

int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width || gate_slot >= slots ||
        !h3_hip_require_bf16(ctx, residual, count, "gate residual") ||
        !h3_hip_require_bf16(ctx, branch, count, "gate branch") ||
        !h3_hip_require_bf16(ctx, modulation, 1, "gate modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "gate row map") ||
        !h3_hip_require_bf16(ctx, output, count, "gate output")) {
        return 0;
    }
    h3_gate_args args = {rows, width, slots, gate_slot};
    return h3_hip_launch_ok(ctx, h3_launch_gate_bf16(
        (const uint16_t *)tensor_ptr(residual)->data,
        (const uint16_t *)tensor_ptr(branch)->data,
        (const uint16_t *)tensor_ptr(modulation)->data,
        (const uint32_t *)tensor_ptr(row_map)->data,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_gate_bf16");
}

int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width || gate_slot >= slots ||
        !h3_hip_require_f32(ctx, residual, count, "gate residual") ||
        !h3_hip_require_f32(ctx, branch, count, "gate branch") ||
        !h3_hip_require_f32(ctx, modulation, 1, "gate modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "gate row map") ||
        !h3_hip_require_f32(ctx, output, count, "gate output")) {
        return 0;
    }
    h3_gate_args args = {rows, width, slots, gate_slot};
    return h3_hip_launch_ok(ctx, h3_launch_gate_f32(
        (const float *)tensor_ptr(residual)->data,
        (const float *)tensor_ptr(branch)->data,
        (const float *)tensor_ptr(modulation)->data,
        (const uint32_t *)tensor_ptr(row_map)->data,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_gate_f32");
}

int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width || shift_slot >= slots ||
        scale_slot >= slots ||
        !h3_hip_require_bf16(ctx, input, count, "AdaLN input") ||
        !h3_hip_require_bf16(ctx, norm_weight, width, "AdaLN norm") ||
        !h3_hip_require_bf16(ctx, modulation, 1, "AdaLN modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "AdaLN row map") ||
        !h3_hip_require_bf16(ctx, output, count, "AdaLN output")) {
        return 0;
    }
    h3_adaln_args args = {rows, width, slots, shift_slot, scale_slot,
                          epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_adaln_bf16(
        (const uint16_t *)tensor_ptr(input)->data,
        (const uint16_t *)tensor_ptr(norm_weight)->data,
        (const uint16_t *)tensor_ptr(modulation)->data,
        (const uint32_t *)tensor_ptr(row_map)->data,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_adaln_bf16");
}

int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width || shift_slot >= slots ||
        scale_slot >= slots ||
        !h3_hip_require_f32(ctx, input, count, "AdaLN input") ||
        !h3_hip_require_f32(ctx, norm_weight, width, "AdaLN norm") ||
        !h3_hip_require_f32(ctx, modulation, 1, "AdaLN modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "AdaLN row map") ||
        !h3_hip_require_f32(ctx, output, count, "AdaLN output")) {
        return 0;
    }
    h3_adaln_args args = {rows, width, slots, shift_slot, scale_slot,
                          epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_adaln_f32(
        (const float *)tensor_ptr(input)->data,
        (const float *)tensor_ptr(norm_weight)->data,
        (const float *)tensor_ptr(modulation)->data,
        (const uint32_t *)tensor_ptr(row_map)->data,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_adaln_f32");
}

int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *norm_weight,
                             const h3_gpu_tensor *modulation,
                             const h3_gpu_tensor *row_map, uint32_t rows,
                             uint32_t width, uint32_t slots,
                             uint32_t shift_slot, uint32_t scale_slot,
                             float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width || shift_slot >= slots ||
        scale_slot >= slots ||
        input_offset > tensor_ptr(input)->elements ||
        count > tensor_ptr(input)->elements - input_offset ||
        !h3_hip_require_bf16(ctx, norm_weight, width, "AdaLN norm") ||
        !h3_hip_require_bf16(ctx, modulation, 1, "AdaLN modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "AdaLN row map") ||
        !h3_hip_require_bf16(ctx, output, count, "AdaLN output")) {
        return 0;
    }
    h3_adaln_args args = {rows, width, slots, shift_slot, scale_slot,
                          epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_adaln_bf16(
        (const uint16_t *)tensor_ptr(input)->data + input_offset,
        (const uint16_t *)tensor_ptr(norm_weight)->data,
        (const uint16_t *)tensor_ptr(modulation)->data,
        (const uint32_t *)tensor_ptr(row_map)->data,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_adaln_bf16_offset");
}

int h3_gpu_gate_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *residual,
                           const h3_gpu_tensor *branch,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *gate_modulation,
                           const h3_gpu_tensor *norm_modulation,
                           const h3_gpu_tensor *row_map, uint32_t rows,
                           uint32_t width, uint32_t slots, uint32_t gate_slot,
                           uint32_t shift_slot, uint32_t scale_slot,
                           float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width || width > 5376u || gate_slot >= slots ||
        shift_slot >= slots || scale_slot >= slots ||
        !h3_hip_require_bf16(ctx, residual, count, "gate AdaLN residual") ||
        !h3_hip_require_bf16(ctx, branch, count, "gate AdaLN branch") ||
        !h3_hip_require_bf16(ctx, norm_weight, width, "gate AdaLN norm") ||
        !h3_hip_require_bf16(ctx, gate_modulation, 1,
                             "gate AdaLN gate modulation") ||
        !h3_hip_require_bf16(ctx, norm_modulation, 1,
                             "gate AdaLN norm modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "gate AdaLN row map") ||
        !h3_hip_require_bf16(ctx, gated_residual, count,
                             "gate AdaLN gated residual") ||
        !h3_hip_require_bf16(ctx, output, count, "gate AdaLN output")) {
        return 0;
    }
    h3_gate_adaln_args args = {rows, width, slots, gate_slot, shift_slot,
                               scale_slot, epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_gate_adaln_bf16(
        (const uint16_t *)tensor_ptr(residual)->data,
        (const uint16_t *)tensor_ptr(branch)->data,
        (const uint16_t *)tensor_ptr(gate_modulation)->data,
        (const uint32_t *)tensor_ptr(row_map)->data,
        (const uint16_t *)tensor_ptr(norm_weight)->data,
        (const uint16_t *)tensor_ptr(norm_modulation)->data,
        (uint16_t *)tensor_ptr(gated_residual)->data,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_gate_adaln_bf16");
}

static int h3_gpu_qkv_rope_bf16_layout(h3_gpu *gpu, h3_gpu_tensor *query,
                                       h3_gpu_tensor *key,
                                       h3_gpu_tensor *value,
                                       const h3_gpu_tensor *qkv,
                                       const h3_gpu_tensor *q_norm,
                                       const h3_gpu_tensor *k_norm,
                                       const h3_gpu_tensor *rope_cos,
                                       const h3_gpu_tensor *rope_sin,
                                       uint32_t sequence, uint32_t heads,
                                       uint32_t head_dim, uint32_t rope_half,
                                       uint32_t grouped, float epsilon,
                                       uint32_t kv_head_major) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t inner = (size_t)heads * head_dim;
    size_t projected = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!ctx || !sequence || !heads || !head_dim ||
        !h3_hip_require_bf16(ctx, qkv, projected * 3, "QKV input") ||
        !h3_hip_require_bf16(ctx, q_norm, head_dim, "Q norm") ||
        !h3_hip_require_bf16(ctx, k_norm, head_dim, "K norm") ||
        !h3_hip_require_bf16(ctx, rope_cos, rope_count, "RoPE cosine") ||
        !h3_hip_require_bf16(ctx, rope_sin, rope_count, "RoPE sine") ||
        !h3_hip_require_bf16(ctx, query, projected, "query output") ||
        !h3_hip_require_bf16(ctx, key, projected, "key output") ||
        !h3_hip_require_bf16(ctx, value, projected, "value output")) {
        return 0;
    }
    if (getenv("H3_SDPA_NO_KV_HM") &&
        strcmp(getenv("H3_SDPA_NO_KV_HM"), "0") != 0)
        kv_head_major = 0;
    h3_qkv_args args = {sequence, heads, head_dim, rope_half, grouped,
                        epsilon, kv_head_major};
    int ok = h3_hip_launch_ok(ctx, h3_launch_qkv_rope_bf16(
        (const uint16_t *)tensor_ptr(qkv)->data,
        (const uint16_t *)tensor_ptr(q_norm)->data,
        (const uint16_t *)tensor_ptr(k_norm)->data,
        (const uint16_t *)tensor_ptr(rope_cos)->data,
        (const uint16_t *)tensor_ptr(rope_sin)->data,
        (uint16_t *)tensor_ptr(query)->data,
        (uint16_t *)tensor_ptr(key)->data,
        (uint16_t *)tensor_ptr(value)->data, &args, ctx->stream),
        "h3_qkv_rope_bf16");
    if (ok && kv_head_major) ctx->sdpa_kv_already_hm = 1;
    return ok;
}

int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                         h3_gpu_tensor *key, h3_gpu_tensor *value,
                         const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon) {
    return h3_gpu_qkv_rope_bf16_layout(gpu, query, key, value, qkv, q_norm,
                                       k_norm, rope_cos, rope_sin, sequence,
                                       heads, head_dim, rope_half, 0,
                                       epsilon, 0);
}

static int h3_gpu_qkv_rope_f32_layout(h3_gpu *gpu, h3_gpu_tensor *query,
                                      h3_gpu_tensor *key,
                                      h3_gpu_tensor *value,
                                      const h3_gpu_tensor *qkv,
                                      const h3_gpu_tensor *q_norm,
                                      const h3_gpu_tensor *k_norm,
                                      const h3_gpu_tensor *rope_cos,
                                      const h3_gpu_tensor *rope_sin,
                                      uint32_t sequence, uint32_t heads,
                                      uint32_t head_dim, uint32_t rope_half,
                                      uint32_t grouped, float epsilon,
                                      uint32_t kv_head_major) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t inner = (size_t)heads * head_dim;
    size_t projected = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!ctx || !sequence || !heads || !head_dim ||
        !h3_hip_require_f32(ctx, qkv, projected * 3, "QKV input") ||
        !h3_hip_require_f32(ctx, q_norm, head_dim, "Q norm") ||
        !h3_hip_require_f32(ctx, k_norm, head_dim, "K norm") ||
        !h3_hip_require_f32(ctx, rope_cos, rope_count, "RoPE cosine") ||
        !h3_hip_require_f32(ctx, rope_sin, rope_count, "RoPE sine") ||
        !h3_hip_require_f32(ctx, query, projected, "query output") ||
        !h3_hip_require_f32(ctx, key, projected, "key output") ||
        !h3_hip_require_f32(ctx, value, projected, "value output")) {
        return 0;
    }
    if (getenv("H3_SDPA_NO_KV_HM") &&
        strcmp(getenv("H3_SDPA_NO_KV_HM"), "0") != 0)
        kv_head_major = 0;
    h3_qkv_args args = {sequence, heads, head_dim, rope_half, grouped,
                        epsilon, kv_head_major};
    int ok = h3_hip_launch_ok(ctx, h3_launch_qkv_rope_f32(
        (const float *)tensor_ptr(qkv)->data,
        (const float *)tensor_ptr(q_norm)->data,
        (const float *)tensor_ptr(k_norm)->data,
        (const float *)tensor_ptr(rope_cos)->data,
        (const float *)tensor_ptr(rope_sin)->data,
        (float *)tensor_ptr(query)->data,
        (float *)tensor_ptr(key)->data,
        (float *)tensor_ptr(value)->data, &args, ctx->stream),
        "h3_qkv_rope_f32");
    if (ok && kv_head_major) ctx->sdpa_kv_already_hm = 1;
    return ok;
}

int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon) {
    return h3_gpu_qkv_rope_f32_layout(gpu, query, key, value, qkv, q_norm,
                                      k_norm, rope_cos, rope_sin, sequence,
                                      heads, head_dim, rope_half, 0, epsilon,
                                      0);
}

int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                h3_gpu_tensor *key, h3_gpu_tensor *value,
                                const h3_gpu_tensor *qkv,
                                const h3_gpu_tensor *q_norm,
                                const h3_gpu_tensor *k_norm,
                                const h3_gpu_tensor *rope_cos,
                                const h3_gpu_tensor *rope_sin,
                                uint32_t sequence, uint32_t heads,
                                uint32_t head_dim, uint32_t rope_half,
                                float epsilon) {
    return h3_gpu_qkv_rope_bf16_layout(gpu, query, key, value, qkv, q_norm,
                                       k_norm, rope_cos, rope_sin, sequence,
                                       heads, head_dim, rope_half, 1,
                                       epsilon, 0);
}

int h3_gpu_vision_qkv_rope_bf16(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
    h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
    const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
    uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!ctx || !sequence || !heads || !head_dim || !rope_half ||
        rope_half * 2 != head_dim ||
        !h3_hip_require_bf16(ctx, qkv, count * 3, "vision QKV") ||
        !h3_hip_require_bf16(ctx, rope_cos, rope_count, "vision RoPE cos") ||
        !h3_hip_require_bf16(ctx, rope_sin, rope_count, "vision RoPE sin") ||
        !h3_hip_require_bf16(ctx, query, count, "vision query") ||
        !h3_hip_require_bf16(ctx, key, count, "vision key") ||
        !h3_hip_require_bf16(ctx, value, count, "vision value")) {
        return 0;
    }
    h3_qkv_args args = {sequence, heads, head_dim, rope_half, 0, 0.0f, 0};
    return h3_hip_launch_ok(ctx, h3_launch_vision_qkv_rope_bf16(
        (const uint16_t *)tensor_ptr(qkv)->data,
        (const uint16_t *)tensor_ptr(rope_cos)->data,
        (const uint16_t *)tensor_ptr(rope_sin)->data,
        (uint16_t *)tensor_ptr(query)->data,
        (uint16_t *)tensor_ptr(key)->data,
        (uint16_t *)tensor_ptr(value)->data, &args, ctx->stream),
        "h3_vision_qkv_rope_bf16");
}

int h3_gpu_video_qkv_rope_f32(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
    h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
    const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
    uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half,
    float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!ctx || !sequence || !heads || !head_dim || !rope_half ||
        rope_half * 2 > head_dim ||
        !h3_hip_require_f32(ctx, qkv, count * 3, "video QKV") ||
        !h3_hip_require_f32(ctx, rope_cos, rope_count, "video RoPE cos") ||
        !h3_hip_require_f32(ctx, rope_sin, rope_count, "video RoPE sin") ||
        !h3_hip_require_f32(ctx, query, count, "video query") ||
        !h3_hip_require_f32(ctx, key, count, "video key") ||
        !h3_hip_require_f32(ctx, value, count, "video value")) {
        return 0;
    }
    h3_qkv_args args = {sequence, heads, head_dim, rope_half, 0, epsilon, 0};
    uint32_t kv_hm = 1;
    if (getenv("H3_SDPA_NO_KV_HM") &&
        strcmp(getenv("H3_SDPA_NO_KV_HM"), "0") != 0)
        kv_hm = 0;
    args.kv_head_major = kv_hm;
    int ok = h3_hip_launch_ok(ctx, h3_launch_video_qkv_rope_f32(
        (const float *)tensor_ptr(qkv)->data,
        (const float *)tensor_ptr(rope_cos)->data,
        (const float *)tensor_ptr(rope_sin)->data,
        (float *)tensor_ptr(query)->data,
        (float *)tensor_ptr(key)->data,
        (float *)tensor_ptr(value)->data, &args, ctx->stream),
        "h3_video_qkv_rope_f32");
    if (ok && kv_hm) ctx->sdpa_kv_already_hm = 1;
    return ok;
}

int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation) {
    return h3_gpu_conv1d_stride_f32(gpu, output, input, weight, bias, batch,
                                    length, input_channels, output_channels,
                                    kernel, 1, padding, dilation);
}

static uint32_t h3_hip_conv1d_output_length(uint32_t length, uint32_t kernel,
                                            uint32_t stride, uint32_t padding,
                                            uint32_t dilation) {
    uint64_t effective = (uint64_t)dilation * (kernel - 1) + 1;
    if ((uint64_t)length + 2 * padding < effective) return 0;
    return (uint32_t)(((uint64_t)length + 2 * padding - effective) / stride +
                      1);
}

int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t batch,
                             uint32_t length, uint32_t input_channels,
                             uint32_t output_channels, uint32_t kernel,
                             uint32_t stride, uint32_t padding,
                             uint32_t dilation) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    uint32_t output_length = h3_hip_conv1d_output_length(
        length, kernel, stride, padding, dilation);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (!ctx || !batch || !length || !input_channels || !output_channels ||
        !kernel || !stride || !dilation || !output_length ||
        !h3_hip_require_f32(ctx, input, input_count, "Conv1d input") ||
        !h3_hip_require_f32(ctx, weight, weight_count, "Conv1d weight") ||
        !h3_hip_require_f32(ctx, output, output_count, "Conv1d output") ||
        (bias && !h3_hip_require_f32(ctx, bias, output_channels,
                                      "Conv1d bias"))) {
        return 0;
    }
    const float *bias_ptr = bias ?
        (const float *)tensor_ptr(bias)->data :
        (const float *)tensor_ptr(input)->data;
    h3_conv1d_args args = {batch, length, output_length, input_channels,
                           output_channels, kernel, stride, padding, dilation,
                           bias ? 1u : 0u, 0u};
    return h3_hip_launch_conv(ctx, h3_launch_conv1d_f32(
        (const float *)tensor_ptr(input)->data,
        (const float *)tensor_ptr(weight)->data, bias_ptr,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_conv1d_stride_f32");
}

int h3_gpu_conv_transpose1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *input,
                                const h3_gpu_tensor *weight,
                                const h3_gpu_tensor *bias, uint32_t batch,
                                uint32_t length, uint32_t input_channels,
                                uint32_t output_channels, uint32_t kernel,
                                uint32_t stride, uint32_t padding) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!batch || !length || !input_channels || !output_channels || !kernel ||
        !stride || (uint64_t)(length - 1) * stride + kernel < 2 * padding) {
        return 0;
    }
    uint32_t output_length = (uint32_t)((uint64_t)(length - 1) * stride +
                                        kernel - 2 * padding);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)input_channels * output_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (!ctx || !output_length ||
        !h3_hip_require_f32(ctx, input, input_count, "ConvTranspose1d input") ||
        !h3_hip_require_f32(ctx, weight, weight_count,
                            "ConvTranspose1d weight") ||
        !h3_hip_require_f32(ctx, output, output_count,
                            "ConvTranspose1d output") ||
        (bias && !h3_hip_require_f32(ctx, bias, output_channels,
                                    "ConvTranspose1d bias"))) {
        return 0;
    }
    const float *bias_ptr = bias ?
        (const float *)tensor_ptr(bias)->data :
        (const float *)tensor_ptr(input)->data;
    h3_conv1d_args args = {batch, length, output_length, input_channels,
                           output_channels, kernel, stride, padding, 1u,
                           bias ? 1u : 0u, 0u};
    return h3_hip_launch_conv(ctx, h3_launch_conv_transpose1d_f32(
        (const float *)tensor_ptr(input)->data,
        (const float *)tensor_ptr(weight)->data, bias_ptr,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_conv_transpose1d_f32");
}

int h3_gpu_alias_free_snake_f32(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *alpha_log, const h3_gpu_tensor *beta_log,
    const h3_gpu_tensor *upsample_filter,
    const h3_gpu_tensor *downsample_filter, uint32_t batch, uint32_t length,
    uint32_t channels) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)batch * length * channels;
    if (!ctx || !batch || !length || !channels ||
        !h3_hip_require_f32(ctx, input, count, "alias-free snake input") ||
        !h3_hip_require_f32(ctx, output, count, "alias-free snake output") ||
        !h3_hip_require_f32(ctx, alpha_log, channels, "alias-free snake alpha") ||
        !h3_hip_require_f32(ctx, beta_log, channels, "alias-free snake beta") ||
        !h3_hip_require_f32(ctx, upsample_filter, 12,
                            "alias-free snake upsample filter") ||
        !h3_hip_require_f32(ctx, downsample_filter, 12,
                            "alias-free snake downsample filter")) {
        return 0;
    }
    h3_audio_activation_args args = {batch, length, channels};
    return h3_hip_launch_ok(ctx, h3_launch_alias_free_snake_f32(
        (const float *)tensor_ptr(input)->data,
        (const float *)tensor_ptr(alpha_log)->data,
        (const float *)tensor_ptr(beta_log)->data,
        (const float *)tensor_ptr(upsample_filter)->data,
        (const float *)tensor_ptr(downsample_filter)->data,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_alias_free_snake_f32");
}

int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)batch * length * channels;
    if (!ctx || !batch || !length || !channels || count > UINT32_MAX ||
        !h3_hip_require_f32(ctx, input, count, "snake1d input") ||
        !h3_hip_require_f32(ctx, alpha, channels, "snake1d alpha") ||
        !h3_hip_require_f32(ctx, output, count, "snake1d output")) {
        return 0;
    }
    h3_audio_activation_args args = {batch, length, channels};
    return h3_hip_launch_ok(ctx, h3_launch_snake1d_f32(
        (const float *)tensor_ptr(input)->data,
        (const float *)tensor_ptr(alpha)->data,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_snake1d_f32");
}

int h3_gpu_audio_qkv_split_f32(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
    h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
    const h3_gpu_tensor *q_bias, const h3_gpu_tensor *k_bias,
    const h3_gpu_tensor *v_bias, uint32_t batch, uint32_t length,
    uint32_t heads, uint32_t head_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t width = (size_t)heads * head_dim;
    size_t count = (size_t)batch * length * width;
    if (!ctx || !batch || !length || !heads || !head_dim ||
        count > UINT32_MAX ||
        !h3_hip_require_f32(ctx, qkv, count * 3, "audio QKV") ||
        !h3_hip_require_f32(ctx, q_bias, width, "audio Q bias") ||
        !h3_hip_require_f32(ctx, k_bias, width, "audio K bias") ||
        !h3_hip_require_f32(ctx, v_bias, width, "audio V bias") ||
        !h3_hip_require_f32(ctx, query, count, "audio query") ||
        !h3_hip_require_f32(ctx, key, count, "audio key") ||
        !h3_hip_require_f32(ctx, value, count, "audio value")) {
        return 0;
    }
    h3_audio_qkv_args args = {batch, length, heads, head_dim};
    return h3_hip_launch_ok(ctx, h3_launch_audio_qkv_split_f32(
        (const float *)tensor_ptr(qkv)->data,
        (const float *)tensor_ptr(q_bias)->data,
        (const float *)tensor_ptr(k_bias)->data,
        (const float *)tensor_ptr(v_bias)->data,
        (float *)tensor_ptr(query)->data,
        (float *)tensor_ptr(key)->data,
        (float *)tensor_ptr(value)->data, &args, ctx->stream),
        "h3_audio_qkv_split_f32");
}

int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t batch,
                           uint32_t sequence, uint32_t heads,
                           uint32_t head_dim, float scale) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)batch * sequence * heads * head_dim;
    if (!ctx || !batch || !sequence || !heads || !head_dim ||
        !h3_hip_require_f32(ctx, query, count, "causal SDPA query") ||
        !h3_hip_require_f32(ctx, key, count, "causal SDPA key") ||
        !h3_hip_require_f32(ctx, value, count, "causal SDPA value") ||
        !h3_hip_require_f32(ctx, output, count, "causal SDPA output")) {
        return 0;
    }
    h3_sdpa_causal_args args = {batch, sequence, heads, head_dim, scale};
    return h3_hip_launch_sdpa(ctx, h3_launch_sdpa_causal_f32(
        (const float *)tensor_ptr(query)->data,
        (const float *)tensor_ptr(key)->data,
        (const float *)tensor_ptr(value)->data,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_sdpa_causal_f32");
}

int h3_gpu_audio_attention_pool_f32(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *attended,
    uint32_t batch, uint32_t length, uint32_t heads, uint32_t head_dim,
    uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t input_count = (size_t)batch * length * heads * head_dim;
    size_t output_count = (size_t)batch * length * output_dim;
    if (!ctx || !batch || !length || !heads || !head_dim || !output_dim ||
        head_dim % output_dim || output_count > UINT32_MAX ||
        !h3_hip_require_f32(ctx, attended, input_count, "audio attended") ||
        !h3_hip_require_f32(ctx, output, output_count, "audio pooled")) {
        return 0;
    }
    h3_audio_pool_args args = {batch, length, heads, head_dim, output_dim};
    return h3_hip_launch_ok(ctx, h3_launch_audio_attention_pool_f32(
        (const float *)tensor_ptr(attended)->data,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_audio_attention_pool_f32");
}

int h3_gpu_vae_encoder_pad_f32(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
    uint32_t channels, uint32_t depth_front, uint32_t height_before,
    uint32_t height_after, uint32_t width_before, uint32_t width_after) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !batch || !depth || height < 2 || width < 2 || !channels ||
        height_before >= height || height_after >= height ||
        width_before >= width || width_after >= width) {
        return 0;
    }
    uint32_t output_depth = depth + depth_front;
    uint32_t output_height = height + height_before + height_after;
    uint32_t output_width = width + width_before + width_after;
    size_t input_count = (size_t)batch * depth * height * width * channels;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * channels;
    if (!h3_hip_require_f32(ctx, input, input_count, "VAE encoder pad input") ||
        !h3_hip_require_f32(ctx, output, output_count,
                            "VAE encoder pad output")) {
        return 0;
    }
    h3_vae_encoder_pad_args args = {
        batch, depth, height, width, channels, depth_front,
        height_before, height_after, width_before, width_after
    };
    return h3_hip_launch_ok(ctx, h3_launch_vae_encoder_pad_f32(
        (const float *)tensor_ptr(input)->data,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_vae_encoder_pad_f32");
}

int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch, uint32_t depth,
                      uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !batch || !depth || !height || !width || !input_channels ||
        !output_channels || !kernel_depth || !kernel_height || !kernel_width ||
        !stride_depth || !stride_height || !stride_width ||
        depth < kernel_depth || height < kernel_height ||
        width < kernel_width) {
        return 0;
    }
    uint32_t output_depth = (depth - kernel_depth) / stride_depth + 1;
    uint32_t output_height = (height - kernel_height) / stride_height + 1;
    uint32_t output_width = (width - kernel_width) / stride_width + 1;
    size_t input_count =
        (size_t)batch * depth * height * width * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels *
                          kernel_depth * kernel_height * kernel_width;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * output_channels;
    if (!h3_hip_require_f32(ctx, input, input_count, "Conv3d input") ||
        !h3_hip_require_f32(ctx, weight, weight_count, "Conv3d weight") ||
        !h3_hip_require_f32(ctx, output, output_count, "Conv3d output") ||
        (bias && !h3_hip_require_f32(ctx, bias, output_channels,
                                    "Conv3d bias"))) {
        return 0;
    }
    const float *bias_ptr = bias ?
        (const float *)tensor_ptr(bias)->data :
        (const float *)tensor_ptr(input)->data;
    h3_conv3d_args args = {batch, depth, height, width, output_depth,
                          output_height, output_width, input_channels,
                          output_channels, kernel_depth, kernel_height,
                          kernel_width, stride_depth, stride_height,
                          stride_width, bias ? 1u : 0u, 0u, 0u};
    return h3_hip_launch_conv(ctx, h3_launch_conv3d_f32(
        (const float *)tensor_ptr(input)->data,
        (const float *)tensor_ptr(weight)->data, bias_ptr,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_conv3d_f32");
}

int h3_gpu_vae_encoder_group_norm_silu_f32(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch,
    uint32_t depth, uint32_t height, uint32_t width, uint32_t channels,
    uint32_t groups, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)batch * depth * height * width * channels;
    if (!ctx || !batch || !depth || !height || !width || !channels ||
        !groups || channels % groups || !(epsilon > 0.0f) ||
        !h3_hip_require_f32(ctx, input, count, "VAE encoder norm input") ||
        !h3_hip_require_f32(ctx, weight, channels, "VAE encoder norm weight") ||
        !h3_hip_require_f32(ctx, bias, channels, "VAE encoder norm bias") ||
        !h3_hip_require_f32(ctx, output, count, "VAE encoder norm output")) {
        return 0;
    }
    uint64_t rows = (uint64_t)batch * depth * groups;
    if (rows > UINT32_MAX) return 0;
    h3_vae_encoder_norm_args args = {batch, depth, height, width, channels,
                                     groups, epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_vae_encoder_group_norm_silu_f32(
        (const float *)tensor_ptr(input)->data,
        (const float *)tensor_ptr(weight)->data,
        (const float *)tensor_ptr(bias)->data,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_vae_encoder_group_norm_silu_f32");
}

int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)sequence * heads * head_dim;
    if (!ctx || !sequence || !heads || !head_dim ||
        !h3_hip_require_bf16(ctx, query, count, "SDPA query") ||
        !h3_hip_require_bf16(ctx, key, count, "SDPA key") ||
        !h3_hip_require_bf16(ctx, value, count, "SDPA value") ||
        !h3_hip_require_bf16(ctx, output, count, "SDPA output")) {
        return 0;
    }
    h3_sdpa_args args = {sequence, heads, head_dim, scale, 0u, 0u};
    const uint16_t *key_ptr = (const uint16_t *)tensor_ptr(key)->data;
    const uint16_t *value_ptr = (const uint16_t *)tensor_ptr(value)->data;
    /* Default: transpose K/V to head-major for wave SDPA locality.
     * H3_SDPA_KV_HM=1: inputs already head-major. H3_SDPA_NO_KV_HM=1: skip.
     * QKV/RoPE may also set ctx->sdpa_kv_already_hm after writing HM. */
    int already_hm = ctx->sdpa_kv_already_hm ||
                     (getenv("H3_SDPA_KV_HM") &&
                      strcmp(getenv("H3_SDPA_KV_HM"), "0") != 0);
    ctx->sdpa_kv_already_hm = 0;
    int disable_hm = getenv("H3_SDPA_NO_KV_HM") &&
                     strcmp(getenv("H3_SDPA_NO_KV_HM"), "0") != 0;
    if (already_hm) {
        args.kv_head_major = 1u;
    } else if (!disable_hm && (head_dim == 128 || head_dim == 64)) {
        size_t need = count * sizeof(uint16_t) * 2u;
        if (need > ctx->kv_hm_scratch_bytes) {
            void *scratch = NULL;
            if (hipHostMalloc(&scratch, need, hipHostMallocDefault) !=
                hipSuccess) {
                h3_hip_set_error(ctx, "cannot allocate SDPA KV scratch");
                return 0;
            }
            if (ctx->kv_hm_scratch) hipHostFree(ctx->kv_hm_scratch);
            ctx->kv_hm_scratch = scratch;
            ctx->kv_hm_scratch_bytes = need;
        }
        uint16_t *key_hm = (uint16_t *)ctx->kv_hm_scratch;
        uint16_t *value_hm = key_hm + count;
        if (!h3_launch_transpose_shd_hsd_bf16(key_ptr, key_hm, sequence, heads,
                                              head_dim, ctx->stream) ||
            !h3_launch_transpose_shd_hsd_bf16(value_ptr, value_hm, sequence,
                                              heads, head_dim, ctx->stream)) {
            h3_hip_set_error(ctx, "SDPA KV transpose failed");
            return 0;
        }
        key_ptr = key_hm;
        value_ptr = value_hm;
        args.kv_head_major = 1u;
    }
    return h3_hip_launch_sdpa(ctx, h3_launch_sdpa_bf16(
        (const uint16_t *)tensor_ptr(query)->data, key_ptr, value_ptr,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_sdpa_bf16");
}

int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)sequence * heads * head_dim;
    if (!ctx || !sequence || !heads || !head_dim ||
        !h3_hip_require_f32(ctx, query, count, "SDPA query") ||
        !h3_hip_require_f32(ctx, key, count, "SDPA key") ||
        !h3_hip_require_f32(ctx, value, count, "SDPA value") ||
        !h3_hip_require_f32(ctx, output, count, "SDPA output")) {
        return 0;
    }
    h3_sdpa_args args = {sequence, heads, head_dim, scale, 0u, 0u};
    const float *key_ptr = (const float *)tensor_ptr(key)->data;
    const float *value_ptr = (const float *)tensor_ptr(value)->data;
    int already_hm = ctx->sdpa_kv_already_hm ||
                     (getenv("H3_SDPA_KV_HM") &&
                      strcmp(getenv("H3_SDPA_KV_HM"), "0") != 0);
    ctx->sdpa_kv_already_hm = 0;
    int disable_hm = getenv("H3_SDPA_NO_KV_HM") &&
                     strcmp(getenv("H3_SDPA_NO_KV_HM"), "0") != 0;
    if (already_hm) {
        args.kv_head_major = 1u;
    } else if (!disable_hm && (head_dim == 128 || head_dim == 64)) {
        size_t need = count * sizeof(float) * 2u;
        if (need > ctx->kv_hm_scratch_bytes) {
            void *scratch = NULL;
            if (hipHostMalloc(&scratch, need, hipHostMallocDefault) !=
                hipSuccess) {
                h3_hip_set_error(ctx, "cannot allocate SDPA KV scratch");
                return 0;
            }
            if (ctx->kv_hm_scratch) hipHostFree(ctx->kv_hm_scratch);
            ctx->kv_hm_scratch = scratch;
            ctx->kv_hm_scratch_bytes = need;
        }
        float *key_hm = (float *)ctx->kv_hm_scratch;
        float *value_hm = key_hm + count;
        if (!h3_launch_transpose_shd_hsd_f32(key_ptr, key_hm, sequence, heads,
                                             head_dim, ctx->stream) ||
            !h3_launch_transpose_shd_hsd_f32(value_ptr, value_hm, sequence,
                                             heads, head_dim, ctx->stream)) {
            h3_hip_set_error(ctx, "SDPA KV transpose failed");
            return 0;
        }
        key_ptr = key_hm;
        value_ptr = value_hm;
        args.kv_head_major = 1u;
    }
    return h3_hip_launch_sdpa(ctx, h3_launch_sdpa_f32(
        (const float *)tensor_ptr(query)->data, key_ptr, value_ptr,
        (float *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_sdpa_f32");
}

int h3_gpu_sdpa_bf16_head_major_output(h3_gpu *gpu, h3_gpu_tensor *output,
                                       const h3_gpu_tensor *query,
                                       const h3_gpu_tensor *key,
                                       const h3_gpu_tensor *value,
                                       uint32_t sequence, uint32_t heads,
                                       uint32_t head_dim, float scale) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)sequence * heads * head_dim;
    if (!ctx || !sequence || !heads || !head_dim ||
        !h3_hip_require_bf16(ctx, query, count, "SDPA query") ||
        !h3_hip_require_bf16(ctx, key, count, "SDPA key") ||
        !h3_hip_require_bf16(ctx, value, count, "SDPA value") ||
        !h3_hip_require_bf16(ctx, output, count, "SDPA output")) {
        return 0;
    }
    h3_sdpa_args args = {sequence, heads, head_dim, scale, 1u, 0u};
    const uint16_t *key_ptr = (const uint16_t *)tensor_ptr(key)->data;
    const uint16_t *value_ptr = (const uint16_t *)tensor_ptr(value)->data;
    int already_hm = ctx->sdpa_kv_already_hm ||
                     (getenv("H3_SDPA_KV_HM") &&
                      strcmp(getenv("H3_SDPA_KV_HM"), "0") != 0);
    ctx->sdpa_kv_already_hm = 0;
    int disable_hm = getenv("H3_SDPA_NO_KV_HM") &&
                     strcmp(getenv("H3_SDPA_NO_KV_HM"), "0") != 0;
    if (already_hm) {
        args.kv_head_major = 1u;
    } else if (!disable_hm && (head_dim == 128 || head_dim == 64)) {
        size_t need = count * sizeof(uint16_t) * 2u;
        if (need > ctx->kv_hm_scratch_bytes) {
            void *scratch = NULL;
            if (hipHostMalloc(&scratch, need, hipHostMallocDefault) !=
                hipSuccess) {
                h3_hip_set_error(ctx, "cannot allocate SDPA KV scratch");
                return 0;
            }
            if (ctx->kv_hm_scratch) hipHostFree(ctx->kv_hm_scratch);
            ctx->kv_hm_scratch = scratch;
            ctx->kv_hm_scratch_bytes = need;
        }
        uint16_t *key_hm = (uint16_t *)ctx->kv_hm_scratch;
        uint16_t *value_hm = key_hm + count;
        if (!h3_launch_transpose_shd_hsd_bf16(key_ptr, key_hm, sequence, heads,
                                              head_dim, ctx->stream) ||
            !h3_launch_transpose_shd_hsd_bf16(value_ptr, value_hm, sequence,
                                              heads, head_dim, ctx->stream)) {
            h3_hip_set_error(ctx, "SDPA KV transpose failed");
            return 0;
        }
        key_ptr = key_hm;
        value_ptr = value_hm;
        args.kv_head_major = 1u;
    }
    return h3_hip_launch_sdpa(ctx, h3_launch_sdpa_bf16(
        (const uint16_t *)tensor_ptr(query)->data, key_ptr, value_ptr,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_sdpa_bf16_head_major_output");
}

int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !rows || !input_dim || !hidden_dim || !output_dim) {
        return 0;
    }
    h3_gpu_tensor *fc1_out = h3_gpu_tensor_new_bf16_device(
        gpu, (size_t)rows * hidden_dim * 2);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_bf16_device(
        gpu, (size_t)rows * hidden_dim);
    if (!fc1_out || !activated) {
        h3_gpu_tensor_free(fc1_out);
        h3_gpu_tensor_free(activated);
        h3_hip_set_error(ctx, "MLP temporary allocation failed");
        return 0;
    }
    int ok = h3_gpu_linear_bf16(gpu, fc1_out, input, fc1_weight, NULL, rows,
                                input_dim, hidden_dim * 2) &&
             h3_gpu_swiglu_bf16(gpu, activated, fc1_out, rows, hidden_dim) &&
             h3_gpu_linear_bf16(gpu, output, activated, fc2_weight, NULL,
                                rows, hidden_dim, output_dim);
    h3_gpu_tensor_free(fc1_out);
    h3_gpu_tensor_free(activated);
    if (!ok) {
        h3_hip_set_error(ctx, "h3_mlp_bf16 failed");
    }
    return ok;
}

static int h3_hip_fc1_swiglu_nax_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, uint32_t rows, uint32_t input_dim,
    uint32_t hidden_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !rows || !input_dim || !hidden_dim ||
        !h3_hip_require_bf16(ctx, input, (size_t)rows * input_dim,
                             "NAX FC1 input") ||
        !h3_hip_require_bf16(ctx, weight,
                             (size_t)hidden_dim * 2 * input_dim,
                             "NAX FC1 weight") ||
        !h3_hip_require_bf16(ctx, output, (size_t)rows * hidden_dim,
                             "NAX FC1 output")) {
        return 0;
    }
    size_t temp_elems = (size_t)rows * hidden_dim * 2;
    if (!ctx->nax_fc1_temp || ctx->nax_fc1_temp_elems < temp_elems) {
        h3_gpu_tensor_free((h3_gpu_tensor *)ctx->nax_fc1_temp);
        ctx->nax_fc1_temp = h3_gpu_tensor_new_bf16_device(gpu, temp_elems);
        ctx->nax_fc1_temp_elems = ctx->nax_fc1_temp ? temp_elems : 0;
    }
    h3_gpu_tensor *fc1_out = (h3_gpu_tensor *)ctx->nax_fc1_temp;
    if (!fc1_out) {
        h3_hip_set_error(ctx, "NAX FC1 temporary allocation failed");
        return 0;
    }
    int ok = h3_gpu_linear_bf16(gpu, fc1_out, input, weight, NULL, rows,
                                input_dim, hidden_dim * 2) &&
             h3_gpu_swiglu_bf16(gpu, output, fc1_out, rows, hidden_dim);
    if (!ok) {
        h3_hip_set_error(ctx, "h3_fc1_swiglu_nax_bf16 failed");
    }
    return ok;
}

int h3_hip_fc1_swiglu_nax_bf16_dispatch(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, uint32_t rows, uint32_t input_dim,
    uint32_t hidden_dim) {
    return h3_hip_fc1_swiglu_nax_bf16(gpu, output, input, weight, rows,
                                      input_dim, hidden_dim);
}

static int h3_hip_linear_bf16_nax(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, uint32_t rows, uint32_t input_dim,
    uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !rows || !input_dim || !output_dim ||
        !h3_hip_require_bf16(ctx, input, (size_t)rows * input_dim,
                             "NAX linear input") ||
        !h3_hip_require_bf16(ctx, weight, (size_t)output_dim * input_dim,
                             "NAX linear weight") ||
        !h3_hip_require_bf16(ctx, output, (size_t)rows * output_dim,
                             "NAX linear output")) {
        return 0;
    }
    if (!h3_gpu_linear_bf16(gpu, output, input, weight, NULL, rows, input_dim,
                            output_dim)) {
        h3_hip_set_error(ctx, "h3_linear_bf16_nax failed");
        return 0;
    }
    return 1;
}

int h3_hip_linear_bf16_nax_dispatch(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, uint32_t rows, uint32_t input_dim,
    uint32_t output_dim) {
    return h3_hip_linear_bf16_nax(gpu, output, input, weight, rows, input_dim,
                                  output_dim);
}

int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * hidden_dim;
    if (!ctx || !rows || !input_dim || !hidden_dim || !output_dim ||
        !h3_hip_require_bf16(ctx, input, (size_t)rows * input_dim,
                             "NAX MLP input") ||
        !h3_hip_require_bf16(ctx, fc1_weight,
                             (size_t)hidden_dim * 2 * input_dim,
                             "NAX MLP FC1 weight") ||
        !h3_hip_require_bf16(ctx, fc2_weight,
                             (size_t)output_dim * hidden_dim,
                             "NAX MLP FC2 weight") ||
        !h3_hip_require_bf16(ctx, activated, count, "NAX MLP activated") ||
        !h3_hip_require_bf16(ctx, output, (size_t)rows * output_dim,
                             "NAX MLP output")) {
        return 0;
    }
    int ok = h3_hip_fc1_swiglu_nax_bf16(gpu, activated, input, fc1_weight, rows,
                                        input_dim, hidden_dim) &&
             h3_hip_linear_bf16_nax(gpu, output, activated, fc2_weight, rows,
                                    hidden_dim, output_dim);
    if (!ok) {
        h3_hip_set_error(ctx, "h3_mlp_nax_bf16 failed");
    }
    return ok;
}

int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_f32(ctx, sample, sample_offset + elements,
                            "Euler sample") ||
        !h3_hip_require_bf16(ctx, last, elements, "Euler last") ||
        !h3_hip_require_bf16(ctx, previous, elements, "Euler previous")) {
        return 0;
    }
    h3_euler_args args = {(uint32_t)sample_offset, elements, delta, ratio};
    return h3_hip_launch_ok(ctx, h3_launch_euler_bf16(
        (float *)tensor_ptr(sample)->data,
        (const uint16_t *)tensor_ptr(last)->data,
        (const uint16_t *)tensor_ptr(previous)->data, &args, ctx->stream),
        "h3_euler_bf16");
}

int h3_gpu_adaln_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             h3_gpu_tensor *inverse,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *norm_weight,
                             const h3_gpu_tensor *modulation,
                             const h3_gpu_tensor *row_map,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t width, uint32_t output_dim,
                             uint32_t slots, uint32_t shift_slot,
                             uint32_t scale_slot, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t input_count = (size_t)rows * width;
    size_t weight_count = (size_t)output_dim * width;
    size_t output_count = (size_t)rows * output_dim;
    if (!ctx || !rows || !width || !output_dim || shift_slot >= slots ||
        scale_slot >= slots ||
        input_offset + input_count > tensor_ptr(input)->elements ||
        !h3_hip_require_bf16(ctx, norm_weight, width, "adaln linear norm") ||
        !h3_hip_require_bf16(ctx, modulation, 1, "adaln linear modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "adaln linear row map") ||
        !h3_hip_require_bf16(ctx, weight, weight_count, "adaln linear weight") ||
        !h3_hip_require_bf16(ctx, output, output_count, "adaln linear output") ||
        !h3_hip_require_f32(ctx, inverse, rows, "adaln linear inverse") ||
        (bias && !h3_hip_require_bf16(ctx, bias, output_dim,
                                      "adaln linear bias"))) {
        return 0;
    }
    const h3_gpu_tensor *bias_tensor = bias ? bias : input;
    h3_norm_args rms_args = {rows, width, epsilon};
    const uint16_t *input_ptr =
        (const uint16_t *)tensor_ptr(input)->data + input_offset;
    if (!h3_hip_launch_ok(ctx, h3_launch_rms_inverse_bf16(
            input_ptr, (float *)tensor_ptr(inverse)->data, &rms_args,
            ctx->stream), "h3_rms_inverse_bf16")) {
        return 0;
    }
    h3_adaln_linear_args args = {rows, width, output_dim, slots, shift_slot,
                                 scale_slot, bias ? 1u : 0u};
    return h3_hip_launch_ok(ctx, h3_launch_adaln_linear_bf16(
        input_ptr, (const float *)tensor_ptr(inverse)->data,
        (const uint16_t *)tensor_ptr(norm_weight)->data,
        (const uint16_t *)tensor_ptr(modulation)->data,
        (const uint32_t *)tensor_ptr(row_map)->data,
        (const uint16_t *)tensor_ptr(weight)->data,
        (const uint16_t *)tensor_ptr(bias_tensor)->data,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_adaln_linear_bf16");
}

int h3_gpu_grouped_qkv_linear_rope_bf16(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
    h3_gpu_tensor *value, h3_gpu_tensor *qkv, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *q_norm,
    const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos,
    const h3_gpu_tensor *rope_sin, uint32_t rows, uint32_t input_dim,
    uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon) {
    uint32_t inner = heads * head_dim;
    return h3_gpu_linear_bf16(gpu, qkv, input, weight, NULL, rows, input_dim,
                                inner * 3) &&
           h3_gpu_qkv_rope_bf16_layout(gpu, query, key, value, qkv, q_norm,
                                       k_norm, rope_cos, rope_sin, rows,
                                       heads, head_dim, rope_half, 1,
                                       epsilon, 1);
}

int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !tokens || !width || !vocab_size ||
        !h3_hip_require_bf16(ctx, weight, (size_t)vocab_size * width,
                             "embedding weight") ||
        !h3_hip_require_u32(ctx, token_ids, tokens, "embedding token ids") ||
        !h3_hip_require_bf16(ctx, output, (size_t)tokens * width,
                             "embedding output")) {
        return 0;
    }
    h3_embedding_args args = {tokens, vocab_size, width};
    return h3_hip_launch_ok(ctx, h3_launch_embedding_bf16(
        (const uint16_t *)tensor_ptr(weight)->data,
        (const uint32_t *)tensor_ptr(token_ids)->data,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_embedding_bf16");
}

int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate, const h3_gpu_tensor *up,
                         uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_bf16(ctx, gate, elements, "SiLU mul gate") ||
        !h3_hip_require_bf16(ctx, up, elements, "SiLU mul up") ||
        !h3_hip_require_bf16(ctx, output, elements, "SiLU mul output")) {
        return 0;
    }
    return h3_hip_launch_ok(ctx, h3_launch_silu_mul_bf16(
        (const uint16_t *)tensor_ptr(gate)->data,
        (const uint16_t *)tensor_ptr(up)->data,
        (uint16_t *)tensor_ptr(output)->data, elements, ctx->stream),
        "h3_silu_mul_bf16");
}

int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t elements = (size_t)rows * width;
    if (!ctx || !input_rows || !rows || rows > input_rows || !width ||
        !h3_hip_require_bf16(ctx, output, elements, "token pool output") ||
        !h3_hip_require_bf16(ctx, input, input_offset + (size_t)input_rows * width,
                             "token pool input") ||
        !h3_hip_require_bf16(ctx, original,
                             original_offset + (size_t)input_rows * width,
                             "token pool original") ||
        !h3_hip_require_bf16(ctx, baseline,
                             baseline_offset + (size_t)baseline_rows * width,
                             "token pool baseline") ||
        !h3_hip_require_u32(ctx, baseline_indices, rows,
                            "token pool baseline indices") ||
        !h3_hip_require_u32(ctx, pairs, (size_t)rows * 2, "token pool pairs")) {
        return 0;
    }
    h3_token_pool_args args = {
        (uint32_t)input_offset, (uint32_t)original_offset,
        (uint32_t)baseline_offset, rows, width
    };
    return h3_hip_launch_ok(ctx, h3_launch_token_pool_bf16(
        (const uint16_t *)tensor_ptr(input)->data,
        (const uint32_t *)tensor_ptr(pairs)->data,
        (uint16_t *)tensor_ptr(output)->data,
        (uint16_t *)tensor_ptr(baseline)->data,
        (const uint32_t *)tensor_ptr(baseline_indices)->data,
        (uint16_t *)tensor_ptr(original)->data, &args, ctx->stream),
        "h3_token_pool_bf16");
}

int h3_gpu_token_pool_adaln_bf16(
    h3_gpu *gpu, h3_gpu_tensor *residual, h3_gpu_tensor *output,
    const h3_gpu_tensor *input, size_t input_offset, h3_gpu_tensor *original,
    size_t original_offset, h3_gpu_tensor *baseline, size_t baseline_offset,
    const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *pairs,
    const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation,
    const h3_gpu_tensor *row_map, uint32_t input_rows, uint32_t rows,
    uint32_t baseline_rows, uint32_t width, uint32_t slots,
    uint32_t shift_slot, uint32_t scale_slot, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t elements = (size_t)rows * width;
    if (!ctx || !rows || !width || width > 5376u || shift_slot >= slots ||
        scale_slot >= slots ||
        !h3_hip_require_bf16(ctx, residual, elements, "token pool residual") ||
        !h3_hip_require_bf16(ctx, output, elements, "token pool output") ||
        !h3_hip_require_bf16(ctx, input,
                             input_offset + (size_t)input_rows * width,
                             "token pool input") ||
        !h3_hip_require_bf16(ctx, original,
                             original_offset + (size_t)input_rows * width,
                             "token pool original") ||
        !h3_hip_require_bf16(ctx, baseline,
                             baseline_offset + (size_t)baseline_rows * width,
                             "token pool baseline") ||
        !h3_hip_require_u32(ctx, baseline_indices, rows,
                            "token pool baseline indices") ||
        !h3_hip_require_u32(ctx, pairs, (size_t)rows * 2, "token pool pairs") ||
        !h3_hip_require_bf16(ctx, norm_weight, width, "token pool norm") ||
        !h3_hip_require_bf16(ctx, modulation, 1, "token pool modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "token pool row map")) {
        return 0;
    }
    h3_token_pool_adaln_args args = {
        (uint32_t)input_offset, (uint32_t)original_offset,
        (uint32_t)baseline_offset, rows, width, slots, shift_slot,
        scale_slot, epsilon
    };
    return h3_hip_launch_ok(ctx, h3_launch_token_pool_adaln_bf16(
        (const uint16_t *)tensor_ptr(input)->data,
        (const uint32_t *)tensor_ptr(pairs)->data,
        (uint16_t *)tensor_ptr(residual)->data,
        (uint16_t *)tensor_ptr(baseline)->data,
        (const uint32_t *)tensor_ptr(baseline_indices)->data,
        (uint16_t *)tensor_ptr(original)->data,
        (const uint16_t *)tensor_ptr(norm_weight)->data,
        (const uint16_t *)tensor_ptr(modulation)->data,
        (const uint32_t *)tensor_ptr(row_map)->data,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_token_pool_adaln_bf16");
}

int h3_gpu_token_expand_delta_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *original,
    size_t original_offset, const h3_gpu_tensor *reduced,
    const h3_gpu_tensor *baseline, size_t baseline_offset,
    const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *parents,
    uint32_t rows, uint32_t reduced_rows, uint32_t baseline_rows,
    uint32_t width, uint32_t exact_prefix_rows, float update_scale) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t elements = (size_t)rows * width;
    if (!ctx || !rows || !width ||
        !h3_hip_require_bf16(ctx, output, elements, "token expand output") ||
        !h3_hip_require_bf16(ctx, original, original_offset + elements,
                             "token expand original") ||
        !h3_hip_require_bf16(ctx, reduced, (size_t)reduced_rows * width,
                             "token expand reduced") ||
        !h3_hip_require_bf16(ctx, baseline,
                             baseline_offset + (size_t)baseline_rows * width,
                             "token expand baseline") ||
        !h3_hip_require_u32(ctx, baseline_indices, reduced_rows,
                            "token expand baseline indices") ||
        !h3_hip_require_u32(ctx, parents, rows, "token expand parents")) {
        return 0;
    }
    h3_token_expand_args args = {
        (uint32_t)original_offset, (uint32_t)baseline_offset,
        rows, width, exact_prefix_rows, update_scale
    };
    return h3_hip_launch_ok(ctx, h3_launch_token_expand_delta_bf16(
        (const uint16_t *)tensor_ptr(original)->data,
        (const uint16_t *)tensor_ptr(reduced)->data,
        (const uint16_t *)tensor_ptr(baseline)->data,
        (const uint32_t *)tensor_ptr(baseline_indices)->data,
        (const uint32_t *)tensor_ptr(parents)->data,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_token_expand_delta_bf16");
}

int h3_gpu_token_expand_adaln_bf16(
    h3_gpu *gpu, h3_gpu_tensor *residual, h3_gpu_tensor *output,
    const h3_gpu_tensor *original, size_t original_offset,
    const h3_gpu_tensor *reduced, const h3_gpu_tensor *baseline,
    size_t baseline_offset, const h3_gpu_tensor *baseline_indices,
    const h3_gpu_tensor *parents, const h3_gpu_tensor *norm_weight,
    const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map,
    uint32_t rows, uint32_t reduced_rows, uint32_t baseline_rows,
    uint32_t width, uint32_t exact_prefix_rows, float update_scale,
    uint32_t slots, uint32_t shift_slot, uint32_t scale_slot,
    float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t elements = (size_t)rows * width;
    if (!ctx || !rows || !width || width > 5376u || shift_slot >= slots ||
        scale_slot >= slots ||
        !h3_hip_require_bf16(ctx, residual, elements,
                             "token expand residual") ||
        !h3_hip_require_bf16(ctx, output, elements, "token expand output") ||
        !h3_hip_require_bf16(ctx, original, original_offset + elements,
                             "token expand original") ||
        !h3_hip_require_bf16(ctx, reduced, (size_t)reduced_rows * width,
                             "token expand reduced") ||
        !h3_hip_require_bf16(ctx, baseline,
                             baseline_offset + (size_t)baseline_rows * width,
                             "token expand baseline") ||
        !h3_hip_require_u32(ctx, baseline_indices, reduced_rows,
                            "token expand baseline indices") ||
        !h3_hip_require_u32(ctx, parents, rows, "token expand parents") ||
        !h3_hip_require_bf16(ctx, norm_weight, width, "token expand norm") ||
        !h3_hip_require_bf16(ctx, modulation, 1, "token expand modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "token expand row map")) {
        return 0;
    }
    h3_token_expand_adaln_args args = {
        (uint32_t)original_offset, (uint32_t)baseline_offset,
        rows, width, exact_prefix_rows, slots, shift_slot, scale_slot,
        update_scale, epsilon
    };
    return h3_hip_launch_ok(ctx, h3_launch_token_expand_adaln_bf16(
        (const uint16_t *)tensor_ptr(original)->data,
        (const uint16_t *)tensor_ptr(reduced)->data,
        (const uint16_t *)tensor_ptr(baseline)->data,
        (const uint32_t *)tensor_ptr(baseline_indices)->data,
        (const uint32_t *)tensor_ptr(parents)->data,
        (uint16_t *)tensor_ptr(residual)->data,
        (const uint16_t *)tensor_ptr(norm_weight)->data,
        (const uint16_t *)tensor_ptr(modulation)->data,
        (const uint32_t *)tensor_ptr(row_map)->data,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_token_expand_adaln_bf16");
}

int h3_gpu_text_qk_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *q_norm,
                             const h3_gpu_tensor *k_norm,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin, uint32_t sequence,
                             uint32_t query_heads, uint32_t kv_heads,
                             uint32_t head_dim, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2);
    if (!ctx || head_dim % 2 || !kv_heads || query_heads % kv_heads ||
        !h3_hip_require_bf16(ctx, query_input, query_count, "text query") ||
        !h3_hip_require_bf16(ctx, key_input, key_count, "text key") ||
        !h3_hip_require_bf16(ctx, q_norm, head_dim, "text Q norm") ||
        !h3_hip_require_bf16(ctx, k_norm, head_dim, "text K norm") ||
        !h3_hip_require_bf16(ctx, rope_cos, rope_count, "text RoPE cosine") ||
        !h3_hip_require_bf16(ctx, rope_sin, rope_count, "text RoPE sine") ||
        !h3_hip_require_bf16(ctx, query_output, query_count,
                             "text query output") ||
        !h3_hip_require_bf16(ctx, key_output, key_count, "text key output")) {
        return 0;
    }
    h3_text_rope_args args = {sequence, query_heads, kv_heads, head_dim,
                              epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_text_qk_rope_bf16(
        (const uint16_t *)tensor_ptr(query_input)->data,
        (const uint16_t *)tensor_ptr(key_input)->data,
        (const uint16_t *)tensor_ptr(q_norm)->data,
        (const uint16_t *)tensor_ptr(k_norm)->data,
        (const uint16_t *)tensor_ptr(rope_cos)->data,
        (const uint16_t *)tensor_ptr(rope_sin)->data,
        (uint16_t *)tensor_ptr(query_output)->data,
        (uint16_t *)tensor_ptr(key_output)->data, &args, ctx->stream),
        "h3_text_qk_rope_bf16");
}

int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight, uint32_t sequence,
                              uint32_t heads, uint32_t head_dim,
                              float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)sequence * heads * head_dim;
    if (!ctx || !sequence || !heads || !head_dim ||
        !h3_hip_require_bf16(ctx, tensor, count, "head norm tensor") ||
        !h3_hip_require_bf16(ctx, weight, head_dim, "head norm weight")) {
        return 0;
    }
    h3_head_norm_args args = {sequence, heads, head_dim, epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_head_rms_norm_bf16(
        (uint16_t *)tensor_ptr(tensor)->data,
        (const uint16_t *)tensor_ptr(weight)->data, &args, ctx->stream),
        "h3_head_rms_norm_bf16");
}

int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32, uint32_t sequence,
                          uint32_t query_heads, uint32_t kv_heads,
                          uint32_t head_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2);
    if (!ctx || head_dim % 2 || !kv_heads || query_heads % kv_heads ||
        !h3_hip_require_bf16(ctx, query, query_count, "RoPE query") ||
        !h3_hip_require_bf16(ctx, key, key_count, "RoPE key") ||
        !h3_hip_require_f32(ctx, rope_cos_f32, rope_count, "RoPE cosine") ||
        !h3_hip_require_f32(ctx, rope_sin_f32, rope_count, "RoPE sine")) {
        return 0;
    }
    h3_text_rope_inplace_args args = {sequence, query_heads, kv_heads,
                                      head_dim};
    return h3_hip_launch_ok(ctx, h3_launch_rope_text_bf16(
        (uint16_t *)tensor_ptr(query)->data,
        (uint16_t *)tensor_ptr(key)->data,
        (const float *)tensor_ptr(rope_cos_f32)->data,
        (const float *)tensor_ptr(rope_sin_f32)->data, &args, ctx->stream),
        "h3_rope_text_bf16");
}

int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t sequence,
                           uint32_t query_heads, uint32_t kv_heads,
                           uint32_t head_dim, float scale) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t kv_count = (size_t)sequence * kv_heads * head_dim;
    if (!ctx || !sequence || !query_heads || !kv_heads || !head_dim ||
        query_heads % kv_heads || head_dim > 128u ||
        !h3_hip_require_bf16(ctx, query, query_count, "GQA query") ||
        !h3_hip_require_bf16(ctx, key, kv_count, "GQA key") ||
        !h3_hip_require_bf16(ctx, value, kv_count, "GQA value") ||
        !h3_hip_require_bf16(ctx, output, query_count, "GQA output")) {
        return 0;
    }
    h3_gqa_args args = {sequence, query_heads, kv_heads, head_dim, scale};
    return h3_hip_launch_sdpa(ctx, h3_launch_gqa_causal_bf16(
        (const uint16_t *)tensor_ptr(query)->data,
        (const uint16_t *)tensor_ptr(key)->data,
        (const uint16_t *)tensor_ptr(value)->data,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_gqa_causal_bf16");
}

int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * columns;
    if (!ctx || !rows || !columns ||
        !h3_hip_require_bf16(ctx, input, count, "BF16 weight to quantize") ||
        !h3_hip_require_i8(ctx, output, count, "int8 quantized output") ||
        !h3_hip_require_f32(ctx, scales, rows, "int8 quantization scales")) {
        return 0;
    }
    h3_int8_quant_args args = {rows, columns, 1.0f};
    return h3_hip_launch_ok(ctx, h3_launch_quantize_bf16_int8_rows(
        (const uint16_t *)tensor_ptr(input)->data,
        (int8_t *)tensor_ptr(output)->data,
        (float *)tensor_ptr(scales)->data, &args, rows, ctx->stream),
        "h3_quantize_bf16_int8_rows");
}

static int h3_hip_quantize_f32_int8_rows(
    struct h3_gpu *ctx, h3_gpu_tensor *output, h3_gpu_tensor *scales,
    const h3_gpu_tensor *input, uint32_t rows, uint32_t columns) {
    size_t count = (size_t)rows * columns;
    if (!ctx || !rows || !columns ||
        !h3_hip_require_f32(ctx, input, count, "F32 weight to quantize") ||
        !h3_hip_require_i8(ctx, output, count, "int8 quantized output") ||
        !h3_hip_require_f32(ctx, scales, rows, "int8 quantization scales")) {
        return 0;
    }
    h3_int8_quant_args args = {rows, columns, 1.0f};
    return h3_hip_launch_ok(ctx, h3_launch_quantize_f32_int8_rows(
        (const float *)tensor_ptr(input)->data,
        (int8_t *)tensor_ptr(output)->data,
        (float *)tensor_ptr(scales)->data, &args, rows, ctx->stream),
        "h3_quantize_f32_int8_rows");
}

int h3_gpu_quantize_weight_f32_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                    h3_gpu_tensor *scales,
                                    const h3_gpu_tensor *input, uint32_t rows,
                                    uint32_t columns) {
    return h3_hip_quantize_f32_int8_rows(gpu_ptr(gpu), output, scales, input,
                                         rows, columns);
}

int h3_gpu_linear_f32_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *weight_scales,
                           const h3_gpu_tensor *bias,
                           h3_gpu_tensor *quantized_input_ws,
                           h3_gpu_tensor *input_scales_ws,
                           uint32_t rows, uint32_t input_dim,
                           uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!ctx || !rows || !input_dim || !output_dim ||
        !h3_hip_require_f32(ctx, input, input_count, "int8-f32 linear input") ||
        !h3_hip_require_i8(ctx, weight, weight_count, "int8-f32 linear weight") ||
        !h3_hip_require_f32(ctx, weight_scales, output_dim,
                            "int8-f32 linear weight scales") ||
        !h3_hip_require_f32(ctx, output, output_count,
                            "int8-f32 linear output") ||
        (bias && !h3_hip_require_f32(ctx, bias, output_dim,
                                     "int8-f32 linear bias"))) {
        return 0;
    }
    int owns_ws = 0;
    h3_gpu_tensor *quantized_input = quantized_input_ws;
    h3_gpu_tensor *input_scales = input_scales_ws;
    if (!quantized_input || !input_scales) {
        quantized_input = h3_gpu_tensor_new_i8_device(gpu, input_count);
        input_scales = h3_gpu_tensor_new_f32_device(gpu, (size_t)rows);
        owns_ws = 1;
    }
    if (!quantized_input || !input_scales) {
        if (owns_ws) {
            h3_gpu_tensor_free(quantized_input);
            h3_gpu_tensor_free(input_scales);
        }
        h3_hip_set_error(ctx, "int8-f32 linear activation allocation failed");
        return 0;
    }
    int ok = h3_hip_quantize_f32_int8_rows(
        ctx, quantized_input, input_scales, input, rows, input_dim);
    if (ok) {
        h3_linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
        ok = h3_hip_launch_ok(ctx, h3_launch_linear_int8_f32(
            (const int8_t *)tensor_ptr(quantized_input)->data,
            (const int8_t *)tensor_ptr(weight)->data,
            (const float *)tensor_ptr(input_scales)->data,
            (const float *)tensor_ptr(weight_scales)->data,
            (float *)tensor_ptr(output)->data, &args, ctx->stream),
            "h3_linear_int8_f32");
    }
    if (owns_ws) {
        h3_gpu_tensor_free(quantized_input);
        h3_gpu_tensor_free(input_scales);
    }
    return ok;
}

static int h3_hip_launch_linear_int8_prequant(
    struct h3_gpu *ctx, h3_gpu_tensor *output,
    const h3_gpu_tensor *quantized_input, const h3_gpu_tensor *input_scales,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    uint32_t rows, uint32_t input_dim, uint32_t output_dim) {
    h3_linear_args args = {rows, input_dim, output_dim, 0};
    const int8_t *input_ptr =
        (const int8_t *)tensor_ptr(quantized_input)->data;
    const int8_t *weight_ptr = (const int8_t *)tensor_ptr(weight)->data;
    const float *input_scale_ptr =
        (const float *)tensor_ptr(input_scales)->data;
    const float *weight_scale_ptr =
        (const float *)tensor_ptr(weight_scales)->data;
    uint16_t *output_ptr = (uint16_t *)tensor_ptr(output)->data;
    if (!(input_dim % 32) && !getenv("H3_INT8_LEGACY")) {
        return h3_hip_launch_linear(ctx, h3_launch_linear_int8_nax_r64(
            input_ptr, weight_ptr, input_scale_ptr, weight_scale_ptr,
            output_ptr, &args, ctx->stream),
            input_dim == 14336 ? "h3_linear_int8_nax_r64_k14336" :
            input_dim == 5376 ? "h3_linear_int8_nax_r64_k5376" :
                                "h3_linear_int8_nax_r64", 0);
    }
    if (!(input_dim % 128) && !(output_dim % 128)) {
        return h3_hip_launch_linear(ctx, h3_launch_linear_int8_nax_r128(
            input_ptr, weight_ptr, input_scale_ptr, weight_scale_ptr,
            output_ptr, &args, ctx->stream), "h3_linear_int8_nax_r128", 0);
    }
    return h3_hip_launch_linear(ctx, h3_launch_linear_int8_bf16_naive(
        input_ptr, weight_ptr, input_scale_ptr, weight_scale_ptr, output_ptr,
        &args, ctx->stream), "h3_linear_int8_bf16_naive", 0);
}

static int h3_hip_launch_fc1_swiglu_int8_prequant(
    struct h3_gpu *ctx, h3_gpu_tensor *output,
    const h3_gpu_tensor *quantized_input, const h3_gpu_tensor *input_scales,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    uint32_t rows, uint32_t input_dim, uint32_t hidden_dim) {
    if (!(input_dim % 32) && !getenv("H3_INT8_LEGACY")) {
        h3_linear_args args = {rows, input_dim, hidden_dim, 0};
        return h3_hip_launch_linear(ctx, h3_launch_fc1_swiglu_int8_nax_r64(
            (const int8_t *)tensor_ptr(quantized_input)->data,
            (const int8_t *)tensor_ptr(weight)->data,
            (const float *)tensor_ptr(input_scales)->data,
            (const float *)tensor_ptr(weight_scales)->data,
            (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
            input_dim == 5376 ? "h3_fc1_swiglu_int8_nax_r64_k5376" :
                                "h3_fc1_swiglu_int8_nax_r64", 0);
    }
    if (input_dim % 128 || hidden_dim % 128) return 0;
    h3_linear_args args = {rows, input_dim, hidden_dim, 0};
    return h3_hip_launch_linear(ctx, h3_launch_fc1_swiglu_int8_nax_r128(
        (const int8_t *)tensor_ptr(quantized_input)->data,
        (const int8_t *)tensor_ptr(weight)->data,
        (const float *)tensor_ptr(input_scales)->data,
        (const float *)tensor_ptr(weight_scales)->data,
        (uint16_t *)tensor_ptr(output)->data, &args, ctx->stream),
        "h3_fc1_swiglu_int8_nax_r128", 0);
}

static int h3_hip_launch_linear_int8_grouped_prequant(
    struct h3_gpu *ctx, h3_gpu_tensor *output,
    const h3_gpu_tensor *quantized_input, const h3_gpu_tensor *input_scales,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    uint32_t rows, uint32_t input_dim, uint32_t output_dim,
    uint32_t group_size) {
    if (!group_size || input_dim % group_size) return 0;
    h3_linear_int8_grouped_args args = {rows, input_dim, output_dim,
                                        group_size};
    const int8_t *input_ptr =
        (const int8_t *)tensor_ptr(quantized_input)->data;
    const int8_t *weight_ptr = (const int8_t *)tensor_ptr(weight)->data;
    const float *input_scale_ptr =
        (const float *)tensor_ptr(input_scales)->data;
    const float *weight_scale_ptr =
        (const float *)tensor_ptr(weight_scales)->data;
    uint16_t *output_ptr = (uint16_t *)tensor_ptr(output)->data;
    if (!(group_size % 32u) && !getenv("H3_INT8_LEGACY")) {
        return h3_hip_launch_linear(ctx, h3_launch_linear_int8_grouped_nax_r64(
            input_ptr, weight_ptr, input_scale_ptr, weight_scale_ptr,
            output_ptr, &args, ctx->stream),
            input_dim == 14336 && group_size == 1024u ?
                "h3_linear_int8_grouped_nax_r64_k14336" :
                "h3_linear_int8_grouped_nax_r64", 0);
    }
    if (group_size == 1024u && !(input_dim % 128) && !(output_dim % 64)) {
        return h3_hip_launch_linear(ctx, h3_launch_linear_int8_grouped_nax_r128x64(
            input_ptr, weight_ptr, input_scale_ptr, weight_scale_ptr,
            output_ptr, &args, ctx->stream),
            "h3_linear_int8_grouped_nax_r128x64", 0);
    }
    return h3_hip_launch_linear(ctx, h3_launch_linear_int8_grouped_naive(
        input_ptr, weight_ptr, input_scale_ptr, weight_scale_ptr, output_ptr,
        &args, ctx->stream), "h3_linear_int8_grouped_naive", 0);
}

int h3_hip_launch_linear_int8_grouped_dispatch(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *quantized_input,
    const h3_gpu_tensor *input_scales, const h3_gpu_tensor *weight,
    const h3_gpu_tensor *weight_scales, uint32_t rows, uint32_t input_dim,
    uint32_t output_dim, uint32_t group_size) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    uint32_t scale_groups = input_dim / group_size;
    if (!ctx || !group_size || !rows || !input_dim || !output_dim ||
        (input_dim % group_size) ||
        !h3_hip_require_i8(ctx, quantized_input,
                           (size_t)((rows + 127u) & ~127u) * input_dim,
                           "grouped int8 linear input") ||
        !h3_hip_require_f32(ctx, input_scales,
                            (size_t)((rows + 127u) & ~127u) * scale_groups,
                            "grouped int8 linear input scales") ||
        !h3_hip_require_i8(ctx, weight, (size_t)output_dim * input_dim,
                           "grouped int8 linear weight") ||
        !h3_hip_require_f32(ctx, weight_scales, output_dim,
                            "grouped int8 linear weight scales") ||
        !h3_hip_require_bf16(ctx, output, (size_t)rows * output_dim,
                             "grouped int8 linear output")) {
        return 0;
    }
    return h3_hip_launch_linear_int8_grouped_prequant(
        ctx, output, quantized_input, input_scales, weight, weight_scales,
        rows, input_dim, output_dim, group_size);
}

static int h3_hip_quantize_bf16_int8_rows(
    struct h3_gpu *ctx, h3_gpu_tensor *quantized,
    h3_gpu_tensor *scales, const h3_gpu_tensor *input, uint32_t rows,
    uint32_t padded_rows, uint32_t columns) {
    h3_int8_quant_args quant_args = {rows, columns, 1.0f};
    return h3_hip_launch_ok(ctx, h3_launch_quantize_bf16_int8_rows(
        (const uint16_t *)tensor_ptr(input)->data,
        (int8_t *)tensor_ptr(quantized)->data,
        (float *)tensor_ptr(scales)->data, &quant_args, padded_rows,
        ctx->stream), "h3_quantize_bf16_int8_rows");
}

static int h3_hip_quantize_bf16_int8_head_major_rows(
    struct h3_gpu *ctx, h3_gpu_tensor *quantized,
    h3_gpu_tensor *scales, const h3_gpu_tensor *input, uint32_t rows,
    uint32_t padded_rows, uint32_t heads, uint32_t head_dim) {
    h3_int8_head_major_quant_args quant_args = {
        rows, padded_rows, heads, head_dim, 1.0f
    };
    return h3_hip_launch_ok(ctx, h3_launch_quantize_bf16_int8_head_major_rows(
        (const uint16_t *)tensor_ptr(input)->data,
        (int8_t *)tensor_ptr(quantized)->data,
        (float *)tensor_ptr(scales)->data, &quant_args, ctx->stream),
        "h3_quantize_bf16_int8_head_major_rows");
}

int h3_hip_quantize_bf16_int8_groups_dispatch(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *scales,
    const h3_gpu_tensor *input, uint32_t rows, uint32_t padded_rows,
    uint32_t columns, uint32_t group_size) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    uint32_t groups = columns / group_size;
    if (!ctx || !group_size || (group_size % 4u) || !rows || !columns ||
        padded_rows < rows || (columns % group_size) ||
        !h3_hip_require_bf16(ctx, input, (size_t)rows * columns,
                             "grouped int8 input") ||
        !h3_hip_require_i8(ctx, output, (size_t)padded_rows * columns,
                           "grouped int8 output") ||
        !h3_hip_require_f32(ctx, scales, (size_t)padded_rows * groups,
                            "grouped int8 scales")) {
        return 0;
    }
    h3_int8_group_quant_args quant_args = {rows, columns, group_size, groups};
    return h3_hip_launch_ok(ctx, h3_launch_quantize_bf16_int8_groups(
        (const uint16_t *)tensor_ptr(input)->data,
        (int8_t *)tensor_ptr(output)->data,
        (float *)tensor_ptr(scales)->data, &quant_args, padded_rows,
        ctx->stream), "h3_quantize_bf16_int8_groups");
}

static int h3_hip_quantize_bf16_int8_groups(
    struct h3_gpu *ctx, h3_gpu_tensor *quantized,
    h3_gpu_tensor *scales, const h3_gpu_tensor *input, uint32_t rows,
    uint32_t padded_rows, uint32_t columns, uint32_t group_size) {
    uint32_t groups = columns / group_size;
    if (!ctx || !group_size || (group_size % 4u) || !rows || !columns ||
        padded_rows < rows || (columns % group_size) ||
        !h3_hip_require_bf16(ctx, input, (size_t)rows * columns,
                             "grouped int8 input") ||
        !h3_hip_require_i8(ctx, quantized, (size_t)padded_rows * columns,
                           "grouped int8 output") ||
        !h3_hip_require_f32(ctx, scales, (size_t)padded_rows * groups,
                            "grouped int8 scales")) {
        return 0;
    }
    h3_int8_group_quant_args quant_args = {rows, columns, group_size, groups};
    return h3_hip_launch_ok(ctx, h3_launch_quantize_bf16_int8_groups(
        (const uint16_t *)tensor_ptr(input)->data,
        (int8_t *)tensor_ptr(quantized)->data,
        (float *)tensor_ptr(scales)->data, &quant_args, padded_rows,
        ctx->stream), "h3_quantize_bf16_int8_groups");
}

int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    uint32_t padded_rows = (rows + 127u) & ~127u;
    size_t activation_count = (size_t)padded_rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    (void)use_slower_uncached_int8_scales;
    if (!ctx || !rows || !input_dim || !output_dim ||
        !h3_hip_require_bf16(ctx, input, (size_t)rows * input_dim,
                             "int8 linear input") ||
        !h3_hip_require_i8(ctx, weight, weight_count, "int8 linear weight") ||
        !h3_hip_require_f32(ctx, weight_scales, output_dim,
                            "int8 linear weight scales") ||
        !h3_hip_require_i8(ctx, quantized_input, activation_count,
                           "int8 linear quantized input") ||
        !h3_hip_require_f32(ctx, input_scales, padded_rows,
                            "int8 linear input scales") ||
        !h3_hip_require_bf16(ctx, output, output_count, "int8 linear output")) {
        return 0;
    }
    h3_int8_quant_args quant_args = {rows, input_dim, 1.0f};
    if (!h3_hip_launch_ok(ctx, h3_launch_quantize_bf16_int8_rows(
            (const uint16_t *)tensor_ptr(input)->data,
            (int8_t *)tensor_ptr(quantized_input)->data,
            (float *)tensor_ptr(input_scales)->data, &quant_args, padded_rows,
            ctx->stream), "h3_quantize_bf16_int8_rows")) {
        return 0;
    }
    return h3_hip_launch_linear_int8_prequant(
        ctx, output, quantized_input, input_scales, weight, weight_scales,
        rows, input_dim, output_dim);
}

int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         h3_gpu_tensor *activated,
                         h3_gpu_tensor *quantized_activation,
                         h3_gpu_tensor *activation_scales,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *fc1_weight,
                         const h3_gpu_tensor *fc1_scales,
                         const h3_gpu_tensor *fc2_weight,
                         const h3_gpu_tensor *fc2_scales,
                         const h3_gpu_tensor *fc1_bf16, const h3_gpu_tensor *fc2_bf16,
                         uint32_t rows, uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim, int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k, int use_int8_row_fc2,
                         int input_is_quantized, h3_gpu_tensor *fc1_out_ws) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    uint32_t padded_rows = (rows + 127u) & ~127u;
    uint32_t fc2_scale_groups = hidden_dim / 1024u;
    uint32_t fc1_output_dim = hidden_dim * 2u;
    size_t activation_capacity = (size_t)padded_rows *
        (input_dim > hidden_dim ? input_dim : hidden_dim);
    size_t fc1_weight_count = (size_t)fc1_output_dim * input_dim;
    size_t fc2_weight_count = (size_t)output_dim * hidden_dim;
    size_t fc1_out_count = (size_t)rows * fc1_output_dim;
    (void)fc1_bf16;
    (void)fc2_bf16;
    (void)use_slower_grouped_quantizer;
    (void)use_slower_dynamic_fc1_k;
    if (!ctx || !rows || !input_dim || !hidden_dim || !output_dim ||
        !h3_hip_require_i8(ctx, quantized_activation, activation_capacity,
                           "int8 MLP activation") ||
        !h3_hip_require_f32(ctx, activation_scales,
                            (size_t)padded_rows *
                                (fc2_scale_groups > 0 ? fc2_scale_groups : 1u),
                            "int8 MLP activation scales") ||
        !h3_hip_require_i8(ctx, fc1_weight, fc1_weight_count,
                           "int8 MLP FC1 weight") ||
        !h3_hip_require_f32(ctx, fc1_scales, fc1_output_dim,
                            "int8 MLP FC1 scales") ||
        !h3_hip_require_i8(ctx, fc2_weight, fc2_weight_count,
                           "int8 MLP FC2 weight") ||
        !h3_hip_require_f32(ctx, fc2_scales, output_dim,
                            "int8 MLP FC2 scales") ||
        (!input_is_quantized &&
         !h3_hip_require_bf16(ctx, input, (size_t)rows * input_dim,
                              "int8 MLP input")) ||
        !h3_hip_require_bf16(ctx, activated, (size_t)rows * hidden_dim,
                             "int8 MLP activated") ||
        !h3_hip_require_bf16(ctx, output, (size_t)rows * output_dim,
                             "int8 MLP output")) {
        return 0;
    }
    h3_gpu_tensor *fc1_out = fc1_out_ws;
    int own_fc1_out = 0;
    if (!fc1_out) {
        fc1_out = h3_gpu_tensor_new_bf16_device(gpu, fc1_out_count);
        if (!fc1_out) {
            h3_hip_set_error(ctx, "int8 MLP FC1 temporary allocation failed");
            return 0;
        }
        own_fc1_out = 1;
    }
    int ok = 1;
    if (!input_is_quantized &&
        !h3_hip_quantize_bf16_int8_rows(
            ctx, quantized_activation, activation_scales, input, rows,
            padded_rows, input_dim)) {
        ok = 0;
    }
    if (ok &&
        !h3_hip_launch_fc1_swiglu_int8_prequant(
            ctx, activated, quantized_activation, activation_scales,
            fc1_weight, fc1_scales, rows, input_dim, hidden_dim)) {
        if (!h3_hip_launch_linear_int8_prequant(
                ctx, fc1_out, quantized_activation, activation_scales,
                fc1_weight, fc1_scales, rows, input_dim, fc1_output_dim) ||
            !h3_gpu_swiglu_bf16(gpu, activated, fc1_out, rows, hidden_dim)) {
            ok = 0;
        }
    }
    /* Match Metal: grouped quantize and row quantize are independent
     * branches. An else-if here re-quantizes the same buffer with one
     * scale per row after a successful group quantize, then grouped FC2
     * indexes scales as [row][group] and the MLP output becomes noise. */
    int grouped_fc2 = !use_int8_row_fc2 && hidden_dim % 1024u == 0;
    if (ok && grouped_fc2 &&
        !h3_hip_quantize_bf16_int8_groups(
            ctx, quantized_activation, activation_scales, activated, rows,
            padded_rows, hidden_dim, 1024u)) {
        ok = 0;
    }
    if (ok && !grouped_fc2 &&
        !h3_hip_quantize_bf16_int8_rows(
            ctx, quantized_activation, activation_scales, activated, rows,
            padded_rows, hidden_dim)) {
        ok = 0;
    }
    if (ok && grouped_fc2) {
        if (!h3_hip_launch_linear_int8_grouped_prequant(
                ctx, output, quantized_activation, activation_scales,
                fc2_weight, fc2_scales, rows, hidden_dim, output_dim, 1024u)) {
            ok = 0;
        }
    } else if (ok && !h3_hip_launch_linear_int8_prequant(
                   ctx, output, quantized_activation, activation_scales,
                   fc2_weight, fc2_scales, rows, hidden_dim, output_dim)) {
        ok = 0;
    }
    if (own_fc1_out) h3_gpu_tensor_free(fc1_out);
    if (!ok) {
        h3_hip_set_error(ctx, "h3_mlp_int8_bf16 failed");
    }
    return ok;
}

int h3_gpu_linear_int8_head_major_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *quantized_input,
    h3_gpu_tensor *input_scales, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    uint32_t rows, uint32_t heads, uint32_t head_dim, uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    uint32_t input_dim = heads * head_dim;
    uint32_t padded_rows = (rows + 127u) & ~127u;
    size_t activation_count = (size_t)padded_rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!ctx || !rows || !heads || !head_dim || !output_dim ||
        !h3_hip_require_bf16(ctx, input, (size_t)rows * input_dim,
                             "head-major int8 linear input") ||
        !h3_hip_require_i8(ctx, weight, weight_count, "int8 linear weight") ||
        !h3_hip_require_f32(ctx, weight_scales, output_dim,
                            "int8 linear weight scales") ||
        !h3_hip_require_i8(ctx, quantized_input, activation_count,
                           "head-major int8 quantized input") ||
        !h3_hip_require_f32(ctx, input_scales, padded_rows,
                            "head-major int8 input scales") ||
        !h3_hip_require_bf16(ctx, output, output_count,
                             "head-major int8 linear output")) {
        return 0;
    }
    if (!h3_hip_quantize_bf16_int8_head_major_rows(
            ctx, quantized_input, input_scales, input, rows, padded_rows,
            heads, head_dim)) {
        return 0;
    }
    return h3_hip_launch_linear_int8_prequant(
        ctx, output, quantized_input, input_scales, weight, weight_scales,
        rows, input_dim, output_dim);
}

int h3_gpu_grouped_qkv_linear_rope_int8(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
    h3_gpu_tensor *value, h3_gpu_tensor *quantized_input,
    h3_gpu_tensor *input_scales, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm,
    const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
    uint32_t rows, uint32_t input_dim, uint32_t heads, uint32_t head_dim,
    uint32_t rope_half, float epsilon, int input_is_quantized,
    int use_slower_unfused_qkv_rope, int use_slower_scalar_qkv_rms,
    int use_slower_uncached_int8_scales, h3_gpu_tensor *qkv_ws) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    uint32_t inner = heads * head_dim;
    uint32_t qkv_dim = inner * 3u;
    uint32_t padded_rows = (rows + 127u) & ~127u;
    size_t activation_count = (size_t)padded_rows * input_dim;
    size_t weight_count = (size_t)qkv_dim * input_dim;
    size_t projected = (size_t)rows * inner;
    size_t rope_count = (size_t)rows * rope_half;
    size_t qkv_count = (size_t)rows * qkv_dim;
    (void)use_slower_unfused_qkv_rope;
    (void)use_slower_scalar_qkv_rms;
    (void)use_slower_uncached_int8_scales;
    if (!ctx || !rows || !input_dim || !heads || !head_dim || !rope_half ||
        !h3_hip_require_i8(ctx, weight, weight_count, "int8 QKV weight") ||
        !h3_hip_require_f32(ctx, weight_scales, qkv_dim, "int8 QKV scales") ||
        !h3_hip_require_i8(ctx, quantized_input, activation_count,
                           "int8 QKV quantized input") ||
        !h3_hip_require_f32(ctx, input_scales, padded_rows,
                            "int8 QKV input scales") ||
        (!input_is_quantized &&
         !h3_hip_require_bf16(ctx, input, (size_t)rows * input_dim,
                              "int8 QKV input")) ||
        !h3_hip_require_bf16(ctx, q_norm, head_dim, "int8 Q norm") ||
        !h3_hip_require_bf16(ctx, k_norm, head_dim, "int8 K norm") ||
        !h3_hip_require_bf16(ctx, rope_cos, rope_count, "int8 RoPE cos") ||
        !h3_hip_require_bf16(ctx, rope_sin, rope_count, "int8 RoPE sin") ||
        !h3_hip_require_bf16(ctx, query, projected, "int8 query") ||
        !h3_hip_require_bf16(ctx, key, projected, "int8 key") ||
        !h3_hip_require_bf16(ctx, value, projected, "int8 value")) {
        return 0;
    }
    h3_gpu_tensor *qkv = qkv_ws;
    int own_qkv = 0;
    if (!qkv) {
        qkv = h3_gpu_tensor_new_bf16_device(gpu, qkv_count);
        if (!qkv) {
            h3_hip_set_error(ctx, "int8 QKV temporary allocation failed");
            return 0;
        }
        own_qkv = 1;
    }
    int ok = 1;
    if (!input_is_quantized &&
        !h3_hip_quantize_bf16_int8_rows(
            ctx, quantized_input, input_scales, input, rows, padded_rows,
            input_dim)) {
        ok = 0;
    }
    if (ok && !h3_hip_launch_linear_int8_prequant(
            ctx, qkv, quantized_input, input_scales, weight, weight_scales,
            rows, input_dim, qkv_dim)) {
        ok = 0;
    }
    if (ok && !h3_gpu_qkv_rope_bf16_layout(
            gpu, query, key, value, qkv, q_norm, k_norm, rope_cos, rope_sin,
            rows, heads, head_dim, rope_half, 1, epsilon, 1)) {
        ok = 0;
    }
    if (own_qkv) h3_gpu_tensor_free(qkv);
    if (!ok) {
        h3_hip_set_error(ctx, "h3_grouped_qkv_linear_rope_int8 failed");
    }
    return ok;
}

int h3_gpu_gate_adaln_quantize_int8(
    h3_gpu *gpu, h3_gpu_tensor *gated_residual,
    h3_gpu_tensor *quantized_output, h3_gpu_tensor *quantized_scales,
    const h3_gpu_tensor *residual, const h3_gpu_tensor *branch,
    const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *gate_modulation,
    const h3_gpu_tensor *norm_modulation, const h3_gpu_tensor *row_map,
    uint32_t rows, uint32_t padded_rows, uint32_t width, uint32_t slots,
    uint32_t gate_slot, uint32_t shift_slot, uint32_t scale_slot,
    float epsilon, h3_gpu_tensor *adaln_ws) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    size_t padded_count = (size_t)padded_rows * width;
    if (!ctx || !rows || !width || padded_rows < rows || width > 5376u ||
        gate_slot >= slots || shift_slot >= slots || scale_slot >= slots ||
        !h3_hip_require_bf16(ctx, residual, count, "int8 gate AdaLN residual") ||
        !h3_hip_require_bf16(ctx, branch, count, "int8 gate AdaLN branch") ||
        !h3_hip_require_bf16(ctx, norm_weight, width, "int8 gate AdaLN norm") ||
        !h3_hip_require_bf16(ctx, gate_modulation, 1,
                             "int8 gate AdaLN gate modulation") ||
        !h3_hip_require_bf16(ctx, norm_modulation, 1,
                             "int8 gate AdaLN norm modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "int8 gate AdaLN row map") ||
        !h3_hip_require_bf16(ctx, gated_residual, count,
                             "int8 gate AdaLN gated residual") ||
        !h3_hip_require_i8(ctx, quantized_output, padded_count,
                           "int8 gate AdaLN quantized output") ||
        !h3_hip_require_f32(ctx, quantized_scales, padded_rows,
                            "int8 gate AdaLN scales")) {
        return 0;
    }
    h3_gpu_tensor *adaln_out = adaln_ws;
    int own_adaln = 0;
    if (!adaln_out) {
        adaln_out = h3_gpu_tensor_new_bf16_device(gpu, count);
        if (!adaln_out) {
            h3_hip_set_error(ctx, "int8 gate AdaLN temporary allocation failed");
            return 0;
        }
        own_adaln = 1;
    }
    int ok = h3_gpu_gate_adaln_bf16(
        gpu, gated_residual, adaln_out, residual, branch, norm_weight,
        gate_modulation, norm_modulation, row_map, rows, width, slots,
        gate_slot, shift_slot, scale_slot, epsilon) &&
             h3_hip_quantize_bf16_int8_rows(
                 ctx, quantized_output, quantized_scales, adaln_out, rows,
                 padded_rows, width);
    if (own_adaln) h3_gpu_tensor_free(adaln_out);
    if (!ok) {
        h3_hip_set_error(ctx, "h3_gate_adaln_quantize_int8 failed");
    }
    return ok;
}

int h3_hip_unimplemented(struct h3_gpu *gpu, const char *name) {
    h3_hip_set_error(gpu, "HIP backend: %s is not implemented yet", name);
    return 0;
}
