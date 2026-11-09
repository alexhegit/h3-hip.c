#ifndef H3_WEIGHT_CACHE_H
#define H3_WEIGHT_CACHE_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

/* Opt-in disk cache for load-time DiT products that are pure functions of the
 * checkpoint: INT8 core weights and AdaLN timestep tables. A cache hit reads
 * the product instead of the BF16 source tensors it was computed from.
 *
 * H3_WEIGHT_CACHE_DIR=/path enables it. Entries live under a directory named
 * after the checkpoint shards (name, size, mtime), so a changed checkpoint
 * never matches. LoRA merges and VSA rewrite weights after load, so the cache
 * stays off while H3_FASTH3_LORA or H3_VSA is set. */

typedef enum {
    H3_WCACHE_BF16 = 2,
    H3_WCACHE_F32 = 4,
    H3_WCACHE_I8 = 1
} h3_wcache_dtype;

typedef struct {
    h3_gpu_tensor *tensor;
    size_t elements;
    h3_wcache_dtype dtype;
} h3_wcache_part;

/* Writes <H3_WEIGHT_CACHE_DIR>/<checkpoint fingerprint> into root and
 * creates it. Returns 0 when the cache is disabled or unusable. */
int h3_wcache_root(const char *weight_directory, char *root, size_t root_size);

uint64_t h3_wcache_hash(const void *data, size_t bytes, uint64_t seed);

/* 1 when path holds an entry whose header matches parts. */
int h3_wcache_present(const char *path, const h3_wcache_part *parts,
                      size_t count);

/* Fill already-allocated tensors from path. Returns 0 on any mismatch. */
int h3_wcache_read(const char *path, const h3_wcache_part *parts, size_t count);

/* Copy the tensors back from the GPU and write them atomically. Failures
 * leave no entry behind and are not fatal to the caller. */
int h3_wcache_write(const char *path, const h3_wcache_part *parts,
                    size_t count);

#endif
