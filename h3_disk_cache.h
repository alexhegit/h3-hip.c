/* Disk cache for exact intermediate results that are expensive to recompute
 * and cheap to store: the per-step AdaLN schedule and prompt embeddings.
 *
 * An entry is found by a hash of its key, but the full key is stored in the
 * file and compared on load, and the payload carries its own hash, so a
 * collision or a truncated file is a miss, never a wrong result. The key
 * always starts with this executable's size and mtime: a rebuild invalidates
 * everything, so kernel changes can never be hidden by a stale entry.
 *
 * H3_DISK_CACHE=0 disables it. H3_CACHE_DIR picks the directory (default
 * $XDG_CACHE_HOME/h3-hip, else ~/.cache/h3-hip). H3_CACHE_MAX_GIB caps the
 * total size (default 8); the least recently used entries go first. */
#ifndef H3_DISK_CACHE_H
#define H3_DISK_CACHE_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    char *text;
    size_t length;
    size_t capacity;
    int failed;
} h3_disk_key;

int h3_disk_cache_enabled(void);

/* Start a key for one kind of entry ("adaln", "text"). */
void h3_disk_key_init(h3_disk_key *key, const char *kind);
void h3_disk_key_add(h3_disk_key *key, const char *format, ...)
    __attribute__((format(printf, 2, 3)));
/* Content hash of a byte range (128-bit) plus its length. */
void h3_disk_key_add_bytes(h3_disk_key *key, const char *label,
                           const void *data, size_t size);
/* Resolved path, size and mtime of a file. */
void h3_disk_key_add_file(h3_disk_key *key, const char *path);
/* Every file in a directory whose name ends in suffix, in name order. */
void h3_disk_key_add_directory(h3_disk_key *key, const char *directory,
                               const char *suffix);
void h3_disk_key_free(h3_disk_key *key);

/* On a hit, *payload is a malloc'd copy the caller frees. Returns 0 on a
 * miss, a disabled cache, or a failed key. */
int h3_disk_cache_load(const h3_disk_key *key, void **payload, size_t *size);
/* Write the parts back to back as one entry. Failures are silent: the cache
 * is an optimisation, so a full disk must not fail a generation. */
void h3_disk_cache_store(const h3_disk_key *key, const void *const *parts,
                         const size_t *sizes, size_t count);

#endif
