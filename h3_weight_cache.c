#include "h3_weight_cache.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* Payload starts on a page so a future O_DIRECT reader needs no copy. */
enum { H3_WCACHE_HEADER = 4096, H3_WCACHE_MAX_PARTS = 256 };
static const char h3_wcache_magic[8] = {'H', '3', 'W', 'C', 'A', 'C', 'H', '1'};

typedef struct {
    char magic[8];
    uint64_t count;
    uint64_t bytes[H3_WCACHE_MAX_PARTS];
} h3_wcache_header;

uint64_t h3_wcache_hash(const void *data, size_t bytes, uint64_t seed) {
    const unsigned char *p = data;
    uint64_t hash = seed ^ 0xcbf29ce484222325ull;
    for (size_t index = 0; index < bytes; index++) {
        hash ^= p[index];
        hash *= 0x100000001b3ull;
    }
    return hash;
}

static int cache_disabled_by_env(void) {
    const char *lora = getenv("H3_FASTH3_LORA");
    const char *vsa = getenv("H3_VSA");
    return (lora && *lora) || (vsa && *vsa && strcmp(vsa, "0") != 0);
}

static int make_directory(const char *path) {
    char partial[1024];
    size_t length = strlen(path);
    if (length >= sizeof(partial)) return 0;
    memcpy(partial, path, length + 1);
    for (size_t index = 1; index <= length; index++) {
        if (partial[index] != '/' && partial[index] != '\0') continue;
        char saved = partial[index];
        partial[index] = '\0';
        if (mkdir(partial, 0755) != 0 && errno != EEXIST) return 0;
        partial[index] = saved;
    }
    return 1;
}

static int compare_names(const void *left, const void *right) {
    return strcmp(*(const char *const *)left, *(const char *const *)right);
}

static int fingerprint(const char *directory, uint64_t *out) {
    DIR *stream = opendir(directory);
    if (!stream) return 0;
    char *names[512];
    size_t count = 0;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL && count < 512) {
        size_t length = strlen(entry->d_name);
        if (length > 12 &&
            !strcmp(entry->d_name + length - 12, ".safetensors")) {
            names[count] = strdup(entry->d_name);
            if (names[count]) count++;
        }
    }
    closedir(stream);
    if (!count) return 0;
    qsort(names, count, sizeof(*names), compare_names);
    uint64_t hash = 0;
    for (size_t index = 0; index < count; index++) {
        char path[2048];
        struct stat status;
        snprintf(path, sizeof(path), "%s/%s", directory, names[index]);
        if (stat(path, &status) == 0) {
            int64_t facts[2] = {(int64_t)status.st_size,
                                (int64_t)status.st_mtime};
            hash = h3_wcache_hash(names[index], strlen(names[index]), hash);
            hash = h3_wcache_hash(facts, sizeof(facts), hash);
        }
        free(names[index]);
    }
    *out = hash;
    return 1;
}

int h3_wcache_root(const char *weight_directory, char *root,
                   size_t root_size) {
    const char *base = getenv("H3_WEIGHT_CACHE_DIR");
    if (!base || !*base || !weight_directory || cache_disabled_by_env())
        return 0;
    uint64_t hash = 0;
    if (!fingerprint(weight_directory, &hash)) return 0;
    int written = snprintf(root, root_size, "%s/%016llx", base,
                           (unsigned long long)hash);
    if (written <= 0 || (size_t)written >= root_size) return 0;
    return make_directory(root);
}

static int expected_header(const h3_wcache_part *parts, size_t count,
                           h3_wcache_header *header) {
    if (!count || count > H3_WCACHE_MAX_PARTS) return 0;
    memset(header, 0, sizeof(*header));
    memcpy(header->magic, h3_wcache_magic, sizeof(header->magic));
    header->count = count;
    for (size_t index = 0; index < count; index++)
        header->bytes[index] = (uint64_t)parts[index].elements *
                               (uint64_t)parts[index].dtype;
    return 1;
}

static int read_header(int fd, h3_wcache_header *header) {
    return pread(fd, header, sizeof(*header), 0) == (ssize_t)sizeof(*header);
}

static int header_matches(int fd, const h3_wcache_part *parts, size_t count) {
    h3_wcache_header want, got;
    struct stat status;
    if (!expected_header(parts, count, &want) || !read_header(fd, &got) ||
        memcmp(&want, &got, sizeof(want)) != 0 || fstat(fd, &status) != 0)
        return 0;
    uint64_t total = H3_WCACHE_HEADER;
    for (size_t index = 0; index < count; index++) total += want.bytes[index];
    return (uint64_t)status.st_size == total;
}

int h3_wcache_present(const char *path, const h3_wcache_part *parts,
                      size_t count) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    int ok = header_matches(fd, parts, count);
    close(fd);
    return ok;
}

int h3_wcache_read(const char *path, const h3_wcache_part *parts,
                   size_t count) {
    if (!h3_wcache_present(path, parts, count)) return 0;
    uint64_t offset = H3_WCACHE_HEADER;
    for (size_t index = 0; index < count; index++) {
        size_t bytes = parts[index].elements * (size_t)parts[index].dtype;
        if (!h3_gpu_tensor_read_file_bytes(parts[index].tensor, path, offset,
                                           bytes, NULL, 0))
            return 0;
        offset += bytes;
    }
    return 1;
}

static int read_back(const h3_wcache_part *part, void *host) {
    switch (part->dtype) {
    case H3_WCACHE_BF16:
        return h3_gpu_tensor_read_bf16(part->tensor, host, part->elements);
    case H3_WCACHE_F32:
        return h3_gpu_tensor_read_f32(part->tensor, host, part->elements);
    case H3_WCACHE_I8:
        return h3_gpu_tensor_read_i8(part->tensor, host, part->elements);
    }
    return 0;
}

static int write_all(int fd, const void *data, size_t bytes) {
    const unsigned char *p = data;
    while (bytes) {
        ssize_t written = write(fd, p, bytes);
        if (written <= 0) return 0;
        p += written;
        bytes -= (size_t)written;
    }
    return 1;
}

int h3_wcache_write(const char *path, const h3_wcache_part *parts,
                    size_t count) {
    h3_wcache_header header;
    if (!expected_header(parts, count, &header)) return 0;
    char temporary[2048];
    int written = snprintf(temporary, sizeof(temporary), "%s.%d.tmp", path,
                           (int)getpid());
    if (written <= 0 || (size_t)written >= sizeof(temporary)) return 0;
    int fd = open(temporary, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return 0;
    unsigned char *page = calloc(1, H3_WCACHE_HEADER);
    int ok = page != NULL;
    if (ok) {
        memcpy(page, &header, sizeof(header));
        ok = write_all(fd, page, H3_WCACHE_HEADER);
    }
    free(page);
    for (size_t index = 0; ok && index < count; index++) {
        size_t bytes = (size_t)header.bytes[index];
        void *host = malloc(bytes ? bytes : 1);
        ok = host && read_back(&parts[index], host) &&
             write_all(fd, host, bytes);
        free(host);
    }
    if (close(fd) != 0) ok = 0;
    if (ok && rename(temporary, path) != 0) ok = 0;
    if (!ok) unlink(temporary);
    return ok;
}
