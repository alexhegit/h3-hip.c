#include "h3_disk_cache.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

static const char cache_magic[8] = {'H', '3', 'D', 'C', 'A', 'C', 'H', '1'};

/* Environment variables that select where results go or how they are
 * reported, never what is computed. Every other H3_* variable is part of the
 * key, so an unknown switch can only cost a miss. */
static const char *const ignored_environment[] = {
    "H3_PROFILE", "H3_DISK_CACHE", "H3_CACHE_DIR", "H3_CACHE_MAX_GIB",
    "H3_HIP_MEMLOG", "H3_INT8_HOST", "H3_INT8_HOST_OVERLAP", "H3_TERMINAL",
    "H3_HIP_DEVICE", "H3_PIKA_SEED", "H3_MODEL", "H3_ADALN_PREFETCH",
};

typedef struct {
    uint64_t first;
    uint64_t second;
} hash128;

static uint64_t rotate_left(uint64_t value, unsigned count) {
    return (value << count) | (value >> (64u - count));
}

static uint64_t finish_lane(uint64_t value) {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return value;
}

/* Not cryptographic: the full key is compared on load, so this only has to
 * spread entries over file names and catch torn or truncated payloads. */
typedef struct {
    uint64_t first;
    uint64_t second;
    uint64_t total;
    unsigned char pending[16];
    size_t buffered;
} hash_state;

static void hash_init(hash_state *state) {
    memset(state, 0, sizeof(*state));
    state->first = 0x9e3779b97f4a7c15ULL;
    state->second = 0xc2b2ae3d27d4eb4fULL;
}

static void hash_block(hash_state *state, const unsigned char *block) {
    uint64_t left, right;
    memcpy(&left, block, sizeof(left));
    memcpy(&right, block + 8, sizeof(right));
    state->first = rotate_left(state->first ^ (left * 0x87c37b91114253d5ULL),
                               31) * 0x4cf5ad432745937fULL;
    state->second = rotate_left(state->second ^
                                (right * 0x4cf5ad432745937fULL), 33) *
                    0x87c37b91114253d5ULL;
    state->first += state->second;
    state->second += state->first;
}

static void hash_update(hash_state *state, const void *data, size_t size) {
    const unsigned char *bytes = data;
    state->total += size;
    if (state->buffered) {
        size_t take = 16 - state->buffered;
        if (take > size) take = size;
        memcpy(state->pending + state->buffered, bytes, take);
        state->buffered += take;
        bytes += take;
        size -= take;
        if (state->buffered < 16) return;
        hash_block(state, state->pending);
        state->buffered = 0;
    }
    for (; size >= 16; bytes += 16, size -= 16) hash_block(state, bytes);
    memcpy(state->pending, bytes, size);
    state->buffered = size;
}

static hash128 hash_final(hash_state *state) {
    uint64_t tail[2] = {0, 0};
    memcpy(tail, state->pending, state->buffered);
    uint64_t first = state->first ^ (tail[0] * 0x87c37b91114253d5ULL) ^
                     state->total;
    uint64_t second = state->second ^ (tail[1] * 0x4cf5ad432745937fULL);
    first = finish_lane(first + second);
    second = finish_lane(second + first);
    return (hash128){first, second};
}

static hash128 hash_bytes(const void *data, size_t size) {
    hash_state state;
    hash_init(&state);
    hash_update(&state, data, size);
    return hash_final(&state);
}

static void key_append_raw(h3_disk_key *key, const char *text, size_t length) {
    if (key->failed) return;
    if (length + 2 > SIZE_MAX - key->length) {
        key->failed = 1;
        return;
    }
    size_t need = key->length + length + 2;
    if (need > key->capacity) {
        size_t capacity = key->capacity ? key->capacity : 256;
        while (capacity < need) capacity *= 2;
        char *grown = realloc(key->text, capacity);
        if (!grown) {
            key->failed = 1;
            return;
        }
        key->text = grown;
        key->capacity = capacity;
    }
    memcpy(key->text + key->length, text, length);
    key->length += length;
    key->text[key->length++] = '\n';
    key->text[key->length] = '\0';
}

void h3_disk_key_add(h3_disk_key *key, const char *format, ...) {
    if (!key || key->failed) return;
    char stack[512];
    va_list arguments;
    va_start(arguments, format);
    int length = vsnprintf(stack, sizeof(stack), format, arguments);
    va_end(arguments);
    if (length < 0) {
        key->failed = 1;
        return;
    }
    if ((size_t)length < sizeof(stack)) {
        key_append_raw(key, stack, (size_t)length);
        return;
    }
    char *heap = malloc((size_t)length + 1);
    if (!heap) {
        key->failed = 1;
        return;
    }
    va_start(arguments, format);
    vsnprintf(heap, (size_t)length + 1, format, arguments);
    va_end(arguments);
    key_append_raw(key, heap, (size_t)length);
    free(heap);
}

void h3_disk_key_add_bytes(h3_disk_key *key, const char *label,
                           const void *data, size_t size) {
    if (!key) return;
    if (size && !data) {
        key->failed = 1;
        return;
    }
    hash128 hash = hash_bytes(data ? data : "", size);
    h3_disk_key_add(key, "%s=%zu:%016" PRIx64 "%016" PRIx64, label, size,
                    hash.first, hash.second);
}

void h3_disk_key_add_file(h3_disk_key *key, const char *path) {
    if (!key || key->failed) return;
    char resolved[PATH_MAX];
    struct stat info;
    if (!path || !realpath(path, resolved) || stat(resolved, &info) != 0) {
        key->failed = 1;
        return;
    }
    h3_disk_key_add(key, "file=%s:%lld:%lld.%09ld", resolved,
                    (long long)info.st_size, (long long)info.st_mtim.tv_sec,
                    info.st_mtim.tv_nsec);
}

static int compare_names(const void *left, const void *right) {
    return strcmp(*(char *const *)left, *(char *const *)right);
}

void h3_disk_key_add_directory(h3_disk_key *key, const char *directory,
                               const char *suffix) {
    if (!key || key->failed) return;
    DIR *stream = directory ? opendir(directory) : NULL;
    if (!stream) {
        key->failed = 1;
        return;
    }
    size_t suffix_length = strlen(suffix);
    char **names = NULL;
    size_t count = 0, capacity = 0;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        size_t length = strlen(entry->d_name);
        if (length < suffix_length ||
            strcmp(entry->d_name + length - suffix_length, suffix)) continue;
        if (count == capacity) {
            capacity = capacity ? capacity * 2 : 16;
            char **grown = realloc(names, capacity * sizeof(*names));
            if (!grown) {
                key->failed = 1;
                break;
            }
            names = grown;
        }
        names[count] = strdup(entry->d_name);
        if (!names[count]) {
            key->failed = 1;
            break;
        }
        count++;
    }
    closedir(stream);
    if (!count) key->failed = 1;
    if (!key->failed) {
        qsort(names, count, sizeof(*names), compare_names);
        for (size_t index = 0; index < count && !key->failed; index++) {
            char path[PATH_MAX];
            if (snprintf(path, sizeof(path), "%s/%s", directory,
                         names[index]) >= (int)sizeof(path)) {
                key->failed = 1;
                break;
            }
            h3_disk_key_add_file(key, path);
        }
    }
    for (size_t index = 0; index < count; index++) free(names[index]);
    free(names);
}

static int environment_ignored(const char *entry) {
    for (size_t index = 0;
         index < sizeof(ignored_environment) / sizeof(*ignored_environment);
         index++) {
        size_t length = strlen(ignored_environment[index]);
        if (!strncmp(entry, ignored_environment[index], length) &&
            entry[length] == '=') return 1;
    }
    return 0;
}

extern char **environ;

static void key_add_environment(h3_disk_key *key) {
    size_t count = 0;
    for (char **entry = environ; entry && *entry; entry++)
        if (!strncmp(*entry, "H3_", 3) && !environment_ignored(*entry))
            count++;
    const char **selected = count ? malloc(count * sizeof(*selected)) : NULL;
    if (count && !selected) {
        key->failed = 1;
        return;
    }
    size_t used = 0;
    for (char **entry = environ; entry && *entry; entry++)
        if (!strncmp(*entry, "H3_", 3) && !environment_ignored(*entry))
            selected[used++] = *entry;
    qsort(selected, used, sizeof(*selected), compare_names);
    for (size_t index = 0; index < used; index++)
        h3_disk_key_add(key, "env=%s", selected[index]);
    free(selected);
}

/* The executable and the GPU libraries it links compute every cached value,
 * so their identities lead each key. */
static void key_add_build(h3_disk_key *key) {
    h3_disk_key_add_file(key, "/proc/self/exe");
    FILE *maps = fopen("/proc/self/maps", "r");
    if (!maps) {
        key->failed = 1;
        return;
    }
    static const char *const libraries[] = {
        "libamdhip64.so", "libhipblas.so", "libhipblaslt.so",
    };
    char *seen[8] = {0};
    size_t seen_count = 0;
    char line[PATH_MAX + 256];
    while (fgets(line, sizeof(line), maps)) {
        char *path = strchr(line, '/');
        if (!path) continue;
        path[strcspn(path, "\n")] = '\0';
        const char *base = strrchr(path, '/') + 1;
        int wanted = 0;
        for (size_t index = 0;
             index < sizeof(libraries) / sizeof(*libraries); index++)
            if (!strncmp(base, libraries[index], strlen(libraries[index])))
                wanted = 1;
        if (!wanted) continue;
        int duplicate = 0;
        for (size_t index = 0; index < seen_count; index++)
            if (!strcmp(seen[index], path)) duplicate = 1;
        if (duplicate || seen_count == sizeof(seen) / sizeof(*seen)) continue;
        seen[seen_count] = strdup(path);
        if (!seen[seen_count]) {
            key->failed = 1;
            break;
        }
        seen_count++;
    }
    fclose(maps);
    qsort(seen, seen_count, sizeof(*seen), compare_names);
    for (size_t index = 0; index < seen_count; index++) {
        h3_disk_key_add_file(key, seen[index]);
        free(seen[index]);
    }
}

int h3_disk_cache_enabled(void) {
    const char *value = getenv("H3_DISK_CACHE");
    return !value || strcmp(value, "0") != 0;
}

void h3_disk_key_init(h3_disk_key *key, const char *kind) {
    memset(key, 0, sizeof(*key));
    if (!h3_disk_cache_enabled()) {
        key->failed = 1;
        return;
    }
    h3_disk_key_add(key, "h3-disk-cache-v1");
    h3_disk_key_add(key, "kind=%s", kind);
    key_add_build(key);
    key_add_environment(key);
}

void h3_disk_key_free(h3_disk_key *key) {
    if (!key) return;
    free(key->text);
    memset(key, 0, sizeof(*key));
}

static int make_directory(const char *path) {
    char partial[PATH_MAX];
    if (snprintf(partial, sizeof(partial), "%s", path) >= (int)sizeof(partial))
        return 0;
    for (char *cursor = partial + 1; *cursor; cursor++) {
        if (*cursor != '/') continue;
        *cursor = '\0';
        if (mkdir(partial, 0700) != 0 && errno != EEXIST) return 0;
        *cursor = '/';
    }
    return mkdir(partial, 0700) == 0 || errno == EEXIST;
}

static int cache_directory(char *path, size_t capacity) {
    const char *explicit = getenv("H3_CACHE_DIR");
    const char *xdg = getenv("XDG_CACHE_HOME");
    const char *home = getenv("HOME");
    int written;
    if (explicit && explicit[0])
        written = snprintf(path, capacity, "%s", explicit);
    else if (xdg && xdg[0])
        written = snprintf(path, capacity, "%s/h3-hip", xdg);
    else if (home && home[0])
        written = snprintf(path, capacity, "%s/.cache/h3-hip", home);
    else
        return 0;
    return written > 0 && (size_t)written < capacity;
}

static int entry_path(const h3_disk_key *key, char *path, size_t capacity) {
    char directory[PATH_MAX];
    if (!cache_directory(directory, sizeof(directory))) return 0;
    hash128 hash = hash_bytes(key->text, key->length);
    int written = snprintf(path, capacity, "%s/%016" PRIx64 "%016" PRIx64
                           ".h3c", directory, hash.first, hash.second);
    return written > 0 && (size_t)written < capacity;
}

static int read_exact(int descriptor, void *data, size_t size) {
    unsigned char *cursor = data;
    while (size) {
        ssize_t got = read(descriptor, cursor, size);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return 0;
        cursor += got;
        size -= (size_t)got;
    }
    return 1;
}

static int write_exact(int descriptor, const void *data, size_t size) {
    const unsigned char *cursor = data;
    while (size) {
        ssize_t put = write(descriptor, cursor, size);
        if (put < 0 && errno == EINTR) continue;
        if (put <= 0) return 0;
        cursor += put;
        size -= (size_t)put;
    }
    return 1;
}

int h3_disk_cache_load(const h3_disk_key *key, void **payload, size_t *size) {
    *payload = NULL;
    *size = 0;
    if (!key || key->failed || !key->text) return 0;
    char path[PATH_MAX];
    if (!entry_path(key, path, sizeof(path))) return 0;
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) return 0;
    char magic[sizeof(cache_magic)];
    uint64_t key_length = 0, payload_length = 0;
    hash128 stored;
    char *stored_key = NULL;
    void *data = NULL;
    int ok = read_exact(descriptor, magic, sizeof(magic)) &&
             !memcmp(magic, cache_magic, sizeof(magic)) &&
             read_exact(descriptor, &key_length, sizeof(key_length)) &&
             key_length == key->length;
    if (ok) {
        stored_key = malloc(key->length);
        ok = stored_key && read_exact(descriptor, stored_key, key->length) &&
             !memcmp(stored_key, key->text, key->length) &&
             read_exact(descriptor, &payload_length, sizeof(payload_length)) &&
             read_exact(descriptor, &stored, sizeof(stored)) &&
             payload_length <= SIZE_MAX;
    }
    if (ok) {
        data = malloc(payload_length ? (size_t)payload_length : 1);
        ok = data && read_exact(descriptor, data, (size_t)payload_length);
        char extra;
        if (ok) ok = read(descriptor, &extra, 1) == 0;
    }
    if (ok) {
        hash128 actual = hash_bytes(data, (size_t)payload_length);
        ok = actual.first == stored.first && actual.second == stored.second;
    }
    close(descriptor);
    free(stored_key);
    if (!ok) {
        free(data);
        return 0;
    }
    (void)utimensat(AT_FDCWD, path, NULL, 0);
    *payload = data;
    *size = (size_t)payload_length;
    return 1;
}

typedef struct {
    char *path;
    off_t size;
    struct timespec used;
} cache_file;

static int compare_used(const void *left, const void *right) {
    const cache_file *a = left;
    const cache_file *b = right;
    if (a->used.tv_sec != b->used.tv_sec)
        return a->used.tv_sec < b->used.tv_sec ? -1 : 1;
    if (a->used.tv_nsec != b->used.tv_nsec)
        return a->used.tv_nsec < b->used.tv_nsec ? -1 : 1;
    return 0;
}

static double cache_limit_bytes(void) {
    const char *value = getenv("H3_CACHE_MAX_GIB");
    double gib = value && value[0] ? atof(value) : 8.0;
    if (!(gib >= 0.0)) gib = 8.0;
    return gib * 1073741824.0;
}

static void evict(const char *directory, const char *keep) {
    DIR *stream = opendir(directory);
    if (!stream) return;
    cache_file *files = NULL;
    size_t count = 0, capacity = 0;
    double total = 0.0;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        size_t length = strlen(entry->d_name);
        if (length < 4 || strcmp(entry->d_name + length - 4, ".h3c")) continue;
        char path[PATH_MAX];
        struct stat info;
        if (snprintf(path, sizeof(path), "%s/%s", directory, entry->d_name) >=
                (int)sizeof(path) || stat(path, &info) != 0) continue;
        if (count == capacity) {
            capacity = capacity ? capacity * 2 : 32;
            cache_file *grown = realloc(files, capacity * sizeof(*files));
            if (!grown) break;
            files = grown;
        }
        files[count].path = strdup(path);
        if (!files[count].path) break;
        files[count].size = info.st_size;
        files[count].used = info.st_mtim;
        total += (double)info.st_size;
        count++;
    }
    closedir(stream);
    if (count) qsort(files, count, sizeof(*files), compare_used);
    double limit = cache_limit_bytes();
    for (size_t index = 0; index < count && total > limit; index++) {
        if (!strcmp(files[index].path, keep)) continue;
        if (unlink(files[index].path) == 0) total -= (double)files[index].size;
    }
    for (size_t index = 0; index < count; index++) free(files[index].path);
    free(files);
}

void h3_disk_cache_store(const h3_disk_key *key, const void *const *parts,
                         const size_t *sizes, size_t count) {
    if (!key || key->failed || !key->text) return;
    size_t total = 0;
    for (size_t index = 0; index < count; index++) {
        if (sizes[index] > SIZE_MAX - total) return;
        total += sizes[index];
    }
    if ((double)total + (double)key->length > cache_limit_bytes()) return;
    char directory[PATH_MAX], path[PATH_MAX], temporary[PATH_MAX];
    if (!cache_directory(directory, sizeof(directory)) ||
        !make_directory(directory) ||
        !entry_path(key, path, sizeof(path)) ||
        snprintf(temporary, sizeof(temporary), "%s.%ld.tmp", path,
                 (long)getpid()) >= (int)sizeof(temporary)) return;
    hash_state state;
    hash_init(&state);
    for (size_t index = 0; index < count; index++)
        if (sizes[index]) hash_update(&state, parts[index], sizes[index]);
    hash128 hash = hash_final(&state);
    int descriptor = open(temporary, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
                          0600);
    if (descriptor < 0) return;
    uint64_t key_length = key->length, payload_length = total;
    int ok = write_exact(descriptor, cache_magic, sizeof(cache_magic)) &&
             write_exact(descriptor, &key_length, sizeof(key_length)) &&
             write_exact(descriptor, key->text, key->length) &&
             write_exact(descriptor, &payload_length, sizeof(payload_length)) &&
             write_exact(descriptor, &hash, sizeof(hash));
    for (size_t index = 0; ok && index < count; index++)
        if (sizes[index]) ok = write_exact(descriptor, parts[index],
                                           sizes[index]);
    if (close(descriptor) != 0) ok = 0;
    if (!ok || rename(temporary, path) != 0) {
        unlink(temporary);
        return;
    }
    evict(directory, path);
}
