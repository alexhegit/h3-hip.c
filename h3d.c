#include "h3d.h"
#include "h3d_config.h"
#include "h3d_http.h"
#include "h3d_json.h"
#include "h3d_job.h"
#include "h3d_worker.h"
#include "h3_host.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* Connections beyond this get a 503 instead of a thread. */
#define H3D_MAX_CONNECTIONS 64

static h3d_ctx *g_ctx = NULL;

static void signal_handler(int sig) {
    (void)sig;
    if (g_ctx) {
        atomic_store(&g_ctx->shutdown, 1);
    }
}

/* ── JSON reader ──────────────────────────────────────────────── */
/* Walks the request structure, so whitespace, key order, and keys that
 * appear inside string values do not matter. Value pointers point at the
 * first byte of the value in the body. */

static const char *json_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    return p;
}

static const char *json_skip_string(const char *p) {
    for (p++; *p && *p != '"'; p++)
        if (*p == '\\' && p[1]) p++;
    return *p == '"' ? p + 1 : NULL;
}

static const char *json_skip_value(const char *p) {
    p = json_ws(p);
    if (*p == '"') return json_skip_string(p);
    if (*p == '{' || *p == '[') {
        int depth = 0;
        while (*p) {
            if (*p == '"') {
                p = json_skip_string(p);
                if (!p) return NULL;
                continue;
            }
            if (*p == '{' || *p == '[') depth++;
            else if ((*p == '}' || *p == ']') && --depth == 0) return p + 1;
            p++;
        }
        return NULL;
    }
    const char *start = p;
    while (*p && !strchr(",}] \t\r\n", *p)) p++;
    return p > start ? p : NULL;
}

/* Value of `key` in the object at `object`, or NULL. */
static const char *json_member(const char *object, const char *key) {
    if (!object) return NULL;
    const char *p = json_ws(object);
    if (*p != '{') return NULL;
    p = json_ws(p + 1);
    size_t key_len = strlen(key);
    while (*p == '"') {
        const char *name = p + 1;
        const char *end = json_skip_string(p);
        if (!end) return NULL;
        p = json_ws(end);
        if (*p != ':') return NULL;
        const char *value = json_ws(p + 1);
        if ((size_t)(end - 1 - name) == key_len && !memcmp(name, key, key_len))
            return value;
        p = json_skip_value(value);
        if (!p) return NULL;
        p = json_ws(p);
        if (*p != ',') break;
        p = json_ws(p + 1);
    }
    return NULL;
}

/* Element `index` of the array at `array`, or NULL. */
static const char *json_element(const char *array, int index) {
    if (!array || *array != '[') return NULL;
    const char *p = json_ws(array + 1);
    if (*p == ']') return NULL;
    for (int i = 0; ; i++) {
        if (i == index) return p;
        p = json_skip_value(p);
        if (!p) return NULL;
        p = json_ws(p);
        if (*p != ',') return NULL;
        p = json_ws(p + 1);
    }
}

/* Present and not null. */
static int json_given(const char *value) {
    return value && strncmp(value, "null", 4) != 0;
}

static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int json_hex4(const char *p, unsigned *out) {
    unsigned v = 0;
    for (int k = 0; k < 4; k++) {
        int d = hex_digit(p[k]);
        if (d < 0) return 0;
        v = v * 16u + (unsigned)d;
    }
    *out = v;
    return 1;
}

/* Decode the string at p into value. Returns 0 if p is not a string or the
 * decoded text does not fit. */
static int json_read_string(const char *p, char *value, size_t value_len) {
    if (!p || *p != '"' || value_len == 0) return 0;
    p++;
    size_t i = 0;
    while (*p && *p != '"') {
        char utf8[4];
        size_t n = 1;
        if (*p != '\\') {
            utf8[0] = *p++;
        } else {
            p++;
            switch (*p) {
            case 'n': utf8[0] = '\n'; p++; break;
            case 't': utf8[0] = '\t'; p++; break;
            case 'r': utf8[0] = '\r'; p++; break;
            case 'b': utf8[0] = '\b'; p++; break;
            case 'f': utf8[0] = '\f'; p++; break;
            case 'u': {
                unsigned cp;
                if (!json_hex4(p + 1, &cp)) return 0;
                p += 5;
                unsigned lo;
                if (cp >= 0xD800 && cp <= 0xDBFF && p[0] == '\\' && p[1] == 'u' &&
                    json_hex4(p + 2, &lo) && lo >= 0xDC00 && lo <= 0xDFFF) {
                    cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                    p += 6;
                }
                if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0xFFFD;
                if (cp == 0) return 0;
                if (cp < 0x80) {
                    utf8[0] = (char)cp;
                } else if (cp < 0x800) {
                    utf8[0] = (char)(0xC0 | (cp >> 6));
                    utf8[1] = (char)(0x80 | (cp & 0x3F));
                    n = 2;
                } else if (cp < 0x10000) {
                    utf8[0] = (char)(0xE0 | (cp >> 12));
                    utf8[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    utf8[2] = (char)(0x80 | (cp & 0x3F));
                    n = 3;
                } else {
                    utf8[0] = (char)(0xF0 | (cp >> 18));
                    utf8[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
                    utf8[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
                    utf8[3] = (char)(0x80 | (cp & 0x3F));
                    n = 4;
                }
                break;
            }
            case '"': case '\\': case '/':
                utf8[0] = *p++;
                break;
            default:
                return 0;
            }
        }
        if (i + n >= value_len) return 0;
        memcpy(value + i, utf8, n);
        i += n;
    }
    if (*p != '"') return 0;
    value[i] = '\0';
    return 1;
}

static int json_number_end(const char *end) {
    return *end && strchr(",}] \t\r\n", *end) != NULL;
}

static int json_read_int(const char *p, int *value) {
    if (!p || !(*p == '-' || (*p >= '0' && *p <= '9'))) return 0;
    char *end = NULL;
    errno = 0;
    long v = strtol(p, &end, 10);
    if (errno || end == p || !json_number_end(end) || v < INT_MIN || v > INT_MAX)
        return 0;
    *value = (int)v;
    return 1;
}

static int json_read_u64(const char *p, uint64_t *value) {
    if (!p || *p < '0' || *p > '9') return 0;
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(p, &end, 10);
    if (errno || end == p || !json_number_end(end)) return 0;
    *value = (uint64_t)v;
    return 1;
}

static int json_read_double(const char *p, double *value) {
    if (!p || !(*p == '-' || (*p >= '0' && *p <= '9'))) return 0;
    char *end = NULL;
    errno = 0;
    double v = strtod(p, &end);
    if (errno || end == p || !json_number_end(end) || !isfinite(v)) return 0;
    *value = v;
    return 1;
}

static int json_read_bool(const char *p, int *value) {
    if (!p) return 0;
    if (!strncmp(p, "true", 4) && json_number_end(p + 4)) { *value = 1; return 1; }
    if (!strncmp(p, "false", 5) && json_number_end(p + 5)) { *value = 0; return 1; }
    return 0;
}

/* ── Responses ───────────────────────────────────────────────── */

static void send_json(int fd, int status, const char *status_text,
                      const char *body, int len) {
    if (len < 0) {
        static const char fallback[] =
            "{\"error\":{\"code\":\"INTERNAL\",\"message\":\"response too large\"}}";
        h3d_http_send_response(fd, 500, "Internal Server Error",
                               "application/json; charset=utf-8",
                               fallback, sizeof(fallback) - 1);
        return;
    }
    h3d_http_send_response(fd, status, status_text,
                           "application/json; charset=utf-8", body, (size_t)len);
}

/* Error body with an optional single detail (field may be NULL). */
static void send_error(int fd, int status, const char *status_text,
                       const char *code, const char *field,
                       const char *format, ...) {
    h3d_error err = {0};
    snprintf(err.code, sizeof(err.code), "%s", code);
    va_list args;
    va_start(args, format);
    vsnprintf(err.message, sizeof(err.message), format, args);
    va_end(args);
    if (field) {
        snprintf(err.details[0].field, sizeof(err.details[0].field), "%s", field);
        snprintf(err.details[0].reason, sizeof(err.details[0].reason), "%.*s",
                 (int)sizeof(err.details[0].reason) - 1, err.message);
        err.detail_count = 1;
    }
    char buf[2048];
    send_json(fd, status, status_text, buf,
              h3d_json_error_response(&err, buf, sizeof(buf)));
}

#define BAD_REQUEST(code, field, ...) \
    send_error(fd, 400, "Bad Request", code, field, __VA_ARGS__)
#define UNPROCESSABLE(code, field, ...) \
    send_error(fd, 422, "Unprocessable Entity", code, field, __VA_ARGS__)

static void send_job(int fd, int status, const char *status_text, h3d_job *job) {
    char *buf = malloc(H3D_JOB_JSON_MAX);
    if (!buf) {
        send_json(fd, 500, "Internal Server Error", NULL, -1);
        return;
    }
    send_json(fd, status, status_text, buf,
              h3d_json_job_response(job, buf, H3D_JOB_JSON_MAX));
    free(buf);
}

/* ── Protocol version check ──────────────────────────────────── */

/* Value of header `name` (case-insensitive) copied into value; 0 if absent. */
static int header_value(const char *headers, const char *name, char *value,
                        size_t value_len) {
    size_t name_len = strlen(name);
    for (const char *line = headers; *line; ) {
        const char *end = strstr(line, "\r\n");
        if (!end) end = line + strlen(line);
        if ((size_t)(end - line) > name_len && line[name_len] == ':' &&
            !strncasecmp(line, name, name_len)) {
            const char *p = line + name_len + 1;
            while (*p == ' ' || *p == '\t') p++;
            size_t n = (size_t)(end - p);
            if (n >= value_len) n = value_len - 1;
            memcpy(value, p, n);
            value[n] = '\0';
            return 1;
        }
        line = *end ? end + 2 : end;
    }
    return 0;
}

static void send_protocol_mismatch(int fd) {
    char body[512];
    int len = snprintf(body, sizeof(body),
        "{\"error\":{\"code\":\"PROTOCOL_MISMATCH\","
        "\"message\":\"Protocol version mismatch\","
        "\"server_protocol\":\"%s\","
        "\"supported_protocols\":[\"%s\"]}}",
        H3D_PROTOCOL, H3D_PROTOCOL);
    send_json(fd, 426, "Upgrade Required", body, len);
}

/* ── Validation helpers ──────────────────────────────────────── */

static int resolve_root(const char *root, char *resolved) {
    char expanded[PATH_MAX];
    if (root[0] == '~') {
        const char *home = getenv("HOME");
        if (!home) return 0;
        snprintf(expanded, sizeof(expanded), "%s%s", home, root + 1);
    } else {
        snprintf(expanded, sizeof(expanded), "%s", root);
    }
    return realpath(expanded, resolved) != NULL;
}

/* Resolve a reference path; it must name an existing regular file under
 * media_root after symlinks and ".." are resolved. Sends the error and
 * returns 0 otherwise. */
static int resolve_media_path(h3d_ctx *ctx, int fd, const char *field,
                              const char *path, char *out, size_t out_len) {
    char root[PATH_MAX], resolved[PATH_MAX];
    if (!resolve_root(ctx->media_root, root)) {
        UNPROCESSABLE("UNSUPPORTED_COMBINATION", field,
                      "media_root does not exist on the daemon host");
        return 0;
    }
    struct stat st;
    if (!realpath(path, resolved) || stat(resolved, &st) < 0 || !S_ISREG(st.st_mode)) {
        BAD_REQUEST("VALIDATION_ERROR", field, "reference file not found: %s", path);
        return 0;
    }
    size_t root_len = strlen(root);
    if (strncmp(resolved, root, root_len) != 0 || resolved[root_len] != '/') {
        BAD_REQUEST("PATH_OUT_OF_ROOT", field, "reference file not in media_root");
        return 0;
    }
    if (strlen(resolved) >= out_len) {
        BAD_REQUEST("VALIDATION_ERROR", field, "reference path too long");
        return 0;
    }
    snprintf(out, out_len, "%s", resolved);
    return 1;
}

/* refs.<key> as a string or an array of strings. Returns the count, or -1
 * after sending an error. */
static int parse_ref_list(h3d_ctx *ctx, int fd, const char *refs,
                          const char *key, h3d_ref_entry *out, int max) {
    char field[64];
    snprintf(field, sizeof(field), "refs.%s", key);
    const char *value = json_member(refs, key);
    if (!json_given(value)) return 0;
    char path[1024];
    if (*value == '"') {
        if (max < 1) {
            BAD_REQUEST("REFS_LIMIT_EXCEEDED", field, "%s accepts no files in this mode", field);
            return -1;
        }
        if (!json_read_string(value, path, sizeof(path))) {
            BAD_REQUEST("VALIDATION_ERROR", field, "%s must be a path string", field);
            return -1;
        }
        return resolve_media_path(ctx, fd, field, path, out[0].path,
                                  sizeof(out[0].path)) ? 1 : -1;
    }
    if (*value != '[') {
        BAD_REQUEST("VALIDATION_ERROR", field, "%s must be a string or an array of strings", field);
        return -1;
    }
    int count = 0;
    for (const char *item; (item = json_element(value, count)); count++) {
        if (count >= max) {
            BAD_REQUEST("REFS_LIMIT_EXCEEDED", field, "%s accepts at most %d files", field, max);
            return -1;
        }
        if (!json_read_string(item, path, sizeof(path))) {
            BAD_REQUEST("VALIDATION_ERROR", field, "%s must contain path strings", field);
            return -1;
        }
        if (!resolve_media_path(ctx, fd, field, path, out[count].path,
                                sizeof(out[count].path)))
            return -1;
    }
    return count;
}

/* refs.<key> as one path string; returns 0 if absent, 1 if set, -1 after an error. */
static int parse_ref_single(h3d_ctx *ctx, int fd, const char *refs,
                            const char *key, char *out, size_t out_len) {
    char field[64];
    snprintf(field, sizeof(field), "refs.%s", key);
    const char *value = json_member(refs, key);
    if (!json_given(value)) return 0;
    char path[1024];
    if (!json_read_string(value, path, sizeof(path))) {
        BAD_REQUEST("VALIDATION_ERROR", field, "%s must be a path string", field);
        return -1;
    }
    return resolve_media_path(ctx, fd, field, path, out, out_len) ? 1 : -1;
}

/* ── Mode parsing ────────────────────────────────────────────── */

static int parse_mode(const char *mode_str, h3d_mode *mode) {
    if (!mode_str[0] || !strcmp(mode_str, "T2VA")) *mode = H3D_MODE_T2VA;
    else if (!strcmp(mode_str, "I2VA")) *mode = H3D_MODE_I2VA;
    else if (!strcmp(mode_str, "FL2VA")) *mode = H3D_MODE_FL2VA;
    else if (!strcmp(mode_str, "Ref2VA")) *mode = H3D_MODE_REF2VA;
    else return 0;
    return 1;
}

/* ── Quality preset resolution ───────────────────────────────── */

static int resolve_quality(h3d_quality *q) {
    int steps, layers, reuse;
    if (!strcmp(q->preset, "fox-s2")) {
        steps = 2; layers = 35; reuse = 1;
    } else if (!strcmp(q->preset, "fox-fast") || !strcmp(q->preset, "cinematic") ||
               !strcmp(q->preset, "custom")) {
        steps = 20; layers = 45; reuse = 2;
    } else {
        return 0;
    }
    if (q->steps == 0) q->steps = steps;
    if (q->layers == 0) q->layers = layers;
    if (q->reuse == 0) q->reuse = reuse;
    return 1;
}

/* ── extra_args ──────────────────────────────────────────────── */

/* Only flags whose h3_params field the daemon sets per job. Anything else
 * is rejected so a request never silently runs without it. */
static int apply_extra_args(h3d_job *job, int fd, char **tokens, int count) {
    char echo[sizeof(job->extra_args)] = {0};
    for (int i = 0; i < count; i++) {
        const char *arg = tokens[i];
        const char *next = i + 1 < count ? tokens[i + 1] : NULL;
        int takes_value = 0;
        if (!strcmp(arg, "--token-reduction")) job->token_reduction = 1;
        else if (!strcmp(arg, "--sol-attn")) job->sol_attn = 1;
        else if (!strcmp(arg, "--fbc")) job->fbc = 1;
        else if (!strcmp(arg, "--vsa")) job->vsa = 1;
        else if (!strcmp(arg, "--core-reuse") || !strcmp(arg, "--render-width") ||
                 !strcmp(arg, "--render-height")) {
            char *end = NULL;
            long v = next ? strtol(next, &end, 10) : 0;
            if (!next || end == next || *end || v <= 0 || v > 65536) {
                UNPROCESSABLE("ARG_REJECTED", "options.extra_args",
                              "%s needs a positive integer", arg);
                return 0;
            }
            if (!strcmp(arg, "--core-reuse")) job->core_reuse = (int)v;
            else if (!strcmp(arg, "--render-width")) job->render_width = (int)v;
            else job->render_height = (int)v;
            takes_value = 1;
        } else if (!strcmp(arg, "--fasth3-lora") || !strcmp(arg, "--taeh3")) {
            struct stat st;
            if (!next || stat(next, &st) < 0 || !S_ISREG(st.st_mode)) {
                UNPROCESSABLE("ARG_REJECTED", "options.extra_args",
                              "%s needs an existing file", arg);
                return 0;
            }
            char *dst = !strcmp(arg, "--taeh3") ? job->taeh3 : job->fasth3_lora;
            if (strlen(next) >= sizeof(job->taeh3)) {
                UNPROCESSABLE("ARG_REJECTED", "options.extra_args", "%s path too long", arg);
                return 0;
            }
            snprintf(dst, sizeof(job->taeh3), "%s", next);
            takes_value = 1;
        } else {
            UNPROCESSABLE("ARG_REJECTED", "options.extra_args",
                          "extra_args does not support %s", arg);
            return 0;
        }
        size_t used = strlen(echo);
        snprintf(echo + used, sizeof(echo) - used, "%s%s%s%s", used ? " " : "",
                 arg, takes_value ? " " : "", takes_value ? next : "");
        i += takes_value;
    }
    snprintf(job->extra_args, sizeof(job->extra_args), "%s", echo);
    return 1;
}

/* options.extra_args as an array of strings, or one space-separated string. */
static int parse_extra_args(h3d_job *job, int fd, const char *value) {
    if (!json_given(value)) return 1;
    char storage[2048];
    char *tokens[64];
    int count = 0;
    if (*value == '"') {
        if (!json_read_string(value, storage, sizeof(storage))) {
            BAD_REQUEST("VALIDATION_ERROR", "options.extra_args", "extra_args is not a valid string");
            return 0;
        }
        char *save = NULL;
        for (char *t = strtok_r(storage, " \t", &save); t && count < 64;
             t = strtok_r(NULL, " \t", &save))
            tokens[count++] = t;
    } else if (*value == '[') {
        size_t used = 0;
        for (const char *item; (item = json_element(value, count)); count++) {
            if (count >= 64 || !json_read_string(item, storage + used, sizeof(storage) - used)) {
                BAD_REQUEST("VALIDATION_ERROR", "options.extra_args",
                            "extra_args must be an array of short strings");
                return 0;
            }
            tokens[count] = storage + used;
            used += strlen(storage + used) + 1;
        }
    } else {
        BAD_REQUEST("VALIDATION_ERROR", "options.extra_args",
                    "extra_args must be an array of strings");
        return 0;
    }
    return apply_extra_args(job, fd, tokens, count);
}

/* ── Build CLI args echo ─────────────────────────────────────── */

static void cli_append(char *buf, size_t len, const char *format, ...) {
    size_t used = strlen(buf);
    if (used + 1 >= len) return;
    va_list args;
    va_start(args, format);
    vsnprintf(buf + used, len - used, format, args);
    va_end(args);
}

static void build_cli_args(char *buf, size_t len, h3d_ctx *ctx, h3d_job *job) {
    buf[0] = '\0';
    char prompt[8192 * 2];
    size_t j = 0;
    for (const char *p = job->prompt; *p && j + 2 < sizeof(prompt); p++) {
        if (*p == '"' || *p == '\\' || *p == '$' || *p == '`') prompt[j++] = '\\';
        prompt[j++] = *p;
    }
    prompt[j] = '\0';
    cli_append(buf, len, "./h3 -d %s -p \"%s\" --width %d --height %d --frames %d"
               " --steps %d --layers %d --reuse %d",
               ctx->model_path, prompt, job->width, job->height, job->frames,
               job->quality.steps, job->quality.layers, job->quality.reuse);
    if (job->seed_was_set) cli_append(buf, len, " --seed %" PRIu64, job->seed);
    if (job->first_frame[0]) cli_append(buf, len, " --first-frame %s", job->first_frame);
    if (job->last_frame[0]) cli_append(buf, len, " --last-frame %s", job->last_frame);
    for (int i = 0; i < job->refs_image_count; i++)
        cli_append(buf, len, " --ref-image %s", job->refs_images[i].path);
    for (int i = 0; i < job->refs_video_count; i++)
        cli_append(buf, len, " --ref-video %s", job->refs_videos[i].path);
    for (int i = 0; i < job->refs_audio_count; i++)
        cli_append(buf, len, " --ref-audio %s", job->refs_audio[i].path);
    if (job->token_reduction && !strstr(job->extra_args, "--token-reduction"))
        cli_append(buf, len, " --token-reduction");
    if (job->fasth3_lora[0] && !strstr(job->extra_args, "--fasth3-lora"))
        cli_append(buf, len, " --fasth3-lora %s", job->fasth3_lora);
    if (job->extra_args[0]) cli_append(buf, len, " %s", job->extra_args);
    cli_append(buf, len, " -o %s", job->mp4_path);
}

/* ── Handlers ────────────────────────────────────────────────── */

static void handle_info(h3d_ctx *ctx, int fd) {
    char body[8192];
    send_json(fd, 200, "OK", body, h3d_json_info_response(ctx, body, sizeof(body)));
}

/* Ask every worker to drop its generation cache. Idle workers act within
 * 100 ms, so wait for them and answer 200 once the memory is back; a busy
 * worker releases after its queue drains and the answer is 202. */
static void handle_cache_release(h3d_ctx *ctx, int fd) {
    for (h3d_worker *w = ctx->workers; w; w = w->next)
        atomic_store(&w->release_requested, 1);
    int released = 0;
    for (int i = 0; i < 100 && !released; i++) {
        released = 1;
        for (h3d_worker *w = ctx->workers; w; w = w->next)
            if (atomic_load(&w->release_requested)) released = 0;
        if (!released) usleep(100000);
    }
    char body[8192];
    int len = h3d_json_cache_release_response(ctx, released, body, sizeof(body));
    if (released) send_json(fd, 200, "OK", body, len);
    else send_json(fd, 202, "Accepted", body, len);
}

#define FIELD(parse, object, key, name, dst, what) do {                        \
    const char *value_ = json_member(object, key);                             \
    if (json_given(value_) && !parse(value_, dst)) {                           \
        BAD_REQUEST("VALIDATION_ERROR", name, "%s must be %s", name, what);    \
        goto reject;                                                            \
    }                                                                           \
} while (0)

static void handle_submit(h3d_ctx *ctx, int fd, const char *body) {
    if (!atomic_load(&ctx->ready)) {
        send_error(fd, 503, "Service Unavailable", "NOT_READY", NULL,
                   "Daemon not ready: %s", ctx->unready_reason);
        return;
    }
    const char *end = json_skip_value(body);
    if (*json_ws(body) != '{' || !end || *json_ws(end)) {
        BAD_REQUEST("VALIDATION_ERROR", NULL, "request body must be one JSON object");
        return;
    }

    h3d_job *job = h3d_job_new(ctx);
    if (!job) {
        send_error(fd, 500, "Internal Server Error", "INTERNAL", NULL, "cannot allocate job");
        return;
    }

    /* prompt */
    const char *value = json_member(body, "prompt");
    if (!json_read_string(value, job->prompt, sizeof(job->prompt)) || !job->prompt[0]) {
        BAD_REQUEST("VALIDATION_ERROR", "prompt", value ? "prompt must be a non-empty string"
                                                        : "prompt is required");
        goto reject;
    }
    if ((int)strlen(job->prompt) > ctx->prompt_max_chars) {
        BAD_REQUEST("VALIDATION_ERROR", "prompt", "prompt exceeds %d characters",
                    ctx->prompt_max_chars);
        goto reject;
    }

    /* mode */
    char mode_str[32] = {0};
    value = json_member(body, "mode");
    if (json_given(value) && (!json_read_string(value, mode_str, sizeof(mode_str)) ||
                              !parse_mode(mode_str, &job->mode))) {
        BAD_REQUEST("VALIDATION_ERROR", "mode", "mode must be T2VA, I2VA, FL2VA or Ref2VA");
        goto reject;
    }

    /* size */
    const char *size = json_member(body, "size");
    job->width = 512;
    job->height = 512;
    FIELD(json_read_int, size, "width", "size.width", &job->width, "an integer");
    FIELD(json_read_int, size, "height", "size.height", &job->height, "an integer");

    /* duration: frames | seconds */
    const char *duration = json_member(body, "duration");
    int has_frames = json_given(json_member(duration, "frames"));
    int has_seconds = json_given(json_member(duration, "seconds"));
    if (has_frames && has_seconds) {
        BAD_REQUEST("VALIDATION_ERROR", "duration", "frames and seconds are mutually exclusive");
        goto reject;
    }
    job->frames = 22;
    FIELD(json_read_int, duration, "frames", "duration.frames", &job->frames, "an integer");
    double seconds = 0.0;
    FIELD(json_read_double, duration, "seconds", "duration.seconds", &seconds, "a number");
    if (has_seconds) {
        if (seconds <= 0.0 || seconds * H3_FPS > INT_MAX) {
            BAD_REQUEST("VALIDATION_ERROR", "duration.seconds", "seconds must be positive");
            goto reject;
        }
        job->seconds = (float)seconds;
        job->frames = (int)llround(seconds * H3_FPS);
    }

    /* quality */
    const char *quality = json_member(body, "quality");
    snprintf(job->quality.preset, sizeof(job->quality.preset), "%s", H3D_QUALITY_DEFAULT_PRESET);
    value = json_member(quality, "preset");
    if (json_given(value) && !json_read_string(value, job->quality.preset,
                                               sizeof(job->quality.preset))) {
        BAD_REQUEST("VALIDATION_ERROR", "quality.preset", "quality.preset must be a string");
        goto reject;
    }
    FIELD(json_read_int, quality, "steps", "quality.steps", &job->quality.steps, "an integer");
    FIELD(json_read_int, quality, "layers", "quality.layers", &job->quality.layers, "an integer");
    FIELD(json_read_int, quality, "reuse", "quality.reuse", &job->quality.reuse, "an integer");
    if (!resolve_quality(&job->quality)) {
        BAD_REQUEST("VALIDATION_ERROR", "quality.preset",
                    "quality.preset must be fox-s2, fox-fast, cinematic or custom");
        goto reject;
    }

    /* seed */
    value = json_member(body, "seed");
    if (json_given(value)) {
        if (!json_read_u64(value, &job->seed)) {
            BAD_REQUEST("VALIDATION_ERROR", "seed", "seed must be a non-negative integer or null");
            goto reject;
        }
        job->seed_was_set = 1;
    }

    /* Same limits h3_generate enforces, so bad requests fail at submit. */
    if (!h3d_geometry_is_valid(job->width, job->height, job->frames)) {
        UNPROCESSABLE("UNSUPPORTED_COMBINATION", "size",
                      "unsupported geometry %dx%d, %d frames: width and height must be "
                      "multiples of %d with at most %d pixels, frames in 5..362",
                      job->width, job->height, job->frames, H3_CANVAS_MULTIPLE,
                      H3_MAX_PIXELS);
        goto reject;
    }
    if (job->quality.steps < 2 || job->quality.steps > H3_MAX_STEPS ||
        job->quality.layers < 35 || job->quality.layers > 50 ||
        job->quality.reuse < 1 || job->quality.reuse > 3) {
        UNPROCESSABLE("UNSUPPORTED_COMBINATION", "quality",
                      "steps must be in 2..%d, layers in 35..50, reuse in 1..3",
                      H3_MAX_STEPS);
        goto reject;
    }

    /* refs */
    const char *refs = json_member(body, "refs");
    int keyframe_mode = job->mode != H3D_MODE_REF2VA;
    job->refs_image_count = parse_ref_list(ctx, fd, refs, "images", job->refs_images,
                                           keyframe_mode ? 2 : H3D_MAX_REF_IMAGES);
    if (job->refs_image_count < 0) goto reject;
    job->refs_video_count = parse_ref_list(ctx, fd, refs, "video", job->refs_videos,
                                           keyframe_mode ? 0 : H3D_MAX_REF_VIDEOS);
    if (job->refs_video_count < 0) goto reject;
    if (!job->refs_video_count) {
        job->refs_video_count = parse_ref_list(ctx, fd, refs, "videos", job->refs_videos,
                                               keyframe_mode ? 0 : H3D_MAX_REF_VIDEOS);
        if (job->refs_video_count < 0) goto reject;
    }
    job->refs_audio_count = parse_ref_list(ctx, fd, refs, "audio", job->refs_audio,
                                           keyframe_mode ? 0 : H3D_MAX_REF_AUDIOS);
    if (job->refs_audio_count < 0) goto reject;
    int has_first = parse_ref_single(ctx, fd, refs, "first_frame", job->first_frame,
                                     sizeof(job->first_frame));
    if (has_first < 0) goto reject;
    int has_last = parse_ref_single(ctx, fd, refs, "last_frame", job->last_frame,
                                    sizeof(job->last_frame));
    if (has_last < 0) goto reject;

    if (keyframe_mode) {
        /* refs.images is the positional form: [first] or [first, last]. */
        if (job->refs_image_count && (has_first || has_last)) {
            BAD_REQUEST("VALIDATION_ERROR", "refs",
                        "use refs.images or refs.first_frame/last_frame, not both");
            goto reject;
        }
        _Static_assert(sizeof(job->first_frame) == sizeof(job->refs_images[0].path),
                       "keyframe and reference paths share one size");
        if (job->refs_image_count >= 1)
            memcpy(job->first_frame, job->refs_images[0].path, sizeof(job->first_frame));
        if (job->refs_image_count == 2)
            memcpy(job->last_frame, job->refs_images[1].path, sizeof(job->last_frame));
        job->refs_image_count = 0;
        if (job->mode == H3D_MODE_T2VA && (job->first_frame[0] || job->last_frame[0])) {
            BAD_REQUEST("VALIDATION_ERROR", "refs", "T2VA takes no keyframes; use FL2VA");
            goto reject;
        }
        if (job->mode == H3D_MODE_I2VA && (!job->first_frame[0] || job->last_frame[0])) {
            BAD_REQUEST("VALIDATION_ERROR", "refs", "I2VA needs exactly one first frame");
            goto reject;
        }
    } else {
        if (has_first || has_last) {
            BAD_REQUEST("VALIDATION_ERROR", "refs", "Ref2VA takes no first/last frame");
            goto reject;
        }
        if (!job->refs_image_count && !job->refs_video_count && !job->refs_audio_count) {
            BAD_REQUEST("VALIDATION_ERROR", "refs", "Ref2VA needs at least one reference");
            goto reject;
        }
    }

    /* options */
    const char *options = json_member(body, "options");
    FIELD(json_read_bool, options, "finalize_partial_on_cancel",
          "options.finalize_partial_on_cancel", &job->finalize_partial_on_cancel, "a boolean");
    FIELD(json_read_bool, options, "write_preview_frames", "options.write_preview_frames",
          &job->write_preview_frames, "a boolean");
    FIELD(json_read_bool, options, "release_after", "options.release_after",
          &job->release_after, "a boolean");
    int token_reduction = -1;
    FIELD(json_read_bool, options, "token_reduction", "options.token_reduction",
          &token_reduction, "a boolean");
    int fasth3 = 0;
    FIELD(json_read_bool, options, "fasth3", "options.fasth3", &fasth3, "a boolean");
    if (json_given(json_member(options, "token_reduction_schedule"))) {
        UNPROCESSABLE("ARG_REJECTED", "options.token_reduction_schedule",
                      "token_reduction_schedule is not supported by this daemon");
        goto reject;
    }
    if (!parse_extra_args(job, fd, json_member(options, "extra_args"))) goto reject;
    if (token_reduction == 0 && job->token_reduction) {
        UNPROCESSABLE("ARG_REJECTED", "options.token_reduction",
                      "token_reduction is false but extra_args has --token-reduction");
        goto reject;
    }
    if (token_reduction == 1) job->token_reduction = 1;
    if (fasth3) {
        if (job->fasth3_lora[0]) {
            UNPROCESSABLE("ARG_REJECTED", "options.fasth3",
                          "use options.fasth3 or --fasth3-lora in extra_args, not both");
            goto reject;
        }
        if (!ctx->fasth3_lora[0]) {
            UNPROCESSABLE("ARG_REJECTED", "options.fasth3",
                          "no FastH3 adapter on this daemon; set H3D_FASTH3_LORA or run "
                          "tools/download_fasth3_lora.sh");
            goto reject;
        }
        snprintf(job->fasth3_lora, sizeof(job->fasth3_lora), "%s", ctx->fasth3_lora);
    }

    /* idempotency */
    value = json_member(body, "idempotency_key");
    if (json_given(value) && !json_read_string(value, job->idempotency_key,
                                               sizeof(job->idempotency_key))) {
        BAD_REQUEST("VALIDATION_ERROR", "idempotency_key", "idempotency_key must be a short string");
        goto reject;
    }
    if (job->idempotency_key[0]) {
        h3d_job *existing = h3d_job_find_by_idempotency(ctx, job->idempotency_key);
        if (existing) {
            h3d_job_free(job);
            send_job(fd, 200, "OK", existing);
            return;
        }
    }

    if (h3d_job_count_by_status(ctx, H3D_JOB_QUEUED) >= ctx->max_queued_per_client) {
        char buf[512];
        send_json(fd, 429, "Too Many Requests", buf,
                  h3d_json_queue_full_response(buf, sizeof(buf)));
        goto free_job;
    }

    build_cli_args(job->cli_args, sizeof(job->cli_args), ctx, job);
    h3d_job_record_event(job, "status", "{\"status\":\"queued\",\"queue_position\":1}");
    if (h3d_job_enqueue(ctx, job) < 0) {
        send_error(fd, 500, "Internal Server Error", "INTERNAL", NULL,
                   "cannot create the job output directory");
        goto free_job;
    }

    char resp[512];
    int rlen = snprintf(resp, sizeof(resp),
        "{\"job_id\":\"%s\",\"status\":\"queued\","
        "\"queue_position\":1,"
        "\"events_url\":\"/v1/jobs/%s/events\","
        "\"submitted_at\":%ld}",
        job->job_id, job->job_id, (long)job->created_at);
    send_json(fd, 202, "Accepted", resp, rlen);
    return;

reject:
free_job:
    h3d_job_free(job);
}

#undef FIELD

static void send_job_not_found(int fd) {
    send_error(fd, 404, "Not Found", "JOB_NOT_FOUND", NULL, "No job with that ID");
}

static void handle_status(h3d_ctx *ctx, int fd, const char *job_id) {
    h3d_job *job = h3d_job_find(ctx, job_id);
    if (!job) {
        send_job_not_found(fd);
        return;
    }
    send_job(fd, 200, "OK", job);
}

static void handle_cancel(h3d_ctx *ctx, int fd, const char *job_id) {
    int result = h3d_job_cancel(ctx, job_id);
    h3d_job *job = h3d_job_find(ctx, job_id);
    if (result < 0 || !job) {
        send_job_not_found(fd);
        return;
    }
    if (result == 0) send_job(fd, 409, "Conflict", job);
    else send_job(fd, 200, "OK", job);
}

static int is_terminal_event(const char *event) {
    return !strncmp(event, "event: done\n", 12) ||
           !strncmp(event, "event: failed\n", 14) ||
           !strncmp(event, "event: cancelled\n", 17) ||
           !strncmp(event, "event: timeout\n", 15);
}

/* Send one recorded event. Terminal events carry the full job object, built
 * now so it is never cut to the ring slot size. Returns -1 once the peer is
 * gone, 1 after a terminal event, 0 otherwise. */
static int send_event(int fd, h3d_job *job, const char *event) {
    if (!is_terminal_event(event))
        return h3d_http_send_chunk(fd, event, strlen(event)) < 0 ? -1 : 0;
    char name[32];
    size_t n = strcspn(event + 7, "\n");
    if (n >= sizeof(name)) n = sizeof(name) - 1;
    memcpy(name, event + 7, n);
    name[n] = '\0';
    size_t cap = H3D_JOB_JSON_MAX + 64;
    char *buf = malloc(cap);
    if (!buf) return -1;
    int len = snprintf(buf, cap, "event: %s\ndata: ", name);
    int json = h3d_json_job_response(job, buf + len, cap - (size_t)len - 2);
    int ok = -1;
    if (json >= 0) {
        len += json;
        buf[len++] = '\n';
        buf[len++] = '\n';
        ok = h3d_http_send_chunk(fd, buf, (size_t)len) < 0 ? -1 : 1;
    }
    free(buf);
    return ok;
}

static void handle_events(h3d_ctx *ctx, int fd, const char *job_id) {
    h3d_job *job = h3d_job_find(ctx, job_id);
    if (!job) {
        send_job_not_found(fd);
        return;
    }
    if (h3d_http_send_sse_headers(fd) < 0) return;

    char event[sizeof(job->events[0])];
    pthread_mutex_lock(&job->lock);
    int next = job->event_tail;
    int pending = job->event_count;
    pthread_mutex_unlock(&job->lock);

    time_t last_heartbeat = time(NULL);
    time_t terminal_since = 0;
    while (!atomic_load(&ctx->shutdown)) {
        /* Drain everything recorded so far, copying each slot under the lock. */
        int sent = 0;
        while (1) {
            pthread_mutex_lock(&job->lock);
            if (pending == 0 && next == job->event_head) {
                pthread_mutex_unlock(&job->lock);
                break;
            }
            memcpy(event, job->events[next], sizeof(event));
            next = (next + 1) % H3D_SSE_EVENT_HISTORY;
            if (pending > 0) pending--;
            pthread_mutex_unlock(&job->lock);
            sent = send_event(fd, job, event);
            if (sent != 0) break;
        }
        if (sent != 0) break;

        pthread_mutex_lock(&job->lock);
        int terminal = h3d_job_is_terminal(job->status);
        pthread_mutex_unlock(&job->lock);
        time_t now = time(NULL);
        /* The worker records the terminal event just after setting the
         * status; stop on our own if it never arrives. */
        if (terminal) {
            if (!terminal_since) terminal_since = now;
            else if (now - terminal_since >= 2) break;
        }
        if (now - last_heartbeat >= H3D_SSE_HEARTBEAT_SEC) {
            if (h3d_http_send_sse_heartbeat(fd) < 0) return;
            last_heartbeat = now;
        }
        usleep(100000);
    }
    h3d_http_end_chunks(fd);
}

static void handle_download(h3d_ctx *ctx, int fd, const char *job_id) {
    h3d_job *job = h3d_job_find(ctx, job_id);
    if (!job) {
        send_job_not_found(fd);
        return;
    }
    pthread_mutex_lock(&job->lock);
    int done = job->status == H3D_JOB_DONE;
    pthread_mutex_unlock(&job->lock);
    if (!done || !job->mp4_path) {
        send_error(fd, 409, "Conflict", "NOT_READY", NULL, "Job not completed or no output");
        return;
    }

    FILE *fp = fopen(job->mp4_path, "rb");
    struct stat st;
    if (!fp || fstat(fileno(fp), &st) < 0) {
        if (fp) fclose(fp);
        send_error(fd, 404, "Not Found", "FILE_NOT_FOUND", NULL, "Output file not found on disk");
        return;
    }

    const char *basename = strrchr(job->mp4_path, '/');
    basename = basename ? basename + 1 : job->mp4_path;
    char header[1024];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: video/mp4\r\n"
        "Content-Length: %lld\r\n"
        "Content-Disposition: attachment; filename=\"%s\"\r\n"
        "Connection: close\r\n"
        "\r\n",
        (long long)st.st_size, basename);
    if (h3d_http_write(fd, header, (size_t)hlen) == 0) {
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), fp)) > 0)
            if (h3d_http_write(fd, buf, n) < 0) break;
    }
    fclose(fp);
}

static void handle_jobs_list(h3d_ctx *ctx, int fd, const char *query) {
    char status_filter[32] = {0};
    int limit = 50;

    /* Parse query string: ?status=running&limit=50 */
    const char *p = query ? strchr(query, '?') : NULL;
    if (p) {
        p++;
        while (*p) {
            if (strncmp(p, "status=", 7) == 0) {
                p += 7;
                size_t i = 0;
                while (*p && *p != '&' && i < sizeof(status_filter) - 1)
                    status_filter[i++] = *p++;
                status_filter[i] = '\0';
            } else if (strncmp(p, "limit=", 6) == 0) {
                limit = atoi(p + 6);
            }
            while (*p && *p != '&') p++;
            if (*p == '&') p++;
        }
    }
    if (limit < 1) limit = 1;
    if (limit > 500) limit = 500;

    char *buf = malloc(131072);
    if (!buf) {
        send_json(fd, 500, "Internal Server Error", NULL, -1);
        return;
    }
    send_json(fd, 200, "OK", buf,
              h3d_json_jobs_list_response(ctx, status_filter[0] ? status_filter : NULL,
                                          limit, buf, 131072));
    free(buf);
}

/* ── Router ──────────────────────────────────────────────────── */

static void handle_request(h3d_ctx *ctx, int fd,
                           const char *method, const char *path,
                           const char *headers, const char *body) {
    char client_proto[32];
    if (header_value(headers, "X-H3-Protocol", client_proto, sizeof(client_proto)) &&
        strcmp(client_proto, H3D_PROTOCOL) != 0) {
        send_protocol_mismatch(fd);
        return;
    }

    if (strcmp(path, "/v1/info") == 0 && strcmp(method, "GET") == 0) {
        handle_info(ctx, fd);
    } else if (strcmp(path, "/v1/cache/release") == 0 && strcmp(method, "POST") == 0) {
        handle_cache_release(ctx, fd);
    } else if (strcmp(path, "/v1/jobs") == 0 && strcmp(method, "POST") == 0) {
        handle_submit(ctx, fd, body);
    } else if (strncmp(path, "/v1/jobs", 8) == 0 &&
               (path[8] == '\0' || path[8] == '?') &&
               strcmp(method, "GET") == 0) {
        handle_jobs_list(ctx, fd, path);
    } else if (strncmp(path, "/v1/jobs/", 9) == 0) {
        const char *job_id_raw = path + 9;
        size_t id_len = strcspn(job_id_raw, "/?");
        const char *suffix = job_id_raw[id_len] == '/' ? job_id_raw + id_len + 1 : NULL;
        char id[H3D_MAX_JOB_ID] = {0};
        if (id_len >= sizeof(id)) id_len = sizeof(id) - 1;
        memcpy(id, job_id_raw, id_len);

        if (suffix && strcmp(suffix, "events") == 0 && strcmp(method, "GET") == 0) {
            handle_events(ctx, fd, id);
        } else if (suffix && strcmp(suffix, "cancel") == 0 && strcmp(method, "POST") == 0) {
            handle_cancel(ctx, fd, id);
        } else if (suffix && strcmp(suffix, "download") == 0 && strcmp(method, "GET") == 0) {
            handle_download(ctx, fd, id);
        } else if (!suffix && strcmp(method, "GET") == 0) {
            handle_status(ctx, fd, id);
        } else {
            send_error(fd, 404, "Not Found", "NOT_FOUND", NULL, "Unknown endpoint");
        }
    } else {
        send_error(fd, 404, "Not Found", "NOT_FOUND", NULL, "Unknown endpoint");
    }
}

/* ── Connections ─────────────────────────────────────────────── */

typedef struct {
    h3d_ctx *ctx;
    int fd;
} h3d_connection;

/* One request per connection: every response says Connection: close, so a
 * slow or idle client only ever holds its own thread. */
static void *connection_main(void *arg) {
    h3d_connection *conn = arg;
    h3d_ctx *ctx = conn->ctx;
    int fd = conn->fd;
    free(conn);

    enum { BODY_MAX = 65536 };
    char *body = malloc(BODY_MAX);
    char method[16], path[4096], headers[8192];
    size_t body_len = 0;
    if (body) {
        int r = h3d_http_read_request(fd, method, sizeof(method), path, sizeof(path),
                                      headers, sizeof(headers), body, BODY_MAX, &body_len);
        if (r == 0) {
            handle_request(ctx, fd, method, path, headers, body);
        } else if (r == H3D_HTTP_TOO_LARGE) {
            send_error(fd, 413, "Payload Too Large", "VALIDATION_ERROR", NULL,
                       "request body exceeds %d bytes", BODY_MAX - 1);
        }
    }
    free(body);
    h3d_http_close(fd);
    atomic_fetch_sub(&ctx->connections, 1);
    return NULL;
}

/* ── Daemon main ─────────────────────────────────────────────── */

int h3d_serve(h3d_ctx *ctx) {
    g_ctx = ctx;

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    signal(SIGHUP, SIG_IGN);
    signal(SIGPIPE, SIG_IGN);

    atomic_store(&ctx->ready, 0);
    atomic_store(&ctx->connections, 0);
    snprintf(ctx->unready_reason, sizeof(ctx->unready_reason), "Initializing workers");

    h3d_jobs_init(ctx);
    if (h3d_workers_init(ctx) < 0) {
        fprintf(stderr, "h3d: no workers available\n");
        return 1;
    }

    ctx->server_fd = h3d_http_listen(ctx->bind_addr, ctx->port);
    if (ctx->server_fd < 0) {
        fprintf(stderr, "h3d: cannot bind %s:%d: %s\n",
                ctx->bind_addr, ctx->port, strerror(errno));
        return 1;
    }

    atomic_store(&ctx->ready, 1);
    ctx->unready_reason[0] = '\0';

    fprintf(stderr, "h3d: listening on %s:%d (%d worker%s) [protocol=%s]\n",
            ctx->bind_addr, ctx->port, ctx->worker_count,
            ctx->worker_count > 1 ? "s" : "", H3D_PROTOCOL);

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    while (!atomic_load(&ctx->shutdown)) {
        int fd = h3d_http_accept(ctx->server_fd);
        if (fd < 0) {
            if (errno == EINTR || errno == ECONNABORTED) continue;
            break;
        }
        if (atomic_fetch_add(&ctx->connections, 1) >= H3D_MAX_CONNECTIONS) {
            atomic_fetch_sub(&ctx->connections, 1);
            send_error(fd, 503, "Service Unavailable", "NOT_READY", NULL,
                       "too many open connections");
            h3d_http_close(fd);
            continue;
        }
        h3d_connection *conn = malloc(sizeof(*conn));
        pthread_t thread;
        if (!conn) {
            atomic_fetch_sub(&ctx->connections, 1);
            h3d_http_close(fd);
            continue;
        }
        conn->ctx = ctx;
        conn->fd = fd;
        if (pthread_create(&thread, &attr, connection_main, conn) != 0) {
            free(conn);
            atomic_fetch_sub(&ctx->connections, 1);
            h3d_http_close(fd);
        }
    }
    pthread_attr_destroy(&attr);

    h3d_http_close(ctx->server_fd);
    h3d_workers_shutdown(ctx);
    fprintf(stderr, "h3d: shutdown complete\n");
    return 0;
}
