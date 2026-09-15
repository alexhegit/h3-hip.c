#include "h3d.h"
#include "h3d_config.h"
#include "h3d_http.h"
#include "h3d_json.h"
#include "h3d_job.h"
#include "h3d_worker.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static h3d_ctx *g_ctx = NULL;

static void signal_handler(int sig) {
    (void)sig;
    if (g_ctx) {
        atomic_store(&g_ctx->shutdown, 1);
    }
}

/* ── JSON parsing helpers ─────────────────────────────────────── */

static const char *json_find_string(const char *json, const char *key,
                                    char *value, size_t value_len) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ') p++;
    if (*p != '"') return NULL;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i < value_len - 1) {
        if (*p != '\\') { value[i++] = *p++; continue; }
        p++;
        if (!*p) break;
        switch (*p) {
        case 'n': value[i++] = '\n'; p++; break;
        case 't': value[i++] = '\t'; p++; break;
        case 'r': value[i++] = '\r'; p++; break;
        case 'b': value[i++] = '\b'; p++; break;
        case 'f': value[i++] = '\f'; p++; break;
        case 'u': {
            /* \uXXXX → UTF-8; surrogate pairs are decoded when both halves
             * are present, otherwise the code unit is replaced. */
            unsigned cp = 0;
            int ok = 1;
            for (int k = 1; k <= 4; k++) {
                char c = p[k];
                if (c >= '0' && c <= '9') cp = cp * 16u + (unsigned)(c - '0');
                else if (c >= 'a' && c <= 'f') cp = cp * 16u + (unsigned)(c - 'a' + 10);
                else if (c >= 'A' && c <= 'F') cp = cp * 16u + (unsigned)(c - 'A' + 10);
                else { ok = 0; break; }
            }
            if (!ok) { value[i++] = *p++; break; }
            p += 5;
            if (cp >= 0xD800 && cp <= 0xDBFF && p[0] == '\\' && p[1] == 'u') {
                unsigned lo = 0;
                int ok2 = 1;
                for (int k = 2; k <= 5; k++) {
                    char c = p[k];
                    if (c >= '0' && c <= '9') lo = lo * 16u + (unsigned)(c - '0');
                    else if (c >= 'a' && c <= 'f') lo = lo * 16u + (unsigned)(c - 'a' + 10);
                    else if (c >= 'A' && c <= 'F') lo = lo * 16u + (unsigned)(c - 'A' + 10);
                    else { ok2 = 0; break; }
                }
                if (ok2 && lo >= 0xDC00 && lo <= 0xDFFF) {
                    cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                    p += 6;
                }
            }
            if (cp >= 0xD800 && cp <= 0xDFFF) cp = 0xFFFD;
            char utf8[4];
            size_t n;
            if (cp < 0x80) { utf8[0] = (char)cp; n = 1; }
            else if (cp < 0x800) {
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
            if (i + n > value_len - 1) goto done;
            memcpy(value + i, utf8, n);
            i += n;
            break;
        }
        default: value[i++] = *p++; break;
        }
    }
done:
    value[i] = '\0';
    /* Leave p on the closing quote so nested lookups resume correctly even
     * when the value was truncated. */
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) p++;
        p++;
    }
    return p;
}

static int json_find_int(const char *json, const char *key, int *value) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(json, pattern);
    if (!p) return 0;
    p += strlen(pattern);
    while (*p == ' ') p++;
    if (*p == 'n') return 0;
    char *end = NULL;
    long v = strtol(p, &end, 10);
    if (end == p) return 0;
    *value = (int)v;
    return 1;
}

static int json_find_u64(const char *json, const char *key, uint64_t *value) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(json, pattern);
    if (!p) return 0;
    p += strlen(pattern);
    while (*p == ' ') p++;
    if (*p == 'n') return 0;
    char *end = NULL;
    unsigned long long v = strtoull(p, &end, 10);
    if (end == p) return 0;
    *value = (uint64_t)v;
    return 1;
}

static int json_find_float(const char *json, const char *key, float *value) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(json, pattern);
    if (!p) return 0;
    p += strlen(pattern);
    while (*p == ' ') p++;
    if (*p == 'n') return 0;
    char *end = NULL;
    float v = strtof(p, &end);
    if (end == p) return 0;
    *value = v;
    return 1;
}

static int json_find_bool(const char *json, const char *key, int *value) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(json, pattern);
    if (!p) return 0;
    p += strlen(pattern);
    while (*p == ' ') p++;
    if (strncmp(p, "true", 4) == 0) { *value = 1; return 1; }
    if (strncmp(p, "false", 5) == 0) { *value = 0; return 1; }
    return 0;
}

/* Parse nested object: "key":{"subkey":...} and find string within */
static const char *json_find_nested_string(const char *json, const char *key,
                                           const char *subkey,
                                           char *value, size_t value_len) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":{", key);
    const char *p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    /* Now find subkey within this object */
    return json_find_string(p, subkey, value, value_len);
}

static int json_find_nested_int(const char *json, const char *key,
                                const char *subkey, int *value) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":{", key);
    const char *p = strstr(json, pattern);
    if (!p) return 0;
    p += strlen(pattern);
    return json_find_int(p, subkey, value);
}

static int json_find_nested_float(const char *json, const char *key,
                                  const char *subkey, float *value) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\":{", key);
    const char *p = strstr(json, pattern);
    if (!p) return 0;
    p += strlen(pattern);
    return json_find_float(p, subkey, value);
}

/* ── Protocol version check ──────────────────────────────────── */

static int check_protocol(const char *headers) {
    /* Look for X-H3-Protocol header */
    const char *p = strstr(headers, "X-H3-Protocol:");
    if (!p) p = strstr(headers, "x-h3-protocol:");
    if (!p) return -1;  /* No header = accept */
    p += 15;
    while (*p == ' ') p++;
    char client_proto[32] = {0};
    size_t i = 0;
    while (*p && *p != '\r' && *p != '\n' && i < sizeof(client_proto) - 1) {
        client_proto[i++] = *p++;
    }
    return strcmp(client_proto, H3D_PROTOCOL) == 0 ? 0 : -2;
}

static void send_protocol_mismatch(int fd, const char *client_proto) {
    (void)client_proto;
    char body[512];
    int len = snprintf(body, sizeof(body),
        "{\"error\":{\"code\":\"PROTOCOL_MISMATCH\","
        "\"message\":\"Protocol version mismatch\","
        "\"server_protocol\":\"%s\","
        "\"supported_protocols\":[\"%s\"]}}",
        H3D_PROTOCOL, H3D_PROTOCOL);
    h3d_http_send_response(fd, 426, "Upgrade Required",
                           "application/json; charset=utf-8",
                           body, (size_t)len);
}

/* ── Request not ready ───────────────────────────────────────── */

static void send_not_ready(h3d_ctx *ctx, int fd) {
    char body[1024];
    char escaped[256];
    h3d_json_escape(escaped, sizeof(escaped), ctx->unready_reason);
    int len = snprintf(body, sizeof(body),
        "{\"error\":{\"code\":\"NOT_READY\","
        "\"message\":\"Daemon not ready: %s\"}}", escaped);
    h3d_http_send_response(fd, 503, "Service Unavailable",
                           "application/json; charset=utf-8",
                           body, (size_t)len);
}

/* ── Validation helpers ──────────────────────────────────────── */

static int is_path_under_root(const char *path, const char *root) {
    if (!path || !root) return 0;
    /* Resolve ~ if present */
    char resolved_root[1024];
    if (root[0] == '~') {
        const char *home = getenv("HOME");
        if (home) snprintf(resolved_root, sizeof(resolved_root), "%s%s", home, root + 1);
        else return 0;
    } else {
        snprintf(resolved_root, sizeof(resolved_root), "%s", root);
    }
    /* Normalize trailing slash */
    size_t rlen = strlen(resolved_root);
    if (rlen > 0 && resolved_root[rlen - 1] == '/')
        resolved_root[rlen - 1] = '\0';

    return strncmp(path, resolved_root, strlen(resolved_root)) == 0 &&
           (path[strlen(resolved_root)] == '/' || path[strlen(resolved_root)] == '\0');
}

/* ── Mode parsing ────────────────────────────────────────────── */

static h3d_mode parse_mode(const char *mode_str) {
    if (!mode_str || !*mode_str) return H3D_MODE_T2VA;
    if (strcmp(mode_str, "T2VA") == 0) return H3D_MODE_T2VA;
    if (strcmp(mode_str, "I2VA") == 0) return H3D_MODE_I2VA;
    if (strcmp(mode_str, "FL2VA") == 0) return H3D_MODE_FL2VA;
    if (strcmp(mode_str, "Ref2VA") == 0) return H3D_MODE_REF2VA;
    return H3D_MODE_T2VA;
}

/* ── Quality preset resolution ───────────────────────────────── */

static void resolve_quality(h3d_quality *q) {
    if (strcmp(q->preset, "fox-s2") == 0) {
        if (q->steps == 0) q->steps = 2;
        if (q->layers == 0) q->layers = 35;
        if (q->reuse == 0) q->reuse = 1;
    } else if (strcmp(q->preset, "fox-fast") == 0) {
        if (q->steps == 0) q->steps = 20;
        if (q->layers == 0) q->layers = 45;
        if (q->reuse == 0) q->reuse = 2;
    } else if (strcmp(q->preset, "cinematic") == 0) {
        if (q->steps == 0) q->steps = 20;
        if (q->layers == 0) q->layers = 45;
        if (q->reuse == 0) q->reuse = 2;
    } else {
        /* custom: defaults if not set */
        if (q->steps == 0) q->steps = 20;
        if (q->layers == 0) q->layers = 45;
        if (q->reuse == 0) q->reuse = 2;
    }
}

/* ── Check for rejected extra_args ───────────────────────────── */

static const char *rejected_args[] = {
    "--frames-dir", "--output", "-o", "--show", "--zoom",
    "--info", "--serve", NULL
};

static int has_rejected_arg(const char *extra_args) {
    if (!extra_args || !*extra_args) return 0;
    for (const char **r = rejected_args; *r; r++) {
        if (strstr(extra_args, *r)) return 1;
    }
    return 0;
}

/* ── Build CLI args echo ─────────────────────────────────────── */

static void build_cli_args(char *buf, size_t len, h3d_ctx *ctx,
                           const char *prompt, h3d_job *job) {
    char escaped_prompt[8192];
    h3d_json_escape(escaped_prompt, sizeof(escaped_prompt), prompt);

    int off = 0;
    int n;
    n = snprintf(buf + off, len - (size_t)off,
        "./h3 -d %s -p \"%s\"", ctx->model_path, escaped_prompt);
    off += n;

    n = snprintf(buf + off, len - (size_t)off,
        " --width %d --height %d", job->width, job->height);
    off += n;

    if (job->frames > 0) {
        n = snprintf(buf + off, len - (size_t)off, " --frames %d", job->frames);
        off += n;
    }
    if (job->seconds > 0) {
        n = snprintf(buf + off, len - (size_t)off, " --seconds %.1f", job->seconds);
        off += n;
    }

    n = snprintf(buf + off, len - (size_t)off,
        " --steps %d --layers %d --reuse %d",
        job->quality.steps, job->quality.layers, job->quality.reuse);
    off += n;

    if (job->seed_was_set) {
        n = snprintf(buf + off, len - (size_t)off, " --seed %" PRIu64, job->seed);
        off += n;
    }

    if (job->extra_args[0]) {
        n = snprintf(buf + off, len - (size_t)off, " %s", job->extra_args);
        off += n;
    }

    snprintf(buf + off, len - (size_t)off, " -o %s", job->mp4_path);
}

/* ── Handlers ────────────────────────────────────────────────── */

static void handle_info(h3d_ctx *ctx, int fd) {
    char body[8192];
    int len = h3d_json_info_response(ctx, body, sizeof(body));
    h3d_http_send_response(fd, 200, "OK", "application/json; charset=utf-8",
                           body, (size_t)len);
}

static void handle_submit(h3d_ctx *ctx, int fd, const char *body,
                          size_t body_len) {
    (void)body_len;

    /* 503 if not ready */
    if (!atomic_load(&ctx->ready)) {
        send_not_ready(ctx, fd);
        return;
    }

    /* ── Parse prompt (required) ── */
    char prompt[8192] = {0};
    if (!json_find_string(body, "prompt", prompt, sizeof(prompt)) || !prompt[0]) {
        h3d_error err = { .code = "VALIDATION_ERROR", .message = "prompt is required" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 400, "Bad Request",
                               "application/json; charset=utf-8", buf, (size_t)len);
        return;
    }
    if ((int)strlen(prompt) > ctx->prompt_max_chars) {
        h3d_error err = { .code = "VALIDATION_ERROR", .message = "prompt exceeds maximum length" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 400, "Bad Request",
                               "application/json; charset=utf-8", buf, (size_t)len);
        return;
    }

    /* ── Parse mode ── */
    char mode_str[32] = {0};
    json_find_string(body, "mode", mode_str, sizeof(mode_str));
    h3d_mode mode = parse_mode(mode_str);

    /* ── Parse size ── */
    int width = 512, height = 512;
    json_find_nested_int(body, "size", "width", &width);
    json_find_nested_int(body, "size", "height", &height);

    /* ── Parse duration (frames | seconds, mutually exclusive) ── */
    int frames = 0;
    float seconds = 0;
    int has_frames = json_find_nested_int(body, "duration", "frames", &frames);
    int has_seconds = json_find_nested_float(body, "duration", "seconds", &seconds);
    if (has_frames && has_seconds) {
        h3d_error err = { .code = "VALIDATION_ERROR",
                          .message = "duration: frames and seconds are mutually exclusive" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 400, "Bad Request",
                               "application/json; charset=utf-8", buf, (size_t)len);
        return;
    }
    if (!has_frames && !has_seconds) {
        frames = 22;  /* default */
    }

    /* ── Parse quality ── */
    h3d_quality quality = {0};
    snprintf(quality.preset, sizeof(quality.preset), "%s", H3D_QUALITY_DEFAULT_PRESET);
    json_find_string(body, "preset", quality.preset, sizeof(quality.preset));
    json_find_nested_int(body, "quality", "steps", &quality.steps);
    json_find_nested_int(body, "quality", "layers", &quality.layers);
    json_find_nested_int(body, "quality", "reuse", &quality.reuse);
    resolve_quality(&quality);

    /* ── Parse seed ── */
    uint64_t seed = 0;
    int seed_was_set = json_find_u64(body, "seed", &seed);

    /* ── Geometry validation ── */
    if (!h3d_geometry_is_valid(ctx, width, height, frames)) {
        char details[1024];
        snprintf(details, sizeof(details),
            "Unsupported geometry: %dx%d %d frames. "
            "See /v1/info for supported combinations.", width, height, frames);
        h3d_error err = { .code = "UNSUPPORTED_COMBINATION", .message = "" };
        snprintf(err.message, sizeof(err.message), "%s", details);
        char buf[1024];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 422, "Unprocessable Entity",
                               "application/json; charset=utf-8", buf, (size_t)len);
        return;
    }

    /* ── Parse refs ── */
    h3d_ref_entry refs_images[H3D_MAX_REFERENCES];
    int refs_image_count = 0;
    h3d_ref_entry refs_video = {0};
    int has_refs_video = 0;
    h3d_ref_entry refs_audio[3];
    int refs_audio_count = 0;

    /* Parse refs.images array */
    const char *images_start = strstr(body, "\"images\"");
    if (images_start) {
        images_start = strchr(images_start, '[');
        if (images_start) {
            images_start++;
            while (*images_start && *images_start != ']' && refs_image_count < H3D_MAX_REFERENCES) {
                while (*images_start == ' ' || *images_start == ',') images_start++;
                if (*images_start == '"') {
                    images_start++;
                    size_t pi = 0;
                    while (*images_start && *images_start != '"' && pi < sizeof(refs_images[0].path) - 1) {
                        refs_images[refs_image_count].path[pi++] = *images_start++;
                    }
                    refs_images[refs_image_count].path[pi] = '\0';
                    if (pi > 0) refs_image_count++;
                } else {
                    break;
                }
            }
        }
    }

    /* Validate refs paths are under media_root */
    for (int i = 0; i < refs_image_count; i++) {
        if (!is_path_under_root(refs_images[i].path, ctx->media_root)) {
            h3d_error err = { .code = "PATH_OUT_OF_ROOT",
                              .message = "Reference file not in media_root" };
            char buf[512];
            int len = h3d_json_error_response(&err, buf, sizeof(buf));
            h3d_http_send_response(fd, 400, "Bad Request",
                                   "application/json; charset=utf-8", buf, (size_t)len);
            return;
        }
    }

    /* Validate Ref2VA limits */
    if (mode == H3D_MODE_REF2VA) {
        if (refs_image_count < 1) {
            h3d_error err = { .code = "VALIDATION_ERROR",
                              .message = "Ref2VA requires at least 1 reference image" };
            char buf[512];
            int len = h3d_json_error_response(&err, buf, sizeof(buf));
            h3d_http_send_response(fd, 400, "Bad Request",
                                   "application/json; charset=utf-8", buf, (size_t)len);
            return;
        }
        if (refs_image_count > 9) {
            h3d_error err = { .code = "REFS_LIMIT_EXCEEDED",
                              .message = "Ref2VA allows max 9 images" };
            char buf[512];
            int len = h3d_json_error_response(&err, buf, sizeof(buf));
            h3d_http_send_response(fd, 400, "Bad Request",
                                   "application/json; charset=utf-8", buf, (size_t)len);
            return;
        }
    }

    /* ── Parse options ── */
    int finalize_partial = 0;
    int write_preview = 0;
    json_find_nested_int(body, "options", "finalize_partial_on_cancel", &finalize_partial);
    json_find_nested_int(body, "options", "write_preview_frames", &write_preview);
    char extra_args[2048] = {0};
    json_find_nested_string(body, "options", "extra_args", extra_args, sizeof(extra_args));

    if (has_rejected_arg(extra_args)) {
        h3d_error err = { .code = "ARG_REJECTED",
                          .message = "extra_args contains rejected parameter" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 422, "Unprocessable Entity",
                               "application/json; charset=utf-8", buf, (size_t)len);
        return;
    }

    /* ── Parse idempotency_key ── */
    char idempotency_key[128] = {0};
    json_find_string(body, "idempotency_key", idempotency_key, sizeof(idempotency_key));

    /* Check idempotency */
    if (idempotency_key[0]) {
        h3d_job *existing = h3d_job_find_by_idempotency(ctx, idempotency_key);
        if (existing) {
            /* Return existing job */
            char resp[4096];
            int rlen = h3d_json_job_response(existing, resp, sizeof(resp));
            h3d_http_send_response(fd, 200, "OK",
                                   "application/json; charset=utf-8",
                                   resp, (size_t)rlen);
            return;
        }
    }

    /* ── Queue limit ── */
    if (h3d_job_count_by_status(ctx, H3D_JOB_QUEUED) >= ctx->max_queued_per_client) {
        char buf[512];
        int len = h3d_json_queue_full_response(buf, sizeof(buf));
        h3d_http_send_response(fd, 429, "Too Many Requests",
                               "application/json; charset=utf-8", buf, (size_t)len);
        return;
    }

    /* ── Create job ── */
    h3d_job *job = h3d_job_create(ctx, prompt, mode, width, height,
                                   frames, seconds, &quality,
                                   seed, seed_was_set,
                                   refs_images, refs_image_count,
                                   &refs_video, has_refs_video,
                                   refs_audio, refs_audio_count,
                                   finalize_partial, write_preview,
                                   extra_args, idempotency_key);
    if (!job) {
        h3d_error err = { .code = "INTERNAL", .message = "Failed to create job" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 500, "Internal Server Error",
                               "application/json; charset=utf-8", buf, (size_t)len);
        return;
    }

    /* Build CLI args for reproduce */
    build_cli_args(job->cli_args, sizeof(job->cli_args), ctx, prompt, job);

    /* Record submission event */
    char data[256];
    snprintf(data, sizeof(data), "{\"status\":\"queued\",\"queue_position\":1}");
    h3d_job_record_event(job, "status", data);

    /* Response 202 */
    char resp[512];
    int rlen = snprintf(resp, sizeof(resp),
        "{\"job_id\":\"%s\",\"status\":\"queued\","
        "\"queue_position\":1,"
        "\"events_url\":\"/v1/jobs/%s/events\","
        "\"submitted_at\":%ld}",
        job->job_id, job->job_id, (long)job->created_at);
    h3d_http_send_response(fd, 202, "Accepted",
                           "application/json; charset=utf-8",
                           resp, (size_t)rlen);
}

static void handle_status(h3d_ctx *ctx, int fd, const char *job_id) {
    h3d_job *job = h3d_job_find(ctx, job_id);
    if (!job) {
        h3d_error err = { .code = "JOB_NOT_FOUND", .message = "No job with that ID" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 404, "Not Found",
                               "application/json; charset=utf-8", buf, (size_t)len);
        return;
    }
    char body[4096];
    int blen = h3d_json_job_response(job, body, sizeof(body));
    h3d_http_send_response(fd, 200, "OK",
                           "application/json; charset=utf-8", body, (size_t)blen);
}

static void handle_cancel(h3d_ctx *ctx, int fd, const char *job_id) {
    int result = h3d_job_cancel(ctx, job_id);
    if (result < 0) {
        h3d_error err = { .code = "JOB_NOT_FOUND", .message = "No job with that ID" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 404, "Not Found",
                               "application/json; charset=utf-8", buf, (size_t)len);
        return;
    }
    if (result == 0) {
        h3d_job *job = h3d_job_find(ctx, job_id);
        if (job) {
            char body[4096];
            int blen = h3d_json_job_response(job, body, sizeof(body));
            h3d_http_send_response(fd, 409, "Conflict",
                                   "application/json; charset=utf-8", body, (size_t)blen);
        }
        return;
    }
    h3d_job *job = h3d_job_find(ctx, job_id);
    if (job) {
        char body[4096];
        int blen = h3d_json_job_response(job, body, sizeof(body));
        h3d_http_send_response(fd, 200, "OK",
                               "application/json; charset=utf-8", body, (size_t)blen);
    }
}

static void handle_events(h3d_ctx *ctx, int fd, const char *job_id) {
    h3d_job *job = h3d_job_find(ctx, job_id);
    if (!job) {
        h3d_error err = { .code = "JOB_NOT_FOUND", .message = "No job with that ID" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 404, "Not Found",
                               "application/json; charset=utf-8", buf, (size_t)len);
        return;
    }

    h3d_http_send_headers(fd, 200, "OK", NULL);

    /* Replay existing events */
    pthread_mutex_lock(&job->lock);
    int count = job->event_count;
    int tail = job->event_tail;
    int head = job->event_head;
    pthread_mutex_unlock(&job->lock);

    if (count > 0) {
        int idx = tail;
        for (int i = 0; i < count; i++) {
            pthread_mutex_lock(&job->lock);
            const char *evt = job->events[idx];
            pthread_mutex_unlock(&job->lock);
            size_t elen = strlen(evt);
            h3d_http_send_chunk(fd, evt, elen);
            idx = (idx + 1) % H3D_SSE_EVENT_HISTORY;
        }
    }

    /* Stream live events */
    time_t last_heartbeat = time(NULL);
    while (1) {
        pthread_mutex_lock(&job->lock);
        h3d_job_status status = job->status;
        pthread_mutex_unlock(&job->lock);

        if (h3d_job_is_terminal(status)) break;

        time_t now = time(NULL);
        if (now - last_heartbeat >= H3D_SSE_HEARTBEAT_SEC) {
            h3d_http_send_sse_heartbeat(fd);
            last_heartbeat = now;
        }

        pthread_mutex_lock(&job->lock);
        int new_head = job->event_head;
        pthread_mutex_unlock(&job->lock);

        if (new_head != head) {
            while (head != new_head) {
                pthread_mutex_lock(&job->lock);
                const char *evt = job->events[head];
                pthread_mutex_unlock(&job->lock);
                size_t elen = strlen(evt);
                h3d_http_send_chunk(fd, evt, elen);
                head = (head + 1) % H3D_SSE_EVENT_HISTORY;
            }
        }

        usleep(100000); /* 100ms poll */
    }

    h3d_http_close(fd);
}

static void handle_download(h3d_ctx *ctx, int fd, const char *job_id) {
    h3d_job *job = h3d_job_find(ctx, job_id);
    if (!job) {
        h3d_error err = { .code = "JOB_NOT_FOUND", .message = "No job with that ID" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 404, "Not Found",
                               "application/json; charset=utf-8", buf, (size_t)len);
        return;
    }

    if (job->status != H3D_JOB_DONE || !job->mp4_path) {
        h3d_error err = { .code = "NOT_READY", .message = "Job not completed or no output" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 409, "Conflict",
                               "application/json; charset=utf-8", buf, (size_t)len);
        return;
    }

    struct stat st;
    if (stat(job->mp4_path, &st) < 0) {
        h3d_error err = { .code = "FILE_NOT_FOUND", .message = "Output file not found on disk" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 404, "Not Found",
                               "application/json; charset=utf-8", buf, (size_t)len);
        return;
    }

    FILE *fp = fopen(job->mp4_path, "rb");
    if (!fp) {
        h3d_error err = { .code = "FILE_ERROR", .message = "Cannot open output file" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 500, "Internal Server Error",
                               "application/json; charset=utf-8", buf, (size_t)len);
        return;
    }

    const char *basename = strrchr(job->mp4_path, '/');
    basename = basename ? basename + 1 : job->mp4_path;

    char extra[512];
    snprintf(extra, sizeof(extra),
        "Content-Length: %ld\r\n"
        "Content-Type: video/mp4\r\n"
        "Content-Disposition: attachment; filename=\"%s\"\r\n",
        (long)st.st_size, basename);
    h3d_http_send_headers(fd, 200, "OK", extra);

    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
        h3d_http_send_chunk(fd, buf, n);
    }
    h3d_http_send_chunk(fd, NULL, 0);
    fclose(fp);
}

static void handle_jobs_list(h3d_ctx *ctx, int fd, const char *query) {
    char status_filter[32] = {0};
    int limit = 50;

    /* Parse query string: ?status=running&limit=50 */
    if (query) {
        const char *p = strchr(query, '?');
        if (p) {
            p++;
            while (*p) {
                if (strncmp(p, "status=", 7) == 0) {
                    p += 7;
                    size_t i = 0;
                    while (*p && *p != '&' && i < sizeof(status_filter) - 1) {
                        status_filter[i++] = *p++;
                    }
                    status_filter[i] = '\0';
                } else if (strncmp(p, "limit=", 6) == 0) {
                    p += 6;
                    limit = atoi(p);
                }
                while (*p && *p != '&') p++;
                if (*p == '&') p++;
            }
        }
    }

    char buf[65536];
    int len = h3d_json_jobs_list_response(ctx,
        status_filter[0] ? status_filter : NULL, limit, buf, sizeof(buf));
    h3d_http_send_response(fd, 200, "OK",
                           "application/json; charset=utf-8", buf, (size_t)len);
}

/* ── Router ──────────────────────────────────────────────────── */

static void handle_request(h3d_ctx *ctx, int fd,
                           const char *method, const char *path,
                           const char *headers,
                           const char *body, size_t body_len) {
    /* Protocol version check (§12) */
    if (check_protocol(headers) == -2) {
        const char *p = strstr(headers, "X-H3-Protocol:");
        if (!p) p = strstr(headers, "x-h3-protocol:");
        char client_proto[32] = {0};
        if (p) {
            p += 15;
            while (*p == ' ') p++;
            size_t i = 0;
            while (*p && *p != '\r' && *p != '\n' && i < sizeof(client_proto) - 1)
                client_proto[i++] = *p++;
        }
        send_protocol_mismatch(fd, client_proto);
        return;
    }

    /* GET /v1/info */
    if (strcmp(path, "/v1/info") == 0 && strcmp(method, "GET") == 0) {
        handle_info(ctx, fd);
    }
    /* POST /v1/jobs */
    else if (strcmp(path, "/v1/jobs") == 0 && strcmp(method, "POST") == 0) {
        handle_submit(ctx, fd, body, body_len);
    }
    /* GET /v1/jobs or /v1/jobs?status=...&limit=... */
    else if (strncmp(path, "/v1/jobs", 8) == 0 &&
             (path[8] == '\0' || path[8] == '?') &&
             strcmp(method, "GET") == 0) {
        handle_jobs_list(ctx, fd, path);
    }
    /* /v1/jobs/{id}... */
    else if (strncmp(path, "/v1/jobs/", 9) == 0) {
        const char *job_id_raw = path + 9;
        const char *suffix = strchr(job_id_raw, '/');
        char id[H3D_MAX_JOB_ID] = {0};
        if (suffix) {
            size_t id_len = (size_t)(suffix - job_id_raw);
            if (id_len >= sizeof(id)) id_len = sizeof(id) - 1;
            memcpy(id, job_id_raw, id_len);
            id[id_len] = '\0';
            suffix++;
        } else {
            snprintf(id, sizeof(id), "%.31s", job_id_raw);
            /* Strip query string from id */
            char *q = strchr(id, '?');
            if (q) *q = '\0';
        }

        if (suffix && strcmp(suffix, "events") == 0 && strcmp(method, "GET") == 0) {
            handle_events(ctx, fd, id);
            return;
        } else if (suffix && strcmp(suffix, "cancel") == 0 && strcmp(method, "POST") == 0) {
            handle_cancel(ctx, fd, id);
        } else if (suffix && strcmp(suffix, "download") == 0 && strcmp(method, "GET") == 0) {
            handle_download(ctx, fd, id);
        } else if (!suffix && strcmp(method, "GET") == 0) {
            handle_status(ctx, fd, id);
        } else {
            h3d_error err = { .code = "NOT_FOUND", .message = "Unknown endpoint" };
            char buf[512];
            int len = h3d_json_error_response(&err, buf, sizeof(buf));
            h3d_http_send_response(fd, 404, "Not Found",
                                   "application/json; charset=utf-8", buf, (size_t)len);
        }
    }
    else {
        h3d_error err = { .code = "NOT_FOUND", .message = "Unknown endpoint" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 404, "Not Found",
                               "application/json; charset=utf-8", buf, (size_t)len);
    }
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

    /* Mark not ready until workers init */
    atomic_store(&ctx->ready, 0);
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

    /* Mark ready */
    atomic_store(&ctx->ready, 1);
    ctx->unready_reason[0] = '\0';

    fprintf(stderr, "h3d: listening on %s:%d (%d worker%s) [protocol=%s]\n",
            ctx->bind_addr, ctx->port, ctx->worker_count,
            ctx->worker_count > 1 ? "s" : "", H3D_PROTOCOL);

    while (!atomic_load(&ctx->shutdown)) {
        int fd = h3d_http_accept(ctx->server_fd);
        if (fd < 0) {
            if (errno == EINTR) continue;
            break;
        }

        while (!atomic_load(&ctx->shutdown)) {
            char method[16], path[4096], headers[8192], body[65536];
            size_t body_len = 0;

            if (h3d_http_read_request(fd, method, sizeof(method),
                                      path, sizeof(path),
                                      headers, sizeof(headers),
                                      body, sizeof(body),
                                      &body_len) < 0) {
                break;
            }

            handle_request(ctx, fd, method, path, headers, body, body_len);

            if (strstr(path, "/events")) break;
        }
        h3d_http_close(fd);
    }

    h3d_workers_shutdown(ctx);
    h3d_http_close(ctx->server_fd);
    fprintf(stderr, "h3d: shutdown complete\n");
    return 0;
}
