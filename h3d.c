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

/* Parse a simple JSON string value: "key":"value" */
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
        if (*p == '\\') { p++; if (!*p) break; }
        value[i++] = *p++;
    }
    value[i] = '\0';
    return p;
}

/* Parse JSON integer: "key":N */
static int json_find_int(const char *json, const char *key, int *value) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(json, pattern);
    if (!p) return 0;
    p += strlen(pattern);
    while (*p == ' ') p++;
    if (*p == 'n') return 0; /* null */
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

/* Handle GET /v1/info */
static void handle_info(h3d_ctx *ctx, int fd) {
    char body[8192];
    int len = h3d_json_info_response(ctx, body, sizeof(body));
    h3d_http_send_response(fd, 200, "OK", "application/json; charset=utf-8",
                           body, (size_t)len);
}

/* Handle POST /v1/jobs */
static void handle_submit(h3d_ctx *ctx, int fd, const char *body,
                          size_t body_len) {
    (void)body_len;
    /* Parse request */
    char prompt[8192] = {0};
    if (!json_find_string(body, "prompt", prompt, sizeof(prompt)) ||
        !prompt[0]) {
        h3d_error err = { .code = "VALIDATION_ERROR",
                          .message = "prompt is required" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 400, "Bad Request",
                               "application/json; charset=utf-8",
                               buf, (size_t)len);
        return;
    }

    if ((int)strlen(prompt) > ctx->prompt_max_chars) {
        h3d_error err = { .code = "VALIDATION_ERROR",
                          .message = "prompt exceeds maximum length" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 400, "Bad Request",
                               "application/json; charset=utf-8",
                               buf, (size_t)len);
        return;
    }

    /* Check queue limit */
    if (h3d_job_count_by_status(ctx, H3D_JOB_QUEUED) >= ctx->max_queued_per_client) {
        char buf[512];
        int len = h3d_json_queue_full_response(buf, sizeof(buf));
        h3d_http_send_response(fd, 429, "Too Many Requests",
                               "application/json; charset=utf-8",
                               buf, (size_t)len);
        return;
    }

    /* Build params */
    h3_params params = H3_PARAMS_DEFAULT;
    json_find_int(body, "width", &params.width);
    json_find_int(body, "height", &params.height);
    json_find_int(body, "frames", &params.frames);
    json_find_int(body, "steps", &params.steps);
    json_find_u64(body, "seed", &params.seed);
    json_find_int(body, "denoise_reuse", &params.denoise_reuse);
    json_find_int(body, "dit_layers", &params.dit_layers);

    char token_red[16] = {0};
    if (json_find_string(body, "token_reduction", token_red, sizeof(token_red))) {
        params.token_reduction = (strcmp(token_red, "true") == 0) ? 1 : 0;
    }

    /* Create job */
    h3d_job *job = h3d_job_create(ctx, prompt, &params, NULL, 0);
    if (!job) {
        h3d_error err = { .code = "INTERNAL",
                          .message = "Failed to create job" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 500, "Internal Server Error",
                               "application/json; charset=utf-8",
                               buf, (size_t)len);
        return;
    }

    /* Record submission event */
    char data[256];
    snprintf(data, sizeof(data),
        "{\"status\":\"queued\",\"queue_position\":1}");
    h3d_job_record_event(job, "status", data);

    /* Response */
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

/* Handle GET /v1/jobs/{id} */
static void handle_status(h3d_ctx *ctx, int fd, const char *job_id) {
    h3d_job *job = h3d_job_find(ctx, job_id);
    if (!job) {
        h3d_error err = { .code = "JOB_NOT_FOUND",
                          .message = "No job with that ID" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 404, "Not Found",
                               "application/json; charset=utf-8",
                               buf, (size_t)len);
        return;
    }
    char body[4096];
    int blen = h3d_json_job_response(job, body, sizeof(body));
    h3d_http_send_response(fd, 200, "OK",
                           "application/json; charset=utf-8",
                           body, (size_t)blen);
}

/* Handle POST /v1/jobs/{id}/cancel */
static void handle_cancel(h3d_ctx *ctx, int fd, const char *job_id) {
    int result = h3d_job_cancel(ctx, job_id);
    if (result < 0) {
        h3d_error err = { .code = "JOB_NOT_FOUND",
                          .message = "No job with that ID" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 404, "Not Found",
                               "application/json; charset=utf-8",
                               buf, (size_t)len);
        return;
    }
    if (result == 0) {
        /* Already terminal */
        h3d_job *job = h3d_job_find(ctx, job_id);
        if (job) {
            char body[4096];
            int blen = h3d_json_job_response(job, body, sizeof(body));
            h3d_http_send_response(fd, 409, "Conflict",
                                   "application/json; charset=utf-8",
                                   body, (size_t)blen);
        }
        return;
    }
    /* Cancel requested */
    h3d_job *job = h3d_job_find(ctx, job_id);
    if (job) {
        char body[4096];
        int blen = h3d_json_job_response(job, body, sizeof(body));
        h3d_http_send_response(fd, 200, "OK",
                               "application/json; charset=utf-8",
                               body, (size_t)blen);
    }
}

/* Handle GET /v1/jobs/{id}/events (SSE) */
static void handle_events(h3d_ctx *ctx, int fd, const char *job_id) {
    h3d_job *job = h3d_job_find(ctx, job_id);
    if (!job) {
        h3d_error err = { .code = "JOB_NOT_FOUND",
                          .message = "No job with that ID" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 404, "Not Found",
                               "application/json; charset=utf-8",
                               buf, (size_t)len);
        return;
    }

    /* Send SSE headers */
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
            /* Send raw event string as chunk */
            size_t elen = strlen(evt);
            h3d_http_send_chunk(fd, evt, elen);
            idx = (idx + 1) % H3D_SSE_EVENT_HISTORY;
        }
    }

    /* Stream live events until job is terminal */
    time_t last_heartbeat = time(NULL);
    while (1) {
        /* Check if job is done */
        pthread_mutex_lock(&job->lock);
        h3d_job_status status = job->status;
        pthread_mutex_unlock(&job->lock);

        if (h3d_job_is_terminal(status)) {
            /* Send final event if not already replayed */
            break;
        }

        /* Heartbeat */
        time_t now = time(NULL);
        if (now - last_heartbeat >= H3D_SSE_HEARTBEAT_SEC) {
            h3d_http_send_sse_heartbeat(fd);
            last_heartbeat = now;
        }

        /* Check for new events */
        pthread_mutex_lock(&job->lock);
        int new_head = job->event_head;
        pthread_mutex_unlock(&job->lock);

        if (new_head != head) {
            /* New event available - replay from head to new_head */
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

/* Handle GET /v1/jobs/{id}/download */
static void handle_download(h3d_ctx *ctx, int fd, const char *job_id) {
    h3d_job *job = h3d_job_find(ctx, job_id);
    if (!job) {
        h3d_error err = { .code = "JOB_NOT_FOUND",
                          .message = "No job with that ID" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 404, "Not Found",
                               "application/json; charset=utf-8",
                               buf, (size_t)len);
        return;
    }

    if (job->status != H3D_JOB_DONE || !job->mp4_path) {
        h3d_error err = { .code = "NOT_READY",
                          .message = "Job not completed or no output" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 409, "Conflict",
                               "application/json; charset=utf-8",
                               buf, (size_t)len);
        return;
    }

    /* Check file exists and get size */
    struct stat st;
    if (stat(job->mp4_path, &st) < 0) {
        h3d_error err = { .code = "FILE_NOT_FOUND",
                          .message = "Output file not found on disk" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 404, "Not Found",
                               "application/json; charset=utf-8",
                               buf, (size_t)len);
        return;
    }

    /* Send file */
    FILE *fp = fopen(job->mp4_path, "rb");
    if (!fp) {
        h3d_error err = { .code = "FILE_ERROR",
                          .message = "Cannot open output file" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 500, "Internal Server Error",
                               "application/json; charset=utf-8",
                               buf, (size_t)len);
        return;
    }

    /* Extract filename from path */
    const char *basename = strrchr(job->mp4_path, '/');
    basename = basename ? basename + 1 : job->mp4_path;

    /* Send headers */
    char extra[512];
    snprintf(extra, sizeof(extra),
        "Content-Length: %ld\r\n"
        "Content-Type: video/mp4\r\n"
        "Content-Disposition: attachment; filename=\"%s\"\r\n",
        (long)st.st_size, basename);
    h3d_http_send_headers(fd, 200, "OK", extra);

    /* Send file in chunks */
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) {
        h3d_http_send_chunk(fd, buf, n);
    }
    /* Send empty chunk to terminate */
    h3d_http_send_chunk(fd, NULL, 0);

    fclose(fp);
}

/* Route a request */
static void handle_request(h3d_ctx *ctx, int fd,
                           const char *method, const char *path,
                           const char *body, size_t body_len) {
    /* Protocol version check */
    /* TODO: check X-H3-Protocol header */

    if (strcmp(path, "/v1/info") == 0 && strcmp(method, "GET") == 0) {
        handle_info(ctx, fd);
    } else if (strcmp(path, "/v1/jobs") == 0 && strcmp(method, "POST") == 0) {
        handle_submit(ctx, fd, body, body_len);
    } else if (strncmp(path, "/v1/jobs/", 9) == 0) {
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
        }

        if (suffix && strcmp(suffix, "events") == 0 && strcmp(method, "GET") == 0) {
            handle_events(ctx, fd, id);
            return; /* SSE closes connection */
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
                                   "application/json; charset=utf-8",
                                   buf, (size_t)len);
        }
    } else {
        h3d_error err = { .code = "NOT_FOUND", .message = "Unknown endpoint" };
        char buf[512];
        int len = h3d_json_error_response(&err, buf, sizeof(buf));
        h3d_http_send_response(fd, 404, "Not Found",
                               "application/json; charset=utf-8",
                               buf, (size_t)len);
    }
}

int h3d_serve(h3d_ctx *ctx) {
    g_ctx = ctx;

    /* Signal handlers */
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    /* Ignore SIGHUP so daemon survives terminal close */
    signal(SIGHUP, SIG_IGN);
    /* Ignore SIGPIPE to avoid crash on client disconnect */
    signal(SIGPIPE, SIG_IGN);

    /* Init subsystems */
    h3d_jobs_init(ctx);
    if (h3d_workers_init(ctx) < 0) {
        fprintf(stderr, "h3d: no workers available\n");
        return 1;
    }

    /* Listen */
    ctx->server_fd = h3d_http_listen(ctx->bind_addr, ctx->port);
    if (ctx->server_fd < 0) {
        fprintf(stderr, "h3d: cannot bind %s:%d: %s\n",
                ctx->bind_addr, ctx->port, strerror(errno));
        return 1;
    }

    fprintf(stderr, "h3d: listening on %s:%d (%d worker%s)\n",
            ctx->bind_addr, ctx->port, ctx->worker_count,
            ctx->worker_count > 1 ? "s" : "");

    /* Accept loop */
    while (!atomic_load(&ctx->shutdown)) {
        int fd = h3d_http_accept(ctx->server_fd);
        if (fd < 0) {
            fprintf(stderr, "h3d: accept failed: %s (errno=%d)\n", strerror(errno), errno);
            if (errno == EINTR) continue;
            break;
        }
        fprintf(stderr, "h3d: accepted connection fd=%d\n", fd);

        /* Handle request (keep-alive loop) */
        while (!atomic_load(&ctx->shutdown)) {
            char method[16], path[4096], headers[8192], body[65536];
            size_t body_len = 0;

            if (h3d_http_read_request(fd, method, sizeof(method),
                                      path, sizeof(path),
                                      headers, sizeof(headers),
                                      body, sizeof(body),
                                      &body_len) < 0) {
                fprintf(stderr, "h3d: read_request failed, breaking inner loop\n");
                break;
            }

            fprintf(stderr, "h3d: %s %s\n", method, path);
            handle_request(ctx, fd, method, path, body, body_len);

            /* For SSE, connection is already closed */
            if (strstr(path, "/events")) break;
        }
        fprintf(stderr, "h3d: closing fd=%d, shutdown=%d\n", fd, atomic_load(&ctx->shutdown));
        h3d_http_close(fd);
    }
    fprintf(stderr, "h3d: exited accept loop, shutdown=%d\n", atomic_load(&ctx->shutdown));

    /* Shutdown */
    h3d_workers_shutdown(ctx);
    h3d_http_close(ctx->server_fd);
    fprintf(stderr, "h3d: shutdown complete\n");
    return 0;
}
