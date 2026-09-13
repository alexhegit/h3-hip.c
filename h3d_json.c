#include "h3d_json.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

int h3d_json_escape(char *dest, size_t dest_len, const char *src) {
    if (!src || !dest || dest_len == 0) return 0;
    size_t di = 0;
    for (size_t si = 0; src[si] && di < dest_len - 1; si++) {
        char c = src[si];
        if (c == '"' || c == '\\') {
            if (di + 2 >= dest_len) break;
            dest[di++] = '\\';
            dest[di++] = c;
        } else if (c == '\n') {
            if (di + 2 >= dest_len) break;
            dest[di++] = '\\';
            dest[di++] = 'n';
        } else if (c == '\r') {
            if (di + 2 >= dest_len) break;
            dest[di++] = '\\';
            dest[di++] = 'r';
        } else if (c == '\t') {
            if (di + 2 >= dest_len) break;
            dest[di++] = '\\';
            dest[di++] = 't';
        } else {
            dest[di++] = c;
        }
    }
    dest[di] = '\0';
    return (int)di;
}

/* ── Info response ───────────────────────────────────────────── */

int h3d_json_info_response(h3d_ctx *ctx, char *buf, size_t len) {
    char escaped_path[1024];
    h3d_json_escape(escaped_path, sizeof(escaped_path), ctx->model_path);
    char escaped_output[1024];
    h3d_json_escape(escaped_output, sizeof(escaped_output), ctx->output_root);
    char escaped_media[1024];
    h3d_json_escape(escaped_media, sizeof(escaped_media), ctx->media_root);

    int off = 0;
    int n;

    /* unready_reason: build separately */
    char unready_json[512] = "null";
    if (ctx->unready_reason[0]) {
        char escaped_reason[256];
        h3d_json_escape(escaped_reason, sizeof(escaped_reason), ctx->unready_reason);
        snprintf(unready_json, sizeof(unready_json), "\"%s\"", escaped_reason);
    }

    n = snprintf(buf + off, len - (size_t)off,
        "{"
        "\"protocol\":\"%s\","
        "\"h3_version\":\"%s\","
        "\"ready\":%s,"
        "\"unready_reason\":%s,"
        "\"model_path\":\"%s\","
        "\"output_root\":\"%s\","
        "\"media_root\":\"%s\","
        "\"quota\":{\"limit_bytes\":%" PRIu64 ",\"used_bytes\":%" PRIu64 "},"
        "\"workers\":[",
        H3D_PROTOCOL, H3D_VERSION,
        atomic_load(&ctx->ready) ? "true" : "false",
        unready_json,
        escaped_path, escaped_output, escaped_media,
        ctx->quota_bytes, ctx->quota_used_bytes);
    off += n;

    /* Workers */
    h3d_worker *w = ctx->workers;
    int first = 1;
    while (w) {
        if (!first) { if (off < (int)len) buf[off++] = ','; }
        first = 0;
        const char *wstate = atomic_load(&w->running) ? "busy" : "idle";
        n = snprintf(buf + off, len - (size_t)off,
            "{\"gpu_index\":%d,\"arch\":\"%s\",\"state\":\"%s\","
            "\"current_job\":null,\"queue_depth\":0}",
            w->gpu_index, w->arch, wstate);
        off += n;
        w = w->next;
    }

    n = snprintf(buf + off, len - (size_t)off,
        "],"
        "\"limits\":{"
        "\"max_queued_jobs_per_client\":%d,"
        "\"prompt_max_chars\":%d,"
        "\"ref_image_max_bytes\":52428800,"
        "\"ref_video_max_bytes\":524288000,"
        "\"ref_audio_max_bytes\":104857600"
        "}}",
        ctx->max_queued_per_client, ctx->prompt_max_chars);
    off += n;

    return off;
}

/* ── Job response ────────────────────────────────────────────── */

int h3d_json_job_response(h3d_job *job, char *buf, size_t len) {
    char escaped_id[H3D_MAX_JOB_ID * 2];
    h3d_json_escape(escaped_id, sizeof(escaped_id), job->job_id);
    char escaped_phase[sizeof(job->phase) * 2];
    h3d_json_escape(escaped_phase, sizeof(escaped_phase), job->phase);

    int off = 0;
    int n;

    /* Status + phase */
    n = snprintf(buf + off, len - (size_t)off,
        "{"
        "\"job_id\":\"%s\","
        "\"status\":\"%s\","
        "\"phase\":\"%s\",",
        escaped_id, h3d_status_name(job->status), escaped_phase);
    off += n;

    /* Progress */
    n = snprintf(buf + off, len - (size_t)off,
        "\"progress\":{"
        "\"step\":%d,\"total_steps\":%d,"
        "\"elapsed_sec\":%ld,\"eta_hint_sec\":%d"
        "},",
        job->progress_step, job->progress_total,
        (long)job->elapsed_sec, job->eta_hint_sec);
    off += n;

    /* Request echo */
    n = snprintf(buf + off, len - (size_t)off,
        "\"request\":{"
        "\"mode\":\"%s\","
        "\"size\":{\"width\":%d,\"height\":%d},"
        "\"duration\":{\"frames\":%d},"
        "\"quality\":{\"preset\":\"%s\",\"steps\":%d,\"layers\":%d,\"reuse\":%d},"
        "\"seed\":%" PRIu64 ""
        "},",
        h3d_mode_name(job->mode),
        job->width, job->height,
        job->frames,
        job->quality.preset,
        job->quality.steps, job->quality.layers, job->quality.reuse,
        job->seed);
    off += n;

    /* Timestamps */
    if (job->finished_at) {
        n = snprintf(buf + off, len - (size_t)off,
            "\"created_at\":%ld,"
            "\"started_at\":%ld,"
            "\"finished_at\":%ld,",
            (long)job->created_at,
            (long)job->started_at,
            (long)job->finished_at);
    } else {
        n = snprintf(buf + off, len - (size_t)off,
            "\"created_at\":%ld,"
            "\"started_at\":%ld,"
            "\"finished_at\":null,",
            (long)job->created_at,
            (long)job->started_at);
    }
    off += n;

    /* Result or null */
    if (job->status == H3D_JOB_DONE && job->mp4_path) {
        char escaped_mp4[2048];
        h3d_json_escape(escaped_mp4, sizeof(escaped_mp4), job->mp4_path);
        char escaped_poster[2048] = {0};
        if (job->poster_path)
            h3d_json_escape(escaped_poster, sizeof(escaped_poster), job->poster_path);
        char escaped_cli[4096];
        h3d_json_escape(escaped_cli, sizeof(escaped_cli), job->cli_args);

        n = snprintf(buf + off, len - (size_t)off,
            "\"result\":{"
            "\"mp4_path\":\"%s\","
            "\"poster_path\":\"%s\","
            "\"duration_sec\":%.2f,"
            "\"width\":%d,\"height\":%d,\"frames\":%d,"
            "\"seed\":%" PRIu64 ","
            "\"partial\":%s,"
            "\"cli\":\"%s\"",
            escaped_mp4,
            job->poster_path ? escaped_poster : "",
            job->duration_sec,
            job->width, job->height, job->frames,
            job->seed,
            job->partial ? "true" : "false",
            escaped_cli);
        off += n;

        /* ffprobe */
        if (job->has_ffprobe && job->ffprobe_json[0]) {
            n = snprintf(buf + off, len - (size_t)off,
                ",\"ffprobe\":%s", job->ffprobe_json);
            off += n;
        }

        if (off < (int)len) buf[off++] = '}';
    } else {
        n = snprintf(buf + off, len - (size_t)off, "\"result\":null");
        off += n;
    }

    /* Error */
    if (job->error.code[0]) {
        char escaped_msg[H3D_MAX_ERROR_MESSAGE * 2];
        h3d_json_escape(escaped_msg, sizeof(escaped_msg), job->error.message);
        n = snprintf(buf + off, len - (size_t)off,
            ",\"error\":{\"code\":\"%s\",\"message\":\"%s\"",
            job->error.code, escaped_msg);
        off += n;

        /* Details */
        if (job->error.detail_count > 0) {
            n = snprintf(buf + off, len - (size_t)off, ",\"details\":[");
            off += n;
            for (int i = 0; i < job->error.detail_count; i++) {
                if (i > 0) {
                    if (off < (int)len) buf[off++] = ',';
                }
                char escaped_field[128];
                h3d_json_escape(escaped_field, sizeof(escaped_field), job->error.details[i].field);
                char escaped_reason[256];
                h3d_json_escape(escaped_reason, sizeof(escaped_reason), job->error.details[i].reason);
                n = snprintf(buf + off, len - (size_t)off,
                    "{\"field\":\"%s\",\"reason\":\"%s\"}",
                    escaped_field, escaped_reason);
                off += n;
            }
            if (off < (int)len) buf[off++] = ']';
        }

        if (off < (int)len) buf[off++] = '}';
    }

    if (off < (int)len) buf[off++] = '}';

    return off;
}

/* ── Error response ──────────────────────────────────────────── */

int h3d_json_error_response(const h3d_error *error, char *buf, size_t len) {
    char escaped_msg[H3D_MAX_ERROR_MESSAGE * 2];
    h3d_json_escape(escaped_msg, sizeof(escaped_msg), error->message);

    int off = 0;
    int n;

    n = snprintf(buf + off, len - (size_t)off,
        "{\"error\":{\"code\":\"%s\",\"message\":\"%s\"",
        error->code, escaped_msg);
    off += n;

    /* Details */
    if (error->detail_count > 0) {
        n = snprintf(buf + off, len - (size_t)off, ",\"details\":[");
        off += n;
        for (int i = 0; i < error->detail_count; i++) {
            if (i > 0) {
                if (off < (int)len) buf[off++] = ',';
            }
            char escaped_field[128];
            h3d_json_escape(escaped_field, sizeof(escaped_field), error->details[i].field);
            char escaped_reason[256];
            h3d_json_escape(escaped_reason, sizeof(escaped_reason), error->details[i].reason);
            n = snprintf(buf + off, len - (size_t)off,
                "{\"field\":\"%s\",\"reason\":\"%s\"}",
                escaped_field, escaped_reason);
            off += n;
        }
        if (off < (int)len) buf[off++] = ']';
    }

    if (off < (int)len) buf[off++] = '}';
    if (off < (int)len) buf[off++] = '}';

    return off;
}

/* ── Queue full response ─────────────────────────────────────── */

int h3d_json_queue_full_response(char *buf, size_t len) {
    return snprintf(buf, len,
        "{\"error\":{\"code\":\"QUEUE_FULL\","
        "\"message\":\"Too many queued jobs for this client\"}}");
}

/* ── Jobs list response ──────────────────────────────────────── */

int h3d_json_jobs_list_response(h3d_ctx *ctx, const char *status_filter,
                                int limit, char *buf, size_t len) {
    int off = 0;
    int n;

    n = snprintf(buf + off, len - (size_t)off, "{\"jobs\":[");
    off += n;

    pthread_mutex_lock(&ctx->jobs_lock);
    int count = 0;
    int first = 1;
    for (size_t i = 0; i < ctx->job_count && count < limit; i++) {
        h3d_job *job = ctx->jobs[i];

        /* Filter by status if provided */
        if (status_filter) {
            const char *status_name = h3d_status_name(job->status);
            if (strcmp(status_name, status_filter) != 0) continue;
        }

        if (!first) {
            if (off < (int)len) buf[off++] = ',';
        }
        first = 0;

        /* Compact job summary */
        char escaped_id[H3D_MAX_JOB_ID * 2];
        h3d_json_escape(escaped_id, sizeof(escaped_id), job->job_id);

        n = snprintf(buf + off, len - (size_t)off,
            "{\"job_id\":\"%s\",\"status\":\"%s\",\"phase\":\"%s\","
            "\"created_at\":%ld}",
            escaped_id, h3d_status_name(job->status), job->phase,
            (long)job->created_at);
        off += n;
        count++;
    }
    pthread_mutex_unlock(&ctx->jobs_lock);

    n = snprintf(buf + off, len - (size_t)off, "],\"count\":%d}", count);
    off += n;

    return off;
}
