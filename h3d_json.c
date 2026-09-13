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

static const char *job_status_name(h3d_job_status status) {
    switch (status) {
    case H3D_JOB_QUEUED: return "queued";
    case H3D_JOB_PREPARING: return "preparing";
    case H3D_JOB_RUNNING: return "running";
    case H3D_JOB_FINALIZING: return "finalizing";
    case H3D_JOB_DONE: return "done";
    case H3D_JOB_FAILED: return "failed";
    case H3D_JOB_CANCELLED: return "cancelled";
    case H3D_JOB_TIMEOUT: return "timeout";
    case H3D_JOB_CANCELLING: return "cancelling";
    }
    return "unknown";
}

int h3d_json_info_response(h3d_ctx *ctx, char *buf, size_t len) {
    char escaped_path[1024];
    h3d_json_escape(escaped_path, sizeof(escaped_path), ctx->model_path);
    char escaped_output[1024];
    h3d_json_escape(escaped_output, sizeof(escaped_output), ctx->output_root);
    char escaped_media[1024];
    h3d_json_escape(escaped_media, sizeof(escaped_media), ctx->media_root);

    int off = 0;
    int n;

    n = snprintf(buf + off, len - (size_t)off,
        "{"
        "\"protocol\":\"%s\","
        "\"h3_version\":\"%s\","
        "\"ready\":true,"
        "\"unready_reason\":null,"
        "\"model_path\":\"%s\","
        "\"output_root\":\"%s\","
        "\"media_root\":\"%s\","
        "\"quota\":{\"limit_bytes\":%" PRIu64 ",\"used_bytes\":%" PRIu64 "},"
        "\"workers\":[",
        H3D_PROTOCOL, H3D_VERSION,
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

int h3d_json_job_response(h3d_job *job, char *buf, size_t len) {
    char escaped_id[H3D_MAX_JOB_ID * 2];
    h3d_json_escape(escaped_id, sizeof(escaped_id), job->job_id);
    char escaped_phase[sizeof(job->phase) * 2];
    h3d_json_escape(escaped_phase, sizeof(escaped_phase), job->phase);

    int off = 0;
    int n;

    n = snprintf(buf + off, len - (size_t)off,
        "{"
        "\"job_id\":\"%s\","
        "\"status\":\"%s\","
        "\"phase\":\"%s\",",
        escaped_id, job_status_name(job->status), escaped_phase);
    if (n > 0 && (size_t)n < len - (size_t)off) off += n; else return off;

    /* Progress */
    n = snprintf(buf + off, len - (size_t)off,
        "\"progress\":{"
        "\"step\":%d,\"total_steps\":%d,"
        "\"elapsed_sec\":%ld,\"eta_hint_sec\":%d"
        "},",
        job->progress_step, job->progress_total,
        (long)job->elapsed_sec, job->eta_hint_sec);
    if (n > 0 && (size_t)n < len - (size_t)off) off += n; else return off;

    /* Request echo */
    n = snprintf(buf + off, len - (size_t)off,
        "\"request\":{"
        "\"width\":%d,\"height\":%d,"
        "\"frames\":%d,\"steps\":%d,"
        "\"seed\":%" PRIu64 ","
        "\"denoise_reuse\":%d,\"dit_layers\":%d,"
        "\"token_reduction\":%s"
        "},",
        job->params.width, job->params.height,
        job->params.frames, job->params.steps,
        job->params.seed,
        job->params.denoise_reuse, job->params.dit_layers,
        job->params.token_reduction ? "true" : "false");
    if (n > 0 && (size_t)n < len - (size_t)off) off += n; else return off;

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

        n = snprintf(buf + off, len - (size_t)off,
            "\"result\":{"
            "\"mp4_path\":\"%s\","
            "\"poster_path\":\"%s\","
            "\"width\":%d,\"height\":%d,\"frames\":%d,"
            "\"seed\":%" PRIu64 ","
            "\"partial\":%s"
            "}",
            escaped_mp4,
            job->poster_path ? escaped_poster : "",
            job->params.width, job->params.height, job->params.frames,
            job->seed,
            job->partial ? "true" : "false");
        off += n;
    } else {
        n = snprintf(buf + off, len - (size_t)off, "\"result\":null");
        off += n;
    }

    /* Error */
    if (job->error.code[0]) {
        char escaped_msg[H3D_MAX_ERROR_MESSAGE * 2];
        h3d_json_escape(escaped_msg, sizeof(escaped_msg), job->error.message);
        n = snprintf(buf + off, len - (size_t)off,
            ",\"error\":{\"code\":\"%s\",\"message\":\"%s\"}",
            job->error.code, escaped_msg);
        off += n;
    }

    if (off < (int)len) buf[off++] = '}';

    return off;
}

int h3d_json_error_response(const h3d_error *error, char *buf, size_t len) {
    char escaped_msg[H3D_MAX_ERROR_MESSAGE * 2];
    h3d_json_escape(escaped_msg, sizeof(escaped_msg), error->message);
    return snprintf(buf, len,
        "{\"error\":{\"code\":\"%s\",\"message\":\"%s\"}}",
        error->code, escaped_msg);
}

int h3d_json_queue_full_response(char *buf, size_t len) {
    return snprintf(buf, len,
        "{\"error\":{\"code\":\"QUEUE_FULL\","
        "\"message\":\"Too many queued jobs for this client\"}}");
}
