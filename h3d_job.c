#include "h3d_job.h"
#include "h3d_json.h"
#include "h3_host.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ── Job ID generation ───────────────────────────────────────── */

static void generate_job_id(char *buf, size_t len) {
    static const char alphabet[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    unsigned int seed = (unsigned int)(time(NULL) ^ (uintptr_t)buf ^ (uintptr_t)getpid());
    for (size_t i = 0; i < len - 1; i++) {
        seed = seed * 1103515245 + 12345;
        buf[i] = alphabet[(seed >> 16) % 36];
    }
    buf[len - 1] = '\0';
}

/* ── Status helpers ──────────────────────────────────────────── */

int h3d_job_is_terminal(h3d_job_status s) {
    return s == H3D_JOB_DONE || s == H3D_JOB_FAILED ||
           s == H3D_JOB_CANCELLED || s == H3D_JOB_TIMEOUT;
}

const char *h3d_status_name(h3d_job_status status) {
    switch (status) {
    case H3D_JOB_QUEUED:     return "queued";
    case H3D_JOB_PREPARING:  return "preparing";
    case H3D_JOB_RUNNING:    return "running";
    case H3D_JOB_FINALIZING: return "finalizing";
    case H3D_JOB_DONE:       return "done";
    case H3D_JOB_FAILED:     return "failed";
    case H3D_JOB_CANCELLED:  return "cancelled";
    case H3D_JOB_TIMEOUT:    return "timeout";
    case H3D_JOB_CANCELLING: return "cancelling";
    }
    return "unknown";
}

const char *h3d_mode_name(h3d_mode mode) {
    switch (mode) {
    case H3D_MODE_T2VA:   return "T2VA";
    case H3D_MODE_I2VA:   return "I2VA";
    case H3D_MODE_FL2VA:  return "FL2VA";
    case H3D_MODE_REF2VA: return "Ref2VA";
    }
    return "T2VA";
}

/* ── Job init ────────────────────────────────────────────────── */

void h3d_jobs_init(h3d_ctx *ctx) {
    ctx->job_capacity = 64;
    ctx->job_count = 0;
    ctx->jobs = calloc(ctx->job_capacity, sizeof(h3d_job *));
}

/* ── Job create ──────────────────────────────────────────────── */

h3d_job *h3d_job_new(h3d_ctx *ctx) {
    h3d_job *job = calloc(1, sizeof(h3d_job));
    if (!job) return NULL;

    pthread_mutex_init(&job->lock, NULL);
    atomic_store(&job->cancel_requested, 0);
    generate_job_id(job->job_id, sizeof(job->job_id));
    job->status = H3D_JOB_QUEUED;
    job->created_at = time(NULL);
    job->core_reuse = 1;

    char output_dir[2048], mp4_path[2048], poster_path[2048];
    if (snprintf(output_dir, sizeof(output_dir), "%s/%s", ctx->output_root,
                 job->job_id) >= (int)sizeof(output_dir) ||
        snprintf(mp4_path, sizeof(mp4_path), "%s/%s.mp4", output_dir,
                 job->job_id) >= (int)sizeof(mp4_path) ||
        snprintf(poster_path, sizeof(poster_path), "%s/%s.poster.png", output_dir,
                 job->job_id) >= (int)sizeof(poster_path)) {
        h3d_job_free(job);
        return NULL;
    }
    job->output_path = strdup(output_dir);
    job->mp4_path = strdup(mp4_path);
    job->poster_path = strdup(poster_path);
    if (!job->output_path || !job->mp4_path || !job->poster_path) {
        h3d_job_free(job);
        return NULL;
    }
    return job;
}

void h3d_job_free(h3d_job *job) {
    if (!job) return;
    free(job->output_path);
    free(job->mp4_path);
    free(job->poster_path);
    pthread_mutex_destroy(&job->lock);
    free(job);
}

int h3d_job_enqueue(h3d_ctx *ctx, h3d_job *job) {
    if (mkdir(job->output_path, 0755) < 0 && errno != EEXIST) return -1;
    pthread_mutex_lock(&ctx->jobs_lock);
    if (ctx->job_count >= ctx->job_capacity) {
        size_t new_cap = ctx->job_capacity * 2;
        h3d_job **new_jobs = realloc(ctx->jobs, new_cap * sizeof(h3d_job *));
        if (!new_jobs) {
            pthread_mutex_unlock(&ctx->jobs_lock);
            return -1;
        }
        ctx->jobs = new_jobs;
        ctx->job_capacity = new_cap;
    }
    ctx->jobs[ctx->job_count++] = job;
    pthread_mutex_unlock(&ctx->jobs_lock);
    return 0;
}

/* ── Job find ────────────────────────────────────────────────── */

h3d_job *h3d_job_find(h3d_ctx *ctx, const char *job_id) {
    pthread_mutex_lock(&ctx->jobs_lock);
    for (size_t i = 0; i < ctx->job_count; i++) {
        if (strcmp(ctx->jobs[i]->job_id, job_id) == 0) {
            h3d_job *job = ctx->jobs[i];
            pthread_mutex_unlock(&ctx->jobs_lock);
            return job;
        }
    }
    pthread_mutex_unlock(&ctx->jobs_lock);
    return NULL;
}

h3d_job *h3d_job_find_by_idempotency(h3d_ctx *ctx, const char *key) {
    if (!key || !*key) return NULL;
    pthread_mutex_lock(&ctx->jobs_lock);
    for (size_t i = 0; i < ctx->job_count; i++) {
        if (strcmp(ctx->jobs[i]->idempotency_key, key) == 0) {
            h3d_job *job = ctx->jobs[i];
            pthread_mutex_unlock(&ctx->jobs_lock);
            return job;
        }
    }
    pthread_mutex_unlock(&ctx->jobs_lock);
    return NULL;
}

/* ── Job cancel ──────────────────────────────────────────────── */

int h3d_job_cancel(h3d_ctx *ctx, const char *job_id) {
    h3d_job *job = h3d_job_find(ctx, job_id);
    if (!job) return -1;

    pthread_mutex_lock(&job->lock);
    if (h3d_job_is_terminal(job->status)) {
        pthread_mutex_unlock(&job->lock);
        return 0; /* Already terminal */
    }
    atomic_store(&job->cancel_requested, 1);
    /* No worker owns a queued job, so nothing else would finish it. Workers
     * only claim QUEUED jobs, and they do so under this same lock. */
    int was_queued = job->status == H3D_JOB_QUEUED;
    if (was_queued) {
        job->status = H3D_JOB_CANCELLED;
        job->finished_at = time(NULL);
        snprintf(job->error.code, sizeof(job->error.code), "CANCELLED");
        snprintf(job->error.message, sizeof(job->error.message),
                 "Job was cancelled before it started");
    } else {
        job->status = H3D_JOB_CANCELLING;
    }
    pthread_mutex_unlock(&job->lock);
    if (was_queued) h3d_job_record_event(job, "cancelled", "{\"status\":\"cancelled\"}");
    return 1;
}

/* ── Job count ───────────────────────────────────────────────── */

int h3d_job_count_by_status(h3d_ctx *ctx, h3d_job_status status) {
    int count = 0;
    pthread_mutex_lock(&ctx->jobs_lock);
    for (size_t i = 0; i < ctx->job_count; i++) {
        if (ctx->jobs[i]->status == status) count++;
    }
    pthread_mutex_unlock(&ctx->jobs_lock);
    return count;
}

/* ── Mark all failed (crash recovery) ────────────────────────── */

void h3d_jobs_mark_all_failed(h3d_ctx *ctx) {
    h3d_error err = { .code = "JOB_LOST_AFTER_RESTART",
                      .message = "Daemon restarted; this job was lost" };
    pthread_mutex_lock(&ctx->jobs_lock);
    for (size_t i = 0; i < ctx->job_count; i++) {
        h3d_job *job = ctx->jobs[i];
        if (!h3d_job_is_terminal(job->status)) {
            pthread_mutex_lock(&job->lock);
            job->status = H3D_JOB_FAILED;
            job->error = err;
            job->finished_at = time(NULL);
            pthread_mutex_unlock(&job->lock);
        }
    }
    pthread_mutex_unlock(&ctx->jobs_lock);
}

/* ── SSE event recording ─────────────────────────────────────── */

void h3d_job_record_event(h3d_job *job, const char *event, const char *data) {
    pthread_mutex_lock(&job->lock);
    int idx = job->event_head;
    snprintf(job->events[idx], sizeof(job->events[idx]),
             "event: %s\ndata: %s\n\n", event, data);
    job->event_head = (idx + 1) % H3D_SSE_EVENT_HISTORY;
    if (job->event_count < H3D_SSE_EVENT_HISTORY)
        job->event_count++;
    else
        job->event_tail = (job->event_tail + 1) % H3D_SSE_EVENT_HISTORY;
    pthread_mutex_unlock(&job->lock);
}

/* ── Geometry validation ─────────────────────────────────────── */

int h3d_geometry_is_valid(int width, int height, int frames) {
    if (width < 32 || height < 32 || width % H3_CANVAS_MULTIPLE ||
        height % H3_CANVAS_MULTIPLE)
        return 0;
    if ((int64_t)width * height > H3_MAX_PIXELS) return 0;
    return frames >= 5 && h3_align_frame_count(frames) <= 362;
}

/* ── Poster generation (ffmpeg extract first frame) ───────────── */

int h3d_generate_poster(const char *mp4_path, const char *poster_path) {
    if (!mp4_path || !poster_path) return -1;

    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "ffmpeg -y -i \"%s\" -vframes 1 -q:v 2 \"%s\" 2>/dev/null",
        mp4_path, poster_path);

    int ret = system(cmd);
    return ret == 0 ? 0 : -1;
}

/* ── ffprobe extraction (compact summary) ─────────────────────── */

int h3d_extract_ffprobe(const char *mp4_path, char *buf, size_t len) {
    if (!mp4_path || !buf || len == 0) return -1;

    /* Run ffprobe with -show_entries for compact output */
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "ffprobe -v quiet -print_format json "
        "-show_entries stream=codec_name,codec_type,width,height,nb_frames "
        "-show_entries format=duration,bit_rate "
        "\"%s\" 2>/dev/null",
        mp4_path);

    FILE *fp = popen(cmd, "r");
    if (!fp) return -1;

    char raw[4096];
    size_t total = 0;
    while (total < sizeof(raw) - 1) {
        size_t n = fread(raw + total, 1, sizeof(raw) - total - 1, fp);
        if (n == 0) break;
        total += n;
    }
    raw[total] = '\0';
    pclose(fp);

    if (total == 0) return -1;

    /* Extract fields from compact JSON */
    char codec[64] = "unknown";
    int width = 0, height = 0;
    float duration_sec = 0.0f;
    int64_t bit_rate = 0;
    int nb_frames = 0;
    int has_audio = 0;

    /* Video stream: find "codec_type": "video" (with possible spaces) */
    const char *video_block = strstr(raw, "\"codec_type\": \"video\"");
    const char *audio_block = strstr(raw, "\"codec_type\": \"audio\"");

    /* codec_name from video stream (find last codec_name before codec_type:video) */
    if (video_block) {
        const char *last_cn = NULL;
        const char *scan = raw;
        while (scan < video_block) {
            const char *cn = strstr(scan, "\"codec_name\":");
            if (!cn || cn >= video_block) break;
            last_cn = cn;
            scan = cn + 13;
        }
        if (last_cn) {
            last_cn += 13;
            while (*last_cn == ' ' || *last_cn == ':') last_cn++;
            if (*last_cn == '"') last_cn++;
            size_t i = 0;
            while (*last_cn && *last_cn != '"' && *last_cn != ',' && i < sizeof(codec) - 1) {
                codec[i++] = *last_cn++;
            }
            codec[i] = '\0';
        }
    }

    /* width, height, nb_frames from video stream */
    if (video_block) {
        const char *p;
        p = strstr(video_block, "\"width\":");
        if (p) { p += 8; while (*p == ' ' || *p == ':' || *p == '"') p++; width = atoi(p); }
        p = strstr(video_block, "\"height\":");
        if (p) { p += 9; while (*p == ' ' || *p == ':' || *p == '"') p++; height = atoi(p); }
        p = strstr(video_block, "\"nb_frames\":");
        if (p) { p += 12; while (*p == ' ' || *p == ':' || *p == '"') p++; nb_frames = atoi(p); }
    }

    /* has_audio */
    if (audio_block) has_audio = 1;

    /* duration, bit_rate from format block */
    const char *format_block = strstr(raw, "\"format\":");
    if (format_block) {
        const char *p;
        p = strstr(format_block, "\"duration\":");
        if (p) { p += 11; while (*p == ' ' || *p == ':' || *p == '"') p++; duration_sec = strtof(p, NULL); }
        p = strstr(format_block, "\"bit_rate\":");
        if (p) { p += 11; while (*p == ' ' || *p == ':' || *p == '"') p++; bit_rate = strtoll(p, NULL, 10); }
    }

    /* Build compact JSON */
    int n = snprintf(buf, len,
        "{\"codec\":\"%s\",\"width\":%d,\"height\":%d,"
        "\"duration_sec\":%.3f,\"bit_rate\":%" PRId64 ","
        "\"nb_frames\":%d,\"has_audio\":%s}",
        codec, width, height, duration_sec, bit_rate,
        nb_frames, has_audio ? "true" : "false");

    return n > 0 ? 0 : -1;
}
