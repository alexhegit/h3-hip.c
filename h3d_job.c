#include "h3d_job.h"
#include "h3d_json.h"

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

h3d_job *h3d_job_create(h3d_ctx *ctx, const char *prompt,
                        const h3d_mode mode,
                        const int width, const int height,
                        const int frames, const float seconds,
                        const h3d_quality *quality,
                        uint64_t seed, int seed_was_set,
                        const h3d_ref_entry *refs_images, int ref_image_count,
                        const h3d_ref_entry *refs_video, int has_video,
                        const h3d_ref_entry *refs_audio, int ref_audio_count,
                        int finalize_partial, int write_preview,
                        const char *extra_args,
                        const char *idempotency_key) {
    h3d_job *job = calloc(1, sizeof(h3d_job));
    if (!job) return NULL;

    pthread_mutex_init(&job->lock, NULL);
    atomic_store(&job->cancel_requested, 0);
    generate_job_id(job->job_id, sizeof(job->job_id));
    job->status = H3D_JOB_QUEUED;
    job->created_at = time(NULL);

    /* Request fields */
    if (prompt) snprintf(job->prompt, sizeof(job->prompt), "%s", prompt);
    job->mode = mode;
    job->width = width;
    job->height = height;
    job->frames = frames;
    job->seconds = seconds;
    if (quality) job->quality = *quality;
    job->seed = seed;
    job->seed_was_set = seed_was_set;
    job->finalize_partial_on_cancel = finalize_partial;
    job->write_preview_frames = write_preview;
    if (extra_args) snprintf(job->extra_args, sizeof(job->extra_args), "%s", extra_args);
    if (idempotency_key) snprintf(job->idempotency_key, sizeof(job->idempotency_key), "%s", idempotency_key);

    /* Copy references */
    job->refs_image_count = ref_image_count;
    for (int i = 0; i < ref_image_count && i < H3D_MAX_REFERENCES; i++) {
        snprintf(job->refs_images[i].path, sizeof(job->refs_images[i].path),
                 "%s", refs_images[i].path);
    }
    if (has_video && refs_video) {
        snprintf(job->refs_video.path, sizeof(job->refs_video.path),
                 "%s", refs_video->path);
        job->has_refs_video = 1;
    }
    job->refs_audio_count = ref_audio_count;
    for (int i = 0; i < ref_audio_count && i < 3; i++) {
        snprintf(job->refs_audio[i].path, sizeof(job->refs_audio[i].path),
                 "%s", refs_audio[i].path);
    }

    /* Generate output paths */
    char output_dir[1024];
    snprintf(output_dir, sizeof(output_dir), "%s/%s", ctx->output_root, job->job_id);
    mkdir(output_dir, 0755);
    job->output_path = strdup(output_dir);

    char mp4_path[1024];
    snprintf(mp4_path, sizeof(mp4_path), "%s/%s.mp4", output_dir, job->job_id);
    job->mp4_path = strdup(mp4_path);

    char poster_path[1024];
    snprintf(poster_path, sizeof(poster_path), "%s/%s.poster.png", output_dir, job->job_id);
    job->poster_path = strdup(poster_path);

    /* Add to job list */
    pthread_mutex_lock(&ctx->jobs_lock);
    if (ctx->job_count >= ctx->job_capacity) {
        size_t new_cap = ctx->job_capacity * 2;
        h3d_job **new_jobs = realloc(ctx->jobs, new_cap * sizeof(h3d_job *));
        if (new_jobs) {
            ctx->jobs = new_jobs;
            ctx->job_capacity = new_cap;
        }
    }
    if (ctx->job_count < ctx->job_capacity) {
        ctx->jobs[ctx->job_count++] = job;
    }
    pthread_mutex_unlock(&ctx->jobs_lock);

    return job;
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
    job->status = H3D_JOB_CANCELLING;
    pthread_mutex_unlock(&job->lock);
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
             "event: %s\ndata: %s\n", event, data);
    job->event_head = (idx + 1) % H3D_SSE_EVENT_HISTORY;
    if (job->event_count < H3D_SSE_EVENT_HISTORY)
        job->event_count++;
    else
        job->event_tail = (job->event_tail + 1) % H3D_SSE_EVENT_HISTORY;
    pthread_mutex_unlock(&job->lock);
}

/* ── Geometry validation ─────────────────────────────────────── */

int h3d_geometry_is_valid(h3d_ctx *ctx, int width, int height, int frames) {
    for (int i = 0; i < ctx->geometry_whitelist_count; i++) {
        h3d_geometry *g = &ctx->geometry_whitelist[i];
        if (g->width == width && g->height == height) {
            if (g->frames == 0) return 1;  /* any frames */
            if (g->frames == frames) return 1;
        }
    }
    return 0;
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
