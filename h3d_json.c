#include "h3d_json.h"
#include "h3_host.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

/* Bounded appender: once anything does not fit, the result is -1 rather
 * than a truncated document. */
typedef struct {
    char *buf;
    size_t len;
    size_t off;
    int overflow;
} jbuf;

static void jb_printf(jbuf *b, const char *format, ...) {
    if (b->overflow) return;
    va_list args;
    va_start(args, format);
    int n = vsnprintf(b->buf + b->off, b->len - b->off, format, args);
    va_end(args);
    if (n < 0 || (size_t)n >= b->len - b->off) {
        b->overflow = 1;
        return;
    }
    b->off += (size_t)n;
}

static void jb_char(jbuf *b, char c) {
    if (b->overflow) return;
    if (b->off + 1 >= b->len) { b->overflow = 1; return; }
    b->buf[b->off++] = c;
    b->buf[b->off] = '\0';
}

/* Quoted, escaped string; NULL becomes null. */
static void jb_str(jbuf *b, const char *s) {
    if (!s) { jb_printf(b, "null"); return; }
    jb_char(b, '"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '"': jb_printf(b, "\\\""); break;
        case '\\': jb_printf(b, "\\\\"); break;
        case '\n': jb_printf(b, "\\n"); break;
        case '\r': jb_printf(b, "\\r"); break;
        case '\t': jb_printf(b, "\\t"); break;
        default:
            if (*p < 0x20) jb_printf(b, "\\u%04x", *p);
            else jb_char(b, (char)*p);
        }
    }
    jb_char(b, '"');
}

static void jb_opt_str(jbuf *b, const char *s) {
    jb_str(b, s && s[0] ? s : NULL);
}

static int jb_end(jbuf *b) {
    if (b->overflow) {
        if (b->len) b->buf[0] = '\0';
        return -1;
    }
    return (int)b->off;
}

static jbuf jb_init(char *buf, size_t len) {
    if (len) buf[0] = '\0';
    return (jbuf){buf, len, 0, len == 0};
}

int h3d_json_escape(char *dest, size_t dest_len, const char *src) {
    if (!src || !dest || dest_len == 0) return 0;
    jbuf b = jb_init(dest, dest_len);
    jb_str(&b, src);
    if (b.overflow || b.off < 2) {
        dest[0] = '\0';
        return 0;
    }
    /* Drop the surrounding quotes. */
    memmove(dest, dest + 1, b.off - 2);
    dest[b.off - 2] = '\0';
    return (int)(b.off - 2);
}

/* ── Info response ───────────────────────────────────────────── */

static void jb_workers(jbuf *b, h3d_ctx *ctx) {
    jb_char(b, '[');
    int first = 1;
    for (h3d_worker *w = ctx->workers; w; w = w->next) {
        if (!first) jb_char(b, ',');
        first = 0;
        h3d_job *current = w->current_job;
        jb_printf(b, "{\"gpu_index\":%d,\"arch\":", w->gpu_index);
        jb_str(b, w->arch);
        jb_printf(b, ",\"state\":\"%s\",\"current_job\":",
                  atomic_load(&w->running) ? "busy" : "idle");
        jb_str(b, current ? current->job_id : NULL);
        pthread_mutex_lock(&w->cache_lock);
        h3_cache_info cache = w->cache;
        pthread_mutex_unlock(&w->cache_lock);
        jb_printf(b, ",\"cache\":{\"prepared_dit\":%s,\"video_decoder\":%s,"
                  "\"embedding_bytes\":%zu,\"release_pending\":%s}}",
                  cache.prepared_dit ? "true" : "false",
                  cache.video_decoder ? "true" : "false",
                  cache.embedding_bytes,
                  atomic_load(&w->release_requested) ? "true" : "false");
    }
    jb_char(b, ']');
}

int h3d_json_cache_release_response(h3d_ctx *ctx, int released, char *buf, size_t len) {
    jbuf b = jb_init(buf, len);
    jb_printf(&b, "{\"released\":%s,\"workers\":", released ? "true" : "false");
    jb_workers(&b, ctx);
    jb_char(&b, '}');
    return jb_end(&b);
}

int h3d_json_info_response(h3d_ctx *ctx, char *buf, size_t len) {
    jbuf b = jb_init(buf, len);
    jb_printf(&b, "{\"protocol\":\"%s\",\"h3_version\":\"%s\",\"ready\":%s,"
              "\"unready_reason\":", H3D_PROTOCOL, H3D_VERSION,
              atomic_load(&ctx->ready) ? "true" : "false");
    jb_opt_str(&b, ctx->unready_reason);
    jb_printf(&b, ",\"model_path\":");
    jb_str(&b, ctx->model_path);
    jb_printf(&b, ",\"output_root\":");
    jb_str(&b, ctx->output_root);
    jb_printf(&b, ",\"media_root\":");
    jb_str(&b, ctx->media_root);
    jb_printf(&b, ",\"quota\":{\"limit_bytes\":%" PRIu64 ",\"used_bytes\":%" PRIu64
              "},\"idle_release_sec\":%d,\"adapters\":{\"fasth3\":%s},\"workers\":",
              ctx->quota_bytes, ctx->quota_used_bytes, ctx->idle_release_sec,
              ctx->fasth3_lora[0] ? "true" : "false");
    jb_workers(&b, ctx);

    jb_printf(&b, ",\"limits\":{"
        "\"max_queued_jobs_per_client\":%d,"
        "\"prompt_max_chars\":%d,"
        "\"ref_images_max\":%d,\"ref_videos_max\":%d,\"ref_audios_max\":%d,"
        "\"ref_image_max_bytes\":52428800,"
        "\"ref_video_max_bytes\":524288000,"
        "\"ref_audio_max_bytes\":104857600},"
        "\"geometry\":{\"multiple\":%d,\"max_pixels\":%d,"
        "\"frames_min\":5,\"frames_max\":362,\"fps\":%d}}",
        ctx->max_queued_per_client, ctx->prompt_max_chars,
        H3D_MAX_REF_IMAGES, H3D_MAX_REF_VIDEOS, H3D_MAX_REF_AUDIOS,
        H3_CANVAS_MULTIPLE, H3_MAX_PIXELS, H3_FPS);
    return jb_end(&b);
}

/* ── Job response ────────────────────────────────────────────── */

static void jb_paths(jbuf *b, const h3d_ref_entry *entries, int count) {
    jb_char(b, '[');
    for (int i = 0; i < count; i++) {
        if (i) jb_char(b, ',');
        jb_str(b, entries[i].path);
    }
    jb_char(b, ']');
}

static void jb_error(jbuf *b, const h3d_error *error) {
    jb_printf(b, "{\"code\":");
    jb_str(b, error->code);
    jb_printf(b, ",\"message\":");
    jb_str(b, error->message);
    if (error->detail_count > 0) {
        jb_printf(b, ",\"details\":[");
        for (int i = 0; i < error->detail_count; i++) {
            if (i) jb_char(b, ',');
            jb_printf(b, "{\"field\":");
            jb_str(b, error->details[i].field);
            jb_printf(b, ",\"reason\":");
            jb_str(b, error->details[i].reason);
            jb_char(b, '}');
        }
        jb_char(b, ']');
    }
    jb_char(b, '}');
}

int h3d_json_job_response(h3d_job *job, char *buf, size_t len) {
    jbuf b = jb_init(buf, len);
    pthread_mutex_lock(&job->lock);

    jb_printf(&b, "{\"job_id\":");
    jb_str(&b, job->job_id);
    jb_printf(&b, ",\"status\":\"%s\",\"phase\":", h3d_status_name(job->status));
    jb_str(&b, job->phase);
    jb_printf(&b, ",\"progress\":{\"step\":%d,\"total_steps\":%d,"
              "\"elapsed_sec\":%ld,\"eta_hint_sec\":%d}",
              job->progress_step, job->progress_total,
              (long)job->elapsed_sec, job->eta_hint_sec);

    jb_printf(&b, ",\"request\":{\"mode\":\"%s\","
              "\"size\":{\"width\":%d,\"height\":%d},\"duration\":{\"frames\":%d",
              h3d_mode_name(job->mode), job->width, job->height, job->frames);
    if (job->seconds > 0) jb_printf(&b, ",\"seconds\":%.3f", job->seconds);
    jb_printf(&b, "},\"quality\":{\"preset\":");
    jb_str(&b, job->quality.preset);
    jb_printf(&b, ",\"steps\":%d,\"layers\":%d,\"reuse\":%d},",
              job->quality.steps, job->quality.layers, job->quality.reuse);
    if (job->seed_was_set) jb_printf(&b, "\"seed\":%" PRIu64 ",", job->seed);
    else jb_printf(&b, "\"seed\":null,");
    jb_printf(&b, "\"refs\":{\"images\":");
    jb_paths(&b, job->refs_images, job->refs_image_count);
    jb_printf(&b, ",\"videos\":");
    jb_paths(&b, job->refs_videos, job->refs_video_count);
    jb_printf(&b, ",\"audio\":");
    jb_paths(&b, job->refs_audio, job->refs_audio_count);
    jb_printf(&b, ",\"first_frame\":");
    jb_opt_str(&b, job->first_frame);
    jb_printf(&b, ",\"last_frame\":");
    jb_opt_str(&b, job->last_frame);
    jb_printf(&b, "},\"options\":{\"finalize_partial_on_cancel\":%s,"
              "\"release_after\":%s,"
              "\"token_reduction\":%s,\"sol_attn\":%s,\"fbc\":%s,\"vsa\":%s,"
              "\"core_reuse\":%d,\"render_width\":%d,\"render_height\":%d,"
              "\"fasth3_lora\":",
              job->finalize_partial_on_cancel ? "true" : "false",
              job->release_after ? "true" : "false",
              job->token_reduction ? "true" : "false",
              job->sol_attn ? "true" : "false",
              job->fbc ? "true" : "false",
              job->vsa ? "true" : "false",
              job->core_reuse, job->render_width, job->render_height);
    jb_opt_str(&b, job->fasth3_lora);
    jb_printf(&b, ",\"taeh3\":");
    jb_opt_str(&b, job->taeh3);
    jb_printf(&b, ",\"extra_args\":");
    jb_str(&b, job->extra_args);
    jb_printf(&b, "}},");

    jb_printf(&b, "\"created_at\":%ld,\"started_at\":%ld,\"finished_at\":",
              (long)job->created_at, (long)job->started_at);
    if (job->finished_at) jb_printf(&b, "%ld", (long)job->finished_at);
    else jb_printf(&b, "null");

    if (job->status == H3D_JOB_DONE && job->mp4_path) {
        jb_printf(&b, ",\"result\":{\"mp4_path\":");
        jb_str(&b, job->mp4_path);
        jb_printf(&b, ",\"poster_path\":");
        jb_str(&b, job->poster_path);
        jb_printf(&b, ",\"duration_sec\":%.2f,\"width\":%d,\"height\":%d,"
                  "\"frames\":%d,\"seed\":%" PRIu64 ",\"partial\":%s,\"cli\":",
                  job->duration_sec, job->width, job->height, job->frames,
                  job->seed, job->partial ? "true" : "false");
        jb_str(&b, job->cli_args);
        if (job->has_ffprobe && job->ffprobe_json[0])
            jb_printf(&b, ",\"ffprobe\":%s", job->ffprobe_json);
        jb_char(&b, '}');
    } else {
        jb_printf(&b, ",\"result\":null");
    }

    if (job->error.code[0]) {
        jb_printf(&b, ",\"error\":");
        jb_error(&b, &job->error);
    }
    jb_char(&b, '}');

    pthread_mutex_unlock(&job->lock);
    return jb_end(&b);
}

/* ── Error response ──────────────────────────────────────────── */

int h3d_json_error_response(const h3d_error *error, char *buf, size_t len) {
    jbuf b = jb_init(buf, len);
    jb_printf(&b, "{\"error\":");
    jb_error(&b, error);
    jb_char(&b, '}');
    return jb_end(&b);
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
    jbuf b = jb_init(buf, len);
    jb_printf(&b, "{\"jobs\":[");

    pthread_mutex_lock(&ctx->jobs_lock);
    int count = 0;
    for (size_t i = 0; i < ctx->job_count && count < limit; i++) {
        h3d_job *job = ctx->jobs[i];
        pthread_mutex_lock(&job->lock);
        const char *status = h3d_status_name(job->status);
        if (!status_filter || strcmp(status, status_filter) == 0) {
            if (count) jb_char(&b, ',');
            jb_printf(&b, "{\"job_id\":");
            jb_str(&b, job->job_id);
            jb_printf(&b, ",\"status\":\"%s\",\"phase\":", status);
            jb_str(&b, job->phase);
            jb_printf(&b, ",\"created_at\":%ld}", (long)job->created_at);
            count++;
        }
        pthread_mutex_unlock(&job->lock);
    }
    pthread_mutex_unlock(&ctx->jobs_lock);

    jb_printf(&b, "],\"count\":%d}", count);
    return jb_end(&b);
}
