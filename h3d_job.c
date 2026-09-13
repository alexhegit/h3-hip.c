#include "h3d_job.h"
#include "h3d_json.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static void generate_job_id(char *buf, size_t len) {
    static const char alphabet[] = "abcdefghijklmnopqrstuvwxyz0123456789";
    unsigned int seed = (unsigned int)(time(NULL) ^ (uintptr_t)buf ^ (uintptr_t)getpid());
    for (size_t i = 0; i < len - 1; i++) {
        seed = seed * 1103515245 + 12345;
        buf[i] = alphabet[(seed >> 16) % 36];
    }
    buf[len - 1] = '\0';
}

int h3d_job_is_terminal(h3d_job_status s) {
    return s == H3D_JOB_DONE || s == H3D_JOB_FAILED ||
           s == H3D_JOB_CANCELLED || s == H3D_JOB_TIMEOUT;
}

void h3d_jobs_init(h3d_ctx *ctx) {
    ctx->job_capacity = 64;
    ctx->job_count = 0;
    ctx->jobs = calloc(ctx->job_capacity, sizeof(h3d_job *));
}

h3d_job *h3d_job_create(h3d_ctx *ctx, const char *prompt,
                        const h3_params *params,
                        const h3_reference *refs, size_t ref_count) {
    h3d_job *job = calloc(1, sizeof(h3d_job));
    if (!job) return NULL;

    pthread_mutex_init(&job->lock, NULL);
    atomic_store(&job->cancel_requested, 0);
    generate_job_id(job->job_id, sizeof(job->job_id));
    job->status = H3D_JOB_QUEUED;
    job->created_at = time(NULL);
    job->params = *params;
    job->seed = params->seed;

    /* Copy prompt */
    job->prompt = strdup(prompt);
    if (!job->prompt) { free(job); return NULL; }

    /* Copy references */
    job->reference_count = ref_count;
    for (size_t i = 0; i < ref_count && i < H3D_MAX_REFERENCES; i++) {
        job->references[i] = refs[i];
        job->references[i].path = refs[i].path ? strdup(refs[i].path) : NULL;
        job->references[i].audio_path = refs[i].audio_path ?
            strdup(refs[i].audio_path) : NULL;
    }

    /* Generate output path */
    char output_dir[1024];
    snprintf(output_dir, sizeof(output_dir), "%s/%s", ctx->output_root, job->job_id);
    mkdir(output_dir, 0755);
    job->output_path = strdup(output_dir);

    char mp4_path[1024];
    snprintf(mp4_path, sizeof(mp4_path), "%s/%s.mp4", output_dir, job->job_id);
    job->mp4_path = strdup(mp4_path);
    job->params.output_path = job->mp4_path;

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

int h3d_job_cancel(h3d_ctx *ctx, const char *job_id) {
    h3d_job *job = h3d_job_find(ctx, job_id);
    if (!job) return -1;

    pthread_mutex_lock(&job->lock);
    if (job->status == H3D_JOB_DONE || job->status == H3D_JOB_FAILED ||
        job->status == H3D_JOB_CANCELLED || job->status == H3D_JOB_TIMEOUT) {
        pthread_mutex_unlock(&job->lock);
        return 0; /* Already terminal */
    }
    atomic_store(&job->cancel_requested, 1);
    job->status = H3D_JOB_CANCELLING;
    pthread_mutex_unlock(&job->lock);
    return 1;
}

int h3d_job_count_by_status(h3d_ctx *ctx, h3d_job_status status) {
    int count = 0;
    pthread_mutex_lock(&ctx->jobs_lock);
    for (size_t i = 0; i < ctx->job_count; i++) {
        if (ctx->jobs[i]->status == status) count++;
    }
    pthread_mutex_unlock(&ctx->jobs_lock);
    return count;
}

void h3d_jobs_mark_all_failed(h3d_ctx *ctx) {
    h3d_error err = { .code = "JOB_LOST_AFTER_RESTART",
                      .message = "Daemon restarted; this job was lost" };
    pthread_mutex_lock(&ctx->jobs_lock);
    for (size_t i = 0; i < ctx->job_count; i++) {
        h3d_job *job = ctx->jobs[i];
        if (job->status != H3D_JOB_DONE && job->status != H3D_JOB_FAILED &&
            job->status != H3D_JOB_CANCELLED && job->status != H3D_JOB_TIMEOUT) {
            pthread_mutex_lock(&job->lock);
            job->status = H3D_JOB_FAILED;
            job->error = err;
            job->finished_at = time(NULL);
            pthread_mutex_unlock(&job->lock);
        }
    }
    pthread_mutex_unlock(&ctx->jobs_lock);
}

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
