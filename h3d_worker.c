#include "h3d_worker.h"
#include "h3d_job.h"
#include "h3d_json.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

/* Progress callback that updates job state and records SSE events. */
static int worker_progress(const char *phase, int completed, int total,
                           void *opaque) {
    h3d_job *job = opaque;
    pthread_mutex_lock(&job->lock);
    snprintf(job->phase, sizeof(job->phase), "%s", phase);
    job->progress_step = completed;
    job->progress_total = total;
    if (job->started_at) {
        job->elapsed_sec = time(NULL) - job->started_at;
        if (completed > 0 && total > completed) {
            double rate = (double)completed / (double)job->elapsed_sec;
            job->eta_hint_sec = (int)(((double)(total - completed)) / rate);
        }
    }
    pthread_mutex_unlock(&job->lock);

    /* Record SSE event */
    char data[512];
    snprintf(data, sizeof(data),
        "{\"phase\":\"%s\",\"step\":%d,\"total_steps\":%d,"
        "\"elapsed_sec\":%ld,\"eta_hint_sec\":%d}",
        phase, completed, total, (long)job->elapsed_sec, job->eta_hint_sec);
    h3d_job_record_event(job, "progress", data);

    /* Check cancel */
    return atomic_load(&job->cancel_requested) ? 1 : 0;
}

static h3d_job *find_next_job(h3d_ctx *dctx) {
    pthread_mutex_lock(&dctx->jobs_lock);
    for (size_t i = 0; i < dctx->job_count; i++) {
        h3d_job *job = dctx->jobs[i];
        if (job->status == H3D_JOB_QUEUED) {
            pthread_mutex_lock(&job->lock);
            job->status = H3D_JOB_PREPARING;
            snprintf(job->phase, sizeof(job->phase), "loading_weights");
            pthread_mutex_unlock(&job->lock);
            pthread_mutex_unlock(&dctx->jobs_lock);
            return job;
        }
    }
    pthread_mutex_unlock(&dctx->jobs_lock);
    return NULL;
}

void *h3d_worker_loop(void *arg) {
    h3d_worker *worker = arg;
    h3d_ctx *dctx = worker->dctx;

    while (!atomic_load(&dctx->shutdown)) {
        h3d_job *job = find_next_job(dctx);
        if (!job) {
            usleep(100000); /* 100ms idle poll */
            continue;
        }

        worker->current_job = job;
        atomic_store(&worker->running, 1);

        /* Record preparing event */
        char data[256];
        snprintf(data, sizeof(data),
            "{\"status\":\"preparing\",\"phase\":\"loading_weights\","
            "\"cache_hint\":\"warm\"}");
        h3d_job_record_event(job, "status", data);

        /* Mark running */
        pthread_mutex_lock(&job->lock);
        job->status = H3D_JOB_RUNNING;
        job->started_at = time(NULL);
        pthread_mutex_unlock(&job->lock);

        h3d_job_record_event(job, "status",
            "{\"status\":\"running\"}");

        /* Set up h3_params for generation */
        h3_params params = job->params;
        params.output_path = job->mp4_path;
        params.on_progress = worker_progress;
        params.callback_opaque = job;
        params.preview_denoise = 0;
        params.on_frame = NULL;
        params.references = job->references;
        params.reference_count = job->reference_count;

        /* Enable cache for weight reuse between jobs */
        h3_cache_set_enabled(worker->ctx, 1);

        /* Generate */
        h3_result *result = h3_generate(worker->ctx, job->prompt, &params);

        if (atomic_load(&job->cancel_requested)) {
            /* Cancelled during generation */
            pthread_mutex_lock(&job->lock);
            job->status = H3D_JOB_CANCELLED;
            job->finished_at = time(NULL);
            snprintf(job->error.code, sizeof(job->error.code), "CANCELLED");
            snprintf(job->error.message, sizeof(job->error.message),
                     "Job was cancelled by user");
            pthread_mutex_unlock(&job->lock);

            h3d_job_record_event(job, "cancelled",
                "{\"status\":\"cancelled\"}");
        } else if (!result) {
            /* Failed */
            const char *err = h3_last_error(worker->ctx);
            pthread_mutex_lock(&job->lock);
            job->status = H3D_JOB_FAILED;
            job->finished_at = time(NULL);
            snprintf(job->error.code, sizeof(job->error.code), "GENERATION_FAILED");
            snprintf(job->error.message, sizeof(job->error.message),
                     "%s", err ? err : "Unknown error");
            pthread_mutex_unlock(&job->lock);

            char err_data[1024];
            char escaped[512];
            h3d_json_escape(escaped, sizeof(escaped), err ? err : "Unknown");
            snprintf(err_data, sizeof(err_data),
                "{\"status\":\"failed\",\"error\":{\"code\":\"GENERATION_FAILED\","
                "\"message\":\"%s\"}}", escaped);
            h3d_job_record_event(job, "failed", err_data);
        } else {
            /* Success */
            pthread_mutex_lock(&job->lock);
            job->status = H3D_JOB_FINALIZING;
            pthread_mutex_unlock(&job->lock);

            h3d_job_record_event(job, "status",
                "{\"status\":\"finalizing\"}");

            /* TODO: extract poster frame from first decoded frame */

            pthread_mutex_lock(&job->lock);
            job->status = H3D_JOB_DONE;
            job->finished_at = time(NULL);
            job->seed = result->seed;
            pthread_mutex_unlock(&job->lock);

            /* Record done event with result */
            char done_data[2048];
            char escaped_mp4[2048];
            h3d_json_escape(escaped_mp4, sizeof(escaped_mp4), job->mp4_path);
            snprintf(done_data, sizeof(done_data),
                "{\"job_id\":\"%s\",\"result\":{"
                "\"mp4_path\":\"%s\","
                "\"width\":%d,\"height\":%d,\"frames\":%d,"
                "\"seed\":%" PRIu64 "}}",
                job->job_id, escaped_mp4,
                result->width, result->height, result->frames, result->seed);
            h3d_job_record_event(job, "done", done_data);

            h3_result_free(result);
        }

        worker->current_job = NULL;
        atomic_store(&worker->running, 0);
    }
    return NULL;
}

int h3d_workers_init(h3d_ctx *ctx) {
    ctx->workers = NULL;
    ctx->worker_count = 0;

    for (int i = 0; i < ctx->gpu_count; i++) {
        h3d_worker *worker = calloc(1, sizeof(h3d_worker));
        if (!worker) return -1;

        worker->gpu_index = ctx->gpu_indices[i];
        worker->dctx = ctx;
        worker->ctx = h3_load_dir(ctx->model_path);
        if (!worker->ctx) {
            fprintf(stderr, "h3d: cannot load model from %s: %s\n",
                    ctx->model_path, h3_last_error(NULL));
            free(worker);
            continue;
        }

        /* Set HIP_VISIBLE_DEVICES for this worker */
        char gpu_env[32];
        snprintf(gpu_env, sizeof(gpu_env), "%d", worker->gpu_index);
        setenv("HIP_VISIBLE_DEVICES", gpu_env, 1);

        snprintf(worker->arch, sizeof(worker->arch), "gpu%d", worker->gpu_index);
        atomic_store(&worker->running, 0);

        /* Prepend to list */
        worker->next = ctx->workers;
        ctx->workers = worker;
        ctx->worker_count++;

        /* Start worker thread */
        if (pthread_create(&worker->thread, NULL, h3d_worker_loop, worker) != 0) {
            fprintf(stderr, "h3d: cannot start worker thread for GPU %d\n",
                    worker->gpu_index);
            h3d_worker *next = worker->next;
            h3_free(worker->ctx);
            free(worker);
            ctx->workers = next;
            ctx->worker_count--;
            continue;
        }
    }

    return ctx->worker_count > 0 ? 0 : -1;
}

void h3d_workers_shutdown(h3d_ctx *ctx) {
    atomic_store(&ctx->shutdown, 1);

    h3d_worker *w = ctx->workers;
    while (w) {
        pthread_join(w->thread, NULL);
        if (w->ctx) h3_free(w->ctx);
        h3d_worker *next = w->next;
        free(w);
        w = next;
    }
    ctx->workers = NULL;
    ctx->worker_count = 0;
}
