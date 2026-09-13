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

/* ── Progress callback ───────────────────────────────────────── */

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

    return atomic_load(&job->cancel_requested) ? 1 : 0;
}

/* ── Find next job ───────────────────────────────────────────── */

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

/* ── Select worker context by mode ───────────────────────────── */

static h3_ctx *select_worker_ctx(h3d_worker *worker, h3d_mode mode) {
    /* Ref2VA uses separate context if loaded */
    if (mode == H3D_MODE_REF2VA && worker->ctx_ref2va) {
        return worker->ctx_ref2va;
    }
    return worker->ctx;
}

/* ── Build h3_params from job ────────────────────────────────── */

static h3_reference g_refs[H3D_MAX_REFERENCES];

static void build_h3_params(h3d_job *job, h3_params *params) {
    /* Zero-init with designated initializer */
    *params = (h3_params){
        .width = job->width,
        .height = job->height,
        .frames = job->frames,
        .steps = job->quality.steps,
        .seed = job->seed,
        .output_path = job->mp4_path,
        .denoise_reuse = job->quality.reuse,
        .dit_layers = job->quality.layers,
        .core_reuse = 1,
        .preview_denoise = 0,
        .on_progress = worker_progress,
        .callback_opaque = job,
    };

    /* Build references array */
    memset(g_refs, 0, sizeof(g_refs));
    int ref_count = 0;

    if (job->refs_image_count > 0) {
        for (int i = 0; i < job->refs_image_count && ref_count < H3D_MAX_REFERENCES; i++) {
            g_refs[ref_count].kind = H3_REFERENCE_IMAGE;
            g_refs[ref_count].path = job->refs_images[i].path;
            g_refs[ref_count].audio_path = NULL;
            g_refs[ref_count].include_embedded_audio = 0;
            ref_count++;
        }
    }
    if (job->has_refs_video) {
        g_refs[ref_count].kind = H3_REFERENCE_VIDEO;
        g_refs[ref_count].path = job->refs_video.path;
        g_refs[ref_count].audio_path = NULL;
        g_refs[ref_count].include_embedded_audio = 0;
        ref_count++;
    }
    for (int i = 0; i < job->refs_audio_count && ref_count < H3D_MAX_REFERENCES; i++) {
        g_refs[ref_count].kind = H3_REFERENCE_AUDIO;
        g_refs[ref_count].path = job->refs_audio[i].path;
        g_refs[ref_count].audio_path = NULL;
        g_refs[ref_count].include_embedded_audio = 0;
        ref_count++;
    }

    params->references = g_refs;
    params->reference_count = (size_t)ref_count;
}

/* ── Worker loop ─────────────────────────────────────────────── */

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

        h3d_job_record_event(job, "status", "{\"status\":\"running\"}");

        /* Select context based on mode */
        h3_ctx *ctx = select_worker_ctx(worker, job->mode);

        /* Enable cache for weight reuse between jobs */
        h3_cache_set_enabled(ctx, 1);

        /* Build params */
        h3_params params;
        build_h3_params(job, &params);

        /* Generate */
        h3_result *result = h3_generate(ctx, job->prompt, &params);

        if (atomic_load(&job->cancel_requested)) {
            /* Cancelled during generation */
            pthread_mutex_lock(&job->lock);
            if (job->finalize_partial_on_cancel && job->partial) {
                job->status = H3D_JOB_DONE;
            } else {
                job->status = H3D_JOB_CANCELLED;
            }
            job->finished_at = time(NULL);
            snprintf(job->error.code, sizeof(job->error.code), "CANCELLED");
            snprintf(job->error.message, sizeof(job->error.message),
                     "Job was cancelled by user");
            pthread_mutex_unlock(&job->lock);

            h3d_job_record_event(job, "cancelled",
                "{\"status\":\"cancelled\"}");
        } else if (!result) {
            /* Failed */
            const char *err = h3_last_error(ctx);
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

            /* Generate poster frame */
            h3d_generate_poster(job->mp4_path, job->poster_path);

            /* Calculate duration */
            if (job->frames > 0 && job->quality.steps > 0) {
                /* Rough estimate: frames / 24 fps */
                job->duration_sec = (float)job->frames / 24.0f;
            }

            pthread_mutex_lock(&job->lock);
            job->status = H3D_JOB_DONE;
            job->finished_at = time(NULL);
            job->seed = result->seed;
            pthread_mutex_unlock(&job->lock);

            /* Record done event with result */
            char done_data[4096];
            char escaped_mp4[2048];
            h3d_json_escape(escaped_mp4, sizeof(escaped_mp4), job->mp4_path);
            char escaped_poster[2048];
            h3d_json_escape(escaped_poster, sizeof(escaped_poster),
                           job->poster_path ? job->poster_path : "");
            char escaped_cli[4096];
            h3d_json_escape(escaped_cli, sizeof(escaped_cli), job->cli_args);

            snprintf(done_data, sizeof(done_data),
                "{\"job_id\":\"%s\",\"result\":{"
                "\"mp4_path\":\"%s\","
                "\"poster_path\":\"%s\","
                "\"duration_sec\":%.2f,"
                "\"width\":%d,\"height\":%d,\"frames\":%d,"
                "\"seed\":%" PRIu64 ","
                "\"cli\":\"%s\"}}",
                job->job_id, escaped_mp4, escaped_poster,
                job->duration_sec,
                result->width, result->height, result->frames, result->seed,
                escaped_cli);
            h3d_job_record_event(job, "done", done_data);

            h3_result_free(result);
        }

        worker->current_job = NULL;
        atomic_store(&worker->running, 0);
    }
    return NULL;
}

/* ── Workers init ────────────────────────────────────────────── */

int h3d_workers_init(h3d_ctx *ctx) {
    ctx->workers = NULL;
    ctx->worker_count = 0;

    for (int i = 0; i < ctx->gpu_count; i++) {
        h3d_worker *worker = calloc(1, sizeof(h3d_worker));
        if (!worker) return -1;

        worker->gpu_index = ctx->gpu_indices[i];
        worker->dctx = ctx;

        /* Load FL2VA model (primary) */
        const char *model_path = ctx->model_path_fl2va[0] ? ctx->model_path_fl2va : ctx->model_path;
        if (!model_path || !model_path[0]) {
            fprintf(stderr, "h3d: no model path configured\n");
            free(worker);
            continue;
        }

        worker->ctx = h3_load_dir(model_path);
        if (!worker->ctx) {
            fprintf(stderr, "h3d: cannot load model from %s: %s\n",
                    model_path, h3_last_error(NULL));
            free(worker);
            continue;
        }

        /* Load Ref2VA model if configured and different */
        if (ctx->model_path_ref2va[0] &&
            strcmp(ctx->model_path_ref2va, model_path) != 0) {
            worker->ctx_ref2va = h3_load_dir(ctx->model_path_ref2va);
            if (!worker->ctx_ref2va) {
                fprintf(stderr, "h3d: warning: cannot load Ref2VA model from %s: %s\n",
                        ctx->model_path_ref2va, h3_last_error(NULL));
                /* Non-fatal: continue without Ref2VA */
            }
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
            if (worker->ctx) h3_free(worker->ctx);
            if (worker->ctx_ref2va) h3_free(worker->ctx_ref2va);
            free(worker);
            ctx->workers = next;
            ctx->worker_count--;
            continue;
        }
    }

    return ctx->worker_count > 0 ? 0 : -1;
}

/* ── Workers shutdown ────────────────────────────────────────── */

void h3d_workers_shutdown(h3d_ctx *ctx) {
    atomic_store(&ctx->shutdown, 1);

    h3d_worker *w = ctx->workers;
    while (w) {
        pthread_join(w->thread, NULL);
        if (w->ctx) h3_free(w->ctx);
        if (w->ctx_ref2va) h3_free(w->ctx_ref2va);
        h3d_worker *next = w->next;
        free(w);
        w = next;
    }
    ctx->workers = NULL;
    ctx->worker_count = 0;
}
