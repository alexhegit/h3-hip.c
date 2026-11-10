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
        pthread_mutex_lock(&job->lock);
        if (job->status == H3D_JOB_QUEUED) {
            job->status = H3D_JOB_PREPARING;
            snprintf(job->phase, sizeof(job->phase), "loading_weights");
            pthread_mutex_unlock(&job->lock);
            pthread_mutex_unlock(&dctx->jobs_lock);
            return job;
        }
        pthread_mutex_unlock(&job->lock);
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

static void add_reference(h3d_job *job, size_t *count, h3_reference_kind kind,
                          const char *path) {
    if (*count >= H3D_MAX_REFERENCES) return;
    /* Videos keep their soundtrack, like the CLI's --ref-video. */
    job->references[*count] = (h3_reference){
        .kind = kind, .path = path, .audio_path = NULL,
        .include_embedded_audio = kind == H3_REFERENCE_VIDEO,
    };
    (*count)++;
}

static void build_h3_params(h3d_job *job, h3_params *params) {
    *params = (h3_params)H3_PARAMS_DEFAULT;
    params->width = job->width;
    params->height = job->height;
    params->frames = job->frames;
    params->steps = job->quality.steps;
    params->seed = job->seed;
    params->output_path = job->mp4_path;
    params->denoise_reuse = job->quality.reuse;
    params->dit_layers = job->quality.layers;
    params->core_reuse = job->core_reuse;
    params->token_reduction = job->token_reduction;
    params->sol_attn = job->sol_attn;
    params->fbc = job->fbc;
    params->vsa = job->vsa;
    params->render_width = job->render_width;
    params->render_height = job->render_height;
    params->fasth3_lora = job->fasth3_lora[0] ? job->fasth3_lora : NULL;
    params->taeh3 = job->taeh3[0] ? job->taeh3 : NULL;
    params->first_frame = job->first_frame[0] ? job->first_frame : NULL;
    params->last_frame = job->last_frame[0] ? job->last_frame : NULL;
    params->on_progress = worker_progress;
    params->callback_opaque = job;

    size_t count = 0;
    for (int i = 0; i < job->refs_image_count; i++)
        add_reference(job, &count, H3_REFERENCE_IMAGE, job->refs_images[i].path);
    for (int i = 0; i < job->refs_video_count; i++)
        add_reference(job, &count, H3_REFERENCE_VIDEO, job->refs_videos[i].path);
    for (int i = 0; i < job->refs_audio_count; i++)
        add_reference(job, &count, H3_REFERENCE_AUDIO, job->refs_audio[i].path);
    params->references = count ? job->references : NULL;
    params->reference_count = count;
}

/* ── Generation cache ────────────────────────────────────────── */

static void publish_cache(h3d_worker *worker) {
    h3_cache_info fl2va, ref2va = {0};
    h3_cache_get_info(worker->ctx, &fl2va);
    if (worker->ctx_ref2va) h3_cache_get_info(worker->ctx_ref2va, &ref2va);
    pthread_mutex_lock(&worker->cache_lock);
    worker->cache.embedding_entries = fl2va.embedding_entries + ref2va.embedding_entries;
    worker->cache.embedding_bytes = fl2va.embedding_bytes + ref2va.embedding_bytes;
    worker->cache.prepared_dit = fl2va.prepared_dit || ref2va.prepared_dit;
    worker->cache.video_decoder = fl2va.video_decoder || ref2va.video_decoder;
    pthread_mutex_unlock(&worker->cache_lock);
}

static int cache_resident(h3d_worker *worker) {
    pthread_mutex_lock(&worker->cache_lock);
    int resident = worker->cache.prepared_dit || worker->cache.video_decoder ||
                   worker->cache.embedding_entries;
    pthread_mutex_unlock(&worker->cache_lock);
    return resident;
}

static void release_cache(h3d_worker *worker, const char *why) {
    int resident = cache_resident(worker);
    h3_cache_clear(worker->ctx);
    if (worker->ctx_ref2va) h3_cache_clear(worker->ctx_ref2va);
    publish_cache(worker);
    if (resident)
        fprintf(stderr, "h3d: GPU %d: released the generation cache (%s)\n",
                worker->gpu_index, why);
}

static void idle_tick(h3d_worker *worker) {
    int idle_release_sec = worker->dctx->idle_release_sec;
    if (atomic_load(&worker->release_requested)) {
        release_cache(worker, "requested");
        atomic_store(&worker->release_requested, 0);
    } else if (idle_release_sec > 0 && worker->idle_since &&
               time(NULL) - worker->idle_since >= idle_release_sec &&
               cache_resident(worker)) {
        release_cache(worker, "idle timeout");
    }
}

/* ── Worker loop ─────────────────────────────────────────────── */

void *h3d_worker_loop(void *arg) {
    h3d_worker *worker = arg;
    h3d_ctx *dctx = worker->dctx;

    while (!atomic_load(&dctx->shutdown)) {
        h3d_job *job = find_next_job(dctx);
        if (!job) {
            idle_tick(worker);
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
        if (job->release_after) release_cache(worker, "release_after");
        else publish_cache(worker);

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
            h3_result_free(result);
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

            h3d_job_record_event(job, "failed", "{\"status\":\"failed\"}");
        } else {
            /* Success */
            pthread_mutex_lock(&job->lock);
            job->status = H3D_JOB_FINALIZING;
            pthread_mutex_unlock(&job->lock);

            h3d_job_record_event(job, "status",
                "{\"status\":\"finalizing\"}");

            /* Generate poster frame */
            h3d_generate_poster(job->mp4_path, job->poster_path);

            /* Extract ffprobe metadata */
            if (h3d_extract_ffprobe(job->mp4_path, job->ffprobe_json, sizeof(job->ffprobe_json)) == 0) {
                job->has_ffprobe = 1;
            }

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

            /* The events stream sends the full job object for this. */
            h3d_job_record_event(job, "done", "{\"status\":\"done\"}");

            h3_result_free(result);
        }

        worker->idle_since = time(NULL);
        worker->current_job = NULL;
        atomic_store(&worker->running, 0);
    }
    return NULL;
}

/* ── Workers init ────────────────────────────────────────────── */

/* The daemon keeps the prepared DiT cached between jobs. On a discrete card
 * below 48 GiB that cached DiT leaves the video VAE too little room, so its
 * tensors and the next job's fall back to pinned host memory (KI-007).
 * Parking the INT8 linears in host RAM gives the same output and fits. An
 * explicit H3_INT8_HOST, or a non-INT8 DiT, is left alone. */
static void default_int8_host_for_cache(const h3_ctx *h3) {
    const h3_device_info *device = h3_device(h3);
    if (!device || device->unified_memory || !device->recommended_working_set)
        return;
    if (getenv("H3_INT8_HOST")) return;
    const char *int8_mlp = getenv("H3_INT8_MLP");
    const char *fp8_mlp = getenv("H3_FP8_MLP");
    if ((int8_mlp && !strcmp(int8_mlp, "0")) || (fp8_mlp && strcmp(fp8_mlp, "0")))
        return;
    double gib = (double)device->recommended_working_set / (1024.0 * 1024.0 * 1024.0);
    if (gib >= 48.0) return;
    setenv("H3_INT8_HOST", "1", 1);
    fprintf(stderr, "h3d: %.1f GiB card: H3_INT8_HOST=1 so the cached DiT leaves "
            "room for the video VAE (set H3_INT8_HOST=0 to keep it on the card)\n",
            gib);
}

int h3d_workers_init(h3d_ctx *ctx) {
    ctx->workers = NULL;
    ctx->worker_count = 0;

    for (int i = 0; i < ctx->gpu_count; i++) {
        h3d_worker *worker = calloc(1, sizeof(h3d_worker));
        if (!worker) return -1;

        worker->gpu_index = ctx->gpu_indices[i];
        worker->dctx = ctx;
        pthread_mutex_init(&worker->cache_lock, NULL);

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
        default_int8_host_for_cache(worker->ctx);

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

    /* A running job would otherwise hold the join for the rest of its
     * denoise; cancelled, it stops before its next step. */
    for (h3d_worker *w = ctx->workers; w; w = w->next) {
        h3d_job *job = w->current_job;
        if (job) h3d_job_cancel(ctx, job->job_id);
    }

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
