#ifndef H3D_H
#define H3D_H

#include "h3.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <time.h>

#define H3D_VERSION "0.12.0-exp"
#define H3D_PROTOCOL "v1alpha"
#define H3D_DEFAULT_PORT 8571
#define H3D_MAX_JOB_ID 32
#define H3D_MAX_REFERENCES 12
#define H3D_MAX_ERROR_MESSAGE 512
#define H3D_SSE_EVENT_HISTORY 50
#define H3D_SSE_HEARTBEAT_SEC 15

/* ── Mode ─────────────────────────────────────────────────────── */
typedef enum {
    H3D_MODE_T2VA = 0,   /* text → video+audio (default) */
    H3D_MODE_I2VA,       /* image+text → video+audio */
    H3D_MODE_FL2VA,      /* first/last frame → video+audio */
    H3D_MODE_REF2VA      /* reference image/video/audio → video+audio */
} h3d_mode;

/* ── Quality presets ──────────────────────────────────────────── */
typedef struct {
    char preset[32];     /* fox-s2 | fox-fast | cinematic | custom */
    int steps;
    int layers;          /* dit_layers */
    int reuse;           /* denoise_reuse */
} h3d_quality;

#define H3D_QUALITY_DEFAULT_PRESET "fox-fast"

/* ── Geometry validation ──────────────────────────────────────── */
typedef struct {
    int width;
    int height;
    int frames;
} h3d_geometry;

/* Pre-validated geometry whitelist (from h3-hip.c testing) */
#define H3D_MAX_WHITELIST 16

/* ── Job status ───────────────────────────────────────────────── */
typedef enum {
    H3D_JOB_QUEUED,
    H3D_JOB_PREPARING,
    H3D_JOB_RUNNING,
    H3D_JOB_FINALIZING,
    H3D_JOB_DONE,
    H3D_JOB_FAILED,
    H3D_JOB_CANCELLED,
    H3D_JOB_TIMEOUT,
    H3D_JOB_CANCELLING
} h3d_job_status;

/* ── Error ────────────────────────────────────────────────────── */
typedef struct {
    char code[64];
    char message[H3D_MAX_ERROR_MESSAGE];
    /* details array for VALIDATION_ERROR etc. */
    struct {
        char field[64];
        char reason[128];
    } details[8];
    int detail_count;
} h3d_error;

/* ── Reference files ──────────────────────────────────────────── */
typedef struct {
    char path[1024];
} h3d_ref_entry;

/* ── Job ──────────────────────────────────────────────────────── */
typedef struct {
    char job_id[H3D_MAX_JOB_ID];
    h3d_job_status status;
    char phase[64];
    int progress_step;
    int progress_total;
    time_t elapsed_sec;
    int eta_hint_sec;
    char cache_hint[16];

    /* Request fields (semantic) */
    h3d_mode mode;
    int width;
    int height;
    int frames;              /* mutually exclusive with seconds */
    float seconds;           /* mutually exclusive with frames */
    h3d_quality quality;
    uint64_t seed;
    int seed_was_set;

    /* References */
    h3d_ref_entry refs_images[H3D_MAX_REFERENCES];
    int refs_image_count;
    h3d_ref_entry refs_video;
    int has_refs_video;
    h3d_ref_entry refs_audio[3];
    int refs_audio_count;

    /* Options */
    int finalize_partial_on_cancel;
    int write_preview_frames;
    char extra_args[2048];

    /* Idempotency */
    char idempotency_key[128];

    /* Prompt */
    char prompt[8192];

    /* CLI echo (for reproduce) */
    char cli_args[4096];

    /* Output */
    char *output_path;
    char *mp4_path;
    char *poster_path;

    /* Result metadata */
    float duration_sec;
    int has_ffprobe;
    char ffprobe_json[1024];

    /* Timestamps */
    time_t created_at;
    time_t started_at;
    time_t finished_at;

    /* Error */
    h3d_error error;
    int partial;

    /* SSE event ring buffer */
    char events[H3D_SSE_EVENT_HISTORY][2048];
    int event_head;
    int event_tail;
    int event_count;
    pthread_mutex_t lock;
    atomic_int cancel_requested;
} h3d_job;

/* ── Worker ───────────────────────────────────────────────────── */
typedef struct h3d_ctx h3d_ctx;

typedef struct h3d_worker {
    int gpu_index;
    char arch[64];
    h3_ctx *ctx;         /* inference context (per-GPU) */
    h3_ctx *ctx_ref2va;  /* Ref2VA inference context (if loaded) */
    h3d_ctx *dctx;       /* daemon context (shared) */
    pthread_t thread;
    atomic_int running;
    h3d_job *current_job;
    struct h3d_worker *next;
} h3d_worker;

/* ── Daemon context ───────────────────────────────────────────── */
struct h3d_ctx {
    /* Config */
    char bind_addr[64];
    int port;
    char model_path_fl2va[1024];
    char model_path_ref2va[1024];
    char model_path[1024];         /* compatibility: shared fallback */
    char output_root[1024];
    char media_root[1024];
    uint64_t quota_bytes;
    int timeout_tier1;
    int timeout_tier2;
    int timeout_tier3;
    char log_level[16];
    char log_dir[1024];
    int max_queued_per_client;
    int prompt_max_chars;
    int gpu_indices[16];
    int gpu_count;

    /* Geometry whitelist */
    h3d_geometry geometry_whitelist[H3D_MAX_WHITELIST];
    int geometry_whitelist_count;

    /* Runtime state */
    h3d_worker *workers;
    int worker_count;
    int server_fd;
    atomic_int shutdown;
    atomic_int ready;              /* 0=not ready, 1=ready */
    char unready_reason[256];
    pthread_mutex_t jobs_lock;
    h3d_job **jobs;
    size_t job_count;
    size_t job_capacity;
    pthread_mutex_t quota_lock;
    uint64_t quota_used_bytes;
};

/* ── API: Config ──────────────────────────────────────────────── */
void h3d_config_init(h3d_ctx *ctx);
int h3d_config_load(h3d_ctx *ctx);

/* ── API: HTTP ────────────────────────────────────────────────── */
int h3d_http_listen(const char *bind, int port);
int h3d_http_accept(int server_fd);
int h3d_http_read_request(int fd, char *method, size_t method_len,
                          char *path, size_t path_len,
                          char *headers, size_t headers_len,
                          char *body, size_t body_len,
                          size_t *body_out_len);
void h3d_http_send_response(int fd, int status, const char *status_text,
                            const char *content_type, const char *body,
                            size_t body_len);
void h3d_http_send_headers(int fd, int status, const char *status_text,
                           const char *extra_headers);
void h3d_http_send_chunk(int fd, const char *data, size_t len);
void h3d_http_send_sse_event(int fd, const char *event, const char *data);
void h3d_http_send_sse_heartbeat(int fd);
void h3d_http_close(int fd);

/* ── API: JSON helpers ────────────────────────────────────────── */
int h3d_json_escape(char *dest, size_t dest_len, const char *src);
int h3d_json_info_response(h3d_ctx *ctx, char *buf, size_t len);
int h3d_json_job_response(h3d_job *job, char *buf, size_t len);
int h3d_json_error_response(const h3d_error *error, char *buf, size_t len);
int h3d_json_queue_full_response(char *buf, size_t len);
int h3d_json_jobs_list_response(h3d_ctx *ctx, const char *status_filter,
                                int limit, char *buf, size_t len);

/* ── API: Job management ──────────────────────────────────────── */
void h3d_jobs_init(h3d_ctx *ctx);
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
                        const char *idempotency_key);
h3d_job *h3d_job_find(h3d_ctx *ctx, const char *job_id);
h3d_job *h3d_job_find_by_idempotency(h3d_ctx *ctx, const char *key);
int h3d_job_cancel(h3d_ctx *ctx, const char *job_id);
int h3d_job_count_by_status(h3d_ctx *ctx, h3d_job_status status);
void h3d_jobs_mark_all_failed(h3d_ctx *ctx);
void h3d_job_record_event(h3d_job *job, const char *event, const char *data);
const char *h3d_mode_name(h3d_mode mode);
const char *h3d_status_name(h3d_job_status status);

/* ── API: Worker ──────────────────────────────────────────────── */
int h3d_workers_init(h3d_ctx *ctx);
void *h3d_worker_loop(void *arg);
void h3d_workers_shutdown(h3d_ctx *ctx);

/* ── API: Geometry validation ─────────────────────────────────── */
int h3d_geometry_is_valid(h3d_ctx *ctx, int width, int height, int frames);

/* ── API: Poster generation ───────────────────────────────────── */
int h3d_generate_poster(const char *mp4_path, const char *poster_path);

/* ── API: Daemon main ─────────────────────────────────────────── */
int h3d_serve(h3d_ctx *ctx);

#endif
