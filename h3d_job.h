#ifndef H3D_JOB_H
#define H3D_JOB_H

#include "h3d.h"

int h3d_job_is_terminal(h3d_job_status status);
void h3d_jobs_init(h3d_ctx *ctx);
h3d_job *h3d_job_new(h3d_ctx *ctx);
int h3d_job_enqueue(h3d_ctx *ctx, h3d_job *job);
void h3d_job_free(h3d_job *job);
h3d_job *h3d_job_find(h3d_ctx *ctx, const char *job_id);
h3d_job *h3d_job_find_by_idempotency(h3d_ctx *ctx, const char *key);
int h3d_job_cancel(h3d_ctx *ctx, const char *job_id);
int h3d_job_count_by_status(h3d_ctx *ctx, h3d_job_status status);
void h3d_jobs_mark_all_failed(h3d_ctx *ctx);
void h3d_job_record_event(h3d_job *job, const char *event, const char *data);
const char *h3d_mode_name(h3d_mode mode);
const char *h3d_status_name(h3d_job_status status);
int h3d_geometry_is_valid(int width, int height, int frames);
int h3d_generate_poster(const char *mp4_path, const char *poster_path);
int h3d_extract_ffprobe(const char *mp4_path, char *buf, size_t len);

#endif
