#ifndef H3D_JOB_H
#define H3D_JOB_H

#include "h3d.h"

int h3d_job_is_terminal(h3d_job_status status);
void h3d_jobs_init(h3d_ctx *ctx);
h3d_job *h3d_job_create(h3d_ctx *ctx, const char *prompt,
                        const h3_params *params,
                        const h3_reference *refs, size_t ref_count);
h3d_job *h3d_job_find(h3d_ctx *ctx, const char *job_id);
int h3d_job_cancel(h3d_ctx *ctx, const char *job_id);
int h3d_job_count_by_status(h3d_ctx *ctx, h3d_job_status status);
void h3d_jobs_mark_all_failed(h3d_ctx *ctx);
void h3d_job_record_event(h3d_job *job, const char *event, const char *data);

#endif
