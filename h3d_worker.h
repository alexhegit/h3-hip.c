#ifndef H3D_WORKER_H
#define H3D_WORKER_H

#include "h3d.h"

int h3d_workers_init(h3d_ctx *ctx);
void *h3d_worker_loop(void *arg);
void h3d_workers_shutdown(h3d_ctx *ctx);

#endif
