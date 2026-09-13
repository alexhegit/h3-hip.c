#ifndef H3D_JSON_H
#define H3D_JSON_H

#include "h3d.h"
#include <stddef.h>

/* Escape a string for JSON. Returns bytes written (not counting NUL). */
int h3d_json_escape(char *dest, size_t dest_len, const char *src);

/* Build /v1/info response. Returns bytes written. */
int h3d_json_info_response(h3d_ctx *ctx, char *buf, size_t len);

/* Build /v1/jobs/{id} response. Returns bytes written. */
int h3d_json_job_response(h3d_job *job, char *buf, size_t len);

/* Build error response. Returns bytes written. */
int h3d_json_error_response(const h3d_error *error, char *buf, size_t len);

/* Build queue full response (429). Returns bytes written. */
int h3d_json_queue_full_response(char *buf, size_t len);

#endif
