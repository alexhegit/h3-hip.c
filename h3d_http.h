#ifndef H3D_HTTP_H
#define H3D_HTTP_H

#include <stddef.h>

/* h3d_http_read_request result when Content-Length exceeds the body buffer. */
#define H3D_HTTP_TOO_LARGE (-2)

int h3d_http_listen(const char *bind, int port);
int h3d_http_accept(int server_fd);

/* Read a complete HTTP request. Returns 0 on success, H3D_HTTP_TOO_LARGE for
 * an oversized body, -1 otherwise. Buffers are caller-allocated. */
int h3d_http_read_request(int fd,
                          char *method, size_t method_len,
                          char *path, size_t path_len,
                          char *headers, size_t headers_len,
                          char *body, size_t body_len,
                          size_t *body_out_len);

/* Write all bytes. Returns 0, or -1 once the peer is gone. */
int h3d_http_write(int fd, const void *buf, size_t len);

/* Send a complete HTTP response. Every response closes the connection. */
void h3d_http_send_response(int fd, int status, const char *status_text,
                            const char *content_type,
                            const char *body, size_t body_len);

/* Streaming SSE response: headers, then chunks, then h3d_http_end_chunks.
 * All return 0, or -1 once the peer is gone. */
int h3d_http_send_sse_headers(int fd);
int h3d_http_send_chunk(int fd, const char *data, size_t len);
int h3d_http_end_chunks(int fd);
int h3d_http_send_sse_event(int fd, const char *event, const char *data);
int h3d_http_send_sse_heartbeat(int fd);

void h3d_http_close(int fd);

#endif
