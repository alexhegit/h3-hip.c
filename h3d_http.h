#ifndef H3D_HTTP_H
#define H3D_HTTP_H

#include <stddef.h>

int h3d_http_listen(const char *bind, int port);
int h3d_http_accept(int server_fd);

/* Read a complete HTTP request. Returns 0 on success.
 * method/path/headers/body are caller-allocated buffers. */
int h3d_http_read_request(int fd,
                          char *method, size_t method_len,
                          char *path, size_t path_len,
                          char *headers, size_t headers_len,
                          char *body, size_t body_len,
                          size_t *body_out_len);

/* Send a complete HTTP response. */
void h3d_http_send_response(int fd, int status, const char *status_text,
                            const char *content_type,
                            const char *body, size_t body_len);

/* Send HTTP headers (for streaming responses). */
void h3d_http_send_headers(int fd, int status, const char *status_text,
                           const char *extra_headers);

/* Send a chunked transfer-encoding chunk. */
void h3d_http_send_chunk(int fd, const char *data, size_t len);

/* Send an SSE event line. */
void h3d_http_send_sse_event(int fd, const char *event, const char *data);

/* Send SSE heartbeat comment. */
void h3d_http_send_sse_heartbeat(int fd);

/* Close connection. */
void h3d_http_close(int fd);

#endif
