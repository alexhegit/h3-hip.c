#include "h3d_http.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

/* Idle reads and stalled writes give up after this, so a client that opens a
 * connection and goes quiet cannot hold a connection thread forever. */
#define H3D_HTTP_IO_TIMEOUT_SEC 30

int h3d_http_write(int fd, const void *buf, size_t len) {
    const char *p = buf;
    size_t done = 0;
    while (done < len) {
        ssize_t n = write(fd, p + done, len - done);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return -1;
        done += (size_t)n;
    }
    return 0;
}

int h3d_http_listen(const char *bind_addr, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (bind_addr && *bind_addr && strcmp(bind_addr, "0.0.0.0") != 0) {
        inet_pton(AF_INET, bind_addr, &addr.sin_addr);
    } else {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    }

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 64) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

int h3d_http_accept(int server_fd) {
    struct sockaddr_in addr;
    socklen_t addr_len = sizeof(addr);
    int fd = accept(server_fd, (struct sockaddr *)&addr, &addr_len);
    if (fd < 0) return -1;

    int opt = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));
    struct timeval timeout = {H3D_HTTP_IO_TIMEOUT_SEC, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    return fd;
}

/* Read a line from fd (up to \n). Returns its length without the line
 * terminator, or -1 on error, timeout, EOF, or a line longer than cap. */
static int read_line(int fd, char *buf, size_t cap) {
    size_t n = 0;
    while (n < cap - 1) {
        char c;
        ssize_t r = read(fd, &c, 1);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        if (c == '\n') {
            if (n > 0 && buf[n - 1] == '\r') n--;
            buf[n] = '\0';
            return (int)n;
        }
        buf[n++] = c;
    }
    return -1;
}

static int read_exact(int fd, char *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t r = read(fd, buf + total, len - total);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return -1;
        total += (size_t)r;
    }
    return 0;
}

int h3d_http_read_request(int fd,
                          char *method, size_t method_len,
                          char *path, size_t path_len,
                          char *headers, size_t headers_len,
                          char *body, size_t body_len,
                          size_t *body_out_len) {
    char line[4096];

    /* Request line: METHOD PATH HTTP/1.x */
    if (read_line(fd, line, sizeof(line)) <= 0) return -1;
    char *sp1 = strchr(line, ' ');
    if (!sp1) return -1;
    *sp1 = '\0';
    char *sp2 = strchr(sp1 + 1, ' ');
    if (!sp2) return -1;
    *sp2 = '\0';

    snprintf(method, method_len, "%s", line);
    snprintf(path, path_len, "%s", sp1 + 1);

    size_t hpos = 0;
    long content_length = 0;
    headers[0] = '\0';
    while (1) {
        int len = read_line(fd, line, sizeof(line));
        if (len < 0) return -1;
        if (len == 0) break;
        if (hpos + (size_t)len + 2 < headers_len) {
            memcpy(headers + hpos, line, (size_t)len);
            hpos += (size_t)len;
            headers[hpos++] = '\r';
            headers[hpos++] = '\n';
            headers[hpos] = '\0';
        }
        if (strncasecmp(line, "Content-Length:", 15) == 0) {
            content_length = strtol(line + 15, NULL, 10);
        }
    }

    *body_out_len = 0;
    body[0] = '\0';
    if (content_length < 0) return -1;
    if (content_length > 0) {
        if ((size_t)content_length >= body_len) return H3D_HTTP_TOO_LARGE;
        if (read_exact(fd, body, (size_t)content_length) < 0) return -1;
        *body_out_len = (size_t)content_length;
        body[content_length] = '\0';
    }
    return 0;
}

void h3d_http_send_response(int fd, int status, const char *status_text,
                            const char *content_type,
                            const char *body, size_t body_len) {
    char header[1024];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text, content_type, body_len);
    if (h3d_http_write(fd, header, (size_t)hlen) == 0 && body_len > 0)
        h3d_http_write(fd, body, body_len);
}

int h3d_http_send_sse_headers(int fd) {
    static const char header[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "Transfer-Encoding: chunked\r\n"
        "\r\n";
    return h3d_http_write(fd, header, sizeof(header) - 1);
}

int h3d_http_send_chunk(int fd, const char *data, size_t len) {
    if (len == 0) return 0;
    char prefix[32];
    int plen = snprintf(prefix, sizeof(prefix), "%zx\r\n", len);
    if (h3d_http_write(fd, prefix, (size_t)plen) < 0) return -1;
    if (h3d_http_write(fd, data, len) < 0) return -1;
    return h3d_http_write(fd, "\r\n", 2);
}

int h3d_http_end_chunks(int fd) {
    return h3d_http_write(fd, "0\r\n\r\n", 5);
}

int h3d_http_send_sse_event(int fd, const char *event, const char *data) {
    char buf[4096];
    int len = snprintf(buf, sizeof(buf), "event: %s\ndata: %s\n\n", event, data);
    if (len < 0) return -1;
    if ((size_t)len >= sizeof(buf)) len = (int)sizeof(buf) - 1;
    return h3d_http_send_chunk(fd, buf, (size_t)len);
}

int h3d_http_send_sse_heartbeat(int fd) {
    return h3d_http_send_sse_event(fd, "ping", "{}");
}

void h3d_http_close(int fd) {
    if (fd >= 0) close(fd);
}
