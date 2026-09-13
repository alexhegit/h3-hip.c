#include "h3d_http.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

/* Ignore write() return value — best-effort send on a streaming socket. */
static void write_all(int fd, const void *buf, size_t len) {
    const char *p = buf;
    size_t done = 0;
    while (done < len) {
        ssize_t n = write(fd, p + done, len - done);
        if (n <= 0) break;
        done += (size_t)n;
    }
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
    if (listen(fd, 16) < 0) {
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
    return fd;
}

/* Read a line from fd (up to \r\n). Returns bytes read, 0 on EOF, -1 on error.
 * The line terminator is NOT included in buf. */
static int read_line(int fd, char *buf, size_t cap) {
    size_t n = 0;
    while (n < cap - 1) {
        char c;
        ssize_t r = read(fd, &c, 1);
        if (r < 0) { fprintf(stderr, "h3d: read_line: read error: %s\n", strerror(errno)); return -1; }
        if (r == 0) { fprintf(stderr, "h3d: read_line: EOF at %zu bytes\n", n); return (int)n > 0 ? (int)n : -1; }
        if (c == '\n') {
            /* Strip trailing \r */
            if (n > 0 && buf[n - 1] == '\r') n--;
            buf[n] = '\0';
            return (int)n;
        }
        buf[n++] = c;
    }
    buf[n] = '\0';
    return (int)n;
}

/* Read exactly len bytes from fd. Returns 0 on success, -1 on error/EOF. */
static int read_exact(int fd, char *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t r = read(fd, buf + total, len - total);
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
    int rlen = read_line(fd, line, sizeof(line));
    fprintf(stderr, "h3d: read_request: first line read=%d\n", rlen);
    if (rlen <= 0) return -1;
    char *sp1 = strchr(line, ' ');
    if (!sp1) return -1;
    *sp1 = '\0';
    char *sp2 = strchr(sp1 + 1, ' ');
    if (!sp2) return -1;
    *sp2 = '\0';

    snprintf(method, method_len, "%s", line);
    snprintf(path, path_len, "%s", sp1 + 1);

    /* Headers */
    size_t hpos = 0;
    int content_length = 0;
    headers[0] = '\0';
    while (1) {
        int len = read_line(fd, line, sizeof(line));
        if (len < 0) return -1;
        if (len == 0 && line[0] == '\0') break; /* Empty line = end of headers */
        if (hpos + (size_t)len + 2 < headers_len) {
            memcpy(headers + hpos, line, (size_t)len);
            hpos += (size_t)len;
            headers[hpos++] = '\r';
            headers[hpos++] = '\n';
            headers[hpos] = '\0';
        }
        /* Parse Content-Length */
        if (strncasecmp(line, "Content-Length:", 15) == 0) {
            content_length = atoi(line + 15);
        }
    }

    /* Body */
    *body_out_len = 0;
    if (content_length > 0) {
        if ((size_t)content_length > body_len) return -1;
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
        "Connection: keep-alive\r\n"
        "\r\n",
        status, status_text, content_type, body_len);
    write_all(fd, header, (size_t)hlen);
    if (body_len > 0) write_all(fd, body, body_len);
}

void h3d_http_send_headers(int fd, int status, const char *status_text,
                           const char *extra_headers) {
    char header[2048];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Transfer-Encoding: chunked\r\n"
        "%s"
        "\r\n",
        status, status_text,
        extra_headers ? extra_headers : "");
    write_all(fd, header, (size_t)hlen);
}

void h3d_http_send_chunk(int fd, const char *data, size_t len) {
    char prefix[32];
    int plen = snprintf(prefix, sizeof(prefix), "%zx\r\n", len);
    write_all(fd, prefix, (size_t)plen);
    if (len > 0) write_all(fd, data, len);
    write_all(fd, "\r\n", 2);
}

void h3d_http_send_sse_event(int fd, const char *event, const char *data) {
    char buf[4096];
    int len = snprintf(buf, sizeof(buf), "event: %s\ndata: %s\n\n", event, data);
    /* Wrap in chunked encoding */
    char chunk_prefix[32];
    int plen = snprintf(chunk_prefix, sizeof(chunk_prefix), "%zx\r\n", (size_t)len);
    write_all(fd, chunk_prefix, (size_t)plen);
    write_all(fd, buf, (size_t)len);
    write_all(fd, "\r\n", 2);
}

void h3d_http_send_sse_heartbeat(int fd) {
    h3d_http_send_sse_event(fd, "ping", "{}");
}

void h3d_http_close(int fd) {
    if (fd >= 0) close(fd);
}
