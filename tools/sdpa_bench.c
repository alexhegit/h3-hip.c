/* Offline SDPA bench on inputs captured with H3_SDPA_DUMP (see
 * backends/h3_gpu_hip.c). Kernel selection comes from the usual H3_SDPA_*
 * environment, so each variant is one process:
 *
 *   ./h3_sdpa_bench DUMP.bin [--iters N] [--write OUT.bin] [--ref REF.bin]
 *
 * --write stores the BF16 output; --ref compares against a stored output
 * (relative L2, cosine, max abs). */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static void die(const char *message) {
    fprintf(stderr, "sdpa_bench: %s\n", message);
    exit(1);
}

static uint16_t *read_bf16(FILE *file, size_t count) {
    uint16_t *data = malloc(count * sizeof(*data));
    if (!data || fread(data, sizeof(*data), count, file) != count)
        die("short read");
    return data;
}

int main(int argc, char **argv) {
    if (argc < 2) die("usage: h3_sdpa_bench DUMP.bin [--iters N] "
                      "[--write OUT.bin] [--ref REF.bin]");
    int iters = 10;
    const char *write_path = NULL, *ref_path = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--write") && i + 1 < argc) write_path = argv[++i];
        else if (!strcmp(argv[i], "--ref") && i + 1 < argc) ref_path = argv[++i];
        else die("unknown argument");
    }

    FILE *file = fopen(argv[1], "rb");
    if (!file) die("cannot open dump");
    uint32_t header[4];
    float scale;
    if (fread(header, sizeof(header), 1, file) != 1 ||
        fread(&scale, sizeof(scale), 1, file) != 1)
        die("bad header");
    uint32_t sequence = header[0], heads = header[1], head_dim = header[2];
    size_t count = (size_t)sequence * heads * head_dim;
    uint16_t *query = read_bf16(file, count);
    uint16_t *key = read_bf16(file, count);
    uint16_t *value = read_bf16(file, count);
    fclose(file);
    /* The dump stores K/V in the layout the kernel read. */
    setenv("H3_SDPA_KV_HM", header[3] ? "1" : "0", 1);

    char error[256];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);
    /* Device memory like the DiT activations: h3_gpu_tensor_from_bf16 is
     * pinned host memory, which the kernel would read over PCIe. */
    h3_gpu_tensor *gq = h3_gpu_tensor_new_bf16_device(gpu, count);
    h3_gpu_tensor *gk = h3_gpu_tensor_new_bf16_device(gpu, count);
    h3_gpu_tensor *gv = h3_gpu_tensor_new_bf16_device(gpu, count);
    h3_gpu_tensor *go = h3_gpu_tensor_new_bf16_device(gpu, count);
    if (!gq || !gk || !gv || !go ||
        !h3_gpu_tensor_write_bf16(gq, query, count) ||
        !h3_gpu_tensor_write_bf16(gk, key, count) ||
        !h3_gpu_tensor_write_bf16(gv, value, count))
        die("tensor allocation failed");

    double best = 1e30, total = 0.0;
    for (int i = -2; i < iters; i++) {
        double start = now_seconds();
        if (!h3_gpu_begin(gpu) ||
            !h3_gpu_sdpa_bf16(gpu, go, gq, gk, gv, sequence, heads, head_dim,
                              scale) ||
            !h3_gpu_submit(gpu))
            die("SDPA failed");
        double elapsed = now_seconds() - start;
        if (i < 0) continue;
        total += elapsed;
        if (elapsed < best) best = elapsed;
    }
    printf("seq=%u heads=%u dim=%u best=%.2f ms mean=%.2f ms\n", sequence,
           heads, head_dim, best * 1e3, total / iters * 1e3);

    uint16_t *out = malloc(count * sizeof(*out));
    if (!out || !h3_gpu_tensor_read_bf16(go, out, count)) die("readback failed");
    if (write_path) {
        FILE *w = fopen(write_path, "wb");
        if (!w || fwrite(out, sizeof(*out), count, w) != count) die("write failed");
        fclose(w);
    }
    if (ref_path) {
        FILE *r = fopen(ref_path, "rb");
        if (!r) die("cannot open ref");
        uint16_t *ref = read_bf16(r, count);
        fclose(r);
        double dot = 0.0, nr = 0.0, ng = 0.0, diff = 0.0, max_abs = 0.0;
        size_t nans = 0;
        for (size_t i = 0; i < count; i++) {
            double a = bf16_to_f32(ref[i]), b = bf16_to_f32(out[i]);
            if (isnan(b)) { nans++; continue; }
            dot += a * b;
            nr += a * a;
            ng += b * b;
            diff += (a - b) * (a - b);
            if (fabs(a - b) > max_abs) max_abs = fabs(a - b);
        }
        printf("vs ref: rel_l2=%.4f%% cos=%.6f max_abs=%.4g nans=%zu\n",
               100.0 * sqrt(diff / (nr > 0 ? nr : 1)),
               dot / sqrt((nr > 0 ? nr : 1) * (ng > 0 ? ng : 1)), max_abs,
               nans);
        free(ref);
    }
    free(out);
    free(query);
    free(key);
    free(value);
    h3_gpu_tensor_free(gq);
    h3_gpu_tensor_free(gk);
    h3_gpu_tensor_free(gv);
    h3_gpu_tensor_free(go);
    h3_gpu_free(gpu);
    return 0;
}
