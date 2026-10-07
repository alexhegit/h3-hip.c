#include "h3_taeh3.h"

#include "h3_safetensors.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef H3_HIP

int h3_taeh3_decode(const char *checkpoint, const float *latent_cthw,
                    int latent_t, int latent_h, int latent_w,
                    h3_video_frames *output, char *error, size_t error_size) {
    (void)checkpoint;
    (void)latent_cthw;
    (void)latent_t;
    (void)latent_h;
    (void)latent_w;
    (void)output;
    if (error && error_size)
        snprintf(error, error_size, "TAEH3 requires the HIP backend");
    return 0;
}

#else

#include "h3_gpu.h"

enum { TAEH3_CHUNK = 5 };

typedef struct {
    char name[64];
    h3_gpu_tensor *weight;
    h3_gpu_tensor *bias;
} tae_layer;

typedef struct {
    uint32_t channels;
    uint32_t height;
    uint32_t width;
    h3_gpu_tensor *frame;
} tae_memory;

static void tae_fail(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message);
}

static float tae_bf16_to_f32(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

static float tae_f16_to_f32(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    uint32_t exp = (bits >> 10) & 0x1fu;
    uint32_t frac = bits & 0x3ffu;
    uint32_t wide;
    if (exp == 0) {
        if (!frac) {
            wide = sign;
        } else {
            exp = 127 - 14;
            while ((frac & 0x400u) == 0) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x3ffu;
            wide = sign | (exp << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        wide = sign | 0x7f800000u | (frac << 13);
    } else {
        wide = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
    }
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

static h3_gpu_tensor *tae_load(h3_gpu *gpu, const h3_st_header *header,
                               const char *name, int expect_bias,
                               h3_gpu_tensor **bias, char *error,
                               size_t error_size) {
    const h3_st_tensor *tensor = h3_st_find(header, name);
    char key[96];
    if (!tensor || tensor->ndim != 4) {
        tae_fail(error, error_size, "TAEH3 weight is missing");
        return NULL;
    }
    uint64_t n = h3_st_tensor_elements(tensor);
    if (!n || n > 64000000ull) {
        tae_fail(error, error_size, "TAEH3 weight has an unexpected size");
        return NULL;
    }
    void *raw = malloc((size_t)n * h3_dtype_size(tensor->dtype));
    float *f32 = malloc((size_t)n * sizeof(float));
    if (!raw || !f32 ||
        !h3_st_read_data(header, tensor, raw, (size_t)n * h3_dtype_size(tensor->dtype),
                         error, error_size)) {
        free(raw);
        free(f32);
        if (error && error_size && !error[0])
            tae_fail(error, error_size, "cannot read TAEH3 weight");
        return NULL;
    }
    if (tensor->dtype == H3_DTYPE_F32) {
        memcpy(f32, raw, (size_t)n * sizeof(float));
    } else if (tensor->dtype == H3_DTYPE_BF16) {
        const uint16_t *src = raw;
        for (uint64_t i = 0; i < n; i++) f32[i] = tae_bf16_to_f32(src[i]);
    } else if (tensor->dtype == H3_DTYPE_F16) {
        const uint16_t *src = raw;
        for (uint64_t i = 0; i < n; i++) f32[i] = tae_f16_to_f32(src[i]);
    } else {
        free(raw);
        free(f32);
        tae_fail(error, error_size, "TAEH3 weight dtype is not f32, bf16, or f16");
        return NULL;
    }
    free(raw);
    h3_gpu_tensor *weight = h3_gpu_tensor_from_f32(gpu, f32, (size_t)n);
    free(f32);
    if (!weight) {
        tae_fail(error, error_size, "cannot upload TAEH3 weight");
        return NULL;
    }
    *bias = NULL;
    if (!expect_bias) return weight;
    snprintf(key, sizeof(key), "%s.bias", name);
    /* name already includes .weight; bias replaces that suffix. */
    size_t len = strlen(name);
    if (len < 7 || strcmp(name + len - 7, ".weight") != 0) {
        h3_gpu_tensor_free(weight);
        tae_fail(error, error_size, "TAEH3 weight name has no .weight suffix");
        return NULL;
    }
    snprintf(key, sizeof(key), "%.*s.bias", (int)(len - 7), name);
    const h3_st_tensor *bias_tensor = h3_st_find(header, key);
    if (!bias_tensor || bias_tensor->ndim != 1) {
        h3_gpu_tensor_free(weight);
        tae_fail(error, error_size, "TAEH3 bias is missing");
        return NULL;
    }
    uint64_t bn = h3_st_tensor_elements(bias_tensor);
    void *braw = malloc((size_t)bn * h3_dtype_size(bias_tensor->dtype));
    float *bf = malloc((size_t)bn * sizeof(float));
    if (!braw || !bf ||
        !h3_st_read_data(header, bias_tensor, braw,
                         (size_t)bn * h3_dtype_size(bias_tensor->dtype),
                         error, error_size)) {
        free(braw);
        free(bf);
        h3_gpu_tensor_free(weight);
        return NULL;
    }
    if (bias_tensor->dtype == H3_DTYPE_F32) {
        memcpy(bf, braw, (size_t)bn * sizeof(float));
    } else if (bias_tensor->dtype == H3_DTYPE_BF16) {
        const uint16_t *src = braw;
        for (uint64_t i = 0; i < bn; i++) bf[i] = tae_bf16_to_f32(src[i]);
    } else if (bias_tensor->dtype == H3_DTYPE_F16) {
        const uint16_t *src = braw;
        for (uint64_t i = 0; i < bn; i++) bf[i] = tae_f16_to_f32(src[i]);
    } else {
        free(braw);
        free(bf);
        h3_gpu_tensor_free(weight);
        tae_fail(error, error_size, "TAEH3 bias dtype is not f32, bf16, or f16");
        return NULL;
    }
    free(braw);
    *bias = h3_gpu_tensor_from_f32(gpu, bf, (size_t)bn);
    free(bf);
    if (!*bias) {
        h3_gpu_tensor_free(weight);
        tae_fail(error, error_size, "cannot upload TAEH3 bias");
        return NULL;
    }
    return weight;
}

static int tae_elems_ok(uint32_t batch, uint32_t channels, uint32_t height,
                        uint32_t width, uint32_t scale, uint32_t *out) {
    uint64_t n = (uint64_t)batch * channels * height * width * scale;
    if (!n || n > 0xffffffffull) return 0;
    *out = (uint32_t)n;
    return 1;
}

static int tae_conv(h3_gpu *gpu, h3_gpu_tensor *out, h3_gpu_tensor *in,
                    h3_gpu_tensor *weight, h3_gpu_tensor *bias, uint32_t batch,
                    uint32_t in_c, uint32_t out_c, uint32_t height,
                    uint32_t width, uint32_t kernel) {
    return h3_gpu_taeh3_conv2d(gpu, out, in, weight, bias, batch, in_c, out_c,
                               height, width, kernel);
}

static int tae_memblock(h3_gpu *gpu, h3_gpu_tensor *x, h3_gpu_tensor *work,
                        h3_gpu_tensor *wide, h3_gpu_tensor *weight0,
                        h3_gpu_tensor *bias0, h3_gpu_tensor *weight2,
                        h3_gpu_tensor *bias2, h3_gpu_tensor *weight4,
                        h3_gpu_tensor *bias4, tae_memory *memory, uint32_t t,
                        uint32_t channels, uint32_t height, uint32_t width) {
    uint32_t count = 0;
    if (!tae_elems_ok(t, channels, height, width, 1u, &count)) return 0;
    if (!memory->frame) {
        memory->frame = h3_gpu_tensor_new_f32_device(
            gpu, (size_t)channels * height * width);
        memory->channels = channels;
        memory->height = height;
        memory->width = width;
        if (!memory->frame) return 0;
    }
    /* wide holds the 2C concat, then is reused as a C-channel buffer.
     * work holds the previous-frame tensor, then the conv outputs.
     * x stays the block input until the residual add copies the result back. */
    if (!h3_gpu_taeh3_time_past(gpu, work, x, memory->frame, t, channels,
                                height, width) ||
        !h3_gpu_taeh3_copy_frame(gpu, memory->frame, x, t - 1u, channels,
                                 height, width) ||
        !h3_gpu_taeh3_cat_channels(gpu, wide, x, work, t, channels, height,
                                   width) ||
        !tae_conv(gpu, work, wide, weight0, bias0, t, channels * 2u, channels,
                  height, width, 3u) ||
        !h3_gpu_taeh3_relu(gpu, work, work, count) ||
        !tae_conv(gpu, wide, work, weight2, bias2, t, channels, channels,
                  height, width, 3u) ||
        !h3_gpu_taeh3_relu(gpu, wide, wide, count) ||
        !tae_conv(gpu, work, wide, weight4, bias4, t, channels, channels,
                  height, width, 3u) ||
        !h3_gpu_taeh3_add_relu(gpu, wide, x, work, count) ||
        !h3_gpu_taeh3_relu(gpu, x, wide, count)) {
        return 0;
    }
    return 1;
}

static int tae_kept(int raw_index) {
    return raw_index % 20 >= 3;
}

int h3_taeh3_decode(const char *checkpoint, const float *latent_cthw,
                    int latent_t, int latent_h, int latent_w,
                    h3_video_frames *output, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!checkpoint || !checkpoint[0] || !latent_cthw || !output ||
        latent_t < 2 || latent_t % 5 != 2 || latent_h < 1 || latent_w < 1) {
        tae_fail(error, error_size, "TAEH3 latent shape is not an H3 video");
        return 0;
    }
    struct timespec begun, stop;
    clock_gettime(CLOCK_MONOTONIC, &begun);
    h3_st_header header;
    memset(&header, 0, sizeof(header));
    h3_gpu *gpu = NULL;
    tae_layer layers[40];
    int layer_count = 0;
    tae_memory memory[9];
    h3_gpu_tensor *buf[3] = {NULL, NULL, NULL};
    float *packed = NULL;
    float *rgb = NULL;
    int ok = 0;
    memset(layers, 0, sizeof(layers));
    memset(memory, 0, sizeof(memory));
    memset(output, 0, sizeof(*output));
    if (!h3_st_read_header(checkpoint, &header, error, error_size)) return 0;
    gpu = h3_gpu_create("h3_shaders.metal", error, error_size);
    if (!gpu) goto done;

    static const struct {
        const char *name;
        int bias;
    } specs[] = {
        {"decoder.1.weight", 1},
        {"decoder.22.weight", 1},
        {"decoder.7.conv.weight", 0},
        {"decoder.8.weight", 0},
        {"decoder.13.conv.weight", 0},
        {"decoder.14.weight", 0},
        {"decoder.19.conv.weight", 0},
        {"decoder.20.weight", 0},
    };
    for (size_t i = 0; i < sizeof(specs) / sizeof(specs[0]); i++) {
        if (layer_count >= 40) goto done;
        layers[layer_count].weight = tae_load(
            gpu, &header, specs[i].name, specs[i].bias,
            &layers[layer_count].bias, error, error_size);
        if (!layers[layer_count].weight) goto done;
        snprintf(layers[layer_count].name, sizeof(layers[layer_count].name),
                 "%s", specs[i].name);
        layer_count++;
    }
    for (int block = 0; block < 9; block++) {
        static const int ids[] = {3, 4, 5, 9, 10, 11, 15, 16, 17};
        for (int conv = 0; conv < 3; conv++) {
            char name[64];
            snprintf(name, sizeof(name), "decoder.%d.conv.%d.weight", ids[block],
                     conv * 2);
            if (layer_count >= 40) goto done;
            layers[layer_count].weight = tae_load(
                gpu, &header, name, 1, &layers[layer_count].bias, error,
                error_size);
            if (!layers[layer_count].weight) goto done;
            snprintf(layers[layer_count].name, sizeof(layers[layer_count].name),
                     "%s", name);
            layer_count++;
        }
    }

    uint32_t ht = (uint32_t)latent_h;
    uint32_t wt = (uint32_t)latent_w;
    /* Largest activation is the stride-2 TGrow output: T*2 frames, 128
     * channels, spatially x8. */
    uint64_t cap64 = (uint64_t)TAEH3_CHUNK * 2ull * 128ull * (uint64_t)ht * 8ull *
                     (uint64_t)wt * 8ull;
    if (cap64 > 0xffffffffull) {
        tae_fail(error, error_size, "TAEH3 frame is too large");
        goto done;
    }
    for (int i = 0; i < 3; i++) {
        buf[i] = h3_gpu_tensor_new_f32_device(gpu, (size_t)cap64);
        if (!buf[i]) {
            tae_fail(error, error_size, "out of memory allocating TAEH3");
            goto done;
        }
    }
    int pixel_h = latent_h * 16;
    int pixel_w = latent_w * 16;
    int raw_frames = latent_t * 4;
    int kept = 0;
    for (int i = 0; i < raw_frames; i++) kept += tae_kept(i);
    rgb = malloc((size_t)kept * (size_t)pixel_h * (size_t)pixel_w * 3u *
                 sizeof(float));
    if (!rgb) {
        tae_fail(error, error_size, "out of memory allocating TAEH3 frames");
        goto done;
    }
    int written = 0;
    for (int start = 0; start < latent_t; start += TAEH3_CHUNK) {
        int chunk = latent_t - start;
        if (chunk > TAEH3_CHUNK) chunk = TAEH3_CHUNK;
        uint32_t t = (uint32_t)chunk;
        uint32_t c = 24u;
        uint32_t h = ht;
        uint32_t w = wt;
        size_t chunk_n = (size_t)t * c * h * w;
        packed = malloc(chunk_n * sizeof(float));
        if (!packed) {
            tae_fail(error, error_size, "out of memory packing TAEH3 latents");
            goto done;
        }
        for (uint32_t ti = 0; ti < t; ti++) {
            for (uint32_t ci = 0; ci < c; ci++) {
                const float *src =
                    latent_cthw +
                    (((size_t)ci * (size_t)latent_t + (size_t)start + ti) *
                         h +
                     0) *
                        w;
                float *dst = packed + ((size_t)ti * c + ci) * h * w;
                memcpy(dst, src, (size_t)h * (size_t)w * sizeof(float));
            }
        }
        h3_gpu_tensor *upload = h3_gpu_tensor_from_f32(gpu, packed, chunk_n);
        free(packed);
        packed = NULL;
        if (!upload) {
            tae_fail(error, error_size, "cannot upload TAEH3 latents");
            goto done;
        }
        if (!h3_gpu_begin(gpu)) {
            h3_gpu_tensor_free(upload);
            tae_fail(error, error_size, "cannot begin TAEH3");
            goto done;
        }
        h3_gpu_tensor *x = buf[0];
        h3_gpu_tensor *spare = buf[1];
        uint32_t count = 0;
        if (!tae_elems_ok(t, c, h, w, 1u, &count) ||
            !h3_gpu_taeh3_tanh_clamp(gpu, x, upload, count)) {
            h3_gpu_tensor_free(upload);
            goto done;
        }
        h3_gpu_tensor_free(upload);
        tae_layer *stem = NULL;
        tae_layer *head = NULL;
        for (int i = 0; i < layer_count; i++) {
            if (!strcmp(layers[i].name, "decoder.1.weight")) stem = &layers[i];
            if (!strcmp(layers[i].name, "decoder.22.weight")) head = &layers[i];
        }
        if (!stem || !head ||
            !tae_conv(gpu, spare, x, stem->weight, stem->bias, t, 24u, 256u, h,
                      w, 3u) ||
            !tae_elems_ok(t, 256u, h, w, 1u, &count) ||
            !h3_gpu_taeh3_relu(gpu, x, spare, count)) {
            goto done;
        }
        c = 256u;
        static const int block_ids[] = {3, 4, 5, 9, 10, 11, 15, 16, 17};
        static const int stage_blocks[] = {0, 3, 6};
        static const int stage_channels[] = {256, 128, 64};
        static const int stage_stride[] = {1, 2, 2};
        static const char *stage_grow[] = {
            "decoder.7.conv.weight", "decoder.13.conv.weight",
            "decoder.19.conv.weight"};
        static const char *stage_proj[] = {
            "decoder.8.weight", "decoder.14.weight", "decoder.20.weight"};
        static const int stage_proj_out[] = {128, 64, 64};
        for (int stage = 0; stage < 3; stage++) {
            uint32_t stage_c = (uint32_t)stage_channels[stage];
            if (c != stage_c) goto done;
            for (int b = 0; b < 3; b++) {
                int id = block_ids[stage_blocks[stage] + b];
                char n0[64], n2[64], n4[64];
                snprintf(n0, sizeof(n0), "decoder.%d.conv.0.weight", id);
                snprintf(n2, sizeof(n2), "decoder.%d.conv.2.weight", id);
                snprintf(n4, sizeof(n4), "decoder.%d.conv.4.weight", id);
                tae_layer *l0 = NULL, *l2 = NULL, *l4 = NULL;
                for (int i = 0; i < layer_count; i++) {
                    if (!strcmp(layers[i].name, n0)) l0 = &layers[i];
                    if (!strcmp(layers[i].name, n2)) l2 = &layers[i];
                    if (!strcmp(layers[i].name, n4)) l4 = &layers[i];
                }
                if (!l0 || !l2 || !l4) goto done;
                tae_memory *slot = &memory[stage_blocks[stage] + b];
                if (x != buf[0]) goto done;
                if (!tae_memblock(gpu, x, buf[1], buf[2], l0->weight, l0->bias,
                                  l2->weight, l2->bias, l4->weight, l4->bias,
                                  slot, t, c, h, w)) {
                    goto done;
                }
            }
            if (!h3_gpu_taeh3_upsample2(gpu, buf[1], x, t, c, h, w)) goto done;
            h *= 2u;
            w *= 2u;
            x = buf[1];
            uint32_t stride = (uint32_t)stage_stride[stage];
            tae_layer *grow = NULL;
            tae_layer *proj = NULL;
            for (int i = 0; i < layer_count; i++) {
                if (!strcmp(layers[i].name, stage_grow[stage])) grow = &layers[i];
                if (!strcmp(layers[i].name, stage_proj[stage])) proj = &layers[i];
            }
            if (!grow || !proj) goto done;
            if (!tae_conv(gpu, buf[0], x, grow->weight, grow->bias, t, c,
                          c * stride, h, w, 1u)) {
                goto done;
            }
            t *= stride;
            x = buf[0];
            uint32_t next_c = (uint32_t)stage_proj_out[stage];
            if (!tae_conv(gpu, buf[1], x, proj->weight, proj->bias, t, c,
                          next_c, h, w, 3u)) {
                goto done;
            }
            c = next_c;
            uint32_t moved = 0;
            if (!tae_elems_ok(t, c, h, w, 1u, &moved) ||
                !h3_gpu_taeh3_copy(gpu, buf[0], buf[1], moved)) {
                goto done;
            }
            x = buf[0];
        }
        uint32_t pre = 0;
        if (c != 64u || !tae_elems_ok(t, c, h, w, 1u, &pre) ||
            !h3_gpu_taeh3_relu(gpu, buf[1], x, pre) ||
            !tae_conv(gpu, buf[2], buf[1], head->weight, head->bias, t, 64u,
                      12u, h, w, 3u)) {
            goto done;
        }
        h3_gpu_tensor *shuffled = buf[0];
        if (!h3_gpu_taeh3_pixel_shuffle2(gpu, shuffled, buf[2], t, h, w) ||
            !h3_gpu_submit(gpu)) {
            goto done;
        }
        uint32_t oh = h * 2u;
        uint32_t ow = w * 2u;
        size_t frame_n = (size_t)3u * oh * ow;
        float *host = malloc((size_t)t * frame_n * sizeof(float));
        if (!host ||
            !h3_gpu_tensor_read_f32(shuffled, host, (size_t)t * frame_n)) {
            free(host);
            tae_fail(error, error_size, "cannot read TAEH3 frames");
            goto done;
        }
        for (uint32_t ti = 0; ti < t; ti++) {
            int raw = start * 4 + (int)ti;
            if (!tae_kept(raw)) continue;
            float *dst = rgb + (size_t)written * (size_t)pixel_h *
                                     (size_t)pixel_w * 3u;
            const float *src = host + (size_t)ti * frame_n;
            for (uint32_t y = 0; y < oh; y++) {
                for (uint32_t px = 0; px < ow; px++) {
                    for (uint32_t ch = 0; ch < 3u; ch++) {
                        dst[((size_t)y * ow + px) * 3u + ch] =
                            src[((size_t)ch * oh + y) * ow + px];
                    }
                }
            }
            written++;
        }
        free(host);
        if (oh != (uint32_t)pixel_h || ow != (uint32_t)pixel_w) goto done;
    }
    if (written != kept) {
        tae_fail(error, error_size, "TAEH3 dropped an unexpected frame count");
        goto done;
    }
    output->frames = kept;
    output->height = pixel_h;
    output->width = pixel_w;
    output->rgb = rgb;
    rgb = NULL;
    ok = 1;
    clock_gettime(CLOCK_MONOTONIC, &stop);
    double wall = (double)(stop.tv_sec - begun.tv_sec) +
                  (double)(stop.tv_nsec - begun.tv_nsec) / 1.0e9;
    fprintf(stderr,
            "h3: TAEH3 decoded %d frames from %d x %d x %d latents in %.3fs\n",
            kept, latent_t, latent_h, latent_w, wall);

done:
    if (!ok && error && error_size && !error[0])
        tae_fail(error, error_size, "TAEH3 decode failed");
    free(packed);
    if (!ok) free(rgb);
    for (int i = 0; i < 3; i++) h3_gpu_tensor_free(buf[i]);
    for (int i = 0; i < 9; i++) h3_gpu_tensor_free(memory[i].frame);
    for (int i = 0; i < layer_count; i++) {
        h3_gpu_tensor_free(layers[i].weight);
        h3_gpu_tensor_free(layers[i].bias);
    }
    h3_gpu_free(gpu);
    h3_st_free_header(&header);
    return ok;
}

#endif
