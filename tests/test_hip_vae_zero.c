#include "h3_video_vae.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { LATENT_CHANNELS = 24 };

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] : "/home/amd/HF-MODELS/MiniMax-H3";
    int latent_t = argc > 2 ? atoi(argv[2]) : 7;
    int latent_h = argc > 3 ? atoi(argv[3]) : 32;
    int latent_w = argc > 4 ? atoi(argv[4]) : 32;
    if (latent_t < 1 || latent_h < 1 || latent_w < 1) return 1;
    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/video_vae/source", model_root);
    size_t latent_count = (size_t)LATENT_CHANNELS * (size_t)latent_t *
                          (size_t)latent_h * (size_t)latent_w;
    float *latent = calloc(latent_count, sizeof(*latent));
    if (!latent) return 1;
    const char *pattern = argc > 5 ? argv[5] : "zero";
    if (strcmp(pattern, "split") == 0) {
        for (int c = 0; c < LATENT_CHANNELS; c++)
            for (int t = 0; t < latent_t; t++)
                for (int y = 0; y < latent_h; y++)
                    for (int x = 0; x < latent_w; x++) {
                        size_t index = (((size_t)c * (size_t)latent_t +
                            (size_t)t) * (size_t)latent_h + (size_t)y) *
                            (size_t)latent_w + (size_t)x;
                        latent[index] = x < latent_w / 2 ? 1.0f : -1.0f;
                    }
    } else if (strcmp(pattern, "gradient") == 0) {
        for (int c = 0; c < LATENT_CHANNELS; c++)
            for (int t = 0; t < latent_t; t++)
                for (int y = 0; y < latent_h; y++)
                    for (int x = 0; x < latent_w; x++) {
                        size_t index = (((size_t)c * (size_t)latent_t +
                            (size_t)t) * (size_t)latent_h + (size_t)y) *
                            (size_t)latent_w + (size_t)x;
                        latent[index] = ((float)x / (float)(latent_w - 1) -
                                         0.5f) * 0.5f;
                    }
    } else if (strcmp(pattern, "file") == 0) {
        const char *path = "/tmp/dit_latent.bin";
        FILE *file = fopen(path, "rb");
        if (!file) {
            fprintf(stderr, "cannot open %s\n", path);
            free(latent);
            return 1;
        }
        int header[4];
        if (fread(header, sizeof(header), 1, file) != 1 ||
            header[0] != LATENT_CHANNELS || header[1] != latent_t ||
            header[2] != latent_h || header[3] != latent_w ||
            fread(latent, sizeof(*latent), latent_count, file) !=
                latent_count) {
            fprintf(stderr, "invalid latent dump %s\n", path);
            fclose(file);
            free(latent);
            return 1;
        }
        fclose(file);
    }
    char error[512];
    h3_video_frames got = {0};
    if (!h3_video_vae_decode(weights, "h3_shaders.metal", latent, latent_t,
                             latent_h, latent_w, NULL, NULL, &got, error,
                             sizeof(error))) {
        fprintf(stderr, "decode failed: %s\n", error);
        free(latent);
        return 1;
    }
    free(latent);
    size_t pixels = (size_t)got.frames * (size_t)got.height * (size_t)got.width;
    double sum[3] = {0}, mn[3] = {1e30, 1e30, 1e30}, mx[3] = {-1e30, -1e30, -1e30};
    int nan_count = 0;
    for (size_t i = 0; i < pixels; i++) {
        for (int c = 0; c < 3; c++) {
            float v = got.rgb[i * 3 + (size_t)c];
            if (v != v) nan_count++;
            sum[c] += (double)v;
            if (v < mn[c]) mn[c] = v;
            if (v > mx[c]) mx[c] = v;
        }
    }
    printf("shape %dx%dx%d pixels %zu nan %d\n", got.frames, got.height,
           got.width, pixels, nan_count);
    printf("R mean/min/max %.4f %.4f %.4f\n", sum[0] / (double)pixels, mn[0], mx[0]);
    printf("G mean/min/max %.4f %.4f %.4f\n", sum[1] / (double)pixels, mn[1], mx[1]);
    printf("B mean/min/max %.4f %.4f %.4f\n", sum[2] / (double)pixels, mn[2], mx[2]);
    const char *ppm_path = "/tmp/vae_zero.ppm";
    if (strcmp(pattern, "split") == 0) ppm_path = "/tmp/vae_split.ppm";
    else if (strcmp(pattern, "gradient") == 0) ppm_path = "/tmp/vae_gradient.ppm";
    else if (strcmp(pattern, "file") == 0) ppm_path = "/tmp/vae_ditfile.ppm";
    FILE *ppm = fopen(ppm_path, "wb");
    if (!ppm) return 1;
    fprintf(ppm, "P6\n%d %d\n255\n", got.width, got.height);
    for (int y = 0; y < got.height; y++) {
        for (int x = 0; x < got.width; x++) {
            size_t i = ((size_t)y * (size_t)got.width + (size_t)x) * 3;
            unsigned char rgb[3];
            for (int c = 0; c < 3; c++) {
                float scaled = got.rgb[i + (size_t)c] * 255.0f;
                if (scaled < 0.0f) scaled = 0.0f;
                if (scaled > 255.0f) scaled = 255.0f;
                rgb[c] = (unsigned char)(scaled + 0.5f);
            }
            fwrite(rgb, 1, 3, ppm);
        }
    }
    fclose(ppm);
    h3_video_frames_free(&got);
    printf("wrote %s (first frame)\n", ppm_path);
    return nan_count ? 1 : 0;
}
