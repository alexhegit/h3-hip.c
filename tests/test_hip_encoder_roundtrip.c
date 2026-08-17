#include "h3_host.h"
#include "h3_video_encoder.h"
#include "h3_video_vae.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    FRAMES = 5,
    HEIGHT = 64,
    WIDTH = 64,
    RGB = 3,
    LATENT_CHANNELS = 24
};

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_hip_encoder_roundtrip.c: %s\n", message);
    exit(1);
}

static void progress(int completed, int total, void *opaque) {
    const char *label = opaque ? (const char *)opaque : "stage";
    if (completed == 1 || completed == total)
        fprintf(stderr, "hip %s: %d/%d\n", label, completed, total);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] :
        "/home/amd/HF-MODELS/MiniMax-H3";
    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/video_vae/source", model_root);

    size_t pixel_count = (size_t)RGB * FRAMES * HEIGHT * WIDTH;
    float *pixels = malloc(pixel_count * sizeof(*pixels));
    if (!pixels) die("out of memory");
    for (int c = 0; c < RGB; c++)
        for (int t = 0; t < FRAMES; t++)
            for (int y = 0; y < HEIGHT; y++)
                for (int x = 0; x < WIDTH; x++) {
                    size_t index = ((((size_t)c * FRAMES + (size_t)t) *
                        HEIGHT + (size_t)y) * WIDTH) + (size_t)x;
                    int left = x < WIDTH / 2;
                    if (c == 0) pixels[index] = left ? 0.85f : 0.05f;
                    else if (c == 2) pixels[index] = left ? 0.05f : 0.85f;
                    else pixels[index] = 0.08f;
                }

    char error[512];
    h3_video_latent latent = {0};
    if (!h3_video_vae_encode(weights, "h3_shaders.metal", pixels, FRAMES,
                             HEIGHT, WIDTH, progress, "encoder tile",
                             &latent, error, sizeof(error)))
        die(error);
    int want_t = h3_video_encoder_latent_t(FRAMES);
    if (latent.time != want_t || latent.height != HEIGHT / 16 ||
        latent.width != WIDTH / 16)
        die("encoder returned the wrong latent shape");
    size_t latent_count = (size_t)LATENT_CHANNELS * (size_t)latent.time *
                          (size_t)latent.height * (size_t)latent.width;
    double sum = 0.0, sum_sq = 0.0;
    for (size_t i = 0; i < latent_count; i++) {
        if (!isfinite(latent.values[i])) die("encoder returned non-finite latents");
        sum += latent.values[i];
        sum_sq += (double)latent.values[i] * latent.values[i];
    }
    double mean = sum / (double)latent_count;
    double var = sum_sq / (double)latent_count - mean * mean;
    printf("visual encoder: shape %d x %d x %d, mean %.4g, var %.4g, "
           "%llu convs, %llu submissions\n",
           latent.time, latent.height, latent.width, mean, var,
           (unsigned long long)latent.gpu_stats.mps_conv_dispatches,
           (unsigned long long)latent.gpu_stats.submissions);
    if (var < 1e-8) die("encoder latents collapsed");
    if (latent.gpu_stats.mps_conv_dispatches != 34)
        die("encoder conv dispatch count mismatch (expected 34 for one tile)");
    if (latent.gpu_stats.submissions != 18)
        die("encoder submission count mismatch (expected 18 for one tile)");

    h3_video_frames frames = {0};
    if (!h3_video_vae_decode(weights, "h3_shaders.metal", latent.values,
                             latent.time, latent.height, latent.width,
                             progress, "decoder block", &frames, error,
                             sizeof(error)))
        die(error);
    if (frames.frames != FRAMES || frames.height != HEIGHT ||
        frames.width != WIDTH)
        die("decoder returned the wrong frame shape");

    double left_r = 0.0, left_b = 0.0, right_r = 0.0, right_b = 0.0;
    size_t left_n = 0, right_n = 0, nan_count = 0;
    for (int t = 0; t < frames.frames; t++)
        for (int y = 0; y < frames.height; y++)
            for (int x = 0; x < frames.width; x++) {
                size_t base = (((size_t)t * (size_t)frames.height +
                    (size_t)y) * (size_t)frames.width + (size_t)x) * 3;
                for (int c = 0; c < 3; c++) {
                    if (!isfinite(frames.rgb[base + (size_t)c])) nan_count++;
                }
                if (x < frames.width / 2) {
                    left_r += frames.rgb[base];
                    left_b += frames.rgb[base + 2];
                    left_n++;
                } else {
                    right_r += frames.rgb[base];
                    right_b += frames.rgb[base + 2];
                    right_n++;
                }
            }
    if (nan_count) die("decoder returned non-finite RGB");
    left_r /= (double)left_n;
    left_b /= (double)left_n;
    right_r /= (double)right_n;
    right_b /= (double)right_n;
    printf("visual roundtrip RGB: left R/B %.3f/%.3f  right R/B %.3f/%.3f  "
           "linears=%llu sdpa=%llu\n",
           left_r, left_b, right_r, right_b,
           (unsigned long long)frames.gpu_stats.mps_linear_dispatches,
           (unsigned long long)frames.gpu_stats.mps_sdpa_dispatches);
    if (left_r <= right_r)
        die("decoded left side is not redder than the right");
    if (right_b <= left_b)
        die("decoded right side is not bluer than the left");

    h3_video_latent_free(&latent);
    h3_video_frames_free(&frames);
    free(pixels);
    puts("ok: HIP visual encoder/decoder roundtrip keeps left-red/right-blue");
    return 0;
}
