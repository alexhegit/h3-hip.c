#include "h3_vision_encoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

enum { HEIGHT = 32, WIDTH = 32 };

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_hip_vision_smoke.c: %s\n", message);
    exit(1);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void progress(int completed, int total, void *opaque) {
    const char *label = opaque ? (const char *)opaque : "vision";
    if (completed == 1 || completed == total || completed % 8 == 0)
        fprintf(stderr, "hip %s: %d/%d layers\n", label, completed, total);
}

static void encode_check(const char *weights, int frames, const char *label) {
    size_t pixel_count = (size_t)frames * 3 * HEIGHT * WIDTH;
    float *pixels = malloc(pixel_count * sizeof(*pixels));
    if (!pixels) die("out of memory");
    for (int t = 0; t < frames; t++)
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < HEIGHT; y++)
                for (int x = 0; x < WIDTH; x++) {
                    size_t index = (((((size_t)t * 3 + (size_t)c) * HEIGHT) +
                        (size_t)y) * WIDTH) + (size_t)x;
                    pixels[index] = ((float)x / (float)(WIDTH - 1)) * 0.6f +
                                    ((float)c) * 0.1f + 0.05f * (float)t;
                }
    char error[512];
    h3_vision_output output = {0};
    if (!h3_vision_encode_bf16(weights, "h3_shaders.metal", pixels, frames,
                               HEIGHT, WIDTH, progress, (void *)label,
                               &output, error, sizeof(error)))
        die(error);
    size_t tokens = (size_t)(HEIGHT / 32) * (size_t)(WIDTH / 32);
    if (output.grid_h != HEIGHT / 16 || output.grid_w != WIDTH / 16 ||
        output.tokens != tokens)
        die("vision output geometry mismatch");
    size_t count = output.tokens * H3_VISION_OUTPUT_WIDTH;
    size_t nan_count = 0, finite = 0;
    double abs_sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        float v = bf16_to_f32(output.merged[i]);
        if (!isfinite(v)) nan_count++;
        else {
            finite++;
            abs_sum += fabs((double)v);
        }
    }
    for (unsigned d = 0; d < H3_VISION_DEEPSTACKS; d++) {
        if (!output.deepstack[d]) die("vision deepstack is missing");
        for (size_t i = 0; i < count; i++) {
            if (!isfinite(bf16_to_f32(output.deepstack[d][i]))) nan_count++;
        }
    }
    printf("vision smoke T=%d: tokens=%zu mean_abs=%.4g nan=%zu "
           "sdpa=%llu submissions=%llu\n",
           frames, output.tokens, finite ? abs_sum / (double)finite : 0.0,
           nan_count,
           (unsigned long long)output.gpu_stats.mps_sdpa_dispatches,
           (unsigned long long)output.gpu_stats.submissions);
    if (nan_count) die("vision encoder returned non-finite values");
    if (abs_sum < 1e-6) die("vision encoder collapsed to zeros");
    if (output.gpu_stats.mps_sdpa_dispatches == 0)
        die("vision encoder reported zero attention dispatches");
    h3_vision_output_free(&output);
    free(pixels);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] :
        "/home/amd/HF-MODELS/MiniMax-H3";
    char weights[1024];
    snprintf(weights, sizeof(weights), "%s/FL2VA/text_encoder", model_root);
    encode_check(weights, 1, "vision T=1");
    encode_check(weights, 2, "vision T=2");
    puts("ok: HIP Qwen vision encoder produces finite tokens for image and video");
    return 0;
}
