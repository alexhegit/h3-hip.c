#include "h3_multimodal.h"
#include "h3_tokenizer.h"
#include "h3_vision_encoder.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HEIGHT = 32, WIDTH = 32, FRAMES = 1 };

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_hip_multimodal_smoke.c: %s\n", message);
    exit(1);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void progress(int completed, int total, void *opaque) {
    const char *label = opaque ? (const char *)opaque : "stage";
    if (completed == 1 || completed == total || completed % 10 == 0)
        fprintf(stderr, "hip %s: %d/%d\n", label, completed, total);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] :
        "/home/amd/HF-MODELS/MiniMax-H3";
    char tokenizer_path[1024], weights[1024];
    snprintf(tokenizer_path, sizeof(tokenizer_path),
             "%s/FL2VA/tokenizer/tokenizer.json", model_root);
    snprintf(weights, sizeof(weights), "%s/FL2VA/text_encoder", model_root);

    size_t pixel_count = (size_t)FRAMES * 3 * HEIGHT * WIDTH;
    float *pixels = malloc(pixel_count * sizeof(*pixels));
    if (!pixels) die("out of memory");
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++) {
                size_t index = ((((size_t)c * HEIGHT) + (size_t)y) * WIDTH) +
                               (size_t)x;
                pixels[index] = ((float)x / (float)(WIDTH - 1)) * 0.5f + 0.2f;
            }

    char error[512];
    h3_vision_output vision = {0};
    if (!h3_vision_encode_bf16(weights, "h3_shaders.metal", pixels, FRAMES,
                               HEIGHT, WIDTH, progress, "vision", &vision,
                               error, sizeof(error)))
        die(error);

    h3_tokenizer *tokenizer = h3_tokenizer_load(tokenizer_path, error,
                                                sizeof(error));
    if (!tokenizer) die(error);
    h3_text_embedding embedding = {0};
    if (!h3_multimodal_encode_fl2va_bf16(
            tokenizer, weights, "h3_shaders.metal", "a red square",
            &vision, 1, progress, "multimodal", &embedding, error,
            sizeof(error)))
        die(error);
    if (embedding.tokens <= vision.tokens ||
        embedding.width != H3_TEXT_HIDDEN_SIZE)
        die("multimodal encoder returned the wrong shape");
    if (!embedding.tags) die("multimodal tags are missing");
    size_t count = embedding.tokens * embedding.width;
    size_t nan_count = 0, vision_tags = 0, text_tags = 0;
    double abs_sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        float v = bf16_to_f32(embedding.values[i]);
        if (!isfinite(v)) nan_count++;
        else abs_sum += fabs((double)v);
    }
    for (size_t i = 0; i < embedding.tokens; i++) {
        if (embedding.tags[i] == 0) vision_tags++;
        else if (embedding.tags[i] == 1) text_tags++;
        else die("multimodal tag is not 0/1");
    }
    printf("multimodal smoke: tokens=%zu vision_tags=%zu text_tags=%zu "
           "mean_abs=%.4g nan=%zu sdpa=%llu submissions=%llu\n",
           embedding.tokens, vision_tags, text_tags,
           abs_sum / (double)count, nan_count,
           (unsigned long long)embedding.gpu_stats.mps_sdpa_dispatches,
           (unsigned long long)embedding.gpu_stats.submissions);
    if (nan_count) die("multimodal encoder returned non-finite values");
    if (abs_sum < 1e-6) die("multimodal encoder collapsed to zeros");
    if (!vision_tags || !text_tags)
        die("multimodal tags did not mix vision and language rows");
    if (embedding.gpu_stats.mps_sdpa_dispatches != 50)
        die("multimodal attention count mismatch (expected 50 layers)");

    h3_text_embedding_free(&embedding);
    h3_vision_output_free(&vision);
    h3_tokenizer_free(tokenizer);
    free(pixels);
    puts("ok: HIP FL2VA multimodal presentation produces mixed vision/text rows");
    return 0;
}
