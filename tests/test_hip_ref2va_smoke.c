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
    fprintf(stderr, "FAIL tests/test_hip_ref2va_smoke.c: %s\n", message);
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
             "%s/Ref2VA/tokenizer/tokenizer.json", model_root);
    snprintf(weights, sizeof(weights), "%s/Ref2VA/text_encoder", model_root);

    size_t pixel_count = (size_t)FRAMES * 3 * HEIGHT * WIDTH;
    float *pixels = malloc(pixel_count * sizeof(*pixels));
    if (!pixels) die("out of memory");
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++) {
                size_t index = ((((size_t)c * HEIGHT) + (size_t)y) * WIDTH) +
                               (size_t)x;
                pixels[index] = 0.15f + 0.4f * (float)x / (float)(WIDTH - 1);
            }

    char error[512];
    h3_vision_output vision = {0};
    if (!h3_vision_encode_bf16(weights, "h3_shaders.metal", pixels, FRAMES,
                               HEIGHT, WIDTH, progress, "ref2va vision",
                               &vision, error, sizeof(error)))
        die(error);

    h3_tokenizer *tokenizer = h3_tokenizer_load(tokenizer_path, error,
                                                sizeof(error));
    if (!tokenizer) die(error);
    h3_reference_presentation reference = {
        H3_PRESENTATION_IMAGE, 0, &vision, 1, NULL
    };
    h3_text_embedding embedding = {0};
    if (!h3_multimodal_encode_ref2va_bf16(
            tokenizer, weights, "h3_shaders.metal", "a red square",
            &reference, 1, progress, "ref2va text", &embedding, error,
            sizeof(error)))
        die(error);
    if (embedding.tokens <= vision.tokens ||
        embedding.width != H3_TEXT_HIDDEN_SIZE || !embedding.tags)
        die("Ref2VA encoder returned the wrong shape");
    size_t nan_count = 0, vision_tags = 0, text_tags = 0;
    double abs_sum = 0.0;
    size_t count = embedding.tokens * embedding.width;
    for (size_t i = 0; i < count; i++) {
        float v = bf16_to_f32(embedding.values[i]);
        if (!isfinite(v)) nan_count++;
        else abs_sum += fabs((double)v);
    }
    for (size_t i = 0; i < embedding.tokens; i++) {
        if (embedding.tags[i] == 0) vision_tags++;
        else if (embedding.tags[i] == 1) text_tags++;
        else die("Ref2VA tag is not 0/1");
    }
    printf("ref2va smoke: tokens=%zu vision_tags=%zu text_tags=%zu "
           "mean_abs=%.4g nan=%zu sdpa=%llu submissions=%llu\n",
           embedding.tokens, vision_tags, text_tags,
           abs_sum / (double)count, nan_count,
           (unsigned long long)embedding.gpu_stats.mps_sdpa_dispatches,
           (unsigned long long)embedding.gpu_stats.submissions);
    if (nan_count) die("Ref2VA encoder returned non-finite values");
    if (abs_sum < 1e-6) die("Ref2VA encoder collapsed to zeros");
    if (!vision_tags || !text_tags)
        die("Ref2VA tags did not mix vision and language rows");
    if (embedding.gpu_stats.mps_sdpa_dispatches != 50)
        die("Ref2VA attention count mismatch (expected 50 layers)");

    h3_text_embedding_free(&embedding);
    h3_vision_output_free(&vision);

    size_t video_pixels = 2 * 3 * HEIGHT * WIDTH;
    float *video = malloc(video_pixels * sizeof(*video));
    if (!video) die("out of memory");
    for (int t = 0; t < 2; t++)
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < HEIGHT; y++)
                for (int x = 0; x < WIDTH; x++) {
                    size_t index = (((((size_t)t * 3 + (size_t)c) * HEIGHT) +
                        (size_t)y) * WIDTH) + (size_t)x;
                    video[index] = 0.2f + 0.3f * (float)t;
                }
    h3_vision_output clip = {0};
    if (!h3_vision_encode_bf16(weights, "h3_shaders.metal", video, 2,
                               HEIGHT, WIDTH, progress, "ref2va video vision",
                               &clip, error, sizeof(error)))
        die(error);
    double timestamp = 0.25;
    h3_reference_presentation video_ref = {
        H3_PRESENTATION_VIDEO, 0, &clip, 1, &timestamp
    };
    h3_text_embedding video_embedding = {0};
    if (!h3_multimodal_encode_ref2va_bf16(
            tokenizer, weights, "h3_shaders.metal", "a red square",
            &video_ref, 1, progress, "ref2va video text", &video_embedding,
            error, sizeof(error)))
        die(error);
    size_t video_nan = 0;
    for (size_t i = 0; i < video_embedding.tokens * video_embedding.width; i++) {
        if (!isfinite(bf16_to_f32(video_embedding.values[i]))) video_nan++;
    }
    printf("ref2va video: tokens=%zu nan=%zu sdpa=%llu\n",
           video_embedding.tokens, video_nan,
           (unsigned long long)video_embedding.gpu_stats.mps_sdpa_dispatches);
    if (video_nan) die("Ref2VA video presentation returned non-finite values");
    if (video_embedding.gpu_stats.mps_sdpa_dispatches != 50)
        die("Ref2VA video attention count mismatch");

    h3_text_embedding_free(&video_embedding);
    h3_vision_output_free(&clip);
    h3_tokenizer_free(tokenizer);
    free(pixels);
    free(video);
    puts("ok: HIP Ref2VA image and video presentations produce mixed rows");
    return 0;
}
