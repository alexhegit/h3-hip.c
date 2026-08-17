#include "h3_text_encoder.h"
#include "h3_tokenizer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_hip_text_smoke.c: %s\n", message);
    exit(1);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static void progress(int completed, int total, void *opaque) {
    (void)opaque;
    if (completed == 1 || completed == total || completed % 10 == 0)
        fprintf(stderr, "hip text: %d/%d layers\n", completed, total);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] :
        "/home/amd/HF-MODELS/MiniMax-H3";
    const char *prompt = argc > 2 ? argv[2] : "A red fox";
    char tokenizer_path[1024], weights[1024];
    snprintf(tokenizer_path, sizeof(tokenizer_path),
             "%s/FL2VA/tokenizer/tokenizer.json", model_root);
    snprintf(weights, sizeof(weights), "%s/FL2VA/text_encoder", model_root);
    char error[512];
    h3_tokenizer *tokenizer = h3_tokenizer_load(tokenizer_path, error,
                                                sizeof(error));
    if (!tokenizer) die(error);
    uint32_t *ids = NULL;
    size_t token_count = 0;
    if (!h3_tokenizer_encode(tokenizer, prompt, 1, &ids, &token_count,
                             error, sizeof(error)))
        die(error);
    if (token_count < 1) die("tokenizer returned no tokens");

    h3_text_embedding embedding = {0};
    if (!h3_text_encode_bf16(weights, "h3_shaders.metal", ids, token_count,
                             progress, NULL, &embedding, error, sizeof(error)))
        die(error);
    if (embedding.tokens != token_count ||
        embedding.width != H3_TEXT_HIDDEN_SIZE)
        die("text encoder returned the wrong shape");
    size_t count = embedding.tokens * embedding.width;
    size_t nan_count = 0;
    double abs_sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        float v = bf16_to_f32(embedding.values[i]);
        if (!isfinite(v)) nan_count++;
        else abs_sum += fabs((double)v);
    }
    printf("text smoke: tokens=%zu mean_abs=%.4g nan=%zu "
           "linears=%llu sdpa=%llu submissions=%llu\n",
           embedding.tokens, abs_sum / (double)count, nan_count,
           (unsigned long long)embedding.gpu_stats.mps_linear_dispatches,
           (unsigned long long)embedding.gpu_stats.mps_sdpa_dispatches,
           (unsigned long long)embedding.gpu_stats.submissions);
    if (nan_count) die("text encoder returned non-finite values");
    if (abs_sum < 1e-6) die("text encoder collapsed to zeros");
    if (embedding.gpu_stats.mps_sdpa_dispatches != 50)
        die("text encoder attention count mismatch (expected 50 layers)");
    if (embedding.gpu_stats.submissions != 51)
        die("text encoder submission count mismatch (expected 51)");

    h3_text_embedding_free(&embedding);
    h3_tokenizer_ids_free(ids);
    h3_tokenizer_free(tokenizer);
    puts("ok: HIP Qwen text encoder produces finite tokens");
    return 0;
}
