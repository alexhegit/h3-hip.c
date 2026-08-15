#include "h3_gpu.h"
#include "h3_weights.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    SEQUENCE = 16,
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    FFN = 14336
};

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_hip_real_dit.c: %s\n", message);
    exit(1);
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static h3_gpu_tensor *load1(h3_weight_store *weights, h3_gpu *gpu,
                            const char *name, uint64_t width) {
    uint64_t shape[] = {width};
    char error[512];
    h3_gpu_tensor *tensor = h3_weight_load_bf16(weights, gpu, name, 1, shape,
                                                error, sizeof(error));
    if (!tensor) die(error);
    return tensor;
}

static h3_gpu_tensor *load2(h3_weight_store *weights, h3_gpu *gpu,
                            const char *name, uint64_t rows, uint64_t columns) {
    uint64_t shape[] = {rows, columns};
    char error[512];
    h3_gpu_tensor *tensor = h3_weight_load_bf16(weights, gpu, name, 2, shape,
                                                error, sizeof(error));
    if (!tensor) die(error);
    return tensor;
}

static void gpu_call(h3_gpu *gpu, int ok, const char *operation) {
    if (ok) return;
    fprintf(stderr, "FAIL tests/test_hip_real_dit.c: %s: %s\n", operation,
            h3_gpu_error(gpu));
    exit(1);
}

int main(int argc, char **argv) {
    const char *model_root = argc > 1 ? argv[1] :
        "/home/amd/HF-MODELS/MiniMax-H3";
    char weights_path[1024];
    snprintf(weights_path, sizeof(weights_path), "%s/FL2VA/transformer",
             model_root);
    char error[512];
    h3_weight_store *weights = h3_weight_store_open(weights_path, error,
                                                    sizeof(error));
    if (!weights) die(error);
    h3_gpu *gpu = h3_gpu_create("kernels/h3_kernels.hip", error, sizeof(error));
    if (!gpu) die(error);

    h3_gpu_tensor *norm1 = load1(weights, gpu, "blocks.0.norm1.weight", HIDDEN);
    h3_gpu_tensor *norm2 = load1(weights, gpu, "blocks.0.norm2.weight", HIDDEN);
    h3_gpu_tensor *qkv_w = load2(weights, gpu, "blocks.0.attn.qkv_proj.weight",
                                 INNER * 3, HIDDEN);
    h3_gpu_tensor *q_norm = load1(weights, gpu, "blocks.0.attn.q_norm.weight",
                                  HEAD_DIM);
    h3_gpu_tensor *k_norm = load1(weights, gpu, "blocks.0.attn.k_norm.weight",
                                  HEAD_DIM);
    h3_gpu_tensor *out_w = load2(weights, gpu, "blocks.0.attn.out_proj.weight",
                                 HIDDEN, INNER);
    h3_gpu_tensor *fc1_w = load2(weights, gpu, "blocks.0.mlp.fc1.weight",
                                 FFN * 2, HIDDEN);
    h3_gpu_tensor *fc2_w = load2(weights, gpu, "blocks.0.mlp.fc2.weight",
                                 HIDDEN, FFN);

    size_t hidden_count = (size_t)SEQUENCE * HIDDEN;
    uint16_t *host_hidden = calloc(hidden_count, sizeof(*host_hidden));
    if (!host_hidden) die("out of memory");
    for (size_t i = 0; i < hidden_count; i++)
        host_hidden[i] = f32_to_bf16(((float)(i % 31) - 15.0f) * 0.02f);

    h3_gpu_tensor *hidden = h3_gpu_tensor_from_bf16(gpu, host_hidden,
                                                    hidden_count);
    h3_gpu_tensor *norm = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *qkv = h3_gpu_tensor_new_bf16(gpu, (size_t)SEQUENCE * INNER * 3);
    h3_gpu_tensor *query = h3_gpu_tensor_new_bf16(gpu, (size_t)SEQUENCE * INNER);
    h3_gpu_tensor *key = h3_gpu_tensor_new_bf16(gpu, (size_t)SEQUENCE * INNER);
    h3_gpu_tensor *value = h3_gpu_tensor_new_bf16(gpu, (size_t)SEQUENCE * INNER);
    h3_gpu_tensor *heads = h3_gpu_tensor_new_bf16(gpu, (size_t)SEQUENCE * INNER);
    h3_gpu_tensor *branch = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *fc1 = h3_gpu_tensor_new_bf16(gpu, (size_t)SEQUENCE * FFN * 2);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_bf16(gpu, (size_t)SEQUENCE * FFN);
    h3_gpu_tensor *dummy_rope = h3_gpu_tensor_new_bf16(gpu, 1);
    if (!hidden || !norm || !qkv || !query || !key || !value || !heads ||
        !branch || !fc1 || !activated || !dummy_rope)
        die("tensor allocation failed");

    gpu_call(gpu, h3_gpu_begin(gpu), "begin");
    gpu_call(gpu, h3_gpu_rms_norm_bf16(gpu, norm, hidden, norm1, SEQUENCE,
                                       HIDDEN, 1e-5f), "rms1");
    gpu_call(gpu, h3_gpu_linear_bf16(gpu, qkv, norm, qkv_w, NULL, SEQUENCE,
                                     HIDDEN, INNER * 3), "qkv");
    gpu_call(gpu, h3_gpu_grouped_qkv_rope_bf16(
                     gpu, query, key, value, qkv, q_norm, k_norm, dummy_rope,
                     dummy_rope, SEQUENCE, HEADS, HEAD_DIM, 0, 1e-5f),
             "qkv split");
    gpu_call(gpu, h3_gpu_sdpa_bf16(gpu, heads, query, key, value, SEQUENCE,
                                   HEADS, HEAD_DIM,
                                   1.0f / sqrtf((float)HEAD_DIM)),
             "sdpa");
    gpu_call(gpu, h3_gpu_linear_bf16(gpu, branch, heads, out_w, NULL, SEQUENCE,
                                     INNER, HIDDEN), "attn out");
    gpu_call(gpu, h3_gpu_add_bf16(gpu, hidden, hidden, branch,
                                  (uint32_t)hidden_count),
             "attn residual");
    gpu_call(gpu, h3_gpu_rms_norm_bf16(gpu, norm, hidden, norm2, SEQUENCE,
                                       HIDDEN, 1e-5f), "rms2");
    gpu_call(gpu, h3_gpu_linear_bf16(gpu, fc1, norm, fc1_w, NULL, SEQUENCE,
                                     HIDDEN, FFN * 2), "fc1");
    gpu_call(gpu, h3_gpu_swiglu_bf16(gpu, activated, fc1, SEQUENCE, FFN),
             "swiglu");
    gpu_call(gpu, h3_gpu_linear_bf16(gpu, branch, activated, fc2_w, NULL,
                                     SEQUENCE, FFN, HIDDEN), "fc2");
    gpu_call(gpu, h3_gpu_add_bf16(gpu, hidden, hidden, branch,
                                  (uint32_t)hidden_count),
             "mlp residual");
    gpu_call(gpu, h3_gpu_submit(gpu), "submit");

    if (!h3_gpu_tensor_read_bf16(hidden, host_hidden, hidden_count))
        die("cannot read block output");
    float max_abs = 0.0f;
    size_t finite = 0;
    for (size_t i = 0; i < hidden_count; i++) {
        float element = bf16_to_f32(host_hidden[i]);
        if (!isfinite(element)) {
            fprintf(stderr, "non-finite at %zu\n", i);
            die("NaN/Inf in DiT block0 output");
        }
        finite++;
        float abs_value = fabsf(element);
        if (abs_value > max_abs) max_abs = abs_value;
    }
    if (finite != hidden_count || max_abs <= 0.0f || max_abs > 1e6f)
        die("block0 output range looks invalid");
    printf("h3_hip_real_dit_smoke ok (seq=%d max_abs=%.4g)\n", SEQUENCE,
           max_abs);

    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(norm);
    h3_gpu_tensor_free(qkv);
    h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(heads);
    h3_gpu_tensor_free(branch);
    h3_gpu_tensor_free(fc1);
    h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(dummy_rope);
    h3_gpu_tensor_free(norm1);
    h3_gpu_tensor_free(norm2);
    h3_gpu_tensor_free(qkv_w);
    h3_gpu_tensor_free(q_norm);
    h3_gpu_tensor_free(k_norm);
    h3_gpu_tensor_free(out_w);
    h3_gpu_tensor_free(fc1_w);
    h3_gpu_tensor_free(fc2_w);
    h3_gpu_free(gpu);
    h3_weight_store_free(weights);
    free(host_hidden);
    return 0;
}
