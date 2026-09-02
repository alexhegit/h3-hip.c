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
    FFN = 14336,
    PADDED_ROWS = 128,
    ROPE_HALF = 48
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

static int report_diff(const char *name, const uint16_t *got,
                       const uint16_t *ref, size_t count, float max_abs_limit,
                       float rel_l2_limit) {
    float max_abs = 0.0f, ref_abs = 0.0f, sum_sq_err = 0.0f, sum_sq_ref = 0.0f;
    size_t nan_count = 0;
    for (size_t i = 0; i < count; i++) {
        float a = bf16_to_f32(got[i]);
        float b = bf16_to_f32(ref[i]);
        if (!isfinite(a) || !isfinite(b)) {
            nan_count++;
            continue;
        }
        float err = fabsf(a - b);
        if (err > max_abs) max_abs = err;
        float rb = fabsf(b);
        if (rb > ref_abs) ref_abs = rb;
        sum_sq_err += err * err;
        sum_sq_ref += b * b;
    }
    float rel = sum_sq_ref > 0.0f ? sqrtf(sum_sq_err / sum_sq_ref) : max_abs;
    printf("%s  max_abs=%.4g  ref_abs=%.4g  rel_l2=%.4g  nan=%zu\n",
           name, max_abs, ref_abs, rel, nan_count);
    if (nan_count || max_abs > max_abs_limit || rel > rel_l2_limit) {
        fprintf(stderr, "FAIL %s exceeds abs %.3g or rel %.3g\n", name,
                max_abs_limit, rel_l2_limit);
        return 1;
    }
    return 0;
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
    h3_gpu_tensor *int8_query = h3_gpu_tensor_new_bf16(gpu, (size_t)SEQUENCE * INNER);
    h3_gpu_tensor *int8_key = h3_gpu_tensor_new_bf16(gpu, (size_t)SEQUENCE * INNER);
    h3_gpu_tensor *int8_value = h3_gpu_tensor_new_bf16(gpu, (size_t)SEQUENCE * INNER);
    h3_gpu_tensor *int8_heads = h3_gpu_tensor_new_bf16(gpu, (size_t)SEQUENCE * INNER);
    h3_gpu_tensor *int8_branch = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *mlp_ref = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *mlp_int8 = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *mlp_rowfc2 = h3_gpu_tensor_new_bf16(gpu, hidden_count);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_bf16(gpu, (size_t)SEQUENCE * FFN);
    h3_gpu_tensor *act_bf16 = h3_gpu_tensor_new_bf16(gpu, (size_t)SEQUENCE * FFN);
    h3_gpu_tensor *act_int8 = h3_gpu_tensor_new_bf16(gpu, (size_t)SEQUENCE * FFN);
    h3_gpu_tensor *fc1_fused = h3_gpu_tensor_new_bf16(gpu, (size_t)SEQUENCE * FFN * 2);
    uint16_t *rope_host = calloc((size_t)SEQUENCE * ROPE_HALF, sizeof(*rope_host));
    uint16_t *rope_sin_host = calloc((size_t)SEQUENCE * ROPE_HALF,
                                     sizeof(*rope_sin_host));
    if (!rope_host || !rope_sin_host) die("out of memory");
    for (size_t i = 0; i < (size_t)SEQUENCE * ROPE_HALF; i++)
        rope_host[i] = f32_to_bf16(1.0f);
    h3_gpu_tensor *rope_cos = h3_gpu_tensor_from_bf16(
        gpu, rope_host, (size_t)SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *rope_sin = h3_gpu_tensor_from_bf16(
        gpu, rope_sin_host, (size_t)SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *qkv_int8 = h3_gpu_tensor_new_i8(gpu, (size_t)INNER * 3 * HIDDEN);
    h3_gpu_tensor *qkv_scales = h3_gpu_tensor_new_f32(gpu, INNER * 3);
    h3_gpu_tensor *out_int8 = h3_gpu_tensor_new_i8(gpu, (size_t)HIDDEN * INNER);
    h3_gpu_tensor *out_scales = h3_gpu_tensor_new_f32(gpu, HIDDEN);
    h3_gpu_tensor *fc1_int8 = h3_gpu_tensor_new_i8(gpu, (size_t)FFN * 2 * HIDDEN);
    h3_gpu_tensor *fc1_scales = h3_gpu_tensor_new_f32(gpu, FFN * 2);
    h3_gpu_tensor *fc2_int8 = h3_gpu_tensor_new_i8(gpu, (size_t)HIDDEN * FFN);
    h3_gpu_tensor *fc2_scales = h3_gpu_tensor_new_f32(gpu, HIDDEN);
    h3_gpu_tensor *quantized = h3_gpu_tensor_new_i8(
        gpu, (size_t)PADDED_ROWS * (FFN > INNER ? FFN : INNER));
    h3_gpu_tensor *input_scales = h3_gpu_tensor_new_f32(
        gpu, (size_t)PADDED_ROWS * (FFN / 1024));
    h3_gpu_tensor *qkv_ws = h3_gpu_tensor_new_bf16(
        gpu, (size_t)SEQUENCE * INNER * 3);
    h3_gpu_tensor *mlp_ws = h3_gpu_tensor_new_bf16(
        gpu, (size_t)SEQUENCE * FFN * 2);
    if (!hidden || !norm || !qkv || !query || !key || !value || !heads ||
        !branch || !int8_query || !int8_key || !int8_value || !int8_heads ||
        !int8_branch || !mlp_ref || !mlp_int8 || !mlp_rowfc2 || !activated ||
        !act_bf16 || !act_int8 || !fc1_fused || !rope_cos ||
        !rope_sin ||
        !qkv_int8 || !qkv_scales || !out_int8 || !out_scales || !fc1_int8 ||
        !fc1_scales || !fc2_int8 || !fc2_scales || !quantized || !input_scales ||
        !qkv_ws || !mlp_ws)
        die("tensor allocation failed");

    gpu_call(gpu, h3_gpu_begin(gpu), "begin");
    gpu_call(gpu, h3_gpu_rms_norm_bf16(gpu, norm, hidden, norm1, SEQUENCE,
                                       HIDDEN, 1e-5f), "rms1");
    gpu_call(gpu, h3_gpu_grouped_qkv_linear_rope_bf16(
                     gpu, query, key, value, qkv, norm, qkv_w, q_norm, k_norm,
                     rope_cos, rope_sin, SEQUENCE, HIDDEN, HEADS, HEAD_DIM,
                     ROPE_HALF, 1e-5f), "bf16 qkv");
    gpu_call(gpu, h3_gpu_quantize_weight_int8(
                     gpu, qkv_int8, qkv_scales, qkv_w, INNER * 3, HIDDEN),
             "quantize qkv");
    gpu_call(gpu, h3_gpu_grouped_qkv_linear_rope_int8(
                     gpu, int8_query, int8_key, int8_value, quantized,
                     input_scales, norm, qkv_int8, qkv_scales, q_norm, k_norm,
                     rope_cos, rope_sin, SEQUENCE, HIDDEN, HEADS, HEAD_DIM,
                     ROPE_HALF, 1e-5f, 0, 0, 0, 0, qkv_ws), "int8 qkv");
    gpu_call(gpu, h3_gpu_sdpa_bf16(gpu, heads, query, key, value, SEQUENCE,
                                   HEADS, HEAD_DIM,
                                   1.0f / sqrtf((float)HEAD_DIM)),
             "bf16 sdpa");
    gpu_call(gpu, h3_gpu_linear_bf16(gpu, branch, heads, out_w, NULL, SEQUENCE,
                                     INNER, HIDDEN), "bf16 attn out");
    gpu_call(gpu, h3_gpu_quantize_weight_int8(
                     gpu, out_int8, out_scales, out_w, HIDDEN, INNER),
             "quantize attn out");
    gpu_call(gpu, h3_gpu_sdpa_bf16_head_major_output(
                     gpu, int8_heads, query, key, value, SEQUENCE, HEADS,
                     HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
             "head-major sdpa");
    gpu_call(gpu, h3_gpu_linear_int8_head_major_bf16(
                     gpu, int8_branch, quantized, input_scales, int8_heads,
                     out_int8, out_scales, SEQUENCE, HEADS, HEAD_DIM, HIDDEN),
             "int8 head-major attn out");
    gpu_call(gpu, h3_gpu_linear_bf16(gpu, fc1_fused, norm, fc1_w, NULL, SEQUENCE,
                                     HIDDEN, FFN * 2), "bf16 fc1");
    gpu_call(gpu, h3_gpu_swiglu_bf16(gpu, act_bf16, fc1_fused, SEQUENCE, FFN),
             "bf16 swiglu");
    gpu_call(gpu, h3_gpu_linear_bf16(gpu, mlp_ref, act_bf16, fc2_w, NULL,
                                     SEQUENCE, FFN, HIDDEN), "bf16 fc2");
    gpu_call(gpu, h3_gpu_quantize_weight_int8(
                     gpu, fc1_int8, fc1_scales, fc1_w, FFN * 2, HIDDEN),
             "quantize fc1");
    gpu_call(gpu, h3_gpu_quantize_weight_int8(
                     gpu, fc2_int8, fc2_scales, fc2_w, HIDDEN, FFN),
             "quantize fc2");
    gpu_call(gpu, h3_gpu_mlp_int8_bf16(
                     gpu, mlp_int8, activated, quantized, input_scales, norm,
                     fc1_int8, fc1_scales, fc2_int8, fc2_scales, fc1_w, fc2_w,
                     SEQUENCE, HIDDEN, FFN, HIDDEN, 0, 0, 0, 0, mlp_ws),
             "int8 mlp");
    gpu_call(gpu, h3_gpu_copy_bf16(gpu, act_int8, 0, activated, 0,
                                   (size_t)SEQUENCE * FFN), "save int8 swiglu");
    gpu_call(gpu, h3_gpu_mlp_int8_bf16(
                     gpu, mlp_rowfc2, activated, quantized, input_scales, norm,
                     fc1_int8, fc1_scales, fc2_int8, fc2_scales, fc1_w, fc2_w,
                     SEQUENCE, HIDDEN, FFN, HIDDEN, 0, 0, 1, 0, mlp_ws),
             "int8 mlp row fc2");
    gpu_call(gpu, h3_gpu_submit(gpu), "submit");

    size_t inner_count = (size_t)SEQUENCE * INNER;
    size_t ffn_count = (size_t)SEQUENCE * FFN;
    size_t max_count = ffn_count > inner_count ? ffn_count : inner_count;
    if (hidden_count > max_count) max_count = hidden_count;
    uint16_t *host_a = calloc(max_count, sizeof(*host_a));
    uint16_t *host_b = calloc(max_count, sizeof(*host_b));
    if (!host_a || !host_b) die("out of memory");
    int failed = 0;
    if (!h3_gpu_tensor_read_bf16(query, host_a, inner_count) ||
        !h3_gpu_tensor_read_bf16(int8_query, host_b, inner_count))
        die("cannot read query");
    failed |= report_diff("qkv.query", host_b, host_a, inner_count, 1.0f, 0.05f);
    if (!h3_gpu_tensor_read_bf16(key, host_a, inner_count) ||
        !h3_gpu_tensor_read_bf16(int8_key, host_b, inner_count))
        die("cannot read key");
    failed |= report_diff("qkv.key", host_b, host_a, inner_count, 1.0f, 0.05f);
    if (!h3_gpu_tensor_read_bf16(value, host_a, inner_count) ||
        !h3_gpu_tensor_read_bf16(int8_value, host_b, inner_count))
        die("cannot read value");
    failed |= report_diff("qkv.value", host_b, host_a, inner_count, 1.0f, 0.05f);
    if (!h3_gpu_tensor_read_bf16(branch, host_a, hidden_count) ||
        !h3_gpu_tensor_read_bf16(int8_branch, host_b, hidden_count))
        die("cannot read attn out");
    failed |= report_diff("attn.out", host_b, host_a, hidden_count, 4.0f, 0.05f);
    if (!h3_gpu_tensor_read_bf16(act_bf16, host_a, ffn_count) ||
        !h3_gpu_tensor_read_bf16(act_int8, host_b, ffn_count))
        die("cannot read swiglu");
    failed |= report_diff("mlp.swiglu", host_b, host_a, ffn_count, 40.0f, 0.05f);
    if (!h3_gpu_tensor_read_bf16(mlp_ref, host_a, hidden_count) ||
        !h3_gpu_tensor_read_bf16(mlp_int8, host_b, hidden_count))
        die("cannot read mlp");
    failed |= report_diff("mlp.grouped_fc2", host_b, host_a, hidden_count,
                          400.0f, 0.05f);
    if (!h3_gpu_tensor_read_bf16(mlp_rowfc2, host_b, hidden_count))
        die("cannot read mlp row fc2");
    failed |= report_diff("mlp.row_fc2", host_b, host_a, hidden_count, 400.0f,
                          0.05f);
    if (failed) die("int8/BF16 production-shape mismatch");
    printf("h3_hip_real_dit_parity ok (seq=%d)\n", SEQUENCE);

    h3_gpu_tensor_free(hidden);
    h3_gpu_tensor_free(norm);
    h3_gpu_tensor_free(qkv);
    h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value);
    h3_gpu_tensor_free(heads);
    h3_gpu_tensor_free(branch);
    h3_gpu_tensor_free(int8_query);
    h3_gpu_tensor_free(int8_key);
    h3_gpu_tensor_free(int8_value);
    h3_gpu_tensor_free(int8_heads);
    h3_gpu_tensor_free(int8_branch);
    h3_gpu_tensor_free(mlp_ref);
    h3_gpu_tensor_free(mlp_int8);
    h3_gpu_tensor_free(mlp_rowfc2);
    h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(act_bf16);
    h3_gpu_tensor_free(act_int8);
    h3_gpu_tensor_free(fc1_fused);
    h3_gpu_tensor_free(rope_cos);
    h3_gpu_tensor_free(rope_sin);
    h3_gpu_tensor_free(qkv_int8);
    h3_gpu_tensor_free(qkv_scales);
    h3_gpu_tensor_free(out_int8);
    h3_gpu_tensor_free(out_scales);
    h3_gpu_tensor_free(fc1_int8);
    h3_gpu_tensor_free(fc1_scales);
    h3_gpu_tensor_free(fc2_int8);
    h3_gpu_tensor_free(fc2_scales);
    h3_gpu_tensor_free(quantized);
    h3_gpu_tensor_free(input_scales);
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
    free(host_a);
    free(host_b);
    free(rope_host);
    free(rope_sin_host);
    return 0;
}
