#include "h3_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

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

static int require_gpu(h3_gpu *gpu, int ok, const char *op) {
    if (ok) return 0;
    fprintf(stderr, "FAIL %s: %s\n", op, h3_gpu_error(gpu));
    return 1;
}

static int test_euler(h3_gpu *gpu) {
    float sample_values[] = {-7.0f, -8.0f, 10.0f, 20.0f,
                             30.0f, 40.0f, -9.0f};
    const float last_values[] = {1.0f, -2.0f, 0.5f, 4.0f};
    const float previous_values[] = {0.5f, -1.0f, 0.25f, 5.0f};
    uint16_t last_bf16[4], previous_bf16[4];
    for (size_t i = 0; i < 4; i++) {
        last_bf16[i] = f32_to_bf16(last_values[i]);
        previous_bf16[i] = f32_to_bf16(previous_values[i]);
    }
    h3_gpu_tensor *sample = h3_gpu_tensor_from_f32(gpu, sample_values, 7);
    h3_gpu_tensor *last = h3_gpu_tensor_from_bf16(gpu, last_bf16, 4);
    h3_gpu_tensor *previous = h3_gpu_tensor_from_bf16(gpu, previous_bf16, 4);
    CHECK(sample && last && previous);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin euler"));
    CHECK(!require_gpu(gpu, h3_gpu_euler_bf16(gpu, sample, 2, last, previous,
                                              4, 0.25f, 0.5f), "euler"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit euler"));
    float got[7];
    const float expected[] = {-7.0f, -8.0f, 10.3125f, 19.375f,
                              30.15625f, 40.875f, -9.0f};
    CHECK(h3_gpu_tensor_read_f32(sample, got, 7));
    CHECK(memcmp(got, expected, sizeof(expected)) == 0);
    h3_gpu_tensor_free(sample);
    h3_gpu_tensor_free(last);
    h3_gpu_tensor_free(previous);
    return 0;
}

static int test_token_pool_expand(h3_gpu *gpu) {
    enum { FULL_ROWS = 6, REDUCED_ROWS = 4, BASELINE_ROWS = 2, WIDTH = 4,
           PADDING = 4 };
    const float input_values[FULL_ROWS * WIDTH] = {
         1.0f,  2.0f,  3.0f,  4.0f,
        10.0f, 12.0f, 14.0f, 16.0f,
        14.0f, 16.0f, 18.0f, 20.0f,
        -8.0f, -4.0f,  0.0f,  4.0f,
        -4.0f,  0.0f,  4.0f,  8.0f,
        20.0f, 21.0f, 22.0f, 23.0f
    };
    const float pooled_values[REDUCED_ROWS * WIDTH] = {
         1.0f,  2.0f,  3.0f,  4.0f,
        12.0f, 14.0f, 16.0f, 18.0f,
        -6.0f, -2.0f,  2.0f,  6.0f,
        20.0f, 21.0f, 22.0f, 23.0f
    };
    const float processed_values[REDUCED_ROWS * WIDTH] = {
         2.0f,  3.0f,  4.0f,  5.0f,
        14.0f, 12.0f, 20.0f, 14.0f,
        -5.0f,  0.0f,  5.0f, 10.0f,
        30.0f, 31.0f, 32.0f, 33.0f
    };
    const float expanded_values[FULL_ROWS * WIDTH] = {
         2.0f,  3.0f,  4.0f,  5.0f,
        12.0f, 10.0f, 18.0f, 12.0f,
        16.0f, 14.0f, 22.0f, 16.0f,
        -7.0f, -2.0f,  3.0f,  8.0f,
        -3.0f,  2.0f,  7.0f, 12.0f,
        30.0f, 31.0f, 32.0f, 33.0f
    };
    const uint32_t pairs[REDUCED_ROWS * 2] = {0, 0, 1, 2, 3, 4, 5, 5};
    const uint32_t baseline_indices[REDUCED_ROWS] = {
        UINT32_MAX, 0, 1, UINT32_MAX
    };
    const uint32_t parents[FULL_ROWS] = {0, 1, 1, 2, 2, 3};
    uint16_t input_bf16[PADDING + FULL_ROWS * WIDTH];
    uint16_t processed_bf16[REDUCED_ROWS * WIDTH];
    uint16_t expected_pooled[REDUCED_ROWS * WIDTH];
    uint16_t expected_expanded[FULL_ROWS * WIDTH];
    for (size_t i = 0; i < PADDING; i++)
        input_bf16[i] = f32_to_bf16(-99.0f);
    for (size_t i = 0; i < FULL_ROWS * WIDTH; i++) {
        input_bf16[PADDING + i] = f32_to_bf16(input_values[i]);
        expected_expanded[i] = f32_to_bf16(expanded_values[i]);
    }
    for (size_t i = 0; i < REDUCED_ROWS * WIDTH; i++) {
        processed_bf16[i] = f32_to_bf16(processed_values[i]);
        expected_pooled[i] = f32_to_bf16(pooled_values[i]);
    }
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(
        gpu, input_bf16, PADDING + FULL_ROWS * WIDTH);
    h3_gpu_tensor *gpu_pairs = h3_gpu_tensor_from_u32(gpu, pairs, REDUCED_ROWS * 2);
    h3_gpu_tensor *gpu_baseline_indices = h3_gpu_tensor_from_u32(
        gpu, baseline_indices, REDUCED_ROWS);
    h3_gpu_tensor *pooled = h3_gpu_tensor_new_bf16(
        gpu, (REDUCED_ROWS + BASELINE_ROWS) * WIDTH);
    h3_gpu_tensor *original = h3_gpu_tensor_new_bf16(
        gpu, PADDING + FULL_ROWS * WIDTH);
    h3_gpu_tensor *processed = h3_gpu_tensor_from_bf16(
        gpu, processed_bf16, REDUCED_ROWS * WIDTH);
    h3_gpu_tensor *gpu_parents = h3_gpu_tensor_from_u32(gpu, parents, FULL_ROWS);
    h3_gpu_tensor *expanded = h3_gpu_tensor_new_bf16(gpu, FULL_ROWS * WIDTH);
    CHECK(input && gpu_pairs && gpu_baseline_indices && pooled && original &&
          processed && gpu_parents && expanded);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin pool"));
    CHECK(!require_gpu(gpu, h3_gpu_token_pool_bf16(
        gpu, pooled, input, PADDING, original, PADDING, pooled,
        REDUCED_ROWS * WIDTH, gpu_baseline_indices, gpu_pairs, FULL_ROWS,
        REDUCED_ROWS, BASELINE_ROWS, WIDTH), "pool"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit pool"));
    uint16_t got_pooled[REDUCED_ROWS * WIDTH];
    CHECK(h3_gpu_tensor_read_bf16(pooled, got_pooled, REDUCED_ROWS * WIDTH));
    CHECK(memcmp(got_pooled, expected_pooled, sizeof(got_pooled)) == 0);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin expand"));
    CHECK(!require_gpu(gpu, h3_gpu_token_expand_delta_bf16(
        gpu, expanded, original, PADDING, processed, pooled,
        REDUCED_ROWS * WIDTH, gpu_baseline_indices, gpu_parents, FULL_ROWS,
        REDUCED_ROWS, BASELINE_ROWS, WIDTH, 1, 1.0f), "expand"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit expand"));
    uint16_t got_expanded[FULL_ROWS * WIDTH];
    CHECK(h3_gpu_tensor_read_bf16(expanded, got_expanded, FULL_ROWS * WIDTH));
    CHECK(memcmp(got_expanded, expected_expanded, sizeof(got_expanded)) == 0);
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(gpu_pairs);
    h3_gpu_tensor_free(gpu_baseline_indices);
    h3_gpu_tensor_free(pooled);
    h3_gpu_tensor_free(original);
    h3_gpu_tensor_free(processed);
    h3_gpu_tensor_free(gpu_parents);
    h3_gpu_tensor_free(expanded);
    return 0;
}

static int test_gate_adaln(h3_gpu *gpu) {
    enum { ROWS = 2, WIDTH = 4, SLOTS = 2 };
    const float residual_f[ROWS * WIDTH] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f
    };
    const float branch_f[ROWS * WIDTH] = {
        0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f
    };
    uint16_t residual[ROWS * WIDTH], branch[ROWS * WIDTH];
    uint16_t norm[WIDTH], gate_mod[ROWS * SLOTS * WIDTH];
    uint16_t norm_mod[ROWS * SLOTS * WIDTH];
    for (size_t i = 0; i < ROWS * WIDTH; i++) {
        residual[i] = f32_to_bf16(residual_f[i]);
        branch[i] = f32_to_bf16(branch_f[i]);
    }
    for (size_t i = 0; i < WIDTH; i++)
        norm[i] = f32_to_bf16(1.0f);
    for (size_t i = 0; i < ROWS * SLOTS * WIDTH; i++) {
        gate_mod[i] = f32_to_bf16(0.5f);
        norm_mod[i] = f32_to_bf16(0.0f);
    }
    const uint32_t row_map[ROWS] = {0, 1};
    h3_gpu_tensor *gpu_residual = h3_gpu_tensor_from_bf16(gpu, residual, ROWS * WIDTH);
    h3_gpu_tensor *gpu_branch = h3_gpu_tensor_from_bf16(gpu, branch, ROWS * WIDTH);
    h3_gpu_tensor *gpu_norm = h3_gpu_tensor_from_bf16(gpu, norm, WIDTH);
    h3_gpu_tensor *gpu_gate_mod = h3_gpu_tensor_from_bf16(gpu, gate_mod, ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *gpu_norm_mod = h3_gpu_tensor_from_bf16(gpu, norm_mod, ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *gpu_row_map = h3_gpu_tensor_from_u32(gpu, row_map, ROWS);
    h3_gpu_tensor *gated = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    CHECK(gpu_residual && gpu_branch && gpu_norm && gpu_gate_mod &&
          gpu_norm_mod && gpu_row_map && gated && out);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin gate adaln"));
    CHECK(!require_gpu(gpu, h3_gpu_gate_adaln_bf16(
        gpu, gated, out, gpu_residual, gpu_branch, gpu_norm, gpu_gate_mod,
        gpu_norm_mod, gpu_row_map, ROWS, WIDTH, SLOTS, 0, 0, 1, 1e-5f),
        "gate adaln"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit gate adaln"));
    h3_gpu_tensor_free(gpu_residual);
    h3_gpu_tensor_free(gpu_branch);
    h3_gpu_tensor_free(gpu_norm);
    h3_gpu_tensor_free(gpu_gate_mod);
    h3_gpu_tensor_free(gpu_norm_mod);
    h3_gpu_tensor_free(gpu_row_map);
    h3_gpu_tensor_free(gated);
    h3_gpu_tensor_free(out);
    return 0;
}

static int test_patch_linear(h3_gpu *gpu) {
    enum {
        PATCH_ROWS = 2, PATCH_IN = 32, PATCH_OUT = 5376, OUTPUT_ROWS = 8
    };
    size_t input_count = (size_t)PATCH_ROWS * PATCH_IN;
    size_t weight_count = (size_t)PATCH_OUT * PATCH_IN;
    float *input = (float *)malloc(input_count * sizeof(*input));
    float *weight = (float *)malloc(weight_count * sizeof(*weight));
    float *bias = (float *)malloc(PATCH_OUT * sizeof(*bias));
    CHECK(input && weight && bias);
    for (size_t i = 0; i < input_count; i++)
        input[i] = (float)((int)(i % 17) - 8) * 0.03125f;
    for (size_t i = 0; i < weight_count; i++)
        weight[i] = (float)((int)(i % 23) - 11) * 0.0078125f;
    for (size_t i = 0; i < PATCH_OUT; i++)
        bias[i] = (float)((int)(i % 7) - 3) * 0.015625f;
    const uint32_t row_map[PATCH_ROWS] = {3, 5};
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_f32(gpu, input, input_count);
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_from_f32(gpu, weight, weight_count);
    h3_gpu_tensor *gpu_bias = h3_gpu_tensor_from_f32(gpu, bias, PATCH_OUT);
    h3_gpu_tensor *gpu_row_map = h3_gpu_tensor_from_u32(gpu, row_map, PATCH_ROWS);
    h3_gpu_tensor *dense = h3_gpu_tensor_new_bf16(
        gpu, (size_t)PATCH_ROWS * PATCH_OUT);
    h3_gpu_tensor *mapped = h3_gpu_tensor_new_bf16(
        gpu, (size_t)OUTPUT_ROWS * PATCH_OUT);
    CHECK(gpu_input && gpu_weight && gpu_bias && gpu_row_map && dense && mapped);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin patch linear"));
    CHECK(!require_gpu(gpu, h3_gpu_patch_linear_bf16(
        gpu, dense, gpu_input, gpu_weight, gpu_bias, PATCH_ROWS, PATCH_IN,
        PATCH_OUT), "patch linear"));
    CHECK(!require_gpu(gpu, h3_gpu_patch_linear_bf16_map(
        gpu, mapped, gpu_input, gpu_weight, gpu_bias, gpu_row_map, OUTPUT_ROWS,
        PATCH_ROWS, PATCH_IN, PATCH_OUT), "patch linear map"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit patch linear"));
    uint16_t *got_dense = (uint16_t *)malloc(
        (size_t)PATCH_ROWS * PATCH_OUT * sizeof(*got_dense));
    uint16_t *got_mapped = (uint16_t *)malloc(
        (size_t)OUTPUT_ROWS * PATCH_OUT * sizeof(*got_mapped));
    CHECK(got_dense && got_mapped);
    CHECK(h3_gpu_tensor_read_bf16(
        dense, got_dense, (size_t)PATCH_ROWS * PATCH_OUT));
    CHECK(h3_gpu_tensor_read_bf16(
        mapped, got_mapped, (size_t)OUTPUT_ROWS * PATCH_OUT));
    for (uint32_t row = 0; row < PATCH_ROWS; row++) {
        CHECK(memcmp(got_mapped + (size_t)row_map[row] * PATCH_OUT,
                     got_dense + (size_t)row * PATCH_OUT,
                     (size_t)PATCH_OUT * sizeof(uint16_t)) == 0);
    }
    free(input);
    free(weight);
    free(bias);
    free(got_dense);
    free(got_mapped);
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_weight);
    h3_gpu_tensor_free(gpu_bias);
    h3_gpu_tensor_free(gpu_row_map);
    h3_gpu_tensor_free(dense);
    h3_gpu_tensor_free(mapped);
    return 0;
}

static void cpu_qkv_rope_bf16(const uint16_t *qkv, const uint16_t *q_weight,
                              const uint16_t *k_weight, const uint16_t *rope_cos,
                              const uint16_t *rope_sin, uint16_t *query,
                              uint16_t *key, uint16_t *value, uint32_t sequence,
                              uint32_t heads, uint32_t head_dim,
                              uint32_t rope_half, float epsilon) {
    uint32_t inner = heads * head_dim;
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            uint32_t row_base = row * inner * 3;
            uint32_t q_base = row_base + head * head_dim;
            uint32_t k_base = q_base + inner;
            uint32_t v_base = q_base + inner * 2;
            float q_sum = 0.0f, k_sum = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                float q = bf16_to_f32(qkv[q_base + d]);
                float k = bf16_to_f32(qkv[k_base + d]);
                q_sum = fmaf(q, q, q_sum);
                k_sum = fmaf(k, k, k_sum);
            }
            float q_inverse = 1.0f / sqrtf(q_sum / (float)head_dim + epsilon);
            float k_inverse = 1.0f / sqrtf(k_sum / (float)head_dim + epsilon);
            for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                float q0 = bf16_to_f32(qkv[q_base + dimension]) * q_inverse *
                    bf16_to_f32(q_weight[dimension]);
                float k0 = bf16_to_f32(qkv[k_base + dimension]) * k_inverse *
                    bf16_to_f32(k_weight[dimension]);
                if (dimension < rope_half) {
                    uint32_t pair = dimension + rope_half;
                    float q1 = bf16_to_f32(qkv[q_base + pair]) * q_inverse *
                        bf16_to_f32(q_weight[pair]);
                    float k1 = bf16_to_f32(qkv[k_base + pair]) * k_inverse *
                        bf16_to_f32(k_weight[pair]);
                    float c = bf16_to_f32(rope_cos[row * rope_half + dimension]);
                    float s = bf16_to_f32(rope_sin[row * rope_half + dimension]);
                    q0 = q0 * c - q1 * s;
                    k0 = k0 * c - k1 * s;
                } else if (dimension < rope_half * 2) {
                    uint32_t pair = dimension - rope_half;
                    float q1 = bf16_to_f32(qkv[q_base + pair]) * q_inverse *
                        bf16_to_f32(q_weight[pair]);
                    float k1 = bf16_to_f32(qkv[k_base + pair]) * k_inverse *
                        bf16_to_f32(k_weight[pair]);
                    float c = bf16_to_f32(rope_cos[row * rope_half + pair]);
                    float s = bf16_to_f32(rope_sin[row * rope_half + pair]);
                    q0 = q0 * c + q1 * s;
                    k0 = k0 * c + k1 * s;
                }
                uint32_t output_index =
                    (row * heads + head) * head_dim + dimension;
                query[output_index] = f32_to_bf16(q0);
                key[output_index] = f32_to_bf16(k0);
                value[output_index] = qkv[v_base + dimension];
            }
        }
    }
}

static int test_qkv_rope(h3_gpu *gpu) {
    enum { SEQUENCE = 2, HEADS = 1, HEAD_DIM = 4, ROPE_HALF = 2 };
    enum { INNER = HEADS * HEAD_DIM, QKV_ELEMS = SEQUENCE * INNER * 3 };
    uint16_t qkv[QKV_ELEMS];
    uint16_t q_norm[HEAD_DIM], k_norm[HEAD_DIM];
    uint16_t rope_cos[SEQUENCE * ROPE_HALF], rope_sin[SEQUENCE * ROPE_HALF];
    memset(qkv, 0, sizeof(qkv));
    for (uint32_t row = 0; row < SEQUENCE; row++) {
        qkv[row * INNER * 3 + 0] = f32_to_bf16(3.0f);
        qkv[row * INNER * 3 + 1] = f32_to_bf16(4.0f);
        qkv[row * INNER * 3 + INNER + 0] = f32_to_bf16(1.0f);
        qkv[row * INNER * 3 + INNER + 1] = f32_to_bf16(2.0f);
        qkv[row * INNER * 3 + INNER * 2 + 2] = f32_to_bf16(5.0f + (float)row);
        rope_cos[row * ROPE_HALF + 0] = f32_to_bf16(0.6f);
        rope_cos[row * ROPE_HALF + 1] = f32_to_bf16(0.8f);
        rope_sin[row * ROPE_HALF + 0] = f32_to_bf16(0.8f);
        rope_sin[row * ROPE_HALF + 1] = f32_to_bf16(-0.6f);
    }
    for (uint32_t i = 0; i < HEAD_DIM; i++) {
        q_norm[i] = f32_to_bf16(1.0f);
        k_norm[i] = f32_to_bf16(1.0f);
    }
    uint16_t expected_q[SEQUENCE * INNER];
    uint16_t expected_k[SEQUENCE * INNER];
    uint16_t expected_v[SEQUENCE * INNER];
    cpu_qkv_rope_bf16(qkv, q_norm, k_norm, rope_cos, rope_sin, expected_q,
                      expected_k, expected_v, SEQUENCE, HEADS, HEAD_DIM,
                      ROPE_HALF, 1e-5f);
    h3_gpu_tensor *gpu_qkv = h3_gpu_tensor_from_bf16(gpu, qkv, QKV_ELEMS);
    h3_gpu_tensor *gpu_q_norm = h3_gpu_tensor_from_bf16(gpu, q_norm, HEAD_DIM);
    h3_gpu_tensor *gpu_k_norm = h3_gpu_tensor_from_bf16(gpu, k_norm, HEAD_DIM);
    h3_gpu_tensor *gpu_cos = h3_gpu_tensor_from_bf16(
        gpu, rope_cos, SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *gpu_sin = h3_gpu_tensor_from_bf16(
        gpu, rope_sin, SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *query = h3_gpu_tensor_new_bf16(gpu, SEQUENCE * INNER);
    h3_gpu_tensor *key = h3_gpu_tensor_new_bf16(gpu, SEQUENCE * INNER);
    h3_gpu_tensor *value = h3_gpu_tensor_new_bf16(gpu, SEQUENCE * INNER);
    CHECK(gpu_qkv && gpu_q_norm && gpu_k_norm && gpu_cos && gpu_sin &&
          query && key && value);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin qkv rope"));
    CHECK(!require_gpu(gpu, h3_gpu_qkv_rope_bf16(
        gpu, query, key, value, gpu_qkv, gpu_q_norm, gpu_k_norm, gpu_cos,
        gpu_sin, SEQUENCE, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f), "qkv rope"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit qkv rope"));
    uint16_t got_q[SEQUENCE * INNER], got_k[SEQUENCE * INNER];
    uint16_t got_v[SEQUENCE * INNER];
    CHECK(h3_gpu_tensor_read_bf16(query, got_q, SEQUENCE * INNER));
    CHECK(h3_gpu_tensor_read_bf16(key, got_k, SEQUENCE * INNER));
    CHECK(h3_gpu_tensor_read_bf16(value, got_v, SEQUENCE * INNER));
    for (size_t i = 0; i < SEQUENCE * INNER; i++) {
        CHECK(fabsf(bf16_to_f32(got_q[i]) - bf16_to_f32(expected_q[i])) < 0.02f);
        CHECK(fabsf(bf16_to_f32(got_k[i]) - bf16_to_f32(expected_k[i])) < 0.02f);
        CHECK(got_v[i] == expected_v[i]);
    }
    h3_gpu_tensor_free(gpu_qkv);
    h3_gpu_tensor_free(gpu_q_norm);
    h3_gpu_tensor_free(gpu_k_norm);
    h3_gpu_tensor_free(gpu_cos);
    h3_gpu_tensor_free(gpu_sin);
    h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value);
    return 0;
}

static void cpu_sdpa_bf16(const uint16_t *query, const uint16_t *key,
                          const uint16_t *value, uint16_t *output,
                          uint32_t sequence, uint32_t heads, uint32_t head_dim,
                          float scale) {
    for (uint32_t head = 0; head < heads; head++) {
        for (uint32_t q_pos = 0; q_pos < sequence; q_pos++) {
            float scores[512];
            float max_score = -1e30f;
            for (uint32_t k_pos = 0; k_pos < sequence; k_pos++) {
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) {
                    uint32_t q_index = (q_pos * heads + head) * head_dim + d;
                    uint32_t k_index = (k_pos * heads + head) * head_dim + d;
                    dot = fmaf(bf16_to_f32(query[q_index]),
                               bf16_to_f32(key[k_index]), dot);
                }
                scores[k_pos] = dot * scale;
                max_score = fmaxf(max_score, scores[k_pos]);
            }
            float sum = 0.0f;
            for (uint32_t k_pos = 0; k_pos < sequence; k_pos++) {
                scores[k_pos] = expf(scores[k_pos] - max_score);
                sum += scores[k_pos];
            }
            float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (uint32_t k_pos = 0; k_pos < sequence; k_pos++) {
                    uint32_t v_index = (k_pos * heads + head) * head_dim + d;
                    acc = fmaf(bf16_to_f32(value[v_index]),
                               scores[k_pos] * inv_sum, acc);
                }
                uint32_t out_index = (q_pos * heads + head) * head_dim + d;
                output[out_index] = f32_to_bf16(acc);
            }
        }
    }
}

static int test_sdpa(h3_gpu *gpu) {
    enum { SEQUENCE = 3, HEADS = 1, HEAD_DIM = 4 };
    enum { COUNT = SEQUENCE * HEADS * HEAD_DIM };
    uint16_t query[COUNT], key[COUNT], value[COUNT];
    for (uint32_t pos = 0; pos < SEQUENCE; pos++) {
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            uint32_t index = (pos * HEADS + 0) * HEAD_DIM + d;
            query[index] = f32_to_bf16((float)(pos + 1) * 0.25f + (float)d * 0.1f);
            key[index] = f32_to_bf16((float)(pos + 1) * 0.5f - (float)d * 0.05f);
            value[index] = f32_to_bf16((float)pos - (float)d * 0.2f);
        }
    }
    uint16_t expected[COUNT];
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    cpu_sdpa_bf16(query, key, value, expected, SEQUENCE, HEADS, HEAD_DIM, scale);
    h3_gpu_tensor *gpu_q = h3_gpu_tensor_from_bf16(gpu, query, COUNT);
    h3_gpu_tensor *gpu_k = h3_gpu_tensor_from_bf16(gpu, key, COUNT);
    h3_gpu_tensor *gpu_v = h3_gpu_tensor_from_bf16(gpu, value, COUNT);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, COUNT);
    CHECK(gpu_q && gpu_k && gpu_v && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin sdpa"));
    CHECK(!require_gpu(gpu, h3_gpu_sdpa_bf16(
        gpu, output, gpu_q, gpu_k, gpu_v, SEQUENCE, HEADS, HEAD_DIM, scale),
        "sdpa"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit sdpa"));
    uint16_t got[COUNT];
    CHECK(h3_gpu_tensor_read_bf16(output, got, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(bf16_to_f32(got[i]) - bf16_to_f32(expected[i])) < 0.05f);
    }
    h3_gpu_tensor_free(gpu_q);
    h3_gpu_tensor_free(gpu_k);
    h3_gpu_tensor_free(gpu_v);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_cast(h3_gpu *gpu) {
    const float values[] = {-2.0f, -0.5f, 0.0f, 1.0f, 3.25f};
    enum { COUNT = 5 };
    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(gpu, values, COUNT);
    h3_gpu_tensor *bf16 = h3_gpu_tensor_new_bf16(gpu, COUNT);
    h3_gpu_tensor *roundtrip = h3_gpu_tensor_new_f32(gpu, COUNT);
    CHECK(input && bf16 && roundtrip);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin cast"));
    CHECK(!require_gpu(gpu, h3_gpu_cast_f32_to_bf16(gpu, bf16, input, COUNT),
                       "cast f32 to bf16"));
    CHECK(!require_gpu(gpu, h3_gpu_cast_bf16_to_f32(gpu, roundtrip, bf16, COUNT),
                       "cast bf16 to f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit cast"));
    float got[COUNT];
    CHECK(h3_gpu_tensor_read_f32(roundtrip, got, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(got[i] - values[i]) < 0.05f);
    }
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(bf16);
    h3_gpu_tensor_free(roundtrip);
    return 0;
}

static int test_mlp(h3_gpu *gpu) {
    enum { ROWS = 2, INPUT_DIM = 4, HIDDEN = 4, OUTPUT_DIM = 4 };
    enum { INPUT_ELEMS = ROWS * INPUT_DIM, FC1_ELEMS = OUTPUT_DIM * INPUT_DIM * 2,
           FC2_ELEMS = OUTPUT_DIM * HIDDEN };
    uint16_t input_bf16[INPUT_ELEMS];
    uint16_t fc1_w_bf16[FC1_ELEMS], fc2_w_bf16[FC2_ELEMS];
    for (size_t i = 0; i < INPUT_ELEMS; i++)
        input_bf16[i] = f32_to_bf16((float)((int)(i % 5) - 2) * 0.25f);
    for (size_t i = 0; i < FC1_ELEMS; i++)
        fc1_w_bf16[i] = f32_to_bf16((float)((int)(i % 7) - 3) * 0.0625f);
    for (size_t i = 0; i < FC2_ELEMS; i++)
        fc2_w_bf16[i] = f32_to_bf16((float)((int)(i % 9) - 4) * 0.03125f);
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(gpu, input_bf16, INPUT_ELEMS);
    h3_gpu_tensor *fc1_w = h3_gpu_tensor_from_bf16(gpu, fc1_w_bf16, FC1_ELEMS);
    h3_gpu_tensor *fc2_w = h3_gpu_tensor_from_bf16(gpu, fc2_w_bf16, FC2_ELEMS);
    h3_gpu_tensor *fc1_out = h3_gpu_tensor_new_bf16(gpu, ROWS * HIDDEN * 2);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_bf16(gpu, ROWS * HIDDEN);
    h3_gpu_tensor *plain = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    h3_gpu_tensor *fused = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    CHECK(input && fc1_w && fc2_w && fc1_out && activated && plain && fused);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin mlp"));
    CHECK(!require_gpu(gpu, h3_gpu_linear_bf16(
        gpu, fc1_out, input, fc1_w, NULL, ROWS, INPUT_DIM, HIDDEN * 2),
        "mlp fc1"));
    CHECK(!require_gpu(gpu, h3_gpu_swiglu_bf16(gpu, activated, fc1_out, ROWS, HIDDEN),
                       "mlp swiglu"));
    CHECK(!require_gpu(gpu, h3_gpu_linear_bf16(
        gpu, plain, activated, fc2_w, NULL, ROWS, HIDDEN, OUTPUT_DIM),
        "mlp fc2"));
    CHECK(!require_gpu(gpu, h3_gpu_mlp_bf16(
        gpu, fused, input, fc1_w, fc2_w, ROWS, INPUT_DIM, HIDDEN, OUTPUT_DIM),
        "fused mlp"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit mlp"));
    uint16_t got_plain[ROWS * OUTPUT_DIM], got_fused[ROWS * OUTPUT_DIM];
    CHECK(h3_gpu_tensor_read_bf16(plain, got_plain, ROWS * OUTPUT_DIM));
    CHECK(h3_gpu_tensor_read_bf16(fused, got_fused, ROWS * OUTPUT_DIM));
    CHECK(memcmp(got_plain, got_fused, sizeof(got_plain)) == 0);
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(fc1_w);
    h3_gpu_tensor_free(fc2_w);
    h3_gpu_tensor_free(fc1_out);
    h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(plain);
    h3_gpu_tensor_free(fused);
    return 0;
}

static int test_adaln_linear(h3_gpu *gpu) {
    enum { ROWS = 2, WIDTH = 4, OUTPUT_DIM = 3, SLOTS = 2 };
    const float input_f[ROWS * WIDTH] = {
        1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 1.5f, 2.5f
    };
    const float norm_f[WIDTH] = {1.0f, 0.5f, 1.5f, 2.0f};
    const float mod_f[ROWS * SLOTS * WIDTH] = {
        0.25f, -0.5f, 0.75f, -1.0f, 0.0f, 0.5f, 0.25f, -0.25f,
        -0.75f, 0.5f, 0.25f, 1.0f, 0.25f, 0.0f, 0.75f, -0.5f
    };
    const float weight_f[OUTPUT_DIM * WIDTH] = {
        0.125f, -0.25f, 0.5f, 0.75f,
        -0.5f, 0.25f, 0.125f, -0.125f,
        0.0625f, 0.0625f, -0.0625f, 0.0625f
    };
    const float bias_f[OUTPUT_DIM] = {0.1f, -0.2f, 0.3f};
    const uint32_t row_map[ROWS] = {0, 1};
    uint16_t input_bf16[ROWS * WIDTH], norm_bf16[WIDTH], mod_bf16[ROWS * SLOTS * WIDTH];
    uint16_t weight_bf16[OUTPUT_DIM * WIDTH], bias_bf16[OUTPUT_DIM];
    for (size_t i = 0; i < ROWS * WIDTH; i++)
        input_bf16[i] = f32_to_bf16(input_f[i]);
    for (size_t i = 0; i < WIDTH; i++)
        norm_bf16[i] = f32_to_bf16(norm_f[i]);
    for (size_t i = 0; i < ROWS * SLOTS * WIDTH; i++)
        mod_bf16[i] = f32_to_bf16(mod_f[i]);
    for (size_t i = 0; i < OUTPUT_DIM * WIDTH; i++)
        weight_bf16[i] = f32_to_bf16(weight_f[i]);
    for (size_t i = 0; i < OUTPUT_DIM; i++)
        bias_bf16[i] = f32_to_bf16(bias_f[i]);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_bf16(gpu, input_bf16, ROWS * WIDTH);
    h3_gpu_tensor *gpu_norm = h3_gpu_tensor_from_bf16(gpu, norm_bf16, WIDTH);
    h3_gpu_tensor *gpu_mod = h3_gpu_tensor_from_bf16(gpu, mod_bf16, ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_from_bf16(gpu, weight_bf16, OUTPUT_DIM * WIDTH);
    h3_gpu_tensor *gpu_bias = h3_gpu_tensor_from_bf16(gpu, bias_bf16, OUTPUT_DIM);
    h3_gpu_tensor *gpu_row_map = h3_gpu_tensor_from_u32(gpu, row_map, ROWS);
    h3_gpu_tensor *inverse = h3_gpu_tensor_new_f32(gpu, ROWS);
    h3_gpu_tensor *adaln = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    h3_gpu_tensor *reference = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    h3_gpu_tensor *fused = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    CHECK(gpu_input && gpu_norm && gpu_mod && gpu_weight && gpu_bias &&
          gpu_row_map && inverse && adaln && reference && fused);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin adaln linear"));
    CHECK(!require_gpu(gpu, h3_gpu_adaln_bf16(
        gpu, adaln, gpu_input, gpu_norm, gpu_mod, gpu_row_map, ROWS, WIDTH,
        SLOTS, 0, 1, 1e-5f), "reference adaln"));
    CHECK(!require_gpu(gpu, h3_gpu_linear_bf16(
        gpu, reference, adaln, gpu_weight, gpu_bias, ROWS, WIDTH, OUTPUT_DIM),
        "reference linear"));
    CHECK(!require_gpu(gpu, h3_gpu_adaln_linear_bf16(
        gpu, fused, inverse, gpu_input, 0, gpu_norm, gpu_mod, gpu_row_map,
        gpu_weight, gpu_bias, ROWS, WIDTH, OUTPUT_DIM, SLOTS, 0, 1, 1e-5f),
        "fused adaln linear"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit adaln linear"));
    uint16_t got_ref[ROWS * OUTPUT_DIM], got_fused[ROWS * OUTPUT_DIM];
    CHECK(h3_gpu_tensor_read_bf16(reference, got_ref, ROWS * OUTPUT_DIM));
    CHECK(h3_gpu_tensor_read_bf16(fused, got_fused, ROWS * OUTPUT_DIM));
    for (size_t i = 0; i < ROWS * OUTPUT_DIM; i++) {
        CHECK(fabsf(bf16_to_f32(got_fused[i]) - bf16_to_f32(got_ref[i])) < 0.05f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_norm);
    h3_gpu_tensor_free(gpu_mod);
    h3_gpu_tensor_free(gpu_weight);
    h3_gpu_tensor_free(gpu_bias);
    h3_gpu_tensor_free(gpu_row_map);
    h3_gpu_tensor_free(inverse);
    h3_gpu_tensor_free(adaln);
    h3_gpu_tensor_free(reference);
    h3_gpu_tensor_free(fused);
    return 0;
}

static int test_embedding(h3_gpu *gpu) {
    enum { VOCAB = 4, WIDTH = 3, TOKENS = 3 };
    const uint32_t token_ids[TOKENS] = {1, 3, 0};
    uint16_t weight[VOCAB * WIDTH];
    for (uint32_t row = 0; row < VOCAB; row++) {
        for (uint32_t col = 0; col < WIDTH; col++)
            weight[row * WIDTH + col] =
                f32_to_bf16((float)(row + 1) * 0.25f + (float)col * 0.1f);
    }
    uint16_t expected[TOKENS * WIDTH];
    for (uint32_t token = 0; token < TOKENS; token++) {
        uint32_t id = token_ids[token];
        for (uint32_t col = 0; col < WIDTH; col++)
            expected[token * WIDTH + col] = weight[id * WIDTH + col];
    }
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_from_bf16(gpu, weight, VOCAB * WIDTH);
    h3_gpu_tensor *gpu_ids = h3_gpu_tensor_from_u32(gpu, token_ids, TOKENS);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, TOKENS * WIDTH);
    CHECK(gpu_weight && gpu_ids && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin embedding"));
    CHECK(!require_gpu(gpu, h3_gpu_embedding_bf16(
        gpu, output, gpu_weight, gpu_ids, TOKENS, VOCAB, WIDTH), "embedding"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit embedding"));
    uint16_t got[TOKENS * WIDTH];
    CHECK(h3_gpu_tensor_read_bf16(output, got, TOKENS * WIDTH));
    CHECK(memcmp(got, expected, sizeof(expected)) == 0);
    h3_gpu_tensor_free(gpu_weight);
    h3_gpu_tensor_free(gpu_ids);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_silu_mul(h3_gpu *gpu) {
    enum { COUNT = 4 };
    const float gate_f[COUNT] = {0.0f, 1.0f, -1.0f, 2.0f};
    const float up_f[COUNT] = {1.0f, 2.0f, 3.0f, 0.5f};
    uint16_t gate[COUNT], up[COUNT], expected[COUNT];
    for (size_t i = 0; i < COUNT; i++) {
        gate[i] = f32_to_bf16(gate_f[i]);
        up[i] = f32_to_bf16(up_f[i]);
        expected[i] = f32_to_bf16(
            gate_f[i] / (1.0f + expf(-gate_f[i])) * up_f[i]);
    }
    h3_gpu_tensor *gpu_gate = h3_gpu_tensor_from_bf16(gpu, gate, COUNT);
    h3_gpu_tensor *gpu_up = h3_gpu_tensor_from_bf16(gpu, up, COUNT);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, COUNT);
    CHECK(gpu_gate && gpu_up && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin silu mul"));
    CHECK(!require_gpu(gpu, h3_gpu_silu_mul_bf16(gpu, output, gpu_gate, gpu_up, COUNT),
                       "silu mul"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit silu mul"));
    uint16_t got[COUNT];
    CHECK(h3_gpu_tensor_read_bf16(output, got, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(bf16_to_f32(got[i]) - bf16_to_f32(expected[i])) < 0.02f);
    }
    h3_gpu_tensor_free(gpu_gate);
    h3_gpu_tensor_free(gpu_up);
    h3_gpu_tensor_free(output);
    return 0;
}

static void layout_grouped_qkv(const uint16_t *conventional, uint16_t *grouped,
                               uint32_t sequence, uint32_t heads,
                               uint32_t head_dim) {
    uint32_t inner = heads * head_dim;
    for (uint32_t row = 0; row < sequence; row++) {
        size_t row_base = (size_t)row * inner * 3;
        for (uint32_t head = 0; head < heads; head++) {
            for (uint32_t kind = 0; kind < 3; kind++) {
                memcpy(grouped + row_base + (size_t)(head * 3 + kind) * head_dim,
                       conventional + row_base + (size_t)kind * inner +
                           (size_t)head * head_dim,
                       (size_t)head_dim * sizeof(uint16_t));
            }
        }
    }
}

static int test_grouped_qkv_rope(h3_gpu *gpu) {
    enum { SEQUENCE = 2, HEADS = 2, HEAD_DIM = 4, ROPE_HALF = 2 };
    enum { INNER = HEADS * HEAD_DIM, QKV_ELEMS = SEQUENCE * INNER * 3 };
    uint16_t conventional[QKV_ELEMS], grouped[QKV_ELEMS];
    uint16_t q_norm[HEAD_DIM], k_norm[HEAD_DIM];
    uint16_t rope_cos[SEQUENCE * ROPE_HALF], rope_sin[SEQUENCE * ROPE_HALF];
    memset(conventional, 0, sizeof(conventional));
    for (uint32_t row = 0; row < SEQUENCE; row++) {
        for (uint32_t head = 0; head < HEADS; head++) {
            size_t q_off = (size_t)row * INNER * 3 + (size_t)head * HEAD_DIM;
            conventional[q_off + 0] = f32_to_bf16(1.0f + (float)head);
            conventional[q_off + 1] = f32_to_bf16(2.0f + (float)head);
            conventional[q_off + INNER + 0] = f32_to_bf16(0.5f + (float)head);
            conventional[q_off + INNER + 1] = f32_to_bf16(1.5f + (float)head);
            conventional[q_off + INNER * 2 + 2] = f32_to_bf16(3.0f + (float)row);
        }
        rope_cos[row * ROPE_HALF + 0] = f32_to_bf16(0.6f);
        rope_cos[row * ROPE_HALF + 1] = f32_to_bf16(0.8f);
        rope_sin[row * ROPE_HALF + 0] = f32_to_bf16(0.8f);
        rope_sin[row * ROPE_HALF + 1] = f32_to_bf16(-0.6f);
    }
    for (uint32_t i = 0; i < HEAD_DIM; i++) {
        q_norm[i] = f32_to_bf16(1.0f);
        k_norm[i] = f32_to_bf16(1.0f);
    }
    layout_grouped_qkv(conventional, grouped, SEQUENCE, HEADS, HEAD_DIM);
    uint16_t expected_q[SEQUENCE * INNER], expected_k[SEQUENCE * INNER];
    uint16_t expected_v[SEQUENCE * INNER];
    cpu_qkv_rope_bf16(conventional, q_norm, k_norm, rope_cos, rope_sin,
                      expected_q, expected_k, expected_v, SEQUENCE, HEADS,
                      HEAD_DIM, ROPE_HALF, 1e-5f);
    h3_gpu_tensor *gpu_conv = h3_gpu_tensor_from_bf16(gpu, conventional, QKV_ELEMS);
    h3_gpu_tensor *gpu_grouped = h3_gpu_tensor_from_bf16(gpu, grouped, QKV_ELEMS);
    h3_gpu_tensor *gpu_q_norm = h3_gpu_tensor_from_bf16(gpu, q_norm, HEAD_DIM);
    h3_gpu_tensor *gpu_k_norm = h3_gpu_tensor_from_bf16(gpu, k_norm, HEAD_DIM);
    h3_gpu_tensor *gpu_cos = h3_gpu_tensor_from_bf16(
        gpu, rope_cos, SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *gpu_sin = h3_gpu_tensor_from_bf16(
        gpu, rope_sin, SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *plain_q = h3_gpu_tensor_new_bf16(gpu, SEQUENCE * INNER);
    h3_gpu_tensor *plain_k = h3_gpu_tensor_new_bf16(gpu, SEQUENCE * INNER);
    h3_gpu_tensor *plain_v = h3_gpu_tensor_new_bf16(gpu, SEQUENCE * INNER);
    h3_gpu_tensor *group_q = h3_gpu_tensor_new_bf16(gpu, SEQUENCE * INNER);
    h3_gpu_tensor *group_k = h3_gpu_tensor_new_bf16(gpu, SEQUENCE * INNER);
    h3_gpu_tensor *group_v = h3_gpu_tensor_new_bf16(gpu, SEQUENCE * INNER);
    CHECK(gpu_conv && gpu_grouped && gpu_q_norm && gpu_k_norm && gpu_cos &&
          gpu_sin && plain_q && plain_k && plain_v && group_q && group_k &&
          group_v);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin grouped qkv"));
    CHECK(!require_gpu(gpu, h3_gpu_qkv_rope_bf16(
        gpu, plain_q, plain_k, plain_v, gpu_conv, gpu_q_norm, gpu_k_norm,
        gpu_cos, gpu_sin, SEQUENCE, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f),
        "plain qkv rope"));
    CHECK(!require_gpu(gpu, h3_gpu_grouped_qkv_rope_bf16(
        gpu, group_q, group_k, group_v, gpu_grouped, gpu_q_norm, gpu_k_norm,
        gpu_cos, gpu_sin, SEQUENCE, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f),
        "grouped qkv rope"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit grouped qkv"));
    uint16_t got_plain_q[SEQUENCE * INNER], got_group_q[SEQUENCE * INNER];
    uint16_t got_plain_k[SEQUENCE * INNER], got_group_k[SEQUENCE * INNER];
    uint16_t got_plain_v[SEQUENCE * INNER], got_group_v[SEQUENCE * INNER];
    CHECK(h3_gpu_tensor_read_bf16(plain_q, got_plain_q, SEQUENCE * INNER));
    CHECK(h3_gpu_tensor_read_bf16(group_q, got_group_q, SEQUENCE * INNER));
    CHECK(h3_gpu_tensor_read_bf16(plain_k, got_plain_k, SEQUENCE * INNER));
    CHECK(h3_gpu_tensor_read_bf16(group_k, got_group_k, SEQUENCE * INNER));
    CHECK(h3_gpu_tensor_read_bf16(plain_v, got_plain_v, SEQUENCE * INNER));
    CHECK(h3_gpu_tensor_read_bf16(group_v, got_group_v, SEQUENCE * INNER));
    CHECK(memcmp(got_plain_q, got_group_q, sizeof(got_plain_q)) == 0);
    CHECK(memcmp(got_plain_k, got_group_k, sizeof(got_plain_k)) == 0);
    CHECK(memcmp(got_plain_v, got_group_v, sizeof(got_plain_v)) == 0);
    for (size_t i = 0; i < SEQUENCE * INNER; i++) {
        CHECK(fabsf(bf16_to_f32(got_plain_q[i]) - bf16_to_f32(expected_q[i])) < 0.02f);
        CHECK(fabsf(bf16_to_f32(got_plain_k[i]) - bf16_to_f32(expected_k[i])) < 0.02f);
        CHECK(got_plain_v[i] == expected_v[i]);
    }
    h3_gpu_tensor_free(gpu_conv);
    h3_gpu_tensor_free(gpu_grouped);
    h3_gpu_tensor_free(gpu_q_norm);
    h3_gpu_tensor_free(gpu_k_norm);
    h3_gpu_tensor_free(gpu_cos);
    h3_gpu_tensor_free(gpu_sin);
    h3_gpu_tensor_free(plain_q);
    h3_gpu_tensor_free(plain_k);
    h3_gpu_tensor_free(plain_v);
    h3_gpu_tensor_free(group_q);
    h3_gpu_tensor_free(group_k);
    h3_gpu_tensor_free(group_v);
    return 0;
}

int main(void) {
    char error[256];
    h3_gpu *gpu = h3_gpu_create("kernels/h3_kernels.hip", error, sizeof(error));
    CHECK(gpu != NULL);
    if (test_euler(gpu) != 0) return 1;
    if (test_token_pool_expand(gpu) != 0) return 1;
    if (test_gate_adaln(gpu) != 0) return 1;
    if (test_patch_linear(gpu) != 0) return 1;
    if (test_qkv_rope(gpu) != 0) return 1;
    if (test_sdpa(gpu) != 0) return 1;
    if (test_cast(gpu) != 0) return 1;
    if (test_mlp(gpu) != 0) return 1;
    if (test_adaln_linear(gpu) != 0) return 1;
    if (test_embedding(gpu) != 0) return 1;
    if (test_silu_mul(gpu) != 0) return 1;
    if (test_grouped_qkv_rope(gpu) != 0) return 1;
    h3_gpu_free(gpu);
    puts("h3_hip_bf16_tests ok");
    return 0;
}
