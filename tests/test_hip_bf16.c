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

static int test_gate_adaln_quantize_int8(h3_gpu *gpu) {
    enum { ROWS = 2, WIDTH = 8, SLOTS = 2, PADDED_ROWS = 128 };
    enum { COUNT = ROWS * WIDTH, Q_ELEMS = PADDED_ROWS * WIDTH };
    const float residual_f[COUNT] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        -1.0f, 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f
    };
    const float branch_f[COUNT] = {
        0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f,
        1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.5f
    };
    uint16_t residual[COUNT], branch[COUNT], norm[WIDTH];
    uint16_t gate_mod[ROWS * SLOTS * WIDTH], norm_mod[ROWS * SLOTS * WIDTH];
    for (size_t i = 0; i < COUNT; i++) {
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
    h3_gpu_tensor *gpu_residual = h3_gpu_tensor_from_bf16(gpu, residual, COUNT);
    h3_gpu_tensor *gpu_branch = h3_gpu_tensor_from_bf16(gpu, branch, COUNT);
    h3_gpu_tensor *gpu_norm = h3_gpu_tensor_from_bf16(gpu, norm, WIDTH);
    h3_gpu_tensor *gpu_gate_mod = h3_gpu_tensor_from_bf16(
        gpu, gate_mod, ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *gpu_norm_mod = h3_gpu_tensor_from_bf16(
        gpu, norm_mod, ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *gpu_row_map = h3_gpu_tensor_from_u32(gpu, row_map, ROWS);
    h3_gpu_tensor *ref_gated = h3_gpu_tensor_new_bf16(gpu, COUNT);
    h3_gpu_tensor *ref_adaln = h3_gpu_tensor_new_bf16(gpu, COUNT);
    h3_gpu_tensor *ref_quant = h3_gpu_tensor_new_i8(gpu, Q_ELEMS);
    h3_gpu_tensor *ref_scales = h3_gpu_tensor_new_f32(gpu, PADDED_ROWS);
    h3_gpu_tensor *got_gated = h3_gpu_tensor_new_bf16(gpu, COUNT);
    h3_gpu_tensor *got_quant = h3_gpu_tensor_new_i8(gpu, Q_ELEMS);
    h3_gpu_tensor *got_scales = h3_gpu_tensor_new_f32(gpu, PADDED_ROWS);
    CHECK(gpu_residual && gpu_branch && gpu_norm && gpu_gate_mod &&
          gpu_norm_mod && gpu_row_map && ref_gated && ref_adaln &&
          ref_quant && ref_scales && got_gated && got_quant && got_scales);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin gate adaln quantize"));
    CHECK(!require_gpu(gpu, h3_gpu_gate_adaln_bf16(
        gpu, ref_gated, ref_adaln, gpu_residual, gpu_branch, gpu_norm,
        gpu_gate_mod, gpu_norm_mod, gpu_row_map, ROWS, WIDTH, SLOTS, 0, 0, 1,
        1e-5f), "reference gate adaln"));
    CHECK(!require_gpu(gpu, h3_gpu_gate_adaln_quantize_int8(
        gpu, got_gated, got_quant, got_scales, gpu_residual, gpu_branch,
        gpu_norm, gpu_gate_mod, gpu_norm_mod, gpu_row_map, ROWS, PADDED_ROWS,
        WIDTH, SLOTS, 0, 0, 1, 1e-5f), "gate adaln quantize int8"));
    CHECK(!require_gpu(gpu, h3_gpu_quantize_weight_int8(
        gpu, ref_quant, ref_scales, ref_adaln, ROWS, WIDTH),
        "reference gate adaln quantize"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit gate adaln quantize"));
    uint16_t read_ref_gated[COUNT], read_got_gated[COUNT];
    CHECK(h3_gpu_tensor_read_bf16(ref_gated, read_ref_gated, COUNT));
    CHECK(h3_gpu_tensor_read_bf16(got_gated, read_got_gated, COUNT));
    CHECK(memcmp(read_ref_gated, read_got_gated, sizeof(read_ref_gated)) == 0);
    int8_t read_ref_quant[Q_ELEMS], read_got_quant[Q_ELEMS];
    float read_ref_scales[PADDED_ROWS], read_got_scales[PADDED_ROWS];
    CHECK(h3_gpu_tensor_read_i8(ref_quant, read_ref_quant, COUNT));
    CHECK(h3_gpu_tensor_read_i8(got_quant, read_got_quant, COUNT));
    CHECK(h3_gpu_tensor_read_f32(ref_scales, read_ref_scales, ROWS));
    CHECK(h3_gpu_tensor_read_f32(got_scales, read_got_scales, ROWS));
    for (size_t i = 0; i < COUNT; i++) {
        float ref = (float)read_ref_quant[i] * read_ref_scales[i / WIDTH];
        float got = (float)read_got_quant[i] * read_got_scales[i / WIDTH];
        CHECK(fabsf(ref - got) < 0.15f);
    }
    h3_gpu_tensor_free(gpu_residual);
    h3_gpu_tensor_free(gpu_branch);
    h3_gpu_tensor_free(gpu_norm);
    h3_gpu_tensor_free(gpu_gate_mod);
    h3_gpu_tensor_free(gpu_norm_mod);
    h3_gpu_tensor_free(gpu_row_map);
    h3_gpu_tensor_free(ref_gated);
    h3_gpu_tensor_free(ref_adaln);
    h3_gpu_tensor_free(ref_quant);
    h3_gpu_tensor_free(ref_scales);
    h3_gpu_tensor_free(got_gated);
    h3_gpu_tensor_free(got_quant);
    h3_gpu_tensor_free(got_scales);
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

static void cpu_qkv_rope_f32(const float *qkv, const float *q_weight,
                             const float *k_weight, const float *rope_cos,
                             const float *rope_sin, float *query, float *key,
                             float *value, uint32_t sequence, uint32_t heads,
                             uint32_t head_dim, uint32_t rope_half,
                             float epsilon) {
    uint32_t inner = heads * head_dim;
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            uint32_t row_base = row * inner * 3;
            uint32_t q_base = row_base + head * head_dim;
            uint32_t k_base = q_base + inner;
            uint32_t v_base = q_base + inner * 2;
            float q_sum = 0.0f, k_sum = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                q_sum = fmaf(qkv[q_base + d], qkv[q_base + d], q_sum);
                k_sum = fmaf(qkv[k_base + d], qkv[k_base + d], k_sum);
            }
            float q_inverse = 1.0f / sqrtf(q_sum / (float)head_dim + epsilon);
            float k_inverse = 1.0f / sqrtf(k_sum / (float)head_dim + epsilon);
            for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                float q0 = qkv[q_base + dimension] * q_inverse *
                    q_weight[dimension];
                float k0 = qkv[k_base + dimension] * k_inverse *
                    k_weight[dimension];
                if (dimension < rope_half) {
                    uint32_t pair = dimension + rope_half;
                    float q1 = qkv[q_base + pair] * q_inverse * q_weight[pair];
                    float k1 = qkv[k_base + pair] * k_inverse * k_weight[pair];
                    float c = rope_cos[row * rope_half + dimension];
                    float s = rope_sin[row * rope_half + dimension];
                    q0 = q0 * c - q1 * s;
                    k0 = k0 * c - k1 * s;
                } else if (dimension < rope_half * 2) {
                    uint32_t pair = dimension - rope_half;
                    float q1 = qkv[q_base + pair] * q_inverse * q_weight[pair];
                    float k1 = qkv[k_base + pair] * k_inverse * k_weight[pair];
                    float c = rope_cos[row * rope_half + pair];
                    float s = rope_sin[row * rope_half + pair];
                    q0 = q0 * c + q1 * s;
                    k0 = k0 * c + k1 * s;
                }
                uint32_t output_index =
                    (row * heads + head) * head_dim + dimension;
                query[output_index] = q0;
                key[output_index] = k0;
                value[output_index] = qkv[v_base + dimension];
            }
        }
    }
}

static int test_qkv_rope_f32(h3_gpu *gpu) {
    enum { SEQUENCE = 2, HEADS = 1, HEAD_DIM = 4, ROPE_HALF = 2 };
    enum { INNER = HEADS * HEAD_DIM, QKV_ELEMS = SEQUENCE * INNER * 3 };
    float qkv[QKV_ELEMS];
    float q_norm[HEAD_DIM], k_norm[HEAD_DIM];
    float rope_cos[SEQUENCE * ROPE_HALF], rope_sin[SEQUENCE * ROPE_HALF];
    memset(qkv, 0, sizeof(qkv));
    for (uint32_t row = 0; row < SEQUENCE; row++) {
        qkv[row * INNER * 3 + 0] = 3.0f;
        qkv[row * INNER * 3 + 1] = 4.0f;
        qkv[row * INNER * 3 + INNER + 0] = 1.0f;
        qkv[row * INNER * 3 + INNER + 1] = 2.0f;
        qkv[row * INNER * 3 + INNER * 2 + 2] = 5.0f + (float)row;
        rope_cos[row * ROPE_HALF + 0] = 0.6f;
        rope_cos[row * ROPE_HALF + 1] = 0.8f;
        rope_sin[row * ROPE_HALF + 0] = 0.8f;
        rope_sin[row * ROPE_HALF + 1] = -0.6f;
    }
    for (uint32_t i = 0; i < HEAD_DIM; i++) {
        q_norm[i] = 1.0f;
        k_norm[i] = 1.0f;
    }
    float expected_q[SEQUENCE * INNER];
    float expected_k[SEQUENCE * INNER];
    float expected_v[SEQUENCE * INNER];
    cpu_qkv_rope_f32(qkv, q_norm, k_norm, rope_cos, rope_sin, expected_q,
                     expected_k, expected_v, SEQUENCE, HEADS, HEAD_DIM,
                     ROPE_HALF, 1e-5f);
    h3_gpu_tensor *gpu_qkv = h3_gpu_tensor_from_f32(gpu, qkv, QKV_ELEMS);
    h3_gpu_tensor *gpu_q_norm = h3_gpu_tensor_from_f32(gpu, q_norm, HEAD_DIM);
    h3_gpu_tensor *gpu_k_norm = h3_gpu_tensor_from_f32(gpu, k_norm, HEAD_DIM);
    h3_gpu_tensor *gpu_cos = h3_gpu_tensor_from_f32(
        gpu, rope_cos, SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *gpu_sin = h3_gpu_tensor_from_f32(
        gpu, rope_sin, SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *query = h3_gpu_tensor_new_f32(gpu, SEQUENCE * INNER);
    h3_gpu_tensor *key = h3_gpu_tensor_new_f32(gpu, SEQUENCE * INNER);
    h3_gpu_tensor *value = h3_gpu_tensor_new_f32(gpu, SEQUENCE * INNER);
    CHECK(gpu_qkv && gpu_q_norm && gpu_k_norm && gpu_cos && gpu_sin &&
          query && key && value);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin qkv rope f32"));
    CHECK(!require_gpu(gpu, h3_gpu_qkv_rope_f32(
        gpu, query, key, value, gpu_qkv, gpu_q_norm, gpu_k_norm, gpu_cos,
        gpu_sin, SEQUENCE, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f), "qkv rope f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit qkv rope f32"));
    float got_q[SEQUENCE * INNER], got_k[SEQUENCE * INNER];
    float got_v[SEQUENCE * INNER];
    CHECK(h3_gpu_tensor_read_f32(query, got_q, SEQUENCE * INNER));
    CHECK(h3_gpu_tensor_read_f32(key, got_k, SEQUENCE * INNER));
    CHECK(h3_gpu_tensor_read_f32(value, got_v, SEQUENCE * INNER));
    for (size_t i = 0; i < SEQUENCE * INNER; i++) {
        CHECK(fabsf(got_q[i] - expected_q[i]) < 1e-4f);
        CHECK(fabsf(got_k[i] - expected_k[i]) < 1e-4f);
        CHECK(fabsf(got_v[i] - expected_v[i]) < 1e-5f);
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

static void cpu_vision_qkv_rope_bf16(
    const uint16_t *qkv, const uint16_t *rope_cos, const uint16_t *rope_sin,
    uint16_t *query, uint16_t *key, uint16_t *value, uint32_t sequence,
    uint32_t heads, uint32_t head_dim, uint32_t rope_half) {
    uint32_t inner = heads * head_dim;
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                uint32_t row_base = row * inner * 3;
                uint32_t q_base = row_base + head * head_dim;
                uint32_t k_base = row_base + inner + head * head_dim;
                uint32_t v_base = row_base + inner * 2 + head * head_dim;
                uint32_t rope_index = row * rope_half + dimension % rope_half;
                float c = bf16_to_f32(rope_cos[rope_index]);
                float s = bf16_to_f32(rope_sin[rope_index]);
                uint32_t pair = dimension < rope_half ?
                    dimension + rope_half : dimension - rope_half;
                float q0 = bf16_to_f32(qkv[q_base + dimension]);
                float k0 = bf16_to_f32(qkv[k_base + dimension]);
                float q1 = bf16_to_f32(qkv[q_base + pair]);
                float k1 = bf16_to_f32(qkv[k_base + pair]);
                float qr = dimension < rope_half ? q0 * c - q1 * s : q0 * c + q1 * s;
                float kr = dimension < rope_half ? k0 * c - k1 * s : k0 * c + k1 * s;
                uint32_t output_index =
                    (row * heads + head) * head_dim + dimension;
                query[output_index] = f32_to_bf16(qr);
                key[output_index] = f32_to_bf16(kr);
                value[output_index] = qkv[v_base + dimension];
            }
        }
    }
}

static void cpu_video_qkv_rope_f32(const float *qkv, const float *rope_cos,
                                   const float *rope_sin, float *query,
                                   float *key, float *value, uint32_t sequence,
                                   uint32_t heads, uint32_t head_dim,
                                   uint32_t rope_half, float epsilon) {
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            uint32_t base = (row * heads + head) * head_dim * 3;
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                q_sum = fmaf(qkv[base + d], qkv[base + d], q_sum);
                k_sum = fmaf(qkv[base + head_dim + d],
                             qkv[base + head_dim + d], k_sum);
            }
            float q_inverse = 1.0f / sqrtf(q_sum / (float)head_dim + epsilon);
            float k_inverse = 1.0f / sqrtf(k_sum / (float)head_dim + epsilon);
            for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                float q0 = qkv[base + dimension] * q_inverse;
                float k0 = qkv[base + head_dim + dimension] * k_inverse;
                if (dimension < rope_half) {
                    uint32_t pair = dimension + rope_half;
                    float q1 = qkv[base + pair] * q_inverse;
                    float k1 = qkv[base + head_dim + pair] * k_inverse;
                    float c = rope_cos[row * rope_half + dimension];
                    float s = rope_sin[row * rope_half + dimension];
                    q0 = q0 * c - q1 * s;
                    k0 = k0 * c - k1 * s;
                } else if (dimension < rope_half * 2) {
                    uint32_t pair = dimension - rope_half;
                    float q1 = qkv[base + pair] * q_inverse;
                    float k1 = qkv[base + head_dim + pair] * k_inverse;
                    float c = rope_cos[row * rope_half + pair];
                    float s = rope_sin[row * rope_half + pair];
                    q0 = q0 * c + q1 * s;
                    k0 = k0 * c + k1 * s;
                }
                uint32_t output_index =
                    (row * heads + head) * head_dim + dimension;
                query[output_index] = q0;
                key[output_index] = k0;
                value[output_index] = qkv[base + head_dim * 2 + dimension];
            }
        }
    }
}

static int test_video_qkv_rope_f32(h3_gpu *gpu) {
    enum { SEQUENCE = 2, HEADS = 2, HEAD_DIM = 4, ROPE_HALF = 2 };
    enum { INNER = HEADS * HEAD_DIM, QKV_ELEMS = SEQUENCE * INNER * 3 };
    float qkv[QKV_ELEMS];
    float rope_cos[SEQUENCE * ROPE_HALF], rope_sin[SEQUENCE * ROPE_HALF];
    for (size_t i = 0; i < QKV_ELEMS; i++) {
        qkv[i] = (float)((int)(i % 11) - 5) * 0.125f;
    }
    for (uint32_t row = 0; row < SEQUENCE; row++) {
        rope_cos[row * ROPE_HALF + 0] = 0.6f;
        rope_cos[row * ROPE_HALF + 1] = 0.8f;
        rope_sin[row * ROPE_HALF + 0] = 0.8f;
        rope_sin[row * ROPE_HALF + 1] = -0.6f;
    }
    float expected_q[SEQUENCE * INNER], expected_k[SEQUENCE * INNER];
    float expected_v[SEQUENCE * INNER];
    cpu_video_qkv_rope_f32(qkv, rope_cos, rope_sin, expected_q, expected_k,
                           expected_v, SEQUENCE, HEADS, HEAD_DIM, ROPE_HALF,
                           1e-5f);
    h3_gpu_tensor *gpu_qkv = h3_gpu_tensor_from_f32(gpu, qkv, QKV_ELEMS);
    h3_gpu_tensor *gpu_cos = h3_gpu_tensor_from_f32(
        gpu, rope_cos, SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *gpu_sin = h3_gpu_tensor_from_f32(
        gpu, rope_sin, SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *query = h3_gpu_tensor_new_f32(gpu, SEQUENCE * INNER);
    h3_gpu_tensor *key = h3_gpu_tensor_new_f32(gpu, SEQUENCE * INNER);
    h3_gpu_tensor *value = h3_gpu_tensor_new_f32(gpu, SEQUENCE * INNER);
    CHECK(gpu_qkv && gpu_cos && gpu_sin && query && key && value);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin video qkv rope f32"));
    CHECK(!require_gpu(gpu, h3_gpu_video_qkv_rope_f32(
        gpu, query, key, value, gpu_qkv, gpu_cos, gpu_sin, SEQUENCE, HEADS,
        HEAD_DIM, ROPE_HALF, 1e-5f), "video qkv rope f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit video qkv rope f32"));
    float got_q[SEQUENCE * INNER], got_k[SEQUENCE * INNER];
    float got_v[SEQUENCE * INNER];
    CHECK(h3_gpu_tensor_read_f32(query, got_q, SEQUENCE * INNER));
    CHECK(h3_gpu_tensor_read_f32(key, got_k, SEQUENCE * INNER));
    CHECK(h3_gpu_tensor_read_f32(value, got_v, SEQUENCE * INNER));
    for (size_t i = 0; i < SEQUENCE * INNER; i++) {
        CHECK(fabsf(got_q[i] - expected_q[i]) < 1e-4f);
        CHECK(fabsf(got_k[i] - expected_k[i]) < 1e-4f);
        CHECK(fabsf(got_v[i] - expected_v[i]) < 1e-5f);
    }
    h3_gpu_tensor_free(gpu_qkv);
    h3_gpu_tensor_free(gpu_cos);
    h3_gpu_tensor_free(gpu_sin);
    h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value);
    return 0;
}

static void cpu_conv1d_stride_f32(const float *input, const float *weight,
                                  const float *bias, float *output,
                                  uint32_t batch, uint32_t input_length,
                                  uint32_t output_length,
                                  uint32_t input_channels,
                                  uint32_t output_channels, uint32_t kernel,
                                  uint32_t stride, uint32_t padding,
                                  uint32_t dilation) {
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t time = 0; time < output_length; time++) {
            for (uint32_t out = 0; out < output_channels; out++) {
                float sum = bias[out];
                for (uint32_t k = 0; k < kernel; k++) {
                    int32_t source = (int32_t)time * (int32_t)stride +
                        (int32_t)(k * dilation) - (int32_t)padding;
                    if (source < 0 || source >= (int32_t)input_length) continue;
                    for (uint32_t in = 0; in < input_channels; in++) {
                        sum = fmaf(
                            input[b * input_length * input_channels +
                                  (uint32_t)source * input_channels + in],
                            weight[(out * input_channels + in) * kernel + k],
                            sum);
                    }
                }
                output[b * output_length * output_channels +
                       time * output_channels + out] = sum;
            }
        }
    }
}

static void cpu_conv_transpose1d_f32(const float *input, const float *weight,
                                     const float *bias, float *output,
                                     uint32_t batch, uint32_t input_length,
                                     uint32_t output_length,
                                     uint32_t input_channels,
                                     uint32_t output_channels, uint32_t kernel,
                                     uint32_t stride, uint32_t padding) {
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t time = 0; time < output_length; time++) {
            for (uint32_t out = 0; out < output_channels; out++) {
                float sum = bias[out];
                for (uint32_t k = 0; k < kernel; k++) {
                    int32_t numerator = (int32_t)time + (int32_t)padding -
                        (int32_t)k;
                    if (numerator < 0 ||
                        (uint32_t)numerator % stride) {
                        continue;
                    }
                    int32_t source = numerator / (int32_t)stride;
                    if (source < 0 || source >= (int32_t)input_length) continue;
                    for (uint32_t in = 0; in < input_channels; in++) {
                        sum = fmaf(
                            input[b * input_length * input_channels +
                                  (uint32_t)source * input_channels + in],
                            weight[(in * output_channels + out) * kernel + k],
                            sum);
                    }
                }
                output[b * output_length * output_channels +
                       time * output_channels + out] = sum;
            }
        }
    }
}

static float cpu_upsample_at(const float *input, const float *filter,
                             uint32_t length, uint32_t channels,
                             uint32_t channel, int up_time) {
    int raw = up_time + 15;
    float result = 0.0f;
    for (int k = 0; k < 12; k++) {
        int numerator = raw - k;
        if (numerator < 0 || numerator % 2) continue;
        int source = numerator / 2 - 5;
        if (source < 0) source = 0;
        if (source >= (int)length) source = (int)length - 1;
        result += input[(uint32_t)source * channels + channel] * 2.0f *
                  filter[k];
    }
    return result;
}

static void cpu_alias_free_snake_f32(const float *input,
                                     const float *alpha_log,
                                     const float *beta_log,
                                     const float *up_filter,
                                     const float *down_filter, float *output,
                                     uint32_t batch, uint32_t length,
                                     uint32_t channels) {
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t time = 0; time < length; time++) {
            for (uint32_t channel = 0; channel < channels; channel++) {
                float alpha = expf(alpha_log[channel]);
                float beta = expf(beta_log[channel]);
                float result = 0.0f;
                for (int k = 0; k < 12; k++) {
                    int up_time = (int)time * 2 + k - 5;
                    if (up_time < 0) up_time = 0;
                    if (up_time >= (int)length * 2) up_time = (int)length * 2 - 1;
                    float value = cpu_upsample_at(
                        input + b * length * channels, up_filter, length,
                        channels, channel, up_time);
                    float sine = sinf(alpha * value);
                    value += sine * sine / (beta + 1e-9f);
                    result += value * down_filter[k];
                }
                output[(b * length + time) * channels + channel] = result;
            }
        }
    }
}

static void cpu_snake1d_f32(const float *input, const float *alpha,
                            float *output, uint32_t batch, uint32_t length,
                            uint32_t channels) {
    uint32_t count = batch * length * channels;
    for (uint32_t index = 0; index < count; index++) {
        float a = alpha[index % channels];
        float x = input[index];
        float wave = sinf(a * x);
        output[index] = x + wave * wave / (a + 1e-9f);
    }
}

static int test_alias_free_snake_f32(h3_gpu *gpu) {
    enum { BATCH = 1, LENGTH = 4, CHANNELS = 2 };
    const float input[] = {
        0.2f, -0.3f, 0.5f, 0.1f, -0.4f, 0.7f, 0.8f, -0.2f
    };
    float filter[12] = {0};
    filter[5] = filter[6] = 0.5f;
    const float alpha_log[] = {0.0f, 0.1f};
    const float beta_log[] = {0.0f, -0.2f};
    float expected[LENGTH * CHANNELS];
    cpu_alias_free_snake_f32(input, alpha_log, beta_log, filter, filter,
                             expected, BATCH, LENGTH, CHANNELS);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_f32(
        gpu, input, LENGTH * CHANNELS);
    h3_gpu_tensor *gpu_alpha = h3_gpu_tensor_from_f32(gpu, alpha_log, CHANNELS);
    h3_gpu_tensor *gpu_beta = h3_gpu_tensor_from_f32(gpu, beta_log, CHANNELS);
    h3_gpu_tensor *gpu_filter = h3_gpu_tensor_from_f32(gpu, filter, 12);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, LENGTH * CHANNELS);
    CHECK(gpu_input && gpu_alpha && gpu_beta && gpu_filter && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin alias-free snake f32"));
    CHECK(!require_gpu(gpu, h3_gpu_alias_free_snake_f32(
        gpu, output, gpu_input, gpu_alpha, gpu_beta, gpu_filter, gpu_filter,
        BATCH, LENGTH, CHANNELS), "alias-free snake f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit alias-free snake f32"));
    float got[LENGTH * CHANNELS];
    CHECK(h3_gpu_tensor_read_f32(output, got, LENGTH * CHANNELS));
    for (size_t i = 0; i < LENGTH * CHANNELS; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 2e-5f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_alpha);
    h3_gpu_tensor_free(gpu_beta);
    h3_gpu_tensor_free(gpu_filter);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_snake1d_f32(h3_gpu *gpu) {
    enum { BATCH = 1, LENGTH = 4, CHANNELS = 2 };
    const float input[] = {
        0.2f, -0.3f, 0.5f, 0.1f, -0.4f, 0.7f, 0.8f, -0.2f
    };
    const float alpha[] = {1.0f, 0.5f};
    float expected[LENGTH * CHANNELS];
    cpu_snake1d_f32(input, alpha, expected, BATCH, LENGTH, CHANNELS);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_f32(
        gpu, input, LENGTH * CHANNELS);
    h3_gpu_tensor *gpu_alpha = h3_gpu_tensor_from_f32(gpu, alpha, CHANNELS);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, LENGTH * CHANNELS);
    CHECK(gpu_input && gpu_alpha && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin snake1d f32"));
    CHECK(!require_gpu(gpu, h3_gpu_snake1d_f32(
        gpu, output, gpu_input, gpu_alpha, BATCH, LENGTH, CHANNELS),
        "snake1d f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit snake1d f32"));
    float got[LENGTH * CHANNELS];
    CHECK(h3_gpu_tensor_read_f32(output, got, LENGTH * CHANNELS));
    for (size_t i = 0; i < LENGTH * CHANNELS; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-6f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_alpha);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_conv1d_f32(h3_gpu *gpu) {
    enum { BATCH = 1, LENGTH = 4, IN_CH = 2, OUT_CH = 2, KERNEL = 3 };
    enum { PAD = 1, DILATION = 1 };
    const float input[] = {
        0.2f, -0.3f, 0.5f, 0.1f, -0.4f, 0.7f, 0.8f, -0.2f
    };
    const float weight[] = {
        0.2f, -0.1f, 0.3f, 0.4f, -0.5f, 0.6f,
        -0.2f, 0.7f, 0.1f, -0.4f, 0.3f, 0.5f
    };
    const float bias[] = {0.1f, -0.2f};
    float expected[LENGTH * OUT_CH];
    cpu_conv1d_stride_f32(input, weight, bias, expected, BATCH, LENGTH, LENGTH,
                          IN_CH, OUT_CH, KERNEL, 1, PAD, DILATION);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_f32(gpu, input, LENGTH * IN_CH);
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_from_f32(
        gpu, weight, OUT_CH * IN_CH * KERNEL);
    h3_gpu_tensor *gpu_bias = h3_gpu_tensor_from_f32(gpu, bias, OUT_CH);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, LENGTH * OUT_CH);
    CHECK(gpu_input && gpu_weight && gpu_bias && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin conv1d f32"));
    CHECK(!require_gpu(gpu, h3_gpu_conv1d_f32(
        gpu, output, gpu_input, gpu_weight, gpu_bias, BATCH, LENGTH, IN_CH,
        OUT_CH, KERNEL, PAD, DILATION), "conv1d f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit conv1d f32"));
    float got[LENGTH * OUT_CH];
    CHECK(h3_gpu_tensor_read_f32(output, got, LENGTH * OUT_CH));
    for (size_t i = 0; i < LENGTH * OUT_CH; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-4f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_weight);
    h3_gpu_tensor_free(gpu_bias);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_conv1d_stride_f32(h3_gpu *gpu) {
    enum { BATCH = 1, LENGTH = 4, OUT_LEN = 2, IN_CH = 2, OUT_CH = 2 };
    enum { KERNEL = 3, STRIDE = 2, PAD = 1, DILATION = 1 };
    const float input[] = {
        0.2f, -0.3f, 0.5f, 0.1f, -0.4f, 0.7f, 0.8f, -0.2f
    };
    const float weight[] = {
        0.2f, -0.1f, 0.3f, 0.4f, -0.5f, 0.6f,
        -0.2f, 0.7f, 0.1f, -0.4f, 0.3f, 0.5f
    };
    const float bias[] = {0.1f, -0.2f};
    float expected[OUT_LEN * OUT_CH];
    cpu_conv1d_stride_f32(input, weight, bias, expected, BATCH, LENGTH, OUT_LEN,
                          IN_CH, OUT_CH, KERNEL, STRIDE, PAD, DILATION);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_f32(gpu, input, LENGTH * IN_CH);
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_from_f32(
        gpu, weight, OUT_CH * IN_CH * KERNEL);
    h3_gpu_tensor *gpu_bias = h3_gpu_tensor_from_f32(gpu, bias, OUT_CH);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, OUT_LEN * OUT_CH);
    CHECK(gpu_input && gpu_weight && gpu_bias && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin conv1d stride f32"));
    CHECK(!require_gpu(gpu, h3_gpu_conv1d_stride_f32(
        gpu, output, gpu_input, gpu_weight, gpu_bias, BATCH, LENGTH, IN_CH,
        OUT_CH, KERNEL, STRIDE, PAD, DILATION), "conv1d stride f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit conv1d stride f32"));
    float got[OUT_LEN * OUT_CH];
    CHECK(h3_gpu_tensor_read_f32(output, got, OUT_LEN * OUT_CH));
    for (size_t i = 0; i < OUT_LEN * OUT_CH; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-4f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_weight);
    h3_gpu_tensor_free(gpu_bias);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_conv_transpose1d_f32(h3_gpu *gpu) {
    enum { BATCH = 1, LENGTH = 3, OUT_LEN = 6, IN_CH = 2, OUT_CH = 1 };
    enum { KERNEL = 4, STRIDE = 2, PAD = 1 };
    const float input[] = {0.2f, -0.3f, 0.5f, 0.1f, -0.4f, 0.7f};
    const float weight[] = {
        0.2f, 0.3f, -0.1f, 0.4f, -0.5f, 0.1f, 0.25f, 0.2f
    };
    const float bias[] = {0.05f};
    float expected[OUT_LEN * OUT_CH];
    cpu_conv_transpose1d_f32(input, weight, bias, expected, BATCH, LENGTH,
                             OUT_LEN, IN_CH, OUT_CH, KERNEL, STRIDE, PAD);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_f32(gpu, input, LENGTH * IN_CH);
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_from_f32(
        gpu, weight, IN_CH * OUT_CH * KERNEL);
    h3_gpu_tensor *gpu_bias = h3_gpu_tensor_from_f32(gpu, bias, OUT_CH);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, OUT_LEN * OUT_CH);
    CHECK(gpu_input && gpu_weight && gpu_bias && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin conv transpose f32"));
    CHECK(!require_gpu(gpu, h3_gpu_conv_transpose1d_f32(
        gpu, output, gpu_input, gpu_weight, gpu_bias, BATCH, LENGTH, IN_CH,
        OUT_CH, KERNEL, STRIDE, PAD), "conv transpose f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit conv transpose f32"));
    float got[OUT_LEN * OUT_CH];
    CHECK(h3_gpu_tensor_read_f32(output, got, OUT_LEN * OUT_CH));
    for (size_t i = 0; i < OUT_LEN * OUT_CH; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-4f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_weight);
    h3_gpu_tensor_free(gpu_bias);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_vision_qkv_rope(h3_gpu *gpu) {
    enum { SEQUENCE = 2, HEADS = 2, HEAD_DIM = 4, ROPE_HALF = 2 };
    enum { INNER = HEADS * HEAD_DIM, QKV_ELEMS = SEQUENCE * INNER * 3 };
    uint16_t qkv[QKV_ELEMS];
    uint16_t rope_cos[SEQUENCE * ROPE_HALF], rope_sin[SEQUENCE * ROPE_HALF];
    for (size_t i = 0; i < QKV_ELEMS; i++)
        qkv[i] = f32_to_bf16((float)((int)(i % 11) - 5) * 0.125f);
    for (uint32_t row = 0; row < SEQUENCE; row++) {
        rope_cos[row * ROPE_HALF + 0] = f32_to_bf16(0.6f);
        rope_cos[row * ROPE_HALF + 1] = f32_to_bf16(0.8f);
        rope_sin[row * ROPE_HALF + 0] = f32_to_bf16(0.8f);
        rope_sin[row * ROPE_HALF + 1] = f32_to_bf16(-0.6f);
    }
    uint16_t expected_q[SEQUENCE * INNER], expected_k[SEQUENCE * INNER];
    uint16_t expected_v[SEQUENCE * INNER];
    cpu_vision_qkv_rope_bf16(qkv, rope_cos, rope_sin, expected_q, expected_k,
                             expected_v, SEQUENCE, HEADS, HEAD_DIM, ROPE_HALF);
    h3_gpu_tensor *gpu_qkv = h3_gpu_tensor_from_bf16(gpu, qkv, QKV_ELEMS);
    h3_gpu_tensor *gpu_cos = h3_gpu_tensor_from_bf16(
        gpu, rope_cos, SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *gpu_sin = h3_gpu_tensor_from_bf16(
        gpu, rope_sin, SEQUENCE * ROPE_HALF);
    h3_gpu_tensor *query = h3_gpu_tensor_new_bf16(gpu, SEQUENCE * INNER);
    h3_gpu_tensor *key = h3_gpu_tensor_new_bf16(gpu, SEQUENCE * INNER);
    h3_gpu_tensor *value = h3_gpu_tensor_new_bf16(gpu, SEQUENCE * INNER);
    CHECK(gpu_qkv && gpu_cos && gpu_sin && query && key && value);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin vision qkv rope"));
    CHECK(!require_gpu(gpu, h3_gpu_vision_qkv_rope_bf16(
        gpu, query, key, value, gpu_qkv, gpu_cos, gpu_sin, SEQUENCE, HEADS,
        HEAD_DIM, ROPE_HALF), "vision qkv rope"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit vision qkv rope"));
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

static void cpu_sdpa_f32(const float *query, const float *key,
                         const float *value, float *output,
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
                    dot = fmaf(query[q_index], key[k_index], dot);
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
                    acc = fmaf(value[v_index], scores[k_pos] * inv_sum, acc);
                }
                uint32_t out_index = (q_pos * heads + head) * head_dim + d;
                output[out_index] = acc;
            }
        }
    }
}

static void cpu_audio_qkv_split_f32(const float *qkv, const float *q_bias,
                                    const float *k_bias, const float *v_bias,
                                    float *query, float *key, float *value,
                                    uint32_t batch, uint32_t length,
                                    uint32_t heads, uint32_t head_dim) {
    uint32_t width = heads * head_dim;
    uint32_t count = batch * length * width;
    for (uint32_t gid = 0; gid < count; gid++) {
        uint32_t column = gid % width;
        uint32_t row = gid / width;
        uint32_t base = row * width * 3;
        query[gid] = qkv[base + column] + q_bias[column];
        key[gid] = qkv[base + width + column] + k_bias[column];
        value[gid] = qkv[base + width * 2 + column] + v_bias[column];
    }
}

static void cpu_sdpa_causal_f32(const float *query, const float *key,
                                const float *value, float *output,
                                uint32_t batch, uint32_t sequence,
                                uint32_t heads, uint32_t head_dim,
                                float scale) {
    uint32_t slice = sequence * heads * head_dim;
    for (uint32_t b = 0; b < batch; b++) {
        const float *q_batch = query + (size_t)b * slice;
        const float *k_batch = key + (size_t)b * slice;
        const float *v_batch = value + (size_t)b * slice;
        float *o_batch = output + (size_t)b * slice;
        for (uint32_t head = 0; head < heads; head++) {
            for (uint32_t q_pos = 0; q_pos < sequence; q_pos++) {
                float scores[512];
                float max_score = -1e30f;
                for (uint32_t k_pos = 0; k_pos <= q_pos; k_pos++) {
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < head_dim; d++) {
                        uint32_t q_index =
                            (q_pos * heads + head) * head_dim + d;
                        uint32_t k_index =
                            (k_pos * heads + head) * head_dim + d;
                        dot = fmaf(q_batch[q_index], k_batch[k_index], dot);
                    }
                    scores[k_pos] = dot * scale;
                    max_score = fmaxf(max_score, scores[k_pos]);
                }
                float sum = 0.0f;
                for (uint32_t k_pos = 0; k_pos <= q_pos; k_pos++) {
                    scores[k_pos] = expf(scores[k_pos] - max_score);
                    sum += scores[k_pos];
                }
                float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) {
                    float acc = 0.0f;
                    for (uint32_t k_pos = 0; k_pos <= q_pos; k_pos++) {
                        uint32_t v_index =
                            (k_pos * heads + head) * head_dim + d;
                        acc = fmaf(v_batch[v_index], scores[k_pos] * inv_sum,
                                   acc);
                    }
                    uint32_t out_index =
                        (q_pos * heads + head) * head_dim + d;
                    o_batch[out_index] = acc;
                }
            }
        }
    }
}

static int test_audio_qkv_split_f32(h3_gpu *gpu) {
    enum { BATCH = 1, LENGTH = 2, HEADS = 2, HEAD_DIM = 2 };
    enum { WIDTH = HEADS * HEAD_DIM, COUNT = BATCH * LENGTH * WIDTH };
    const float qkv[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f,
        22.0f, 23.0f, 24.0f
    };
    const float q_bias[] = {0.1f, 0.2f, 0.3f, 0.4f};
    const float k_bias[] = {-0.1f, -0.2f, -0.3f, -0.4f};
    const float v_bias[] = {0.05f, 0.15f, 0.25f, 0.35f};
    float expected_q[COUNT], expected_k[COUNT], expected_v[COUNT];
    cpu_audio_qkv_split_f32(qkv, q_bias, k_bias, v_bias, expected_q,
                            expected_k, expected_v, BATCH, LENGTH, HEADS,
                            HEAD_DIM);
    h3_gpu_tensor *gpu_qkv = h3_gpu_tensor_from_f32(gpu, qkv, COUNT * 3);
    h3_gpu_tensor *gpu_q_bias = h3_gpu_tensor_from_f32(gpu, q_bias, WIDTH);
    h3_gpu_tensor *gpu_k_bias = h3_gpu_tensor_from_f32(gpu, k_bias, WIDTH);
    h3_gpu_tensor *gpu_v_bias = h3_gpu_tensor_from_f32(gpu, v_bias, WIDTH);
    h3_gpu_tensor *query = h3_gpu_tensor_new_f32(gpu, COUNT);
    h3_gpu_tensor *key = h3_gpu_tensor_new_f32(gpu, COUNT);
    h3_gpu_tensor *value = h3_gpu_tensor_new_f32(gpu, COUNT);
    CHECK(gpu_qkv && gpu_q_bias && gpu_k_bias && gpu_v_bias && query && key &&
          value);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin audio qkv split f32"));
    CHECK(!require_gpu(gpu, h3_gpu_audio_qkv_split_f32(
        gpu, query, key, value, gpu_qkv, gpu_q_bias, gpu_k_bias, gpu_v_bias,
        BATCH, LENGTH, HEADS, HEAD_DIM), "audio qkv split f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit audio qkv split f32"));
    float got_q[COUNT], got_k[COUNT], got_v[COUNT];
    CHECK(h3_gpu_tensor_read_f32(query, got_q, COUNT));
    CHECK(h3_gpu_tensor_read_f32(key, got_k, COUNT));
    CHECK(h3_gpu_tensor_read_f32(value, got_v, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(got_q[i] - expected_q[i]) < 1e-6f);
        CHECK(fabsf(got_k[i] - expected_k[i]) < 1e-6f);
        CHECK(fabsf(got_v[i] - expected_v[i]) < 1e-6f);
    }
    h3_gpu_tensor_free(gpu_qkv);
    h3_gpu_tensor_free(gpu_q_bias);
    h3_gpu_tensor_free(gpu_k_bias);
    h3_gpu_tensor_free(gpu_v_bias);
    h3_gpu_tensor_free(query);
    h3_gpu_tensor_free(key);
    h3_gpu_tensor_free(value);
    return 0;
}

static void cpu_audio_attention_pool_f32(const float *attended, float *output,
                                         uint32_t batch, uint32_t length,
                                         uint32_t heads, uint32_t head_dim,
                                         uint32_t output_dim) {
    uint32_t pool = head_dim / output_dim;
    uint32_t count = batch * length * output_dim;
    for (uint32_t gid = 0; gid < count; gid++) {
        uint32_t column = gid % output_dim;
        uint32_t row = gid / output_dim;
        float sum = 0.0f;
        for (uint32_t head = 0; head < heads; head++) {
            uint32_t base =
                (row * heads + head) * head_dim + column * pool;
            for (uint32_t item = 0; item < pool; item++)
                sum += attended[base + item];
        }
        output[gid] = sum / (float)(heads * pool);
    }
}

static int test_audio_attention_pool_f32(h3_gpu *gpu) {
    enum { BATCH = 1, LENGTH = 2, HEADS = 2, HEAD_DIM = 4, OUT_DIM = 2 };
    enum { IN_COUNT = BATCH * LENGTH * HEADS * HEAD_DIM };
    enum { OUT_COUNT = BATCH * LENGTH * OUT_DIM };
    const float attended[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f
    };
    float expected[OUT_COUNT];
    cpu_audio_attention_pool_f32(attended, expected, BATCH, LENGTH, HEADS,
                                 HEAD_DIM, OUT_DIM);
    h3_gpu_tensor *gpu_attended = h3_gpu_tensor_from_f32(gpu, attended, IN_COUNT);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, OUT_COUNT);
    CHECK(gpu_attended && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin audio attention pool f32"));
    CHECK(!require_gpu(gpu, h3_gpu_audio_attention_pool_f32(
        gpu, output, gpu_attended, BATCH, LENGTH, HEADS, HEAD_DIM, OUT_DIM),
        "audio attention pool f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit audio attention pool f32"));
    float got[OUT_COUNT];
    CHECK(h3_gpu_tensor_read_f32(output, got, OUT_COUNT));
    for (size_t i = 0; i < OUT_COUNT; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-6f);
    }
    h3_gpu_tensor_free(gpu_attended);
    h3_gpu_tensor_free(output);
    return 0;
}

static int h3_reflect_coordinate(int coordinate, int length) {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= length) return 2 * length - coordinate - 2;
    return coordinate;
}

static void cpu_vae_encoder_pad_f32(const float *input, float *output,
                                    uint32_t batch, uint32_t depth,
                                    uint32_t height, uint32_t width,
                                    uint32_t channels, uint32_t depth_front,
                                    uint32_t height_before,
                                    uint32_t height_after,
                                    uint32_t width_before,
                                    uint32_t width_after) {
    uint32_t out_depth = depth + depth_front;
    uint32_t out_height = height + height_before + height_after;
    uint32_t out_width = width + width_before + width_after;
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t out_t = 0; out_t < out_depth; out_t++) {
            for (uint32_t out_y = 0; out_y < out_height; out_y++) {
                for (uint32_t out_x = 0; out_x < out_width; out_x++) {
                    for (uint32_t channel = 0; channel < channels; channel++) {
                        size_t destination =
                            ((((size_t)b * out_depth + out_t) * out_height +
                              out_y) * out_width + out_x) * channels + channel;
                        if (out_t < depth_front) {
                            output[destination] = 0.0f;
                            continue;
                        }
                        int source_y = h3_reflect_coordinate(
                            (int)out_y - (int)height_before, (int)height);
                        int source_x = h3_reflect_coordinate(
                            (int)out_x - (int)width_before, (int)width);
                        uint32_t source_t = out_t - depth_front;
                        size_t source =
                            ((((size_t)b * depth + source_t) * height +
                              (uint32_t)source_y) * width +
                             (uint32_t)source_x) * channels + channel;
                        output[destination] = input[source];
                    }
                }
            }
        }
    }
}

static void cpu_conv3d_f32(const float *input, const float *weight,
                           const float *bias, float *output, uint32_t batch,
                           uint32_t depth, uint32_t height, uint32_t width,
                           uint32_t input_channels, uint32_t output_channels,
                           uint32_t kernel_depth, uint32_t kernel_height,
                           uint32_t kernel_width, uint32_t stride_depth,
                           uint32_t stride_height, uint32_t stride_width) {
    uint32_t output_depth = (depth - kernel_depth) / stride_depth + 1;
    uint32_t output_height = (height - kernel_height) / stride_height + 1;
    uint32_t output_width = (width - kernel_width) / stride_width + 1;
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t out_t = 0; out_t < output_depth; out_t++) {
            for (uint32_t out_y = 0; out_y < output_height; out_y++) {
                for (uint32_t out_x = 0; out_x < output_width; out_x++) {
                    for (uint32_t out = 0; out < output_channels; out++) {
                        float sum = bias[out];
                        for (uint32_t kd = 0; kd < kernel_depth; kd++) {
                            uint32_t source_t = out_t * stride_depth + kd;
                            for (uint32_t kh = 0; kh < kernel_height; kh++) {
                                uint32_t source_y = out_y * stride_height + kh;
                                for (uint32_t kw = 0; kw < kernel_width; kw++) {
                                    uint32_t source_x =
                                        out_x * stride_width + kw;
                                    for (uint32_t in = 0; in < input_channels;
                                         in++) {
                                        size_t input_index =
                                            (((((size_t)b * depth + source_t) *
                                                   height + source_y) *
                                                  width + source_x) *
                                                 input_channels + in);
                                        size_t weight_index =
                                            (((((size_t)out * input_channels +
                                                    in) * kernel_depth + kd) *
                                               kernel_height + kh) *
                                              kernel_width + kw);
                                        sum = fmaf(input[input_index],
                                                   weight[weight_index], sum);
                                    }
                                }
                            }
                        }
                        size_t output_index =
                            (((((size_t)b * output_depth + out_t) *
                                   output_height + out_y) *
                                  output_width + out_x) *
                                 output_channels + out);
                        output[output_index] = sum;
                    }
                }
            }
        }
    }
}

static int test_vae_encoder_pad_f32(h3_gpu *gpu) {
    enum { BATCH = 1, DEPTH = 1, HEIGHT = 4, WIDTH = 4, CHANNELS = 2 };
    enum { IN_COUNT = BATCH * DEPTH * HEIGHT * WIDTH * CHANNELS };
    enum {
        DEPTH_FRONT = 1, HEIGHT_BEFORE = 1, HEIGHT_AFTER = 1,
        WIDTH_BEFORE = 1, WIDTH_AFTER = 1
    };
    enum {
        OUT_DEPTH = DEPTH + DEPTH_FRONT,
        OUT_HEIGHT = HEIGHT + HEIGHT_BEFORE + HEIGHT_AFTER,
        OUT_WIDTH = WIDTH + WIDTH_BEFORE + WIDTH_AFTER,
        OUT_COUNT = BATCH * OUT_DEPTH * OUT_HEIGHT * OUT_WIDTH * CHANNELS
    };
    float input[IN_COUNT];
    for (size_t i = 0; i < IN_COUNT; i++) input[i] = (float)(i + 1) * 0.1f;
    float expected[OUT_COUNT];
    cpu_vae_encoder_pad_f32(input, expected, BATCH, DEPTH, HEIGHT, WIDTH,
                            CHANNELS, DEPTH_FRONT, HEIGHT_BEFORE,
                            HEIGHT_AFTER, WIDTH_BEFORE, WIDTH_AFTER);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_f32(gpu, input, IN_COUNT);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, OUT_COUNT);
    CHECK(gpu_input && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin vae encoder pad f32"));
    CHECK(!require_gpu(gpu, h3_gpu_vae_encoder_pad_f32(
        gpu, output, gpu_input, BATCH, DEPTH, HEIGHT, WIDTH, CHANNELS,
        DEPTH_FRONT, HEIGHT_BEFORE, HEIGHT_AFTER, WIDTH_BEFORE, WIDTH_AFTER),
        "vae encoder pad f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit vae encoder pad f32"));
    float got[OUT_COUNT];
    CHECK(h3_gpu_tensor_read_f32(output, got, OUT_COUNT));
    for (size_t i = 0; i < OUT_COUNT; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-6f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_conv3d_f32(h3_gpu *gpu) {
    enum { BATCH = 1, DEPTH = 2, HEIGHT = 3, WIDTH = 3 };
    enum { IN_CH = 2, OUT_CH = 2, KERNEL = 2, STRIDE = 1 };
    enum { IN_COUNT = BATCH * DEPTH * HEIGHT * WIDTH * IN_CH };
    enum { OUT_DEPTH = 1, OUT_HEIGHT = 2, OUT_WIDTH = 2 };
    enum { OUT_COUNT = BATCH * OUT_DEPTH * OUT_HEIGHT * OUT_WIDTH * OUT_CH };
    const float input[] = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f,
        1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f,
        2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3.0f,
        3.1f, 3.2f, 3.3f, 3.4f, 3.5f, 3.6f
    };
    const float weight[] = {
        0.2f, -0.1f, 0.3f, 0.4f, -0.5f, 0.6f, -0.2f, 0.7f,
        0.1f, -0.4f, 0.3f, 0.5f, 0.2f, -0.3f, 0.4f, 0.1f,
        -0.2f, 0.6f, 0.3f, -0.1f, 0.5f, 0.2f, -0.4f, 0.3f,
        0.1f, 0.2f, -0.3f, 0.4f, 0.5f, -0.6f, 0.7f, -0.8f
    };
    const float bias[] = {0.05f, -0.1f};
    float expected[OUT_COUNT];
    cpu_conv3d_f32(input, weight, bias, expected, BATCH, DEPTH, HEIGHT, WIDTH,
                   IN_CH, OUT_CH, KERNEL, KERNEL, KERNEL, STRIDE, STRIDE,
                   STRIDE);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_f32(gpu, input, IN_COUNT);
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_from_f32(
        gpu, weight, OUT_CH * IN_CH * KERNEL * KERNEL * KERNEL);
    h3_gpu_tensor *gpu_bias = h3_gpu_tensor_from_f32(gpu, bias, OUT_CH);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, OUT_COUNT);
    CHECK(gpu_input && gpu_weight && gpu_bias && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin conv3d f32"));
    CHECK(!require_gpu(gpu, h3_gpu_conv3d_f32(
        gpu, output, gpu_input, gpu_weight, gpu_bias, BATCH, DEPTH, HEIGHT,
        WIDTH, IN_CH, OUT_CH, KERNEL, KERNEL, KERNEL, STRIDE, STRIDE,
        STRIDE), "conv3d f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit conv3d f32"));
    float got[OUT_COUNT];
    CHECK(h3_gpu_tensor_read_f32(output, got, OUT_COUNT));
    for (size_t i = 0; i < OUT_COUNT; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-4f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_weight);
    h3_gpu_tensor_free(gpu_bias);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_sdpa_causal_f32(h3_gpu *gpu) {
    enum { BATCH = 2, SEQUENCE = 3, HEADS = 1, HEAD_DIM = 4 };
    enum { SLICE = SEQUENCE * HEADS * HEAD_DIM, COUNT = BATCH * SLICE };
    float query[COUNT], key[COUNT], value[COUNT];
    for (uint32_t b = 0; b < BATCH; b++) {
        for (uint32_t pos = 0; pos < SEQUENCE; pos++) {
            for (uint32_t d = 0; d < HEAD_DIM; d++) {
                uint32_t index = b * SLICE + (pos * HEADS + 0) * HEAD_DIM + d;
                query[index] = (float)(b + 1) * 0.1f + (float)(pos + 1) * 0.25f +
                    (float)d * 0.05f;
                key[index] = (float)(b + 1) * 0.2f + (float)(pos + 1) * 0.5f -
                    (float)d * 0.03f;
                value[index] = (float)pos - (float)d * 0.2f + (float)b * 0.1f;
            }
        }
    }
    float expected[COUNT];
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    cpu_sdpa_causal_f32(query, key, value, expected, BATCH, SEQUENCE, HEADS,
                        HEAD_DIM, scale);
    h3_gpu_tensor *gpu_q = h3_gpu_tensor_from_f32(gpu, query, COUNT);
    h3_gpu_tensor *gpu_k = h3_gpu_tensor_from_f32(gpu, key, COUNT);
    h3_gpu_tensor *gpu_v = h3_gpu_tensor_from_f32(gpu, value, COUNT);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, COUNT);
    CHECK(gpu_q && gpu_k && gpu_v && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin sdpa causal f32"));
    CHECK(!require_gpu(gpu, h3_gpu_sdpa_causal_f32(
        gpu, output, gpu_q, gpu_k, gpu_v, BATCH, SEQUENCE, HEADS, HEAD_DIM,
        scale), "sdpa causal f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit sdpa causal f32"));
    float got[COUNT];
    CHECK(h3_gpu_tensor_read_f32(output, got, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-4f);
    }
    h3_gpu_tensor_free(gpu_q);
    h3_gpu_tensor_free(gpu_k);
    h3_gpu_tensor_free(gpu_v);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_sdpa_f32(h3_gpu *gpu) {
    enum { SEQUENCE = 3, HEADS = 1, HEAD_DIM = 4 };
    enum { COUNT = SEQUENCE * HEADS * HEAD_DIM };
    float query[COUNT], key[COUNT], value[COUNT];
    for (uint32_t pos = 0; pos < SEQUENCE; pos++) {
        for (uint32_t d = 0; d < HEAD_DIM; d++) {
            uint32_t index = (pos * HEADS + 0) * HEAD_DIM + d;
            query[index] = (float)(pos + 1) * 0.25f + (float)d * 0.1f;
            key[index] = (float)(pos + 1) * 0.5f - (float)d * 0.05f;
            value[index] = (float)pos - (float)d * 0.2f;
        }
    }
    float expected[COUNT];
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    cpu_sdpa_f32(query, key, value, expected, SEQUENCE, HEADS, HEAD_DIM, scale);
    h3_gpu_tensor *gpu_q = h3_gpu_tensor_from_f32(gpu, query, COUNT);
    h3_gpu_tensor *gpu_k = h3_gpu_tensor_from_f32(gpu, key, COUNT);
    h3_gpu_tensor *gpu_v = h3_gpu_tensor_from_f32(gpu, value, COUNT);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, COUNT);
    CHECK(gpu_q && gpu_k && gpu_v && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin sdpa f32"));
    CHECK(!require_gpu(gpu, h3_gpu_sdpa_f32(
        gpu, output, gpu_q, gpu_k, gpu_v, SEQUENCE, HEADS, HEAD_DIM, scale),
        "sdpa f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit sdpa f32"));
    float got[COUNT];
    CHECK(h3_gpu_tensor_read_f32(output, got, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-4f);
    }
    h3_gpu_tensor_free(gpu_q);
    h3_gpu_tensor_free(gpu_k);
    h3_gpu_tensor_free(gpu_v);
    h3_gpu_tensor_free(output);
    return 0;
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

extern int h3_hip_fc1_swiglu_nax_bf16_dispatch(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, uint32_t rows, uint32_t input_dim,
    uint32_t hidden_dim);

static int test_fc1_swiglu_nax(h3_gpu *gpu) {
    enum { ROWS = 2, INPUT_DIM = 4, HIDDEN = 4 };
    enum { INPUT_ELEMS = ROWS * INPUT_DIM, FC1_ELEMS = HIDDEN * 2 * INPUT_DIM };
    uint16_t input_bf16[INPUT_ELEMS], fc1_w_bf16[FC1_ELEMS];
    for (size_t i = 0; i < INPUT_ELEMS; i++)
        input_bf16[i] = f32_to_bf16((float)((int)(i % 5) - 2) * 0.25f);
    for (size_t i = 0; i < FC1_ELEMS; i++)
        fc1_w_bf16[i] = f32_to_bf16((float)((int)(i % 7) - 3) * 0.0625f);
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(gpu, input_bf16, INPUT_ELEMS);
    h3_gpu_tensor *fc1_w = h3_gpu_tensor_from_bf16(gpu, fc1_w_bf16, FC1_ELEMS);
    h3_gpu_tensor *fc1_out = h3_gpu_tensor_new_bf16(gpu, ROWS * HIDDEN * 2);
    h3_gpu_tensor *reference = h3_gpu_tensor_new_bf16(gpu, ROWS * HIDDEN);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, ROWS * HIDDEN);
    CHECK(input && fc1_w && fc1_out && reference && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin fc1 swiglu nax"));
    CHECK(!require_gpu(gpu, h3_gpu_linear_bf16(
        gpu, fc1_out, input, fc1_w, NULL, ROWS, INPUT_DIM, HIDDEN * 2),
        "fc1 swiglu nax fc1"));
    CHECK(!require_gpu(gpu, h3_gpu_swiglu_bf16(
        gpu, reference, fc1_out, ROWS, HIDDEN), "fc1 swiglu nax reference"));
    CHECK(!require_gpu(gpu, h3_hip_fc1_swiglu_nax_bf16_dispatch(
        gpu, output, input, fc1_w, ROWS, INPUT_DIM, HIDDEN),
        "fc1 swiglu nax"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit fc1 swiglu nax"));
    uint16_t got[ROWS * HIDDEN], got_ref[ROWS * HIDDEN];
    CHECK(h3_gpu_tensor_read_bf16(output, got, ROWS * HIDDEN));
    CHECK(h3_gpu_tensor_read_bf16(reference, got_ref, ROWS * HIDDEN));
    CHECK(memcmp(got, got_ref, sizeof(got)) == 0);
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(fc1_w);
    h3_gpu_tensor_free(fc1_out);
    h3_gpu_tensor_free(reference);
    h3_gpu_tensor_free(output);
    return 0;
}

extern int h3_hip_linear_bf16_nax_dispatch(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, uint32_t rows, uint32_t input_dim,
    uint32_t output_dim);

static int test_linear_f32(h3_gpu *gpu) {
    enum { ROWS = 2, INPUT_DIM = 4, OUTPUT_DIM = 3 };
    enum { INPUT_ELEMS = ROWS * INPUT_DIM, WEIGHT_ELEMS = OUTPUT_DIM * INPUT_DIM };
    float input[INPUT_ELEMS], weight[WEIGHT_ELEMS], bias[OUTPUT_DIM];
    float expected[ROWS * OUTPUT_DIM];
    for (size_t i = 0; i < INPUT_ELEMS; i++)
        input[i] = (float)((int)(i % 7) - 3) * 0.125f;
    for (size_t i = 0; i < WEIGHT_ELEMS; i++)
        weight[i] = (float)((int)(i % 5) - 2) * 0.0625f;
    for (size_t i = 0; i < OUTPUT_DIM; i++)
        bias[i] = (float)((int)(i % 3) - 1) * 0.25f;
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t col = 0; col < OUTPUT_DIM; col++) {
            float sum = bias[col];
            for (uint32_t k = 0; k < INPUT_DIM; k++) {
                sum = fmaf(input[row * INPUT_DIM + k],
                           weight[col * INPUT_DIM + k], sum);
            }
            expected[row * OUTPUT_DIM + col] = sum;
        }
    }
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_f32(gpu, input, INPUT_ELEMS);
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_from_f32(gpu, weight, WEIGHT_ELEMS);
    h3_gpu_tensor *gpu_bias = h3_gpu_tensor_from_f32(gpu, bias, OUTPUT_DIM);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, ROWS * OUTPUT_DIM);
    CHECK(gpu_input && gpu_weight && gpu_bias && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin linear f32"));
    CHECK(!require_gpu(gpu, h3_gpu_linear_f32(
        gpu, output, gpu_input, gpu_weight, gpu_bias, ROWS, INPUT_DIM,
        OUTPUT_DIM), "linear f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit linear f32"));
    float got[ROWS * OUTPUT_DIM];
    CHECK(h3_gpu_tensor_read_f32(output, got, ROWS * OUTPUT_DIM));
    for (size_t i = 0; i < ROWS * OUTPUT_DIM; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-4f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_weight);
    h3_gpu_tensor_free(gpu_bias);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_linear_bf16_nax(h3_gpu *gpu) {
    enum { ROWS = 2, INPUT_DIM = 4, OUTPUT_DIM = 3 };
    enum { INPUT_ELEMS = ROWS * INPUT_DIM, WEIGHT_ELEMS = OUTPUT_DIM * INPUT_DIM };
    uint16_t input_bf16[INPUT_ELEMS], weight_bf16[WEIGHT_ELEMS];
    for (size_t i = 0; i < INPUT_ELEMS; i++)
        input_bf16[i] = f32_to_bf16((float)((int)(i % 7) - 3) * 0.125f);
    for (size_t i = 0; i < WEIGHT_ELEMS; i++)
        weight_bf16[i] = f32_to_bf16((float)((int)(i % 5) - 2) * 0.0625f);
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(gpu, input_bf16, INPUT_ELEMS);
    h3_gpu_tensor *weight = h3_gpu_tensor_from_bf16(gpu, weight_bf16, WEIGHT_ELEMS);
    h3_gpu_tensor *reference = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    CHECK(input && weight && reference && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin linear bf16 nax"));
    CHECK(!require_gpu(gpu, h3_gpu_linear_bf16(
        gpu, reference, input, weight, NULL, ROWS, INPUT_DIM, OUTPUT_DIM),
        "linear bf16 nax reference"));
    CHECK(!require_gpu(gpu, h3_hip_linear_bf16_nax_dispatch(
        gpu, output, input, weight, ROWS, INPUT_DIM, OUTPUT_DIM),
        "linear bf16 nax"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit linear bf16 nax"));
    uint16_t got[ROWS * OUTPUT_DIM], got_ref[ROWS * OUTPUT_DIM];
    CHECK(h3_gpu_tensor_read_bf16(output, got, ROWS * OUTPUT_DIM));
    CHECK(h3_gpu_tensor_read_bf16(reference, got_ref, ROWS * OUTPUT_DIM));
    CHECK(memcmp(got, got_ref, sizeof(got)) == 0);
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(reference);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_mlp_nax(h3_gpu *gpu) {
    enum { ROWS = 2, INPUT_DIM = 4, HIDDEN = 4, OUTPUT_DIM = 4 };
    enum { INPUT_ELEMS = ROWS * INPUT_DIM, FC1_ELEMS = HIDDEN * 2 * INPUT_DIM,
           FC2_ELEMS = OUTPUT_DIM * HIDDEN };
    uint16_t input_bf16[INPUT_ELEMS], fc1_w_bf16[FC1_ELEMS], fc2_w_bf16[FC2_ELEMS];
    for (size_t i = 0; i < INPUT_ELEMS; i++)
        input_bf16[i] = f32_to_bf16((float)((int)(i % 5) - 2) * 0.25f);
    for (size_t i = 0; i < FC1_ELEMS; i++)
        fc1_w_bf16[i] = f32_to_bf16((float)((int)(i % 7) - 3) * 0.0625f);
    for (size_t i = 0; i < FC2_ELEMS; i++)
        fc2_w_bf16[i] = f32_to_bf16((float)((int)(i % 9) - 4) * 0.03125f);
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(gpu, input_bf16, INPUT_ELEMS);
    h3_gpu_tensor *fc1_w = h3_gpu_tensor_from_bf16(gpu, fc1_w_bf16, FC1_ELEMS);
    h3_gpu_tensor *fc2_w = h3_gpu_tensor_from_bf16(gpu, fc2_w_bf16, FC2_ELEMS);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_bf16(gpu, ROWS * HIDDEN);
    h3_gpu_tensor *reference = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    CHECK(input && fc1_w && fc2_w && activated && reference && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin mlp nax"));
    CHECK(!require_gpu(gpu, h3_gpu_mlp_bf16(
        gpu, reference, input, fc1_w, fc2_w, ROWS, INPUT_DIM, HIDDEN,
        OUTPUT_DIM), "reference mlp"));
    CHECK(!require_gpu(gpu, h3_gpu_mlp_nax_bf16(
        gpu, output, activated, input, fc1_w, fc2_w, ROWS, INPUT_DIM, HIDDEN,
        OUTPUT_DIM), "mlp nax"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit mlp nax"));
    uint16_t got[ROWS * OUTPUT_DIM], got_ref[ROWS * OUTPUT_DIM];
    CHECK(h3_gpu_tensor_read_bf16(output, got, ROWS * OUTPUT_DIM));
    CHECK(h3_gpu_tensor_read_bf16(reference, got_ref, ROWS * OUTPUT_DIM));
    CHECK(memcmp(got, got_ref, sizeof(got)) == 0);
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(fc1_w);
    h3_gpu_tensor_free(fc2_w);
    h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(reference);
    h3_gpu_tensor_free(output);
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

static void cpu_rms_norm_bf16(const uint16_t *input, const uint16_t *weight,
                              uint16_t *output, uint32_t rows, uint32_t width,
                              float epsilon) {
    for (uint32_t row = 0; row < rows; row++) {
        const uint16_t *x = input + row * width;
        float sum = 0.0f;
        for (uint32_t column = 0; column < width; column++) {
            float value = bf16_to_f32(x[column]);
            sum = fmaf(value, value, sum);
        }
        float inverse = 1.0f / sqrtf(sum / (float)width + epsilon);
        for (uint32_t column = 0; column < width; column++) {
            float normalized = bf16_to_f32(x[column]) * inverse;
            output[row * width + column] =
                f32_to_bf16(normalized * bf16_to_f32(weight[column]));
        }
    }
}

static void cpu_layer_norm_bf16(const uint16_t *input, const uint16_t *weight,
                                const uint16_t *bias, uint16_t *output,
                                uint32_t rows, uint32_t width, float epsilon) {
    for (uint32_t row = 0; row < rows; row++) {
        const uint16_t *x = input + row * width;
        float sum = 0.0f;
        for (uint32_t column = 0; column < width; column++)
            sum += bf16_to_f32(x[column]);
        float mean = sum / (float)width;
        float square = 0.0f;
        for (uint32_t column = 0; column < width; column++) {
            float centered = bf16_to_f32(x[column]) - mean;
            square = fmaf(centered, centered, square);
        }
        float inverse = 1.0f / sqrtf(square / (float)width + epsilon);
        for (uint32_t column = 0; column < width; column++) {
            float normalized = (bf16_to_f32(x[column]) - mean) * inverse;
            output[row * width + column] = f32_to_bf16(
                fmaf(normalized, bf16_to_f32(weight[column]),
                     bf16_to_f32(bias[column])));
        }
    }
}

static void cpu_rms_norm_f32(const float *input, const float *weight,
                             float *output, uint32_t rows, uint32_t width,
                             float epsilon) {
    for (uint32_t row = 0; row < rows; row++) {
        const float *x = input + row * width;
        float sum = 0.0f;
        for (uint32_t column = 0; column < width; column++) {
            sum = fmaf(x[column], x[column], sum);
        }
        float inverse = 1.0f / sqrtf(sum / (float)width + epsilon);
        for (uint32_t column = 0; column < width; column++) {
            output[row * width + column] =
                x[column] * inverse * weight[column];
        }
    }
}

static void cpu_layer_norm_f32(const float *input, const float *weight,
                               const float *bias, float *output,
                               uint32_t rows, uint32_t width, float epsilon) {
    for (uint32_t row = 0; row < rows; row++) {
        const float *x = input + row * width;
        float sum = 0.0f;
        for (uint32_t column = 0; column < width; column++) {
            sum += x[column];
        }
        float mean = sum / (float)width;
        float square = 0.0f;
        for (uint32_t column = 0; column < width; column++) {
            float centered = x[column] - mean;
            square = fmaf(centered, centered, square);
        }
        float inverse = 1.0f / sqrtf(square / (float)width + epsilon);
        for (uint32_t column = 0; column < width; column++) {
            float normalized = (x[column] - mean) * inverse;
            output[row * width + column] =
                fmaf(normalized, weight[column], bias[column]);
        }
    }
}

static int test_rms_norm_f32(h3_gpu *gpu) {
    enum { ROWS = 2, WIDTH = 4 };
    const float input[ROWS * WIDTH] = {
        1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 1.5f, 2.5f
    };
    const float weight[WIDTH] = {1.0f, 0.5f, 1.5f, 2.0f};
    float expected[ROWS * WIDTH];
    cpu_rms_norm_f32(input, weight, expected, ROWS, WIDTH, 1e-5f);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_f32(gpu, input, ROWS * WIDTH);
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_from_f32(gpu, weight, WIDTH);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, ROWS * WIDTH);
    CHECK(gpu_input && gpu_weight && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin rms norm f32"));
    CHECK(!require_gpu(gpu, h3_gpu_rms_norm_f32(
        gpu, output, gpu_input, gpu_weight, ROWS, WIDTH, 1e-5f), "rms norm f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit rms norm f32"));
    float got[ROWS * WIDTH];
    CHECK(h3_gpu_tensor_read_f32(output, got, ROWS * WIDTH));
    for (size_t i = 0; i < ROWS * WIDTH; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-4f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_weight);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_layer_norm_f32(h3_gpu *gpu) {
    enum { ROWS = 2, WIDTH = 4 };
    const float input[ROWS * WIDTH] = {
        1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 1.5f, 2.5f
    };
    const float weight[WIDTH] = {1.0f, 0.5f, 1.5f, 2.0f};
    const float bias[WIDTH] = {0.1f, -0.2f, 0.3f, -0.4f};
    float expected[ROWS * WIDTH];
    cpu_layer_norm_f32(input, weight, bias, expected, ROWS, WIDTH, 1e-5f);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_f32(gpu, input, ROWS * WIDTH);
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_from_f32(gpu, weight, WIDTH);
    h3_gpu_tensor *gpu_bias = h3_gpu_tensor_from_f32(gpu, bias, WIDTH);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, ROWS * WIDTH);
    CHECK(gpu_input && gpu_weight && gpu_bias && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin layer norm f32"));
    CHECK(!require_gpu(gpu, h3_gpu_layer_norm_f32(
        gpu, output, gpu_input, gpu_weight, gpu_bias, ROWS, WIDTH, 1e-5f),
        "layer norm f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit layer norm f32"));
    float got[ROWS * WIDTH];
    CHECK(h3_gpu_tensor_read_f32(output, got, ROWS * WIDTH));
    for (size_t i = 0; i < ROWS * WIDTH; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-4f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_weight);
    h3_gpu_tensor_free(gpu_bias);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_rms_norm(h3_gpu *gpu) {
    enum { ROWS = 2, WIDTH = 4 };
    const float input_f[ROWS * WIDTH] = {
        1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 1.5f, 2.5f
    };
    const float weight_f[WIDTH] = {1.0f, 0.5f, 1.5f, 2.0f};
    uint16_t input[ROWS * WIDTH], weight[WIDTH], expected[ROWS * WIDTH];
    for (size_t i = 0; i < ROWS * WIDTH; i++)
        input[i] = f32_to_bf16(input_f[i]);
    for (size_t i = 0; i < WIDTH; i++)
        weight[i] = f32_to_bf16(weight_f[i]);
    cpu_rms_norm_bf16(input, weight, expected, ROWS, WIDTH, 1e-5f);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_bf16(gpu, input, ROWS * WIDTH);
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_from_bf16(gpu, weight, WIDTH);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    CHECK(gpu_input && gpu_weight && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin rms norm"));
    CHECK(!require_gpu(gpu, h3_gpu_rms_norm_bf16(
        gpu, output, gpu_input, gpu_weight, ROWS, WIDTH, 1e-5f), "rms norm"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit rms norm"));
    uint16_t got[ROWS * WIDTH];
    CHECK(h3_gpu_tensor_read_bf16(output, got, ROWS * WIDTH));
    for (size_t i = 0; i < ROWS * WIDTH; i++) {
        CHECK(fabsf(bf16_to_f32(got[i]) - bf16_to_f32(expected[i])) < 0.02f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_weight);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_layer_norm(h3_gpu *gpu) {
    enum { ROWS = 2, WIDTH = 4 };
    const float input_f[ROWS * WIDTH] = {
        1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 1.5f, 2.5f
    };
    const float weight_f[WIDTH] = {1.0f, 0.5f, 1.5f, 2.0f};
    const float bias_f[WIDTH] = {0.1f, -0.2f, 0.3f, -0.4f};
    uint16_t input[ROWS * WIDTH], weight[WIDTH], bias[WIDTH];
    uint16_t expected[ROWS * WIDTH];
    for (size_t i = 0; i < ROWS * WIDTH; i++)
        input[i] = f32_to_bf16(input_f[i]);
    for (size_t i = 0; i < WIDTH; i++) {
        weight[i] = f32_to_bf16(weight_f[i]);
        bias[i] = f32_to_bf16(bias_f[i]);
    }
    cpu_layer_norm_bf16(input, weight, bias, expected, ROWS, WIDTH, 1e-5f);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_bf16(gpu, input, ROWS * WIDTH);
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_from_bf16(gpu, weight, WIDTH);
    h3_gpu_tensor *gpu_bias = h3_gpu_tensor_from_bf16(gpu, bias, WIDTH);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    CHECK(gpu_input && gpu_weight && gpu_bias && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin layer norm"));
    CHECK(!require_gpu(gpu, h3_gpu_layer_norm_bf16(
        gpu, output, gpu_input, gpu_weight, gpu_bias, ROWS, WIDTH, 1e-5f),
        "layer norm"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit layer norm"));
    uint16_t got[ROWS * WIDTH];
    CHECK(h3_gpu_tensor_read_bf16(output, got, ROWS * WIDTH));
    for (size_t i = 0; i < ROWS * WIDTH; i++) {
        CHECK(fabsf(bf16_to_f32(got[i]) - bf16_to_f32(expected[i])) < 0.02f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_weight);
    h3_gpu_tensor_free(gpu_bias);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_grouped_qkv_linear_rope(h3_gpu *gpu) {
    enum { ROWS = 2, INPUT_DIM = 8, HEADS = 2, HEAD_DIM = 4, ROPE_HALF = 2 };
    enum { INNER = HEADS * HEAD_DIM, QKV_DIM = INNER * 3,
           INPUT_ELEMS = ROWS * INPUT_DIM, WEIGHT_ELEMS = QKV_DIM * INPUT_DIM };
    uint16_t input[INPUT_ELEMS], weight[WEIGHT_ELEMS];
    uint16_t q_norm[HEAD_DIM], k_norm[HEAD_DIM];
    uint16_t rope_cos[ROWS * ROPE_HALF], rope_sin[ROWS * ROPE_HALF];
    for (size_t i = 0; i < INPUT_ELEMS; i++)
        input[i] = f32_to_bf16((float)((int)(i % 11) - 5) * 0.125f);
    for (size_t i = 0; i < WEIGHT_ELEMS; i++)
        weight[i] = f32_to_bf16((float)((int)(i % 13) - 6) * 0.03125f);
    for (uint32_t row = 0; row < ROWS; row++) {
        rope_cos[row * ROPE_HALF + 0] = f32_to_bf16(0.6f);
        rope_cos[row * ROPE_HALF + 1] = f32_to_bf16(0.8f);
        rope_sin[row * ROPE_HALF + 0] = f32_to_bf16(0.8f);
        rope_sin[row * ROPE_HALF + 1] = f32_to_bf16(-0.6f);
    }
    for (uint32_t i = 0; i < HEAD_DIM; i++) {
        q_norm[i] = f32_to_bf16(1.0f);
        k_norm[i] = f32_to_bf16(1.0f);
    }
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_bf16(gpu, input, INPUT_ELEMS);
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_from_bf16(gpu, weight, WEIGHT_ELEMS);
    h3_gpu_tensor *gpu_q_norm = h3_gpu_tensor_from_bf16(gpu, q_norm, HEAD_DIM);
    h3_gpu_tensor *gpu_k_norm = h3_gpu_tensor_from_bf16(gpu, k_norm, HEAD_DIM);
    h3_gpu_tensor *gpu_cos = h3_gpu_tensor_from_bf16(gpu, rope_cos, ROWS * ROPE_HALF);
    h3_gpu_tensor *gpu_sin = h3_gpu_tensor_from_bf16(gpu, rope_sin, ROWS * ROPE_HALF);
    h3_gpu_tensor *qkv = h3_gpu_tensor_new_bf16(gpu, ROWS * QKV_DIM);
    h3_gpu_tensor *plain_q = h3_gpu_tensor_new_bf16(gpu, ROWS * INNER);
    h3_gpu_tensor *plain_k = h3_gpu_tensor_new_bf16(gpu, ROWS * INNER);
    h3_gpu_tensor *plain_v = h3_gpu_tensor_new_bf16(gpu, ROWS * INNER);
    h3_gpu_tensor *fused_q = h3_gpu_tensor_new_bf16(gpu, ROWS * INNER);
    h3_gpu_tensor *fused_k = h3_gpu_tensor_new_bf16(gpu, ROWS * INNER);
    h3_gpu_tensor *fused_v = h3_gpu_tensor_new_bf16(gpu, ROWS * INNER);
    CHECK(gpu_input && gpu_weight && gpu_q_norm && gpu_k_norm && gpu_cos &&
          gpu_sin && qkv && plain_q && plain_k && plain_v && fused_q &&
          fused_k && fused_v);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin grouped qkv linear"));
    CHECK(!require_gpu(gpu, h3_gpu_linear_bf16(
        gpu, qkv, gpu_input, gpu_weight, NULL, ROWS, INPUT_DIM, QKV_DIM),
        "grouped qkv linear"));
    CHECK(!require_gpu(gpu, h3_gpu_grouped_qkv_rope_bf16(
        gpu, plain_q, plain_k, plain_v, qkv, gpu_q_norm, gpu_k_norm, gpu_cos,
        gpu_sin, ROWS, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f), "grouped qkv rope"));
    CHECK(!require_gpu(gpu, h3_gpu_grouped_qkv_linear_rope_bf16(
        gpu, fused_q, fused_k, fused_v, qkv, gpu_input, gpu_weight, gpu_q_norm,
        gpu_k_norm, gpu_cos, gpu_sin, ROWS, INPUT_DIM, HEADS, HEAD_DIM,
        ROPE_HALF, 1e-5f), "fused grouped qkv linear rope"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit grouped qkv linear"));
    uint16_t got_plain_q[ROWS * INNER], got_fused_q[ROWS * INNER];
    uint16_t got_plain_k[ROWS * INNER], got_fused_k[ROWS * INNER];
    uint16_t got_plain_v[ROWS * INNER], got_fused_v[ROWS * INNER];
    CHECK(h3_gpu_tensor_read_bf16(plain_q, got_plain_q, ROWS * INNER));
    CHECK(h3_gpu_tensor_read_bf16(fused_q, got_fused_q, ROWS * INNER));
    CHECK(h3_gpu_tensor_read_bf16(plain_k, got_plain_k, ROWS * INNER));
    CHECK(h3_gpu_tensor_read_bf16(fused_k, got_fused_k, ROWS * INNER));
    CHECK(h3_gpu_tensor_read_bf16(plain_v, got_plain_v, ROWS * INNER));
    CHECK(h3_gpu_tensor_read_bf16(fused_v, got_fused_v, ROWS * INNER));
    CHECK(memcmp(got_plain_q, got_fused_q, sizeof(got_plain_q)) == 0);
    CHECK(memcmp(got_plain_k, got_fused_k, sizeof(got_plain_k)) == 0);
    CHECK(memcmp(got_plain_v, got_fused_v, sizeof(got_plain_v)) == 0);
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_weight);
    h3_gpu_tensor_free(gpu_q_norm);
    h3_gpu_tensor_free(gpu_k_norm);
    h3_gpu_tensor_free(gpu_cos);
    h3_gpu_tensor_free(gpu_sin);
    h3_gpu_tensor_free(qkv);
    h3_gpu_tensor_free(plain_q);
    h3_gpu_tensor_free(plain_k);
    h3_gpu_tensor_free(plain_v);
    h3_gpu_tensor_free(fused_q);
    h3_gpu_tensor_free(fused_k);
    h3_gpu_tensor_free(fused_v);
    return 0;
}

static void cpu_gate_bf16(const uint16_t *residual, const uint16_t *branch,
                          const uint16_t *modulation, const uint32_t *row_map,
                          uint16_t *output, uint32_t rows, uint32_t width,
                          uint32_t slots, uint32_t gate_slot) {
    for (uint32_t row = 0; row < rows; row++) {
        uint32_t base = row_map[row] * slots * width;
        for (uint32_t column = 0; column < width; column++) {
            float gate = bf16_to_f32(
                modulation[base + gate_slot * width + column]);
            uint32_t index = row * width + column;
            output[index] = f32_to_bf16(
                bf16_to_f32(residual[index]) + bf16_to_f32(branch[index]) * gate);
        }
    }
}

static void cpu_adaln_bf16(const uint16_t *input, const uint16_t *weight,
                           const uint16_t *modulation, const uint32_t *row_map,
                           uint16_t *output, uint32_t rows, uint32_t width,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon) {
    for (uint32_t row = 0; row < rows; row++) {
        const uint16_t *x = input + row * width;
        float sum = 0.0f;
        for (uint32_t column = 0; column < width; column++) {
            float value = bf16_to_f32(x[column]);
            sum = fmaf(value, value, sum);
        }
        float inverse = 1.0f / sqrtf(sum / (float)width + epsilon);
        uint32_t base = row_map[row] * slots * width;
        for (uint32_t column = 0; column < width; column++) {
            float normalized = bf16_to_f32(x[column]) * inverse *
                bf16_to_f32(weight[column]);
            float shift = bf16_to_f32(
                modulation[base + shift_slot * width + column]);
            float scale = bf16_to_f32(
                modulation[base + scale_slot * width + column]);
            output[row * width + column] =
                f32_to_bf16(normalized * (1.0f + scale) + shift);
        }
    }
}

static void cpu_gate_f32(const float *residual, const float *branch,
                         const float *modulation, const uint32_t *row_map,
                         float *output, uint32_t rows, uint32_t width,
                         uint32_t slots, uint32_t gate_slot) {
    for (uint32_t row = 0; row < rows; row++) {
        uint32_t base = row_map[row] * slots * width;
        for (uint32_t column = 0; column < width; column++) {
            float gate = modulation[base + gate_slot * width + column];
            uint32_t index = row * width + column;
            output[index] = residual[index] + branch[index] * gate;
        }
    }
}

static void cpu_adaln_f32(const float *input, const float *weight,
                          const float *modulation, const uint32_t *row_map,
                          float *output, uint32_t rows, uint32_t width,
                          uint32_t slots, uint32_t shift_slot,
                          uint32_t scale_slot, float epsilon) {
    for (uint32_t row = 0; row < rows; row++) {
        const float *x = input + row * width;
        float sum = 0.0f;
        for (uint32_t column = 0; column < width; column++) {
            sum = fmaf(x[column], x[column], sum);
        }
        float inverse = 1.0f / sqrtf(sum / (float)width + epsilon);
        uint32_t base = row_map[row] * slots * width;
        for (uint32_t column = 0; column < width; column++) {
            float normalized = x[column] * inverse * weight[column];
            float shift = modulation[base + shift_slot * width + column];
            float scale = modulation[base + scale_slot * width + column];
            output[row * width + column] =
                normalized * (1.0f + scale) + shift;
        }
    }
}

static int test_gate_f32(h3_gpu *gpu) {
    enum { ROWS = 2, WIDTH = 4, SLOTS = 2 };
    const float residual[ROWS * WIDTH] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f
    };
    const float branch[ROWS * WIDTH] = {
        0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f
    };
    const float mod[ROWS * SLOTS * WIDTH] = {
        0.5f, 0.5f, 0.5f, 0.5f, 0.25f, 0.25f, 0.25f, 0.25f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    const uint32_t row_map[ROWS] = {0, 1};
    float expected[ROWS * WIDTH];
    cpu_gate_f32(residual, branch, mod, row_map, expected, ROWS, WIDTH, SLOTS, 0);
    h3_gpu_tensor *gpu_residual = h3_gpu_tensor_from_f32(gpu, residual, ROWS * WIDTH);
    h3_gpu_tensor *gpu_branch = h3_gpu_tensor_from_f32(gpu, branch, ROWS * WIDTH);
    h3_gpu_tensor *gpu_mod = h3_gpu_tensor_from_f32(gpu, mod, ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *gpu_row_map = h3_gpu_tensor_from_u32(gpu, row_map, ROWS);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, ROWS * WIDTH);
    CHECK(gpu_residual && gpu_branch && gpu_mod && gpu_row_map && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin gate f32"));
    CHECK(!require_gpu(gpu, h3_gpu_gate_f32(
        gpu, output, gpu_residual, gpu_branch, gpu_mod, gpu_row_map, ROWS, WIDTH,
        SLOTS, 0), "gate f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit gate f32"));
    float got[ROWS * WIDTH];
    CHECK(h3_gpu_tensor_read_f32(output, got, ROWS * WIDTH));
    for (size_t i = 0; i < ROWS * WIDTH; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-5f);
    }
    h3_gpu_tensor_free(gpu_residual);
    h3_gpu_tensor_free(gpu_branch);
    h3_gpu_tensor_free(gpu_mod);
    h3_gpu_tensor_free(gpu_row_map);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_adaln_f32(h3_gpu *gpu) {
    enum { ROWS = 2, WIDTH = 4, SLOTS = 2 };
    const float input[ROWS * WIDTH] = {
        1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f, 1.5f, 2.5f
    };
    const float weight[WIDTH] = {1.0f, 0.5f, 1.5f, 2.0f};
    const float mod[ROWS * SLOTS * WIDTH] = {
        0.25f, -0.5f, 0.75f, -1.0f, 0.0f, 0.5f, 0.25f, -0.25f,
        -0.75f, 0.5f, 0.25f, 1.0f, 0.25f, 0.0f, 0.75f, -0.5f
    };
    const uint32_t row_map[ROWS] = {0, 1};
    float expected[ROWS * WIDTH];
    cpu_adaln_f32(input, weight, mod, row_map, expected, ROWS, WIDTH, SLOTS,
                  0, 1, 1e-5f);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_f32(gpu, input, ROWS * WIDTH);
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_from_f32(gpu, weight, WIDTH);
    h3_gpu_tensor *gpu_mod = h3_gpu_tensor_from_f32(gpu, mod, ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *gpu_row_map = h3_gpu_tensor_from_u32(gpu, row_map, ROWS);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, ROWS * WIDTH);
    CHECK(gpu_input && gpu_weight && gpu_mod && gpu_row_map && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin adaln f32"));
    CHECK(!require_gpu(gpu, h3_gpu_adaln_f32(
        gpu, output, gpu_input, gpu_weight, gpu_mod, gpu_row_map, ROWS, WIDTH,
        SLOTS, 0, 1, 1e-5f), "adaln f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit adaln f32"));
    float got[ROWS * WIDTH];
    CHECK(h3_gpu_tensor_read_f32(output, got, ROWS * WIDTH));
    for (size_t i = 0; i < ROWS * WIDTH; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-4f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_weight);
    h3_gpu_tensor_free(gpu_mod);
    h3_gpu_tensor_free(gpu_row_map);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_gate(h3_gpu *gpu) {
    enum { ROWS = 2, WIDTH = 4, SLOTS = 2 };
    const float residual_f[ROWS * WIDTH] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f
    };
    const float branch_f[ROWS * WIDTH] = {
        0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f
    };
    const float mod_f[ROWS * SLOTS * WIDTH] = {
        0.5f, 0.5f, 0.5f, 0.5f, 0.25f, 0.25f, 0.25f, 0.25f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    uint16_t residual[ROWS * WIDTH], branch[ROWS * WIDTH], mod[ROWS * SLOTS * WIDTH];
    uint16_t expected[ROWS * WIDTH];
    const uint32_t row_map[ROWS] = {0, 1};
    for (size_t i = 0; i < ROWS * WIDTH; i++) {
        residual[i] = f32_to_bf16(residual_f[i]);
        branch[i] = f32_to_bf16(branch_f[i]);
    }
    for (size_t i = 0; i < ROWS * SLOTS * WIDTH; i++)
        mod[i] = f32_to_bf16(mod_f[i]);
    cpu_gate_bf16(residual, branch, mod, row_map, expected, ROWS, WIDTH, SLOTS, 0);
    h3_gpu_tensor *gpu_residual = h3_gpu_tensor_from_bf16(gpu, residual, ROWS * WIDTH);
    h3_gpu_tensor *gpu_branch = h3_gpu_tensor_from_bf16(gpu, branch, ROWS * WIDTH);
    h3_gpu_tensor *gpu_mod = h3_gpu_tensor_from_bf16(gpu, mod, ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *gpu_row_map = h3_gpu_tensor_from_u32(gpu, row_map, ROWS);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    CHECK(gpu_residual && gpu_branch && gpu_mod && gpu_row_map && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin gate"));
    CHECK(!require_gpu(gpu, h3_gpu_gate_bf16(
        gpu, output, gpu_residual, gpu_branch, gpu_mod, gpu_row_map, ROWS, WIDTH,
        SLOTS, 0), "gate"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit gate"));
    uint16_t got[ROWS * WIDTH];
    CHECK(h3_gpu_tensor_read_bf16(output, got, ROWS * WIDTH));
    for (size_t i = 0; i < ROWS * WIDTH; i++) {
        CHECK(fabsf(bf16_to_f32(got[i]) - bf16_to_f32(expected[i])) < 0.02f);
    }
    h3_gpu_tensor_free(gpu_residual);
    h3_gpu_tensor_free(gpu_branch);
    h3_gpu_tensor_free(gpu_mod);
    h3_gpu_tensor_free(gpu_row_map);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_adaln_offset(h3_gpu *gpu) {
    enum { PADDING = 2, ROWS = 2, WIDTH = 4, SLOTS = 2 };
    const float mod_f[ROWS * SLOTS * WIDTH] = {
        0.25f, -0.5f, 0.75f, -1.0f, 0.0f, 0.5f, 0.25f, -0.25f,
        -0.75f, 0.5f, 0.25f, 1.0f, 0.25f, 0.0f, 0.75f, -0.5f
    };
    uint16_t padded[PADDING + ROWS * WIDTH];
    uint16_t norm[WIDTH], mod[ROWS * SLOTS * WIDTH];
    uint16_t expected[ROWS * WIDTH], got_offset[ROWS * WIDTH];
    const uint32_t row_map[ROWS] = {0, 1};
    for (size_t i = 0; i < PADDING; i++)
        padded[i] = f32_to_bf16(-99.0f);
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t col = 0; col < WIDTH; col++) {
            float value = (float)(row + 1) + (float)col * 0.5f;
            padded[PADDING + row * WIDTH + col] = f32_to_bf16(value);
        }
    }
    for (uint32_t i = 0; i < WIDTH; i++)
        norm[i] = f32_to_bf16(1.0f);
    for (size_t i = 0; i < ROWS * SLOTS * WIDTH; i++)
        mod[i] = f32_to_bf16(mod_f[i]);
    cpu_adaln_bf16(padded + PADDING, norm, mod, row_map, expected, ROWS, WIDTH,
                   SLOTS, 0, 1, 1e-5f);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_bf16(
        gpu, padded, PADDING + ROWS * WIDTH);
    h3_gpu_tensor *gpu_norm = h3_gpu_tensor_from_bf16(gpu, norm, WIDTH);
    h3_gpu_tensor *gpu_mod = h3_gpu_tensor_from_bf16(gpu, mod, ROWS * SLOTS * WIDTH);
    h3_gpu_tensor *gpu_row_map = h3_gpu_tensor_from_u32(gpu, row_map, ROWS);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    CHECK(gpu_input && gpu_norm && gpu_mod && gpu_row_map && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin adaln offset"));
    CHECK(!require_gpu(gpu, h3_gpu_adaln_bf16_offset(
        gpu, output, gpu_input, PADDING, gpu_norm, gpu_mod, gpu_row_map, ROWS,
        WIDTH, SLOTS, 0, 1, 1e-5f), "adaln offset"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit adaln offset"));
    CHECK(h3_gpu_tensor_read_bf16(output, got_offset, ROWS * WIDTH));
    for (size_t i = 0; i < ROWS * WIDTH; i++) {
        CHECK(fabsf(bf16_to_f32(got_offset[i]) - bf16_to_f32(expected[i])) < 0.02f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_norm);
    h3_gpu_tensor_free(gpu_mod);
    h3_gpu_tensor_free(gpu_row_map);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_linear_bias(h3_gpu *gpu) {
    enum { ROWS = 2, INPUT_DIM = 4, OUTPUT_DIM = 4 };
    const uint16_t input[ROWS * INPUT_DIM] = {
        0x3f80, 0x0000, 0x4000, 0x0000,
        0x0000, 0x3f80, 0x0000, 0x4000
    };
    const uint16_t weight[OUTPUT_DIM * INPUT_DIM] = {
        0x3f80, 0x0000, 0x0000, 0x3f80,
        0x0000, 0x3f80, 0x3f80, 0x0000,
        0x3f80, 0x3f80, 0x0000, 0x0000,
        0x0000, 0x0000, 0x3f80, 0x3f80
    };
    const uint16_t bias[OUTPUT_DIM] = {0x3f00, 0xbf00, 0x0000, 0x3f80};
    uint16_t expected[ROWS * OUTPUT_DIM];
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t col = 0; col < OUTPUT_DIM; col++) {
            float sum = bf16_to_f32(bias[col]);
            for (uint32_t k = 0; k < INPUT_DIM; k++) {
                sum = fmaf(bf16_to_f32(input[row * INPUT_DIM + k]),
                           bf16_to_f32(weight[col * INPUT_DIM + k]), sum);
            }
            expected[row * OUTPUT_DIM + col] = f32_to_bf16(sum);
        }
    }
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_bf16(gpu, input, ROWS * INPUT_DIM);
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_from_bf16(
        gpu, weight, OUTPUT_DIM * INPUT_DIM);
    h3_gpu_tensor *gpu_bias = h3_gpu_tensor_from_bf16(gpu, bias, OUTPUT_DIM);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    CHECK(gpu_input && gpu_weight && gpu_bias && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin linear bias"));
    CHECK(!require_gpu(gpu, h3_gpu_linear_bf16(
        gpu, output, gpu_input, gpu_weight, gpu_bias, ROWS, INPUT_DIM,
        OUTPUT_DIM), "linear bias"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit linear bias"));
    uint16_t got[ROWS * OUTPUT_DIM];
    CHECK(h3_gpu_tensor_read_bf16(output, got, ROWS * OUTPUT_DIM));
    for (size_t i = 0; i < ROWS * OUTPUT_DIM; i++) {
        CHECK(fabsf(bf16_to_f32(got[i]) - bf16_to_f32(expected[i])) < 0.02f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_weight);
    h3_gpu_tensor_free(gpu_bias);
    h3_gpu_tensor_free(output);
    return 0;
}

static void cpu_text_qk_rope_bf16(
    const uint16_t *query_input, const uint16_t *key_input,
    const uint16_t *q_weight, const uint16_t *k_weight,
    const uint16_t *rope_cos, const uint16_t *rope_sin,
    uint16_t *query_output, uint16_t *key_output, uint32_t sequence,
    uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim,
    float epsilon) {
    uint32_t half_dim = head_dim / 2;
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < query_heads; head++) {
            for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                uint32_t pair = dimension < half_dim ? dimension + half_dim :
                                                       dimension - half_dim;
                float c = bf16_to_f32(
                    rope_cos[row * half_dim + (dimension % half_dim)]);
                float s = bf16_to_f32(
                    rope_sin[row * half_dim + (dimension % half_dim)]);
                uint32_t q_base = (row * query_heads + head) * head_dim;
                float q_sum = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) {
                    float value = bf16_to_f32(query_input[q_base + d]);
                    q_sum = fmaf(value, value, q_sum);
                }
                float q_inverse = 1.0f / sqrtf(q_sum / (float)head_dim + epsilon);
                float q0 = bf16_to_f32(query_input[q_base + dimension]) *
                    q_inverse * bf16_to_f32(q_weight[dimension]);
                float q1 = bf16_to_f32(query_input[q_base + pair]) * q_inverse *
                    bf16_to_f32(q_weight[pair]);
                float q_rotated = dimension < half_dim ? q0 * c - q1 * s :
                                                         q0 * c + q1 * s;
                query_output[q_base + dimension] = f32_to_bf16(q_rotated);
            }
        }
        for (uint32_t head = 0; head < kv_heads; head++) {
            for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                uint32_t pair = dimension < half_dim ? dimension + half_dim :
                                                       dimension - half_dim;
                float c = bf16_to_f32(
                    rope_cos[row * half_dim + (dimension % half_dim)]);
                float s = bf16_to_f32(
                    rope_sin[row * half_dim + (dimension % half_dim)]);
                uint32_t k_base = (row * kv_heads + head) * head_dim;
                float k_sum = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) {
                    float value = bf16_to_f32(key_input[k_base + d]);
                    k_sum = fmaf(value, value, k_sum);
                }
                float k_inverse = 1.0f / sqrtf(k_sum / (float)head_dim + epsilon);
                float k0 = bf16_to_f32(key_input[k_base + dimension]) * k_inverse *
                    bf16_to_f32(k_weight[dimension]);
                float k1 = bf16_to_f32(key_input[k_base + pair]) * k_inverse *
                    bf16_to_f32(k_weight[pair]);
                float k_rotated = dimension < half_dim ? k0 * c - k1 * s :
                                                         k0 * c + k1 * s;
                key_output[k_base + dimension] = f32_to_bf16(k_rotated);
            }
        }
    }
}

static void cpu_gqa_causal_bf16(const uint16_t *query, const uint16_t *key,
                                const uint16_t *value, uint16_t *output,
                                uint32_t sequence, uint32_t query_heads,
                                uint32_t kv_heads, uint32_t head_dim,
                                float scale) {
    for (uint32_t query_row = 0; query_row < sequence; query_row++) {
        for (uint32_t query_head = 0; query_head < query_heads; query_head++) {
            uint32_t kv_head = query_head / (query_heads / kv_heads);
            uint32_t q_base =
                (query_row * query_heads + query_head) * head_dim;
            uint32_t key_count = query_row + 1;
            float scores[512];
            float max_score = -1e30f;
            for (uint32_t key_row = 0; key_row < key_count; key_row++) {
                uint32_t k_base = (key_row * kv_heads + kv_head) * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) {
                    float q = bf16_to_f32(f32_to_bf16(
                        bf16_to_f32(query[q_base + d]) * scale));
                    dot = fmaf(q, bf16_to_f32(key[k_base + d]), dot);
                }
                scores[key_row] = dot;
                max_score = fmaxf(max_score, dot);
            }
            float sum = 0.0f;
            for (uint32_t key_row = 0; key_row < key_count; key_row++) {
                scores[key_row] = expf(scores[key_row] - max_score);
                sum += scores[key_row];
            }
            float inv_sum = 1.0f / sum;
            for (uint32_t d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (uint32_t key_row = 0; key_row < key_count; key_row++) {
                    uint32_t v_index =
                        (key_row * kv_heads + kv_head) * head_dim + d;
                    acc = fmaf(scores[key_row] * inv_sum,
                               bf16_to_f32(value[v_index]), acc);
                }
                output[q_base + d] = f32_to_bf16(acc);
            }
        }
    }
}

static int test_text_qk_rope(h3_gpu *gpu) {
    enum { SEQUENCE = 2, QUERY_HEADS = 4, KV_HEADS = 2, HEAD_DIM = 4 };
    enum { Q_ELEMS = SEQUENCE * QUERY_HEADS * HEAD_DIM,
           K_ELEMS = SEQUENCE * KV_HEADS * HEAD_DIM };
    uint16_t query_in[Q_ELEMS], key_in[K_ELEMS];
    uint16_t q_norm[HEAD_DIM], k_norm[HEAD_DIM];
    uint16_t rope_cos[SEQUENCE * HEAD_DIM / 2], rope_sin[SEQUENCE * HEAD_DIM / 2];
    for (size_t i = 0; i < Q_ELEMS; i++)
        query_in[i] = f32_to_bf16((float)((int)(i % 9) - 4) * 0.25f);
    for (size_t i = 0; i < K_ELEMS; i++)
        key_in[i] = f32_to_bf16((float)((int)(i % 7) - 3) * 0.125f);
    for (uint32_t i = 0; i < HEAD_DIM; i++) {
        q_norm[i] = f32_to_bf16(1.0f);
        k_norm[i] = f32_to_bf16(1.0f);
    }
    for (uint32_t row = 0; row < SEQUENCE; row++) {
        rope_cos[row * 2 + 0] = f32_to_bf16(0.6f);
        rope_cos[row * 2 + 1] = f32_to_bf16(0.8f);
        rope_sin[row * 2 + 0] = f32_to_bf16(0.8f);
        rope_sin[row * 2 + 1] = f32_to_bf16(-0.6f);
    }
    uint16_t expected_q[Q_ELEMS], expected_k[K_ELEMS];
    cpu_text_qk_rope_bf16(query_in, key_in, q_norm, k_norm, rope_cos, rope_sin,
                          expected_q, expected_k, SEQUENCE, QUERY_HEADS,
                          KV_HEADS, HEAD_DIM, 1e-5f);
    h3_gpu_tensor *gpu_q_in = h3_gpu_tensor_from_bf16(gpu, query_in, Q_ELEMS);
    h3_gpu_tensor *gpu_k_in = h3_gpu_tensor_from_bf16(gpu, key_in, K_ELEMS);
    h3_gpu_tensor *gpu_q_norm = h3_gpu_tensor_from_bf16(gpu, q_norm, HEAD_DIM);
    h3_gpu_tensor *gpu_k_norm = h3_gpu_tensor_from_bf16(gpu, k_norm, HEAD_DIM);
    h3_gpu_tensor *gpu_cos = h3_gpu_tensor_from_bf16(gpu, rope_cos, SEQUENCE * 2);
    h3_gpu_tensor *gpu_sin = h3_gpu_tensor_from_bf16(gpu, rope_sin, SEQUENCE * 2);
    h3_gpu_tensor *query_out = h3_gpu_tensor_new_bf16(gpu, Q_ELEMS);
    h3_gpu_tensor *key_out = h3_gpu_tensor_new_bf16(gpu, K_ELEMS);
    CHECK(gpu_q_in && gpu_k_in && gpu_q_norm && gpu_k_norm && gpu_cos &&
          gpu_sin && query_out && key_out);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin text qk rope"));
    CHECK(!require_gpu(gpu, h3_gpu_text_qk_rope_bf16(
        gpu, query_out, key_out, gpu_q_in, gpu_k_in, gpu_q_norm, gpu_k_norm,
        gpu_cos, gpu_sin, SEQUENCE, QUERY_HEADS, KV_HEADS, HEAD_DIM, 1e-5f),
        "text qk rope"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit text qk rope"));
    uint16_t got_q[Q_ELEMS], got_k[K_ELEMS];
    CHECK(h3_gpu_tensor_read_bf16(query_out, got_q, Q_ELEMS));
    CHECK(h3_gpu_tensor_read_bf16(key_out, got_k, K_ELEMS));
    for (size_t i = 0; i < Q_ELEMS; i++)
        CHECK(fabsf(bf16_to_f32(got_q[i]) - bf16_to_f32(expected_q[i])) < 0.03f);
    for (size_t i = 0; i < K_ELEMS; i++)
        CHECK(fabsf(bf16_to_f32(got_k[i]) - bf16_to_f32(expected_k[i])) < 0.03f);
    h3_gpu_tensor_free(gpu_q_in);
    h3_gpu_tensor_free(gpu_k_in);
    h3_gpu_tensor_free(gpu_q_norm);
    h3_gpu_tensor_free(gpu_k_norm);
    h3_gpu_tensor_free(gpu_cos);
    h3_gpu_tensor_free(gpu_sin);
    h3_gpu_tensor_free(query_out);
    h3_gpu_tensor_free(key_out);
    return 0;
}

static int test_gqa_causal(h3_gpu *gpu) {
    enum { SEQUENCE = 4, QUERY_HEADS = 4, KV_HEADS = 2, HEAD_DIM = 4 };
    enum { Q_ELEMS = SEQUENCE * QUERY_HEADS * HEAD_DIM,
           KV_ELEMS = SEQUENCE * KV_HEADS * HEAD_DIM };
    uint16_t query[Q_ELEMS], key[KV_ELEMS], value[KV_ELEMS], expected[Q_ELEMS];
    for (size_t i = 0; i < Q_ELEMS; i++)
        query[i] = f32_to_bf16((float)((int)(i % 11) - 5) * 0.1f);
    for (size_t i = 0; i < KV_ELEMS; i++) {
        key[i] = f32_to_bf16((float)((int)(i % 9) - 4) * 0.15f);
        value[i] = f32_to_bf16((float)((int)(i % 7) - 3) * 0.2f);
    }
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    cpu_gqa_causal_bf16(query, key, value, expected, SEQUENCE, QUERY_HEADS,
                        KV_HEADS, HEAD_DIM, scale);
    h3_gpu_tensor *gpu_q = h3_gpu_tensor_from_bf16(gpu, query, Q_ELEMS);
    h3_gpu_tensor *gpu_k = h3_gpu_tensor_from_bf16(gpu, key, KV_ELEMS);
    h3_gpu_tensor *gpu_v = h3_gpu_tensor_from_bf16(gpu, value, KV_ELEMS);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, Q_ELEMS);
    CHECK(gpu_q && gpu_k && gpu_v && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin gqa causal"));
    CHECK(!require_gpu(gpu, h3_gpu_gqa_causal_bf16(
        gpu, output, gpu_q, gpu_k, gpu_v, SEQUENCE, QUERY_HEADS, KV_HEADS,
        HEAD_DIM, scale), "gqa causal"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit gqa causal"));
    uint16_t got[Q_ELEMS];
    CHECK(h3_gpu_tensor_read_bf16(output, got, Q_ELEMS));
    for (size_t i = 0; i < Q_ELEMS; i++)
        CHECK(fabsf(bf16_to_f32(got[i]) - bf16_to_f32(expected[i])) < 0.05f);
    h3_gpu_tensor_free(gpu_q);
    h3_gpu_tensor_free(gpu_k);
    h3_gpu_tensor_free(gpu_v);
    h3_gpu_tensor_free(output);
    return 0;
}

static void cpu_head_rms_norm_bf16(uint16_t *tensor, const uint16_t *weight,
                                   uint32_t sequence, uint32_t heads,
                                   uint32_t head_dim, float epsilon) {
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < heads; head++) {
            uint32_t base = (row * heads + head) * head_dim;
            float sum = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) {
                float value = bf16_to_f32(tensor[base + d]);
                sum = fmaf(value, value, sum);
            }
            float inverse = 1.0f / sqrtf(sum / (float)head_dim + epsilon);
            for (uint32_t d = 0; d < head_dim; d++) {
                float value = bf16_to_f32(tensor[base + d]);
                tensor[base + d] = f32_to_bf16(
                    value * inverse * bf16_to_f32(weight[d]));
            }
        }
    }
}

static void cpu_rope_text_bf16(uint16_t *query, uint16_t *key,
                               const float *rope_cos, const float *rope_sin,
                               uint32_t sequence, uint32_t query_heads,
                               uint32_t kv_heads, uint32_t head_dim) {
    uint32_t half_dim = head_dim / 2;
    for (uint32_t row = 0; row < sequence; row++) {
        for (uint32_t head = 0; head < query_heads; head++) {
            uint32_t base = (row * query_heads + head) * head_dim;
            for (uint32_t d = 0; d < half_dim; d++) {
                float first = bf16_to_f32(query[base + d]);
                float second = bf16_to_f32(query[base + half_dim + d]);
                float c = rope_cos[row * half_dim + d];
                float s = rope_sin[row * half_dim + d];
                query[base + d] = f32_to_bf16(first * c - second * s);
                query[base + half_dim + d] = f32_to_bf16(second * c + first * s);
            }
        }
        for (uint32_t head = 0; head < kv_heads; head++) {
            uint32_t base = (row * kv_heads + head) * head_dim;
            for (uint32_t d = 0; d < half_dim; d++) {
                float first = bf16_to_f32(key[base + d]);
                float second = bf16_to_f32(key[base + half_dim + d]);
                float c = rope_cos[row * half_dim + d];
                float s = rope_sin[row * half_dim + d];
                key[base + d] = f32_to_bf16(first * c - second * s);
                key[base + half_dim + d] = f32_to_bf16(second * c + first * s);
            }
        }
    }
}

static int test_head_rms_norm(h3_gpu *gpu) {
    enum { SEQUENCE = 2, HEADS = 2, HEAD_DIM = 4 };
    enum { COUNT = SEQUENCE * HEADS * HEAD_DIM };
    uint16_t tensor[COUNT], weight[HEAD_DIM], expected[COUNT];
    for (size_t i = 0; i < COUNT; i++)
        tensor[i] = f32_to_bf16((float)((int)(i % 9) - 4) * 0.25f);
    for (uint32_t i = 0; i < HEAD_DIM; i++)
        weight[i] = f32_to_bf16(0.5f + (float)i * 0.25f);
    memcpy(expected, tensor, sizeof(expected));
    cpu_head_rms_norm_bf16(expected, weight, SEQUENCE, HEADS, HEAD_DIM, 1e-5f);
    h3_gpu_tensor *gpu_tensor = h3_gpu_tensor_from_bf16(gpu, tensor, COUNT);
    h3_gpu_tensor *gpu_weight = h3_gpu_tensor_from_bf16(gpu, weight, HEAD_DIM);
    CHECK(gpu_tensor && gpu_weight);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin head rms norm"));
    CHECK(!require_gpu(gpu, h3_gpu_head_rms_norm_bf16(
        gpu, gpu_tensor, gpu_weight, SEQUENCE, HEADS, HEAD_DIM, 1e-5f),
        "head rms norm"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit head rms norm"));
    uint16_t got[COUNT];
    CHECK(h3_gpu_tensor_read_bf16(gpu_tensor, got, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(bf16_to_f32(got[i]) - bf16_to_f32(expected[i])) < 0.02f);
    }
    h3_gpu_tensor_free(gpu_tensor);
    h3_gpu_tensor_free(gpu_weight);
    return 0;
}

static int test_rope_text(h3_gpu *gpu) {
    enum { SEQUENCE = 2, QUERY_HEADS = 4, KV_HEADS = 2, HEAD_DIM = 4 };
    enum { Q_ELEMS = SEQUENCE * QUERY_HEADS * HEAD_DIM,
           K_ELEMS = SEQUENCE * KV_HEADS * HEAD_DIM };
    uint16_t query[Q_ELEMS], key[K_ELEMS];
    float rope_cos[SEQUENCE * HEAD_DIM / 2], rope_sin[SEQUENCE * HEAD_DIM / 2];
    uint16_t expected_q[Q_ELEMS], expected_k[K_ELEMS];
    for (size_t i = 0; i < Q_ELEMS; i++)
        query[i] = f32_to_bf16((float)((int)(i % 11) - 5) * 0.1f);
    for (size_t i = 0; i < K_ELEMS; i++)
        key[i] = f32_to_bf16((float)((int)(i % 7) - 3) * 0.15f);
    for (uint32_t row = 0; row < SEQUENCE; row++) {
        rope_cos[row * 2 + 0] = 0.6f;
        rope_cos[row * 2 + 1] = 0.8f;
        rope_sin[row * 2 + 0] = 0.8f;
        rope_sin[row * 2 + 1] = -0.6f;
    }
    memcpy(expected_q, query, sizeof(expected_q));
    memcpy(expected_k, key, sizeof(expected_k));
    cpu_rope_text_bf16(expected_q, expected_k, rope_cos, rope_sin, SEQUENCE,
                       QUERY_HEADS, KV_HEADS, HEAD_DIM);
    h3_gpu_tensor *gpu_q = h3_gpu_tensor_from_bf16(gpu, query, Q_ELEMS);
    h3_gpu_tensor *gpu_k = h3_gpu_tensor_from_bf16(gpu, key, K_ELEMS);
    h3_gpu_tensor *gpu_cos = h3_gpu_tensor_from_f32(gpu, rope_cos, SEQUENCE * 2);
    h3_gpu_tensor *gpu_sin = h3_gpu_tensor_from_f32(gpu, rope_sin, SEQUENCE * 2);
    CHECK(gpu_q && gpu_k && gpu_cos && gpu_sin);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin rope text"));
    CHECK(!require_gpu(gpu, h3_gpu_rope_text_bf16(
        gpu, gpu_q, gpu_k, gpu_cos, gpu_sin, SEQUENCE, QUERY_HEADS, KV_HEADS,
        HEAD_DIM), "rope text"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit rope text"));
    uint16_t got_q[Q_ELEMS], got_k[K_ELEMS];
    CHECK(h3_gpu_tensor_read_bf16(gpu_q, got_q, Q_ELEMS));
    CHECK(h3_gpu_tensor_read_bf16(gpu_k, got_k, K_ELEMS));
    for (size_t i = 0; i < Q_ELEMS; i++) {
        CHECK(fabsf(bf16_to_f32(got_q[i]) - bf16_to_f32(expected_q[i])) < 0.02f);
    }
    for (size_t i = 0; i < K_ELEMS; i++) {
        CHECK(fabsf(bf16_to_f32(got_k[i]) - bf16_to_f32(expected_k[i])) < 0.02f);
    }
    h3_gpu_tensor_free(gpu_q);
    h3_gpu_tensor_free(gpu_k);
    h3_gpu_tensor_free(gpu_cos);
    h3_gpu_tensor_free(gpu_sin);
    return 0;
}

static int test_copy_bf16(h3_gpu *gpu) {
    enum { PADDING = 2, COUNT = 4 };
    uint16_t source[PADDING + COUNT];
    for (size_t i = 0; i < PADDING; i++)
        source[i] = f32_to_bf16(-99.0f);
    for (size_t i = 0; i < COUNT; i++)
        source[PADDING + i] = f32_to_bf16((float)(i + 1) * 0.5f);
    h3_gpu_tensor *gpu_source = h3_gpu_tensor_from_bf16(
        gpu, source, PADDING + COUNT);
    h3_gpu_tensor *dest = h3_gpu_tensor_new_bf16(gpu, PADDING + COUNT);
    CHECK(gpu_source && dest);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin copy bf16"));
    CHECK(!require_gpu(gpu, h3_gpu_copy_bf16(
        gpu, dest, PADDING, gpu_source, PADDING, COUNT), "copy bf16"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit copy bf16"));
    uint16_t got[PADDING + COUNT];
    CHECK(h3_gpu_tensor_read_bf16(dest, got, PADDING + COUNT));
    CHECK(memcmp(got + PADDING, source + PADDING, COUNT * sizeof(uint16_t)) == 0);
    h3_gpu_tensor_free(gpu_source);
    h3_gpu_tensor_free(dest);
    return 0;
}

static int test_sub_bf16(h3_gpu *gpu) {
    enum { COUNT = 4 };
    const float left_f[COUNT] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float right_f[COUNT] = {0.5f, 1.5f, 2.5f, 3.5f};
    uint16_t left[COUNT], right[COUNT], expected[COUNT];
    for (size_t i = 0; i < COUNT; i++) {
        left[i] = f32_to_bf16(left_f[i]);
        right[i] = f32_to_bf16(right_f[i]);
        expected[i] = f32_to_bf16(left_f[i] - right_f[i]);
    }
    h3_gpu_tensor *gpu_left = h3_gpu_tensor_from_bf16(gpu, left, COUNT);
    h3_gpu_tensor *gpu_right = h3_gpu_tensor_from_bf16(gpu, right, COUNT);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, COUNT);
    CHECK(gpu_left && gpu_right && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin sub bf16"));
    CHECK(!require_gpu(gpu, h3_gpu_sub_bf16(
        gpu, output, gpu_left, gpu_right, COUNT), "sub bf16"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit sub bf16"));
    uint16_t got[COUNT];
    CHECK(h3_gpu_tensor_read_bf16(output, got, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(bf16_to_f32(got[i]) - bf16_to_f32(expected[i])) < 0.02f);
    }
    h3_gpu_tensor_free(gpu_left);
    h3_gpu_tensor_free(gpu_right);
    h3_gpu_tensor_free(output);
    return 0;
}

static float cpu_gelu_approx(float value) {
    float inner = 0.7978845608028654f *
        (value + 0.044715f * value * value * value);
    if (inner <= -10.0f) return 0.0f;
    if (inner >= 10.0f) return value;
    return 0.5f * value * (1.0f + tanhf(inner));
}

static int test_gelu(h3_gpu *gpu) {
    enum { COUNT = 4 };
    const float values[COUNT] = {-2.0f, -0.5f, 0.0f, 1.5f};
    uint16_t input[COUNT], expected[COUNT];
    for (size_t i = 0; i < COUNT; i++) {
        input[i] = f32_to_bf16(values[i]);
        expected[i] = f32_to_bf16(cpu_gelu_approx(values[i]));
    }
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_bf16(gpu, input, COUNT);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, COUNT);
    CHECK(gpu_input && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin gelu"));
    CHECK(!require_gpu(gpu, h3_gpu_gelu_bf16(gpu, output, gpu_input, COUNT, 1),
                       "gelu"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit gelu"));
    uint16_t got[COUNT];
    CHECK(h3_gpu_tensor_read_bf16(output, got, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(bf16_to_f32(got[i]) - bf16_to_f32(expected[i])) < 0.02f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(output);
    return 0;
}

static void cpu_quantize_weight_int8(const uint16_t *input, int8_t *output,
                                     float *scales, uint32_t rows,
                                     uint32_t columns, float clip) {
    for (uint32_t row = 0; row < rows; row++) {
        float max_abs = 0.0f;
        for (uint32_t column = 0; column < columns; column++) {
            float value = bf16_to_f32(input[row * columns + column]);
            max_abs = fmaxf(max_abs, fabsf(value));
        }
        float clipped_max = max_abs * clip;
        float scale = clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
        float inverse = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
        scales[row] = scale;
        for (uint32_t column = 0; column < columns; column++) {
            int quantized = (int)rintf(
                bf16_to_f32(input[row * columns + column]) * inverse);
            if (quantized > 127) quantized = 127;
            if (quantized < -127) quantized = -127;
            output[row * columns + column] = (int8_t)quantized;
        }
    }
}

static void cpu_quantize_bf16_int8_groups(
    const uint16_t *input, int8_t *output, float *scales, uint32_t rows,
    uint32_t columns, uint32_t group_size) {
    uint32_t groups = columns / group_size;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t group = 0; group < groups; group++) {
            float max_abs = 0.0f;
            uint32_t start = group * group_size;
            for (uint32_t local = 0; local < group_size; local++) {
                float value = bf16_to_f32(input[row * columns + start + local]);
                max_abs = fmaxf(max_abs, fabsf(value));
            }
            float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
            float inverse = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
            scales[row * groups + group] = scale;
            for (uint32_t local = 0; local < group_size; local++) {
                int quantized = (int)rintf(
                    bf16_to_f32(input[row * columns + start + local]) * inverse);
                if (quantized > 127) quantized = 127;
                if (quantized < -127) quantized = -127;
                output[row * columns + start + local] = (int8_t)quantized;
            }
        }
    }
}

extern int h3_hip_quantize_bf16_int8_groups_dispatch(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *scales,
    const h3_gpu_tensor *input, uint32_t rows, uint32_t padded_rows,
    uint32_t columns, uint32_t group_size);

static int test_quantize_bf16_int8_groups(h3_gpu *gpu) {
    enum { ROWS = 2, COLS = 8, GROUP_SIZE = 4, GROUPS = COLS / GROUP_SIZE,
           PADDED_ROWS = 128 };
    enum { COUNT = ROWS * COLS, Q_ELEMS = PADDED_ROWS * COLS,
           SCALE_ELEMS = PADDED_ROWS * GROUPS };
    uint16_t input[COUNT];
    int8_t expected[COUNT];
    float expected_scales[ROWS * GROUPS];
    for (size_t i = 0; i < COUNT; i++)
        input[i] = f32_to_bf16((float)((int)(i % 13) - 6) * 0.125f);
    cpu_quantize_bf16_int8_groups(input, expected, expected_scales, ROWS, COLS,
                                  GROUP_SIZE);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_bf16(gpu, input, COUNT);
    h3_gpu_tensor *output = h3_gpu_tensor_new_i8(gpu, Q_ELEMS);
    h3_gpu_tensor *scales = h3_gpu_tensor_new_f32(gpu, SCALE_ELEMS);
    CHECK(gpu_input && output && scales);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin grouped quantize"));
    CHECK(!require_gpu(gpu, h3_hip_quantize_bf16_int8_groups_dispatch(
        gpu, output, scales, gpu_input, ROWS, PADDED_ROWS, COLS, GROUP_SIZE),
        "grouped quantize"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit grouped quantize"));
    int8_t got[COUNT];
    float got_scales[ROWS * GROUPS];
    CHECK(h3_gpu_tensor_read_i8(output, got, COUNT));
    CHECK(h3_gpu_tensor_read_f32(scales, got_scales, ROWS * GROUPS));
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t group = 0; group < GROUPS; group++) {
            CHECK(fabsf(got_scales[row * GROUPS + group] -
                        expected_scales[row * GROUPS + group]) < 1e-5f);
        }
    }
    CHECK(memcmp(got, expected, sizeof(got)) == 0);
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(scales);
    return 0;
}

static int test_quantize_weight_int8(h3_gpu *gpu) {
    enum { ROWS = 2, COLS = 8 };
    enum { COUNT = ROWS * COLS };
    uint16_t input[COUNT];
    int8_t expected[COUNT];
    float expected_scales[ROWS];
    for (size_t i = 0; i < COUNT; i++)
        input[i] = f32_to_bf16((float)((int)(i % 13) - 6) * 0.125f);
    cpu_quantize_weight_int8(input, expected, expected_scales, ROWS, COLS, 1.0f);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_bf16(gpu, input, COUNT);
    h3_gpu_tensor *output = h3_gpu_tensor_new_i8(gpu, COUNT);
    h3_gpu_tensor *scales = h3_gpu_tensor_new_f32(gpu, ROWS);
    CHECK(gpu_input && output && scales);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin quantize weight"));
    CHECK(!require_gpu(gpu, h3_gpu_quantize_weight_int8(
        gpu, output, scales, gpu_input, ROWS, COLS), "quantize weight"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit quantize weight"));
    int8_t got[COUNT];
    float got_scales[ROWS];
    CHECK(h3_gpu_tensor_read_i8(output, got, COUNT));
    CHECK(h3_gpu_tensor_read_f32(scales, got_scales, ROWS));
    for (uint32_t row = 0; row < ROWS; row++) {
        CHECK(fabsf(got_scales[row] - expected_scales[row]) < 1e-5f);
    }
    CHECK(memcmp(got, expected, sizeof(got)) == 0);
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(scales);
    return 0;
}

static int test_add_bf16(h3_gpu *gpu) {
    enum { COUNT = 4 };
    const float left_f[COUNT] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float right_f[COUNT] = {0.5f, 1.5f, 2.5f, 3.5f};
    uint16_t left[COUNT], right[COUNT], expected[COUNT];
    for (size_t i = 0; i < COUNT; i++) {
        left[i] = f32_to_bf16(left_f[i]);
        right[i] = f32_to_bf16(right_f[i]);
        expected[i] = f32_to_bf16(left_f[i] + right_f[i]);
    }
    h3_gpu_tensor *gpu_left = h3_gpu_tensor_from_bf16(gpu, left, COUNT);
    h3_gpu_tensor *gpu_right = h3_gpu_tensor_from_bf16(gpu, right, COUNT);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, COUNT);
    CHECK(gpu_left && gpu_right && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin add bf16"));
    CHECK(!require_gpu(gpu, h3_gpu_add_bf16(
        gpu, output, gpu_left, gpu_right, COUNT), "add bf16"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit add bf16"));
    uint16_t got[COUNT];
    CHECK(h3_gpu_tensor_read_bf16(output, got, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(bf16_to_f32(got[i]) - bf16_to_f32(expected[i])) < 0.02f);
    }
    h3_gpu_tensor_free(gpu_left);
    h3_gpu_tensor_free(gpu_right);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_linear_int8(h3_gpu *gpu) {
    enum { ROWS = 2, INPUT_DIM = 8, OUTPUT_DIM = 4, PADDED_ROWS = 128 };
    enum { INPUT_ELEMS = ROWS * INPUT_DIM, WEIGHT_ELEMS = OUTPUT_DIM * INPUT_DIM,
           Q_ELEMS = PADDED_ROWS * INPUT_DIM };
    uint16_t input[INPUT_ELEMS], weight_bf16[WEIGHT_ELEMS];
    for (size_t i = 0; i < INPUT_ELEMS; i++)
        input[i] = f32_to_bf16((float)((int)(i % 11) - 5) * 0.125f);
    for (size_t i = 0; i < WEIGHT_ELEMS; i++)
        weight_bf16[i] = f32_to_bf16((float)((int)(i % 9) - 4) * 0.0625f);
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_bf16(gpu, input, INPUT_ELEMS);
    h3_gpu_tensor *gpu_weight_bf16 = h3_gpu_tensor_from_bf16(
        gpu, weight_bf16, WEIGHT_ELEMS);
    h3_gpu_tensor *weight = h3_gpu_tensor_new_i8(gpu, WEIGHT_ELEMS);
    h3_gpu_tensor *weight_scales = h3_gpu_tensor_new_f32(gpu, OUTPUT_DIM);
    h3_gpu_tensor *quantized = h3_gpu_tensor_new_i8(gpu, Q_ELEMS);
    h3_gpu_tensor *input_scales = h3_gpu_tensor_new_f32(gpu, PADDED_ROWS);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    h3_gpu_tensor *reference = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    CHECK(gpu_input && gpu_weight_bf16 && weight && weight_scales &&
          quantized && input_scales && output && reference);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin linear int8 setup"));
    CHECK(!require_gpu(gpu, h3_gpu_quantize_weight_int8(
        gpu, weight, weight_scales, gpu_weight_bf16, OUTPUT_DIM, INPUT_DIM),
        "quantize int8 weight"));
    CHECK(!require_gpu(gpu, h3_gpu_linear_bf16(
        gpu, reference, gpu_input, gpu_weight_bf16, NULL, ROWS, INPUT_DIM,
        OUTPUT_DIM), "reference bf16 linear"));
    CHECK(!require_gpu(gpu, h3_gpu_linear_int8_bf16(
        gpu, output, quantized, input_scales, gpu_input, weight, weight_scales,
        ROWS, INPUT_DIM, OUTPUT_DIM, 0), "linear int8"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit linear int8"));
    uint16_t got[ROWS * OUTPUT_DIM], got_ref[ROWS * OUTPUT_DIM];
    CHECK(h3_gpu_tensor_read_bf16(output, got, ROWS * OUTPUT_DIM));
    CHECK(h3_gpu_tensor_read_bf16(reference, got_ref, ROWS * OUTPUT_DIM));
    for (size_t i = 0; i < ROWS * OUTPUT_DIM; i++) {
        CHECK(fabsf(bf16_to_f32(got[i]) - bf16_to_f32(got_ref[i])) < 0.15f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_weight_bf16);
    h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(weight_scales);
    h3_gpu_tensor_free(quantized);
    h3_gpu_tensor_free(input_scales);
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(reference);
    return 0;
}

static int test_grouped_qkv_linear_rope_int8(h3_gpu *gpu) {
    enum { ROWS = 2, INPUT_DIM = 8, HEADS = 2, HEAD_DIM = 4, ROPE_HALF = 2,
           PADDED_ROWS = 128 };
    enum { INNER = HEADS * HEAD_DIM, QKV_DIM = INNER * 3,
           INPUT_ELEMS = ROWS * INPUT_DIM, WEIGHT_ELEMS = QKV_DIM * INPUT_DIM,
           Q_ELEMS = PADDED_ROWS * INPUT_DIM };
    uint16_t input[INPUT_ELEMS], weight_bf16[WEIGHT_ELEMS];
    uint16_t q_norm[HEAD_DIM], k_norm[HEAD_DIM];
    uint16_t rope_cos[ROWS * ROPE_HALF], rope_sin[ROWS * ROPE_HALF];
    for (size_t i = 0; i < INPUT_ELEMS; i++)
        input[i] = f32_to_bf16((float)((int)(i % 11) - 5) * 0.125f);
    for (size_t i = 0; i < WEIGHT_ELEMS; i++)
        weight_bf16[i] = f32_to_bf16((float)((int)(i % 13) - 6) * 0.03125f);
    for (uint32_t row = 0; row < ROWS; row++) {
        rope_cos[row * ROPE_HALF + 0] = f32_to_bf16(0.6f);
        rope_cos[row * ROPE_HALF + 1] = f32_to_bf16(0.8f);
        rope_sin[row * ROPE_HALF + 0] = f32_to_bf16(0.8f);
        rope_sin[row * ROPE_HALF + 1] = f32_to_bf16(-0.6f);
    }
    for (uint32_t i = 0; i < HEAD_DIM; i++) {
        q_norm[i] = f32_to_bf16(1.0f);
        k_norm[i] = f32_to_bf16(1.0f);
    }
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_bf16(gpu, input, INPUT_ELEMS);
    h3_gpu_tensor *gpu_weight_bf16 = h3_gpu_tensor_from_bf16(
        gpu, weight_bf16, WEIGHT_ELEMS);
    h3_gpu_tensor *gpu_q_norm = h3_gpu_tensor_from_bf16(gpu, q_norm, HEAD_DIM);
    h3_gpu_tensor *gpu_k_norm = h3_gpu_tensor_from_bf16(gpu, k_norm, HEAD_DIM);
    h3_gpu_tensor *gpu_cos = h3_gpu_tensor_from_bf16(gpu, rope_cos, ROWS * ROPE_HALF);
    h3_gpu_tensor *gpu_sin = h3_gpu_tensor_from_bf16(gpu, rope_sin, ROWS * ROPE_HALF);
    h3_gpu_tensor *weight = h3_gpu_tensor_new_i8(gpu, WEIGHT_ELEMS);
    h3_gpu_tensor *weight_scales = h3_gpu_tensor_new_f32(gpu, QKV_DIM);
    h3_gpu_tensor *quantized = h3_gpu_tensor_new_i8(gpu, Q_ELEMS);
    h3_gpu_tensor *input_scales = h3_gpu_tensor_new_f32(gpu, PADDED_ROWS);
    h3_gpu_tensor *qkv = h3_gpu_tensor_new_bf16(gpu, ROWS * QKV_DIM);
    h3_gpu_tensor *ref_q = h3_gpu_tensor_new_bf16(gpu, ROWS * INNER);
    h3_gpu_tensor *ref_k = h3_gpu_tensor_new_bf16(gpu, ROWS * INNER);
    h3_gpu_tensor *ref_v = h3_gpu_tensor_new_bf16(gpu, ROWS * INNER);
    h3_gpu_tensor *got_q = h3_gpu_tensor_new_bf16(gpu, ROWS * INNER);
    h3_gpu_tensor *got_k = h3_gpu_tensor_new_bf16(gpu, ROWS * INNER);
    h3_gpu_tensor *got_v = h3_gpu_tensor_new_bf16(gpu, ROWS * INNER);
    CHECK(gpu_input && gpu_weight_bf16 && gpu_q_norm && gpu_k_norm && gpu_cos &&
          gpu_sin && weight && weight_scales && quantized && input_scales &&
          qkv && ref_q && ref_k && ref_v && got_q && got_k && got_v);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin grouped qkv int8"));
    CHECK(!require_gpu(gpu, h3_gpu_quantize_weight_int8(
        gpu, weight, weight_scales, gpu_weight_bf16, QKV_DIM, INPUT_DIM),
        "quantize grouped qkv int8 weight"));
    CHECK(!require_gpu(gpu, h3_gpu_grouped_qkv_linear_rope_bf16(
        gpu, ref_q, ref_k, ref_v, qkv, gpu_input, gpu_weight_bf16, gpu_q_norm,
        gpu_k_norm, gpu_cos, gpu_sin, ROWS, INPUT_DIM, HEADS, HEAD_DIM,
        ROPE_HALF, 1e-5f), "reference grouped qkv linear rope"));
    CHECK(!require_gpu(gpu, h3_gpu_grouped_qkv_linear_rope_int8(
        gpu, got_q, got_k, got_v, quantized, input_scales, gpu_input, weight,
        weight_scales, gpu_q_norm, gpu_k_norm, gpu_cos, gpu_sin, ROWS,
        INPUT_DIM, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f, 0, 0, 0, 0),
        "grouped qkv linear rope int8"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit grouped qkv int8"));
    uint16_t read_ref_q[ROWS * INNER], read_got_q[ROWS * INNER];
    uint16_t read_ref_k[ROWS * INNER], read_got_k[ROWS * INNER];
    uint16_t read_ref_v[ROWS * INNER], read_got_v[ROWS * INNER];
    CHECK(h3_gpu_tensor_read_bf16(ref_q, read_ref_q, ROWS * INNER));
    CHECK(h3_gpu_tensor_read_bf16(got_q, read_got_q, ROWS * INNER));
    CHECK(h3_gpu_tensor_read_bf16(ref_k, read_ref_k, ROWS * INNER));
    CHECK(h3_gpu_tensor_read_bf16(got_k, read_got_k, ROWS * INNER));
    CHECK(h3_gpu_tensor_read_bf16(ref_v, read_ref_v, ROWS * INNER));
    CHECK(h3_gpu_tensor_read_bf16(got_v, read_got_v, ROWS * INNER));
    for (size_t i = 0; i < ROWS * INNER; i++) {
        CHECK(fabsf(bf16_to_f32(read_got_q[i]) - bf16_to_f32(read_ref_q[i])) <
              0.25f);
        CHECK(fabsf(bf16_to_f32(read_got_k[i]) - bf16_to_f32(read_ref_k[i])) <
              0.25f);
        CHECK(fabsf(bf16_to_f32(read_got_v[i]) - bf16_to_f32(read_ref_v[i])) <
              0.25f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(gpu_weight_bf16);
    h3_gpu_tensor_free(gpu_q_norm);
    h3_gpu_tensor_free(gpu_k_norm);
    h3_gpu_tensor_free(gpu_cos);
    h3_gpu_tensor_free(gpu_sin);
    h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(weight_scales);
    h3_gpu_tensor_free(quantized);
    h3_gpu_tensor_free(input_scales);
    h3_gpu_tensor_free(qkv);
    h3_gpu_tensor_free(ref_q);
    h3_gpu_tensor_free(ref_k);
    h3_gpu_tensor_free(ref_v);
    h3_gpu_tensor_free(got_q);
    h3_gpu_tensor_free(got_k);
    h3_gpu_tensor_free(got_v);
    return 0;
}

static int test_linear_int8_head_major(h3_gpu *gpu) {
    enum { ROWS = 2, HEADS = 2, HEAD_DIM = 4, OUTPUT_DIM = 4,
           INPUT_DIM = HEADS * HEAD_DIM, PADDED_ROWS = 128 };
    enum { HM_ELEMS = HEADS * ROWS * HEAD_DIM,
           WEIGHT_ELEMS = OUTPUT_DIM * INPUT_DIM,
           Q_ELEMS = PADDED_ROWS * INPUT_DIM,
           RM_ELEMS = ROWS * INPUT_DIM };
    const float hm_f[HM_ELEMS] = {
        0.0f, 0.25f, -0.25f, 0.5f, 1.0f, -0.5f, 0.125f, -0.125f,
        0.75f, -1.0f, 0.5f, -0.75f, 0.25f, 0.25f, -0.5f, 1.0f
    };
    uint16_t head_major[HM_ELEMS], row_major[RM_ELEMS], weight_bf16[WEIGHT_ELEMS];
    for (size_t row = 0; row < ROWS; row++) {
        for (size_t head = 0; head < HEADS; head++) {
            for (size_t dim = 0; dim < HEAD_DIM; dim++) {
                size_t hm = (head * ROWS + row) * HEAD_DIM + dim;
                size_t rm = row * INPUT_DIM + head * HEAD_DIM + dim;
                head_major[hm] = f32_to_bf16(hm_f[hm]);
                row_major[rm] = head_major[hm];
            }
        }
    }
    for (size_t i = 0; i < WEIGHT_ELEMS; i++)
        weight_bf16[i] = f32_to_bf16((float)((int)(i % 9) - 4) * 0.0625f);
    h3_gpu_tensor *gpu_head_major = h3_gpu_tensor_from_bf16(
        gpu, head_major, HM_ELEMS);
    h3_gpu_tensor *gpu_row_major = h3_gpu_tensor_from_bf16(
        gpu, row_major, RM_ELEMS);
    h3_gpu_tensor *gpu_weight_bf16 = h3_gpu_tensor_from_bf16(
        gpu, weight_bf16, WEIGHT_ELEMS);
    h3_gpu_tensor *weight = h3_gpu_tensor_new_i8(gpu, WEIGHT_ELEMS);
    h3_gpu_tensor *weight_scales = h3_gpu_tensor_new_f32(gpu, OUTPUT_DIM);
    h3_gpu_tensor *quantized = h3_gpu_tensor_new_i8(gpu, Q_ELEMS);
    h3_gpu_tensor *input_scales = h3_gpu_tensor_new_f32(gpu, PADDED_ROWS);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    h3_gpu_tensor *reference = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    CHECK(gpu_head_major && gpu_row_major && gpu_weight_bf16 && weight &&
          weight_scales && quantized && input_scales && output && reference);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin head-major int8"));
    CHECK(!require_gpu(gpu, h3_gpu_quantize_weight_int8(
        gpu, weight, weight_scales, gpu_weight_bf16, OUTPUT_DIM, INPUT_DIM),
        "quantize head-major int8 weight"));
    CHECK(!require_gpu(gpu, h3_gpu_linear_int8_bf16(
        gpu, reference, quantized, input_scales, gpu_row_major, weight,
        weight_scales, ROWS, INPUT_DIM, OUTPUT_DIM, 0),
        "reference row-major int8 linear"));
    CHECK(!require_gpu(gpu, h3_gpu_linear_int8_head_major_bf16(
        gpu, output, quantized, input_scales, gpu_head_major, weight,
        weight_scales, ROWS, HEADS, HEAD_DIM, OUTPUT_DIM),
        "head-major int8 linear"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit head-major int8"));
    uint16_t got[ROWS * OUTPUT_DIM], got_ref[ROWS * OUTPUT_DIM];
    CHECK(h3_gpu_tensor_read_bf16(output, got, ROWS * OUTPUT_DIM));
    CHECK(h3_gpu_tensor_read_bf16(reference, got_ref, ROWS * OUTPUT_DIM));
    for (size_t i = 0; i < ROWS * OUTPUT_DIM; i++) {
        CHECK(fabsf(bf16_to_f32(got[i]) - bf16_to_f32(got_ref[i])) < 0.15f);
    }
    h3_gpu_tensor_free(gpu_head_major);
    h3_gpu_tensor_free(gpu_row_major);
    h3_gpu_tensor_free(gpu_weight_bf16);
    h3_gpu_tensor_free(weight);
    h3_gpu_tensor_free(weight_scales);
    h3_gpu_tensor_free(quantized);
    h3_gpu_tensor_free(input_scales);
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(reference);
    return 0;
}

static int test_mlp_int8(h3_gpu *gpu) {
    enum { ROWS = 2, INPUT_DIM = 8, HIDDEN = 4, OUTPUT_DIM = 4,
           PADDED_ROWS = 128 };
    enum { INPUT_ELEMS = ROWS * INPUT_DIM,
           FC1_ELEMS = HIDDEN * 2 * INPUT_DIM,
           FC2_ELEMS = OUTPUT_DIM * HIDDEN,
           Q_ELEMS = PADDED_ROWS * INPUT_DIM };
    uint16_t input_bf16[INPUT_ELEMS], fc1_w_bf16[FC1_ELEMS], fc2_w_bf16[FC2_ELEMS];
    for (size_t i = 0; i < INPUT_ELEMS; i++)
        input_bf16[i] = f32_to_bf16((float)((int)(i % 5) - 2) * 0.25f);
    for (size_t i = 0; i < FC1_ELEMS; i++)
        fc1_w_bf16[i] = f32_to_bf16((float)((int)(i % 7) - 3) * 0.0625f);
    for (size_t i = 0; i < FC2_ELEMS; i++)
        fc2_w_bf16[i] = f32_to_bf16((float)((int)(i % 9) - 4) * 0.03125f);
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(gpu, input_bf16, INPUT_ELEMS);
    h3_gpu_tensor *fc1_w = h3_gpu_tensor_from_bf16(gpu, fc1_w_bf16, FC1_ELEMS);
    h3_gpu_tensor *fc2_w = h3_gpu_tensor_from_bf16(gpu, fc2_w_bf16, FC2_ELEMS);
    h3_gpu_tensor *fc1_int8 = h3_gpu_tensor_new_i8(gpu, FC1_ELEMS);
    h3_gpu_tensor *fc1_scales = h3_gpu_tensor_new_f32(gpu, HIDDEN * 2);
    h3_gpu_tensor *fc2_int8 = h3_gpu_tensor_new_i8(gpu, FC2_ELEMS);
    h3_gpu_tensor *fc2_scales = h3_gpu_tensor_new_f32(gpu, OUTPUT_DIM);
    h3_gpu_tensor *quantized = h3_gpu_tensor_new_i8(gpu, Q_ELEMS);
    h3_gpu_tensor *activation_scales = h3_gpu_tensor_new_f32(gpu, PADDED_ROWS);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_bf16(gpu, ROWS * HIDDEN);
    h3_gpu_tensor *reference = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, ROWS * OUTPUT_DIM);
    CHECK(input && fc1_w && fc2_w && fc1_int8 && fc1_scales && fc2_int8 &&
          fc2_scales && quantized && activation_scales && activated &&
          reference && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin mlp int8"));
    CHECK(!require_gpu(gpu, h3_gpu_quantize_weight_int8(
        gpu, fc1_int8, fc1_scales, fc1_w, HIDDEN * 2, INPUT_DIM),
        "quantize mlp fc1"));
    CHECK(!require_gpu(gpu, h3_gpu_quantize_weight_int8(
        gpu, fc2_int8, fc2_scales, fc2_w, OUTPUT_DIM, HIDDEN),
        "quantize mlp fc2"));
    CHECK(!require_gpu(gpu, h3_gpu_mlp_bf16(
        gpu, reference, input, fc1_w, fc2_w, ROWS, INPUT_DIM, HIDDEN,
        OUTPUT_DIM), "reference bf16 mlp"));
    CHECK(!require_gpu(gpu, h3_gpu_mlp_int8_bf16(
        gpu, output, activated, quantized, activation_scales, input,
        fc1_int8, fc1_scales, fc2_int8, fc2_scales, fc1_w, fc2_w, ROWS,
        INPUT_DIM, HIDDEN, OUTPUT_DIM, 0, 0, 0, 0), "mlp int8"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit mlp int8"));
    uint16_t got[ROWS * OUTPUT_DIM], got_ref[ROWS * OUTPUT_DIM];
    CHECK(h3_gpu_tensor_read_bf16(output, got, ROWS * OUTPUT_DIM));
    CHECK(h3_gpu_tensor_read_bf16(reference, got_ref, ROWS * OUTPUT_DIM));
    for (size_t i = 0; i < ROWS * OUTPUT_DIM; i++) {
        CHECK(fabsf(bf16_to_f32(got[i]) - bf16_to_f32(got_ref[i])) < 0.25f);
    }
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(fc1_w);
    h3_gpu_tensor_free(fc2_w);
    h3_gpu_tensor_free(fc1_int8);
    h3_gpu_tensor_free(fc1_scales);
    h3_gpu_tensor_free(fc2_int8);
    h3_gpu_tensor_free(fc2_scales);
    h3_gpu_tensor_free(quantized);
    h3_gpu_tensor_free(activation_scales);
    h3_gpu_tensor_free(activated);
    h3_gpu_tensor_free(reference);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_weight_norm_f32(h3_gpu *gpu) {
    enum { OUTER = 2, INNER = 6 };
    enum { COUNT = OUTER * INNER };
    const float vector[COUNT] = {
        0.2f, -0.1f, 0.3f, 0.4f, -0.5f, 0.6f,
        -0.2f, 0.7f, 0.1f, -0.4f, 0.3f, 0.5f
    };
    const float magnitude[OUTER] = {1.5f, 0.75f};
    float expected[COUNT];
    for (uint32_t outer = 0; outer < OUTER; outer++) {
        float square = 0.0f;
        for (uint32_t index = 0; index < INNER; index++) {
            float value = vector[outer * INNER + index];
            square = fmaf(value, value, square);
        }
        float scale = magnitude[outer] / sqrtf(square);
        for (uint32_t index = 0; index < INNER; index++) {
            expected[outer * INNER + index] =
                vector[outer * INNER + index] * scale;
        }
    }
    h3_gpu_tensor *gpu_vector = h3_gpu_tensor_from_f32(gpu, vector, COUNT);
    h3_gpu_tensor *gpu_magnitude = h3_gpu_tensor_from_f32(gpu, magnitude, OUTER);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, COUNT);
    CHECK(gpu_vector && gpu_magnitude && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin weight norm f32"));
    CHECK(!require_gpu(gpu, h3_gpu_weight_norm_f32(
        gpu, output, gpu_vector, gpu_magnitude, OUTER, INNER), "weight norm f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit weight norm f32"));
    float got[COUNT];
    CHECK(h3_gpu_tensor_read_f32(output, got, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-5f);
    }
    h3_gpu_tensor_free(gpu_vector);
    h3_gpu_tensor_free(gpu_magnitude);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_geglu_f32(h3_gpu *gpu) {
    enum { COUNT = 3 };
    const float gate[COUNT] = {-1.0f, 0.0f, 1.5f};
    const float linear[COUNT] = {2.0f, -1.0f, 0.5f};
    float expected[COUNT];
    for (size_t i = 0; i < COUNT; i++) {
        float x = gate[i];
        float cube = x * x * x;
        float gelu = 0.5f * x *
            (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * cube)));
        expected[i] = gelu * linear[i];
    }
    h3_gpu_tensor *gpu_gate = h3_gpu_tensor_from_f32(gpu, gate, COUNT);
    h3_gpu_tensor *gpu_linear = h3_gpu_tensor_from_f32(gpu, linear, COUNT);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, COUNT);
    CHECK(gpu_gate && gpu_linear && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin geglu f32"));
    CHECK(!require_gpu(gpu, h3_gpu_geglu_f32(gpu, output, gpu_gate, gpu_linear,
                                              COUNT), "geglu f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit geglu f32"));
    float got[COUNT];
    CHECK(h3_gpu_tensor_read_f32(output, got, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-5f);
    }
    h3_gpu_tensor_free(gpu_gate);
    h3_gpu_tensor_free(gpu_linear);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_silu_f32(h3_gpu *gpu) {
    enum { COUNT = 3 };
    const float input[COUNT] = {-1.0f, 0.0f, 2.0f};
    float expected[COUNT];
    for (size_t i = 0; i < COUNT; i++) {
        expected[i] = input[i] / (1.0f + expf(-input[i]));
    }
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_f32(gpu, input, COUNT);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, COUNT);
    CHECK(gpu_input && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin silu f32"));
    CHECK(!require_gpu(gpu, h3_gpu_silu_f32(gpu, output, gpu_input, COUNT),
                       "silu f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit silu f32"));
    float got[COUNT];
    CHECK(h3_gpu_tensor_read_f32(output, got, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-5f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_swiglu_f32(h3_gpu *gpu) {
    enum { ROWS = 2, WIDTH = 3 };
    enum { FUSED = ROWS * WIDTH * 2 };
    const float fused[FUSED] = {
        0.0f, 1.0f, -1.0f, 2.0f, 3.0f, 4.0f,
        -0.5f, 0.5f, 1.5f, 1.0f, -1.0f, 2.0f
    };
    float expected[ROWS * WIDTH];
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t col = 0; col < WIDTH; col++) {
            uint32_t base = row * WIDTH * 2;
            float gate = fused[base + col];
            float up = fused[base + WIDTH + col];
            expected[row * WIDTH + col] =
                gate / (1.0f + expf(-gate)) * up;
        }
    }
    h3_gpu_tensor *gpu_fused = h3_gpu_tensor_from_f32(gpu, fused, FUSED);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, ROWS * WIDTH);
    CHECK(gpu_fused && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin swiglu f32"));
    CHECK(!require_gpu(gpu, h3_gpu_swiglu_f32(gpu, output, gpu_fused, ROWS,
                                                WIDTH), "swiglu f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit swiglu f32"));
    float got[ROWS * WIDTH];
    CHECK(h3_gpu_tensor_read_f32(output, got, ROWS * WIDTH));
    for (size_t i = 0; i < ROWS * WIDTH; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-4f);
    }
    h3_gpu_tensor_free(gpu_fused);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_clip_f32(h3_gpu *gpu) {
    enum { COUNT = 4 };
    const float input[COUNT] = {-2.0f, -0.5f, 0.5f, 2.0f};
    const float expected[COUNT] = {-1.0f, -0.5f, 0.5f, 1.0f};
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_f32(gpu, input, COUNT);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, COUNT);
    CHECK(gpu_input && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin clip f32"));
    CHECK(!require_gpu(gpu, h3_gpu_clip_f32(gpu, output, gpu_input, COUNT,
                                              -1.0f, 1.0f), "clip f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit clip f32"));
    float got[COUNT];
    CHECK(h3_gpu_tensor_read_f32(output, got, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-6f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_add_scaled_f32(h3_gpu *gpu) {
    enum { COUNT = 3 };
    const float left[COUNT] = {1.0f, 2.0f, 3.0f};
    const float right[COUNT] = {4.0f, 5.0f, 6.0f};
    const float expected[COUNT] = {2.5f, 3.5f, 4.5f};
    h3_gpu_tensor *gpu_left = h3_gpu_tensor_from_f32(gpu, left, COUNT);
    h3_gpu_tensor *gpu_right = h3_gpu_tensor_from_f32(gpu, right, COUNT);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, COUNT);
    CHECK(gpu_left && gpu_right && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin add scaled f32"));
    CHECK(!require_gpu(gpu, h3_gpu_add_scaled_f32(
        gpu, output, gpu_left, gpu_right, 0.5f, 0.5f, COUNT), "add scaled f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit add scaled f32"));
    float got[COUNT];
    CHECK(h3_gpu_tensor_read_f32(output, got, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-5f);
    }
    h3_gpu_tensor_free(gpu_left);
    h3_gpu_tensor_free(gpu_right);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_scale_add_f32(h3_gpu *gpu) {
    enum { ROWS = 2, WIDTH = 2 };
    enum { COUNT = ROWS * WIDTH };
    const float residual[COUNT] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float branch[COUNT] = {0.5f, 1.0f, 1.5f, 2.0f};
    const float scale[WIDTH] = {2.0f, 0.5f};
    const float expected[COUNT] = {2.0f, 2.5f, 6.0f, 5.0f};
    h3_gpu_tensor *gpu_residual = h3_gpu_tensor_from_f32(gpu, residual, COUNT);
    h3_gpu_tensor *gpu_branch = h3_gpu_tensor_from_f32(gpu, branch, COUNT);
    h3_gpu_tensor *gpu_scale = h3_gpu_tensor_from_f32(gpu, scale, WIDTH);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, COUNT);
    CHECK(gpu_residual && gpu_branch && gpu_scale && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin scale add f32"));
    CHECK(!require_gpu(gpu, h3_gpu_scale_add_f32(
        gpu, output, gpu_residual, gpu_branch, gpu_scale, ROWS, WIDTH),
        "scale add f32"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit scale add f32"));
    float got[COUNT];
    CHECK(h3_gpu_tensor_read_f32(output, got, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(got[i] - expected[i]) < 1e-5f);
    }
    h3_gpu_tensor_free(gpu_residual);
    h3_gpu_tensor_free(gpu_branch);
    h3_gpu_tensor_free(gpu_scale);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_silu(h3_gpu *gpu) {
    enum { COUNT = 3 };
    const float values[COUNT] = {-1.0f, 0.0f, 2.0f};
    uint16_t input[COUNT], expected[COUNT];
    for (size_t i = 0; i < COUNT; i++) {
        input[i] = f32_to_bf16(values[i]);
        expected[i] = f32_to_bf16(values[i] / (1.0f + expf(-values[i])));
    }
    h3_gpu_tensor *gpu_input = h3_gpu_tensor_from_bf16(gpu, input, COUNT);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, COUNT);
    CHECK(gpu_input && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin silu"));
    CHECK(!require_gpu(gpu, h3_gpu_silu_bf16(gpu, output, gpu_input, COUNT),
                       "silu"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit silu"));
    uint16_t got[COUNT];
    CHECK(h3_gpu_tensor_read_bf16(output, got, COUNT));
    for (size_t i = 0; i < COUNT; i++) {
        CHECK(fabsf(bf16_to_f32(got[i]) - bf16_to_f32(expected[i])) < 0.02f);
    }
    h3_gpu_tensor_free(gpu_input);
    h3_gpu_tensor_free(output);
    return 0;
}

static int test_swiglu(h3_gpu *gpu) {
    enum { ROWS = 2, WIDTH = 3 };
    enum { FUSED = ROWS * WIDTH * 2 };
    const float fused_f[FUSED] = {
        0.0f, 1.0f, -1.0f, 2.0f, 3.0f, 4.0f,
        -0.5f, 0.5f, 1.5f, 1.0f, -1.0f, 2.0f
    };
    uint16_t fused[FUSED], expected[ROWS * WIDTH];
    for (size_t i = 0; i < FUSED; i++)
        fused[i] = f32_to_bf16(fused_f[i]);
    for (uint32_t row = 0; row < ROWS; row++) {
        for (uint32_t col = 0; col < WIDTH; col++) {
            float gate = fused_f[row * WIDTH * 2 + col];
            float up = fused_f[row * WIDTH * 2 + WIDTH + col];
            expected[row * WIDTH + col] =
                f32_to_bf16(gate / (1.0f + expf(-gate)) * up);
        }
    }
    h3_gpu_tensor *gpu_fused = h3_gpu_tensor_from_bf16(gpu, fused, FUSED);
    h3_gpu_tensor *output = h3_gpu_tensor_new_bf16(gpu, ROWS * WIDTH);
    CHECK(gpu_fused && output);
    CHECK(!require_gpu(gpu, h3_gpu_begin(gpu), "begin swiglu"));
    CHECK(!require_gpu(gpu, h3_gpu_swiglu_bf16(gpu, output, gpu_fused, ROWS, WIDTH),
                       "swiglu"));
    CHECK(!require_gpu(gpu, h3_gpu_submit(gpu), "submit swiglu"));
    uint16_t got[ROWS * WIDTH];
    CHECK(h3_gpu_tensor_read_bf16(output, got, ROWS * WIDTH));
    for (size_t i = 0; i < ROWS * WIDTH; i++) {
        CHECK(fabsf(bf16_to_f32(got[i]) - bf16_to_f32(expected[i])) < 0.02f);
    }
    h3_gpu_tensor_free(gpu_fused);
    h3_gpu_tensor_free(output);
    return 0;
}

int main(void) {
    char error[256];
    h3_gpu *gpu = h3_gpu_create("kernels/h3_kernels.hip", error, sizeof(error));
    CHECK(gpu != NULL);
    if (test_euler(gpu) != 0) return 1;
    if (test_token_pool_expand(gpu) != 0) return 1;
    if (test_gate_adaln(gpu) != 0) return 1;
    if (test_gate_adaln_quantize_int8(gpu) != 0) return 1;
    if (test_patch_linear(gpu) != 0) return 1;
    if (test_qkv_rope(gpu) != 0) return 1;
    if (test_qkv_rope_f32(gpu) != 0) return 1;
    if (test_vision_qkv_rope(gpu) != 0) return 1;
    if (test_video_qkv_rope_f32(gpu) != 0) return 1;
    if (test_conv1d_f32(gpu) != 0) return 1;
    if (test_conv1d_stride_f32(gpu) != 0) return 1;
    if (test_conv_transpose1d_f32(gpu) != 0) return 1;
    if (test_alias_free_snake_f32(gpu) != 0) return 1;
    if (test_snake1d_f32(gpu) != 0) return 1;
    if (test_audio_qkv_split_f32(gpu) != 0) return 1;
    if (test_sdpa_causal_f32(gpu) != 0) return 1;
    if (test_audio_attention_pool_f32(gpu) != 0) return 1;
    if (test_vae_encoder_pad_f32(gpu) != 0) return 1;
    if (test_conv3d_f32(gpu) != 0) return 1;
    if (test_sdpa(gpu) != 0) return 1;
    if (test_sdpa_f32(gpu) != 0) return 1;
    if (test_cast(gpu) != 0) return 1;
    if (test_mlp(gpu) != 0) return 1;
    if (test_mlp_nax(gpu) != 0) return 1;
    if (test_fc1_swiglu_nax(gpu) != 0) return 1;
    if (test_linear_bf16_nax(gpu) != 0) return 1;
    if (test_linear_f32(gpu) != 0) return 1;
    if (test_silu_f32(gpu) != 0) return 1;
    if (test_swiglu_f32(gpu) != 0) return 1;
    if (test_clip_f32(gpu) != 0) return 1;
    if (test_add_scaled_f32(gpu) != 0) return 1;
    if (test_scale_add_f32(gpu) != 0) return 1;
    if (test_weight_norm_f32(gpu) != 0) return 1;
    if (test_geglu_f32(gpu) != 0) return 1;
    if (test_adaln_linear(gpu) != 0) return 1;
    if (test_embedding(gpu) != 0) return 1;
    if (test_silu_mul(gpu) != 0) return 1;
    if (test_grouped_qkv_rope(gpu) != 0) return 1;
    if (test_rms_norm(gpu) != 0) return 1;
    if (test_rms_norm_f32(gpu) != 0) return 1;
    if (test_layer_norm(gpu) != 0) return 1;
    if (test_layer_norm_f32(gpu) != 0) return 1;
    if (test_grouped_qkv_linear_rope(gpu) != 0) return 1;
    if (test_gate(gpu) != 0) return 1;
    if (test_gate_f32(gpu) != 0) return 1;
    if (test_adaln_offset(gpu) != 0) return 1;
    if (test_adaln_f32(gpu) != 0) return 1;
    if (test_linear_bias(gpu) != 0) return 1;
    if (test_text_qk_rope(gpu) != 0) return 1;
    if (test_gqa_causal(gpu) != 0) return 1;
    if (test_head_rms_norm(gpu) != 0) return 1;
    if (test_rope_text(gpu) != 0) return 1;
    if (test_copy_bf16(gpu) != 0) return 1;
    if (test_sub_bf16(gpu) != 0) return 1;
    if (test_gelu(gpu) != 0) return 1;
    if (test_quantize_weight_int8(gpu) != 0) return 1;
    if (test_quantize_bf16_int8_groups(gpu) != 0) return 1;
    if (test_add_bf16(gpu) != 0) return 1;
    if (test_linear_int8(gpu) != 0) return 1;
    if (test_linear_int8_head_major(gpu) != 0) return 1;
    if (test_mlp_int8(gpu) != 0) return 1;
    if (test_grouped_qkv_linear_rope_int8(gpu) != 0) return 1;
    if (test_silu(gpu) != 0) return 1;
    if (test_swiglu(gpu) != 0) return 1;
    h3_gpu_free(gpu);
    puts("h3_hip_bf16_tests ok");
    return 0;
}
