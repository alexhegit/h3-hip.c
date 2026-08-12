#include "h3_gpu.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

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

int main(void) {
    char error[256];
    h3_gpu *gpu = h3_gpu_create("kernels/h3_kernels.hip", error, sizeof(error));
    CHECK(gpu != NULL);
    if (test_euler(gpu) != 0) return 1;
    if (test_token_pool_expand(gpu) != 0) return 1;
    if (test_gate_adaln(gpu) != 0) return 1;
    h3_gpu_free(gpu);
    puts("h3_hip_bf16_tests ok");
    return 0;
}
