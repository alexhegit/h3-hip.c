#include "h3_gpu.h"

#include <stdio.h>
#include <string.h>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

int main(void) {
    char error[256];
    h3_gpu *gpu = h3_gpu_create("kernels/h3_kernels.hip", error, sizeof(error));
    CHECK(gpu != NULL);

    const float left_f[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float right_f[4] = {0.5f, 1.5f, 2.5f, 3.5f};
    h3_gpu_tensor *left = h3_gpu_tensor_from_f32(gpu, left_f, 4);
    h3_gpu_tensor *right = h3_gpu_tensor_from_f32(gpu, right_f, 4);
    h3_gpu_tensor *left_bf16 = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *right_bf16 = h3_gpu_tensor_new_bf16(gpu, 4);
    h3_gpu_tensor *sum = h3_gpu_tensor_new_bf16(gpu, 4);
    CHECK(left && right && left_bf16 && right_bf16 && sum);

    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_cast_f32_to_bf16(gpu, left_bf16, left, 4));
    CHECK(h3_gpu_cast_f32_to_bf16(gpu, right_bf16, right, 4));
    CHECK(h3_gpu_add_bf16(gpu, sum, left_bf16, right_bf16, 4));
    CHECK(h3_gpu_submit(gpu));

    float out[4];
    h3_gpu_tensor *out_tensor = h3_gpu_tensor_new_f32(gpu, 4);
    CHECK(out_tensor != NULL);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_cast_bf16_to_f32(gpu, out_tensor, sum, 4));
    CHECK(h3_gpu_submit(gpu));
    CHECK(h3_gpu_tensor_read_f32(out_tensor, out, 4));

    for (int i = 0; i < 4; i++) {
        float expected = left_f[i] + right_f[i];
        if (out[i] < expected - 0.05f || out[i] > expected + 0.05f) {
            fprintf(stderr, "unexpected bf16 add at %d: got %f expected %f\n",
                    i, out[i], expected);
            return 1;
        }
    }

    const uint16_t silu_in[2] = {0x0000, 0x3f80};
    h3_gpu_tensor *silu_input = h3_gpu_tensor_from_bf16(gpu, silu_in, 2);
    h3_gpu_tensor *silu_output = h3_gpu_tensor_new_bf16(gpu, 2);
    CHECK(silu_input && silu_output);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_silu_bf16(gpu, silu_output, silu_input, 2));
    CHECK(h3_gpu_submit(gpu));
    uint16_t silu_bf16[2];
    CHECK(h3_gpu_tensor_read_bf16(silu_output, silu_bf16, 2));
    CHECK(silu_bf16[0] == 0x0000);

    const uint16_t linear_input[4] = {0x3f80, 0x0000, 0x4000, 0x0000};
    const uint16_t linear_weight[8] = {
        0x3f80, 0x0000, 0x0000, 0x3f80,
        0x0000, 0x3f80, 0x3f80, 0x0000
    };
    h3_gpu_tensor *lin_in = h3_gpu_tensor_from_bf16(gpu, linear_input, 4);
    h3_gpu_tensor *lin_w = h3_gpu_tensor_from_bf16(gpu, linear_weight, 8);
    h3_gpu_tensor *lin_out = h3_gpu_tensor_new_bf16(gpu, 4);
    CHECK(lin_in && lin_w && lin_out);
    CHECK(h3_gpu_begin(gpu));
    CHECK(h3_gpu_linear_bf16(gpu, lin_out, lin_in, lin_w, NULL, 2, 2, 2));
    CHECK(h3_gpu_submit(gpu));
    uint16_t linear_out[4];
    CHECK(h3_gpu_tensor_read_bf16(lin_out, linear_out, 4));
    CHECK(linear_out[0] == 0x3f80);
    CHECK(linear_out[2] == 0x4000);

    h3_gpu_tensor_free(silu_input);
    h3_gpu_tensor_free(silu_output);
    h3_gpu_tensor_free(lin_in);
    h3_gpu_tensor_free(lin_w);
    h3_gpu_tensor_free(lin_out);

    h3_gpu_tensor_free(left);
    h3_gpu_tensor_free(right);
    h3_gpu_tensor_free(left_bf16);
    h3_gpu_tensor_free(right_bf16);
    h3_gpu_tensor_free(sum);
    h3_gpu_tensor_free(out_tensor);
    h3_gpu_free(gpu);
    puts("h3_hip_smoke ok");
    return 0;
}
