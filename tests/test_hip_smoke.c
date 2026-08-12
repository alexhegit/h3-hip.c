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
