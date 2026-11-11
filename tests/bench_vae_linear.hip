/* Video VAE linear shapes through the production launchers: times each
 * H3_F32_WMMA_PIPE value and checks its output is bitwise the default's.
 * Build: make HIP_ARCH=... h3_bench_vae_linear */
#include <hip/hip_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
#include "kernels/h3_kernels.h"
}

#define HIP_CHECK(call) do {                                                  \
    hipError_t status_ = (call);                                              \
    if (status_ != hipSuccess) {                                              \
        fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(status_));      \
        exit(1);                                                              \
    }                                                                         \
} while (0)

static uint16_t random_bf16(uint32_t *state, float scale) {
    *state = *state * 1664525u + 1013904223u;
    float value = ((float)(*state >> 8) / 16777216.0f - 0.5f) * scale;
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

static int launch(int bf16_input, const void *input, const uint16_t *weight,
                  const float *bias, float *output, const h3_linear_args *args) {
    return bf16_input
        ? h3_launch_linear_bf16a_wbf16((const uint16_t *)input, weight, bias,
                                       output, args, 0)
        : h3_launch_linear_f32_wbf16((const float *)input, weight, bias,
                                     output, args, 0);
}

int main(int argc, char **argv) {
    const struct { uint32_t m, k, n; int bf16_input; const char *name; } shapes[] = {
        {2028, 2048, 6144, 1, "qkv"},
        {2028, 2048, 2048, 0, "attn-out"},
        {2028, 2048, 16384, 1, "fc1"},
        {2028, 8192, 2048, 1, "fc2"},
        {300, 2048, 2048, 1, "short-rows"},
    };
    const char *pipes[] = {"0", "1", "2"};
    const char *configs[] = {"3", "1", "4", "5", "6", "9"};
    int iterations = argc > 1 ? atoi(argv[1]) : 20;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(*shapes); s++) {
        uint32_t m = shapes[s].m, k = shapes[s].k, n = shapes[s].n;
        size_t in_n = (size_t)m * k, w_n = (size_t)n * k, out_n = (size_t)m * n;
        uint32_t state = 12345u + (uint32_t)s;
        uint16_t *host_in16 = (uint16_t *)malloc(in_n * 2);
        float *host_in32 = (float *)malloc(in_n * 4);
        uint16_t *host_w = (uint16_t *)malloc(w_n * 2);
        float *host_b = (float *)malloc(n * 4);
        float *reference = (float *)malloc(out_n * 4);
        float *result = (float *)malloc(out_n * 4);
        for (size_t i = 0; i < in_n; i++) {
            host_in16[i] = random_bf16(&state, 4.0f);
            uint32_t bits = (uint32_t)host_in16[i] << 16;
            memcpy(&host_in32[i], &bits, 4);
            host_in32[i] *= 1.0f + 1e-3f * (float)(i % 7);
        }
        for (size_t i = 0; i < w_n; i++) host_w[i] = random_bf16(&state, 0.1f);
        for (uint32_t i = 0; i < n; i++) host_b[i] = 0.01f * (float)(i % 13);
        void *in;
        uint16_t *w;
        float *b, *out;
        size_t in_bytes = shapes[s].bf16_input ? in_n * 2 : in_n * 4;
        HIP_CHECK(hipMalloc(&in, in_bytes));
        HIP_CHECK(hipMalloc((void **)&w, w_n * 2));
        HIP_CHECK(hipMalloc((void **)&b, n * 4));
        HIP_CHECK(hipMalloc((void **)&out, out_n * 4));
        HIP_CHECK(hipMemcpy(in, shapes[s].bf16_input ? (void *)host_in16
                                                      : (void *)host_in32,
                            in_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(w, host_w, w_n * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(b, host_b, n * 4, hipMemcpyHostToDevice));
        h3_linear_args args = {m, k, n, 1u};
        double flop = 2.0 * m * (double)k * n;
        for (size_t c = 0; c < sizeof(configs) / sizeof(*configs); c++)
        for (size_t p = 0; p < sizeof(pipes) / sizeof(*pipes); p++) {
            setenv("H3_F32_WMMA_CFG", configs[c], 1);
            setenv("H3_F32_WMMA_PIPE", pipes[p], 1);
            int first = !c && !p;
            HIP_CHECK(hipMemset(out, 0xff, out_n * 4));
            if (!launch(shapes[s].bf16_input, in, w, b, out, &args)) {
                fprintf(stderr, "launch failed\n");
                return 1;
            }
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(first ? reference : result, out, out_n * 4,
                                hipMemcpyDeviceToHost));
            int same = first || !memcmp(result, reference, out_n * 4);
            hipEvent_t start, stop;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));
            for (int i = 0; i < 3; i++)
                launch(shapes[s].bf16_input, in, w, b, out, &args);
            HIP_CHECK(hipEventRecord(start, 0));
            for (int i = 0; i < iterations; i++)
                launch(shapes[s].bf16_input, in, w, b, out, &args);
            HIP_CHECK(hipEventRecord(stop, 0));
            HIP_CHECK(hipEventSynchronize(stop));
            float ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
            ms /= (float)iterations;
            printf("%-10s M=%u K=%u N=%u cfg=%s pipe=%s %8.3f ms %6.1f "
                   "TFLOP/s %s\n", shapes[s].name, m, k, n, configs[c],
                   pipes[p], ms, flop / (ms * 1e-3) / 1e12,
                   first ? "reference" :
                   (same ? "bitwise-same" : "DIFFERENT"));
            HIP_CHECK(hipEventDestroy(start));
            HIP_CHECK(hipEventDestroy(stop));
        }
        HIP_CHECK(hipFree(in));
        HIP_CHECK(hipFree(w));
        HIP_CHECK(hipFree(b));
        HIP_CHECK(hipFree(out));
        free(host_in16);
        free(host_in32);
        free(host_w);
        free(host_b);
        free(reference);
        free(result);
    }
    return 0;
}
