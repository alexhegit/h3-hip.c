#include "h3_gpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hip/hip_runtime_api.h>

static void die(const char *msg) {
    fprintf(stderr, "FAIL %s\n", msg);
    exit(1);
}

static void cpu_sdpa_f32(const float *query, const float *key,
                         const float *value, float *output,
                         uint32_t sequence, uint32_t heads, uint32_t head_dim,
                         float scale) {
    float *scores = malloc((size_t)sequence * sizeof(*scores));
    if (!scores) die("oom");
    for (uint32_t head = 0; head < heads; head++) {
        for (uint32_t q_pos = 0; q_pos < sequence; q_pos++) {
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
                output[(q_pos * heads + head) * head_dim + d] = acc;
            }
        }
    }
    free(scores);
}

static int run_case(h3_gpu *gpu, uint32_t sequence, uint32_t heads,
                    uint32_t head_dim, const char *label) {
    size_t count = (size_t)sequence * heads * head_dim;
    float *query = calloc(count, sizeof(*query));
    float *key = calloc(count, sizeof(*key));
    float *value = calloc(count, sizeof(*value));
    float *expected = calloc(count, sizeof(*expected));
    float *got = calloc(count, sizeof(*got));
    if (!query || !key || !value || !expected || !got) die("oom");
    for (uint32_t pos = 0; pos < sequence; pos++) {
        for (uint32_t head = 0; head < heads; head++) {
            for (uint32_t d = 0; d < head_dim; d++) {
                uint32_t index = (pos * heads + head) * head_dim + d;
                query[index] = (float)((int)((pos + head + d) % 13) - 6) * 0.03125f;
                key[index] = (float)((int)((pos * 3 + d) % 11) - 5) * 0.015625f;
                value[index] = (float)((int)((head + d + pos) % 7) - 3) * 0.0625f;
            }
        }
    }
    float scale = 1.0f / sqrtf((float)head_dim);
    cpu_sdpa_f32(query, key, value, expected, sequence, heads, head_dim, scale);
    h3_gpu_tensor *gpu_q = h3_gpu_tensor_from_f32(gpu, query, count);
    h3_gpu_tensor *gpu_k = h3_gpu_tensor_from_f32(gpu, key, count);
    h3_gpu_tensor *gpu_v = h3_gpu_tensor_from_f32(gpu, value, count);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(gpu, count);
    if (!gpu_q || !gpu_k || !gpu_v || !output) die("tensor alloc");
    if (!h3_gpu_begin(gpu) ||
        !h3_gpu_sdpa_f32(gpu, output, gpu_q, gpu_k, gpu_v, sequence, heads,
                         head_dim, scale) ||
        !h3_gpu_submit(gpu)) {
        die(h3_gpu_error(gpu));
    }
    if (!h3_gpu_tensor_read_f32(output, got, count)) die("read");
    double max_abs = 0.0, num = 0.0, den = 0.0;
    size_t nans = 0;
    for (size_t i = 0; i < count; i++) {
        if (got[i] != got[i] || expected[i] != expected[i]) nans++;
        double d = fabs((double)got[i] - (double)expected[i]);
        if (d > max_abs) max_abs = d;
        num += d * d;
        den += (double)expected[i] * (double)expected[i];
    }
    double rel = den > 0.0 ? sqrt(num / den) : sqrt(num);
    printf("%s seq=%u heads=%u dim=%u max_abs=%.4g rel_l2=%.4g nans=%zu\n",
           label, sequence, heads, head_dim, max_abs, rel, nans);
    int ok = nans == 0 && max_abs < 2e-3 && rel < 1e-3;
    h3_gpu_tensor_free(gpu_q);
    h3_gpu_tensor_free(gpu_k);
    h3_gpu_tensor_free(gpu_v);
    h3_gpu_tensor_free(output);
    free(query); free(key); free(value); free(expected); free(got);
    return ok ? 0 : 1;
}

int main(void) {
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, 0) == hipSuccess) {
        printf("device %s arch %s warpSize %d maxGrid %d %d %d maxThreads %d\n",
               props.name, props.gcnArchName, props.warpSize,
               props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2],
               props.maxThreadsPerBlock);
    }
    char error[256];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof(error));
    if (!gpu) die(error);
    int failed = 0;
    failed |= run_case(gpu, 16, 2, 64, "wave-or-legacy");
    failed |= run_case(gpu, 64, 4, 64, "wave-or-legacy");
    failed |= run_case(gpu, 256, 2, 64, "wave-or-legacy");
    failed |= run_case(gpu, 512, 2, 64, "wave-or-legacy");
    h3_gpu_free(gpu);
    return failed;
}
