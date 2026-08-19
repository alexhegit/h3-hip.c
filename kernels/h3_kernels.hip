#include "h3_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

__device__ __host__ inline float h3_bf16_to_f32(uint16_t value) {
    /* Exact bf16→f32: place mantissa/exp in the high half of an f32. Avoids
     * a library convert on the SDPA/GEMM hot paths. */
#if defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float((uint32_t)value << 16);
#else
    uint32_t bits = (uint32_t)value << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
#endif
}

__device__ __host__ inline uint16_t h3_f32_to_bf16(float value) {
    __hip_bfloat16 bf16 = __float2bfloat16(value);
    return *reinterpret_cast<uint16_t *>(&bf16);
}

__device__ inline float h3_erf_approx(float value) {
    float sign = value < 0.0f ? -1.0f : 1.0f;
    float x = fabsf(value);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = (((((1.061405429f * t - 1.453152027f) * t) +
                           1.421413741f) * t - 0.284496736f) * t +
                           0.254829592f) * t;
    return sign * (1.0f - polynomial * expf(-x * x));
}

__global__ void h3_copy_f32_kernel(const float *input, float *output,
                                   uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) output[gid] = input[gid];
}

__global__ void h3_copy_u16_kernel(const uint16_t *input, uint16_t *output,
                                   uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) output[gid] = input[gid];
}

__global__ void h3_cast_f32_to_bf16_kernel(const float *input, uint16_t *output,
                                           uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) {
        output[gid] = h3_f32_to_bf16(input[gid]);
    }
}

__global__ void h3_cast_bf16_to_f32_kernel(const uint16_t *input, float *output,
                                           uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) {
        output[gid] = h3_bf16_to_f32(input[gid]);
    }
}

__global__ void h3_add_bf16_kernel(const uint16_t *left, const uint16_t *right,
                                   uint16_t *output, uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) {
        output[gid] = h3_f32_to_bf16(h3_bf16_to_f32(left[gid]) +
                                     h3_bf16_to_f32(right[gid]));
    }
}

__global__ void h3_sub_bf16_kernel(const uint16_t *left, const uint16_t *right,
                                   uint16_t *output, uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) {
        output[gid] = h3_f32_to_bf16(h3_bf16_to_f32(left[gid]) -
                                     h3_bf16_to_f32(right[gid]));
    }
}

__global__ void h3_silu_bf16_kernel(const uint16_t *input, uint16_t *output,
                                    uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) {
        float value = h3_bf16_to_f32(input[gid]);
        output[gid] = h3_f32_to_bf16(value / (1.0f + expf(-value)));
    }
}

__global__ void h3_silu_f32_kernel(const float *input, float *output,
                                   uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) {
        float value = input[gid];
        output[gid] = value / (1.0f + expf(-value));
    }
}

__global__ void h3_gelu_bf16_kernel(const uint16_t *input, uint16_t *output,
                                    h3_gelu_bf16_args args) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= args.elements) return;
    float value = h3_bf16_to_f32(input[index]);
    float activated;
    if (args.approximate) {
        float inner = 0.7978845608028654f *
            (value + 0.044715f * value * value * value);
        activated = inner <= -10.0f ? 0.0f :
                    inner >= 10.0f ? value :
                    0.5f * value * (1.0f + tanhf(inner));
    } else {
        activated = value <= -10.0f ? 0.0f :
                    value >= 10.0f ? value :
                    0.5f * value *
                    (1.0f + h3_erf_approx(value * 0.7071067811865475f));
    }
    output[index] = h3_f32_to_bf16(activated);
}

__global__ void h3_swiglu_bf16_kernel(const uint16_t *fused, uint16_t *output,
                                      h3_swiglu_args args) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || column >= args.width) return;
    uint32_t base = row * args.width * 2;
    float gate = h3_bf16_to_f32(fused[base + column]);
    float up = h3_bf16_to_f32(fused[base + args.width + column]);
    output[row * args.width + column] =
        h3_f32_to_bf16(gate / (1.0f + expf(-gate)) * up);
}

__global__ void h3_swiglu_f32_kernel(const float *fused, float *output,
                                     h3_swiglu_args args) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || column >= args.width) return;
    uint32_t base = row * args.width * 2;
    float gate = fused[base + column];
    float up = fused[base + args.width + column];
    output[row * args.width + column] =
        gate / (1.0f + expf(-gate)) * up;
}

__global__ void h3_clip_f32_kernel(const float *input, float *output,
                                   h3_clip_f32_args args) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= args.elements) return;
    float value = input[index];
    output[index] = fminf(fmaxf(value, args.minimum), args.maximum);
}

__global__ void h3_alias_free_snake_f32_kernel(
    const float *input, const float *alpha_log, const float *beta_log,
    const float *upsample_filter, const float *downsample_filter, float *output,
    h3_audio_activation_args args) {
    uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t time = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t batch = blockIdx.z;
    if (channel >= args.channels || time >= args.length || batch >= args.batch)
        return;
    float alpha = expf(alpha_log[channel]);
    float beta = expf(beta_log[channel]);
    float result = 0.0f;
    for (int down_k = 0; down_k < 12; down_k++) {
        int up_time = (int)time * 2 + down_k - 5;
        up_time = max(up_time, 0);
        up_time = min(up_time, (int)args.length * 2 - 1);
        int raw_time = up_time + 15;
        float upsampled = 0.0f;
        for (int up_k = 0; up_k < 12; up_k++) {
            int numerator = raw_time - up_k;
            if (numerator < 0 || (numerator & 1)) continue;
            int padded_time = numerator / 2;
            int source_time = padded_time - 5;
            source_time = max(source_time, 0);
            source_time = min(source_time, (int)args.length - 1);
            uint32_t source = (batch * args.length + (uint32_t)source_time) *
                              args.channels + channel;
            upsampled = fmaf(input[source], 2.0f * upsample_filter[up_k],
                             upsampled);
        }
        float sine = sinf(alpha * upsampled);
        float activated = upsampled + sine * sine / (beta + 1e-9f);
        result = fmaf(activated, downsample_filter[down_k], result);
    }
    uint32_t destination =
        (batch * args.length + time) * args.channels + channel;
    output[destination] = result;
}

__global__ void h3_snake1d_f32_kernel(const float *input, const float *alpha,
                                      float *output,
                                      h3_audio_activation_args args) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t count = args.batch * args.length * args.channels;
    if (index >= count) return;
    float a = alpha[index % args.channels];
    float x = input[index];
    float wave = sinf(a * x);
    output[index] = x + wave * wave / (a + 1e-9f);
}

__global__ void h3_audio_qkv_split_f32_kernel(
    const float *qkv, const float *q_bias, const float *k_bias,
    const float *v_bias, float *query, float *key, float *value,
    h3_audio_qkv_args args) {
    uint32_t width = args.heads * args.head_dim;
    uint32_t count = args.batch * args.length * width;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    uint32_t column = gid % width;
    uint32_t row = gid / width;
    uint32_t base = row * width * 3;
    query[gid] = qkv[base + column] + q_bias[column];
    key[gid] = qkv[base + width + column] + k_bias[column];
    value[gid] = qkv[base + width * 2 + column] + v_bias[column];
}

__global__ void h3_sdpa_causal_f32_kernel(const float *query, const float *key,
                                          const float *value, float *output,
                                          h3_sdpa_causal_args args) {
    uint32_t head = blockIdx.x;
    uint32_t q_pos = blockIdx.y;
    uint32_t batch = blockIdx.z;
    if (head >= args.heads || q_pos >= args.sequence || batch >= args.batch)
        return;

    uint32_t slice = batch * args.sequence * args.heads * args.head_dim;
    float scores[512];
    float max_score = -1e30f;
    for (uint32_t k_pos = 0; k_pos <= q_pos; k_pos++) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < args.head_dim; d++) {
            uint32_t q_index =
                slice + (q_pos * args.heads + head) * args.head_dim + d;
            uint32_t k_index =
                slice + (k_pos * args.heads + head) * args.head_dim + d;
            dot = fmaf(query[q_index], key[k_index], dot);
        }
        scores[k_pos] = dot * args.scale;
        max_score = fmaxf(max_score, scores[k_pos]);
    }
    float sum = 0.0f;
    for (uint32_t k_pos = 0; k_pos <= q_pos; k_pos++) {
        scores[k_pos] = expf(scores[k_pos] - max_score);
        sum += scores[k_pos];
    }
    float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint32_t d = 0; d < args.head_dim; d++) {
        float acc = 0.0f;
        for (uint32_t k_pos = 0; k_pos <= q_pos; k_pos++) {
            uint32_t v_index =
                slice + (k_pos * args.heads + head) * args.head_dim + d;
            acc = fmaf(scores[k_pos] * inv_sum, value[v_index], acc);
        }
        uint32_t out_index =
            slice + (q_pos * args.heads + head) * args.head_dim + d;
        output[out_index] = acc;
    }
}

__global__ void h3_audio_attention_pool_f32_kernel(const float *attended,
                                                   float *output,
                                                   h3_audio_pool_args args) {
    uint32_t count = args.batch * args.length * args.output_dim;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    uint32_t column = gid % args.output_dim;
    uint32_t row = gid / args.output_dim;
    uint32_t pool = args.head_dim / args.output_dim;
    float sum = 0.0f;
    for (uint32_t head = 0; head < args.heads; head++) {
        uint32_t base = (row * args.heads + head) * args.head_dim + column * pool;
        for (uint32_t item = 0; item < pool; item++) sum += attended[base + item];
    }
    output[gid] = sum / float(args.heads * pool);
}

__device__ inline int h3_reflect_coordinate(int coordinate, int length) {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= length) return 2 * length - coordinate - 2;
    return coordinate;
}

__global__ void h3_vae_encoder_pad_f32_kernel(const float *input, float *output,
                                              h3_vae_encoder_pad_args args) {
    uint32_t out_height = args.height + args.height_before + args.height_after;
    uint32_t out_width = args.width + args.width_before + args.width_after;
    uint32_t out_depth = args.depth + args.depth_front;
    uint64_t spatial = (uint64_t)out_depth * out_height * out_width;
    uint64_t total = (uint64_t)args.batch * spatial * args.channels;
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    uint32_t channel = (uint32_t)(gid % args.channels);
    uint64_t rem = gid / args.channels;
    uint32_t out_x = (uint32_t)(rem % out_width);
    rem /= out_width;
    uint32_t out_y = (uint32_t)(rem % out_height);
    rem /= out_height;
    uint32_t out_t = (uint32_t)(rem % out_depth);
    uint32_t batch = (uint32_t)(rem / out_depth);
    size_t destination = ((((size_t)batch * out_depth + out_t) * out_height +
                           out_y) * out_width + out_x) * args.channels +
                         channel;
    if (out_t < args.depth_front) {
        output[destination] = 0.0f;
        return;
    }
    int source_y = h3_reflect_coordinate(
        (int)out_y - (int)args.height_before, (int)args.height);
    int source_x = h3_reflect_coordinate(
        (int)out_x - (int)args.width_before, (int)args.width);
    uint32_t source_t = out_t - args.depth_front;
    size_t source = ((((size_t)batch * args.depth + source_t) * args.height +
                      (uint32_t)source_y) * args.width + (uint32_t)source_x) *
                    args.channels + channel;
    output[destination] = input[source];
}

__global__ void h3_vae_encoder_group_norm_silu_f32_kernel(
    const float *input, const float *weight, const float *bias, float *output,
    h3_vae_encoder_norm_args args) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    uint32_t rows = args.batch * args.depth * args.groups;
    if (row >= rows) return;
    __shared__ float reductions[256];
    uint32_t channels_per_group = args.channels / args.groups;
    uint32_t group_index = row % args.groups;
    uint32_t temporal_plane = row / args.groups;
    uint32_t elements = args.height * args.width * channels_per_group;
    float local = 0.0f;
    for (uint32_t index = tid; index < elements; index += threads) {
        uint32_t spatial = index / channels_per_group;
        uint32_t channel = group_index * channels_per_group +
                           index % channels_per_group;
        size_t source = ((size_t)temporal_plane * args.height * args.width +
                         spatial) * args.channels + channel;
        local += input[source];
    }
    reductions[tid] = local;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float mean = reductions[0] / float(elements);
    local = 0.0f;
    for (uint32_t index = tid; index < elements; index += threads) {
        uint32_t spatial = index / channels_per_group;
        uint32_t channel = group_index * channels_per_group +
                           index % channels_per_group;
        size_t source = ((size_t)temporal_plane * args.height * args.width +
                         spatial) * args.channels + channel;
        float centered = input[source] - mean;
        local = fmaf(centered, centered, local);
    }
    reductions[tid] = local;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reductions[0] / float(elements) + args.epsilon);
    for (uint32_t index = tid; index < elements; index += threads) {
        uint32_t spatial = index / channels_per_group;
        uint32_t channel = group_index * channels_per_group +
                           index % channels_per_group;
        size_t destination =
            ((size_t)temporal_plane * args.height * args.width + spatial) *
            args.channels + channel;
        float value = (input[destination] - mean) * inverse * weight[channel] +
                      bias[channel];
        output[destination] = value / (1.0f + expf(-value));
    }
}

__global__ void h3_add_scaled_f32_kernel(const float *left, const float *right,
                                       float *output,
                                       h3_add_scaled_f32_args args) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= args.elements) return;
    output[index] = left[index] * args.left_scale +
                    right[index] * args.right_scale;
}

__global__ void h3_scale_add_f32_kernel(const float *residual,
                                        const float *branch,
                                        const float *scale, float *output,
                                        h3_swiglu_args args) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || column >= args.width) return;
    uint32_t index = row * args.width + column;
    output[index] = residual[index] + branch[index] * scale[column];
}

__global__ void h3_weight_norm_f32_kernel(const float *vector,
                                          const float *magnitude, float *output,
                                          h3_weight_norm_args args) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= args.outer) return;
    uint32_t base = row * args.inner;
    float square_sum = 0.0f;
    for (uint32_t index = 0; index < args.inner; index++) {
        float value = vector[base + index];
        square_sum = fmaf(value, value, square_sum);
    }
    float scale = magnitude[row] * rsqrtf(square_sum);
    for (uint32_t index = 0; index < args.inner; index++) {
        output[base + index] = vector[base + index] * scale;
    }
}

__global__ void h3_geglu_f32_kernel(const float *gate, const float *linear,
                                    float *output, uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = gate[gid];
    float cube = x * x * x;
    float gelu = 0.5f * x *
        (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * cube)));
    output[gid] = gelu * linear[gid];
}

__global__ void h3_rms_norm_f32_kernel(const float *input, const float *weight,
                                       float *output, h3_norm_args args) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    const float *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint32_t k = tid; k < args.width; k += threads) {
        local_sum = fmaf(x[k], x[k], local_sum);
    }
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reductions[0] / float(args.width) + args.epsilon);
    for (uint32_t column = tid; column < args.width; column += threads) {
        output[row * args.width + column] =
            x[column] * inverse * weight[column];
    }
}

__global__ void h3_layer_norm_f32_kernel(const float *input, const float *weight,
                                         const float *bias, float *output,
                                         h3_norm_args args) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    const float *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        local_sum += x[column];
    }
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float mean = reductions[0] / float(args.width);
    __syncthreads();
    float local_square = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        float centered = x[column] - mean;
        local_square = fmaf(centered, centered, local_square);
    }
    reductions[tid] = local_square;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reductions[0] / float(args.width) + args.epsilon);
    for (uint32_t column = tid; column < args.width; column += threads) {
        float normalized = (x[column] - mean) * inverse;
        output[row * args.width + column] =
            fmaf(normalized, weight[column], bias[column]);
    }
}

__global__ void h3_rms_norm_bf16_kernel(const uint16_t *input,
                                        const uint16_t *weight,
                                        uint16_t *output, h3_norm_args args) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    const uint16_t *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint32_t k = tid; k < args.width; k += threads) {
        float value = h3_bf16_to_f32(x[k]);
        local_sum = fmaf(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reductions[0] / float(args.width) + args.epsilon);
    for (uint32_t column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(x[column]) * inverse;
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * h3_bf16_to_f32(weight[column]));
    }
}

__global__ void h3_layer_norm_bf16_kernel(const uint16_t *input,
                                          const uint16_t *weight,
                                          const uint16_t *bias,
                                          uint16_t *output, h3_norm_args args) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    const uint16_t *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads)
        local_sum += h3_bf16_to_f32(x[column]);
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float mean = reductions[0] / float(args.width);
    __syncthreads();
    float local_square = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        float centered = h3_bf16_to_f32(x[column]) - mean;
        local_square = fmaf(centered, centered, local_square);
    }
    reductions[tid] = local_square;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reductions[0] / float(args.width) + args.epsilon);
    for (uint32_t column = tid; column < args.width; column += threads) {
        float normalized = (h3_bf16_to_f32(x[column]) - mean) * inverse;
        float value = fmaf(normalized, h3_bf16_to_f32(weight[column]),
                           h3_bf16_to_f32(bias[column]));
        output[row * args.width + column] = h3_f32_to_bf16(value);
    }
}

__global__ void h3_linear_bf16_kernel(const uint16_t *input,
                                      const uint16_t *weight,
                                      const uint16_t *bias, uint16_t *output,
                                      h3_linear_args args) {
    __shared__ float input_tile[16][16];
    __shared__ float weight_tile[16][16];
    uint32_t row = blockIdx.y * 16 + threadIdx.y;
    uint32_t column = blockIdx.x * 16 + threadIdx.x;
    float sum = args.has_bias && column < args.output_dim ?
        h3_bf16_to_f32(bias[column]) : 0.0f;
    uint32_t tile_count = (args.input_dim + 15) / 16;
    for (uint32_t tile = 0; tile < tile_count; tile++) {
        uint32_t input_k = tile * 16 + threadIdx.x;
        input_tile[threadIdx.y][threadIdx.x] =
            row < args.rows && input_k < args.input_dim ?
            h3_bf16_to_f32(input[row * args.input_dim + input_k]) : 0.0f;
        uint32_t weight_k = tile * 16 + threadIdx.y;
        weight_tile[threadIdx.y][threadIdx.x] =
            column < args.output_dim && weight_k < args.input_dim ?
            h3_bf16_to_f32(weight[column * args.input_dim + weight_k]) : 0.0f;
        __syncthreads();
        for (uint32_t k = 0; k < 16; k++) {
            sum = fmaf(input_tile[threadIdx.y][k], weight_tile[k][threadIdx.x],
                       sum);
        }
        __syncthreads();
    }
    if (row < args.rows && column < args.output_dim) {
        output[row * args.output_dim + column] = h3_f32_to_bf16(sum);
    }
}

__global__ void h3_gate_bf16_kernel(const uint16_t *residual,
                                    const uint16_t *branch,
                                    const uint16_t *modulation,
                                    const uint32_t *row_map, uint16_t *output,
                                    h3_gate_args args) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || column >= args.width) return;
    uint32_t base = row_map[row] * args.slots * args.width;
    float gate = h3_bf16_to_f32(
        modulation[base + args.gate_slot * args.width + column]);
    uint32_t index = row * args.width + column;
    float value = h3_bf16_to_f32(residual[index]) +
                  h3_bf16_to_f32(branch[index]) * gate;
    output[index] = h3_f32_to_bf16(value);
}

__global__ void h3_gate_f32_kernel(const float *residual, const float *branch,
                                   const float *modulation,
                                   const uint32_t *row_map, float *output,
                                   h3_gate_args args) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || column >= args.width) return;
    uint32_t base = row_map[row] * args.slots * args.width;
    float gate = modulation[base + args.gate_slot * args.width + column];
    uint32_t index = row * args.width + column;
    output[index] = residual[index] + branch[index] * gate;
}

__global__ void h3_adaln_f32_kernel(const float *input, const float *weight,
                                    const float *modulation,
                                    const uint32_t *row_map, float *output,
                                    h3_adaln_args args) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    const float *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint32_t k = tid; k < args.width; k += threads) {
        local_sum = fmaf(x[k], x[k], local_sum);
    }
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reductions[0] / float(args.width) + args.epsilon);
    uint32_t base = row_map[row] * args.slots * args.width;
    for (uint32_t column = tid; column < args.width; column += threads) {
        float normalized = x[column] * inverse * weight[column];
        float shift = modulation[base + args.shift_slot * args.width + column];
        float scale = modulation[base + args.scale_slot * args.width + column];
        output[row * args.width + column] = normalized * (1.0f + scale) + shift;
    }
}

__global__ void h3_adaln_bf16_kernel(const uint16_t *input,
                                     const uint16_t *weight,
                                     const uint16_t *modulation,
                                     const uint32_t *row_map, uint16_t *output,
                                     h3_adaln_args args) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    const uint16_t *x = input + row * args.width;
    float local_sum = 0.0f;
    for (uint32_t k = tid; k < args.width; k += threads) {
        float value = h3_bf16_to_f32(x[k]);
        local_sum = fmaf(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reductions[0] / float(args.width) + args.epsilon);
    uint32_t base = row_map[row] * args.slots * args.width;
    for (uint32_t column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(x[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

__global__ void h3_gate_adaln_bf16_kernel(const uint16_t *residual,
                                          const uint16_t *branch,
                                          const uint16_t *gate_modulation,
                                          const uint32_t *row_map,
                                          const uint16_t *weight,
                                          const uint16_t *norm_modulation,
                                          uint16_t *gated_residual,
                                          uint16_t *output,
                                          h3_gate_adaln_args args) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    __shared__ uint16_t gated_values[5376];
    uint32_t base = row_map[row] * args.slots * args.width;
    float local_sum = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        uint32_t index = row * args.width + column;
        float gate = h3_bf16_to_f32(
            gate_modulation[base + args.gate_slot * args.width + column]);
        uint16_t gated = h3_f32_to_bf16(
            h3_bf16_to_f32(residual[index]) +
            h3_bf16_to_f32(branch[index]) * gate);
        gated_residual[index] = gated;
        gated_values[column] = gated;
        float value = h3_bf16_to_f32(gated);
        local_sum = fmaf(value, value, local_sum);
    }
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reductions[0] / float(args.width) + args.epsilon);
    for (uint32_t column = tid; column < args.width; column += threads) {
        float normalized = h3_bf16_to_f32(gated_values[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            norm_modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            norm_modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

__global__ void h3_qkv_rope_bf16_kernel(const uint16_t *qkv,
                                        const uint16_t *q_weight,
                                        const uint16_t *k_weight,
                                        const uint16_t *rope_cos,
                                        const uint16_t *rope_sin,
                                        uint16_t *query, uint16_t *key,
                                        uint16_t *value, h3_qkv_args args) {
    uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.z;
    if (dimension >= args.head_dim || head >= args.heads ||
        row >= args.sequence) return;
    uint32_t inner = args.heads * args.head_dim;
    uint32_t row_base = row * inner * 3;
    uint32_t q_base = row_base + head * args.head_dim;
    uint32_t k_base = q_base + inner;
    uint32_t v_base = q_base + inner * 2;
    if (args.grouped) {
        q_base = row_base + head * args.head_dim * 3;
        k_base = q_base + args.head_dim;
        v_base = k_base + args.head_dim;
    }
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint32_t d = 0; d < args.head_dim; d++) {
        float q = h3_bf16_to_f32(qkv[q_base + d]);
        float k = h3_bf16_to_f32(qkv[k_base + d]);
        q_sum = fmaf(q, q, q_sum);
        k_sum = fmaf(k, k, k_sum);
    }
    float q_inverse = rsqrtf(q_sum / float(args.head_dim) + args.epsilon);
    float k_inverse = rsqrtf(k_sum / float(args.head_dim) + args.epsilon);
    float q0 = h3_bf16_to_f32(qkv[q_base + dimension]) * q_inverse *
               h3_bf16_to_f32(q_weight[dimension]);
    float k0 = h3_bf16_to_f32(qkv[k_base + dimension]) * k_inverse *
               h3_bf16_to_f32(k_weight[dimension]);
    if (dimension < args.rope_half) {
        uint32_t pair = dimension + args.rope_half;
        float q1 = h3_bf16_to_f32(qkv[q_base + pair]) * q_inverse *
                   h3_bf16_to_f32(q_weight[pair]);
        float k1 = h3_bf16_to_f32(qkv[k_base + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float c = h3_bf16_to_f32(rope_cos[row * args.rope_half + dimension]);
        float s = h3_bf16_to_f32(rope_sin[row * args.rope_half + dimension]);
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint32_t pair = dimension - args.rope_half;
        float q1 = h3_bf16_to_f32(qkv[q_base + pair]) * q_inverse *
                   h3_bf16_to_f32(q_weight[pair]);
        float k1 = h3_bf16_to_f32(qkv[k_base + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float c = h3_bf16_to_f32(rope_cos[row * args.rope_half + pair]);
        float s = h3_bf16_to_f32(rope_sin[row * args.rope_half + pair]);
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint32_t output_index = (row * args.heads + head) * args.head_dim + dimension;
    uint32_t kv_index = args.kv_head_major ?
        (head * args.sequence + row) * args.head_dim + dimension : output_index;
    query[output_index] = h3_f32_to_bf16(q0);
    key[kv_index] = h3_f32_to_bf16(k0);
    value[kv_index] = qkv[v_base + dimension];
}

__global__ void h3_qkv_rope_f32_kernel(const float *qkv, const float *q_weight,
                                       const float *k_weight,
                                       const float *rope_cos,
                                       const float *rope_sin, float *query,
                                       float *key, float *value,
                                       h3_qkv_args args) {
    uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.z;
    if (dimension >= args.head_dim || head >= args.heads ||
        row >= args.sequence) return;
    uint32_t inner = args.heads * args.head_dim;
    uint32_t row_base = row * inner * 3;
    uint32_t q_base = row_base + head * args.head_dim;
    uint32_t k_base = q_base + inner;
    uint32_t v_base = q_base + inner * 2;
    if (args.grouped) {
        q_base = row_base + head * args.head_dim * 3;
        k_base = q_base + args.head_dim;
        v_base = k_base + args.head_dim;
    }
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint32_t d = 0; d < args.head_dim; d++) {
        q_sum = fmaf(qkv[q_base + d], qkv[q_base + d], q_sum);
        k_sum = fmaf(qkv[k_base + d], qkv[k_base + d], k_sum);
    }
    float q_inverse = rsqrtf(q_sum / float(args.head_dim) + args.epsilon);
    float k_inverse = rsqrtf(k_sum / float(args.head_dim) + args.epsilon);
    float q0 = qkv[q_base + dimension] * q_inverse * q_weight[dimension];
    float k0 = qkv[k_base + dimension] * k_inverse * k_weight[dimension];
    if (dimension < args.rope_half) {
        uint32_t pair = dimension + args.rope_half;
        float q1 = qkv[q_base + pair] * q_inverse * q_weight[pair];
        float k1 = qkv[k_base + pair] * k_inverse * k_weight[pair];
        float c = rope_cos[row * args.rope_half + dimension];
        float s = rope_sin[row * args.rope_half + dimension];
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint32_t pair = dimension - args.rope_half;
        float q1 = qkv[q_base + pair] * q_inverse * q_weight[pair];
        float k1 = qkv[k_base + pair] * k_inverse * k_weight[pair];
        float c = rope_cos[row * args.rope_half + pair];
        float s = rope_sin[row * args.rope_half + pair];
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint32_t output_index = (row * args.heads + head) * args.head_dim + dimension;
    uint32_t kv_index = args.kv_head_major ?
        (head * args.sequence + row) * args.head_dim + dimension : output_index;
    query[output_index] = q0;
    key[kv_index] = k0;
    value[kv_index] = qkv[v_base + dimension];
}

__global__ void h3_video_qkv_rope_f32_kernel(
    const float *qkv, const float *rope_cos, const float *rope_sin,
    float *query, float *key, float *value, h3_qkv_args args) {
    uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.z;
    if (dimension >= args.head_dim || head >= args.heads ||
        row >= args.sequence) return;
    uint32_t base = (row * args.heads + head) * args.head_dim * 3;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint32_t d = 0; d < args.head_dim; d++) {
        q_sum = fmaf(qkv[base + d], qkv[base + d], q_sum);
        k_sum = fmaf(qkv[base + args.head_dim + d],
                     qkv[base + args.head_dim + d], k_sum);
    }
    float q_inverse = rsqrtf(q_sum / float(args.head_dim) + args.epsilon);
    float k_inverse = rsqrtf(k_sum / float(args.head_dim) + args.epsilon);
    float q0 = qkv[base + dimension] * q_inverse;
    float k0 = qkv[base + args.head_dim + dimension] * k_inverse;
    if (dimension < args.rope_half) {
        uint32_t pair = dimension + args.rope_half;
        float q1 = qkv[base + pair] * q_inverse;
        float k1 = qkv[base + args.head_dim + pair] * k_inverse;
        float c = rope_cos[row * args.rope_half + dimension];
        float s = rope_sin[row * args.rope_half + dimension];
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint32_t pair = dimension - args.rope_half;
        float q1 = qkv[base + pair] * q_inverse;
        float k1 = qkv[base + args.head_dim + pair] * k_inverse;
        float c = rope_cos[row * args.rope_half + pair];
        float s = rope_sin[row * args.rope_half + pair];
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint32_t output_index =
        (row * args.heads + head) * args.head_dim + dimension;
    uint32_t kv_index = args.kv_head_major ?
        (head * args.sequence + row) * args.head_dim + dimension : output_index;
    query[output_index] = q0;
    key[kv_index] = k0;
    value[kv_index] = qkv[base + args.head_dim * 2 + dimension];
}

__global__ void h3_vision_qkv_rope_bf16_kernel(
    const uint16_t *qkv, const uint16_t *rope_cos, const uint16_t *rope_sin,
    uint16_t *query, uint16_t *key, uint16_t *value, h3_qkv_args args) {
    uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.z;
    if (dimension >= args.head_dim || head >= args.heads ||
        row >= args.sequence) return;
    uint32_t inner = args.heads * args.head_dim;
    uint32_t row_base = row * inner * 3;
    uint32_t q_base = row_base + head * args.head_dim;
    uint32_t k_base = row_base + inner + head * args.head_dim;
    uint32_t v_base = row_base + inner * 2 + head * args.head_dim;
    uint32_t half_dim = args.rope_half;
    uint32_t rope_index = row * half_dim + dimension % half_dim;
    float c = h3_bf16_to_f32(rope_cos[rope_index]);
    float s = h3_bf16_to_f32(rope_sin[rope_index]);
    uint32_t pair = dimension < half_dim ? dimension + half_dim :
                                         dimension - half_dim;
    float q0 = h3_bf16_to_f32(qkv[q_base + dimension]);
    float k0 = h3_bf16_to_f32(qkv[k_base + dimension]);
    float q1 = h3_bf16_to_f32(qkv[q_base + pair]);
    float k1 = h3_bf16_to_f32(qkv[k_base + pair]);
    float qr = dimension < half_dim ? q0 * c - q1 * s : q0 * c + q1 * s;
    float kr = dimension < half_dim ? k0 * c - k1 * s : k0 * c + k1 * s;
    uint32_t output_index = (row * args.heads + head) * args.head_dim + dimension;
    query[output_index] = h3_f32_to_bf16(qr);
    key[output_index] = h3_f32_to_bf16(kr);
    value[output_index] = qkv[v_base + dimension];
}

__device__ inline uint32_t h3_sdpa_output_index(h3_sdpa_args args,
                                                uint32_t q_pos, uint32_t head,
                                                uint32_t dimension) {
    uint32_t packed = args.head_major_output ?
        head * args.sequence + q_pos : q_pos * args.heads + head;
    return packed * args.head_dim + dimension;
}

__global__ void h3_sdpa_bf16_kernel(const uint16_t *query, const uint16_t *key,
                                    const uint16_t *value, uint16_t *output,
                                    h3_sdpa_args args) {
    uint32_t head = blockIdx.x;
    uint32_t q_pos = blockIdx.y;
    if (head >= args.heads || q_pos >= args.sequence) return;

    float scores[512];
    float max_score = -1e30f;
    for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < args.head_dim; d++) {
            uint32_t q_index =
                (q_pos * args.heads + head) * args.head_dim + d;
            uint32_t k_index =
                (k_pos * args.heads + head) * args.head_dim + d;
            dot = fmaf(h3_bf16_to_f32(query[q_index]),
                       h3_bf16_to_f32(key[k_index]), dot);
        }
        scores[k_pos] = dot * args.scale;
        max_score = fmaxf(max_score, scores[k_pos]);
    }
    float sum = 0.0f;
    for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
        scores[k_pos] = expf(scores[k_pos] - max_score);
        sum += scores[k_pos];
    }
    float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint32_t d = 0; d < args.head_dim; d++) {
        float acc = 0.0f;
        for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
            uint32_t v_index =
                (k_pos * args.heads + head) * args.head_dim + d;
            acc = fmaf(scores[k_pos] * inv_sum,
                       h3_bf16_to_f32(value[v_index]), acc);
        }
        uint32_t out_index = h3_sdpa_output_index(args, q_pos, head, d);
        output[out_index] = h3_f32_to_bf16(acc);
    }
}

__global__ void h3_sdpa_f32_kernel(const float *query, const float *key,
                                   const float *value, float *output,
                                   h3_sdpa_args args) {
    uint32_t head = blockIdx.x;
    uint32_t q_pos = blockIdx.y;
    if (head >= args.heads || q_pos >= args.sequence) return;

    float scores[512];
    float max_score = -1e30f;
    for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < args.head_dim; d++) {
            uint32_t q_index =
                (q_pos * args.heads + head) * args.head_dim + d;
            uint32_t k_index =
                (k_pos * args.heads + head) * args.head_dim + d;
            dot = fmaf(query[q_index], key[k_index], dot);
        }
        scores[k_pos] = dot * args.scale;
        max_score = fmaxf(max_score, scores[k_pos]);
    }
    float sum = 0.0f;
    for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
        scores[k_pos] = expf(scores[k_pos] - max_score);
        sum += scores[k_pos];
    }
    float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint32_t d = 0; d < args.head_dim; d++) {
        float acc = 0.0f;
        for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
            uint32_t v_index =
                (k_pos * args.heads + head) * args.head_dim + d;
            acc = fmaf(scores[k_pos] * inv_sum, value[v_index], acc);
        }
        uint32_t out_index =
            (q_pos * args.heads + head) * args.head_dim + d;
        output[out_index] = acc;
    }
}

#define H3_SDPA_TILE 512u

static uint32_t h3_sdpa_threads(uint32_t sequence) {
    uint32_t threads = 1;
    while (threads * 2u <= 128u && threads < sequence) threads *= 2u;
    return threads ? threads : 1u;
}

__device__ inline float h3_warp_reduce_sum(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        value += __shfl_xor(value, mask);
    return value;
}

__device__ inline float h3_fast_exp(float x) {
    return exp2f(x * 1.4426950408889634f);
}

/* AudioVAE encoder uses 8 heads x 256; VAE/text stay <= 128. One warp covers
 * 32-wide chunks, so 8 chunks reach head_dim 256. */
__global__ void __launch_bounds__(32)
h3_sdpa_causal_f32_wave_kernel(const float *query, const float *key,
                               const float *value, float *output,
                               h3_sdpa_causal_args args) {
    uint32_t head = blockIdx.x;
    uint32_t q_pos = blockIdx.y;
    uint32_t batch = blockIdx.z;
    uint32_t lane = threadIdx.x;
    uint32_t slice = batch * args.sequence * args.heads * args.head_dim;
    uint32_t q_base = slice + (q_pos * args.heads + head) * args.head_dim;
    float q[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int item = 0; item < 8; item++) {
        uint32_t dimension = lane + (uint32_t)item * 32u;
        if (dimension < args.head_dim)
            q[item] = query[q_base + dimension] * args.scale;
    }
    float maximum = -1e30f;
    float sum = 0.0f;
    for (uint32_t k_pos = 0; k_pos <= q_pos; k_pos++) {
        uint32_t k_base = slice + (k_pos * args.heads + head) * args.head_dim;
        float partial = 0.0f;
#pragma unroll
        for (int item = 0; item < 8; item++) {
            uint32_t dimension = lane + (uint32_t)item * 32u;
            if (dimension < args.head_dim)
                partial = fmaf(q[item], key[k_base + dimension], partial);
        }
        float score = h3_warp_reduce_sum(partial);
        float new_max = fmaxf(maximum, score);
        float alpha = h3_fast_exp(maximum - new_max);
        float probability = h3_fast_exp(score - new_max);
#pragma unroll
        for (int item = 0; item < 8; item++) {
            uint32_t dimension = lane + (uint32_t)item * 32u;
            float v = dimension < args.head_dim ? value[k_base + dimension] : 0.0f;
            acc[item] = fmaf(probability, v, acc[item] * alpha);
        }
        sum = fmaf(sum, alpha, probability);
        maximum = new_max;
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
#pragma unroll
    for (int item = 0; item < 8; item++) {
        uint32_t dimension = lane + (uint32_t)item * 32u;
        if (dimension < args.head_dim)
            output[q_base + dimension] = acc[item] * inv;
    }
}

__global__ void __launch_bounds__(32)
h3_sdpa_bf16_wave_d128_kernel(const uint16_t *query, const uint16_t *key,
                              const uint16_t *value, uint16_t *output,
                              h3_sdpa_args args) {
    uint32_t q_pos = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    uint32_t q_base = (q_pos * args.heads + head) * 128u;
    float q0 = h3_bf16_to_f32(query[q_base + lane]) * args.scale;
    float q1 = h3_bf16_to_f32(query[q_base + 32u + lane]) * args.scale;
    float q2 = h3_bf16_to_f32(query[q_base + 64u + lane]) * args.scale;
    float q3 = h3_bf16_to_f32(query[q_base + 96u + lane]) * args.scale;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float maximum = -1e30f;
    float sum = 0.0f;
    /* Seq-major K/V: stride = heads*128. Head-major: contiguous per head. */
    uint32_t stride = args.kv_head_major ? 128u : args.heads * 128u;
    uint32_t k_base = args.kv_head_major ? head * args.sequence * 128u
                                         : head * 128u;
    for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
        float k0 = h3_bf16_to_f32(key[k_base + lane]);
        float k1v = h3_bf16_to_f32(key[k_base + 32u + lane]);
        float k2 = h3_bf16_to_f32(key[k_base + 64u + lane]);
        float k3 = h3_bf16_to_f32(key[k_base + 96u + lane]);
        float v0 = h3_bf16_to_f32(value[k_base + lane]);
        float v1 = h3_bf16_to_f32(value[k_base + 32u + lane]);
        float v2 = h3_bf16_to_f32(value[k_base + 64u + lane]);
        float v3 = h3_bf16_to_f32(value[k_base + 96u + lane]);
        float partial = q0 * k0 + q1 * k1v + q2 * k2 + q3 * k3;
        float score = h3_warp_reduce_sum(partial);
        float new_max = fmaxf(maximum, score);
        float alpha = h3_fast_exp(maximum - new_max);
        float probability = h3_fast_exp(score - new_max);
        acc0 = fmaf(probability, v0, acc0 * alpha);
        acc1 = fmaf(probability, v1, acc1 * alpha);
        acc2 = fmaf(probability, v2, acc2 * alpha);
        acc3 = fmaf(probability, v3, acc3 * alpha);
        sum = fmaf(sum, alpha, probability);
        maximum = new_max;
        k_base += stride;
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    output[h3_sdpa_output_index(args, q_pos, head, lane)] =
        h3_f32_to_bf16(acc0 * inv);
    output[h3_sdpa_output_index(args, q_pos, head, 32u + lane)] =
        h3_f32_to_bf16(acc1 * inv);
    output[h3_sdpa_output_index(args, q_pos, head, 64u + lane)] =
        h3_f32_to_bf16(acc2 * inv);
    output[h3_sdpa_output_index(args, q_pos, head, 96u + lane)] =
        h3_f32_to_bf16(acc3 * inv);
}

/* Flash-style blocked SDPA for DiT head_dim=128: several Q rows share each
 * K/V tile in LDS so global traffic drops by ~Q_TILE. Wave32 RDNA path.
 * Rows are padded to 132 floats to avoid 4-way LDS bank conflicts when a
 * warp reads lane, lane+32, lane+64, lane+96. */
#define H3_SDPA_FLASH_Q 4u
#define H3_SDPA_FLASH_K 32u
#define H3_SDPA_FLASH_STRIDE 132u

__global__ void __launch_bounds__(128)
h3_sdpa_bf16_flash_d128_kernel(const uint16_t *query, const uint16_t *key,
                               const uint16_t *value, uint16_t *output,
                               h3_sdpa_args args) {
    __shared__ float smem_k[H3_SDPA_FLASH_K * H3_SDPA_FLASH_STRIDE];
    __shared__ float smem_v[H3_SDPA_FLASH_K * H3_SDPA_FLASH_STRIDE];

    uint32_t head = blockIdx.x;
    uint32_t q_tile = blockIdx.y;
    uint32_t lane = threadIdx.x;
    uint32_t q_local = threadIdx.y;
    uint32_t q_pos = q_tile * H3_SDPA_FLASH_Q + q_local;
    uint32_t tid = q_local * 32u + lane;
    int active = q_pos < args.sequence;

    float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float maximum = -1e30f;
    float sum = 0.0f;
    if (active) {
        uint32_t q_base = (q_pos * args.heads + head) * 128u;
        q0 = h3_bf16_to_f32(query[q_base + lane]) * args.scale;
        q1 = h3_bf16_to_f32(query[q_base + 32u + lane]) * args.scale;
        q2 = h3_bf16_to_f32(query[q_base + 64u + lane]) * args.scale;
        q3 = h3_bf16_to_f32(query[q_base + 96u + lane]) * args.scale;
    }

    uint32_t stride = args.kv_head_major ? 128u : args.heads * 128u;
    for (uint32_t k_start = 0; k_start < args.sequence;
         k_start += H3_SDPA_FLASH_K) {
        uint32_t tile = args.sequence - k_start;
        if (tile > H3_SDPA_FLASH_K) tile = H3_SDPA_FLASH_K;
        for (uint32_t index = tid; index < tile * 128u; index += 128u) {
            uint32_t k_local = index >> 7;
            uint32_t dimension = index & 127u;
            uint32_t src = args.kv_head_major ?
                (head * args.sequence + k_start + k_local) * 128u + dimension :
                (k_start + k_local) * stride + head * 128u + dimension;
            uint32_t dst = k_local * H3_SDPA_FLASH_STRIDE + dimension;
            smem_k[dst] = h3_bf16_to_f32(key[src]);
            smem_v[dst] = h3_bf16_to_f32(value[src]);
        }
        __syncthreads();

        if (active) {
            for (uint32_t k_local = 0; k_local < tile; k_local++) {
                const float *k_row = smem_k + k_local * H3_SDPA_FLASH_STRIDE;
                const float *v_row = smem_v + k_local * H3_SDPA_FLASH_STRIDE;
                float partial = q0 * k_row[lane] + q1 * k_row[32u + lane] +
                                q2 * k_row[64u + lane] +
                                q3 * k_row[96u + lane];
                float score = h3_warp_reduce_sum(partial);
                float new_max = fmaxf(maximum, score);
                float alpha = h3_fast_exp(maximum - new_max);
                float probability = h3_fast_exp(score - new_max);
                acc0 = fmaf(probability, v_row[lane], acc0 * alpha);
                acc1 = fmaf(probability, v_row[32u + lane], acc1 * alpha);
                acc2 = fmaf(probability, v_row[64u + lane], acc2 * alpha);
                acc3 = fmaf(probability, v_row[96u + lane], acc3 * alpha);
                sum = fmaf(sum, alpha, probability);
                maximum = new_max;
            }
        }
        __syncthreads();
    }

    if (active) {
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        output[h3_sdpa_output_index(args, q_pos, head, lane)] =
            h3_f32_to_bf16(acc0 * inv);
        output[h3_sdpa_output_index(args, q_pos, head, 32u + lane)] =
            h3_f32_to_bf16(acc1 * inv);
        output[h3_sdpa_output_index(args, q_pos, head, 64u + lane)] =
            h3_f32_to_bf16(acc2 * inv);
        output[h3_sdpa_output_index(args, q_pos, head, 96u + lane)] =
            h3_f32_to_bf16(acc3 * inv);
    }
}

__global__ void __launch_bounds__(128)
h3_sdpa_f32_flash_d128_kernel(const float *query, const float *key,
                              const float *value, float *output,
                              h3_sdpa_args args) {
    __shared__ float smem_k[H3_SDPA_FLASH_K * H3_SDPA_FLASH_STRIDE];
    __shared__ float smem_v[H3_SDPA_FLASH_K * H3_SDPA_FLASH_STRIDE];

    uint32_t head = blockIdx.x;
    uint32_t q_tile = blockIdx.y;
    uint32_t lane = threadIdx.x;
    uint32_t q_local = threadIdx.y;
    uint32_t q_pos = q_tile * H3_SDPA_FLASH_Q + q_local;
    uint32_t tid = q_local * 32u + lane;
    int active = q_pos < args.sequence;

    float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float maximum = -1e30f;
    float sum = 0.0f;
    if (active) {
        uint32_t q_base = (q_pos * args.heads + head) * 128u;
        q0 = query[q_base + lane] * args.scale;
        q1 = query[q_base + 32u + lane] * args.scale;
        q2 = query[q_base + 64u + lane] * args.scale;
        q3 = query[q_base + 96u + lane] * args.scale;
    }

    uint32_t stride = args.kv_head_major ? 128u : args.heads * 128u;
    for (uint32_t k_start = 0; k_start < args.sequence;
         k_start += H3_SDPA_FLASH_K) {
        uint32_t tile = args.sequence - k_start;
        if (tile > H3_SDPA_FLASH_K) tile = H3_SDPA_FLASH_K;
        for (uint32_t index = tid; index < tile * 128u; index += 128u) {
            uint32_t k_local = index >> 7;
            uint32_t dimension = index & 127u;
            uint32_t src = args.kv_head_major ?
                (head * args.sequence + k_start + k_local) * 128u + dimension :
                (k_start + k_local) * stride + head * 128u + dimension;
            uint32_t dst = k_local * H3_SDPA_FLASH_STRIDE + dimension;
            smem_k[dst] = key[src];
            smem_v[dst] = value[src];
        }
        __syncthreads();

        if (active) {
            for (uint32_t k_local = 0; k_local < tile; k_local++) {
                const float *k_row = smem_k + k_local * H3_SDPA_FLASH_STRIDE;
                const float *v_row = smem_v + k_local * H3_SDPA_FLASH_STRIDE;
                float partial = q0 * k_row[lane] + q1 * k_row[32u + lane] +
                                q2 * k_row[64u + lane] +
                                q3 * k_row[96u + lane];
                float score = h3_warp_reduce_sum(partial);
                float new_max = fmaxf(maximum, score);
                float alpha = h3_fast_exp(maximum - new_max);
                float probability = h3_fast_exp(score - new_max);
                acc0 = fmaf(probability, v_row[lane], acc0 * alpha);
                acc1 = fmaf(probability, v_row[32u + lane], acc1 * alpha);
                acc2 = fmaf(probability, v_row[64u + lane], acc2 * alpha);
                acc3 = fmaf(probability, v_row[96u + lane], acc3 * alpha);
                sum = fmaf(sum, alpha, probability);
                maximum = new_max;
            }
        }
        __syncthreads();
    }

    if (active) {
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        uint32_t q_base = (q_pos * args.heads + head) * 128u;
        output[q_base + lane] = acc0 * inv;
        output[q_base + 32u + lane] = acc1 * inv;
        output[q_base + 64u + lane] = acc2 * inv;
        output[q_base + 96u + lane] = acc3 * inv;
    }
}

__global__ void __launch_bounds__(32)
h3_sdpa_f32_wave_d128_kernel(const float *query, const float *key,
                             const float *value, float *output,
                             h3_sdpa_args args) {
    uint32_t q_pos = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    uint32_t q_base = (q_pos * args.heads + head) * 128u;
    float q0 = query[q_base + lane] * args.scale;
    float q1 = query[q_base + 32u + lane] * args.scale;
    float q2 = query[q_base + 64u + lane] * args.scale;
    float q3 = query[q_base + 96u + lane] * args.scale;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float maximum = -1e30f;
    float sum = 0.0f;
    uint32_t stride = args.kv_head_major ? 128u : args.heads * 128u;
    uint32_t k_base = args.kv_head_major ? head * args.sequence * 128u
                                         : head * 128u;
    for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
        float k0 = key[k_base + lane];
        float k1v = key[k_base + 32u + lane];
        float k2 = key[k_base + 64u + lane];
        float k3 = key[k_base + 96u + lane];
        float v0 = value[k_base + lane];
        float v1 = value[k_base + 32u + lane];
        float v2 = value[k_base + 64u + lane];
        float v3 = value[k_base + 96u + lane];
        float partial = q0 * k0 + q1 * k1v + q2 * k2 + q3 * k3;
        float score = h3_warp_reduce_sum(partial);
        float new_max = fmaxf(maximum, score);
        float alpha = h3_fast_exp(maximum - new_max);
        float probability = h3_fast_exp(score - new_max);
        acc0 = fmaf(probability, v0, acc0 * alpha);
        acc1 = fmaf(probability, v1, acc1 * alpha);
        acc2 = fmaf(probability, v2, acc2 * alpha);
        acc3 = fmaf(probability, v3, acc3 * alpha);
        sum = fmaf(sum, alpha, probability);
        maximum = new_max;
        k_base += stride;
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    output[q_base + lane] = acc0 * inv;
    output[q_base + 32u + lane] = acc1 * inv;
    output[q_base + 64u + lane] = acc2 * inv;
    output[q_base + 96u + lane] = acc3 * inv;
}

__global__ void __launch_bounds__(32)
h3_sdpa_f32_wave_d64_kernel(const float *query, const float *key,
                            const float *value, float *output,
                            h3_sdpa_args args) {
    uint32_t q_pos = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    uint32_t q_base = (q_pos * args.heads + head) * 64u;
    float q0 = query[q_base + lane] * args.scale;
    float q1 = query[q_base + 32u + lane] * args.scale;
    float acc0 = 0.0f, acc1 = 0.0f;
    float maximum = -1e30f;
    float sum = 0.0f;
    uint32_t stride = args.kv_head_major ? 64u : args.heads * 64u;
    uint32_t k_base = args.kv_head_major ? head * args.sequence * 64u
                                         : head * 64u;
    /* Four-key online softmax on VAE d64 (2-key helped; try 4). */
    uint32_t k_pos = 0;
    for (; k_pos + 3u < args.sequence; k_pos += 4u) {
        float score[4];
        float v0s[4], v1s[4];
#pragma unroll
        for (int t = 0; t < 4; t++) {
            uint32_t base = k_base + (uint32_t)t * stride;
            float k0 = key[base + lane];
            float k1v = key[base + 32u + lane];
            v0s[t] = value[base + lane];
            v1s[t] = value[base + 32u + lane];
            score[t] = h3_warp_reduce_sum(q0 * k0 + q1 * k1v);
        }
        float new_max = fmaxf(maximum, fmaxf(fmaxf(score[0], score[1]),
                                             fmaxf(score[2], score[3])));
        float alpha = h3_fast_exp(maximum - new_max);
        float p0 = h3_fast_exp(score[0] - new_max);
        float p1 = h3_fast_exp(score[1] - new_max);
        float p2 = h3_fast_exp(score[2] - new_max);
        float p3 = h3_fast_exp(score[3] - new_max);
        acc0 = fmaf(p0, v0s[0],
                    fmaf(p1, v0s[1],
                         fmaf(p2, v0s[2], fmaf(p3, v0s[3], acc0 * alpha))));
        acc1 = fmaf(p0, v1s[0],
                    fmaf(p1, v1s[1],
                         fmaf(p2, v1s[2], fmaf(p3, v1s[3], acc1 * alpha))));
        sum = fmaf(sum, alpha, p0 + p1 + p2 + p3);
        maximum = new_max;
        k_base += 4u * stride;
    }
    for (; k_pos + 1u < args.sequence; k_pos += 2u) {
        float k0a = key[k_base + lane];
        float k1a = key[k_base + 32u + lane];
        float v0a = value[k_base + lane];
        float v1a = value[k_base + 32u + lane];
        uint32_t k_base1 = k_base + stride;
        float k0b = key[k_base1 + lane];
        float k1b = key[k_base1 + 32u + lane];
        float v0b = value[k_base1 + lane];
        float v1b = value[k_base1 + 32u + lane];
        float score_a = h3_warp_reduce_sum(q0 * k0a + q1 * k1a);
        float score_b = h3_warp_reduce_sum(q0 * k0b + q1 * k1b);
        float new_max = fmaxf(maximum, fmaxf(score_a, score_b));
        float alpha = h3_fast_exp(maximum - new_max);
        float pa = h3_fast_exp(score_a - new_max);
        float pb = h3_fast_exp(score_b - new_max);
        acc0 = fmaf(pa, v0a, fmaf(pb, v0b, acc0 * alpha));
        acc1 = fmaf(pa, v1a, fmaf(pb, v1b, acc1 * alpha));
        sum = fmaf(sum, alpha, pa + pb);
        maximum = new_max;
        k_base = k_base1 + stride;
    }
    for (; k_pos < args.sequence; k_pos++) {
        float k0 = key[k_base + lane];
        float k1v = key[k_base + 32u + lane];
        float v0 = value[k_base + lane];
        float v1 = value[k_base + 32u + lane];
        float score = h3_warp_reduce_sum(q0 * k0 + q1 * k1v);
        float new_max = fmaxf(maximum, score);
        float alpha = h3_fast_exp(maximum - new_max);
        float probability = h3_fast_exp(score - new_max);
        acc0 = fmaf(probability, v0, acc0 * alpha);
        acc1 = fmaf(probability, v1, acc1 * alpha);
        sum = fmaf(sum, alpha, probability);
        maximum = new_max;
        k_base += stride;
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    output[q_base + lane] = acc0 * inv;
    output[q_base + 32u + lane] = acc1 * inv;
}

__global__ void __launch_bounds__(32)
h3_sdpa_bf16_wave_kernel(const uint16_t *query, const uint16_t *key,
                         const uint16_t *value, uint16_t *output,
                         h3_sdpa_args args) {
    uint32_t q_pos = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    uint32_t q_base = (q_pos * args.heads + head) * args.head_dim;
    float q[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int item = 0; item < 4; item++) {
        uint32_t dimension = lane + (uint32_t)item * 32u;
        if (dimension < args.head_dim)
            q[item] = h3_bf16_to_f32(query[q_base + dimension]) * args.scale;
    }
    float maximum = -1e30f;
    float sum = 0.0f;
    uint32_t stride = args.kv_head_major ? args.head_dim
                                         : args.heads * args.head_dim;
    uint32_t k_base = args.kv_head_major ? head * args.sequence * args.head_dim
                                         : head * args.head_dim;
    for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
        float partial = 0.0f;
#pragma unroll
        for (int item = 0; item < 4; item++) {
            uint32_t dimension = lane + (uint32_t)item * 32u;
            if (dimension < args.head_dim)
                partial = fmaf(q[item], h3_bf16_to_f32(key[k_base + dimension]),
                               partial);
        }
        float score = h3_warp_reduce_sum(partial);
        float new_max = fmaxf(maximum, score);
        float alpha = h3_fast_exp(maximum - new_max);
        float probability = h3_fast_exp(score - new_max);
#pragma unroll
        for (int item = 0; item < 4; item++) {
            uint32_t dimension = lane + (uint32_t)item * 32u;
            float v = 0.0f;
            if (dimension < args.head_dim)
                v = h3_bf16_to_f32(value[k_base + dimension]);
            acc[item] = fmaf(probability, v, acc[item] * alpha);
        }
        sum = fmaf(sum, alpha, probability);
        maximum = new_max;
        k_base += stride;
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        uint32_t dimension = lane + (uint32_t)item * 32u;
        if (dimension < args.head_dim)
            output[h3_sdpa_output_index(args, q_pos, head, dimension)] =
                h3_f32_to_bf16(acc[item] * inv);
    }
}

__global__ void __launch_bounds__(32)
h3_sdpa_f32_wave_kernel(const float *query, const float *key,
                        const float *value, float *output, h3_sdpa_args args) {
    uint32_t q_pos = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    uint32_t q_base = (q_pos * args.heads + head) * args.head_dim;
    float q[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int item = 0; item < 4; item++) {
        uint32_t dimension = lane + (uint32_t)item * 32u;
        if (dimension < args.head_dim)
            q[item] = query[q_base + dimension] * args.scale;
    }
    float maximum = -1e30f;
    float sum = 0.0f;
    uint32_t stride = args.kv_head_major ? args.head_dim
                                         : args.heads * args.head_dim;
    uint32_t k_base = args.kv_head_major ? head * args.sequence * args.head_dim
                                         : head * args.head_dim;
    for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
        float partial = 0.0f;
#pragma unroll
        for (int item = 0; item < 4; item++) {
            uint32_t dimension = lane + (uint32_t)item * 32u;
            if (dimension < args.head_dim)
                partial = fmaf(q[item], key[k_base + dimension], partial);
        }
        float score = h3_warp_reduce_sum(partial);
        float new_max = fmaxf(maximum, score);
        float alpha = h3_fast_exp(maximum - new_max);
        float probability = h3_fast_exp(score - new_max);
#pragma unroll
        for (int item = 0; item < 4; item++) {
            uint32_t dimension = lane + (uint32_t)item * 32u;
            float v = dimension < args.head_dim ? value[k_base + dimension] : 0.0f;
            acc[item] = fmaf(probability, v, acc[item] * alpha);
        }
        sum = fmaf(sum, alpha, probability);
        maximum = new_max;
        k_base += stride;
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        uint32_t dimension = lane + (uint32_t)item * 32u;
        if (dimension < args.head_dim) output[q_base + dimension] = acc[item] * inv;
    }
}

__global__ void h3_sdpa_bf16_tiled_kernel(const uint16_t *query,
                                          const uint16_t *key,
                                          const uint16_t *value,
                                          uint16_t *output,
                                          h3_sdpa_args args) {
    __shared__ float reductions[128];
    __shared__ float shared_query[128];
    __shared__ float tile_scores[H3_SDPA_TILE];
    __shared__ float acc[128];
    __shared__ float softmax_max;
    __shared__ float softmax_sum;

    uint32_t head = blockIdx.x;
    uint32_t q_pos = blockIdx.y;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (head >= args.heads || q_pos >= args.sequence || args.head_dim > 128u)
        return;

    uint32_t q_base = (q_pos * args.heads + head) * args.head_dim;
    for (uint32_t d = tid; d < args.head_dim; d += threads) {
        shared_query[d] = h3_bf16_to_f32(query[q_base + d]) * args.scale;
        acc[d] = 0.0f;
    }
    if (tid == 0) {
        softmax_max = -1e30f;
        softmax_sum = 0.0f;
    }
    __syncthreads();

    for (uint32_t tile_start = 0; tile_start < args.sequence;
         tile_start += H3_SDPA_TILE) {
        uint32_t tile_count = args.sequence - tile_start;
        if (tile_count > H3_SDPA_TILE) tile_count = H3_SDPA_TILE;

        float local_max = -1e30f;
        for (uint32_t ki = tid; ki < tile_count; ki += threads) {
            uint32_t k_pos = tile_start + ki;
            uint32_t k_base = (k_pos * args.heads + head) * args.head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < args.head_dim; d++) {
                dot = fmaf(shared_query[d], h3_bf16_to_f32(key[k_base + d]),
                           dot);
            }
            tile_scores[ki] = dot;
            local_max = fmaxf(local_max, dot);
        }
        reductions[tid] = local_max;
        __syncthreads();
        for (uint32_t stride = threads / 2; stride; stride >>= 1) {
            if (tid < stride) {
                reductions[tid] =
                    fmaxf(reductions[tid], reductions[tid + stride]);
            }
            __syncthreads();
        }
        float tile_max = reductions[0];

        if (tid == 0) {
            float new_max = fmaxf(softmax_max, tile_max);
            reductions[0] = expf(softmax_max - new_max);
            softmax_max = new_max;
        }
        __syncthreads();
        float rescale = reductions[0];

        for (uint32_t d = tid; d < args.head_dim; d += threads) acc[d] *= rescale;
        __syncthreads();

        for (uint32_t d = tid; d < args.head_dim; d += threads) {
            float sum = 0.0f;
            for (uint32_t ki = 0; ki < tile_count; ki++) {
                uint32_t k_pos = tile_start + ki;
                float prob = expf(tile_scores[ki] - softmax_max);
                uint32_t v_index =
                    (k_pos * args.heads + head) * args.head_dim + d;
                sum = fmaf(prob, h3_bf16_to_f32(value[v_index]), sum);
            }
            acc[d] += sum;
        }
        __syncthreads();

        float local_prob_sum = 0.0f;
        for (uint32_t ki = tid; ki < tile_count; ki += threads) {
            local_prob_sum += expf(tile_scores[ki] - softmax_max);
        }
        reductions[tid] = local_prob_sum;
        __syncthreads();
        for (uint32_t stride = threads / 2; stride; stride >>= 1) {
            if (tid < stride) reductions[tid] += reductions[tid + stride];
            __syncthreads();
        }
        if (tid == 0) softmax_sum = softmax_sum * rescale + reductions[0];
        __syncthreads();
    }

    float inv_sum = softmax_sum > 0.0f ? 1.0f / softmax_sum : 0.0f;
    for (uint32_t d = tid; d < args.head_dim; d += threads) {
        output[h3_sdpa_output_index(args, q_pos, head, d)] =
            h3_f32_to_bf16(acc[d] * inv_sum);
    }
}

__global__ void h3_sdpa_f32_tiled_kernel(const float *query, const float *key,
                                         const float *value, float *output,
                                         h3_sdpa_args args) {
    __shared__ float reductions[128];
    __shared__ float shared_query[128];
    __shared__ float tile_scores[H3_SDPA_TILE];
    __shared__ float acc[128];
    __shared__ float softmax_max;
    __shared__ float softmax_sum;

    uint32_t head = blockIdx.x;
    uint32_t q_pos = blockIdx.y;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (head >= args.heads || q_pos >= args.sequence || args.head_dim > 128u)
        return;

    uint32_t q_base = (q_pos * args.heads + head) * args.head_dim;
    for (uint32_t d = tid; d < args.head_dim; d += threads) {
        shared_query[d] = query[q_base + d] * args.scale;
        acc[d] = 0.0f;
    }
    if (tid == 0) {
        softmax_max = -1e30f;
        softmax_sum = 0.0f;
    }
    __syncthreads();

    for (uint32_t tile_start = 0; tile_start < args.sequence;
         tile_start += H3_SDPA_TILE) {
        uint32_t tile_count = args.sequence - tile_start;
        if (tile_count > H3_SDPA_TILE) tile_count = H3_SDPA_TILE;

        float local_max = -1e30f;
        for (uint32_t ki = tid; ki < tile_count; ki += threads) {
            uint32_t k_pos = tile_start + ki;
            uint32_t k_base = (k_pos * args.heads + head) * args.head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < args.head_dim; d++) {
                dot = fmaf(shared_query[d], key[k_base + d], dot);
            }
            tile_scores[ki] = dot;
            local_max = fmaxf(local_max, dot);
        }
        reductions[tid] = local_max;
        __syncthreads();
        for (uint32_t stride = threads / 2; stride; stride >>= 1) {
            if (tid < stride) {
                reductions[tid] =
                    fmaxf(reductions[tid], reductions[tid + stride]);
            }
            __syncthreads();
        }
        float tile_max = reductions[0];

        if (tid == 0) {
            float new_max = fmaxf(softmax_max, tile_max);
            reductions[0] = expf(softmax_max - new_max);
            softmax_max = new_max;
        }
        __syncthreads();
        float rescale = reductions[0];

        for (uint32_t d = tid; d < args.head_dim; d += threads) acc[d] *= rescale;
        __syncthreads();

        for (uint32_t d = tid; d < args.head_dim; d += threads) {
            float sum = 0.0f;
            for (uint32_t ki = 0; ki < tile_count; ki++) {
                uint32_t k_pos = tile_start + ki;
                float prob = expf(tile_scores[ki] - softmax_max);
                uint32_t v_index =
                    (k_pos * args.heads + head) * args.head_dim + d;
                sum = fmaf(prob, value[v_index], sum);
            }
            acc[d] += sum;
        }
        __syncthreads();

        float local_prob_sum = 0.0f;
        for (uint32_t ki = tid; ki < tile_count; ki += threads) {
            local_prob_sum += expf(tile_scores[ki] - softmax_max);
        }
        reductions[tid] = local_prob_sum;
        __syncthreads();
        for (uint32_t stride = threads / 2; stride; stride >>= 1) {
            if (tid < stride) reductions[tid] += reductions[tid + stride];
            __syncthreads();
        }
        if (tid == 0) softmax_sum = softmax_sum * rescale + reductions[0];
        __syncthreads();
    }

    float inv_sum = softmax_sum > 0.0f ? 1.0f / softmax_sum : 0.0f;
    for (uint32_t d = tid; d < args.head_dim; d += threads) {
        output[q_base + d] = acc[d] * inv_sum;
    }
}

__global__ void h3_euler_bf16_kernel(float *sample, const uint16_t *last,
                                   const uint16_t *previous,
                                   h3_euler_args args) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= args.elements) return;
    float last_value = h3_bf16_to_f32(last[gid]);
    float velocity = fmaf(args.ratio,
                          last_value - h3_bf16_to_f32(previous[gid]),
                          last_value);
    uint32_t sample_index = args.sample_offset + gid;
    sample[sample_index] = fmaf(args.delta, velocity, sample[sample_index]);
}

static int h3_launch_1d(uint32_t count, hipStream_t stream,
                        const void *kernel, size_t args_size, void **args) {
    if (!count) return 1;
    uint32_t block = 256;
    uint32_t grid = (count + block - 1u) / block;
    return hipLaunchKernel(kernel, dim3(grid), dim3(block), args, args_size,
                           stream) == hipSuccess;
}

static uint32_t h3_row_threads(uint32_t width) {
    uint32_t threads = 1;
    while (threads * 2u <= 256u) threads *= 2u;
    if (threads > width) {
        threads = 1;
        while (threads < width && threads * 2u <= 256u) threads *= 2u;
    }
    return threads ? threads : 1u;
}

extern "C" int h3_launch_copy_f32(const float *input, float *output,
                                  uint32_t count, hipStream_t stream) {
    void *args[] = {&input, &output, &count};
    return h3_launch_1d(count, stream,
                        reinterpret_cast<const void *>(h3_copy_f32_kernel),
                        sizeof(args), args);
}

extern "C" int h3_launch_copy_u16(const uint16_t *input, uint16_t *output,
                                  uint32_t count, hipStream_t stream) {
    void *args[] = {&input, &output, &count};
    return h3_launch_1d(count, stream,
                        reinterpret_cast<const void *>(h3_copy_u16_kernel),
                        sizeof(args), args);
}

extern "C" int h3_launch_cast_f32_to_bf16(const float *input, uint16_t *output,
                                          uint32_t count, hipStream_t stream) {
    void *args[] = {&input, &output, &count};
    return h3_launch_1d(count, stream,
                        reinterpret_cast<const void *>(h3_cast_f32_to_bf16_kernel),
                        sizeof(args), args);
}

extern "C" int h3_launch_cast_bf16_to_f32(const uint16_t *input, float *output,
                                          uint32_t count, hipStream_t stream) {
    void *args[] = {&input, &output, &count};
    return h3_launch_1d(count, stream,
                        reinterpret_cast<const void *>(h3_cast_bf16_to_f32_kernel),
                        sizeof(args), args);
}

extern "C" int h3_launch_add_bf16(const uint16_t *left, const uint16_t *right,
                                  uint16_t *output, uint32_t count,
                                  hipStream_t stream) {
    void *args[] = {&left, &right, &output, &count};
    return h3_launch_1d(count, stream,
                        reinterpret_cast<const void *>(h3_add_bf16_kernel),
                        sizeof(args), args);
}

extern "C" int h3_launch_sub_bf16(const uint16_t *left, const uint16_t *right,
                                  uint16_t *output, uint32_t count,
                                  hipStream_t stream) {
    void *args[] = {&left, &right, &output, &count};
    return h3_launch_1d(count, stream,
                        reinterpret_cast<const void *>(h3_sub_bf16_kernel),
                        sizeof(args), args);
}

extern "C" int h3_launch_silu_bf16(const uint16_t *input, uint16_t *output,
                                     uint32_t count, hipStream_t stream) {
    void *args[] = {&input, &output, &count};
    return h3_launch_1d(count, stream,
                        reinterpret_cast<const void *>(h3_silu_bf16_kernel),
                        sizeof(args), args);
}

extern "C" int h3_launch_silu_f32(const float *input, float *output,
                                  uint32_t count, hipStream_t stream) {
    void *args[] = {&input, &output, &count};
    return h3_launch_1d(count, stream,
                        reinterpret_cast<const void *>(h3_silu_f32_kernel),
                        sizeof(args), args);
}

extern "C" int h3_launch_gelu_bf16(const uint16_t *input, uint16_t *output,
                                   const h3_gelu_bf16_args *args,
                                   hipStream_t stream) {
    h3_gelu_bf16_args host_args = *args;
    void *kernel_args[] = {&input, &output, &host_args};
    return h3_launch_1d(args->elements, stream,
                        reinterpret_cast<const void *>(h3_gelu_bf16_kernel),
                        sizeof(kernel_args), kernel_args);
}

extern "C" int h3_launch_swiglu_bf16(const uint16_t *fused, uint16_t *output,
                                     const h3_swiglu_args *args,
                                     hipStream_t stream) {
    h3_swiglu_args host_args = *args;
    void *kernel_args[] = {&fused, &output, &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.width + 15) / 16, (host_args.rows + 15) / 16, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_swiglu_bf16_kernel), grid,
               block, kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_swiglu_f32(const float *fused, float *output,
                                    const h3_swiglu_args *args,
                                    hipStream_t stream) {
    h3_swiglu_args host_args = *args;
    void *kernel_args[] = {&fused, &output, &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.width + 15) / 16, (host_args.rows + 15) / 16, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_swiglu_f32_kernel), grid,
               block, kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_clip_f32(const float *input, float *output,
                                  const h3_clip_f32_args *args,
                                  hipStream_t stream) {
    h3_clip_f32_args host_args = *args;
    void *kernel_args[] = {&input, &output, &host_args};
    return h3_launch_1d(args->elements, stream,
                        reinterpret_cast<const void *>(h3_clip_f32_kernel),
                        sizeof(kernel_args), kernel_args);
}

extern "C" int h3_launch_alias_free_snake_f32(
    const float *input, const float *alpha_log, const float *beta_log,
    const float *upsample_filter, const float *downsample_filter, float *output,
    const h3_audio_activation_args *args, hipStream_t stream) {
    h3_audio_activation_args host_args = *args;
    void *kernel_args[] = {&input,     &alpha_log, &beta_log,
                           &upsample_filter, &downsample_filter, &output,
                           &host_args};
    dim3 block(8, 8, 1);
    dim3 grid((host_args.channels + 7) / 8, (host_args.length + 7) / 8,
              host_args.batch);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_alias_free_snake_f32_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}

extern "C" int h3_launch_snake1d_f32(const float *input, const float *alpha,
                                     float *output,
                                     const h3_audio_activation_args *args,
                                     hipStream_t stream) {
    h3_audio_activation_args host_args = *args;
    void *kernel_args[] = {&input, &alpha, &output, &host_args};
    return h3_launch_1d(host_args.batch * host_args.length * host_args.channels,
                        stream,
                        reinterpret_cast<const void *>(h3_snake1d_f32_kernel),
                        sizeof(kernel_args), kernel_args);
}

extern "C" int h3_launch_audio_qkv_split_f32(
    const float *qkv, const float *q_bias, const float *k_bias,
    const float *v_bias, float *query, float *key, float *value,
    const h3_audio_qkv_args *args, hipStream_t stream) {
    h3_audio_qkv_args host_args = *args;
    void *kernel_args[] = {&qkv,     &q_bias, &k_bias, &v_bias,
                           &query,   &key,    &value,  &host_args};
    return h3_launch_1d(host_args.batch * host_args.length *
                            host_args.heads * host_args.head_dim,
                        stream,
                        reinterpret_cast<const void *>(
                            h3_audio_qkv_split_f32_kernel),
                        sizeof(kernel_args), kernel_args);
}

extern "C" int h3_launch_sdpa_causal_f32(const float *query, const float *key,
                                         const float *value, float *output,
                                         const h3_sdpa_causal_args *args,
                                         hipStream_t stream) {
    if (!args->sequence || !args->heads || !args->head_dim || !args->batch ||
        args->head_dim > 256) {
        return 0;
    }
    h3_sdpa_causal_args host_args = *args;
    void *kernel_args[] = {&query, &key, &value, &output, &host_args};
    dim3 grid(host_args.heads, host_args.sequence, host_args.batch);
    if (!getenv("H3_SDPA_LEGACY")) {
        return hipLaunchKernel(
                   reinterpret_cast<const void *>(h3_sdpa_causal_f32_wave_kernel),
                   grid, dim3(32), kernel_args, sizeof(kernel_args),
                   stream) == hipSuccess;
    }
    if (host_args.sequence > 512) return 0;
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_sdpa_causal_f32_kernel), grid,
               dim3(1), kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_audio_attention_pool_f32(
    const float *attended, float *output, const h3_audio_pool_args *args,
    hipStream_t stream) {
    h3_audio_pool_args host_args = *args;
    void *kernel_args[] = {&attended, &output, &host_args};
    return h3_launch_1d(host_args.batch * host_args.length * host_args.output_dim,
                        stream,
                        reinterpret_cast<const void *>(
                            h3_audio_attention_pool_f32_kernel),
                        sizeof(kernel_args), kernel_args);
}

extern "C" int h3_launch_vae_encoder_pad_f32(
    const float *input, float *output, const h3_vae_encoder_pad_args *args,
    hipStream_t stream) {
    h3_vae_encoder_pad_args host_args = *args;
    void *kernel_args[] = {&input, &output, &host_args};
    uint64_t out_height = (uint64_t)host_args.height + host_args.height_before +
                          host_args.height_after;
    uint64_t out_width = (uint64_t)host_args.width + host_args.width_before +
                         host_args.width_after;
    uint64_t out_depth = (uint64_t)host_args.depth + host_args.depth_front;
    uint64_t total = (uint64_t)host_args.batch * out_depth * out_height *
                     out_width * host_args.channels;
    if (!total) return 1;
    if (total > UINT32_MAX) return 0;
    return h3_launch_1d((uint32_t)total, stream,
                        reinterpret_cast<const void *>(
                            h3_vae_encoder_pad_f32_kernel),
                        sizeof(kernel_args), kernel_args);
}

extern "C" int h3_launch_vae_encoder_group_norm_silu_f32(
    const float *input, const float *weight, const float *bias, float *output,
    const h3_vae_encoder_norm_args *args, hipStream_t stream) {
    h3_vae_encoder_norm_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &bias, &output, &host_args};
    uint32_t rows = host_args.batch * host_args.depth * host_args.groups;
    uint32_t elements = host_args.height * host_args.width *
                        (host_args.channels / host_args.groups);
    uint32_t threads = h3_row_threads(elements);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(
                   h3_vae_encoder_group_norm_silu_f32_kernel),
               dim3(rows), dim3(threads), kernel_args, sizeof(kernel_args),
               stream) == hipSuccess;
}

extern "C" int h3_launch_add_scaled_f32(const float *left, const float *right,
                                      float *output,
                                      const h3_add_scaled_f32_args *args,
                                      hipStream_t stream) {
    h3_add_scaled_f32_args host_args = *args;
    void *kernel_args[] = {&left, &right, &output, &host_args};
    return h3_launch_1d(args->elements, stream,
                        reinterpret_cast<const void *>(h3_add_scaled_f32_kernel),
                        sizeof(kernel_args), kernel_args);
}

extern "C" int h3_launch_scale_add_f32(const float *residual,
                                       const float *branch, const float *scale,
                                       float *output, const h3_swiglu_args *args,
                                       hipStream_t stream) {
    h3_swiglu_args host_args = *args;
    void *kernel_args[] = {&residual, &branch, &scale, &output, &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.width + 15) / 16, (host_args.rows + 15) / 16, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_scale_add_f32_kernel), grid,
               block, kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_weight_norm_f32(const float *vector,
                                         const float *magnitude, float *output,
                                         const h3_weight_norm_args *args,
                                         hipStream_t stream) {
    h3_weight_norm_args host_args = *args;
    void *kernel_args[] = {&vector, &magnitude, &output, &host_args};
    return h3_launch_1d(args->outer, stream,
                        reinterpret_cast<const void *>(h3_weight_norm_f32_kernel),
                        sizeof(kernel_args), kernel_args);
}

extern "C" int h3_launch_geglu_f32(const float *gate, const float *linear,
                                   float *output, uint32_t count,
                                   hipStream_t stream) {
    void *kernel_args[] = {&gate, &linear, &output, &count};
    return h3_launch_1d(count, stream,
                        reinterpret_cast<const void *>(h3_geglu_f32_kernel),
                        sizeof(kernel_args), kernel_args);
}

extern "C" int h3_launch_rms_norm_bf16(const uint16_t *input,
                                       const uint16_t *weight, uint16_t *output,
                                       const h3_norm_args *args,
                                       hipStream_t stream) {
    h3_norm_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &output, &host_args};
    uint32_t threads = h3_row_threads(host_args.width);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_rms_norm_bf16_kernel),
               dim3(host_args.rows), dim3(threads), kernel_args,
               sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_rms_norm_f32(const float *input, const float *weight,
                                      float *output, const h3_norm_args *args,
                                      hipStream_t stream) {
    h3_norm_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &output, &host_args};
    uint32_t threads = h3_row_threads(host_args.width);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_rms_norm_f32_kernel),
               dim3(host_args.rows), dim3(threads), kernel_args,
               sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_layer_norm_bf16(const uint16_t *input,
                                         const uint16_t *weight,
                                         const uint16_t *bias,
                                         uint16_t *output,
                                         const h3_norm_args *args,
                                         hipStream_t stream) {
    h3_norm_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &bias, &output, &host_args};
    uint32_t threads = h3_row_threads(host_args.width);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_layer_norm_bf16_kernel),
               dim3(host_args.rows), dim3(threads), kernel_args,
               sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_layer_norm_f32(const float *input, const float *weight,
                                        const float *bias, float *output,
                                        const h3_norm_args *args,
                                        hipStream_t stream) {
    h3_norm_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &bias, &output, &host_args};
    uint32_t threads = h3_row_threads(host_args.width);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_layer_norm_f32_kernel),
               dim3(host_args.rows), dim3(threads), kernel_args,
               sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_linear_bf16(const uint16_t *input,
                                     const uint16_t *weight,
                                     const uint16_t *bias, uint16_t *output,
                                     const h3_linear_args *args,
                                     hipStream_t stream) {
    h3_linear_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &bias, &output, &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.output_dim + 15) / 16, (host_args.rows + 15) / 16, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_linear_bf16_kernel), grid,
               block, kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_gate_bf16(const uint16_t *residual,
                                   const uint16_t *branch,
                                   const uint16_t *modulation,
                                   const uint32_t *row_map, uint16_t *output,
                                   const h3_gate_args *args,
                                   hipStream_t stream) {
    h3_gate_args host_args = *args;
    void *kernel_args[] = {&residual, &branch, &modulation, &row_map, &output,
                           &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.width + 15) / 16, (host_args.rows + 15) / 16, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_gate_bf16_kernel), grid, block,
               kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_gate_f32(const float *residual, const float *branch,
                                  const float *modulation,
                                  const uint32_t *row_map, float *output,
                                  const h3_gate_args *args, hipStream_t stream) {
    h3_gate_args host_args = *args;
    void *kernel_args[] = {&residual, &branch, &modulation, &row_map, &output,
                           &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.width + 15) / 16, (host_args.rows + 15) / 16, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_gate_f32_kernel), grid, block,
               kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_adaln_bf16(const uint16_t *input,
                                    const uint16_t *weight,
                                    const uint16_t *modulation,
                                    const uint32_t *row_map, uint16_t *output,
                                    const h3_adaln_args *args,
                                    hipStream_t stream) {
    h3_adaln_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &modulation, &row_map, &output,
                           &host_args};
    uint32_t threads = h3_row_threads(host_args.width);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_adaln_bf16_kernel),
               dim3(host_args.rows), dim3(threads), kernel_args,
               sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_adaln_f32(const float *input, const float *weight,
                                   const float *modulation,
                                   const uint32_t *row_map, float *output,
                                   const h3_adaln_args *args,
                                   hipStream_t stream) {
    h3_adaln_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &modulation, &row_map, &output,
                           &host_args};
    uint32_t threads = h3_row_threads(host_args.width);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_adaln_f32_kernel),
               dim3(host_args.rows), dim3(threads), kernel_args,
               sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_gate_adaln_bf16(const uint16_t *residual,
                                         const uint16_t *branch,
                                         const uint16_t *gate_modulation,
                                         const uint32_t *row_map,
                                         const uint16_t *weight,
                                         const uint16_t *norm_modulation,
                                         uint16_t *gated_residual,
                                         uint16_t *output,
                                         const h3_gate_adaln_args *args,
                                         hipStream_t stream) {
    h3_gate_adaln_args host_args = *args;
    void *kernel_args[] = {&residual, &branch, &gate_modulation, &row_map,
                           &weight, &norm_modulation, &gated_residual, &output,
                           &host_args};
    uint32_t threads = h3_row_threads(host_args.width);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_gate_adaln_bf16_kernel),
               dim3(host_args.rows), dim3(threads), kernel_args,
               sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_qkv_rope_bf16(const uint16_t *qkv,
                                       const uint16_t *q_weight,
                                       const uint16_t *k_weight,
                                       const uint16_t *rope_cos,
                                       const uint16_t *rope_sin,
                                       uint16_t *query, uint16_t *key,
                                       uint16_t *value, const h3_qkv_args *args,
                                       hipStream_t stream) {
    h3_qkv_args host_args = *args;
    void *kernel_args[] = {&qkv, &q_weight, &k_weight, &rope_cos, &rope_sin,
                           &query, &key, &value, &host_args};
    dim3 block(host_args.head_dim, 1, 1);
    dim3 grid(1, host_args.heads, host_args.sequence);
    if (host_args.head_dim > 256) return 0;
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_qkv_rope_bf16_kernel), grid,
               block, kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_qkv_rope_f32(const float *qkv, const float *q_weight,
                                      const float *k_weight,
                                      const float *rope_cos,
                                      const float *rope_sin, float *query,
                                      float *key, float *value,
                                      const h3_qkv_args *args,
                                      hipStream_t stream) {
    h3_qkv_args host_args = *args;
    void *kernel_args[] = {&qkv, &q_weight, &k_weight, &rope_cos, &rope_sin,
                           &query, &key, &value, &host_args};
    dim3 block(host_args.head_dim, 1, 1);
    dim3 grid(1, host_args.heads, host_args.sequence);
    if (host_args.head_dim > 256) return 0;
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_qkv_rope_f32_kernel), grid,
               block, kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_video_qkv_rope_f32(const float *qkv,
                                            const float *rope_cos,
                                            const float *rope_sin,
                                            float *query, float *key,
                                            float *value, const h3_qkv_args *args,
                                            hipStream_t stream) {
    h3_qkv_args host_args = *args;
    void *kernel_args[] = {&qkv, &rope_cos, &rope_sin, &query, &key, &value,
                           &host_args};
    dim3 block(host_args.head_dim, 1, 1);
    dim3 grid(1, host_args.heads, host_args.sequence);
    if (host_args.head_dim > 256) return 0;
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_video_qkv_rope_f32_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}

extern "C" int h3_launch_vision_qkv_rope_bf16(
    const uint16_t *qkv, const uint16_t *rope_cos, const uint16_t *rope_sin,
    uint16_t *query, uint16_t *key, uint16_t *value, const h3_qkv_args *args,
    hipStream_t stream) {
    h3_qkv_args host_args = *args;
    void *kernel_args[] = {&qkv, &rope_cos, &rope_sin, &query, &key, &value,
                           &host_args};
    dim3 block(host_args.head_dim, 1, 1);
    dim3 grid(1, host_args.heads, host_args.sequence);
    if (host_args.head_dim > 256) return 0;
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_vision_qkv_rope_bf16_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}

__global__ void h3_transpose_shd_hsd_bf16_kernel(const uint16_t *input,
                                                 uint16_t *output,
                                                 uint32_t sequence,
                                                 uint32_t heads,
                                                 uint32_t head_dim) {
    uint32_t dimension = threadIdx.x;
    uint32_t head = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (dimension >= head_dim || head >= heads || row >= sequence) return;
    uint32_t src = (row * heads + head) * head_dim + dimension;
    uint32_t dst = (head * sequence + row) * head_dim + dimension;
    output[dst] = input[src];
}

__global__ void h3_transpose_shd_hsd_f32_kernel(const float *input,
                                                float *output,
                                                uint32_t sequence,
                                                uint32_t heads,
                                                uint32_t head_dim) {
    uint32_t dimension = threadIdx.x;
    uint32_t head = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (dimension >= head_dim || head >= heads || row >= sequence) return;
    uint32_t src = (row * heads + head) * head_dim + dimension;
    uint32_t dst = (head * sequence + row) * head_dim + dimension;
    output[dst] = input[src];
}

extern "C" int h3_launch_transpose_shd_hsd_bf16(
    const uint16_t *input, uint16_t *output, uint32_t sequence, uint32_t heads,
    uint32_t head_dim, hipStream_t stream) {
    if (!sequence || !heads || !head_dim || head_dim > 256) return 0;
    void *kernel_args[] = {&input, &output, &sequence, &heads, &head_dim};
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_transpose_shd_hsd_bf16_kernel),
               dim3(heads, sequence, 1), dim3(head_dim, 1, 1), kernel_args,
               sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_transpose_shd_hsd_f32(const float *input,
                                               float *output,
                                               uint32_t sequence,
                                               uint32_t heads,
                                               uint32_t head_dim,
                                               hipStream_t stream) {
    if (!sequence || !heads || !head_dim || head_dim > 256) return 0;
    void *kernel_args[] = {&input, &output, &sequence, &heads, &head_dim};
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_transpose_shd_hsd_f32_kernel),
               dim3(heads, sequence, 1), dim3(head_dim, 1, 1), kernel_args,
               sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_sdpa_bf16(const uint16_t *query, const uint16_t *key,
                                   const uint16_t *value, uint16_t *output,
                                   const h3_sdpa_args *args,
                                   hipStream_t stream) {
    if (!args->sequence || !args->heads || !args->head_dim ||
        args->head_dim > 128) {
        return 0;
    }
    h3_sdpa_args host_args = *args;
    void *kernel_args[] = {&query, &key, &value, &output, &host_args};
    if (!getenv("H3_SDPA_LEGACY")) {
        /* Default: one-warp-per-query online softmax. Optional flash path
         * (H3_SDPA_FLASH=1) stages K/V tiles in LDS for multi-Q reuse; kept
         * opt-in after regressing vs wave on gfx1151 DiT shapes. */
        if (host_args.head_dim == 128 && getenv("H3_SDPA_FLASH") &&
            strcmp(getenv("H3_SDPA_FLASH"), "0") != 0) {
            dim3 grid(host_args.heads,
                      (host_args.sequence + H3_SDPA_FLASH_Q - 1u) /
                          H3_SDPA_FLASH_Q,
                      1);
            dim3 block(32, H3_SDPA_FLASH_Q, 1);
            return hipLaunchKernel(
                       reinterpret_cast<const void *>(
                           h3_sdpa_bf16_flash_d128_kernel),
                       grid, block, kernel_args, sizeof(kernel_args),
                       stream) == hipSuccess;
        }
        /* Grid (sequence, heads): consecutive blocks share one KV head so
         * head-major K/V stays hot in L2 (gfx1151). */
        dim3 grid(host_args.sequence, host_args.heads, 1);
        const void *kernel = host_args.head_dim == 128 ?
            reinterpret_cast<const void *>(h3_sdpa_bf16_wave_d128_kernel) :
            reinterpret_cast<const void *>(h3_sdpa_bf16_wave_kernel);
        return hipLaunchKernel(kernel, grid, dim3(32), kernel_args,
                               sizeof(kernel_args), stream) == hipSuccess;
    }
    dim3 grid(host_args.heads, host_args.sequence, 1);
    if (host_args.sequence <= H3_SDPA_TILE) {
        return hipLaunchKernel(
                   reinterpret_cast<const void *>(h3_sdpa_bf16_kernel), grid,
                   dim3(1), kernel_args, sizeof(kernel_args), stream) ==
               hipSuccess;
    }
    dim3 block(h3_sdpa_threads(host_args.sequence), 1, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_sdpa_bf16_tiled_kernel), grid,
               block, kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_sdpa_f32(const float *query, const float *key,
                                  const float *value, float *output,
                                  const h3_sdpa_args *args,
                                  hipStream_t stream) {
    if (!args->sequence || !args->heads || !args->head_dim ||
        args->head_dim > 128) {
        return 0;
    }
    h3_sdpa_args host_args = *args;
    void *kernel_args[] = {&query, &key, &value, &output, &host_args};
    if (!getenv("H3_SDPA_LEGACY")) {
        if (host_args.head_dim == 128 && getenv("H3_SDPA_FLASH") &&
            strcmp(getenv("H3_SDPA_FLASH"), "0") != 0) {
            dim3 grid(host_args.heads,
                      (host_args.sequence + H3_SDPA_FLASH_Q - 1u) /
                          H3_SDPA_FLASH_Q,
                      1);
            dim3 block(32, H3_SDPA_FLASH_Q, 1);
            return hipLaunchKernel(
                       reinterpret_cast<const void *>(
                           h3_sdpa_f32_flash_d128_kernel),
                       grid, block, kernel_args, sizeof(kernel_args),
                       stream) == hipSuccess;
        }
        dim3 grid(host_args.sequence, host_args.heads, 1);
        const void *kernel = host_args.head_dim == 128 ?
            reinterpret_cast<const void *>(h3_sdpa_f32_wave_d128_kernel) :
            host_args.head_dim == 64 ?
            reinterpret_cast<const void *>(h3_sdpa_f32_wave_d64_kernel) :
            reinterpret_cast<const void *>(h3_sdpa_f32_wave_kernel);
        return hipLaunchKernel(kernel, grid, dim3(32), kernel_args,
                               sizeof(kernel_args), stream) == hipSuccess;
    }
    dim3 grid(host_args.heads, host_args.sequence, 1);
    if (host_args.sequence <= H3_SDPA_TILE) {
        return hipLaunchKernel(
                   reinterpret_cast<const void *>(h3_sdpa_f32_kernel), grid,
                   dim3(1), kernel_args, sizeof(kernel_args), stream) ==
               hipSuccess;
    }
    dim3 block(h3_sdpa_threads(host_args.sequence), 1, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_sdpa_f32_tiled_kernel), grid,
               block, kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_euler_bf16(float *sample, const uint16_t *last,
                                  const uint16_t *previous,
                                  const h3_euler_args *args,
                                  hipStream_t stream) {
    h3_euler_args host_args = *args;
    void *kernel_args[] = {&sample, &last, &previous, &host_args};
    return h3_launch_1d(host_args.elements, stream,
                        reinterpret_cast<const void *>(h3_euler_bf16_kernel),
                        sizeof(kernel_args), kernel_args);
}
