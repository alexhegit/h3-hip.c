#include "h3_kernels.h"
#include "h3_wave32.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>

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

/* Two bf16 in one dword so an LDS tile can be filled with b32 stores. */
__device__ inline uint32_t h3_pack_bf16x2(float low, float high) {
    return (uint32_t)h3_f32_to_bf16(low) |
           ((uint32_t)h3_f32_to_bf16(high) << 16);
}

__device__ inline uint32_t h3_pack_f16x2(float low, float high) {
    union {
        _Float16 h;
        uint16_t u;
    } a, b;
    a.h = (_Float16)low;
    b.h = (_Float16)high;
    return (uint32_t)a.u | ((uint32_t)b.u << 16);
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
    uint32_t gid = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (gid + 3u < count) {
        uint2 a = *reinterpret_cast<const uint2 *>(left + gid);
        uint2 b = *reinterpret_cast<const uint2 *>(right + gid);
        uint16_t av[4] = {(uint16_t)(a.x & 0xffffu), (uint16_t)(a.x >> 16),
                          (uint16_t)(a.y & 0xffffu), (uint16_t)(a.y >> 16)};
        uint16_t bv[4] = {(uint16_t)(b.x & 0xffffu), (uint16_t)(b.x >> 16),
                          (uint16_t)(b.y & 0xffffu), (uint16_t)(b.y >> 16)};
        uint32_t o0 =
            (uint32_t)h3_f32_to_bf16(h3_bf16_to_f32(av[0]) + h3_bf16_to_f32(bv[0])) |
            ((uint32_t)h3_f32_to_bf16(h3_bf16_to_f32(av[1]) +
                                      h3_bf16_to_f32(bv[1]))
             << 16);
        uint32_t o1 =
            (uint32_t)h3_f32_to_bf16(h3_bf16_to_f32(av[2]) + h3_bf16_to_f32(bv[2])) |
            ((uint32_t)h3_f32_to_bf16(h3_bf16_to_f32(av[3]) +
                                      h3_bf16_to_f32(bv[3]))
             << 16);
        uint2 out{o0, o1};
        *reinterpret_cast<uint2 *>(output + gid) = out;
        return;
    }
    for (uint32_t i = 0; gid + i < count; i++)
        output[gid + i] = h3_f32_to_bf16(h3_bf16_to_f32(left[gid + i]) +
                                         h3_bf16_to_f32(right[gid + i]));
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
    if ((args.width & 3u) == 0u) {
        for (uint32_t k = tid * 4u; k < args.width; k += threads * 4u) {
            uint2 packed = *reinterpret_cast<const uint2 *>(x + k);
            float f0 = h3_bf16_to_f32((uint16_t)(packed.x & 0xffffu));
            float f1 = h3_bf16_to_f32((uint16_t)(packed.x >> 16));
            float f2 = h3_bf16_to_f32((uint16_t)(packed.y & 0xffffu));
            float f3 = h3_bf16_to_f32((uint16_t)(packed.y >> 16));
            local_sum = fmaf(f0, f0, local_sum);
            local_sum = fmaf(f1, f1, local_sum);
            local_sum = fmaf(f2, f2, local_sum);
            local_sum = fmaf(f3, f3, local_sum);
        }
    } else {
        for (uint32_t k = tid; k < args.width; k += threads) {
            float value = h3_bf16_to_f32(x[k]);
            local_sum = fmaf(value, value, local_sum);
        }
    }
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse = rsqrtf(reductions[0] / float(args.width) + args.epsilon);
    if ((args.width & 3u) == 0u) {
        uint16_t *row_out = output + row * args.width;
        for (uint32_t column = tid * 4u; column < args.width;
             column += threads * 4u) {
            uint2 xv = *reinterpret_cast<const uint2 *>(x + column);
            uint2 wv = *reinterpret_cast<const uint2 *>(weight + column);
            uint16_t xs[4] = {(uint16_t)(xv.x & 0xffffu), (uint16_t)(xv.x >> 16),
                              (uint16_t)(xv.y & 0xffffu), (uint16_t)(xv.y >> 16)};
            uint16_t ws[4] = {(uint16_t)(wv.x & 0xffffu), (uint16_t)(wv.x >> 16),
                              (uint16_t)(wv.y & 0xffffu), (uint16_t)(wv.y >> 16)};
            uint32_t o0 =
                (uint32_t)h3_f32_to_bf16(h3_bf16_to_f32(xs[0]) * inverse *
                                         h3_bf16_to_f32(ws[0])) |
                ((uint32_t)h3_f32_to_bf16(h3_bf16_to_f32(xs[1]) * inverse *
                                          h3_bf16_to_f32(ws[1]))
                 << 16);
            uint32_t o1 =
                (uint32_t)h3_f32_to_bf16(h3_bf16_to_f32(xs[2]) * inverse *
                                         h3_bf16_to_f32(ws[2])) |
                ((uint32_t)h3_f32_to_bf16(h3_bf16_to_f32(xs[3]) * inverse *
                                          h3_bf16_to_f32(ws[3]))
                 << 16);
            uint2 packed{o0, o1};
            *reinterpret_cast<uint2 *>(row_out + column) = packed;
        }
    } else {
        for (uint32_t column = tid; column < args.width; column += threads) {
            float normalized = h3_bf16_to_f32(x[column]) * inverse;
            output[row * args.width + column] =
                h3_f32_to_bf16(normalized * h3_bf16_to_f32(weight[column]));
        }
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

__device__ inline float h3_warp_reduce_sum(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        value += h3_shfl_xor32(value, mask);
    return value;
}

__device__ inline float h3_fast_exp(float x) {
    return exp2f(x * 1.4426950408889634f);
}

/* 4 consecutive bf16 values (8 bytes). Lane L owns dims 4L..4L+3 so one
 * uint2 load covers the whole d=128 row (32*8 = 256 bytes). */
__device__ inline void h3_load_bf16x4(const uint16_t *ptr, float out[4]) {
    uint2 packed = *reinterpret_cast<const uint2 *>(ptr);
    out[0] = h3_bf16_to_f32((uint16_t)packed.x);
    out[1] = h3_bf16_to_f32((uint16_t)(packed.x >> 16));
    out[2] = h3_bf16_to_f32((uint16_t)packed.y);
    out[3] = h3_bf16_to_f32((uint16_t)(packed.y >> 16));
}

__device__ inline void h3_store_bf16x4(uint16_t *ptr, float x0, float x1,
                                       float x2, float x3) {
    uint32_t o0 = (uint32_t)h3_f32_to_bf16(x0) |
                  ((uint32_t)h3_f32_to_bf16(x1) << 16);
    uint32_t o1 = (uint32_t)h3_f32_to_bf16(x2) |
                  ((uint32_t)h3_f32_to_bf16(x3) << 16);
    *reinterpret_cast<uint2 *>(ptr) = uint2{o0, o1};
}

#define H3_QKV_ROPE_MAX_DIM 256
#define H3_QKV_ROPE_WAVES 8u

/* Vectorized head_dim=128 path: one wave per (row, head) with four consecutive
 * dims per lane, so every qkv / weight / rope / output access is a uint2. The
 * RMS sums are a wave reduction and the RoPE partner dim lives rope_half/4
 * lanes away, so there is no LDS and no barrier. Requires rope_half % 4 == 0
 * (true for 48 and 64) so each lane's four dims fall in one rope region. */
__global__ void __launch_bounds__(H3_QKV_ROPE_WAVES * 32u)
h3_qkv_rope_bf16_d128_kernel(const uint16_t *qkv, const uint16_t *q_weight,
                             const uint16_t *k_weight,
                             const uint16_t *rope_cos, const uint16_t *rope_sin,
                             uint16_t *query, uint16_t *key, uint16_t *value,
                             h3_qkv_args args) {
    const uint32_t wave = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t slot = blockIdx.x * H3_QKV_ROPE_WAVES + wave;
    if (slot >= args.sequence * args.heads) return;
    const uint32_t row = slot / args.heads;
    const uint32_t head = slot - row * args.heads;
    const uint32_t d0 = lane * 4u;
    const uint32_t inner = args.heads * 128u;
    uint32_t q_base = row * inner * 3u;
    q_base += args.grouped ? head * 384u : head * 128u;
    const uint32_t k_base = q_base + (args.grouped ? 128u : inner);
    const uint32_t v_base = q_base + (args.grouped ? 256u : inner * 2u);

    float qv[4], kv[4], wq[4], wk[4];
    h3_load_bf16x4(qkv + q_base + d0, qv);
    h3_load_bf16x4(qkv + k_base + d0, kv);
    h3_load_bf16x4(q_weight + d0, wq);
    h3_load_bf16x4(k_weight + d0, wk);
    float qs = 0.0f, ks = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        qs = fmaf(qv[j], qv[j], qs);
        ks = fmaf(kv[j], kv[j], ks);
    }
    float q_inverse = rsqrtf(h3_warp_reduce_sum(qs) / 128.0f + args.epsilon);
    float k_inverse = rsqrtf(h3_warp_reduce_sum(ks) / 128.0f + args.epsilon);
    float qw[4], kw[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        qw[j] = qv[j] * q_inverse * wq[j];
        kw[j] = kv[j] * k_inverse * wk[j];
    }

    /* Shuffle outside the rope branch: a source lane must be active, which it
     * would not be inside the divergent region below. */
    const uint32_t shift = args.rope_half >> 2;
    int src = (int)lane;
    if (lane < shift) src = (int)(lane + shift);
    else if (lane < shift * 2u) src = (int)(lane - shift);
    float qp[4], kp[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        qp[j] = h3_shfl32(qw[j], src);
        kp[j] = h3_shfl32(kw[j], src);
    }

    float qo[4], ko[4];
    if (lane < shift * 2u) {
        /* Lower half rotates with -sin, upper half with +sin. */
        float sign = lane < shift ? -1.0f : 1.0f;
        uint32_t rope_index = lane < shift ? d0 : d0 - args.rope_half;
        float c[4], s[4];
        h3_load_bf16x4(rope_cos + row * args.rope_half + rope_index, c);
        h3_load_bf16x4(rope_sin + row * args.rope_half + rope_index, s);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            qo[j] = fmaf(sign * qp[j], s[j], qw[j] * c[j]);
            ko[j] = fmaf(sign * kp[j], s[j], kw[j] * c[j]);
        }
    } else {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            qo[j] = qw[j];
            ko[j] = kw[j];
        }
    }

    uint32_t out_index = (row * args.heads + head) * 128u + d0;
    uint32_t kv_index = args.kv_head_major
                            ? (head * args.sequence + row) * 128u + d0
                            : out_index;
    h3_store_bf16x4(query + out_index, qo[0], qo[1], qo[2], qo[3]);
    h3_store_bf16x4(key + kv_index, ko[0], ko[1], ko[2], ko[3]);
    *reinterpret_cast<uint2 *>(value + kv_index) =
        *reinterpret_cast<const uint2 *>(qkv + v_base + d0);
}

/* One block per (row, head), one thread per dimension. The q/k row is read
 * once into LDS and the RMS sums come from a block reduction; the RoPE partner
 * dimension is then an LDS read. Doing the sums per thread instead would read
 * each row head_dim times, which is what made this ~9x slower than the 165 MB
 * of traffic it actually needs. */
__global__ void h3_qkv_rope_bf16_kernel(const uint16_t *qkv,
                                        const uint16_t *q_weight,
                                        const uint16_t *k_weight,
                                        const uint16_t *rope_cos,
                                        const uint16_t *rope_sin,
                                        uint16_t *query, uint16_t *key,
                                        uint16_t *value, h3_qkv_args args) {
    __shared__ float sq[H3_QKV_ROPE_MAX_DIM];
    __shared__ float sk[H3_QKV_ROPE_MAX_DIM];
    __shared__ float rq[H3_QKV_ROPE_MAX_DIM];
    __shared__ float rk[H3_QKV_ROPE_MAX_DIM];
    uint32_t dimension = threadIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t row = blockIdx.z;
    if (head >= args.heads || row >= args.sequence) return;
    /* Launcher contract: blockDim.x == head_dim <= H3_QKV_ROPE_MAX_DIM, so no
     * thread may exit early ahead of the barriers below. */
    uint32_t n = args.head_dim;
    uint32_t inner = args.heads * n;
    uint32_t row_base = row * inner * 3;
    uint32_t q_base = row_base + head * n;
    uint32_t k_base = q_base + inner;
    uint32_t v_base = q_base + inner * 2;
    if (args.grouped) {
        q_base = row_base + head * n * 3;
        k_base = q_base + n;
        v_base = k_base + n;
    }

    float qv = h3_bf16_to_f32(qkv[q_base + dimension]);
    float kv = h3_bf16_to_f32(qkv[k_base + dimension]);
    sq[dimension] = qv;
    sk[dimension] = kv;
    rq[dimension] = qv * qv;
    rk[dimension] = kv * kv;
    /* Pad the reduction to a power of two; pow2 < 2n, so dimension+n covers
     * the whole padded tail. */
    uint32_t pow2 = 1u;
    while (pow2 < n) pow2 <<= 1;
    if (dimension + n < pow2) {
        rq[dimension + n] = 0.0f;
        rk[dimension + n] = 0.0f;
    }
    __syncthreads();
    for (uint32_t stride = pow2 >> 1; stride; stride >>= 1) {
        if (dimension < stride) {
            rq[dimension] += rq[dimension + stride];
            rk[dimension] += rk[dimension + stride];
        }
        __syncthreads();
    }
    float q_inverse = rsqrtf(rq[0] / float(n) + args.epsilon);
    float k_inverse = rsqrtf(rk[0] / float(n) + args.epsilon);

    float q0 = qv * q_inverse * h3_bf16_to_f32(q_weight[dimension]);
    float k0 = kv * k_inverse * h3_bf16_to_f32(k_weight[dimension]);
    if (dimension < args.rope_half) {
        uint32_t pair = dimension + args.rope_half;
        float q1 = sq[pair] * q_inverse * h3_bf16_to_f32(q_weight[pair]);
        float k1 = sk[pair] * k_inverse * h3_bf16_to_f32(k_weight[pair]);
        float c = h3_bf16_to_f32(rope_cos[row * args.rope_half + dimension]);
        float s = h3_bf16_to_f32(rope_sin[row * args.rope_half + dimension]);
        q0 = q0 * c - q1 * s;
        k0 = k0 * c - k1 * s;
    } else if (dimension < args.rope_half * 2) {
        uint32_t pair = dimension - args.rope_half;
        float q1 = sq[pair] * q_inverse * h3_bf16_to_f32(q_weight[pair]);
        float k1 = sk[pair] * k_inverse * h3_bf16_to_f32(k_weight[pair]);
        float c = h3_bf16_to_f32(rope_cos[row * args.rope_half + pair]);
        float s = h3_bf16_to_f32(rope_sin[row * args.rope_half + pair]);
        q0 = q0 * c + q1 * s;
        k0 = k0 * c + k1 * s;
    }
    uint32_t output_index = (row * args.heads + head) * n + dimension;
    uint32_t kv_index = args.kv_head_major ?
        (head * args.sequence + row) * n + dimension : output_index;
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

__device__ inline float2 h3_load_f32x2(const float *ptr) {
    return *reinterpret_cast<const float2 *>(ptr);
}

__device__ inline void h3_store_f32x2(float *ptr, float x, float y) {
    *reinterpret_cast<float2 *>(ptr) = float2{x, y};
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

/* Two queries share each K/V load (d128 DiT). Not 2-key/4-key unroll.
 * Opt in: H3_SDPA_D128_Q2=1. Default is Q3. */
__global__ void __launch_bounds__(32)
h3_sdpa_bf16_wave_d128_q2_kernel(const uint16_t *query, const uint16_t *key,
                                 const uint16_t *value, uint16_t *output,
                                 h3_sdpa_args args) {
    uint32_t q_pos = blockIdx.x * 2u;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    int q1_live = (q_pos + 1u) < args.sequence;
    uint32_t q_base0 = (q_pos * args.heads + head) * 128u;
    uint32_t q_base1 = q_base0 + args.heads * 128u;
    float a0 = h3_bf16_to_f32(query[q_base0 + lane]) * args.scale;
    float a1 = h3_bf16_to_f32(query[q_base0 + 32u + lane]) * args.scale;
    float a2 = h3_bf16_to_f32(query[q_base0 + 64u + lane]) * args.scale;
    float a3 = h3_bf16_to_f32(query[q_base0 + 96u + lane]) * args.scale;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    if (q1_live) {
        b0 = h3_bf16_to_f32(query[q_base1 + lane]) * args.scale;
        b1 = h3_bf16_to_f32(query[q_base1 + 32u + lane]) * args.scale;
        b2 = h3_bf16_to_f32(query[q_base1 + 64u + lane]) * args.scale;
        b3 = h3_bf16_to_f32(query[q_base1 + 96u + lane]) * args.scale;
    }
    float acc_a0 = 0.0f, acc_a1 = 0.0f, acc_a2 = 0.0f, acc_a3 = 0.0f;
    float acc_b0 = 0.0f, acc_b1 = 0.0f, acc_b2 = 0.0f, acc_b3 = 0.0f;
    float max_a = -1e30f, max_b = -1e30f, sum_a = 0.0f, sum_b = 0.0f;
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
        float score_a = h3_warp_reduce_sum(a0 * k0 + a1 * k1v + a2 * k2 + a3 * k3);
        float new_a = fmaxf(max_a, score_a);
        float alpha_a = h3_fast_exp(max_a - new_a);
        float pa = h3_fast_exp(score_a - new_a);
        acc_a0 = fmaf(pa, v0, acc_a0 * alpha_a);
        acc_a1 = fmaf(pa, v1, acc_a1 * alpha_a);
        acc_a2 = fmaf(pa, v2, acc_a2 * alpha_a);
        acc_a3 = fmaf(pa, v3, acc_a3 * alpha_a);
        sum_a = fmaf(sum_a, alpha_a, pa);
        max_a = new_a;
        if (q1_live) {
            float score_b =
                h3_warp_reduce_sum(b0 * k0 + b1 * k1v + b2 * k2 + b3 * k3);
            float new_b = fmaxf(max_b, score_b);
            float alpha_b = h3_fast_exp(max_b - new_b);
            float pb = h3_fast_exp(score_b - new_b);
            acc_b0 = fmaf(pb, v0, acc_b0 * alpha_b);
            acc_b1 = fmaf(pb, v1, acc_b1 * alpha_b);
            acc_b2 = fmaf(pb, v2, acc_b2 * alpha_b);
            acc_b3 = fmaf(pb, v3, acc_b3 * alpha_b);
            sum_b = fmaf(sum_b, alpha_b, pb);
            max_b = new_b;
        }
        k_base += stride;
    }
    float inv_a = sum_a > 0.0f ? 1.0f / sum_a : 0.0f;
    output[h3_sdpa_output_index(args, q_pos, head, lane)] =
        h3_f32_to_bf16(acc_a0 * inv_a);
    output[h3_sdpa_output_index(args, q_pos, head, 32u + lane)] =
        h3_f32_to_bf16(acc_a1 * inv_a);
    output[h3_sdpa_output_index(args, q_pos, head, 64u + lane)] =
        h3_f32_to_bf16(acc_a2 * inv_a);
    output[h3_sdpa_output_index(args, q_pos, head, 96u + lane)] =
        h3_f32_to_bf16(acc_a3 * inv_a);
    if (q1_live) {
        float inv_b = sum_b > 0.0f ? 1.0f / sum_b : 0.0f;
        output[h3_sdpa_output_index(args, q_pos + 1u, head, lane)] =
            h3_f32_to_bf16(acc_b0 * inv_b);
        output[h3_sdpa_output_index(args, q_pos + 1u, head, 32u + lane)] =
            h3_f32_to_bf16(acc_b1 * inv_b);
        output[h3_sdpa_output_index(args, q_pos + 1u, head, 64u + lane)] =
            h3_f32_to_bf16(acc_b2 * inv_b);
        output[h3_sdpa_output_index(args, q_pos + 1u, head, 96u + lane)] =
            h3_f32_to_bf16(acc_b3 * inv_b);
    }
}

/* Three queries share each K/V load. Opt in: H3_SDPA_D128_Q3=1. */
__global__ void __launch_bounds__(32)
h3_sdpa_bf16_wave_d128_q3_kernel(const uint16_t *query, const uint16_t *key,
                                 const uint16_t *value, uint16_t *output,
                                 h3_sdpa_args args) {
    uint32_t q_pos = blockIdx.x * 3u;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    int q1_live = (q_pos + 1u) < args.sequence;
    int q2_live = (q_pos + 2u) < args.sequence;
    uint32_t q_base0 = (q_pos * args.heads + head) * 128u;
    uint32_t q_stride = args.heads * 128u;
    float a0 = h3_bf16_to_f32(query[q_base0 + lane]) * args.scale;
    float a1 = h3_bf16_to_f32(query[q_base0 + 32u + lane]) * args.scale;
    float a2 = h3_bf16_to_f32(query[q_base0 + 64u + lane]) * args.scale;
    float a3 = h3_bf16_to_f32(query[q_base0 + 96u + lane]) * args.scale;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    if (q1_live) {
        uint32_t q_base1 = q_base0 + q_stride;
        b0 = h3_bf16_to_f32(query[q_base1 + lane]) * args.scale;
        b1 = h3_bf16_to_f32(query[q_base1 + 32u + lane]) * args.scale;
        b2 = h3_bf16_to_f32(query[q_base1 + 64u + lane]) * args.scale;
        b3 = h3_bf16_to_f32(query[q_base1 + 96u + lane]) * args.scale;
    }
    if (q2_live) {
        uint32_t q_base2 = q_base0 + 2u * q_stride;
        c0 = h3_bf16_to_f32(query[q_base2 + lane]) * args.scale;
        c1 = h3_bf16_to_f32(query[q_base2 + 32u + lane]) * args.scale;
        c2 = h3_bf16_to_f32(query[q_base2 + 64u + lane]) * args.scale;
        c3 = h3_bf16_to_f32(query[q_base2 + 96u + lane]) * args.scale;
    }
    float acc_a0 = 0.0f, acc_a1 = 0.0f, acc_a2 = 0.0f, acc_a3 = 0.0f;
    float acc_b0 = 0.0f, acc_b1 = 0.0f, acc_b2 = 0.0f, acc_b3 = 0.0f;
    float acc_c0 = 0.0f, acc_c1 = 0.0f, acc_c2 = 0.0f, acc_c3 = 0.0f;
    float max_a = -1e30f, max_b = -1e30f, max_c = -1e30f;
    float sum_a = 0.0f, sum_b = 0.0f, sum_c = 0.0f;
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
        float score_a = h3_warp_reduce_sum(a0 * k0 + a1 * k1v + a2 * k2 + a3 * k3);
        float new_a = fmaxf(max_a, score_a);
        float alpha_a = h3_fast_exp(max_a - new_a);
        float pa = h3_fast_exp(score_a - new_a);
        acc_a0 = fmaf(pa, v0, acc_a0 * alpha_a);
        acc_a1 = fmaf(pa, v1, acc_a1 * alpha_a);
        acc_a2 = fmaf(pa, v2, acc_a2 * alpha_a);
        acc_a3 = fmaf(pa, v3, acc_a3 * alpha_a);
        sum_a = fmaf(sum_a, alpha_a, pa);
        max_a = new_a;
        if (q1_live) {
            float score_b =
                h3_warp_reduce_sum(b0 * k0 + b1 * k1v + b2 * k2 + b3 * k3);
            float new_b = fmaxf(max_b, score_b);
            float alpha_b = h3_fast_exp(max_b - new_b);
            float pb = h3_fast_exp(score_b - new_b);
            acc_b0 = fmaf(pb, v0, acc_b0 * alpha_b);
            acc_b1 = fmaf(pb, v1, acc_b1 * alpha_b);
            acc_b2 = fmaf(pb, v2, acc_b2 * alpha_b);
            acc_b3 = fmaf(pb, v3, acc_b3 * alpha_b);
            sum_b = fmaf(sum_b, alpha_b, pb);
            max_b = new_b;
        }
        if (q2_live) {
            float score_c =
                h3_warp_reduce_sum(c0 * k0 + c1 * k1v + c2 * k2 + c3 * k3);
            float new_c = fmaxf(max_c, score_c);
            float alpha_c = h3_fast_exp(max_c - new_c);
            float pc = h3_fast_exp(score_c - new_c);
            acc_c0 = fmaf(pc, v0, acc_c0 * alpha_c);
            acc_c1 = fmaf(pc, v1, acc_c1 * alpha_c);
            acc_c2 = fmaf(pc, v2, acc_c2 * alpha_c);
            acc_c3 = fmaf(pc, v3, acc_c3 * alpha_c);
            sum_c = fmaf(sum_c, alpha_c, pc);
            max_c = new_c;
        }
        k_base += stride;
    }
    float inv_a = sum_a > 0.0f ? 1.0f / sum_a : 0.0f;
    output[h3_sdpa_output_index(args, q_pos, head, lane)] =
        h3_f32_to_bf16(acc_a0 * inv_a);
    output[h3_sdpa_output_index(args, q_pos, head, 32u + lane)] =
        h3_f32_to_bf16(acc_a1 * inv_a);
    output[h3_sdpa_output_index(args, q_pos, head, 64u + lane)] =
        h3_f32_to_bf16(acc_a2 * inv_a);
    output[h3_sdpa_output_index(args, q_pos, head, 96u + lane)] =
        h3_f32_to_bf16(acc_a3 * inv_a);
    if (q1_live) {
        float inv_b = sum_b > 0.0f ? 1.0f / sum_b : 0.0f;
        output[h3_sdpa_output_index(args, q_pos + 1u, head, lane)] =
            h3_f32_to_bf16(acc_b0 * inv_b);
        output[h3_sdpa_output_index(args, q_pos + 1u, head, 32u + lane)] =
            h3_f32_to_bf16(acc_b1 * inv_b);
        output[h3_sdpa_output_index(args, q_pos + 1u, head, 64u + lane)] =
            h3_f32_to_bf16(acc_b2 * inv_b);
        output[h3_sdpa_output_index(args, q_pos + 1u, head, 96u + lane)] =
            h3_f32_to_bf16(acc_b3 * inv_b);
    }
    if (q2_live) {
        float inv_c = sum_c > 0.0f ? 1.0f / sum_c : 0.0f;
        output[h3_sdpa_output_index(args, q_pos + 2u, head, lane)] =
            h3_f32_to_bf16(acc_c0 * inv_c);
        output[h3_sdpa_output_index(args, q_pos + 2u, head, 32u + lane)] =
            h3_f32_to_bf16(acc_c1 * inv_c);
        output[h3_sdpa_output_index(args, q_pos + 2u, head, 64u + lane)] =
            h3_f32_to_bf16(acc_c2 * inv_c);
        output[h3_sdpa_output_index(args, q_pos + 2u, head, 96u + lane)] =
            h3_f32_to_bf16(acc_c3 * inv_c);
    }
}

/* Four queries share each K/V load. Opt in: H3_SDPA_D128_Q4=1. */
__global__ void __launch_bounds__(32)
h3_sdpa_bf16_wave_d128_q4_kernel(const uint16_t *query, const uint16_t *key,
                                 const uint16_t *value, uint16_t *output,
                                 h3_sdpa_args args) {
    uint32_t q_pos = blockIdx.x * 4u;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    int q1_live = (q_pos + 1u) < args.sequence;
    int q2_live = (q_pos + 2u) < args.sequence;
    int q3_live = (q_pos + 3u) < args.sequence;
    uint32_t q_base0 = (q_pos * args.heads + head) * 128u;
    uint32_t q_stride = args.heads * 128u;
    float a0 = h3_bf16_to_f32(query[q_base0 + lane]) * args.scale;
    float a1 = h3_bf16_to_f32(query[q_base0 + 32u + lane]) * args.scale;
    float a2 = h3_bf16_to_f32(query[q_base0 + 64u + lane]) * args.scale;
    float a3 = h3_bf16_to_f32(query[q_base0 + 96u + lane]) * args.scale;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    if (q1_live) {
        uint32_t q_base1 = q_base0 + q_stride;
        b0 = h3_bf16_to_f32(query[q_base1 + lane]) * args.scale;
        b1 = h3_bf16_to_f32(query[q_base1 + 32u + lane]) * args.scale;
        b2 = h3_bf16_to_f32(query[q_base1 + 64u + lane]) * args.scale;
        b3 = h3_bf16_to_f32(query[q_base1 + 96u + lane]) * args.scale;
    }
    if (q2_live) {
        uint32_t q_base2 = q_base0 + 2u * q_stride;
        c0 = h3_bf16_to_f32(query[q_base2 + lane]) * args.scale;
        c1 = h3_bf16_to_f32(query[q_base2 + 32u + lane]) * args.scale;
        c2 = h3_bf16_to_f32(query[q_base2 + 64u + lane]) * args.scale;
        c3 = h3_bf16_to_f32(query[q_base2 + 96u + lane]) * args.scale;
    }
    if (q3_live) {
        uint32_t q_base3 = q_base0 + 3u * q_stride;
        d0 = h3_bf16_to_f32(query[q_base3 + lane]) * args.scale;
        d1 = h3_bf16_to_f32(query[q_base3 + 32u + lane]) * args.scale;
        d2 = h3_bf16_to_f32(query[q_base3 + 64u + lane]) * args.scale;
        d3 = h3_bf16_to_f32(query[q_base3 + 96u + lane]) * args.scale;
    }
    float acc_a0 = 0.0f, acc_a1 = 0.0f, acc_a2 = 0.0f, acc_a3 = 0.0f;
    float acc_b0 = 0.0f, acc_b1 = 0.0f, acc_b2 = 0.0f, acc_b3 = 0.0f;
    float acc_c0 = 0.0f, acc_c1 = 0.0f, acc_c2 = 0.0f, acc_c3 = 0.0f;
    float acc_d0 = 0.0f, acc_d1 = 0.0f, acc_d2 = 0.0f, acc_d3 = 0.0f;
    float max_a = -1e30f, max_b = -1e30f, max_c = -1e30f, max_d = -1e30f;
    float sum_a = 0.0f, sum_b = 0.0f, sum_c = 0.0f, sum_d = 0.0f;
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
        float score_a = h3_warp_reduce_sum(a0 * k0 + a1 * k1v + a2 * k2 + a3 * k3);
        float new_a = fmaxf(max_a, score_a);
        float alpha_a = h3_fast_exp(max_a - new_a);
        float pa = h3_fast_exp(score_a - new_a);
        acc_a0 = fmaf(pa, v0, acc_a0 * alpha_a);
        acc_a1 = fmaf(pa, v1, acc_a1 * alpha_a);
        acc_a2 = fmaf(pa, v2, acc_a2 * alpha_a);
        acc_a3 = fmaf(pa, v3, acc_a3 * alpha_a);
        sum_a = fmaf(sum_a, alpha_a, pa);
        max_a = new_a;
        if (q1_live) {
            float score_b =
                h3_warp_reduce_sum(b0 * k0 + b1 * k1v + b2 * k2 + b3 * k3);
            float new_b = fmaxf(max_b, score_b);
            float alpha_b = h3_fast_exp(max_b - new_b);
            float pb = h3_fast_exp(score_b - new_b);
            acc_b0 = fmaf(pb, v0, acc_b0 * alpha_b);
            acc_b1 = fmaf(pb, v1, acc_b1 * alpha_b);
            acc_b2 = fmaf(pb, v2, acc_b2 * alpha_b);
            acc_b3 = fmaf(pb, v3, acc_b3 * alpha_b);
            sum_b = fmaf(sum_b, alpha_b, pb);
            max_b = new_b;
        }
        if (q2_live) {
            float score_c =
                h3_warp_reduce_sum(c0 * k0 + c1 * k1v + c2 * k2 + c3 * k3);
            float new_c = fmaxf(max_c, score_c);
            float alpha_c = h3_fast_exp(max_c - new_c);
            float pc = h3_fast_exp(score_c - new_c);
            acc_c0 = fmaf(pc, v0, acc_c0 * alpha_c);
            acc_c1 = fmaf(pc, v1, acc_c1 * alpha_c);
            acc_c2 = fmaf(pc, v2, acc_c2 * alpha_c);
            acc_c3 = fmaf(pc, v3, acc_c3 * alpha_c);
            sum_c = fmaf(sum_c, alpha_c, pc);
            max_c = new_c;
        }
        if (q3_live) {
            float score_d =
                h3_warp_reduce_sum(d0 * k0 + d1 * k1v + d2 * k2 + d3 * k3);
            float new_d = fmaxf(max_d, score_d);
            float alpha_d = h3_fast_exp(max_d - new_d);
            float pd = h3_fast_exp(score_d - new_d);
            acc_d0 = fmaf(pd, v0, acc_d0 * alpha_d);
            acc_d1 = fmaf(pd, v1, acc_d1 * alpha_d);
            acc_d2 = fmaf(pd, v2, acc_d2 * alpha_d);
            acc_d3 = fmaf(pd, v3, acc_d3 * alpha_d);
            sum_d = fmaf(sum_d, alpha_d, pd);
            max_d = new_d;
        }
        k_base += stride;
    }
    float inv_a = sum_a > 0.0f ? 1.0f / sum_a : 0.0f;
    output[h3_sdpa_output_index(args, q_pos, head, lane)] =
        h3_f32_to_bf16(acc_a0 * inv_a);
    output[h3_sdpa_output_index(args, q_pos, head, 32u + lane)] =
        h3_f32_to_bf16(acc_a1 * inv_a);
    output[h3_sdpa_output_index(args, q_pos, head, 64u + lane)] =
        h3_f32_to_bf16(acc_a2 * inv_a);
    output[h3_sdpa_output_index(args, q_pos, head, 96u + lane)] =
        h3_f32_to_bf16(acc_a3 * inv_a);
    if (q1_live) {
        float inv_b = sum_b > 0.0f ? 1.0f / sum_b : 0.0f;
        output[h3_sdpa_output_index(args, q_pos + 1u, head, lane)] =
            h3_f32_to_bf16(acc_b0 * inv_b);
        output[h3_sdpa_output_index(args, q_pos + 1u, head, 32u + lane)] =
            h3_f32_to_bf16(acc_b1 * inv_b);
        output[h3_sdpa_output_index(args, q_pos + 1u, head, 64u + lane)] =
            h3_f32_to_bf16(acc_b2 * inv_b);
        output[h3_sdpa_output_index(args, q_pos + 1u, head, 96u + lane)] =
            h3_f32_to_bf16(acc_b3 * inv_b);
    }
    if (q2_live) {
        float inv_c = sum_c > 0.0f ? 1.0f / sum_c : 0.0f;
        output[h3_sdpa_output_index(args, q_pos + 2u, head, lane)] =
            h3_f32_to_bf16(acc_c0 * inv_c);
        output[h3_sdpa_output_index(args, q_pos + 2u, head, 32u + lane)] =
            h3_f32_to_bf16(acc_c1 * inv_c);
        output[h3_sdpa_output_index(args, q_pos + 2u, head, 64u + lane)] =
            h3_f32_to_bf16(acc_c2 * inv_c);
        output[h3_sdpa_output_index(args, q_pos + 2u, head, 96u + lane)] =
            h3_f32_to_bf16(acc_c3 * inv_c);
    }
    if (q3_live) {
        float inv_d = sum_d > 0.0f ? 1.0f / sum_d : 0.0f;
        output[h3_sdpa_output_index(args, q_pos + 3u, head, lane)] =
            h3_f32_to_bf16(acc_d0 * inv_d);
        output[h3_sdpa_output_index(args, q_pos + 3u, head, 32u + lane)] =
            h3_f32_to_bf16(acc_d1 * inv_d);
        output[h3_sdpa_output_index(args, q_pos + 3u, head, 64u + lane)] =
            h3_f32_to_bf16(acc_d2 * inv_d);
        output[h3_sdpa_output_index(args, q_pos + 3u, head, 96u + lane)] =
            h3_f32_to_bf16(acc_d3 * inv_d);
    }
}

/* Five queries share each K/V load. Opt in: H3_SDPA_D128_Q5=1. */
__global__ void __launch_bounds__(32)
h3_sdpa_bf16_wave_d128_q5_kernel(const uint16_t *query, const uint16_t *key,
                                 const uint16_t *value, uint16_t *output,
                                 h3_sdpa_args args) {
    uint32_t q_pos = blockIdx.x * 5u;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    int q1_live = (q_pos + 1u) < args.sequence;
    int q2_live = (q_pos + 2u) < args.sequence;
    int q3_live = (q_pos + 3u) < args.sequence;
    int q4_live = (q_pos + 4u) < args.sequence;
    uint32_t q_base0 = (q_pos * args.heads + head) * 128u;
    uint32_t q_stride = args.heads * 128u;
    float a0 = h3_bf16_to_f32(query[q_base0 + lane]) * args.scale;
    float a1 = h3_bf16_to_f32(query[q_base0 + 32u + lane]) * args.scale;
    float a2 = h3_bf16_to_f32(query[q_base0 + 64u + lane]) * args.scale;
    float a3 = h3_bf16_to_f32(query[q_base0 + 96u + lane]) * args.scale;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f, e3 = 0.0f;
    if (q1_live) {
        uint32_t qb = q_base0 + q_stride;
        b0 = h3_bf16_to_f32(query[qb + lane]) * args.scale;
        b1 = h3_bf16_to_f32(query[qb + 32u + lane]) * args.scale;
        b2 = h3_bf16_to_f32(query[qb + 64u + lane]) * args.scale;
        b3 = h3_bf16_to_f32(query[qb + 96u + lane]) * args.scale;
    }
    if (q2_live) {
        uint32_t qb = q_base0 + 2u * q_stride;
        c0 = h3_bf16_to_f32(query[qb + lane]) * args.scale;
        c1 = h3_bf16_to_f32(query[qb + 32u + lane]) * args.scale;
        c2 = h3_bf16_to_f32(query[qb + 64u + lane]) * args.scale;
        c3 = h3_bf16_to_f32(query[qb + 96u + lane]) * args.scale;
    }
    if (q3_live) {
        uint32_t qb = q_base0 + 3u * q_stride;
        d0 = h3_bf16_to_f32(query[qb + lane]) * args.scale;
        d1 = h3_bf16_to_f32(query[qb + 32u + lane]) * args.scale;
        d2 = h3_bf16_to_f32(query[qb + 64u + lane]) * args.scale;
        d3 = h3_bf16_to_f32(query[qb + 96u + lane]) * args.scale;
    }
    if (q4_live) {
        uint32_t qb = q_base0 + 4u * q_stride;
        e0 = h3_bf16_to_f32(query[qb + lane]) * args.scale;
        e1 = h3_bf16_to_f32(query[qb + 32u + lane]) * args.scale;
        e2 = h3_bf16_to_f32(query[qb + 64u + lane]) * args.scale;
        e3 = h3_bf16_to_f32(query[qb + 96u + lane]) * args.scale;
    }
    float acc_a0 = 0.0f, acc_a1 = 0.0f, acc_a2 = 0.0f, acc_a3 = 0.0f;
    float acc_b0 = 0.0f, acc_b1 = 0.0f, acc_b2 = 0.0f, acc_b3 = 0.0f;
    float acc_c0 = 0.0f, acc_c1 = 0.0f, acc_c2 = 0.0f, acc_c3 = 0.0f;
    float acc_d0 = 0.0f, acc_d1 = 0.0f, acc_d2 = 0.0f, acc_d3 = 0.0f;
    float acc_e0 = 0.0f, acc_e1 = 0.0f, acc_e2 = 0.0f, acc_e3 = 0.0f;
    float max_a = -1e30f, max_b = -1e30f, max_c = -1e30f, max_d = -1e30f,
          max_e = -1e30f;
    float sum_a = 0.0f, sum_b = 0.0f, sum_c = 0.0f, sum_d = 0.0f, sum_e = 0.0f;
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
        float score_a = h3_warp_reduce_sum(a0 * k0 + a1 * k1v + a2 * k2 + a3 * k3);
        float new_a = fmaxf(max_a, score_a);
        float alpha_a = h3_fast_exp(max_a - new_a);
        float pa = h3_fast_exp(score_a - new_a);
        acc_a0 = fmaf(pa, v0, acc_a0 * alpha_a);
        acc_a1 = fmaf(pa, v1, acc_a1 * alpha_a);
        acc_a2 = fmaf(pa, v2, acc_a2 * alpha_a);
        acc_a3 = fmaf(pa, v3, acc_a3 * alpha_a);
        sum_a = fmaf(sum_a, alpha_a, pa);
        max_a = new_a;
        if (q1_live) {
            float score_b =
                h3_warp_reduce_sum(b0 * k0 + b1 * k1v + b2 * k2 + b3 * k3);
            float new_b = fmaxf(max_b, score_b);
            float alpha_b = h3_fast_exp(max_b - new_b);
            float pb = h3_fast_exp(score_b - new_b);
            acc_b0 = fmaf(pb, v0, acc_b0 * alpha_b);
            acc_b1 = fmaf(pb, v1, acc_b1 * alpha_b);
            acc_b2 = fmaf(pb, v2, acc_b2 * alpha_b);
            acc_b3 = fmaf(pb, v3, acc_b3 * alpha_b);
            sum_b = fmaf(sum_b, alpha_b, pb);
            max_b = new_b;
        }
        if (q2_live) {
            float score_c =
                h3_warp_reduce_sum(c0 * k0 + c1 * k1v + c2 * k2 + c3 * k3);
            float new_c = fmaxf(max_c, score_c);
            float alpha_c = h3_fast_exp(max_c - new_c);
            float pc = h3_fast_exp(score_c - new_c);
            acc_c0 = fmaf(pc, v0, acc_c0 * alpha_c);
            acc_c1 = fmaf(pc, v1, acc_c1 * alpha_c);
            acc_c2 = fmaf(pc, v2, acc_c2 * alpha_c);
            acc_c3 = fmaf(pc, v3, acc_c3 * alpha_c);
            sum_c = fmaf(sum_c, alpha_c, pc);
            max_c = new_c;
        }
        if (q3_live) {
            float score_d =
                h3_warp_reduce_sum(d0 * k0 + d1 * k1v + d2 * k2 + d3 * k3);
            float new_d = fmaxf(max_d, score_d);
            float alpha_d = h3_fast_exp(max_d - new_d);
            float pd = h3_fast_exp(score_d - new_d);
            acc_d0 = fmaf(pd, v0, acc_d0 * alpha_d);
            acc_d1 = fmaf(pd, v1, acc_d1 * alpha_d);
            acc_d2 = fmaf(pd, v2, acc_d2 * alpha_d);
            acc_d3 = fmaf(pd, v3, acc_d3 * alpha_d);
            sum_d = fmaf(sum_d, alpha_d, pd);
            max_d = new_d;
        }
        if (q4_live) {
            float score_e =
                h3_warp_reduce_sum(e0 * k0 + e1 * k1v + e2 * k2 + e3 * k3);
            float new_e = fmaxf(max_e, score_e);
            float alpha_e = h3_fast_exp(max_e - new_e);
            float pe = h3_fast_exp(score_e - new_e);
            acc_e0 = fmaf(pe, v0, acc_e0 * alpha_e);
            acc_e1 = fmaf(pe, v1, acc_e1 * alpha_e);
            acc_e2 = fmaf(pe, v2, acc_e2 * alpha_e);
            acc_e3 = fmaf(pe, v3, acc_e3 * alpha_e);
            sum_e = fmaf(sum_e, alpha_e, pe);
            max_e = new_e;
        }
        k_base += stride;
    }
    float inv_a = sum_a > 0.0f ? 1.0f / sum_a : 0.0f;
    output[h3_sdpa_output_index(args, q_pos, head, lane)] =
        h3_f32_to_bf16(acc_a0 * inv_a);
    output[h3_sdpa_output_index(args, q_pos, head, 32u + lane)] =
        h3_f32_to_bf16(acc_a1 * inv_a);
    output[h3_sdpa_output_index(args, q_pos, head, 64u + lane)] =
        h3_f32_to_bf16(acc_a2 * inv_a);
    output[h3_sdpa_output_index(args, q_pos, head, 96u + lane)] =
        h3_f32_to_bf16(acc_a3 * inv_a);
    if (q1_live) {
        float inv_b = sum_b > 0.0f ? 1.0f / sum_b : 0.0f;
        output[h3_sdpa_output_index(args, q_pos + 1u, head, lane)] =
            h3_f32_to_bf16(acc_b0 * inv_b);
        output[h3_sdpa_output_index(args, q_pos + 1u, head, 32u + lane)] =
            h3_f32_to_bf16(acc_b1 * inv_b);
        output[h3_sdpa_output_index(args, q_pos + 1u, head, 64u + lane)] =
            h3_f32_to_bf16(acc_b2 * inv_b);
        output[h3_sdpa_output_index(args, q_pos + 1u, head, 96u + lane)] =
            h3_f32_to_bf16(acc_b3 * inv_b);
    }
    if (q2_live) {
        float inv_c = sum_c > 0.0f ? 1.0f / sum_c : 0.0f;
        output[h3_sdpa_output_index(args, q_pos + 2u, head, lane)] =
            h3_f32_to_bf16(acc_c0 * inv_c);
        output[h3_sdpa_output_index(args, q_pos + 2u, head, 32u + lane)] =
            h3_f32_to_bf16(acc_c1 * inv_c);
        output[h3_sdpa_output_index(args, q_pos + 2u, head, 64u + lane)] =
            h3_f32_to_bf16(acc_c2 * inv_c);
        output[h3_sdpa_output_index(args, q_pos + 2u, head, 96u + lane)] =
            h3_f32_to_bf16(acc_c3 * inv_c);
    }
    if (q3_live) {
        float inv_d = sum_d > 0.0f ? 1.0f / sum_d : 0.0f;
        output[h3_sdpa_output_index(args, q_pos + 3u, head, lane)] =
            h3_f32_to_bf16(acc_d0 * inv_d);
        output[h3_sdpa_output_index(args, q_pos + 3u, head, 32u + lane)] =
            h3_f32_to_bf16(acc_d1 * inv_d);
        output[h3_sdpa_output_index(args, q_pos + 3u, head, 64u + lane)] =
            h3_f32_to_bf16(acc_d2 * inv_d);
        output[h3_sdpa_output_index(args, q_pos + 3u, head, 96u + lane)] =
            h3_f32_to_bf16(acc_d3 * inv_d);
    }
    if (q4_live) {
        float inv_e = sum_e > 0.0f ? 1.0f / sum_e : 0.0f;
        output[h3_sdpa_output_index(args, q_pos + 4u, head, lane)] =
            h3_f32_to_bf16(acc_e0 * inv_e);
        output[h3_sdpa_output_index(args, q_pos + 4u, head, 32u + lane)] =
            h3_f32_to_bf16(acc_e1 * inv_e);
        output[h3_sdpa_output_index(args, q_pos + 4u, head, 64u + lane)] =
            h3_f32_to_bf16(acc_e2 * inv_e);
        output[h3_sdpa_output_index(args, q_pos + 4u, head, 96u + lane)] =
            h3_f32_to_bf16(acc_e3 * inv_e);
    }
}


/* Six queries share each K/V load. Default for d128. Opt out:
 * H3_SDPA_D128_Q5=1, Q4, Q3, Q2, or Q1. */
__global__ void __launch_bounds__(32)
h3_sdpa_bf16_wave_d128_q6_kernel(const uint16_t *query, const uint16_t *key,
                                 const uint16_t *value, uint16_t *output,
                                 h3_sdpa_args args) {
    uint32_t q_pos = blockIdx.x * 6u;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    int live[6];
    for (int i = 0; i < 6; i++)
        live[i] = (q_pos + (uint32_t)i) < args.sequence;
    uint32_t q_base0 = (q_pos * args.heads + head) * 128u;
    uint32_t q_stride = args.heads * 128u;
    float q[6][4];
#pragma unroll
    for (int qi = 0; qi < 6; qi++) {
#pragma unroll
        for (int d = 0; d < 4; d++) q[qi][d] = 0.0f;
        if (!live[qi]) continue;
        uint32_t qb = q_base0 + (uint32_t)qi * q_stride;
#pragma unroll
        for (int d = 0; d < 4; d++)
            q[qi][d] = h3_bf16_to_f32(query[qb + 32u * (uint32_t)d + lane]) *
                       args.scale;
    }
    float acc[6][4];
    float mx[6], sm[6];
#pragma unroll
    for (int qi = 0; qi < 6; qi++) {
        mx[qi] = -1e30f;
        sm[qi] = 0.0f;
#pragma unroll
        for (int d = 0; d < 4; d++) acc[qi][d] = 0.0f;
    }
    uint32_t stride = args.kv_head_major ? 128u : args.heads * 128u;
    uint32_t k_base = args.kv_head_major ? head * args.sequence * 128u
                                         : head * 128u;
    for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
        float kv[4], vv[4];
#pragma unroll
        for (int d = 0; d < 4; d++) {
            kv[d] = h3_bf16_to_f32(key[k_base + 32u * (uint32_t)d + lane]);
            vv[d] = h3_bf16_to_f32(value[k_base + 32u * (uint32_t)d + lane]);
        }
#pragma unroll
        for (int qi = 0; qi < 6; qi++) {
            if (qi && !live[qi]) continue;
            float score = h3_warp_reduce_sum(q[qi][0] * kv[0] + q[qi][1] * kv[1] +
                                             q[qi][2] * kv[2] + q[qi][3] * kv[3]);
            float nw = fmaxf(mx[qi], score);
            float alpha = h3_fast_exp(mx[qi] - nw);
            float p = h3_fast_exp(score - nw);
#pragma unroll
            for (int d = 0; d < 4; d++)
                acc[qi][d] = fmaf(p, vv[d], acc[qi][d] * alpha);
            sm[qi] = fmaf(sm[qi], alpha, p);
            mx[qi] = nw;
        }
        k_base += stride;
    }
#pragma unroll
    for (int qi = 0; qi < 6; qi++) {
        if (!live[qi]) continue;
        float inv = sm[qi] > 0.0f ? 1.0f / sm[qi] : 0.0f;
#pragma unroll
        for (int d = 0; d < 4; d++)
            output[h3_sdpa_output_index(args, q_pos + (uint32_t)qi, head,
                                        32u * (uint32_t)d + lane)] =
                h3_f32_to_bf16(acc[qi][d] * inv);
    }
}

/* Q6 with consecutive-4 dims per lane so K/V/Q/out are uint2 vector loads.
 * Default. Opt out: H3_SDPA_Q6_SCALAR=1 (strided 32/64/96 layout).
 * Tail=1 keeps per-query live masks; Tail=0 is for sequence % 6 == 0. */
template <int Tail>
__global__ void __launch_bounds__(32)
h3_sdpa_bf16_wave_d128_q6_vec_kernel(const uint16_t *query, const uint16_t *key,
                                     const uint16_t *value, uint16_t *output,
                                     h3_sdpa_args args) {
    uint32_t q_pos = blockIdx.x * 6u;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    uint32_t d0 = lane * 4u;
    int live[6];
#pragma unroll
    for (int i = 0; i < 6; i++)
        live[i] = !Tail || (q_pos + (uint32_t)i) < args.sequence;
    uint32_t q_base0 = (q_pos * args.heads + head) * 128u;
    uint32_t q_stride = args.heads * 128u;
    float q[6][4];
#pragma unroll
    for (int qi = 0; qi < 6; qi++) {
#pragma unroll
        for (int d = 0; d < 4; d++) q[qi][d] = 0.0f;
        if (Tail && !live[qi]) continue;
        h3_load_bf16x4(query + q_base0 + (uint32_t)qi * q_stride + d0, q[qi]);
#pragma unroll
        for (int d = 0; d < 4; d++) q[qi][d] *= args.scale;
    }
    float acc[6][4];
    float mx[6], sm[6];
#pragma unroll
    for (int qi = 0; qi < 6; qi++) {
        mx[qi] = -1e30f;
        sm[qi] = 0.0f;
#pragma unroll
        for (int d = 0; d < 4; d++) acc[qi][d] = 0.0f;
    }
    uint32_t stride = args.kv_head_major ? 128u : args.heads * 128u;
    uint32_t k_base = args.kv_head_major ? head * args.sequence * 128u
                                         : head * 128u;
    for (uint32_t k_pos = 0; k_pos < args.sequence; k_pos++) {
        float kv[4], vv[4];
        h3_load_bf16x4(key + k_base + d0, kv);
        h3_load_bf16x4(value + k_base + d0, vv);
#pragma unroll
        for (int qi = 0; qi < 6; qi++) {
            if (Tail && qi && !live[qi]) continue;
            float score = h3_warp_reduce_sum(q[qi][0] * kv[0] + q[qi][1] * kv[1] +
                                             q[qi][2] * kv[2] + q[qi][3] * kv[3]);
            float nw = fmaxf(mx[qi], score);
            float alpha = h3_fast_exp(mx[qi] - nw);
            float p = h3_fast_exp(score - nw);
#pragma unroll
            for (int d = 0; d < 4; d++)
                acc[qi][d] = fmaf(p, vv[d], acc[qi][d] * alpha);
            sm[qi] = fmaf(sm[qi], alpha, p);
            mx[qi] = nw;
        }
        k_base += stride;
    }
#pragma unroll
    for (int qi = 0; qi < 6; qi++) {
        if (Tail && !live[qi]) continue;
        float inv = sm[qi] > 0.0f ? 1.0f / sm[qi] : 0.0f;
        uint32_t out_i = h3_sdpa_output_index(args, q_pos + (uint32_t)qi, head,
                                              d0);
        h3_store_bf16x4(output + out_i, acc[qi][0] * inv, acc[qi][1] * inv,
                        acc[qi][2] * inv, acc[qi][3] * inv);
    }
}

/* Tiled WMMA flash attention for head_dim=128 (wave32 RDNA3/3.5).
 *
 * Both Q@K^T and P@V run on the matrix units via v_wmma_f32_16x16x16_bf16,
 * which removes the per-(query,key) cross-lane dot-product reduction that
 * limits the wave Q6 kernel, and a 32-key K/V tile shared by all waves in the
 * block cuts global traffic by the query-block size instead of by 6.
 *
 * Fragment layouts probed on gfx1151 (rocWMMA 2.2, wave32):
 *   matrix_a row_major  lane l -> row l&15,  element i -> k 8*(l>>4)+i
 *   matrix_b col_major  lane l -> col l&15,  element i -> k 8*(l>>4)+i
 *   accumulator f32     lane l -> col l&15,  element i -> row 2*i+(l>>4)
 * The accumulator row mapping is what lets the online-softmax rescale stay in
 * registers: element i of every O fragment belongs to row 2*i+(l>>4), so one
 * alpha per element index covers all eight 16-dim output slices.
 *
 * Rows are reduced across the 16 lanes sharing a row; XOR masks 1/2/4/8 never
 * flip lane bit 4, so the two row groups stay independent. */
#define H3_SDPA_WMMA_QROWS 16u
/* 8 halves of padding so the 16-row fragment loads skew across LDS banks. */
#define H3_SDPA_WMMA_LD 136u

/* WAVES sets the query block (WAVES*16 rows), which is the K/V reuse factor;
 * BK sets the key tile staged in LDS per iteration. */
template <uint32_t WAVES, uint32_t BK>
__global__ void __launch_bounds__(WAVES * 32u)
h3_sdpa_bf16_wmma_d128_kernel(const uint16_t *query, const uint16_t *key,
                              const uint16_t *value, uint16_t *output,
                              h3_sdpa_args args) {
    constexpr uint32_t D = 128u;
    constexpr uint32_t LD = H3_SDPA_WMMA_LD;
    constexpr uint32_t QROWS = H3_SDPA_WMMA_QROWS;
    constexpr uint32_t THREADS = WAVES * 32u;
    constexpr uint32_t NT = BK / 16u;
    typedef rocwmma::bfloat16_t bf16;

    __shared__ uint16_t sk[BK * LD];
    __shared__ uint16_t sv[BK * LD];
    __shared__ uint16_t sp[WAVES][QROWS * BK];

    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t col = lane & 15u;
    const uint32_t rowg = lane >> 4;
    const uint32_t head = blockIdx.y;
    const uint32_t sequence = args.sequence;

    /* Clamp the query block so a short tail re-computes overlapping rows: that
     * keeps the query fragments loadable straight from global. The recomputed
     * rows must not be stored twice, though -- two waves computing one row
     * disagree in the last bits, and which store lands last is a race -- so
     * each wave stores only the rows at or after its unclamped start, which
     * makes the union exact and disjoint. Requires sequence >= QROWS, enforced
     * by the dispatch. */
    uint32_t q_row0 = blockIdx.x * (WAVES * QROWS) + wave * QROWS;
    uint32_t qbase = q_row0 + QROWS <= sequence ? q_row0 : sequence - QROWS;

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bf16, rocwmma::row_major>
        fq[8];
    const uint16_t *qp = query + (size_t)(qbase * args.heads + head) * D;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        rocwmma::load_matrix_sync(
            fq[i], reinterpret_cast<const bf16 *>(qp + (uint32_t)i * 16u),
            args.heads * D);
    }

    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fo[8];
#pragma unroll
    for (int j = 0; j < 8; j++) rocwmma::fill_fragment(fo[j], 0.0f);
    float row_max[8], row_sum[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        row_max[i] = -1e30f;
        row_sum[i] = 0.0f;
    }

    const uint32_t kv_stride = args.kv_head_major ? D : args.heads * D;
    const size_t kv_base = args.kv_head_major
                               ? (size_t)head * sequence * D
                               : (size_t)head * D;
    const uint16_t *kp = key + kv_base;
    const uint16_t *vp = value + kv_base;

    for (uint32_t k0 = 0; k0 < sequence; k0 += BK) {
        __syncthreads();
        /* Coalesced uint2 staging: consecutive threads take consecutive
         * 8-byte chunks; rows past the sequence end are zero filled. */
        for (uint32_t idx = tid * 4u; idx < BK * D; idx += THREADS * 4u) {
            uint32_t r = idx >> 7;
            uint32_t d = idx & 127u;
            uint32_t g = k0 + r;
            uint2 kk = uint2{0u, 0u};
            uint2 vv = uint2{0u, 0u};
            if (g < sequence) {
                size_t off = (size_t)g * kv_stride + d;
                kk = *reinterpret_cast<const uint2 *>(kp + off);
                vv = *reinterpret_cast<const uint2 *>(vp + off);
            }
            *reinterpret_cast<uint2 *>(sk + r * LD + d) = kk;
            *reinterpret_cast<uint2 *>(sv + r * LD + d) = vv;
        }
        __syncthreads();

        /* S = Q @ K^T. K is row-major [key][dim] in LDS, so reading it as
         * col_major with the same leading dimension yields K^T directly. */
        rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fs[NT];
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            rocwmma::fill_fragment(fs[t], 0.0f);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bf16,
                                  rocwmma::col_major> fk;
                rocwmma::load_matrix_sync(
                    fk, reinterpret_cast<const bf16 *>(
                            sk + t * 16u * LD + (uint32_t)i * 16u),
                    LD);
                rocwmma::mma_sync(fs[t], fq[i], fk, fs[t]);
            }
        }

        float alpha[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            float s[NT];
            float m = -1e30f;
#pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                s[t] = k0 + t * 16u + col < sequence
                           ? fs[t].x[i] * args.scale
                           : -1e30f;
                m = fmaxf(m, s[t]);
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
                m = fmaxf(m, h3_shfl_xor32(m, off));
            float new_max = fmaxf(row_max[i], m);
            alpha[i] = h3_fast_exp(row_max[i] - new_max);
            uint32_t prow = 2u * (uint32_t)i + rowg;
            float ps = 0.0f;
#pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                float p = h3_fast_exp(s[t] - new_max);
                ps += p;
                sp[wave][prow * BK + t * 16u + col] = h3_f32_to_bf16(p);
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) ps += h3_shfl_xor32(ps, off);
            row_sum[i] = fmaf(row_sum[i], alpha[i], ps);
            row_max[i] = new_max;
        }
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int i = 0; i < 8; i++) fo[j].x[i] *= alpha[i];

        /* O += P @ V. sp is written and read by the same wave, so the
         * compiler's LDS dependency tracking covers it without a barrier. */
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bf16,
                              rocwmma::row_major> fp;
            rocwmma::load_matrix_sync(
                fp, reinterpret_cast<const bf16 *>(sp[wave] + t * 16u), BK);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bf16,
                                  rocwmma::row_major> fv;
                rocwmma::load_matrix_sync(
                    fv, reinterpret_cast<const bf16 *>(
                            sv + t * 16u * LD + (uint32_t)j * 16u),
                    LD);
                rocwmma::mma_sync(fo[j], fp, fv, fo[j]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        float inv = row_sum[i] > 0.0f ? 1.0f / row_sum[i] : 0.0f;
        uint32_t q_pos = qbase + 2u * (uint32_t)i + rowg;
        if (q_pos < q_row0) continue;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            uint32_t d = (uint32_t)j * 16u + col;
            output[h3_sdpa_output_index(args, q_pos, head, d)] =
                h3_f32_to_bf16(fo[j].x[i] * inv);
        }
    }
}

/* SageAttention INT8 QK kernel (MI300X first).
 *
 * Per-tile INT8 quantization for Q and K (one absmax scale per QROWS×128 tile).
 * Scalar INT8 dot product for QK^T (v1 correctness baseline).
 * Dequant to FP32 for online softmax. FP32 PV accumulation.
 *
 * Opt-in: H3_SAGE_SDPA=1. Default off.
 *
 * Data layout: [sequence, heads, head_dim] (row-major, head_dim=128)
 */
template <uint32_t WAVES, uint32_t BK>
__global__ void __launch_bounds__(WAVES * 64u)
h3_sdpa_sage_int8_d128_kernel(const uint16_t *query, const uint16_t *key,
                               const uint16_t *value, uint16_t *output,
                               h3_sdpa_args args) {
    constexpr uint32_t D = 128u;
    constexpr uint32_t QROWS = H3_SDPA_WMMA_QROWS;
    constexpr uint32_t THREADS = WAVES * 64u;
    constexpr float I8_MAX = 127.0f;

    /* Per-wave output accumulators: 4 rows × 128 dims in 8 fragments of 16 */
    float acc[4][D];
    float row_max[4], row_sum[4];
    for (int i = 0; i < 4; i++) {
        row_max[i] = -1e30f;
        row_sum[i] = 0.0f;
        for (uint32_t d = 0; d < D; d++) acc[i][d] = 0.0f;
    }

    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid / THREADS * (threadIdx.x / THREADS);
    const uint32_t head = blockIdx.y;
    const uint32_t sequence = args.sequence;
    const uint32_t q_row0 = blockIdx.x * (WAVES * QROWS) + (tid / 64u) * QROWS;
    const uint32_t qbase = q_row0 + QROWS <= sequence ? q_row0
                                                       : (sequence > QROWS ? sequence - QROWS : 0u);

    const uint32_t kv_stride = args.kv_head_major ? D : args.heads * D;
    const size_t kv_base = args.kv_head_major
                               ? (size_t)head * sequence * D
                               : (size_t)head * D;
    const uint16_t *qp = query + (size_t)(qbase * args.heads + head) * D;
    const uint16_t *kp = key + kv_base;
    const uint16_t *vp = value + kv_base;

    /* Shared memory for Q tile */
    __shared__ int8_t sq_i8[QROWS][D];
    __shared__ float sq_scale;

    /* Load Q tile and quantize to INT8 (per-tile absmax) */
    {
        float local_max = 0.0f;
        for (uint32_t idx = tid; idx < QROWS * D; idx += THREADS) {
            uint32_t r = idx / D;
            uint32_t d = idx % D;
            if (qbase + r < sequence) {
                float v = fabsf(h3_bf16_to_f32(qp[r * D + d]));
                local_max = fmaxf(local_max, v);
            }
        }
        /* Warp reduce max */
        for (uint32_t off = 16; off > 0; off >>= 1)
            local_max = fmaxf(local_max, __shfl_xor(local_max, off, 16));
        /* Block reduce max */
        __shared__ float warp_maxes[16];
        if ((tid & 31u) == 0u) warp_maxes[tid >> 5] = local_max;
        __syncthreads();
        if (tid == 0u) {
            float m = 0.0f;
            for (uint32_t i = 0; i < THREADS / 32u; i++)
                m = fmaxf(m, warp_maxes[i]);
            sq_scale = m > 0.0f ? m / I8_MAX : 1.0f;
        }
        __syncthreads();

        float scale = sq_scale;
        for (uint32_t idx = tid; idx < QROWS * D; idx += THREADS) {
            uint32_t r = idx / D;
            uint32_t d = idx % D;
            if (qbase + r < sequence) {
                float v = h3_bf16_to_f32(qp[r * D + d]);
                float q = roundf(v / scale);
                q = fminf(fmaxf(q, -I8_MAX), I8_MAX);
                sq_i8[r][d] = (int8_t)q;
            } else {
                sq_i8[r][d] = 0;
            }
        }
    }
    __syncthreads();

    /* Main loop: process K tiles */
    __shared__ int8_t sk_i8[BK][D];
    __shared__ float sk_scale;
    __shared__ uint16_t sv_tile[BK][D];

    for (uint32_t k0 = 0; k0 < sequence; k0 += BK) {
        /* Load K tile and quantize to INT8 (per-tile absmax) */
        {
            float local_max = 0.0f;
            for (uint32_t idx = tid; idx < BK * D; idx += THREADS) {
                uint32_t r = idx / D;
                uint32_t d = idx % D;
                uint32_t g = k0 + r;
                if (g < sequence) {
                    float v = fabsf(h3_bf16_to_f32(kp[(size_t)g * kv_stride + d]));
                    local_max = fmaxf(local_max, v);
                }
            }
            for (uint32_t off = 16; off > 0; off >>= 1)
                local_max = fmaxf(local_max, __shfl_xor(local_max, off, 16));
            __shared__ float k_warp_maxes[16];
            if ((tid & 31u) == 0u) k_warp_maxes[tid >> 5] = local_max;
            __syncthreads();
            if (tid == 0u) {
                float m = 0.0f;
                for (uint32_t i = 0; i < THREADS / 32u; i++)
                    m = fmaxf(m, k_warp_maxes[i]);
                sk_scale = m > 0.0f ? m / I8_MAX : 1.0f;
            }
            __syncthreads();

            float scale = sk_scale;
            for (uint32_t idx = tid; idx < BK * D; idx += THREADS) {
                uint32_t r = idx / D;
                uint32_t d = idx % D;
                uint32_t g = k0 + r;
                if (g < sequence) {
                    float v = h3_bf16_to_f32(kp[(size_t)g * kv_stride + d]);
                    float q = roundf(v / scale);
                    q = fminf(fmaxf(q, -I8_MAX), I8_MAX);
                    sk_i8[r][d] = (int8_t)q;
                } else {
                    sk_i8[r][d] = 0;
                }
            }

            /* Load V tile */
            for (uint32_t idx = tid; idx < BK * D; idx += THREADS) {
                uint32_t r = idx / D;
                uint32_t d = idx % D;
                uint32_t g = k0 + r;
                if (g < sequence) {
                    sv_tile[r][d] = vp[(size_t)g * kv_stride + d];
                } else {
                    sv_tile[r][d] = 0u;
                }
            }
        }
        __syncthreads();

        float qscale = sq_scale;
        float kscale = sk_scale;

        /* INT8 QK^T: each thread handles one (qi, ki) pair */
        /* Use thread layout: THREADS threads across QROWS × BK tiles */
        uint32_t total_pairs = QROWS * BK;
        for (uint32_t pair = tid; pair < total_pairs; pair += THREADS) {
            uint32_t qi = pair / BK;
            uint32_t ki = pair % BK;
            uint32_t q_pos = qbase + qi;
            uint32_t k_pos = k0 + ki;
            if (q_pos >= sequence || k_pos >= sequence) continue;

            /* INT8 dot product over 128 dimensions */
            int32_t dot = 0;
            for (uint32_t d = 0; d < D; d++) {
                dot += (int32_t)sq_i8[qi][d] * (int32_t)sk_i8[ki][d];
            }

            /* Dequant to FP32 */
            float score = (float)dot * qscale * kscale * args.scale;

            /* Online softmax update (per-row) */
            float old_max = row_max[qi];
            float new_max = fmaxf(old_max, score);
            float alpha = h3_fast_exp(old_max - new_max);
            float p = h3_fast_exp(score - new_max);

            row_sum[qi] = row_sum[qi] * alpha + p;

            /* Rescale previous accumulators */
            for (uint32_t d = 0; d < D; d++)
                acc[qi][d] *= alpha;

            /* PV: accumulate p * V[ki] */
            for (uint32_t d = 0; d < D; d++)
                acc[qi][d] += p * h3_bf16_to_f32(sv_tile[ki][d]);

            row_max[qi] = new_max;
        }
        __syncthreads();
    }

    /* Write output */
    for (int qi = 0; qi < 4; qi++) {
        uint32_t q_pos = qbase + qi;
        if (q_pos >= q_row0 && q_pos < sequence) {
            float inv = row_sum[qi] > 0.0f ? 1.0f / row_sum[qi] : 0.0f;
            for (uint32_t d = tid; d < D; d += THREADS) {
                output[h3_sdpa_output_index(args, q_pos, head, d)] =
                    h3_f32_to_bf16(acc[qi][d] * inv);
            }
        }
    }
}

/* gfx90a/CDNA2 counterpart of the wave32 WMMA kernel above. rocWMMA maps a
 * 16x16 f32 accumulator over a wave64 as:
 *   lane l -> columns l&15, rows 4*(l>>4) + fragment element [0..3].
 * Keeping that mapping explicit lets online-softmax rescale the PV
 * accumulators in registers and avoids materializing the score matrix in HBM.
 */
template <uint32_t WAVES, uint32_t BK, bool FP16_PV = false>
__global__ void __launch_bounds__(WAVES * 64u)
h3_sdpa_bf16_mfma_d128_kernel(const uint16_t *query, const uint16_t *key,
                              const uint16_t *value, uint16_t *output,
                              h3_sdpa_args args) {
    constexpr uint32_t D = 128u;
    constexpr uint32_t LD = H3_SDPA_WMMA_LD;
    constexpr uint32_t QROWS = H3_SDPA_WMMA_QROWS;
    constexpr uint32_t THREADS = WAVES * 64u;
    constexpr uint32_t NT = BK / 16u;
    typedef rocwmma::bfloat16_t bf16;
    typedef typename std::conditional<FP16_PV, _Float16, bf16>::type pv16;

    constexpr uint32_t STAGES = 1u;
    __shared__ __align__(16) uint16_t sk[STAGES][BK * LD];
    __shared__ __align__(16) uint16_t sv[STAGES][BK * LD];
    __shared__ __align__(16) uint16_t sp[WAVES][QROWS * BK];

    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 6;
    const uint32_t lane = tid & 63u;
    const uint32_t col = lane & 15u;
    const uint32_t rowg = lane >> 4;
    const uint32_t head = blockIdx.y;
    const uint32_t sequence = args.sequence;

    uint32_t q_row0 = blockIdx.x * (WAVES * QROWS) + wave * QROWS;
    uint32_t qbase = q_row0 + QROWS <= sequence ? q_row0 : sequence - QROWS;

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bf16,
                      rocwmma::row_major> fq[8];
    const uint16_t *qp = query + (size_t)(qbase * args.heads + head) * D;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        rocwmma::load_matrix_sync(
            fq[i], reinterpret_cast<const bf16 *>(qp + (uint32_t)i * 16u),
            args.heads * D);
    }

    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fo[8];
#pragma unroll
    for (int j = 0; j < 8; j++) rocwmma::fill_fragment(fo[j], 0.0f);
    float row_max[4], row_sum[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        row_max[i] = -1e30f;
        row_sum[i] = 0.0f;
    }

    const uint32_t kv_stride = args.kv_head_major ? D : args.heads * D;
    const size_t kv_base = args.kv_head_major
                               ? (size_t)head * sequence * D
                               : (size_t)head * D;
    const uint16_t *kp = key + kv_base;
    const uint16_t *vp = value + kv_base;

    auto load_kv = [&](uint32_t k0, uint32_t st) {
        uint16_t *sks = sk[st];
        uint16_t *svs = sv[st];
        for (uint32_t idx = tid * 4u; idx < BK * D; idx += THREADS * 4u) {
            uint32_t r = idx >> 7;
            uint32_t d = idx & 127u;
            uint32_t g = k0 + r;
            uint2 kk = uint2{0u, 0u};
            if (g < sequence) {
                size_t off = (size_t)g * kv_stride + d;
                kk = *reinterpret_cast<const uint2 *>(kp + off);
                if constexpr (FP16_PV) {
                    uint32_t *dst =
                        reinterpret_cast<uint32_t *>(svs + r * LD + d);
                    dst[0] = h3_pack_f16x2(h3_bf16_to_f32(vp[off]),
                                           h3_bf16_to_f32(vp[off + 1u]));
                    dst[1] = h3_pack_f16x2(h3_bf16_to_f32(vp[off + 2u]),
                                           h3_bf16_to_f32(vp[off + 3u]));
                } else {
                    *reinterpret_cast<uint2 *>(svs + r * LD + d) =
                        *reinterpret_cast<const uint2 *>(vp + off);
                }
            } else {
                *reinterpret_cast<uint2 *>(svs + r * LD + d) = uint2{0u, 0u};
            }
            *reinterpret_cast<uint2 *>(sks + r * LD + d) = kk;
        }
    };

    uint32_t stage = 0;
    load_kv(0, 0);
    __syncthreads();
    for (uint32_t k0 = 0; k0 < sequence; k0 += BK) {

        rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fs[NT];
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            rocwmma::fill_fragment(fs[t], 0.0f);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bf16,
                                  rocwmma::col_major> fk;
                rocwmma::load_matrix_sync(
                    fk, reinterpret_cast<const bf16 *>(
                            sk[stage] + t * 16u * LD + (uint32_t)i * 16u),
                    LD);
                rocwmma::mma_sync(fs[t], fq[i], fk, fs[t]);
            }
        }

        float alpha[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float s[NT];
            float m = -1e30f;
#pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                s[t] = k0 + t * 16u + col < sequence
                           ? fs[t].x[i] * args.scale
                           : -1e30f;
                m = fmaxf(m, s[t]);
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
                m = fmaxf(m, __shfl_xor(m, off, 16));
            float new_max = fmaxf(row_max[i], m);
            alpha[i] = h3_fast_exp(row_max[i] - new_max);
            uint32_t prow = rowg * 4u + (uint32_t)i;
            float ps = 0.0f;
#pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                float p = h3_fast_exp(s[t] - new_max);
                ps += p;
                uint16_t *pdst = sp[wave] + prow * BK + t * 16u + col;
                if constexpr (FP16_PV)
                    *reinterpret_cast<_Float16 *>(pdst) = (_Float16)p;
                else
                    *pdst = h3_f32_to_bf16(p);
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
                ps += __shfl_xor(ps, off, 16);
            row_sum[i] = fmaf(row_sum[i], alpha[i], ps);
            row_max[i] = new_max;
        }
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int i = 0; i < 4; i++) fo[j].x[i] *= alpha[i];

        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, pv16,
                              rocwmma::row_major> fp;
            rocwmma::load_matrix_sync(
                fp, reinterpret_cast<const pv16 *>(sp[wave] + t * 16u), BK);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, pv16,
                                  rocwmma::row_major> fv;
                rocwmma::load_matrix_sync(
                    fv, reinterpret_cast<const pv16 *>(
                            sv[stage] + t * 16u * LD + (uint32_t)j * 16u),
                    LD);
                rocwmma::mma_sync(fo[j], fp, fv, fo[j]);
            }
        }
        uint32_t k_next = k0 + BK;
        if (k_next < sequence) {
            if constexpr (STAGES == 2u) {
                load_kv(k_next, stage ^ 1u);
            } else {
                __syncthreads();
                load_kv(k_next, 0);
            }
        }
        __syncthreads();
        if constexpr (STAGES == 2u) stage ^= 1u;
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
        float inv = row_sum[i] > 0.0f ? 1.0f / row_sum[i] : 0.0f;
        uint32_t q_pos = qbase + rowg * 4u + (uint32_t)i;
        if (q_pos < q_row0) continue;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            uint32_t d = (uint32_t)j * 16u + col;
            output[h3_sdpa_output_index(args, q_pos, head, d)] =
                h3_f32_to_bf16(fo[j].x[i] * inv);
        }
    }
}

/* Same tiled-WMMA flash attention for the f32 head_dim=64 path used by the
 * video VAE decoder. The wave kernel it replaces spends a 5-shuffle
 * h3_warp_reduce_sum on every (query, key) score, which held it to 1.28
 * TFLOP/s; here the scores come out of the matrix units instead.
 *
 * Q/K/V arrive as f32, so all three are rounded to bf16 on the way into LDS
 * (the accumulator and the output stay f32). Q has to go through LDS as well,
 * since load_matrix_sync cannot read f32 global memory as bf16 fragments.
 *
 * 64 dims of padding-free bf16 would put every 16-row fragment load in the same
 * LDS bank group, so rows are padded to 72. */
#define H3_SDPA_WMMA_D64_LD 72u

template <uint32_t WAVES, uint32_t BK>
__global__ void __launch_bounds__(WAVES * 32u)
h3_sdpa_f32_wmma_d64_kernel(const float *query, const float *key,
                            const float *value, float *output,
                            h3_sdpa_args args) {
    constexpr uint32_t D = 64u;
    constexpr uint32_t ND = D / 16u; /* dim slices */
    constexpr uint32_t LD = H3_SDPA_WMMA_D64_LD;
    constexpr uint32_t QROWS = H3_SDPA_WMMA_QROWS;
    constexpr uint32_t THREADS = WAVES * 32u;
    constexpr uint32_t NT = BK / 16u;
    typedef rocwmma::bfloat16_t bf16;

    __shared__ __align__(16) uint16_t sk[BK * LD];
    __shared__ __align__(16) uint16_t sv[BK * LD];
    __shared__ __align__(16) uint16_t sq[WAVES][QROWS * LD];
    __shared__ __align__(16) uint16_t sp[WAVES][QROWS * BK];

    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 5;
    const uint32_t lane = tid & 31u;
    const uint32_t col = lane & 15u;
    const uint32_t rowg = lane >> 4;
    const uint32_t head = blockIdx.y;
    const uint32_t sequence = args.sequence;

    /* A partial query block is masked, not clamped back onto the previous one:
     * letting two waves recompute the same rows made them disagree in the last
     * bits, and which store landed last was a race. Rows past the end read as
     * zero and are never stored. */
    uint32_t qbase = blockIdx.x * (WAVES * QROWS) + wave * QROWS;

    /* Stage this wave's 16 query rows, then build the dim fragments from LDS. */
    const float *qp = query + (size_t)head * D;
    for (uint32_t idx = lane * 4u; idx < QROWS * D; idx += 32u * 4u) {
        uint32_t r = idx / D, d = idx % D;
        uint32_t g = qbase + r;
        float4 q = make_float4(0.f, 0.f, 0.f, 0.f);
        if (g < sequence) {
            q = *reinterpret_cast<const float4 *>(
                qp + (size_t)g * args.heads * D + d);
        }
        uint32_t *dst = reinterpret_cast<uint32_t *>(sq[wave] + r * LD + d);
        dst[0] = h3_pack_bf16x2(q.x, q.y);
        dst[1] = h3_pack_bf16x2(q.z, q.w);
    }
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();
    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bf16, rocwmma::row_major>
        fq[ND];
#pragma unroll
    for (uint32_t i = 0; i < ND; i++)
        rocwmma::load_matrix_sync(
            fq[i], reinterpret_cast<const bf16 *>(sq[wave] + i * 16u), LD);

    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fo[ND];
#pragma unroll
    for (uint32_t j = 0; j < ND; j++) rocwmma::fill_fragment(fo[j], 0.0f);
    float row_max[8], row_sum[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        row_max[i] = -1e30f;
        row_sum[i] = 0.0f;
    }

    const uint32_t kv_stride = args.kv_head_major ? D : args.heads * D;
    const size_t kv_base = args.kv_head_major ? (size_t)head * sequence * D
                                              : (size_t)head * D;
    const float *kp = key + kv_base;
    const float *vp = value + kv_base;

    for (uint32_t k0 = 0; k0 < sequence; k0 += BK) {
        __syncthreads();
        for (uint32_t idx = tid * 4u; idx < BK * D; idx += THREADS * 4u) {
            uint32_t r = idx / D, d = idx % D;
            uint32_t g = k0 + r;
            float4 kk = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 vv = kk;
            if (g < sequence) {
                size_t off = (size_t)g * kv_stride + d;
                kk = *reinterpret_cast<const float4 *>(kp + off);
                vv = *reinterpret_cast<const float4 *>(vp + off);
            }
            uint32_t *dk = reinterpret_cast<uint32_t *>(sk + r * LD + d);
            dk[0] = h3_pack_bf16x2(kk.x, kk.y);
            dk[1] = h3_pack_bf16x2(kk.z, kk.w);
            uint32_t *dv = reinterpret_cast<uint32_t *>(sv + r * LD + d);
            dv[0] = h3_pack_bf16x2(vv.x, vv.y);
            dv[1] = h3_pack_bf16x2(vv.z, vv.w);
        }
        __syncthreads();

        rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fs[NT];
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            rocwmma::fill_fragment(fs[t], 0.0f);
#pragma unroll
            for (uint32_t i = 0; i < ND; i++) {
                rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bf16,
                                  rocwmma::col_major> fk;
                rocwmma::load_matrix_sync(
                    fk, reinterpret_cast<const bf16 *>(sk + t * 16u * LD +
                                                       i * 16u),
                    LD);
                rocwmma::mma_sync(fs[t], fq[i], fk, fs[t]);
            }
        }

        float alpha[8];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            float s[NT];
            float m = -1e30f;
#pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                s[t] = k0 + t * 16u + col < sequence
                           ? fs[t].x[i] * args.scale
                           : -1e30f;
                m = fmaxf(m, s[t]);
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
                m = fmaxf(m, h3_shfl_xor32(m, off));
            float new_max = fmaxf(row_max[i], m);
            alpha[i] = h3_fast_exp(row_max[i] - new_max);
            uint32_t prow = 2u * (uint32_t)i + rowg;
            float ps = 0.0f;
#pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                float p = h3_fast_exp(s[t] - new_max);
                ps += p;
                sp[wave][prow * BK + t * 16u + col] = h3_f32_to_bf16(p);
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) ps += h3_shfl_xor32(ps, off);
            row_sum[i] = fmaf(row_sum[i], alpha[i], ps);
            row_max[i] = new_max;
        }
#pragma unroll
        for (uint32_t j = 0; j < ND; j++)
#pragma unroll
            for (int i = 0; i < 8; i++) fo[j].x[i] *= alpha[i];

        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bf16,
                              rocwmma::row_major> fp;
            rocwmma::load_matrix_sync(
                fp, reinterpret_cast<const bf16 *>(sp[wave] + t * 16u), BK);
#pragma unroll
            for (uint32_t j = 0; j < ND; j++) {
                rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bf16,
                                  rocwmma::row_major> fv;
                rocwmma::load_matrix_sync(
                    fv, reinterpret_cast<const bf16 *>(sv + t * 16u * LD +
                                                       j * 16u),
                    LD);
                rocwmma::mma_sync(fo[j], fp, fv, fo[j]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 8; i++) {
        float inv = row_sum[i] > 0.0f ? 1.0f / row_sum[i] : 0.0f;
        uint32_t q_pos = qbase + 2u * (uint32_t)i + rowg;
        if (q_pos >= sequence) continue;
#pragma unroll
        for (uint32_t j = 0; j < ND; j++) {
            uint32_t d = j * 16u + col;
            output[h3_sdpa_output_index(args, q_pos, head, d)] =
                fo[j].x[i] * inv;
        }
    }
}

/* gfx90a wave64 counterpart of the f32 d64 WMMA kernel. Same online softmax
 * and LDS K/V tiling; accumulator rows are 4*(lane>>4)+i. */
template <uint32_t WAVES, uint32_t BK>
__global__ void __launch_bounds__(WAVES * 64u)
h3_sdpa_f32_mfma_d64_kernel(const float *query, const float *key,
                            const float *value, float *output,
                            h3_sdpa_args args) {
    constexpr uint32_t D = 64u;
    constexpr uint32_t ND = D / 16u;
    constexpr uint32_t LD = H3_SDPA_WMMA_D64_LD;
    constexpr uint32_t QROWS = H3_SDPA_WMMA_QROWS;
    constexpr uint32_t THREADS = WAVES * 64u;
    constexpr uint32_t NT = BK / 16u;
    typedef rocwmma::bfloat16_t bf16;

    __shared__ __align__(16) uint16_t sk[BK * LD];
    __shared__ __align__(16) uint16_t sv[BK * LD];
    __shared__ __align__(16) uint16_t sq[WAVES][QROWS * LD];
    __shared__ __align__(16) uint16_t sp[WAVES][QROWS * BK];

    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 6;
    const uint32_t lane = tid & 63u;
    const uint32_t col = lane & 15u;
    const uint32_t rowg = lane >> 4;
    const uint32_t head = blockIdx.y;
    const uint32_t sequence = args.sequence;
    uint32_t qbase = blockIdx.x * (WAVES * QROWS) + wave * QROWS;

    const float *qp = query + (size_t)head * D;
    for (uint32_t idx = lane * 4u; idx < QROWS * D; idx += 64u * 4u) {
        uint32_t r = idx / D, d = idx % D;
        uint32_t g = qbase + r;
        float4 q = make_float4(0.f, 0.f, 0.f, 0.f);
        if (g < sequence) {
            q = *reinterpret_cast<const float4 *>(
                qp + (size_t)g * args.heads * D + d);
        }
        uint32_t *dst = reinterpret_cast<uint32_t *>(sq[wave] + r * LD + d);
        dst[0] = h3_pack_bf16x2(q.x, q.y);
        dst[1] = h3_pack_bf16x2(q.z, q.w);
    }
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();
    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bf16, rocwmma::row_major>
        fq[ND];
#pragma unroll
    for (uint32_t i = 0; i < ND; i++)
        rocwmma::load_matrix_sync(
            fq[i], reinterpret_cast<const bf16 *>(sq[wave] + i * 16u), LD);

    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fo[ND];
#pragma unroll
    for (uint32_t j = 0; j < ND; j++) rocwmma::fill_fragment(fo[j], 0.0f);
    float row_max[4], row_sum[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        row_max[i] = -1e30f;
        row_sum[i] = 0.0f;
    }

    const uint32_t kv_stride = args.kv_head_major ? D : args.heads * D;
    const size_t kv_base = args.kv_head_major ? (size_t)head * sequence * D
                                              : (size_t)head * D;
    const float *kp = key + kv_base;
    const float *vp = value + kv_base;

    for (uint32_t k0 = 0; k0 < sequence; k0 += BK) {
        __syncthreads();
        for (uint32_t idx = tid * 4u; idx < BK * D; idx += THREADS * 4u) {
            uint32_t r = idx / D, d = idx % D;
            uint32_t g = k0 + r;
            float4 kk = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 vv = kk;
            if (g < sequence) {
                size_t off = (size_t)g * kv_stride + d;
                kk = *reinterpret_cast<const float4 *>(kp + off);
                vv = *reinterpret_cast<const float4 *>(vp + off);
            }
            uint32_t *dk = reinterpret_cast<uint32_t *>(sk + r * LD + d);
            dk[0] = h3_pack_bf16x2(kk.x, kk.y);
            dk[1] = h3_pack_bf16x2(kk.z, kk.w);
            uint32_t *dv = reinterpret_cast<uint32_t *>(sv + r * LD + d);
            dv[0] = h3_pack_bf16x2(vv.x, vv.y);
            dv[1] = h3_pack_bf16x2(vv.z, vv.w);
        }
        __syncthreads();

        rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fs[NT];
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            rocwmma::fill_fragment(fs[t], 0.0f);
#pragma unroll
            for (uint32_t i = 0; i < ND; i++) {
                rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bf16,
                                  rocwmma::col_major> fk;
                rocwmma::load_matrix_sync(
                    fk, reinterpret_cast<const bf16 *>(sk + t * 16u * LD +
                                                       i * 16u),
                    LD);
                rocwmma::mma_sync(fs[t], fq[i], fk, fs[t]);
            }
        }

        float alpha[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float s[NT];
            float m = -1e30f;
#pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                s[t] = k0 + t * 16u + col < sequence
                           ? fs[t].x[i] * args.scale
                           : -1e30f;
                m = fmaxf(m, s[t]);
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
                m = fmaxf(m, __shfl_xor(m, off, 16));
            float new_max = fmaxf(row_max[i], m);
            alpha[i] = h3_fast_exp(row_max[i] - new_max);
            uint32_t prow = rowg * 4u + (uint32_t)i;
            float ps = 0.0f;
#pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                float p = h3_fast_exp(s[t] - new_max);
                ps += p;
                sp[wave][prow * BK + t * 16u + col] = h3_f32_to_bf16(p);
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
                ps += __shfl_xor(ps, off, 16);
            row_sum[i] = fmaf(row_sum[i], alpha[i], ps);
            row_max[i] = new_max;
        }
#pragma unroll
        for (uint32_t j = 0; j < ND; j++)
#pragma unroll
            for (int i = 0; i < 4; i++) fo[j].x[i] *= alpha[i];

        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bf16,
                              rocwmma::row_major> fp;
            rocwmma::load_matrix_sync(
                fp, reinterpret_cast<const bf16 *>(sp[wave] + t * 16u), BK);
#pragma unroll
            for (uint32_t j = 0; j < ND; j++) {
                rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bf16,
                                  rocwmma::row_major> fv;
                rocwmma::load_matrix_sync(
                    fv, reinterpret_cast<const bf16 *>(sv + t * 16u * LD +
                                                       j * 16u),
                    LD);
                rocwmma::mma_sync(fo[j], fp, fv, fo[j]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
        float inv = row_sum[i] > 0.0f ? 1.0f / row_sum[i] : 0.0f;
        uint32_t q_pos = qbase + rowg * 4u + (uint32_t)i;
        if (q_pos >= sequence) continue;
#pragma unroll
        for (uint32_t j = 0; j < ND; j++) {
            uint32_t d = j * 16u + col;
            output[h3_sdpa_output_index(args, q_pos, head, d)] =
                fo[j].x[i] * inv;
        }
    }
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
    /* Four-key online softmax on VAE d64 (2-key helped; 8-key flat). */
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

/* Two queries share each K/V load (d64 VAE). Opt out: H3_SDPA_D64_Q1=1. */
__global__ void __launch_bounds__(32)
h3_sdpa_f32_wave_d64_q2_kernel(const float *query, const float *key,
                               const float *value, float *output,
                               h3_sdpa_args args) {
    uint32_t q_pos = blockIdx.x * 2u;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    int q1_live = (q_pos + 1u) < args.sequence;
    uint32_t q_base0 = (q_pos * args.heads + head) * 64u;
    uint32_t q_base1 = q_base0 + args.heads * 64u;
    uint32_t d0 = lane * 2u;
    float2 aq = h3_load_f32x2(query + q_base0 + d0);
    float a0 = aq.x * args.scale;
    float a1 = aq.y * args.scale;
    float b0 = 0.0f, b1 = 0.0f;
    if (q1_live) {
        float2 bq = h3_load_f32x2(query + q_base1 + d0);
        b0 = bq.x * args.scale;
        b1 = bq.y * args.scale;
    }
    float acc_a0 = 0.0f, acc_a1 = 0.0f, acc_b0 = 0.0f, acc_b1 = 0.0f;
    float max_a = -1e30f, max_b = -1e30f, sum_a = 0.0f, sum_b = 0.0f;
    uint32_t stride = args.kv_head_major ? 64u : args.heads * 64u;
    uint32_t k_base = args.kv_head_major ? head * args.sequence * 64u
                                         : head * 64u;
    uint32_t k_pos = 0;
    for (; k_pos + 3u < args.sequence; k_pos += 4u) {
        float v0s[4], v1s[4], sa[4], sb[4];
#pragma unroll
        for (int t = 0; t < 4; t++) {
            uint32_t base = k_base + (uint32_t)t * stride;
            float2 kk = h3_load_f32x2(key + base + d0);
            float2 vv = h3_load_f32x2(value + base + d0);
            v0s[t] = vv.x;
            v1s[t] = vv.y;
            sa[t] = h3_warp_reduce_sum(a0 * kk.x + a1 * kk.y);
            sb[t] = q1_live ? h3_warp_reduce_sum(b0 * kk.x + b1 * kk.y) : 0.0f;
        }
        float new_a = fmaxf(max_a, fmaxf(fmaxf(sa[0], sa[1]),
                                         fmaxf(sa[2], sa[3])));
        float alpha_a = h3_fast_exp(max_a - new_a);
        float pa0 = h3_fast_exp(sa[0] - new_a);
        float pa1 = h3_fast_exp(sa[1] - new_a);
        float pa2 = h3_fast_exp(sa[2] - new_a);
        float pa3 = h3_fast_exp(sa[3] - new_a);
        acc_a0 = fmaf(pa0, v0s[0], fmaf(pa1, v0s[1],
                     fmaf(pa2, v0s[2], fmaf(pa3, v0s[3], acc_a0 * alpha_a))));
        acc_a1 = fmaf(pa0, v1s[0], fmaf(pa1, v1s[1],
                     fmaf(pa2, v1s[2], fmaf(pa3, v1s[3], acc_a1 * alpha_a))));
        sum_a = fmaf(sum_a, alpha_a, pa0 + pa1 + pa2 + pa3);
        max_a = new_a;
        if (q1_live) {
            float new_b = fmaxf(max_b, fmaxf(fmaxf(sb[0], sb[1]),
                                             fmaxf(sb[2], sb[3])));
            float alpha_b = h3_fast_exp(max_b - new_b);
            float pb0 = h3_fast_exp(sb[0] - new_b);
            float pb1 = h3_fast_exp(sb[1] - new_b);
            float pb2 = h3_fast_exp(sb[2] - new_b);
            float pb3 = h3_fast_exp(sb[3] - new_b);
            acc_b0 = fmaf(pb0, v0s[0], fmaf(pb1, v0s[1],
                         fmaf(pb2, v0s[2], fmaf(pb3, v0s[3], acc_b0 * alpha_b))));
            acc_b1 = fmaf(pb0, v1s[0], fmaf(pb1, v1s[1],
                         fmaf(pb2, v1s[2], fmaf(pb3, v1s[3], acc_b1 * alpha_b))));
            sum_b = fmaf(sum_b, alpha_b, pb0 + pb1 + pb2 + pb3);
            max_b = new_b;
        }
        k_base += 4u * stride;
    }
    for (; k_pos < args.sequence; k_pos++) {
        float2 kk = h3_load_f32x2(key + k_base + d0);
        float2 vv = h3_load_f32x2(value + k_base + d0);
        float score_a = h3_warp_reduce_sum(a0 * kk.x + a1 * kk.y);
        float new_a = fmaxf(max_a, score_a);
        float alpha_a = h3_fast_exp(max_a - new_a);
        float pa = h3_fast_exp(score_a - new_a);
        acc_a0 = fmaf(pa, vv.x, acc_a0 * alpha_a);
        acc_a1 = fmaf(pa, vv.y, acc_a1 * alpha_a);
        sum_a = fmaf(sum_a, alpha_a, pa);
        max_a = new_a;
        if (q1_live) {
            float score_b = h3_warp_reduce_sum(b0 * kk.x + b1 * kk.y);
            float new_b = fmaxf(max_b, score_b);
            float alpha_b = h3_fast_exp(max_b - new_b);
            float pb = h3_fast_exp(score_b - new_b);
            acc_b0 = fmaf(pb, vv.x, acc_b0 * alpha_b);
            acc_b1 = fmaf(pb, vv.y, acc_b1 * alpha_b);
            sum_b = fmaf(sum_b, alpha_b, pb);
            max_b = new_b;
        }
        k_base += stride;
    }
    float inv_a = sum_a > 0.0f ? 1.0f / sum_a : 0.0f;
    h3_store_f32x2(output + q_base0 + d0, acc_a0 * inv_a, acc_a1 * inv_a);
    if (q1_live) {
        float inv_b = sum_b > 0.0f ? 1.0f / sum_b : 0.0f;
        h3_store_f32x2(output + q_base1 + d0, acc_b0 * inv_b, acc_b1 * inv_b);
    }
}

/* N queries share each K/V load (d64 VAE). Default N=4.
 * H3_SDPA_D64_Q8=1 is slower on gfx1151 (REJECT as default).
 * Opt out: H3_SDPA_D64_Q2=1 or H3_SDPA_D64_Q1=1. */
template <int QN>
__global__ void __launch_bounds__(32)
h3_sdpa_f32_wave_d64_qn_kernel(const float *query, const float *key,
                               const float *value, float *output,
                               h3_sdpa_args args) {
    uint32_t q_pos = blockIdx.x * (uint32_t)QN;
    uint32_t head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    uint32_t d0 = lane * 2u;
    uint32_t q_stride = args.heads * 64u;
    uint32_t q_base0 = (q_pos * args.heads + head) * 64u;
    int live[QN];
    float q0[QN], q1[QN], acc0[QN], acc1[QN], mx[QN], sm[QN];
#pragma unroll
    for (int qi = 0; qi < QN; qi++) {
        live[qi] = (q_pos + (uint32_t)qi) < args.sequence;
        q0[qi] = 0.0f;
        q1[qi] = 0.0f;
        acc0[qi] = 0.0f;
        acc1[qi] = 0.0f;
        mx[qi] = -1e30f;
        sm[qi] = 0.0f;
        if (!live[qi]) continue;
        float2 qq = h3_load_f32x2(query + q_base0 + (uint32_t)qi * q_stride + d0);
        q0[qi] = qq.x * args.scale;
        q1[qi] = qq.y * args.scale;
    }
    uint32_t stride = args.kv_head_major ? 64u : q_stride;
    uint32_t k_base = args.kv_head_major ? head * args.sequence * 64u
                                         : head * 64u;
    uint32_t k_pos = 0;
    for (; k_pos + 3u < args.sequence; k_pos += 4u) {
        float v0s[4], v1s[4], sc[QN][4];
#pragma unroll
        for (int t = 0; t < 4; t++) {
            uint32_t base = k_base + (uint32_t)t * stride;
            float2 kk = h3_load_f32x2(key + base + d0);
            float2 vv = h3_load_f32x2(value + base + d0);
            v0s[t] = vv.x;
            v1s[t] = vv.y;
#pragma unroll
            for (int qi = 0; qi < QN; qi++)
                sc[qi][t] = (qi && !live[qi]) ? 0.0f :
                    h3_warp_reduce_sum(q0[qi] * kk.x + q1[qi] * kk.y);
        }
#pragma unroll
        for (int qi = 0; qi < QN; qi++) {
            if (qi && !live[qi]) continue;
            float nw = fmaxf(mx[qi], fmaxf(fmaxf(sc[qi][0], sc[qi][1]),
                                           fmaxf(sc[qi][2], sc[qi][3])));
            float alpha = h3_fast_exp(mx[qi] - nw);
            float p0 = h3_fast_exp(sc[qi][0] - nw);
            float p1 = h3_fast_exp(sc[qi][1] - nw);
            float p2 = h3_fast_exp(sc[qi][2] - nw);
            float p3 = h3_fast_exp(sc[qi][3] - nw);
            acc0[qi] = fmaf(p0, v0s[0], fmaf(p1, v0s[1],
                         fmaf(p2, v0s[2], fmaf(p3, v0s[3], acc0[qi] * alpha))));
            acc1[qi] = fmaf(p0, v1s[0], fmaf(p1, v1s[1],
                         fmaf(p2, v1s[2], fmaf(p3, v1s[3], acc1[qi] * alpha))));
            sm[qi] = fmaf(sm[qi], alpha, p0 + p1 + p2 + p3);
            mx[qi] = nw;
        }
        k_base += 4u * stride;
    }
    for (; k_pos < args.sequence; k_pos++) {
        float2 kk = h3_load_f32x2(key + k_base + d0);
        float2 vv = h3_load_f32x2(value + k_base + d0);
#pragma unroll
        for (int qi = 0; qi < QN; qi++) {
            if (qi && !live[qi]) continue;
            float score = h3_warp_reduce_sum(q0[qi] * kk.x + q1[qi] * kk.y);
            float nw = fmaxf(mx[qi], score);
            float alpha = h3_fast_exp(mx[qi] - nw);
            float p = h3_fast_exp(score - nw);
            acc0[qi] = fmaf(p, vv.x, acc0[qi] * alpha);
            acc1[qi] = fmaf(p, vv.y, acc1[qi] * alpha);
            sm[qi] = fmaf(sm[qi], alpha, p);
            mx[qi] = nw;
        }
        k_base += stride;
    }
#pragma unroll
    for (int qi = 0; qi < QN; qi++) {
        if (!live[qi]) continue;
        float inv = sm[qi] > 0.0f ? 1.0f / sm[qi] : 0.0f;
        h3_store_f32x2(output + q_base0 + (uint32_t)qi * q_stride + d0,
                       acc0[qi] * inv, acc1[qi] * inv);
    }
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
    return h3_launch_1d((count + 3u) / 4u, stream,
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
    if (h3_try_linear_bf16_hipblas(input, weight, bias, output, args, stream))
        return 1;
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
    if (host_args.head_dim > H3_QKV_ROPE_MAX_DIM) return 0;
    /* Opt out of the vectorized wave path: H3_QKV_ROPE_SCALAR=1. */
    static int scalar = -1;
    if (scalar < 0) {
        const char *e = getenv("H3_QKV_ROPE_SCALAR");
        scalar = (e && strcmp(e, "0") != 0) ? 1 : 0;
    }
    if (!scalar && host_args.head_dim == 128 && (host_args.rope_half % 4u) == 0u &&
        host_args.rope_half * 2u <= 128u) {
        uint32_t slots = host_args.sequence * host_args.heads;
        dim3 grid((slots + H3_QKV_ROPE_WAVES - 1u) / H3_QKV_ROPE_WAVES, 1, 1);
        return hipLaunchKernel(
                   reinterpret_cast<const void *>(
                       h3_qkv_rope_bf16_d128_kernel),
                   grid, dim3(H3_QKV_ROPE_WAVES * 32u), kernel_args,
                   sizeof(kernel_args), stream) == hipSuccess;
    }
    dim3 block(host_args.head_dim, 1, 1);
    dim3 grid(1, host_args.heads, host_args.sequence);
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

static uint32_t h3_env_u32(const char *name, uint32_t fallback) {
    const char *value = getenv(name);
    if (!value || !*value) return fallback;
    unsigned long parsed = strtoul(value, NULL, 10);
    return parsed ? (uint32_t)parsed : fallback;
}

static int h3_hip_gfx90a(void) {
    static int cached = -1;
    if (cached >= 0) return cached;
    int device = 0;
    hipDeviceProp_t props;
    /* Match all CDNA ISAs: gfx90a (MI210/MI250X) and gfx942 (MI300X). */
    cached = hipGetDevice(&device) == hipSuccess &&
             hipGetDeviceProperties(&props, device) == hipSuccess &&
             (strncmp(props.gcnArchName, "gfx90a", 6) == 0 ||
              strncmp(props.gcnArchName, "gfx942", 6) == 0);
    return cached;
}

extern "C" int h3_hip_rdna_wmma_default(void) {
    static int cached = -1;
    if (cached >= 0) return cached;
    int device = 0;
    hipDeviceProp_t props;
    if (hipGetDevice(&device) != hipSuccess) device = 0;
    if (hipGetDeviceProperties(&props, device) != hipSuccess) {
        cached = 0;
        return cached;
    }
    /* gfx11/gfx12: wave32 rocWMMA layouts probed on gfx1151. gfx9 CDNA is
     * wave64 MFMA, so it must use a separately validated kernel. */
    cached = (strncmp(props.gcnArchName, "gfx11", 5) == 0 ||
              strncmp(props.gcnArchName, "gfx12", 5) == 0) ? 1 : 0;
    return cached;
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
    /* SageAttention INT8 QK opt-in (MI300X first) */
    if (getenv("H3_SAGE_SDPA") && strcmp(getenv("H3_SAGE_SDPA"), "0") != 0) {
        if (h3_hip_gfx90a() && host_args.head_dim == 128 &&
            host_args.sequence >= 128u) {
            uint32_t waves = 8u;
            uint32_t bk = 32u;
            const void *kernel =
                reinterpret_cast<const void *>(
                    h3_sdpa_sage_int8_d128_kernel<8u, 32u>);
            uint32_t bq = waves * H3_SDPA_WMMA_QROWS;
            dim3 grid((host_args.sequence + bq - 1u) / bq,
                      host_args.heads, 1);
            return hipLaunchKernel(kernel, grid, dim3(waves * 64u),
                                   kernel_args, sizeof(kernel_args),
                                   stream) == hipSuccess;
        }
        /* Fallback to BF16 flash for other ISAs or short sequences */
    }
    if (!getenv("H3_SDPA_LEGACY")) {
        const char *cdna_flash = getenv("H3_SDPA_CDNA_FLASH");
        if (h3_hip_gfx90a() && host_args.head_dim == 128 &&
            host_args.sequence >= H3_SDPA_WMMA_QROWS &&
            (!cdna_flash || strcmp(cdna_flash, "0") != 0)) {
            uint32_t waves = h3_env_u32("H3_SDPA_CDNA_WAVES", 8u);
            uint32_t bk = h3_env_u32("H3_SDPA_CDNA_BK", 32u);
            const char *fp16_pv_env = getenv("H3_SDPA_CDNA_FP16_PV");
            int fp16_pv =
                !fp16_pv_env || strcmp(fp16_pv_env, "0") != 0;
            const void *kernel = NULL;
#define H3_SDPA_MFMA_PICK(w, b)                                              \
    if (waves == (w) && bk == (b)) {                                         \
        kernel = fp16_pv                                                     \
            ? reinterpret_cast<const void *>(                                \
                  h3_sdpa_bf16_mfma_d128_kernel<w, b, true>)                  \
            : reinterpret_cast<const void *>(                                \
                  h3_sdpa_bf16_mfma_d128_kernel<w, b, false>);                \
    }
            H3_SDPA_MFMA_PICK(4u, 32u)
            H3_SDPA_MFMA_PICK(4u, 64u)
            H3_SDPA_MFMA_PICK(8u, 16u)
            H3_SDPA_MFMA_PICK(8u, 32u)
            H3_SDPA_MFMA_PICK(8u, 64u)
            H3_SDPA_MFMA_PICK(16u, 32u)
#undef H3_SDPA_MFMA_PICK
            if (!kernel) {
                waves = 8u;
                bk = 32u;
                kernel = fp16_pv
                    ? reinterpret_cast<const void *>(
                          h3_sdpa_bf16_mfma_d128_kernel<8u, 32u, true>)
                    : reinterpret_cast<const void *>(
                          h3_sdpa_bf16_mfma_d128_kernel<8u, 32u, false>);
            }
            uint32_t bq = waves * H3_SDPA_WMMA_QROWS;
            dim3 grid((host_args.sequence + bq - 1u) / bq,
                      host_args.heads, 1);
            return hipLaunchKernel(kernel, grid, dim3(waves * 64u),
                                   kernel_args, sizeof(kernel_args),
                                   stream) == hipSuccess;
        }
        if ((host_args.head_dim == 64 || host_args.head_dim == 128) &&
            host_args.sequence >= 128u &&
            h3_launch_sdpa_bf16_hipblas(query, key, value, output, &host_args,
                                        stream)) {
            return 1;
        }
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
        /* Default d128 path: tiled WMMA flash (matrix units + LDS K/V reuse).
         * Opt out: H3_SDPA_WMMA=0, or force any wave Q-variant below. */
        static int wmma = -1;
        if (wmma < 0) {
            const char *e = getenv("H3_SDPA_WMMA");
            static const char *const forced[] = {
                "H3_SDPA_D128_Q1", "H3_SDPA_D128_Q2", "H3_SDPA_D128_Q3",
                "H3_SDPA_D128_Q4", "H3_SDPA_D128_Q5", "H3_SDPA_Q6_SCALAR"};
            int rdna_wmma = h3_hip_rdna_wmma_default();
            if (e && strcmp(e, "0") == 0) wmma = 0;
            else wmma = rdna_wmma;
            for (size_t i = 0; wmma && i < sizeof(forced) / sizeof(*forced);
                 i++) {
                const char *f = getenv(forced[i]);
                if (f && strcmp(f, "0") != 0) wmma = 0;
            }
        }
        if (wmma && host_args.head_dim == 128 &&
            host_args.sequence >= H3_SDPA_WMMA_QROWS) {
            /* Tile shape knobs for A/B: H3_SDPA_WMMA_WAVES (query block =
             * waves*16) and H3_SDPA_WMMA_BK (key tile). */
            static uint32_t waves = 0u, bk = 0u;
            if (!waves) {
                /* 8 waves (128-row query block) x 32-key tile measured best
                 * on gfx1151 at the DiT shape; see docs/perf-day9. */
                waves = h3_env_u32("H3_SDPA_WMMA_WAVES", 8u);
                bk = h3_env_u32("H3_SDPA_WMMA_BK", 32u);
            }
            const void *kernel = NULL;
#define H3_SDPA_WMMA_PICK(w, b)                                              \
    if (waves == (w) && bk == (b))                                           \
        kernel = reinterpret_cast<const void *>(                             \
            h3_sdpa_bf16_wmma_d128_kernel<w, b>);
            H3_SDPA_WMMA_PICK(2u, 32u)
            H3_SDPA_WMMA_PICK(4u, 16u)
            H3_SDPA_WMMA_PICK(4u, 32u)
            H3_SDPA_WMMA_PICK(4u, 64u)
            H3_SDPA_WMMA_PICK(8u, 16u)
            H3_SDPA_WMMA_PICK(8u, 32u)
            H3_SDPA_WMMA_PICK(8u, 64u)
            H3_SDPA_WMMA_PICK(16u, 16u)
            H3_SDPA_WMMA_PICK(16u, 32u)
#undef H3_SDPA_WMMA_PICK
            if (!kernel) {
                waves = 8u;
                bk = 32u;
                kernel = reinterpret_cast<const void *>(
                    h3_sdpa_bf16_wmma_d128_kernel<8u, 32u>);
            }
            uint32_t bq = waves * H3_SDPA_WMMA_QROWS;
            dim3 grid((host_args.sequence + bq - 1u) / bq, host_args.heads, 1);
            return hipLaunchKernel(kernel, grid, dim3(waves * 32u), kernel_args,
                                   sizeof(kernel_args), stream) == hipSuccess;
        }
        /* Grid (sequence, heads): consecutive blocks share one KV head so
         * head-major K/V stays hot in L2 (gfx1151). */
        dim3 grid(host_args.sequence, host_args.heads, 1);
        const void *kernel = host_args.head_dim == 128 ?
            reinterpret_cast<const void *>(h3_sdpa_bf16_wave_d128_kernel) :
            reinterpret_cast<const void *>(h3_sdpa_bf16_wave_kernel);
        if (host_args.head_dim == 128 &&
            !(getenv("H3_SDPA_D128_Q1") &&
              strcmp(getenv("H3_SDPA_D128_Q1"), "0") != 0)) {
            int force_q2 = getenv("H3_SDPA_D128_Q2") &&
                           strcmp(getenv("H3_SDPA_D128_Q2"), "0") != 0;
            int force_q3 = getenv("H3_SDPA_D128_Q3") &&
                           strcmp(getenv("H3_SDPA_D128_Q3"), "0") != 0;
            int force_q4 = getenv("H3_SDPA_D128_Q4") &&
                           strcmp(getenv("H3_SDPA_D128_Q4"), "0") != 0;
            int force_q5 = getenv("H3_SDPA_D128_Q5") &&
                           strcmp(getenv("H3_SDPA_D128_Q5"), "0") != 0;
            if (force_q2) {
                grid = dim3((host_args.sequence + 1u) / 2u, host_args.heads, 1);
                kernel = reinterpret_cast<const void *>(
                    h3_sdpa_bf16_wave_d128_q2_kernel);
            } else if (force_q3) {
                grid = dim3((host_args.sequence + 2u) / 3u, host_args.heads, 1);
                kernel = reinterpret_cast<const void *>(
                    h3_sdpa_bf16_wave_d128_q3_kernel);
            } else if (force_q4) {
                grid = dim3((host_args.sequence + 3u) / 4u, host_args.heads, 1);
                kernel = reinterpret_cast<const void *>(
                    h3_sdpa_bf16_wave_d128_q4_kernel);
            } else if (force_q5) {
                grid = dim3((host_args.sequence + 4u) / 5u, host_args.heads, 1);
                kernel = reinterpret_cast<const void *>(
                    h3_sdpa_bf16_wave_d128_q5_kernel);
            } else {
                grid = dim3((host_args.sequence + 5u) / 6u, host_args.heads, 1);
                static int q6_scalar = -1;
                if (q6_scalar < 0) {
                    const char *e = getenv("H3_SDPA_Q6_SCALAR");
                    q6_scalar = (e && strcmp(e, "0") != 0) ? 1 : 0;
                }
                if (q6_scalar) {
                    kernel = reinterpret_cast<const void *>(
                        h3_sdpa_bf16_wave_d128_q6_kernel);
                } else if ((host_args.sequence % 6u) == 0u) {
                    kernel = reinterpret_cast<const void *>(
                        h3_sdpa_bf16_wave_d128_q6_vec_kernel<0>);
                } else {
                    kernel = reinterpret_cast<const void *>(
                        h3_sdpa_bf16_wave_d128_q6_vec_kernel<1>);
                }
            }
        }
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

static int h3_sdpa_f32_wmma_on(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("H3_SDPA_F32_WMMA");
        int rdna_wmma = h3_hip_rdna_wmma_default();
        if (e && strcmp(e, "0") == 0) cached = 0;
        else cached = rdna_wmma;
    }
    return cached;
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
        /* CDNA: fused d64 flash before hipBLAS score materialization. */
        const char *cdna_flash = getenv("H3_SDPA_CDNA_FLASH");
        if (h3_hip_gfx90a() && host_args.head_dim == 64 &&
            host_args.sequence >= 256u &&
            (!cdna_flash || strcmp(cdna_flash, "0") != 0)) {
            uint32_t waves = h3_env_u32("H3_SDPA_CDNA_D64_WAVES", 16u);
            uint32_t bk = h3_env_u32("H3_SDPA_CDNA_D64_BK", 32u);
            const void *kernel = NULL;
#define H3_SDPA_F32_MFMA_PICK(w, b)                                          \
    if (waves == (w) && bk == (b))                                           \
        kernel = reinterpret_cast<const void *>(                             \
            h3_sdpa_f32_mfma_d64_kernel<w, b>);
            H3_SDPA_F32_MFMA_PICK(4u, 32u)
            H3_SDPA_F32_MFMA_PICK(8u, 32u)
            H3_SDPA_F32_MFMA_PICK(8u, 64u)
            H3_SDPA_F32_MFMA_PICK(16u, 32u)
#undef H3_SDPA_F32_MFMA_PICK
            if (!kernel) {
                waves = 16u;
                bk = 32u;
                kernel = reinterpret_cast<const void *>(
                    h3_sdpa_f32_mfma_d64_kernel<16u, 32u>);
            }
            uint32_t bq = waves * H3_SDPA_WMMA_QROWS;
            dim3 grid((host_args.sequence + bq - 1u) / bq, host_args.heads, 1);
            return hipLaunchKernel(kernel, grid, dim3(waves * 64u), kernel_args,
                                   sizeof(kernel_args), stream) == hipSuccess;
        }
        /* CDNA fallback: materialize QK^T with hipBLAS, then softmax + PV. */
        if ((host_args.head_dim == 64 || host_args.head_dim == 128) &&
            host_args.sequence >= 128u &&
            h3_launch_sdpa_f32_hipblas(query, key, value, output, &host_args,
                                       stream)) {
            return 1;
        }
        /* Tiled WMMA for the long d64 sequences the video VAE decoder runs.
         * Short sequences stay on the exact wave kernel: they gain nothing from
         * a 128-row query block, and keeping them bit-identical means the bf16
         * operand rounding can only ever touch the decoder. */
        if (host_args.head_dim == 64 && h3_sdpa_f32_wmma_on() &&
            host_args.sequence >= 256u) {
            static uint32_t waves = 0u, bk = 0u;
            if (!waves) {
                /* 16 waves (256-row query block) x 32-key tile measured best
                 * at the video VAE shape; see docs/perf-day9. */
                waves = h3_env_u32("H3_SDPA_F32_WMMA_WAVES", 16u);
                bk = h3_env_u32("H3_SDPA_F32_WMMA_BK", 32u);
            }
            const void *kernel = NULL;
#define H3_SDPA_F32_WMMA_PICK(w, b)                                          \
    if (waves == (w) && bk == (b))                                           \
        kernel = reinterpret_cast<const void *>(                             \
            h3_sdpa_f32_wmma_d64_kernel<w, b>);
            H3_SDPA_F32_WMMA_PICK(4u, 32u)
            H3_SDPA_F32_WMMA_PICK(4u, 64u)
            H3_SDPA_F32_WMMA_PICK(4u, 128u)
            H3_SDPA_F32_WMMA_PICK(8u, 32u)
            H3_SDPA_F32_WMMA_PICK(8u, 64u)
            H3_SDPA_F32_WMMA_PICK(16u, 32u)
#undef H3_SDPA_F32_WMMA_PICK
            if (!kernel) {
                waves = 16u;
                bk = 32u;
                kernel = reinterpret_cast<const void *>(
                    h3_sdpa_f32_wmma_d64_kernel<16u, 32u>);
            }
            uint32_t bq = waves * H3_SDPA_WMMA_QROWS;
            dim3 grid((host_args.sequence + bq - 1u) / bq, host_args.heads, 1);
            if (hipLaunchKernel(kernel, grid, dim3(waves * 32u), kernel_args,
                                sizeof(kernel_args), stream) != hipSuccess) {
                return 0;
            }
            return 1;
        }
        dim3 grid(host_args.sequence, host_args.heads, 1);
        const void *kernel = host_args.head_dim == 128 ?
            reinterpret_cast<const void *>(h3_sdpa_f32_wave_d128_kernel) :
            host_args.head_dim == 64 ?
            reinterpret_cast<const void *>(h3_sdpa_f32_wave_d64_kernel) :
            reinterpret_cast<const void *>(h3_sdpa_f32_wave_kernel);
        if (host_args.head_dim == 64 &&
            !(getenv("H3_SDPA_D64_Q1") &&
              strcmp(getenv("H3_SDPA_D64_Q1"), "0") != 0)) {
            int force_q2 = getenv("H3_SDPA_D64_Q2") &&
                           strcmp(getenv("H3_SDPA_D64_Q2"), "0") != 0;
            int force_q8 = getenv("H3_SDPA_D64_Q8") &&
                           strcmp(getenv("H3_SDPA_D64_Q8"), "0") != 0;
            if (force_q2) {
                grid = dim3((host_args.sequence + 1u) / 2u, host_args.heads, 1);
                kernel = reinterpret_cast<const void *>(
                    h3_sdpa_f32_wave_d64_q2_kernel);
            } else if (force_q8) {
                grid = dim3((host_args.sequence + 7u) / 8u, host_args.heads, 1);
                kernel = reinterpret_cast<const void *>(
                    h3_sdpa_f32_wave_d64_qn_kernel<8>);
            } else {
                grid = dim3((host_args.sequence + 3u) / 4u, host_args.heads, 1);
                kernel = reinterpret_cast<const void *>(
                    h3_sdpa_f32_wave_d64_qn_kernel<4>);
            }
        }
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
