#include "h3_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <math.h>

__device__ __host__ inline float h3_bf16_to_f32(uint16_t value) {
    return __bfloat162float(*reinterpret_cast<__hip_bfloat16 *>(&value));
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
    uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t out_x = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t out_height = args.height + args.height_before + args.height_after;
    uint32_t out_width = args.width + args.width_before + args.width_after;
    uint32_t out_depth = args.depth + args.depth_front;
    uint32_t plane = blockIdx.z;
    if (channel >= args.channels || out_x >= out_width ||
        plane >= args.batch * out_depth * out_height) {
        return;
    }
    uint32_t out_y = plane % out_height;
    uint32_t temporal_plane = plane / out_height;
    uint32_t out_t = temporal_plane % out_depth;
    uint32_t batch = temporal_plane / out_depth;
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
    query[output_index] = h3_f32_to_bf16(q0);
    key[output_index] = h3_f32_to_bf16(k0);
    value[output_index] = qkv[v_base + dimension];
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
    query[output_index] = q0;
    key[output_index] = k0;
    value[output_index] = qkv[v_base + dimension];
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
    query[output_index] = q0;
    key[output_index] = k0;
    value[output_index] = qkv[base + args.head_dim * 2 + dimension];
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
        uint32_t out_index =
            (q_pos * args.heads + head) * args.head_dim + d;
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
    if (args->sequence > 512) return 0;
    h3_sdpa_causal_args host_args = *args;
    void *kernel_args[] = {&query, &key, &value, &output, &host_args};
    dim3 grid(host_args.heads, host_args.sequence, host_args.batch);
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
    uint32_t out_height = host_args.height + host_args.height_before +
                          host_args.height_after;
    uint32_t out_width = host_args.width + host_args.width_before +
                         host_args.width_after;
    uint32_t out_depth = host_args.depth + host_args.depth_front;
    dim3 block(8, 8, 1);
    dim3 grid((host_args.channels + 7) / 8, (out_width + 7) / 8,
              host_args.batch * out_depth * out_height);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_vae_encoder_pad_f32_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
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

extern "C" int h3_launch_sdpa_bf16(const uint16_t *query, const uint16_t *key,
                                   const uint16_t *value, uint16_t *output,
                                   const h3_sdpa_args *args,
                                   hipStream_t stream) {
    if (args->sequence > 512) return 0;
    h3_sdpa_args host_args = *args;
    void *kernel_args[] = {&query, &key, &value, &output, &host_args};
    dim3 grid(host_args.heads, host_args.sequence, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_sdpa_bf16_kernel), grid,
               dim3(1), kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_sdpa_f32(const float *query, const float *key,
                                  const float *value, float *output,
                                  const h3_sdpa_args *args,
                                  hipStream_t stream) {
    if (args->sequence > 512) return 0;
    h3_sdpa_args host_args = *args;
    void *kernel_args[] = {&query, &key, &value, &output, &host_args};
    dim3 grid(host_args.heads, host_args.sequence, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_sdpa_f32_kernel), grid,
               dim3(1), kernel_args, sizeof(kernel_args), stream) == hipSuccess;
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
