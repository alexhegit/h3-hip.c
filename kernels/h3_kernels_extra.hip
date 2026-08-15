#include "h3_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <math.h>
#include <stdlib.h>

__device__ __host__ inline float h3_bf16_to_f32(uint16_t value) {
    return __bfloat162float(*reinterpret_cast<__hip_bfloat16 *>(&value));
}

__device__ __host__ inline uint16_t h3_f32_to_bf16(float value) {
    __hip_bfloat16 bf16 = __float2bfloat16(value);
    return *reinterpret_cast<uint16_t *>(&bf16);
}

__global__ void h3_rms_inverse_bf16_kernel(const uint16_t *input, float *inverse,
                                           h3_norm_args args) {
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
    if (tid == 0) {
        inverse[row] = rsqrtf(reductions[0] / float(args.width) + args.epsilon);
    }
}

__global__ void h3_adaln_linear_bf16_kernel(
    const uint16_t *input, const float *inverse, const uint16_t *norm_weight,
    const uint16_t *modulation, const uint32_t *row_map, const uint16_t *weight,
    const uint16_t *bias, uint16_t *output, h3_adaln_linear_args args) {
    __shared__ uint16_t input_tile[16][16];
    __shared__ float weight_tile[16][16];
    uint32_t row = blockIdx.y * 16 + threadIdx.y;
    uint32_t column = blockIdx.x * 16 + threadIdx.x;
    float sum = args.has_bias && column < args.output_dim ?
        h3_bf16_to_f32(bias[column]) : 0.0f;
    uint32_t base = row < args.rows ? row_map[row] * args.slots * args.width : 0;
    uint32_t tile_count = (args.width + 15) / 16;
    for (uint32_t tile = 0; tile < tile_count; tile++) {
        uint32_t input_k = tile * 16 + threadIdx.x;
        uint16_t normalized = 0;
        if (row < args.rows && input_k < args.width) {
            float value = h3_bf16_to_f32(input[row * args.width + input_k]);
            float shift = h3_bf16_to_f32(
                modulation[base + args.shift_slot * args.width + input_k]);
            float scale = h3_bf16_to_f32(
                modulation[base + args.scale_slot * args.width + input_k]);
            float normed = value * inverse[row] *
                h3_bf16_to_f32(norm_weight[input_k]);
            normalized = h3_f32_to_bf16(normed * (1.0f + scale) + shift);
        }
        input_tile[threadIdx.y][threadIdx.x] = normalized;
        uint32_t weight_k = tile * 16 + threadIdx.y;
        weight_tile[threadIdx.y][threadIdx.x] =
            column < args.output_dim && weight_k < args.width ?
            h3_bf16_to_f32(weight[column * args.width + weight_k]) : 0.0f;
        __syncthreads();
        for (uint32_t k = 0; k < 16; k++) {
            sum = fmaf(h3_bf16_to_f32(input_tile[threadIdx.y][k]),
                       weight_tile[k][threadIdx.x], sum);
        }
        __syncthreads();
    }
    if (row < args.rows && column < args.output_dim) {
        output[row * args.output_dim + column] = h3_f32_to_bf16(sum);
    }
}

__global__ void h3_token_pool_bf16_kernel(const uint16_t *input,
                                          const uint32_t *pairs,
                                          uint16_t *output, uint16_t *baseline,
                                          const uint32_t *baseline_indices,
                                          uint16_t *original,
                                          h3_token_pool_args args) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || column >= args.width) return;
    uint32_t pair_x = pairs[row * 2];
    uint32_t pair_y = pairs[row * 2 + 1];
    uint16_t first = input[args.input_offset + pair_x * args.width + column];
    original[args.original_offset + pair_x * args.width + column] = first;
    uint16_t pooled = first;
    if (pair_x != pair_y) {
        uint16_t second =
            input[args.input_offset + pair_y * args.width + column];
        original[args.original_offset + pair_y * args.width + column] = second;
        pooled = h3_f32_to_bf16((h3_bf16_to_f32(first) +
                                 h3_bf16_to_f32(second)) * 0.5f);
    }
    output[row * args.width + column] = pooled;
    uint32_t baseline_index = baseline_indices[row];
    if (baseline_index != 0xffffffffu) {
        baseline[args.baseline_offset + baseline_index * args.width + column] =
            pooled;
    }
}

__global__ void h3_token_pool_adaln_bf16_kernel(
    const uint16_t *input, const uint32_t *pairs, uint16_t *residual,
    uint16_t *baseline, const uint32_t *baseline_indices, uint16_t *original,
    const uint16_t *weight, const uint16_t *modulation,
    const uint32_t *row_map, uint16_t *output, h3_token_pool_adaln_args args) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    __shared__ uint16_t pooled_values[5376];
    uint32_t pair_x = pairs[row * 2];
    uint32_t pair_y = pairs[row * 2 + 1];
    uint32_t baseline_index = baseline_indices[row];
    float local_sum = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        uint16_t first = input[args.input_offset + pair_x * args.width + column];
        original[args.original_offset + pair_x * args.width + column] = first;
        uint16_t pooled = first;
        if (pair_x != pair_y) {
            uint16_t second =
                input[args.input_offset + pair_y * args.width + column];
            original[args.original_offset + pair_y * args.width + column] =
                second;
            pooled = h3_f32_to_bf16((h3_bf16_to_f32(first) +
                                     h3_bf16_to_f32(second)) * 0.5f);
        }
        uint32_t destination = row * args.width + column;
        residual[destination] = pooled;
        pooled_values[column] = pooled;
        if (baseline_index != 0xffffffffu) {
            baseline[args.baseline_offset + baseline_index * args.width +
                     column] = pooled;
        }
        float value = h3_bf16_to_f32(pooled);
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
        float normalized = h3_bf16_to_f32(pooled_values[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

__global__ void h3_token_expand_delta_bf16_kernel(
    const uint16_t *original, const uint16_t *reduced, const uint16_t *baseline,
    const uint32_t *baseline_indices, const uint32_t *parents,
    uint16_t *output, h3_token_expand_args args) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || column >= args.width) return;
    uint32_t parent = parents[row];
    uint32_t destination = row * args.width + column;
    uint32_t reduced_index = parent * args.width + column;
    if (row < args.exact_prefix_rows) {
        output[destination] = reduced[reduced_index];
    } else {
        uint32_t baseline_row = baseline_indices[parent];
        if (baseline_row == 0xffffffffu) {
            output[destination] = reduced[reduced_index];
            return;
        }
        uint32_t baseline_index =
            args.baseline_offset + baseline_row * args.width + column;
        float update = h3_bf16_to_f32(reduced[reduced_index]) -
                       h3_bf16_to_f32(baseline[baseline_index]);
        output[destination] = h3_f32_to_bf16(
            h3_bf16_to_f32(original[args.original_offset + destination]) +
            args.update_scale * update);
    }
}

__global__ void h3_token_expand_adaln_bf16_kernel(
    const uint16_t *original, const uint16_t *reduced, const uint16_t *baseline,
    const uint32_t *baseline_indices, const uint32_t *parents,
    uint16_t *residual, const uint16_t *weight, const uint16_t *modulation,
    const uint32_t *row_map, uint16_t *output, h3_token_expand_adaln_args args) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    if (row >= args.rows) return;
    __shared__ float reductions[256];
    __shared__ uint16_t restored_values[5376];
    uint32_t parent = parents[row];
    uint32_t baseline_row = baseline_indices[parent];
    bool direct = row < args.exact_prefix_rows || baseline_row == 0xffffffffu;
    float local_sum = 0.0f;
    for (uint32_t column = tid; column < args.width; column += threads) {
        uint32_t destination = row * args.width + column;
        uint32_t reduced_index = parent * args.width + column;
        uint16_t restored = reduced[reduced_index];
        if (!direct) {
            uint32_t baseline_index =
                args.baseline_offset + baseline_row * args.width + column;
            float update = h3_bf16_to_f32(restored) -
                           h3_bf16_to_f32(baseline[baseline_index]);
            restored = h3_f32_to_bf16(
                h3_bf16_to_f32(original[args.original_offset + destination]) +
                args.update_scale * update);
        }
        restored_values[column] = restored;
        residual[destination] = restored;
        float value = h3_bf16_to_f32(restored);
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
        float normalized = h3_bf16_to_f32(restored_values[column]) * inverse *
            h3_bf16_to_f32(weight[column]);
        float shift = h3_bf16_to_f32(
            modulation[base + args.shift_slot * args.width + column]);
        float scale = h3_bf16_to_f32(
            modulation[base + args.scale_slot * args.width + column]);
        output[row * args.width + column] =
            h3_f32_to_bf16(normalized * (1.0f + scale) + shift);
    }
}

__global__ void h3_embedding_bf16_kernel(const uint16_t *weight,
                                         const uint32_t *token_ids,
                                         uint16_t *output,
                                         h3_embedding_args args) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t token = blockIdx.y * blockDim.y + threadIdx.y;
    if (token >= args.tokens || column >= args.width) return;
    uint32_t identifier = token_ids[token];
    output[token * args.width + column] = identifier < args.vocab_size ?
        weight[identifier * args.width + column] : uint16_t(0);
}

__global__ void h3_silu_mul_bf16_kernel(const uint16_t *gate,
                                        const uint16_t *up, uint16_t *output,
                                        uint32_t count) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float value = h3_bf16_to_f32(gate[gid]);
    float other = h3_bf16_to_f32(up[gid]);
    output[gid] = h3_f32_to_bf16(value / (1.0f + expf(-value)) * other);
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

extern "C" int h3_launch_rms_inverse_bf16(const uint16_t *input, float *inverse,
                                          const h3_norm_args *args,
                                          hipStream_t stream) {
    h3_norm_args host_args = *args;
    void *kernel_args[] = {&input, &inverse, &host_args};
    uint32_t threads = h3_row_threads(host_args.width);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_rms_inverse_bf16_kernel),
               dim3(host_args.rows), dim3(threads), kernel_args,
               sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_adaln_linear_bf16(
    const uint16_t *input, const float *inverse, const uint16_t *norm_weight,
    const uint16_t *modulation, const uint32_t *row_map, const uint16_t *weight,
    const uint16_t *bias, uint16_t *output, const h3_adaln_linear_args *args,
    hipStream_t stream) {
    h3_adaln_linear_args host_args = *args;
    void *kernel_args[] = {&input, &inverse, &norm_weight, &modulation,
                           &row_map, &weight, &bias, &output, &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.output_dim + 15) / 16, (host_args.rows + 15) / 16, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_adaln_linear_bf16_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}

extern "C" int h3_launch_token_pool_bf16(
    const uint16_t *input, const uint32_t *pairs, uint16_t *output,
    uint16_t *baseline, const uint32_t *baseline_indices, uint16_t *original,
    const h3_token_pool_args *args, hipStream_t stream) {
    h3_token_pool_args host_args = *args;
    void *kernel_args[] = {&input, &pairs, &output, &baseline,
                           &baseline_indices, &original, &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.width + 15) / 16, (host_args.rows + 15) / 16, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_token_pool_bf16_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}

extern "C" int h3_launch_token_pool_adaln_bf16(
    const uint16_t *input, const uint32_t *pairs, uint16_t *residual,
    uint16_t *baseline, const uint32_t *baseline_indices, uint16_t *original,
    const uint16_t *weight, const uint16_t *modulation,
    const uint32_t *row_map, uint16_t *output,
    const h3_token_pool_adaln_args *args, hipStream_t stream) {
    h3_token_pool_adaln_args host_args = *args;
    void *kernel_args[] = {&input, &pairs, &residual, &baseline,
                           &baseline_indices, &original, &weight, &modulation,
                           &row_map, &output, &host_args};
    uint32_t threads = h3_row_threads(host_args.width);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_token_pool_adaln_bf16_kernel),
               dim3(host_args.rows), dim3(threads), kernel_args,
               sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_token_expand_delta_bf16(
    const uint16_t *original, const uint16_t *reduced, const uint16_t *baseline,
    const uint32_t *baseline_indices, const uint32_t *parents,
    uint16_t *output, const h3_token_expand_args *args, hipStream_t stream) {
    h3_token_expand_args host_args = *args;
    void *kernel_args[] = {&original, &reduced, &baseline, &baseline_indices,
                           &parents, &output, &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.width + 15) / 16, (host_args.rows + 15) / 16, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_token_expand_delta_bf16_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}

extern "C" int h3_launch_token_expand_adaln_bf16(
    const uint16_t *original, const uint16_t *reduced, const uint16_t *baseline,
    const uint32_t *baseline_indices, const uint32_t *parents,
    uint16_t *residual, const uint16_t *weight, const uint16_t *modulation,
    const uint32_t *row_map, uint16_t *output,
    const h3_token_expand_adaln_args *args, hipStream_t stream) {
    h3_token_expand_adaln_args host_args = *args;
    void *kernel_args[] = {&original, &reduced, &baseline, &baseline_indices,
                           &parents, &residual, &weight, &modulation,
                           &row_map, &output, &host_args};
    uint32_t threads = h3_row_threads(host_args.width);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_token_expand_adaln_bf16_kernel),
               dim3(host_args.rows), dim3(threads), kernel_args,
               sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_embedding_bf16(const uint16_t *weight,
                                        const uint32_t *token_ids,
                                        uint16_t *output,
                                        const h3_embedding_args *args,
                                        hipStream_t stream) {
    h3_embedding_args host_args = *args;
    void *kernel_args[] = {&weight, &token_ids, &output, &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.width + 15) / 16, (host_args.tokens + 15) / 16, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_embedding_bf16_kernel), grid,
               block, kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_silu_mul_bf16(const uint16_t *gate,
                                       const uint16_t *up, uint16_t *output,
                                       uint32_t count, hipStream_t stream) {
    void *kernel_args[] = {&gate, &up, &output, &count};
    uint32_t block = 256;
    uint32_t grid = (count + block - 1u) / block;
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_silu_mul_bf16_kernel),
               dim3(grid), dim3(block), kernel_args, sizeof(kernel_args),
               stream) ==            hipSuccess;
}

__global__ void h3_linear_f32_kernel(const float *input, const float *weight,
                                     const float *bias, float *output,
                                     h3_linear_args args) {
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.rows || column >= args.output_dim) return;
    float sum = args.has_bias ? bias[column] : 0.0f;
    const float *x = input + row * args.input_dim;
    const float *w = weight + column * args.input_dim;
    for (uint32_t k = 0; k < args.input_dim; k++) {
        sum = fmaf(x[k], w[k], sum);
    }
    output[row * args.output_dim + column] = sum;
}

__global__ void h3_conv1d_f32_kernel(const float *input, const float *weight,
                                     const float *bias, float *output,
                                     h3_conv1d_args args) {
    uint32_t out = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t time = blockIdx.y;
    uint32_t batch = blockIdx.z;
    if (out >= args.output_channels || time >= args.output_length ||
        batch >= args.batch) {
        return;
    }
    float sum = args.has_bias ? bias[out] : 0.0f;
    uint32_t input_batch = batch * args.input_length * args.input_channels;
    uint32_t output_batch = batch * args.output_length * args.output_channels;
    for (uint32_t k = 0; k < args.kernel; k++) {
        int32_t source = (int32_t)time * (int32_t)args.stride +
                         (int32_t)(k * args.dilation) - (int32_t)args.padding;
        if (source < 0 || source >= (int32_t)args.input_length) continue;
        for (uint32_t in = 0; in < args.input_channels; in++) {
            sum = fmaf(input[input_batch + (uint32_t)source * args.input_channels +
                               in],
                       weight[(out * args.input_channels + in) * args.kernel + k],
                       sum);
        }
    }
    output[output_batch + time * args.output_channels + out] = sum;
}

__global__ void h3_conv_transpose1d_f32_kernel(const float *input,
                                               const float *weight,
                                               const float *bias, float *output,
                                               h3_conv1d_args args) {
    uint32_t out = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t time = blockIdx.y;
    uint32_t batch = blockIdx.z;
    if (out >= args.output_channels || time >= args.output_length ||
        batch >= args.batch) {
        return;
    }
    float sum = args.has_bias ? bias[out] : 0.0f;
    uint32_t input_batch = batch * args.input_length * args.input_channels;
    uint32_t output_batch = batch * args.output_length * args.output_channels;
    for (uint32_t k = 0; k < args.kernel; k++) {
        int32_t numerator = (int32_t)time + (int32_t)args.padding -
                            (int32_t)k;
        if (numerator < 0 || (uint32_t)numerator % args.stride) continue;
        int32_t source = numerator / (int32_t)args.stride;
        if (source < 0 || source >= (int32_t)args.input_length) continue;
        for (uint32_t in = 0; in < args.input_channels; in++) {
            sum = fmaf(input[input_batch +
                               (uint32_t)source * args.input_channels + in],
                       weight[(in * args.output_channels + out) * args.kernel +
                              k],
                       sum);
        }
    }
    output[output_batch + time * args.output_channels + out] = sum;
}

__global__ void h3_conv3d_f32_kernel(const float *input, const float *weight,
                                     const float *bias, float *output,
                                     h3_conv3d_args args) {
    uint32_t out = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t spatial = blockIdx.y;
    uint32_t batch = blockIdx.z;
    uint32_t spatial_count =
        args.output_depth * args.output_height * args.output_width;
    if (out >= args.output_channels || spatial >= spatial_count ||
        batch >= args.batch) {
        return;
    }
    uint32_t out_x = spatial % args.output_width;
    uint32_t plane = spatial / args.output_width;
    uint32_t out_y = plane % args.output_height;
    uint32_t out_t = plane / args.output_height;
    float sum = args.has_bias ? bias[out] : 0.0f;
    uint32_t input_batch = batch * args.input_depth * args.input_height *
                           args.input_width * args.input_channels;
    uint32_t output_batch = batch * spatial_count * args.output_channels;
    for (uint32_t kd = 0; kd < args.kernel_depth; kd++) {
        uint32_t source_t = out_t * args.stride_depth + kd;
        for (uint32_t kh = 0; kh < args.kernel_height; kh++) {
            uint32_t source_y = out_y * args.stride_height + kh;
            for (uint32_t kw = 0; kw < args.kernel_width; kw++) {
                uint32_t source_x = out_x * args.stride_width + kw;
                for (uint32_t in = 0; in < args.input_channels; in++) {
                    size_t input_index = input_batch +
                        (((size_t)source_t * args.input_height + source_y) *
                             args.input_width + source_x) *
                            args.input_channels + in;
                    size_t weight_index =
                        (((((size_t)out * args.input_channels + in) *
                               args.kernel_depth + kd) *
                              args.kernel_height + kh) *
                             args.kernel_width + kw);
                    sum = fmaf(input[input_index], weight[weight_index], sum);
                }
            }
        }
    }
    output[output_batch + spatial * args.output_channels + out] = sum;
}

extern "C" int h3_launch_conv1d_f32(const float *input, const float *weight,
                                    const float *bias, float *output,
                                    const h3_conv1d_args *args,
                                    hipStream_t stream) {
    h3_conv1d_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &bias, &output, &host_args};
    dim3 block(16, 1, 1);
    dim3 grid((host_args.output_channels + 15) / 16, host_args.output_length,
              host_args.batch);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_conv1d_f32_kernel), grid,
               block, kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_conv_transpose1d_f32(const float *input,
                                              const float *weight,
                                              const float *bias, float *output,
                                              const h3_conv1d_args *args,
                                              hipStream_t stream) {
    h3_conv1d_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &bias, &output, &host_args};
    dim3 block(16, 1, 1);
    dim3 grid((host_args.output_channels + 15) / 16, host_args.output_length,
              host_args.batch);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_conv_transpose1d_f32_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}

extern "C" int h3_launch_conv3d_f32(const float *input, const float *weight,
                                    const float *bias, float *output,
                                    const h3_conv3d_args *args,
                                    hipStream_t stream) {
    h3_conv3d_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &bias, &output, &host_args};
    uint32_t spatial_count = host_args.output_depth * host_args.output_height *
                             host_args.output_width;
    dim3 block(16, 1, 1);
    dim3 grid((host_args.output_channels + 15) / 16, spatial_count,
              host_args.batch);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_conv3d_f32_kernel), grid,
               block, kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_linear_f32(const float *input, const float *weight,
                                    const float *bias, float *output,
                                    const h3_linear_args *args,
                                    hipStream_t stream) {
    h3_linear_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &bias, &output, &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.output_dim + 15) / 16, (host_args.rows + 15) / 16, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_linear_f32_kernel), grid,
               block, kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

__global__ void h3_linear_f32_tiled_bf16_kernel(
    const float *input, const float *weight, const float *bias,
    uint16_t *output, h3_linear_args args) {
    __shared__ float input_tile[16][16];
    __shared__ float weight_tile[16][16];
    uint32_t row = blockIdx.y * 16 + threadIdx.y;
    uint32_t column = blockIdx.x * 16 + threadIdx.x;
    float sum = args.has_bias && column < args.output_dim ?
        bias[column] : 0.0f;
    uint32_t tile_count = (args.input_dim + 15) / 16;
    for (uint32_t tile = 0; tile < tile_count; tile++) {
        uint32_t input_k = tile * 16 + threadIdx.x;
        input_tile[threadIdx.y][threadIdx.x] =
            row < args.rows && input_k < args.input_dim ?
            input[row * args.input_dim + input_k] : 0.0f;
        uint32_t weight_k = tile * 16 + threadIdx.y;
        weight_tile[threadIdx.y][threadIdx.x] =
            column < args.output_dim && weight_k < args.input_dim ?
            weight[column * args.input_dim + weight_k] : 0.0f;
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

extern "C" int h3_launch_linear_f32_tiled_bf16(
    const float *input, const float *weight, const float *bias,
    uint16_t *output, const h3_linear_args *args, hipStream_t stream) {
    h3_linear_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &bias, &output, &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.output_dim + 15) / 16, (host_args.rows + 15) / 16, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_linear_f32_tiled_bf16_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}

__global__ void h3_linear_f32_tiled_bf16_map_kernel(
    const float *input, const float *weight, const float *bias,
    uint16_t *output, const uint32_t *row_map, h3_linear_args args) {
    __shared__ float input_tile[16][16];
    __shared__ float weight_tile[16][16];
    __shared__ uint32_t output_rows[16];
    uint32_t row = blockIdx.y * 16 + threadIdx.y;
    uint32_t column = blockIdx.x * 16 + threadIdx.x;
    if (threadIdx.x == 0) {
        output_rows[threadIdx.y] = row < args.rows ? row_map[row] : 0;
    }
    __syncthreads();
    float sum = args.has_bias && column < args.output_dim ?
        bias[column] : 0.0f;
    uint32_t tile_count = (args.input_dim + 15) / 16;
    for (uint32_t tile = 0; tile < tile_count; tile++) {
        uint32_t input_k = tile * 16 + threadIdx.x;
        input_tile[threadIdx.y][threadIdx.x] =
            row < args.rows && input_k < args.input_dim ?
            input[row * args.input_dim + input_k] : 0.0f;
        uint32_t weight_k = tile * 16 + threadIdx.y;
        weight_tile[threadIdx.y][threadIdx.x] =
            column < args.output_dim && weight_k < args.input_dim ?
            weight[column * args.input_dim + weight_k] : 0.0f;
        __syncthreads();
        for (uint32_t k = 0; k < 16; k++) {
            sum = fmaf(input_tile[threadIdx.y][k], weight_tile[k][threadIdx.x],
                       sum);
        }
        __syncthreads();
    }
    if (row < args.rows && column < args.output_dim) {
        output[output_rows[threadIdx.y] * args.output_dim + column] =
            h3_f32_to_bf16(sum);
    }
}

extern "C" int h3_launch_linear_f32_tiled_bf16_map(
    const float *input, const float *weight, const float *bias,
    uint16_t *output, const uint32_t *row_map, const h3_linear_args *args,
    hipStream_t stream) {
    h3_linear_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &bias, &output, &row_map,
                           &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.output_dim + 15) / 16, (host_args.rows + 15) / 16, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(
                   h3_linear_f32_tiled_bf16_map_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}

__global__ void h3_text_qk_rope_bf16_kernel(
    const uint16_t *query_input, const uint16_t *key_input,
    const uint16_t *q_weight, const uint16_t *k_weight,
    const uint16_t *rope_cos, const uint16_t *rope_sin,
    uint16_t *query_output, uint16_t *key_output, h3_text_rope_args args) {
    uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t row = blockIdx.z * blockDim.z + threadIdx.z;
    if (dimension >= args.head_dim || head >= args.query_heads ||
        row >= args.sequence) {
        return;
    }
    uint32_t half_dim = args.head_dim / 2;
    uint32_t pair = dimension < half_dim ? dimension + half_dim :
                                          dimension - half_dim;
    float c = h3_bf16_to_f32(rope_cos[row * half_dim + (dimension % half_dim)]);
    float s = h3_bf16_to_f32(rope_sin[row * half_dim + (dimension % half_dim)]);
    uint32_t q_base = (row * args.query_heads + head) * args.head_dim;
    float q_sum = 0.0f;
    for (uint32_t d = 0; d < args.head_dim; d++) {
        float value = h3_bf16_to_f32(query_input[q_base + d]);
        q_sum = fmaf(value, value, q_sum);
    }
    float q_inverse = rsqrtf(q_sum / float(args.head_dim) + args.epsilon);
    float q0 = h3_bf16_to_f32(query_input[q_base + dimension]) * q_inverse *
               h3_bf16_to_f32(q_weight[dimension]);
    float q1 = h3_bf16_to_f32(query_input[q_base + pair]) * q_inverse *
               h3_bf16_to_f32(q_weight[pair]);
    float q_rotated = dimension < half_dim ? q0 * c - q1 * s : q0 * c + q1 * s;
    query_output[q_base + dimension] = h3_f32_to_bf16(q_rotated);
    if (head < args.kv_heads) {
        uint32_t k_base = (row * args.kv_heads + head) * args.head_dim;
        float k_sum = 0.0f;
        for (uint32_t d = 0; d < args.head_dim; d++) {
            float value = h3_bf16_to_f32(key_input[k_base + d]);
            k_sum = fmaf(value, value, k_sum);
        }
        float k_inverse = rsqrtf(k_sum / float(args.head_dim) + args.epsilon);
        float k0 = h3_bf16_to_f32(key_input[k_base + dimension]) * k_inverse *
                   h3_bf16_to_f32(k_weight[dimension]);
        float k1 = h3_bf16_to_f32(key_input[k_base + pair]) * k_inverse *
                   h3_bf16_to_f32(k_weight[pair]);
        float k_rotated = dimension < half_dim ? k0 * c - k1 * s : k0 * c + k1 * s;
        key_output[k_base + dimension] = h3_f32_to_bf16(k_rotated);
    }
}

__global__ void h3_head_rms_norm_bf16_kernel(uint16_t *tensor,
                                             const uint16_t *weight,
                                             h3_head_norm_args args) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.sequence || head >= args.heads) return;
    uint32_t base = (row * args.heads + head) * args.head_dim;
    float sum = 0.0f;
    for (uint32_t d = 0; d < args.head_dim; d++) {
        float value = h3_bf16_to_f32(tensor[base + d]);
        sum = fmaf(value, value, sum);
    }
    float inverse = rsqrtf(sum / float(args.head_dim) + args.epsilon);
    for (uint32_t d = 0; d < args.head_dim; d++) {
        float value = h3_bf16_to_f32(tensor[base + d]);
        tensor[base + d] = h3_f32_to_bf16(
            value * inverse * h3_bf16_to_f32(weight[d]));
    }
}

__global__ void h3_rope_text_bf16_kernel(uint16_t *query, uint16_t *key,
                                         const float *rope_cos,
                                         const float *rope_sin,
                                         h3_text_rope_inplace_args args) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.sequence) return;
    uint32_t half_dim = args.head_dim / 2;
    if (head < args.query_heads) {
        uint32_t base = (row * args.query_heads + head) * args.head_dim;
        for (uint32_t d = 0; d < half_dim; d++) {
            float first = h3_bf16_to_f32(query[base + d]);
            float second = h3_bf16_to_f32(query[base + half_dim + d]);
            float c = rope_cos[row * half_dim + d];
            float s = rope_sin[row * half_dim + d];
            query[base + d] = h3_f32_to_bf16(first * c - second * s);
            query[base + half_dim + d] = h3_f32_to_bf16(second * c + first * s);
        }
    }
    if (head < args.kv_heads) {
        uint32_t base = (row * args.kv_heads + head) * args.head_dim;
        for (uint32_t d = 0; d < half_dim; d++) {
            float first = h3_bf16_to_f32(key[base + d]);
            float second = h3_bf16_to_f32(key[base + half_dim + d]);
            float c = rope_cos[row * half_dim + d];
            float s = rope_sin[row * half_dim + d];
            key[base + d] = h3_f32_to_bf16(first * c - second * s);
            key[base + half_dim + d] = h3_f32_to_bf16(second * c + first * s);
        }
    }
}

__global__ void h3_gqa_causal_bf16_kernel(const uint16_t *query,
                                          const uint16_t *key,
                                          const uint16_t *value,
                                          uint16_t *output, h3_gqa_args args) {
    __shared__ float reductions[128];
    __shared__ float shared_query[128];
    __shared__ float scores[512];
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    uint32_t query_row = blockIdx.x;
    uint32_t query_head = blockIdx.y;
    if (query_row >= args.sequence || query_head >= args.query_heads) return;
    uint32_t kv_head = query_head / (args.query_heads / args.kv_heads);
    uint32_t q_base = (query_row * args.query_heads + query_head) * args.head_dim;
    uint32_t key_count = query_row + 1;
    for (uint32_t d = tid; d < args.head_dim; d += threads) {
        shared_query[d] = h3_bf16_to_f32(h3_f32_to_bf16(
            h3_bf16_to_f32(query[q_base + d]) * args.scale));
    }
    __syncthreads();
    float local_max = -1e30f;
    for (uint32_t key_row = tid; key_row < key_count; key_row += threads) {
        uint32_t k_base = (key_row * args.kv_heads + kv_head) * args.head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < args.head_dim; d++) {
            dot = fmaf(shared_query[d], h3_bf16_to_f32(key[k_base + d]), dot);
        }
        scores[key_row] = dot;
        local_max = fmaxf(local_max, dot);
    }
    reductions[tid] = local_max;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] = fmaxf(reductions[tid], reductions[tid + stride]);
        __syncthreads();
    }
    float maximum = reductions[0];
    float local_sum = 0.0f;
    for (uint32_t key_row = tid; key_row < key_count; key_row += threads) {
        float probability = expf(scores[key_row] - maximum);
        scores[key_row] = probability;
        local_sum += probability;
    }
    reductions[tid] = local_sum;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) reductions[tid] += reductions[tid + stride];
        __syncthreads();
    }
    float inverse_sum = 1.0f / reductions[0];
    for (uint32_t d = tid; d < args.head_dim; d += threads) {
        float sum = 0.0f;
        for (uint32_t key_row = 0; key_row < key_count; key_row++) {
            uint32_t v_index =
                (key_row * args.kv_heads + kv_head) * args.head_dim + d;
            sum = fmaf(scores[key_row] * inverse_sum,
                       h3_bf16_to_f32(value[v_index]), sum);
        }
        output[q_base + d] = h3_f32_to_bf16(sum);
    }
}

__device__ inline float h3_gqa_warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        value += __shfl_xor(value, mask);
    return value;
}

__global__ void __launch_bounds__(32)
h3_gqa_causal_bf16_wave_kernel(const uint16_t *query, const uint16_t *key,
                               const uint16_t *value, uint16_t *output,
                               h3_gqa_args args) {
    uint32_t query_row = blockIdx.x;
    uint32_t query_head = blockIdx.y;
    uint32_t lane = threadIdx.x;
    uint32_t kv_head = query_head / (args.query_heads / args.kv_heads);
    uint32_t q_base =
        (query_row * args.query_heads + query_head) * args.head_dim;
    float q[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int item = 0; item < 4; item++) {
        uint32_t dimension = lane + (uint32_t)item * 32u;
        if (dimension < args.head_dim) {
            q[item] = h3_bf16_to_f32(h3_f32_to_bf16(
                h3_bf16_to_f32(query[q_base + dimension]) * args.scale));
        }
    }
    float maximum = -1e30f;
    float sum = 0.0f;
    uint32_t key_count = query_row + 1;
    for (uint32_t key_row = 0; key_row < key_count; key_row++) {
        uint32_t k_base = (key_row * args.kv_heads + kv_head) * args.head_dim;
        float partial = 0.0f;
#pragma unroll
        for (int item = 0; item < 4; item++) {
            uint32_t dimension = lane + (uint32_t)item * 32u;
            if (dimension < args.head_dim)
                partial = fmaf(q[item], h3_bf16_to_f32(key[k_base + dimension]),
                               partial);
        }
        float score = h3_gqa_warp_sum(partial);
        float new_max = fmaxf(maximum, score);
        float alpha = expf(maximum - new_max);
        float probability = expf(score - new_max);
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
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
#pragma unroll
    for (int item = 0; item < 4; item++) {
        uint32_t dimension = lane + (uint32_t)item * 32u;
        if (dimension < args.head_dim)
            output[q_base + dimension] = h3_f32_to_bf16(acc[item] * inv);
    }
}

extern "C" int h3_launch_text_qk_rope_bf16(
    const uint16_t *query_input, const uint16_t *key_input,
    const uint16_t *q_weight, const uint16_t *k_weight,
    const uint16_t *rope_cos, const uint16_t *rope_sin,
    uint16_t *query_output, uint16_t *key_output,
    const h3_text_rope_args *args, hipStream_t stream) {
    h3_text_rope_args host_args = *args;
    void *kernel_args[] = {&query_input, &key_input, &q_weight, &k_weight,
                           &rope_cos, &rope_sin, &query_output, &key_output,
                           &host_args};
    dim3 block(16, 4, 1);
    dim3 grid((host_args.head_dim + 15) / 16, host_args.query_heads,
              host_args.sequence);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_text_qk_rope_bf16_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}

extern "C" int h3_launch_head_rms_norm_bf16(uint16_t *tensor,
                                            const uint16_t *weight,
                                            const h3_head_norm_args *args,
                                            hipStream_t stream) {
    h3_head_norm_args host_args = *args;
    void *kernel_args[] = {&tensor, &weight, &host_args};
    dim3 block(1, 1, 1);
    dim3 grid(host_args.sequence, host_args.heads, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_head_rms_norm_bf16_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}

extern "C" int h3_launch_rope_text_bf16(uint16_t *query, uint16_t *key,
                                        const float *rope_cos,
                                        const float *rope_sin,
                                        const h3_text_rope_inplace_args *args,
                                        hipStream_t stream) {
    h3_text_rope_inplace_args host_args = *args;
    void *kernel_args[] = {&query, &key, &rope_cos, &rope_sin, &host_args};
    uint32_t maximum_heads = host_args.query_heads > host_args.kv_heads ?
        host_args.query_heads : host_args.kv_heads;
    dim3 block(1, 1, 1);
    dim3 grid(host_args.sequence, maximum_heads, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_rope_text_bf16_kernel), grid,
               block, kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

extern "C" int h3_launch_gqa_causal_bf16(const uint16_t *query,
                                         const uint16_t *key,
                                         const uint16_t *value,
                                         uint16_t *output,
                                         const h3_gqa_args *args,
                                         hipStream_t stream) {
    if (!args->sequence || !args->query_heads || !args->kv_heads ||
        !args->head_dim || args->head_dim > 128 ||
        args->query_heads % args->kv_heads != 0) {
        return 0;
    }
    h3_gqa_args host_args = *args;
    void *kernel_args[] = {&query, &key, &value, &output, &host_args};
    dim3 grid(host_args.sequence, host_args.query_heads, 1);
    if (!getenv("H3_GQA_LEGACY")) {
        return hipLaunchKernel(
                   reinterpret_cast<const void *>(
                       h3_gqa_causal_bf16_wave_kernel),
                   grid, dim3(32), kernel_args, sizeof(kernel_args),
                   stream) == hipSuccess;
    }
    if (host_args.sequence > 512) return 0;
    uint32_t threads = 1;
    while (threads * 2u <= 128u && threads < host_args.sequence) threads *= 2u;
    dim3 block(threads, 1, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_gqa_causal_bf16_kernel), grid,
               block, kernel_args, sizeof(kernel_args), stream) == hipSuccess;
}

__global__ void h3_quantize_bf16_int8_rows_kernel(
    const uint16_t *input, int8_t *output, float *scales,
    h3_int8_quant_args args) {
    __shared__ float reductions[256];
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    uint32_t base = row * args.columns;
    if (row >= args.rows) {
        for (uint32_t column = tid; column < args.columns; column += threads)
            output[base + column] = 0;
        if (tid == 0) scales[row] = 1.0f;
        return;
    }
    float local_max = 0.0f;
    for (uint32_t column = tid; column < args.columns; column += threads) {
        local_max = fmaxf(local_max,
                          fabsf(h3_bf16_to_f32(input[base + column])));
    }
    reductions[tid] = local_max;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) {
            reductions[tid] = fmaxf(reductions[tid], reductions[tid + stride]);
        }
        __syncthreads();
    }
    float clipped_max = reductions[0] * args.clip;
    float scale = clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
    float inverse = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
    if (tid == 0) scales[row] = scale;
    for (uint32_t column = tid; column < args.columns; column += threads) {
        int quantized =
            (int)rintf(h3_bf16_to_f32(input[base + column]) * inverse);
        if (quantized > 127) quantized = 127;
        if (quantized < -127) quantized = -127;
        output[base + column] = (int8_t)quantized;
    }
}

extern "C" int h3_launch_quantize_bf16_int8_rows(
    const uint16_t *input, int8_t *output, float *scales,
    const h3_int8_quant_args *args, uint32_t dispatch_rows,
    hipStream_t stream) {
    h3_int8_quant_args host_args = *args;
    void *kernel_args[] = {&input, &output, &scales, &host_args};
    dim3 block(256, 1, 1);
    dim3 grid(dispatch_rows, 1, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_quantize_bf16_int8_rows_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}

__global__ void h3_quantize_bf16_int8_head_major_rows_kernel(
    const uint16_t *input, int8_t *output, float *scales,
    h3_int8_head_major_quant_args args) {
    __shared__ float reductions[256];
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    uint32_t input_dim = args.heads * args.head_dim;
    uint32_t out_base = row * input_dim;
    if (row >= args.rows) {
        for (uint32_t column = tid; column < input_dim; column += threads)
            output[out_base + column] = 0;
        if (tid == 0) scales[row] = 1.0f;
        return;
    }
    float local_max = 0.0f;
    for (uint32_t column = tid; column < input_dim; column += threads) {
        uint32_t head = column / args.head_dim;
        uint32_t dim = column % args.head_dim;
        uint32_t hm_index = (head * args.rows + row) * args.head_dim + dim;
        local_max = fmaxf(local_max,
                          fabsf(h3_bf16_to_f32(input[hm_index])));
    }
    reductions[tid] = local_max;
    __syncthreads();
    for (uint32_t stride = threads / 2; stride; stride >>= 1) {
        if (tid < stride) {
            reductions[tid] = fmaxf(reductions[tid], reductions[tid + stride]);
        }
        __syncthreads();
    }
    float clipped_max = reductions[0] * args.clip;
    float scale = clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
    float inverse = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
    if (tid == 0) scales[row] = scale;
    for (uint32_t column = tid; column < input_dim; column += threads) {
        uint32_t head = column / args.head_dim;
        uint32_t dim = column % args.head_dim;
        uint32_t hm_index = (head * args.rows + row) * args.head_dim + dim;
        int quantized =
            (int)rintf(h3_bf16_to_f32(input[hm_index]) * inverse);
        if (quantized > 127) quantized = 127;
        if (quantized < -127) quantized = -127;
        output[out_base + column] = (int8_t)quantized;
    }
}

extern "C" int h3_launch_quantize_bf16_int8_head_major_rows(
    const uint16_t *input, int8_t *output, float *scales,
    const h3_int8_head_major_quant_args *args, hipStream_t stream) {
    h3_int8_head_major_quant_args host_args = *args;
    void *kernel_args[] = {&input, &output, &scales, &host_args};
    dim3 block(256, 1, 1);
    dim3 grid(host_args.padded_rows, 1, 1);
    return hipLaunchKernel(reinterpret_cast<const void *>(
                               h3_quantize_bf16_int8_head_major_rows_kernel),
                           grid, block, kernel_args, sizeof(kernel_args),
                           stream) == hipSuccess;
}

__global__ void h3_quantize_bf16_int8_groups_scalar_kernel(
    const uint16_t *input, int8_t *output, float *scales,
    h3_int8_group_quant_args args) {
    __shared__ float reductions[256];
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    uint32_t row_base = row * args.columns;
    if (row >= args.rows) {
        for (uint32_t column = tid; column < args.columns; column += threads)
            output[row_base + column] = 0;
        for (uint32_t group = tid; group < args.groups; group += threads)
            scales[row * args.groups + group] = 1.0f;
        return;
    }
    for (uint32_t group = 0; group < args.groups; group++) {
        uint32_t start = group * args.group_size;
        float local_max = 0.0f;
        for (uint32_t local = tid; local < args.group_size; local += threads) {
            local_max = fmaxf(local_max, fabsf(h3_bf16_to_f32(
                input[row_base + start + local])));
        }
        reductions[tid] = local_max;
        __syncthreads();
        for (uint32_t stride = threads / 2; stride; stride >>= 1) {
            if (tid < stride) {
                reductions[tid] = fmaxf(reductions[tid], reductions[tid + stride]);
            }
            __syncthreads();
        }
        float scale = reductions[0] > 0.0f ? reductions[0] / 127.0f : 1.0f / 127.0f;
        float inverse = reductions[0] > 0.0f ? 127.0f / reductions[0] : 127.0f;
        if (tid == 0) scales[row * args.groups + group] = scale;
        for (uint32_t local = tid; local < args.group_size; local += threads) {
            int quantized = (int)rintf(
                h3_bf16_to_f32(input[row_base + start + local]) * inverse);
            if (quantized > 127) quantized = 127;
            if (quantized < -127) quantized = -127;
            output[row_base + start + local] = (int8_t)quantized;
        }
        __syncthreads();
    }
}

extern "C" int h3_launch_quantize_bf16_int8_groups(
    const uint16_t *input, int8_t *output, float *scales,
    const h3_int8_group_quant_args *args, uint32_t dispatch_rows,
    hipStream_t stream) {
    h3_int8_group_quant_args host_args = *args;
    void *kernel_args[] = {&input, &output, &scales, &host_args};
    dim3 block(256, 1, 1);
    dim3 grid(dispatch_rows, 1, 1);
    return hipLaunchKernel(reinterpret_cast<const void *>(
                               h3_quantize_bf16_int8_groups_scalar_kernel),
                           grid, block, kernel_args, sizeof(kernel_args),
                           stream) == hipSuccess;
}

__global__ void h3_linear_int8_bf16_naive_kernel(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output, h3_linear_args args) {
    uint32_t row = blockIdx.y;
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= args.rows || column >= args.output_dim) return;
    int32_t acc = 0;
    for (uint32_t k = 0; k < args.input_dim; k++) {
        acc += (int32_t)input[row * args.input_dim + k] *
               (int32_t)weight[column * args.input_dim + k];
    }
    float value = (float)acc * input_scales[row] * weight_scales[column];
    output[row * args.output_dim + column] = h3_f32_to_bf16(value);
}

__global__ void h3_linear_int8_nax_r128_kernel(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output, h3_linear_args args) {
    __shared__ int8_t input_tile[16][16];
    __shared__ int8_t weight_tile[16][16];
    uint32_t row = blockIdx.y * 16 + threadIdx.y;
    uint32_t column = blockIdx.x * 16 + threadIdx.x;
    int32_t sum = 0;
    uint32_t tile_count = args.input_dim / 16;
    for (uint32_t tile = 0; tile < tile_count; tile++) {
        uint32_t input_k = tile * 16 + threadIdx.x;
        input_tile[threadIdx.y][threadIdx.x] =
            input_k < args.input_dim ?
            input[row * args.input_dim + input_k] : (int8_t)0;
        uint32_t weight_k = tile * 16 + threadIdx.y;
        weight_tile[threadIdx.y][threadIdx.x] =
            column < args.output_dim && weight_k < args.input_dim ?
            weight[column * args.input_dim + weight_k] : (int8_t)0;
        __syncthreads();
        for (uint32_t k = 0; k < 16; k++) {
            sum += (int32_t)input_tile[threadIdx.y][k] *
                   (int32_t)weight_tile[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < args.rows && column < args.output_dim) {
        float value = (float)sum * input_scales[row] * weight_scales[column];
        output[row * args.output_dim + column] = h3_f32_to_bf16(value);
    }
}

__global__ void h3_fc1_swiglu_int8_nax_r128_kernel(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output, h3_linear_args args) {
    __shared__ int8_t input_tile[16][16];
    __shared__ int8_t gate_weight_tile[16][16];
    __shared__ int8_t up_weight_tile[16][16];
    uint32_t row = blockIdx.y * 16 + threadIdx.y;
    uint32_t column = blockIdx.x * 16 + threadIdx.x;
    uint32_t hidden_dim = args.output_dim;
    int32_t gate_sum = 0;
    int32_t up_sum = 0;
    uint32_t tile_count = args.input_dim / 16;
    for (uint32_t tile = 0; tile < tile_count; tile++) {
        uint32_t input_k = tile * 16 + threadIdx.x;
        input_tile[threadIdx.y][threadIdx.x] =
            input_k < args.input_dim ?
            input[row * args.input_dim + input_k] : (int8_t)0;
        uint32_t weight_k = tile * 16 + threadIdx.y;
        gate_weight_tile[threadIdx.y][threadIdx.x] =
            column < hidden_dim && weight_k < args.input_dim ?
            weight[column * args.input_dim + weight_k] : (int8_t)0;
        up_weight_tile[threadIdx.y][threadIdx.x] =
            column < hidden_dim && weight_k < args.input_dim ?
            weight[(hidden_dim + column) * args.input_dim + weight_k] :
            (int8_t)0;
        __syncthreads();
        for (uint32_t k = 0; k < 16; k++) {
            gate_sum += (int32_t)input_tile[threadIdx.y][k] *
                        (int32_t)gate_weight_tile[k][threadIdx.x];
            up_sum += (int32_t)input_tile[threadIdx.y][k] *
                      (int32_t)up_weight_tile[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < args.rows && column < hidden_dim) {
        float gate = (float)gate_sum * input_scales[row] *
                     weight_scales[column];
        float up = (float)up_sum * input_scales[row] *
                   weight_scales[hidden_dim + column];
        float value = gate / (1.0f + expf(-gate)) * up;
        output[row * hidden_dim + column] = h3_f32_to_bf16(value);
    }
}

extern "C" int h3_launch_linear_int8_bf16_naive(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output, const h3_linear_args *args,
    hipStream_t stream) {
    h3_linear_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &input_scales, &weight_scales,
                           &output, &host_args};
    dim3 block(16, 1, 1);
    dim3 grid((host_args.output_dim + 15) / 16, host_args.rows, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_linear_int8_bf16_naive_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}

extern "C" int h3_launch_linear_int8_nax_r128(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output, const h3_linear_args *args,
    hipStream_t stream) {
    if (args->input_dim % 128 || args->output_dim % 128) return 0;
    h3_linear_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &input_scales, &weight_scales,
                           &output, &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.output_dim + 15) / 16, (host_args.rows + 15) / 16, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_linear_int8_nax_r128_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}

extern "C" int h3_launch_fc1_swiglu_int8_nax_r128(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output, const h3_linear_args *args,
    hipStream_t stream) {
    if (args->input_dim % 128 || args->output_dim % 128) return 0;
    h3_linear_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &input_scales, &weight_scales,
                           &output, &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.output_dim + 15) / 16, (host_args.rows + 15) / 16, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_fc1_swiglu_int8_nax_r128_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}

__global__ void h3_linear_int8_grouped_naive_kernel(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output,
    h3_linear_int8_grouped_args args) {
    uint32_t row = blockIdx.y;
    uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= args.rows || column >= args.output_dim) return;
    uint32_t scale_groups = args.input_dim / args.group_size;
    float total = 0.0f;
    for (uint32_t group = 0; group < scale_groups; group++) {
        uint32_t k0 = group * args.group_size;
        int32_t acc = 0;
        for (uint32_t k = 0; k < args.group_size; k++) {
            acc += (int32_t)input[row * args.input_dim + k0 + k] *
                   (int32_t)weight[column * args.input_dim + k0 + k];
        }
        total = fmaf((float)acc,
                     input_scales[row * scale_groups + group] *
                         weight_scales[column],
                     total);
    }
    output[row * args.output_dim + column] = h3_f32_to_bf16(total);
}

__global__ void h3_linear_int8_grouped_nax_r128x64_kernel(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output,
    h3_linear_int8_grouped_args args) {
    __shared__ int8_t input_tile[16][16];
    __shared__ int8_t weight_tile[16][16];
    uint32_t row = blockIdx.y * 16 + threadIdx.y;
    uint32_t column = blockIdx.x * 16 + threadIdx.x;
    uint32_t scale_groups = args.input_dim / args.group_size;
    float total = 0.0f;
    for (uint32_t group = 0; group < scale_groups; group++) {
        uint32_t k0 = group * args.group_size;
        int32_t sum = 0;
        uint32_t tile_count = args.group_size / 16;
        for (uint32_t tile = 0; tile < tile_count; tile++) {
            uint32_t input_k = k0 + tile * 16 + threadIdx.x;
            input_tile[threadIdx.y][threadIdx.x] =
                row < args.rows && input_k < args.input_dim ?
                input[row * args.input_dim + input_k] : (int8_t)0;
            uint32_t weight_k = k0 + tile * 16 + threadIdx.y;
            weight_tile[threadIdx.y][threadIdx.x] =
                column < args.output_dim && weight_k < args.input_dim ?
                weight[column * args.input_dim + weight_k] : (int8_t)0;
            __syncthreads();
            for (uint32_t k = 0; k < 16; k++) {
                sum += (int32_t)input_tile[threadIdx.y][k] *
                       (int32_t)weight_tile[k][threadIdx.x];
            }
            __syncthreads();
        }
        if (row < args.rows && column < args.output_dim) {
            total = fmaf((float)sum,
                         input_scales[row * scale_groups + group] *
                             weight_scales[column],
                         total);
        }
    }
    if (row < args.rows && column < args.output_dim) {
        output[row * args.output_dim + column] = h3_f32_to_bf16(total);
    }
}

extern "C" int h3_launch_linear_int8_grouped_naive(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output,
    const h3_linear_int8_grouped_args *args, hipStream_t stream) {
    h3_linear_int8_grouped_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &input_scales, &weight_scales,
                           &output, &host_args};
    dim3 block(16, 1, 1);
    dim3 grid((host_args.output_dim + 15) / 16, host_args.rows, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_linear_int8_grouped_naive_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}

extern "C" int h3_launch_linear_int8_grouped_nax_r128x64(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output,
    const h3_linear_int8_grouped_args *args, hipStream_t stream) {
    if (args->input_dim % 128 || args->output_dim % 64 ||
        args->group_size % 1024 || args->input_dim % args->group_size) {
        return 0;
    }
    h3_linear_int8_grouped_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &input_scales, &weight_scales,
                           &output, &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.output_dim + 15) / 16, (host_args.rows + 15) / 16, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(
                   h3_linear_int8_grouped_nax_r128x64_kernel),
               grid, block, kernel_args, sizeof(kernel_args), stream) ==
           hipSuccess;
}
