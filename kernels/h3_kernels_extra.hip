#include "h3_kernels.h"
#include "h3_wave32.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <rocwmma/rocwmma.hpp>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int h3_f32_wmma_on(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("H3_F32_WMMA");
        int rdna_wmma = h3_hip_rdna_wmma_default();
        if (e && strcmp(e, "0") == 0) cached = 0;
        else cached = rdna_wmma;
    }
    return cached;
}

/* CDNA: large VAE GEMMs go through hipBLAS SGEMM (gfx1151 used rocWMMA).
 * Opt out: H3_F32_HIPBLAS=0. Force on RDNA: H3_F32_HIPBLAS=1. */
static int h3_f32_hipblas_on(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("H3_F32_HIPBLAS");
        if (e && strcmp(e, "0") == 0) cached = 0;
        else if (e && strcmp(e, "1") == 0) cached = 1;
        else cached = !h3_hip_rdna_wmma_default();
    }
    return cached;
}

static int h3_f32_hipblas_linear(const float *input, const float *weight,
                                 const float *bias, float *output,
                                 const h3_linear_args *args,
                                 hipStream_t stream);

static int h3_f32_stats_on(void) {
    const char *e = getenv("H3_F32_STATS");
    return e && strcmp(e, "0") != 0;
}

static void h3_f32_stats_add(const char *path, const h3_linear_args *args) {
    if (!h3_f32_stats_on() || !args) return;
    static unsigned calls;
    static unsigned long long flop;
    if (calls < 24u) {
        fprintf(stderr, "f32-linear %s M=%u K=%u N=%u\n", path, args->rows,
                args->input_dim, args->output_dim);
    }
    calls++;
    flop += 2ull * (unsigned long long)args->rows *
            (unsigned long long)args->input_dim *
            (unsigned long long)args->output_dim;
    if ((calls & 63u) == 0u)
        fprintf(stderr, "f32-linear stats calls=%u flops=%.2f TFLOP last=%s\n",
                calls, (double)flop * 1e-12, path);
}

static int h3_int8_split_on(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("H3_INT8_SPLIT");
        cached = (e && strcmp(e, "0") != 0) ? 1 : 0;
    }
    return cached;
}

static void h3_int8_split_begin(hipEvent_t *start, hipEvent_t *stop,
                                hipStream_t stream) {
    if (!*start) {
        (void)hipEventCreate(start);
        (void)hipEventCreate(stop);
    }
    (void)hipEventRecord(*start, stream);
}

static void h3_int8_split_end(hipEvent_t start, hipEvent_t stop,
                              hipStream_t stream, const char *tag) {
    (void)hipEventRecord(stop, stream);
    (void)hipEventSynchronize(stop);
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, start, stop);
    fprintf(stderr, "int8-split %s %.3f ms\n", tag, ms);
}

__device__ __host__ inline float h3_bf16_to_f32(uint16_t value) {
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

template <int BM, int BK, int LDS_K>
__device__ inline void h3_load_f32_tile_bmxk(
    float dst[BM][LDS_K], const float *src, uint32_t block_index, uint32_t k0,
    uint32_t bound, uint32_t stride, uint32_t tid) {
    /* 256 threads. BM=64: 4 threads/row × float4; BM=128: 2 threads/row. */
    constexpr uint32_t threads_per_row = 256u / (uint32_t)BM;
    constexpr uint32_t vecs_per_thread = (uint32_t)BK / (threads_per_row * 4u);
    uint32_t local_row = tid / threads_per_row;
    uint32_t index = block_index + local_row;
#pragma unroll
    for (uint32_t vec = 0; vec < vecs_per_thread; vec++) {
        uint32_t local_k =
            (tid % threads_per_row) * 4u + vec * (threads_per_row * 4u);
        float4 packed = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (index < bound)
            packed = *reinterpret_cast<const float4 *>(
                src + (size_t)index * stride + k0 + local_k);
        dst[local_row][local_k + 0u] = packed.x;
        dst[local_row][local_k + 1u] = packed.y;
        dst[local_row][local_k + 2u] = packed.z;
        dst[local_row][local_k + 3u] = packed.w;
    }
}

template <int BK, int LDS_K>
__device__ inline void h3_load_f32_tile_64xk(
    float dst[64][LDS_K], const float *src, uint32_t block_index, uint32_t k0,
    uint32_t bound, uint32_t stride, uint32_t tid) {
    h3_load_f32_tile_bmxk<64, BK, LDS_K>(dst, src, block_index, k0, bound,
                                          stride, tid);
}

template <int BK>
__global__ void h3_linear_f32_r64_kernel(const float *input, const float *weight,
                                         const float *bias, float *output,
                                         h3_linear_args args) {
    constexpr int BM = 64, BN = 64;
    /* +1 pads LDS rows so As[r][k] and As[r+1][k] avoid 32-bank collisions
     * when BK is a multiple of 32 (default BK=32). */
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BN][BK + 1];
    uint32_t block_row = blockIdx.y * (uint32_t)BM;
    uint32_t block_col = blockIdx.x * (uint32_t)BN;
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t tid = ty * 16u + tx;
    uint32_t k_tiles = args.input_dim / (uint32_t)BK;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
#pragma unroll 1
    for (uint32_t kt = 0; kt < k_tiles; kt++) {
        uint32_t k0 = kt * (uint32_t)BK;
        h3_load_f32_tile_64xk<BK, BK + 1>(As, input, block_row, k0, args.rows,
                                           args.input_dim, tid);
        h3_load_f32_tile_64xk<BK, BK + 1>(Bs, weight, block_col, k0,
                                           args.output_dim, args.input_dim,
                                           tid);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; k++) {
            float av[4], bv[4];
#pragma unroll
            for (int i = 0; i < 4; i++)
                av[i] = As[ty * 4u + (uint32_t)i][k];
#pragma unroll
            for (int j = 0; j < 4; j++)
                bv[j] = Bs[tx * 4u + (uint32_t)j][k];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) {
        uint32_t row = block_row + ty * 4u + (uint32_t)i;
        if (row < args.rows) {
            uint32_t column0 = block_col + tx * 4u;
            if (column0 + 3u < args.output_dim && (column0 % 4u) == 0u) {
                float4 out;
                out.x = acc[i][0];
                out.y = acc[i][1];
                out.z = acc[i][2];
                out.w = acc[i][3];
                if (args.has_bias) {
                    float4 b = *reinterpret_cast<const float4 *>(bias + column0);
                    out.x += b.x;
                    out.y += b.y;
                    out.z += b.z;
                    out.w += b.w;
                }
                *reinterpret_cast<float4 *>(
                    output + (size_t)row * args.output_dim + column0) = out;
            } else {
                for (int j = 0; j < 4; j++) {
                    uint32_t column = column0 + (uint32_t)j;
                    if (column < args.output_dim) {
                        float value = acc[i][j];
                        if (args.has_bias) value += bias[column];
                        output[(size_t)row * args.output_dim + column] = value;
                    }
                }
            }
        }
    }
}

/* 128×128 output tile, 16×16 threads, 8×8 FMA. Fewer blocks, more ILP than r64.
 * Opt out with H3_F32_R64=1. */
template <int BK>
__global__ void __launch_bounds__(256)
h3_linear_f32_r128_kernel(const float *input, const float *weight,
                          const float *bias, float *output,
                          h3_linear_args args) {
    constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
    constexpr int LDS_K = BK + 1;
    __shared__ float As[BM][LDS_K];
    __shared__ float Bs[BN][LDS_K];
    uint32_t block_row = blockIdx.y * (uint32_t)BM;
    uint32_t block_col = blockIdx.x * (uint32_t)BN;
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t tid = ty * 16u + tx;
    uint32_t k_tiles = args.input_dim / (uint32_t)BK;
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;
#pragma unroll 1
    for (uint32_t kt = 0; kt < k_tiles; kt++) {
        uint32_t k0 = kt * (uint32_t)BK;
        h3_load_f32_tile_bmxk<BM, BK, LDS_K>(As, input, block_row, k0,
                                              args.rows, args.input_dim, tid);
        h3_load_f32_tile_bmxk<BN, BK, LDS_K>(Bs, weight, block_col, k0,
                                              args.output_dim, args.input_dim,
                                              tid);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; k++) {
            float av[TM], bv[TN];
#pragma unroll
            for (int i = 0; i < TM; i++)
                av[i] = As[ty * (uint32_t)TM + (uint32_t)i][k];
#pragma unroll
            for (int j = 0; j < TN; j++)
                bv[j] = Bs[tx * (uint32_t)TN + (uint32_t)j][k];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; i++) {
        uint32_t row = block_row + ty * (uint32_t)TM + (uint32_t)i;
        if (row >= args.rows) continue;
        uint32_t column0 = block_col + tx * (uint32_t)TN;
        if (column0 + (uint32_t)TN - 1u < args.output_dim &&
            (column0 % 4u) == 0u) {
#pragma unroll
            for (int v = 0; v < TN / 4; v++) {
                uint32_t column = column0 + (uint32_t)v * 4u;
                float4 out;
                out.x = acc[i][v * 4 + 0];
                out.y = acc[i][v * 4 + 1];
                out.z = acc[i][v * 4 + 2];
                out.w = acc[i][v * 4 + 3];
                if (args.has_bias) {
                    float4 b =
                        *reinterpret_cast<const float4 *>(bias + column);
                    out.x += b.x;
                    out.y += b.y;
                    out.z += b.z;
                    out.w += b.w;
                }
                *reinterpret_cast<float4 *>(
                    output + (size_t)row * args.output_dim + column) = out;
            }
        } else {
#pragma unroll
            for (int j = 0; j < TN; j++) {
                uint32_t column = column0 + (uint32_t)j;
                if (column >= args.output_dim) continue;
                float value = acc[i][j];
                if (args.has_bias) value += bias[column];
                output[(size_t)row * args.output_dim + column] = value;
            }
        }
    }
}

/* F32 linear on the bf16 matrix cores. Operands are rounded to bf16 on the way
 * into LDS; the accumulator and the output stay f32, so the only precision loss
 * is the input rounding (~1.1e-4 relative to the largest output on the VAE
 * shapes). The FP32 FMA ceiling on gfx1151 is 7.4 TFLOP/s and r128 already
 * reaches 58% of it, whereas bf16 WMMA peaks at 57, so this is the only way
 * left to make the video-VAE decoder GEMMs meaningfully faster.
 *
 * Tile: 256x128 output, BK=32, one 64x64 quadrant per wave, K tiles prefetched
 * into registers so the global loads overlap the mma chain. Opt out with
 * H3_F32_WMMA=0. */

template <int BM, int BN, int BK, int WM, int WN, int PAD>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32)
h3_linear_f32_wmma_kernel(const float *input, const float *weight,
                          const float *bias, float *output,
                          h3_linear_args args) {
    using bf16 = rocwmma::bfloat16_t;
    constexpr int LD = BK + PAD;
    constexpr int WAVES_N = BN / WN;
    constexpr int THREADS = (BM / WM) * WAVES_N * 32;
    constexpr int FM = WM / 16, FN = WN / 16;
    constexpr int KQ = BK / 4;
    constexpr int AI = BM * KQ / THREADS;
    constexpr int BI = BN * KQ / THREADS;
    constexpr int ROW_STEP = THREADS / KQ;

    __shared__ uint16_t As[BM * LD];
    __shared__ uint16_t Bs[BN * LD];

    uint32_t tid = threadIdx.x;
    uint32_t wave = tid >> 5;
    uint32_t wm = wave / (uint32_t)WAVES_N, wn = wave % (uint32_t)WAVES_N;
    uint32_t block_row = blockIdx.y * (uint32_t)BM;
    uint32_t block_col = blockIdx.x * (uint32_t)BN;
    uint32_t rows = args.rows, cols = args.output_dim, K = args.input_dim;

    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) rocwmma::fill_fragment(acc[i][j], 0.0f);

    const int row0 = (int)tid / KQ, kq0 = ((int)tid % KQ) * 4;
    const bf16 *pA = reinterpret_cast<const bf16 *>(As);
    const bf16 *pB = reinterpret_cast<const bf16 *>(Bs);

    float4 a_next[AI], b_next[BI];
    auto fetch = [&](uint32_t k0) {
#pragma unroll
        for (int it = 0; it < AI; it++) {
            uint32_t r = block_row + (uint32_t)(row0 + it * ROW_STEP);
            a_next[it] = r < rows ? *reinterpret_cast<const float4 *>(
                                        input + (size_t)r * K + k0 + kq0)
                                  : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int it = 0; it < BI; it++) {
            uint32_t r = block_col + (uint32_t)(row0 + it * ROW_STEP);
            b_next[it] = r < cols ? *reinterpret_cast<const float4 *>(
                                        weight + (size_t)r * K + k0 + kq0)
                                  : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    };
    auto stash = [&]() {
#pragma unroll
        for (int it = 0; it < AI; it++) {
            uint32_t *d = reinterpret_cast<uint32_t *>(
                As + (row0 + it * ROW_STEP) * LD + kq0);
            d[0] = h3_pack_bf16x2(a_next[it].x, a_next[it].y);
            d[1] = h3_pack_bf16x2(a_next[it].z, a_next[it].w);
        }
#pragma unroll
        for (int it = 0; it < BI; it++) {
            uint32_t *d = reinterpret_cast<uint32_t *>(
                Bs + (row0 + it * ROW_STEP) * LD + kq0);
            d[0] = h3_pack_bf16x2(b_next[it].x, b_next[it].y);
            d[1] = h3_pack_bf16x2(b_next[it].z, b_next[it].w);
        }
    };
    auto compute = [&]() {
#pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bf16,
                              rocwmma::row_major> fa[FM];
            rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bf16,
                              rocwmma::col_major> fb[FN];
#pragma unroll
            for (int i = 0; i < FM; i++)
                rocwmma::load_matrix_sync(
                    fa[i], pA + ((int)(wm * WM) + i * 16) * LD + ks * 16, LD);
#pragma unroll
            for (int j = 0; j < FN; j++)
                rocwmma::load_matrix_sync(
                    fb[j], pB + ((int)(wn * WN) + j * 16) * LD + ks * 16, LD);
#pragma unroll
            for (int i = 0; i < FM; i++)
#pragma unroll
                for (int j = 0; j < FN; j++)
                    rocwmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
    };

    uint32_t ktiles = K / (uint32_t)BK;
    fetch(0u);
    stash();
    for (uint32_t kt = 1; kt < ktiles; kt++) {
        fetch(kt * (uint32_t)BK);
        __syncthreads();
        compute();
        __syncthreads();
        stash();
    }
    __syncthreads();
    compute();

    /* Accumulator element i of every fragment belongs to row 2*i + (lane>>4),
     * lane&15 picks the column. */
    uint32_t lane = tid & 31u;
    uint32_t frag_col = lane & 15u, row_half = lane >> 4;
#pragma unroll
    for (int j = 0; j < FN; j++) {
        uint32_t col = block_col + wn * (uint32_t)WN + (uint32_t)j * 16u +
                       frag_col;
        if (col >= cols) continue;
        float b = args.has_bias ? bias[col] : 0.0f;
#pragma unroll
        for (int i = 0; i < FM; i++) {
#pragma unroll
            for (int e = 0; e < 8; e++) {
                uint32_t row = block_row + wm * (uint32_t)WM +
                               (uint32_t)i * 16u + 2u * (uint32_t)e + row_half;
                if (row < rows)
                    output[(size_t)row * cols + col] = acc[i][j].x[e] + b;
            }
        }
    }
}

/* Tile shapes differ mostly in register pressure: a 64x64 per-wave tile needs
 * 16 accumulator fragments (128 VGPRs) and spills, while 32x64 needs 8. Pick
 * with H3_F32_WMMA_CFG; see docs/perf-day9/STATUS.md for the sweep. */
static const void *h3_f32_wmma_kernel(uint32_t *bm, uint32_t *bn,
                                      uint32_t *threads) {
    const char *cfg = getenv("H3_F32_WMMA_CFG");
    unsigned choice = cfg ? (unsigned)strtoul(cfg, NULL, 10) : 3u;
#define H3_WMMA_CASE(BM, BN, WM, WN)                                           \
    do {                                                                       \
        *bm = BM;                                                              \
        *bn = BN;                                                              \
        *threads = (BM / WM) * (BN / WN) * 32u;                                 \
        return reinterpret_cast<const void *>(                                 \
            h3_linear_f32_wmma_kernel<BM, BN, 32, WM, WN, 8>);                  \
    } while (0)
    switch (choice) {
    case 1: H3_WMMA_CASE(256, 128, 64, 64);
    case 2: H3_WMMA_CASE(256, 256, 64, 64);
    case 4: H3_WMMA_CASE(128, 128, 32, 64);
    case 5: H3_WMMA_CASE(128, 256, 64, 64);
    case 6: H3_WMMA_CASE(256, 128, 64, 32);
    case 7: H3_WMMA_CASE(256, 256, 32, 64);
    case 8: H3_WMMA_CASE(512, 128, 32, 64);
    case 9: H3_WMMA_CASE(128, 128, 32, 32);
    case 3:
    default: H3_WMMA_CASE(256, 128, 32, 64);
    }
#undef H3_WMMA_CASE
}

__global__ void h3_conv1d_f32_kernel(const float *input, const float *weight,
                                     const float *bias, float *output,
                                     h3_conv1d_args args) {
    uint32_t out = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t time = args.time_offset + blockIdx.y;
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

/* Time-blocked conv1d. The scalar kernel above gives each thread one output
 * element, so every weight is re-fetched once per time step -- at the audio
 * VAE's shapes that is gigabytes of weight traffic for a few hundred MFLOP.
 * Here a block owns BC output channels over BC*BY*TT time steps and stages the
 * weight slice in LDS, so each weight crosses the bus once per BY*TT outputs.
 *
 * Threads are (channel, time) with the channel fastest, and each thread's TT
 * time steps are strided by BY, so for any fixed step the block writes one
 * contiguous run of the time-major output. Input is left in global: the whole
 * activation is under a megabyte, so it stays L2-resident, and a wave that
 * spans one time step reads it as a broadcast.
 *
 * The (in, k) accumulation order is transposed from the scalar kernel's to
 * match the weight's memory layout, so results differ at f32 rounding. */
template <uint32_t BC, uint32_t BY, uint32_t TT, uint32_t WSLICE>
__global__ __launch_bounds__(BC *BY) void h3_conv1d_f32_tiled_kernel(
    const float *__restrict__ input, const float *__restrict__ weight,
    const float *__restrict__ bias, float *__restrict__ output,
    h3_conv1d_args args) {
    __shared__ float sw[BC * WSLICE];
    const uint32_t channel = blockIdx.x * BC + threadIdx.x;
    const uint32_t time_base = args.time_offset + blockIdx.y * (BY * TT);
    const uint32_t batch = blockIdx.z;
    const uint32_t taps = args.kernel;
    /* WSLICE has to hold a whole number of taps so the LDS slice can be
     * indexed as (in - in0) * taps + k. */
    const uint32_t in_slice = WSLICE / taps;
    const uint32_t lane = threadIdx.y * BC + threadIdx.x;
    const float bias_value =
        (args.has_bias && channel < args.output_channels) ? bias[channel]
                                                          : 0.0f;
    float acc[TT];
    for (uint32_t t = 0; t < TT; t++) acc[t] = bias_value;
    const size_t input_batch =
        (size_t)batch * args.input_length * args.input_channels;
    const size_t weight_row = (size_t)args.input_channels * taps;
    for (uint32_t in0 = 0; in0 < args.input_channels; in0 += in_slice) {
        uint32_t ins = args.input_channels - in0;
        if (ins > in_slice) ins = in_slice;
        uint32_t span = ins * taps;
        __syncthreads();
        /* One wave per output channel walking a contiguous weight run. */
        for (uint32_t c = threadIdx.y; c < BC; c += BY) {
            uint32_t row = blockIdx.x * BC + c;
            const float *src =
                weight + (size_t)row * weight_row + (size_t)in0 * taps;
            int valid = row < args.output_channels;
            for (uint32_t j = threadIdx.x; j < span; j += BC) {
                sw[c * WSLICE + j] = valid ? src[j] : 0.0f;
            }
        }
        __syncthreads();
        for (uint32_t in = 0; in < ins; in++) {
            for (uint32_t k = 0; k < taps; k++) {
                float w = sw[threadIdx.x * WSLICE + in * taps + k];
                int32_t tap = (int32_t)(k * args.dilation) -
                              (int32_t)args.padding;
                for (uint32_t t = 0; t < TT; t++) {
                    uint32_t time = time_base + t * BY + threadIdx.y;
                    if (time >= args.output_length) continue;
                    int32_t source = (int32_t)time * (int32_t)args.stride + tap;
                    if (source < 0 || source >= (int32_t)args.input_length) {
                        continue;
                    }
                    acc[t] = fmaf(input[input_batch +
                                        (size_t)source * args.input_channels +
                                        in0 + in],
                                  w, acc[t]);
                }
            }
        }
    }
    if (channel >= args.output_channels) return;
    const size_t output_batch =
        (size_t)batch * args.output_length * args.output_channels;
    for (uint32_t t = 0; t < TT; t++) {
        uint32_t time = time_base + t * BY + threadIdx.y;
        if (time >= args.output_length) continue;
        output[output_batch + (size_t)time * args.output_channels + channel] =
            acc[t];
    }
    (void)lane;
}

__global__ void h3_conv_transpose1d_f32_kernel(const float *input,
                                               const float *weight,
                                               const float *bias, float *output,
                                               h3_conv1d_args args) {
    uint32_t out = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t time = args.time_offset + blockIdx.y;
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
    uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t work = args.spatial_span * args.output_channels;
    if (local >= work) return;
    uint32_t out = local % args.output_channels;
    uint32_t spatial_in_chunk = local / args.output_channels;
    uint64_t spatial_count = (uint64_t)args.output_depth * args.output_height *
                             args.output_width;
    uint64_t spatial = (uint64_t)args.spatial_offset + spatial_in_chunk;
    uint32_t batch = blockIdx.y;
    if (spatial >= spatial_count || batch >= args.batch) return;
    uint32_t out_x = (uint32_t)(spatial % args.output_width);
    uint64_t plane = spatial / args.output_width;
    uint32_t out_y = (uint32_t)(plane % args.output_height);
    uint32_t out_t = (uint32_t)(plane / args.output_height);
    float sum = args.has_bias ? bias[out] : 0.0f;
    size_t input_batch = ((size_t)batch * args.input_depth * args.input_height *
                          args.input_width) * args.input_channels;
    size_t output_batch = ((size_t)batch * spatial_count) * args.output_channels;
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

/* HIP/CUDA grid.y and grid.z are limited to 65535. A 22-frame 256px encoder
 * tile has T*H*W = 1,441,792, and a single huge launch wedged gfx1151
 * (amdgpu GPU reset). Chunk the grid so each packet stays in limits.
 * Do not synchronize between chunks: hipLaunchKernel copies by-value args
 * immediately, and the extra waits serialized VAE convs (~88 syncs/layer
 * at 256px). Execution errors still surface on the next h3_gpu_submit. */
enum { H3_HIP_MAX_GRID_YZ = 65535u };
enum { H3_HIP_CONV3D_SPATIAL_CHUNK = 16384u };
enum { H3_HIP_CONV1D_TIME_CHUNK = 8192u };
enum { H3_CONV1D_TT = 8u, H3_CONV1D_WSLICE = 64u };

static int h3_conv1d_tiled_on(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("H3_CONV1D_TILED");
        cached = (!e || strcmp(e, "0") != 0) ? 1 : 0;
    }
    return cached;
}

static int h3_hip_launch(const void *kernel, dim3 grid, dim3 block,
                         void **kernel_args, hipStream_t stream) {
    return hipLaunchKernel(kernel, grid, block, kernel_args, 0, stream) ==
           hipSuccess;
}

extern "C" int h3_launch_conv1d_f32(const float *input, const float *weight,
                                    const float *bias, float *output,
                                    const h3_conv1d_args *args,
                                    hipStream_t stream) {
    h3_conv1d_args host_args = *args;
    if (!host_args.batch || !host_args.output_length ||
        !host_args.output_channels) {
        return 1;
    }
    if (h3_conv1d_tiled_on() && host_args.kernel <= H3_CONV1D_WSLICE) {
        uint32_t bc = 64u, by = 4u;
        if (host_args.output_channels <= 8u) { bc = 8u; by = 32u; }
        else if (host_args.output_channels <= 16u) { bc = 16u; by = 16u; }
        else if (host_args.output_channels <= 32u) { bc = 32u; by = 8u; }
        const void *kernel = NULL;
#define H3_CONV1D_TILED_PICK(c, y)                                           \
    if (bc == (c))                                                           \
        kernel = reinterpret_cast<const void *>(                             \
            h3_conv1d_f32_tiled_kernel<c, y, H3_CONV1D_TT,                   \
                                       H3_CONV1D_WSLICE>);
        H3_CONV1D_TILED_PICK(8u, 32u)
        H3_CONV1D_TILED_PICK(16u, 16u)
        H3_CONV1D_TILED_PICK(32u, 8u)
        H3_CONV1D_TILED_PICK(64u, 4u)
#undef H3_CONV1D_TILED_PICK
        uint32_t per_block = by * H3_CONV1D_TT;
        uint32_t chunk = H3_HIP_MAX_GRID_YZ * per_block;
        for (uint32_t offset = 0; offset < host_args.output_length; ) {
            uint32_t span = host_args.output_length - offset;
            if (span > chunk) span = chunk;
            h3_conv1d_args chunk_args = host_args;
            chunk_args.time_offset = offset;
            void *kernel_args[] = {&input, &weight, &bias, &output,
                                   &chunk_args};
            dim3 grid((chunk_args.output_channels + bc - 1u) / bc,
                      (span + per_block - 1u) / per_block, chunk_args.batch);
            if (!h3_hip_launch(kernel, grid, dim3(bc, by, 1), kernel_args,
                               stream)) {
                return 0;
            }
            offset += span;
        }
        return 1;
    }
    dim3 block(16, 1, 1);
    uint32_t chunk = H3_HIP_CONV1D_TIME_CHUNK;
    if (chunk > H3_HIP_MAX_GRID_YZ) chunk = H3_HIP_MAX_GRID_YZ;
    for (uint32_t offset = 0; offset < host_args.output_length; ) {
        uint32_t span = host_args.output_length - offset;
        if (span > chunk) span = chunk;
        h3_conv1d_args chunk_args = host_args;
        chunk_args.time_offset = offset;
        void *kernel_args[] = {&input, &weight, &bias, &output, &chunk_args};
        dim3 grid((chunk_args.output_channels + 15) / 16, span,
                  chunk_args.batch);
        if (!h3_hip_launch(
                reinterpret_cast<const void *>(h3_conv1d_f32_kernel), grid,
                block, kernel_args, stream)) {
            return 0;
        }
        offset += span;
    }
    return 1;
}

extern "C" int h3_launch_conv_transpose1d_f32(const float *input,
                                              const float *weight,
                                              const float *bias, float *output,
                                              const h3_conv1d_args *args,
                                              hipStream_t stream) {
    h3_conv1d_args host_args = *args;
    if (!host_args.batch || !host_args.output_length ||
        !host_args.output_channels) {
        return 1;
    }
    dim3 block(16, 1, 1);
    uint32_t chunk = H3_HIP_CONV1D_TIME_CHUNK;
    if (chunk > H3_HIP_MAX_GRID_YZ) chunk = H3_HIP_MAX_GRID_YZ;
    for (uint32_t offset = 0; offset < host_args.output_length; ) {
        uint32_t span = host_args.output_length - offset;
        if (span > chunk) span = chunk;
        h3_conv1d_args chunk_args = host_args;
        chunk_args.time_offset = offset;
        void *kernel_args[] = {&input, &weight, &bias, &output, &chunk_args};
        dim3 grid((chunk_args.output_channels + 15) / 16, span,
                  chunk_args.batch);
        if (!h3_hip_launch(
                reinterpret_cast<const void *>(h3_conv_transpose1d_f32_kernel),
                grid, block, kernel_args, stream)) {
            return 0;
        }
        offset += span;
    }
    return 1;
}

extern "C" int h3_launch_conv3d_f32(const float *input, const float *weight,
                                    const float *bias, float *output,
                                    const h3_conv3d_args *args,
                                    hipStream_t stream) {
    h3_conv3d_args host_args = *args;
    uint64_t spatial_count = (uint64_t)host_args.output_depth *
                             host_args.output_height * host_args.output_width;
    if (!host_args.batch || !host_args.output_channels || !spatial_count) {
        return 1;
    }
    if (spatial_count > UINT32_MAX) return 0;
    dim3 block(256, 1, 1);
    uint32_t total_spatial = (uint32_t)spatial_count;
    uint32_t chunk = H3_HIP_CONV3D_SPATIAL_CHUNK;
    if (chunk > H3_HIP_MAX_GRID_YZ) chunk = H3_HIP_MAX_GRID_YZ;
    for (uint32_t offset = 0; offset < total_spatial; ) {
        uint32_t span = total_spatial - offset;
        if (span > chunk) span = chunk;
        h3_conv3d_args chunk_args = host_args;
        chunk_args.spatial_offset = offset;
        chunk_args.spatial_span = span;
        void *kernel_args[] = {&input, &weight, &bias, &output, &chunk_args};
        uint32_t work = span * chunk_args.output_channels;
        dim3 grid((work + 255u) / 256u, chunk_args.batch, 1);
        if (!h3_hip_launch(
                reinterpret_cast<const void *>(h3_conv3d_f32_kernel), grid,
                block, kernel_args, stream)) {
            return 0;
        }
        offset += span;
    }
    return 1;
}

extern "C" int h3_launch_linear_f32(const float *input, const float *weight,
                                    const float *bias, float *output,
                                    const h3_linear_args *args,
                                    hipStream_t stream) {
    h3_linear_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &bias, &output, &host_args};
    if (host_args.input_dim % 16u == 0 && !getenv("H3_F32_LEGACY")) {
        dim3 block(16, 16, 1);
        int force_r64 = getenv("H3_F32_R64") &&
                        strcmp(getenv("H3_F32_R64"), "0") != 0;
        int use_r128 = !force_r64 && host_args.input_dim % 32u == 0;
        /* Only the decoder-sized GEMMs go to the matrix cores. Anything
         * narrower than one 256x128 tile has nothing to gain, and the small
         * f32 projections that survive here feed the DiT conditioning, where a
         * bf16 perturbation changes the sampled trajectory rather than just
         * the last digits. */
        int wmma_worth_it = host_args.rows >= 256u &&
                            host_args.input_dim >= 512u &&
                            host_args.output_dim >= 512u;
        if (wmma_worth_it && h3_f32_hipblas_on() &&
            h3_f32_hipblas_linear(input, weight, bias, output, &host_args,
                                  stream)) {
            h3_f32_stats_add("hipblas", &host_args);
            return 1;
        }
        if (use_r128 && wmma_worth_it && h3_f32_wmma_on()) {
            uint32_t bm = 0, bn = 0, threads = 0;
            const void *kernel = h3_f32_wmma_kernel(&bm, &bn, &threads);
            dim3 grid((host_args.output_dim + bn - 1u) / bn,
                      (host_args.rows + bm - 1u) / bm, 1);
            h3_f32_stats_add("wmma", &host_args);
            return hipLaunchKernel(kernel, grid, dim3(threads), kernel_args, 0,
                                   stream) == hipSuccess;
        }
        if (use_r128) {
            dim3 grid((host_args.output_dim + 127u) / 128u,
                      (host_args.rows + 127u) / 128u, 1);
            h3_f32_stats_add("r128", &host_args);
            return hipLaunchKernel(
                       reinterpret_cast<const void *>(
                           h3_linear_f32_r128_kernel<32>),
                       grid, block, kernel_args, 0, stream) == hipSuccess;
        }
        dim3 grid((host_args.output_dim + 63u) / 64u,
                  (host_args.rows + 63u) / 64u, 1);
        /* BK=32 beats BK=64 on gfx1151 VAE shapes; BK=16 for odd alignments. */
        const void *kernel = host_args.input_dim % 32u == 0 ?
            reinterpret_cast<const void *>(h3_linear_f32_r64_kernel<32>) :
            reinterpret_cast<const void *>(h3_linear_f32_r64_kernel<16>);
        h3_f32_stats_add("r64", &host_args);
    return hipLaunchKernel(kernel, grid, block, kernel_args, 0, stream) ==
           hipSuccess;
}
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
        value += h3_shfl_xor32(value, mask);
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
    __shared__ float warp_max[16];
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
    if ((args.columns & 3u) == 0u) {
        /* Four BF16 values per iteration (uint2 = 4×u16). */
        for (uint32_t column = tid * 4u; column < args.columns;
             column += threads * 4u) {
            uint2 packed =
                *reinterpret_cast<const uint2 *>(input + base + column);
            uint16_t a0 = (uint16_t)(packed.x & 0xffffu);
            uint16_t a1 = (uint16_t)(packed.x >> 16);
            uint16_t a2 = (uint16_t)(packed.y & 0xffffu);
            uint16_t a3 = (uint16_t)(packed.y >> 16);
            local_max = fmaxf(local_max, fabsf(h3_bf16_to_f32(a0)));
            local_max = fmaxf(local_max, fabsf(h3_bf16_to_f32(a1)));
            local_max = fmaxf(local_max, fabsf(h3_bf16_to_f32(a2)));
            local_max = fmaxf(local_max, fabsf(h3_bf16_to_f32(a3)));
        }
    } else {
        for (uint32_t column = tid; column < args.columns; column += threads) {
            local_max = fmaxf(local_max,
                              fabsf(h3_bf16_to_f32(input[base + column])));
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, h3_shfl_xor32(local_max, offset));
    if ((tid & 31u) == 0u) warp_max[tid >> 5] = local_max;
    __syncthreads();
    if (tid < 16u) {
        float value = (tid < (threads >> 5)) ? warp_max[tid] : 0.0f;
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1)
            value = fmaxf(value, h3_shfl_xor32(value, offset));
        if (tid == 0) warp_max[0] = value;
    }
    __syncthreads();
    float clipped_max = warp_max[0] * args.clip;
    float scale = clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
    float inverse = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
    if (tid == 0) scales[row] = scale;
    if ((args.columns & 3u) == 0u) {
        for (uint32_t column = tid * 4u; column < args.columns;
             column += threads * 4u) {
            uint2 packed =
                *reinterpret_cast<const uint2 *>(input + base + column);
            uint16_t vals[4] = {
                (uint16_t)(packed.x & 0xffffu), (uint16_t)(packed.x >> 16),
                (uint16_t)(packed.y & 0xffffu), (uint16_t)(packed.y >> 16)};
            uint32_t qpack = 0;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                int quantized =
                    (int)rintf(h3_bf16_to_f32(vals[i]) * inverse);
                if (quantized > 127) quantized = 127;
                if (quantized < -127) quantized = -127;
                qpack |= (uint32_t)(uint8_t)(int8_t)quantized << (8 * i);
            }
            *reinterpret_cast<uint32_t *>(output + base + column) = qpack;
        }
    } else {
        for (uint32_t column = tid; column < args.columns; column += threads) {
            int quantized =
                (int)rintf(h3_bf16_to_f32(input[base + column]) * inverse);
            if (quantized > 127) quantized = 127;
            if (quantized < -127) quantized = -127;
            output[base + column] = (int8_t)quantized;
        }
    }
}

/* Stage the BF16 row in LDS so the max pass and the quantize pass share one
 * global load. Opt-in only: H3_INT8_QUANT_LDS=1 (slightly slower). */
__global__ void h3_quantize_bf16_int8_rows_lds_kernel(
    const uint16_t *input, int8_t *output, float *scales,
    h3_int8_quant_args args) {
    extern __shared__ uint16_t smem[];
    __shared__ float warp_max[16];
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
    for (uint32_t column = tid * 4u; column < args.columns;
         column += threads * 4u) {
        uint2 packed = *reinterpret_cast<const uint2 *>(input + base + column);
        *reinterpret_cast<uint2 *>(smem + column) = packed;
        uint16_t a0 = (uint16_t)(packed.x & 0xffffu);
        uint16_t a1 = (uint16_t)(packed.x >> 16);
        uint16_t a2 = (uint16_t)(packed.y & 0xffffu);
        uint16_t a3 = (uint16_t)(packed.y >> 16);
        local_max = fmaxf(local_max, fabsf(h3_bf16_to_f32(a0)));
        local_max = fmaxf(local_max, fabsf(h3_bf16_to_f32(a1)));
        local_max = fmaxf(local_max, fabsf(h3_bf16_to_f32(a2)));
        local_max = fmaxf(local_max, fabsf(h3_bf16_to_f32(a3)));
    }
    __syncthreads();
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, h3_shfl_xor32(local_max, offset));
    if ((tid & 31u) == 0u) warp_max[tid >> 5] = local_max;
    __syncthreads();
    if (tid < 16u) {
        float value = (tid < (threads >> 5)) ? warp_max[tid] : 0.0f;
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1)
            value = fmaxf(value, h3_shfl_xor32(value, offset));
        if (tid == 0) warp_max[0] = value;
    }
    __syncthreads();
    float clipped_max = warp_max[0] * args.clip;
    float scale = clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
    float inverse = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
    if (tid == 0) scales[row] = scale;
    for (uint32_t column = tid * 4u; column < args.columns;
         column += threads * 4u) {
        uint2 packed = *reinterpret_cast<const uint2 *>(smem + column);
        uint16_t vals[4] = {
            (uint16_t)(packed.x & 0xffffu), (uint16_t)(packed.x >> 16),
            (uint16_t)(packed.y & 0xffffu), (uint16_t)(packed.y >> 16)};
        uint32_t qpack = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int quantized = (int)rintf(h3_bf16_to_f32(vals[i]) * inverse);
            if (quantized > 127) quantized = 127;
            if (quantized < -127) quantized = -127;
            qpack |= (uint32_t)(uint8_t)(int8_t)quantized << (8 * i);
        }
        *reinterpret_cast<uint32_t *>(output + base + column) = qpack;
    }
}

extern "C" int h3_launch_quantize_bf16_int8_rows(
    const uint16_t *input, int8_t *output, float *scales,
    const h3_int8_quant_args *args, uint32_t dispatch_rows,
    hipStream_t stream) {
    h3_int8_quant_args host_args = *args;
    void *kernel_args[] = {&input, &output, &scales, &host_args};
    /* 512 threads when wide rows (DiT FC / QKV); else 256. */
    uint32_t block_threads =
        host_args.columns >= 4096u && (host_args.columns % 4u) == 0u ? 512u
                                                                     : 256u;
    dim3 block(block_threads, 1, 1);
    dim3 grid(dispatch_rows, 1, 1);
    static int use_lds = -1;
    if (use_lds < 0) {
        const char *e = getenv("H3_INT8_QUANT_LDS");
        /* Opt-in only: LDS staging was slightly slower on the full linear. */
        use_lds = (e && strcmp(e, "0") != 0) ? 1 : 0;
    }
    size_t smem = (size_t)host_args.columns * sizeof(uint16_t);
    int lds = use_lds && (host_args.columns % 4u) == 0u && smem > 0 &&
              smem <= 49152u;
    const void *kernel = lds ?
        reinterpret_cast<const void *>(h3_quantize_bf16_int8_rows_lds_kernel) :
        reinterpret_cast<const void *>(h3_quantize_bf16_int8_rows_kernel);
    size_t dyn = lds ? smem : sizeof(kernel_args);
    static hipEvent_t split_start = NULL, split_stop = NULL;
    int split = h3_int8_split_on();
    if (split) h3_int8_split_begin(&split_start, &split_stop, stream);
    int ok = hipLaunchKernel(kernel, grid, block, kernel_args, dyn, stream) ==
             hipSuccess;
    if (split && ok)
        h3_int8_split_end(split_start, split_stop, stream,
                          lds ? "quantize-lds" : "quantize");
    return ok;
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

/* ===========================================================================
 * FP8 E4M3 FNUZ quantize kernel (gfx942 / CDNA3 only).
 *
 * FP8 E4M3 FNUZ has a max representable value of 240.0.
 * Quantization: fp8_val = clamp(round(bf16_val * inverse), -240, 240)
 * Dequantization: bf16_val = fp8_val * scale
 * =========================================================================== */
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_fp8.h>
#endif

__global__ void h3_quantize_bf16_fp8_rows_kernel(
    const uint16_t *input, void *output, float *scales,
    h3_int8_quant_args args) {
    __shared__ float warp_max[16];
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t threads = blockDim.x;
    uint32_t base = row * args.columns;
    if (row >= args.rows) {
        for (uint32_t column = tid; column < args.columns; column += threads)
            ((uint8_t *)output)[base + column] = 0;
        if (tid == 0) scales[row] = 1.0f;
        return;
    }
    float local_max = 0.0f;
    if ((args.columns & 3u) == 0u) {
        for (uint32_t column = tid * 4u; column < args.columns;
             column += threads * 4u) {
            uint2 packed =
                *reinterpret_cast<const uint2 *>(input + base + column);
            uint16_t a0 = (uint16_t)(packed.x & 0xffffu);
            uint16_t a1 = (uint16_t)(packed.x >> 16);
            uint16_t a2 = (uint16_t)(packed.y & 0xffffu);
            uint16_t a3 = (uint16_t)(packed.y >> 16);
            local_max = fmaxf(local_max, fabsf(h3_bf16_to_f32(a0)));
            local_max = fmaxf(local_max, fabsf(h3_bf16_to_f32(a1)));
            local_max = fmaxf(local_max, fabsf(h3_bf16_to_f32(a2)));
            local_max = fmaxf(local_max, fabsf(h3_bf16_to_f32(a3)));
        }
    } else {
        for (uint32_t column = tid; column < args.columns; column += threads)
            local_max = fmaxf(local_max,
                              fabsf(h3_bf16_to_f32(input[base + column])));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, h3_shfl_xor32(local_max, offset));
    if ((tid & 31u) == 0u) warp_max[tid >> 5] = local_max;
    __syncthreads();
    if (tid < 16u) {
        float value = (tid < (threads >> 5)) ? warp_max[tid] : 0.0f;
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1)
            value = fmaxf(value, h3_shfl_xor32(value, offset));
        if (tid == 0) warp_max[0] = value;
    }
    __syncthreads();
    float clipped_max = warp_max[0] * args.clip;
    /* FP8 E4M3 FNUZ max is 240.0, not 127.0 as INT8. */
    static const float FP8_MAX = 240.0f;
    float scale = clipped_max > 0.0f ? clipped_max / FP8_MAX : 1.0f / FP8_MAX;
    float inverse = clipped_max > 0.0f ? FP8_MAX / clipped_max : FP8_MAX;
    if (tid == 0) scales[row] = scale;
    if ((args.columns & 3u) == 0u) {
        for (uint32_t column = tid * 4u; column < args.columns;
             column += threads * 4u) {
            uint2 packed =
                *reinterpret_cast<const uint2 *>(input + base + column);
            uint16_t vals[4] = {
                (uint16_t)(packed.x & 0xffffu), (uint16_t)(packed.x >> 16),
                (uint16_t)(packed.y & 0xffffu), (uint16_t)(packed.y >> 16)};
            uint32_t qpack = 0;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float val = h3_bf16_to_f32(vals[i]) * inverse;
                if (val > FP8_MAX) val = FP8_MAX;
                if (val < -FP8_MAX) val = -FP8_MAX;
                __hip_fp8_e4m3_fnuz fp8_val = (__hip_fp8_e4m3_fnuz)val;
                qpack |= (uint32_t)(fp8_val.__x) << (8 * i);
            }
            *reinterpret_cast<uint32_t *>((uint8_t *)output + base + column) = qpack;
        }
    } else {
        for (uint32_t column = tid; column < args.columns; column += threads) {
            float val = h3_bf16_to_f32(input[base + column]) * inverse;
            if (val > FP8_MAX) val = FP8_MAX;
            if (val < -FP8_MAX) val = -FP8_MAX;
            __hip_fp8_e4m3_fnuz fp8_val = (__hip_fp8_e4m3_fnuz)val;
            ((uint8_t *)output)[base + column] = fp8_val.__x;
        }
    }
}

extern "C" int h3_launch_quantize_bf16_fp8_rows(
    const uint16_t *input, void *output, float *scales,
    const h3_int8_quant_args *args, uint32_t dispatch_rows,
    hipStream_t stream) {
    h3_int8_quant_args host_args = *args;
    void *kernel_args[] = {&input, &output, &scales, &host_args};
    uint32_t block_threads =
        host_args.columns >= 4096u && (host_args.columns % 4u) == 0u ? 512u
                                                                     : 256u;
    dim3 block(block_threads, 1, 1);
    dim3 grid(dispatch_rows, 1, 1);
    return hipLaunchKernel(
               reinterpret_cast<const void *>(
                   h3_quantize_bf16_fp8_rows_kernel),
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

__device__ inline float h3_fast_exp(float x) {
    return exp2f(x * 1.4426950408889634f);
}

__device__ inline float h3_silu(float gate) {
    return gate * (1.0f / (1.0f + h3_fast_exp(-gate)));
}

__device__ inline int32_t h3_i8dot4_scalar(int32_t a, int32_t b, int32_t acc) {
    acc += (int32_t)(int8_t)a * (int32_t)(int8_t)b;
    acc += (int32_t)(int8_t)(a >> 8) * (int32_t)(int8_t)(b >> 8);
    acc += (int32_t)(int8_t)(a >> 16) * (int32_t)(int8_t)(b >> 16);
    acc += (int32_t)(int8_t)(a >> 24) * (int32_t)(int8_t)(b >> 24);
    return acc;
}

/* gfx11/1151: packed signed int8 dot. CDNA: sdot4. Scalar fallback otherwise. */
__device__ inline int32_t h3_i8dot4(int32_t a, int32_t b, int32_t acc) {
#if defined(__gfx1151__) || defined(__gfx1150__) || defined(__gfx1100__) || \
    defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    return __builtin_amdgcn_sudot4(true, a, true, b, acc, false);
#elif defined(__has_builtin) && __has_builtin(__builtin_amdgcn_sdot4)
    return __builtin_amdgcn_sdot4(a, b, acc, false);
#else
    return h3_i8dot4_scalar(a, b, acc);
#endif
}

template <int BM, int LDS_K, int BK>
__device__ inline void h3_load_i8_tile_bmxk(
    int8_t dst[BM][LDS_K], const int8_t *src, uint32_t block_index, uint32_t k0,
    uint32_t bound, uint32_t stride, uint32_t tid) {
    constexpr uint32_t tpr = 256u / (uint32_t)BM;
    constexpr uint32_t chunks = (uint32_t)BK / (tpr * 8u);
    uint32_t local_row = tid / tpr;
    uint32_t index = block_index + local_row;
#pragma unroll
    for (uint32_t chunk = 0; chunk < chunks; chunk++) {
        uint32_t local_k = (tid % tpr) * 8u + chunk * (tpr * 8u);
        int64_t packed = 0;
        if (index < bound)
            packed = *reinterpret_cast<const int64_t *>(
                src + (size_t)index * stride + k0 + local_k);
        *reinterpret_cast<int64_t *>(&dst[local_row][local_k]) = packed;
    }
}

template <int LDS_K, int BK>
__device__ inline void h3_load_i8_tile_64xk(
    int8_t dst[64][LDS_K], const int8_t *src, uint32_t block_index, uint32_t k0,
    uint32_t bound, uint32_t stride, uint32_t tid) {
    h3_load_i8_tile_bmxk<64, LDS_K, BK>(dst, src, block_index, k0, bound,
                                         stride, tid);
}

template <int LDS_K>
__device__ inline void h3_load_i8_tile_64x32(
    int8_t dst[64][LDS_K], const int8_t *src, uint32_t block_index, uint32_t k0,
    uint32_t bound, uint32_t stride, uint32_t tid) {
    h3_load_i8_tile_64xk<LDS_K, 32>(dst, src, block_index, k0, bound, stride,
                                     tid);
}

template <int K_CONST, int BK>
__global__ void __launch_bounds__(256)
h3_linear_int8_r64_kernel(const int8_t *input, const int8_t *weight,
                          const float *input_scales, const float *weight_scales,
                          uint16_t *output, h3_linear_args args) {
    constexpr int BM = 64, BN = 64;
    static_assert(BK == 32 || BK == 64 || BK == 128, "INT8 BK 32/64/128");
    /* +4 pads each LDS row so int32 column reads avoid 32-bank collisions. */
    constexpr int LDS_K = BK + 4;
    __shared__ int8_t As[BM][LDS_K];
    __shared__ int8_t Bs[BN][LDS_K];
    __shared__ float scale_a[BM];
    __shared__ float scale_b[BN];
    uint32_t block_row = blockIdx.y * (uint32_t)BM;
    uint32_t block_col = blockIdx.x * (uint32_t)BN;
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t tid = ty * 16u + tx;
    uint32_t input_dim = K_CONST ? (uint32_t)K_CONST : args.input_dim;
    uint32_t k_tiles = input_dim / (uint32_t)BK;
    if (tid < (uint32_t)BM) {
        uint32_t row = block_row + tid;
        scale_a[tid] = row < args.rows ? input_scales[row] : 0.0f;
    }
    if (tid < (uint32_t)BN) {
        uint32_t column = block_col + tid;
        scale_b[tid] =
            column < args.output_dim ? weight_scales[column] : 0.0f;
    }
    int32_t acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0;
    for (uint32_t kt = 0; kt < k_tiles; kt++) {
        uint32_t k0 = kt * (uint32_t)BK;
        h3_load_i8_tile_64xk<LDS_K, BK>(As, input, block_row, k0, args.rows,
                                         input_dim, tid);
        h3_load_i8_tile_64xk<LDS_K, BK>(Bs, weight, block_col, k0,
                                         args.output_dim, input_dim, tid);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; k += 4) {
            int32_t av[4], bv[4];
#pragma unroll
            for (int i = 0; i < 4; i++)
                av[i] = *reinterpret_cast<const int32_t *>(
                    &As[ty * 4u + (uint32_t)i][k]);
#pragma unroll
            for (int j = 0; j < 4; j++)
                bv[j] = *reinterpret_cast<const int32_t *>(
                    &Bs[tx * 4u + (uint32_t)j][k]);
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++)
                    acc[i][j] = h3_i8dot4(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t row = block_row + ty * 4u + (uint32_t)i;
        if (row >= args.rows) continue;
        uint32_t column0 = block_col + tx * 4u;
        if (column0 + 3u < args.output_dim) {
            float sa = scale_a[ty * 4u + (uint32_t)i];
            uint16_t o0 = h3_f32_to_bf16((float)acc[i][0] * sa *
                                         scale_b[tx * 4u + 0u]);
            uint16_t o1 = h3_f32_to_bf16((float)acc[i][1] * sa *
                                         scale_b[tx * 4u + 1u]);
            uint16_t o2 = h3_f32_to_bf16((float)acc[i][2] * sa *
                                         scale_b[tx * 4u + 2u]);
            uint16_t o3 = h3_f32_to_bf16((float)acc[i][3] * sa *
                                         scale_b[tx * 4u + 3u]);
            uint2 packed;
            packed.x = (uint32_t)o0 | ((uint32_t)o1 << 16);
            packed.y = (uint32_t)o2 | ((uint32_t)o3 << 16);
            *reinterpret_cast<uint2 *>(output + (size_t)row * args.output_dim +
                                       column0) = packed;
        } else {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t column = column0 + (uint32_t)j;
                if (column >= args.output_dim) continue;
                float value = (float)acc[i][j] * scale_a[ty * 4u + (uint32_t)i] *
                              scale_b[tx * 4u + (uint32_t)j];
                output[row * args.output_dim + column] = h3_f32_to_bf16(value);
            }
        }
    }
}

/* 128×128 output tile / 8×8 sudot4. Opt out: H3_INT8_R64=1. */
template <int K_CONST, int BK>
__global__ void __launch_bounds__(256, 2)
h3_linear_int8_t128_kernel(const int8_t *input, const int8_t *weight,
                           const float *input_scales, const float *weight_scales,
                           uint16_t *output, h3_linear_args args) {
    constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
    static_assert(BK == 32 || BK == 64 || BK == 128, "INT8 BK 32/64/128");
    constexpr int LDS_K = BK + 4;
    __shared__ int8_t As[BM][LDS_K];
    __shared__ int8_t Bs[BN][LDS_K];
    __shared__ float scale_a[BM];
    __shared__ float scale_b[BN];
    uint32_t block_row = blockIdx.y * (uint32_t)BM;
    uint32_t block_col = blockIdx.x * (uint32_t)BN;
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t tid = ty * 16u + tx;
    uint32_t input_dim = K_CONST ? (uint32_t)K_CONST : args.input_dim;
    uint32_t k_tiles = input_dim / (uint32_t)BK;
    if (tid < (uint32_t)BM) {
        uint32_t row = block_row + tid;
        scale_a[tid] = row < args.rows ? input_scales[row] : 0.0f;
    }
    if (tid < (uint32_t)BN) {
        uint32_t column = block_col + tid;
        scale_b[tid] =
            column < args.output_dim ? weight_scales[column] : 0.0f;
    }
    int32_t acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0;
    for (uint32_t kt = 0; kt < k_tiles; kt++) {
        uint32_t k0 = kt * (uint32_t)BK;
        h3_load_i8_tile_bmxk<BM, LDS_K, BK>(As, input, block_row, k0, args.rows,
                                             input_dim, tid);
        h3_load_i8_tile_bmxk<BN, LDS_K, BK>(Bs, weight, block_col, k0,
                                             args.output_dim, input_dim, tid);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; k += 16) {
            int32_t av0[TM], av1[TM], av2[TM], av3[TM];
            int32_t bv0[TN], bv1[TN], bv2[TN], bv3[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) {
                uint64_t p0 = *reinterpret_cast<const uint64_t *>(
                    &As[ty * (uint32_t)TM + (uint32_t)i][k]);
                uint64_t p1 = *reinterpret_cast<const uint64_t *>(
                    &As[ty * (uint32_t)TM + (uint32_t)i][k + 8]);
                av0[i] = (int32_t)p0;
                av1[i] = (int32_t)(p0 >> 32);
                av2[i] = (int32_t)p1;
                av3[i] = (int32_t)(p1 >> 32);
            }
#pragma unroll
            for (int j = 0; j < TN; j++) {
                uint64_t p0 = *reinterpret_cast<const uint64_t *>(
                    &Bs[tx * (uint32_t)TN + (uint32_t)j][k]);
                uint64_t p1 = *reinterpret_cast<const uint64_t *>(
                    &Bs[tx * (uint32_t)TN + (uint32_t)j][k + 8]);
                bv0[j] = (int32_t)p0;
                bv1[j] = (int32_t)(p0 >> 32);
                bv2[j] = (int32_t)p1;
                bv3[j] = (int32_t)(p1 >> 32);
            }
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i][j] = h3_i8dot4(av0[i], bv0[j], acc[i][j]);
                    acc[i][j] = h3_i8dot4(av1[i], bv1[j], acc[i][j]);
                    acc[i][j] = h3_i8dot4(av2[i], bv2[j], acc[i][j]);
                    acc[i][j] = h3_i8dot4(av3[i], bv3[j], acc[i][j]);
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; i++) {
        uint32_t row = block_row + ty * (uint32_t)TM + (uint32_t)i;
        if (row >= args.rows) continue;
        uint32_t column0 = block_col + tx * (uint32_t)TN;
        if (column0 + (uint32_t)TN - 1u < args.output_dim) {
            float sa = scale_a[ty * (uint32_t)TM + (uint32_t)i];
            uint16_t outs[TN];
#pragma unroll
            for (int j = 0; j < TN; j++)
                outs[j] = h3_f32_to_bf16(
                    (float)acc[i][j] * sa *
                    scale_b[tx * (uint32_t)TN + (uint32_t)j]);
#pragma unroll
            for (int v = 0; v < TN / 4; v++) {
                uint2 packed;
                packed.x = (uint32_t)outs[v * 4 + 0] |
                           ((uint32_t)outs[v * 4 + 1] << 16);
                packed.y = (uint32_t)outs[v * 4 + 2] |
                           ((uint32_t)outs[v * 4 + 3] << 16);
                *reinterpret_cast<uint2 *>(
                    output + (size_t)row * args.output_dim + column0 +
                    (uint32_t)v * 4u) = packed;
            }
        } else {
#pragma unroll
            for (int j = 0; j < TN; j++) {
                uint32_t column = column0 + (uint32_t)j;
                if (column >= args.output_dim) continue;
                float value = (float)acc[i][j] *
                              scale_a[ty * (uint32_t)TM + (uint32_t)i] *
                              scale_b[tx * (uint32_t)TN + (uint32_t)j];
                output[row * args.output_dim + column] = h3_f32_to_bf16(value);
            }
        }
    }
}

template <int K_CONST, int BK>
__global__ void __launch_bounds__(256)
h3_fc1_swiglu_int8_r64_kernel(const int8_t *input, const int8_t *weight,
                              const float *input_scales,
                              const float *weight_scales, uint16_t *output,
                              h3_linear_args args) {
    constexpr int BM = 64, BN = 64;
    static_assert(BK == 32 || BK == 64 || BK == 128, "INT8 BK 32/64/128");
    constexpr int LDS_K = BK + 4;
    __shared__ int8_t As[BM][LDS_K];
    __shared__ int8_t gate_bs[BN][LDS_K];
    __shared__ int8_t up_bs[BN][LDS_K];
    __shared__ float scale_a[BM];
    __shared__ float gate_scale[BN];
    __shared__ float up_scale[BN];
    uint32_t block_row = blockIdx.y * (uint32_t)BM;
    uint32_t block_col = blockIdx.x * (uint32_t)BN;
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t tid = ty * 16u + tx;
    uint32_t input_dim = K_CONST ? (uint32_t)K_CONST : args.input_dim;
    uint32_t hidden_dim = args.output_dim;
    uint32_t k_tiles = input_dim / (uint32_t)BK;
    if (tid < (uint32_t)BM) {
        uint32_t row = block_row + tid;
        scale_a[tid] = row < args.rows ? input_scales[row] : 0.0f;
    }
    if (tid < (uint32_t)BN) {
        uint32_t column = block_col + tid;
        gate_scale[tid] =
            column < hidden_dim ? weight_scales[column] : 0.0f;
        up_scale[tid] = column < hidden_dim ?
            weight_scales[hidden_dim + column] : 0.0f;
    }
    int32_t gate_acc[4][4];
    int32_t up_acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            gate_acc[i][j] = 0;
            up_acc[i][j] = 0;
        }
    for (uint32_t kt = 0; kt < k_tiles; kt++) {
        uint32_t k0 = kt * (uint32_t)BK;
        h3_load_i8_tile_64xk<LDS_K, BK>(As, input, block_row, k0, args.rows,
                                         input_dim, tid);
        h3_load_i8_tile_64xk<LDS_K, BK>(gate_bs, weight, block_col, k0,
                                         hidden_dim, input_dim, tid);
        h3_load_i8_tile_64xk<LDS_K, BK>(
            up_bs, weight + (size_t)hidden_dim * input_dim, block_col, k0,
            hidden_dim, input_dim, tid);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; k += 4) {
            int32_t av[4], gv[4], uv[4];
#pragma unroll
            for (int i = 0; i < 4; i++)
                av[i] = *reinterpret_cast<const int32_t *>(
                    &As[ty * 4u + (uint32_t)i][k]);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                gv[j] = *reinterpret_cast<const int32_t *>(
                    &gate_bs[tx * 4u + (uint32_t)j][k]);
                uv[j] = *reinterpret_cast<const int32_t *>(
                    &up_bs[tx * 4u + (uint32_t)j][k]);
            }
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    gate_acc[i][j] = h3_i8dot4(av[i], gv[j], gate_acc[i][j]);
                    up_acc[i][j] = h3_i8dot4(av[i], uv[j], up_acc[i][j]);
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t row = block_row + ty * 4u + (uint32_t)i;
        if (row >= args.rows) continue;
        uint32_t column0 = block_col + tx * 4u;
        if (column0 + 3u < hidden_dim) {
            float sa = scale_a[ty * 4u + (uint32_t)i];
            uint16_t outs[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                float gate =
                    (float)gate_acc[i][j] * sa * gate_scale[tx * 4u + (uint32_t)j];
                float up =
                    (float)up_acc[i][j] * sa * up_scale[tx * 4u + (uint32_t)j];
                outs[j] = h3_f32_to_bf16(h3_silu(gate) * up);
            }
            uint2 packed;
            packed.x = (uint32_t)outs[0] | ((uint32_t)outs[1] << 16);
            packed.y = (uint32_t)outs[2] | ((uint32_t)outs[3] << 16);
            *reinterpret_cast<uint2 *>(output + (size_t)row * hidden_dim +
                                       column0) = packed;
        } else {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                uint32_t column = column0 + (uint32_t)j;
                if (column >= hidden_dim) continue;
                float gate = (float)gate_acc[i][j] * scale_a[ty * 4u + (uint32_t)i] *
                             gate_scale[tx * 4u + (uint32_t)j];
                float up = (float)up_acc[i][j] * scale_a[ty * 4u + (uint32_t)i] *
                           up_scale[tx * 4u + (uint32_t)j];
                output[row * hidden_dim + column] =
                    h3_f32_to_bf16(h3_silu(gate) * up);
            }
        }
    }
}

/* fc1 SwiGLU: 128×64 output, two 64-wide B tiles. Not 128×128 (rejected).
 * Opt out: H3_INT8_FC1_R64=1. */
template <int K_CONST, int BK>
__global__ void __launch_bounds__(256)
h3_fc1_swiglu_int8_r128x64_kernel(const int8_t *input, const int8_t *weight,
                                  const float *input_scales,
                                  const float *weight_scales, uint16_t *output,
                                  h3_linear_args args) {
    constexpr int BM = 128, BN = 64, TM = 8, TN = 4;
    static_assert(BK == 32 || BK == 64 || BK == 128, "INT8 BK 32/64/128");
    constexpr int LDS_K = BK + 4;
    __shared__ int8_t As[BM][LDS_K];
    __shared__ int8_t gate_bs[BN][LDS_K];
    __shared__ int8_t up_bs[BN][LDS_K];
    __shared__ float scale_a[BM];
    __shared__ float gate_scale[BN];
    __shared__ float up_scale[BN];
    uint32_t block_row = blockIdx.y * (uint32_t)BM;
    uint32_t block_col = blockIdx.x * (uint32_t)BN;
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t tid = ty * 16u + tx;
    uint32_t input_dim = K_CONST ? (uint32_t)K_CONST : args.input_dim;
    uint32_t hidden_dim = args.output_dim;
    uint32_t k_tiles = input_dim / (uint32_t)BK;
    if (tid < (uint32_t)BM) {
        uint32_t row = block_row + tid;
        scale_a[tid] = row < args.rows ? input_scales[row] : 0.0f;
    }
    if (tid < (uint32_t)BN) {
        uint32_t column = block_col + tid;
        gate_scale[tid] =
            column < hidden_dim ? weight_scales[column] : 0.0f;
        up_scale[tid] = column < hidden_dim ?
            weight_scales[hidden_dim + column] : 0.0f;
    }
    int32_t gate_acc[TM][TN];
    int32_t up_acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) {
            gate_acc[i][j] = 0;
            up_acc[i][j] = 0;
        }
    for (uint32_t kt = 0; kt < k_tiles; kt++) {
        uint32_t k0 = kt * (uint32_t)BK;
        h3_load_i8_tile_bmxk<BM, LDS_K, BK>(As, input, block_row, k0, args.rows,
                                             input_dim, tid);
        h3_load_i8_tile_bmxk<BN, LDS_K, BK>(gate_bs, weight, block_col, k0,
                                             hidden_dim, input_dim, tid);
        h3_load_i8_tile_bmxk<BN, LDS_K, BK>(
            up_bs, weight + (size_t)hidden_dim * input_dim, block_col, k0,
            hidden_dim, input_dim, tid);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; k += 16) {
            int32_t av0[TM], av1[TM], av2[TM], av3[TM];
            int32_t gv0[TN], gv1[TN], gv2[TN], gv3[TN];
            int32_t uv0[TN], uv1[TN], uv2[TN], uv3[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) {
                uint64_t p0 = *reinterpret_cast<const uint64_t *>(
                    &As[ty * (uint32_t)TM + (uint32_t)i][k]);
                uint64_t p1 = *reinterpret_cast<const uint64_t *>(
                    &As[ty * (uint32_t)TM + (uint32_t)i][k + 8]);
                av0[i] = (int32_t)p0;
                av1[i] = (int32_t)(p0 >> 32);
                av2[i] = (int32_t)p1;
                av3[i] = (int32_t)(p1 >> 32);
            }
#pragma unroll
            for (int j = 0; j < TN; j++) {
                uint64_t gp0 = *reinterpret_cast<const uint64_t *>(
                    &gate_bs[tx * (uint32_t)TN + (uint32_t)j][k]);
                uint64_t gp1 = *reinterpret_cast<const uint64_t *>(
                    &gate_bs[tx * (uint32_t)TN + (uint32_t)j][k + 8]);
                uint64_t up0 = *reinterpret_cast<const uint64_t *>(
                    &up_bs[tx * (uint32_t)TN + (uint32_t)j][k]);
                uint64_t up1 = *reinterpret_cast<const uint64_t *>(
                    &up_bs[tx * (uint32_t)TN + (uint32_t)j][k + 8]);
                gv0[j] = (int32_t)gp0;
                gv1[j] = (int32_t)(gp0 >> 32);
                gv2[j] = (int32_t)gp1;
                gv3[j] = (int32_t)(gp1 >> 32);
                uv0[j] = (int32_t)up0;
                uv1[j] = (int32_t)(up0 >> 32);
                uv2[j] = (int32_t)up1;
                uv3[j] = (int32_t)(up1 >> 32);
            }
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    gate_acc[i][j] = h3_i8dot4(av0[i], gv0[j], gate_acc[i][j]);
                    up_acc[i][j] = h3_i8dot4(av0[i], uv0[j], up_acc[i][j]);
                    gate_acc[i][j] = h3_i8dot4(av1[i], gv1[j], gate_acc[i][j]);
                    up_acc[i][j] = h3_i8dot4(av1[i], uv1[j], up_acc[i][j]);
                    gate_acc[i][j] = h3_i8dot4(av2[i], gv2[j], gate_acc[i][j]);
                    up_acc[i][j] = h3_i8dot4(av2[i], uv2[j], up_acc[i][j]);
                    gate_acc[i][j] = h3_i8dot4(av3[i], gv3[j], gate_acc[i][j]);
                    up_acc[i][j] = h3_i8dot4(av3[i], uv3[j], up_acc[i][j]);
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; i++) {
        uint32_t row = block_row + ty * (uint32_t)TM + (uint32_t)i;
        if (row >= args.rows) continue;
        uint32_t column0 = block_col + tx * (uint32_t)TN;
        if (column0 + (uint32_t)TN - 1u < hidden_dim) {
            float sa = scale_a[ty * (uint32_t)TM + (uint32_t)i];
            uint16_t outs[TN];
#pragma unroll
            for (int j = 0; j < TN; j++) {
                float gate = (float)gate_acc[i][j] * sa *
                             gate_scale[tx * (uint32_t)TN + (uint32_t)j];
                float up = (float)up_acc[i][j] * sa *
                           up_scale[tx * (uint32_t)TN + (uint32_t)j];
                outs[j] = h3_f32_to_bf16(h3_silu(gate) * up);
            }
            uint2 packed;
            packed.x = (uint32_t)outs[0] | ((uint32_t)outs[1] << 16);
            packed.y = (uint32_t)outs[2] | ((uint32_t)outs[3] << 16);
            *reinterpret_cast<uint2 *>(output + (size_t)row * hidden_dim +
                                       column0) = packed;
        } else {
#pragma unroll
            for (int j = 0; j < TN; j++) {
                uint32_t column = column0 + (uint32_t)j;
                if (column >= hidden_dim) continue;
                float gate =
                    (float)gate_acc[i][j] *
                    scale_a[ty * (uint32_t)TM + (uint32_t)i] *
                    gate_scale[tx * (uint32_t)TN + (uint32_t)j];
                float up = (float)up_acc[i][j] *
                           scale_a[ty * (uint32_t)TM + (uint32_t)i] *
                           up_scale[tx * (uint32_t)TN + (uint32_t)j];
                output[row * hidden_dim + column] =
                    h3_f32_to_bf16(h3_silu(gate) * up);
            }
        }
    }
}

static hipblasHandle_t h3_hipblas = NULL;
static int32_t *h3_int8_i32_scratch = NULL;
static size_t h3_int8_i32_scratch_bytes = 0;
static float *h3_int8_f32_scratch = NULL;
static size_t h3_int8_f32_scratch_bytes = 0;

static int h3_int8_ensure_hipblas(hipStream_t stream) {
    if (!h3_hipblas && hipblasCreate(&h3_hipblas) != HIPBLAS_STATUS_SUCCESS)
        return 0;
    return hipblasSetStream(h3_hipblas, stream) == HIPBLAS_STATUS_SUCCESS;
}

__global__ void __launch_bounds__(256)
h3_bias_f32_kernel(float *output, const float *bias, uint32_t rows,
                   uint32_t cols) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= rows || col >= cols) return;
    output[(size_t)row * cols + col] += bias[col];
}

/* Same layout as INT8 hipBLAS: C[M,N] = A[M,K] * W[N,K]^T (row-major). */
static int h3_f32_hipblas_linear(const float *input, const float *weight,
                                 const float *bias, float *output,
                                 const h3_linear_args *args,
                                 hipStream_t stream) {
    if (!h3_int8_ensure_hipblas(stream)) return 0;
    int m = (int)args->output_dim;
    int n = (int)args->rows;
    int k = (int)args->input_dim;
    const float alpha = 1.0f, beta = 0.0f;
    hipblasStatus_t st =
        hipblasSgemm(h3_hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, &alpha,
                     weight, k, input, k, &beta, output, m);
    if (st != HIPBLAS_STATUS_SUCCESS) {
        if (h3_f32_stats_on())
            fprintf(stderr, "f32-linear hipblasSgemm status=%d M=%d N=%d K=%d\n",
                    (int)st, n, m, k);
        return 0;
    }
    if (!args->has_bias || !bias) return 1;
    uint32_t rows = args->rows, cols = args->output_dim;
    dim3 block(256, 1, 1);
    dim3 grid((cols + 255u) / 256u, rows, 1);
    void *bias_args[] = {&output, &bias, &rows, &cols};
    return hipLaunchKernel(reinterpret_cast<const void *>(h3_bias_f32_kernel),
                           grid, block, bias_args, 0, stream) == hipSuccess;
}

static int h3_bf16_hipblas_on(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("H3_BF16_HIPBLAS");
        if (e && strcmp(e, "0") == 0) cached = 0;
        else if (e && strcmp(e, "1") == 0) cached = 1;
        else cached = !h3_hip_rdna_wmma_default();
    }
    return cached;
}

__global__ void __launch_bounds__(256)
h3_bias_bf16_kernel(uint16_t *output, const uint16_t *bias, uint32_t rows,
                    uint32_t cols) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= rows || col >= cols) return;
    size_t index = (size_t)row * cols + col;
    output[index] = h3_f32_to_bf16(h3_bf16_to_f32(output[index]) +
                                   h3_bf16_to_f32(bias[col]));
}

extern "C" int h3_try_linear_bf16_hipblas(const uint16_t *input,
                                          const uint16_t *weight,
                                          const uint16_t *bias,
                                          uint16_t *output,
                                          const h3_linear_args *args,
                                          hipStream_t stream) {
    if (!h3_bf16_hipblas_on() || !args) return 0;
    if (args->rows < 64u || args->input_dim < 64u || args->output_dim < 16u)
        return 0;
    if (args->input_dim % 2u) return 0;
    if (!h3_int8_ensure_hipblas(stream)) return 0;
    int m = (int)args->output_dim;
    int n = (int)args->rows;
    int k = (int)args->input_dim;
    const float alpha = 1.0f, beta = 0.0f;
    hipblasStatus_t st = hipblasGemmEx(
        h3_hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, &alpha, weight,
        HIP_R_16BF, k, input, HIP_R_16BF, k, &beta, output, HIP_R_16BF, m,
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    if (st != HIPBLAS_STATUS_SUCCESS) return 0;
    if (!args->has_bias || !bias) return 1;
    uint32_t rows = args->rows, cols = args->output_dim;
    dim3 block(256, 1, 1);
    dim3 grid((cols + 255u) / 256u, rows, 1);
    void *bias_args[] = {&output, &bias, &rows, &cols};
    return hipLaunchKernel(reinterpret_cast<const void *>(h3_bias_bf16_kernel),
                           grid, block, bias_args, 0, stream) == hipSuccess;
}

static float *h3_sdpa_scratch = NULL;
static size_t h3_sdpa_scratch_bytes = 0;

static int h3_sdpa_ensure_scratch(size_t bytes) {
    if (bytes <= h3_sdpa_scratch_bytes) return 1;
    float *next = NULL;
    if (hipMalloc(&next, bytes) != hipSuccess) return 0;
    if (h3_sdpa_scratch) (void)hipFree(h3_sdpa_scratch);
    h3_sdpa_scratch = next;
    h3_sdpa_scratch_bytes = bytes;
    return 1;
}

static int h3_sdpa_f32_hipblas_on(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("H3_SDPA_HIPBLAS");
        if (e && strcmp(e, "0") == 0) cached = 0;
        else if (e && strcmp(e, "1") == 0) cached = 1;
        else cached = !h3_hip_rdna_wmma_default();
    }
    return cached;
}

__global__ void h3_transpose_hsd_shd_f32_kernel(const float *input,
                                                float *output,
                                                uint32_t sequence,
                                                uint32_t heads,
                                                uint32_t head_dim) {
    uint32_t dimension = threadIdx.x;
    uint32_t head = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (dimension >= head_dim || head >= heads || row >= sequence) return;
    uint32_t src = (head * sequence + row) * head_dim + dimension;
    uint32_t dst = (row * heads + head) * head_dim + dimension;
    output[dst] = input[src];
}

__global__ void __launch_bounds__(256)
h3_softmax_rows_f32_kernel(float *scores, uint32_t cols) {
    uint32_t row = blockIdx.x;
    float *line = scores + (size_t)row * cols;
    float row_max = -1e30f;
    uint32_t vec = cols >> 2;
    int vectorized = (cols & 3u) == 0u;
    if (vectorized) {
        float4 *line4 = reinterpret_cast<float4 *>(line);
        for (uint32_t col = threadIdx.x; col < vec; col += blockDim.x) {
            float4 value = line4[col];
            row_max = fmaxf(row_max,
                            fmaxf(fmaxf(value.x, value.y),
                                  fmaxf(value.z, value.w)));
        }
    } else {
        for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x)
            row_max = fmaxf(row_max, line[col]);
    }
    __shared__ float red[256];
    red[threadIdx.x] = row_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        __syncthreads();
    }
    row_max = red[0];
    float sum = 0.0f;
    if (vectorized) {
        float4 *line4 = reinterpret_cast<float4 *>(line);
        for (uint32_t col = threadIdx.x; col < vec; col += blockDim.x) {
            float4 value = line4[col];
            value.x = expf(value.x - row_max);
            value.y = expf(value.y - row_max);
            value.z = expf(value.z - row_max);
            value.w = expf(value.w - row_max);
            line4[col] = value;
            sum += value.x + value.y + value.z + value.w;
        }
    } else {
        for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
            float p = expf(line[col] - row_max);
            line[col] = p;
            sum += p;
        }
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    float inv = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;
    if (vectorized) {
        float4 *line4 = reinterpret_cast<float4 *>(line);
        for (uint32_t col = threadIdx.x; col < vec; col += blockDim.x) {
            float4 value = line4[col];
            value.x *= inv;
            value.y *= inv;
            value.z *= inv;
            value.w *= inv;
            line4[col] = value;
        }
    } else {
        for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x)
            line[col] *= inv;
    }
}

extern "C" int h3_launch_sdpa_f32_hipblas(const float *query, const float *key,
                                          const float *value, float *output,
                                          const h3_sdpa_args *args,
                                          hipStream_t stream) {
    if (!h3_sdpa_f32_hipblas_on() || !args) return 0;
    uint32_t seq = args->sequence, heads = args->heads, dim = args->head_dim;
    if ((dim != 64u && dim != 128u) || seq < 128u || seq > 8192u || !heads)
        return 0;
    if (!h3_int8_ensure_hipblas(stream)) return 0;
    size_t hs_d = (size_t)heads * seq * dim;
    size_t hs_s = (size_t)heads * seq * seq;
    if (hs_d > UINT32_MAX || hs_s > UINT32_MAX) return 0;
    int need_k = !args->kv_head_major;
    size_t bytes = (hs_d + hs_s + hs_d) * sizeof(float);
    if (need_k) bytes += 2u * hs_d * sizeof(float);
    if (bytes > (size_t)2u << 30) return 0;
    if (!h3_sdpa_ensure_scratch(bytes)) return 0;
    float *q_hm = h3_sdpa_scratch;
    float *scores = q_hm + hs_d;
    float *out_hm = scores + hs_s;
    const float *k_hm = key;
    const float *v_hm = value;
    if (need_k) {
        float *k_buf = out_hm + hs_d;
        float *v_buf = k_buf + hs_d;
        if (!h3_launch_transpose_shd_hsd_f32(key, k_buf, seq, heads, dim,
                                             stream) ||
            !h3_launch_transpose_shd_hsd_f32(value, v_buf, seq, heads, dim,
                                             stream)) {
            return 0;
        }
        k_hm = k_buf;
        v_hm = v_buf;
    }
    if (!h3_launch_transpose_shd_hsd_f32(query, q_hm, seq, heads, dim, stream))
        return 0;
    int m = (int)seq, n = (int)seq, k = (int)dim;
    const float alpha = args->scale, beta = 0.0f;
    hipblasStride stride_q = (hipblasStride)((size_t)seq * dim);
    hipblasStride stride_s = (hipblasStride)((size_t)seq * seq);
    if (hipblasSgemmStridedBatched(h3_hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n,
                                   k, &alpha, k_hm, k, stride_q, q_hm, k,
                                   stride_q, &beta, scores, m, stride_s,
                                   (int)heads) != HIPBLAS_STATUS_SUCCESS) {
        return 0;
    }
    dim3 soft_grid(heads * seq, 1, 1);
    void *soft_args[] = {&scores, &seq};
    if (hipLaunchKernel(reinterpret_cast<const void *>(h3_softmax_rows_f32_kernel),
                        soft_grid, dim3(256), soft_args, 0,
                        stream) != hipSuccess) {
        return 0;
    }
    const float one = 1.0f;
    if (hipblasSgemmStridedBatched(h3_hipblas, HIPBLAS_OP_N, HIPBLAS_OP_N,
                                   (int)dim, (int)seq, (int)seq, &one, v_hm,
                                   (int)dim, stride_q, scores, (int)seq,
                                   stride_s, &beta, out_hm, (int)dim, stride_q,
                                   (int)heads) != HIPBLAS_STATUS_SUCCESS) {
        return 0;
    }
    if (args->head_major_output) {
        return hipMemcpyAsync(output, out_hm, hs_d * sizeof(float),
                              hipMemcpyDeviceToDevice, stream) == hipSuccess;
    }
    void *tr_args[] = {&out_hm, &output, &seq, &heads, &dim};
    return hipLaunchKernel(
               reinterpret_cast<const void *>(h3_transpose_hsd_shd_f32_kernel),
               dim3(heads, seq, 1), dim3(dim, 1, 1), tr_args, 0,
               stream) == hipSuccess;
}

static float *h3_sdpa_bf16_scratch = NULL;
static size_t h3_sdpa_bf16_scratch_bytes = 0;
static unsigned char *h3_sdpa_bf16_direct_scratch = NULL;
static size_t h3_sdpa_bf16_direct_scratch_bytes = 0;
static hipblasHandle_t h3_sdpa_bf16_direct_hipblas = NULL;

static int h3_sdpa_bf16_ensure_scratch(size_t bytes) {
    if (bytes <= h3_sdpa_bf16_scratch_bytes) return 1;
    float *next = NULL;
    if (hipMalloc(&next, bytes) != hipSuccess) return 0;
    if (h3_sdpa_bf16_scratch) (void)hipFree(h3_sdpa_bf16_scratch);
    h3_sdpa_bf16_scratch = next;
    h3_sdpa_bf16_scratch_bytes = bytes;
    return 1;
}

static int h3_sdpa_bf16_direct_ensure_scratch(size_t bytes) {
    if (bytes <= h3_sdpa_bf16_direct_scratch_bytes) return 1;
    unsigned char *next = NULL;
    if (hipMalloc(&next, bytes) != hipSuccess) return 0;
    if (h3_sdpa_bf16_direct_scratch)
        (void)hipFree(h3_sdpa_bf16_direct_scratch);
    h3_sdpa_bf16_direct_scratch = next;
    h3_sdpa_bf16_direct_scratch_bytes = bytes;
    return 1;
}

__global__ void h3_scatter_f32_to_bf16_hsd_kernel(
    const float *input, uint16_t *output, uint32_t query_start,
    uint32_t query_rows, uint32_t sequence, uint32_t heads,
    uint32_t head_dim, uint32_t count) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t dimension = index % head_dim;
    uint32_t query = (index / head_dim) % query_rows;
    uint32_t head = index / (head_dim * query_rows);
    size_t destination =
        ((size_t)head * sequence + query_start + query) * head_dim + dimension;
    reinterpret_cast<__hip_bfloat16 *>(output)[destination] =
        (__hip_bfloat16)input[index];
}

__global__ void h3_scatter_f32_to_bf16_shd_kernel(
    const float *input, uint16_t *output, uint32_t query_start,
    uint32_t query_rows, uint32_t heads, uint32_t head_dim,
    uint32_t count) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint32_t dimension = index % head_dim;
    uint32_t query = (index / head_dim) % query_rows;
    uint32_t head = index / (head_dim * query_rows);
    size_t destination =
        ((size_t)(query_start + query) * heads + head) * head_dim + dimension;
    reinterpret_cast<__hip_bfloat16 *>(output)[destination] =
        (__hip_bfloat16)input[index];
}

static int h3_sdpa_bf16_direct_on(void) {
    const char *e = getenv("H3_SDPA_BF16_DIRECT");
    if (e) return strcmp(e, "0") != 0;
    static int cached = -1;
    if (cached >= 0) return cached;
    int device = 0;
    hipDeviceProp_t props;
    cached = hipGetDevice(&device) == hipSuccess &&
             hipGetDeviceProperties(&props, device) == hipSuccess &&
             (strncmp(props.gcnArchName, "gfx90a", 6) == 0 ||
              strncmp(props.gcnArchName, "gfx942", 6) == 0);
    return cached;
}

static int h3_sdpa_bf16_direct_ensure_hipblas(hipStream_t stream) {
    if (!h3_sdpa_bf16_direct_hipblas &&
        hipblasCreate(&h3_sdpa_bf16_direct_hipblas) != HIPBLAS_STATUS_SUCCESS) {
        return 0;
    }
    return hipblasSetStream(h3_sdpa_bf16_direct_hipblas, stream) ==
           HIPBLAS_STATUS_SUCCESS;
}

__global__ void h3_sdpa_online_init_kernel(float *acc, float *m, float *l,
                                           uint32_t dim) {
    uint32_t row = blockIdx.x;
    if (threadIdx.x == 0) {
        m[row] = -1e30f;
        l[row] = 0.0f;
    }
    float *line = acc + (size_t)row * dim;
    for (uint32_t col = threadIdx.x; col < dim; col += blockDim.x)
        line[col] = 0.0f;
}

__global__ void __launch_bounds__(256)
h3_sdpa_online_prepare_p_kernel(float *scores, float *acc, float *m, float *l,
                                uint32_t cols, uint32_t dim) {
    uint32_t row = blockIdx.x;
    float *line = scores + (size_t)row * cols;
    float row_max = -1e30f;
    uint32_t vec = cols >> 2;
    int vectorized = (cols & 3u) == 0u;
    if (vectorized) {
        float4 *line4 = reinterpret_cast<float4 *>(line);
        for (uint32_t col = threadIdx.x; col < vec; col += blockDim.x) {
            float4 value = line4[col];
            row_max = fmaxf(row_max,
                            fmaxf(fmaxf(value.x, value.y),
                                  fmaxf(value.z, value.w)));
        }
    } else {
        for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x)
            row_max = fmaxf(row_max, line[col]);
    }
    __shared__ float red[256];
    red[threadIdx.x] = row_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        __syncthreads();
    }
    row_max = red[0];
    float m_old = m[row];
    float m_new = fmaxf(m_old, row_max);
    float alpha = expf(m_old - m_new);
    float *out = acc + (size_t)row * dim;
    for (uint32_t col = threadIdx.x; col < dim; col += blockDim.x)
        out[col] *= alpha;
    float sum = 0.0f;
    if (vectorized) {
        float4 *line4 = reinterpret_cast<float4 *>(line);
        for (uint32_t col = threadIdx.x; col < vec; col += blockDim.x) {
            float4 value = line4[col];
            value.x = expf(value.x - m_new);
            value.y = expf(value.y - m_new);
            value.z = expf(value.z - m_new);
            value.w = expf(value.w - m_new);
            line4[col] = value;
            sum += value.x + value.y + value.z + value.w;
        }
    } else {
        for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
            float p = expf(line[col] - m_new);
            line[col] = p;
            sum += p;
        }
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride)
            red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        m[row] = m_new;
        l[row] = l[row] * alpha + red[0];
    }
}

__global__ void h3_sdpa_online_finalize_kernel(float *acc, const float *l,
                                               uint32_t dim) {
    uint32_t row = blockIdx.x;
    float inv = l[row] > 0.0f ? 1.0f / l[row] : 0.0f;
    float *line = acc + (size_t)row * dim;
    for (uint32_t col = threadIdx.x; col < dim; col += blockDim.x)
        line[col] *= inv;
}

/* CDNA matrix-core QK with F32 softmax and PV. Long sequences are split by
 * query row, bounding the score matrix without changing attention semantics.
 * Sequences above 8192 also tile the key axis with online softmax so the
 * score buffer stays cache-sized. */
static int h3_launch_sdpa_bf16_hipblas_direct(
    const uint16_t *query, const uint16_t *key, const uint16_t *value,
    uint16_t *output, const h3_sdpa_args *args, hipStream_t stream) {
    if (!h3_sdpa_bf16_direct_on() || !args) return 0;
    uint32_t seq = args->sequence, heads = args->heads, dim = args->head_dim;
    if ((dim != 64u && dim != 128u) || seq < 128u || !heads ||
        !args->kv_head_major)
        return 0;
    if (!h3_sdpa_bf16_direct_ensure_hipblas(stream)) return 0;
    size_t hs_d = (size_t)heads * seq * dim;
    if (hs_d > UINT32_MAX) return 0;
    uint32_t query_rows = seq;
    if (seq > 8192u) query_rows = 128u;
    const char *configured = getenv("H3_SDPA_BF16_QROWS");
    if (configured && *configured) {
        unsigned long parsed = strtoul(configured, NULL, 10);
        if (parsed && parsed <= seq) query_rows = (uint32_t)parsed;
    }
    uint32_t k_tile = seq;
    if (seq > 8192u) k_tile = 4096u;
    const char *k_configured = getenv("H3_SDPA_BF16_KTILE");
    if (k_configured && *k_configured) {
        unsigned long parsed = strtoul(k_configured, NULL, 10);
        if (parsed) k_tile = (uint32_t)parsed;
    }
    if (k_tile > seq) k_tile = seq;
    int online = k_tile < seq;
    int need_out = !args->head_major_output;
    size_t score_elements = (size_t)heads * query_rows * k_tile;
    size_t output_elements = (size_t)heads * query_rows * dim;
    size_t stat_elements = (size_t)heads * query_rows;
    if (score_elements > UINT32_MAX || output_elements > UINT32_MAX)
        return 0;
    size_t bytes = hs_d * sizeof(uint16_t) +
                   (hs_d + score_elements + output_elements) * sizeof(float);
    if (online) bytes += 2u * stat_elements * sizeof(float);
    if (!h3_sdpa_bf16_direct_ensure_scratch(bytes)) {
        return 0;
    }
    uint16_t *q_hm =
        reinterpret_cast<uint16_t *>(h3_sdpa_bf16_direct_scratch);
    float *v32 = reinterpret_cast<float *>(q_hm + hs_d);
    float *scores = v32 + hs_d;
    float *o32 = scores + score_elements;
    float *m_stats = online ? o32 + output_elements : NULL;
    float *l_stats = online ? m_stats + stat_elements : NULL;
    if (!h3_launch_transpose_shd_hsd_bf16(
            query, q_hm, seq, heads, dim, stream) ||
        !h3_launch_cast_bf16_to_f32(
            value, v32, (uint32_t)hs_d, stream)) {
        return 0;
    }
    const float alpha = args->scale, beta = 0.0f;
    hipblasStride stride_q = (hipblasStride)((size_t)seq * dim);
    const float one = 1.0f;
    for (uint32_t start = 0; start < seq; start += query_rows) {
        uint32_t rows = seq - start;
        if (rows > query_rows) rows = query_rows;
        uint32_t stat_rows = heads * rows;
        hipblasStride stride_o = (hipblasStride)((size_t)rows * dim);
        if (online) {
            void *init_args[] = {&o32, &m_stats, &l_stats, &dim};
            if (hipLaunchKernel(
                    reinterpret_cast<const void *>(h3_sdpa_online_init_kernel),
                    dim3(stat_rows), dim3(64), init_args, 0,
                    stream) != hipSuccess) {
                return 0;
            }
        }
        for (uint32_t k0 = 0; k0 < seq; k0 += k_tile) {
            uint32_t keys = seq - k0;
            if (keys > k_tile) keys = k_tile;
            hipblasStride stride_s = (hipblasStride)((size_t)keys * rows);
            hipblasStatus_t status = hipblasGemmStridedBatchedEx(
                h3_sdpa_bf16_direct_hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N,
                (int)keys, (int)rows, (int)dim, &alpha,
                key + (size_t)k0 * dim, HIP_R_16BF, (int)dim, stride_q,
                q_hm + (size_t)start * dim, HIP_R_16BF, (int)dim, stride_q,
                &beta, scores, HIP_R_32F, (int)keys, stride_s, (int)heads,
                HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
            if (status != HIPBLAS_STATUS_SUCCESS) return 0;
            if (online) {
                void *prep_args[] = {&scores, &o32, &m_stats, &l_stats, &keys,
                                     &dim};
                if (hipLaunchKernel(
                        reinterpret_cast<const void *>(
                            h3_sdpa_online_prepare_p_kernel),
                        dim3(stat_rows), dim3(256), prep_args, 0,
                        stream) != hipSuccess) {
                    return 0;
                }
            } else {
                void *soft_args[] = {&scores, &keys};
                if (hipLaunchKernel(
                        reinterpret_cast<const void *>(
                            h3_softmax_rows_f32_kernel),
                        dim3(stat_rows), dim3(256), soft_args, 0,
                        stream) != hipSuccess) {
                    return 0;
                }
            }
            const float *pv_beta = online ? &one : &beta;
            status = hipblasSgemmStridedBatched(
                h3_sdpa_bf16_direct_hipblas, HIPBLAS_OP_N, HIPBLAS_OP_N,
                (int)dim, (int)rows, (int)keys, &one,
                v32 + (size_t)k0 * dim, (int)dim, stride_q,
                scores, (int)keys, stride_s, pv_beta, o32, (int)dim, stride_o,
                (int)heads);
            if (status != HIPBLAS_STATUS_SUCCESS) return 0;
        }
        if (online) {
            void *fin_args[] = {&o32, &l_stats, &dim};
            if (hipLaunchKernel(
                    reinterpret_cast<const void *>(
                        h3_sdpa_online_finalize_kernel),
                    dim3(stat_rows), dim3(64), fin_args, 0,
                    stream) != hipSuccess) {
                return 0;
            }
        }
        uint32_t count = heads * rows * dim;
        dim3 scatter_grid((count + 255u) / 256u);
        if (need_out) {
            void *scatter_args[] = {&o32, &output, &start, &rows, &heads,
                                    &dim, &count};
            if (hipLaunchKernel(
                    reinterpret_cast<const void *>(
                        h3_scatter_f32_to_bf16_shd_kernel),
                    scatter_grid, dim3(256), scatter_args, 0,
                    stream) != hipSuccess) {
                return 0;
            }
        } else {
            void *scatter_args[] = {&o32, &output, &start, &rows, &seq,
                                    &heads, &dim, &count};
            if (hipLaunchKernel(
                    reinterpret_cast<const void *>(
                        h3_scatter_f32_to_bf16_hsd_kernel),
                    scatter_grid, dim3(256), scatter_args, 0,
                    stream) != hipSuccess) {
                return 0;
            }
        }
    }
    return 1;
}

extern "C" int h3_launch_sdpa_bf16_hipblas(const uint16_t *query,
                                           const uint16_t *key,
                                           const uint16_t *value,
                                           uint16_t *output,
                                           const h3_sdpa_args *args,
                                           hipStream_t stream) {
    if (!h3_sdpa_f32_hipblas_on() || !args) return 0;
    if (h3_launch_sdpa_bf16_hipblas_direct(
            query, key, value, output, args, stream)) {
        return 1;
    }
    uint32_t seq = args->sequence, heads = args->heads, dim = args->head_dim;
    if ((dim != 64u && dim != 128u) || seq < 128u || seq > 8192u || !heads)
        return 0;
    size_t n = (size_t)seq * heads * dim;
    if (!h3_sdpa_bf16_ensure_scratch(n * 4u * sizeof(float))) return 0;
    float *q32 = h3_sdpa_bf16_scratch;
    float *k32 = q32 + n;
    float *v32 = k32 + n;
    float *o32 = v32 + n;
    if (!h3_launch_cast_bf16_to_f32(query, q32, (uint32_t)n, stream) ||
        !h3_launch_cast_bf16_to_f32(key, k32, (uint32_t)n, stream) ||
        !h3_launch_cast_bf16_to_f32(value, v32, (uint32_t)n, stream)) {
        return 0;
    }
    if (!h3_launch_sdpa_f32_hipblas(q32, k32, v32, o32, args, stream))
        return 0;
    return h3_launch_cast_f32_to_bf16(o32, output, (uint32_t)n, stream);
}

static int h3_int8_ensure_i32_scratch(size_t bytes) {
    if (bytes <= h3_int8_i32_scratch_bytes) return 1;
    int32_t *next = NULL;
    if (hipMalloc(&next, bytes) != hipSuccess) return 0;
    if (h3_int8_i32_scratch) (void)hipFree(h3_int8_i32_scratch);
    h3_int8_i32_scratch = next;
    h3_int8_i32_scratch_bytes = bytes;
    return 1;
}

static int h3_int8_ensure_f32_scratch(size_t bytes) {
    if (bytes <= h3_int8_f32_scratch_bytes) return 1;
    float *next = NULL;
    if (hipMalloc(&next, bytes) != hipSuccess) return 0;
    if (h3_int8_f32_scratch) (void)hipFree(h3_int8_f32_scratch);
    h3_int8_f32_scratch = next;
    h3_int8_f32_scratch_bytes = bytes;
    return 1;
}

static int8_t *h3_int8_pack_scratch = NULL;
static size_t h3_int8_pack_scratch_bytes = 0;

static int h3_int8_ensure_i8_scratch(size_t bytes) {
    if (bytes <= h3_int8_pack_scratch_bytes) return 1;
    int8_t *next = NULL;
    if (hipMalloc(&next, bytes) != hipSuccess) return 0;
    if (h3_int8_pack_scratch) (void)hipFree(h3_int8_pack_scratch);
    h3_int8_pack_scratch = next;
    h3_int8_pack_scratch_bytes = bytes;
    return 1;
}

/* [rows, groups, G] -> [groups, rows, G] so each GEMM batch has lda = G. */
__global__ void h3_i8_pack_groups_kernel(const int8_t *in, int8_t *out,
                                         uint32_t rows, uint32_t groups,
                                         uint32_t group_size) {
    uint32_t g = blockIdx.z;
    uint32_t row = blockIdx.y;
    uint32_t k = (blockIdx.x * blockDim.x + threadIdx.x) * 16u;
    if (g >= groups || row >= rows || k >= group_size) return;
    const int8_t *src = in + ((size_t)row * groups + g) * group_size + k;
    int8_t *dst = out + ((size_t)g * rows + row) * group_size + k;
    *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(src);
}

static int h3_i8_pack_groups(const int8_t *in, int8_t *out, uint32_t rows,
                             uint32_t groups, uint32_t group_size,
                             hipStream_t stream) {
    if (group_size % 16u) return 0;
    dim3 block(256, 1, 1);
    dim3 grid(((group_size / 16u) + 255u) / 256u, rows, groups);
    void *args[] = {&in, &out, &rows, &groups, &group_size};
    return hipLaunchKernel(reinterpret_cast<const void *>(h3_i8_pack_groups_kernel),
                           grid, block, args, 0, stream) == hipSuccess;
}

/* 4-wide i32 load + packed bf16 store. D[m,n] = acc[m,n] * sa[m] * sb[n]. */
__global__ void __launch_bounds__(256)
h3_i32_scale_to_bf16_kernel(const int32_t *acc, const float *sa, const float *sb,
                            uint16_t *out, uint32_t rows, uint32_t cols) {
    uint32_t row = blockIdx.y;
    uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (row >= rows || col >= cols) return;
    float srow = sa[row];
    const int32_t *in = acc + (size_t)row * cols + col;
    uint16_t *dst = out + (size_t)row * cols + col;
    if (col + 3u < cols) {
        int4 v = *reinterpret_cast<const int4 *>(in);
        float4 scale = *reinterpret_cast<const float4 *>(sb + col);
        uint16_t o0 = h3_f32_to_bf16((float)v.x * srow * scale.x);
        uint16_t o1 = h3_f32_to_bf16((float)v.y * srow * scale.y);
        uint16_t o2 = h3_f32_to_bf16((float)v.z * srow * scale.z);
        uint16_t o3 = h3_f32_to_bf16((float)v.w * srow * scale.w);
        uint2 packed;
        packed.x = (uint32_t)o0 | ((uint32_t)o1 << 16);
        packed.y = (uint32_t)o2 | ((uint32_t)o3 << 16);
        *reinterpret_cast<uint2 *>(dst) = packed;
    } else {
        for (uint32_t j = 0; col + j < cols; j++)
            dst[j] = h3_f32_to_bf16((float)in[j] * srow * sb[col + j]);
    }
}

/* acc is [rows, 2H] from gemm; gate = [0:H), up = [H:2H). */
__global__ void __launch_bounds__(256)
h3_i32_swiglu_to_bf16_kernel(const int32_t *acc, const float *sa,
                             const float *sb, uint16_t *out, uint32_t rows,
                             uint32_t hidden) {
    uint32_t row = blockIdx.y;
    uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (row >= rows || col >= hidden) return;
    float srow = sa[row];
    const int32_t *gate_in =
        acc + ((size_t)row * hidden * 2u) + col;
    const int32_t *up_in = gate_in + hidden;
    uint16_t *dst = out + (size_t)row * hidden + col;
    if (col + 3u < hidden) {
        int4 g = *reinterpret_cast<const int4 *>(gate_in);
        int4 u = *reinterpret_cast<const int4 *>(up_in);
        uint16_t o0 = h3_f32_to_bf16(
            h3_silu((float)g.x * srow * sb[col]) *
            ((float)u.x * srow * sb[hidden + col]));
        uint16_t o1 = h3_f32_to_bf16(
            h3_silu((float)g.y * srow * sb[col + 1u]) *
            ((float)u.y * srow * sb[hidden + col + 1u]));
        uint16_t o2 = h3_f32_to_bf16(
            h3_silu((float)g.z * srow * sb[col + 2u]) *
            ((float)u.z * srow * sb[hidden + col + 2u]));
        uint16_t o3 = h3_f32_to_bf16(
            h3_silu((float)g.w * srow * sb[col + 3u]) *
            ((float)u.w * srow * sb[hidden + col + 3u]));
        uint2 packed;
        packed.x = (uint32_t)o0 | ((uint32_t)o1 << 16);
        packed.y = (uint32_t)o2 | ((uint32_t)o3 << 16);
        *reinterpret_cast<uint2 *>(dst) = packed;
    } else {
        for (uint32_t j = 0; col + j < hidden; j++) {
            float gate = (float)gate_in[j] * srow * sb[col + j];
            float up = (float)up_in[j] * srow * sb[hidden + col + j];
            dst[j] = h3_f32_to_bf16(h3_silu(gate) * up);
        }
    }
}

/* Running grouped epilogue: out += acc * sa[row, g] * sb[col]. last writes bf16. */
__global__ void __launch_bounds__(256)
h3_i32_grouped_scale_epilogue_kernel(
    const int32_t *acc, const float *sa, const float *sb, float *partial,
    uint16_t *out, uint32_t rows, uint32_t cols, uint32_t groups,
    uint32_t group, int first, int last) {
    uint32_t row = blockIdx.y;
    uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (row >= rows || col >= cols) return;
    float srow = sa[(size_t)row * groups + group];
    const int32_t *in = acc + (size_t)row * cols + col;
    float *psum = partial ? partial + (size_t)row * cols + col : NULL;
    uint16_t *dst = out + (size_t)row * cols + col;
    if (col + 3u < cols) {
        int4 v = *reinterpret_cast<const int4 *>(in);
        float t0 = (float)v.x * srow * sb[col];
        float t1 = (float)v.y * srow * sb[col + 1u];
        float t2 = (float)v.z * srow * sb[col + 2u];
        float t3 = (float)v.w * srow * sb[col + 3u];
        if (!first) {
            float4 p = *reinterpret_cast<const float4 *>(psum);
            t0 += p.x;
            t1 += p.y;
            t2 += p.z;
            t3 += p.w;
        }
        if (last) {
            uint16_t o0 = h3_f32_to_bf16(t0);
            uint16_t o1 = h3_f32_to_bf16(t1);
            uint16_t o2 = h3_f32_to_bf16(t2);
            uint16_t o3 = h3_f32_to_bf16(t3);
            uint2 packed;
            packed.x = (uint32_t)o0 | ((uint32_t)o1 << 16);
            packed.y = (uint32_t)o2 | ((uint32_t)o3 << 16);
            *reinterpret_cast<uint2 *>(dst) = packed;
        } else {
            float4 o;
            o.x = t0;
            o.y = t1;
            o.z = t2;
            o.w = t3;
            *reinterpret_cast<float4 *>(psum) = o;
        }
    } else {
        for (uint32_t j = 0; col + j < cols; j++) {
            float t = (float)in[j] * srow * sb[col + j];
            if (!first) t += psum[j];
            if (last) dst[j] = h3_f32_to_bf16(t);
            else psum[j] = t;
        }
    }
}

/* Reduce batched i32 planes[g][row, col] * sa[row, g] * sb[col] -> bf16. */
__global__ void __launch_bounds__(256)
h3_i32_grouped_reduce_to_bf16_kernel(const int32_t *acc, const float *sa,
                                     const float *sb, uint16_t *out,
                                     uint32_t rows, uint32_t cols,
                                     uint32_t groups) {
    uint32_t row = blockIdx.y;
    uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (row >= rows || col >= cols) return;
    size_t plane = (size_t)rows * cols;
    uint16_t *dst = out + (size_t)row * cols + col;
    if (col + 3u < cols) {
        float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
        float b0 = sb[col], b1 = sb[col + 1u], b2 = sb[col + 2u],
              b3 = sb[col + 3u];
        for (uint32_t g = 0; g < groups; g++) {
            float srow = sa[(size_t)row * groups + g];
            const int32_t *in =
                acc + (size_t)g * plane + (size_t)row * cols + col;
            int4 v = *reinterpret_cast<const int4 *>(in);
            t0 = fmaf((float)v.x, srow * b0, t0);
            t1 = fmaf((float)v.y, srow * b1, t1);
            t2 = fmaf((float)v.z, srow * b2, t2);
            t3 = fmaf((float)v.w, srow * b3, t3);
        }
        uint16_t o0 = h3_f32_to_bf16(t0);
        uint16_t o1 = h3_f32_to_bf16(t1);
        uint16_t o2 = h3_f32_to_bf16(t2);
        uint16_t o3 = h3_f32_to_bf16(t3);
        uint2 packed;
        packed.x = (uint32_t)o0 | ((uint32_t)o1 << 16);
        packed.y = (uint32_t)o2 | ((uint32_t)o3 << 16);
        *reinterpret_cast<uint2 *>(dst) = packed;
    } else {
        for (uint32_t j = 0; col + j < cols; j++) {
            float total = 0.0f;
            float b = sb[col + j];
            for (uint32_t g = 0; g < groups; g++) {
                float srow = sa[(size_t)row * groups + g];
                int32_t v = acc[(size_t)g * plane + (size_t)row * cols + col + j];
                total = fmaf((float)v, srow * b, total);
            }
            dst[j] = h3_f32_to_bf16(total);
        }
    }
}

static int h3_int8_hipblas_opt_in(int fc1_swiglu) {
    static int plain = -1;
    static int fc1 = -1;
    if (plain < 0) {
        const char *t128 = getenv("H3_INT8_T128");
        const char *off = getenv("H3_INT8_HIPBLAS");
        plain = 1;
        if (t128 && strcmp(t128, "0") != 0) plain = 0;
        if (off && strcmp(off, "0") == 0) plain = 0;
        const char *fc1_off = getenv("H3_INT8_FC1_HIPBLAS");
        fc1 = plain;
        if (fc1_off && strcmp(fc1_off, "0") == 0) fc1 = 0;
        if (getenv("H3_INT8_FC1_R64") &&
            strcmp(getenv("H3_INT8_FC1_R64"), "0") != 0)
            fc1 = 0;
    }
    return fc1_swiglu ? fc1 : plain;
}

static int h3_int8_hipblas_linear(const int8_t *input, const int8_t *weight,
                                  const float *input_scales,
                                  const float *weight_scales, uint16_t *output,
                                  const h3_linear_args *args, hipStream_t stream,
                                  int fc1_swiglu) {
    if (!h3_int8_hipblas_opt_in(fc1_swiglu)) return 0;
    uint32_t hidden = args->output_dim;
    uint32_t gemm_n = fc1_swiglu ? hidden * 2u : hidden;
    if (args->input_dim % 4u || args->input_dim < 32u || gemm_n < 16u)
        return 0;
    if (!h3_int8_ensure_hipblas(stream)) return 0;
    size_t bytes = (size_t)args->rows * gemm_n * sizeof(int32_t);
    if (!h3_int8_ensure_i32_scratch(bytes)) return 0;
    int32_t alpha = 1, beta = 0;
    int m = (int)gemm_n;
    int n = (int)args->rows;
    int k = (int)args->input_dim;
    int split = h3_int8_split_on();
    static hipEvent_t gemm_start = NULL, gemm_stop = NULL;
    static hipEvent_t epi_start = NULL, epi_stop = NULL;
    if (split) h3_int8_split_begin(&gemm_start, &gemm_stop, stream);
    if (hipblasGemmEx(h3_hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, &alpha,
                      weight, HIP_R_8I, k, input, HIP_R_8I, k, &beta,
                      h3_int8_i32_scratch, HIP_R_32I, m, HIPBLAS_COMPUTE_32I,
                      HIPBLAS_GEMM_DEFAULT) != HIPBLAS_STATUS_SUCCESS) {
        return 0;
    }
    if (split) h3_int8_split_end(gemm_start, gemm_stop, stream, "gemm");
    uint32_t rows = args->rows;
    uint32_t cols = hidden;
    dim3 block(256, 1, 1);
    dim3 grid(((cols + 3u) / 4u + 255u) / 256u, rows, 1);
    void *ep_host[] = {&h3_int8_i32_scratch, &input_scales, &weight_scales,
                       &output, &rows, &cols};
    const void *ep = fc1_swiglu ?
        reinterpret_cast<const void *>(h3_i32_swiglu_to_bf16_kernel) :
        reinterpret_cast<const void *>(h3_i32_scale_to_bf16_kernel);
    if (split) h3_int8_split_begin(&epi_start, &epi_stop, stream);
    int ok = hipLaunchKernel(ep, grid, block, ep_host, 0, stream) == hipSuccess;
    if (split && ok) h3_int8_split_end(epi_start, epi_stop, stream, "epilogue");
    return ok;
}

static int h3_int8_grouped_hipblas_on(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *off = getenv("H3_INT8_HIPBLAS");
        const char *g_off = getenv("H3_INT8_GROUPED_HIPBLAS");
        cached = 1;
        if (off && strcmp(off, "0") == 0) cached = 0;
        if (g_off && strcmp(g_off, "0") == 0) cached = 0;
    }
    return cached;
}

static int h3_int8_hipblas_grouped_linear(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output,
    const h3_linear_int8_grouped_args *args, hipStream_t stream) {
    if (!h3_int8_grouped_hipblas_on()) return 0;
    uint32_t rows = args->rows;
    uint32_t k_full = args->input_dim;
    uint32_t cols = args->output_dim;
    uint32_t group_size = args->group_size;
    if (!rows || !k_full || !cols || !group_size) return 0;
    if (k_full % group_size || group_size % 4u || group_size < 32u || cols < 16u)
        return 0;
    uint32_t groups = k_full / group_size;
    if (!h3_int8_ensure_hipblas(stream)) return 0;

    int32_t alpha = 1, beta = 0;
    int m = (int)cols;
    int n = (int)rows;
    int k = (int)group_size;
    int lda = (int)k_full;
    dim3 block(256, 1, 1);
    dim3 grid(((cols + 3u) / 4u + 255u) / 256u, rows, 1);
    int split = h3_int8_split_on();
    static hipEvent_t gemm_start = NULL, gemm_stop = NULL;
    static hipEvent_t epi_start = NULL, epi_stop = NULL;

    const char *force_loop = getenv("H3_INT8_GROUPED_LOOP");
    int want_loop = force_loop && strcmp(force_loop, "0") != 0;
    const char *no_pack = getenv("H3_INT8_GROUPED_PACK");
    int want_pack = !no_pack || strcmp(no_pack, "0") != 0;
    size_t plane_bytes = (size_t)rows * cols * sizeof(int32_t);
    size_t batched_bytes = plane_bytes * (size_t)groups;
    const size_t batched_cap = (size_t)768 * 1024 * 1024;
    int use_batched = !want_loop && groups > 1u && batched_bytes <= batched_cap;

    if (use_batched && want_pack && group_size % 16u == 0u) {
        size_t pack_in = (size_t)groups * rows * group_size;
        size_t pack_w = (size_t)groups * cols * group_size;
        if (h3_int8_ensure_i8_scratch(pack_in + pack_w) &&
            h3_int8_ensure_i32_scratch(batched_bytes)) {
            int8_t *packed_in = h3_int8_pack_scratch;
            int8_t *packed_w = h3_int8_pack_scratch + pack_in;
            if (split) h3_int8_split_begin(&gemm_start, &gemm_stop, stream);
            int packed = h3_i8_pack_groups(input, packed_in, rows, groups,
                                           group_size, stream) &&
                         h3_i8_pack_groups(weight, packed_w, cols, groups,
                                           group_size, stream);
            hipblasStatus_t st = HIPBLAS_STATUS_INTERNAL_ERROR;
            if (packed) {
                st = hipblasGemmStridedBatchedEx(
                    h3_hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, &alpha,
                    packed_w, HIP_R_8I, k, (hipblasStride)((size_t)cols * group_size),
                    packed_in, HIP_R_8I, k, (hipblasStride)((size_t)rows * group_size),
                    &beta, h3_int8_i32_scratch, HIP_R_32I, m,
                    (hipblasStride)((size_t)m * (size_t)n), (int)groups,
                    HIPBLAS_COMPUTE_32I, HIPBLAS_GEMM_DEFAULT);
            }
            if (split)
                h3_int8_split_end(gemm_start, gemm_stop, stream,
                                  "grouped-gemm-packed");
            if (st == HIPBLAS_STATUS_SUCCESS) {
                void *red[] = {&h3_int8_i32_scratch, &input_scales,
                               &weight_scales, &output, &rows, &cols, &groups};
                if (split) h3_int8_split_begin(&epi_start, &epi_stop, stream);
                int ok = hipLaunchKernel(
                    reinterpret_cast<const void *>(
                        h3_i32_grouped_reduce_to_bf16_kernel),
                    grid, block, red, 0, stream) == hipSuccess;
                if (split && ok)
                    h3_int8_split_end(epi_start, epi_stop, stream,
                                      "grouped-reduce");
                return ok;
            }
        }
    }

    if (use_batched && h3_int8_ensure_i32_scratch(batched_bytes)) {
        if (split) h3_int8_split_begin(&gemm_start, &gemm_stop, stream);
        hipblasStatus_t st = hipblasGemmStridedBatchedEx(
            h3_hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, &alpha, weight,
            HIP_R_8I, lda, (hipblasStride)group_size, input, HIP_R_8I, lda,
            (hipblasStride)group_size, &beta, h3_int8_i32_scratch, HIP_R_32I, m,
            (hipblasStride)((size_t)m * (size_t)n), (int)groups,
            HIPBLAS_COMPUTE_32I, HIPBLAS_GEMM_DEFAULT);
        if (split) h3_int8_split_end(gemm_start, gemm_stop, stream,
                                     "grouped-gemm-batched");
        if (st == HIPBLAS_STATUS_SUCCESS) {
            void *red[] = {&h3_int8_i32_scratch, &input_scales, &weight_scales,
                           &output, &rows, &cols, &groups};
            if (split) h3_int8_split_begin(&epi_start, &epi_stop, stream);
            int ok = hipLaunchKernel(
                reinterpret_cast<const void *>(h3_i32_grouped_reduce_to_bf16_kernel),
                grid, block, red, 0, stream) == hipSuccess;
            if (split && ok)
                h3_int8_split_end(epi_start, epi_stop, stream, "grouped-reduce");
            return ok;
        }
    }

    if (!h3_int8_ensure_i32_scratch(plane_bytes)) return 0;
    int need_partial = groups > 1u;
    if (need_partial &&
        !h3_int8_ensure_f32_scratch((size_t)rows * cols * sizeof(float)))
        return 0;
    for (uint32_t g = 0; g < groups; g++) {
        if (split) h3_int8_split_begin(&gemm_start, &gemm_stop, stream);
        hipblasStatus_t st = hipblasGemmEx(
            h3_hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N, m, n, k, &alpha,
            weight + (size_t)g * group_size, HIP_R_8I, lda,
            input + (size_t)g * group_size, HIP_R_8I, lda, &beta,
            h3_int8_i32_scratch, HIP_R_32I, m, HIPBLAS_COMPUTE_32I,
            HIPBLAS_GEMM_DEFAULT);
        if (split) h3_int8_split_end(gemm_start, gemm_stop, stream,
                                     "grouped-gemm");
        if (st != HIPBLAS_STATUS_SUCCESS) return 0;
        uint32_t group = g;
        int first = (g == 0u);
        int last = (g + 1u == groups);
        void *ep[] = {&h3_int8_i32_scratch, &input_scales, &weight_scales,
                      &h3_int8_f32_scratch, &output, &rows, &cols, &groups,
                      &group, &first, &last};
        if (split) h3_int8_split_begin(&epi_start, &epi_stop, stream);
        int ok = hipLaunchKernel(
            reinterpret_cast<const void *>(h3_i32_grouped_scale_epilogue_kernel),
            grid, block, ep, 0, stream) == hipSuccess;
        if (split && ok)
            h3_int8_split_end(epi_start, epi_stop, stream, "grouped-epilogue");
        if (!ok) return 0;
    }
    return 1;
}

/* ===========================================================================
 * FP8 E4M3 FNUZ GEMM via hipBLASLt (gfx942 / CDNA3 only).
 *
 * Flow: quantize BF16 input → FP8, then hipBLASLt FP8×FP8 → FP32 GEMM,
 * then epilogue: out_bf16[m][n] = fp32_acc[m][n] * input_scale[m] * weight_scale[n].
 *
 * Weight is pre-quantized to FP8 at load time. Both input and weight are
 * FP8 E4M3 FNUZ (max=240.0). The GEMM accumulates in FP32 and the epilogue
 * dequantizes to BF16.
 * =========================================================================== */
#if __has_include(<hipblaslt/hipblaslt.h>)
#include <hipblaslt/hipblaslt.h>

static hipblasLtHandle_t h3_fp8_hipblaslt = NULL;
static int h3_fp8_hipblaslt_inited = 0;

static int h3_fp8_ensure_hiplt(hipStream_t stream) {
    if (h3_fp8_hipblaslt_inited) return h3_fp8_hipblaslt != NULL;
    h3_fp8_hipblaslt_inited = 1;
    if (hipblasLtCreate(&h3_fp8_hipblaslt) != HIPBLAS_STATUS_SUCCESS) {
        h3_fp8_hipblaslt = NULL;
        return 0;
    }
    return 1;
}

/* 4-wide f32 load + packed bf16 store. D[m,n] = acc[m,n] * sa[m] * sb[n]. */
__global__ void __launch_bounds__(256)
h3_fp32_scale_to_bf16_kernel(const float *acc, const float *sa, const float *sb,
                             uint16_t *out, uint32_t rows, uint32_t cols) {
    uint32_t row = blockIdx.y;
    uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (row >= rows || col >= cols) return;
    float srow = sa[row];
    const float *in = acc + (size_t)row * cols + col;
    uint16_t *dst = out + (size_t)row * cols + col;
    if (col + 3u < cols) {
        float4 v = *reinterpret_cast<const float4 *>(in);
        float4 scale = *reinterpret_cast<const float4 *>(sb + col);
        uint16_t o0 = h3_f32_to_bf16(v.x * srow * scale.x);
        uint16_t o1 = h3_f32_to_bf16(v.y * srow * scale.y);
        uint16_t o2 = h3_f32_to_bf16(v.z * srow * scale.z);
        uint16_t o3 = h3_f32_to_bf16(v.w * srow * scale.w);
        uint2 packed;
        packed.x = (uint32_t)o0 | ((uint32_t)o1 << 16);
        packed.y = (uint32_t)o2 | ((uint32_t)o3 << 16);
        *reinterpret_cast<uint2 *>(dst) = packed;
    } else {
        for (uint32_t j = 0; col + j < cols; j++)
            dst[j] = h3_f32_to_bf16(in[j] * srow * sb[col + j]);
    }
}

/* acc is [rows, 2H] from gemm; gate = [0:H), up = [H:2H). */
__global__ void __launch_bounds__(256)
h3_fp32_swiglu_to_bf16_kernel(
    const float *acc, const float *sa, const float *sb,
    uint16_t *out, uint32_t rows, uint32_t hidden) {
    uint32_t row = blockIdx.y;
    uint32_t col = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
    if (row >= rows || col >= hidden) return;
    float srow = sa[row];
    const float *gate_in = acc + (size_t)row * hidden * 2 + col;
    const float *up_in = acc + (size_t)row * hidden * 2 + hidden + col;
    uint16_t *dst = out + (size_t)row * hidden + col;
    if (col + 3u < hidden) {
        float4 gv = *reinterpret_cast<const float4 *>(gate_in);
        float4 uv = *reinterpret_cast<const float4 *>(up_in);
        float4 gate_scale = *reinterpret_cast<const float4 *>(sb + col);
        float4 up_scale = *reinterpret_cast<const float4 *>(sb + hidden + col);
        uint16_t o0 = h3_f32_to_bf16(
            h3_silu(gv.x * srow * gate_scale.x) *
            (uv.x * srow * up_scale.x));
        uint16_t o1 = h3_f32_to_bf16(
            h3_silu(gv.y * srow * gate_scale.y) *
            (uv.y * srow * up_scale.y));
        uint16_t o2 = h3_f32_to_bf16(
            h3_silu(gv.z * srow * gate_scale.z) *
            (uv.z * srow * up_scale.z));
        uint16_t o3 = h3_f32_to_bf16(
            h3_silu(gv.w * srow * gate_scale.w) *
            (uv.w * srow * up_scale.w));
        uint2 packed;
        packed.x = (uint32_t)o0 | ((uint32_t)o1 << 16);
        packed.y = (uint32_t)o2 | ((uint32_t)o3 << 16);
        *reinterpret_cast<uint2 *>(dst) = packed;
    } else {
        for (uint32_t j = 0; col + j < hidden; j++) {
            float gate = gate_in[j] * srow * sb[col + j];
            float up = up_in[j] * srow * sb[hidden + col + j];
            dst[j] = h3_f32_to_bf16(h3_silu(gate) * up);
        }
    }
}

static int h3_fp8_gemm_run(
    const void *input, const void *weight, uint32_t m, uint32_t n,
    uint32_t k, float *fp32_acc, hipStream_t stream) {
    float alpha = 1.0f, beta = 0.0f;
    hipblasOperation_t opT = HIPBLAS_OP_T;
    hipblasOperation_t opN = HIPBLAS_OP_N;
    hipblasLtMatmulDesc_t matmulDesc = NULL;
    hipblasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL;
    hipblasLtMatrixLayout_t Cdesc = NULL, Ddesc = NULL;
    hipblasLtMatmulPreference_t pref = NULL;
    void *f32_ws = NULL;
    size_t ws_size = 0;
    int ok = 0;

    if (hipblasLtMatmulDescCreate(&matmulDesc, HIPBLAS_COMPUTE_32F,
                                  HIPBLAS_R_32F) != HIPBLAS_STATUS_SUCCESS)
        goto done;
    (void)hipblasLtMatmulDescSetAttribute(
        matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    (void)hipblasLtMatmulDescSetAttribute(
        matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    if (hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_8F_E4M3_FNUZ, k, m, k) !=
            HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&Bdesc, HIP_R_8F_E4M3_FNUZ, k, n, k) !=
            HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&Cdesc, HIP_R_32F, m, n, m) !=
            HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatrixLayoutCreate(&Ddesc, HIP_R_32F, m, n, m) !=
            HIPBLAS_STATUS_SUCCESS)
        goto done;

    if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS)
        goto done;
    {
        hipblasLtMatmulHeuristicResult_t heuristic[1];
        int returned = 0;
        if (hipblasLtMatmulAlgoGetHeuristic(h3_fp8_hipblaslt, matmulDesc,
                                            Adesc, Bdesc, Cdesc, Ddesc, pref,
                                            1, heuristic, &returned) !=
                HIPBLAS_STATUS_SUCCESS ||
            returned < 1 || heuristic[0].state != HIPBLAS_STATUS_SUCCESS)
            goto done;
        ws_size = heuristic[0].workspaceSize;
    }
    if (ws_size > 0) {
        if (hipMalloc(&f32_ws, ws_size) != hipSuccess) {
            f32_ws = NULL;
            ws_size = 0;
        }
    }
    {
        hipblasLtMatmulHeuristicResult_t heuristic[1];
        int returned = 0;
        (void)hipblasLtMatmulAlgoGetHeuristic(
            h3_fp8_hipblaslt, matmulDesc, Adesc, Bdesc, Cdesc, Ddesc,
            pref, 1, heuristic, &returned);
        hipblasStatus_t st = hipblasLtMatmul(
            h3_fp8_hipblaslt, matmulDesc, &alpha, weight, Adesc,
            input, Bdesc, &beta, fp32_acc, Cdesc, fp32_acc, Ddesc,
            &heuristic[0].algo, f32_ws, ws_size, stream);
        ok = (st == HIPBLAS_STATUS_SUCCESS);
    }

done:
    if (f32_ws) (void)hipFree(f32_ws);
    if (pref) hipblasLtMatmulPreferenceDestroy(pref);
    if (Ddesc) hipblasLtMatrixLayoutDestroy(Ddesc);
    if (Cdesc) hipblasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) hipblasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) hipblasLtMatrixLayoutDestroy(Adesc);
    if (matmulDesc) hipblasLtMatmulDescDestroy(matmulDesc);
    return ok;
}

extern "C" int h3_launch_linear_fp8_bf16(
    const void *input, const void *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output,
    const h3_linear_args *args, hipStream_t stream) {
    if (!h3_fp8_ensure_hiplt(stream)) return 0;
    uint32_t m = args->output_dim;
    uint32_t n = args->rows;
    uint32_t k = args->input_dim;
    if (!m || !n || !k) return 0;
    uint32_t fp32_count = n * m;
    float *fp32_acc = NULL;
    if (hipMalloc(&fp32_acc, (size_t)fp32_count * sizeof(float)) != hipSuccess)
        return 0;
    if (!h3_fp8_gemm_run(input, weight, m, n, k, fp32_acc, stream)) {
        (void)hipFree(fp32_acc);
        return 0;
    }
    dim3 block(256, 1, 1);
    dim3 grid(((m + 3u) / 4u + 255u) / 256u, n, 1);
    void *ep_args[] = {&fp32_acc, &input_scales, &weight_scales,
                       &output, &n, &m};
    hipLaunchKernel(
        reinterpret_cast<const void *>(h3_fp32_scale_to_bf16_kernel),
        grid, block, ep_args, 0, stream);
    (void)hipFree(fp32_acc);
    return 1;
}

extern "C" int h3_launch_linear_fp8_fc1_swiglu_bf16(
    const void *input, const void *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output,
    const h3_linear_args *args, hipStream_t stream) {
    if (!h3_fp8_ensure_hiplt(stream)) return 0;
    uint32_t m = args->output_dim;
    uint32_t n = args->rows;
    uint32_t k = args->input_dim;
    if (!m || !n || !k) return 0;
    uint32_t fp32_count = n * m;
    float *fp32_acc = NULL;
    if (hipMalloc(&fp32_acc, (size_t)fp32_count * sizeof(float)) != hipSuccess)
        return 0;
    if (!h3_fp8_gemm_run(input, weight, m, n, k, fp32_acc, stream)) {
        (void)hipFree(fp32_acc);
        return 0;
    }
    dim3 block(256, 1, 1);
    dim3 grid(((m + 3u) / 4u + 255u) / 256u, n, 1);
    uint32_t hidden = m / 2;
    void *ep_args[] = {&fp32_acc, &input_scales, &weight_scales,
                       &output, &n, &hidden};
    hipLaunchKernel(
        reinterpret_cast<const void *>(h3_fp32_swiglu_to_bf16_kernel),
        grid, block, ep_args, 0, stream);
    (void)hipFree(fp32_acc);
    return 1;
}

#else
extern "C" int h3_launch_linear_fp8_bf16(
    const void *input, const void *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output,
    const h3_linear_args *args, hipStream_t stream) {
    (void)input; (void)weight; (void)input_scales; (void)weight_scales;
    (void)output; (void)args; (void)stream;
    return 0;
}
extern "C" int h3_launch_linear_fp8_fc1_swiglu_bf16(
    const void *input, const void *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output,
    const h3_linear_args *args, hipStream_t stream) {
    (void)input; (void)weight; (void)input_scales; (void)weight_scales;
    (void)output; (void)args; (void)stream;
    return 0;
}
#endif

static int h3_launch_linear_int8_r64_dispatch(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output, const h3_linear_args *args,
    hipStream_t stream, int fc1_swiglu) {
    if (!args->rows || !args->input_dim || !args->output_dim ||
        args->input_dim % 32u) {
        return 0;
    }
    /* Prefer larger BK when K divides evenly. Opt out: H3_INT8_BK32 / BK64. */
    int force_bk32 = getenv("H3_INT8_BK32") &&
                     strcmp(getenv("H3_INT8_BK32"), "0") != 0;
    int force_bk64 = getenv("H3_INT8_BK64") &&
                     strcmp(getenv("H3_INT8_BK64"), "0") != 0;
    int use_bk128 = (args->input_dim % 128u) == 0 && !force_bk32 && !force_bk64;
    int use_bk64 = (args->input_dim % 64u) == 0 && !force_bk32 && !use_bk128;
    int force_r64 = getenv("H3_INT8_R64") &&
                    strcmp(getenv("H3_INT8_R64"), "0") != 0;
    int force_fc1_r64 = getenv("H3_INT8_FC1_R64") &&
                        strcmp(getenv("H3_INT8_FC1_R64"), "0") != 0;
    if (h3_int8_hipblas_linear(input, weight, input_scales, weight_scales,
                               output, args, stream, fc1_swiglu)) {
        return 1;
    }
    int use_t128 = !fc1_swiglu && !force_r64;
    int use_fc1_r128x64 = fc1_swiglu && !force_fc1_r64 && !force_r64;
    h3_linear_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &input_scales, &weight_scales,
                           &output, &host_args};
    dim3 block(16, 16, 1);
    uint32_t tile_n = use_t128 ? 128u : 64u;
    uint32_t tile_m = (use_t128 || use_fc1_r128x64) ? 128u : 64u;
    dim3 grid((host_args.output_dim + tile_n - 1u) / tile_n,
              (host_args.rows + tile_m - 1u) / tile_m, 1);
    if (use_t128) {
        const void *t128 = NULL;
        if (use_bk128) {
            if (host_args.input_dim == 14336)
                t128 = reinterpret_cast<const void *>(
                    h3_linear_int8_t128_kernel<14336, 128>);
            else if (host_args.input_dim == 5376)
                t128 = reinterpret_cast<const void *>(
                    h3_linear_int8_t128_kernel<5376, 128>);
            else
                t128 = reinterpret_cast<const void *>(
                    h3_linear_int8_t128_kernel<0, 128>);
        } else if (use_bk64) {
            if (host_args.input_dim == 14336)
                t128 = reinterpret_cast<const void *>(
                    h3_linear_int8_t128_kernel<14336, 64>);
            else if (host_args.input_dim == 5376)
                t128 = reinterpret_cast<const void *>(
                    h3_linear_int8_t128_kernel<5376, 64>);
            else
                t128 = reinterpret_cast<const void *>(
                    h3_linear_int8_t128_kernel<0, 64>);
        } else if (host_args.input_dim == 14336) {
            t128 = reinterpret_cast<const void *>(
                h3_linear_int8_t128_kernel<14336, 32>);
        } else if (host_args.input_dim == 5376) {
            t128 = reinterpret_cast<const void *>(
                h3_linear_int8_t128_kernel<5376, 32>);
        } else {
            t128 = reinterpret_cast<const void *>(
                h3_linear_int8_t128_kernel<0, 32>);
        }
        return hipLaunchKernel(t128, grid, block, kernel_args,
                               sizeof(kernel_args), stream) == hipSuccess;
    }
    const void *kernel = NULL;
    if (use_fc1_r128x64) {
        if (use_bk128) {
            if (host_args.input_dim == 5376)
                kernel = reinterpret_cast<const void *>(
                    h3_fc1_swiglu_int8_r128x64_kernel<5376, 128>);
            else
                kernel = reinterpret_cast<const void *>(
                    h3_fc1_swiglu_int8_r128x64_kernel<0, 128>);
        } else if (use_bk64) {
            if (host_args.input_dim == 5376)
                kernel = reinterpret_cast<const void *>(
                    h3_fc1_swiglu_int8_r128x64_kernel<5376, 64>);
            else
                kernel = reinterpret_cast<const void *>(
                    h3_fc1_swiglu_int8_r128x64_kernel<0, 64>);
        } else if (host_args.input_dim == 5376) {
            kernel = reinterpret_cast<const void *>(
                h3_fc1_swiglu_int8_r128x64_kernel<5376, 32>);
        } else {
            kernel = reinterpret_cast<const void *>(
                h3_fc1_swiglu_int8_r128x64_kernel<0, 32>);
        }
        return hipLaunchKernel(kernel, grid, block, kernel_args,
                               sizeof(kernel_args), stream) == hipSuccess;
    }
    if (fc1_swiglu) {
        if (use_bk128) {
            if (host_args.input_dim == 5376)
                kernel = reinterpret_cast<const void *>(
                    h3_fc1_swiglu_int8_r64_kernel<5376, 128>);
            else
                kernel = reinterpret_cast<const void *>(
                    h3_fc1_swiglu_int8_r64_kernel<0, 128>);
        } else if (use_bk64) {
            if (host_args.input_dim == 5376)
                kernel = reinterpret_cast<const void *>(
                    h3_fc1_swiglu_int8_r64_kernel<5376, 64>);
            else
                kernel = reinterpret_cast<const void *>(
                    h3_fc1_swiglu_int8_r64_kernel<0, 64>);
        } else if (host_args.input_dim == 5376) {
            kernel = reinterpret_cast<const void *>(
                h3_fc1_swiglu_int8_r64_kernel<5376, 32>);
        } else {
            kernel = reinterpret_cast<const void *>(
                h3_fc1_swiglu_int8_r64_kernel<0, 32>);
        }
    } else if (use_bk128) {
        if (host_args.input_dim == 14336)
            kernel = reinterpret_cast<const void *>(
                h3_linear_int8_r64_kernel<14336, 128>);
        else if (host_args.input_dim == 5376)
            kernel = reinterpret_cast<const void *>(
                h3_linear_int8_r64_kernel<5376, 128>);
        else
            kernel = reinterpret_cast<const void *>(
                h3_linear_int8_r64_kernel<0, 128>);
    } else if (use_bk64) {
        if (host_args.input_dim == 14336)
            kernel = reinterpret_cast<const void *>(
                h3_linear_int8_r64_kernel<14336, 64>);
        else if (host_args.input_dim == 5376)
            kernel = reinterpret_cast<const void *>(
                h3_linear_int8_r64_kernel<5376, 64>);
        else
            kernel = reinterpret_cast<const void *>(
                h3_linear_int8_r64_kernel<0, 64>);
    } else if (host_args.input_dim == 14336) {
        kernel = reinterpret_cast<const void *>(
            h3_linear_int8_r64_kernel<14336, 32>);
    } else if (host_args.input_dim == 5376) {
        kernel = reinterpret_cast<const void *>(
            h3_linear_int8_r64_kernel<5376, 32>);
    } else {
        kernel = reinterpret_cast<const void *>(
            h3_linear_int8_r64_kernel<0, 32>);
    }
    return hipLaunchKernel(kernel, grid, block, kernel_args,
                           sizeof(kernel_args), stream) == hipSuccess;
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

extern "C" int h3_launch_linear_int8_nax_r64(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output, const h3_linear_args *args,
    hipStream_t stream) {
    return h3_launch_linear_int8_r64_dispatch(input, weight, input_scales,
                                              weight_scales, output, args,
                                              stream, 0);
}

extern "C" int h3_launch_fc1_swiglu_int8_nax_r64(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output, const h3_linear_args *args,
    hipStream_t stream) {
    return h3_launch_linear_int8_r64_dispatch(input, weight, input_scales,
                                              weight_scales, output, args,
                                              stream, 1);
}

template <int K_CONST, int GROUP_CONST, int BK>
__global__ void __launch_bounds__(256)
h3_linear_int8_grouped_r64_kernel(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output,
    h3_linear_int8_grouped_args args) {
    constexpr int BM = 64, BN = 64;
    static_assert(BK == 32 || BK == 64 || BK == 128, "INT8 BK 32/64/128");
    constexpr int LDS_K = BK + 4;
    __shared__ int8_t As[BM][LDS_K];
    __shared__ int8_t Bs[BN][LDS_K];
    __shared__ float scale_a[BM];
    __shared__ float scale_b[BN];
    uint32_t block_row = blockIdx.y * (uint32_t)BM;
    uint32_t block_col = blockIdx.x * (uint32_t)BN;
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t tid = ty * 16u + tx;
    uint32_t input_dim = K_CONST ? (uint32_t)K_CONST : args.input_dim;
    uint32_t group_size = GROUP_CONST ? (uint32_t)GROUP_CONST : args.group_size;
    uint32_t scale_groups = input_dim / group_size;
    uint32_t k_tiles = group_size / (uint32_t)BK;
    if (tid < (uint32_t)BN) {
        uint32_t column = block_col + tid;
        scale_b[tid] =
            column < args.output_dim ? weight_scales[column] : 0.0f;
    }
    float total[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) total[i][j] = 0.0f;
    for (uint32_t group = 0; group < scale_groups; group++) {
        if (tid < (uint32_t)BM) {
            uint32_t row = block_row + tid;
            scale_a[tid] = row < args.rows ?
                input_scales[row * scale_groups + group] : 0.0f;
        }
        int32_t acc[4][4];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) acc[i][j] = 0;
        uint32_t group_k0 = group * group_size;
        for (uint32_t kt = 0; kt < k_tiles; kt++) {
            uint32_t k0 = group_k0 + kt * (uint32_t)BK;
            h3_load_i8_tile_64xk<LDS_K, BK>(As, input, block_row, k0, args.rows,
                                             input_dim, tid);
            h3_load_i8_tile_64xk<LDS_K, BK>(Bs, weight, block_col, k0,
                                             args.output_dim, input_dim, tid);
            __syncthreads();
#pragma unroll
            for (int k = 0; k < BK; k += 8) {
                int32_t av0[4], av1[4], bv0[4], bv1[4];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    uint64_t packed = *reinterpret_cast<const uint64_t *>(
                        &As[ty * 4u + (uint32_t)i][k]);
                    av0[i] = (int32_t)packed;
                    av1[i] = (int32_t)(packed >> 32);
                }
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    uint64_t packed = *reinterpret_cast<const uint64_t *>(
                        &Bs[tx * 4u + (uint32_t)j][k]);
                    bv0[j] = (int32_t)packed;
                    bv1[j] = (int32_t)(packed >> 32);
                }
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        acc[i][j] = h3_i8dot4(av0[i], bv0[j], acc[i][j]);
                        acc[i][j] = h3_i8dot4(av1[i], bv1[j], acc[i][j]);
                    }
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                total[i][j] = fmaf((float)acc[i][j],
                                   scale_a[ty * 4u + (uint32_t)i] *
                                       scale_b[tx * 4u + (uint32_t)j],
                                   total[i][j]);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t row = block_row + ty * 4u + (uint32_t)i;
        if (row >= args.rows) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t column = block_col + tx * 4u + (uint32_t)j;
            if (column >= args.output_dim) continue;
            output[row * args.output_dim + column] =
                h3_f32_to_bf16(total[i][j]);
        }
    }
}

extern "C" int h3_launch_linear_int8_grouped_nax_r64(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output,
    const h3_linear_int8_grouped_args *args, hipStream_t stream) {
    if (!args->rows || !args->input_dim || !args->output_dim ||
        !args->group_size || args->group_size % 32u ||
        args->input_dim % args->group_size) {
        return 0;
    }
    if (h3_int8_hipblas_grouped_linear(input, weight, input_scales,
                                       weight_scales, output, args, stream)) {
        return 1;
    }
    int force_bk32 = getenv("H3_INT8_BK32") &&
                     strcmp(getenv("H3_INT8_BK32"), "0") != 0;
    int force_bk64 = getenv("H3_INT8_BK64") &&
                     strcmp(getenv("H3_INT8_BK64"), "0") != 0;
    int use_bk128 =
        (args->group_size % 128u) == 0 && !force_bk32 && !force_bk64;
    int use_bk64 = (args->group_size % 64u) == 0 && !force_bk32 && !use_bk128;
    h3_linear_int8_grouped_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &input_scales, &weight_scales,
                           &output, &host_args};
    dim3 block(16, 16, 1);
    dim3 grid((host_args.output_dim + 63u) / 64u,
              (host_args.rows + 63u) / 64u, 1);
    const void *kernel = NULL;
    if (use_bk128) {
        if (host_args.input_dim == 14336 && host_args.group_size == 1024)
            kernel = reinterpret_cast<const void *>(
                h3_linear_int8_grouped_r64_kernel<14336, 1024, 128>);
        else if (host_args.group_size == 1024)
            kernel = reinterpret_cast<const void *>(
                h3_linear_int8_grouped_r64_kernel<0, 1024, 128>);
        else
            kernel = reinterpret_cast<const void *>(
                h3_linear_int8_grouped_r64_kernel<0, 0, 128>);
    } else if (use_bk64) {
        if (host_args.input_dim == 14336 && host_args.group_size == 1024)
            kernel = reinterpret_cast<const void *>(
                h3_linear_int8_grouped_r64_kernel<14336, 1024, 64>);
        else if (host_args.group_size == 1024)
            kernel = reinterpret_cast<const void *>(
                h3_linear_int8_grouped_r64_kernel<0, 1024, 64>);
        else
            kernel = reinterpret_cast<const void *>(
                h3_linear_int8_grouped_r64_kernel<0, 0, 64>);
    } else if (host_args.input_dim == 14336 && host_args.group_size == 1024) {
        kernel = reinterpret_cast<const void *>(
            h3_linear_int8_grouped_r64_kernel<14336, 1024, 32>);
    } else if (host_args.group_size == 1024) {
        kernel = reinterpret_cast<const void *>(
            h3_linear_int8_grouped_r64_kernel<0, 1024, 32>);
    } else {
        kernel = reinterpret_cast<const void *>(
            h3_linear_int8_grouped_r64_kernel<0, 0, 32>);
    }
    return hipLaunchKernel(kernel, grid, block, kernel_args,
                           sizeof(kernel_args), stream) == hipSuccess;
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

/* ── F32→INT8 weight quantization (per-row symmetric) ─────────────────── */

__global__ void __launch_bounds__(256)
h3_quantize_f32_int8_rows_kernel(
    const float *input, int8_t *output, float *scales,
    h3_int8_quant_args args) {
    uint32_t row = blockIdx.x;
    uint32_t columns = args.columns;
    float clip = args.clip;
    const float *row_ptr = input + (size_t)row * columns;
    int8_t *out_ptr = output + (size_t)row * columns;

    /* Pass 1: find max abs using float4 vectorized loads */
    float local_max = 0.0f;
    uint32_t col4 = columns / 4u;
    uint32_t tail_start = col4 * 4u;
    const float4 *row4 = reinterpret_cast<const float4 *>(row_ptr);
    for (uint32_t i = threadIdx.x; i < col4; i += blockDim.x) {
        float4 v = row4[i];
        float a0 = fabsf(v.x), a1 = fabsf(v.y);
        float a2 = fabsf(v.z), a3 = fabsf(v.w);
        float m = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));
        if (m > local_max) local_max = m;
    }
    for (uint32_t i = tail_start + threadIdx.x; i < columns; i += blockDim.x) {
        float a = fabsf(row_ptr[i]);
        if (a > local_max) local_max = a;
    }
    /* Wave-level max reduction */
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, h3_shfl_xor32(local_max, offset));
    /* Shared-mem only for cross-wave reduction (tiny: blockDim.x/32 floats) */
    __shared__ float smax[16];
    if ((threadIdx.x & 31u) == 0u) smax[threadIdx.x >> 5] = local_max;
    __syncthreads();
    if (threadIdx.x < 16u) {
        float v = (threadIdx.x < (blockDim.x >> 5)) ? smax[threadIdx.x] : 0.0f;
        for (int offset = 8; offset > 0; offset >>= 1)
            v = fmaxf(v, h3_shfl_xor32(v, offset));
        if (threadIdx.x == 0) smax[0] = v;
    }
    __syncthreads();
    float clipped_max = smax[0] * clip;
    float scale = clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
    float inverse = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
    if (threadIdx.x == 0) scales[row] = scale;

    /* Pass 2: quantize with float4 loads + int32 stores (4 int8 at once) */
    int32_t *out4 = reinterpret_cast<int32_t *>(out_ptr);
    for (uint32_t i = threadIdx.x; i < col4; i += blockDim.x) {
        float4 v = row4[i];
        int q0 = max(-127, min(127, (int)rintf(v.x * inverse)));
        int q1 = max(-127, min(127, (int)rintf(v.y * inverse)));
        int q2 = max(-127, min(127, (int)rintf(v.z * inverse)));
        int q3 = max(-127, min(127, (int)rintf(v.w * inverse)));
        out4[i] = (uint32_t)(uint8_t)q0 | ((uint32_t)(uint8_t)q1 << 8) |
                  ((uint32_t)(uint8_t)q2 << 16) | ((uint32_t)(uint8_t)q3 << 24);
    }
    for (uint32_t i = tail_start + threadIdx.x; i < columns; i += blockDim.x) {
        int q = max(-127, min(127, (int)rintf(row_ptr[i] * inverse)));
        out_ptr[i] = (int8_t)q;
    }
}

extern "C" int h3_launch_quantize_f32_int8_rows(
    const float *input, int8_t *output, float *scales,
    const h3_int8_quant_args *args, uint32_t dispatch_rows,
    hipStream_t stream) {
    h3_int8_quant_args host_args = *args;
    void *kernel_args[] = {&input, &output, &scales, &host_args};
    dim3 block(256, 1, 1);
    dim3 grid(dispatch_rows, 1, 1);
    return hipLaunchKernel(
        reinterpret_cast<const void *>(h3_quantize_f32_int8_rows_kernel),
        grid, block, kernel_args, 0, stream) == hipSuccess;
}

/* ── INT8 GEMM with F32 input, F32 output ──────────────────────────── */

template <int K_CONST, int BK>
__global__ void __launch_bounds__(256, 2)
h3_linear_int8_f32_t128_kernel(const int8_t *input, const int8_t *weight,
                               const float *input_scales,
                               const float *weight_scales, float *output,
                               h3_linear_args args) {
    constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
    static_assert(BK == 32 || BK == 64 || BK == 128, "INT8 BK 32/64/128");
    constexpr int LDS_K = BK + 4;
    __shared__ int8_t As[BM][LDS_K];
    __shared__ int8_t Bs[BN][LDS_K];
    __shared__ float scale_a[BM];
    __shared__ float scale_b[BN];
    uint32_t block_row = blockIdx.y * (uint32_t)BM;
    uint32_t block_col = blockIdx.x * (uint32_t)BN;
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t tid = ty * 16u + tx;
    uint32_t input_dim = K_CONST ? (uint32_t)K_CONST : args.input_dim;
    uint32_t k_tiles = input_dim / (uint32_t)BK;
    if (tid < (uint32_t)BM) {
        uint32_t row = block_row + tid;
        scale_a[tid] = row < args.rows ? input_scales[row] : 0.0f;
    }
    if (tid < (uint32_t)BN) {
        uint32_t column = block_col + tid;
        scale_b[tid] =
            column < args.output_dim ? weight_scales[column] : 0.0f;
    }
    int32_t acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0;
    for (uint32_t kt = 0; kt < k_tiles; kt++) {
        uint32_t k0 = kt * (uint32_t)BK;
        h3_load_i8_tile_bmxk<BM, LDS_K, BK>(As, input, block_row, k0, args.rows,
                                             input_dim, tid);
        h3_load_i8_tile_bmxk<BN, LDS_K, BK>(Bs, weight, block_col, k0,
                                             args.output_dim, input_dim, tid);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; k += 16) {
            int32_t av0[TM], av1[TM], av2[TM], av3[TM];
            int32_t bv0[TN], bv1[TN], bv2[TN], bv3[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) {
                uint64_t p0 = *reinterpret_cast<const uint64_t *>(
                    &As[ty * (uint32_t)TM + (uint32_t)i][k]);
                uint64_t p1 = *reinterpret_cast<const uint64_t *>(
                    &As[ty * (uint32_t)TM + (uint32_t)i][k + 8]);
                av0[i] = (int32_t)p0;
                av1[i] = (int32_t)(p0 >> 32);
                av2[i] = (int32_t)p1;
                av3[i] = (int32_t)(p1 >> 32);
            }
#pragma unroll
            for (int j = 0; j < TN; j++) {
                uint64_t p0 = *reinterpret_cast<const uint64_t *>(
                    &Bs[tx * (uint32_t)TN + (uint32_t)j][k]);
                uint64_t p1 = *reinterpret_cast<const uint64_t *>(
                    &Bs[tx * (uint32_t)TN + (uint32_t)j][k + 8]);
                bv0[j] = (int32_t)p0;
                bv1[j] = (int32_t)(p0 >> 32);
                bv2[j] = (int32_t)p1;
                bv3[j] = (int32_t)(p1 >> 32);
            }
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i][j] = h3_i8dot4(av0[i], bv0[j], acc[i][j]);
                    acc[i][j] = h3_i8dot4(av1[i], bv1[j], acc[i][j]);
                    acc[i][j] = h3_i8dot4(av2[i], bv2[j], acc[i][j]);
                    acc[i][j] = h3_i8dot4(av3[i], bv3[j], acc[i][j]);
                }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; i++) {
        uint32_t row = block_row + ty * (uint32_t)TM + (uint32_t)i;
        if (row >= args.rows) continue;
        uint32_t column0 = block_col + tx * (uint32_t)TN;
        float sa = scale_a[ty * (uint32_t)TM + (uint32_t)i];
        for (int j = 0; j < TN; j++) {
            uint32_t column = column0 + (uint32_t)j;
            if (column >= args.output_dim) continue;
            float value = (float)acc[i][j] * sa *
                          scale_b[tx * (uint32_t)TN + (uint32_t)j];
            output[row * args.output_dim + column] = value;
        }
    }
}

extern "C" int h3_launch_linear_int8_f32(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, float *output, const h3_linear_args *args,
    hipStream_t stream) {
    h3_linear_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &input_scales, &weight_scales,
                           &output, &host_args};
    constexpr int BM = 128, BN = 128;
    dim3 block(16, 16, 1);
    dim3 grid((host_args.output_dim + BN - 1u) / BN,
              (host_args.rows + BM - 1u) / BM, 1);
    uint32_t input_dim = host_args.input_dim;
    const void *kernel = nullptr;
    if (input_dim % 128 == 0)
        kernel = reinterpret_cast<const void *>(
            h3_linear_int8_f32_t128_kernel<0, 128>);
    else if (input_dim % 64 == 0)
        kernel = reinterpret_cast<const void *>(
            h3_linear_int8_f32_t128_kernel<0, 64>);
    else if (input_dim % 32 == 0)
        kernel = reinterpret_cast<const void *>(
            h3_linear_int8_f32_t128_kernel<0, 32>);
    else
        kernel = reinterpret_cast<const void *>(
            h3_linear_int8_f32_t128_kernel<0, 32>);
    return hipLaunchKernel(kernel, grid, block, kernel_args, 0, stream) ==
           hipSuccess;
}

/* ── Fused F32→INT8 quantize + INT8 GEMM with F32 output ────────────── *
 *
 * Quantizes F32 input on-the-fly and performs INT8 GEMM in a single kernel
 * launch. Eliminates the separate quantize kernel and INT8 intermediate
 * buffer for the VAE path.
 *
 * Two phases per block:
 *   Phase 1 (scales): cooperatively find per-row max across all input_dim
 *                     columns, compute row scales, store to global memory.
 *   Phase 2 (GEMM):   in the k-tile loop, quantize F32→INT8 on-the-fly from
 *                     global memory into shared memory, run existing INT8 GEMM.
 *
 * Shared memory: (BK+4)*128*2 = 33 KB (BK=128) — fits all ISAs.
 * Grid: (output_dim/128, rows/128, 1). Block: (16, 16, 1) = 256 threads.
 * ─────────────────────────────────────────────────────────────────────── */

template <int K_CONST, int BK>
__global__ void __launch_bounds__(256, 2)
h3_linear_int8_f32_fused_t128_kernel(const float *input, const int8_t *weight,
                                     const float *weight_scales, float *output,
                                     h3_linear_args args) {
    constexpr int BM = 128, BN = 128, TM = 8, TN = 8;
    static_assert(BK == 32 || BK == 64 || BK == 128, "fused INT8 BK 32/64/128");
    constexpr int LDS_K = BK + 4;
    __shared__ int8_t As[BM][LDS_K];
    __shared__ int8_t Bs[BN][LDS_K];
    __shared__ float scale_a[BM];
    __shared__ float scale_b[BN];

    uint32_t block_row = blockIdx.y * (uint32_t)BM;
    uint32_t block_col = blockIdx.x * (uint32_t)BN;
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t tid = ty * 16u + tx;
    uint32_t input_dim = K_CONST ? (uint32_t)K_CONST : args.input_dim;
    uint32_t k_tiles = input_dim / (uint32_t)BK;

    /* ── Phase 1: compute per-row scales from F32 input ── */
    for (uint32_t row_base = tid; row_base < (uint32_t)BM; row_base += 256u) {
        uint32_t row = block_row + row_base;
        float local_max = 0.0f;
        if (row < args.rows) {
            const float *row_ptr = input + (size_t)row * input_dim;
            uint32_t col4 = input_dim / 4u;
            const float4 *row4 = reinterpret_cast<const float4 *>(row_ptr);
            for (uint32_t i = 0; i < col4; i += 64u) {
                float4 v = __ldg(row4 + i);
                float a0 = fabsf(v.x), a1 = fabsf(v.y);
                float a2 = fabsf(v.z), a3 = fabsf(v.w);
                float m = fmaxf(fmaxf(a0, a1), fmaxf(a2, a3));
                if (m > local_max) local_max = m;
            }
            uint32_t tail = col4 * 4u;
            for (uint32_t i = tail + (row_base % 4u); i < input_dim; i += 4u) {
                float a = fabsf(__ldg(row_ptr + i));
                if (a > local_max) local_max = a;
            }
        }
        /* Wave-level max reduction (32 threads) */
        for (int offset = 16; offset > 0; offset >>= 1)
            local_max = fmaxf(local_max, h3_shfl_xor32(local_max, offset));
        /* Cross-wave reduction via shared memory */
        __shared__ float smax[8];
        if ((tid & 31u) == 0u) smax[tid >> 5] = local_max;
        __syncthreads();
        if (tid < 8u) {
            float v = (tid < (256u >> 5)) ? smax[tid] : 0.0f;
            for (int offset = 4; offset > 0; offset >>= 1)
                v = fmaxf(v, h3_shfl_xor32(v, offset));
            if (tid == 0) smax[0] = v;
        }
        __syncthreads();
        float clipped_max = smax[0] * 0.999f;
        float inv = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
        if (row < args.rows) scale_a[row_base] = inv;
    }
    __syncthreads();

    /* Load weight scales */
    if (tid < (uint32_t)BN) {
        uint32_t column = block_col + tid;
        scale_b[tid] =
            column < args.output_dim ? weight_scales[column] : 0.0f;
    }

    int32_t acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0;

    /* ── Phase 2: fused quantize + INT8 GEMM ── */
    for (uint32_t kt = 0; kt < k_tiles; kt++) {
        uint32_t k0 = kt * (uint32_t)BK;

        /* Quantize F32 input tile → INT8 in shared memory */
        {
            constexpr int TOTAL_I8 = BM * LDS_K; /* 128 * 132 = 16896 */
            constexpr int I8_PER_THREAD = (TOTAL_I8 + 255) / 256; /* = 66 */
            for (uint32_t ii = 0; ii < (uint32_t)I8_PER_THREAD; ii++) {
                uint32_t idx = tid + ii * 256u;
                if (idx < (uint32_t)TOTAL_I8) {
                    uint32_t r = idx / (uint32_t)LDS_K;
                    uint32_t c = idx % (uint32_t)LDS_K;
                    uint32_t row = block_row + r;
                    uint32_t col = k0 + c;
                    float val = 0.0f;
                    if (row < args.rows && c < (uint32_t)BK && col < input_dim)
                        val = __ldg(input + (size_t)row * input_dim + col);
                    float inv_a = (r < (uint32_t)BM) ? scale_a[r] : 0.0f;
                    int q = max(-127, min(127, (int)rintf(val * inv_a)));
                    As[r][c] = (int8_t)q;
                }
            }
        }

        /* Load weight tile (unchanged) */
        h3_load_i8_tile_bmxk<BN, LDS_K, BK>(Bs, weight, block_col, k0,
                                             args.output_dim, input_dim, tid);
        __syncthreads();

        /* INT8 dot-product accumulate */
#pragma unroll
        for (int k = 0; k < BK; k += 16) {
            int32_t av0[TM], av1[TM], av2[TM], av3[TM];
            int32_t bv0[TN], bv1[TN], bv2[TN], bv3[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) {
                uint64_t p0 = *reinterpret_cast<const uint64_t *>(
                    &As[ty * (uint32_t)TM + (uint32_t)i][k]);
                uint64_t p1 = *reinterpret_cast<const uint64_t *>(
                    &As[ty * (uint32_t)TM + (uint32_t)i][k + 8]);
                av0[i] = (int32_t)p0;
                av1[i] = (int32_t)(p0 >> 32);
                av2[i] = (int32_t)p1;
                av3[i] = (int32_t)(p1 >> 32);
            }
#pragma unroll
            for (int j = 0; j < TN; j++) {
                uint64_t p0 = *reinterpret_cast<const uint64_t *>(
                    &Bs[tx * (uint32_t)TN + (uint32_t)j][k]);
                uint64_t p1 = *reinterpret_cast<const uint64_t *>(
                    &Bs[tx * (uint32_t)TN + (uint32_t)j][k + 8]);
                bv0[j] = (int32_t)p0;
                bv1[j] = (int32_t)(p0 >> 32);
                bv2[j] = (int32_t)p1;
                bv3[j] = (int32_t)(p1 >> 32);
            }
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++) {
                    acc[i][j] = h3_i8dot4(av0[i], bv0[j], acc[i][j]);
                    acc[i][j] = h3_i8dot4(av1[i], bv1[j], acc[i][j]);
                    acc[i][j] = h3_i8dot4(av2[i], bv2[j], acc[i][j]);
                    acc[i][j] = h3_i8dot4(av3[i], bv3[j], acc[i][j]);
                }
        }
        __syncthreads();
    }

    /* ── Write output with dequantization ── */
#pragma unroll
    for (int i = 0; i < TM; i++) {
        uint32_t row = block_row + ty * (uint32_t)TM + (uint32_t)i;
        if (row >= args.rows) continue;
        uint32_t column0 = block_col + tx * (uint32_t)TN;
        float sa = scale_a[ty * (uint32_t)TM + (uint32_t)i];
        for (int j = 0; j < TN; j++) {
            uint32_t column = column0 + (uint32_t)j;
            if (column >= args.output_dim) continue;
            float value = (float)acc[i][j] * sa *
                          scale_b[tx * (uint32_t)TN + (uint32_t)j];
            output[row * args.output_dim + column] = value;
        }
    }
}

extern "C" int h3_launch_linear_int8_f32_fused(
    const float *input, const int8_t *weight, const float *weight_scales,
    float *output, const h3_linear_args *args, hipStream_t stream) {
    h3_linear_args host_args = *args;
    void *kernel_args[] = {&input, &weight, &weight_scales, &output,
                           &host_args};
    constexpr int BM = 128, BN = 128;
    dim3 block(16, 16, 1);
    dim3 grid((host_args.output_dim + BN - 1u) / BN,
              (host_args.rows + BM - 1u) / BM, 1);
    uint32_t input_dim = host_args.input_dim;
    const void *kernel = nullptr;
    if (input_dim % 128 == 0)
        kernel = reinterpret_cast<const void *>(
            h3_linear_int8_f32_fused_t128_kernel<0, 128>);
    else if (input_dim % 64 == 0)
        kernel = reinterpret_cast<const void *>(
            h3_linear_int8_f32_fused_t128_kernel<0, 64>);
    else if (input_dim % 32 == 0)
        kernel = reinterpret_cast<const void *>(
            h3_linear_int8_f32_fused_t128_kernel<0, 32>);
    else
        kernel = reinterpret_cast<const void *>(
            h3_linear_int8_f32_fused_t128_kernel<0, 32>);
    return hipLaunchKernel(kernel, grid, block, kernel_args, 0, stream) ==
           hipSuccess;
}
