#include "h3_kernels.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

__device__ __host__ inline float h3_bf16_to_f32(uint16_t value) {
    return __bfloat162float(*reinterpret_cast<__hip_bfloat16 *>(&value));
}

__device__ __host__ inline uint16_t h3_f32_to_bf16(float value) {
    __hip_bfloat16 bf16 = __float2bfloat16(value);
    return *reinterpret_cast<uint16_t *>(&bf16);
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

static int h3_launch_1d(uint32_t count, hipStream_t stream,
                        const void *kernel, size_t args_size, void **args) {
    if (!count) return 1;
    uint32_t block = 256;
    uint32_t grid = (count + block - 1u) / block;
    return hipLaunchKernel(kernel, dim3(grid), dim3(block), args, args_size,
                           stream) == hipSuccess;
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
