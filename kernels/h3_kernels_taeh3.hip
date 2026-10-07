#include "h3_kernels.h"

#include <math.h>

namespace {

uint32_t taeh3_grid(uint64_t n) {
    uint64_t blocks = (n + 255ull) / 256ull;
    if (blocks < 1ull) blocks = 1ull;
    if (blocks > 2048ull) blocks = 2048ull;
    return (uint32_t)blocks;
}

__global__ void h3_taeh3_conv2d_kernel(const float *input, const float *weight,
                                       const float *bias, float *output,
                                       h3_taeh3_conv2d_args args) {
    uint64_t spatial = (uint64_t)args.height * args.width;
    uint64_t nout = (uint64_t)args.batch * args.out_channels * spatial;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < nout; idx += stride) {
        uint32_t x = (uint32_t)(idx % args.width);
        uint64_t rest = idx / args.width;
        uint32_t y = (uint32_t)(rest % args.height);
        rest /= args.height;
        uint32_t oc = (uint32_t)(rest % args.out_channels);
        uint32_t n = (uint32_t)(rest / args.out_channels);
        float sum = args.has_bias ? bias[oc] : 0.0f;
        int pad = (int)args.pad;
        for (uint32_t ic = 0; ic < args.in_channels; ic++) {
            for (uint32_t kh = 0; kh < args.kernel; kh++) {
                int sy = (int)y + (int)kh - pad;
                if (sy < 0 || sy >= (int)args.height) continue;
                for (uint32_t kw = 0; kw < args.kernel; kw++) {
                    int sx = (int)x + (int)kw - pad;
                    if (sx < 0 || sx >= (int)args.width) continue;
                    uint64_t in_index =
                        ((((uint64_t)n * args.in_channels + ic) * args.height +
                          (uint32_t)sy) *
                             args.width +
                         (uint32_t)sx);
                    uint64_t w_index =
                        ((((uint64_t)oc * args.in_channels + ic) * args.kernel +
                          kh) *
                             args.kernel +
                         kw);
                    sum = fmaf(input[in_index], weight[w_index], sum);
                }
            }
        }
        output[idx] = sum;
    }
}

__global__ void h3_taeh3_tanh_clamp_kernel(const float *input, float *output,
                                           uint32_t elements) {
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
         i += stride) {
        output[i] = tanhf(input[i] / 3.0f) * 3.0f;
    }
}

__global__ void h3_taeh3_relu_kernel(const float *input, float *output,
                                     uint32_t elements) {
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
         i += stride) {
        float v = input[i];
        output[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void h3_taeh3_add_relu_kernel(const float *left, const float *right,
                                         float *output, uint32_t elements) {
    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
         i += stride) {
        float v = left[i] + right[i];
        output[i] = v > 0.0f ? v : 0.0f;
    }
}

__global__ void h3_taeh3_upsample2_kernel(const float *input, float *output,
                                          h3_taeh3_nchw_args args) {
    uint32_t oh = args.height * 2u;
    uint32_t ow = args.width * 2u;
    uint64_t nout = (uint64_t)args.batch * args.channels * oh * ow;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < nout; idx += stride) {
        uint32_t x = (uint32_t)(idx % ow);
        uint64_t rest = idx / ow;
        uint32_t y = (uint32_t)(rest % oh);
        rest /= oh;
        uint32_t c = (uint32_t)(rest % args.channels);
        uint32_t n = (uint32_t)(rest / args.channels);
        uint32_t sy = y / 2u;
        uint32_t sx = x / 2u;
        uint64_t in_index =
            ((((uint64_t)n * args.channels + c) * args.height + sy) *
                 args.width +
             sx);
        output[idx] = input[in_index];
    }
}

__global__ void h3_taeh3_time_past_kernel(const float *input,
                                          const float *memory, float *output,
                                          h3_taeh3_nchw_args args) {
    uint64_t frame = (uint64_t)args.channels * args.height * args.width;
    uint64_t nout = (uint64_t)args.batch * frame;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < nout; idx += stride) {
        uint32_t n = (uint32_t)(idx / frame);
        uint64_t inner = idx % frame;
        if (n == 0u) output[idx] = memory[inner];
        else output[idx] = input[(uint64_t)(n - 1u) * frame + inner];
    }
}

__global__ void h3_taeh3_cat_channels_kernel(const float *left,
                                             const float *right, float *output,
                                             h3_taeh3_nchw_args args) {
    uint32_t out_c = args.channels * 2u;
    uint64_t plane = (uint64_t)args.height * args.width;
    uint64_t nout = (uint64_t)args.batch * out_c * plane;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < nout; idx += stride) {
        uint32_t x = (uint32_t)(idx % args.width);
        uint64_t rest = idx / args.width;
        uint32_t y = (uint32_t)(rest % args.height);
        rest /= args.height;
        uint32_t c = (uint32_t)(rest % out_c);
        uint32_t n = (uint32_t)(rest / out_c);
        const float *src = c < args.channels ? left : right;
        uint32_t sc = c < args.channels ? c : c - args.channels;
        uint64_t in_index =
            ((((uint64_t)n * args.channels + sc) * args.height + y) *
                 args.width +
             x);
        output[idx] = src[in_index];
    }
}

__global__ void h3_taeh3_pixel_shuffle2_kernel(const float *input,
                                               float *output,
                                               h3_taeh3_nchw_args args) {
    /* Input is NCHW with 12 channels. Output is NCHW RGB, spatially x2,
     * clamped to [0, 1]. */
    uint32_t oh = args.height * 2u;
    uint32_t ow = args.width * 2u;
    uint64_t nout = (uint64_t)args.batch * 3u * oh * ow;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < nout; idx += stride) {
        uint32_t x = (uint32_t)(idx % ow);
        uint64_t rest = idx / ow;
        uint32_t y = (uint32_t)(rest % oh);
        rest /= oh;
        uint32_t c = (uint32_t)(rest % 3u);
        uint32_t n = (uint32_t)(rest / 3u);
        uint32_t sy = y / 2u;
        uint32_t sx = x / 2u;
        uint32_t ic = (c * 2u + (y & 1u)) * 2u + (x & 1u);
        uint64_t in_index =
            ((((uint64_t)n * 12u + ic) * args.height + sy) * args.width + sx);
        float v = input[in_index];
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        output[idx] = v;
    }
}

int launch_count(const void *kernel, uint64_t n, void **args,
                 hipStream_t stream) {
    if (!n) return 1;
    return hipLaunchKernel(kernel, dim3(taeh3_grid(n)), dim3(256), args, 0,
                           stream) == hipSuccess;
}

}  // namespace

extern "C" int h3_launch_taeh3_conv2d(const float *input, const float *weight,
                                      const float *bias, float *output,
                                      const h3_taeh3_conv2d_args *args,
                                      hipStream_t stream) {
    h3_taeh3_conv2d_args host = *args;
    uint64_t n = (uint64_t)host.batch * host.out_channels * host.height *
                 host.width;
    void *kargs[] = {&input, &weight, &bias, &output, &host};
    return launch_count(
        reinterpret_cast<const void *>(h3_taeh3_conv2d_kernel), n, kargs,
        stream);
}

extern "C" int h3_launch_taeh3_tanh_clamp(const float *input, float *output,
                                          uint32_t elements,
                                          hipStream_t stream) {
    void *kargs[] = {&input, &output, &elements};
    return launch_count(
        reinterpret_cast<const void *>(h3_taeh3_tanh_clamp_kernel), elements,
        kargs, stream);
}

extern "C" int h3_launch_taeh3_relu(const float *input, float *output,
                                    uint32_t elements, hipStream_t stream) {
    void *kargs[] = {&input, &output, &elements};
    return launch_count(reinterpret_cast<const void *>(h3_taeh3_relu_kernel),
                        elements, kargs, stream);
}

extern "C" int h3_launch_taeh3_add_relu(const float *left, const float *right,
                                        float *output, uint32_t elements,
                                        hipStream_t stream) {
    void *kargs[] = {&left, &right, &output, &elements};
    return launch_count(
        reinterpret_cast<const void *>(h3_taeh3_add_relu_kernel), elements,
        kargs, stream);
}

extern "C" int h3_launch_taeh3_upsample2(const float *input, float *output,
                                         const h3_taeh3_nchw_args *args,
                                         hipStream_t stream) {
    h3_taeh3_nchw_args host = *args;
    uint64_t n = (uint64_t)host.batch * host.channels * host.height *
                 host.width * 4ull;
    void *kargs[] = {&input, &output, &host};
    return launch_count(
        reinterpret_cast<const void *>(h3_taeh3_upsample2_kernel), n, kargs,
        stream);
}

extern "C" int h3_launch_taeh3_time_past(const float *input,
                                         const float *memory, float *output,
                                         const h3_taeh3_nchw_args *args,
                                         hipStream_t stream) {
    h3_taeh3_nchw_args host = *args;
    uint64_t n = (uint64_t)host.batch * host.channels * host.height *
                 host.width;
    void *kargs[] = {&input, &memory, &output, &host};
    return launch_count(
        reinterpret_cast<const void *>(h3_taeh3_time_past_kernel), n, kargs,
        stream);
}

extern "C" int h3_launch_taeh3_cat_channels(const float *left,
                                            const float *right, float *output,
                                            const h3_taeh3_nchw_args *args,
                                            hipStream_t stream) {
    h3_taeh3_nchw_args host = *args;
    uint64_t n = (uint64_t)host.batch * host.channels * 2ull * host.height *
                 host.width;
    void *kargs[] = {&left, &right, &output, &host};
    return launch_count(
        reinterpret_cast<const void *>(h3_taeh3_cat_channels_kernel), n, kargs,
        stream);
}

extern "C" int h3_launch_taeh3_pixel_shuffle2(const float *input,
                                              float *output,
                                              const h3_taeh3_nchw_args *args,
                                              hipStream_t stream) {
    h3_taeh3_nchw_args host = *args;
    uint64_t n = (uint64_t)host.batch * 3ull * host.height * host.width * 4ull;
    void *kargs[] = {&input, &output, &host};
    return launch_count(
        reinterpret_cast<const void *>(h3_taeh3_pixel_shuffle2_kernel), n,
        kargs, stream);
}
