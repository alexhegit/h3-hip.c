#include "h3_kernels.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

namespace {

uint32_t taeh3_grid(uint64_t n) {
    uint64_t blocks = (n + 255ull) / 256ull;
    if (blocks < 1ull) blocks = 1ull;
    if (blocks > 2048ull) blocks = 2048ull;
    return (uint32_t)blocks;
}

/* One 16x8 spatial tile and a group of output channels. The input halo stays
 * in LDS and is reused across that group, instead of once per channel. */
__global__ void __launch_bounds__(128)
h3_taeh3_conv3x3_tiled_kernel(const float *input, const float *weight,
                              const float *bias, float *output,
                              h3_taeh3_conv2d_args args) {
    constexpr int TW = 16;
    constexpr int TH = 8;
    constexpr int IC = 4;
    constexpr int OC = 32;
    constexpr int HALO = 1;
    __shared__ float tile[IC][TH + 2][TW + 2];
    __shared__ float taps[OC][IC][9];

    const int tid = (int)threadIdx.x;
    const int tx = tid % TW;
    const int ty = tid / TW;
    const int x0 = (int)blockIdx.x * TW;
    const int y0 = (int)blockIdx.y * TH;
    const int out_c = (int)args.out_channels;
    const int groups = (out_c + OC - 1) / OC;
    const int n = (int)blockIdx.z / groups;
    const int group = (int)blockIdx.z - n * groups;
    const int oc0 = group * OC;
    const int ox = x0 + tx;
    const int oy = y0 + ty;
    const int in_c = (int)args.in_channels;
    const int height = (int)args.height;
    const int width = (int)args.width;
    const int live = ox < width && oy < height;
    const int nout = out_c - oc0 < OC ? out_c - oc0 : OC;
    float acc[OC];
#pragma unroll
    for (int oc = 0; oc < OC; oc++) {
        int ocg = oc0 + oc;
        acc[oc] = (oc < nout && args.has_bias) ? bias[ocg] : 0.0f;
    }

    for (int ic0 = 0; ic0 < in_c; ic0 += IC) {
        for (int i = tid; i < IC * (TH + 2) * (TW + 2); i += 128) {
            int ic = i / ((TH + 2) * (TW + 2));
            int rem = i - ic * (TH + 2) * (TW + 2);
            int ly = rem / (TW + 2);
            int lx = rem - ly * (TW + 2);
            int gy = y0 + ly - HALO;
            int gx = x0 + lx - HALO;
            int icg = ic0 + ic;
            float v = 0.0f;
            if (icg < in_c && gy >= 0 && gx >= 0 && gy < height && gx < width) {
                uint64_t in_index =
                    ((((uint64_t)n * in_c + icg) * height + gy) * width + gx);
                v = input[in_index];
            }
            tile[ic][ly][lx] = v;
        }
        for (int i = tid; i < OC * IC * 9; i += 128) {
            int oc = i / (IC * 9);
            int rem = i - oc * IC * 9;
            int ic = rem / 9;
            int k = rem - ic * 9;
            int ocg = oc0 + oc;
            int icg = ic0 + ic;
            float w = 0.0f;
            if (ocg < out_c && icg < in_c) {
                uint64_t w_index =
                    ((((uint64_t)ocg * (uint32_t)in_c + (uint32_t)icg) * 3u +
                      (uint32_t)(k / 3)) *
                         3u +
                     (uint32_t)(k % 3));
                w = weight[w_index];
            }
            taps[oc][ic][k] = w;
        }
        __syncthreads();
        if (live) {
#pragma unroll
            for (int oc = 0; oc < OC; oc++) {
                if (oc >= nout) continue;
#pragma unroll
                for (int ic = 0; ic < IC; ic++) {
#pragma unroll
                    for (int kh = 0; kh < 3; kh++) {
#pragma unroll
                        for (int kw = 0; kw < 3; kw++) {
                            acc[oc] = fmaf(tile[ic][ty + kh][tx + kw],
                                           taps[oc][ic][kh * 3 + kw], acc[oc]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    if (live) {
        for (int oc = 0; oc < nout; oc++) {
            uint64_t out_index =
                ((((uint64_t)n * out_c + (oc0 + oc)) * (uint32_t)height +
                  (uint32_t)oy) *
                     (uint32_t)width +
                 (uint32_t)ox);
            output[out_index] = acc[oc];
        }
    }
}

__global__ void __launch_bounds__(128)
h3_taeh3_conv1x1_tiled_kernel(const float *input, const float *weight,
                              const float *bias, float *output,
                              h3_taeh3_conv2d_args args) {
    constexpr int TW = 16;
    constexpr int TH = 8;
    constexpr int IC = 16;
    constexpr int OC = 16;
    __shared__ float tile[IC][TH][TW];
    __shared__ float taps[OC][IC];

    const int tid = (int)threadIdx.x;
    const int tx = tid % TW;
    const int ty = tid / TW;
    const int x0 = (int)blockIdx.x * TW;
    const int y0 = (int)blockIdx.y * TH;
    const int out_c = (int)args.out_channels;
    const int groups = (out_c + OC - 1) / OC;
    const int n = (int)blockIdx.z / groups;
    const int group = (int)blockIdx.z - n * groups;
    const int oc0 = group * OC;
    const int ox = x0 + tx;
    const int oy = y0 + ty;
    const int in_c = (int)args.in_channels;
    const int height = (int)args.height;
    const int width = (int)args.width;
    const int live = ox < width && oy < height;
    const int nout = out_c - oc0 < OC ? out_c - oc0 : OC;
    float acc[OC];
#pragma unroll
    for (int oc = 0; oc < OC; oc++) {
        int ocg = oc0 + oc;
        acc[oc] = (oc < nout && args.has_bias) ? bias[ocg] : 0.0f;
    }

    for (int ic0 = 0; ic0 < in_c; ic0 += IC) {
        for (int i = tid; i < IC * TH * TW; i += 128) {
            int ic = i / (TH * TW);
            int rem = i - ic * TH * TW;
            int ly = rem / TW;
            int lx = rem - ly * TW;
            int gy = y0 + ly;
            int gx = x0 + lx;
            int icg = ic0 + ic;
            float v = 0.0f;
            if (icg < in_c && gy < height && gx < width) {
                uint64_t in_index =
                    ((((uint64_t)n * in_c + icg) * height + gy) * width + gx);
                v = input[in_index];
            }
            tile[ic][ly][lx] = v;
        }
        for (int i = tid; i < OC * IC; i += 128) {
            int oc = i / IC;
            int ic = i - oc * IC;
            int ocg = oc0 + oc;
            int icg = ic0 + ic;
            taps[oc][ic] =
                (ocg < out_c && icg < in_c)
                    ? weight[(uint64_t)ocg * (uint32_t)in_c + (uint32_t)icg]
                    : 0.0f;
        }
        __syncthreads();
        if (live) {
#pragma unroll
            for (int oc = 0; oc < OC; oc++) {
                if (oc >= nout) continue;
#pragma unroll
                for (int ic = 0; ic < IC; ic++)
                    acc[oc] = fmaf(tile[ic][ty][tx], taps[oc][ic], acc[oc]);
            }
        }
        __syncthreads();
    }
    if (live) {
        for (int oc = 0; oc < nout; oc++) {
            uint64_t out_index =
                ((((uint64_t)n * out_c + (oc0 + oc)) * (uint32_t)height +
                  (uint32_t)oy) *
                     (uint32_t)width +
                 (uint32_t)ox);
            output[out_index] = acc[oc];
        }
    }
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
    const char *naive = getenv("H3_TAEH3_NAIVE");
    int use_naive = naive && strcmp(naive, "0") != 0;
    uint32_t group = host.kernel == 1u ? 16u : 32u;
    uint32_t groups = (host.out_channels + group - 1u) / group;
    uint64_t z = (uint64_t)host.batch * groups;
    int tiled = !use_naive && z > 0 && z <= 65535ull && host.height > 0 &&
                host.width > 0 &&
                ((host.kernel == 3u && host.pad == 1u) ||
                 (host.kernel == 1u && host.pad == 0u));
    if (tiled) {
        dim3 grid((host.width + 15u) / 16u, (host.height + 7u) / 8u, (uint32_t)z);
        const void *kernel =
            host.kernel == 1u
                ? reinterpret_cast<const void *>(h3_taeh3_conv1x1_tiled_kernel)
                : reinterpret_cast<const void *>(h3_taeh3_conv3x3_tiled_kernel);
        return hipLaunchKernel(kernel, grid, dim3(128), kargs, 0, stream) ==
               hipSuccess;
    }
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
