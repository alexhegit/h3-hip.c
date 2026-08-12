#include "h3_gpu.h"
#include "kernels/h3_kernels.h"

#include <hip/hip_runtime_api.h>

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct h3_gpu_tensor {
    void *host;
    size_t elements;
    size_t bytes;
    h3_gpu_dtype dtype;
    struct h3_gpu *owner;
};

struct h3_gpu {
    int device_id;
    hipStream_t stream;
    char last_error[512];
    h3_gpu_stats stats;
    char profile_label[128];
    int in_command;
};

static struct h3_gpu_tensor *tensor_ptr(const h3_gpu_tensor *tensor) {
    return (struct h3_gpu_tensor *)(void *)tensor;
}

static struct h3_gpu *gpu_ptr(const h3_gpu *gpu) {
    return (struct h3_gpu *)(void *)gpu;
}

static void h3_hip_set_error(struct h3_gpu *gpu, const char *format, ...) {
    if (!gpu) return;
    va_list args;
    va_start(args, format);
    vsnprintf(gpu->last_error, sizeof(gpu->last_error), format, args);
    va_end(args);
}

int h3_hip_unimplemented(struct h3_gpu *gpu, const char *name);

static struct h3_gpu_tensor *h3_hip_tensor_new(struct h3_gpu *gpu,
                                               const void *values,
                                               size_t elements,
                                               size_t item_size,
                                               h3_gpu_dtype dtype) {
    if (!gpu || elements > SIZE_MAX / item_size) {
        return NULL;
    }
    size_t bytes = elements * item_size;
    struct h3_gpu_tensor *tensor = calloc(1, sizeof(*tensor));
    if (!tensor) {
        h3_hip_set_error(gpu, "out of memory allocating tensor metadata");
        return NULL;
    }
    if (hipHostMalloc(&tensor->host, bytes > 0 ? bytes : 1, hipHostMallocDefault)
        != hipSuccess) {
        free(tensor);
        h3_hip_set_error(gpu, "cannot allocate %zu-byte HIP buffer", bytes);
        return NULL;
    }
    if (values && bytes) {
        memcpy(tensor->host, values, bytes);
    }
    tensor->elements = elements;
    tensor->bytes = bytes;
    tensor->dtype = dtype;
    tensor->owner = gpu;
    gpu->stats.allocated_bytes += bytes;
    gpu->stats.live_bytes += bytes;
    if (gpu->stats.live_bytes > gpu->stats.peak_live_bytes) {
        gpu->stats.peak_live_bytes = gpu->stats.live_bytes;
    }
    gpu->stats.tensor_allocations++;
    return tensor;
}

static int h3_hip_require_bf16(struct h3_gpu *gpu, const h3_gpu_tensor *tensor,
                               size_t elements, const char *label) {
    if (!tensor || tensor_ptr(tensor)->dtype != H3_GPU_BF16) {
        h3_hip_set_error(gpu, "%s tensor dtype mismatch", label);
        return 0;
    }
    if (elements && h3_gpu_tensor_elements(tensor) < elements) {
        h3_hip_set_error(gpu, "%s tensor is too small", label);
        return 0;
    }
    return 1;
}

static int h3_hip_require_f32(struct h3_gpu *gpu, const h3_gpu_tensor *tensor,
                              size_t elements, const char *label) {
    if (!tensor || tensor_ptr(tensor)->dtype != H3_GPU_F32) {
        h3_hip_set_error(gpu, "%s tensor dtype mismatch", label);
        return 0;
    }
    if (elements && h3_gpu_tensor_elements(tensor) < elements) {
        h3_hip_set_error(gpu, "%s tensor is too small", label);
        return 0;
    }
    return 1;
}

h3_gpu *h3_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size) {
    (void)shader_source_path;
    struct h3_gpu *gpu = calloc(1, sizeof(*gpu));
    if (!gpu) {
        if (error && error_size) {
            snprintf(error, error_size, "out of memory creating HIP context");
        }
        return NULL;
    }
    if (hipSetDevice(0) != hipSuccess ||
        hipStreamCreate(&gpu->stream) != hipSuccess) {
        if (error && error_size) {
            snprintf(error, error_size, "cannot initialize HIP device/stream");
        }
        free(gpu);
        return NULL;
    }
    gpu->device_id = 0;
    snprintf(gpu->profile_label, sizeof(gpu->profile_label), "HIP context");
    return (h3_gpu *)gpu;
}

void h3_gpu_free(h3_gpu *gpu) {
    if (!gpu) return;
    struct h3_gpu *ctx = gpu_ptr(gpu);
    hipStreamDestroy(ctx->stream);
    free(ctx);
}

int h3_gpu_is_m5(const h3_gpu *gpu) {
    (void)gpu;
    return 0;
}

int h3_gpu_has_nax_mlp(const h3_gpu *gpu) {
    (void)gpu;
    return 0;
}

int h3_gpu_has_int8_mlp(const h3_gpu *gpu) {
    (void)gpu;
    return 0;
}

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements) {
    return (h3_gpu_tensor *)h3_hip_tensor_new(gpu_ptr(gpu), NULL, elements,
                                            sizeof(float), H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements) {
    return (h3_gpu_tensor *)h3_hip_tensor_new(gpu_ptr(gpu), NULL, elements,
                                            sizeof(uint16_t), H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements) {
    return (h3_gpu_tensor *)h3_hip_tensor_new(gpu_ptr(gpu), NULL, elements,
                                            sizeof(int8_t), H3_GPU_I8);
}

h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements) {
    return (h3_gpu_tensor *)h3_hip_tensor_new(gpu_ptr(gpu), values, elements,
                                            sizeof(float), H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements) {
    return (h3_gpu_tensor *)h3_hip_tensor_new(gpu_ptr(gpu), values, elements,
                                            sizeof(uint16_t), H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements) {
    return (h3_gpu_tensor *)h3_hip_tensor_new(gpu_ptr(gpu), values, elements,
                                            sizeof(uint32_t), H3_GPU_U32);
}

h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_bf16(gpu, elements);
    if (!tensor) return NULL;
    char detail[256];
    if (!h3_gpu_tensor_read_file_bf16(tensor, path, file_offset, elements,
                                      detail, sizeof(detail))) {
        h3_hip_set_error(ctx, "%s", detail);
        h3_gpu_tensor_free(tensor);
        return NULL;
    }
    return tensor;
}

h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    h3_gpu_tensor *tensor = h3_gpu_tensor_new_f32(gpu, elements);
    if (!tensor) return NULL;
    if (elements > SIZE_MAX / sizeof(float)) {
        h3_gpu_tensor_free(tensor);
        h3_hip_set_error(ctx, "F32 weight load size overflow");
        return NULL;
    }
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        h3_gpu_tensor_free(tensor);
        h3_hip_set_error(ctx, "cannot open %s: %s", path, strerror(errno));
        return NULL;
    }
    ssize_t read_bytes = pread(fd, tensor_ptr(tensor)->host,
                               elements * sizeof(float), (off_t)file_offset);
    close(fd);
    if (read_bytes < 0 ||
        (size_t)read_bytes != elements * sizeof(float)) {
        h3_gpu_tensor_free(tensor);
        h3_hip_set_error(ctx, "cannot read %zu F32 values from %s", elements,
                         path);
        return NULL;
    }
    return tensor;
}

int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || !path || !*path || elements > obj->elements) {
        if (error && error_size) {
            snprintf(error, error_size, "invalid BF16 file read request");
        }
        return 0;
    }
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        if (error && error_size) {
            snprintf(error, error_size, "cannot open %s: %s", path,
                     strerror(errno));
        }
        return 0;
    }
    ssize_t read_bytes = pread(fd, obj->host, elements * sizeof(uint16_t),
                               (off_t)file_offset);
    close(fd);
    if (read_bytes < 0 ||
        (size_t)read_bytes != elements * sizeof(uint16_t)) {
        if (error && error_size) {
            snprintf(error, error_size, "cannot read %zu BF16 values from %s",
                     elements, path);
        }
        return 0;
    }
    return 1;
}

int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size) {
    return h3_gpu_tensor_read_file_bf16(tensor, path, file_offset, elements,
                                       error, error_size);
}

void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj) return;
    struct h3_gpu *gpu = obj->owner;
    if (obj->host) {
        hipHostFree(obj->host);
    }
    if (gpu) {
        gpu->stats.live_bytes -= obj->bytes;
    }
    free(obj);
}

size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor) {
    return tensor ? tensor_ptr(tensor)->elements : 0;
}

h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor) {
    return tensor ? tensor_ptr(tensor)->dtype : H3_GPU_F32;
}

int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || obj->dtype != H3_GPU_F32 || !values ||
        elements > obj->elements) {
        return 0;
    }
    memcpy(values, obj->host, elements * sizeof(float));
    return 1;
}

int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || obj->dtype != H3_GPU_F32 || !values ||
        source_offset > obj->elements ||
        elements > obj->elements - source_offset) {
        return 0;
    }
    memcpy(values, (float *)obj->host + source_offset,
           elements * sizeof(float));
    return 1;
}

int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || obj->dtype != H3_GPU_BF16 || !values ||
        elements > obj->elements) {
        return 0;
    }
    memcpy(values, obj->host, elements * sizeof(uint16_t));
    return 1;
}

int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || obj->dtype != H3_GPU_F32 || !values ||
        elements > obj->elements) {
        return 0;
    }
    memcpy(obj->host, values, elements * sizeof(float));
    return 1;
}

int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || obj->dtype != H3_GPU_F32 || !values ||
        destination_offset > obj->elements ||
        elements > obj->elements - destination_offset) {
        return 0;
    }
    memcpy((float *)obj->host + destination_offset, values,
           elements * sizeof(float));
    return 1;
}

int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || obj->dtype != H3_GPU_BF16 || !values ||
        elements > obj->elements) {
        return 0;
    }
    memcpy(obj->host, values, elements * sizeof(uint16_t));
    return 1;
}

int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || obj->dtype != H3_GPU_BF16 || !values ||
        destination_offset > obj->elements ||
        elements > obj->elements - destination_offset) {
        return 0;
    }
    memcpy((uint16_t *)obj->host + destination_offset, values,
           elements * sizeof(uint16_t));
    return 1;
}

int h3_gpu_begin(h3_gpu *gpu) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx) return 0;
    ctx->in_command = 1;
    return 1;
}

int h3_gpu_continue(h3_gpu *gpu) {
    return h3_gpu_submit(gpu) && h3_gpu_begin(gpu);
}

int h3_gpu_submit(h3_gpu *gpu) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx) return 0;
    if (hipStreamSynchronize(ctx->stream) != hipSuccess) {
        h3_hip_set_error(ctx, "HIP stream synchronization failed");
        return 0;
    }
    ctx->stats.submissions++;
    ctx->in_command = 0;
    return 1;
}

const char *h3_gpu_error(const h3_gpu *gpu) {
    return gpu ? gpu_ptr(gpu)->last_error : "invalid HIP context";
}

int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats) {
    if (!gpu || !stats) return 0;
    *stats = gpu_ptr(gpu)->stats;
    return 1;
}

void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label) {
    if (!gpu || !label) return;
    snprintf(gpu_ptr(gpu)->profile_label, sizeof(gpu_ptr(gpu)->profile_label),
             "%s", label);
}

void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase) {
    (void)gpu;
    (void)phase;
}

int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_f32(ctx, input, elements, "cast input") ||
        !h3_hip_require_bf16(ctx, output, elements, "cast output")) {
        return 0;
    }
    if (!h3_launch_cast_f32_to_bf16((const float *)tensor_ptr(input)->host,
                                    (uint16_t *)tensor_ptr(output)->host,
                                    elements, ctx->stream)) {
        h3_hip_set_error(ctx, "h3_cast_f32_to_bf16 launch failed");
        return 0;
    }
    ctx->stats.direct_dispatches++;
    return 1;
}

int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_bf16(ctx, input, elements, "cast input") ||
        !h3_hip_require_f32(ctx, output, elements, "cast output")) {
        return 0;
    }
    if (!h3_launch_cast_bf16_to_f32((const uint16_t *)tensor_ptr(input)->host,
                                    (float *)tensor_ptr(output)->host,
                                    elements, ctx->stream)) {
        h3_hip_set_error(ctx, "h3_cast_bf16_to_f32 launch failed");
        return 0;
    }
    ctx->stats.direct_dispatches++;
    return 1;
}

int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_bf16(ctx, left, elements, "add left") ||
        !h3_hip_require_bf16(ctx, right, elements, "add right") ||
        !h3_hip_require_bf16(ctx, output, elements, "add output")) {
        return 0;
    }
    if (!h3_launch_add_bf16((const uint16_t *)tensor_ptr(left)->host,
                            (const uint16_t *)tensor_ptr(right)->host,
                            (uint16_t *)tensor_ptr(output)->host, elements,
                            ctx->stream)) {
        h3_hip_set_error(ctx, "h3_add_bf16 launch failed");
        return 0;
    }
    ctx->stats.direct_dispatches++;
    return 1;
}

int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination,
                     size_t destination_offset,
                     const h3_gpu_tensor *source, size_t source_offset,
                     size_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_bf16(ctx, source, source_offset + elements,
                             "copy source") ||
        !h3_hip_require_bf16(ctx, destination, destination_offset + elements,
                             "copy destination")) {
        return 0;
    }
    memcpy((uint16_t *)tensor_ptr(destination)->host + destination_offset,
           (const uint16_t *)tensor_ptr(source)->host + source_offset,
           elements * sizeof(uint16_t));
    ctx->stats.blit_copies++;
    return 1;
}

static int h3_hip_require_u32(struct h3_gpu *gpu, const h3_gpu_tensor *tensor,
                              size_t elements, const char *label) {
    if (!tensor || tensor_ptr(tensor)->dtype != H3_GPU_U32) {
        h3_hip_set_error(gpu, "%s tensor dtype mismatch", label);
        return 0;
    }
    if (elements && h3_gpu_tensor_elements(tensor) < elements) {
        h3_hip_set_error(gpu, "%s tensor is too small", label);
        return 0;
    }
    return 1;
}

static int h3_hip_launch_ok(struct h3_gpu *gpu, int ok, const char *name) {
    if (ok) {
        gpu->stats.direct_dispatches++;
        return 1;
    }
    h3_hip_set_error(gpu, "%s launch failed", name);
    return 0;
}

int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_bf16(ctx, left, elements, "sub left") ||
        !h3_hip_require_bf16(ctx, right, elements, "sub right") ||
        !h3_hip_require_bf16(ctx, output, elements, "sub output")) {
        return 0;
    }
    return h3_hip_launch_ok(ctx, h3_launch_sub_bf16(
        (const uint16_t *)tensor_ptr(left)->host,
        (const uint16_t *)tensor_ptr(right)->host,
        (uint16_t *)tensor_ptr(output)->host, elements, ctx->stream),
        "h3_sub_bf16");
}

int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_bf16(ctx, input, elements, "SiLU input") ||
        !h3_hip_require_bf16(ctx, output, elements, "SiLU output")) {
        return 0;
    }
    return h3_hip_launch_ok(ctx, h3_launch_silu_bf16(
        (const uint16_t *)tensor_ptr(input)->host,
        (uint16_t *)tensor_ptr(output)->host, elements, ctx->stream),
        "h3_silu_bf16");
}

int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_bf16(ctx, input, elements, "GELU input") ||
        !h3_hip_require_bf16(ctx, output, elements, "GELU output")) {
        return 0;
    }
    h3_gelu_bf16_args args = {elements, approximate ? 1u : 0u};
    return h3_hip_launch_ok(ctx, h3_launch_gelu_bf16(
        (const uint16_t *)tensor_ptr(input)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_gelu_bf16");
}

int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t fused_count = (size_t)rows * width * 2;
    size_t out_count = (size_t)rows * width;
    if (!ctx || !rows || !width ||
        !h3_hip_require_bf16(ctx, fused, fused_count, "SwiGLU input") ||
        !h3_hip_require_bf16(ctx, output, out_count, "SwiGLU output")) {
        return 0;
    }
    h3_swiglu_args args = {rows, width};
    return h3_hip_launch_ok(ctx, h3_launch_swiglu_bf16(
        (const uint16_t *)tensor_ptr(fused)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_swiglu_bf16");
}

int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width ||
        !h3_hip_require_bf16(ctx, input, count, "RMSNorm input") ||
        !h3_hip_require_bf16(ctx, weight, width, "RMSNorm weight") ||
        !h3_hip_require_bf16(ctx, output, count, "RMSNorm output")) {
        return 0;
    }
    h3_norm_args args = {rows, width, epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_rms_norm_bf16(
        (const uint16_t *)tensor_ptr(input)->host,
        (const uint16_t *)tensor_ptr(weight)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_rms_norm_bf16");
}

int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width ||
        !h3_hip_require_bf16(ctx, input, count, "LayerNorm input") ||
        !h3_hip_require_bf16(ctx, weight, width, "LayerNorm weight") ||
        !h3_hip_require_bf16(ctx, bias, width, "LayerNorm bias") ||
        !h3_hip_require_bf16(ctx, output, count, "LayerNorm output")) {
        return 0;
    }
    h3_norm_args args = {rows, width, epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_layer_norm_bf16(
        (const uint16_t *)tensor_ptr(input)->host,
        (const uint16_t *)tensor_ptr(weight)->host,
        (const uint16_t *)tensor_ptr(bias)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_layer_norm_bf16");
}

int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!ctx || !rows || !input_dim || !output_dim ||
        !h3_hip_require_bf16(ctx, input, input_count, "linear input") ||
        !h3_hip_require_bf16(ctx, weight, weight_count, "linear weight") ||
        !h3_hip_require_bf16(ctx, output, output_count, "linear output") ||
        (bias && !h3_hip_require_bf16(ctx, bias, output_dim, "linear bias"))) {
        return 0;
    }
    const h3_gpu_tensor *bias_tensor = bias ? bias : input;
    h3_linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    return h3_hip_launch_ok(ctx, h3_launch_linear_bf16(
        (const uint16_t *)tensor_ptr(input)->host,
        (const uint16_t *)tensor_ptr(weight)->host,
        (const uint16_t *)tensor_ptr(bias_tensor)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_linear_bf16");
}

int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width || gate_slot >= slots ||
        !h3_hip_require_bf16(ctx, residual, count, "gate residual") ||
        !h3_hip_require_bf16(ctx, branch, count, "gate branch") ||
        !h3_hip_require_bf16(ctx, modulation, 1, "gate modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "gate row map") ||
        !h3_hip_require_bf16(ctx, output, count, "gate output")) {
        return 0;
    }
    h3_gate_args args = {rows, width, slots, gate_slot};
    return h3_hip_launch_ok(ctx, h3_launch_gate_bf16(
        (const uint16_t *)tensor_ptr(residual)->host,
        (const uint16_t *)tensor_ptr(branch)->host,
        (const uint16_t *)tensor_ptr(modulation)->host,
        (const uint32_t *)tensor_ptr(row_map)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_gate_bf16");
}

int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width || shift_slot >= slots ||
        scale_slot >= slots ||
        !h3_hip_require_bf16(ctx, input, count, "AdaLN input") ||
        !h3_hip_require_bf16(ctx, norm_weight, width, "AdaLN norm") ||
        !h3_hip_require_bf16(ctx, modulation, 1, "AdaLN modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "AdaLN row map") ||
        !h3_hip_require_bf16(ctx, output, count, "AdaLN output")) {
        return 0;
    }
    h3_adaln_args args = {rows, width, slots, shift_slot, scale_slot,
                          epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_adaln_bf16(
        (const uint16_t *)tensor_ptr(input)->host,
        (const uint16_t *)tensor_ptr(norm_weight)->host,
        (const uint16_t *)tensor_ptr(modulation)->host,
        (const uint32_t *)tensor_ptr(row_map)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_adaln_bf16");
}

int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *norm_weight,
                             const h3_gpu_tensor *modulation,
                             const h3_gpu_tensor *row_map, uint32_t rows,
                             uint32_t width, uint32_t slots,
                             uint32_t shift_slot, uint32_t scale_slot,
                             float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width || shift_slot >= slots ||
        scale_slot >= slots ||
        input_offset > tensor_ptr(input)->elements ||
        count > tensor_ptr(input)->elements - input_offset ||
        !h3_hip_require_bf16(ctx, norm_weight, width, "AdaLN norm") ||
        !h3_hip_require_bf16(ctx, modulation, 1, "AdaLN modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "AdaLN row map") ||
        !h3_hip_require_bf16(ctx, output, count, "AdaLN output")) {
        return 0;
    }
    h3_adaln_args args = {rows, width, slots, shift_slot, scale_slot,
                          epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_adaln_bf16(
        (const uint16_t *)tensor_ptr(input)->host + input_offset,
        (const uint16_t *)tensor_ptr(norm_weight)->host,
        (const uint16_t *)tensor_ptr(modulation)->host,
        (const uint32_t *)tensor_ptr(row_map)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_adaln_bf16_offset");
}

int h3_gpu_gate_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *residual,
                           const h3_gpu_tensor *branch,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *gate_modulation,
                           const h3_gpu_tensor *norm_modulation,
                           const h3_gpu_tensor *row_map, uint32_t rows,
                           uint32_t width, uint32_t slots, uint32_t gate_slot,
                           uint32_t shift_slot, uint32_t scale_slot,
                           float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width || width > 5376u || gate_slot >= slots ||
        shift_slot >= slots || scale_slot >= slots ||
        !h3_hip_require_bf16(ctx, residual, count, "gate AdaLN residual") ||
        !h3_hip_require_bf16(ctx, branch, count, "gate AdaLN branch") ||
        !h3_hip_require_bf16(ctx, norm_weight, width, "gate AdaLN norm") ||
        !h3_hip_require_bf16(ctx, gate_modulation, 1,
                             "gate AdaLN gate modulation") ||
        !h3_hip_require_bf16(ctx, norm_modulation, 1,
                             "gate AdaLN norm modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "gate AdaLN row map") ||
        !h3_hip_require_bf16(ctx, gated_residual, count,
                             "gate AdaLN gated residual") ||
        !h3_hip_require_bf16(ctx, output, count, "gate AdaLN output")) {
        return 0;
    }
    h3_gate_adaln_args args = {rows, width, slots, gate_slot, shift_slot,
                               scale_slot, epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_gate_adaln_bf16(
        (const uint16_t *)tensor_ptr(residual)->host,
        (const uint16_t *)tensor_ptr(branch)->host,
        (const uint16_t *)tensor_ptr(gate_modulation)->host,
        (const uint32_t *)tensor_ptr(row_map)->host,
        (const uint16_t *)tensor_ptr(norm_weight)->host,
        (const uint16_t *)tensor_ptr(norm_modulation)->host,
        (uint16_t *)tensor_ptr(gated_residual)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_gate_adaln_bf16");
}

static int h3_gpu_qkv_rope_bf16_layout(h3_gpu *gpu, h3_gpu_tensor *query,
                                       h3_gpu_tensor *key,
                                       h3_gpu_tensor *value,
                                       const h3_gpu_tensor *qkv,
                                       const h3_gpu_tensor *q_norm,
                                       const h3_gpu_tensor *k_norm,
                                       const h3_gpu_tensor *rope_cos,
                                       const h3_gpu_tensor *rope_sin,
                                       uint32_t sequence, uint32_t heads,
                                       uint32_t head_dim, uint32_t rope_half,
                                       uint32_t grouped, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t inner = (size_t)heads * head_dim;
    size_t projected = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!ctx || !sequence || !heads || !head_dim ||
        !h3_hip_require_bf16(ctx, qkv, projected * 3, "QKV input") ||
        !h3_hip_require_bf16(ctx, q_norm, head_dim, "Q norm") ||
        !h3_hip_require_bf16(ctx, k_norm, head_dim, "K norm") ||
        !h3_hip_require_bf16(ctx, rope_cos, rope_count, "RoPE cosine") ||
        !h3_hip_require_bf16(ctx, rope_sin, rope_count, "RoPE sine") ||
        !h3_hip_require_bf16(ctx, query, projected, "query output") ||
        !h3_hip_require_bf16(ctx, key, projected, "key output") ||
        !h3_hip_require_bf16(ctx, value, projected, "value output")) {
        return 0;
    }
    h3_qkv_args args = {sequence, heads, head_dim, rope_half, grouped,
                        epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_qkv_rope_bf16(
        (const uint16_t *)tensor_ptr(qkv)->host,
        (const uint16_t *)tensor_ptr(q_norm)->host,
        (const uint16_t *)tensor_ptr(k_norm)->host,
        (const uint16_t *)tensor_ptr(rope_cos)->host,
        (const uint16_t *)tensor_ptr(rope_sin)->host,
        (uint16_t *)tensor_ptr(query)->host,
        (uint16_t *)tensor_ptr(key)->host,
        (uint16_t *)tensor_ptr(value)->host, &args, ctx->stream),
        "h3_qkv_rope_bf16");
}

int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                         h3_gpu_tensor *key, h3_gpu_tensor *value,
                         const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon) {
    return h3_gpu_qkv_rope_bf16_layout(gpu, query, key, value, qkv, q_norm,
                                       k_norm, rope_cos, rope_sin, sequence,
                                       heads, head_dim, rope_half, 0,
                                       epsilon);
}

int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                h3_gpu_tensor *key, h3_gpu_tensor *value,
                                const h3_gpu_tensor *qkv,
                                const h3_gpu_tensor *q_norm,
                                const h3_gpu_tensor *k_norm,
                                const h3_gpu_tensor *rope_cos,
                                const h3_gpu_tensor *rope_sin,
                                uint32_t sequence, uint32_t heads,
                                uint32_t head_dim, uint32_t rope_half,
                                float epsilon) {
    return h3_gpu_qkv_rope_bf16_layout(gpu, query, key, value, qkv, q_norm,
                                       k_norm, rope_cos, rope_sin, sequence,
                                       heads, head_dim, rope_half, 1,
                                       epsilon);
}

int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)sequence * heads * head_dim;
    if (!ctx || !sequence || !heads || !head_dim ||
        !h3_hip_require_bf16(ctx, query, count, "SDPA query") ||
        !h3_hip_require_bf16(ctx, key, count, "SDPA key") ||
        !h3_hip_require_bf16(ctx, value, count, "SDPA value") ||
        !h3_hip_require_bf16(ctx, output, count, "SDPA output")) {
        return 0;
    }
    h3_sdpa_args args = {sequence, heads, head_dim, scale};
    return h3_hip_launch_ok(ctx, h3_launch_sdpa_bf16(
        (const uint16_t *)tensor_ptr(query)->host,
        (const uint16_t *)tensor_ptr(key)->host,
        (const uint16_t *)tensor_ptr(value)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_sdpa_bf16");
}

int h3_gpu_sdpa_bf16_head_major_output(h3_gpu *gpu, h3_gpu_tensor *output,
                                       const h3_gpu_tensor *query,
                                       const h3_gpu_tensor *key,
                                       const h3_gpu_tensor *value,
                                       uint32_t sequence, uint32_t heads,
                                       uint32_t head_dim, float scale) {
    return h3_gpu_sdpa_bf16(gpu, output, query, key, value, sequence, heads,
                            head_dim, scale);
}

int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !rows || !input_dim || !hidden_dim || !output_dim) {
        return 0;
    }
    h3_gpu_tensor *fc1_out = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * hidden_dim * 2);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * hidden_dim);
    if (!fc1_out || !activated) {
        h3_gpu_tensor_free(fc1_out);
        h3_gpu_tensor_free(activated);
        h3_hip_set_error(ctx, "MLP temporary allocation failed");
        return 0;
    }
    int ok = h3_gpu_linear_bf16(gpu, fc1_out, input, fc1_weight, NULL, rows,
                                input_dim, hidden_dim * 2) &&
             h3_gpu_swiglu_bf16(gpu, activated, fc1_out, rows, hidden_dim) &&
             h3_gpu_linear_bf16(gpu, output, activated, fc2_weight, NULL,
                                rows, hidden_dim, output_dim);
    h3_gpu_tensor_free(fc1_out);
    h3_gpu_tensor_free(activated);
    if (!ok) {
        h3_hip_set_error(ctx, "h3_mlp_bf16 failed");
    }
    return ok;
}

int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_f32(ctx, sample, sample_offset + elements,
                            "Euler sample") ||
        !h3_hip_require_bf16(ctx, last, elements, "Euler last") ||
        !h3_hip_require_bf16(ctx, previous, elements, "Euler previous")) {
        return 0;
    }
    h3_euler_args args = {(uint32_t)sample_offset, elements, delta, ratio};
    return h3_hip_launch_ok(ctx, h3_launch_euler_bf16(
        (float *)tensor_ptr(sample)->host,
        (const uint16_t *)tensor_ptr(last)->host,
        (const uint16_t *)tensor_ptr(previous)->host, &args, ctx->stream),
        "h3_euler_bf16");
}

int h3_hip_unimplemented(struct h3_gpu *gpu, const char *name) {
    h3_hip_set_error(gpu, "HIP backend: %s is not implemented yet", name);
    return 0;
}
