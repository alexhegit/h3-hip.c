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
    return 1;
}

int h3_gpu_has_int8_mlp(const h3_gpu *gpu) {
    (void)gpu;
    return 1;
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

int h3_gpu_tensor_read_i8(const h3_gpu_tensor *tensor, int8_t *values,
                          size_t elements) {
    struct h3_gpu_tensor *obj = tensor_ptr(tensor);
    if (!obj || obj->dtype != H3_GPU_I8 || !values ||
        elements > obj->elements) {
        return 0;
    }
    memcpy(values, obj->host, elements * sizeof(int8_t));
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

static int h3_hip_require_i8(struct h3_gpu *gpu, const h3_gpu_tensor *tensor,
                             size_t elements, const char *label) {
    if (!tensor || tensor_ptr(tensor)->dtype != H3_GPU_I8) {
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

static int h3_hip_require_f32_tensor(struct h3_gpu *gpu,
                                     const h3_gpu_tensor *tensor,
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

int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination,
                    size_t destination_offset, const h3_gpu_tensor *source,
                    size_t source_offset, size_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_f32(ctx, source, source_offset + elements,
                            "copy source") ||
        !h3_hip_require_f32(ctx, destination, destination_offset + elements,
                             "copy destination")) {
        return 0;
    }
    memcpy((float *)tensor_ptr(destination)->host + destination_offset,
           (const float *)tensor_ptr(source)->host + source_offset,
           elements * sizeof(float));
    ctx->stats.blit_copies++;
    return 1;
}

int h3_gpu_patch_linear_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                                    size_t output_offset,
                                    const h3_gpu_tensor *input,
                                    size_t input_offset,
                                    const h3_gpu_tensor *weight,
                                    const h3_gpu_tensor *bias, uint32_t rows,
                                    uint32_t input_dim, uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!ctx || output_dim != 5376u ||
        (input_dim != 32u && input_dim != 96u) ||
        input_offset + input_count > tensor_ptr(input)->elements ||
        output_offset + output_count > tensor_ptr(output)->elements ||
        !h3_hip_require_f32_tensor(ctx, input, input_offset + input_count,
                                   "patch input") ||
        !h3_hip_require_f32_tensor(ctx, weight, weight_count,
                                   "patch weight") ||
        !h3_hip_require_bf16(ctx, output, output_offset + output_count,
                            "patch output") ||
        (bias && !h3_hip_require_f32_tensor(ctx, bias, output_dim,
                                            "patch bias"))) {
        return 0;
    }
    const float *bias_ptr = bias ?
        (const float *)tensor_ptr(bias)->host :
        (const float *)tensor_ptr(input)->host;
    h3_linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    return h3_hip_launch_ok(ctx, h3_launch_linear_f32_tiled_bf16(
        (const float *)tensor_ptr(input)->host + input_offset,
        (const float *)tensor_ptr(weight)->host, bias_ptr,
        (uint16_t *)tensor_ptr(output)->host + output_offset, &args,
        ctx->stream), "h3_linear_f32_tiled_bf16");
}

int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!ctx || !rows || !input_dim || !output_dim ||
        !h3_hip_require_f32(ctx, input, input_count, "linear input") ||
        !h3_hip_require_f32(ctx, weight, weight_count, "linear weight") ||
        !h3_hip_require_f32(ctx, output, output_count, "linear output") ||
        (bias && !h3_hip_require_f32(ctx, bias, output_dim, "linear bias"))) {
        return 0;
    }
    const float *bias_ptr = bias ?
        (const float *)tensor_ptr(bias)->host :
        (const float *)tensor_ptr(input)->host;
    h3_linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    return h3_hip_launch_ok(ctx, h3_launch_linear_f32(
        (const float *)tensor_ptr(input)->host,
        (const float *)tensor_ptr(weight)->host, bias_ptr,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_linear_f32");
}

int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_patch_linear_bf16_offset(gpu, output, 0, input, 0, weight,
                                           bias, rows, input_dim, output_dim);
}

int h3_gpu_patch_linear_bf16_map(h3_gpu *gpu, h3_gpu_tensor *output,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *bias,
                                 const h3_gpu_tensor *row_map,
                                 uint32_t output_rows, uint32_t rows,
                                 uint32_t input_dim, uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)output_rows * output_dim;
    if (!ctx || output_dim != 5376u ||
        (input_dim != 32u && input_dim != 96u) ||
        !h3_hip_require_f32_tensor(ctx, input, input_count, "mapped patch input") ||
        !h3_hip_require_f32_tensor(ctx, weight, weight_count,
                                   "mapped patch weight") ||
        !h3_hip_require_bf16(ctx, output, output_count, "mapped patch output") ||
        !h3_hip_require_u32(ctx, row_map, rows, "mapped patch row map") ||
        (bias && !h3_hip_require_f32_tensor(ctx, bias, output_dim,
                                            "mapped patch bias"))) {
        return 0;
    }
    const float *bias_ptr = bias ?
        (const float *)tensor_ptr(bias)->host :
        (const float *)tensor_ptr(input)->host;
    h3_linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    return h3_hip_launch_ok(ctx, h3_launch_linear_f32_tiled_bf16_map(
        (const float *)tensor_ptr(input)->host,
        (const float *)tensor_ptr(weight)->host, bias_ptr,
        (uint16_t *)tensor_ptr(output)->host,
        (const uint32_t *)tensor_ptr(row_map)->host, &args, ctx->stream),
        "h3_linear_f32_tiled_bf16_map");
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

int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_f32(ctx, input, elements, "SiLU input") ||
        !h3_hip_require_f32(ctx, output, elements, "SiLU output")) {
        return 0;
    }
    return h3_hip_launch_ok(ctx, h3_launch_silu_f32(
        (const float *)tensor_ptr(input)->host,
        (float *)tensor_ptr(output)->host, elements, ctx->stream),
        "h3_silu_f32");
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

int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t fused_count = (size_t)rows * width * 2;
    size_t out_count = (size_t)rows * width;
    if (!ctx || !rows || !width ||
        !h3_hip_require_f32(ctx, fused, fused_count, "SwiGLU input") ||
        !h3_hip_require_f32(ctx, output, out_count, "SwiGLU output")) {
        return 0;
    }
    h3_swiglu_args args = {rows, width};
    return h3_hip_launch_ok(ctx, h3_launch_swiglu_f32(
        (const float *)tensor_ptr(fused)->host,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_swiglu_f32");
}

int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width ||
        !h3_hip_require_f32(ctx, residual, count, "scale-add residual") ||
        !h3_hip_require_f32(ctx, branch, count, "scale-add branch") ||
        !h3_hip_require_f32(ctx, scale, width, "scale-add scale") ||
        !h3_hip_require_f32(ctx, output, count, "scale-add output")) {
        return 0;
    }
    h3_swiglu_args args = {rows, width};
    return h3_hip_launch_ok(ctx, h3_launch_scale_add_f32(
        (const float *)tensor_ptr(residual)->host,
        (const float *)tensor_ptr(branch)->host,
        (const float *)tensor_ptr(scale)->host,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_scale_add_f32");
}

int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_f32(ctx, left, elements, "add-scaled left") ||
        !h3_hip_require_f32(ctx, right, elements, "add-scaled right") ||
        !h3_hip_require_f32(ctx, output, elements, "add-scaled output")) {
        return 0;
    }
    h3_add_scaled_f32_args args = {elements, left_scale, right_scale};
    return h3_hip_launch_ok(ctx, h3_launch_add_scaled_f32(
        (const float *)tensor_ptr(left)->host,
        (const float *)tensor_ptr(right)->host,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_add_scaled_f32");
}

int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_f32(ctx, input, elements, "clip input") ||
        !h3_hip_require_f32(ctx, output, elements, "clip output")) {
        return 0;
    }
    h3_clip_f32_args args = {elements, minimum, maximum};
    return h3_hip_launch_ok(ctx, h3_launch_clip_f32(
        (const float *)tensor_ptr(input)->host,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_clip_f32");
}

int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)outer * inner;
    if (!ctx || !outer || !inner ||
        !h3_hip_require_f32(ctx, vector, count, "weight norm vector") ||
        !h3_hip_require_f32(ctx, magnitude, outer, "weight norm magnitude") ||
        !h3_hip_require_f32(ctx, output, count, "weight norm output")) {
        return 0;
    }
    h3_weight_norm_args args = {outer, inner};
    return h3_hip_launch_ok(ctx, h3_launch_weight_norm_f32(
        (const float *)tensor_ptr(vector)->host,
        (const float *)tensor_ptr(magnitude)->host,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_weight_norm_f32");
}

int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate,
                     const h3_gpu_tensor *linear, uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_f32(ctx, gate, elements, "GeGLU gate") ||
        !h3_hip_require_f32(ctx, linear, elements, "GeGLU linear") ||
        !h3_hip_require_f32(ctx, output, elements, "GeGLU output")) {
        return 0;
    }
    return h3_hip_launch_ok(ctx, h3_launch_geglu_f32(
        (const float *)tensor_ptr(gate)->host,
        (const float *)tensor_ptr(linear)->host,
        (float *)tensor_ptr(output)->host, elements, ctx->stream),
        "h3_geglu_f32");
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

int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width ||
        !h3_hip_require_f32(ctx, input, count, "RMSNorm input") ||
        !h3_hip_require_f32(ctx, weight, width, "RMSNorm weight") ||
        !h3_hip_require_f32(ctx, output, count, "RMSNorm output")) {
        return 0;
    }
    h3_norm_args args = {rows, width, epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_rms_norm_f32(
        (const float *)tensor_ptr(input)->host,
        (const float *)tensor_ptr(weight)->host,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_rms_norm_f32");
}

int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width ||
        !h3_hip_require_f32(ctx, input, count, "LayerNorm input") ||
        !h3_hip_require_f32(ctx, weight, width, "LayerNorm weight") ||
        !h3_hip_require_f32(ctx, bias, width, "LayerNorm bias") ||
        !h3_hip_require_f32(ctx, output, count, "LayerNorm output")) {
        return 0;
    }
    h3_norm_args args = {rows, width, epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_layer_norm_f32(
        (const float *)tensor_ptr(input)->host,
        (const float *)tensor_ptr(weight)->host,
        (const float *)tensor_ptr(bias)->host,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_layer_norm_f32");
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

int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    if (!ctx || !rows || !width || gate_slot >= slots ||
        !h3_hip_require_f32(ctx, residual, count, "gate residual") ||
        !h3_hip_require_f32(ctx, branch, count, "gate branch") ||
        !h3_hip_require_f32(ctx, modulation, 1, "gate modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "gate row map") ||
        !h3_hip_require_f32(ctx, output, count, "gate output")) {
        return 0;
    }
    h3_gate_args args = {rows, width, slots, gate_slot};
    return h3_hip_launch_ok(ctx, h3_launch_gate_f32(
        (const float *)tensor_ptr(residual)->host,
        (const float *)tensor_ptr(branch)->host,
        (const float *)tensor_ptr(modulation)->host,
        (const uint32_t *)tensor_ptr(row_map)->host,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_gate_f32");
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

int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output,
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
        !h3_hip_require_f32(ctx, input, count, "AdaLN input") ||
        !h3_hip_require_f32(ctx, norm_weight, width, "AdaLN norm") ||
        !h3_hip_require_f32(ctx, modulation, 1, "AdaLN modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "AdaLN row map") ||
        !h3_hip_require_f32(ctx, output, count, "AdaLN output")) {
        return 0;
    }
    h3_adaln_args args = {rows, width, slots, shift_slot, scale_slot,
                          epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_adaln_f32(
        (const float *)tensor_ptr(input)->host,
        (const float *)tensor_ptr(norm_weight)->host,
        (const float *)tensor_ptr(modulation)->host,
        (const uint32_t *)tensor_ptr(row_map)->host,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_adaln_f32");
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

static int h3_gpu_qkv_rope_f32_layout(h3_gpu *gpu, h3_gpu_tensor *query,
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
        !h3_hip_require_f32(ctx, qkv, projected * 3, "QKV input") ||
        !h3_hip_require_f32(ctx, q_norm, head_dim, "Q norm") ||
        !h3_hip_require_f32(ctx, k_norm, head_dim, "K norm") ||
        !h3_hip_require_f32(ctx, rope_cos, rope_count, "RoPE cosine") ||
        !h3_hip_require_f32(ctx, rope_sin, rope_count, "RoPE sine") ||
        !h3_hip_require_f32(ctx, query, projected, "query output") ||
        !h3_hip_require_f32(ctx, key, projected, "key output") ||
        !h3_hip_require_f32(ctx, value, projected, "value output")) {
        return 0;
    }
    h3_qkv_args args = {sequence, heads, head_dim, rope_half, grouped,
                        epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_qkv_rope_f32(
        (const float *)tensor_ptr(qkv)->host,
        (const float *)tensor_ptr(q_norm)->host,
        (const float *)tensor_ptr(k_norm)->host,
        (const float *)tensor_ptr(rope_cos)->host,
        (const float *)tensor_ptr(rope_sin)->host,
        (float *)tensor_ptr(query)->host,
        (float *)tensor_ptr(key)->host,
        (float *)tensor_ptr(value)->host, &args, ctx->stream),
        "h3_qkv_rope_f32");
}

int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon) {
    return h3_gpu_qkv_rope_f32_layout(gpu, query, key, value, qkv, q_norm,
                                      k_norm, rope_cos, rope_sin, sequence,
                                      heads, head_dim, rope_half, 0, epsilon);
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

int h3_gpu_vision_qkv_rope_bf16(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
    h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
    const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
    uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!ctx || !sequence || !heads || !head_dim || !rope_half ||
        rope_half * 2 != head_dim ||
        !h3_hip_require_bf16(ctx, qkv, count * 3, "vision QKV") ||
        !h3_hip_require_bf16(ctx, rope_cos, rope_count, "vision RoPE cos") ||
        !h3_hip_require_bf16(ctx, rope_sin, rope_count, "vision RoPE sin") ||
        !h3_hip_require_bf16(ctx, query, count, "vision query") ||
        !h3_hip_require_bf16(ctx, key, count, "vision key") ||
        !h3_hip_require_bf16(ctx, value, count, "vision value")) {
        return 0;
    }
    h3_qkv_args args = {sequence, heads, head_dim, rope_half, 0, 0.0f};
    return h3_hip_launch_ok(ctx, h3_launch_vision_qkv_rope_bf16(
        (const uint16_t *)tensor_ptr(qkv)->host,
        (const uint16_t *)tensor_ptr(rope_cos)->host,
        (const uint16_t *)tensor_ptr(rope_sin)->host,
        (uint16_t *)tensor_ptr(query)->host,
        (uint16_t *)tensor_ptr(key)->host,
        (uint16_t *)tensor_ptr(value)->host, &args, ctx->stream),
        "h3_vision_qkv_rope_bf16");
}

int h3_gpu_video_qkv_rope_f32(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
    h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
    const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
    uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half,
    float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!ctx || !sequence || !heads || !head_dim || !rope_half ||
        rope_half * 2 > head_dim ||
        !h3_hip_require_f32(ctx, qkv, count * 3, "video QKV") ||
        !h3_hip_require_f32(ctx, rope_cos, rope_count, "video RoPE cos") ||
        !h3_hip_require_f32(ctx, rope_sin, rope_count, "video RoPE sin") ||
        !h3_hip_require_f32(ctx, query, count, "video query") ||
        !h3_hip_require_f32(ctx, key, count, "video key") ||
        !h3_hip_require_f32(ctx, value, count, "video value")) {
        return 0;
    }
    h3_qkv_args args = {sequence, heads, head_dim, rope_half, 0, epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_video_qkv_rope_f32(
        (const float *)tensor_ptr(qkv)->host,
        (const float *)tensor_ptr(rope_cos)->host,
        (const float *)tensor_ptr(rope_sin)->host,
        (float *)tensor_ptr(query)->host,
        (float *)tensor_ptr(key)->host,
        (float *)tensor_ptr(value)->host, &args, ctx->stream),
        "h3_video_qkv_rope_f32");
}

int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation) {
    return h3_gpu_conv1d_stride_f32(gpu, output, input, weight, bias, batch,
                                    length, input_channels, output_channels,
                                    kernel, 1, padding, dilation);
}

static uint32_t h3_hip_conv1d_output_length(uint32_t length, uint32_t kernel,
                                            uint32_t stride, uint32_t padding,
                                            uint32_t dilation) {
    uint64_t effective = (uint64_t)dilation * (kernel - 1) + 1;
    if ((uint64_t)length + 2 * padding < effective) return 0;
    return (uint32_t)(((uint64_t)length + 2 * padding - effective) / stride +
                      1);
}

int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t batch,
                             uint32_t length, uint32_t input_channels,
                             uint32_t output_channels, uint32_t kernel,
                             uint32_t stride, uint32_t padding,
                             uint32_t dilation) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    uint32_t output_length = h3_hip_conv1d_output_length(
        length, kernel, stride, padding, dilation);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (!ctx || !batch || !length || !input_channels || !output_channels ||
        !kernel || !stride || !dilation || !output_length ||
        !h3_hip_require_f32(ctx, input, input_count, "Conv1d input") ||
        !h3_hip_require_f32(ctx, weight, weight_count, "Conv1d weight") ||
        !h3_hip_require_f32(ctx, output, output_count, "Conv1d output") ||
        (bias && !h3_hip_require_f32(ctx, bias, output_channels,
                                      "Conv1d bias"))) {
        return 0;
    }
    const float *bias_ptr = bias ?
        (const float *)tensor_ptr(bias)->host :
        (const float *)tensor_ptr(input)->host;
    h3_conv1d_args args = {batch, length, output_length, input_channels,
                           output_channels, kernel, stride, padding, dilation,
                           bias ? 1u : 0u};
    return h3_hip_launch_ok(ctx, h3_launch_conv1d_f32(
        (const float *)tensor_ptr(input)->host,
        (const float *)tensor_ptr(weight)->host, bias_ptr,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_conv1d_stride_f32");
}

int h3_gpu_conv_transpose1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                                const h3_gpu_tensor *input,
                                const h3_gpu_tensor *weight,
                                const h3_gpu_tensor *bias, uint32_t batch,
                                uint32_t length, uint32_t input_channels,
                                uint32_t output_channels, uint32_t kernel,
                                uint32_t stride, uint32_t padding) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!batch || !length || !input_channels || !output_channels || !kernel ||
        !stride || (uint64_t)(length - 1) * stride + kernel < 2 * padding) {
        return 0;
    }
    uint32_t output_length = (uint32_t)((uint64_t)(length - 1) * stride +
                                        kernel - 2 * padding);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)input_channels * output_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (!ctx || !output_length ||
        !h3_hip_require_f32(ctx, input, input_count, "ConvTranspose1d input") ||
        !h3_hip_require_f32(ctx, weight, weight_count,
                            "ConvTranspose1d weight") ||
        !h3_hip_require_f32(ctx, output, output_count,
                            "ConvTranspose1d output") ||
        (bias && !h3_hip_require_f32(ctx, bias, output_channels,
                                    "ConvTranspose1d bias"))) {
        return 0;
    }
    const float *bias_ptr = bias ?
        (const float *)tensor_ptr(bias)->host :
        (const float *)tensor_ptr(input)->host;
    h3_conv1d_args args = {batch, length, output_length, input_channels,
                           output_channels, kernel, stride, padding, 1u,
                           bias ? 1u : 0u};
    return h3_hip_launch_ok(ctx, h3_launch_conv_transpose1d_f32(
        (const float *)tensor_ptr(input)->host,
        (const float *)tensor_ptr(weight)->host, bias_ptr,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_conv_transpose1d_f32");
}

int h3_gpu_alias_free_snake_f32(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *alpha_log, const h3_gpu_tensor *beta_log,
    const h3_gpu_tensor *upsample_filter,
    const h3_gpu_tensor *downsample_filter, uint32_t batch, uint32_t length,
    uint32_t channels) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)batch * length * channels;
    if (!ctx || !batch || !length || !channels ||
        !h3_hip_require_f32(ctx, input, count, "alias-free snake input") ||
        !h3_hip_require_f32(ctx, output, count, "alias-free snake output") ||
        !h3_hip_require_f32(ctx, alpha_log, channels, "alias-free snake alpha") ||
        !h3_hip_require_f32(ctx, beta_log, channels, "alias-free snake beta") ||
        !h3_hip_require_f32(ctx, upsample_filter, 12,
                            "alias-free snake upsample filter") ||
        !h3_hip_require_f32(ctx, downsample_filter, 12,
                            "alias-free snake downsample filter")) {
        return 0;
    }
    h3_audio_activation_args args = {batch, length, channels};
    return h3_hip_launch_ok(ctx, h3_launch_alias_free_snake_f32(
        (const float *)tensor_ptr(input)->host,
        (const float *)tensor_ptr(alpha_log)->host,
        (const float *)tensor_ptr(beta_log)->host,
        (const float *)tensor_ptr(upsample_filter)->host,
        (const float *)tensor_ptr(downsample_filter)->host,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_alias_free_snake_f32");
}

int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)batch * length * channels;
    if (!ctx || !batch || !length || !channels || count > UINT32_MAX ||
        !h3_hip_require_f32(ctx, input, count, "snake1d input") ||
        !h3_hip_require_f32(ctx, alpha, channels, "snake1d alpha") ||
        !h3_hip_require_f32(ctx, output, count, "snake1d output")) {
        return 0;
    }
    h3_audio_activation_args args = {batch, length, channels};
    return h3_hip_launch_ok(ctx, h3_launch_snake1d_f32(
        (const float *)tensor_ptr(input)->host,
        (const float *)tensor_ptr(alpha)->host,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_snake1d_f32");
}

int h3_gpu_audio_qkv_split_f32(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
    h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
    const h3_gpu_tensor *q_bias, const h3_gpu_tensor *k_bias,
    const h3_gpu_tensor *v_bias, uint32_t batch, uint32_t length,
    uint32_t heads, uint32_t head_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t width = (size_t)heads * head_dim;
    size_t count = (size_t)batch * length * width;
    if (!ctx || !batch || !length || !heads || !head_dim ||
        count > UINT32_MAX ||
        !h3_hip_require_f32(ctx, qkv, count * 3, "audio QKV") ||
        !h3_hip_require_f32(ctx, q_bias, width, "audio Q bias") ||
        !h3_hip_require_f32(ctx, k_bias, width, "audio K bias") ||
        !h3_hip_require_f32(ctx, v_bias, width, "audio V bias") ||
        !h3_hip_require_f32(ctx, query, count, "audio query") ||
        !h3_hip_require_f32(ctx, key, count, "audio key") ||
        !h3_hip_require_f32(ctx, value, count, "audio value")) {
        return 0;
    }
    h3_audio_qkv_args args = {batch, length, heads, head_dim};
    return h3_hip_launch_ok(ctx, h3_launch_audio_qkv_split_f32(
        (const float *)tensor_ptr(qkv)->host,
        (const float *)tensor_ptr(q_bias)->host,
        (const float *)tensor_ptr(k_bias)->host,
        (const float *)tensor_ptr(v_bias)->host,
        (float *)tensor_ptr(query)->host,
        (float *)tensor_ptr(key)->host,
        (float *)tensor_ptr(value)->host, &args, ctx->stream),
        "h3_audio_qkv_split_f32");
}

int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t batch,
                           uint32_t sequence, uint32_t heads,
                           uint32_t head_dim, float scale) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)batch * sequence * heads * head_dim;
    if (!ctx || !batch || !sequence || !heads || !head_dim ||
        !h3_hip_require_f32(ctx, query, count, "causal SDPA query") ||
        !h3_hip_require_f32(ctx, key, count, "causal SDPA key") ||
        !h3_hip_require_f32(ctx, value, count, "causal SDPA value") ||
        !h3_hip_require_f32(ctx, output, count, "causal SDPA output")) {
        return 0;
    }
    h3_sdpa_causal_args args = {batch, sequence, heads, head_dim, scale};
    return h3_hip_launch_ok(ctx, h3_launch_sdpa_causal_f32(
        (const float *)tensor_ptr(query)->host,
        (const float *)tensor_ptr(key)->host,
        (const float *)tensor_ptr(value)->host,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_sdpa_causal_f32");
}

int h3_gpu_audio_attention_pool_f32(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *attended,
    uint32_t batch, uint32_t length, uint32_t heads, uint32_t head_dim,
    uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t input_count = (size_t)batch * length * heads * head_dim;
    size_t output_count = (size_t)batch * length * output_dim;
    if (!ctx || !batch || !length || !heads || !head_dim || !output_dim ||
        head_dim % output_dim || output_count > UINT32_MAX ||
        !h3_hip_require_f32(ctx, attended, input_count, "audio attended") ||
        !h3_hip_require_f32(ctx, output, output_count, "audio pooled")) {
        return 0;
    }
    h3_audio_pool_args args = {batch, length, heads, head_dim, output_dim};
    return h3_hip_launch_ok(ctx, h3_launch_audio_attention_pool_f32(
        (const float *)tensor_ptr(attended)->host,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_audio_attention_pool_f32");
}

int h3_gpu_vae_encoder_pad_f32(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
    uint32_t channels, uint32_t depth_front, uint32_t height_before,
    uint32_t height_after, uint32_t width_before, uint32_t width_after) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !batch || !depth || height < 2 || width < 2 || !channels ||
        height_before >= height || height_after >= height ||
        width_before >= width || width_after >= width) {
        return 0;
    }
    uint32_t output_depth = depth + depth_front;
    uint32_t output_height = height + height_before + height_after;
    uint32_t output_width = width + width_before + width_after;
    size_t input_count = (size_t)batch * depth * height * width * channels;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * channels;
    if (!h3_hip_require_f32(ctx, input, input_count, "VAE encoder pad input") ||
        !h3_hip_require_f32(ctx, output, output_count,
                            "VAE encoder pad output")) {
        return 0;
    }
    h3_vae_encoder_pad_args args = {
        batch, depth, height, width, channels, depth_front,
        height_before, height_after, width_before, width_after
    };
    return h3_hip_launch_ok(ctx, h3_launch_vae_encoder_pad_f32(
        (const float *)tensor_ptr(input)->host,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_vae_encoder_pad_f32");
}

int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch, uint32_t depth,
                      uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !batch || !depth || !height || !width || !input_channels ||
        !output_channels || !kernel_depth || !kernel_height || !kernel_width ||
        !stride_depth || !stride_height || !stride_width ||
        depth < kernel_depth || height < kernel_height ||
        width < kernel_width) {
        return 0;
    }
    uint32_t output_depth = (depth - kernel_depth) / stride_depth + 1;
    uint32_t output_height = (height - kernel_height) / stride_height + 1;
    uint32_t output_width = (width - kernel_width) / stride_width + 1;
    size_t input_count =
        (size_t)batch * depth * height * width * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels *
                          kernel_depth * kernel_height * kernel_width;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * output_channels;
    if (!h3_hip_require_f32(ctx, input, input_count, "Conv3d input") ||
        !h3_hip_require_f32(ctx, weight, weight_count, "Conv3d weight") ||
        !h3_hip_require_f32(ctx, output, output_count, "Conv3d output") ||
        (bias && !h3_hip_require_f32(ctx, bias, output_channels,
                                    "Conv3d bias"))) {
        return 0;
    }
    const float *bias_ptr = bias ?
        (const float *)tensor_ptr(bias)->host :
        (const float *)tensor_ptr(input)->host;
    h3_conv3d_args args = {batch, depth, height, width, output_depth,
                          output_height, output_width, input_channels,
                          output_channels, kernel_depth, kernel_height,
                          kernel_width, stride_depth, stride_height,
                          stride_width, bias ? 1u : 0u};
    return h3_hip_launch_ok(ctx, h3_launch_conv3d_f32(
        (const float *)tensor_ptr(input)->host,
        (const float *)tensor_ptr(weight)->host, bias_ptr,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_conv3d_f32");
}

int h3_gpu_vae_encoder_group_norm_silu_f32(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch,
    uint32_t depth, uint32_t height, uint32_t width, uint32_t channels,
    uint32_t groups, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)batch * depth * height * width * channels;
    if (!ctx || !batch || !depth || !height || !width || !channels ||
        !groups || channels % groups || !(epsilon > 0.0f) ||
        !h3_hip_require_f32(ctx, input, count, "VAE encoder norm input") ||
        !h3_hip_require_f32(ctx, weight, channels, "VAE encoder norm weight") ||
        !h3_hip_require_f32(ctx, bias, channels, "VAE encoder norm bias") ||
        !h3_hip_require_f32(ctx, output, count, "VAE encoder norm output")) {
        return 0;
    }
    uint64_t rows = (uint64_t)batch * depth * groups;
    if (rows > UINT32_MAX) return 0;
    h3_vae_encoder_norm_args args = {batch, depth, height, width, channels,
                                     groups, epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_vae_encoder_group_norm_silu_f32(
        (const float *)tensor_ptr(input)->host,
        (const float *)tensor_ptr(weight)->host,
        (const float *)tensor_ptr(bias)->host,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_vae_encoder_group_norm_silu_f32");
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

int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)sequence * heads * head_dim;
    if (!ctx || !sequence || !heads || !head_dim ||
        !h3_hip_require_f32(ctx, query, count, "SDPA query") ||
        !h3_hip_require_f32(ctx, key, count, "SDPA key") ||
        !h3_hip_require_f32(ctx, value, count, "SDPA value") ||
        !h3_hip_require_f32(ctx, output, count, "SDPA output")) {
        return 0;
    }
    h3_sdpa_args args = {sequence, heads, head_dim, scale};
    return h3_hip_launch_ok(ctx, h3_launch_sdpa_f32(
        (const float *)tensor_ptr(query)->host,
        (const float *)tensor_ptr(key)->host,
        (const float *)tensor_ptr(value)->host,
        (float *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_sdpa_f32");
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

static int h3_hip_fc1_swiglu_nax_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, uint32_t rows, uint32_t input_dim,
    uint32_t hidden_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !rows || !input_dim || !hidden_dim ||
        !h3_hip_require_bf16(ctx, input, (size_t)rows * input_dim,
                             "NAX FC1 input") ||
        !h3_hip_require_bf16(ctx, weight,
                             (size_t)hidden_dim * 2 * input_dim,
                             "NAX FC1 weight") ||
        !h3_hip_require_bf16(ctx, output, (size_t)rows * hidden_dim,
                             "NAX FC1 output")) {
        return 0;
    }
    h3_gpu_tensor *fc1_out = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * hidden_dim * 2);
    if (!fc1_out) {
        h3_hip_set_error(ctx, "NAX FC1 temporary allocation failed");
        return 0;
    }
    int ok = h3_gpu_linear_bf16(gpu, fc1_out, input, weight, NULL, rows,
                                input_dim, hidden_dim * 2) &&
             h3_gpu_swiglu_bf16(gpu, output, fc1_out, rows, hidden_dim);
    h3_gpu_tensor_free(fc1_out);
    if (!ok) {
        h3_hip_set_error(ctx, "h3_fc1_swiglu_nax_bf16 failed");
    }
    return ok;
}

int h3_hip_fc1_swiglu_nax_bf16_dispatch(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, uint32_t rows, uint32_t input_dim,
    uint32_t hidden_dim) {
    return h3_hip_fc1_swiglu_nax_bf16(gpu, output, input, weight, rows,
                                      input_dim, hidden_dim);
}

static int h3_hip_linear_bf16_nax(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, uint32_t rows, uint32_t input_dim,
    uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !rows || !input_dim || !output_dim ||
        !h3_hip_require_bf16(ctx, input, (size_t)rows * input_dim,
                             "NAX linear input") ||
        !h3_hip_require_bf16(ctx, weight, (size_t)output_dim * input_dim,
                             "NAX linear weight") ||
        !h3_hip_require_bf16(ctx, output, (size_t)rows * output_dim,
                             "NAX linear output")) {
        return 0;
    }
    if (!h3_gpu_linear_bf16(gpu, output, input, weight, NULL, rows, input_dim,
                            output_dim)) {
        h3_hip_set_error(ctx, "h3_linear_bf16_nax failed");
        return 0;
    }
    return 1;
}

int h3_hip_linear_bf16_nax_dispatch(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, uint32_t rows, uint32_t input_dim,
    uint32_t output_dim) {
    return h3_hip_linear_bf16_nax(gpu, output, input, weight, rows, input_dim,
                                  output_dim);
}

int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * hidden_dim;
    if (!ctx || !rows || !input_dim || !hidden_dim || !output_dim ||
        !h3_hip_require_bf16(ctx, input, (size_t)rows * input_dim,
                             "NAX MLP input") ||
        !h3_hip_require_bf16(ctx, fc1_weight,
                             (size_t)hidden_dim * 2 * input_dim,
                             "NAX MLP FC1 weight") ||
        !h3_hip_require_bf16(ctx, fc2_weight,
                             (size_t)output_dim * hidden_dim,
                             "NAX MLP FC2 weight") ||
        !h3_hip_require_bf16(ctx, activated, count, "NAX MLP activated") ||
        !h3_hip_require_bf16(ctx, output, (size_t)rows * output_dim,
                             "NAX MLP output")) {
        return 0;
    }
    int ok = h3_hip_fc1_swiglu_nax_bf16(gpu, activated, input, fc1_weight, rows,
                                        input_dim, hidden_dim) &&
             h3_hip_linear_bf16_nax(gpu, output, activated, fc2_weight, rows,
                                    hidden_dim, output_dim);
    if (!ok) {
        h3_hip_set_error(ctx, "h3_mlp_nax_bf16 failed");
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

int h3_gpu_adaln_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             h3_gpu_tensor *inverse,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *norm_weight,
                             const h3_gpu_tensor *modulation,
                             const h3_gpu_tensor *row_map,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t width, uint32_t output_dim,
                             uint32_t slots, uint32_t shift_slot,
                             uint32_t scale_slot, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t input_count = (size_t)rows * width;
    size_t weight_count = (size_t)output_dim * width;
    size_t output_count = (size_t)rows * output_dim;
    if (!ctx || !rows || !width || !output_dim || shift_slot >= slots ||
        scale_slot >= slots ||
        input_offset + input_count > tensor_ptr(input)->elements ||
        !h3_hip_require_bf16(ctx, norm_weight, width, "adaln linear norm") ||
        !h3_hip_require_bf16(ctx, modulation, 1, "adaln linear modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "adaln linear row map") ||
        !h3_hip_require_bf16(ctx, weight, weight_count, "adaln linear weight") ||
        !h3_hip_require_bf16(ctx, output, output_count, "adaln linear output") ||
        !h3_hip_require_f32(ctx, inverse, rows, "adaln linear inverse") ||
        (bias && !h3_hip_require_bf16(ctx, bias, output_dim,
                                      "adaln linear bias"))) {
        return 0;
    }
    const h3_gpu_tensor *bias_tensor = bias ? bias : input;
    h3_norm_args rms_args = {rows, width, epsilon};
    const uint16_t *input_ptr =
        (const uint16_t *)tensor_ptr(input)->host + input_offset;
    if (!h3_hip_launch_ok(ctx, h3_launch_rms_inverse_bf16(
            input_ptr, (float *)tensor_ptr(inverse)->host, &rms_args,
            ctx->stream), "h3_rms_inverse_bf16")) {
        return 0;
    }
    h3_adaln_linear_args args = {rows, width, output_dim, slots, shift_slot,
                                 scale_slot, bias ? 1u : 0u};
    return h3_hip_launch_ok(ctx, h3_launch_adaln_linear_bf16(
        input_ptr, (const float *)tensor_ptr(inverse)->host,
        (const uint16_t *)tensor_ptr(norm_weight)->host,
        (const uint16_t *)tensor_ptr(modulation)->host,
        (const uint32_t *)tensor_ptr(row_map)->host,
        (const uint16_t *)tensor_ptr(weight)->host,
        (const uint16_t *)tensor_ptr(bias_tensor)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_adaln_linear_bf16");
}

int h3_gpu_grouped_qkv_linear_rope_bf16(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
    h3_gpu_tensor *value, h3_gpu_tensor *qkv, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *q_norm,
    const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos,
    const h3_gpu_tensor *rope_sin, uint32_t rows, uint32_t input_dim,
    uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon) {
    uint32_t inner = heads * head_dim;
    return h3_gpu_linear_bf16(gpu, qkv, input, weight, NULL, rows, input_dim,
                                inner * 3) &&
           h3_gpu_grouped_qkv_rope_bf16(gpu, query, key, value, qkv, q_norm,
                                        k_norm, rope_cos, rope_sin, rows,
                                        heads, head_dim, rope_half, epsilon);
}

int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !tokens || !width || !vocab_size ||
        !h3_hip_require_bf16(ctx, weight, (size_t)vocab_size * width,
                             "embedding weight") ||
        !h3_hip_require_u32(ctx, token_ids, tokens, "embedding token ids") ||
        !h3_hip_require_bf16(ctx, output, (size_t)tokens * width,
                             "embedding output")) {
        return 0;
    }
    h3_embedding_args args = {tokens, vocab_size, width};
    return h3_hip_launch_ok(ctx, h3_launch_embedding_bf16(
        (const uint16_t *)tensor_ptr(weight)->host,
        (const uint32_t *)tensor_ptr(token_ids)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_embedding_bf16");
}

int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate, const h3_gpu_tensor *up,
                         uint32_t elements) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    if (!ctx || !elements ||
        !h3_hip_require_bf16(ctx, gate, elements, "SiLU mul gate") ||
        !h3_hip_require_bf16(ctx, up, elements, "SiLU mul up") ||
        !h3_hip_require_bf16(ctx, output, elements, "SiLU mul output")) {
        return 0;
    }
    return h3_hip_launch_ok(ctx, h3_launch_silu_mul_bf16(
        (const uint16_t *)tensor_ptr(gate)->host,
        (const uint16_t *)tensor_ptr(up)->host,
        (uint16_t *)tensor_ptr(output)->host, elements, ctx->stream),
        "h3_silu_mul_bf16");
}

int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t elements = (size_t)rows * width;
    if (!ctx || !input_rows || !rows || rows > input_rows || !width ||
        !h3_hip_require_bf16(ctx, output, elements, "token pool output") ||
        !h3_hip_require_bf16(ctx, input, input_offset + (size_t)input_rows * width,
                             "token pool input") ||
        !h3_hip_require_bf16(ctx, original,
                             original_offset + (size_t)input_rows * width,
                             "token pool original") ||
        !h3_hip_require_bf16(ctx, baseline,
                             baseline_offset + (size_t)baseline_rows * width,
                             "token pool baseline") ||
        !h3_hip_require_u32(ctx, baseline_indices, rows,
                            "token pool baseline indices") ||
        !h3_hip_require_u32(ctx, pairs, (size_t)rows * 2, "token pool pairs")) {
        return 0;
    }
    h3_token_pool_args args = {
        (uint32_t)input_offset, (uint32_t)original_offset,
        (uint32_t)baseline_offset, rows, width
    };
    return h3_hip_launch_ok(ctx, h3_launch_token_pool_bf16(
        (const uint16_t *)tensor_ptr(input)->host,
        (const uint32_t *)tensor_ptr(pairs)->host,
        (uint16_t *)tensor_ptr(output)->host,
        (uint16_t *)tensor_ptr(baseline)->host,
        (const uint32_t *)tensor_ptr(baseline_indices)->host,
        (uint16_t *)tensor_ptr(original)->host, &args, ctx->stream),
        "h3_token_pool_bf16");
}

int h3_gpu_token_pool_adaln_bf16(
    h3_gpu *gpu, h3_gpu_tensor *residual, h3_gpu_tensor *output,
    const h3_gpu_tensor *input, size_t input_offset, h3_gpu_tensor *original,
    size_t original_offset, h3_gpu_tensor *baseline, size_t baseline_offset,
    const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *pairs,
    const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation,
    const h3_gpu_tensor *row_map, uint32_t input_rows, uint32_t rows,
    uint32_t baseline_rows, uint32_t width, uint32_t slots,
    uint32_t shift_slot, uint32_t scale_slot, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t elements = (size_t)rows * width;
    if (!ctx || !rows || !width || width > 5376u || shift_slot >= slots ||
        scale_slot >= slots ||
        !h3_hip_require_bf16(ctx, residual, elements, "token pool residual") ||
        !h3_hip_require_bf16(ctx, output, elements, "token pool output") ||
        !h3_hip_require_bf16(ctx, input,
                             input_offset + (size_t)input_rows * width,
                             "token pool input") ||
        !h3_hip_require_bf16(ctx, original,
                             original_offset + (size_t)input_rows * width,
                             "token pool original") ||
        !h3_hip_require_bf16(ctx, baseline,
                             baseline_offset + (size_t)baseline_rows * width,
                             "token pool baseline") ||
        !h3_hip_require_u32(ctx, baseline_indices, rows,
                            "token pool baseline indices") ||
        !h3_hip_require_u32(ctx, pairs, (size_t)rows * 2, "token pool pairs") ||
        !h3_hip_require_bf16(ctx, norm_weight, width, "token pool norm") ||
        !h3_hip_require_bf16(ctx, modulation, 1, "token pool modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "token pool row map")) {
        return 0;
    }
    h3_token_pool_adaln_args args = {
        (uint32_t)input_offset, (uint32_t)original_offset,
        (uint32_t)baseline_offset, rows, width, slots, shift_slot,
        scale_slot, epsilon
    };
    return h3_hip_launch_ok(ctx, h3_launch_token_pool_adaln_bf16(
        (const uint16_t *)tensor_ptr(input)->host,
        (const uint32_t *)tensor_ptr(pairs)->host,
        (uint16_t *)tensor_ptr(residual)->host,
        (uint16_t *)tensor_ptr(baseline)->host,
        (const uint32_t *)tensor_ptr(baseline_indices)->host,
        (uint16_t *)tensor_ptr(original)->host,
        (const uint16_t *)tensor_ptr(norm_weight)->host,
        (const uint16_t *)tensor_ptr(modulation)->host,
        (const uint32_t *)tensor_ptr(row_map)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_token_pool_adaln_bf16");
}

int h3_gpu_token_expand_delta_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *original,
    size_t original_offset, const h3_gpu_tensor *reduced,
    const h3_gpu_tensor *baseline, size_t baseline_offset,
    const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *parents,
    uint32_t rows, uint32_t reduced_rows, uint32_t baseline_rows,
    uint32_t width, uint32_t exact_prefix_rows, float update_scale) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t elements = (size_t)rows * width;
    if (!ctx || !rows || !width ||
        !h3_hip_require_bf16(ctx, output, elements, "token expand output") ||
        !h3_hip_require_bf16(ctx, original, original_offset + elements,
                             "token expand original") ||
        !h3_hip_require_bf16(ctx, reduced, (size_t)reduced_rows * width,
                             "token expand reduced") ||
        !h3_hip_require_bf16(ctx, baseline,
                             baseline_offset + (size_t)baseline_rows * width,
                             "token expand baseline") ||
        !h3_hip_require_u32(ctx, baseline_indices, reduced_rows,
                            "token expand baseline indices") ||
        !h3_hip_require_u32(ctx, parents, rows, "token expand parents")) {
        return 0;
    }
    h3_token_expand_args args = {
        (uint32_t)original_offset, (uint32_t)baseline_offset,
        rows, width, exact_prefix_rows, update_scale
    };
    return h3_hip_launch_ok(ctx, h3_launch_token_expand_delta_bf16(
        (const uint16_t *)tensor_ptr(original)->host,
        (const uint16_t *)tensor_ptr(reduced)->host,
        (const uint16_t *)tensor_ptr(baseline)->host,
        (const uint32_t *)tensor_ptr(baseline_indices)->host,
        (const uint32_t *)tensor_ptr(parents)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_token_expand_delta_bf16");
}

int h3_gpu_token_expand_adaln_bf16(
    h3_gpu *gpu, h3_gpu_tensor *residual, h3_gpu_tensor *output,
    const h3_gpu_tensor *original, size_t original_offset,
    const h3_gpu_tensor *reduced, const h3_gpu_tensor *baseline,
    size_t baseline_offset, const h3_gpu_tensor *baseline_indices,
    const h3_gpu_tensor *parents, const h3_gpu_tensor *norm_weight,
    const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map,
    uint32_t rows, uint32_t reduced_rows, uint32_t baseline_rows,
    uint32_t width, uint32_t exact_prefix_rows, float update_scale,
    uint32_t slots, uint32_t shift_slot, uint32_t scale_slot,
    float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t elements = (size_t)rows * width;
    if (!ctx || !rows || !width || width > 5376u || shift_slot >= slots ||
        scale_slot >= slots ||
        !h3_hip_require_bf16(ctx, residual, elements,
                             "token expand residual") ||
        !h3_hip_require_bf16(ctx, output, elements, "token expand output") ||
        !h3_hip_require_bf16(ctx, original, original_offset + elements,
                             "token expand original") ||
        !h3_hip_require_bf16(ctx, reduced, (size_t)reduced_rows * width,
                             "token expand reduced") ||
        !h3_hip_require_bf16(ctx, baseline,
                             baseline_offset + (size_t)baseline_rows * width,
                             "token expand baseline") ||
        !h3_hip_require_u32(ctx, baseline_indices, reduced_rows,
                            "token expand baseline indices") ||
        !h3_hip_require_u32(ctx, parents, rows, "token expand parents") ||
        !h3_hip_require_bf16(ctx, norm_weight, width, "token expand norm") ||
        !h3_hip_require_bf16(ctx, modulation, 1, "token expand modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "token expand row map")) {
        return 0;
    }
    h3_token_expand_adaln_args args = {
        (uint32_t)original_offset, (uint32_t)baseline_offset,
        rows, width, exact_prefix_rows, slots, shift_slot, scale_slot,
        update_scale, epsilon
    };
    return h3_hip_launch_ok(ctx, h3_launch_token_expand_adaln_bf16(
        (const uint16_t *)tensor_ptr(original)->host,
        (const uint16_t *)tensor_ptr(reduced)->host,
        (const uint16_t *)tensor_ptr(baseline)->host,
        (const uint32_t *)tensor_ptr(baseline_indices)->host,
        (const uint32_t *)tensor_ptr(parents)->host,
        (uint16_t *)tensor_ptr(residual)->host,
        (const uint16_t *)tensor_ptr(norm_weight)->host,
        (const uint16_t *)tensor_ptr(modulation)->host,
        (const uint32_t *)tensor_ptr(row_map)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_token_expand_adaln_bf16");
}

int h3_gpu_text_qk_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *q_norm,
                             const h3_gpu_tensor *k_norm,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin, uint32_t sequence,
                             uint32_t query_heads, uint32_t kv_heads,
                             uint32_t head_dim, float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2);
    if (!ctx || head_dim % 2 || !kv_heads || query_heads % kv_heads ||
        !h3_hip_require_bf16(ctx, query_input, query_count, "text query") ||
        !h3_hip_require_bf16(ctx, key_input, key_count, "text key") ||
        !h3_hip_require_bf16(ctx, q_norm, head_dim, "text Q norm") ||
        !h3_hip_require_bf16(ctx, k_norm, head_dim, "text K norm") ||
        !h3_hip_require_bf16(ctx, rope_cos, rope_count, "text RoPE cosine") ||
        !h3_hip_require_bf16(ctx, rope_sin, rope_count, "text RoPE sine") ||
        !h3_hip_require_bf16(ctx, query_output, query_count,
                             "text query output") ||
        !h3_hip_require_bf16(ctx, key_output, key_count, "text key output")) {
        return 0;
    }
    h3_text_rope_args args = {sequence, query_heads, kv_heads, head_dim,
                              epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_text_qk_rope_bf16(
        (const uint16_t *)tensor_ptr(query_input)->host,
        (const uint16_t *)tensor_ptr(key_input)->host,
        (const uint16_t *)tensor_ptr(q_norm)->host,
        (const uint16_t *)tensor_ptr(k_norm)->host,
        (const uint16_t *)tensor_ptr(rope_cos)->host,
        (const uint16_t *)tensor_ptr(rope_sin)->host,
        (uint16_t *)tensor_ptr(query_output)->host,
        (uint16_t *)tensor_ptr(key_output)->host, &args, ctx->stream),
        "h3_text_qk_rope_bf16");
}

int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight, uint32_t sequence,
                              uint32_t heads, uint32_t head_dim,
                              float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)sequence * heads * head_dim;
    if (!ctx || !sequence || !heads || !head_dim ||
        !h3_hip_require_bf16(ctx, tensor, count, "head norm tensor") ||
        !h3_hip_require_bf16(ctx, weight, head_dim, "head norm weight")) {
        return 0;
    }
    h3_head_norm_args args = {sequence, heads, head_dim, epsilon};
    return h3_hip_launch_ok(ctx, h3_launch_head_rms_norm_bf16(
        (uint16_t *)tensor_ptr(tensor)->host,
        (const uint16_t *)tensor_ptr(weight)->host, &args, ctx->stream),
        "h3_head_rms_norm_bf16");
}

int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32, uint32_t sequence,
                          uint32_t query_heads, uint32_t kv_heads,
                          uint32_t head_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2);
    if (!ctx || head_dim % 2 || !kv_heads || query_heads % kv_heads ||
        !h3_hip_require_bf16(ctx, query, query_count, "RoPE query") ||
        !h3_hip_require_bf16(ctx, key, key_count, "RoPE key") ||
        !h3_hip_require_f32(ctx, rope_cos_f32, rope_count, "RoPE cosine") ||
        !h3_hip_require_f32(ctx, rope_sin_f32, rope_count, "RoPE sine")) {
        return 0;
    }
    h3_text_rope_inplace_args args = {sequence, query_heads, kv_heads,
                                      head_dim};
    return h3_hip_launch_ok(ctx, h3_launch_rope_text_bf16(
        (uint16_t *)tensor_ptr(query)->host,
        (uint16_t *)tensor_ptr(key)->host,
        (const float *)tensor_ptr(rope_cos_f32)->host,
        (const float *)tensor_ptr(rope_sin_f32)->host, &args, ctx->stream),
        "h3_rope_text_bf16");
}

int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value, uint32_t sequence,
                           uint32_t query_heads, uint32_t kv_heads,
                           uint32_t head_dim, float scale) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t kv_count = (size_t)sequence * kv_heads * head_dim;
    if (!ctx || !sequence || !query_heads || !kv_heads || !head_dim ||
        query_heads % kv_heads || head_dim > 128u ||
        !h3_hip_require_bf16(ctx, query, query_count, "GQA query") ||
        !h3_hip_require_bf16(ctx, key, kv_count, "GQA key") ||
        !h3_hip_require_bf16(ctx, value, kv_count, "GQA value") ||
        !h3_hip_require_bf16(ctx, output, query_count, "GQA output")) {
        return 0;
    }
    h3_gqa_args args = {sequence, query_heads, kv_heads, head_dim, scale};
    return h3_hip_launch_ok(ctx, h3_launch_gqa_causal_bf16(
        (const uint16_t *)tensor_ptr(query)->host,
        (const uint16_t *)tensor_ptr(key)->host,
        (const uint16_t *)tensor_ptr(value)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_gqa_causal_bf16");
}

int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * columns;
    if (!ctx || !rows || !columns ||
        !h3_hip_require_bf16(ctx, input, count, "BF16 weight to quantize") ||
        !h3_hip_require_i8(ctx, output, count, "int8 quantized output") ||
        !h3_hip_require_f32(ctx, scales, rows, "int8 quantization scales")) {
        return 0;
    }
    h3_int8_quant_args args = {rows, columns, 1.0f};
    return h3_hip_launch_ok(ctx, h3_launch_quantize_bf16_int8_rows(
        (const uint16_t *)tensor_ptr(input)->host,
        (int8_t *)tensor_ptr(output)->host,
        (float *)tensor_ptr(scales)->host, &args, rows, ctx->stream),
        "h3_quantize_bf16_int8_rows");
}

static int h3_hip_launch_linear_int8_prequant(
    struct h3_gpu *ctx, h3_gpu_tensor *output,
    const h3_gpu_tensor *quantized_input, const h3_gpu_tensor *input_scales,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    uint32_t rows, uint32_t input_dim, uint32_t output_dim) {
    h3_linear_args args = {rows, input_dim, output_dim, 0};
    const int8_t *input_ptr =
        (const int8_t *)tensor_ptr(quantized_input)->host;
    const int8_t *weight_ptr = (const int8_t *)tensor_ptr(weight)->host;
    const float *input_scale_ptr =
        (const float *)tensor_ptr(input_scales)->host;
    const float *weight_scale_ptr =
        (const float *)tensor_ptr(weight_scales)->host;
    uint16_t *output_ptr = (uint16_t *)tensor_ptr(output)->host;
    if (!(input_dim % 32) && !getenv("H3_INT8_LEGACY")) {
        return h3_hip_launch_ok(ctx, h3_launch_linear_int8_nax_r64(
            input_ptr, weight_ptr, input_scale_ptr, weight_scale_ptr,
            output_ptr, &args, ctx->stream),
            input_dim == 14336 ? "h3_linear_int8_nax_r64_k14336" :
            input_dim == 5376 ? "h3_linear_int8_nax_r64_k5376" :
                                "h3_linear_int8_nax_r64");
    }
    if (!(input_dim % 128) && !(output_dim % 128)) {
        return h3_hip_launch_ok(ctx, h3_launch_linear_int8_nax_r128(
            input_ptr, weight_ptr, input_scale_ptr, weight_scale_ptr,
            output_ptr, &args, ctx->stream), "h3_linear_int8_nax_r128");
    }
    return h3_hip_launch_ok(ctx, h3_launch_linear_int8_bf16_naive(
        input_ptr, weight_ptr, input_scale_ptr, weight_scale_ptr, output_ptr,
        &args, ctx->stream), "h3_linear_int8_bf16_naive");
}

static int h3_hip_launch_fc1_swiglu_int8_prequant(
    struct h3_gpu *ctx, h3_gpu_tensor *output,
    const h3_gpu_tensor *quantized_input, const h3_gpu_tensor *input_scales,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    uint32_t rows, uint32_t input_dim, uint32_t hidden_dim) {
    if (!(input_dim % 32) && !getenv("H3_INT8_LEGACY")) {
        h3_linear_args args = {rows, input_dim, hidden_dim, 0};
        return h3_hip_launch_ok(ctx, h3_launch_fc1_swiglu_int8_nax_r64(
            (const int8_t *)tensor_ptr(quantized_input)->host,
            (const int8_t *)tensor_ptr(weight)->host,
            (const float *)tensor_ptr(input_scales)->host,
            (const float *)tensor_ptr(weight_scales)->host,
            (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
            input_dim == 5376 ? "h3_fc1_swiglu_int8_nax_r64_k5376" :
                                "h3_fc1_swiglu_int8_nax_r64");
    }
    if (input_dim % 128 || hidden_dim % 128) return 0;
    h3_linear_args args = {rows, input_dim, hidden_dim, 0};
    return h3_hip_launch_ok(ctx, h3_launch_fc1_swiglu_int8_nax_r128(
        (const int8_t *)tensor_ptr(quantized_input)->host,
        (const int8_t *)tensor_ptr(weight)->host,
        (const float *)tensor_ptr(input_scales)->host,
        (const float *)tensor_ptr(weight_scales)->host,
        (uint16_t *)tensor_ptr(output)->host, &args, ctx->stream),
        "h3_fc1_swiglu_int8_nax_r128");
}

static int h3_hip_launch_linear_int8_grouped_prequant(
    struct h3_gpu *ctx, h3_gpu_tensor *output,
    const h3_gpu_tensor *quantized_input, const h3_gpu_tensor *input_scales,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    uint32_t rows, uint32_t input_dim, uint32_t output_dim,
    uint32_t group_size) {
    if (!group_size || input_dim % group_size) return 0;
    h3_linear_int8_grouped_args args = {rows, input_dim, output_dim,
                                        group_size};
    const int8_t *input_ptr =
        (const int8_t *)tensor_ptr(quantized_input)->host;
    const int8_t *weight_ptr = (const int8_t *)tensor_ptr(weight)->host;
    const float *input_scale_ptr =
        (const float *)tensor_ptr(input_scales)->host;
    const float *weight_scale_ptr =
        (const float *)tensor_ptr(weight_scales)->host;
    uint16_t *output_ptr = (uint16_t *)tensor_ptr(output)->host;
    if (group_size == 1024u && !(input_dim % 128) && !(output_dim % 64)) {
        return h3_hip_launch_ok(ctx, h3_launch_linear_int8_grouped_nax_r128x64(
            input_ptr, weight_ptr, input_scale_ptr, weight_scale_ptr,
            output_ptr, &args, ctx->stream),
            "h3_linear_int8_grouped_nax_r128x64");
    }
    return h3_hip_launch_ok(ctx, h3_launch_linear_int8_grouped_naive(
        input_ptr, weight_ptr, input_scale_ptr, weight_scale_ptr, output_ptr,
        &args, ctx->stream), "h3_linear_int8_grouped_naive");
}

int h3_hip_launch_linear_int8_grouped_dispatch(
    h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *quantized_input,
    const h3_gpu_tensor *input_scales, const h3_gpu_tensor *weight,
    const h3_gpu_tensor *weight_scales, uint32_t rows, uint32_t input_dim,
    uint32_t output_dim, uint32_t group_size) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    uint32_t scale_groups = input_dim / group_size;
    if (!ctx || !group_size || !rows || !input_dim || !output_dim ||
        (input_dim % group_size) ||
        !h3_hip_require_i8(ctx, quantized_input,
                           (size_t)((rows + 127u) & ~127u) * input_dim,
                           "grouped int8 linear input") ||
        !h3_hip_require_f32(ctx, input_scales,
                            (size_t)((rows + 127u) & ~127u) * scale_groups,
                            "grouped int8 linear input scales") ||
        !h3_hip_require_i8(ctx, weight, (size_t)output_dim * input_dim,
                           "grouped int8 linear weight") ||
        !h3_hip_require_f32(ctx, weight_scales, output_dim,
                            "grouped int8 linear weight scales") ||
        !h3_hip_require_bf16(ctx, output, (size_t)rows * output_dim,
                             "grouped int8 linear output")) {
        return 0;
    }
    return h3_hip_launch_linear_int8_grouped_prequant(
        ctx, output, quantized_input, input_scales, weight, weight_scales,
        rows, input_dim, output_dim, group_size);
}

static int h3_hip_quantize_bf16_int8_rows(
    struct h3_gpu *ctx, h3_gpu_tensor *quantized,
    h3_gpu_tensor *scales, const h3_gpu_tensor *input, uint32_t rows,
    uint32_t padded_rows, uint32_t columns) {
    h3_int8_quant_args quant_args = {rows, columns, 1.0f};
    return h3_hip_launch_ok(ctx, h3_launch_quantize_bf16_int8_rows(
        (const uint16_t *)tensor_ptr(input)->host,
        (int8_t *)tensor_ptr(quantized)->host,
        (float *)tensor_ptr(scales)->host, &quant_args, padded_rows,
        ctx->stream), "h3_quantize_bf16_int8_rows");
}

static int h3_hip_quantize_bf16_int8_head_major_rows(
    struct h3_gpu *ctx, h3_gpu_tensor *quantized,
    h3_gpu_tensor *scales, const h3_gpu_tensor *input, uint32_t rows,
    uint32_t padded_rows, uint32_t heads, uint32_t head_dim) {
    h3_int8_head_major_quant_args quant_args = {
        rows, padded_rows, heads, head_dim, 1.0f
    };
    return h3_hip_launch_ok(ctx, h3_launch_quantize_bf16_int8_head_major_rows(
        (const uint16_t *)tensor_ptr(input)->host,
        (int8_t *)tensor_ptr(quantized)->host,
        (float *)tensor_ptr(scales)->host, &quant_args, ctx->stream),
        "h3_quantize_bf16_int8_head_major_rows");
}

int h3_hip_quantize_bf16_int8_groups_dispatch(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *scales,
    const h3_gpu_tensor *input, uint32_t rows, uint32_t padded_rows,
    uint32_t columns, uint32_t group_size) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    uint32_t groups = columns / group_size;
    if (!ctx || !group_size || (group_size % 4u) || !rows || !columns ||
        padded_rows < rows || (columns % group_size) ||
        !h3_hip_require_bf16(ctx, input, (size_t)rows * columns,
                             "grouped int8 input") ||
        !h3_hip_require_i8(ctx, output, (size_t)padded_rows * columns,
                           "grouped int8 output") ||
        !h3_hip_require_f32(ctx, scales, (size_t)padded_rows * groups,
                            "grouped int8 scales")) {
        return 0;
    }
    h3_int8_group_quant_args quant_args = {rows, columns, group_size, groups};
    return h3_hip_launch_ok(ctx, h3_launch_quantize_bf16_int8_groups(
        (const uint16_t *)tensor_ptr(input)->host,
        (int8_t *)tensor_ptr(output)->host,
        (float *)tensor_ptr(scales)->host, &quant_args, padded_rows,
        ctx->stream), "h3_quantize_bf16_int8_groups");
}

static int h3_hip_quantize_bf16_int8_groups(
    struct h3_gpu *ctx, h3_gpu_tensor *quantized,
    h3_gpu_tensor *scales, const h3_gpu_tensor *input, uint32_t rows,
    uint32_t padded_rows, uint32_t columns, uint32_t group_size) {
    uint32_t groups = columns / group_size;
    if (!ctx || !group_size || (group_size % 4u) || !rows || !columns ||
        padded_rows < rows || (columns % group_size) ||
        !h3_hip_require_bf16(ctx, input, (size_t)rows * columns,
                             "grouped int8 input") ||
        !h3_hip_require_i8(ctx, quantized, (size_t)padded_rows * columns,
                           "grouped int8 output") ||
        !h3_hip_require_f32(ctx, scales, (size_t)padded_rows * groups,
                            "grouped int8 scales")) {
        return 0;
    }
    h3_int8_group_quant_args quant_args = {rows, columns, group_size, groups};
    return h3_hip_launch_ok(ctx, h3_launch_quantize_bf16_int8_groups(
        (const uint16_t *)tensor_ptr(input)->host,
        (int8_t *)tensor_ptr(quantized)->host,
        (float *)tensor_ptr(scales)->host, &quant_args, padded_rows,
        ctx->stream), "h3_quantize_bf16_int8_groups");
}

int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    uint32_t padded_rows = (rows + 127u) & ~127u;
    size_t activation_count = (size_t)padded_rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    (void)use_slower_uncached_int8_scales;
    if (!ctx || !rows || !input_dim || !output_dim ||
        !h3_hip_require_bf16(ctx, input, (size_t)rows * input_dim,
                             "int8 linear input") ||
        !h3_hip_require_i8(ctx, weight, weight_count, "int8 linear weight") ||
        !h3_hip_require_f32(ctx, weight_scales, output_dim,
                            "int8 linear weight scales") ||
        !h3_hip_require_i8(ctx, quantized_input, activation_count,
                           "int8 linear quantized input") ||
        !h3_hip_require_f32(ctx, input_scales, padded_rows,
                            "int8 linear input scales") ||
        !h3_hip_require_bf16(ctx, output, output_count, "int8 linear output")) {
        return 0;
    }
    h3_int8_quant_args quant_args = {rows, input_dim, 1.0f};
    if (!h3_hip_launch_ok(ctx, h3_launch_quantize_bf16_int8_rows(
            (const uint16_t *)tensor_ptr(input)->host,
            (int8_t *)tensor_ptr(quantized_input)->host,
            (float *)tensor_ptr(input_scales)->host, &quant_args, padded_rows,
            ctx->stream), "h3_quantize_bf16_int8_rows")) {
        return 0;
    }
    return h3_hip_launch_linear_int8_prequant(
        ctx, output, quantized_input, input_scales, weight, weight_scales,
        rows, input_dim, output_dim);
}

int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         h3_gpu_tensor *activated,
                         h3_gpu_tensor *quantized_activation,
                         h3_gpu_tensor *activation_scales,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *fc1_weight,
                         const h3_gpu_tensor *fc1_scales,
                         const h3_gpu_tensor *fc2_weight,
                         const h3_gpu_tensor *fc2_scales,
                         const h3_gpu_tensor *fc1_bf16, const h3_gpu_tensor *fc2_bf16,
                         uint32_t rows, uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim, int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k, int use_int8_row_fc2,
                         int input_is_quantized) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    uint32_t padded_rows = (rows + 127u) & ~127u;
    uint32_t fc2_scale_groups = hidden_dim / 1024u;
    uint32_t fc1_output_dim = hidden_dim * 2u;
    size_t activation_capacity = (size_t)padded_rows *
        (input_dim > hidden_dim ? input_dim : hidden_dim);
    size_t fc1_weight_count = (size_t)fc1_output_dim * input_dim;
    size_t fc2_weight_count = (size_t)output_dim * hidden_dim;
    (void)fc1_bf16;
    (void)fc2_bf16;
    (void)use_slower_grouped_quantizer;
    (void)use_slower_dynamic_fc1_k;
    (void)use_int8_row_fc2;
    if (!ctx || !rows || !input_dim || !hidden_dim || !output_dim ||
        !h3_hip_require_i8(ctx, quantized_activation, activation_capacity,
                           "int8 MLP activation") ||
        !h3_hip_require_f32(ctx, activation_scales,
                            (size_t)padded_rows *
                                (fc2_scale_groups > 0 ? fc2_scale_groups : 1u),
                            "int8 MLP activation scales") ||
        !h3_hip_require_i8(ctx, fc1_weight, fc1_weight_count,
                           "int8 MLP FC1 weight") ||
        !h3_hip_require_f32(ctx, fc1_scales, fc1_output_dim,
                            "int8 MLP FC1 scales") ||
        !h3_hip_require_i8(ctx, fc2_weight, fc2_weight_count,
                           "int8 MLP FC2 weight") ||
        !h3_hip_require_f32(ctx, fc2_scales, output_dim,
                            "int8 MLP FC2 scales") ||
        (!input_is_quantized &&
         !h3_hip_require_bf16(ctx, input, (size_t)rows * input_dim,
                              "int8 MLP input")) ||
        !h3_hip_require_bf16(ctx, activated, (size_t)rows * hidden_dim,
                             "int8 MLP activated") ||
        !h3_hip_require_bf16(ctx, output, (size_t)rows * output_dim,
                             "int8 MLP output")) {
        return 0;
    }
    h3_gpu_tensor *fc1_out = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * fc1_output_dim);
    if (!fc1_out) {
        h3_hip_set_error(ctx, "int8 MLP FC1 temporary allocation failed");
        return 0;
    }
    int ok = 1;
    if (!input_is_quantized &&
        !h3_hip_quantize_bf16_int8_rows(
            ctx, quantized_activation, activation_scales, input, rows,
            padded_rows, input_dim)) {
        ok = 0;
    }
    if (ok &&
        !h3_hip_launch_fc1_swiglu_int8_prequant(
            ctx, activated, quantized_activation, activation_scales,
            fc1_weight, fc1_scales, rows, input_dim, hidden_dim)) {
        if (!h3_hip_launch_linear_int8_prequant(
                ctx, fc1_out, quantized_activation, activation_scales,
                fc1_weight, fc1_scales, rows, input_dim, fc1_output_dim) ||
            !h3_gpu_swiglu_bf16(gpu, activated, fc1_out, rows, hidden_dim)) {
            ok = 0;
        }
    }
    if (ok && !use_int8_row_fc2 && hidden_dim % 1024u == 0 &&
        !h3_hip_quantize_bf16_int8_groups(
            ctx, quantized_activation, activation_scales, activated, rows,
            padded_rows, hidden_dim, 1024u)) {
        ok = 0;
    } else if (ok && !h3_hip_quantize_bf16_int8_rows(
                   ctx, quantized_activation, activation_scales, activated,
                   rows, padded_rows, hidden_dim)) {
        ok = 0;
    }
    if (ok && !use_int8_row_fc2 && hidden_dim % 1024u == 0) {
        if (!h3_hip_launch_linear_int8_grouped_prequant(
                ctx, output, quantized_activation, activation_scales,
                fc2_weight, fc2_scales, rows, hidden_dim, output_dim, 1024u)) {
            ok = 0;
        }
    } else if (ok && !h3_hip_launch_linear_int8_prequant(
                   ctx, output, quantized_activation, activation_scales,
                   fc2_weight, fc2_scales, rows, hidden_dim, output_dim)) {
        ok = 0;
    }
    h3_gpu_tensor_free(fc1_out);
    if (!ok) {
        h3_hip_set_error(ctx, "h3_mlp_int8_bf16 failed");
    }
    return ok;
}

int h3_gpu_linear_int8_head_major_bf16(
    h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *quantized_input,
    h3_gpu_tensor *input_scales, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    uint32_t rows, uint32_t heads, uint32_t head_dim, uint32_t output_dim) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    uint32_t input_dim = heads * head_dim;
    uint32_t padded_rows = (rows + 127u) & ~127u;
    size_t activation_count = (size_t)padded_rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!ctx || !rows || !heads || !head_dim || !output_dim ||
        !h3_hip_require_bf16(ctx, input, (size_t)rows * input_dim,
                             "head-major int8 linear input") ||
        !h3_hip_require_i8(ctx, weight, weight_count, "int8 linear weight") ||
        !h3_hip_require_f32(ctx, weight_scales, output_dim,
                            "int8 linear weight scales") ||
        !h3_hip_require_i8(ctx, quantized_input, activation_count,
                           "head-major int8 quantized input") ||
        !h3_hip_require_f32(ctx, input_scales, padded_rows,
                            "head-major int8 input scales") ||
        !h3_hip_require_bf16(ctx, output, output_count,
                             "head-major int8 linear output")) {
        return 0;
    }
    if (!h3_hip_quantize_bf16_int8_head_major_rows(
            ctx, quantized_input, input_scales, input, rows, padded_rows,
            heads, head_dim)) {
        return 0;
    }
    return h3_hip_launch_linear_int8_prequant(
        ctx, output, quantized_input, input_scales, weight, weight_scales,
        rows, input_dim, output_dim);
}

int h3_gpu_grouped_qkv_linear_rope_int8(
    h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key,
    h3_gpu_tensor *value, h3_gpu_tensor *quantized_input,
    h3_gpu_tensor *input_scales, const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales,
    const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm,
    const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin,
    uint32_t rows, uint32_t input_dim, uint32_t heads, uint32_t head_dim,
    uint32_t rope_half, float epsilon, int input_is_quantized,
    int use_slower_unfused_qkv_rope, int use_slower_scalar_qkv_rms,
    int use_slower_uncached_int8_scales) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    uint32_t inner = heads * head_dim;
    uint32_t qkv_dim = inner * 3u;
    uint32_t padded_rows = (rows + 127u) & ~127u;
    size_t activation_count = (size_t)padded_rows * input_dim;
    size_t weight_count = (size_t)qkv_dim * input_dim;
    size_t projected = (size_t)rows * inner;
    size_t rope_count = (size_t)rows * rope_half;
    (void)use_slower_unfused_qkv_rope;
    (void)use_slower_scalar_qkv_rms;
    (void)use_slower_uncached_int8_scales;
    if (!ctx || !rows || !input_dim || !heads || !head_dim || !rope_half ||
        !h3_hip_require_i8(ctx, weight, weight_count, "int8 QKV weight") ||
        !h3_hip_require_f32(ctx, weight_scales, qkv_dim, "int8 QKV scales") ||
        !h3_hip_require_i8(ctx, quantized_input, activation_count,
                           "int8 QKV quantized input") ||
        !h3_hip_require_f32(ctx, input_scales, padded_rows,
                            "int8 QKV input scales") ||
        (!input_is_quantized &&
         !h3_hip_require_bf16(ctx, input, (size_t)rows * input_dim,
                              "int8 QKV input")) ||
        !h3_hip_require_bf16(ctx, q_norm, head_dim, "int8 Q norm") ||
        !h3_hip_require_bf16(ctx, k_norm, head_dim, "int8 K norm") ||
        !h3_hip_require_bf16(ctx, rope_cos, rope_count, "int8 RoPE cos") ||
        !h3_hip_require_bf16(ctx, rope_sin, rope_count, "int8 RoPE sin") ||
        !h3_hip_require_bf16(ctx, query, projected, "int8 query") ||
        !h3_hip_require_bf16(ctx, key, projected, "int8 key") ||
        !h3_hip_require_bf16(ctx, value, projected, "int8 value")) {
        return 0;
    }
    h3_gpu_tensor *qkv = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * qkv_dim);
    if (!qkv) {
        h3_hip_set_error(ctx, "int8 QKV temporary allocation failed");
        return 0;
    }
    int ok = 1;
    if (!input_is_quantized &&
        !h3_hip_quantize_bf16_int8_rows(
            ctx, quantized_input, input_scales, input, rows, padded_rows,
            input_dim)) {
        ok = 0;
    }
    if (ok && !h3_hip_launch_linear_int8_prequant(
            ctx, qkv, quantized_input, input_scales, weight, weight_scales,
            rows, input_dim, qkv_dim)) {
        ok = 0;
    }
    if (ok && !h3_gpu_grouped_qkv_rope_bf16(
            gpu, query, key, value, qkv, q_norm, k_norm, rope_cos, rope_sin,
            rows, heads, head_dim, rope_half, epsilon)) {
        ok = 0;
    }
    h3_gpu_tensor_free(qkv);
    if (!ok) {
        h3_hip_set_error(ctx, "h3_grouped_qkv_linear_rope_int8 failed");
    }
    return ok;
}

int h3_gpu_gate_adaln_quantize_int8(
    h3_gpu *gpu, h3_gpu_tensor *gated_residual,
    h3_gpu_tensor *quantized_output, h3_gpu_tensor *quantized_scales,
    const h3_gpu_tensor *residual, const h3_gpu_tensor *branch,
    const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *gate_modulation,
    const h3_gpu_tensor *norm_modulation, const h3_gpu_tensor *row_map,
    uint32_t rows, uint32_t padded_rows, uint32_t width, uint32_t slots,
    uint32_t gate_slot, uint32_t shift_slot, uint32_t scale_slot,
    float epsilon) {
    struct h3_gpu *ctx = gpu_ptr(gpu);
    size_t count = (size_t)rows * width;
    size_t padded_count = (size_t)padded_rows * width;
    if (!ctx || !rows || !width || padded_rows < rows || width > 5376u ||
        gate_slot >= slots || shift_slot >= slots || scale_slot >= slots ||
        !h3_hip_require_bf16(ctx, residual, count, "int8 gate AdaLN residual") ||
        !h3_hip_require_bf16(ctx, branch, count, "int8 gate AdaLN branch") ||
        !h3_hip_require_bf16(ctx, norm_weight, width, "int8 gate AdaLN norm") ||
        !h3_hip_require_bf16(ctx, gate_modulation, 1,
                             "int8 gate AdaLN gate modulation") ||
        !h3_hip_require_bf16(ctx, norm_modulation, 1,
                             "int8 gate AdaLN norm modulation") ||
        !h3_hip_require_u32(ctx, row_map, rows, "int8 gate AdaLN row map") ||
        !h3_hip_require_bf16(ctx, gated_residual, count,
                             "int8 gate AdaLN gated residual") ||
        !h3_hip_require_i8(ctx, quantized_output, padded_count,
                           "int8 gate AdaLN quantized output") ||
        !h3_hip_require_f32(ctx, quantized_scales, padded_rows,
                            "int8 gate AdaLN scales")) {
        return 0;
    }
    h3_gpu_tensor *adaln_out = h3_gpu_tensor_new_bf16(gpu, count);
    if (!adaln_out) {
        h3_hip_set_error(ctx, "int8 gate AdaLN temporary allocation failed");
        return 0;
    }
    int ok = h3_gpu_gate_adaln_bf16(
        gpu, gated_residual, adaln_out, residual, branch, norm_weight,
        gate_modulation, norm_modulation, row_map, rows, width, slots,
        gate_slot, shift_slot, scale_slot, epsilon) &&
             h3_hip_quantize_bf16_int8_rows(
                 ctx, quantized_output, quantized_scales, adaln_out, rows,
                 padded_rows, width);
    h3_gpu_tensor_free(adaln_out);
    if (!ok) {
        h3_hip_set_error(ctx, "h3_gate_adaln_quantize_int8 failed");
    }
    return ok;
}

int h3_hip_unimplemented(struct h3_gpu *gpu, const char *name) {
    h3_hip_set_error(gpu, "HIP backend: %s is not implemented yet", name);
    return 0;
}
