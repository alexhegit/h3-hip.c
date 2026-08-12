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

int h3_hip_unimplemented(struct h3_gpu *gpu, const char *name) {
    h3_hip_set_error(gpu, "HIP backend: %s is not implemented yet", name);
    return 0;
}
