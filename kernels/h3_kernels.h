#ifndef H3_KERNELS_H
#define H3_KERNELS_H

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int h3_launch_cast_f32_to_bf16(const float *input, uint16_t *output,
                               uint32_t count, hipStream_t stream);
int h3_launch_cast_bf16_to_f32(const uint16_t *input, float *output,
                               uint32_t count, hipStream_t stream);
int h3_launch_add_bf16(const uint16_t *left, const uint16_t *right,
                       uint16_t *output, uint32_t count, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
