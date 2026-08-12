#ifndef H3_KERNELS_H
#define H3_KERNELS_H

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t rows;
    uint32_t input_dim;
    uint32_t output_dim;
    uint32_t has_bias;
} h3_linear_args;

typedef struct {
    uint32_t rows;
    uint32_t width;
    float epsilon;
} h3_norm_args;

typedef struct {
    uint32_t elements;
    uint32_t approximate;
} h3_gelu_bf16_args;

typedef struct {
    uint32_t rows;
    uint32_t width;
} h3_swiglu_args;

typedef struct {
    uint32_t rows;
    uint32_t width;
    uint32_t slots;
    uint32_t gate_slot;
} h3_gate_args;

typedef struct {
    uint32_t rows;
    uint32_t width;
    uint32_t slots;
    uint32_t shift_slot;
    uint32_t scale_slot;
    float epsilon;
} h3_adaln_args;

typedef struct {
    uint32_t rows;
    uint32_t width;
    uint32_t slots;
    uint32_t gate_slot;
    uint32_t shift_slot;
    uint32_t scale_slot;
    float epsilon;
} h3_gate_adaln_args;

typedef struct {
    uint32_t sequence;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t rope_half;
    uint32_t grouped;
    float epsilon;
} h3_qkv_args;

typedef struct {
    uint32_t sequence;
    uint32_t heads;
    uint32_t head_dim;
    float scale;
} h3_sdpa_args;

typedef struct {
    uint32_t sample_offset;
    uint32_t elements;
    float delta;
    float ratio;
} h3_euler_args;

int h3_launch_cast_f32_to_bf16(const float *input, uint16_t *output,
                               uint32_t count, hipStream_t stream);
int h3_launch_cast_bf16_to_f32(const uint16_t *input, float *output,
                               uint32_t count, hipStream_t stream);
int h3_launch_add_bf16(const uint16_t *left, const uint16_t *right,
                       uint16_t *output, uint32_t count, hipStream_t stream);
int h3_launch_sub_bf16(const uint16_t *left, const uint16_t *right,
                       uint16_t *output, uint32_t count, hipStream_t stream);
int h3_launch_silu_bf16(const uint16_t *input, uint16_t *output,
                        uint32_t count, hipStream_t stream);
int h3_launch_gelu_bf16(const uint16_t *input, uint16_t *output,
                        const h3_gelu_bf16_args *args, hipStream_t stream);
int h3_launch_swiglu_bf16(const uint16_t *fused, uint16_t *output,
                          const h3_swiglu_args *args, hipStream_t stream);
int h3_launch_rms_norm_bf16(const uint16_t *input, const uint16_t *weight,
                            uint16_t *output, const h3_norm_args *args,
                            hipStream_t stream);
int h3_launch_layer_norm_bf16(const uint16_t *input, const uint16_t *weight,
                              const uint16_t *bias, uint16_t *output,
                              const h3_norm_args *args, hipStream_t stream);
int h3_launch_linear_bf16(const uint16_t *input, const uint16_t *weight,
                          const uint16_t *bias, uint16_t *output,
                          const h3_linear_args *args, hipStream_t stream);
int h3_launch_gate_bf16(const uint16_t *residual, const uint16_t *branch,
                        const uint16_t *modulation, const uint32_t *row_map,
                        uint16_t *output, const h3_gate_args *args,
                        hipStream_t stream);
int h3_launch_adaln_bf16(const uint16_t *input, const uint16_t *weight,
                         const uint16_t *modulation, const uint32_t *row_map,
                         uint16_t *output, const h3_adaln_args *args,
                         hipStream_t stream);
int h3_launch_gate_adaln_bf16(const uint16_t *residual, const uint16_t *branch,
                              const uint16_t *gate_modulation,
                              const uint32_t *row_map, const uint16_t *weight,
                              const uint16_t *norm_modulation,
                              uint16_t *gated_residual, uint16_t *output,
                              const h3_gate_adaln_args *args,
                              hipStream_t stream);
int h3_launch_qkv_rope_bf16(const uint16_t *qkv, const uint16_t *q_weight,
                            const uint16_t *k_weight, const uint16_t *rope_cos,
                            const uint16_t *rope_sin, uint16_t *query,
                            uint16_t *key, uint16_t *value,
                            const h3_qkv_args *args, hipStream_t stream);
int h3_launch_sdpa_bf16(const uint16_t *query, const uint16_t *key,
                        const uint16_t *value, uint16_t *output,
                        const h3_sdpa_args *args, hipStream_t stream);
int h3_launch_euler_bf16(float *sample, const uint16_t *last,
                         const uint16_t *previous, const h3_euler_args *args,
                         hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
