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
    uint32_t elements;
    float left_scale;
    float right_scale;
} h3_add_scaled_f32_args;

typedef struct {
    uint32_t elements;
    float minimum;
    float maximum;
} h3_clip_f32_args;

typedef struct {
    uint32_t outer;
    uint32_t inner;
} h3_weight_norm_args;

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

typedef struct {
    uint32_t input_offset;
    uint32_t original_offset;
    uint32_t baseline_offset;
    uint32_t rows;
    uint32_t width;
} h3_token_pool_args;

typedef struct {
    uint32_t input_offset;
    uint32_t original_offset;
    uint32_t baseline_offset;
    uint32_t rows;
    uint32_t width;
    uint32_t slots;
    uint32_t shift_slot;
    uint32_t scale_slot;
    float epsilon;
} h3_token_pool_adaln_args;

typedef struct {
    uint32_t original_offset;
    uint32_t baseline_offset;
    uint32_t rows;
    uint32_t width;
    uint32_t exact_prefix_rows;
    float update_scale;
} h3_token_expand_args;

typedef struct {
    uint32_t original_offset;
    uint32_t baseline_offset;
    uint32_t rows;
    uint32_t width;
    uint32_t exact_prefix_rows;
    uint32_t slots;
    uint32_t shift_slot;
    uint32_t scale_slot;
    float update_scale;
    float epsilon;
} h3_token_expand_adaln_args;

typedef struct {
    uint32_t rows;
    uint32_t width;
    uint32_t output_dim;
    uint32_t slots;
    uint32_t shift_slot;
    uint32_t scale_slot;
    uint32_t has_bias;
} h3_adaln_linear_args;

typedef struct {
    uint32_t tokens;
    uint32_t vocab_size;
    uint32_t width;
} h3_embedding_args;

typedef struct {
    uint32_t sequence;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    float epsilon;
} h3_text_rope_args;

typedef struct {
    uint32_t sequence;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
} h3_text_rope_inplace_args;

typedef struct {
    uint32_t sequence;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    float scale;
} h3_gqa_args;

typedef struct {
    uint32_t sequence;
    uint32_t heads;
    uint32_t head_dim;
    float epsilon;
} h3_head_norm_args;

typedef struct {
    uint32_t rows;
    uint32_t columns;
    float clip;
} h3_int8_quant_args;

typedef struct {
    uint32_t rows;
    uint32_t padded_rows;
    uint32_t heads;
    uint32_t head_dim;
    float clip;
} h3_int8_head_major_quant_args;

typedef struct {
    uint32_t rows;
    uint32_t columns;
    uint32_t group_size;
    uint32_t groups;
} h3_int8_group_quant_args;

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
int h3_launch_silu_f32(const float *input, float *output, uint32_t count,
                       hipStream_t stream);
int h3_launch_gelu_bf16(const uint16_t *input, uint16_t *output,
                        const h3_gelu_bf16_args *args, hipStream_t stream);
int h3_launch_swiglu_bf16(const uint16_t *fused, uint16_t *output,
                          const h3_swiglu_args *args, hipStream_t stream);
int h3_launch_swiglu_f32(const float *fused, float *output,
                         const h3_swiglu_args *args, hipStream_t stream);
int h3_launch_clip_f32(const float *input, float *output,
                       const h3_clip_f32_args *args, hipStream_t stream);
int h3_launch_add_scaled_f32(const float *left, const float *right,
                             float *output, const h3_add_scaled_f32_args *args,
                             hipStream_t stream);
int h3_launch_scale_add_f32(const float *residual, const float *branch,
                            const float *scale, float *output,
                            const h3_swiglu_args *args, hipStream_t stream);
int h3_launch_weight_norm_f32(const float *vector, const float *magnitude,
                              float *output, const h3_weight_norm_args *args,
                              hipStream_t stream);
int h3_launch_geglu_f32(const float *gate, const float *linear, float *output,
                        uint32_t count, hipStream_t stream);
int h3_launch_rms_norm_bf16(const uint16_t *input, const uint16_t *weight,
                            uint16_t *output, const h3_norm_args *args,
                            hipStream_t stream);
int h3_launch_rms_norm_f32(const float *input, const float *weight,
                           float *output, const h3_norm_args *args,
                           hipStream_t stream);
int h3_launch_layer_norm_bf16(const uint16_t *input, const uint16_t *weight,
                              const uint16_t *bias, uint16_t *output,
                              const h3_norm_args *args, hipStream_t stream);
int h3_launch_layer_norm_f32(const float *input, const float *weight,
                             const float *bias, float *output,
                             const h3_norm_args *args, hipStream_t stream);
int h3_launch_linear_bf16(const uint16_t *input, const uint16_t *weight,
                          const uint16_t *bias, uint16_t *output,
                          const h3_linear_args *args, hipStream_t stream);
int h3_launch_linear_f32(const float *input, const float *weight,
                         const float *bias, float *output,
                         const h3_linear_args *args, hipStream_t stream);
int h3_launch_gate_bf16(const uint16_t *residual, const uint16_t *branch,
                        const uint16_t *modulation, const uint32_t *row_map,
                        uint16_t *output, const h3_gate_args *args,
                        hipStream_t stream);
int h3_launch_gate_f32(const float *residual, const float *branch,
                       const float *modulation, const uint32_t *row_map,
                       float *output, const h3_gate_args *args,
                       hipStream_t stream);
int h3_launch_adaln_bf16(const uint16_t *input, const uint16_t *weight,
                         const uint16_t *modulation, const uint32_t *row_map,
                         uint16_t *output, const h3_adaln_args *args,
                         hipStream_t stream);
int h3_launch_adaln_f32(const float *input, const float *weight,
                        const float *modulation, const uint32_t *row_map,
                        float *output, const h3_adaln_args *args,
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
int h3_launch_vision_qkv_rope_bf16(const uint16_t *qkv,
                                   const uint16_t *rope_cos,
                                   const uint16_t *rope_sin, uint16_t *query,
                                   uint16_t *key, uint16_t *value,
                                   const h3_qkv_args *args, hipStream_t stream);
int h3_launch_sdpa_bf16(const uint16_t *query, const uint16_t *key,
                        const uint16_t *value, uint16_t *output,
                        const h3_sdpa_args *args, hipStream_t stream);
int h3_launch_euler_bf16(float *sample, const uint16_t *last,
                         const uint16_t *previous, const h3_euler_args *args,
                         hipStream_t stream);
int h3_launch_rms_inverse_bf16(const uint16_t *input, float *inverse,
                               const h3_norm_args *args, hipStream_t stream);
int h3_launch_adaln_linear_bf16(const uint16_t *input, const float *inverse,
                                const uint16_t *norm_weight,
                                const uint16_t *modulation,
                                const uint32_t *row_map, const uint16_t *weight,
                                const uint16_t *bias, uint16_t *output,
                                const h3_adaln_linear_args *args,
                                hipStream_t stream);
int h3_launch_token_pool_bf16(const uint16_t *input, const uint32_t *pairs,
                              uint16_t *output, uint16_t *baseline,
                              const uint32_t *baseline_indices,
                              uint16_t *original, const h3_token_pool_args *args,
                              hipStream_t stream);
int h3_launch_token_pool_adaln_bf16(
    const uint16_t *input, const uint32_t *pairs, uint16_t *residual,
    uint16_t *baseline, const uint32_t *baseline_indices, uint16_t *original,
    const uint16_t *weight, const uint16_t *modulation,
    const uint32_t *row_map, uint16_t *output,
    const h3_token_pool_adaln_args *args, hipStream_t stream);
int h3_launch_token_expand_delta_bf16(
    const uint16_t *original, const uint16_t *reduced, const uint16_t *baseline,
    const uint32_t *baseline_indices, const uint32_t *parents,
    uint16_t *output, const h3_token_expand_args *args, hipStream_t stream);
int h3_launch_token_expand_adaln_bf16(
    const uint16_t *original, const uint16_t *reduced, const uint16_t *baseline,
    const uint32_t *baseline_indices, const uint32_t *parents,
    uint16_t *residual, const uint16_t *weight, const uint16_t *modulation,
    const uint32_t *row_map, uint16_t *output,
    const h3_token_expand_adaln_args *args, hipStream_t stream);
int h3_launch_embedding_bf16(const uint16_t *weight, const uint32_t *token_ids,
                               uint16_t *output, const h3_embedding_args *args,
                               hipStream_t stream);
int h3_launch_silu_mul_bf16(const uint16_t *gate, const uint16_t *up,
                            uint16_t *output, uint32_t count,
                            hipStream_t stream);
int h3_launch_linear_f32_tiled_bf16(const float *input, const float *weight,
                                    const float *bias, uint16_t *output,
                                    const h3_linear_args *args,
                                    hipStream_t stream);
int h3_launch_linear_f32_tiled_bf16_map(
    const float *input, const float *weight, const float *bias,
    uint16_t *output, const uint32_t *row_map, const h3_linear_args *args,
    hipStream_t stream);
int h3_launch_text_qk_rope_bf16(
    const uint16_t *query_input, const uint16_t *key_input,
    const uint16_t *q_weight, const uint16_t *k_weight,
    const uint16_t *rope_cos, const uint16_t *rope_sin,
    uint16_t *query_output, uint16_t *key_output,
    const h3_text_rope_args *args, hipStream_t stream);
int h3_launch_head_rms_norm_bf16(uint16_t *tensor, const uint16_t *weight,
                                 const h3_head_norm_args *args,
                                 hipStream_t stream);
int h3_launch_rope_text_bf16(uint16_t *query, uint16_t *key,
                             const float *rope_cos, const float *rope_sin,
                             const h3_text_rope_inplace_args *args,
                             hipStream_t stream);
int h3_launch_gqa_causal_bf16(const uint16_t *query, const uint16_t *key,
                              const uint16_t *value, uint16_t *output,
                              const h3_gqa_args *args, hipStream_t stream);
int h3_launch_quantize_bf16_int8_rows(const uint16_t *input, int8_t *output,
                                      float *scales,
                                      const h3_int8_quant_args *args,
                                      uint32_t dispatch_rows,
                                      hipStream_t stream);
int h3_launch_quantize_bf16_int8_head_major_rows(
    const uint16_t *input, int8_t *output, float *scales,
    const h3_int8_head_major_quant_args *args, hipStream_t stream);
int h3_launch_quantize_bf16_int8_groups(
    const uint16_t *input, int8_t *output, float *scales,
    const h3_int8_group_quant_args *args, uint32_t dispatch_rows,
    hipStream_t stream);
int h3_launch_linear_int8_bf16_naive(
    const int8_t *input, const int8_t *weight, const float *input_scales,
    const float *weight_scales, uint16_t *output, const h3_linear_args *args,
    hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
