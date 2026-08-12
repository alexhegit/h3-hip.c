#include "h3_gpu.h"

struct h3_gpu;
int h3_hip_unimplemented(struct h3_gpu *gpu, const char *name);

int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows, uint32_t input_dim, uint32_t output_dim) { return h3_hip_unimplemented(NULL, "h3_gpu_linear_f32"); }

int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows, uint32_t input_dim, uint32_t output_dim) { return h3_hip_unimplemented(NULL, "h3_gpu_patch_linear_bf16"); }

int h3_gpu_patch_linear_bf16_offset( h3_gpu *gpu, h3_gpu_tensor *output, size_t output_offset, const h3_gpu_tensor *input, size_t input_offset, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows, uint32_t input_dim, uint32_t output_dim) { return h3_hip_unimplemented(NULL, "h3_gpu_patch_linear_bf16_offset"); }

int h3_gpu_patch_linear_bf16_map( h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, const h3_gpu_tensor *row_map, uint32_t output_rows, uint32_t rows, uint32_t input_dim, uint32_t output_dim) { return h3_hip_unimplemented(NULL, "h3_gpu_patch_linear_bf16_map"); }

int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t elements) { return h3_hip_unimplemented(NULL, "h3_gpu_silu_f32"); }

int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination, size_t destination_offset, const h3_gpu_tensor *source, size_t source_offset, size_t elements) { return h3_hip_unimplemented(NULL, "h3_gpu_copy_f32"); }

int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, uint32_t rows, uint32_t width, float epsilon) { return h3_hip_unimplemented(NULL, "h3_gpu_rms_norm_f32"); }

int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width, uint32_t slots, uint32_t shift_slot, uint32_t scale_slot, float epsilon) { return h3_hip_unimplemented(NULL, "h3_gpu_adaln_f32"); }

int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *residual, const h3_gpu_tensor *branch, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width, uint32_t slots, uint32_t gate_slot) { return h3_hip_unimplemented(NULL, "h3_gpu_gate_f32"); }

int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv, const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon) { return h3_hip_unimplemented(NULL, "h3_gpu_qkv_rope_f32"); }

int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query, const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t sequence, uint32_t heads, uint32_t head_dim, float scale) { return h3_hip_unimplemented(NULL, "h3_gpu_sdpa_f32"); }

int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *fused, uint32_t rows, uint32_t width) { return h3_hip_unimplemented(NULL, "h3_gpu_swiglu_f32"); }

int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *residual, const h3_gpu_tensor *branch, const h3_gpu_tensor *scale, uint32_t rows, uint32_t width) { return h3_hip_unimplemented(NULL, "h3_gpu_scale_add_f32"); }

int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows, uint32_t width, float epsilon) { return h3_hip_unimplemented(NULL, "h3_gpu_layer_norm_f32"); }

int h3_gpu_video_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon) { return h3_hip_unimplemented(NULL, "h3_gpu_video_qkv_rope_f32"); }

int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t length, uint32_t input_channels, uint32_t output_channels, uint32_t kernel, uint32_t padding, uint32_t dilation) { return h3_hip_unimplemented(NULL, "h3_gpu_conv1d_f32"); }

int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t length, uint32_t input_channels, uint32_t output_channels, uint32_t kernel, uint32_t stride, uint32_t padding, uint32_t dilation) { return h3_hip_unimplemented(NULL, "h3_gpu_conv1d_stride_f32"); }

int h3_gpu_conv_transpose1d_f32( h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t length, uint32_t input_channels, uint32_t output_channels, uint32_t kernel, uint32_t stride, uint32_t padding) { return h3_hip_unimplemented(NULL, "h3_gpu_conv_transpose1d_f32"); }

int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *vector, const h3_gpu_tensor *magnitude, uint32_t outer, uint32_t inner) { return h3_hip_unimplemented(NULL, "h3_gpu_weight_norm_f32"); }

int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *left, const h3_gpu_tensor *right, float left_scale, float right_scale, uint32_t elements) { return h3_hip_unimplemented(NULL, "h3_gpu_add_scaled_f32"); }

int h3_gpu_alias_free_snake_f32( h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *alpha_log, const h3_gpu_tensor *beta_log, const h3_gpu_tensor *upsample_filter, const h3_gpu_tensor *downsample_filter, uint32_t batch, uint32_t length, uint32_t channels) { return h3_hip_unimplemented(NULL, "h3_gpu_alias_free_snake_f32"); }

int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *alpha, uint32_t batch, uint32_t length, uint32_t channels) { return h3_hip_unimplemented(NULL, "h3_gpu_snake1d_f32"); }

int h3_gpu_audio_qkv_split_f32(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv, const h3_gpu_tensor *q_bias, const h3_gpu_tensor *k_bias, const h3_gpu_tensor *v_bias, uint32_t batch, uint32_t length, uint32_t heads, uint32_t head_dim) { return h3_hip_unimplemented(NULL, "h3_gpu_audio_qkv_split_f32"); }

int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query, const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t batch, uint32_t sequence, uint32_t heads, uint32_t head_dim, float scale) { return h3_hip_unimplemented(NULL, "h3_gpu_sdpa_causal_f32"); }

int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *attended, uint32_t batch, uint32_t length, uint32_t heads, uint32_t head_dim, uint32_t output_dim) { return h3_hip_unimplemented(NULL, "h3_gpu_audio_attention_pool_f32"); }

int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *gate, const h3_gpu_tensor *linear, uint32_t elements) { return h3_hip_unimplemented(NULL, "h3_gpu_geglu_f32"); }

int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t elements, float minimum, float maximum) { return h3_hip_unimplemented(NULL, "h3_gpu_clip_f32"); }

int h3_gpu_vae_encoder_pad_f32( h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t batch, uint32_t depth, uint32_t height, uint32_t width, uint32_t channels, uint32_t depth_front, uint32_t height_before, uint32_t height_after, uint32_t width_before, uint32_t width_after) { return h3_hip_unimplemented(NULL, "h3_gpu_vae_encoder_pad_f32"); }

int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t depth, uint32_t height, uint32_t width, uint32_t input_channels, uint32_t output_channels, uint32_t kernel_depth, uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_depth, uint32_t stride_height, uint32_t stride_width) { return h3_hip_unimplemented(NULL, "h3_gpu_conv3d_f32"); }

int h3_gpu_vae_encoder_group_norm_silu_f32( h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t depth, uint32_t height, uint32_t width, uint32_t channels, uint32_t groups, float epsilon) { return h3_hip_unimplemented(NULL, "h3_gpu_vae_encoder_group_norm_silu_f32"); }

int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *activated, const h3_gpu_tensor *input, const h3_gpu_tensor *fc1_weight, const h3_gpu_tensor *fc2_weight, uint32_t rows, uint32_t input_dim, uint32_t hidden_dim, uint32_t output_dim) { return h3_hip_unimplemented(NULL, "h3_gpu_mlp_nax_bf16"); }

int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *scales, const h3_gpu_tensor *input, uint32_t rows, uint32_t columns) { return h3_hip_unimplemented(NULL, "h3_gpu_quantize_weight_int8"); }

int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales, uint32_t rows, uint32_t input_dim, uint32_t output_dim, int use_slower_uncached_int8_scales) { return h3_hip_unimplemented(NULL, "h3_gpu_linear_int8_bf16"); }

int h3_gpu_linear_int8_head_major_bf16( h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales, uint32_t rows, uint32_t heads, uint32_t head_dim, uint32_t output_dim) { return h3_hip_unimplemented(NULL, "h3_gpu_linear_int8_head_major_bf16"); }

int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *activated, h3_gpu_tensor *quantized_activation, h3_gpu_tensor *activation_scales, const h3_gpu_tensor *input, const h3_gpu_tensor *fc1_weight, const h3_gpu_tensor *fc1_scales, const h3_gpu_tensor *fc2_weight, const h3_gpu_tensor *fc2_scales, const h3_gpu_tensor *fc1_bf16, const h3_gpu_tensor *fc2_bf16, uint32_t rows, uint32_t input_dim, uint32_t hidden_dim, uint32_t output_dim, int use_slower_grouped_quantizer, int use_slower_dynamic_fc1_k, int use_int8_row_fc2, int input_is_quantized) { return h3_hip_unimplemented(NULL, "h3_gpu_mlp_int8_bf16"); }

int h3_gpu_vision_qkv_rope_bf16( h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half) { return h3_hip_unimplemented(NULL, "h3_gpu_vision_qkv_rope_bf16"); }

int h3_gpu_gate_adaln_quantize_int8( h3_gpu *gpu, h3_gpu_tensor *gated_residual, h3_gpu_tensor *quantized_output, h3_gpu_tensor *quantized_scales, const h3_gpu_tensor *residual, const h3_gpu_tensor *branch, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *gate_modulation, const h3_gpu_tensor *norm_modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t padded_rows, uint32_t width, uint32_t slots, uint32_t gate_slot, uint32_t shift_slot, uint32_t scale_slot, float epsilon) { return h3_hip_unimplemented(NULL, "h3_gpu_gate_adaln_quantize_int8"); }

int h3_gpu_grouped_qkv_linear_rope_int8( h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales, const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t rows, uint32_t input_dim, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon, int input_is_quantized, int use_slower_unfused_qkv_rope, int use_slower_scalar_qkv_rms, int use_slower_uncached_int8_scales) { return h3_hip_unimplemented(NULL, "h3_gpu_grouped_qkv_linear_rope_int8"); }

int h3_gpu_text_qk_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query_output, h3_gpu_tensor *key_output, const h3_gpu_tensor *query_input, const h3_gpu_tensor *key_input, const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim, float epsilon) { return h3_hip_unimplemented(NULL, "h3_gpu_text_qk_rope_bf16"); }

int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor, const h3_gpu_tensor *weight, uint32_t sequence, uint32_t heads, uint32_t head_dim, float epsilon) { return h3_hip_unimplemented(NULL, "h3_gpu_head_rms_norm_bf16"); }

int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, const h3_gpu_tensor *rope_cos_f32, const h3_gpu_tensor *rope_sin_f32, uint32_t sequence, uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim) { return h3_hip_unimplemented(NULL, "h3_gpu_rope_text_bf16"); }

int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query, const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t sequence, uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim, float scale) { return h3_hip_unimplemented(NULL, "h3_gpu_gqa_causal_bf16"); }
