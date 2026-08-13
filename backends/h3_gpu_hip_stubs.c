#include "h3_gpu.h"

struct h3_gpu;
int h3_hip_unimplemented(struct h3_gpu *gpu, const char *name);

int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *attended, uint32_t batch, uint32_t length, uint32_t heads, uint32_t head_dim, uint32_t output_dim) { return h3_hip_unimplemented(NULL, "h3_gpu_audio_attention_pool_f32"); }

int h3_gpu_vae_encoder_pad_f32( h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t batch, uint32_t depth, uint32_t height, uint32_t width, uint32_t channels, uint32_t depth_front, uint32_t height_before, uint32_t height_after, uint32_t width_before, uint32_t width_after) { return h3_hip_unimplemented(NULL, "h3_gpu_vae_encoder_pad_f32"); }

int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t depth, uint32_t height, uint32_t width, uint32_t input_channels, uint32_t output_channels, uint32_t kernel_depth, uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_depth, uint32_t stride_height, uint32_t stride_width) { return h3_hip_unimplemented(NULL, "h3_gpu_conv3d_f32"); }

int h3_gpu_vae_encoder_group_norm_silu_f32( h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t depth, uint32_t height, uint32_t width, uint32_t channels, uint32_t groups, float epsilon) { return h3_hip_unimplemented(NULL, "h3_gpu_vae_encoder_group_norm_silu_f32"); }

