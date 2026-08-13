#include "h3_gpu.h"

struct h3_gpu;
int h3_hip_unimplemented(struct h3_gpu *gpu, const char *name);

int h3_gpu_vae_encoder_group_norm_silu_f32( h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t depth, uint32_t height, uint32_t width, uint32_t channels, uint32_t groups, float epsilon) { return h3_hip_unimplemented(NULL, "h3_gpu_vae_encoder_group_norm_silu_f32"); }

