#ifndef H3_FASTH3_H
#define H3_FASTH3_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

/* Merge a FastH3 dense-datafree adapter into one loaded checkpoint tensor.
 * No-op when H3_FASTH3_LORA is unset. QKV is scattered into the per-head
 * interleaved layout; fc1 swaps Diffusers' [value; gate] back to [gate; value].
 * weight_f32 is 1 when the resident tensor is float32. */
int h3_fasth3_apply(h3_gpu *gpu, h3_gpu_tensor *weight, const char *name,
                    uint32_t rows, uint32_t cols, int weight_f32,
                    char *error, size_t error_size);

/* Fail if any adapter tensor was never applied. No-op when FastH3 is off. */
int h3_fasth3_finish(char *error, size_t error_size);

#endif
