#ifndef H3_WAVE32_H
#define H3_WAVE32_H

/* Logical 32-lane warps. CDNA (gfx90a / gfx942) is wave64; RDNA3.5 is wave32.
 * Kernels still launch 32-thread groups and index head_dim as lane + k*32, so
 * shuffles must use width 32 and ignore the upper half of a CDNA wavefront. */

__device__ inline float h3_shfl_xor32(float value, int mask) {
    return __shfl_xor(value, mask, 32);
}

__device__ inline float h3_shfl32(float value, int src) {
    return __shfl(value, src, 32);
}

#endif
