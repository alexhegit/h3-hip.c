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

/* XOR exchange inside each 16-lane row. RDNA has DPP row_xmask, a single VALU
 * op; __shfl_xor lowers to ds_swizzle/ds_bpermute, which go through the LDS
 * pipe and stall the WMMA loop. gfx9 has no row_xmask, so CDNA keeps the
 * shuffle. MASK must be below 16. */
template <int MASK>
__device__ inline float h3_xor16(float value) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
    defined(__gfx1152__) || defined(__gfx1153__) || defined(__gfx1200__) || \
    defined(__gfx1201__)
    return __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(value), 0x160 + MASK, 0xf, 0xf, false));
#else
    return __shfl_xor(value, MASK, 16);
#endif
}

#endif
