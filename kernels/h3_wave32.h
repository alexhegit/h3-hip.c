#ifndef H3_WAVE32_H
#define H3_WAVE32_H

/* Logical 32-lane warps. CDNA (gfx90a / gfx942) is wave64; RDNA3.5 (gfx1151)
 * and RDNA4 (gfx1200 / gfx1201) are wave32. Kernels still launch 32-thread
 * groups and index head_dim as lane + k*32, so shuffles must use width 32 and
 * ignore the upper half of a CDNA wavefront.
 *
 * rocWMMA 16x16 f32 accumulator, wave32. Column is lane&15 on both.
 *   gfx11 (probed on gfx1151): element i -> row 2*i + (lane>>4)
 *   gfx12 (probed on gfx1201): element i -> row i + 8*(lane>>4)
 * CDNA wave64 kernels keep their own 4*(lane>>4)+i map and do not use this. */
#if defined(__gfx1200__) || defined(__gfx1201__)
#define H3_WMMA_ACC_ROW(elem, row_group) \
    ((uint32_t)(elem) + 8u * (uint32_t)(row_group))
#else
#define H3_WMMA_ACC_ROW(elem, row_group) \
    (2u * (uint32_t)(elem) + (uint32_t)(row_group))
#endif

__device__ inline float h3_shfl_xor32(float value, int mask) {
    return __shfl_xor(value, mask, 32);
}

__device__ inline float h3_shfl32(float value, int src) {
    return __shfl(value, src, 32);
}

/* lane ^ MASK inside each 16-lane row via DPP row_xmask (gfx10+), the same
 * value __shfl_xor(v, MASK, 32) returns for MASK < 16 without the LDS
 * round trip of ds_bpermute. */
template <int MASK>
__device__ inline float h3_row16_xor(float value) {
    static_assert(MASK > 0 && MASK < 16, "row_xmask covers one 16-lane row");
#if defined(__GFX9__)
    return __shfl_xor(value, MASK, 32);
#else
    return __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(value), 0x160 + MASK, 0xf, 0xf, false));
#endif
}

#endif
