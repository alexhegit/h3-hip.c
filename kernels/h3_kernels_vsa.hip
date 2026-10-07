/* Opt-in FastH3 VSA. Video tokens are 4x4x4 cubes (64). Prefix segments
 * (text, keyframes, audio) are tiled without straddling a boundary and
 * stay dense. Video queries keep every prefix tile and the top
 * (1 - sparsity) video tiles. The learned gate adds a dense tile-mean
 * branch: output = fine + coarse * gate. Sol-Attn is not used. */
#include "h3_kernels.h"
#include "h3_wave32.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <math.h>
#include <stdlib.h>
#include <string.h>

namespace {

__device__ float vsa_bf16(uint16_t bits) {
    return __uint_as_float((uint32_t)bits << 16);
}

__device__ uint16_t vsa_f32(float value) {
    __hip_bfloat16 bf = __float2bfloat16(value);
    return *reinterpret_cast<uint16_t *>(&bf);
}

uint32_t vsa_grid(uint64_t n) {
    uint64_t blocks = (n + 255ull) / 256ull;
    if (blocks < 1ull) blocks = 1ull;
    if (blocks > 2048ull) blocks = 2048ull;
    return (uint32_t)blocks;
}

__device__ float vsa_warp_sum(float value) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        value += h3_shfl_xor32(value, off);
    return value;
}

__device__ float vsa_warp_max(float value) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        value = fmaxf(value, h3_shfl_xor32(value, off));
    return value;
}

__device__ void vsa_load4(const uint16_t *p, float &a, float &b, float &c,
                          float &d) {
    uint2 raw = *reinterpret_cast<const uint2 *>(p);
    a = vsa_bf16(static_cast<uint16_t>(raw.x));
    b = vsa_bf16(static_cast<uint16_t>(raw.x >> 16));
    c = vsa_bf16(static_cast<uint16_t>(raw.y));
    d = vsa_bf16(static_cast<uint16_t>(raw.y >> 16));
}

__device__ void vsa_store4(uint16_t *p, float a, float b, float c, float d) {
    uint2 raw;
    raw.x = (uint32_t)vsa_f32(a) | ((uint32_t)vsa_f32(b) << 16);
    raw.y = (uint32_t)vsa_f32(c) | ((uint32_t)vsa_f32(d) << 16);
    *reinterpret_cast<uint2 *>(p) = raw;
}

__global__ void h3_vsa_scatter_kernel(const uint16_t *src, uint16_t *dst,
                                      const uint32_t *slot, uint32_t sequence,
                                      uint32_t heads, uint32_t dim,
                                      uint32_t padded, uint32_t kv_head_major) {
    uint64_t n = (uint64_t)sequence * heads * dim;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += stride) {
        uint32_t d = (uint32_t)(idx % dim);
        uint64_t rest = idx / dim;
        uint32_t head = (uint32_t)(rest % heads);
        uint32_t token = (uint32_t)(rest / heads);
        uint32_t dest = slot[token];
        size_t from = kv_head_major
                          ? ((size_t)head * sequence + token) * dim + d
                          : ((size_t)token * heads + head) * dim + d;
        dst[((size_t)head * padded + dest) * dim + d] = src[from];
    }
}

__global__ void h3_vsa_pool_kernel(const uint16_t *tiled, float *pooled,
                                   const uint32_t *block_sizes,
                                   uint32_t n_tiles, uint32_t heads,
                                   uint32_t dim) {
    constexpr uint32_t TILE = 64u;
    const uint32_t lane = threadIdx.x;
    const uint32_t tile = (uint32_t)blockIdx.x;
    const uint32_t head = (uint32_t)blockIdx.y;
    if (lane >= 32u || tile >= n_tiles || head >= heads || dim != 128u) return;
    const uint32_t padded = n_tiles * TILE;
    const uint32_t count = block_sizes[tile];
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        float a, b, c, d;
        vsa_load4(tiled + (((size_t)head * padded + (size_t)tile * TILE + i) *
                               dim +
                           lane * 4u),
                  a, b, c, d);
        s0 += a;
        s1 += b;
        s2 += c;
        s3 += d;
    }
    float inv = count ? 1.0f / (float)count : 0.0f;
    float *out = pooled + ((size_t)head * n_tiles + tile) * dim + lane * 4u;
    out[0] = s0 * inv;
    out[1] = s1 * inv;
    out[2] = s2 * inv;
    out[3] = s3 * inv;
}

__global__ void h3_vsa_score_kernel(const float *query, const float *key,
                                    float *scores, uint32_t n_tiles,
                                    uint32_t heads, uint32_t dim, float scale) {
    const uint32_t q_tile = (uint32_t)blockIdx.x;
    const uint32_t head = (uint32_t)blockIdx.y;
    if (q_tile >= n_tiles || head >= heads || dim != 128u) return;
    const float *q = query + ((size_t)head * n_tiles + q_tile) * dim;
    for (uint32_t k_tile = threadIdx.x; k_tile < n_tiles; k_tile += blockDim.x) {
        const float *k = key + ((size_t)head * n_tiles + k_tile) * dim;
        float sum = 0.0f;
        for (uint32_t d = 0; d < dim; d += 4u) {
            float4 qv = *reinterpret_cast<const float4 *>(q + d);
            float4 kv = *reinterpret_cast<const float4 *>(k + d);
            sum = fmaf(qv.x, kv.x, sum);
            sum = fmaf(qv.y, kv.y, sum);
            sum = fmaf(qv.z, kv.z, sum);
            sum = fmaf(qv.w, kv.w, sum);
        }
        scores[((size_t)head * n_tiles + q_tile) * n_tiles + k_tile] =
            sum * scale;
    }
}

__global__ void h3_vsa_mask_kernel(const float *scores, uint8_t *mask,
                                   uint32_t n_tiles, uint32_t n_prefix,
                                   uint32_t keep, uint32_t heads) {
    uint32_t q_tile = (uint32_t)blockIdx.x;
    uint32_t head = (uint32_t)blockIdx.y;
    if (q_tile >= n_tiles || head >= heads || threadIdx.x != 0u) return;
    uint8_t *row = mask + ((size_t)head * n_tiles + q_tile) * n_tiles;
    const float *score = scores + ((size_t)head * n_tiles + q_tile) * n_tiles;
    if (q_tile < n_prefix || keep >= n_tiles - n_prefix) {
        for (uint32_t col = 0; col < n_tiles; col++) row[col] = 1u;
        return;
    }
    for (uint32_t col = 0; col < n_prefix; col++) row[col] = 1u;
    for (uint32_t col = n_prefix; col < n_tiles; col++) row[col] = 0u;
    uint32_t video = n_tiles - n_prefix;
    uint32_t k = keep < video ? keep : video;
    for (uint32_t chosen = 0; chosen < k; chosen++) {
        float best = -INFINITY;
        uint32_t best_col = n_prefix;
        for (uint32_t col = n_prefix; col < n_tiles; col++) {
            if (row[col]) continue;
            float value = score[col];
            if (value > best || (value == best && col < best_col)) {
                best = value;
                best_col = col;
            }
        }
        row[best_col] = 1u;
    }
}

__global__ void h3_vsa_coarse_kernel(const float *scores,
                                     const float *value_pool, float *coarse,
                                     uint32_t n_tiles, uint32_t heads,
                                     uint32_t dim) {
    const uint32_t lane = threadIdx.x;
    const uint32_t q_tile = (uint32_t)blockIdx.x;
    const uint32_t head = (uint32_t)blockIdx.y;
    if (lane >= 32u || q_tile >= n_tiles || head >= heads || dim != 128u)
        return;
    const float *srow = scores + ((size_t)head * n_tiles + q_tile) * n_tiles;
    float max_score = -INFINITY;
    for (uint32_t kv = lane; kv < n_tiles; kv += 32u)
        max_score = fmaxf(max_score, srow[kv]);
    max_score = vsa_warp_max(max_score);
    float denom = 0.0f;
    for (uint32_t kv = lane; kv < n_tiles; kv += 32u)
        denom += __expf(srow[kv] - max_score);
    denom = vsa_warp_sum(denom);
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    if (denom > 0.0f) {
        for (uint32_t kv = 0; kv < n_tiles; kv++) {
            float prob = __expf(srow[kv] - max_score) / denom;
            float4 value = *reinterpret_cast<const float4 *>(
                value_pool + ((size_t)head * n_tiles + kv) * dim + lane * 4u);
            a0 = fmaf(prob, value.x, a0);
            a1 = fmaf(prob, value.y, a1);
            a2 = fmaf(prob, value.z, a2);
            a3 = fmaf(prob, value.w, a3);
        }
    }
    float *out = coarse + ((size_t)head * n_tiles + q_tile) * dim + lane * 4u;
    out[0] = a0;
    out[1] = a1;
    out[2] = a2;
    out[3] = a3;
}

/* CDNA tile kernel. One 256-thread block owns one query tile (64 rows) of
 * one head and walks only the selected 64-token KV tiles. The QK/PV
 * fragment order matches the dense gfx942 flash kernel. */
__global__ void __launch_bounds__(256)
h3_vsa_fine_mfma_kernel(const uint16_t *query, const uint16_t *key,
                        const uint16_t *value, const uint16_t *gate,
                        const float *coarse, const uint8_t *mask,
                        const uint32_t *block_sizes, uint16_t *output,
                        uint32_t n_tiles, float scale) {
    constexpr uint32_t D = 128u;
    constexpr uint32_t LD = 136u;
    constexpr uint32_t TILE = 64u;
    constexpr uint32_t BK = 32u;
    constexpr uint32_t WAVES = 4u;
    constexpr uint32_t QROWS = 16u;
    constexpr uint32_t THREADS = WAVES * 64u;
    constexpr uint32_t NT = BK / 16u;
    constexpr uint32_t VEC = 4u;
    constexpr uint32_t CHUNKS = BK * D / VEC;
    constexpr uint32_t RPT = CHUNKS / THREADS;
    constexpr uint32_t SP_LD = BK + 8u;
    constexpr uint32_t SEL_CAP = 2048u;
    typedef rocwmma::bfloat16_t bf16;

    __shared__ __align__(16) uint16_t sk[BK * LD];
    __shared__ __align__(16) uint16_t sv[BK * LD];
    __shared__ __align__(16) uint16_t sp[WAVES][QROWS * SP_LD];
    __shared__ uint16_t sel[SEL_CAP];
    __shared__ uint32_t sel_n;

    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 6;
    const uint32_t lane = tid & 63u;
    const uint32_t col = lane & 15u;
    const uint32_t rowg = lane >> 4;
    const uint32_t q_tile = (uint32_t)blockIdx.x;
    const uint32_t head = (uint32_t)blockIdx.y;
    if (q_tile >= n_tiles || head >= 56u || n_tiles > SEL_CAP) return;
    const uint32_t q_count = block_sizes[q_tile];
    if (!q_count) return;
    const uint32_t padded = n_tiles * TILE;
    const uint32_t qbase = wave * QROWS;
    const uint16_t *qp =
        query + ((size_t)head * padded + (size_t)q_tile * TILE + qbase) * D;

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bf16, rocwmma::row_major>
        fq[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        rocwmma::load_matrix_sync(fq[i],
                                  reinterpret_cast<const bf16 *>(qp + (uint32_t)i * 16u),
                                  D);
    }
    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fo[8];
#pragma unroll
    for (int j = 0; j < 8; j++) rocwmma::fill_fragment(fo[j], 0.0f);
    float row_max[4], row_sum[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        row_max[i] = -1e30f;
        row_sum[i] = 0.0f;
    }

    const uint8_t *row = mask + ((size_t)head * n_tiles + q_tile) * n_tiles;
    if (tid == 0u) {
        uint32_t n = 0;
        for (uint32_t kv = 0; kv < n_tiles; kv++) {
            if (row[kv] && block_sizes[kv]) sel[n++] = (uint16_t)kv;
        }
        sel_n = n;
    }
    __syncthreads();

    uint2 kreg[RPT], vreg[RPT];
    auto load_regs = [&](uint32_t kv, uint32_t row0, uint32_t nb) {
        const uint16_t *kbase =
            key + ((size_t)head * padded + (size_t)kv * TILE) * D;
        const uint16_t *vbase =
            value + ((size_t)head * padded + (size_t)kv * TILE) * D;
#pragma unroll
        for (uint32_t u = 0; u < RPT; u++) {
            uint32_t c = tid + u * THREADS;
            uint32_t r = row0 + ((c * VEC) >> 7);
            uint32_t d = (c * VEC) & 127u;
            kreg[u] = uint2{0u, 0u};
            vreg[u] = uint2{0u, 0u};
            if (r < nb) {
                kreg[u] = *reinterpret_cast<const uint2 *>(kbase + r * D + d);
                vreg[u] = *reinterpret_cast<const uint2 *>(vbase + r * D + d);
            }
        }
    };
    auto store_lds = [&](void) {
#pragma unroll
        for (uint32_t u = 0; u < RPT; u++) {
            uint32_t c = tid + u * THREADS;
            uint32_t r = (c * VEC) >> 7;
            uint32_t d = (c * VEC) & 127u;
            *reinterpret_cast<uint2 *>(sk + r * LD + d) = kreg[u];
            *reinterpret_cast<uint2 *>(sv + r * LD + d) = vreg[u];
        }
    };
    auto attend = [&](uint32_t row0, uint32_t nb) {
        rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fs[NT];
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            rocwmma::fill_fragment(fs[t], 0.0f);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bf16,
                                  rocwmma::col_major>
                    fk;
                rocwmma::load_matrix_sync(
                    fk, reinterpret_cast<const bf16 *>(sk + t * 16u * LD +
                                                       (uint32_t)i * 16u),
                    LD);
                rocwmma::mma_sync(fs[t], fq[i], fk, fs[t]);
            }
        }
        float alpha[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float s[NT];
            float m = -1e30f;
#pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                uint32_t kcol = row0 + t * 16u + col;
                s[t] = kcol < nb ? fs[t].x[i] * scale : -1e30f;
                m = fmaxf(m, s[t]);
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
                m = fmaxf(m, __shfl_xor(m, off, 16));
            float new_max = fmaxf(row_max[i], m);
            alpha[i] = __expf(row_max[i] - new_max);
            uint32_t prow = rowg * 4u + (uint32_t)i;
            float ps = 0.0f;
#pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                float p = __expf(s[t] - new_max);
                ps += p;
                sp[wave][prow * SP_LD + t * 16u + col] = vsa_f32(p);
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
                ps += __shfl_xor(ps, off, 16);
            row_sum[i] = fmaf(row_sum[i], alpha[i], ps);
            row_max[i] = new_max;
        }
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int i = 0; i < 4; i++) fo[j].x[i] *= alpha[i];
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bf16,
                              rocwmma::row_major>
                fp;
            rocwmma::load_matrix_sync(
                fp, reinterpret_cast<const bf16 *>(sp[wave] + t * 16u), SP_LD);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bf16,
                                  rocwmma::row_major>
                    fv;
                rocwmma::load_matrix_sync(
                    fv, reinterpret_cast<const bf16 *>(sv + t * 16u * LD +
                                                       (uint32_t)j * 16u),
                    LD);
                rocwmma::mma_sync(fo[j], fp, fv, fo[j]);
            }
        }
    };

    if (sel_n) {
        uint32_t kv0 = sel[0];
        uint32_t nb0 = block_sizes[kv0];
        load_regs(kv0, 0u, nb0);
        store_lds();
        __syncthreads();
        for (uint32_t si = 0; si < sel_n; si++) {
            uint32_t kv = sel[si];
            uint32_t nb = block_sizes[kv];
            uint32_t chunks = nb > BK ? 2u : 1u;
            for (uint32_t ch = 0; ch < chunks; ch++) {
                uint32_t row0 = ch * BK;
                uint32_t nkv = 0, nrow = 0, nnb = 0;
                int has_next = 0;
                if (ch + 1u < chunks) {
                    has_next = 1;
                    nkv = kv;
                    nrow = BK;
                    nnb = nb;
                } else if (si + 1u < sel_n) {
                    has_next = 1;
                    nkv = sel[si + 1u];
                    nrow = 0u;
                    nnb = block_sizes[nkv];
                }
                if (has_next) load_regs(nkv, nrow, nnb);
                attend(row0, nb);
                if (has_next) {
                    __syncthreads();
                    store_lds();
                    __syncthreads();
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
        float inv = row_sum[i] > 0.0f ? 1.0f / row_sum[i] : 0.0f;
        uint32_t q_local = qbase + rowg * 4u + (uint32_t)i;
        if (q_local >= q_count) continue;
        const uint16_t *grow =
            gate + ((size_t)head * padded + (size_t)q_tile * TILE + q_local) * D;
        uint16_t *orow =
            output + ((size_t)head * padded + (size_t)q_tile * TILE + q_local) * D;
        const float *crow =
            coarse + ((size_t)head * n_tiles + q_tile) * D;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            uint32_t d = (uint32_t)j * 16u + col;
            float mixed = fmaf(crow[d], vsa_bf16(grow[d]), fo[j].x[i] * inv);
            orow[d] = vsa_f32(mixed);
        }
    }
}

__global__ void h3_vsa_fine_kernel(const uint16_t *query, const uint16_t *key,
                                   const uint16_t *value, const uint16_t *gate,
                                   const float *coarse, const uint8_t *mask,
                                   const uint32_t *block_sizes, uint16_t *output,
                                   float *fine_acc, uint32_t n_tiles,
                                   uint32_t dim, float scale) {
    constexpr uint32_t TILE = 64u;
    const uint32_t lane = threadIdx.x;
    const uint32_t q_slot = (uint32_t)blockIdx.x;
    const uint32_t head = (uint32_t)blockIdx.y;
    const uint32_t q_tile = q_slot / TILE;
    const uint32_t q_local = q_slot - q_tile * TILE;
    (void)fine_acc;
    if (lane >= 32u || dim != 128u || q_tile >= n_tiles || head >= 56u) return;
    if (q_local >= block_sizes[q_tile]) return;
    const uint32_t padded = n_tiles * TILE;
    float q0, q1, q2, q3;
    vsa_load4(query + ((size_t)head * padded + q_slot) * dim + lane * 4u, q0, q1,
              q2, q3);
    const uint8_t *row = mask + ((size_t)head * n_tiles + q_tile) * n_tiles;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (uint32_t kv = 0; kv < n_tiles; kv++) {
        if (!row[kv]) continue;
        uint32_t nb = block_sizes[kv];
        const uint16_t *kbase =
            key + ((size_t)head * padded + (size_t)kv * TILE) * dim;
        const uint16_t *vbase =
            value + ((size_t)head * padded + (size_t)kv * TILE) * dim;
        for (uint32_t k = 0; k < nb; k++) {
            float k0, k1, k2, k3;
            vsa_load4(kbase + (size_t)k * dim + lane * 4u, k0, k1, k2, k3);
            float score = vsa_warp_sum(q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3) *
                          scale;
            if (!(score > -INFINITY)) continue;
            float next_max = fmaxf(running_max, score);
            float alpha = running_max > -INFINITY ? __expf(running_max - next_max)
                                                  : 0.0f;
            a0 *= alpha;
            a1 *= alpha;
            a2 *= alpha;
            a3 *= alpha;
            running_sum *= alpha;
            float prob = __expf(score - next_max);
            running_sum += prob;
            float v0, v1, v2, v3;
            vsa_load4(vbase + (size_t)k * dim + lane * 4u, v0, v1, v2, v3);
            a0 = fmaf(prob, v0, a0);
            a1 = fmaf(prob, v1, a1);
            a2 = fmaf(prob, v2, a2);
            a3 = fmaf(prob, v3, a3);
            running_max = next_max;
        }
    }
    float inv = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    float4 crow = *reinterpret_cast<const float4 *>(
        coarse + ((size_t)head * n_tiles + q_tile) * dim + lane * 4u);
    float g0, g1, g2, g3;
    vsa_load4(gate + ((size_t)head * padded + q_slot) * dim + lane * 4u, g0, g1,
              g2, g3);
    vsa_store4(output + ((size_t)head * padded + q_slot) * dim + lane * 4u,
               fmaf(crow.x, g0, a0 * inv), fmaf(crow.y, g1, a1 * inv),
               fmaf(crow.z, g2, a2 * inv), fmaf(crow.w, g3, a3 * inv));
}

__global__ void h3_vsa_gather_kernel(const uint16_t *tiled, uint16_t *output,
                                     const uint32_t *slot, uint32_t sequence,
                                     uint32_t heads, uint32_t dim,
                                     uint32_t padded) {
    uint64_t n = (uint64_t)sequence * heads * dim;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += stride) {
        uint32_t d = (uint32_t)(idx % dim);
        uint64_t rest = idx / dim;
        uint32_t head = (uint32_t)(rest % heads);
        uint32_t token = (uint32_t)(rest / heads);
        uint32_t src = slot[token];
        output[((size_t)token * heads + head) * dim + d] =
            tiled[((size_t)head * padded + src) * dim + d];
    }
}

}  // namespace

extern "C" int h3_launch_vsa_bf16(
    const uint16_t *query, const uint16_t *key, const uint16_t *value,
    const uint16_t *gate, uint16_t *output, const uint32_t *slot,
    const uint32_t *block_sizes, uint16_t *tiled_q, uint16_t *tiled_k,
    uint16_t *tiled_v, uint16_t *tiled_g, uint16_t *tiled_o, float *pool_q,
    float *pool_k, float *pool_v, float *scores, float *fine_acc, uint8_t *mask,
    uint32_t sequence, uint32_t heads, uint32_t dim, uint32_t padded,
    uint32_t n_tiles, uint32_t n_prefix, uint32_t keep, float scale,
    uint32_t kv_head_major, hipStream_t stream) {
    if (!sequence || !heads || dim != 128u || !n_tiles || padded != n_tiles * 64u)
        return 0;
    size_t tiled_bytes = (size_t)heads * padded * dim * sizeof(uint16_t);
    if (hipMemsetAsync(tiled_q, 0, tiled_bytes, stream) != hipSuccess ||
        hipMemsetAsync(tiled_k, 0, tiled_bytes, stream) != hipSuccess ||
        hipMemsetAsync(tiled_v, 0, tiled_bytes, stream) != hipSuccess ||
        hipMemsetAsync(tiled_g, 0, tiled_bytes, stream) != hipSuccess)
        return 0;
    auto scatter = [&](const uint16_t *src, uint16_t *dst, uint32_t hm) {
        uint64_t n = (uint64_t)sequence * heads * dim;
        void *args[] = {&src, &dst, &slot, &sequence, &heads, &dim, &padded, &hm};
        return hipLaunchKernel(reinterpret_cast<const void *>(h3_vsa_scatter_kernel),
                               dim3(vsa_grid(n)), dim3(256), args, 0,
                               stream) == hipSuccess;
    };
    uint32_t q_hm = 0u;
    if (!scatter(query, tiled_q, q_hm) || !scatter(key, tiled_k, kv_head_major) ||
        !scatter(value, tiled_v, kv_head_major) || !scatter(gate, tiled_g, q_hm))
        return 0;
    auto pool = [&](const uint16_t *tiled, float *dest) {
        void *args[] = {&tiled, &dest, &block_sizes, &n_tiles, &heads, &dim};
        return hipLaunchKernel(reinterpret_cast<const void *>(h3_vsa_pool_kernel),
                               dim3(n_tiles, heads), dim3(32), args, 0,
                               stream) == hipSuccess;
    };
    if (!pool(tiled_q, pool_q) || !pool(tiled_k, pool_k) || !pool(tiled_v, pool_v))
        return 0;
    {
        void *args[] = {&pool_q, &pool_k, &scores, &n_tiles, &heads, &dim, &scale};
        if (hipLaunchKernel(reinterpret_cast<const void *>(h3_vsa_score_kernel),
                            dim3(n_tiles, heads), dim3(128), args, 0,
                            stream) != hipSuccess)
            return 0;
    }
    {
        void *args[] = {&scores, &mask, &n_tiles, &n_prefix, &keep, &heads};
        if (hipLaunchKernel(reinterpret_cast<const void *>(h3_vsa_mask_kernel),
                            dim3(n_tiles, heads), dim3(1), args, 0,
                            stream) != hipSuccess)
            return 0;
    }
    {
        void *args[] = {&scores, &pool_v, &pool_q, &n_tiles, &heads, &dim};
        if (hipLaunchKernel(reinterpret_cast<const void *>(h3_vsa_coarse_kernel),
                            dim3(n_tiles, heads), dim3(32), args, 0,
                            stream) != hipSuccess)
            return 0;
    }
    int mfma = 0;
    {
        const char *env = getenv("H3_VSA_MFMA");
        if (!(env && strcmp(env, "0") == 0)) {
            int device = 0;
            hipDeviceProp_t props;
            mfma = n_tiles <= 2048u &&
                   hipGetDevice(&device) == hipSuccess &&
                   hipGetDeviceProperties(&props, device) == hipSuccess &&
                   (strncmp(props.gcnArchName, "gfx90a", 6) == 0 ||
                    strncmp(props.gcnArchName, "gfx942", 6) == 0);
        }
    }
    if (mfma) {
        void *args[] = {&tiled_q, &tiled_k, &tiled_v, &tiled_g, &pool_q, &mask,
                        &block_sizes, &tiled_o, &n_tiles, &scale};
        if (hipLaunchKernel(
                reinterpret_cast<const void *>(h3_vsa_fine_mfma_kernel),
                dim3(n_tiles, heads), dim3(256), args, 0, stream) != hipSuccess)
            return 0;
    } else {
        void *args[] = {&tiled_q, &tiled_k, &tiled_v, &tiled_g, &pool_q, &mask,
                        &block_sizes, &tiled_o, &fine_acc, &n_tiles, &dim,
                        &scale};
        if (hipLaunchKernel(reinterpret_cast<const void *>(h3_vsa_fine_kernel),
                            dim3(n_tiles * 64u, heads), dim3(32), args, 0,
                            stream) != hipSuccess)
            return 0;
    }
    {
        uint64_t n = (uint64_t)sequence * heads * dim;
        void *args[] = {&tiled_o, &output, &slot, &sequence, &heads, &dim, &padded};
        if (hipLaunchKernel(reinterpret_cast<const void *>(h3_vsa_gather_kernel),
                            dim3(vsa_grid(n)), dim3(256), args, 0,
                            stream) != hipSuccess)
            return 0;
    }
    return 1;
}
