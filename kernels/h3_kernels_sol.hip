/* Opt-in Sol-Attn: 64-token KV tiles, cheap Q-bar routing, exact MMA on
 * kept tiles, pooled K/V in the online softmax for skipped tiles.
 * Included from h3_kernels.hip so it shares the CDNA flash helpers. */

#define H3_SOL_KV_TILE 64u
#define H3_SOL_MAX_KV_BLOCKS 1024u

__global__ void h3_sdpa_kv_block_summary_kernel(
    const uint16_t *__restrict__ key, const uint16_t *__restrict__ value,
    float *__restrict__ k_mean, float *__restrict__ v_sum, uint32_t sequence,
    uint32_t heads, uint32_t n_blocks, uint32_t kv_head_major) {
    const uint32_t block = (uint32_t)blockIdx.x;
    const uint32_t head = (uint32_t)blockIdx.y;
    const uint32_t dim = (uint32_t)threadIdx.x;
    if (block >= n_blocks || head >= heads || dim >= 128u) return;
    const uint32_t n0 = block * H3_SOL_KV_TILE;
    const uint32_t n1 = n0 + H3_SOL_KV_TILE < sequence ? n0 + H3_SOL_KV_TILE
                                                       : sequence;
    const uint32_t kv_stride = kv_head_major ? 128u : heads * 128u;
    const size_t kv_base = kv_head_major ? (size_t)head * sequence * 128u
                                         : (size_t)head * 128u;
    float sum_k = 0.0f;
    float sum_v = 0.0f;
    for (uint32_t token = n0; token < n1; token++) {
        size_t index = kv_base + (size_t)token * kv_stride + dim;
        sum_k += h3_bf16_to_f32(key[index]);
        sum_v += h3_bf16_to_f32(value[index]);
    }
    float count = (float)(n1 - n0);
    size_t out = ((size_t)head * n_blocks + block) * 128u + dim;
    k_mean[out] = count > 0.0f ? sum_k / count : 0.0f;
    v_sum[out] = sum_v;
}

template <uint32_t WAVES, uint32_t BK, bool FP16_PV = false>
__global__ void __launch_bounds__(WAVES * 64u)
h3_sdpa_bf16_mfma_d128_sol_kernel(
    const uint16_t *query, const uint16_t *key, const uint16_t *value,
    uint16_t *output, h3_sdpa_args args, const float *k_mean,
    const float *v_sum, h3_sdpa_sol_cfg cfg) {
    constexpr uint32_t D = 128u;
    constexpr uint32_t LD = H3_SDPA_WMMA_LD;
    constexpr uint32_t QROWS = H3_SDPA_WMMA_QROWS;
    constexpr uint32_t THREADS = WAVES * 64u;
    constexpr uint32_t NT = BK / 16u;
    constexpr uint32_t BQ = WAVES * QROWS;
    typedef rocwmma::bfloat16_t bf16;
    typedef typename std::conditional<FP16_PV, _Float16, bf16>::type pv16;

    __shared__ __align__(16) uint16_t sk[BK * LD];
    __shared__ __align__(16) uint16_t sv[BK * LD];
    __shared__ __align__(16) uint16_t sp[WAVES][QROWS * BK];
    __shared__ float q_bar[128];
    __shared__ float proxy[H3_SOL_MAX_KV_BLOCKS];
    __shared__ uint16_t skipped[H3_SOL_MAX_KV_BLOCKS];
    __shared__ uint32_t skipped_count;

    const uint32_t tid = threadIdx.x;
    const uint32_t wave = tid >> 6;
    const uint32_t lane = tid & 63u;
    const uint32_t col = lane & 15u;
    const uint32_t rowg = lane >> 4;
    const uint32_t head = blockIdx.y;
    const uint32_t sequence = args.sequence;
    const uint32_t n_blocks = cfg.n_blocks;
    uint32_t q_row0 = blockIdx.x * BQ + wave * QROWS;
    uint32_t qbase = q_row0 + QROWS <= sequence ? q_row0 : sequence - QROWS;
    const uint32_t q_block = (blockIdx.x * BQ) / H3_SOL_KV_TILE;

    rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, bf16,
                      rocwmma::row_major> fq[8];
    const uint16_t *qp = query + (size_t)(qbase * args.heads + head) * D;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        rocwmma::load_matrix_sync(
            fq[i], reinterpret_cast<const bf16 *>(qp + (uint32_t)i * 16u),
            args.heads * D);
    }

    if (tid < D) {
        float sum = 0.0f;
        uint32_t live = 0;
        uint32_t q0 = blockIdx.x * BQ;
        for (uint32_t r = 0; r < BQ; r++) {
            uint32_t g = q0 + r;
            if (g < sequence) {
                size_t index = ((size_t)g * args.heads + head) * D + tid;
                sum += h3_bf16_to_f32(query[index]);
                live++;
            }
        }
        q_bar[tid] = live ? sum / (float)live : 0.0f;
    }
    __syncthreads();

    for (uint32_t block = tid; block < n_blocks; block += THREADS) {
        const float *centroid =
            k_mean + ((size_t)head * n_blocks + block) * D;
        float score = 0.0f;
#pragma unroll
        for (uint32_t dim = 0; dim < D; dim++)
            score = fmaf(q_bar[dim], centroid[dim], score);
        proxy[block] = score * args.scale;
    }
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        float sumsq = 0.0f;
        for (uint32_t block = 0; block < n_blocks; block++) {
            float score = proxy[block];
            sum += score;
            sumsq += score * score;
        }
        float mean = sum / (float)n_blocks;
        float var = fmaxf(sumsq / (float)n_blocks - mean * mean, 0.0f);
        float threshold = mean + cfg.tau * sqrtf(var);
        for (uint32_t block = 0; block < n_blocks; block++) {
            int local = (int)block >= (int)q_block - cfg.band &&
                        (int)block <= (int)q_block + cfg.band;
            int exact_prefix =
                (int)block < cfg.prefix || (int)q_block < cfg.prefix;
            proxy[block] =
                (local || exact_prefix || proxy[block] >= threshold) ? 1.0f
                                                                     : 0.0f;
        }
        skipped_count = 0;
        if (!cfg.drop_unselected) {
            for (uint32_t block = 0; block < n_blocks; block++) {
                if (proxy[block] <= 0.5f)
                    skipped[skipped_count++] = (uint16_t)block;
            }
        }
        if (cfg.route_stats) {
            uint64_t kept = 0;
            for (uint32_t block = 0; block < n_blocks; block++)
                kept += proxy[block] > 0.5f ? 1u : 0u;
            atomicAdd(reinterpret_cast<unsigned long long *>(
                          cfg.route_stats), (unsigned long long)kept);
            atomicAdd(reinterpret_cast<unsigned long long *>(
                          cfg.route_stats + 1),
                      (unsigned long long)n_blocks);
        }
    }
    __syncthreads();

    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fo[8];
#pragma unroll
    for (int j = 0; j < 8; j++) rocwmma::fill_fragment(fo[j], 0.0f);
    float row_max[4], row_sum[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        row_max[i] = -1e30f;
        row_sum[i] = 0.0f;
    }

    const uint32_t kv_stride = args.kv_head_major ? D : args.heads * D;
    const size_t kv_base = args.kv_head_major
                               ? (size_t)head * sequence * D
                               : (size_t)head * D;
    const uint16_t *kp = key + kv_base;
    const uint16_t *vp = value + kv_base;

    auto load_kv = [&](uint32_t k0) {
        for (uint32_t idx = tid * 4u; idx < BK * D; idx += THREADS * 4u) {
            uint32_t r = idx >> 7;
            uint32_t d = idx & 127u;
            uint32_t g = k0 + r;
            uint2 kk = uint2{0u, 0u};
            if (g < sequence) {
                size_t off = (size_t)g * kv_stride + d;
                kk = *reinterpret_cast<const uint2 *>(kp + off);
                if constexpr (FP16_PV) {
                    uint32_t *dst =
                        reinterpret_cast<uint32_t *>(sv + r * LD + d);
                    dst[0] = h3_pack_f16x2(h3_bf16_to_f32(vp[off]),
                                           h3_bf16_to_f32(vp[off + 1u]));
                    dst[1] = h3_pack_f16x2(h3_bf16_to_f32(vp[off + 2u]),
                                           h3_bf16_to_f32(vp[off + 3u]));
                } else {
                    *reinterpret_cast<uint2 *>(sv + r * LD + d) =
                        *reinterpret_cast<const uint2 *>(vp + off);
                }
            } else {
                *reinterpret_cast<uint2 *>(sv + r * LD + d) = uint2{0u, 0u};
            }
            *reinterpret_cast<uint2 *>(sk + r * LD + d) = kk;
        }
    };

    for (uint32_t k0 = 0; k0 < sequence; k0 += BK) {
        uint32_t kv_block = k0 / H3_SOL_KV_TILE;
        int keep = kv_block < n_blocks && proxy[kv_block] > 0.5f;
        if (!keep) {
            __syncthreads();
            /* BK=32 visits each 64-token routing tile twice. Skip the empty
             * second visit; pooled contributions are handled in MFMA batches
             * after all exact tiles. */
            if constexpr (BK < H3_SOL_KV_TILE)
                if ((k0 % H3_SOL_KV_TILE) == 0u)
                    k0 += H3_SOL_KV_TILE - BK;
            continue;
        }

        __syncthreads();
        load_kv(k0);
        __syncthreads();

        rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fs[NT];
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            rocwmma::fill_fragment(fs[t], 0.0f);
#pragma unroll
            for (int i = 0; i < 8; i++) {
                rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bf16,
                                  rocwmma::col_major> fk;
                rocwmma::load_matrix_sync(
                    fk, reinterpret_cast<const bf16 *>(
                            sk + t * 16u * LD + (uint32_t)i * 16u),
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
                s[t] = k0 + t * 16u + col < sequence
                           ? fs[t].x[i] * args.scale
                           : -1e30f;
                m = fmaxf(m, s[t]);
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
                m = fmaxf(m, __shfl_xor(m, off, 16));
            float new_max = fmaxf(row_max[i], m);
            alpha[i] = h3_fast_exp(row_max[i] - new_max);
            uint32_t prow = rowg * 4u + (uint32_t)i;
            float ps = 0.0f;
#pragma unroll
            for (uint32_t t = 0; t < NT; t++) {
                float p = h3_fast_exp(s[t] - new_max);
                ps += p;
                uint16_t *pdst = sp[wave] + prow * BK + t * 16u + col;
                if constexpr (FP16_PV)
                    *reinterpret_cast<_Float16 *>(pdst) = (_Float16)p;
                else
                    *pdst = h3_f32_to_bf16(p);
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
            rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, pv16,
                              rocwmma::row_major> fp;
            rocwmma::load_matrix_sync(
                fp, reinterpret_cast<const pv16 *>(sp[wave] + t * 16u), BK);
#pragma unroll
            for (int j = 0; j < 8; j++) {
                rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, pv16,
                                  rocwmma::row_major> fv;
                rocwmma::load_matrix_sync(
                    fv, reinterpret_cast<const pv16 *>(
                            sv + t * 16u * LD + (uint32_t)j * 16u),
                    LD);
                rocwmma::mma_sync(fo[j], fp, fv, fo[j]);
            }
        }
        __syncthreads();
    }

    /* Treat 16 skipped 64-token tiles as 16 pooled pseudo-KV tokens. QK and
     * PV then use the same MFMA fragments as exact attention. Multiplicity
     * enters only the softmax denominator; V summaries are already sums. */
    for (uint32_t sb0 = 0; sb0 < skipped_count; sb0 += 16u) {
        uint32_t batch = skipped_count - sb0;
        if (batch > 16u) batch = 16u;
        __syncthreads();
        for (uint32_t idx = tid; idx < 16u * D; idx += THREADS) {
            uint32_t r = idx >> 7;
            uint32_t d = idx & 127u;
            float km = 0.0f;
            float vs = 0.0f;
            if (r < batch) {
                uint32_t block = (uint32_t)skipped[sb0 + r];
                size_t off = ((size_t)head * n_blocks + block) * D + d;
                km = k_mean[off];
                vs = v_sum[off];
            }
            sk[r * LD + d] = h3_f32_to_bf16(km);
            if constexpr (FP16_PV)
                *reinterpret_cast<_Float16 *>(sv + r * LD + d) =
                    (_Float16)vs;
            else
                sv[r * LD + d] = h3_f32_to_bf16(vs);
        }
        __syncthreads();

        rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fs;
        rocwmma::fill_fragment(fs, 0.0f);
#pragma unroll
        for (int slice = 0; slice < 8; slice++) {
            rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, bf16,
                              rocwmma::col_major> fk;
            rocwmma::load_matrix_sync(
                fk, reinterpret_cast<const bf16 *>(
                        sk + (uint32_t)slice * 16u),
                LD);
            rocwmma::mma_sync(fs, fq[slice], fk, fs);
        }

        float alpha[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float score = col < batch ? fs.x[i] * args.scale : -1e30f;
            float tile_max = score;
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
                tile_max = fmaxf(tile_max, __shfl_xor(tile_max, off, 16));
            float new_max = fmaxf(row_max[i], tile_max);
            alpha[i] = h3_fast_exp(row_max[i] - new_max);
            float p = col < batch ? h3_fast_exp(score - new_max) : 0.0f;
            uint32_t live = 0;
            if (col < batch) {
                uint32_t block = (uint32_t)skipped[sb0 + col];
                uint32_t first = block * H3_SOL_KV_TILE;
                live = sequence - first;
                if (live > H3_SOL_KV_TILE) live = H3_SOL_KV_TILE;
            }
            float ps = p * (float)live;
#pragma unroll
            for (int off = 1; off < 16; off <<= 1)
                ps += __shfl_xor(ps, off, 16);
            row_sum[i] = fmaf(row_sum[i], alpha[i], ps);
            row_max[i] = new_max;
            uint32_t prow = rowg * 4u + (uint32_t)i;
            uint16_t *pdst = sp[wave] + prow * BK + col;
            if constexpr (FP16_PV)
                *reinterpret_cast<_Float16 *>(pdst) = (_Float16)p;
            else
                *pdst = h3_f32_to_bf16(p);
        }
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int i = 0; i < 4; i++) fo[j].x[i] *= alpha[i];

        __builtin_amdgcn_wave_barrier();
        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, pv16,
                          rocwmma::row_major> fp;
        rocwmma::load_matrix_sync(
            fp, reinterpret_cast<const pv16 *>(sp[wave]), BK);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, pv16,
                              rocwmma::row_major> fv;
            rocwmma::load_matrix_sync(
                fv, reinterpret_cast<const pv16 *>(
                        sv + (uint32_t)j * 16u),
                LD);
            rocwmma::mma_sync(fo[j], fp, fv, fo[j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
        float inv = row_sum[i] > 0.0f ? 1.0f / row_sum[i] : 0.0f;
        uint32_t q_pos = qbase + rowg * 4u + (uint32_t)i;
        if (q_pos < q_row0) continue;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            uint32_t d = (uint32_t)j * 16u + col;
            output[h3_sdpa_output_index(args, q_pos, head, d)] =
                h3_f32_to_bf16(fo[j].x[i] * inv);
        }
    }
}

extern "C" int h3_launch_sdpa_bf16_sol(const uint16_t *query,
                                       const uint16_t *key,
                                       const uint16_t *value,
                                       uint16_t *output,
                                       const h3_sdpa_args *args,
                                       float *k_mean, float *v_sum,
                                       const h3_sdpa_sol_cfg *cfg,
                                       hipStream_t stream) {
    if (!args || !cfg || !args->sequence || !args->heads ||
        args->head_dim != 128 || !cfg->n_blocks ||
        cfg->n_blocks > H3_SOL_MAX_KV_BLOCKS) {
        return 0;
    }
    if (!h3_hip_gfx90a())
        return h3_launch_sdpa_bf16(query, key, value, output, args, stream);
    h3_sdpa_args host_args = *args;
    h3_sdpa_sol_cfg host_cfg = *cfg;
    uint32_t sequence = host_args.sequence;
    uint32_t heads = host_args.heads;
    uint32_t n_blocks = host_cfg.n_blocks;
    uint32_t kv_hm = host_args.kv_head_major;
    void *summary_args[] = {&key, &value, &k_mean, &v_sum, &sequence, &heads,
                            &n_blocks, &kv_hm};
    if (hipLaunchKernel(reinterpret_cast<const void *>(
                            h3_sdpa_kv_block_summary_kernel),
                        dim3(n_blocks, heads, 1), dim3(128u), summary_args, 0,
                        stream) != hipSuccess) {
        return 0;
    }
    uint32_t waves = h3_env_u32("H3_SDPA_CDNA_WAVES", 8u);
    uint32_t bk = h3_env_u32("H3_SDPA_CDNA_BK", 32u);
    const char *fp16_pv_env = getenv("H3_SDPA_CDNA_FP16_PV");
    int fp16_pv = !fp16_pv_env || strcmp(fp16_pv_env, "0") != 0;
    const void *kernel = NULL;
#define H3_SDPA_SOL_PICK(w, b)                                               \
    if (waves == (w) && bk == (b)) {                                         \
        kernel = fp16_pv                                                     \
            ? reinterpret_cast<const void *>(                                \
                  h3_sdpa_bf16_mfma_d128_sol_kernel<w, b, true>)             \
            : reinterpret_cast<const void *>(                                \
                  h3_sdpa_bf16_mfma_d128_sol_kernel<w, b, false>);           \
    }
    H3_SDPA_SOL_PICK(4u, 32u)
    H3_SDPA_SOL_PICK(4u, 64u)
    H3_SDPA_SOL_PICK(8u, 32u)
    H3_SDPA_SOL_PICK(8u, 64u)
#undef H3_SDPA_SOL_PICK
    if (!kernel) {
        waves = 8u;
        bk = 32u;
        kernel = fp16_pv
            ? reinterpret_cast<const void *>(
                  h3_sdpa_bf16_mfma_d128_sol_kernel<8u, 32u, true>)
            : reinterpret_cast<const void *>(
                  h3_sdpa_bf16_mfma_d128_sol_kernel<8u, 32u, false>);
    }
    uint32_t bq = waves * H3_SDPA_WMMA_QROWS;
    dim3 grid((host_args.sequence + bq - 1u) / bq, host_args.heads, 1);
    void *kernel_args[] = {&query, &key, &value, &output, &host_args, &k_mean,
                           &v_sum, &host_cfg};
    return hipLaunchKernel(kernel, grid, dim3(waves * 64u), kernel_args,
                           sizeof(kernel_args), stream) == hipSuccess;
}
