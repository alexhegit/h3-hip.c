#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* f32-in / f32-out GEMM via half-precision WMMA, with an optional
 * register-prefetch double buffer. C[M,N] = A[M,K] * B[N,K]^T. */

__device__ inline uint32_t bf16_u16(float x) {
    union { float f; uint32_t u; } v;
    v.f = x;
    return (v.u + 0x7fffu + ((v.u >> 16) & 1u)) >> 16;
}

__device__ inline uint32_t f16_u16(float x) {
    _Float16 h = (_Float16)x;
    union { _Float16 h; uint16_t u; } v;
    v.h = h;
    return v.u;
}

template <typename T> struct conv;
template <> struct conv<rocwmma::bfloat16_t> {
    __device__ static uint32_t pack(float a, float b) {
        return bf16_u16(a) | (bf16_u16(b) << 16);
    }
};
template <> struct conv<_Float16> {
    __device__ static uint32_t pack(float a, float b) {
        return f16_u16(a) | (f16_u16(b) << 16);
    }
};

/* ---------------------------- baseline r128 --------------------------- */

__global__ void __launch_bounds__(256)
r128(const float *A, const float *B, float *C, uint32_t M, uint32_t K,
     uint32_t N) {
    constexpr int BM = 128, BN = 128, BK = 32, TM = 8, TN = 8;
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BN][BK + 1];
    uint32_t block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
    uint32_t tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16u + tx;
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;
#pragma unroll 1
    for (uint32_t kt = 0; kt < K / BK; kt++) {
        uint32_t k0 = kt * BK;
#pragma unroll
        for (int i = 0; i < BM * BK / 256; i++) {
            int linear = (int)tid + i * 256;
            int r = linear / BK, k = linear % BK;
            uint32_t row = block_row + r;
            As[r][k] = row < M ? A[(size_t)row * K + k0 + k] : 0.0f;
            row = block_col + r;
            Bs[r][k] = row < N ? B[(size_t)row * K + k0 + k] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; k++) {
            float av[TM], bv[TN];
#pragma unroll
            for (int i = 0; i < TM; i++) av[i] = As[ty * TM + i][k];
#pragma unroll
            for (int j = 0; j < TN; j++) bv[j] = Bs[tx * TN + j][k];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; i++) {
        uint32_t row = block_row + ty * TM + i;
        if (row >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; j++) {
            uint32_t col = block_col + tx * TN + j;
            if (col < N) C[(size_t)row * N + col] = acc[i][j];
        }
    }
}

/* ------------------------------ WMMA GEMM ----------------------------- */

template <typename H, int BM, int BN, int BK, int WM, int WN, int PAD,
          int DBUF>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32)
wmma_gemm(const float *A, const float *B, float *C, uint32_t M, uint32_t K,
          uint32_t N) {
    constexpr int LD = BK + PAD;
    constexpr int WAVES_N = BN / WN;
    constexpr int THREADS = (BM / WM) * WAVES_N * 32;
    constexpr int FM = WM / 16, FN = WN / 16;
    constexpr int KQ = BK / 4;
    constexpr int AI = BM * KQ / THREADS;
    constexpr int BI = BN * KQ / THREADS;

    __shared__ uint16_t As[BM * LD];
    __shared__ uint16_t Bs[BN * LD];

    uint32_t tid = threadIdx.x;
    uint32_t wave = tid >> 5;
    uint32_t wm = wave / WAVES_N, wn = wave % WAVES_N;
    uint32_t block_row = blockIdx.y * (uint32_t)BM;
    uint32_t block_col = blockIdx.x * (uint32_t)BN;

    rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) rocwmma::fill_fragment(acc[i][j], 0.0f);

    /* Index arithmetic is recomputed per use: holding it in arrays costs
     * enough VGPRs to drop a wave per SIMD. */
    const int row0 = (int)tid / KQ, kq0 = ((int)tid % KQ) * 4;
    constexpr int ROW_STEP = THREADS / KQ;

    const H *pA = reinterpret_cast<const H *>(As);
    const H *pB = reinterpret_cast<const H *>(Bs);
    uint32_t ktiles = K / (uint32_t)BK;

    float4 a_next[AI], b_next[BI];
    auto fetch = [&](uint32_t k0) {
#pragma unroll
        for (int it = 0; it < AI; it++) {
            uint32_t row = block_row + (uint32_t)(row0 + it * ROW_STEP);
            a_next[it] = row < M ? *reinterpret_cast<const float4 *>(
                                       A + (size_t)row * K + k0 + kq0)
                                 : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int it = 0; it < BI; it++) {
            uint32_t row = block_col + (uint32_t)(row0 + it * ROW_STEP);
            b_next[it] = row < N ? *reinterpret_cast<const float4 *>(
                                       B + (size_t)row * K + k0 + kq0)
                                 : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    };
    auto stash = [&]() {
#pragma unroll
        for (int it = 0; it < AI; it++) {
            uint32_t *d = reinterpret_cast<uint32_t *>(
                As + (row0 + it * ROW_STEP) * LD + kq0);
            d[0] = conv<H>::pack(a_next[it].x, a_next[it].y);
            d[1] = conv<H>::pack(a_next[it].z, a_next[it].w);
        }
#pragma unroll
        for (int it = 0; it < BI; it++) {
            uint32_t *d = reinterpret_cast<uint32_t *>(
                Bs + (row0 + it * ROW_STEP) * LD + kq0);
            d[0] = conv<H>::pack(b_next[it].x, b_next[it].y);
            d[1] = conv<H>::pack(b_next[it].z, b_next[it].w);
        }
    };
    auto compute = [&]() {
#pragma unroll
        for (int ks = 0; ks < BK / 16; ks++) {
            rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, H,
                              rocwmma::row_major> fa[FM];
            rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, H,
                              rocwmma::col_major> fb[FN];
#pragma unroll
            for (int i = 0; i < FM; i++)
                rocwmma::load_matrix_sync(
                    fa[i], pA + ((int)(wm * WM) + i * 16) * LD + ks * 16, LD);
#pragma unroll
            for (int j = 0; j < FN; j++)
                rocwmma::load_matrix_sync(
                    fb[j], pB + ((int)(wn * WN) + j * 16) * LD + ks * 16, LD);
#pragma unroll
            for (int i = 0; i < FM; i++)
#pragma unroll
                for (int j = 0; j < FN; j++)
                    rocwmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
    };

    if (DBUF) {
        fetch(0);
        stash();
        for (uint32_t kt = 1; kt < ktiles; kt++) {
            fetch(kt * (uint32_t)BK);
            __syncthreads();
            compute();
            __syncthreads();
            stash();
        }
        __syncthreads();
        compute();
    } else {
        for (uint32_t kt = 0; kt < ktiles; kt++) {
            fetch(kt * (uint32_t)BK);
            stash();
            __syncthreads();
            compute();
            __syncthreads();
        }
    }

    uint32_t lane = tid & 31u;
    uint32_t frag_col = lane & 15u, row_half = lane >> 4;
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) {
            uint32_t col = block_col + wn * WN + j * 16 + frag_col;
            if (col >= N) continue;
#pragma unroll
            for (int e = 0; e < 8; e++) {
                uint32_t row = block_row + wm * WM + i * 16 + 2u * e + row_half;
                if (row < M) C[(size_t)row * N + col] = acc[i][j].x[e];
            }
        }
}

/* ------------------------------- driver ------------------------------- */

static float *ref = NULL, *hC = NULL;

static void report(const char *label, double ms, double flops,
                   const float *C, uint32_t M, uint32_t N, int compare) {
    double max_abs = 0.0, ref_max = 0.0;
    if (compare)
        for (uint32_t r = 0; r < M; r += 61)
            for (uint32_t c = 0; c < N; c += 53) {
                double d = fabs((double)C[(size_t)r * N + c] -
                                ref[(size_t)r * N + c]);
                if (d > max_abs) max_abs = d;
                double a = fabs((double)ref[(size_t)r * N + c]);
                if (a > ref_max) ref_max = a;
            }
    printf("  %-38s %8.2f ms  %6.2f TFLOP/s", label, ms,
           flops / (ms * 1e-3) / 1e12);
    if (compare) printf("   err %.2e rel-max", max_abs / ref_max);
    printf("\n");
}

template <typename H, int BM, int BN, int BK, int WM, int WN, int PAD,
          int DBUF>
static void run(const char *tname, const float *A, const float *B, float *C,
                uint32_t M, uint32_t K, uint32_t N, double flops, size_t nc) {
    constexpr int THREADS = (BM / WM) * (BN / WN) * 32;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
    auto kern = wmma_gemm<H, BM, BN, BK, WM, WN, PAD, DBUF>;
    hipLaunchKernelGGL(kern, grid, dim3(THREADS), 0, 0, A, B, C, M, K, N);
    if (hipDeviceSynchronize() != hipSuccess) {
        printf("  %s %dx%d BK%d w%dx%d p%d db%d FAILED\n", tname, BM, BN, BK,
               WM, WN, PAD, DBUF);
        (void)hipGetLastError();
        return;
    }
    /* Best of three timing rounds: single rounds vary by ~10% on this part. */
    double best = 1e30;
    for (int round = 0; round < 3; round++) {
        hipEvent_t a, b;
        (void)hipEventCreate(&a);
        (void)hipEventCreate(&b);
        (void)hipEventRecord(a);
        for (int i = 0; i < 10; i++)
            hipLaunchKernelGGL(kern, grid, dim3(THREADS), 0, 0, A, B, C, M, K,
                               N);
        (void)hipEventRecord(b);
        (void)hipDeviceSynchronize();
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, a, b);
        (void)hipEventDestroy(a);
        (void)hipEventDestroy(b);
        if (ms / 10.0 < best) best = ms / 10.0;
    }
    (void)hipMemcpy(hC, C, nc * 4, hipMemcpyDeviceToHost);
    char label[96];
    snprintf(label, sizeof(label), "%-4s %3dx%-3d BK%-2d w%2dx%-3d p%-2d %s",
             tname, BM, BN, BK, WM, WN, PAD, DBUF ? "dbuf" : "    ");
    report(label, best, flops, hC, M, N, 1);
}

int main(int argc, char **argv) {
    using bf = rocwmma::bfloat16_t;
    struct { uint32_t M, K, N; const char *tag; } shapes[] = {
        {2304, 2048, 16384, "VAE geglu  "},
        {2304, 8192, 2048, "VAE ff-down"},
        {2304, 2048, 2048, "VAE out    "},
    };
    int ns = 3;
    if (argc == 4) {
        shapes[0].M = atoi(argv[1]);
        shapes[0].K = atoi(argv[2]);
        shapes[0].N = atoi(argv[3]);
        shapes[0].tag = "custom     ";
        ns = 1;
    }
    for (int s = 0; s < ns; s++) {
        uint32_t M = shapes[s].M, K = shapes[s].K, N = shapes[s].N;
        size_t na = (size_t)M * K, nb = (size_t)N * K, nc = (size_t)M * N;
        float *hA = (float *)malloc(na * 4), *hB = (float *)malloc(nb * 4);
        hC = (float *)malloc(nc * 4);
        ref = (float *)malloc(nc * 4);
        for (size_t i = 0; i < na; i++)
            hA[i] = (float)((int)(i % 251) - 125) * 0.011f;
        for (size_t i = 0; i < nb; i++)
            hB[i] = (float)((int)(i % 241) - 120) * 0.009f;
        float *A, *B, *C;
        (void)hipMalloc(&A, na * 4);
        (void)hipMalloc(&B, nb * 4);
        (void)hipMalloc(&C, nc * 4);
        (void)hipMemcpy(A, hA, na * 4, hipMemcpyHostToDevice);
        (void)hipMemcpy(B, hB, nb * 4, hipMemcpyHostToDevice);
        double flops = 2.0 * M * K * (double)N;
        printf("%s M=%u K=%u N=%u  (%.1f GFLOP)\n", shapes[s].tag, M, K, N,
               flops / 1e9);

        dim3 g((N + 127) / 128, (M + 127) / 128, 1);
        hipLaunchKernelGGL(r128, g, dim3(16, 16, 1), 0, 0, A, B, C, M, K, N);
        (void)hipDeviceSynchronize();
        hipEvent_t ea, eb;
        (void)hipEventCreate(&ea);
        (void)hipEventCreate(&eb);
        (void)hipEventRecord(ea);
        for (int i = 0; i < 5; i++)
            hipLaunchKernelGGL(r128, g, dim3(16, 16, 1), 0, 0, A, B, C, M, K,
                               N);
        (void)hipEventRecord(eb);
        (void)hipDeviceSynchronize();
        float ms = 0.0f;
        (void)hipEventElapsedTime(&ms, ea, eb);
        (void)hipMemcpy(ref, C, nc * 4, hipMemcpyDeviceToHost);
        report("r128 (current f32 FMA)", ms / 5.0, flops, ref, M, N, 0);

        run<bf, 256, 128, 32, 64, 64, 8, 0>("bf16", A, B, C, M, K, N, flops, nc);
        run<bf, 256, 128, 32, 64, 64, 8, 1>("bf16", A, B, C, M, K, N, flops, nc);
        run<bf, 256, 128, 32, 64, 64, 4, 1>("bf16", A, B, C, M, K, N, flops, nc);
        run<bf, 256, 128, 64, 64, 64, 8, 1>("bf16", A, B, C, M, K, N, flops, nc);
        run<bf, 128, 128, 32, 32, 64, 8, 1>("bf16", A, B, C, M, K, N, flops, nc);
        run<bf, 128, 128, 32, 64, 32, 8, 1>("bf16", A, B, C, M, K, N, flops, nc);
        run<bf, 128, 128, 64, 32, 64, 8, 1>("bf16", A, B, C, M, K, N, flops, nc);
        run<bf, 256, 256, 32, 64, 64, 8, 1>("bf16", A, B, C, M, K, N, flops, nc);
        run<bf, 512, 128, 32, 64, 64, 8, 1>("bf16", A, B, C, M, K, N, flops, nc);
        run<_Float16, 256, 128, 32, 64, 64, 8, 1>("f16", A, B, C, M, K, N, flops,
                                                  nc);
        run<_Float16, 128, 128, 32, 32, 64, 8, 1>("f16", A, B, C, M, K, N, flops,
                                                  nc);

        (void)hipFree(A);
        (void)hipFree(B);
        (void)hipFree(C);
        free(hA);
        free(hB);
        free(hC);
        free(ref);
        printf("\n");
    }
    return 0;
}
