/* Kernel microbenchmark used for the W7900 (gfx1100) analysis in
 * docs/perf-runs/W7900_2026-11-09.md: device bandwidth, FP32 FMA, the DiT
 * SDPA at long sequences (optionally with Q/K/V in pinned host memory), and
 * the INT8 GEMM paths at DiT shapes. It samples the active sclk/mclk levels
 * from sysfs while each test runs.
 *
 * Build after `make HIP_ARCH=gfx1100 h3` (kernel objects must exist):
 *   hipcc -std=c++17 -O3 -DH3_HIP -I. -Ikernels -I$ROCM/include \
 *     -I<rocwmma include> --offload-arch=gfx1100 -c tests/bench_kernels.hip \
 *     -o tests/bench_kernels.o
 *   hipcc tests/bench_kernels.o kernels/h3_kernels*.o -lamdhip64 -lhipblas \
 *     -lhipblaslt -o h3_bench_kernels
 *
 * Usage: ./h3_bench_kernels {bw|fma|sdpa|gemm|int8|all}
 *   SDPA_SEQ / SDPA_HEADS   single SDPA shape (default: a 4k..109k sweep)
 *   SDPA_HOST=1             Q/K/V/O in hipHostMalloc memory
 *   GEMM_HOST=1             GEMM activations and output in host memory
 *   INT8_ROWS / INT8_ONLY   rows and shape subset (qkv,fc1,fc2,out) for int8
 *   INT8_TAG                suffix for the dumped outputs in /tmp
 *   H3_BENCH_SYSFS          DRM device dir for clock sampling
 *                           (default /sys/class/drm/card1/device) */
#include "h3_kernels.h"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <atomic>
#include <functional>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#define HIPCHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP %s at %s:%d: %s\n", #x, __FILE__, __LINE__, \
            hipGetErrorString(e_)); exit(1); } } while (0)

static const char *g_card = "/sys/class/drm/card1/device";

static std::atomic<bool> g_sample{false};
static std::atomic<bool> g_quit{false};
static std::map<std::string, int> g_sclk, g_mclk;
static int g_samples = 0;

static std::string active_level(const char *file) {
    char path[256];
    snprintf(path, sizeof(path), "%s/%s", g_card, file);
    FILE *f = fopen(path, "r");
    if (!f) return "?";
    char line[128];
    std::string out = "?";
    while (fgets(line, sizeof(line), f)) {
        if (strchr(line, '*')) {
            char *colon = strchr(line, ':');
            std::string s = colon ? colon + 1 : line;
            size_t a = s.find_first_not_of(' ');
            size_t b = s.find('*');
            out = s.substr(a, b - a);
            while (!out.empty() && out.back() == ' ') out.pop_back();
        }
    }
    fclose(f);
    return out;
}

static void sampler() {
    while (!g_quit) {
        if (g_sample) {
            g_sclk[active_level("pp_dpm_sclk")]++;
            g_mclk[active_level("pp_dpm_mclk")]++;
            g_samples++;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

static void begin_sample() {
    g_sclk.clear();
    g_mclk.clear();
    g_samples = 0;
    g_sample = true;
}

static void end_sample(const char *label) {
    g_sample = false;
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    printf("    clocks during %s (%d samples): sclk", label, g_samples);
    for (auto &kv : g_sclk) printf(" %s=%d", kv.first.c_str(), kv.second);
    printf(" | mclk");
    for (auto &kv : g_mclk) printf(" %s=%d", kv.first.c_str(), kv.second);
    printf("\n");
}

__global__ void fma_kernel(float *out, int iters) {
    float a = threadIdx.x * 1e-3f, b = 1.0001f, c = 0.9999f, d = 0.5f;
    float e = a + 1, f = a + 2, g = a + 3, h = a + 4;
    for (int i = 0; i < iters; i++) {
        a = fmaf(a, b, c); e = fmaf(e, b, c); f = fmaf(f, b, c); g = fmaf(g, b, c);
        h = fmaf(h, b, c); d = fmaf(d, b, c);
        a = fmaf(a, c, b); e = fmaf(e, c, b); f = fmaf(f, c, b); g = fmaf(g, c, b);
        h = fmaf(h, c, b); d = fmaf(d, c, b);
    }
    if (a + e + f + g + h + d == 12345.0f) out[threadIdx.x] = a;
}

__global__ void read_kernel(const float4 *in, size_t n, float *out) {
    float acc = 0.0f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        float4 v = in[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.0f) out[0] = acc;
}

__global__ void fill_bf16(uint16_t *p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed;
        x ^= x >> 13;
        float v = ((float)(x & 0xffff) / 65535.0f - 0.5f) * 0.25f;
        uint32_t bits;
        memcpy(&bits, &v, 4);
        p[i] = (uint16_t)(bits >> 16);
    }
}

static float time_ms(hipStream_t s, const std::function<void()> &fn, int reps) {
    hipEvent_t a, b;
    HIPCHECK(hipEventCreate(&a));
    HIPCHECK(hipEventCreate(&b));
    fn();
    HIPCHECK(hipStreamSynchronize(s));
    HIPCHECK(hipEventRecord(a, s));
    for (int i = 0; i < reps; i++) fn();
    HIPCHECK(hipEventRecord(b, s));
    HIPCHECK(hipEventSynchronize(b));
    float ms = 0;
    HIPCHECK(hipEventElapsedTime(&ms, a, b));
    hipEventDestroy(a);
    hipEventDestroy(b);
    return ms / reps;
}

int main(int argc, char **argv) {
    const char *only = argc > 1 ? argv[1] : "all";
    if (getenv("H3_BENCH_SYSFS")) g_card = getenv("H3_BENCH_SYSFS");
    hipDeviceProp_t p;
    HIPCHECK(hipGetDeviceProperties(&p, 0));
    printf("device %s %s CU=%d clock=%d kHz mem_clock=%d kHz bus=%d bit L2=%d KiB\n",
           p.name, p.gcnArchName, p.multiProcessorCount, p.clockRate,
           p.memoryClockRate, p.memoryBusWidth, p.l2CacheSize / 1024);
    std::thread th(sampler);
    hipStream_t s;
    HIPCHECK(hipStreamCreate(&s));
    float *sink;
    HIPCHECK(hipMalloc(&sink, 4096));

    if (!strcmp(only, "all") || !strcmp(only, "bw")) {
        size_t bytes = (size_t)4 << 30;
        void *a, *b;
        HIPCHECK(hipMalloc(&a, bytes));
        HIPCHECK(hipMalloc(&b, bytes));
        HIPCHECK(hipMemset(a, 1, bytes));
        begin_sample();
        float ms = time_ms(s, [&] {
            HIPCHECK(hipMemcpyAsync(b, a, bytes, hipMemcpyDeviceToDevice, s));
        }, 10);
        end_sample("D2D copy");
        printf("bw   D2D copy 4 GiB: %.2f ms -> %.1f GB/s (read+write)\n", ms,
               2.0 * bytes / ms / 1e6);
        size_t n4 = bytes / 16;
        begin_sample();
        ms = time_ms(s, [&] {
            read_kernel<<<p.multiProcessorCount * 32, 256, 0, s>>>(
                (const float4 *)a, n4, sink);
        }, 10);
        end_sample("read");
        printf("bw   read-only 4 GiB: %.2f ms -> %.1f GB/s\n", ms, bytes / ms / 1e6);
        hipFree(a);
        hipFree(b);
    }

    if (!strcmp(only, "all") || !strcmp(only, "fma")) {
        int iters = 20000;
        int blocks = p.multiProcessorCount * 64;
        begin_sample();
        float ms = time_ms(s, [&] {
            fma_kernel<<<blocks, 256, 0, s>>>(sink, iters);
        }, 5);
        end_sample("fma");
        double flops = 2.0 * 12 * iters * (double)blocks * 256;
        printf("fma  FP32: %.2f ms -> %.2f TFLOPS\n", ms, flops / ms / 1e9);
    }

    if (!strcmp(only, "all") || !strcmp(only, "sdpa")) {
        struct Case { uint32_t seq, heads; int reps; };
        std::vector<Case> cases = {{4096, 56, 5}, {16384, 56, 2},
                                   {44000, 8, 2}, {109334, 8, 1}};
        if (getenv("SDPA_SEQ")) {
            cases = {{(uint32_t)atoi(getenv("SDPA_SEQ")),
                      (uint32_t)atoi(getenv("SDPA_HEADS") ? getenv("SDPA_HEADS") : "8"), 1}};
        }
        for (auto c : cases) {
            size_t n = (size_t)c.seq * c.heads * 128;
            uint16_t *q, *k, *v, *o;
            int host = getenv("SDPA_HOST") && atoi(getenv("SDPA_HOST"));
            if (host) {
                HIPCHECK(hipHostMalloc(&q, n * 2, hipHostMallocDefault));
                HIPCHECK(hipHostMalloc(&k, n * 2, hipHostMallocDefault));
                HIPCHECK(hipHostMalloc(&v, n * 2, hipHostMallocDefault));
                HIPCHECK(hipHostMalloc(&o, n * 2, hipHostMallocDefault));
            } else {
                HIPCHECK(hipMalloc(&q, n * 2));
                HIPCHECK(hipMalloc(&k, n * 2));
                HIPCHECK(hipMalloc(&v, n * 2));
                HIPCHECK(hipMalloc(&o, n * 2));
            }
            printf("    buffers in %s memory\n", host ? "pinned HOST" : "device");
            fill_bf16<<<1024, 256, 0, s>>>(q, n, 1);
            fill_bf16<<<1024, 256, 0, s>>>(k, n, 2);
            fill_bf16<<<1024, 256, 0, s>>>(v, n, 3);
            h3_sdpa_args args = {c.seq, c.heads, 128, 0.08838834764f, 0, 1};
            begin_sample();
            float ms = time_ms(s, [&] {
                if (!h3_launch_sdpa_bf16(q, k, v, o, &args, s)) {
                    fprintf(stderr, "sdpa launch failed\n");
                    exit(1);
                }
            }, c.reps);
            char label[64];
            snprintf(label, sizeof(label), "sdpa L=%u", c.seq);
            end_sample(label);
            double flops = 4.0 * c.seq * (double)c.seq * 128 * c.heads;
            printf("sdpa L=%6u heads=%2u: %10.2f ms -> %6.2f TFLOPS  (x56 heads est %.1f s/call)\n",
                   c.seq, c.heads, ms, flops / ms / 1e9,
                   ms / 1000.0 * 56.0 / c.heads);
            if (host) {
                hipHostFree(q); hipHostFree(k); hipHostFree(v); hipHostFree(o);
            } else {
                hipFree(q); hipFree(k); hipFree(v); hipFree(o);
            }
        }
    }

    if (!strcmp(only, "all") || !strcmp(only, "gemm")) {
        hipblasHandle_t h;
        hipblasCreate(&h);
        hipblasSetStream(h, s);
        struct G { int m, n, k; const char *what; };
        std::vector<G> gs = {{16128, 8192, 5376, "qkv"},
                             {28672, 8192, 5376, "fc1"},
                             {5376, 8192, 14336, "fc2"}};
        for (auto g : gs) {
            int8_t *A, *B;
            int32_t *C;
            int ghost = getenv("GEMM_HOST") && atoi(getenv("GEMM_HOST"));
            HIPCHECK(hipMalloc(&A, (size_t)g.m * g.k));
            if (ghost) {
                HIPCHECK(hipHostMalloc(&B, (size_t)g.n * g.k, hipHostMallocDefault));
                HIPCHECK(hipHostMalloc(&C, (size_t)g.m * g.n * 4, hipHostMallocDefault));
            } else {
                HIPCHECK(hipMalloc(&B, (size_t)g.n * g.k));
                HIPCHECK(hipMalloc(&C, (size_t)g.m * g.n * 4));
            }
            HIPCHECK(hipMemset(A, 1, (size_t)g.m * g.k));
            HIPCHECK(hipMemset(B, 1, (size_t)g.n * g.k));
            int32_t one = 1, zero = 0;
            begin_sample();
            float ms = time_ms(s, [&] {
                hipblasStatus_t st = hipblasGemmEx(
                    h, HIPBLAS_OP_T, HIPBLAS_OP_N, g.m, g.n, g.k, &one, A,
                    HIP_R_8I, g.k, B, HIP_R_8I, g.k, &zero, C, HIP_R_32I, g.m,
                    HIPBLAS_COMPUTE_32I, HIPBLAS_GEMM_DEFAULT);
                if (st != HIPBLAS_STATUS_SUCCESS) {
                    fprintf(stderr, "int8 gemm status %d\n", (int)st);
                    exit(1);
                }
            }, 3);
            end_sample("int8 gemm");
            double ops = 2.0 * g.m * (double)g.n * g.k;
            printf("gemm int8 %s m=%d n=%d k=%d (act %s): %8.2f ms -> %6.2f TOPS\n",
                   g.what, g.m, g.n, g.k, ghost ? "HOST" : "device", ms,
                   ops / ms / 1e9);
            hipFree(A);
            if (ghost) { hipHostFree(B); hipHostFree(C); }
            else { hipFree(B); hipFree(C); }
            if (ghost) continue;

            uint16_t *Ab, *Bb, *Cb;
            HIPCHECK(hipMalloc(&Ab, (size_t)g.m * g.k * 2));
            HIPCHECK(hipMalloc(&Bb, (size_t)g.n * g.k * 2));
            HIPCHECK(hipMalloc(&Cb, (size_t)g.m * g.n * 2));
            fill_bf16<<<1024, 256, 0, s>>>(Ab, (size_t)g.m * g.k, 4);
            fill_bf16<<<1024, 256, 0, s>>>(Bb, (size_t)g.n * g.k, 5);
            float fone = 1.0f, fzero = 0.0f;
            begin_sample();
            ms = time_ms(s, [&] {
                hipblasStatus_t st = hipblasGemmEx(
                    h, HIPBLAS_OP_T, HIPBLAS_OP_N, g.m, g.n, g.k, &fone, Ab,
                    HIP_R_16BF, g.k, Bb, HIP_R_16BF, g.k, &fzero, Cb,
                    HIP_R_16BF, g.m, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
                if (st != HIPBLAS_STATUS_SUCCESS) {
                    fprintf(stderr, "bf16 gemm status %d\n", (int)st);
                    exit(1);
                }
            }, 3);
            end_sample("bf16 gemm");
            printf("gemm bf16 %s m=%d n=%d k=%d: %8.2f ms -> %6.2f TFLOPS\n",
                   g.what, g.m, g.n, g.k, ms, ops / ms / 1e9);
            hipFree(Ab); hipFree(Bb); hipFree(Cb);
        }
        hipblasDestroy(h);
    }

    if (!strcmp(only, "int8")) {
        const char *tag = getenv("INT8_TAG") ? getenv("INT8_TAG") : "run";
        uint32_t rows = getenv("INT8_ROWS") ? (uint32_t)atoi(getenv("INT8_ROWS")) : 70000u;
        struct L { const char *name; uint32_t k, out; int kind; };
        std::vector<L> ls = {{"qkv", 5376, 21504, 0},
                             {"fc1", 5376, 14336, 1},
                             {"fc2", 14336, 5376, 2},
                             {"out", 7168, 5376, 2}};
        if (getenv("INT8_ONLY")) {
            std::vector<L> keep;
            for (auto l : ls) if (strstr(getenv("INT8_ONLY"), l.name)) keep.push_back(l);
            ls = keep;
        }
        for (auto l : ls) {
            uint32_t gemm_n = l.kind == 1 ? l.out * 2 : l.out;
            size_t in_n = (size_t)rows * l.k, w_n = (size_t)gemm_n * l.k;
            uint32_t groups = l.kind == 2 ? l.k / 1024u : 1u;
            std::vector<int8_t> hin(in_n), hw(w_n);
            uint32_t x = 12345;
            for (auto &v : hin) { x = x * 1664525u + 1013904223u; v = (int8_t)(x >> 24); }
            for (auto &v : hw) { x = x * 1664525u + 1013904223u; v = (int8_t)(x >> 24); }
            std::vector<float> hsa((size_t)rows * groups), hsb(gemm_n);
            for (auto &v : hsa) { x = x * 1664525u + 1013904223u; v = 1e-4f * (1 + (x >> 28)); }
            for (auto &v : hsb) { x = x * 1664525u + 1013904223u; v = 1e-3f * (1 + (x >> 28)); }
            int8_t *din, *dw; float *dsa, *dsb; uint16_t *dout;
            HIPCHECK(hipMalloc(&din, in_n)); HIPCHECK(hipMalloc(&dw, w_n));
            HIPCHECK(hipMalloc(&dsa, hsa.size() * 4)); HIPCHECK(hipMalloc(&dsb, hsb.size() * 4));
            size_t out_n = (size_t)rows * l.out;
            HIPCHECK(hipMalloc(&dout, out_n * 2));
            HIPCHECK(hipMemcpy(din, hin.data(), in_n, hipMemcpyHostToDevice));
            HIPCHECK(hipMemcpy(dw, hw.data(), w_n, hipMemcpyHostToDevice));
            HIPCHECK(hipMemcpy(dsa, hsa.data(), hsa.size() * 4, hipMemcpyHostToDevice));
            HIPCHECK(hipMemcpy(dsb, hsb.data(), hsb.size() * 4, hipMemcpyHostToDevice));
            h3_linear_args a = {rows, l.k, l.out, 0};
            h3_linear_int8_grouped_args ga = {rows, l.k, l.out, 1024};
            float ms = time_ms(s, [&] {
                int ok = l.kind == 0 ? h3_launch_linear_int8_nax_r64(din, dw, dsa, dsb, dout, &a, s)
                       : l.kind == 1 ? h3_launch_fc1_swiglu_int8_nax_r64(din, dw, dsa, dsb, dout, &a, s)
                       : h3_launch_linear_int8_grouped_nax_r64(din, dw, dsa, dsb, dout, &ga, s);
                if (!ok) { fprintf(stderr, "int8 launch failed %s\n", l.name); exit(1); }
            }, 2);
            HIPCHECK(hipStreamSynchronize(s));
            std::vector<uint16_t> hout(out_n);
            HIPCHECK(hipMemcpy(hout.data(), dout, out_n * 2, hipMemcpyDeviceToHost));
            char path[128];
            snprintf(path, sizeof(path), "/tmp/h3_bench_int8_%s_%s.bin", l.name, tag);
            FILE *f = fopen(path, "wb"); fwrite(hout.data(), 2, out_n, f); fclose(f);
            double ops = 2.0 * rows * (double)gemm_n * l.k;
            printf("int8 %s rows=%u k=%u n=%u [%s]: %8.2f ms -> %6.2f TOPS\n",
                   l.name, rows, l.k, gemm_n, tag, ms, ops / ms / 1e9);
            hipFree(din); hipFree(dw); hipFree(dsa); hipFree(dsb); hipFree(dout);
        }
    }

    g_quit = true;
    th.join();
    return 0;
}
