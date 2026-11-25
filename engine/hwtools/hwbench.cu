// Hardware microbenchmarks for the ceiling calculation.
// GPU: device copy bandwidth, read-only bandwidth, H2D/D2H pinned PCIe, cuBLAS GEMM (fp16, int8).
// CPU: multithreaded AVX2 streaming read bandwidth over a large buffer.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <immintrin.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

__global__ void read_kernel(const int4* __restrict__ p, size_t n, int4* out) {
    int4 acc = make_int4(0, 0, 0, 0);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        int4 v = __ldg(p + i);
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if (acc.x == 0x12345678) out[0] = acc;
}

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static double cpu_read_bw(size_t bytes, int threads) {
    size_t n = bytes / 32;
    __m256i* buf = (__m256i*)_aligned_malloc(n * 32, 64);
    // first touch in parallel
    std::vector<std::thread> th;
    for (int t = 0; t < threads; t++) th.emplace_back([=] { size_t a = n * t / threads, b = n * (t + 1) / threads; for (size_t i = a; i < b; i++) buf[i] = _mm256_set1_epi32((int)i); });
    for (auto& x : th) x.join();
    th.clear();
    double best = 0;
    for (int rep = 0; rep < 6; rep++) {
        std::vector<long long> sink(threads * 8);
        double t0 = now();
        for (int t = 0; t < threads; t++) th.emplace_back([=, &sink] {
            size_t a = n * t / threads, b = n * (t + 1) / threads;
            __m256i x0 = _mm256_setzero_si256(), x1 = x0, x2 = x0, x3 = x0;
            for (size_t i = a; i + 4 <= b; i += 4) {
                x0 = _mm256_xor_si256(x0, _mm256_load_si256(buf + i));
                x1 = _mm256_xor_si256(x1, _mm256_load_si256(buf + i + 1));
                x2 = _mm256_xor_si256(x2, _mm256_load_si256(buf + i + 2));
                x3 = _mm256_xor_si256(x3, _mm256_load_si256(buf + i + 3));
            }
            x0 = _mm256_xor_si256(_mm256_xor_si256(x0, x1), _mm256_xor_si256(x2, x3));
            sink[t * 8] = _mm256_extract_epi64(x0, 0);
        });
        for (auto& x : th) x.join();
        th.clear();
        double dt = now() - t0;
        best = std::max(best, bytes / dt / 1e9);
    }
    _aligned_free(buf);
    return best;
}

int main() {
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU %s sm_%d%d SMs=%d L2=%d MB\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize >> 20);
    size_t freeb, totb; CK(cudaMemGetInfo(&freeb, &totb));
    printf("cudaMemGetInfo free=%.0f MiB total=%.0f MiB\n", freeb / 1048576.0, totb / 1048576.0);

    const size_t B = 1ull << 30;  // 1 GiB
    void *a, *b; CK(cudaMalloc(&a, B)); CK(cudaMalloc(&b, B));
    CK(cudaMemset(a, 1, B)); CK(cudaMemset(b, 2, B));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float ms;
    // D2D copy (counts read + write)
    double best = 0;
    for (int i = 0; i < 10; i++) { cudaEventRecord(e0); CK(cudaMemcpy(b, a, B, cudaMemcpyDeviceToDevice)); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); best = std::max(best, 2.0 * B / (ms / 1e3) / 1e9); }
    printf("D2D copy bandwidth (read+write): %.1f GB/s\n", best);
    // read-only
    int4* out; CK(cudaMalloc(&out, 64));
    best = 0;
    for (int i = 0; i < 10; i++) { cudaEventRecord(e0); read_kernel<<<prop.multiProcessorCount * 8, 512>>>((int4*)a, B / 16, out); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); best = std::max(best, B / (ms / 1e3) / 1e9); }
    printf("GPU read-only bandwidth: %.1f GB/s\n", best);
    // smaller read (fits partly L2) - 64 MB
    best = 0;
    for (int i = 0; i < 20; i++) { cudaEventRecord(e0); read_kernel<<<prop.multiProcessorCount * 8, 512>>>((int4*)a, (64ull << 20) / 16, out); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); best = std::max(best, (64ull << 20) / (ms / 1e3) / 1e9); }
    printf("GPU read-only bandwidth (64 MB): %.1f GB/s\n", best);

    // PCIe pinned
    void* h; CK(cudaMallocHost(&h, 512ull << 20)); memset(h, 3, 512ull << 20);
    best = 0;
    for (int i = 0; i < 8; i++) { cudaEventRecord(e0); CK(cudaMemcpy(a, h, 512ull << 20, cudaMemcpyHostToDevice)); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); best = std::max(best, (512.0 * 1048576) / (ms / 1e3) / 1e9); }
    printf("H2D pinned: %.1f GB/s\n", best);
    best = 0;
    for (int i = 0; i < 8; i++) { cudaEventRecord(e0); CK(cudaMemcpy(h, a, 512ull << 20, cudaMemcpyDeviceToHost)); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); best = std::max(best, (512.0 * 1048576) / (ms / 1e3) / 1e9); }
    printf("D2H pinned: %.1f GB/s\n", best);
    // small transfer latency (9 KB hidden state roundtrip)
    {
        double t0 = now();
        for (int i = 0; i < 1000; i++) { CK(cudaMemcpy(a, h, 9216, cudaMemcpyHostToDevice)); CK(cudaMemcpy(h, a, 9216, cudaMemcpyDeviceToHost)); }
        printf("9 KB H2D+D2H roundtrip: %.1f us\n", (now() - t0) * 1e6 / 1000);
    }
    CK(cudaFreeHost(h));

    // cuBLAS GEMMs
    cublasHandle_t hb; cublasCreate(&hb);
    for (int N : {4096, 8192}) {
        void *A, *Bm, *C;
        CK(cudaMalloc(&A, (size_t)N * N * 2)); CK(cudaMalloc(&Bm, (size_t)N * N * 2)); CK(cudaMalloc(&C, (size_t)N * N * 4));
        cudaMemset(A, 0, (size_t)N * N * 2); cudaMemset(Bm, 0, (size_t)N * N * 2);
        float alpha = 1, beta = 0;
        // fp16 in, fp32 accumulate
        for (int w = 0; w < 3; w++) cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, CUDA_R_16F, N, Bm, CUDA_R_16F, N, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        best = 0;
        for (int i = 0; i < 5; i++) { cudaEventRecord(e0); cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, CUDA_R_16F, N, Bm, CUDA_R_16F, N, &beta, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); best = std::max(best, 2.0 * N * N * N / (ms / 1e3) / 1e12); }
        printf("GEMM fp16 (fp32 acc) %d^3: %.1f TFLOPS\n", N, best);
        // fp16 accumulate
        __half ah = __float2half(1.f), bh = __float2half(0.f);
        best = 0;
        for (int i = 0; i < 5; i++) { cudaEventRecord(e0); cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &ah, A, CUDA_R_16F, N, Bm, CUDA_R_16F, N, &bh, C, CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); best = std::max(best, 2.0 * N * N * N / (ms / 1e3) / 1e12); }
        printf("GEMM fp16 (fp16 acc) %d^3: %.1f TFLOPS\n", N, best);
        // int8 -> int32
        int ia = 1, ib = 0;
        best = 0;
        for (int i = 0; i < 5; i++) { cudaEventRecord(e0); cublasStatus_t st = cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &ia, A, CUDA_R_8I, N, Bm, CUDA_R_8I, N, &ib, C, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); if (st) { printf("int8 gemm status %d\n", st); break; } best = std::max(best, 2.0 * N * N * N / (ms / 1e3) / 1e12); }
        printf("GEMM int8 (int32 acc) %d^3: %.1f TOPS\n", N, best);
        cudaFree(A); cudaFree(Bm); cudaFree(C);
    }
    int clk; cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, 0);
    printf("SM clock attr: %d MHz\n", clk / 1000);

    unsigned nt = std::thread::hardware_concurrency();
    for (int t : {1, 4, 8, (int)nt}) printf("CPU read bandwidth, %2d threads (2 GiB): %.1f GB/s\n", t, cpu_read_bw(2ull << 30, t));
    return 0;
}
