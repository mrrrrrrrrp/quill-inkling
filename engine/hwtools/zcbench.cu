// Concurrent bandwidth: CPU threads streaming one pinned buffer while a GPU kernel streams another pinned buffer
// over PCIe (zero-copy, mapped host memory). Tells whether GPU-side reads of host experts add bandwidth on top of
// the CPU tier or just share the DRAM.
#include <cuda_runtime.h>
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

__global__ void zc_read(const int4* __restrict__ p, size_t n, int reps, int4* out) {
  int4 acc = make_int4(0, 0, 0, 0);
  for (int r = 0; r < reps; ++r)
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
      int4 v = __ldg(p + i);
      acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
  if (acc.x == 0x7531) out[0] = acc;
}

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
  const size_t B = 1ull << 30;  // 1 GiB each
  void *hc, *hg;
  CK(cudaHostAlloc(&hc, B, cudaHostAllocMapped));
  CK(cudaHostAlloc(&hg, B, cudaHostAllocMapped));
  memset(hc, 1, B); memset(hg, 2, B);
  void* dg; CK(cudaHostGetDevicePointer(&dg, hg, 0));
  int4* out; CK(cudaMalloc(&out, 64));
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  auto gpu_run = [&](int blocks, int reps) -> double {
    cudaEventRecord(e0);
    zc_read<<<blocks, 512>>>((const int4*)dg, B / 16, reps, out);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    return reps * B / (ms / 1e3) / 1e9;
  };
  for (int blocks : {56, 112, 224, 448}) printf("GPU zero-copy read alone, %d blocks: %.1f GB/s\n", blocks, gpu_run(blocks, 2));
  auto cpu_read = [&](int threads, double seconds, std::atomic<bool>& go) -> double {
    std::vector<std::thread> th; std::atomic<size_t> bytes{0};
    for (int t = 0; t < threads; t++) th.emplace_back([&, t] {
      const __m256i* p = (const __m256i*)hc; size_t n = B / 32; size_t a = n * t / threads, b = n * (t + 1) / threads;
      __m256i acc = _mm256_setzero_si256(); size_t done = 0;
      while (!go.load()) {}
      double t0 = now();
      while (now() - t0 < seconds) { for (size_t i = a; i < b; i++) acc = _mm256_xor_si256(acc, _mm256_load_si256(p + i)); done += (b - a) * 32; }
      bytes += done + (_mm256_extract_epi64(acc, 0) == 12345);
    });
    go = true; for (auto& x : th) x.join();
    return bytes / seconds / 1e9;
  };
  { std::atomic<bool> go{false}; printf("CPU read alone, 7 threads: %.1f GB/s\n", cpu_read(7, 2.0, go)); }
  {
    std::atomic<bool> go{false};
    double cpu_bw = 0;
    std::thread t([&] { cpu_bw = cpu_read(7, 3.0, go); });
    while (!go.load()) {}
    double g = gpu_run(224, 4);
    t.join();
    printf("concurrent: CPU %.1f GB/s + GPU zero-copy %.1f GB/s\n", cpu_bw, g);
  }
  return 0;
}
