// quill: CPU tier of the hybrid MoE decode.
//
// Inside a captured decode step, per layer visit, a GPU kernel (kernels/moe/hybrid.cu: moe_dispatch) writes
// the experts the GPU does not hold into a request in mapped pinned memory and rings a doorbell (seq). Worker
// threads spinning on the doorbell compute those experts' outputs y = bf16(W2 . act(W13 . x)) per (token, k)
// on AVX2, write them to the response area and publish ack = seq; moe_join (GPU) spins on ack and copies the
// outputs next to the GPU experts' before the weighted sum. No CUDA call happens on the CPU side: the GPU
// never waits for the host driver, only for this memory handshake (~2 us each way over PCIe).
#pragma once

#include <cstdio>

#include "core/host_memory.h"
#include "model/experts.h"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

namespace ingine::model {

inline constexpr int CPU_MAX_ENTRIES = 64;  // (token, k) slots per call: M <= 8 tokens x top-8

// Request (GPU -> CPU) and response (CPU -> GPU) in one mapped pinned buffer. The writer of each half
// publishes its sequence number last.
struct alignas(64) CpuRequest {
  std::uint32_t seq;  // doorbell
  std::int32_t layer, M, n;
  std::int32_t visit;  // quill: decoder visit index (pre-gating hints)
  std::int32_t pad[11];
  std::int32_t slot[CPU_MAX_ENTRIES];    // m * 8 + k of each entry
  std::int32_t expert[CPU_MAX_ENTRIES];  // expert of each entry
  std::uint16_t x[8][HIDDEN];            // bf16 inputs of the M tokens
};
struct alignas(64) CpuResponse {
  std::uint32_t ack;
  std::int32_t pad[15];
  float y[CPU_MAX_ENTRIES][HIDDEN];  // BF16-rounded expert outputs, by entry
};
struct alignas(64) CpuExchange {
  CpuRequest req;
  CpuResponse resp;
};

// quill diagnostics: counters of the live CPU tier (zeros when none), reset by cpu_tier_reset_stats().
struct CpuTierStats {
  std::uint64_t calls = 0, entries = 0;
  std::uint64_t uniques = 0;  // quill: distinct experts read (the tier's DDR traffic: uniques x 4.5 MB)
  double busy_s = 0;
};
CpuTierStats cpu_tier_stats();
void cpu_tier_reset_stats();

class CpuTier {
 public:
  CpuTier(const ExpertStore& store, int threads);
  ~CpuTier();
  CpuTier(const CpuTier&) = delete;
  CpuTier& operator=(const CpuTier&) = delete;

  // Device view of the exchange buffer (for the dispatch / join kernels).
  CpuExchange* device_exchange() const { return static_cast<CpuExchange*>(exch_.device_ptr()); }
  // Workers spin only while active (decode in flight); idle workers sleep. Reference counted.
  // quill: pre-gating prefetch: every layer's router (bf16 [LAYERS][128][HIDDEN]) and bias ([LAYERS][128])
  void set_routers(std::vector<std::uint16_t> w, std::vector<float> bias);
  void activate();
  void deactivate();

  // Synchronous CPU evaluation of one expert for tests: y = bf16(W2 . act(W13 . x)), x bf16 [H].
  void compute_expert(int layer, int expert, const std::uint16_t* x, float* y) const;

  std::uint64_t calls() const { return calls_.load(); }
  std::uint64_t entries() const { return entries_.load(); }
  double busy_seconds() const { return busy_ns_.load() * 1e-9; }
  // quill diagnostics: the QUILL_TIER_TRACE file (worker 0 writes the "C" call lines; SpecLoop appends the
  // round's "V" routing lines and the "R" round line through the same handle)
  std::FILE* trace_file() const { return trace_; }

 private:
  void worker(int id);
  void run_request(int id, const CpuRequest& rq);
  void predict_next(int id, int visit, int M, std::uint32_t seq);
  std::vector<std::uint16_t> rw_;
  std::vector<float> rb_;
  int pred_mode_ = 0;  // QUILL_PREFETCH: 0 off, 1 hints traced, 2 hints prefetched
  int pf_max_ = 128;   // QUILL_PF_MAX: experts prefetched per prediction (highest scores first)
  bool nt_stream_ = true;  // quill: non-temporal hints for never-reused tier loads (QUILL_NT_STREAM=0 off)
  alignas(64) float pkey_[8][128];
  std::atomic<std::uint64_t> pf_bytes_{0}, pf_hints_{0}, pf_runs_{0}, pf_ns_{0};

  const ExpertStore& store_;
  int nthreads_;
  MappedHostBuffer exch_;
  CpuExchange* host_ = nullptr;
  std::vector<std::thread> threads_;
  std::atomic<bool> stop_{false};
  std::atomic<int> active_{0};
  // work sharing within one request
  std::atomic<std::uint32_t> phase_arrive_{0};
  std::atomic<std::uint32_t> phase_gen_{0};
  std::vector<float> act_;  // [CPU_MAX_ENTRIES][INTER] activations of the current request
  std::vector<float> xf_;   // [8][HIDDEN] fp32 inputs
  // quill: int8 path buffers (XQ<HIDDEN> x 8, XQ<INTER> x CPU_MAX_ENTRIES) in 32-byte aligned storage
  struct alignas(32) Blk32 {
    char b[32];
  };
  std::vector<Blk32> xq_, actq_;
  bool int8_ = true;
  std::FILE* trace_ = nullptr;  // QUILL_TIER_TRACE
  std::uint64_t* phase_ns_ = nullptr;  // QUILL_TIER_PHASES: cumulative ns since call start at each phase end
  std::atomic<std::uint64_t> calls_{0}, entries_{0}, busy_ns_{0}, uniques_{0};
  std::atomic<int> next1_{0}, next2_{0};  // quill: dynamic row chunks of the two weight phases
  bool static_split_ = false;
  friend CpuTierStats cpu_tier_stats();
  friend void cpu_tier_reset_stats();
  void barrier();
};

}  // namespace ingine::model
