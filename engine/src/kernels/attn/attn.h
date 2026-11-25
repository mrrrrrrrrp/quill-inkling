// Attention core of one layer visit for M = 1..8 consecutive decode tokens (the PyTorch extension's kernel ported
// to a raw-pointer launcher; contract in native/KERNELS.md, area `attn`): q/rel post-processing, key/value convs +
// state update, KV cache write, split-K attention with relative bias, sink and output gate, in one kernel.
// Always launched with programmatic dependent launch (as the Python binding does); graph capturable.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ingine::kernels::attn {

struct DecodeArgs {
  const float* proj = nullptr;  // fp32 [M, R]: R = 8600 (sliding) or 8664 (full), SEG order
  int M = 1;                    // 1..8
  const float* q_scale = nullptr;   // fp32 [24, 128]
  const float* r_scale = nullptr;   // fp32 [24, 16]
  const float* rel_proj = nullptr;  // fp32 [16, 1024]
  const float* k_conv = nullptr;    // sliding: fp32 [1024, 4] (conv1d order); full: k_static [4, 1024] by lag
  const float* v_conv = nullptr;
  float* conv_k_state = nullptr;    // fp32 [3, 1024]
  float* conv_v_state = nullptr;
  void* k_cache = nullptr;          // [rows, 8, 128] bf16, or float8_e4m3fn (full layers only)
  void* v_cache = nullptr;
  bool cache_fp8 = false;
  bool cache_q4 = false;            // quill: full-layer IQ4_NL q4 rows (q4_row_bytes() per row)
  bool cache_q8 = false;            // quill: sliding ring rows as int8 + fp16 scale per 32 (1088 B)
  std::int64_t q4_stride = 576;     // quill: bytes between consecutive positions of a q4 cache
  std::int64_t cache_rows = 0;      // 1024 (sliding) or max_ctx (full), at most 2^30
  std::int64_t* kpos = nullptr;     // sliding: int64 [1024] slot positions (-1 = empty); full: unused
  const std::int64_t* pos = nullptr;  // int64 device, pos[0] = position of the first token
  int window = 0;                   // 1024 (sliding ring) or 0 (full layer, linear cache)
  bool log_scaling = false;         // full layers: q, rel *= tau(p)
  void* workspace = nullptr;        // workspace_bytes(M) bytes, zeroed once at allocation
  std::size_t workspace_size = 0;
  __nv_bfloat16* out = nullptr;     // bf16 [M, 3072]
};

// Bytes of workspace attn_decode needs for up to max_m tokens: a counter (zero the workspace once; the
// kernel resets the counter itself) and the split partials. One workspace serves every layer of a
// stream-ordered decode step.
std::size_t workspace_bytes(int max_m);

// How many of the kernel's 16-block clusters (one per KV head, 8 per call) fit on the GPU at once.
int max_active_clusters(int M, int window);

void attn_decode(const DecodeArgs& a, cudaStream_t stream);
// quill diagnostics: per-block phase timers of the q8 sliding split kernel
void debug_enable(bool on);
void debug_report(const char* tag, int launches);

// quill: bytes of one q4 cache row (8 heads x 128 dims of keys or values).
std::size_t q4_row_bytes();

}  // namespace ingine::kernels::attn
