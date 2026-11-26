// Routed-expert kernels (the PyTorch extension's kernels ported to raw-pointer launchers; contracts in
// native/KERNELS.md, area `moe`) and the expert decode to BF16. Launchers run on `stream`, never
// allocate or synchronise, read routing from device memory and are CUDA-graph capturable. Shapes are
// row-major (Python order); H = 2304, I = 1152, 128 experts, top-8. Invalid arguments throw
// std::invalid_argument.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::kernels::moe {

enum class Format : int { NVFP4 = 0, MXFP8 = 1 };

// moe_up: W13 GEMV of the selected experts + softcapped SwiGLU. x bf16 [M, H], idx int32 [M, 8],
// out bf16 [M, 8, I]. NVFP4: w13 u8 [128, 2I, H/2], w13_scale e4m3 [128, 2I, H/16], w13_factor fp32 [128];
// MXFP8: w13 e4m3 [128, 2I, H], w13_scale u8 E8M0 [128, 2I, H/32], factor unused (may be null).
// pdl: the kernel waits for its predecessor before reading anything.
void moe_up(const __nv_bfloat16* x, int M, const int* idx, const void* w13, const void* w13_scale,
            const float* w13_factor, Format fmt, __nv_bfloat16* out, bool pdl, cudaStream_t stream);

// moe_down: W2 GEMV + routing-weighted sum + post-MLP conv with state update. act bf16 [M, 8, I],
// wts fp32 [M, 8], w2 [128, H, I(/2)] + scales [128, H, I/16 | I/32] (+ factor), conv_w fp32 [H, 4],
// conv_state fp32 [3, H], out fp32 [M, H], conv_hist fp32 [M, H] (BF16-rounded conv inputs) or null.
// pdl: valid only right after moe_up (idx is read before the PDL wait).
void moe_down(const __nv_bfloat16* act, int M, const int* idx, const float* wts, const void* w2,
              const void* w2_scale, const float* w2_factor, Format fmt, const float* conv_w, float* conv_state,
              float* out, float* conv_hist, bool pdl, cudaStream_t stream);

// dequant_experts: decode experts[0..n) (int32 device) of one weight ([128, rows, cols] in checkpoint
// encoding: NVFP4 codes u8 [128, rows, cols/2] + e4m3 scales [128, rows, cols/16] + factor fp32 [128];
// MXFP8 codes e4m3 [128, rows, cols] + E8M0 scales [128, rows, cols/32]) into out bf16 [n, rows, cols],
// exactly as model.decode_experts (FP32 product, one BF16 rounding).
// quill: hybrid decode MoE. Device view of the CPU tier's mapped exchange (model/cpu_tier.h CpuExchange).
struct HybridExchange {
  unsigned* req_hdr = nullptr;   // {seq, layer, M, n}
  int* req_slot = nullptr;       // [64]
  int* req_expert = nullptr;     // [64]
  uint16_t* req_x = nullptr;     // bf16 [8, H]
  unsigned* resp_ack = nullptr;
  float* resp_y = nullptr;       // [64, H]
  // quill (lossy, off at 0): a non-resident expert whose routing weight is below skip_below (fraction of the
  // token's total, weights sum to 8) is dropped instead of sent to the CPU tier; skip_renorm rescales the
  // token's kept weights back to a sum of 8
  float skip_below = 0.f;
  bool skip_renorm = true;
  // quill: GPU zero-copy assist: the first round(share * U) of a call's U unique non-resident experts (U >=
  // assist_min, at least one left to the CPU) are computed on the GPU from the host arrays instead
  float assist_share = 0.f;  // quill: off since the int8 tier + 41k draft head (teacher-forced 109.2 vs 108.4 at 0.34); QUILL_ASSIST=x
  int assist_min = 3;
  bool assist_by_tokens = false;  // quill: share-based assist takes the misses with the most rows
  int assist_min_tokens = 0;      // quill: > 0: assist every miss used by at least this many rows (share ignored)
};
// quill: a layer's host expert arrays (device-mapped pinned memory): expert e's rows at w + e * (rows * code
// bytes), scales likewise, factors fac[2 e] (w13) / fac[2 e + 1] (w2); flags[e] != 0: computed on the GPU
struct HostAssist {
  const unsigned char* flags = nullptr;
  const unsigned char* w = nullptr;
  const unsigned char* s = nullptr;
  const float* fac = nullptr;
};
// quill: regions pulled into L2 on a side stream while the join waits for the CPU tier (next visit's weights)
struct L2Prefetch {
  const void* p[8] = {};
  unsigned long long n[8] = {};
  int count = 0;
  unsigned* sink = nullptr;  // never written in practice (keeps the loads alive)
};
struct HybridArgs {
  const __nv_bfloat16* x = nullptr;  // [M, H]
  int M = 1, layer = 0;
  const int* idx = nullptr;          // [M, 8]
  const float* wts = nullptr;        // [M, 8]
  const int* slot_map = nullptr;     // [128] -> pool slot or -1
  Format fmt = Format::NVFP4;
  const void* w13 = nullptr;         // pool arrays (indexed by slot)
  const void* w13_scale = nullptr;
  const float* w13_factor = nullptr;
  const void* w2 = nullptr;
  const void* w2_scale = nullptr;
  const float* w2_factor = nullptr;
  __nv_bfloat16* act = nullptr;      // [M, 8, I] scratch
  float* y = nullptr;                // [M * 8, H] scratch: per-slot expert outputs
  HybridExchange ex;
  unsigned* ctl = nullptr;           // device [2]: seq, n of the current call
  const float* conv_w = nullptr;
  float* conv_state = nullptr;
  float* out = nullptr;              // [M, H]
  float* conv_hist = nullptr;
  unsigned* route_counts = nullptr;  // [24][128] selection counts (profiling), or null
  long long slot_stride = 0;         // bytes between slots when the pool is a shared region (0: per-array layout)
  unsigned char* assist = nullptr;   // quill: device [128] flags (written by dispatch), null: no GPU assist
  HostAssist host_up, host_down;     // quill: this layer's host w13 / w2 arrays (flags set by hybrid_moe)
  int visit = -1;                    // quill: visit index, written to the request header (pre-gating hints)
  // quill diagnostics (QUILL_TIER_TRACE): device record of this visit's full routing, [VTRACE_INTS] ints:
  // {layer, n (CPU-tier entries), M, visit, then (expert << 1) | resident bit of every (row, k) slot in
  // idx order, M * 8 values (the slots past M stay 0)}. Null when not tracing.
  int* vtrace = nullptr;
  L2Prefetch pf;                     // quill: launched on pf_stream after moe_down (pf_event: the fork)
  cudaStream_t pf_stream = nullptr;
  cudaEvent_t pf_event = nullptr;
};
// quill diagnostics (QUILL_TIER_TRACE): ints per visit in the routing record HybridArgs.vtrace points to
inline constexpr int VTRACE_INTS = 4 + 8 * 8;  // header (layer, n, M, visit) + M * 8 packed (row, k) slots

// dispatch (CPU tier request) -> moe_up / moe_down over resident experts -> join (wait for the CPU tier) ->
// weighted sum + post-MLP conv. Same results as moe_up + moe_down for any residency.
void hybrid_moe(const HybridArgs& a, cudaStream_t stream);
// quill diagnostics: GPU-side join timing (report prints and resets the sums)
void join_debug(bool enable, bool report);

void dequant_experts(const void* codes, const void* scales, const float* factor, const std::int32_t* experts, int n,
                     int rows, int cols, Format fmt, __nv_bfloat16* out, cudaStream_t stream);

}  // namespace ingine::kernels::moe
