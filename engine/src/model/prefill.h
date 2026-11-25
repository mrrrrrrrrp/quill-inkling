// Native prompt processing: model.Model.prefill for one chunk, without PyTorch.
#pragma once

#include "core/arena.h"
#include "model/model_data.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace ingine::kernels::prefill {
class Gemm;
}

namespace ingine::model {

// Processes prompt tokens [T] (device int32) at positions pos0..pos0+T-1 on top of `state` (exactly
// model.Model.prefill for one chunk): writes caches/conv state, sets state.length = pos0 + T, returns the last
// token's logits (device fp32 [UNPADDED_VOCAB], softcapped like the head) and, if taps != nullptr, every row's
// DSpark taps (bf16 [T, 4, H], model.TAP_VISITS + final normed hidden).
//
// The visit sequence is fast.FastDecoder.decode's (the same block-state wiring and mix / message-gate kernels)
// with T rows; projections are cuBLASLt BF16 GEMMs over the dequantized weight rows (weights.dequant), the
// attention a flash-attention prefill kernel, the MoE grouped GEMMs over the experts decoded on the fly.
// Stream-ordered on `stream`; run() only synchronises for its first use of a block-pointer table.
// Embedding rows computed elsewhere (image tokens from the vision tower) for rows [row, row + count) of a
// chunk: src bf16 [count, HIDDEN] on the device replaces the token embeddings of those rows (which the model
// feeds in as they are: no embedding norm).
struct EmbedRows {
  int row = 0, count = 0;
  const __nv_bfloat16* src = nullptr;
};

class Prefiller {
 public:
  // lent: the scratch is not allocated here; bind() places it (quill: in expert slots of the shared region,
  // reclaimed for each prefill, see Runtime::lend_scratch) and zero() must run before each prefill
  // taps_bytes: a buffer for the drafter's prefill taps inside the scratch (taps_scratch()), 0: none
  Prefiller(const Weights& w, const Ablation& abl, int max_chunk, cudaStream_t stream, bool lent = false,
            std::size_t taps_bytes = 0);
  void* taps_scratch() const { return taps_; }
  ~Prefiller();
  Prefiller(const Prefiller&) = delete;
  Prefiller& operator=(const Prefiller&) = delete;

  // Tokens with a negative id must be covered by `rows` (their embedding is the image placeholder's otherwise).
  void run(State& state, const std::int32_t* tokens, int T, int pos0, float* logits, Tensor* taps,
           std::span<const EmbedRows> rows = {});

  int max_chunk() const { return max_chunk_; }
  std::size_t device_bytes() const;  // scratch held by this object
  std::size_t scratch_bytes() const { return scratch_bytes_; }
  void bind(void* base);
  void zero(cudaStream_t s);

 private:
  const float* const* block_table(const std::vector<const float*>& blocks);

  const Weights& w_;
  const Ablation& abl_;
  int max_chunk_;
  cudaStream_t stream_;
  std::unique_ptr<kernels::prefill::Gemm> gemm_;
  std::size_t layout(char* base);
  DeviceBuffer own_;
  std::size_t taps_bytes_ = 0;
  void* taps_ = nullptr;
  void* gemm_ws_ = nullptr;
  char* base_ = nullptr;
  std::size_t scratch_bytes_ = 0;
  std::map<std::vector<const float*>, DeviceBuffer> tables_;

  // activations (rows = max_chunk), see prefill.cpp
  float *B0_, *B1_, *B2_[MAX_PASSES], *E_[MAX_PASSES], *hid_, *part_, *y_, *y2_;  // B2_/E_: first passes() set
  __nv_bfloat16 *xin_, *xm_, *xf_;
  float *mixf_, *msgs_, *mixm_;
  __nv_bfloat16 *proj_, *q_, *rel16_, *rel_, *krows_, *vrows_, *attn_out_, *wo_out_;
  int* kpos_;
  int4* bounds_;
  __nv_bfloat16 *gin_, *g16_;
  float *fpre_, *gpre_;
  float *router_logits_, *wts_;
  int* idx_;
  int *counts_, *tiles_n_, *pair_token_, *pair_dst_;
  int4* tiles_;
  __nv_bfloat16 *act_, *down_, *moe_sum_;
  __nv_bfloat16* wscratch_;  // dequantized FP8 rows of wproj / wo (dense_fp8)
  std::int64_t* ids_;
  int kpos_rows_;
  // quill: staged experts (host counts, per-group slot maps)
  static constexpr int kMaxGroups = 24;
  cudaStream_t copy_ = nullptr;  // staged expert uploads
  cudaEvent_t ev_start_ = nullptr, ev_up_[kMaxGroups] = {}, ev_done_[kMaxGroups] = {};
  int* hcounts_ = nullptr;  // pinned [128]
  int* hmaps_ = nullptr;    // pinned [kMaxGroups][128]
  int* dmaps_ = nullptr;    // device [kMaxGroups][128]
  std::size_t staged_bytes_ = 0;  // H2D expert bytes of the last run (stats)
 public:
  std::size_t staged_bytes() const { return staged_bytes_; }
  // quill: per (layer, expert) token counts of the prefills since the last take (24 * 128)
  std::vector<std::uint32_t> take_counts() {
    std::vector<std::uint32_t> c(counts_acc_.begin(), counts_acc_.end());
    std::fill(counts_acc_.begin(), counts_acc_.end(), 0u);
    return c;
  }
 private:
  std::vector<std::uint32_t> counts_acc_ = std::vector<std::uint32_t>(LAYERS * EXPERTS, 0u);
};

}  // namespace ingine::model
