// quill: DFlash2 drafter (incoai/Inkling-2-Mini-DFlash2) behind the DraftModel interface. Kernels and the model
// math: kernels/draft/dflash2.h. Context lives in one ring of WIN = 2048 positions per layer (fixed shape: one
// CUDA graph serves every context length).
#pragma once

#include "core/arena.h"
#include "core/host_memory.h"

#include <memory>
#include "kernels/dense/dense.h"
#include "kernels/draft/dflash2.h"
#include "model/draft_model.h"
#include "model/model_data.h"

#include <filesystem>
#include <string>
#include <vector>

namespace ingine::model {

class DFlash2 final : public DraftModel {
 public:
  DFlash2(const std::filesystem::path& draft_dir, const Weights& w, cudaStream_t stream);
  ~DFlash2() override;

  void capture() override;
  void reset() override;
  int length() const override { return length_; }
  using DraftModel::commit;
  void commit(const __nv_bfloat16* feats, int n, int start) override;
  Draft draft(int anchor, int P, float temperature) override;
  void set_seed(std::uint64_t seed) override;
  void note_tokens(std::span<const std::int32_t>) override {}
  void bind_round(const kernels::draft::RoundInputs& in) override;
  Draft advance(int start) override;
  void advanced(int n) override { length_ = pending_start_ + n; }
  std::size_t checkpoint_bytes() const override;
  void save_checkpoint(void* dst) override;
  void restore_checkpoint(const void* src) override;
  std::array<Region, 3> checkpoint_regions(int length) const override;
  const std::vector<std::int32_t>& checkpoint_context() const override { return no_context_; }
  void adopt_checkpoint(const void* src, int length, std::span<const std::int32_t> context) override;
  int vocab() const override { return V_; }
  const AcceptScratch& accept_scratch() const override { return accept_; }
  cudaStream_t stream() const override { return stream_; }
  std::size_t device_bytes() const;

 private:
  struct Layer {
    const __nv_bfloat16 *in_norm, *post_norm, *q_norm, *k_norm;
    kernels::dense::Matrix qkv, kv, cpa, cpm, wo, gate_up, down;
    const float *base_a, *base_m;  // fp32 [2][2][H]
    __nv_bfloat16 *k_ring, *v_ring;  // [SLOTS, 8, 128]
  };
  void* alloc(std::size_t bytes);
  void load(const std::filesystem::path& dir);
  void run_inject();  // the staged rows in feats_ -> ring
  void run_draft(float* debug_x = nullptr);
  void dump(const std::string& dir);
  bool dumped_ = false;
  int last_commit_start_ = 0, last_commit_n_ = 0;
  void run_advance();
  Draft outputs() const { return {tokens_, draft_logits_, conf_, qstats_}; }

  const Weights& w_;
  cudaStream_t stream_;
  int V_ = 0;
  int length_ = 0, pending_start_ = 0;
  std::vector<DeviceBuffer> storage_;
  Layer layers_[kernels::dflash2::LAYERS]{};
  kernels::dense::Matrix fc_{}, sel_{}, head_{};
  const __nv_bfloat16 *hidden_norm_ = nullptr, *final_norm_ = nullptr, *pred_ = nullptr, *succ_ = nullptr;
  kernels::dflash2::Df2Params* params_ = nullptr;
  std::int32_t* ring_pos_ = nullptr;
  int *seg_qkv_ = nullptr, *seg_kv_ = nullptr, *seg_dyn_ = nullptr, *seg_gu_ = nullptr, *seg_head_ = nullptr,
      *seg_sel_ = nullptr;
  // activations
  __nv_bfloat16 *feats_ = nullptr, *g_ = nullptr, *xn_ = nullptr, *xc_ = nullptr, *q_ = nullptr, *attn_ = nullptr,
                *act_ = nullptr, *h_ = nullptr;
  float *fcout_ = nullptr, *kvout_ = nullptr, *x_ = nullptr, *dyn_ = nullptr, *qkv_out_ = nullptr, *o_ = nullptr,
        *gu_ = nullptr, *logits_ = nullptr, *cand_val_ = nullptr, *gate_ = nullptr, *draft_logits_ = nullptr,
        *conf_ = nullptr, *qstats_ = nullptr;
  std::int32_t *cand_id_ = nullptr, *prev_cand_ = nullptr, *tokens_ = nullptr;
  const std::int32_t* head_tok_ = nullptr;  // quill: head row -> token (frequency-ordered rows), null: identity
  AcceptScratch accept_{};
  cudaGraphExec_t draft_graph_ = nullptr, commit_graph_ = nullptr, advance_graph_ = nullptr;
  kernels::draft::RoundInputs round_{};
  bool round_bound_ = false;
  const void* checkpoint_ = nullptr;
  int checkpoint_length_ = 0;
  std::vector<std::int32_t> no_context_;
  std::vector<std::shared_ptr<MappedHostBuffer>> host_bufs_;
  float draft_ts_ = 1.f;
  bool draft_fp4_ = true;  // quill: drafter matrices as FP4 E2M1 rows (QUILL_DRAFT_FP4=0: FP6); +1.4-2.2% with the tier CPU-bound
  bool draft_fp6_ = true;  // quill: drafter matrices as FP6 E2M3 rows (QUILL_DRAFT_FP6)
  int Vh_ = 0;  // quill: rows of the draft head (QUILL_DRAFT_VOCAB), <= V_  // quill: draft temperature / sampling temperature (QUILL_DRAFT_TS)
};

}  // namespace ingine::model
