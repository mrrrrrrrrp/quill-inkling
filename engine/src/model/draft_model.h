// quill: the drafter interface of the speculative loop and the generator, implemented by the DSpark drafter
// (drafter.h, mtp.safetensors) and the DFlash2 drafter (dflash2.h, incoai/Inkling-2-Mini-DFlash2).
#pragma once

#include "core/tensor.h"
#include "kernels/draft/draft.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace ingine::model {

class DraftModel {
 public:
  struct Draft {
    const std::int32_t* tokens;  // int32 [7]
    const float* logits;         // fp32 [7, V]: the draft distributions' logits, -inf outside their support
    const float* conf;           // fp32 [7]: max softmax probability per row
    const float* qstats;         // fp32 [7, 2]: sampled drafts' q statistics
  };
  struct Region {
    std::size_t offset, bytes;
  };
  struct AcceptScratch {
    std::int32_t* target;
    float* probs;
    float* resid;
    float* uniforms;
    float* ws;
    kernels::draft::AcceptResult* result;
  };

  virtual ~DraftModel() = default;
  virtual void capture() = 0;
  virtual void reset() = 0;
  virtual int length() const = 0;
  virtual void commit(const __nv_bfloat16* feats, int n, int start) = 0;
  void commit(const Tensor& feats, int start) {
    if (feats.dtype != DType::BF16 || feats.ne[0] != 2304 || feats.ne[1] != 4 || !feats.is_contiguous())
      throw std::invalid_argument("DraftModel::commit: feats must be contiguous bf16 [n, 4, 2304]");
    commit(static_cast<const __nv_bfloat16*>(feats.data), feats.ne[2] * feats.ne[3], start);
  }
  virtual Draft draft(int anchor, int P, float temperature) = 0;
  virtual void set_seed(std::uint64_t seed) = 0;
  virtual void note_tokens(std::span<const std::int32_t> ids) = 0;
  virtual void bind_round(const kernels::draft::RoundInputs& in) = 0;
  virtual Draft advance(int start) = 0;
  virtual void advanced(int n) = 0;
  virtual std::size_t checkpoint_bytes() const = 0;
  virtual void save_checkpoint(void* dst) = 0;
  virtual void restore_checkpoint(const void* src) = 0;
  virtual std::array<Region, 3> checkpoint_regions(int length) const = 0;
  virtual const std::vector<std::int32_t>& checkpoint_context() const = 0;
  virtual void adopt_checkpoint(const void* src, int length, std::span<const std::int32_t> context) = 0;
  virtual int vocab() const = 0;
  virtual const AcceptScratch& accept_scratch() const = 0;
  virtual cudaStream_t stream() const = 0;
};

}  // namespace ingine::model
