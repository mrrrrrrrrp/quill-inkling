// quill: hybrid routed-expert storage for a GPU that cannot hold every expert.
//
// Every expert of every layer lives in pinned host memory in its checkpoint encoding (NVFP4: codes + E4M3
// block scales + an FP32 factor; MXFP8: E4M3 codes + E8M0 block scales). The GPU holds a pool of expert
// slots per format, each slot one expert in the same per-expert layout the kernels index by row:
//   w13 [slots, 2I, H/2 | H] codes, w13_scale [slots, 2I, H/16 | H/32], w2 [slots, H, I/2 | I],
//   w2_scale [slots, H, I/16 | I/32], factor13 / factor2 fp32 [slots] (NVFP4).
// slot_map (device int32 [24][128]) maps (layer, expert) -> resident slot or -1. The last `staging` slots of
// each pool are scratch for the prefill, which streams non-resident experts into them per layer visit.
// Decode: resident experts run on the GPU, the others on the CPU tier (cpu_tier.h).
#pragma once

#include "core/arena.h"
#include "core/host_memory.h"
#include "model/model_data.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace ingine::model {

struct ExpertBytes {  // bytes of one expert's four arrays
  std::size_t w13 = 0, w13s = 0, w2 = 0, w2s = 0;
  std::size_t total() const { return w13 + w13s + w2 + w2s; }
};
ExpertBytes expert_bytes(ExpertFormat fmt);

struct HostLayerExperts {
  ExpertFormat fmt = ExpertFormat::NVFP4;
  std::byte* w13 = nullptr;   // [128] experts, each ExpertBytes::w13
  std::byte* w13s = nullptr;
  std::byte* w2 = nullptr;
  std::byte* w2s = nullptr;
  std::array<float, EXPERTS> f13{}, f2{};  // NVFP4 factors (1 for MXFP8)
  const std::byte* expert_w13(int e) const;
  const std::byte* expert_w13s(int e) const;
  const std::byte* expert_w2(int e) const;
  const std::byte* expert_w2s(int e) const;
};

struct ExpertPool {
  ExpertFormat fmt = ExpertFormat::NVFP4;
  int slots = 0;    // resident + staging
  int staging = 0;  // the last `staging` slots
  DeviceBuffer w13, w13s, w2, w2s, f13, f2;
  int resident_slots() const { return slots - staging; }
  // quill shared region: slot s is one record [w13 | w13s | w2 | w2s] at base + s * stride (stride 0: the four
  // per-array buffers above). The region's low addresses hold the full-attention KV, growing upwards.
  char* base = nullptr;
  long long stride = 0;
  std::size_t off13s = 0, off2 = 0, off2s = 0;
  const void* w13p() const { return stride ? static_cast<const void*>(base) : w13.p; }
  const void* w13sp() const { return stride ? static_cast<const void*>(base + off13s) : w13s.p; }
  const void* w2p() const { return stride ? static_cast<const void*>(base + off2) : w2.p; }
  const void* w2sp() const { return stride ? static_cast<const void*>(base + off2s) : w2s.p; }
};

class ExpertStore {
 public:
  ExpertStore() = default;
  ExpertStore(const ExpertStore&) = delete;
  ExpertStore& operator=(const ExpertStore&) = delete;

  // Reads every layer's experts from the checkpoint into pinned host memory.
  // mxfp8_as_nvfp4: MXFP8 layers (20-23) are re-encoded as NVFP4 (half the bytes; lossy) and formats() says so.
  void load_host(const std::filesystem::path& model_dir, const std::array<ExpertFormat, LAYERS>& formats,
                 bool mxfp8_as_nvfp4 = false);

  // Allocates the device pools: `budget_bytes` of resident slots split over the formats by priority, plus
  // `staging` slots per format. Fills the resident slots from `priority` (layer, expert) pairs, best first
  // (empty: loop layers first, then the others, NVFP4 before MXFP8).
  struct Pair {
    int layer, expert;
  };
  void build_pools(std::size_t budget_bytes, int staging, std::vector<Pair> priority, cudaStream_t stream);
  // quill: NVFP4 experts only. Slots fill [region, region + bytes) from the top; [region, region + kv_max) may be
  // claimed by the KV cache (ensure below). Throws when kv_max plus the staging slots do not fit.
  void build_pools_shared(char* region, std::size_t bytes, std::size_t kv_max, int staging,
                          std::vector<Pair> priority, cudaStream_t stream);
  bool shared() const { return dev_region_ != nullptr; }
  // Slots below the KV front (region + front_bytes) are unmapped (stream-ordered); returns how many.
  int evict_below(std::size_t front_bytes, cudaStream_t stream);
  // First resident slot of the NVFP4 pool at or above the KV front (shared region; 0 otherwise).
  int first_usable(std::size_t front_bytes) const;

  const HostLayerExperts& host(int l) const { return host_[l]; }
  // quill: device-visible alias of a pointer into the host expert region (registered, mapped), else nullptr
  const unsigned char* host_device(const void* p) const {
    if (!region_.device_ptr()) return nullptr;
    return static_cast<const unsigned char*>(region_.device_ptr()) + (static_cast<const std::byte*>(p) - region_.data());
  }
  const float* pinned_factors_device(int l);  // pinned [128][2] factors of layer l (device-visible)
  ExpertPool& pool(ExpertFormat f) { return pools_[int(f)]; }
  const ExpertPool& pool(ExpertFormat f) const { return pools_[int(f)]; }
  const std::int32_t* slot_map(int l) const { return static_cast<const std::int32_t*>(slot_map_.p) + l * EXPERTS; }
  std::int32_t host_slot(int l, int e) const { return host_map_[l][e]; }
  int resident_count(int l) const;
  int resident_total() const;

  // Copies (layer, expert) from host memory into `slot` of its format's pool (stream-ordered H2D).
  // factors = false: the slot's f13 / f2 are set elsewhere (prefill staging: set_stage_factors)
  void upload(int l, int e, int slot, cudaStream_t stream, bool factors = true) const;
  // Default priority order (see build_pools).
  static std::vector<Pair> default_priority(const std::array<ExpertFormat, LAYERS>& formats);

  // quill adaptive residency: (layer, expert) -> slot (or -1) in the host map and, stream-ordered, the device map
  // (4-byte copies from a pinned ring). slot_owner(f, slot) = layer * 128 + expert or -1.
  void set_slot(int l, int e, int slot, cudaStream_t stream);
  int slot_owner(ExpertFormat f, int slot) const { return owners_[int(f)][slot]; }

  std::size_t host_bytes() const { return host_bytes_; }
  std::size_t device_bytes() const;
  const std::array<ExpertFormat, LAYERS>& formats() const { return formats_; }

 private:
  HostRegion region_;
  std::size_t host_bytes_ = 0;
  std::array<ExpertFormat, LAYERS> formats_{};
  std::array<HostLayerExperts, LAYERS> host_{};
  std::array<ExpertPool, 2> pools_{};
  DeviceBuffer slot_map_;
  std::array<std::array<std::int32_t, EXPERTS>, LAYERS> host_map_{};
  std::array<std::vector<int>, 2> owners_;
  std::int32_t* ring_ = nullptr;  // pinned [4096] sources of the device map updates
  float* fac_ = nullptr;          // pinned [LAYERS][EXPERTS][2] (f13, f2): async-safe upload sources
  const float* pinned_factors(int l, int e);
  char* dev_region_ = nullptr;
  std::size_t dev_region_bytes_ = 0;
  int ring_pos_ = 0;
 public:
  ~ExpertStore();
};

}  // namespace ingine::model
