// quill: a device region of fixed virtual addresses backed by physical chunks that can be released and re-created
// at run time (CUDA virtual memory management). The shared KV/expert region lives in one, so the engine can give
// VRAM back when other applications grow and take it again when they shrink (Runtime::vram_check), instead of
// holding a large static reserve under the cap.
#pragma once

#include <cstddef>
#include <vector>

namespace ingine::model {

class VmmRegion {
 public:
  // false (and nothing reserved) when the device or driver lacks virtual memory management
  bool init(std::size_t bytes, std::size_t chunk_bytes);
  ~VmmRegion();
  VmmRegion() = default;
  VmmRegion(const VmmRegion&) = delete;
  VmmRegion& operator=(const VmmRegion&) = delete;

  char* base() const { return base_; }
  std::size_t bytes() const { return chunk_ * handles_.size(); }
  std::size_t chunk() const { return chunk_; }
  std::size_t chunks() const { return handles_.size(); }
  bool mapped(std::size_t i) const { return handles_[i] != 0; }
  std::size_t mapped_bytes() const;
  // chunk i: physical memory created and mapped / unmapped and released (no-ops when already in that state).
  // unmap: the caller guarantees no kernel touches the chunk any more (stream synchronized).
  bool map(std::size_t i);
  void unmap(std::size_t i);

 private:
  char* base_ = nullptr;
  std::size_t chunk_ = 0, reserved_ = 0;
  std::vector<unsigned long long> handles_;  // CUmemGenericAllocationHandle, 0 = unmapped
  int device_ = 0;
};

}  // namespace ingine::model
