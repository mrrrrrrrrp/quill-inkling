// CPU twin of VisionEncoder (model/vision.h) for RuntimeOptions::vision_cpu: the same tower, token layout and
// preprocessing, computed on the host in FP32. The matrices stay BF16 in host memory as the checkpoint stores them
// (636 MB) and are widened inside the products, so the tower takes no GPU memory. encode() returns the rows as the
// device buffer VisionEncoder::encode does (BF16 [padded tokens, 2304]), so the generator treats both alike.
#pragma once

#include "core/arena.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>

namespace ingine::model {

namespace detail {
struct VisionCpuState;  // weights and the worker pool (vision_cpu.cpp)
}

class VisionCpuEncoder {
 public:
  // threads: the tower's threads in all, the calling thread included
  VisionCpuEncoder(const std::filesystem::path& model_dir, int threads);
  ~VisionCpuEncoder();
  VisionCpuEncoder(const VisionCpuEncoder&) = delete;
  VisionCpuEncoder& operator=(const VisionCpuEncoder&) = delete;

  // rgb u8 [height, width, 3] (host) -> bf16 [padded_tokens(N), 2304] on the device; rows [0, N) are the image's
  // N = ceil(height / 40) * ceil(width / 40) tokens in raster order.
  DeviceBuffer encode(const std::uint8_t* rgb, int height, int width);

  [[nodiscard]] std::size_t host_bytes() const;

 private:
  std::unique_ptr<detail::VisionCpuState> state_;
};

}  // namespace ingine::model
