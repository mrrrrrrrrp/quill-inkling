#include "model/vmm_region.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

namespace ingine::model {
namespace {

CUmemAllocationProp prop_for(int device) {
  CUmemAllocationProp p{};
  p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  p.location.id = device;
  return p;
}

}  // namespace

bool VmmRegion::init(std::size_t bytes, std::size_t chunk_bytes) {
  if (cudaGetDevice(&device_) != cudaSuccess) return false;
  cudaFree(nullptr);  // the primary context is current for the driver calls
  CUdevice dev = 0;
  if (cuDeviceGet(&dev, device_) != CUDA_SUCCESS) return false;
  int vmm = 0;
  if (cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev) != CUDA_SUCCESS || !vmm)
    return false;
  const CUmemAllocationProp p = prop_for(device_);
  std::size_t g = 0;
  if (cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS || g == 0) return false;
  chunk_ = (chunk_bytes + g - 1) / g * g;
  const std::size_t n = bytes / chunk_;
  if (n == 0) return false;
  reserved_ = n * chunk_;
  CUdeviceptr va = 0;
  if (cuMemAddressReserve(&va, reserved_, 0, 0, 0) != CUDA_SUCCESS) return false;
  base_ = reinterpret_cast<char*>(va);
  handles_.assign(n, 0);
  for (std::size_t i = 0; i < n; ++i)
    if (!map(i)) {
      spdlog::warn("vmm: mapping chunk {} of {} failed; falling back to a plain allocation", i, n);
      for (std::size_t j = 0; j < n; ++j) unmap(j);
      cuMemAddressFree(va, reserved_);
      base_ = nullptr;
      handles_.clear();
      return false;
    }
  return true;
}

bool VmmRegion::map(std::size_t i) {
  if (handles_[i]) return true;
  const CUmemAllocationProp p = prop_for(device_);
  CUmemGenericAllocationHandle h = 0;
  if (cuMemCreate(&h, chunk_, &p, 0) != CUDA_SUCCESS) return false;
  const CUdeviceptr at = reinterpret_cast<CUdeviceptr>(base_) + i * chunk_;
  if (cuMemMap(at, chunk_, 0, h, 0) != CUDA_SUCCESS) {
    cuMemRelease(h);
    return false;
  }
  CUmemAccessDesc d{};
  d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  d.location.id = device_;
  d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (cuMemSetAccess(at, chunk_, &d, 1) != CUDA_SUCCESS) {
    cuMemUnmap(at, chunk_);
    cuMemRelease(h);
    return false;
  }
  handles_[i] = h;
  return true;
}

void VmmRegion::unmap(std::size_t i) {
  if (!handles_[i]) return;
  const CUdeviceptr at = reinterpret_cast<CUdeviceptr>(base_) + i * chunk_;
  cuMemUnmap(at, chunk_);
  cuMemRelease(static_cast<CUmemGenericAllocationHandle>(handles_[i]));
  handles_[i] = 0;
}

std::size_t VmmRegion::mapped_bytes() const {
  std::size_t n = 0;
  for (auto h : handles_) n += h ? chunk_ : 0;
  return n;
}

VmmRegion::~VmmRegion() {
  if (!base_) return;
  cudaDeviceSynchronize();
  for (std::size_t i = 0; i < handles_.size(); ++i) unmap(i);
  cuMemAddressFree(reinterpret_cast<CUdeviceptr>(base_), reserved_);
}

}  // namespace ingine::model
