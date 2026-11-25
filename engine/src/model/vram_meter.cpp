// quill: device memory in use as the driver reports it to nvidia-smi and Task Manager (NVML, all processes).
//
// On Windows (WDDM) cudaMemGetInfo is a per-process view: measured on this machine it read 50-660 MiB below
// nvidia-smi, and the gap moves as allocations become resident. The VRAM cap is a Task Manager number, so the
// expert pools are sized against NVML. nvml.dll ships with the driver; it is loaded at run time, and without it
// the meter falls back to cudaMemGetInfo.
#include "model/vram_meter.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "core/arena.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <chrono>
#include <thread>

namespace ingine::model {
namespace {

struct NvmlMemory {
  unsigned long long total, free, used;
};

class Nvml {
 public:
  Nvml() {
#ifdef _WIN32
    HMODULE h = LoadLibraryA("nvml.dll");
    if (!h) h = LoadLibraryA("C:\\Windows\\System32\\nvml.dll");
    if (!h) return;
    auto sym = [&](const char* n) { return reinterpret_cast<void*>(GetProcAddress(h, n)); };
#else
    void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!h) return;
    auto sym = [&](const char* n) { return dlsym(h, n); };
#endif
    auto init = reinterpret_cast<int (*)()>(sym("nvmlInit_v2"));
    auto by_pci = reinterpret_cast<int (*)(const char*, void**)>(sym("nvmlDeviceGetHandleByPciBusId_v2"));
    mem_ = reinterpret_cast<int (*)(void*, NvmlMemory*)>(sym("nvmlDeviceGetMemoryInfo"));
    if (!init || !by_pci || !mem_ || init() != 0) return;
    int dev = 0;
    char bus[32] = {};
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetPCIBusId(bus, sizeof bus, dev) != cudaSuccess) return;
    if (by_pci(bus, &device_) != 0) device_ = nullptr;
  }
  bool ok() const { return device_ && mem_; }
  std::size_t used() const {
    NvmlMemory m{};
    return ok() && mem_(device_, &m) == 0 ? std::size_t(m.used) : 0;
  }

 private:
  void* device_ = nullptr;
  int (*mem_)(void*, NvmlMemory*) = nullptr;
};

const Nvml& nvml() {
  static const Nvml n;
  return n;
}

std::size_t cuda_used() {
  std::size_t f = 0, t = 0;
  return cudaMemGetInfo(&f, &t) == cudaSuccess ? t - f : 0;
}

}  // namespace

bool vram_meter_nvml() { return nvml().ok(); }

std::size_t vram_used_bytes() { return std::max(nvml().used(), cuda_used()); }

std::size_t vram_used_settled_bytes(int max_ms) {
  // NVML picks up a new allocation with a delay: wait for two reads 50 ms apart within 4 MiB
  std::size_t prev = vram_used_bytes();
  for (int waited = 0; waited < max_ms; waited += 50) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const std::size_t now = vram_used_bytes();
    const std::size_t d = now > prev ? now - prev : prev - now;
    if (d < (std::size_t(4) << 20)) return std::max(now, prev);
    prev = now;
  }
  return prev;
}

void vram_phase(const char* what) {
  static std::size_t last_nvml = 0, last_live = 0;
  cudaDeviceSynchronize();
  const std::size_t nv = vram_used_settled_bytes(800), live = device_buffer_live_bytes();
  spdlog::info("vram phase {:<28s} NVML {:6.0f} MiB ({:+6.0f})  buffers {:6.0f} MiB ({:+6.0f})  hidden {:+6.0f}", what,
               nv / 1048576.0, (double(nv) - double(last_nvml)) / 1048576.0, live / 1048576.0,
               (double(live) - double(last_live)) / 1048576.0,
               ((double(nv) - double(last_nvml)) - (double(live) - double(last_live))) / 1048576.0);
  last_nvml = nv, last_live = live;
}

}  // namespace ingine::model
