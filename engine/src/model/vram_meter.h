// quill: the VRAM meter the cap is enforced with (see vram_meter.cpp).
#pragma once

#include <cstddef>

namespace ingine::model {

// true when NVML is available (else the meter is cudaMemGetInfo alone)
bool vram_meter_nvml();
// device memory in use by all processes: max(NVML, cudaMemGetInfo)
std::size_t vram_used_bytes();
// the same once the reading has stopped moving (NVML lags new allocations), waiting at most max_ms
std::size_t vram_used_settled_bytes(int max_ms = 1500);
// quill diagnostics: logs NVML (settled) and the live DeviceBuffer bytes with their deltas since the last call
void vram_phase(const char* what);

}  // namespace ingine::model
