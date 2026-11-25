// quill: IQ4_NL q4 rows of the full-attention caches, shared by the decode attention (attn89.cu) and the
// prefill (rows.cu, attention.cu) so both write and read exactly the same values.
//
// Row layout (one position, 8 KV heads x 128 dims): bytes [0, 512) codes, head h at 64 h, dims (2i, 2i+1) of a
// head in byte i (low, high nibble); bytes [512, 576) fp16 block scales, head h at 512 + 8 h, one per 32 dims.
// Value = kIQ4NLv[code] * scale. Quantisation of a block of 32 (llama.cpp cpy_blck_f32_iq4_nl): scale from the
// signed absolute maximum, nearest table entries, then the least-squares scale weighted by x^2.
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

namespace ingine::kernels {

inline constexpr int kQ4Row = 576, kQ4Codes = 512;

__device__ __forceinline__ float iq4nl_value(int i) {
  constexpr float t[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
                           1.f,    13.f,   25.f,  38.f,  53.f,  69.f,  89.f,  113.f};
  return t[i];
}

// x: 32 values (overwritten with the dequantised values); codes: 16 bytes; scale: one fp16
__device__ inline void iq4nl_quant32(float* x, uint8_t* codes, __half* scale) {
  float amax = 0.f, vmax = 0.f;
  for (int j = 0; j < 32; ++j)
    if (amax < fabsf(x[j])) amax = fabsf(x[j]), vmax = x[j];
  const float d0 = vmax / -127.f;
  const float id = d0 != 0.f ? 1.f / d0 : 0.f;
  uint8_t q[32];
  float sumqx = 0.f, sumq2 = 0.f;
  for (int j = 0; j < 32; ++j) {
    const float v = x[j] * id;
    int best = 0;
    float bd = fabsf(v - iq4nl_value(0));
    for (int k = 1; k < 16; ++k) {
      const float dd = fabsf(v - iq4nl_value(k));
      if (dd < bd) bd = dd, best = k;
    }
    q[j] = uint8_t(best);
    const float w = x[j] * x[j], t = iq4nl_value(best);
    sumqx += w * t * x[j];
    sumq2 += w * t * t;
  }
  const float d = sumq2 > 0.f ? sumqx / sumq2 : d0;
  const __half dh = __float2half_rn(d);
  const float df = __half2float(dh);
  for (int j = 0; j < 16; ++j) codes[j] = uint8_t(q[2 * j] | (q[2 * j + 1] << 4));
  for (int j = 0; j < 32; ++j) x[j] = iq4nl_value(q[j]) * df;
  *scale = dh;
}

}  // namespace ingine::kernels
