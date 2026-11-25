// quill: decode attention core for sm_89 (RTX 4070 SUPER), contract of native/KERNELS.md area `attn`.
//
// Three graph-capturable kernels per layer visit (M = 1..8 consecutive tokens):
//  1. attn89_prep   grid (8 KV heads, 5): y = 0: the head's key/value short convolutions, key clip, value RMS
//                   norm, rounding to the cache precision (BF16, or IQ4_NL q4 blocks for full layers), the new
//                   conv state, and q / rel post-processing of query heads 3g..3g+2 (scale, L2 clip, BF16, log
//                   scaling); y = 1..4: the relative-bias table rel_logits[m][h][d] for d in a quarter of 0..1023.
//  2. attn89_split  grid (SMAX, 8, M): flash-decoding over a contiguous range of cached keys; the last active
//                   split also takes the call's new keys (causal within the call). Half a warp per key, lane c
//                   holds dims 8c..8c+7; online softmax in the log2 domain, FP32 accumulators.
//  3. attn89_combine grid (M, 8): merges the split partials with the sink logit, applies the sigmoid gate
//                   (bf16(bf16(o) * sigmoid(gate))) and then writes the call's keys / values into the cache:
//                   sliding ring slots p % 1024 + kpos, or full-layer rows. Writing after every split has read
//                   keeps ring entries that earlier tokens of the same call still need (M > 1).
// q4 cache rows (full layers): [rows][576 B] = 8 heads x 64 B of 4-bit codes (dims 2i, 2i+1 in byte i: low,
// high nibble) then 8 heads x 4 fp16 block scales (one per 32 dims); values = kIQ4NL[code] * scale.
// Every read of a q4 entry, the call's own new keys included, uses the dequantized value the cache holds.
#include "kernels/attn/attn.h"

#include "core/device.h"
#include "kernels/common/iq4nl.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <type_traits>

namespace ingine::kernels::attn {
namespace {

constexpr int HQ = 24, HKV = 8, D = 128, DREL = 16, EXT = 1024, WIN = 1024, GQA = 3, CONV_C = HKV * D;
constexpr int OFF_GATE = 3072, OFF_REL = 6144, OFF_SINK = 6528, OFF_K = 6552, OFF_V = 7576, OFF_KDYN = 8600,
              OFF_VDYN = 8632;
constexpr int R_SLIDING = 8600, R_FULL = 8664;
constexpr int MAXM = 8;
constexpr float CLIP_QK = 22.627416997969522f, CLIP_REL = 8.f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float QSCALE = 0.08838834764831845f * LOG2E;  // 1/sqrt(128) * log2(e)
constexpr float NEG_INIT = -1e30f;
constexpr float INV_SQRT5 = 1.f / 2.2360679774997896f;
constexpr float INV_3 = 1.f / 3.f;
constexpr int SL_CHUNK = 128;  // quill: sliding-window split (keys per block; WIN / SL_CHUNK splits)
constexpr int SL_STAGE = SL_CHUNK * (2 * 128 + 2 * 8) + SL_CHUNK * 4 + 3 * SL_CHUNK * 4;  // q8 staging bytes
constexpr int RED_O_BYTES = 16 * 3 * 128 * 4;
constexpr int SMAX = 64;     // max splits per KV head (quill: 128 x 128-key chunks measured neutral at 22k)
constexpr int PSTR = D + 4;  // partial: o[128], m, l, pad
constexpr int Q4ROW = 576;   // bytes per q4 cache row (8 heads)
constexpr int Q4CODES = 512;

__constant__ float kIQ4NL[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
                                 1.f,    13.f,   25.f,  38.f,  53.f,  69.f,  89.f,  113.f};

// workspace layout (bytes)
constexpr size_t WS_QS = 0;                                        // fp32 [8][24][128] q * QSCALE
constexpr size_t WS_RELB = WS_QS + size_t(MAXM) * HQ * D * 4;      // fp32 [8][24][1024] bias * LOG2E
constexpr size_t WS_NK = WS_RELB + size_t(MAXM) * HQ * EXT * 4;    // fp32 [8][8][128] new keys (cache values)
constexpr size_t WS_NV = WS_NK + size_t(MAXM) * HKV * D * 4;       // fp32 [8][8][128] new values
constexpr size_t WS_NQ = WS_NV + size_t(MAXM) * HKV * D * 4;       // q4 rows [8][2][576]
constexpr int Q8ROW = 1088;  // quill: int8 ring rows: 8 heads x 128 B codes, then 8 x 4 fp16 scales (per 32 dims)
constexpr int Q8CODES = 1024;
constexpr size_t WS_PART = WS_NQ + size_t(MAXM) * 2 * Q8ROW;       // fp32 [8][24][SMAX][PSTR]
constexpr size_t WS_END = WS_PART + size_t(MAXM) * HQ * SMAX * PSTR * 4;

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float wsum(float x) {
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
  return x;
}
__device__ __forceinline__ float hsum16(float x) {
#pragma unroll
  for (int o = 1; o < 16; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
  return x;
}
__device__ __forceinline__ float clip_factor(float ss, float t) {
  const float norm = ss > 0.f ? ss * rsqrtf(ss) : 0.f;
  return fminf(1.f, __fdividef(t, norm + 1e-6f));
}
__device__ __forceinline__ float tau_at(int p, int log_scaling) {
  return log_scaling ? 1.f + 0.1f * logf(fmaxf(float(p + 1) * (1.f / 8192.f), 1.f)) : 1.f;
}

// Split geometry shared by split and combine kernels (from the context on the device).
__constant__ int kSplits = SMAX, kChunkMin = 256;
__device__ __forceinline__ void split_geom(int L_old, bool full, int& chunk, int& n_active) {
  if (!full) {
    chunk = SL_CHUNK;
    n_active = WIN / SL_CHUNK;
    return;
  }
  int c = (L_old + kSplits - 1) / kSplits;
  c = (c + 63) & ~63;
  chunk = c < kChunkMin ? kChunkMin : c;
  n_active = (L_old + chunk - 1) / chunk;
  if (n_active < 1) n_active = 1;
}

struct PrepArgs {
  const float* proj;
  int R;
  int M;
  int full;
  int q4;
  long long cap;
  const float* q_scale;
  const float* r_scale;
  const float* rel_proj;
  const float* k_conv;
  const float* v_conv;
  float* conv_k;
  float* conv_v;
  const long long* pos;
  char* ws;
  int q8;
};

__device__ inline void quant_iq4nl_32(float* x, uint8_t* codes, __half* scale) {
  kernels::iq4nl_quant32(x, codes, scale);
}

template <int M>
__global__ void __launch_bounds__(256) attn89_prep(const PrepArgs a) {
  const int g = blockIdx.x, part = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int p0 = int(a.pos[0]);
  if (a.full && p0 + M > a.cap) return;
  char* ws = a.ws;
  __shared__ __align__(16) float rel_s[MAXM][GQA][DREL];  // processed rel vectors (bf16 values)
  __shared__ __align__(16) float kv_s[2][MAXM][D];        // conv outputs, then cache values
  // ---- rel vectors of heads 3g..3g+2 (all parts need them): two (token, head) items per warp, one per half
  for (int it = warp; it < (M * GQA + 1) / 2; it += 8) {
    const int item = it * 2 + (lane >> 4), c = lane & 15;
    const bool valid = item < M * GQA;
    const int m = valid ? item / GQA : 0, i = valid ? item % GQA : 0, h = g * GQA + i;
    const float r = a.proj[size_t(m) * a.R + OFF_REL + h * DREL + c] * a.r_scale[h * DREL + c];
    const float ss = hsum16(r * r);
    float y = bf16r(r * clip_factor(ss, CLIP_REL));
    if (a.full) y = bf16r(y * tau_at(p0 + m, 1));
    if (valid) rel_s[m][i][c] = y;
  }
  __syncthreads();
  if (part > 0) {  // ---- relative-bias table: d in [256 (part-1), 256 part)
    float* relb = reinterpret_cast<float*>(ws + WS_RELB);
    const int d = (part - 1) * 256 + tid;
    float rp[DREL];
#pragma unroll
    for (int c = 0; c < DREL; ++c) rp[c] = __ldg(a.rel_proj + c * EXT + d);
    for (int m = 0; m < M; ++m)
#pragma unroll
      for (int i = 0; i < GQA; ++i) {
        float s = 0.f;
#pragma unroll
        for (int c = 0; c < DREL; ++c) s = fmaf(rel_s[m][i][c], rp[c], s);
        relb[(size_t(m) * HQ + g * GQA + i) * EXT + d] = bf16r(s) * LOG2E;
      }
    return;
  }
  // ---- q rows: one warp per (token, head), lane = 4 dims
  float* qs = reinterpret_cast<float*>(ws + WS_QS);
  for (int it = warp; it < M * GQA; it += 8) {
    const int m = it / GQA, i = it % GQA, h = g * GQA + i;
    const float4 q = reinterpret_cast<const float4*>(a.proj + size_t(m) * a.R + h * D)[lane];
    const float4 s = reinterpret_cast<const float4*>(a.q_scale + h * D)[lane];
    float x[4] = {q.x * s.x, q.y * s.y, q.z * s.z, q.w * s.w};
    const float f = clip_factor(wsum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]), CLIP_QK);
    const float tau = tau_at(p0 + m, a.full);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      x[e] = bf16r(x[e] * f);
      if (a.full) x[e] = bf16r(x[e] * tau);
    }
    reinterpret_cast<float4*>(qs + (size_t(m) * HQ + h) * D)[lane] =
        make_float4(x[0] * QSCALE, x[1] * QSCALE, x[2] * QSCALE, x[3] * QSCALE);
  }
  // ---- key/value convolutions of KV head g (one thread per (k|v, channel))
  {
    const int kv = tid / D, ch = tid % D, cc = g * D + ch;
    float* st = kv ? a.conv_v : a.conv_k;
    float e[3 + MAXM];
#pragma unroll
    for (int j = 0; j < 3; ++j) e[j] = st[j * CONV_C + cc];
#pragma unroll
    for (int m = 0; m < M; ++m) e[3 + m] = a.proj[size_t(m) * a.R + (kv ? OFF_V : OFF_K) + cc];
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float y;
      if (a.full) {  // dynamic conv by lag (0 = current), / 3
        const float* w = kv ? a.v_conv : a.k_conv;  // static [4][1024] by lag
        const float* dyn = a.proj + size_t(m) * a.R + (kv ? OFF_VDYN : OFF_KDYN) + g * 4;
        float o = 0.f;
#pragma unroll
        for (int lag = 0; lag < 4; ++lag) {
          const float wl = __fadd_rn(w[lag * CONV_C + cc], dyn[lag]);
          o = __fadd_rn(o, __fmul_rn(wl, e[m + 3 - lag]));
        }
        y = __fmul_rn(__fadd_rn(o, e[m + 3]), INV_3);
      } else {  // static conv, conv1d order (tap 3 = current), / sqrt(5)
        const float* w = (kv ? a.v_conv : a.k_conv) + cc * 4;
        float o = __fmul_rn(w[0], e[m]);
#pragma unroll
        for (int j = 1; j < 4; ++j) o = __fadd_rn(o, __fmul_rn(w[j], e[m + j]));
        y = __fmul_rn(__fadd_rn(o, e[m + 3]), INV_SQRT5);
      }
      kv_s[kv][m][ch] = bf16r(y);
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) st[j * CONV_C + cc] = e[M + j];
  }
  __syncthreads();
  // key clip / value RMS norm, BF16: one warp per (k|v, token)
  for (int pr = warp; pr < 2 * M; pr += 8) {
    const int kv = pr / M, m = pr % M;
    float4 x = reinterpret_cast<float4*>(&kv_s[kv][m][0])[lane];
    const float ss = wsum(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
    const float f = kv == 0 ? clip_factor(ss, CLIP_QK) : rsqrtf(ss / float(D) + 1e-6f);
    x = make_float4(bf16r(x.x * f), bf16r(x.y * f), bf16r(x.z * f), bf16r(x.w * f));
    __syncwarp();
    reinterpret_cast<float4*>(&kv_s[kv][m][0])[lane] = x;
  }
  __syncthreads();
  if (a.q4) {  // one thread per 32-value block: 2 * M * 4 blocks (quill: sliding rings too)
    uint8_t* nq = reinterpret_cast<uint8_t*>(ws + WS_NQ);
    if (tid < 2 * M * 4) {
      const int kv = tid / (M * 4), m = (tid / 4) % M, b = tid % 4;
      uint8_t* row = nq + (size_t(m) * 2 + kv) * Q4ROW;
      quant_iq4nl_32(&kv_s[kv][m][b * 32], row + g * 64 + b * 16,
                     reinterpret_cast<__half*>(row + Q4CODES) + g * 4 + b);
    }
    __syncthreads();
  }
  if (a.q8) {  // quill: one thread per 32-value block: int8 codes, fp16 scale; kv_s gets the dequantized values
    uint8_t* nq = reinterpret_cast<uint8_t*>(ws + WS_NQ);
    if (tid < 2 * M * 4) {
      const int kv = tid / (M * 4), m = (tid / 4) % M, b = tid % 4;
      float* x = &kv_s[kv][m][b * 32];
      float am = 0.f;
      for (int j = 0; j < 32; ++j) am = fmaxf(am, fabsf(x[j]));
      const __half sh = __float2half(am / 127.f);
      const float sf = __half2float(sh);
      uint8_t* row = nq + (size_t(m) * 2 + kv) * Q8ROW;
      for (int j = 0; j < 32; ++j) {
        const float q = sf > 0.f ? fminf(fmaxf(rintf(x[j] / sf), -127.f), 127.f) : 0.f;
        row[g * 128 + b * 32 + j] = uint8_t(int8_t(q));
        x[j] = q * sf;
      }
      reinterpret_cast<__half*>(row + Q8CODES)[g * 4 + b] = sh;
    }
    __syncthreads();
  }
  float* nk = reinterpret_cast<float*>(ws + WS_NK);
  float* nv = reinterpret_cast<float*>(ws + WS_NV);
  for (int i = tid; i < M * D; i += 256) {
    const int m = i / D, ch = i % D;
    nk[(size_t(m) * HKV + g) * D + ch] = kv_s[0][m][ch];
    nv[(size_t(m) * HKV + g) * D + ch] = kv_s[1][m][ch];
  }
}

struct SplitArgs {
  const void* k_cache;
  const void* v_cache;
  const long long* kpos;
  const long long* pos;
  long long cap;
  int full;
  int q4;
  int M;
  char* ws;
  long long q4_stride;
  int q8;
};

// 8 dims (lane c of a half warp) of one cached key or value row as floats
template <int Q4>  // quill: 0 BF16, 1 IQ4_NL q4, 2 int8 (q8)
__device__ __forceinline__ void load8(const void* cache, long long row, int g, int c, const float* lut, float (&f)[8],
                                      long long stride = Q4ROW) {
  if constexpr (Q4 == 2) {
    const uint8_t* r = static_cast<const uint8_t*>(cache) + size_t(row) * size_t(stride);
    const uint2 q = __ldcs(reinterpret_cast<const uint2*>(r + g * 128 + c * 8));
    const float s = __half2float(__ldcs(reinterpret_cast<const __half*>(r + Q8CODES) + g * 4 + (c >> 2)));
    const uint32_t w[2] = {q.x, q.y};
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = float(int8_t((w[e >> 2] >> (8 * (e & 3))) & 0xffu)) * s;
  } else if constexpr (Q4 == 1) {
    const uint8_t* r = static_cast<const uint8_t*>(cache) + size_t(row) * size_t(stride);
    const uint32_t q = __ldcs(reinterpret_cast<const unsigned int*>(r + g * 64 + c * 4));
    const float s = __half2float(__ldcs(reinterpret_cast<const __half*>(r + Q4CODES) + g * 4 + (c >> 2)));
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = lut[(q >> (4 * e)) & 15u] * s;
  } else {
    const uint4 u = __ldcs(reinterpret_cast<const uint4*>(static_cast<const __nv_bfloat16*>(cache) + (size_t(row) * HKV + g) * D + c * 8));
    f[0] = __uint_as_float(u.x << 16), f[1] = __uint_as_float(u.x & 0xffff0000u);
    f[2] = __uint_as_float(u.y << 16), f[3] = __uint_as_float(u.y & 0xffff0000u);
    f[4] = __uint_as_float(u.z << 16), f[5] = __uint_as_float(u.z & 0xffff0000u);
    f[6] = __uint_as_float(u.w << 16), f[7] = __uint_as_float(u.w & 0xffff0000u);
  }
}

__device__ int g_dbg_on;
__device__ unsigned long long g_dbg_acc[8];  // sums: staged-start, loop-staged, end-loop, end-start, count, kernel span
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ void cpa16(void* sm, const void* g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(unsigned(__cvta_generic_to_shared(sm))), "l"(g) : "memory");
}
__device__ __forceinline__ void cpa8(void* sm, const void* g) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(unsigned(__cvta_generic_to_shared(sm))), "l"(g) : "memory");
}

template <bool FULL, int Q4>
__global__ void __launch_bounds__(256) attn89_split(const SplitArgs a) {
  constexpr int NH = 16;  // half-warps per block
  // quill: the call's rows are the fastest grid dimension: the M blocks of one (split, head group) read the same
  // keys back to back (L2 hits) instead of M passes over the cache (8x the reads at M = 8, ~6 ms per round at 22k)
  const int m = blockIdx.x, g = blockIdx.y, s = blockIdx.z;  // quill: (rows, head groups, splits): one
  // split's key rows are read by all 8 head-group blocks back to back (DRAM page / L2 sector locality)
  const int tid = threadIdx.x, hw = tid >> 4, c = tid & 15;
  const int p0 = int(a.pos[0]);
  if (FULL && p0 + a.M > a.cap) return;
  const int L_old = FULL ? p0 : WIN;
  int chunk, n_active;
  split_geom(L_old, FULL, chunk, n_active);
  if (s >= n_active) return;
  const int P = p0 + m;
  const char* ws = a.ws;
  __shared__ float lut[16];
  __shared__ float red_m[NH][GQA], red_l[NH][GQA];
  extern __shared__ __align__(16) float dyn_sm[];  // quill: red_o [NH][GQA][D] (q8 sliding: the staged split first)
  float (*red_o)[GQA][D] = reinterpret_cast<float (*)[GQA][D]>(dyn_sm);
  if (tid < 16) lut[tid] = kIQ4NL[tid];
  // q (pre-scaled) of the 3 heads, this lane's 8 dims
  float q[GQA][8];
  const float* qs = reinterpret_cast<const float*>(ws + WS_QS);
#pragma unroll
  for (int i = 0; i < GQA; ++i) {
    const float4* qp = reinterpret_cast<const float4*>(qs + (size_t(m) * HQ + g * GQA + i) * D + c * 8);
    const float4 a0 = qp[0], a1 = qp[1];
    q[i][0] = a0.x, q[i][1] = a0.y, q[i][2] = a0.z, q[i][3] = a0.w;
    q[i][4] = a1.x, q[i][5] = a1.y, q[i][6] = a1.z, q[i][7] = a1.w;
  }
  const float* relb = reinterpret_cast<const float*>(ws + WS_RELB) + (size_t(m) * HQ + g * GQA) * EXT;
  __syncthreads();
  float mr[GQA], lr[GQA], acc[GQA][8];
#pragma unroll
  for (int i = 0; i < GQA; ++i) {
    mr[i] = NEG_INIT, lr[i] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
  }
  // every lane of the block runs the same iterations (full-mask shuffles); masked keys score -inf
  auto attend = [&](const float (&kf)[8], const float (&vf)[8], int d, bool ok) {
    float sc[GQA];
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
      float t = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) t = fmaf(q[i][e], kf[e], t);
      t = hsum16(t);
      sc[i] = ok ? t + (d < EXT ? __ldg(relb + i * EXT + d) : 0.f) : -INFINITY;
    }
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
      const float mn = fmaxf(mr[i], sc[i]);
      const float alpha = exp2f(mr[i] - mn), pexp = exp2f(sc[i] - mn);
      lr[i] = lr[i] * alpha + pexp;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][e] = fmaf(acc[i][e], alpha, pexp * vf[e]);
      mr[i] = mn;
    }
  };
  auto attend_b = [&](const float (&kf)[8], const float (&vf)[8], const float (&bias)[GQA]) {
    float sc[GQA];
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
      float t = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) t = fmaf(q[i][e], kf[e], t);
      sc[i] = hsum16(t) + bias[i];  // bias -inf: masked
    }
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
      const float mn = fmaxf(mr[i], sc[i]);
      const float alpha = exp2f(mr[i] - mn), pexp = exp2f(sc[i] - mn);
      lr[i] = lr[i] * alpha + pexp;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][e] = fmaf(acc[i][e], alpha, pexp * vf[e]);
      mr[i] = mn;
    }
  };
  // ---- cached keys [k0, k1)
  const int k0 = s * chunk, k1 = min(L_old, k0 + chunk);
  unsigned long long t_a = 0, t_b = 0, t_c = 0;
  if constexpr (!FULL && Q4 == 2) {  // quill: stage the split, then attend from shared memory
    constexpr int CK = SL_CHUNK;
    if (tid == 0 && g_dbg_on) t_a = gtimer();
    uint8_t* ks = reinterpret_cast<uint8_t*>(dyn_sm);
    uint8_t* vs = ks + CK * 128;
    __half* kss = reinterpret_cast<__half*>(vs + CK * 128);
    __half* vss = kss + CK * 4;
    int* kps = reinterpret_cast<int*>(vss + CK * 4);
    float* rbs = reinterpret_cast<float*>(kps + CK);
    const uint8_t* kc = static_cast<const uint8_t*>(a.k_cache);
    const uint8_t* vc = static_cast<const uint8_t*>(a.v_cache);
    for (int i = tid; i < CK * 18; i += 256) {
      const int j = i / 18, pt = i % 18;
      const size_t row = size_t(k0 + j) * size_t(a.q4_stride);
      if (pt < 8) cpa16(ks + j * 128 + pt * 16, kc + row + g * 128 + pt * 16);
      else if (pt < 16) cpa16(vs + j * 128 + (pt - 8) * 16, vc + row + g * 128 + (pt - 8) * 16);
      else if (pt == 16) cpa8(kss + j * 4, kc + row + Q8CODES + g * 8);
      else cpa8(vss + j * 4, vc + row + Q8CODES + g * 8);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    for (int j = tid; j < CK; j += 256) kps[j] = int(a.kpos[k0 + j]);
    __syncthreads();
    for (int i = tid; i < GQA * CK; i += 256) {
      const int hh = i / CK, j = i % CK;
      const int kp = kps[j], d = P - kp;
      const bool ok = kp >= 0 && kp < p0 && d >= 0 && d < WIN;
      rbs[hh * CK + j] = ok ? __ldg(relb + hh * EXT + d) : -INFINITY;
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    if (tid == 0 && g_dbg_on) t_b = gtimer();
#pragma unroll 2
    for (int jj = hw; jj < CK; jj += NH) {
      const uint2 kq = *reinterpret_cast<const uint2*>(ks + jj * 128 + c * 8);
      const uint2 vq = *reinterpret_cast<const uint2*>(vs + jj * 128 + c * 8);
      const float ksc = __half2float(kss[jj * 4 + (c >> 2)]), vsc = __half2float(vss[jj * 4 + (c >> 2)]);
      const uint32_t kw[2] = {kq.x, kq.y}, vw[2] = {vq.x, vq.y};
      float kf[8], vf[8], bias[GQA];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        kf[e] = float(int8_t((kw[e >> 2] >> (8 * (e & 3))) & 0xffu)) * ksc;
        vf[e] = float(int8_t((vw[e >> 2] >> (8 * (e & 3))) & 0xffu)) * vsc;
      }
#pragma unroll
      for (int i = 0; i < GQA; ++i) bias[i] = rbs[i * CK + jj];
      attend_b(kf, vf, bias);
    }
    if (tid == 0 && g_dbg_on) t_c = gtimer();
  } else
  for (int base = k0; base < k1; base += 2 * NH) {
    float kf[2][8], vf[2][8];
    int kp[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int j = base + u * NH + hw;
      kp[u] = -1;
#pragma unroll
      for (int e = 0; e < 8; ++e) kf[u][e] = vf[u][e] = 0.f;
      if (j < k1) {
        kp[u] = FULL ? j : int(a.kpos[j]);
        load8<Q4>(a.k_cache, j, g, c, lut, kf[u], a.q4_stride);
        load8<Q4>(a.v_cache, j, g, c, lut, vf[u], a.q4_stride);
      }
    }
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int d = P - kp[u];
      const bool ok = kp[u] >= 0 && kp[u] < p0 && d >= 0 && (FULL || d < WIN);
      attend(kf[u], vf[u], ok ? d : EXT, ok);
    }
  }
  // ---- the call's own keys 0..m (last active split)
  if (s == n_active - 1) {  // block-uniform
    const float* nk = reinterpret_cast<const float*>(ws + WS_NK);
    const float* nv = reinterpret_cast<const float*>(ws + WS_NV);
    for (int base = 0; base <= m; base += NH) {
      const int mm = base + hw;
      const bool ok = mm <= m;
      const int ms = ok ? mm : m;
      float kf[8], vf[8];
      const float4* kp4 = reinterpret_cast<const float4*>(nk + (size_t(ms) * HKV + g) * D + c * 8);
      const float4* vp4 = reinterpret_cast<const float4*>(nv + (size_t(ms) * HKV + g) * D + c * 8);
      const float4 k0v = __ldcg(kp4), k1v = __ldcg(kp4 + 1), v0v = __ldcg(vp4), v1v = __ldcg(vp4 + 1);
      kf[0] = k0v.x, kf[1] = k0v.y, kf[2] = k0v.z, kf[3] = k0v.w, kf[4] = k1v.x, kf[5] = k1v.y, kf[6] = k1v.z, kf[7] = k1v.w;
      vf[0] = v0v.x, vf[1] = v0v.y, vf[2] = v0v.z, vf[3] = v0v.w, vf[4] = v1v.x, vf[5] = v1v.y, vf[6] = v1v.z, vf[7] = v1v.w;
      attend(kf, vf, m - ms, ok);
    }
  }
  // ---- merge the 16 half-warps
  if constexpr (!FULL && Q4 == 2) __syncthreads();  // red_o overwrites the staged split
  if (c == 0) {
#pragma unroll
    for (int i = 0; i < GQA; ++i) red_m[hw][i] = mr[i], red_l[hw][i] = lr[i];
  }
#pragma unroll
  for (int i = 0; i < GQA; ++i) {
    float4* o = reinterpret_cast<float4*>(&red_o[hw][i][c * 8]);
    o[0] = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    o[1] = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
  }
  __syncthreads();
  float* part = reinterpret_cast<float*>(a.ws + WS_PART);
  for (int it = tid; it < GQA * D; it += 256) {
    const int i = it / D, d = it % D;
    float M_ = NEG_INIT;
#pragma unroll
    for (int w = 0; w < NH; ++w) M_ = fmaxf(M_, red_m[w][i]);
    float l = 0.f, o = 0.f;
#pragma unroll
    for (int w = 0; w < NH; ++w) {
      const float f = exp2f(red_m[w][i] - M_);
      l = fmaf(red_l[w][i], f, l);
      o = fmaf(red_o[w][i][d], f, o);
    }
    float* pp = part + ((size_t(m) * HQ + g * GQA + i) * SMAX + s) * PSTR;
    pp[d] = o;
    if (d == 0) pp[D] = M_, pp[D + 1] = l;
  }
  if constexpr (!FULL && Q4 == 2) {
    if (tid == 0 && g_dbg_on) {
      const unsigned long long t_d = gtimer();
      atomicAdd(&g_dbg_acc[0], t_b - t_a), atomicAdd(&g_dbg_acc[1], t_c - t_b), atomicAdd(&g_dbg_acc[2], t_d - t_c);
      atomicAdd(&g_dbg_acc[3], t_d - t_a), atomicAdd(&g_dbg_acc[4], 1ull);
      atomicMin(&g_dbg_acc[5], t_a), atomicMax(&g_dbg_acc[6], t_d);
    }
  }
}


// quill: q8 sliding window, all rows per block. Grid (HKV, WIN / SL_CHUNK). Half-warp hw serves row r = hw % M (rank
// k = hw / M of the row's nr half-warps); every half-warp runs the same (masked) trip counts (full-mask shuffles).
constexpr int SL8_STAGE = SL_CHUNK * (2 * 128 + 2 * 8) + SL_CHUNK * 4 + MAXM * 3 * SL_CHUNK * 4;
__global__ void __launch_bounds__(256) attn89_slide8(const SplitArgs a) {
  constexpr int NH = 16, CK = SL_CHUNK;
  const int g = blockIdx.x, s = blockIdx.y;
  const int tid = threadIdx.x, hw = tid >> 4, c = tid & 15;
  const int M = a.M;
  const int p0 = int(a.pos[0]);
  const int r = hw % M, k = hw / M, nr = (NH - r + M - 1) / M, nmin = NH / M;
  const int n_active = WIN / CK;
  __shared__ float red_m[NH][GQA], red_l[NH][GQA];
  extern __shared__ __align__(16) float dyn_sm[];
  uint8_t* ks = reinterpret_cast<uint8_t*>(dyn_sm);
  uint8_t* vs = ks + CK * 128;
  __half* kss = reinterpret_cast<__half*>(vs + CK * 128);
  __half* vss = kss + CK * 4;
  int* kps = reinterpret_cast<int*>(vss + CK * 4);
  float* rbs = reinterpret_cast<float*>(kps + CK);  // [M][GQA][CK]
  const int k0 = s * CK;
  const unsigned long long t_a = (tid == 0 && g_dbg_on) ? gtimer() : 0ull;
  {
    const uint8_t* kc = static_cast<const uint8_t*>(a.k_cache);
    const uint8_t* vc = static_cast<const uint8_t*>(a.v_cache);
    for (int i = tid; i < CK * 18; i += 256) {
      const int j = i / 18, pt = i % 18;
      const size_t row = size_t(k0 + j) * size_t(a.q4_stride);
      if (pt < 8) cpa16(ks + j * 128 + pt * 16, kc + row + g * 128 + pt * 16);
      else if (pt < 16) cpa16(vs + j * 128 + (pt - 8) * 16, vc + row + g * 128 + (pt - 8) * 16);
      else if (pt == 16) cpa8(kss + j * 4, kc + row + Q8CODES + g * 8);
      else cpa8(vss + j * 4, vc + row + Q8CODES + g * 8);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  for (int j = tid; j < CK; j += 256) kps[j] = int(a.kpos[k0 + j]);
  const char* ws = a.ws;
  const float* relb0 = reinterpret_cast<const float*>(ws + WS_RELB) + size_t(g) * GQA * EXT;  // + (mm * HQ + i) * EXT
  // q (pre-scaled) of row r, 3 heads, this lane's 8 dims
  float q[GQA][8];
  const float* qs = reinterpret_cast<const float*>(ws + WS_QS);
#pragma unroll
  for (int i = 0; i < GQA; ++i) {
    const float4* qp = reinterpret_cast<const float4*>(qs + (size_t(r) * HQ + g * GQA + i) * D + c * 8);
    const float4 a0 = qp[0], a1 = qp[1];
    q[i][0] = a0.x, q[i][1] = a0.y, q[i][2] = a0.z, q[i][3] = a0.w;
    q[i][4] = a1.x, q[i][5] = a1.y, q[i][6] = a1.z, q[i][7] = a1.w;
  }
  __syncthreads();
  for (int i = tid; i < M * GQA * CK; i += 256) {
    const int mm = i / (GQA * CK), hh = (i / CK) % GQA, j = i % CK;
    const int kp = kps[j], d = p0 + mm - kp;
    const bool ok = kp >= 0 && kp < p0 && d >= 0 && d < WIN;
    rbs[i] = ok ? __ldg(relb0 + (size_t(mm) * HQ + hh) * EXT + d) : -INFINITY;
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  const unsigned long long t_b = (tid == 0 && g_dbg_on) ? gtimer() : 0ull;
  float mr[GQA], lr[GQA], acc[GQA][8];
#pragma unroll
  for (int i = 0; i < GQA; ++i) {
    mr[i] = NEG_INIT, lr[i] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
  }
  auto attend_b = [&](const float (&kf)[8], const float (&vf)[8], const float (&bias)[GQA]) {
    float sc[GQA];
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
      float t = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) t = fmaf(q[i][e], kf[e], t);
      sc[i] = hsum16(t) + bias[i];
    }
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
      const float mn = fmaxf(mr[i], sc[i]);
      const float alpha = exp2f(mr[i] - mn), pexp = exp2f(sc[i] - mn);
      lr[i] = lr[i] * alpha + pexp;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][e] = fmaf(acc[i][e], alpha, pexp * vf[e]);
      mr[i] = mn;
    }
  };
  const float* rb = rbs + size_t(r) * GQA * CK;
  const int T = (CK + nmin - 1) / nmin;  // block-uniform
  // quill: int8 -> float by the mantissa trick (PRMT + FADD, full rate; I2F is quarter rate), scales applied after
  // the dot (K) / folded into the probability (V), accumulators rescaled only when the running max moves
  auto i8f = [](uint32_t w, float* f) {
    const uint32_t u = w ^ 0x80808080u;
    f[0] = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7650)) - 8388736.f;
    f[1] = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7651)) - 8388736.f;
    f[2] = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7652)) - 8388736.f;
    f[3] = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7653)) - 8388736.f;
  };
#pragma unroll 2
  for (int it = 0; it < T; ++it) {
    const int jj = k + it * nr;
    const bool valid = jj < CK;
    const int j = valid ? jj : 0;
    const uint2 kq = *reinterpret_cast<const uint2*>(ks + j * 128 + c * 8);
    const uint2 vq = *reinterpret_cast<const uint2*>(vs + j * 128 + c * 8);
    const float ksc = __half2float(kss[j * 4 + (c >> 2)]), vsc = __half2float(vss[j * 4 + (c >> 2)]);
    float kf[8], vf[8];
    i8f(kq.x, kf), i8f(kq.y, kf + 4), i8f(vq.x, vf), i8f(vq.y, vf + 4);
    float sc[GQA];
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
      float t = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) t = fmaf(q[i][e], kf[e], t);
      sc[i] = hsum16(t * ksc) + (valid ? rb[i * CK + j] : -INFINITY);
    }
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
      if (sc[i] > mr[i]) {
        const float al = exp2f(mr[i] - sc[i]);
        lr[i] *= al;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[i][e] *= al;
        mr[i] = sc[i];
      }
      const float pe = exp2f(sc[i] - mr[i]);
      lr[i] += pe;
      const float ps = pe * vsc;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][e] = fmaf(ps, vf[e], acc[i][e]);
    }
  }
  const unsigned long long t_c = (tid == 0 && g_dbg_on) ? gtimer() : 0ull;
  // the call's own keys 0..r (last split), shared over the row's half-warps
  if (s == n_active - 1) {
    const float* nk = reinterpret_cast<const float*>(ws + WS_NK);
    const float* nv = reinterpret_cast<const float*>(ws + WS_NV);
    const int T2 = (M + nmin - 1) / nmin;
    for (int it = 0; it < T2; ++it) {
      const int mm = k + it * nr;
      const bool ok = mm <= r;
      const int ms = ok ? mm : r;
      float kf[8], vf[8], bias[GQA];
      const float4* kp4 = reinterpret_cast<const float4*>(nk + (size_t(ms) * HKV + g) * D + c * 8);
      const float4* vp4 = reinterpret_cast<const float4*>(nv + (size_t(ms) * HKV + g) * D + c * 8);
      const float4 k0v = __ldcg(kp4), k1v = __ldcg(kp4 + 1), v0v = __ldcg(vp4), v1v = __ldcg(vp4 + 1);
      kf[0] = k0v.x, kf[1] = k0v.y, kf[2] = k0v.z, kf[3] = k0v.w, kf[4] = k1v.x, kf[5] = k1v.y, kf[6] = k1v.z, kf[7] = k1v.w;
      vf[0] = v0v.x, vf[1] = v0v.y, vf[2] = v0v.z, vf[3] = v0v.w, vf[4] = v1v.x, vf[5] = v1v.y, vf[6] = v1v.z, vf[7] = v1v.w;
#pragma unroll
      for (int i = 0; i < GQA; ++i) bias[i] = ok ? __ldg(relb0 + (size_t(r) * HQ + i) * EXT + (r - ms)) : -INFINITY;
      attend_b(kf, vf, bias);
    }
  }
  // merge each row's half-warps
  if (c == 0) {
#pragma unroll
    for (int i = 0; i < GQA; ++i) red_m[hw][i] = mr[i], red_l[hw][i] = lr[i];
  }
  __syncthreads();  // the staged split is dead: red_o aliases it
  float (*red_o)[GQA][D] = reinterpret_cast<float (*)[GQA][D]>(dyn_sm);
#pragma unroll
  for (int i = 0; i < GQA; ++i) {
    float4* o = reinterpret_cast<float4*>(&red_o[hw][i][c * 8]);
    o[0] = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    o[1] = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
  }
  __syncthreads();
  float* part = reinterpret_cast<float*>(a.ws + WS_PART);
  for (int it = tid; it < M * GQA * D; it += 256) {
    const int mm = it / (GQA * D), i = (it / D) % GQA, d = it % D;
    float M_ = NEG_INIT;
    for (int w = mm; w < NH; w += M) M_ = fmaxf(M_, red_m[w][i]);
    float l = 0.f, o = 0.f;
    for (int w = mm; w < NH; w += M) {
      const float f = exp2f(red_m[w][i] - M_);
      l = fmaf(red_l[w][i], f, l);
      o = fmaf(red_o[w][i][d], f, o);
    }
    float* pp = part + ((size_t(mm) * HQ + g * GQA + i) * SMAX + s) * PSTR;
    pp[d] = o;
    if (d == 0) pp[D] = M_, pp[D + 1] = l;
  }
  if (tid == 0 && g_dbg_on) {
    const unsigned long long t_d = gtimer();
    atomicAdd(&g_dbg_acc[0], t_b - t_a), atomicAdd(&g_dbg_acc[1], t_c - t_b), atomicAdd(&g_dbg_acc[2], t_d - t_c);
    atomicAdd(&g_dbg_acc[3], t_d - t_a), atomicAdd(&g_dbg_acc[4], 1ull);
    atomicMin(&g_dbg_acc[5], t_a), atomicMax(&g_dbg_acc[6], t_d);
  }
}


// quill: full attention over the q4 cache, all rows per block. Grid (HKV, SMAX); split geometry as split_geom.
constexpr int FT = 128;                                       // keys per staged tile
constexpr int F8_STAGE = FT * (2 * 64 + 2 * 8);               // K/V codes + scales of one head group
constexpr int F8_SMEM = F8_STAGE > RED_O_BYTES ? F8_STAGE : RED_O_BYTES;
__global__ void __launch_bounds__(256) attn89_full8(const SplitArgs a) {
  constexpr int NH = 16;
  const int g = blockIdx.x, s = blockIdx.y;
  const int tid = threadIdx.x, hw = tid >> 4, c = tid & 15;
  const int M = a.M;
  const int p0 = int(a.pos[0]);
  if (p0 + M > a.cap) return;
  const int L_old = p0;
  int chunk, n_active;
  split_geom(L_old, true, chunk, n_active);
  if (s >= n_active) return;
  const int r = hw % M, k = hw / M, nr = (NH - r + M - 1) / M, nmin = NH / M;
  const int P = p0 + r;
  __shared__ float lut[16];
  __shared__ float red_m[NH][GQA], red_l[NH][GQA];
  extern __shared__ __align__(16) float dyn_sm[];
  uint8_t* ks = reinterpret_cast<uint8_t*>(dyn_sm);  // [FT][64]
  uint8_t* vs = ks + FT * 64;                        // [FT][64]
  __half* kss = reinterpret_cast<__half*>(vs + FT * 64);  // [FT][4]
  __half* vss = kss + FT * 4;
  if (tid < 16) lut[tid] = kIQ4NL[tid];
  const char* ws = a.ws;
  const float* relb0 = reinterpret_cast<const float*>(ws + WS_RELB) + size_t(g) * GQA * EXT;
  float q[GQA][8];
  const float* qs = reinterpret_cast<const float*>(ws + WS_QS);
#pragma unroll
  for (int i = 0; i < GQA; ++i) {
    const float4* qp = reinterpret_cast<const float4*>(qs + (size_t(r) * HQ + g * GQA + i) * D + c * 8);
    const float4 a0 = qp[0], a1 = qp[1];
    q[i][0] = a0.x, q[i][1] = a0.y, q[i][2] = a0.z, q[i][3] = a0.w;
    q[i][4] = a1.x, q[i][5] = a1.y, q[i][6] = a1.z, q[i][7] = a1.w;
  }
  float mr[GQA], lr[GQA], acc[GQA][8];
#pragma unroll
  for (int i = 0; i < GQA; ++i) {
    mr[i] = NEG_INIT, lr[i] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
  }
  auto attend_b = [&](const float (&kf)[8], const float (&vf)[8], const float (&bias)[GQA]) {
    float sc[GQA];
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
      float t = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) t = fmaf(q[i][e], kf[e], t);
      sc[i] = hsum16(t) + bias[i];
    }
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
      if (sc[i] > mr[i]) {
        const float al = exp2f(mr[i] - sc[i]);
        lr[i] *= al;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[i][e] *= al;
        mr[i] = sc[i];
      }
      const float pe = exp2f(sc[i] - mr[i]);
      lr[i] += pe;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][e] = fmaf(pe, vf[e], acc[i][e]);
    }
  };
  const uint8_t* kc = static_cast<const uint8_t*>(a.k_cache);
  const uint8_t* vc = static_cast<const uint8_t*>(a.v_cache);
  const int k0 = s * chunk, k1 = min(L_old, k0 + chunk);
  for (int t0 = k0; t0 < k1; t0 += FT) {
    const int nt = min(FT, k1 - t0);
    for (int i = tid; i < nt * 10; i += 256) {
      const int j = i / 10, pt = i % 10;
      const size_t row = size_t(t0 + j) * size_t(a.q4_stride);
      if (pt < 4) cpa16(ks + j * 64 + pt * 16, kc + row + g * 64 + pt * 16);
      else if (pt < 8) cpa16(vs + j * 64 + (pt - 4) * 16, vc + row + g * 64 + (pt - 4) * 16);
      else if (pt == 8) cpa8(kss + j * 4, kc + row + Q4CODES + g * 8);
      else cpa8(vss + j * 4, vc + row + Q4CODES + g * 8);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    const int T = (FT + nmin - 1) / nmin;  // block-uniform (padded with masked keys)
#pragma unroll 2
    for (int it = 0; it < T; ++it) {
      const int jj = k + it * nr;
      const bool valid = jj < nt;
      const int j = valid ? jj : 0;
      const uint32_t kq = *reinterpret_cast<const uint32_t*>(ks + j * 64 + c * 4);
      const uint32_t vq = *reinterpret_cast<const uint32_t*>(vs + j * 64 + c * 4);
      const float ksc = __half2float(kss[j * 4 + (c >> 2)]), vsc = __half2float(vss[j * 4 + (c >> 2)]);
      float kf[8], vf[8], bias[GQA];
#pragma unroll
      for (int e = 0; e < 8; ++e) kf[e] = lut[(kq >> (4 * e)) & 15u] * ksc, vf[e] = lut[(vq >> (4 * e)) & 15u] * vsc;
      const int d = P - (t0 + j);
#pragma unroll
      for (int i = 0; i < GQA; ++i)
        bias[i] = !valid ? -INFINITY : d < EXT ? __ldg(relb0 + (size_t(r) * HQ + i) * EXT + d) : 0.f;
      attend_b(kf, vf, bias);
    }
    __syncthreads();  // the tile is restaged next
  }
  if (s == n_active - 1) {  // the call's own keys 0..r
    const float* nk = reinterpret_cast<const float*>(ws + WS_NK);
    const float* nv = reinterpret_cast<const float*>(ws + WS_NV);
    const int T2 = (M + nmin - 1) / nmin;
    for (int it = 0; it < T2; ++it) {
      const int mm = k + it * nr;
      const bool ok = mm <= r;
      const int ms = ok ? mm : r;
      float kf[8], vf[8], bias[GQA];
      const float4* kp4 = reinterpret_cast<const float4*>(nk + (size_t(ms) * HKV + g) * D + c * 8);
      const float4* vp4 = reinterpret_cast<const float4*>(nv + (size_t(ms) * HKV + g) * D + c * 8);
      const float4 k0v = __ldcg(kp4), k1v = __ldcg(kp4 + 1), v0v = __ldcg(vp4), v1v = __ldcg(vp4 + 1);
      kf[0] = k0v.x, kf[1] = k0v.y, kf[2] = k0v.z, kf[3] = k0v.w, kf[4] = k1v.x, kf[5] = k1v.y, kf[6] = k1v.z, kf[7] = k1v.w;
      vf[0] = v0v.x, vf[1] = v0v.y, vf[2] = v0v.z, vf[3] = v0v.w, vf[4] = v1v.x, vf[5] = v1v.y, vf[6] = v1v.z, vf[7] = v1v.w;
#pragma unroll
      for (int i = 0; i < GQA; ++i) bias[i] = ok ? __ldg(relb0 + (size_t(r) * HQ + i) * EXT + (r - ms)) : -INFINITY;
      attend_b(kf, vf, bias);
    }
  }
  if (c == 0) {
#pragma unroll
    for (int i = 0; i < GQA; ++i) red_m[hw][i] = mr[i], red_l[hw][i] = lr[i];
  }
  __syncthreads();
  float (*red_o)[GQA][D] = reinterpret_cast<float (*)[GQA][D]>(dyn_sm);
#pragma unroll
  for (int i = 0; i < GQA; ++i) {
    float4* o = reinterpret_cast<float4*>(&red_o[hw][i][c * 8]);
    o[0] = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    o[1] = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
  }
  __syncthreads();
  float* part = reinterpret_cast<float*>(a.ws + WS_PART);
  for (int it = tid; it < M * GQA * D; it += 256) {
    const int mm = it / (GQA * D), i = (it / D) % GQA, dd = it % D;
    float M_ = NEG_INIT;
    for (int w = mm; w < NH; w += M) M_ = fmaxf(M_, red_m[w][i]);
    float l = 0.f, o = 0.f;
    for (int w = mm; w < NH; w += M) {
      const float f = exp2f(red_m[w][i] - M_);
      l = fmaf(red_l[w][i], f, l);
      o = fmaf(red_o[w][i][dd], f, o);
    }
    float* pp = part + ((size_t(mm) * HQ + g * GQA + i) * SMAX + s) * PSTR;
    pp[dd] = o;
    if (dd == 0) pp[D] = M_, pp[D + 1] = l;
  }
}


// ---- quill: tensor-core full attention (q4 cache) ----
constexpr int FM_KS = 136;  // f16 row stride of a staged 16-key tile (128 + 8: conflict-free ldmatrix)
constexpr int FM_TILE_HALVES = 16 * FM_KS;
constexpr int FM_SMEM = 4 * 2 * FM_TILE_HALVES * 2 + 256 * 4;  // 4 warps x (K, V) tiles + nibble-pair table
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(r0), "=r"(r1) : "r"(unsigned(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ void ldsm_x2_t(uint32_t& r0, uint32_t& r1, const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
               : "=r"(r0), "=r"(r1) : "r"(unsigned(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t hmul2u(uint32_t a, __half2 s) {
  __half2 h = *reinterpret_cast<const __half2*>(&a);
  h = __hmul2(h, s);
  return *reinterpret_cast<const uint32_t*>(&h);
}

__global__ void __launch_bounds__(128) attn89_full_mma(const SplitArgs a) {
  const int g = blockIdx.x, s = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, qg = lane >> 2, qt = lane & 3;
  const int M = a.M;
  const int p0 = int(a.pos[0]);
  if (p0 + M > a.cap) return;
  const int L_old = p0;
  int chunk, n_active;
  split_geom(L_old, true, chunk, n_active);
  if (s >= n_active) return;
  const int R3 = 3 * M, ntile = R3 > 16 ? 2 : 1;
  const int tile = ntile == 2 ? warp >> 1 : 0, kg = ntile == 2 ? (warp & 1) : warp, nkg = ntile == 2 ? 2 : 4;
  extern __shared__ __align__(16) unsigned char fm_sm[];
  __half* kt = reinterpret_cast<__half*>(fm_sm) + warp * 2 * FM_TILE_HALVES;  // this warp's K tile [16][FM_KS]
  __half* vt = kt + FM_TILE_HALVES;                                           // and V tile
  uint32_t* tbl = reinterpret_cast<uint32_t*>(fm_sm + 4 * 2 * FM_TILE_HALVES * 2);  // byte -> f16x2 (lo nibble first)
  for (int b = tid; b < 256; b += 128) tbl[b] = pack_h2(kIQ4NL[b & 15], kIQ4NL[b >> 4]);
  const char* ws = a.ws;
  const float* qs = reinterpret_cast<const float*>(ws + WS_QS);
  const float* relb0 = reinterpret_cast<const float*>(ws + WS_RELB);
  // A fragments of Q (rows rho = 16 tile + qg, + 8; columns 16 ks + 2 qt (+1, +8, +9)), zero past 3M
  uint32_t qa[8][4];
  {
    const int r0 = 16 * tile + qg, r1 = r0 + 8;
    const float* q0 = r0 < R3 ? qs + (size_t(r0 / 3) * HQ + g * GQA + r0 % 3) * D : nullptr;
    const float* q1 = r1 < R3 ? qs + (size_t(r1 / 3) * HQ + g * GQA + r1 % 3) * D : nullptr;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int c0 = 16 * ks + 2 * qt;
      qa[ks][0] = q0 ? pack_h2(q0[c0], q0[c0 + 1]) : 0u;
      qa[ks][1] = q1 ? pack_h2(q1[c0], q1[c0 + 1]) : 0u;
      qa[ks][2] = q0 ? pack_h2(q0[c0 + 8], q0[c0 + 9]) : 0u;
      qa[ks][3] = q1 ? pack_h2(q1[c0 + 8], q1[c0 + 9]) : 0u;
    }
  }
  // row bookkeeping of this lane's two rows
  const int rA = 16 * tile + qg, rB = rA + 8;
  const bool okA = rA < R3, okB = rB < R3;
  const int mA = okA ? rA / 3 : 0, mB = okB ? rB / 3 : 0;
  const float* rbA = relb0 + (size_t(mA) * HQ + g * GQA + (okA ? rA % 3 : 0)) * EXT;
  const float* rbB = relb0 + (size_t(mB) * HQ + g * GQA + (okB ? rB % 3 : 0)) * EXT;
  float o[16][4];
#pragma unroll
  for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float mxA = NEG_INIT, mxB = NEG_INIT, lA = 0.f, lB = 0.f;
  __syncthreads();  // tbl

  // one 16-key tile from the staged f16 K / V (key positions kpos0 + 0..15; nvalid keys; own: the call's new keys)
  auto process = [&](int kpos0, int nvalid, bool own) {
    __syncwarp();
    float sc[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t b0, b1;
        // matrices: keys 8 nt + 0..7 x dims 16 ks + 0..7 (lanes 0-7) and + 8..15 (lanes 8-15)
        ldsm_x2(b0, b1, kt + (8 * nt + (lane & 7)) * FM_KS + 16 * ks + ((lane >> 3) & 1) * 8);
        mma16816(sc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b0, b1);
      }
    }
    // bias, mask; columns 8 nt + 2 qt (+1); rows rA (elements 0, 1) and rB (2, 3)
    float mnA = mxA, mnB = mxB;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int col = 8 * nt + 2 * qt + (e & 1);
        const bool rowB = e >= 2;
        const bool okr = rowB ? okB : okA;
        const int mm = rowB ? mB : mA;
        float v;
        if (!okr || col >= nvalid) {
          v = -INFINITY;
        } else if (own) {  // the call's key col (row mm sees cols <= mm)
          v = col <= mm ? sc[nt][e] + __ldg((rowB ? rbB : rbA) + (mm - col)) : -INFINITY;
        } else {
          const int d = p0 + mm - (kpos0 + col);
          v = sc[nt][e] + (d < EXT ? __ldg((rowB ? rbB : rbA) + d) : 0.f);
        }
        sc[nt][e] = v;
        if (rowB) mnB = fmaxf(mnB, v);
        else mnA = fmaxf(mnA, v);
      }
    mnA = fmaxf(mnA, __shfl_xor_sync(0xffffffffu, mnA, 1));
    mnA = fmaxf(mnA, __shfl_xor_sync(0xffffffffu, mnA, 2));
    mnB = fmaxf(mnB, __shfl_xor_sync(0xffffffffu, mnB, 1));
    mnB = fmaxf(mnB, __shfl_xor_sync(0xffffffffu, mnB, 2));
    const float alA = exp2f(mxA - mnA), alB = exp2f(mxB - mnB);
    mxA = mnA, mxB = mnB;
    lA *= alA, lB *= alB;
#pragma unroll
    for (int n = 0; n < 16; ++n) o[n][0] *= alA, o[n][1] *= alA, o[n][2] *= alB, o[n][3] *= alB;
    float pr[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      pr[nt][0] = exp2f(sc[nt][0] - mnA), pr[nt][1] = exp2f(sc[nt][1] - mnA);
      pr[nt][2] = exp2f(sc[nt][2] - mnB), pr[nt][3] = exp2f(sc[nt][3] - mnB);
      lA += pr[nt][0] + pr[nt][1], lB += pr[nt][2] + pr[nt][3];
    }
    const uint32_t pa0 = pack_h2(pr[0][0], pr[0][1]), pa1 = pack_h2(pr[0][2], pr[0][3]);
    const uint32_t pa2 = pack_h2(pr[1][0], pr[1][1]), pa3 = pack_h2(pr[1][2], pr[1][3]);
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      uint32_t b0, b1;
      // V [keys 0..15][dims 8 n .. 8 n + 7]: transposed 8x8 matrices (keys 0-7: lanes 0-7, 8-15: lanes 8-15)
      ldsm_x2_t(b0, b1, vt + (lane & 15) * FM_KS + 8 * n);
      mma16816(o[n], pa0, pa1, pa2, pa3, b0, b1);
    }
    __syncwarp();
  };

  const uint8_t* kc = static_cast<const uint8_t*>(a.k_cache);
  const uint8_t* vc = static_cast<const uint8_t*>(a.v_cache);
  const int k0 = s * chunk, k1 = min(L_old, k0 + chunk);
  const int ntiles = (k1 - k0 + 15) / 16;
  // lane -> (key lane >> 1, half lane & 1): 32 code bytes (64 dims) + 2 scales of K and of V
  const int dk = lane >> 1, dh = lane & 1;
  auto load_codes = [&](int t, uint4 (&ck)[2], uint4 (&cv)[2], uint32_t& sk, uint32_t& sv) {
    const int j = min(k0 + 16 * t + dk, k1 - 1);
    const uint8_t* rk = kc + size_t(j) * size_t(a.q4_stride);
    const uint8_t* rv = vc + size_t(j) * size_t(a.q4_stride);
    ck[0] = __ldcs(reinterpret_cast<const uint4*>(rk + g * 64 + dh * 32));
    ck[1] = __ldcs(reinterpret_cast<const uint4*>(rk + g * 64 + dh * 32 + 16));
    cv[0] = __ldcs(reinterpret_cast<const uint4*>(rv + g * 64 + dh * 32));
    cv[1] = __ldcs(reinterpret_cast<const uint4*>(rv + g * 64 + dh * 32 + 16));
    sk = __ldcs(reinterpret_cast<const unsigned int*>(rk + Q4CODES + g * 8 + dh * 4));
    sv = __ldcs(reinterpret_cast<const unsigned int*>(rv + Q4CODES + g * 8 + dh * 4));
  };
  auto stage = [&](const uint4 (&cc)[2], uint32_t sc2, __half* dst) {
    const __half2 s0 = __half2half2(*reinterpret_cast<const __half*>(&sc2));
    const __half2 s1 = __half2half2(*(reinterpret_cast<const __half*>(&sc2) + 1));
    const uint32_t w[8] = {cc[0].x, cc[0].y, cc[0].z, cc[0].w, cc[1].x, cc[1].y, cc[1].z, cc[1].w};
    uint32_t h[32];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int b = 0; b < 4; ++b) h[4 * i + b] = hmul2u(tbl[(w[i] >> (8 * b)) & 0xffu], i < 4 ? s0 : s1);
    uint4* d4 = reinterpret_cast<uint4*>(dst + dk * FM_KS + dh * 64);
#pragma unroll
    for (int i = 0; i < 8; ++i) d4[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
  };
  uint4 ck[2], cv[2];
  uint32_t sk = 0, sv = 0;
  int t = kg;
  if (t < ntiles) load_codes(t, ck, cv, sk, sv);
  for (; t < ntiles; t += nkg) {
    stage(ck, sk, kt), stage(cv, sv, vt);
    if (t + nkg < ntiles) load_codes(t + nkg, ck, cv, sk, sv);  // next tile's codes in flight during the math
    process(k0 + 16 * t, min(16, k1 - k0 - 16 * t), false);
  }
  // the call's own keys (last split; key group 0 of each row tile), fp32 -> f16
  if (s == n_active - 1 && kg == 0) {
    const float* nk = reinterpret_cast<const float*>(ws + WS_NK);
    const float* nv = reinterpret_cast<const float*>(ws + WS_NV);
    for (int i = lane; i < 16 * D; i += 32) {
      const int kk = i / D, dd = i % D;
      kt[kk * FM_KS + dd] = __float2half_rn(kk < M ? nk[(size_t(kk) * HKV + g) * D + dd] : 0.f);
      vt[kk * FM_KS + dd] = __float2half_rn(kk < M ? nv[(size_t(kk) * HKV + g) * D + dd] : 0.f);
    }
    process(0, M, true);
  }
  // quad sums of l; merge the key groups of each row tile through shared memory
  lA += __shfl_xor_sync(0xffffffffu, lA, 1), lA += __shfl_xor_sync(0xffffffffu, lA, 2);
  lB += __shfl_xor_sync(0xffffffffu, lB, 1), lB += __shfl_xor_sync(0xffffffffu, lB, 2);
  __syncthreads();  // the tiles are dead: reuse as [4 warps][16 rows][D] O + m, l
  float* so = reinterpret_cast<float*>(fm_sm);                // [4][16][D]
  float* sml = so + 4 * 16 * D;                               // [4][16][2]
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    const int c = 8 * n + 2 * qt;
    so[(warp * 16 + qg) * D + c] = o[n][0], so[(warp * 16 + qg) * D + c + 1] = o[n][1];
    so[(warp * 16 + qg + 8) * D + c] = o[n][2], so[(warp * 16 + qg + 8) * D + c + 1] = o[n][3];
  }
  if (qt == 0) {
    sml[(warp * 16 + qg) * 2] = mxA, sml[(warp * 16 + qg) * 2 + 1] = lA;
    sml[(warp * 16 + qg + 8) * 2] = mxB, sml[(warp * 16 + qg + 8) * 2 + 1] = lB;
  }
  __syncthreads();
  float* part = reinterpret_cast<float*>(a.ws + WS_PART);
  for (int it = tid; it < R3 * D; it += 128) {
    const int rho = it / D, dd = it % D;
    const int tl = rho / 16, rr = rho % 16;
    const int w0 = ntile == 2 ? 2 * tl : 0, nw = ntile == 2 ? 2 : 4;
    float M_ = NEG_INIT;
    for (int w = w0; w < w0 + nw; ++w) M_ = fmaxf(M_, sml[(w * 16 + rr) * 2]);
    float l = 0.f, ov = 0.f;
    for (int w = w0; w < w0 + nw; ++w) {
      const float f = exp2f(sml[(w * 16 + rr) * 2] - M_);
      l = fmaf(sml[(w * 16 + rr) * 2 + 1], f, l);
      ov = fmaf(so[(w * 16 + rr) * D + dd], f, ov);
    }
    const int mm = rho / 3, i = rho % 3;
    float* pp = part + ((size_t(mm) * HQ + g * GQA + i) * SMAX + s) * PSTR;
    pp[dd] = ov;
    if (dd == 0) pp[D] = M_, pp[D + 1] = l;
  }
}

__global__ void __launch_bounds__(128) attn89_slide_mma(const SplitArgs a) {  // quill: q8 sliding ring
  const int g = blockIdx.x, s = blockIdx.y;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, qg = lane >> 2, qt = lane & 3;
  const int M = a.M;
  const int p0 = int(a.pos[0]);
  const int L_old = WIN;
  int chunk, n_active;
  split_geom(L_old, false, chunk, n_active);
  if (s >= n_active) return;
  const int R3 = 3 * M, ntile = R3 > 16 ? 2 : 1;
  const int tile = ntile == 2 ? warp >> 1 : 0, kg = ntile == 2 ? (warp & 1) : warp, nkg = ntile == 2 ? 2 : 4;
  extern __shared__ __align__(16) unsigned char fm_sm[];
  __half* kt = reinterpret_cast<__half*>(fm_sm) + warp * 2 * FM_TILE_HALVES;  // this warp's K tile [16][FM_KS]
  __half* vt = kt + FM_TILE_HALVES;                                           // and V tile
  uint32_t* tbl = reinterpret_cast<uint32_t*>(fm_sm + 4 * 2 * FM_TILE_HALVES * 2);  // byte -> f16x2 (lo nibble first)
  for (int b = tid; b < 256; b += 128) tbl[b] = pack_h2(kIQ4NL[b & 15], kIQ4NL[b >> 4]);
  const char* ws = a.ws;
  const float* qs = reinterpret_cast<const float*>(ws + WS_QS);
  const float* relb0 = reinterpret_cast<const float*>(ws + WS_RELB);
  // A fragments of Q (rows rho = 16 tile + qg, + 8; columns 16 ks + 2 qt (+1, +8, +9)), zero past 3M
  uint32_t qa[8][4];
  {
    const int r0 = 16 * tile + qg, r1 = r0 + 8;
    const float* q0 = r0 < R3 ? qs + (size_t(r0 / 3) * HQ + g * GQA + r0 % 3) * D : nullptr;
    const float* q1 = r1 < R3 ? qs + (size_t(r1 / 3) * HQ + g * GQA + r1 % 3) * D : nullptr;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int c0 = 16 * ks + 2 * qt;
      qa[ks][0] = q0 ? pack_h2(q0[c0], q0[c0 + 1]) : 0u;
      qa[ks][1] = q1 ? pack_h2(q1[c0], q1[c0 + 1]) : 0u;
      qa[ks][2] = q0 ? pack_h2(q0[c0 + 8], q0[c0 + 9]) : 0u;
      qa[ks][3] = q1 ? pack_h2(q1[c0 + 8], q1[c0 + 9]) : 0u;
    }
  }
  // row bookkeeping of this lane's two rows
  const int rA = 16 * tile + qg, rB = rA + 8;
  const bool okA = rA < R3, okB = rB < R3;
  const int mA = okA ? rA / 3 : 0, mB = okB ? rB / 3 : 0;
  const float* rbA = relb0 + (size_t(mA) * HQ + g * GQA + (okA ? rA % 3 : 0)) * EXT;
  const float* rbB = relb0 + (size_t(mB) * HQ + g * GQA + (okB ? rB % 3 : 0)) * EXT;
  float o[16][4];
#pragma unroll
  for (int n = 0; n < 16; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float mxA = NEG_INIT, mxB = NEG_INIT, lA = 0.f, lB = 0.f;
  __syncthreads();  // tbl

  // one 16-key tile from the staged f16 K / V (key positions kpos0 + 0..15; nvalid keys; own: the call's new keys)
  auto process = [&](int kpos0, int nvalid, bool own) {
    __syncwarp();
    float sc[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t b0, b1;
        // matrices: keys 8 nt + 0..7 x dims 16 ks + 0..7 (lanes 0-7) and + 8..15 (lanes 8-15)
        ldsm_x2(b0, b1, kt + (8 * nt + (lane & 7)) * FM_KS + 16 * ks + ((lane >> 3) & 1) * 8);
        mma16816(sc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], b0, b1);
      }
    }
    // bias, mask; columns 8 nt + 2 qt (+1); rows rA (elements 0, 1) and rB (2, 3)
    float mnA = mxA, mnB = mxB;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int col = 8 * nt + 2 * qt + (e & 1);
        const bool rowB = e >= 2;
        const bool okr = rowB ? okB : okA;
        const int mm = rowB ? mB : mA;
        float v;
        if (!okr || col >= nvalid) {
          v = -INFINITY;
        } else if (own) {  // the call's key col (row mm sees cols <= mm)
          v = col <= mm ? sc[nt][e] + __ldg((rowB ? rbB : rbA) + (mm - col)) : -INFINITY;
        } else {
          const int kp = int(__ldg(a.kpos + kpos0 + col)), d = p0 + mm - kp;
          v = kp >= 0 && kp < p0 && d >= 0 && d < WIN ? sc[nt][e] + __ldg((rowB ? rbB : rbA) + d) : -INFINITY;
        }
        sc[nt][e] = v;
        if (rowB) mnB = fmaxf(mnB, v);
        else mnA = fmaxf(mnA, v);
      }
    mnA = fmaxf(mnA, __shfl_xor_sync(0xffffffffu, mnA, 1));
    mnA = fmaxf(mnA, __shfl_xor_sync(0xffffffffu, mnA, 2));
    mnB = fmaxf(mnB, __shfl_xor_sync(0xffffffffu, mnB, 1));
    mnB = fmaxf(mnB, __shfl_xor_sync(0xffffffffu, mnB, 2));
    const float alA = exp2f(mxA - mnA), alB = exp2f(mxB - mnB);
    mxA = mnA, mxB = mnB;
    lA *= alA, lB *= alB;
#pragma unroll
    for (int n = 0; n < 16; ++n) o[n][0] *= alA, o[n][1] *= alA, o[n][2] *= alB, o[n][3] *= alB;
    float pr[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      pr[nt][0] = exp2f(sc[nt][0] - mnA), pr[nt][1] = exp2f(sc[nt][1] - mnA);
      pr[nt][2] = exp2f(sc[nt][2] - mnB), pr[nt][3] = exp2f(sc[nt][3] - mnB);
      lA += pr[nt][0] + pr[nt][1], lB += pr[nt][2] + pr[nt][3];
    }
    const uint32_t pa0 = pack_h2(pr[0][0], pr[0][1]), pa1 = pack_h2(pr[0][2], pr[0][3]);
    const uint32_t pa2 = pack_h2(pr[1][0], pr[1][1]), pa3 = pack_h2(pr[1][2], pr[1][3]);
#pragma unroll
    for (int n = 0; n < 16; ++n) {
      uint32_t b0, b1;
      // V [keys 0..15][dims 8 n .. 8 n + 7]: transposed 8x8 matrices (keys 0-7: lanes 0-7, 8-15: lanes 8-15)
      ldsm_x2_t(b0, b1, vt + (lane & 15) * FM_KS + 8 * n);
      mma16816(o[n], pa0, pa1, pa2, pa3, b0, b1);
    }
    __syncwarp();
  };

  const uint8_t* kc = static_cast<const uint8_t*>(a.k_cache);
  const uint8_t* vc = static_cast<const uint8_t*>(a.v_cache);
  const int k0 = s * chunk, k1 = min(L_old, k0 + chunk);
  const int ntiles = (k1 - k0 + 15) / 16;
  // lane -> (key lane >> 1, half lane & 1): 32 code bytes (64 dims) + 2 scales of K and of V
  const int dk = lane >> 1, dh = lane & 1;
  auto load_codes = [&](int t, uint4 (&ck)[4], uint4 (&cv)[4], uint32_t& sk, uint32_t& sv) {
    const int j = min(k0 + 16 * t + dk, k1 - 1);
    const uint8_t* rk = kc + size_t(j) * size_t(a.q4_stride);
    const uint8_t* rv = vc + size_t(j) * size_t(a.q4_stride);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ck[i] = __ldcs(reinterpret_cast<const uint4*>(rk + g * 128 + dh * 64 + 16 * i));
      cv[i] = __ldcs(reinterpret_cast<const uint4*>(rv + g * 128 + dh * 64 + 16 * i));
    }
    sk = __ldcs(reinterpret_cast<const unsigned int*>(rk + Q8CODES + g * 8 + dh * 4));
    sv = __ldcs(reinterpret_cast<const unsigned int*>(rv + Q8CODES + g * 8 + dh * 4));
  };
  auto stage = [&](const uint4 (&cc)[4], uint32_t sc2, __half* dst) {
    // int8 x -> f16 1024 + (x + 128) by byte placement, then (h - 1152) * s in one HFMA2
    const __half2 s0 = __half2half2(*reinterpret_cast<const __half*>(&sc2));
    const __half2 s1 = __half2half2(*(reinterpret_cast<const __half*>(&sc2) + 1));
    const __half2 c0 = __hmul2(s0, __float2half2_rn(-1152.f)), c1 = __hmul2(s1, __float2half2_rn(-1152.f));
    const uint32_t w[16] = {cc[0].x, cc[0].y, cc[0].z, cc[0].w, cc[1].x, cc[1].y, cc[1].z, cc[1].w,
                            cc[2].x, cc[2].y, cc[2].z, cc[2].w, cc[3].x, cc[3].y, cc[3].z, cc[3].w};
    uint32_t h[32];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint32_t u = w[i] ^ 0x80808080u;
      const __half2 sv2 = i < 8 ? s0 : s1, cv2 = i < 8 ? c0 : c1;
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const uint32_t bits = __byte_perm(u, 0x64646464u, b ? 0x7362 : 0x5140);
        const __half2 hv = __hfma2(*reinterpret_cast<const __half2*>(&bits), sv2, cv2);
        h[2 * i + b] = *reinterpret_cast<const uint32_t*>(&hv);
      }
    }
    uint4* d4 = reinterpret_cast<uint4*>(dst + dk * FM_KS + dh * 64);
#pragma unroll
    for (int i = 0; i < 8; ++i) d4[i] = make_uint4(h[4 * i], h[4 * i + 1], h[4 * i + 2], h[4 * i + 3]);
  };
  uint4 ck[4], cv[4];
  uint32_t sk = 0, sv = 0;
  int t = kg;
  if (t < ntiles) load_codes(t, ck, cv, sk, sv);
  for (; t < ntiles; t += nkg) {
    stage(ck, sk, kt), stage(cv, sv, vt);
    if (t + nkg < ntiles) load_codes(t + nkg, ck, cv, sk, sv);  // next tile's codes in flight during the math
    process(k0 + 16 * t, min(16, k1 - k0 - 16 * t), false);
  }
  // the call's own keys (last split; key group 0 of each row tile), fp32 -> f16
  if (s == n_active - 1 && kg == 0) {
    const float* nk = reinterpret_cast<const float*>(ws + WS_NK);
    const float* nv = reinterpret_cast<const float*>(ws + WS_NV);
    for (int i = lane; i < 16 * D; i += 32) {
      const int kk = i / D, dd = i % D;
      kt[kk * FM_KS + dd] = __float2half_rn(kk < M ? nk[(size_t(kk) * HKV + g) * D + dd] : 0.f);
      vt[kk * FM_KS + dd] = __float2half_rn(kk < M ? nv[(size_t(kk) * HKV + g) * D + dd] : 0.f);
    }
    process(0, M, true);
  }
  // quad sums of l; merge the key groups of each row tile through shared memory
  lA += __shfl_xor_sync(0xffffffffu, lA, 1), lA += __shfl_xor_sync(0xffffffffu, lA, 2);
  lB += __shfl_xor_sync(0xffffffffu, lB, 1), lB += __shfl_xor_sync(0xffffffffu, lB, 2);
  __syncthreads();  // the tiles are dead: reuse as [4 warps][16 rows][D] O + m, l
  float* so = reinterpret_cast<float*>(fm_sm);                // [4][16][D]
  float* sml = so + 4 * 16 * D;                               // [4][16][2]
#pragma unroll
  for (int n = 0; n < 16; ++n) {
    const int c = 8 * n + 2 * qt;
    so[(warp * 16 + qg) * D + c] = o[n][0], so[(warp * 16 + qg) * D + c + 1] = o[n][1];
    so[(warp * 16 + qg + 8) * D + c] = o[n][2], so[(warp * 16 + qg + 8) * D + c + 1] = o[n][3];
  }
  if (qt == 0) {
    sml[(warp * 16 + qg) * 2] = mxA, sml[(warp * 16 + qg) * 2 + 1] = lA;
    sml[(warp * 16 + qg + 8) * 2] = mxB, sml[(warp * 16 + qg + 8) * 2 + 1] = lB;
  }
  __syncthreads();
  float* part = reinterpret_cast<float*>(a.ws + WS_PART);
  for (int it = tid; it < R3 * D; it += 128) {
    const int rho = it / D, dd = it % D;
    const int tl = rho / 16, rr = rho % 16;
    const int w0 = ntile == 2 ? 2 * tl : 0, nw = ntile == 2 ? 2 : 4;
    float M_ = NEG_INIT;
    for (int w = w0; w < w0 + nw; ++w) M_ = fmaxf(M_, sml[(w * 16 + rr) * 2]);
    float l = 0.f, ov = 0.f;
    for (int w = w0; w < w0 + nw; ++w) {
      const float f = exp2f(sml[(w * 16 + rr) * 2] - M_);
      l = fmaf(sml[(w * 16 + rr) * 2 + 1], f, l);
      ov = fmaf(so[(w * 16 + rr) * D + dd], f, ov);
    }
    const int mm = rho / 3, i = rho % 3;
    float* pp = part + ((size_t(mm) * HQ + g * GQA + i) * SMAX + s) * PSTR;
    pp[dd] = ov;
    if (dd == 0) pp[D] = M_, pp[D + 1] = l;
  }
}

struct CombineArgs {
  long long q4_stride;
  const float* proj;
  int R;
  void* k_cache;
  void* v_cache;
  long long* kpos;
  const long long* pos;
  long long cap;
  int full;
  int q4;
  int M;
  const char* ws;
  __nv_bfloat16* out;
  int q8;
};

__global__ void __launch_bounds__(D) attn89_combine(const CombineArgs a) {
  const int m = blockIdx.x, g = blockIdx.y, d = threadIdx.x;
  const int p0 = int(a.pos[0]);
  if (a.full && p0 + a.M > a.cap) return;
  const int L_old = a.full ? p0 : WIN;
  int chunk, n_active;
  split_geom(L_old, a.full, chunk, n_active);
  const float* part = reinterpret_cast<const float*>(a.ws + WS_PART);
#pragma unroll
  for (int i = 0; i < GQA; ++i) {
    const int h = g * GQA + i;
    const float* pp = part + (size_t(m) * HQ + h) * SMAX * PSTR;
    const float sink = a.proj[size_t(m) * a.R + OFF_SINK + h] * LOG2E;
    float M_ = sink;
    for (int s = 0; s < n_active; ++s) M_ = fmaxf(M_, __ldcg(pp + s * PSTR + D));
    float l = exp2f(sink - M_), o = 0.f;
    for (int s = 0; s < n_active; ++s) {
      const float f = exp2f(__ldcg(pp + s * PSTR + D) - M_);
      l = fmaf(__ldcg(pp + s * PSTR + D + 1), f, l);
      o = fmaf(__ldcg(pp + s * PSTR + d), f, o);
    }
    const float gate = a.proj[size_t(m) * a.R + OFF_GATE + h * D + d];
    const float sg = 1.f / (1.f + __expf(-gate));
    a.out[size_t(m) * (HQ * D) + h * D + d] = __float2bfloat16_rn(bf16r(o / l) * sg);
  }
  // ---- write the call's key / value of head g into the cache
  const int P = p0 + m;
  const float* nk = reinterpret_cast<const float*>(a.ws + WS_NK);
  const float* nv = reinterpret_cast<const float*>(a.ws + WS_NV);
  if (!a.full && a.q8) {  // quill: q8 ring row at slot P % 1024
    const int slot = P & (WIN - 1);
    const uint8_t* nq = reinterpret_cast<const uint8_t*>(a.ws + WS_NQ);
    uint8_t* kr = static_cast<uint8_t*>(a.k_cache) + size_t(slot) * Q8ROW;
    uint8_t* vr = static_cast<uint8_t*>(a.v_cache) + size_t(slot) * Q8ROW;
    const uint8_t* sk = nq + (size_t(m) * 2 + 0) * Q8ROW;
    const uint8_t* sv = nq + (size_t(m) * 2 + 1) * Q8ROW;
    kr[g * 128 + d] = sk[g * 128 + d], vr[g * 128 + d] = sv[g * 128 + d];
    if (d < 8) kr[Q8CODES + g * 8 + d] = sk[Q8CODES + g * 8 + d], vr[Q8CODES + g * 8 + d] = sv[Q8CODES + g * 8 + d];
    if (g == 0 && d == 0) a.kpos[slot] = P;
  } else if (!a.full && a.q4) {  // quill: q4 ring row at slot P % 1024
    const int slot = P & (WIN - 1);
    const uint8_t* nq = reinterpret_cast<const uint8_t*>(a.ws + WS_NQ);
    uint8_t* kr = static_cast<uint8_t*>(a.k_cache) + size_t(slot) * Q4ROW;
    uint8_t* vr = static_cast<uint8_t*>(a.v_cache) + size_t(slot) * Q4ROW;
    const uint8_t* sk = nq + (size_t(m) * 2 + 0) * Q4ROW;
    const uint8_t* sv = nq + (size_t(m) * 2 + 1) * Q4ROW;
    if (d < 64) kr[g * 64 + d] = sk[g * 64 + d], vr[g * 64 + d] = sv[g * 64 + d];
    else if (d < 72) kr[Q4CODES + g * 8 + d - 64] = sk[Q4CODES + g * 8 + d - 64], vr[Q4CODES + g * 8 + d - 64] = sv[Q4CODES + g * 8 + d - 64];
    if (g == 0 && d == 0) a.kpos[slot] = P;
  } else if (!a.full) {
    const int slot = P & (WIN - 1);
    __nv_bfloat16* kc = static_cast<__nv_bfloat16*>(a.k_cache);
    __nv_bfloat16* vc = static_cast<__nv_bfloat16*>(a.v_cache);
    kc[(size_t(slot) * HKV + g) * D + d] = __float2bfloat16_rn(nk[(size_t(m) * HKV + g) * D + d]);
    vc[(size_t(slot) * HKV + g) * D + d] = __float2bfloat16_rn(nv[(size_t(m) * HKV + g) * D + d]);
    if (g == 0 && d == 0) a.kpos[slot] = P;
  } else if (a.q4) {
    const uint8_t* nq = reinterpret_cast<const uint8_t*>(a.ws + WS_NQ);
    uint8_t* kr = static_cast<uint8_t*>(a.k_cache) + size_t(P) * size_t(a.q4_stride);
    uint8_t* vr = static_cast<uint8_t*>(a.v_cache) + size_t(P) * size_t(a.q4_stride);
    const uint8_t* sk = nq + (size_t(m) * 2 + 0) * Q4ROW;
    const uint8_t* sv = nq + (size_t(m) * 2 + 1) * Q4ROW;
    if (d < 64) kr[g * 64 + d] = sk[g * 64 + d], vr[g * 64 + d] = sv[g * 64 + d];
    else if (d < 72) kr[Q4CODES + g * 8 + d - 64] = sk[Q4CODES + g * 8 + d - 64], vr[Q4CODES + g * 8 + d - 64] = sv[Q4CODES + g * 8 + d - 64];
  } else {
    __nv_bfloat16* kc = static_cast<__nv_bfloat16*>(a.k_cache);
    __nv_bfloat16* vc = static_cast<__nv_bfloat16*>(a.v_cache);
    kc[(size_t(P) * HKV + g) * D + d] = __float2bfloat16_rn(nk[(size_t(m) * HKV + g) * D + d]);
    vc[(size_t(P) * HKV + g) * D + d] = __float2bfloat16_rn(nv[(size_t(m) * HKV + g) * D + d]);
  }
}

template <class F>
void dispatch_m(int M, F&& f) {
  switch (M) {
    case 1: f(std::integral_constant<int, 1>{}); break;
    case 2: f(std::integral_constant<int, 2>{}); break;
    case 3: f(std::integral_constant<int, 3>{}); break;
    case 4: f(std::integral_constant<int, 4>{}); break;
    case 5: f(std::integral_constant<int, 5>{}); break;
    case 6: f(std::integral_constant<int, 6>{}); break;
    case 7: f(std::integral_constant<int, 7>{}); break;
    case 8: f(std::integral_constant<int, 8>{}); break;
    default: throw std::invalid_argument("attn_decode: M must be in 1..8");
  }
}

}  // namespace

std::size_t workspace_bytes(int /*max_m*/) { return WS_END; }

int max_active_clusters(int /*M*/, int /*window*/) { return 8; }

std::size_t q4_row_bytes() { return Q4ROW; }

void debug_enable(bool on) {
  const int v = on ? 1 : 0;
  CUDA_CHECK(cudaMemcpyToSymbol(g_dbg_on, &v, sizeof(v)));
  unsigned long long z[8] = {0, 0, 0, 0, 0, ~0ull, 0, 0};
  CUDA_CHECK(cudaMemcpyToSymbol(g_dbg_acc, z, sizeof(z)));
}
void debug_report(const char* tag, int launches) {
  CUDA_CHECK(cudaDeviceSynchronize());
  unsigned long long h[8];
  CUDA_CHECK(cudaMemcpyFromSymbol(h, g_dbg_acc, sizeof(h)));
  const double n = double(h[4] ? h[4] : 1);
  std::fprintf(stderr, "[attn dbg %s] blocks %llu: staging %.2f us, loop %.2f us, merge+write %.2f us, block total %.2f us; first start -> last end %.1f us over %d launches (%.2f us each)\n",
               tag, h[4], h[0] / n / 1e3, h[1] / n / 1e3, h[2] / n / 1e3, h[3] / n / 1e3, (h[6] - h[5]) / 1e3, launches,
               launches ? (h[6] - h[5]) / 1e3 / launches : 0.0);
}

void attn_decode(const DecodeArgs& a, cudaStream_t stream) {
  const bool full = a.window == 0;
  if (a.M < 1 || a.M > MAXM) throw std::invalid_argument("attn_decode: M must be in 1..8");
  if (a.workspace_size < WS_END) throw std::invalid_argument("attn_decode: workspace too small");
  if (!full && a.cache_fp8) throw std::invalid_argument("attn_decode: sliding rings are BF16 or q4");
  if (a.cache_fp8) throw std::invalid_argument("attn_decode(sm89): FP8 caches are not supported; use q4 or BF16");
  char* ws = static_cast<char*>(a.workspace);
  PrepArgs pa{a.proj, full ? R_FULL : R_SLIDING, a.M, full, a.cache_q4, a.cache_rows, a.q_scale, a.r_scale,
              a.rel_proj, a.k_conv, a.v_conv, a.conv_k_state, a.conv_v_state, a.pos, ws, a.cache_q8};
  dispatch_m(a.M, [&](auto mc) { attn89_prep<decltype(mc)::value><<<dim3(HKV, 5), 256, 0, stream>>>(pa); });
  SplitArgs sa{a.k_cache, a.v_cache, a.kpos, a.pos, a.cache_rows, full, a.cache_q4, a.M, ws, a.q4_stride, a.cache_q8};
  const dim3 grid(a.M, HKV, full ? SMAX : WIN / SL_CHUNK);
  constexpr int stage_bytes = SL_STAGE > RED_O_BYTES ? SL_STAGE : RED_O_BYTES;
  static const bool slide8 = !(std::getenv("QUILL_ATTN_SLIDE8") && std::atoi(std::getenv("QUILL_ATTN_SLIDE8")) == 0);
  static const bool smma = !(std::getenv("QUILL_ATTN_SMMA") && std::atoi(std::getenv("QUILL_ATTN_SMMA")) == 0);
  if (!full && a.cache_q8 && smma)  // quill: tensor cores
    attn89_slide_mma<<<dim3(HKV, WIN / SL_CHUNK), 128, FM_SMEM, stream>>>(sa);
  else if (!full && a.cache_q8 && slide8)  // quill: all rows per block
    attn89_slide8<<<dim3(HKV, WIN / SL_CHUNK), 256, SL8_STAGE, stream>>>(sa);
  else if (!full && a.cache_q8) attn89_split<false, 2><<<grid, 256, stage_bytes, stream>>>(sa);  // quill: q8 ring
  else if (!full && a.cache_q4) attn89_split<false, 1><<<grid, 256, RED_O_BYTES, stream>>>(sa);  // quill: q4 ring
  else if (!full) attn89_split<false, 0><<<grid, 256, RED_O_BYTES, stream>>>(sa);
  else if (a.cache_q4 && !(std::getenv("QUILL_ATTN_FMMA") && std::atoi(std::getenv("QUILL_ATTN_FMMA")) == 0))
    attn89_full_mma<<<dim3(HKV, SMAX), 128, FM_SMEM, stream>>>(sa);  // quill: tensor cores
  else if (a.cache_q4 && !(std::getenv("QUILL_ATTN_FULL8") && std::atoi(std::getenv("QUILL_ATTN_FULL8")) == 0))
    attn89_full8<<<dim3(HKV, SMAX), 256, F8_SMEM, stream>>>(sa);  // quill: all rows per block
  else if (a.cache_q4) attn89_split<true, 1><<<grid, 256, RED_O_BYTES, stream>>>(sa);
  else attn89_split<true, 0><<<grid, 256, RED_O_BYTES, stream>>>(sa);
  CombineArgs ca{a.q4_stride, a.proj, full ? R_FULL : R_SLIDING, a.k_cache, a.v_cache, a.kpos, a.pos, a.cache_rows, full,
                 a.cache_q4, a.M, ws, a.out, a.cache_q8};
  attn89_combine<<<dim3(a.M, HKV), D, 0, stream>>>(ca);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::attn
