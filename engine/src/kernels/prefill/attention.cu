// Prefill attention (flex.attend + model._visit's output gate) as a flash-attention kernel for 24 query /
// 8 KV heads, head dim 128.
//
// Design. A CTA owns one KV head and 128 query rows of the chunk, rows ordered (token, head of the GQA group):
// row R is token R / 3, head 3 g + R % 3, so all of them read the same keys and values and every key/value
// tile in shared memory serves 128 rows. 8 warps of 16 rows. Q (BF16) stays in registers as m16n8k16 A
// fragments. Per 64-key tile: S = Q K^T with BF16 MMA and FP32 accumulation (BF16 values are exact, as in
// flex's TF32 dots), scale 1/sqrt(128), relative bias rel_logits[t, h, d] for distances d < 1024, mask
// (causal, window, empty slots), online softmax in FP32 (base 2), P = FP16(p * 2^15) (an FP16 / TF32 10-bit
// mantissa without FP16's underflow below 2^-14), O += P V with FP16 MMA (values converted BF16/E4M3 -> FP16
// when staged: exact for the model's RMS-normed values), FP32 accumulation. The row sum uses the FP32 p.
// Epilogue: o = O 2^-15 / l, lse = ln(sum e^s), o * sigmoid(lse - sink) rounded to BF16, times
// sigmoid(gate), rounded to BF16.
// Keys/values are staged through registers into double-buffered swizzled shared memory: tile i + 1 is loaded
// from global memory at the start of tile i and stored to the other buffer right after tile i's Q.K^T (one
// barrier per tile); fragment loads of a k16 group are issued before its MMAs. Keys are either implicit (key j at
// position j: the full-attention cache prefix) or explicit (kpos, the sliding ring + chunk rows) with per-tile
// position bounds deciding which tiles are skipped, need no mask, or need the bias. CTAs run heaviest row tiles
// (latest tokens) first.
#include "kernels/prefill/common.cuh"
#include "kernels/prefill/prefill.h"
#include "kernels/common/iq4nl.cuh"

#include <cmath>
#include <stdexcept>

namespace ingine::kernels::prefill {
namespace {
using ingine::kernels::iq4nl_value;
using ingine::kernels::kQ4Row;
using ingine::kernels::kQ4Codes;

constexpr int D = kHeadDim;   // 128
constexpr int DCH = D / 8;    // 16-byte chunks per 16-bit row
constexpr int BK = kAttnKeyTile;
constexpr int NWARPS = 8, NT = NWARPS * 32, BR = NWARPS * 16;  // 128 query rows per CTA
constexpr int GQA = kHeads / kKvHeads;
constexpr int GATE_COL = kHeads * kHeadDim;                       // 3072
constexpr int SINK_COL = 2 * kHeads * kHeadDim + kHeads * kRel;   // 6528
constexpr float kScale = 0.08838834764831845f;                    // 128 ** -0.5
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;
constexpr float kPScale = 32768.f, kPUnscale = 1.f / 32768.f;

struct Smem {
  uint16_t k[2][BK * D];  // BF16 keys, swizzled [64][16 chunks]
  uint16_t v[2][BK * D];  // FP16 values
  int kpos[2][BK];
  int4 bounds[kAttnMaxTiles];
  int list[kAttnMaxTiles];
  int nlist;
  float lut[16];  // quill: IQ4_NL values
};

template <int KVT>
struct KvRegs {
  static constexpr int N = KVT == 1 ? 2 : KVT == 2 ? 1 : 4;  // 16-byte loads per thread per matrix
  uint4 k[N], v[N];
  __half ks, vs;  // q4 block scales
  int kp;
};

template <int KVT, bool EXPLICIT>
__device__ __forceinline__ void load_tile(const AttnParams& p, int g, int j0, KvRegs<KVT>& r) {
  const int tid = threadIdx.x;
  if constexpr (KVT == 2) {  // quill: q4 rows; thread = (key, 32-dim block)
    const uint8_t* K = static_cast<const uint8_t*>(p.k);
    const uint8_t* V = static_cast<const uint8_t*>(p.v);
    const int row = tid >> 2, c = tid & 3, j = j0 + row;
    if (j < p.L) {
      const size_t o = size_t(j) * size_t(p.kv_stride);
      r.k[0] = __ldg(reinterpret_cast<const uint4*>(K + o + g * 64 + c * 16));
      r.v[0] = __ldg(reinterpret_cast<const uint4*>(V + o + g * 64 + c * 16));
      r.ks = __ldg(reinterpret_cast<const __half*>(K + o + kQ4Codes + g * 8 + c * 2));
      r.vs = __ldg(reinterpret_cast<const __half*>(V + o + kQ4Codes + g * 8 + c * 2));
    } else {
      r.k[0] = r.v[0] = make_uint4(0, 0, 0, 0);
      r.ks = r.vs = __float2half(0.f);
    }
  } else if constexpr (KVT == 1) {
    const uint8_t* K = static_cast<const uint8_t*>(p.k);
    const uint8_t* V = static_cast<const uint8_t*>(p.v);
#pragma unroll
    for (int i = 0; i < r.N; ++i) {
      const int id = tid + i * NT, row = id >> 3, c = id & 7;
      const int j = j0 + row;
      if (j < p.L) {
        const size_t off = size_t(j) * kKvDim + g * D + c * 16;
        r.k[i] = __ldg(reinterpret_cast<const uint4*>(K + off));
        r.v[i] = __ldg(reinterpret_cast<const uint4*>(V + off));
      } else {
        r.k[i] = r.v[i] = make_uint4(0, 0, 0, 0);
      }
    }
  } else {
    const uint16_t* K = static_cast<const uint16_t*>(p.k);
    const uint16_t* V = static_cast<const uint16_t*>(p.v);
#pragma unroll
    for (int i = 0; i < r.N; ++i) {
      const int id = tid + i * NT, row = id >> 4, c = id & 15;
      const int j = j0 + row;
      if (j < p.L) {
        const size_t off = size_t(j) * kKvDim + g * D + c * 8;
        r.k[i] = __ldg(reinterpret_cast<const uint4*>(K + off));
        r.v[i] = __ldg(reinterpret_cast<const uint4*>(V + off));
      } else {
        r.k[i] = r.v[i] = make_uint4(0, 0, 0, 0);
      }
    }
  }
  if constexpr (EXPLICIT) {
    if (tid < BK) r.kp = __ldg(p.kpos + j0 + tid);
  }
}

__device__ __forceinline__ uint32_t bf16x2_to_half2(uint32_t u) { return pack_half2(bf_lo(u), bf_hi(u)); }

__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint16_t u) {
  const float2 f = e4m3x2_to_float2(u);
  return pack_bf16x2(f.x, f.y);
}

template <int KVT, bool EXPLICIT>
__device__ __forceinline__ void store_tile(Smem& s, int buf, const KvRegs<KVT>& r) {
  const int tid = threadIdx.x;
  if constexpr (KVT == 2) {  // quill: dequantise the thread's 32-dim block into 4 chunks of K (bf16) and V (fp16)
    const int row = tid >> 2, c = tid & 3;
    const float ks = __half2float(r.ks), vs = __half2float(r.vs);
    const uint32_t kw[4] = {r.k[0].x, r.k[0].y, r.k[0].z, r.k[0].w};
    const uint32_t vw[4] = {r.v[0].x, r.v[0].y, r.v[0].z, r.v[0].w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {  // 8 dims (4 bytes) per chunk
      float kf[8], vf[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        kf[e] = s.lut[(kw[q] >> (4 * e)) & 15u] * ks;
        vf[e] = s.lut[(vw[q] >> (4 * e)) & 15u] * vs;
      }
      uint4 kq, vq;
      kq.x = pack_bf16x2(kf[0], kf[1]); kq.y = pack_bf16x2(kf[2], kf[3]);
      kq.z = pack_bf16x2(kf[4], kf[5]); kq.w = pack_bf16x2(kf[6], kf[7]);
      vq.x = pack_half2(vf[0], vf[1]); vq.y = pack_half2(vf[2], vf[3]);
      vq.z = pack_half2(vf[4], vf[5]); vq.w = pack_half2(vf[6], vf[7]);
      *reinterpret_cast<uint4*>(&s.k[buf][swz<DCH>(row, 4 * c + q)]) = kq;
      *reinterpret_cast<uint4*>(&s.v[buf][swz<DCH>(row, 4 * c + q)]) = vq;
    }
  } else if constexpr (KVT == 1) {
#pragma unroll
    for (int i = 0; i < r.N; ++i) {
      const int id = tid + i * NT, row = id >> 3, c = id & 7;
      const uint32_t kw[4] = {r.k[i].x, r.k[i].y, r.k[i].z, r.k[i].w};
      const uint32_t vw[4] = {r.v[i].x, r.v[i].y, r.v[i].z, r.v[i].w};
      uint4 k0, k1, v0, v1;
      k0.x = e4m3x2_to_bf16x2(kw[0] & 0xffff); k0.y = e4m3x2_to_bf16x2(kw[0] >> 16);
      k0.z = e4m3x2_to_bf16x2(kw[1] & 0xffff); k0.w = e4m3x2_to_bf16x2(kw[1] >> 16);
      k1.x = e4m3x2_to_bf16x2(kw[2] & 0xffff); k1.y = e4m3x2_to_bf16x2(kw[2] >> 16);
      k1.z = e4m3x2_to_bf16x2(kw[3] & 0xffff); k1.w = e4m3x2_to_bf16x2(kw[3] >> 16);
      v0.x = e4m3x2_to_half2(vw[0] & 0xffff); v0.y = e4m3x2_to_half2(vw[0] >> 16);
      v0.z = e4m3x2_to_half2(vw[1] & 0xffff); v0.w = e4m3x2_to_half2(vw[1] >> 16);
      v1.x = e4m3x2_to_half2(vw[2] & 0xffff); v1.y = e4m3x2_to_half2(vw[2] >> 16);
      v1.z = e4m3x2_to_half2(vw[3] & 0xffff); v1.w = e4m3x2_to_half2(vw[3] >> 16);
      *reinterpret_cast<uint4*>(&s.k[buf][swz<DCH>(row, 2 * c)]) = k0;
      *reinterpret_cast<uint4*>(&s.k[buf][swz<DCH>(row, 2 * c + 1)]) = k1;
      *reinterpret_cast<uint4*>(&s.v[buf][swz<DCH>(row, 2 * c)]) = v0;
      *reinterpret_cast<uint4*>(&s.v[buf][swz<DCH>(row, 2 * c + 1)]) = v1;
    }
  } else {
#pragma unroll
    for (int i = 0; i < r.N; ++i) {
      const int id = tid + i * NT, row = id >> 4, c = id & 15;
      *reinterpret_cast<uint4*>(&s.k[buf][swz<DCH>(row, c)]) = r.k[i];
      uint4 v;
      v.x = bf16x2_to_half2(r.v[i].x);
      v.y = bf16x2_to_half2(r.v[i].y);
      v.z = bf16x2_to_half2(r.v[i].z);
      v.w = bf16x2_to_half2(r.v[i].w);
      *reinterpret_cast<uint4*>(&s.v[buf][swz<DCH>(row, c)]) = v;
    }
  }
  if constexpr (EXPLICIT) {
    if (tid < BK) s.kpos[buf][tid] = r.kp;
  }
}

template <int KVT, bool EXPLICIT>
__global__ void __launch_bounds__(NT, 1) attn_kernel(const AttnParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = blockIdx.y;
  const int rows = GQA * p.T;
  const int R0 = (gridDim.x - 1 - blockIdx.x) * BR;  // heaviest (latest) rows first
  if (tid < 16) s.lut[tid] = iq4nl_value(tid);
  __syncthreads();

  // ---- Q tile -> registers (staged through the key buffers)
  {
    uint16_t* stage = &s.k[0][0];
#pragma unroll
    for (int i = 0; i < (BR * DCH) / NT; ++i) {
      const int id = tid + i * NT, r = id >> 4, c = id & 15;
      const int R = min(R0 + r, rows - 1);
      const int t = R / GQA, h = g * GQA + R % GQA;
      cp_async16(stage + swz<DCH>(r, c), p.q + (size_t(t) * kHeads + h) * D + c * 8);
    }
    cp_commit();
  }
  // per-thread rows: lane / 4 and lane / 4 + 8 of the warp's 16
  int qpos[2];
  const uint16_t* relp[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int R = min(R0 + warp * 16 + (lane >> 2) + r * 8, rows - 1);
    const int t = R / GQA, h = g * GQA + R % GQA;
    qpos[r] = p.pos0 + t;
    relp[r] = reinterpret_cast<const uint16_t*>(p.rel) + (size_t(t) * kHeads + h) * kRelExtent;
  }
  const int qmin = p.pos0 + R0 / GQA;
  const int qmax = p.pos0 + min(R0 + BR - 1, rows - 1) / GQA;

  // ---- key tiles to visit
  int first = 0, ntiles = 0;
  if constexpr (EXPLICIT) {
    const int total = (p.L + BK - 1) / BK;
    if (tid < total) s.bounds[tid] = p.bounds[tid];
    __syncthreads();
    if (tid == 0) {
      int n = 0;
      for (int i = 0; i < total; ++i) {
        const int4 b = s.bounds[i];
        const bool reach = b.w && qmax - b.x >= 0 && (p.window == 0 || qmin - b.y < p.window);
        if (reach) s.list[n++] = i;
      }
      s.nlist = n;
    }
  } else {
    const int lo = p.window ? max(0, qmin - p.window + 1) : 0;
    const int hi = min(qmax, p.L - 1);
    first = lo / BK;
    ntiles = hi >= lo ? hi / BK - first + 1 : 0;
  }
  cp_wait<0>();
  __syncthreads();
  if constexpr (EXPLICIT) ntiles = s.nlist;

  unsigned qf[DCH / 2][4];
#pragma unroll
  for (int kk = 0; kk < DCH / 2; ++kk)
    ldmatrix_x4(qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3],
                smem_addr(&s.k[0][0] + swz<DCH>(warp * 16 + (lane & 15), kk * 2 + (lane >> 4))));
  __syncthreads();  // Q staging is overwritten by the first key tile

  auto tile_at = [&](int i) { return EXPLICIT ? s.list[i] : first + i; };

  float o[D / 8][4];
#pragma unroll
  for (int i = 0; i < D / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};

  KvRegs<KVT> regs;
  if (ntiles > 0) {
    load_tile<KVT, EXPLICIT>(p, g, tile_at(0) * BK, regs);
    store_tile<KVT, EXPLICIT>(s, 0, regs);
  }
  __syncthreads();

#pragma unroll 1
  for (int i = 0; i < ntiles; ++i) {
    const int kt = tile_at(i), buf = i & 1;
    if (i + 1 < ntiles) load_tile<KVT, EXPLICIT>(p, g, tile_at(i + 1) * BK, regs);

    // tile class (uniform over the CTA)
    int kmin, kmax;
    bool allv;
    if constexpr (EXPLICIT) {
      const int4 b = s.bounds[kt];
      kmin = b.x, kmax = b.y, allv = b.z != 0;
    } else {
      kmin = kt * BK;
      kmax = min(kt * BK + BK - 1, p.L - 1);
      allv = kt * BK + BK <= p.L;
    }
    const int dmin = qmin - kmax, dmax = qmax - kmin;
    const bool need_mask = !(allv && dmin >= 0 && (p.window == 0 || dmax < p.window));
    const bool need_bias = dmin < kRelExtent;

    // ---- S = Q K^T
    float sc[BK / 8][4];
#pragma unroll
    for (int nt = 0; nt < BK / 8; ++nt) sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
    const uint16_t* Ks = s.k[buf];
#pragma unroll
    for (int kk = 0; kk < DCH / 2; ++kk) {
      // all of the k16 group's key fragments first, then its 8 MMAs (each MMA no longer waits on the ldmatrix
      // issued right before it)
      unsigned b[BK / 16][4];
#pragma unroll
      for (int np = 0; np < BK / 16; ++np)
        ldmatrix_x4(b[np][0], b[np][1], b[np][2], b[np][3],
                    smem_addr(Ks + swz<DCH>(np * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1))));
#pragma unroll
      for (int np = 0; np < BK / 16; ++np) {
        mma_bf16(sc[2 * np], qf[kk], b[np][0], b[np][1]);
        mma_bf16(sc[2 * np + 1], qf[kk], b[np][2], b[np][3]);
      }
    }
    // stage the next tile now (its buffer was last read before the previous barrier): the conversion and the
    // shared stores overlap the Q.K^T MMAs in flight instead of trailing the P.V MMAs
    if (i + 1 < ntiles) store_tile<KVT, EXPLICIT>(s, buf ^ 1, regs);

    // ---- scores (log2 units), bias, mask
    float rmax[2] = {m[0], m[1]};
#pragma unroll
    for (int nt = 0; nt < BK / 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int r = e >> 1, jl = nt * 8 + (lane & 3) * 2 + (e & 1);
        float x = sc[nt][e] * kScale;
        if (need_mask || need_bias) {
          const int kp = EXPLICIT ? s.kpos[buf][jl] : kt * BK + jl;
          const int d = qpos[r] - kp;
          const bool ok = kp >= 0 && d >= 0 && (p.window == 0 || d < p.window) && (EXPLICIT || kp < p.L);
          if (need_bias && ok && d < kRelExtent) x += bf2f(__ldg(relp[r] + d));
          x = ok ? x * kLog2e : -INFINITY;
        } else {
          x *= kLog2e;
        }
        sc[nt][e] = x;
        rmax[r] = fmaxf(rmax[r], x);
      }
    }
    float mref[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 1));
      rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 2));
      mref[r] = rmax[r] == -INFINITY ? 0.f : rmax[r];
      const float alpha = exp2f(m[r] - mref[r]);
      m[r] = rmax[r];
      l[r] *= alpha;
#pragma unroll
      for (int dt = 0; dt < D / 8; ++dt) {
        o[dt][2 * r] *= alpha;
        o[dt][2 * r + 1] *= alpha;
      }
    }
#pragma unroll
    for (int nt = 0; nt < BK / 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float pv = exp2f(sc[nt][e] - mref[e >> 1]);
        l[e >> 1] += pv;
        sc[nt][e] = pv * kPScale;
      }
    }

    // ---- O += P V
    const uint16_t* Vs = s.v[buf];
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      const unsigned a[4] = {pack_half2(sc[2 * kk][0], sc[2 * kk][1]), pack_half2(sc[2 * kk][2], sc[2 * kk][3]),
                             pack_half2(sc[2 * kk + 1][0], sc[2 * kk + 1][1]),
                             pack_half2(sc[2 * kk + 1][2], sc[2 * kk + 1][3])};
#pragma unroll
      for (int dq = 0; dq < D / 32; ++dq) {  // two value fragment loads, then their four MMAs
        unsigned b[2][4];
#pragma unroll
        for (int h = 0; h < 2; ++h)
          ldmatrix_x4_t(b[h][0], b[h][1], b[h][2], b[h][3],
                        smem_addr(Vs + swz<DCH>(kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8,
                                                (2 * dq + h) * 2 + (lane >> 4))));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          mma_f16(o[2 * (2 * dq + h)], a, b[h][0], b[h][1]);
          mma_f16(o[2 * (2 * dq + h) + 1], a, b[h][2], b[h][3]);
        }
      }
    }

    __syncthreads();
  }

  // ---- epilogue
  const uint16_t* proj = reinterpret_cast<const uint16_t*>(p.proj);
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float lr = l[r];
    lr += __shfl_xor_sync(0xffffffffu, lr, 1);
    lr += __shfl_xor_sync(0xffffffffu, lr, 2);
    const int R = R0 + warp * 16 + (lane >> 2) + r * 8;
    if (R >= rows) continue;
    const int t = R / GQA, h = g * GQA + R % GQA;
    const uint16_t* prow = proj + size_t(t) * p.ld;
    const float lse = (m[r] + log2f(lr)) * kLn2;
    const float factor = 1.f / (1.f + expf(-(lse - bf2f(prow[SINK_COL + h]))));
    uint16_t* out = reinterpret_cast<uint16_t*>(p.out) + (size_t(t) * kHeads + h) * D;
#pragma unroll
    for (int dt = 0; dt < D / 8; ++dt) {
      const int col = dt * 8 + (lane & 3) * 2;
      const uint32_t gw = *reinterpret_cast<const uint32_t*>(prow + GATE_COL + h * D + col);
      const float g0 = bf_lo(gw), g1 = bf_hi(gw);
      const float a0 = round_bf16(o[dt][2 * r] * kPUnscale / lr * factor);
      const float a1 = round_bf16(o[dt][2 * r + 1] * kPUnscale / lr * factor);
      const float s0 = 1.f / (1.f + expf(-g0)), s1 = 1.f / (1.f + expf(-g1));
      *reinterpret_cast<uint32_t*>(out + col) = pack_bf16x2(a0 * s0, a1 * s1);
    }
  }
}

template <int KVT, bool EXPLICIT>
void launch(const AttnParams& p, cudaStream_t stream) {
  static const cudaError_t attr =
      cudaFuncSetAttribute(attn_kernel<KVT, EXPLICIT>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem)));
  if (attr != cudaSuccess) throw std::runtime_error("prefill attention: shared memory attribute failed");
  const dim3 grid((GQA * p.T + BR - 1) / BR, kKvHeads);
  attn_kernel<KVT, EXPLICIT><<<grid, NT, sizeof(Smem), stream>>>(p);
}

// ---- sliding-visit key rows

constexpr int kRowVecs = kKvDim * 2 / 16;  // 16-byte vectors per key row (128)

__global__ void ring_gather_kernel(const uint4* __restrict__ k_ring, const uint4* __restrict__ v_ring,
                                   const int64_t* __restrict__ ring_pos, int pos0, int T, uint4* __restrict__ k_rows,
                                   uint4* __restrict__ v_rows, int* __restrict__ kpos, int kpos_rows) {
  const int i = blockIdx.x, slot = (pos0 + i) & (kRelExtent - 1);
  k_rows[size_t(i) * kRowVecs + threadIdx.x] = k_ring[size_t(slot) * kRowVecs + threadIdx.x];
  v_rows[size_t(i) * kRowVecs + threadIdx.x] = v_ring[size_t(slot) * kRowVecs + threadIdx.x];
  if (threadIdx.x == 0) kpos[i] = int(ring_pos[slot]);
  for (int r = kRelExtent + i * kRowVecs + threadIdx.x; r < kpos_rows; r += gridDim.x * kRowVecs)
    kpos[r] = r - kRelExtent < T ? pos0 + r - kRelExtent : -1;
}

__global__ void ring_update_kernel(const uint4* __restrict__ k_rows, const uint4* __restrict__ v_rows, int pos0, int T,
                                   uint4* __restrict__ k_ring, uint4* __restrict__ v_ring, int64_t* __restrict__ ring_pos) {
  const int n = min(T, kRelExtent);
  const int c = T - n + blockIdx.x;  // chunk row
  const int pos = pos0 + c, slot = pos & (kRelExtent - 1);
  k_ring[size_t(slot) * kRowVecs + threadIdx.x] = k_rows[size_t(kRelExtent + c) * kRowVecs + threadIdx.x];
  v_ring[size_t(slot) * kRowVecs + threadIdx.x] = v_rows[size_t(kRelExtent + c) * kRowVecs + threadIdx.x];
  if (threadIdx.x == 0) ring_pos[slot] = pos;
}

// quill: q4 ring rows (576 B: 8 heads x 64 B codes, then 8 x 4 fp16 scales). gather: dequantized into BF16 rows
// (the values decode attention reads); update: the chunk's rows quantized into the ring (IQ4_NL, as decode)
__global__ void ring_gather_q4_kernel(const uint8_t* __restrict__ k_ring, const uint8_t* __restrict__ v_ring,
                                      const int64_t* __restrict__ ring_pos, int pos0, int T, uint4* __restrict__ k_rows,
                                      uint4* __restrict__ v_rows, int* __restrict__ kpos, int kpos_rows) {
  const int i = blockIdx.x, slot = (pos0 + i) & (kRelExtent - 1);
  const int t = threadIdx.x, h = t / 16, c = t % 16;  // 128 threads: head h, dims c*8 .. c*8+7
  for (int kv = 0; kv < 2; ++kv) {
    const uint8_t* r = (kv ? v_ring : k_ring) + size_t(slot) * 576;
    const uint32_t q = *reinterpret_cast<const uint32_t*>(r + h * 64 + c * 4);
    const float s = __half2float(reinterpret_cast<const __half*>(r + 512)[h * 4 + (c >> 2)]);
    __align__(16) __nv_bfloat16 v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = __float2bfloat16_rn(kernels::iq4nl_value(int((q >> (4 * e)) & 15u)) * s);
    (kv ? v_rows : k_rows)[size_t(i) * kRowVecs + t] = *reinterpret_cast<const uint4*>(v);
  }
  if (t == 0) kpos[i] = int(ring_pos[slot]);
  for (int r = kRelExtent + i * kRowVecs + t; r < kpos_rows; r += gridDim.x * kRowVecs)
    kpos[r] = r - kRelExtent < T ? pos0 + r - kRelExtent : -1;
}

__global__ void ring_update_q4_kernel(const __nv_bfloat16* __restrict__ k_rows, const __nv_bfloat16* __restrict__ v_rows,
                                      int pos0, int T, uint8_t* __restrict__ k_ring, uint8_t* __restrict__ v_ring,
                                      int64_t* __restrict__ ring_pos) {
  const int n = min(T, kRelExtent);
  const int c = T - n + blockIdx.x;  // chunk row
  const int pos = pos0 + c, slot = pos & (kRelExtent - 1);
  const int t = threadIdx.x;         // 64 threads: (k|v, head, 32-dim block)
  const int kv = t / 32, h = (t / 4) % 8, b = t % 4;
  const __nv_bfloat16* src = (kv ? v_rows : k_rows) + size_t(kRelExtent + c) * kKvDim + h * 128 + b * 32;
  float x[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) x[j] = __bfloat162float(src[j]);
  uint8_t codes[16];
  __half sc;
  kernels::iq4nl_quant32(x, codes, &sc);
  uint8_t* row = (kv ? v_ring : k_ring) + size_t(slot) * 576;
#pragma unroll
  for (int j = 0; j < 16; ++j) row[h * 64 + b * 16 + j] = codes[j];
  reinterpret_cast<__half*>(row + 512)[h * 4 + b] = sc;
  if (t == 0) ring_pos[slot] = pos;
}

// quill: q8 ring rows (1088 B: 8 heads x 128 int8, then 8 x 4 fp16 scales)
__global__ void ring_gather_q8_kernel(const uint8_t* __restrict__ k_ring, const uint8_t* __restrict__ v_ring,
                                      const int64_t* __restrict__ ring_pos, int pos0, int T, uint4* __restrict__ k_rows,
                                      uint4* __restrict__ v_rows, int* __restrict__ kpos, int kpos_rows) {
  const int i = blockIdx.x, slot = (pos0 + i) & (kRelExtent - 1);
  const int t = threadIdx.x, h = t / 16, c = t % 16;
  for (int kv = 0; kv < 2; ++kv) {
    const uint8_t* r = (kv ? v_ring : k_ring) + size_t(slot) * 1088;
    const uint2 q = *reinterpret_cast<const uint2*>(r + h * 128 + c * 8);
    const float s = __half2float(reinterpret_cast<const __half*>(r + 1024)[h * 4 + (c >> 2)]);
    const uint32_t w[2] = {q.x, q.y};
    __align__(16) __nv_bfloat16 v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = __float2bfloat16_rn(float(int8_t((w[e >> 2] >> (8 * (e & 3))) & 0xffu)) * s);
    (kv ? v_rows : k_rows)[size_t(i) * kRowVecs + t] = *reinterpret_cast<const uint4*>(v);
  }
  if (t == 0) kpos[i] = int(ring_pos[slot]);
  for (int r = kRelExtent + i * kRowVecs + t; r < kpos_rows; r += gridDim.x * kRowVecs)
    kpos[r] = r - kRelExtent < T ? pos0 + r - kRelExtent : -1;
}

__global__ void ring_update_q8_kernel(const __nv_bfloat16* __restrict__ k_rows, const __nv_bfloat16* __restrict__ v_rows,
                                      int pos0, int T, uint8_t* __restrict__ k_ring, uint8_t* __restrict__ v_ring,
                                      int64_t* __restrict__ ring_pos) {
  const int n = min(T, kRelExtent);
  const int c = T - n + blockIdx.x;
  const int pos = pos0 + c, slot = pos & (kRelExtent - 1);
  const int t = threadIdx.x, kv = t / 32, h = (t / 4) % 8, b = t % 4;
  const __nv_bfloat16* src = (kv ? v_rows : k_rows) + size_t(kRelExtent + c) * kKvDim + h * 128 + b * 32;
  float am = 0.f;
  for (int j = 0; j < 32; ++j) am = fmaxf(am, fabsf(__bfloat162float(src[j])));
  const __half sh = __float2half(am / 127.f);
  const float sf = __half2float(sh);
  uint8_t* row = (kv ? v_ring : k_ring) + size_t(slot) * 1088;
  for (int j = 0; j < 32; ++j) {
    const float q = sf > 0.f ? fminf(fmaxf(rintf(__bfloat162float(src[j]) / sf), -127.f), 127.f) : 0.f;
    row[h * 128 + b * 32 + j] = uint8_t(int8_t(q));
  }
  reinterpret_cast<__half*>(row + 1024)[h * 4 + b] = sh;
  if (t == 0) ring_pos[slot] = pos;
}

__global__ void bounds_kernel(const int* __restrict__ kpos, int L, int4* __restrict__ bounds) {
  const int tile = blockIdx.x, lane = threadIdx.x;
  int lo = INT_MAX, hi = INT_MIN, valid = 0;
#pragma unroll
  for (int i = 0; i < BK / 32; ++i) {
    const int j = tile * BK + i * 32 + lane;
    const int kp = j < L ? kpos[j] : -1;
    if (kp >= 0) {
      lo = min(lo, kp);
      hi = max(hi, kp);
      ++valid;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    lo = min(lo, __shfl_xor_sync(0xffffffffu, lo, o));
    hi = max(hi, __shfl_xor_sync(0xffffffffu, hi, o));
    valid += __shfl_xor_sync(0xffffffffu, valid, o);
  }
  if (lane == 0) bounds[tile] = make_int4(lo, hi, valid == BK ? 1 : 0, valid > 0 ? 1 : 0);
}

}  // namespace

void attention(const AttnParams& p, cudaStream_t stream) {
  if (p.T <= 0 || p.L <= 0 || !p.q || !p.k || !p.v || !p.rel || !p.proj || !p.out)
    throw std::invalid_argument("prefill attention: missing argument");
  const bool expl = p.kpos != nullptr;
  if (expl && (!p.bounds || (p.L + BK - 1) / BK > kAttnMaxTiles))
    throw std::invalid_argument("prefill attention: explicit keys need bounds and at most 64 tiles");
  if (p.kv_q4) {
    if (expl) throw std::invalid_argument("prefill attention: q4 caches are full-layer (implicit keys) only");
    launch<2, false>(p, stream);
  } else if (p.kv_fp8) {
    if (expl) launch<1, true>(p, stream);
    else launch<1, false>(p, stream);
  } else {
    if (expl) launch<0, true>(p, stream);
    else launch<0, false>(p, stream);
  }
}

void ring_gather(const __nv_bfloat16* k_ring, const __nv_bfloat16* v_ring, const int64_t* ring_pos, int pos0, int T,
                 __nv_bfloat16* k_rows, __nv_bfloat16* v_rows, int* kpos, int kpos_rows, cudaStream_t stream) {
  ring_gather_kernel<<<kRelExtent, kRowVecs, 0, stream>>>(
      reinterpret_cast<const uint4*>(k_ring), reinterpret_cast<const uint4*>(v_ring), ring_pos, pos0, T,
      reinterpret_cast<uint4*>(k_rows), reinterpret_cast<uint4*>(v_rows), kpos, kpos_rows);
}

void ring_update(const __nv_bfloat16* k_rows, const __nv_bfloat16* v_rows, int pos0, int T, __nv_bfloat16* k_ring,
                 __nv_bfloat16* v_ring, int64_t* ring_pos, cudaStream_t stream) {
  ring_update_kernel<<<std::min(T, kRelExtent), kRowVecs, 0, stream>>>(
      reinterpret_cast<const uint4*>(k_rows), reinterpret_cast<const uint4*>(v_rows), pos0, T,
      reinterpret_cast<uint4*>(k_ring), reinterpret_cast<uint4*>(v_ring), ring_pos);
}

void ring_gather_q4(const void* k_ring, const void* v_ring, const int64_t* ring_pos, int pos0, int T,
                    __nv_bfloat16* k_rows, __nv_bfloat16* v_rows, int* kpos, int kpos_rows, cudaStream_t stream) {
  ring_gather_q4_kernel<<<kRelExtent, kRowVecs, 0, stream>>>(
      static_cast<const uint8_t*>(k_ring), static_cast<const uint8_t*>(v_ring), ring_pos, pos0, T,
      reinterpret_cast<uint4*>(k_rows), reinterpret_cast<uint4*>(v_rows), kpos, kpos_rows);
}

void ring_update_q4(const __nv_bfloat16* k_rows, const __nv_bfloat16* v_rows, int pos0, int T, void* k_ring, void* v_ring,
                    int64_t* ring_pos, cudaStream_t stream) {
  ring_update_q4_kernel<<<std::min(T, kRelExtent), 64, 0, stream>>>(k_rows, v_rows, pos0, T, static_cast<uint8_t*>(k_ring),
                                                                    static_cast<uint8_t*>(v_ring), ring_pos);
}

void ring_gather_q8(const void* k_ring, const void* v_ring, const int64_t* ring_pos, int pos0, int T,
                    __nv_bfloat16* k_rows, __nv_bfloat16* v_rows, int* kpos, int kpos_rows, cudaStream_t stream) {
  ring_gather_q8_kernel<<<kRelExtent, kRowVecs, 0, stream>>>(
      static_cast<const uint8_t*>(k_ring), static_cast<const uint8_t*>(v_ring), ring_pos, pos0, T,
      reinterpret_cast<uint4*>(k_rows), reinterpret_cast<uint4*>(v_rows), kpos, kpos_rows);
}

void ring_update_q8(const __nv_bfloat16* k_rows, const __nv_bfloat16* v_rows, int pos0, int T, void* k_ring, void* v_ring,
                    int64_t* ring_pos, cudaStream_t stream) {
  ring_update_q8_kernel<<<std::min(T, kRelExtent), 64, 0, stream>>>(k_rows, v_rows, pos0, T, static_cast<uint8_t*>(k_ring),
                                                                    static_cast<uint8_t*>(v_ring), ring_pos);
}

void key_tile_bounds(const int* kpos, int L, int4* bounds, cudaStream_t stream) {
  bounds_kernel<<<(L + BK - 1) / BK, 32, 0, stream>>>(kpos, L, bounds);
}

}  // namespace ingine::kernels::prefill
