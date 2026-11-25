// quill: sm_89 (Ada) versions of the two cluster kernels of the dense area: router and message_gates.
//
// The Blackwell originals (router.cu, gates.cu) exchange partial results between the CTAs of a thread-block
// cluster through distributed shared memory. Ada has neither clusters nor DSMEM, so the same work split is
// kept and the exchanges go through global memory (L2) instead:
// * router: CL = 16 CTAs each compute 8 experts' logits for the M tokens into a workspace; the last CTA to
//   finish (atomic ticket, reset by itself) selects the top-8 of every token. Selection is the original's:
//   v_e = sigmoid(logit_e) + bias_e, ties to the lower expert, weights 8 s / (sum s + 1e-20).
// * message_gates: per gate CL = 16 CTAs, two grid barriers (sense-reversing, counter + generation in
//   global memory) replace the two mbarrier exchanges. Grids are G * 16 <= 48 CTAs, all co-resident on the
//   56 SMs of an RTX 4070 SUPER (one stream, no concurrent kernels).
// Per-thread arithmetic and the order of every FP32 sum follow gates.cu / router.cu.
#include "kernels/dense/dense.cuh"
#include "kernels/dense/dense.h"
#include "kernels/dense/reduce.cuh"

#include "core/device.h"

#include <cuda_bf16.h>

#include <stdexcept>
#include <type_traits>

namespace ingine::kernels::dense {
namespace {

inline void req(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("dense(sm89): ") + what);
}

// Global scratch shared by the launches of one process (decode steps are stream ordered).
struct Scratch {
  float* logits = nullptr;     // [8][128] router logits
  unsigned* ticket = nullptr;  // router ticket
  float* gall = nullptr;       // [3][8][64] gate down outputs (bf16 values)
  float* ss = nullptr;         // [3][16][8] per-CTA sums of squares
  unsigned* bar = nullptr;     // [3][2] count, generation per gate
};

Scratch& scratch() {
  static Scratch s = [] {
    Scratch t;
    const size_t bytes = 8 * 128 * 4 + 64 + 3 * 8 * 64 * 4 + 3 * 16 * 8 * 4 + 64;
    char* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    CUDA_CHECK(cudaMemset(p, 0, bytes));
    t.logits = reinterpret_cast<float*>(p);
    t.ticket = reinterpret_cast<unsigned*>(p + 8 * 128 * 4);
    t.gall = reinterpret_cast<float*>(p + 8 * 128 * 4 + 64);
    t.ss = t.gall + 3 * 8 * 64;
    t.bar = reinterpret_cast<unsigned*>(t.ss + 3 * 16 * 8);
    return t;
  }();
  return s;
}

__device__ __forceinline__ float bf16_rn(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// ------------------------------------------------------------------------------------------------ router

constexpr int EXPERTS = 128, TOPK = 8, RCL = 16, RPC = EXPERTS / RCL;  // 8 experts per CTA

template <bool WF32, int M>
__global__ void __launch_bounds__(32 * RPC) router89_kernel(const __nv_bfloat16* __restrict__ x, const void* __restrict__ w,
                                                            const float* __restrict__ bias, int* __restrict__ idx,
                                                            float* __restrict__ wts, float* __restrict__ glog,
                                                            unsigned* __restrict__ ticket) {
  constexpr int EPC = WF32 ? 4 : 8;
  constexpr int CH = H / (EPC * 32);
  __shared__ __align__(16) __nv_bfloat16 xs[M][H];
  __shared__ bool last;
  __shared__ unsigned long long keys[EXPERTS];
  __shared__ float sv[EXPERTS];
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int e = blockIdx.x * RPC + warp;
  // the expert's weight row into registers
  uint4 wv[CH];
  const uint4* wr = reinterpret_cast<const uint4*>(static_cast<const char*>(w) + size_t(e) * H * (WF32 ? 4 : 2));
#pragma unroll
  for (int i = 0; i < CH; ++i) wv[i] = __ldg(wr + i * 32 + lane);
  for (int i = t; i < M * H / 8; i += 32 * RPC) reinterpret_cast<uint4*>(&xs[0][0])[i] = reinterpret_cast<const uint4*>(x)[i];
  __syncthreads();
#pragma unroll
  for (int m = 0; m < M; ++m) {
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < CH; ++i) {
      const int col = (i * 32 + lane) * EPC;
      float xf[8];
      unpack_bf16x8(*reinterpret_cast<const uint4*>(&xs[m][col & ~7]), xf);
      if constexpr (WF32) {
        const float* xo = xf + (col & 7);
        acc = fmaf(__uint_as_float(wv[i].x), xo[0], acc);
        acc = fmaf(__uint_as_float(wv[i].y), xo[1], acc);
        acc = fmaf(__uint_as_float(wv[i].z), xo[2], acc);
        acc = fmaf(__uint_as_float(wv[i].w), xo[3], acc);
      } else {
        float wf[8];
        unpack_bf16x8(wv[i], wf);
#pragma unroll
        for (int k = 0; k < 8; ++k) acc = fmaf(wf[k], xf[k], acc);
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) glog[m * EXPERTS + e] = acc;
  }
  // ticket: the last CTA selects
  __threadfence();
  __syncthreads();
  if (t == 0) last = atomicAdd(ticket, 1u) == RCL - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int m = 0; m < M; ++m) {
    if (t < EXPERTS) {
      const float lg = __ldcg(glog + m * EXPERTS + t);
      const float s = 1.f / (1.f + __expf(-lg));
      const float v = s + bias[t];
      sv[t] = s;
      // order-preserving bits of v, ties to the lower expert
      unsigned b = __float_as_uint(v);
      b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
      keys[t] = (static_cast<unsigned long long>(b) << 32) | unsigned(EXPERTS - 1 - t);
    }
    __syncthreads();
    if (t < EXPERTS) {
      const unsigned long long k = keys[t];
      int r = 0;
      for (int j = 0; j < EXPERTS; ++j) r += keys[j] > k;
      if (r < TOPK) {
        idx[m * TOPK + r] = t;
        wts[m * TOPK + r] = sv[t];  // raw s; normalised below
      }
    }
    __syncthreads();
    if (t < TOPK) {
      const float my_s = wts[m * TOPK + t];
      float sum = my_s;
#pragma unroll
      for (int o = 4; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffu, sum, o);
      wts[m * TOPK + t] = my_s / (sum + 1e-20f) * 8.f;
    }
    __syncthreads();
  }
  if (t == 0) *ticket = 0;
}

// --------------------------------------------------------------------------------------------- gates

constexpr int RANK = 64, GCL = 16, GRPC = RANK / GCL;  // 4 down rows per CTA
constexpr int GNW = 2 * GRPC, GNT = 32 * GNW;            // 8 warps: (row, half)
constexpr int CPC = H / GCL;                             // 144 channels per CTA

__device__ __forceinline__ void grid_barrier(unsigned* count, unsigned* gen, unsigned n) {
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned g = *reinterpret_cast<volatile unsigned*>(gen);
    __threadfence();
    if (atomicAdd(count, 1u) == n - 1) {
      *reinterpret_cast<volatile unsigned*>(count) = 0;
      __threadfence();
      atomicAdd(gen, 1u);
    } else {
      while (*reinterpret_cast<volatile unsigned*>(gen) == g) {
      }
    }
    __threadfence();
  }
  __syncthreads();
}

struct Gate89Args {
  const float* h;
  bool h_shared;
  const float* msg;
  const __nv_bfloat16* w_down;
  const void* w_forget;
  const void* w_gain;
  const float* norm_w;
  __nv_bfloat16* out_bf16;
  float* out_f32;
  float* gall;  // [G][M][64]
  float* ss;    // [G][16][M]
  unsigned* bar;
};

template <bool FG32, int M>
__global__ void __launch_bounds__(GNT) gates89_kernel(Gate89Args a) {
  constexpr int DCH = H / (8 * 32);  // 9 chunks of 8 down weights per lane
  constexpr int XPT = H / GNT;       // 9
  constexpr int FGB = FG32 ? 4 : 2, EPC = 16 / FGB;
  __shared__ __align__(16) __nv_bfloat16 xs[2 * H];
  __shared__ __align__(16) float gl[M][RANK];
  __shared__ float part[M][2][GRPC];
  __shared__ float red[GNW * 32];
  const int g = blockIdx.x / GCL, rank = blockIdx.x % GCL;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int rl = warp >> 1, half = warp & 1;
  const float* hg = a.h + (a.h_shared ? size_t(0) : size_t(g) * M * H);
  const float* mg = a.msg + size_t(g) * M * H;
  unsigned* cnt = a.bar + 2 * g;
  unsigned* gen = cnt + 1;

  uint4 wv[DCH];
  {
    const __nv_bfloat16* wr = a.w_down + (size_t(g) * RANK + rank * GRPC + rl) * 2 * H + half * H;
#pragma unroll
    for (int i = 0; i < DCH; ++i) wv[i] = __ldg(reinterpret_cast<const uint4*>(wr + (i * 32 + lane) * 8));
  }
  // row norms and down projection, one token at a time
  for (int m = 0; m < M; ++m) {
    float xh[XPT], xm[XPT];
    float ss[2] = {0.f, 0.f};
#pragma unroll
    for (int k = 0; k < XPT; ++k) {
      xh[k] = hg[size_t(m) * H + k * GNT + t];
      xm[k] = mg[size_t(m) * H + k * GNT + t];
      ss[0] = fmaf(xh[k], xh[k], ss[0]);
      ss[1] = fmaf(xm[k], xm[k], ss[1]);
    }
    block_sum_bcast<2, GNW>(ss, red);
    const float rh = rsqrtf(ss[0] / H + EPS), rm = rsqrtf(ss[1] / H + EPS);
#pragma unroll
    for (int k = 0; k < XPT; ++k) {
      xs[k * GNT + t] = __float2bfloat16_rn(xh[k] * rh);
      xs[H + k * GNT + t] = __float2bfloat16_rn(xm[k] * rm);
    }
    __syncthreads();
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < DCH; ++i) {
      float wf[8], xf[8];
      unpack_bf16x8(wv[i], wf);
      unpack_bf16x8(*reinterpret_cast<const uint4*>(&xs[half * H + (i * 32 + lane) * 8]), xf);
#pragma unroll
      for (int e = 0; e < 8; ++e) acc = fmaf(wf[e], xf[e], acc);
    }
    acc = group_sum<32>(acc);
    if (lane == 0) part[m][half][rl] = acc;
    __syncthreads();
  }
  if (t < M * GRPC) {
    const int m = t / GRPC, r = t % GRPC;
    a.gall[(size_t(g) * M + m) * RANK + rank * GRPC + r] = bf16_rn(part[m][0][r] + part[m][1][r]);
  }
  grid_barrier(cnt, gen, GCL);
  for (int i = t; i < M * RANK; i += GNT) (&gl[0][0])[i] = __ldcg(a.gall + size_t(g) * M * RANK + i);
  __syncthreads();

  const int c = rank * CPC + t;
  const bool own = t < CPC;
  float hv[M], mv[M], sq[M];
#pragma unroll
  for (int m = 0; m < M; ++m) hv[m] = mv[m] = sq[m] = 0.f;
  float nw = 0.f;
  if (own) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      hv[m] = hg[size_t(m) * H + c];
      mv[m] = mg[size_t(m) * H + c];
    }
    nw = a.norm_w[size_t(g) * H + c];
    const uint8_t* fr = static_cast<const uint8_t*>(a.w_forget) + (size_t(g) * H + c) * RANK * FGB;
    const uint8_t* gr = static_cast<const uint8_t*>(a.w_gain) + (size_t(g) * H + c) * RANK * FGB;
    float df[M], dg[M];
#pragma unroll
    for (int m = 0; m < M; ++m) df[m] = dg[m] = 0.f;
#pragma unroll 4
    for (int k = 0; k < RANK / EPC; ++k) {
      const uint4 uf = __ldg(reinterpret_cast<const uint4*>(fr + k * 16));
      const uint4 ug = __ldg(reinterpret_cast<const uint4*>(gr + k * 16));
      float wf[EPC], wg[EPC];
      if constexpr (FG32) {
        wf[0] = __uint_as_float(uf.x); wf[1] = __uint_as_float(uf.y); wf[2] = __uint_as_float(uf.z); wf[3] = __uint_as_float(uf.w);
        wg[0] = __uint_as_float(ug.x); wg[1] = __uint_as_float(ug.y); wg[2] = __uint_as_float(ug.z); wg[3] = __uint_as_float(ug.w);
      } else {
        unpack_bf16x8(uf, wf);
        unpack_bf16x8(ug, wg);
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int e = 0; e < EPC; e += 4) {
          const float4 gv = *reinterpret_cast<const float4*>(&gl[m][k * EPC + e]);
          df[m] = fmaf(wf[e], gv.x, fmaf(wf[e + 1], gv.y, fmaf(wf[e + 2], gv.z, fmaf(wf[e + 3], gv.w, df[m]))));
          dg[m] = fmaf(wg[e], gv.x, fmaf(wg[e + 1], gv.y, fmaf(wg[e + 2], gv.z, fmaf(wg[e + 3], gv.w, dg[m]))));
        }
      }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      hv[m] = (1.f - tanhf(df[m])) * hv[m] + tanhf(dg[m]) * mv[m];
      sq[m] = hv[m] * hv[m];
      if (a.out_f32) a.out_f32[(size_t(g) * M + m) * H + c] = hv[m];
    }
  }
  block_sum_bcast<M, GNW>(sq, red);
  if (t < M) a.ss[(size_t(g) * GCL + rank) * 8 + t] = sq[t];
  grid_barrier(cnt, gen, GCL);
  if (own) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float ss = 0.f;
#pragma unroll
      for (int d = 0; d < GCL; ++d) ss += __ldcg(a.ss + (size_t(g) * GCL + d) * 8 + m);
      a.out_bf16[(size_t(g) * M + m) * H + c] = __float2bfloat16_rn(hv[m] * rsqrtf(ss / H + EPS) * nw);
    }
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
    default: req(false, "M must be in 1..8");
  }
}

}  // namespace

void router(const __nv_bfloat16* x, int M, const void* w, bool w_fp32, const float* bias, int* idx, float* wts,
            bool /*pdl*/, cudaStream_t stream) {
  req(x && w && bias && idx && wts, "router: null argument");
  Scratch& s = scratch();
  dispatch_m(M, [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    if (w_fp32) router89_kernel<true, MM><<<RCL, 32 * RPC, 0, stream>>>(x, w, bias, idx, wts, s.logits, s.ticket);
    else router89_kernel<false, MM><<<RCL, 32 * RPC, 0, stream>>>(x, w, bias, idx, wts, s.logits, s.ticket);
  });
  CUDA_CHECK(cudaGetLastError());
}

void message_gates(const float* h, bool h_shared, const float* msg, int G, int M, const __nv_bfloat16* w_down,
                   const void* w_forget, const void* w_gain, bool fg_fp32, const float* norm_w,
                   __nv_bfloat16* out_bf16, float* out_f32, bool /*pdl*/, cudaStream_t stream) {
  req(G >= 1 && G <= 3 && M >= 1 && M <= 8, "message_gates: G in 1..3, M in 1..8");
  req(h && msg && w_down && w_forget && w_gain && norm_w && out_bf16, "message_gates: null argument");
  Scratch& s = scratch();
  Gate89Args a{h, h_shared, msg, w_down, w_forget, w_gain, norm_w, out_bf16, out_f32, s.gall, s.ss, s.bar};
  dispatch_m(M, [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    if (fg_fp32) gates89_kernel<true, MM><<<G * GCL, GNT, 0, stream>>>(a);
    else gates89_kernel<false, MM><<<G * GCL, GNT, 0, stream>>>(a);
  });
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::dense
