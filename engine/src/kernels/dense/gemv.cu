// proj, proj_conv, head, fp8_rows: weight-streaming GEMVs for M = 1..8 tokens.
//
// Design. Decode GEMVs are pure weight streams (M <= 8 activations, K = 2304 or 3072): the goal is to keep
// DRAM busy with the fewest instructions per weight byte. Two kernels share the epilogues:
//
// M = 1, `gemv_kernel` (CUDA cores):
// * One row per lane group: a full warp per BF16 row, a half warp per FP8 row, so every lane owns exactly
//   CH = K * bytes / (16 * LANES) 16-byte chunks (BF16 K=2304: 9; FP8 K=2304: 9; K=3072: 12). Chunk i of
//   lane l is bytes [(i * LANES + l) * 16, +16) of the row: each load instruction of a group reads one
//   contiguous 256/512-byte span (fully coalesced).
// * The CH chunk loads of a row are issued at the top of the kernel (read-only streaming loads, no L1
//   allocation; ptxas interleaves them with the FMAs of earlier chunks, leaving ~4 chunks = 2 KB in
//   flight per warp), and with ~32 resident warps per SM (low register count) that is ~64 KB per SM,
//   several times the bandwidth-delay product. (Measured alternatives, no gain: persistent grids, L2 bulk
//   prefetch of the row, 128/512-thread blocks.)
// * The activation is read with 16-byte L1-cached loads at the matching columns; FP32 FMAs, then a
//   warp reduction.
//
// M >= 2, `gemv_mma_kernel` (tensor cores): on CUDA cores every weight costs M FMAs plus the unpacking of
// M activations, which made M = 8 instruction-bound (FP8 proj lost 20-40%; staging the activations in
// shared memory or giving each lane two rows was measured and did not help: both raise register or
// shared-memory pressure and cut occupancy). Instead each warp runs mma.m16n8k16 (BF16 x BF16 -> FP32):
// A = 16 weight rows, B = the activations of up to 8 tokens (N = 8 = max M; missing tokens are zero),
// so one instruction does 16 x 8 x 16 multiply-adds with FP32 accumulation, exact BF16 products as on the
// CUDA-core path. The reduction order over k is free, so the fragments are fed straight from coalesced
// 16-byte loads: lane (g, t) loads columns [c + CPL t, +CPL) of rows g and g + 8 (CPL = 8 BF16 or 16
// FP8 converted exactly to BF16) and the same columns of token g's activation; any k-permutation that A
// and B share gives the same dot products. A block of SPLITK warps owns 16 rows; warp w covers a 1/SPLITK
// slice of K (enough warps in flight for small row counts), and the partial 16 x 8 tiles are summed in
// shared memory in a fixed order. proj blocks that straddle a segment boundary with different inputs run
// one pass per input with the other rows masked to zero.
//
// Epilogues: proj rounds to BF16 (input vector per row from the device-side segment table), head applies
// the softcap, proj_conv applies the post-attention causal conv for its channel (= row) over the call's
// M tokens, rewrites that channel's conv state and optionally records the conv inputs (conv_hist), so
// every op is one launch with no grid-wide dependency. M is a template parameter; FP8 rows multiply the
// dot by the row scale.
//
// Programmatic dependent launch (pdl=True): the M = 1 kernel issues all its row's weight loads, the
// M >= 2 kernel its first batch, before griddepcontrol.wait, so they overlap the previous kernel's tail
// (pdl_wait_fenced: a plain wait lets ptxas sink the loads below it; measured M = 1 proj 13.7 -> 11.8 us
// behind a 2 us predecessor); activations are read (coherent, non-hoistable loads: ld_act) and
// outputs/state written only after it, and the next kernel is released right after the wait.
// L2: the weight rows are read once per step, so their lines are filled evict-first (kernels/common/l2.cuh):
// they stop evicting the activations, KV rings and the small kernels' evict-last constants.
// Ported from engine/csrc/dense/gemv.cu (kernel bodies unchanged; torch launchers -> raw pointers).
#include "kernels/common/l2.cuh"
#include "kernels/dense/cluster.cuh"
#include "kernels/dense/dense.cuh"
#include "kernels/dense/dense.h"

#include <cstdint>
#include <type_traits>

namespace ingine::kernels::dense {
namespace {

constexpr l2::Evict WEIGHT_L2 = l2::Evict::first;

// ---- epilogues ----------------------------------------------------------------------------------------
// input_of(row): input index of a row (-1: row not computed); x_of(i): that input, bf16 [M, K].
// kScatter epilogues store one (row, token) value at a time (store1); proj_conv needs every token of a
// row together (store: lane-parallel for the CUDA-core kernel; store_row: one thread).

template <int M>
struct ProjEpi {
  static constexpr int kInputs = 3;
  static constexpr bool kScatter = true;
  const __nv_bfloat16* xin;  // [3, M, H]
  const int* segments;       // [S, 3]
  int nseg, R;
  float* out;                // [M, R]
  __device__ int input_of(int row) const {
    for (int s = 0; s < nseg; ++s) {
      const int b = __ldg(segments + 3 * s), e = __ldg(segments + 3 * s + 1);
      if (row >= b && row < e) return __ldg(segments + 3 * s + 2);
    }
    return -1;
  }
  __device__ const __nv_bfloat16* x_of(int i) const { return xin + size_t(i) * M * H; }
  __device__ void store1(int row, int m, float v) const { out[size_t(m) * R + row] = round_bf16(v); }
};

template <int M>
struct HeadEpi {
  static constexpr int kInputs = 1;
  static constexpr bool kScatter = true;
  const __nv_bfloat16* x;
  int V;
  float* out;  // [M, V]
  const int* tok;  // quill: row -> token id (rows in frequency order), null: identity
  __device__ int input_of(int) const { return 0; }
  __device__ const __nv_bfloat16* x_of(int) const { return x; }
  __device__ void store1(int row, int m, float v) const {
    out[size_t(m) * V + (tok ? __ldg(tok + row) : row)] = tanhf(round_bf16(v) / 30.f) * 30.f;
  }
};

// plain GEMV (drafter): one input x [M, K] (row stride K), out [M, R] = bf16(w . x)
template <int M>
struct GemvEpi {
  static constexpr int kInputs = 1;
  static constexpr bool kScatter = true;
  const __nv_bfloat16* x;
  int R;
  float* out;  // [M, R]
  __device__ int input_of(int) const { return 0; }
  __device__ const __nv_bfloat16* x_of(int) const { return x; }
  __device__ void store1(int row, int m, float v) const { out[size_t(m) * R + row] = round_bf16(v); }
};

template <int M>
struct ConvEpi {
  static constexpr int kInputs = 1;
  static constexpr bool kScatter = false;  // the conv of token m needs y of tokens m-3 .. m
  const __nv_bfloat16* x;  // [M, 3072]
  const float* conv_w;     // [H, 4], tap 3 = current token
  float* state;            // [3, H], oldest first
  float* out;              // [M, H]
  float* hist;             // [M, H] BF16-rounded conv inputs y, or null
  __device__ int input_of(int) const { return 0; }
  __device__ const __nv_bfloat16* x_of(int) const { return x; }
  __device__ void load_ext(int c, const float (&acc)[M], float (&ext)[M + 3]) const {
#pragma unroll
    for (int j = 0; j < 3; ++j) ext[j] = state[j * H + c];
#pragma unroll
    for (int m = 0; m < M; ++m) ext[3 + m] = round_bf16(acc[m]);
  }
  __device__ void token(int c, int m, const float (&ext)[M + 3]) const {
    const float4 cw = __ldg(reinterpret_cast<const float4*>(conv_w) + c);
    const float s = cw.x * ext[m] + cw.y * ext[m + 1] + cw.z * ext[m + 2] + cw.w * ext[m + 3];
    out[size_t(m) * H + c] = (s + ext[m + 3]) * (1.f / sqrtf(5.f));
    if (hist) hist[size_t(m) * H + c] = ext[m + 3];
  }
  // lane-parallel (all lanes of the group hold every token's sum)
  __device__ void store(int c, const float (&acc)[M], int gl, unsigned mask) const {
    float ext[M + 3];
    load_ext(c, acc, ext);
#pragma unroll
    for (int m = 0; m < M; ++m)
      if (gl == m) token(c, m, ext);
    __syncwarp(mask);  // the group has read the old state before it is overwritten
#pragma unroll
    for (int j = 0; j < 3; ++j)
      if (gl == 8 + j) state[j * H + c] = ext[M + j];
  }
  // one thread per channel
  __device__ void store_row(int c, const float (&acc)[M]) const {
    float ext[M + 3];
    load_ext(c, acc, ext);
#pragma unroll
    for (int m = 0; m < M; ++m) token(c, m, ext);
#pragma unroll
    for (int j = 0; j < 3; ++j) state[j * H + c] = ext[M + j];
  }
};

// ---- M = 1: CUDA cores --------------------------------------------------------------------------------

template <bool FP8, int K, int LANES, int BLOCK, class Epi>
__global__ void __launch_bounds__(BLOCK) gemv_kernel(const uint8_t* __restrict__ w,
                                                     const float* __restrict__ w_scale, int rows, uint32_t zero,
                                                     Epi epi) {
  constexpr int EPC = FP8 ? 16 : 8;  // elements per 16-byte chunk
  constexpr int CH = K / (EPC * LANES);
  static_assert(K % (EPC * LANES) == 0, "row must split evenly over the lane group");
  constexpr size_t ROW_BYTES = size_t(K) * (FP8 ? 1 : 2);

  const int gl = threadIdx.x % LANES;
  const int row = (blockIdx.x * BLOCK + threadIdx.x) / LANES;
  // A whole group shares `row`, so groups exit together and the shuffles below stay within live groups.
  if (row >= rows) return;
  const int in = epi.input_of(row);
  if (in < 0) return;
  const __nv_bfloat16* x = epi.x_of(in);  // pointer arithmetic only: the segment table is a constant

  const uint8_t* wr = w + size_t(row) * ROW_BYTES;
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
  uint4 wv[CH];
#pragma unroll
  for (int i = 0; i < CH; ++i) wv[i] = l2::ld_stream<WEIGHT_L2>(wr + (size_t(i) * LANES + gl) * 16, wpol);
  pdl_wait_fenced(zero);  // the whole row is in flight; the activation comes from an earlier kernel
  pdl_trigger();

  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    const int col = (i * LANES + gl) * EPC;
    float wf[EPC], xf[EPC];
    if constexpr (FP8) unpack_e4m3x16(wv[i], wf);
    else unpack_bf16x8(wv[i], wf);
    unpack_bf16x8(ld_act(x + col), xf);
    if constexpr (FP8) unpack_bf16x8(ld_act(x + col + 8), xf + 8);
#pragma unroll
    for (int j = 0; j < EPC; ++j) acc = fmaf(wf[j], xf[j], acc);
  }

  float v[1] = {group_sum<LANES>(acc) * (FP8 ? __ldg(w_scale + row) : 1.f)};
  if constexpr (Epi::kScatter) {
    if (gl == 0) epi.store1(row, 0, v[0]);
  } else {
    epi.store(row, v, gl, group_mask<LANES>());
  }
}

// ---- M >= 2: tensor cores -----------------------------------------------------------------------------

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 2 E4M3 (low 16 bits of u) -> BF16x2, exact: one instruction (cvt.rn.bf16x2.e4m3x2; through FP16 and
// FP32 it took four, 16 per mma)
__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint32_t u) {
  uint32_t r;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(static_cast<unsigned short>(u)));
#else  // quill (sm_89): E4M3 -> f16x2 (one cvt), then exact f16 -> bf16
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u & 0xffffu), __NV_E4M3);
  const float2 f = __half22float2(__half2(h));
  const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
  r = *reinterpret_cast<const uint32_t*>(&b);
#endif
  return r;
}

#ifndef MMA_U_BF16
#define MMA_U_BF16 6
#endif
#ifndef MMA_U_FP8
#define MMA_U_FP8 3
#endif
// `zero` is 0 at run time (opaque to the compiler): every mma of a batch takes an operand xor-ed with
// (xor of all the batch's loads) & zero, so ptxas cannot interleave the mma with the batch's loads and
// all U steps' loads of a warp are in flight together (otherwise it issues one step's loads at a time,
// ~1 KB per warp, and scattered 64-byte requests from 16 rows run at ~40% of DRAM bandwidth).
template <bool FP8, int K, int M, int SPLITK, int U, class Epi>
__global__ void __launch_bounds__(32 * SPLITK) gemv_mma_kernel(const uint8_t* __restrict__ w,
                                                              const float* __restrict__ w_scale, int rows,
                                                              uint32_t zero, Epi epi) {
  constexpr int CPL = FP8 ? 16 : 8;  // columns per lane per 16-byte weight load
  constexpr int SPAN = 4 * CPL;      // columns per step (4 lanes t)
  constexpr int KS = K / SPLITK;     // columns per warp
  static_assert(KS % (SPAN * U) == 0, "K slice must split into unrolled steps");
  constexpr int NJ = KS / SPAN;
  constexpr size_t ROW_BYTES = size_t(K) * (FP8 ? 1 : 2);
  __shared__ float cs[SPLITK][16][8];

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.x * 16;
  const int ra = r0 + g, rb = r0 + g + 8;
  const int ia = ra < rows ? epi.input_of(ra) : -1, ib = rb < rows ? epi.input_of(rb) : -1;
  const int c0 = warp * KS + t * CPL;

  float c[4] = {0.f, 0.f, 0.f, 0.f};
  bool waited = false;
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
  auto pass = [&](int cur) {
    const bool la = ia == cur, lb = ib == cur, lx = g < M;
    const uint8_t* wa = w + size_t(la ? ra : 0) * ROW_BYTES;
    const uint8_t* wb = w + size_t(lb ? rb : 0) * ROW_BYTES;
    const __nv_bfloat16* xg = epi.x_of(cur) + size_t(lx ? g : 0) * K;
#pragma unroll 1
    for (int jb = 0; jb < NJ; jb += U) {
      uint4 A[U], B[U], X[U][CPL / 8];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const size_t off = size_t(c0 + (jb + u) * SPAN) * (FP8 ? 1 : 2);
        A[u] = la ? l2::ld_stream<WEIGHT_L2>(wa + off, wpol) : make_uint4(0, 0, 0, 0);
        B[u] = lb ? l2::ld_stream<WEIGHT_L2>(wb + off, wpol) : make_uint4(0, 0, 0, 0);
      }
      if (!waited) {  // the first batch of weights is in flight; activations come from earlier kernels
        pdl_wait_fenced(zero);
        pdl_trigger();
        waited = true;
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int col = c0 + (jb + u) * SPAN;
#pragma unroll
        for (int h = 0; h < CPL / 8; ++h) X[u][h] = lx ? ld_act(xg + col + 8 * h) : make_uint4(0, 0, 0, 0);
      }
      uint32_t dep = 0;
#pragma unroll
      for (int u = 0; u < U; ++u) {
        dep ^= A[u].x ^ A[u].w ^ B[u].x ^ B[u].w;
#pragma unroll
        for (int h = 0; h < CPL / 8; ++h) dep ^= X[u][h].x;
      }
      dep &= zero;
#pragma unroll
      for (int u = 0; u < U; ++u) {
        if constexpr (FP8) {
          // 16 columns = 4 k-steps of 4 columns (2 k-pairs) each
          const uint32_t aw[4] = {A[u].x, A[u].y, A[u].z, A[u].w}, bw[4] = {B[u].x, B[u].y, B[u].z, B[u].w};
          const uint32_t xw[8] = {X[u][0].x, X[u][0].y, X[u][0].z, X[u][0].w,
                                  X[u][1].x, X[u][1].y, X[u][1].z, X[u][1].w};
#pragma unroll
          for (int s = 0; s < 4; ++s)
            mma_bf16(c, e4m3x2_to_bf16x2(aw[s]), e4m3x2_to_bf16x2(bw[s]), e4m3x2_to_bf16x2(aw[s] >> 16),
                     e4m3x2_to_bf16x2(bw[s] >> 16), xw[2 * s] ^ dep, xw[2 * s + 1]);
        } else {
          mma_bf16(c, A[u].x, B[u].x, A[u].y, B[u].y, X[u][0].x ^ dep, X[u][0].y);
          mma_bf16(c, A[u].z, B[u].z, A[u].w, B[u].w, X[u][0].z ^ dep, X[u][0].w);
        }
      }
    }
  };
  if constexpr (Epi::kInputs == 1) {
    pass(0);
  } else {
    // inputs present among the block's 16 rows (warp-uniform; every warp sees the same rows)
#pragma unroll 1
    for (int i = 0; i < Epi::kInputs; ++i)
      if (__any_sync(0xffffffffu, ia == i || ib == i)) pass(i);
  }

  // C fragment: c0, c1 = rows g, tokens 2t, 2t+1; c2, c3 = row g + 8
  cs[warp][g][2 * t] = c[0];
  cs[warp][g][2 * t + 1] = c[1];
  cs[warp][g + 8][2 * t] = c[2];
  cs[warp][g + 8][2 * t + 1] = c[3];
  __syncthreads();
  auto total = [&](int r, int m) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < SPLITK; ++k) s += cs[k][r][m];
    return s;
  };
  if constexpr (Epi::kScatter) {
    for (int k = threadIdx.x; k < 16 * M; k += 32 * SPLITK) {
      const int r = k / M, m = k % M, row = r0 + r;
      if (row < rows && epi.input_of(row) >= 0)
        epi.store1(row, m, total(r, m) * (FP8 ? __ldg(w_scale + row) : 1.f));
    }
  } else {
    const int r = threadIdx.x, row = r0 + r;
    if (r < 16 && row < rows) {
      const float scale = FP8 ? __ldg(w_scale + row) : 1.f;
      float y[M];
#pragma unroll
      for (int m = 0; m < M; ++m) y[m] = total(r, m) * scale;
      epi.store_row(row, y);
    }
  }
}

#include "kernels/dense/fp6.cuh"
#include "kernels/dense/fp4.cuh"

// ---- launch helpers -----------------------------------------------------------------------------------

// BLOCK: threads per block of the M = 1 kernel (0: M = 1 runs the tensor-core kernel too, for K whose
// whole row per lane would not fit in registers); SPLITK: warps per 16-row block of the M >= 2 kernel;
// UFP8: steps per load batch of its FP8 instantiation.
template <int K, int BLOCK, int SPLITK, int UFP8, int UBF16, template <int> class Epi, class Make>
void launch_gemv(const Matrix& w, int M, bool pdl, cudaStream_t stream, Make make) {
  const bool fp4 = w.bscale != nullptr && w.fp4;
  const bool fp6 = w.bscale != nullptr && !w.fp4;
  const bool fp8 = !fp6 && !fp4 && w.scale != nullptr;
  require(w.w != nullptr && w.rows > 0, "weight matrix is empty");
  const int rows = w.rows;
  const float* sc = w.scale;
  const uint8_t* wp = static_cast<const uint8_t*>(w.w);
  auto go = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    auto epi = make(std::integral_constant<int, MM>{});
    using E = decltype(epi);
    constexpr int KS6 = K / SPLITK;  // quill: FP6 steps of 128 columns per warp slice
    constexpr bool fp6_mma_ok = KS6 % 128 == 0;
    constexpr int U6 = fp6_mma_ok && (KS6 / 128) % 3 == 0 ? 3 : 1;
    if constexpr (MM == 1 && BLOCK > 0) {
      constexpr int B = BLOCK > 0 ? BLOCK : 32;  // (the branch is discarded for BLOCK = 0)
      if (fp4) {
        if constexpr (K % 256 == 0)
          launch_ex(gemv4_kernel<K, B, E>, dim3((rows * 16 + B - 1) / B), B, 0, 1, pdl, stream, wp, w.bscale, sc, rows, 0u, epi);
        else
          require(false, "FP4 rows: K must be a multiple of 256");
      } else if (fp6) {
        if constexpr (K % 256 == 0)
          launch_ex(gemv6_kernel<K, B, E>, dim3((rows * 16 + B - 1) / B), B, 0, 1, pdl, stream, wp, w.bscale, sc, rows, 0u, epi);
        else
          require(false, "FP6 rows: K must be a multiple of 256");
      } else if (fp8) launch_ex(gemv_kernel<true, K, 16, B, E>, dim3((rows * 16 + B - 1) / B), B, 0, 1, pdl, stream, wp, sc, rows, 0u, epi);
      else launch_ex(gemv_kernel<false, K, 32, B, E>, dim3((rows * 32 + B - 1) / B), B, 0, 1, pdl, stream, wp, sc, rows, 0u, epi);
    } else {
      const dim3 blocks((rows + 15) / 16);
      if (fp4) {
        if constexpr (fp6_mma_ok)
          launch_ex(gemv4_mma_kernel<K, MM, SPLITK, U6, E>, blocks, 32 * SPLITK, 0, 1, pdl, stream, wp, w.bscale, sc, rows, 0u, epi);
        else
          require(false, "FP4 rows: K slice per warp must be a multiple of 128 columns");
      } else if (fp6) {
        if constexpr (fp6_mma_ok)
          launch_ex(gemv6_mma_kernel<K, MM, SPLITK, U6, E>, blocks, 32 * SPLITK, 0, 1, pdl, stream, wp, w.bscale, sc, rows, 0u, epi);
        else
          require(false, "FP6 rows: K slice per warp must be a multiple of 128 columns");
      } else if (fp8) {
        if constexpr ((K / SPLITK) % (64 * UFP8) == 0)  // quill: an FP6-only split may not fit the FP8 steps
          launch_ex(gemv_mma_kernel<true, K, MM, SPLITK, UFP8, E>, blocks, 32 * SPLITK, 0, 1, pdl, stream, wp, sc, rows, 0u, epi);
        else
          require(false, "FP8 rows: K slice does not split into the unrolled steps");
      } else {
        if constexpr ((K / SPLITK) % (32 * UBF16) == 0)
          launch_ex(gemv_mma_kernel<false, K, MM, SPLITK, UBF16, E>, blocks, 32 * SPLITK, 0, 1, pdl, stream, wp, sc, rows, 0u, epi);
        else
          require(false, "BF16 rows: K slice does not split into the unrolled steps");
      }
    }
  };
  switch (M) {
    case 1: go(std::integral_constant<int, 1>{}); break;
    case 2: go(std::integral_constant<int, 2>{}); break;
    case 3: go(std::integral_constant<int, 3>{}); break;
    case 4: go(std::integral_constant<int, 4>{}); break;
    case 5: go(std::integral_constant<int, 5>{}); break;
    case 6: go(std::integral_constant<int, 6>{}); break;
    case 7: go(std::integral_constant<int, 7>{}); break;
    case 8: go(std::integral_constant<int, 8>{}); break;
    default: require(false, "M must be in 1..8");
  }
}

// Warps per 16-row block of the M >= 2 kernel (K slice per warp must hold a whole number of U-step
// batches). Measured at M = 2 and 8 (shared GPU): proj 6 > 4 > 2 (FP8 +4-8%), head 6 ~ 4 >> 2 >> 1
// (+20-30%: 12.5K blocks need short K slices to keep enough rows in flight), proj_conv 4 ~ 8 ~ 16.
#ifndef PROJ_SPLITK
#define PROJ_SPLITK 6
#endif
#ifndef CONV_SPLITK
#define CONV_SPLITK 8
#endif
// FP8 steps per load batch of proj_conv's M >= 2 kernel: 6 = a warp's whole K slice. Its 144 blocks are all
// resident at once (2 per SM at 120 registers), so every weight byte is requested before the PDL wait, as in
// the M = 1 kernel (measured against 3 steps: 5.9 -> 4.9 us per call at M = 2, 6.6 -> 6.0 at M = 8). proj
// (538 blocks) loses with 6: half its blocks would wait for a slot (14.0 -> 14.5 us at M = 2).
#ifndef CONV_U_FP8
#define CONV_U_FP8 6
#endif
#ifndef HEAD_SPLITK
#define HEAD_SPLITK 6
#endif

__global__ void fp8_rows_kernel(const __nv_bfloat16* __restrict__ w, int rows, int K, __nv_fp8_storage_t* q,
                                float* scale) {
  const int lane = threadIdx.x & 31;
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= rows) return;
  const __nv_bfloat16* wr = w + size_t(row) * K;
  float amax = 0.f;
  for (int k = lane; k < K; k += 32) amax = fmaxf(amax, fabsf(__bfloat162float(wr[k])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float s = amax / 448.f;
  if (lane == 0) scale[row] = s;
  for (int k = lane; k < K; k += 32)
    q[size_t(row) * K + k] =
        s > 0.f ? __nv_cvt_float_to_fp8(__bfloat162float(wr[k]) / s, __NV_SATFINITE, __NV_E4M3) : 0;
}

// weights.dequant: bf16(float(q) * scale[row]); one thread per 16 elements (K % 16 == 0).
__global__ void dequant_rows_kernel(const __nv_fp8_storage_t* __restrict__ q, const float* __restrict__ scale,
                                    int K, long long total16, __nv_bfloat16* __restrict__ out) {
  const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= total16) return;
  const long long e = i * 16;
  const float s = scale[e / K];
  const uint4 u = *reinterpret_cast<const uint4*>(q + e);
  float f[16];
  unpack_e4m3x16(u, f);
  __align__(16) __nv_bfloat16 v[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) v[j] = __float2bfloat16_rn(f[j] * s);
  uint4* dst = reinterpret_cast<uint4*>(out + e);
  dst[0] = reinterpret_cast<const uint4*>(v)[0];
  dst[1] = reinterpret_cast<const uint4*>(v)[1];
}

}  // namespace

void proj(const __nv_bfloat16* xin, int M, Matrix w, const int* segments, int nseg, float* out, bool pdl,
          cudaStream_t stream) {
  require(xin && segments && out && nseg > 0, "proj: null argument");
  const int R = w.rows;
  launch_gemv<H, 256, PROJ_SPLITK, MMA_U_FP8, MMA_U_BF16, ProjEpi>(w, M, pdl, stream, [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    return ProjEpi<MM>{xin, segments, nseg, R, out};
  });
}

void proj_conv(const __nv_bfloat16* x, int M, Matrix w, const float* conv_w, float* conv_state, float* out,
               float* conv_hist, bool pdl, cudaStream_t stream) {
  require(x && conv_w && conv_state && out, "proj_conv: null argument");
  require(w.rows == H, "proj_conv: w must have 2304 rows");
  // M = 1: 2304 rows = 1152-2304 warps; 32-thread blocks spread them evenly over the SMs (measured: equal
  // or better than 64-256)
  launch_gemv<3072, 32, CONV_SPLITK, CONV_U_FP8, MMA_U_BF16, ConvEpi>(w, M, pdl, stream, [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    return ConvEpi<MM>{x, conv_w, conv_state, out, conv_hist};
  });
}

void head(const __nv_bfloat16* x, int M, Matrix w, float* out, bool pdl, cudaStream_t stream, const int* row_tok,
          int ldo) {
  require(x && out, "head: null argument");
  const int V = ldo > 0 ? ldo : w.rows;
  launch_gemv<H, 256, HEAD_SPLITK, MMA_U_FP8, MMA_U_BF16, HeadEpi>(w, M, pdl, stream, [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    return HeadEpi<MM>{x, V, out, row_tok};
  });
}

void gemv(const __nv_bfloat16* x, int M, int K, Matrix w, float* out, bool pdl, cudaStream_t stream) {
  require(x && out, "gemv: null argument");
  const int R = w.rows;
  auto make = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    return GemvEpi<MM>{x, R, out};
  };
  // 2304 rows = 144 blocks of 16 rows: 16 warps per block (K slices of 192 / 576 columns) keep enough
  // bytes in flight (measured M = 8, BF16: K = 3072 10.4 -> 8.4 us, K = 9216 28.3 -> 26.0 us vs 8 warps).
  // K = 3072, M = 1: as proj_conv; K = 9216: 36 chunks per lane would spill, so M = 1 uses tensor cores.
  if (K == 3072) launch_gemv<3072, 32, 16, MMA_U_FP8, MMA_U_BF16, GemvEpi>(w, M, pdl, stream, make);
  else if (K == 9216 && w.bscale) launch_gemv<9216, 0, 18, MMA_U_FP8, MMA_U_BF16, GemvEpi>(w, M, pdl, stream, make);  // FP6: 128-col steps
  else if (K == 9216) launch_gemv<9216, 0, 16, MMA_U_FP8, MMA_U_BF16, GemvEpi>(w, M, pdl, stream, make);
  else if (K == 4096) launch_gemv<4096, 0, 16, 4, 8, GemvEpi>(w, M, pdl, stream, make);  // quill: DFlash2 o_proj
  else if (K == 6912 && w.bscale) launch_gemv<6912, 0, 18, MMA_U_FP8, MMA_U_BF16, GemvEpi>(w, M, pdl, stream, make);  // FP6
  else if (K == 6912) launch_gemv<6912, 0, 12, MMA_U_FP8, MMA_U_BF16, GemvEpi>(w, M, pdl, stream, make);  // quill: DFlash2 down
  else require(false, "gemv: K must be 3072, 4096, 6912 or 9216");
}

void fp8_rows(const __nv_bfloat16* w, int rows, int K, void* q, float* scale, cudaStream_t stream) {
  require(rows >= 0 && K > 0, "fp8_rows: bad shape");
  if (rows > 0)
    fp8_rows_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(w, rows, K, static_cast<__nv_fp8_storage_t*>(q), scale);
  CUDA_CHECK(cudaGetLastError());
}

void fp4_rows(const __nv_bfloat16* w, int rows, int K, void* q, std::uint8_t* bscale, float* scale, cudaStream_t stream) {
  require(rows >= 0 && K > 0 && K % 256 == 0, "fp4_rows: K must be a multiple of 256");
  if (rows > 0) fp4_rows_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(w, rows, K, static_cast<uint8_t*>(q), bscale, scale);
  CUDA_CHECK(cudaGetLastError());
}

void fp6_rows(const __nv_bfloat16* w, int rows, int K, void* q, std::uint8_t* bscale, float* scale, cudaStream_t stream) {
  require(rows >= 0 && K > 0 && K % 256 == 0, "fp6_rows: K must be a multiple of 256");
  if (rows > 0) fp6_rows_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(w, rows, K, static_cast<uint8_t*>(q), bscale, scale);
  CUDA_CHECK(cudaGetLastError());
}

void dequant_rows6(const void* q, const std::uint8_t* bscale, const float* scale, int rows, int K, __nv_bfloat16* out,
                   cudaStream_t stream) {
  require(rows >= 0 && K % 256 == 0, "dequant_rows6: K must be a multiple of 256");
  const long long total16 = (long long)rows * K / 16;
  if (total16 > 0)
    dequant_rows6_kernel<<<unsigned((total16 + 255) / 256), 256, 0, stream>>>(static_cast<const uint8_t*>(q), bscale,
                                                                            scale, K, total16, out);
  CUDA_CHECK(cudaGetLastError());
}

void dequant_rows(const void* q, const float* scale, int rows, int K, __nv_bfloat16* out, cudaStream_t stream) {
  require(rows >= 0 && K > 0 && K % 16 == 0, "dequant_rows: K must be a multiple of 16");
  const long long total16 = (long long)rows * K / 16;
  if (total16 > 0)
    dequant_rows_kernel<<<unsigned((total16 + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_fp8_storage_t*>(q), scale, K, total16, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::dense

// ---------------------------------------------------------------------------------------------------------------
// quill experiment: NVFP4 round trip of a BF16 matrix in place (quality measurement of 4-bit dense weights).
// Per row an FP32 scale g = amax / (448 * 6); per 16 values an E4M3 scale s = e4m3(amax16 / 6 / g); values
// rounded to the nearest E2M1 magnitude {0, .5, 1, 1.5, 2, 3, 4, 6} * s * g.
namespace ingine::kernels::dense {
namespace {
__device__ __forceinline__ float e2m1_round(float a) {  // a >= 0, in units of the block scale
  const float lv[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float best = 0.f, bd = 1e30f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float d = fabsf(a - lv[i]);
    if (d < bd) bd = d, best = lv[i];
  }
  return best;
}
__global__ void __launch_bounds__(256) nvfp4_roundtrip_kernel(__nv_bfloat16* w, int K) {
  __nv_bfloat16* row = w + std::size_t(blockIdx.x) * K;
  __shared__ float red[8];
  float amax = 0.f;
  for (int c = threadIdx.x; c < K; c += blockDim.x) amax = fmaxf(amax, fabsf(__bfloat162float(row[c])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
  __syncthreads();
  amax = 0.f;
  for (int i = 0; i < 8; ++i) amax = fmaxf(amax, red[i]);
  const float g = amax > 0.f ? amax / (448.f * 6.f) : 1.f;
  for (int b = threadIdx.x; b < K / 16; b += blockDim.x) {
    float v[16], bmax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = __bfloat162float(row[b * 16 + i]), bmax = fmaxf(bmax, fabsf(v[i]));
    const float s = float(__nv_fp8_e4m3(bmax / 6.f / g)) * g;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float q = s > 0.f ? copysignf(e2m1_round(fabsf(v[i]) / s), v[i]) * s : 0.f;
      row[b * 16 + i] = __float2bfloat16_rn(q);
    }
  }
}
}  // namespace
void nvfp4_roundtrip(__nv_bfloat16* w, int rows, int K, cudaStream_t stream) {
  nvfp4_roundtrip_kernel<<<rows, 256, 0, stream>>>(w, K);
  CUDA_CHECK(cudaGetLastError());
}
}  // namespace ingine::kernels::dense

// quill experiment: FP6 round trip (E2M3: max 7.5, E3M2: max 28) in blocks of `blk` values with an E4M3 block scale
// under a per-row FP32 scale (the NVFP4 layout with 6-bit values).
namespace ingine::kernels::dense {
namespace {
// fmt 0 E2M3 (6-bit), 1 E3M2 (6-bit), 2 E2M2 (5-bit: subnormal step .25, max 7)
__device__ __forceinline__ float fp6_round(float a, int fmt) {  // a >= 0, in block-scale units
  if (fmt == 2) {
    if (a >= 7.f) return 7.f;
    const float step = a < 1.f ? 0.25f : exp2f(floorf(log2f(a))) / 4.f;
    return rintf(a / step) * step;
  }
  const bool e3m2 = fmt == 1;
  // E2M3: exponent bias 1, 3 mantissa bits, subnormal step 0.125, max 7.5; E3M2: bias 3, 2 mantissa bits, step
  // 0.0625, max 28
  const float maxv = e3m2 ? 28.f : 7.5f;
  if (a >= maxv) return maxv;
  const int mb = e3m2 ? 2 : 3;
  const float minnorm = e3m2 ? 0.25f : 1.f;
  float step;
  if (a < minnorm) step = minnorm / float(1 << mb);
  else step = exp2f(floorf(log2f(a))) / float(1 << mb);
  return rintf(a / step) * step;
}
__global__ void __launch_bounds__(256) fp6_roundtrip_kernel(__nv_bfloat16* w, int K, int blk, int fmt) {
  __nv_bfloat16* row = w + std::size_t(blockIdx.x) * K;
  __shared__ float red[8];
  float amax = 0.f;
  for (int c = threadIdx.x; c < K; c += blockDim.x) amax = fmaxf(amax, fabsf(__bfloat162float(row[c])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
  __syncthreads();
  amax = 0.f;
  for (int i = 0; i < 8; ++i) amax = fmaxf(amax, red[i]);
  const float maxv = fmt == 1 ? 28.f : fmt == 2 ? 7.f : 7.5f;
  const float g = amax > 0.f ? amax / (448.f * maxv) : 1.f;
  for (int b = threadIdx.x; b < K / blk; b += blockDim.x) {
    float bmax = 0.f;
    for (int i = 0; i < blk; ++i) bmax = fmaxf(bmax, fabsf(__bfloat162float(row[b * blk + i])));
    const float s = float(__nv_fp8_e4m3(bmax / maxv / g)) * g;
    for (int i = 0; i < blk; ++i) {
      const float v = __bfloat162float(row[b * blk + i]);
      const float q = s > 0.f ? copysignf(fp6_round(fabsf(v) / s, fmt), v) * s : 0.f;
      row[b * blk + i] = __float2bfloat16_rn(q);
    }
  }
}
}  // namespace
void fp6_roundtrip(__nv_bfloat16* w, int rows, int K, int blk, int fmt, cudaStream_t stream) {
  fp6_roundtrip_kernel<<<rows, 256, 0, stream>>>(w, K, blk, fmt);
  CUDA_CHECK(cudaGetLastError());
}
}  // namespace ingine::kernels::dense
