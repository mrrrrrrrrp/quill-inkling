// quill: FP4 (E2M1, NVFP4-style) dense rows. Included by gemv.cu after fp6.cuh (uses its helpers).
//
// Format (per row of K values): [K/2 bytes: codes s e1 e0 m0, value 2i in the low nibble of byte i]; per 16 values one
// E4M3 block scale (separate array [rows][K/16]); per row one FP32 scale (x 64). Decoding: the byte
// s<<7 | (e1 e0 m0)<<2 read as E4M3 is exactly value(c) x 2^-6 (E2M1 0.5 = E4M3 subnormal 4/8 x 2^-6, ...), so the
// hardware E4M3 -> FP16 conversion does it and the row scale carries the 2^6. 4.5 bits per value (FP6: 6.25).

// 8 codes -> 8 E4M3 bytes (values 0-3 in w0, 4-7 in w1, element 0 in the low byte)
__device__ __forceinline__ void fp4_e4m3x8(uint32_t n, uint32_t& w0, uint32_t& w1) {
  const uint32_t lo = n & 0x0f0f0f0fu, hi = (n >> 4) & 0x0f0f0f0fu;
  const uint32_t a = __byte_perm(lo, hi, 0x5140), b = __byte_perm(lo, hi, 0x7362);
  w0 = ((a & 0x08080808u) << 4) | ((a & 0x07070707u) << 2);
  w1 = ((b & 0x08080808u) << 4) | ((b & 0x07070707u) << 2);
}

// ---- M = 1: 16 lanes per row, 16-value chunks (8 bytes + one block scale each) ----
template <int K, int BLOCK, class Epi>
__global__ void __launch_bounds__(BLOCK) gemv4_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ bsc,
                                                      const float* __restrict__ w_scale, int rows, uint32_t zero,
                                                      Epi epi) {
  constexpr int LANES = 16, CH = K / (16 * LANES);
  static_assert(K % (16 * LANES) == 0, "row must split evenly over the lane group");
  constexpr size_t ROW_BYTES = size_t(K) / 2;
  const int gl = threadIdx.x % LANES;
  const int row = (blockIdx.x * BLOCK + threadIdx.x) / LANES;
  if (row >= rows) return;
  const int in = epi.input_of(row);
  if (in < 0) return;
  const __nv_bfloat16* x = epi.x_of(in);
  const uint8_t* wr = w + size_t(row) * ROW_BYTES;
  const uint8_t* br = bsc + size_t(row) * (K / 16);
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
  uint2 lo[CH];
  uint32_t bs[CH];
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    const int ch = i * LANES + gl;
    lo[i] = ld_stream8(wr + size_t(ch) * 8, wpol);
    bs[i] = __ldg(br + ch);
  }
  pdl_wait_fenced(zero);
  pdl_trigger();
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    const int col = (i * LANES + gl) * 16;
    float part = 0.f;
#pragma unroll
    for (int q = 0; q < 2; ++q) {
      uint32_t w0, w1;
      fp4_e4m3x8(q ? lo[i].y : lo[i].x, w0, w1);
      float wf[8], xf[8];
      unpack_e4m3x4(w0, wf);
      unpack_e4m3x4(w1, wf + 4);
      unpack_bf16x8(ld_act(x + col + 8 * q), xf);
#pragma unroll
      for (int j = 0; j < 8; ++j) part = fmaf(wf[j], xf[j], part);
    }
    acc = fmaf(part, e4m3_float(bs[i]), acc);
  }
  float v[1] = {group_sum<LANES>(acc) * __ldg(w_scale + row)};
  if constexpr (Epi::kScatter) {
    if (gl == 0) epi.store1(row, 0, v[0]);
  } else {
    epi.store(row, v, gl, group_mask<LANES>());
  }
}

// ---- M >= 2: tensor cores, 32 columns per lane per step (SPAN 128), as gemv6_mma_kernel ----
template <int K, int M, int SPLITK, int U, class Epi>
__global__ void __launch_bounds__(32 * SPLITK) gemv4_mma_kernel(const uint8_t* __restrict__ w,
                                                               const uint8_t* __restrict__ bsc,
                                                               const float* __restrict__ w_scale, int rows,
                                                               uint32_t zero, Epi epi) {
  constexpr int CPL = 32, SPAN = 4 * CPL, KS = K / SPLITK;
  static_assert(KS % (SPAN * U) == 0, "K slice must split into unrolled steps");
  constexpr int NJ = KS / SPAN;
  constexpr size_t ROW_BYTES = size_t(K) / 2;
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
    const size_t rowa = la ? ra : 0, rowb = lb ? rb : 0;
    const uint8_t* wa = w + rowa * ROW_BYTES;
    const uint8_t* wb = w + rowb * ROW_BYTES;
    const uint8_t* sa = bsc + rowa * (K / 16);
    const uint8_t* sb = bsc + rowb * (K / 16);
    const __nv_bfloat16* xg = epi.x_of(cur) + size_t(lx ? g : 0) * K;
#pragma unroll 1
    for (int jb = 0; jb < NJ; jb += U) {
      uint4 AL[U], BL[U], X[U][4];
      uint32_t AS[U], BS[U];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int col = c0 + (jb + u) * SPAN;
        AL[u] = la ? l2::ld_stream<WEIGHT_L2>(wa + col / 2, wpol) : make_uint4(0, 0, 0, 0);
        BL[u] = lb ? l2::ld_stream<WEIGHT_L2>(wb + col / 2, wpol) : make_uint4(0, 0, 0, 0);
        AS[u] = la ? uint32_t(__ldg(reinterpret_cast<const unsigned short*>(sa + col / 16))) : 0u;
        BS[u] = lb ? uint32_t(__ldg(reinterpret_cast<const unsigned short*>(sb + col / 16))) : 0u;
      }
      if (!waited) {
        pdl_wait_fenced(zero);
        pdl_trigger();
        waited = true;
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int col = c0 + (jb + u) * SPAN;
#pragma unroll
        for (int h = 0; h < 4; ++h) X[u][h] = lx ? ld_act(xg + col + 8 * h) : make_uint4(0, 0, 0, 0);
      }
      uint32_t dep = 0;
#pragma unroll
      for (int u = 0; u < U; ++u) dep ^= AL[u].x ^ AL[u].w ^ BL[u].x ^ BL[u].w ^ X[u][0].x ^ X[u][3].w;
      dep &= zero;
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const float fa0 = e4m3_float(AS[u]), fa1 = e4m3_float(AS[u] >> 8);
        const float fb0 = e4m3_float(BS[u]), fb1 = e4m3_float(BS[u] >> 8);
        const uint32_t an[4] = {AL[u].x, AL[u].y, AL[u].z, AL[u].w}, bn[4] = {BL[u].x, BL[u].y, BL[u].z, BL[u].w};
        const uint32_t xw[16] = {X[u][0].x, X[u][0].y, X[u][0].z, X[u][0].w, X[u][1].x, X[u][1].y, X[u][1].z, X[u][1].w,
                                 X[u][2].x, X[u][2].y, X[u][2].z, X[u][2].w, X[u][3].x, X[u][3].y, X[u][3].z, X[u][3].w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {  // 8 values per word; block scale 0 for values 0-15, 1 for 16-31
          const float fa = q < 2 ? fa0 : fa1, fb = q < 2 ? fb0 : fb1;
          uint32_t a0, a1, b0, b1;
          fp4_e4m3x8(an[q], a0, a1);
          fp4_e4m3x8(bn[q], b0, b1);
          mma_bf16(c, e4m3x2_scaled_bf16x2(a0, fa), e4m3x2_scaled_bf16x2(b0, fb), e4m3x2_scaled_bf16x2(a0 >> 16, fa),
                   e4m3x2_scaled_bf16x2(b0 >> 16, fb), xw[4 * q] ^ dep, xw[4 * q + 1]);
          mma_bf16(c, e4m3x2_scaled_bf16x2(a1, fa), e4m3x2_scaled_bf16x2(b1, fb), e4m3x2_scaled_bf16x2(a1 >> 16, fa),
                   e4m3x2_scaled_bf16x2(b1 >> 16, fb), xw[4 * q + 2] ^ dep, xw[4 * q + 3]);
        }
      }
    }
  };
  if constexpr (Epi::kInputs == 1) {
    pass(0);
  } else {
#pragma unroll 1
    for (int i = 0; i < Epi::kInputs; ++i)
      if (__any_sync(0xffffffffu, ia == i || ib == i)) pass(i);
  }
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
      if (row < rows && epi.input_of(row) >= 0) epi.store1(row, m, total(r, m) * __ldg(w_scale + row));
    }
  } else {
    const int r = threadIdx.x, row = r0 + r;
    if (r < 16 && row < rows) {
      const float scale = __ldg(w_scale + row);
      float y[M];
#pragma unroll
      for (int m = 0; m < M; ++m) y[m] = total(r, m) * scale;
      epi.store_row(row, y);
    }
  }
}

// ---- conversion (load time): one warp per row (row scale), one lane per 16-value block ----
__device__ __forceinline__ uint32_t e2m1_code(float v) {  // v in block-scale units; nearest, ties to the smaller
  const float a = fabsf(v);
  const uint32_t i = (a > 0.25f) + (a > 0.75f) + (a > 1.25f) + (a > 1.75f) + (a > 2.5f) + (a > 3.5f) + (a > 5.f);
  return (v < 0.f && i ? 8u : 0u) | i;
}

__global__ void fp4_rows_kernel(const __nv_bfloat16* __restrict__ w, int rows, int K, uint8_t* q, uint8_t* bsc,
                                float* scale) {
  const int lane = threadIdx.x & 31;
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= rows) return;
  const __nv_bfloat16* wr = w + size_t(row) * K;
  float amax = 0.f;
  for (int k = lane; k < K; k += 32) amax = fmaxf(amax, fabsf(__bfloat162float(wr[k])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float g = amax > 0.f ? amax / (448.f * 6.f) : 1.f;
  if (lane == 0) scale[row] = g * 64.f;  // the E4M3 decoding trick reads values x 2^-6
  uint8_t* qr = q + size_t(row) * (K / 2);
  for (int b = lane; b < K / 16; b += 32) {
    float v[16], bmax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = __bfloat162float(wr[16 * b + i]), bmax = fmaxf(bmax, fabsf(v[i]));
    const __nv_fp8_storage_t sb = __nv_cvt_float_to_fp8(bmax / 6.f / g, __NV_SATFINITE, __NV_E4M3);
    bsc[size_t(row) * (K / 16) + b] = sb;
    const float s = e4m3_float(sb) * g;
#pragma unroll
    for (int i = 0; i < 16; i += 2) {
      const uint32_t c0 = s > 0.f ? e2m1_code(v[i] / s) : 0u, c1 = s > 0.f ? e2m1_code(v[i + 1] / s) : 0u;
      qr[8 * b + i / 2] = uint8_t(c0 | (c1 << 4));
    }
  }
}
