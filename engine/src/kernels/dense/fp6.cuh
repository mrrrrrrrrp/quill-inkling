// quill: FP6 (E2M3) dense weights. Included by gemv.cu inside its anonymous namespace (uses its helpers).
//
// Format (per row of K values): [K/2 bytes: low nibbles (e0 m2 m1 m0) of the codes, value 2i in the low nibble of
// byte i] [K/4 bytes: per 16 values one 32-bit word, bits 0-15 = e1 of values 0-15, bits 16-31 = their signs]; per
// 32 values one E4M3 block scale (separate array [rows][K/32]); per row one FP32 scale (x 64, see below).
// E2M3 code c = s e1 e0 m2 m1 m0 (bias 1, subnormal step 1/8, max 7.5). Decoding: the byte s<<7 | (c & 0x1f) read
// as E4M3 is exactly value(c) x 2^-6 (both have 3 mantissa bits, subnormals included), so the hardware
// E4M3 -> FP16 conversion does it; the row scale carries the 2^6. The block scale multiplies in FP32 (E2M3 x E4M3
// products have <= 8 significant bits: exact in BF16 for the tensor-core path).
// 22% fewer bytes than FP8 rows; KLD (all dense) 0.00734 vs FP8 rows ~0.0075 on the WikiText harness.

// 4-byte / 8-byte streaming weight loads (as l2::ld_stream)
__device__ __forceinline__ uint32_t ld_stream4(const void* p, std::uint64_t pol) {
  uint32_t r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.u32 %0, [%1], %2;" : "=r"(r) : "l"(p), "l"(pol));
  return r;
}
__device__ __forceinline__ uint2 ld_stream8(const void* p, std::uint64_t pol) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v2.u32 {%0,%1}, [%2], %3;"
               : "=r"(r.x), "=r"(r.y)
               : "l"(p), "l"(pol));
  return r;
}

// 8 codes -> 8 E4M3 bytes (values 0-3 in w0, 4-7 in w1, element 0 in the low byte): n = their 8 nibbles, e1 / sg
// = their e1 / sign bits in bits 0-7. The bit spreads multiply 4 bits into bit 4 (e1) or 7 (sign) of each byte
// (no carries: the partial products never meet).
__device__ __forceinline__ void fp6_e4m3x8(uint32_t n, uint32_t e1, uint32_t sg, uint32_t& w0, uint32_t& w1) {
  const uint32_t lo = n & 0x0f0f0f0fu, hi = (n >> 4) & 0x0f0f0f0fu;
  w0 = __byte_perm(lo, hi, 0x5140) | (((e1 & 0xfu) * 0x02040810u) & 0x10101010u) |
       (((sg & 0xfu) * 0x10204080u) & 0x80808080u);
  w1 = __byte_perm(lo, hi, 0x7362) | ((((e1 >> 4) & 0xfu) * 0x02040810u) & 0x10101010u) |
       ((((sg >> 4) & 0xfu) * 0x10204080u) & 0x80808080u);
}

__device__ __forceinline__ float e4m3_float(uint32_t b) {
  const __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b & 0xffu), __NV_E4M3);
  return __half2float(__half(h));
}

// 2 E4M3 (low 16 bits of u) x s -> BF16x2
__device__ __forceinline__ uint32_t e4m3x2_scaled_bf16x2(uint32_t u, float s) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(u & 0xffffu), __NV_E4M3);
  const float2 f = __half22float2(__half2(h));
  const __nv_bfloat162 b = __floats2bfloat162_rn(f.x * s, f.y * s);
  return *reinterpret_cast<const uint32_t*>(&b);
}

// ---- M = 1: 16 lanes per row, 16-value chunks (CH = K / 256 per lane, as the FP8 kernel) ----
template <int K, int BLOCK, class Epi>
__global__ void __launch_bounds__(BLOCK) gemv6_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ bsc,
                                                      const float* __restrict__ w_scale, int rows, uint32_t zero,
                                                      Epi epi) {
  constexpr int LANES = 16, CH = K / (16 * LANES);
  static_assert(K % (16 * LANES) == 0, "row must split evenly over the lane group");
  constexpr size_t ROW_BYTES = size_t(K) * 3 / 4;
  const int gl = threadIdx.x % LANES;
  const int row = (blockIdx.x * BLOCK + threadIdx.x) / LANES;
  if (row >= rows) return;
  const int in = epi.input_of(row);
  if (in < 0) return;
  const __nv_bfloat16* x = epi.x_of(in);
  const uint8_t* wr = w + size_t(row) * ROW_BYTES;
  const uint8_t* br = bsc + size_t(row) * (K / 32);
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
  uint2 lo[CH];
  uint32_t hi[CH], bs[CH];
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    const int ch = i * LANES + gl;  // 16-value chunk
    lo[i] = ld_stream8(wr + size_t(ch) * 8, wpol);
    hi[i] = ld_stream4(wr + K / 2 + size_t(ch) * 4, wpol);
    bs[i] = __ldg(br + ch / 2);
  }
  pdl_wait_fenced(zero);
  pdl_trigger();
  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    const int col = (i * LANES + gl) * 16;
    float part = 0.f;
#pragma unroll
    for (int q = 0; q < 2; ++q) {  // 8 values
      uint32_t w0, w1;
      fp6_e4m3x8(q ? lo[i].y : lo[i].x, hi[i] >> (8 * q), hi[i] >> (16 + 8 * q), w0, w1);
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

// ---- M >= 2: tensor cores, 32 columns per lane per step (SPAN 128), as gemv_mma_kernel ----
template <int K, int M, int SPLITK, int U, class Epi>
__global__ void __launch_bounds__(32 * SPLITK) gemv6_mma_kernel(const uint8_t* __restrict__ w,
                                                               const uint8_t* __restrict__ bsc,
                                                               const float* __restrict__ w_scale, int rows,
                                                               uint32_t zero, Epi epi) {
  constexpr int CPL = 32, SPAN = 4 * CPL, KS = K / SPLITK;
  static_assert(KS % (SPAN * U) == 0, "K slice must split into unrolled steps");
  constexpr int NJ = KS / SPAN;
  constexpr size_t ROW_BYTES = size_t(K) * 3 / 4;
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
    const uint8_t* sa = bsc + rowa * (K / 32);
    const uint8_t* sb = bsc + rowb * (K / 32);
    const __nv_bfloat16* xg = epi.x_of(cur) + size_t(lx ? g : 0) * K;
#pragma unroll 1
    for (int jb = 0; jb < NJ; jb += U) {
      uint4 AL[U], BL[U], X[U][4];
      uint2 AH[U], BH[U];
      uint32_t AS[U], BS[U];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int col = c0 + (jb + u) * SPAN;
        AL[u] = la ? l2::ld_stream<WEIGHT_L2>(wa + col / 2, wpol) : make_uint4(0, 0, 0, 0);
        BL[u] = lb ? l2::ld_stream<WEIGHT_L2>(wb + col / 2, wpol) : make_uint4(0, 0, 0, 0);
        AH[u] = la ? ld_stream8(wa + K / 2 + col / 4, wpol) : make_uint2(0, 0);
        BH[u] = lb ? ld_stream8(wb + K / 2 + col / 4, wpol) : make_uint2(0, 0);
        AS[u] = la ? __ldg(sa + col / 32) : 0u;
        BS[u] = lb ? __ldg(sb + col / 32) : 0u;
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
      for (int u = 0; u < U; ++u) dep ^= AL[u].x ^ AL[u].w ^ BL[u].x ^ BL[u].w ^ AH[u].x ^ BH[u].y ^ X[u][0].x ^ X[u][3].w;
      dep &= zero;
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const float fa = e4m3_float(AS[u]), fb = e4m3_float(BS[u]);
        const uint32_t an[4] = {AL[u].x, AL[u].y, AL[u].z, AL[u].w}, bn[4] = {BL[u].x, BL[u].y, BL[u].z, BL[u].w};
        const uint32_t xw[16] = {X[u][0].x, X[u][0].y, X[u][0].z, X[u][0].w, X[u][1].x, X[u][1].y, X[u][1].z, X[u][1].w,
                                 X[u][2].x, X[u][2].y, X[u][2].z, X[u][2].w, X[u][3].x, X[u][3].y, X[u][3].z, X[u][3].w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {  // 8 values: q-th byte of the nibble words, hi word q / 2
          const uint32_t ah = q < 2 ? AH[u].x : AH[u].y, bh = q < 2 ? BH[u].x : BH[u].y;
          const int sh = 8 * (q & 1);
          uint32_t a0, a1, b0, b1;
          fp6_e4m3x8(an[q], ah >> sh, ah >> (16 + sh), a0, a1);
          fp6_e4m3x8(bn[q], bh >> sh, bh >> (16 + sh), b0, b1);
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

// ---- conversion (load time) and dequantization (prefill) ----
__device__ __forceinline__ uint32_t e2m3_encode(float v) {  // v in block-scale units
  const float a = fminf(fabsf(v), 7.5f);
  uint32_t e, m;
  if (a < 1.f) {
    m = uint32_t(rintf(a * 8.f));
    e = 0;
    if (m == 8) e = 1, m = 0;
  } else if (a < 2.f) {
    m = uint32_t(rintf((a - 1.f) * 8.f));
    e = 1;
    if (m == 8) e = 2, m = 0;
  } else if (a < 4.f) {
    m = uint32_t(rintf((a - 2.f) * 4.f));
    e = 2;
    if (m == 8) e = 3, m = 0;
  } else {
    m = min(uint32_t(rintf((a - 4.f) * 2.f)), 7u);
    e = 3;
  }
  return (v < 0.f && (e | m) ? 0x20u : 0u) | (e << 3) | m;
}

// one warp per row: row scale, then one lane per 32-value block
__global__ void fp6_rows_kernel(const __nv_bfloat16* __restrict__ w, int rows, int K, uint8_t* q, uint8_t* bsc,
                                float* scale) {
  const int lane = threadIdx.x & 31;
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= rows) return;
  const __nv_bfloat16* wr = w + size_t(row) * K;
  float amax = 0.f;
  for (int k = lane; k < K; k += 32) amax = fmaxf(amax, fabsf(__bfloat162float(wr[k])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float g = amax > 0.f ? amax / (448.f * 7.5f) : 1.f;
  if (lane == 0) scale[row] = g * 64.f;  // the E4M3 decoding trick reads values x 2^-6
  uint8_t* lo = q + size_t(row) * (K * 3 / 4);
  uint32_t* hw = reinterpret_cast<uint32_t*>(lo + K / 2);
  for (int b = lane; b < K / 32; b += 32) {
    float bmax = 0.f;
    for (int i = 0; i < 32; ++i) bmax = fmaxf(bmax, fabsf(__bfloat162float(wr[32 * b + i])));
    const __nv_fp8_storage_t sb = __nv_cvt_float_to_fp8(bmax / 7.5f / g, __NV_SATFINITE, __NV_E4M3);
    bsc[size_t(row) * (K / 32) + b] = sb;
    const float s = e4m3_float(sb) * g;
    uint32_t e1w[2] = {0, 0}, sw[2] = {0, 0};
    for (int i = 0; i < 32; i += 2) {
      const uint32_t c0 = s > 0.f ? e2m3_encode(__bfloat162float(wr[32 * b + i]) / s) : 0u;
      const uint32_t c1 = s > 0.f ? e2m3_encode(__bfloat162float(wr[32 * b + i + 1]) / s) : 0u;
      lo[16 * b + i / 2] = uint8_t((c0 & 0xf) | ((c1 & 0xf) << 4));
      const int h = i / 16, k = i % 16;
      e1w[h] |= ((c0 >> 4) & 1u) << k | ((c1 >> 4) & 1u) << (k + 1);
      sw[h] |= (c0 >> 5) << k | (c1 >> 5) << (k + 1);
    }
    hw[2 * b] = e1w[0] | (sw[0] << 16);
    hw[2 * b + 1] = e1w[1] | (sw[1] << 16);
  }
}

// bf16(value) of 16 values per thread
__global__ void dequant_rows6_kernel(const uint8_t* __restrict__ q, const uint8_t* __restrict__ bsc,
                                     const float* __restrict__ scale, int K, long long total16,
                                     __nv_bfloat16* __restrict__ out) {
  const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= total16) return;
  const long long e = i * 16;
  const long long row = e / K;
  const int col = int(e % K);
  const uint8_t* lo = q + row * (K * 3 / 4);
  const uint2 n = *reinterpret_cast<const uint2*>(lo + col / 2);
  const uint32_t h = *reinterpret_cast<const uint32_t*>(lo + K / 2 + col / 4);
  const float s = e4m3_float(bsc[row * (K / 32) + col / 32]) * scale[row];
  __align__(16) __nv_bfloat16 v[16];
#pragma unroll
  for (int q8 = 0; q8 < 2; ++q8) {
    uint32_t w0, w1;
    fp6_e4m3x8(q8 ? n.y : n.x, h >> (8 * q8), h >> (16 + 8 * q8), w0, w1);
    float f[8];
    unpack_e4m3x4(w0, f);
    unpack_e4m3x4(w1, f + 4);
#pragma unroll
    for (int j = 0; j < 8; ++j) v[8 * q8 + j] = __float2bfloat16_rn(f[j] * s);
  }
  uint4* dst = reinterpret_cast<uint4*>(out + e);
  dst[0] = reinterpret_cast<const uint4*>(v)[0];
  dst[1] = reinterpret_cast<const uint4*>(v)[1];
}
