// quill: CPU tier of the hybrid MoE decode (cpu_tier.h). AVX2 + F16C + FMA kernels for NVFP4 and MXFP8
// expert rows; values decoded as model.decode_experts (code x block scale x factor) in FP32.
#include "model/cpu_tier.h"
#include "kernels/moe/moe.h"

#include <spdlog/spdlog.h>

#include <immintrin.h>

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

namespace ingine::model {
namespace {

struct Luts {
  float e4m3[256];
  Luts() {
    for (int b = 0; b < 256; ++b) {
      const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
      float v;
      if (e == 0) v = std::ldexp(float(m) / 8.f, -6);
      else v = std::ldexp(1.f + float(m) / 8.f, e - 7);
      if (e == 15 && m == 7) v = 0.f;  // NaN encoding: never produced in weights
      e4m3[b] = s ? -v : v;
    }
  }
};
const Luts& luts() {
  static const Luts l;
  return l;
}

inline float bf16r(float x) {
  std::uint32_t u;
  std::memcpy(&u, &x, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return x;
  u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
  float r;
  std::memcpy(&r, &u, 4);
  return r;
}
inline float hsum(__m256 v) {
  __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  a = _mm_add_ps(a, _mm_movehl_ps(a, a));
  a = _mm_add_ss(a, _mm_movehdup_ps(a));
  return _mm_cvtss_f32(a);
}

// dot products of one expert row (K values) with T input vectors xs[t] (fp32, K), into out[t]. T is a template
// parameter so the accumulators stay in registers; T = 1 runs four independent FMA chains.
inline void nvfp4_decode16(const std::uint8_t* codes, std::uint8_t scale, const float* e4, __m256& w0, __m256& w1) {
  const __m128i lut = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);  // E2M1 x 2
  const __m128i m4 = _mm_set1_epi8(0x0f);
  const __m128i c = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes));
  const __m128i lo = _mm_and_si128(c, m4), hi = _mm_and_si128(_mm_srli_epi16(c, 4), m4);
  const __m128i v8 = _mm_shuffle_epi8(lut, _mm_unpacklo_epi8(lo, hi));
  const __m256 sv = _mm256_set1_ps(e4[scale] * 0.5f);
  w0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v8)), sv);
  w1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v8, 8))), sv);
}

template <int K, int T>
inline void nvfp4_dot_t(const std::uint8_t* codes, const std::uint8_t* scales, const float* const* xs, float* out) {
  const float* e4 = luts().e4m3;
  if constexpr (T == 1) {
    const float* x = xs[0];
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (int b = 0; b < K / 16; b += 2) {
      __m256 w0, w1, w2, w3;
      nvfp4_decode16(codes + 8 * b, scales[b], e4, w0, w1);
      nvfp4_decode16(codes + 8 * b + 8, scales[b + 1], e4, w2, w3);
      a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + 16 * b), a0);
      a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x + 16 * b + 8), a1);
      a2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(x + 16 * b + 16), a2);
      a3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(x + 16 * b + 24), a3);
    }
    out[0] = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
  } else {
    __m256 acc[T];
    for (int t = 0; t < T; ++t) acc[t] = _mm256_setzero_ps();
    for (int b = 0; b < K / 16; ++b) {
      __m256 w0, w1;
      nvfp4_decode16(codes + 8 * b, scales[b], e4, w0, w1);
      for (int t = 0; t < T; ++t) {
        acc[t] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xs[t] + 16 * b), acc[t]);
        acc[t] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xs[t] + 16 * b + 8), acc[t]);
      }
    }
    for (int t = 0; t < T; ++t) out[t] = hsum(acc[t]);
  }
}

inline __m256 mxfp8_decode8(const std::uint8_t* codes, __m256 s) {
  const __m128i m80 = _mm_set1_epi16(0x80), m7f = _mm_set1_epi16(0x7f);
  const __m128i u = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes)));
  const __m128i h = _mm_or_si128(_mm_slli_epi16(_mm_and_si128(u, m80), 8), _mm_slli_epi16(_mm_and_si128(u, m7f), 7));
  return _mm256_mul_ps(_mm256_cvtph_ps(h), s);
}
inline __m256 e8m0x256(std::uint8_t se) {  // E8M0 scale x 2^8 (the f16 trick yields e4m3 / 256)
  float sc;
  const std::uint32_t bits = se ? std::uint32_t(se) << 23 : 0x00400000u;
  std::memcpy(&sc, &bits, 4);
  return _mm256_set1_ps(sc * 256.f);
}

template <int K, int T>
inline void mxfp8_dot_t(const std::uint8_t* codes, const std::uint8_t* scales, const float* const* xs, float* out) {
  if constexpr (T == 1) {
    const float* x = xs[0];
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    for (int b = 0; b < K / 32; ++b) {
      const __m256 s = e8m0x256(scales[b]);
      const std::uint8_t* c = codes + 32 * b;
      a0 = _mm256_fmadd_ps(mxfp8_decode8(c, s), _mm256_loadu_ps(x + 32 * b), a0);
      a1 = _mm256_fmadd_ps(mxfp8_decode8(c + 8, s), _mm256_loadu_ps(x + 32 * b + 8), a1);
      a2 = _mm256_fmadd_ps(mxfp8_decode8(c + 16, s), _mm256_loadu_ps(x + 32 * b + 16), a2);
      a3 = _mm256_fmadd_ps(mxfp8_decode8(c + 24, s), _mm256_loadu_ps(x + 32 * b + 24), a3);
    }
    out[0] = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
  } else {
    __m256 acc[T];
    for (int t = 0; t < T; ++t) acc[t] = _mm256_setzero_ps();
    for (int b = 0; b < K / 32; ++b) {
      const __m256 s = e8m0x256(scales[b]);
      for (int q = 0; q < 4; ++q) {
        const __m256 w = mxfp8_decode8(codes + 32 * b + 8 * q, s);
        for (int t = 0; t < T; ++t) acc[t] = _mm256_fmadd_ps(w, _mm256_loadu_ps(xs[t] + 32 * b + 8 * q), acc[t]);
      }
    }
    for (int t = 0; t < T; ++t) out[t] = hsum(acc[t]);
  }
}

template <int K>
void dot_nvfp4(const std::uint8_t* codes, const std::uint8_t* scales, float factor, const float* const* xs, int T,
               float* out) {
  switch (T) {
    case 1: nvfp4_dot_t<K, 1>(codes, scales, xs, out); break;
    case 2: nvfp4_dot_t<K, 2>(codes, scales, xs, out); break;
    case 3: nvfp4_dot_t<K, 3>(codes, scales, xs, out); break;
    case 4: nvfp4_dot_t<K, 4>(codes, scales, xs, out); break;
    case 5: nvfp4_dot_t<K, 5>(codes, scales, xs, out); break;
    case 6: nvfp4_dot_t<K, 6>(codes, scales, xs, out); break;
    case 7: nvfp4_dot_t<K, 7>(codes, scales, xs, out); break;
    default: nvfp4_dot_t<K, 8>(codes, scales, xs, out); break;
  }
  for (int t = 0; t < T; ++t) out[t] *= factor;
}

template <int K>
void dot_mxfp8(const std::uint8_t* codes, const std::uint8_t* scales, float /*factor*/, const float* const* xs, int T,
               float* out) {
  switch (T) {
    case 1: mxfp8_dot_t<K, 1>(codes, scales, xs, out); break;
    case 2: mxfp8_dot_t<K, 2>(codes, scales, xs, out); break;
    case 3: mxfp8_dot_t<K, 3>(codes, scales, xs, out); break;
    case 4: mxfp8_dot_t<K, 4>(codes, scales, xs, out); break;
    case 5: mxfp8_dot_t<K, 5>(codes, scales, xs, out); break;
    case 6: mxfp8_dot_t<K, 6>(codes, scales, xs, out); break;
    case 7: mxfp8_dot_t<K, 7>(codes, scales, xs, out); break;
    default: mxfp8_dot_t<K, 8>(codes, scales, xs, out); break;
  }
}

// ---- quill: int8 path (default; QUILL_CPU_FP32=1 restores the fp32 one). Inputs are quantized per 32 values
// (int8 + fp32 scale); E2M1 codes x 2 become int8 by one pshufb per 32 weights; |w| (u8) x sign(x, w) (s8) through
// maddubs / madd gives int32 sums of 4 products (lanes 0-3: the first 16 weights, 4-7: the next 16), scaled per
// 16-weight block. 1.8x the fp32 path per core (10 vs 5.5 GB/s), 69 GB/s on 7 cores from L3.
template <int K>
struct XQ {
  alignas(32) std::int8_t q[K];
  float s[K / 32];
};
inline void quant32(const float* x, std::int8_t* q, float* s) {
  const __m256 sign = _mm256_set1_ps(-0.f);
  const __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8), v2 = _mm256_loadu_ps(x + 16), v3 = _mm256_loadu_ps(x + 24);
  const __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)),
                                 _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
  __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
  h = _mm_max_ps(h, _mm_movehl_ps(h, h));
  h = _mm_max_ss(h, _mm_movehdup_ps(h));
  const float am = _mm_cvtss_f32(h);
  *s = am / 127.f;
  const __m256 inv = _mm256_set1_ps(am > 0.f ? 127.f / am : 0.f);
  const __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, inv)), i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, inv));
  const __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, inv)), i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, inv));
  __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
  p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(q), p);
}
template <int K, int T>
inline void nvfp4_dot_i8_t(const std::uint8_t* codes, const std::uint8_t* scales, const XQ<K>* const* xq, float* out) {
  const float* e4 = luts().e4m3;
  const __m256i lut = _mm256_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
                                       0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
  const __m128i m4 = _mm_set1_epi8(0x0f);
  const __m256i ones = _mm256_set1_epi16(1);
  __m256 acc[T];
  for (int t = 0; t < T; ++t) acc[t] = _mm256_setzero_ps();
  for (int b = 0; b < K / 32; ++b) {
    const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + 16 * b));
    const __m128i lo = _mm_and_si128(c, m4), hi = _mm_and_si128(_mm_srli_epi16(c, 4), m4);
    const __m256i w = _mm256_shuffle_epi8(lut, _mm256_set_m128i(_mm_unpackhi_epi8(lo, hi), _mm_unpacklo_epi8(lo, hi)));
    const __m256i aw = _mm256_abs_epi8(w);
    const __m256 ws = _mm256_set_m128(_mm_set1_ps(e4[scales[2 * b + 1]] * 0.5f), _mm_set1_ps(e4[scales[2 * b]] * 0.5f));
    for (int t = 0; t < T; ++t) {
      const __m256i x = _mm256_load_si256(reinterpret_cast<const __m256i*>(xq[t]->q + 32 * b));
      const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(x, w)), ones);
      acc[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_mul_ps(ws, _mm256_set1_ps(xq[t]->s[b])), acc[t]);
    }
  }
  for (int t = 0; t < T; ++t) out[t] = hsum(acc[t]);
}
template <int K>
void dot_nvfp4_i8(const std::uint8_t* codes, const std::uint8_t* scales, float factor, const XQ<K>* const* xq, int T,
                  float* out) {
  for (int t0 = 0; t0 < T; t0 += 4) {  // more than 4 rows spill the accumulators: passes of up to 4
    const int n = std::min(4, T - t0);
    switch (n) {
      case 1: nvfp4_dot_i8_t<K, 1>(codes, scales, xq + t0, out + t0); break;
      case 2: nvfp4_dot_i8_t<K, 2>(codes, scales, xq + t0, out + t0); break;
      case 3: nvfp4_dot_i8_t<K, 3>(codes, scales, xq + t0, out + t0); break;
      default: nvfp4_dot_i8_t<K, 4>(codes, scales, xq + t0, out + t0); break;
    }
  }
  for (int t = 0; t < T; ++t) out[t] *= factor;
}

inline float act_fn(float g, float u) {  // bf16 inputs; softcap(g,4) sigmoid(g) softcap(u,25)
  const float a = std::tanh(g / 4.f) * 4.f * (1.f / (1.f + std::exp(-g))) * (std::tanh(u / 25.f) * 25.f);
  return bf16r(a);
}

struct Unique {
  int expert;
  int n;                 // entries using it
  int entry[8];          // entry indices
  int token[8];          // token of each entry
};

CpuTier* g_tier = nullptr;
inline std::uint64_t ns_since(std::chrono::steady_clock::time_point t0) {
  return std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}

}  // namespace

CpuTierStats cpu_tier_stats() {
  CpuTierStats s;
  if (g_tier) s.calls = g_tier->calls_.load(), s.entries = g_tier->entries_.load(), s.busy_s = g_tier->busy_ns_.load() * 1e-9,
              s.uniques = g_tier->uniques_.load();
  return s;
}
void cpu_tier_reset_stats() {
  if (g_tier) g_tier->calls_ = 0, g_tier->entries_ = 0, g_tier->busy_ns_ = 0, g_tier->uniques_ = 0;
}

CpuTier::CpuTier(const ExpertStore& store, int threads)
    : store_(store), nthreads_(std::max(1, threads)), exch_(sizeof(CpuExchange)) {
  g_tier = this;
  host_ = static_cast<CpuExchange*>(exch_.data());
  act_.resize(std::size_t(CPU_MAX_ENTRIES) * INTER);
  xf_.resize(std::size_t(8) * HIDDEN);
  xq_.resize((8 * sizeof(XQ<HIDDEN>) + 31) / 32);
  actq_.resize((std::size_t(CPU_MAX_ENTRIES) * sizeof(XQ<INTER>) + 31) / 32);
  int8_ = std::getenv("QUILL_CPU_FP32") == nullptr;
  static_split_ = std::getenv("QUILL_CPU_STATIC") != nullptr;
  if (const char* tp = std::getenv("QUILL_TIER_TRACE")) trace_ = std::fopen(tp, "w");
  if (std::getenv("QUILL_TIER_PHASES")) phase_ns_ = new std::uint64_t[5]{};
  if (const char* pm = std::getenv("QUILL_PREFETCH")) pred_mode_ = std::atoi(pm);
  if (const char* pm = std::getenv("QUILL_PF_MAX")) pf_max_ = std::atoi(pm);
  (void)luts();
  if (std::getenv("QUILL_JOIN_DBG")) kernels::moe::join_debug(true, false);
  for (int i = 0; i < nthreads_; ++i) threads_.emplace_back([this, i] { worker(i); });
  spdlog::info("cpu tier: {} worker threads, {} dot products", nthreads_, int8_ ? "int8" : "fp32");
}

CpuTier::~CpuTier() {
  if (phase_ns_) {
    const std::uint64_t c = std::max<std::uint64_t>(calls_.load(), 1);
    std::fprintf(stderr, "[cpu tier phases] per call (cumulative us): in %.1f, inq %.1f, w13 %.1f, actq %.1f, w2 %.1f | calls %llu, unique/call %.2f\n",
                 phase_ns_[0] / 1e3 / c, phase_ns_[1] / 1e3 / c, phase_ns_[2] / 1e3 / c, phase_ns_[3] / 1e3 / c,
                 phase_ns_[4] / 1e3 / c, (unsigned long long)c, double(uniques_.load()) / c);
    delete[] phase_ns_;
  }
  if (std::getenv("QUILL_JOIN_DBG")) kernels::moe::join_debug(false, true);
  if (pred_mode_ && pf_runs_.load())
    std::fprintf(stderr, "[cpu tier prefetch] mode %d: %llu predictions, %.2f hinted experts each, %.1f MB prefetched each, %.1f us per worker each (%.1f GB/s)\n",
                 pred_mode_, (unsigned long long)pf_runs_.load(), double(pf_hints_.load()) / pf_runs_.load(),
                 double(pf_bytes_.load()) / 1e6 / pf_runs_.load(), double(pf_ns_.load()) / 1e3 / pf_runs_.load() / nthreads_,
                 double(pf_bytes_.load()) / std::max(1.0, double(pf_ns_.load()) / nthreads_));
  if (g_tier == this) g_tier = nullptr;
  stop_.store(true);
  for (auto& t : threads_) t.join();
}

void CpuTier::set_routers(std::vector<std::uint16_t> w, std::vector<float> bias) {
  rw_ = std::move(w), rb_ = std::move(bias);
  if (pred_mode_) spdlog::info("cpu tier: pre-gating {} (next visit's router on the call's inputs)",
                               pred_mode_ >= 2 ? "prefetch" : "hints traced");
}

namespace {
inline float pg_hsum8(__m256 v) {
  __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  a = _mm_add_ps(a, _mm_movehl_ps(a, a));
  a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 1));
  return _mm_cvtss_f32(a);
}
}  // namespace

// quill: after the call of `visit` (ack published), every worker: router keys (sigmoid + bias) of visit + 1 for
// its share of the 128 experts over the call's M inputs; barrier; top-8 per token (each worker, same result);
// the non-resident ones are prefetched (4 KB chunks round robin over the workers) until the next request.
void CpuTier::predict_next(int id, int visit, int M, std::uint32_t seq) {
  const int nl = visit_at(visit + 1).layer;
  const float* bias = rb_.data() + std::size_t(nl) * EXPERTS;
  const int r0 = EXPERTS * id / nthreads_, r1 = EXPERTS * (id + 1) / nthreads_;
  for (int e = r0; e < r1; ++e) {
    const std::uint16_t* w = rw_.data() + (std::size_t(nl) * EXPERTS + e) * HIDDEN;
    __m256 acc[8];
    for (int t = 0; t < 8; ++t) acc[t] = _mm256_setzero_ps();
    for (int k = 0; k < HIDDEN; k += 8) {
      const __m256 wv = _mm256_castsi256_ps(
          _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + k))), 16));
      for (int t = 0; t < M; ++t) acc[t] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(&xf_[std::size_t(t) * HIDDEN + k]), acc[t]);
    }
    for (int t = 0; t < M; ++t) pkey_[t][e] = 1.f / (1.f + std::exp(-pg_hsum8(acc[t]))) + bias[e];
  }
  barrier();
  float score[EXPERTS] = {};  // tokens selecting the expert (+ its best key / 16 as the tie break)
  for (int t = 0; t < M; ++t) {
    float key[EXPERTS];
    std::memcpy(key, pkey_[t], sizeof(key));
    for (int k = 0; k < TOP_K; ++k) {
      int b = 0;
      for (int e = 1; e < EXPERTS; ++e) b = key[e] > key[b] ? e : b;
      score[b] += 1.f + std::max(0.f, pkey_[t][b]) / 16.f, key[b] = -1e30f;
    }
  }
  int hint[EXPERTS], nh = 0;
  for (int e = 0; e < EXPERTS; ++e)
    if (score[e] > 0.f && store_.host_slot(nl, e) < 0) hint[nh++] = e;
  std::stable_sort(hint, hint + nh, [&](int a, int b) { return score[a] > score[b]; });
  nh = std::min(nh, pf_max_);
  if (id == 0) {
    pf_runs_.fetch_add(1, std::memory_order_relaxed), pf_hints_.fetch_add(std::uint64_t(nh), std::memory_order_relaxed);
    if (trace_) {
      std::fprintf(trace_, "P %d %d", visit + 1, nh);
      for (int i = 0; i < nh; ++i) std::fprintf(trace_, " %d", hint[i]);
      std::fputc(10, trace_);
    }
  }
  if (pred_mode_ < 2 || nh == 0) return;
  const HostLayerExperts& h = store_.host(nl);
  const ExpertBytes eb = expert_bytes(h.fmt);
  struct Seg {
    const char* p;
    std::size_t n;
  } seg[4 * EXPERTS];
  int ns = 0;
  for (int i = 0; i < nh; ++i)  // whole experts in rank order
    seg[ns++] = {reinterpret_cast<const char*>(h.expert_w13(hint[i])), eb.w13},
    seg[ns++] = {reinterpret_cast<const char*>(h.expert_w13s(hint[i])), eb.w13s},
    seg[ns++] = {reinterpret_cast<const char*>(h.expert_w2(hint[i])), eb.w2},
    seg[ns++] = {reinterpret_cast<const char*>(h.expert_w2s(hint[i])), eb.w2s};
  constexpr std::size_t CH = 4096;
  volatile std::uint32_t* seqp = &host_->req.seq;
  std::size_t chunk = 0, bytes = 0;
  const auto pt0 = std::chrono::steady_clock::now();
  std::uint64_t sink = 0;
  for (int s = 0; s < ns; ++s)
    for (std::size_t off = 0; off < seg[s].n; off += CH, ++chunk) {
      if (int(chunk % std::size_t(nthreads_)) != id) continue;
      if (*seqp != seq) goto out;  // the next request is here: stop
      const char* p = seg[s].p + off;
      const std::size_t n = std::min(CH, seg[s].n - off);
      if (pred_mode_ >= 3) {  // real loads: one 8-byte read per line, summed (cannot be dropped like a prefetch)
        std::uint64_t a0 = 0, a1 = 0;
        for (std::size_t o = 0; o < n; o += 128)
          a0 += *reinterpret_cast<const volatile std::uint64_t*>(p + o),
          a1 += *reinterpret_cast<const volatile std::uint64_t*>(p + o + 64);
        sink += a0 ^ a1;
      } else {
        for (std::size_t o = 0; o < n; o += 64) _mm_prefetch(p + o, _MM_HINT_T1);
      }
      bytes += n;
    }
out:
  pf_bytes_.fetch_add(bytes, std::memory_order_relaxed);
  pf_ns_.fetch_add(std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - pt0).count()),
                   std::memory_order_relaxed);
  if (sink == 0x123456789abcdefull) std::fputc(0, stderr);
}

void CpuTier::activate() { active_.fetch_add(1); }
void CpuTier::deactivate() { active_.fetch_sub(1); }

void CpuTier::barrier() {
  const std::uint32_t gen = phase_gen_.load(std::memory_order_acquire);
  if (phase_arrive_.fetch_add(1, std::memory_order_acq_rel) == std::uint32_t(nthreads_ - 1)) {
    phase_arrive_.store(0, std::memory_order_relaxed);
    phase_gen_.store(gen + 1, std::memory_order_release);
  } else {
    while (phase_gen_.load(std::memory_order_acquire) == gen) _mm_pause();
  }
}

void CpuTier::worker(int id) {
#ifdef _WIN32
  // workers 0-6 on cores 1-7 (logical 2, 4, .., 14), workers 7-13 on their SMT siblings (3, 5, .., 15); core 0
  // stays with the thread that launches the GPU work
  const int logical = id < 7 ? 2 * (id + 1) : 2 * (id - 7 + 1) + 1;
  SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << (logical % 16));
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif
  // A request is pending while req.seq != resp.ack (the ack is published after every worker finished it), so a
  // worker that slept through the doorbell still takes part. Inactive workers poll every 200 us instead of
  // spinning: a decode step run outside an active period (tests, warm-up) still completes.
  volatile std::uint32_t* seqp = &host_->req.seq;
  volatile std::uint32_t* ackp = &host_->resp.ack;
  std::uint32_t done = *ackp;  // last request this worker finished
  while (!stop_.load(std::memory_order_relaxed)) {
    const std::uint32_t s = *seqp;
    if (s == *ackp || s == done) {
      if (active_.load(std::memory_order_relaxed) == 0) std::this_thread::sleep_for(std::chrono::microseconds(200));
      else _mm_pause();
      continue;
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    run_request(id, host_->req);
    done = s;
  }
}

void CpuTier::run_request(int id, const CpuRequest& rq) {
  const auto t0 = std::chrono::steady_clock::now();
  const int layer = rq.layer, M = rq.M, n = rq.n, visit = rq.visit;
  const std::uint32_t seq = rq.seq;
  const HostLayerExperts& h = store_.host(layer);
  const bool nv = h.fmt == ExpertFormat::NVFP4;
  const ExpertBytes eb = expert_bytes(h.fmt);
  // group the entries by expert (same order on every thread)
  Unique uq[CPU_MAX_ENTRIES];
  int U = 0;
  for (int i = 0; i < n; ++i) {
    const int e = rq.expert[i];
    int u = 0;
    while (u < U && uq[u].expert != e) ++u;
    if (u == U) uq[U++] = Unique{e, 0, {}, {}};
    uq[u].entry[uq[u].n] = i;
    uq[u].token[uq[u].n] = rq.slot[i] / 8;
    ++uq[u].n;
  }
  if (id == 0) next1_.store(0, std::memory_order_relaxed), next2_.store(0, std::memory_order_relaxed);
  // phase 0: inputs to fp32 (and, on the int8 path, to int8 per 32 values)
  const bool i8 = int8_ && nv;
  for (int i = id; i < M * HIDDEN / 8; i += nthreads_) {
    const int m = i / (HIDDEN / 8), c = (i % (HIDDEN / 8)) * 8;
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&rq.x[m][c]));
    const __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(b), 16);
    _mm256_storeu_ps(&xf_[std::size_t(m) * HIDDEN + c], _mm256_castsi256_ps(w));
  }
  barrier();
  if (id == 0 && phase_ns_) phase_ns_[0] += ns_since(t0);
  if (i8) {
    auto* xq = reinterpret_cast<XQ<HIDDEN>*>(xq_.data());
    for (int i = id; i < M * HIDDEN / 32; i += nthreads_) {
      const int m = i / (HIDDEN / 32), blk = i % (HIDDEN / 32);
      quant32(&xf_[std::size_t(m) * HIDDEN + 32 * blk], xq[m].q + 32 * blk, &xq[m].s[blk]);
    }
    barrier();
    if (id == 0 && phase_ns_) phase_ns_[1] += ns_since(t0);
  }
  // phase 1: W13 pairs (gate row 2i, up row 2i+1) over the unique experts
  {
    const int total = U * INTER;
    const std::size_t crow = nv ? HIDDEN / 2 : HIDDEN, srow = nv ? HIDDEN / 16 : HIDDEN / 32;
    constexpr int CHUNK = 16;  // quill: rows per grab (dynamic split: no thread waits at the barrier)
    const bool stat = static_split_;
    int a = stat ? int(std::int64_t(total) * id / nthreads_) : 0, b = stat ? int(std::int64_t(total) * (id + 1) / nthreads_) : 0;
    for (;;) {
      if (!stat) {
        a = next1_.fetch_add(CHUNK, std::memory_order_relaxed);
        if (a >= total) break;
        b = std::min(total, a + CHUNK);
      }
    for (int it = a; it < b; ++it) {
      const Unique& q = uq[it / INTER];
      const int i = it % INTER;
      const auto* codes = reinterpret_cast<const std::uint8_t*>(h.expert_w13(q.expert));
      const auto* scales = reinterpret_cast<const std::uint8_t*>(h.expert_w13s(q.expert));
      const float* xs[8];
      for (int t = 0; t < q.n; ++t) xs[t] = &xf_[std::size_t(q.token[t]) * HIDDEN];
      float g[8], u[8];
      const float f = h.f13[q.expert];
      if (i8) {
        const auto* xqa = reinterpret_cast<const XQ<HIDDEN>*>(xq_.data());
        const XQ<HIDDEN>* xq[8];
        for (int t = 0; t < q.n; ++t) xq[t] = &xqa[q.token[t]];
        dot_nvfp4_i8<HIDDEN>(codes + (2 * i) * crow, scales + (2 * i) * srow, f, xq, q.n, g);
        dot_nvfp4_i8<HIDDEN>(codes + (2 * i + 1) * crow, scales + (2 * i + 1) * srow, f, xq, q.n, u);
      } else if (nv) {
        dot_nvfp4<HIDDEN>(codes + (2 * i) * crow, scales + (2 * i) * srow, f, xs, q.n, g);
        dot_nvfp4<HIDDEN>(codes + (2 * i + 1) * crow, scales + (2 * i + 1) * srow, f, xs, q.n, u);
      } else {
        dot_mxfp8<HIDDEN>(codes + (2 * i) * crow, scales + (2 * i) * srow, f, xs, q.n, g);
        dot_mxfp8<HIDDEN>(codes + (2 * i + 1) * crow, scales + (2 * i + 1) * srow, f, xs, q.n, u);
      }
      for (int t = 0; t < q.n; ++t) act_[std::size_t(q.entry[t]) * INTER + i] = act_fn(bf16r(g[t]), bf16r(u[t]));
    }
      if (stat) break;
    }
  }
  barrier();
  if (id == 0 && phase_ns_) phase_ns_[2] += ns_since(t0);
  if (i8) {  // phase 1.5: the activations to int8 per 32 values
    auto* aq = reinterpret_cast<XQ<INTER>*>(actq_.data());
    for (int i = id; i < n * INTER / 32; i += nthreads_) {
      const int e = i / (INTER / 32), blk = i % (INTER / 32);
      quant32(&act_[std::size_t(e) * INTER + 32 * blk], aq[e].q + 32 * blk, &aq[e].s[blk]);
    }
    barrier();
    if (id == 0 && phase_ns_) phase_ns_[3] += ns_since(t0);
  }
  // phase 2: W2 rows
  {
    const int total = U * HIDDEN;
    const std::size_t crow = nv ? INTER / 2 : INTER, srow = nv ? INTER / 16 : INTER / 32;
    CpuResponse& rs = host_->resp;
    constexpr int CHUNK = 16;  // quill: rows per grab (dynamic split: no thread waits at the barrier)
    const bool stat = static_split_;
    int a = stat ? int(std::int64_t(total) * id / nthreads_) : 0, b = stat ? int(std::int64_t(total) * (id + 1) / nthreads_) : 0;
    for (;;) {
      if (!stat) {
        a = next2_.fetch_add(CHUNK, std::memory_order_relaxed);
        if (a >= total) break;
        b = std::min(total, a + CHUNK);
      }
    for (int it = a; it < b; ++it) {
      const Unique& q = uq[it / HIDDEN];
      const int r = it % HIDDEN;
      const auto* codes = reinterpret_cast<const std::uint8_t*>(h.expert_w2(q.expert));
      const auto* scales = reinterpret_cast<const std::uint8_t*>(h.expert_w2s(q.expert));
      const float* xs[8];
      for (int t = 0; t < q.n; ++t) xs[t] = &act_[std::size_t(q.entry[t]) * INTER];
      float y[8];
      if (i8) {
        const auto* aqa = reinterpret_cast<const XQ<INTER>*>(actq_.data());
        const XQ<INTER>* aq[8];
        for (int t = 0; t < q.n; ++t) aq[t] = &aqa[q.entry[t]];
        dot_nvfp4_i8<INTER>(codes + r * crow, scales + r * srow, h.f2[q.expert], aq, q.n, y);
      } else if (nv) dot_nvfp4<INTER>(codes + r * crow, scales + r * srow, h.f2[q.expert], xs, q.n, y);
      else dot_mxfp8<INTER>(codes + r * crow, scales + r * srow, 1.f, xs, q.n, y);
      for (int t = 0; t < q.n; ++t) rs.y[q.entry[t]][r] = bf16r(y[t]);
    }
      if (stat) break;
    }
  }
  (void)eb;
  barrier();
  if (id == 0 && phase_ns_) phase_ns_[4] += ns_since(t0);
  if (id == 0 && trace_) {  // quill diagnostics (QUILL_TIER_TRACE=file): layer, rows, then the unique experts
    const auto us = [](std::chrono::steady_clock::time_point t) {
      return (long long)std::chrono::duration_cast<std::chrono::microseconds>(t.time_since_epoch()).count();
    };
    std::fprintf(trace_, "C %d %lld %lld %d %d", visit, us(t0), us(std::chrono::steady_clock::now()), layer, M);
    for (int u = 0; u < U; ++u) std::fprintf(trace_, " %d", uq[u].expert);
    std::fputc(10, trace_);  // newline
  }
  if (id == 0) {
    std::atomic_thread_fence(std::memory_order_release);
    *reinterpret_cast<volatile std::uint32_t*>(&host_->resp.ack) = rq.seq;
    calls_.fetch_add(1, std::memory_order_relaxed);
    entries_.fetch_add(std::uint64_t(n), std::memory_order_relaxed);
    uniques_.fetch_add(std::uint64_t(U), std::memory_order_relaxed);
    busy_ns_.fetch_add(std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()),
                       std::memory_order_relaxed);
  }
  if (pred_mode_ && !rw_.empty() && visit >= 0 && visit + 1 < num_visits()) predict_next(id, visit, M, seq);
}

void CpuTier::compute_expert(int layer, int expert, const std::uint16_t* x, float* y) const {
  const HostLayerExperts& h = store_.host(layer);
  const bool nv = h.fmt == ExpertFormat::NVFP4;
  std::vector<float> xf(HIDDEN), act(INTER);
  for (int c = 0; c < HIDDEN; ++c) {
    const std::uint32_t w = std::uint32_t(x[c]) << 16;
    std::memcpy(&xf[c], &w, 4);
  }
  const float* xs[1] = {xf.data()};
  const auto* c13 = reinterpret_cast<const std::uint8_t*>(h.expert_w13(expert));
  const auto* s13 = reinterpret_cast<const std::uint8_t*>(h.expert_w13s(expert));
  const std::size_t crow = nv ? HIDDEN / 2 : HIDDEN, srow = nv ? HIDDEN / 16 : HIDDEN / 32;
  for (int i = 0; i < INTER; ++i) {
    float g, u;
    if (nv) {
      dot_nvfp4<HIDDEN>(c13 + 2 * i * crow, s13 + 2 * i * srow, h.f13[expert], xs, 1, &g);
      dot_nvfp4<HIDDEN>(c13 + (2 * i + 1) * crow, s13 + (2 * i + 1) * srow, h.f13[expert], xs, 1, &u);
    } else {
      dot_mxfp8<HIDDEN>(c13 + 2 * i * crow, s13 + 2 * i * srow, 1.f, xs, 1, &g);
      dot_mxfp8<HIDDEN>(c13 + (2 * i + 1) * crow, s13 + (2 * i + 1) * srow, 1.f, xs, 1, &u);
    }
    act[i] = act_fn(bf16r(g), bf16r(u));
  }
  const float* as[1] = {act.data()};
  const auto* c2 = reinterpret_cast<const std::uint8_t*>(h.expert_w2(expert));
  const auto* s2 = reinterpret_cast<const std::uint8_t*>(h.expert_w2s(expert));
  const std::size_t crow2 = nv ? INTER / 2 : INTER, srow2 = nv ? INTER / 16 : INTER / 32;
  for (int r = 0; r < HIDDEN; ++r) {
    float v;
    if (nv) dot_nvfp4<INTER>(c2 + r * crow2, s2 + r * srow2, h.f2[expert], as, 1, &v);
    else dot_mxfp8<INTER>(c2 + r * crow2, s2 + r * srow2, 1.f, as, 1, &v);
    y[r] = bf16r(v);
  }
}

}  // namespace ingine::model
