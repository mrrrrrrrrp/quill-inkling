#include "model/vision_cpu.h"

#include "core/safetensors.h"
#include "kernels/vision/vision.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ingine::model {

namespace detail {

// `threads` threads in all: the calling thread (worker 0) and threads - 1 workers. run() hands out items 0..n-1.
class Pool {
 public:
  explicit Pool(int threads) : threads_(std::max(1, threads)) {
    for (int id = 1; id < threads_; ++id) workers_.emplace_back([this, id] { loop(id); });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> lock(m_);
      quit_ = true;
    }
    cv_.notify_all();
    for (std::thread& t : workers_) t.join();
  }
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  int size() const { return threads_; }

  // fn(item, worker) for the items 0..n-1, worker < size(); returns when every item has run.
  void run(int n, const std::function<void(int, int)>& fn) {
    if (n <= 0) return;
    if (threads_ == 1 || n == 1) {
      for (int i = 0; i < n; ++i) fn(i, 0);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(m_);
      fn_ = &fn;
      n_ = n;
      next_ = 0;
      busy_ = threads_ - 1;
      ++gen_;
    }
    cv_.notify_all();
    work(0);
    std::unique_lock<std::mutex> lock(m_);
    done_.wait(lock, [this] { return busy_ == 0; });
  }

 private:
  void work(int worker) {
    for (;;) {
      const int i = next_.fetch_add(1);
      if (i >= n_) break;
      (*fn_)(i, worker);
    }
  }

  void loop(int worker) {
    std::uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [this, seen] { return quit_ || gen_ != seen; });
        if (quit_) return;
        seen = gen_;
      }
      work(worker);
      std::lock_guard<std::mutex> lock(m_);
      if (--busy_ == 0) done_.notify_all();
    }
  }

  int threads_;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_, done_;
  const std::function<void(int, int)>* fn_ = nullptr;
  int n_ = 0;
  std::atomic<int> next_{0};
  int busy_ = 0;
  std::uint64_t gen_ = 0;
  bool quit_ = false;
};

struct Layer {
  std::vector<float> ln1w, ln1b, ln2w, ln2b, ls1, ls2, bqkv, bo, bfc, bproj;
  // bf16, row-major [out, in]: qkv [3072, 1024] (q, k, v stacked), wo [1024, 1024], fc [4096, 1024], proj [1024, 4096]
  std::vector<std::uint16_t> wqkv, wo, fc, proj;
};

struct VisionCpuState {
  explicit VisionCpuState(int threads) : pool(threads) {}
  Pool pool;
  std::vector<std::uint16_t> patch_w, proj_in, proj_out;  // bf16
  std::vector<float> pos, cls, ln_pre_w, ln_pre_b, inv_freq;
  std::vector<Layer> layers;
  std::size_t host_bytes = 0;
};

}  // namespace detail

namespace {

namespace vk = kernels::vision;

using detail::Layer;
using detail::Pool;
using detail::VisionCpuState;

constexpr int H = vk::HIDDEN;         // 1024: residual width
constexpr int HEAD = vk::HEAD_DIM;    // 64
constexpr int HEADS = vk::HEADS;      // 16
constexpr int MLP = vk::MLP_DIM;      // 4096
constexpr int MERGED = vk::MERGED;    // 4096
constexpr int OUT_DIM = 2304;             // the language model's hidden size
constexpr int LAYERS = 23;            // encoder layers
constexpr int ROWS = 1024;            // token rows per pass over a GEMM (bounds the buffers)
constexpr int QROWS = 64;             // query rows per attention task

int round_up(int v, int to) { return (v + to - 1) / to * to; }

// token 1 + p of an image `cols` tiles wide: its row and column on the 2 x 2 patch grid of each tile
void patch_position(int p, int cols, int& row, int& col) {
  const int tile = p >> 2, in = p & 3;
  row = (tile / cols) * 2 + (in >> 1);
  col = (tile % cols) * 2 + (in & 1);
}

// one axis of the bilinear resampling of the 32-entry position table to `size` positions (as the GPU kernel)
void axis_taps(int index, int size, int& t0, int& t1, float& w1) {
  const float src = (float(index) + 0.5f) * float(vk::POS_GRID) / float(size) - 0.5f;
  const float low = std::floor(src);
  w1 = src - low;
  const int l = int(low);
  t0 = std::clamp(l, 0, vk::POS_GRID - 1);
  t1 = std::clamp(l + 1, 0, vk::POS_GRID - 1);
}

inline float bf16_to_f32(std::uint16_t b) {
  const std::uint32_t w = std::uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

inline std::uint16_t f32_to_bf16(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;  // NaN
  u += 0x7fffu + ((u >> 16) & 1u);                      // round to nearest, ties to even
  return std::uint16_t(u >> 16);
}

// 8 bf16 values at p as 8 floats
inline __m256 load8_bf16(const std::uint16_t* p) {
  const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}

inline float hsum(__m256 v) {
  const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  const __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
  return _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 1)));
}

// y[i * ldy + j] = sum_k x[i * ldx + k] * w[j * ldw + k], i < MR, j < 4; w bf16, K a multiple of 8
template <int MR>
void dot_bf16(const float* x, int ldx, const std::uint16_t* w, int ldw, int K, float* y, int ldy) {
  __m256 acc[MR][4];
  for (int i = 0; i < MR; ++i)
    for (int j = 0; j < 4; ++j) acc[i][j] = _mm256_setzero_ps();
  for (int k = 0; k < K; k += 8) {
    __m256 wv[4];
    for (int j = 0; j < 4; ++j) wv[j] = load8_bf16(w + std::size_t(j) * ldw + k);
    for (int i = 0; i < MR; ++i) {
      const __m256 xv = _mm256_loadu_ps(x + std::size_t(i) * ldx + k);
      for (int j = 0; j < 4; ++j) acc[i][j] = _mm256_fmadd_ps(xv, wv[j], acc[i][j]);
    }
  }
  for (int i = 0; i < MR; ++i)
    for (int j = 0; j < 4; ++j) y[std::size_t(i) * ldy + j] = hsum(acc[i][j]);
}

// the same with FP32 operands: y[i][j] = sum_k a[i][k] * b[j][k]
template <int MR>
void dot_f32(const float* a, int lda, const float* b, int ldb, int K, float* y, int ldy) {
  __m256 acc[MR][4];
  for (int i = 0; i < MR; ++i)
    for (int j = 0; j < 4; ++j) acc[i][j] = _mm256_setzero_ps();
  for (int k = 0; k < K; k += 8) {
    __m256 bv[4];
    for (int j = 0; j < 4; ++j) bv[j] = _mm256_loadu_ps(b + std::size_t(j) * ldb + k);
    for (int i = 0; i < MR; ++i) {
      const __m256 av = _mm256_loadu_ps(a + std::size_t(i) * lda + k);
      for (int j = 0; j < 4; ++j) acc[i][j] = _mm256_fmadd_ps(av, bv[j], acc[i][j]);
    }
  }
  for (int i = 0; i < MR; ++i)
    for (int j = 0; j < 4; ++j) y[std::size_t(i) * ldy + j] = hsum(acc[i][j]);
}

// Y[M][N] = X[M][K] * W[N][K]^T (+ bias[N]), then exact GELU if asked. X and Y are FP32 row-major, W bf16.
void linear(Pool& pool, const float* X, int M, int K, const std::uint16_t* W, int N, float* Y, const float* bias,
            bool gelu) {
  if (N % 4 != 0 || K % 8 != 0) throw std::logic_error("vision cpu: GEMM shape");
  pool.run((M + 31) / 32, [&](int blk, int) {
    const int m0 = blk * 32, m1 = std::min(M, m0 + 32);
    for (int m = m0; m < m1;) {
      const int mr = std::min(4, m1 - m);
      const float* x = X + std::size_t(m) * K;
      float* y = Y + std::size_t(m) * N;
      for (int n = 0; n < N; n += 4) {
        const std::uint16_t* w = W + std::size_t(n) * K;
        if (mr == 4) {
          dot_bf16<4>(x, K, w, K, K, y + n, N);
        } else {
          for (int i = 0; i < mr; ++i) dot_bf16<1>(x + std::size_t(i) * K, K, w, K, K, y + std::size_t(i) * N + n, N);
        }
      }
      m += mr;
    }
    for (int m = m0; m < m1; ++m) {
      float* y = Y + std::size_t(m) * N;
      for (int n = 0; n < N; ++n) {
        const float v = y[n] + (bias ? bias[n] : 0.f);
        y[n] = gelu ? 0.5f * v * (1.f + std::erf(v * 0.70710678118654752f)) : v;
      }
    }
  });
}

// rows of x (M of them, H values each) normalised over their H values: out = (x - mean) / sqrt(var + eps) * w + b
void layernorm(Pool& pool, const float* x, const float* w, const float* b, int M, float* out) {
  pool.run((M + 63) / 64, [&](int blk, int) {
    const int end = std::min(M, blk * 64 + 64);
    for (int m = blk * 64; m < end; ++m) {
      const float* xr = x + std::size_t(m) * H;
      float mean = 0.f;
      for (int c = 0; c < H; ++c) mean += xr[c];
      mean /= float(H);
      float var = 0.f;
      for (int c = 0; c < H; ++c) var += (xr[c] - mean) * (xr[c] - mean);
      var /= float(H);
      const float r = 1.f / std::sqrt(var + 1e-5f);
      float* orow = out + std::size_t(m) * H;
      for (int c = 0; c < H; ++c) orow[c] = (xr[c] - mean) * r * w[c] + b[c];
    }
  });
}

// h += gamma * y for M rows of H values (layer scale and residual)
void residual(Pool& pool, float* h, const float* y, const float* gamma, int M) {
  pool.run((M + 63) / 64, [&](int blk, int) {
    const int end = std::min(M, blk * 64 + 64);
    for (int m = blk * 64; m < end; ++m) {
      float* hr = h + std::size_t(m) * H;
      const float* yr = y + std::size_t(m) * H;
      for (int c = 0; c < H; ++c) hr[c] += gamma[c] * yr[c];
    }
  });
}

// out[i][h * 64 + d] = softmax_j(scale * q_h[i] . k_h[j]) . v_h[j][d] over the S keys, for the S query rows and all
// heads. q, k: f32 [Sp][1024] (rows >= S zero); vt: f32 [1024][Sp] (values transposed, columns >= S zero).
void attention(Pool& pool, const float* q, const float* k, const float* vt, int S, int Sp, float* out, float scale) {
  const int chunks = (S + QROWS - 1) / QROWS;
  std::vector<std::vector<float>> scratch(std::size_t(pool.size()), std::vector<float>(std::size_t(QROWS) * Sp));
  pool.run(HEADS * chunks, [&](int item, int worker) {
    const int h = item / chunks, c0 = (item % chunks) * QROWS;
    const int Q = std::min(QROWS, S - c0);
    float* p = scratch[std::size_t(worker)].data();  // [Q][Sp] scores, then probabilities
    const float* qh = q + std::size_t(c0) * H + h * HEAD;
    const float* kh = k + h * HEAD;
    for (int i0 = 0; i0 < Q; i0 += 4) {
      const int mr = std::min(4, Q - i0);
      for (int j = 0; j < Sp; j += 4) {
        const float* a = qh + std::size_t(i0) * H;
        const float* b = kh + std::size_t(j) * H;
        float* dst = p + std::size_t(i0) * Sp + j;
        if (mr == 4) {
          dot_f32<4>(a, H, b, H, HEAD, dst, Sp);
        } else {
          for (int i = 0; i < mr; ++i) dot_f32<1>(a + std::size_t(i) * H, H, b, H, HEAD, dst + std::size_t(i) * Sp, Sp);
        }
      }
    }
    for (int i = 0; i < Q; ++i) {
      float* r = p + std::size_t(i) * Sp;
      float m = -INFINITY;
      for (int j = 0; j < S; ++j) m = std::max(m, r[j] * scale);
      float sum = 0.f;
      for (int j = 0; j < S; ++j) {
        r[j] = std::exp(r[j] * scale - m);
        sum += r[j];
      }
      const float inv = 1.f / sum;
      for (int j = 0; j < S; ++j) r[j] *= inv;
      for (int j = S; j < Sp; ++j) r[j] = 0.f;
    }
    for (int i0 = 0; i0 < Q; i0 += 4) {
      const int mr = std::min(4, Q - i0);
      for (int d = 0; d < HEAD; d += 4) {
        const float* a = p + std::size_t(i0) * Sp;
        const float* b = vt + std::size_t(h * HEAD + d) * Sp;
        float* dst = out + std::size_t(c0 + i0) * H + h * HEAD + d;
        if (mr == 4) {
          dot_f32<4>(a, Sp, b, Sp, Sp, dst, H);
        } else {
          for (int i = 0; i < mr; ++i) dot_f32<1>(a + std::size_t(i) * Sp, Sp, b, Sp, Sp, dst + std::size_t(i) * H, H);
        }
      }
    }
  });
}

std::vector<std::uint16_t> bf16_values(const SafetensorsModel& m, const std::string& name, std::int64_t numel) {
  const SafetensorsRef ref = m.at(name);
  if (ref->dtype != "BF16" || ref->numel() != numel) {
    throw std::runtime_error("vision weights: " + name + " is not BF16 with " + std::to_string(numel) + " values");
  }
  const std::vector<std::byte> raw = m.read(name);
  std::vector<std::uint16_t> out(static_cast<std::size_t>(numel));
  std::memcpy(out.data(), raw.data(), out.size() * 2);
  return out;
}

std::vector<std::uint16_t> bf16_matrix(const SafetensorsModel& m, const std::string& name, std::int64_t rows,
                                       std::int64_t cols) {
  const SafetensorsRef ref = m.at(name);
  if (ref->dtype != "BF16" || ref->shape != std::vector<std::int64_t>{rows, cols}) {
    throw std::runtime_error("vision weights: " + name + " is not BF16 [" + std::to_string(rows) + ", " +
                             std::to_string(cols) + "]");
  }
  return bf16_values(m, name, rows * cols);
}

std::vector<float> f32_values(const SafetensorsModel& m, const std::string& name, std::int64_t numel) {
  const std::vector<std::uint16_t> bits = bf16_values(m, name, numel);
  std::vector<float> out(bits.size());
  for (std::size_t i = 0; i < bits.size(); ++i) out[i] = bf16_to_f32(bits[i]);
  return out;
}

void load(const std::filesystem::path& dir, VisionCpuState& s) {
  const SafetensorsModel m(dir);
  if (!m.contains("model.visual.proj_out.weight")) {
    throw std::runtime_error("the checkpoint in " + dir.string() + " has no vision tower (model.visual.*)");
  }
  const std::string emb = "model.visual.backbone.embeddings.";
  s.patch_w = bf16_matrix(m, emb + "patch_embedding.weight", H, vk::PATCH_DIM);
  s.pos = f32_values(m, emb + "position_embedding", std::int64_t(vk::POS_GRID) * vk::POS_GRID * H);
  s.cls = f32_values(m, emb + "class_embedding", H);
  const std::vector<float> cls_pos = f32_values(m, emb + "cls_position_embedding", H);
  for (int c = 0; c < H; ++c) s.cls[std::size_t(c)] += cls_pos[std::size_t(c)];
  s.ln_pre_w = f32_values(m, "model.visual.backbone.ln_pre.weight", H);
  s.ln_pre_b = f32_values(m, "model.visual.backbone.ln_pre.bias", H);
  s.layers.resize(LAYERS);
  for (int i = 0; i < LAYERS; ++i) {
    const std::string p = "model.visual.backbone.layers." + std::to_string(i) + ".";
    Layer& L = s.layers[std::size_t(i)];
    L.ln1w = f32_values(m, p + "ln_1.weight", H);
    L.ln1b = f32_values(m, p + "ln_1.bias", H);
    L.ln2w = f32_values(m, p + "ln_2.weight", H);
    L.ln2b = f32_values(m, p + "ln_2.bias", H);
    L.ls1 = f32_values(m, p + "ls_1.gamma", H);
    L.ls2 = f32_values(m, p + "ls_2.gamma", H);
    for (const char* proj : {"q_proj", "k_proj", "v_proj"}) {
      const std::vector<std::uint16_t> w = bf16_matrix(m, p + "self_attn." + proj + ".weight", H, H);
      L.wqkv.insert(L.wqkv.end(), w.begin(), w.end());
      const std::vector<float> b = f32_values(m, p + "self_attn." + proj + ".bias", H);
      L.bqkv.insert(L.bqkv.end(), b.begin(), b.end());
    }
    L.wo = bf16_matrix(m, p + "self_attn.out_proj.weight", H, H);
    L.bo = f32_values(m, p + "self_attn.out_proj.bias", H);
    L.fc = bf16_matrix(m, p + "mlp.c_fc.weight", MLP, H);
    L.bfc = f32_values(m, p + "mlp.c_fc.bias", MLP);
    L.proj = bf16_matrix(m, p + "mlp.c_proj.weight", H, MLP);
    L.bproj = f32_values(m, p + "mlp.c_proj.bias", H);
  }
  s.proj_in = bf16_matrix(m, "model.visual.proj_in.weight", MERGED, MERGED);
  s.proj_out = bf16_matrix(m, "model.visual.proj_out.weight", OUT_DIM, MERGED);
  s.inv_freq.resize(vk::ROPE_FREQS);  // 1 / 10000^(2i / 32), in FP32 as the reference computes it
  for (int i = 0; i < vk::ROPE_FREQS; ++i) {
    s.inv_freq[std::size_t(i)] = 1.0f / std::pow(10000.0f, float(2 * i) / float(2 * vk::ROPE_FREQS));
  }

  std::size_t bytes = (s.patch_w.size() + s.proj_in.size() + s.proj_out.size()) * 2;
  bytes += (s.pos.size() + s.cls.size() + s.ln_pre_w.size() + s.ln_pre_b.size() + s.inv_freq.size()) * 4;
  for (const Layer& L : s.layers) {
    bytes += (L.wqkv.size() + L.wo.size() + L.fc.size() + L.proj.size()) * 2;
    bytes += (L.ln1w.size() + L.ln1b.size() + L.ln2w.size() + L.ln2b.size() + L.ls1.size() + L.ls2.size() +
              L.bqkv.size() + L.bo.size() + L.bfc.size() + L.bproj.size()) * 4;
  }
  s.host_bytes = bytes;
}

DeviceBuffer encode_image(VisionCpuState& s, const std::uint8_t* rgb, int height, int width) {
  if (!rgb || height < 1 || width < 1) throw std::invalid_argument("VisionCpuEncoder::encode: empty image");
  Pool& pool = s.pool;
  const int rows = (height + vk::TILE - 1) / vk::TILE, cols = (width + vk::TILE - 1) / vk::TILE;
  const int N = rows * cols;      // image tokens
  const int P = 4 * N;            // patches
  const int S = 1 + P;            // tokens of the tower: the class token, then the patches
  const int Sp = round_up(S, 8);  // padded rows of the attention operands
  const int Np = round_up(N, 16); // padded output rows (VisionEncoder::padded_tokens)

  // patches [P][1200]: (v - 127.5) / 127.5, v = -1 outside the image (the processor pads to whole tiles)
  std::vector<float> patches(std::size_t(P) * vk::PATCH_DIM);
  pool.run(P, [&](int p, int) {
    int row, col;
    patch_position(p, cols, row, col);
    float* dst = patches.data() + std::size_t(p) * vk::PATCH_DIM;
    for (int e = 0; e < vk::PATCH_DIM; ++e) {
      const int y = row * vk::PATCH + e / (vk::PATCH * 3), x = col * vk::PATCH + (e / 3) % vk::PATCH, c = e % 3;
      const float v = (y < height && x < width) ? float(rgb[(std::size_t(y) * width + x) * 3 + c]) : -1.f;
      dst[e] = (v - 127.5f) / 127.5f;
    }
  });

  // h [Sp][1024]: the class token; then each patch's embedding plus the position table resampled bilinearly to the
  // 2 x 2 grid of its tile (half-pixel centres, edge clamp), as the GPU embed kernel does
  std::vector<float> emb(std::size_t(P) * H);
  linear(pool, patches.data(), P, vk::PATCH_DIM, s.patch_w.data(), H, emb.data(), nullptr, false);
  std::vector<float> h(std::size_t(Sp) * H, 0.f);
  std::copy(s.cls.begin(), s.cls.end(), h.begin());
  pool.run(P, [&](int p, int) {
    int row, col;
    patch_position(p, cols, row, col);
    int r0, r1, c0, c1;
    float wr, wc;
    axis_taps(row, rows * 2, r0, r1, wr);
    axis_taps(col, cols * 2, c0, c1, wc);
    const float w00 = (1.f - wr) * (1.f - wc), w01 = (1.f - wr) * wc, w10 = wr * (1.f - wc), w11 = wr * wc;
    const float* pa = s.pos.data() + std::size_t(r0 * vk::POS_GRID + c0) * H;
    const float* pb = s.pos.data() + std::size_t(r0 * vk::POS_GRID + c1) * H;
    const float* pd = s.pos.data() + std::size_t(r1 * vk::POS_GRID + c0) * H;
    const float* pe = s.pos.data() + std::size_t(r1 * vk::POS_GRID + c1) * H;
    const float* x = emb.data() + std::size_t(p) * H;
    float* out = h.data() + std::size_t(1 + p) * H;
    for (int c = 0; c < H; ++c) out[c] = x[c] + (((pa[c] * w00 + pb[c] * w01) + pd[c] * w10) + pe[c] * w11);
  });
  layernorm(pool, h.data(), s.ln_pre_w.data(), s.ln_pre_b.data(), S, h.data());

  // per-layer buffers; GEMMs run over ROWS-token chunks so their temporaries stay small
  std::vector<float> xn(std::size_t(S) * H);
  std::vector<float> q(std::size_t(Sp) * H, 0.f), k(std::size_t(Sp) * H, 0.f);
  std::vector<float> vt(std::size_t(H) * Sp, 0.f), attn(std::size_t(Sp) * H, 0.f);
  std::vector<float> chunk_qkv(std::size_t(ROWS) * 3 * H), chunk_mlp(std::size_t(ROWS) * MLP);
  std::vector<float> chunk_out(std::size_t(ROWS) * MERGED);  // holds OUT_DIM (2304) columns at the merger
  const float scale = 1.f / std::sqrt(float(HEAD));

  for (const Layer& L : s.layers) {
    // h += ls1 * out_proj(attention(ln1(h)))
    layernorm(pool, h.data(), L.ln1w.data(), L.ln1b.data(), S, xn.data());
    for (int r = 0; r < S; r += ROWS) {
      const int m = std::min(ROWS, S - r);
      linear(pool, xn.data() + std::size_t(r) * H, m, H, L.wqkv.data(), 3 * H, chunk_qkv.data(), nullptr, false);
      // rotary angles of q and k (class token at (0, 0), patch at (row + 1, col + 1)); bias; v transposed
      pool.run(m, [&](int t, int) {
        const int tok = r + t;
        float pos_row = 0.f, pos_col = 0.f;
        if (tok > 0) {
          int prow, pcol;
          patch_position(tok - 1, cols, prow, pcol);
          pos_row = float(prow + 1);
          pos_col = float(pcol + 1);
        }
        const float* x = chunk_qkv.data() + std::size_t(t) * 3 * H;
        float* qo = q.data() + std::size_t(tok) * H;
        float* ko = k.data() + std::size_t(tok) * H;
        for (int c = 0; c < H; c += 2) {
          const int j = (c % HEAD) / 2;
          const float angle = (j < vk::ROPE_FREQS ? pos_col : pos_row) * s.inv_freq[std::size_t(j % vk::ROPE_FREQS)];
          const float sn = std::sin(angle), cs = std::cos(angle);
          const float qa = x[c] + L.bqkv[std::size_t(c)], qb = x[c + 1] + L.bqkv[std::size_t(c + 1)];
          const float ka = x[H + c] + L.bqkv[std::size_t(H + c)], kb = x[H + c + 1] + L.bqkv[std::size_t(H + c + 1)];
          qo[c] = qa * cs - qb * sn;
          qo[c + 1] = qb * cs + qa * sn;
          ko[c] = ka * cs - kb * sn;
          ko[c + 1] = kb * cs + ka * sn;
        }
        for (int c = 0; c < H; ++c) vt[std::size_t(c) * Sp + tok] = x[2 * H + c] + L.bqkv[std::size_t(2 * H + c)];
      });
    }
    attention(pool, q.data(), k.data(), vt.data(), S, Sp, attn.data(), scale);
    for (int r = 0; r < S; r += ROWS) {
      const int m = std::min(ROWS, S - r);
      linear(pool, attn.data() + std::size_t(r) * H, m, H, L.wo.data(), H, chunk_out.data(), L.bo.data(), false);
      residual(pool, h.data() + std::size_t(r) * H, chunk_out.data(), L.ls1.data(), m);
    }
    // h += ls2 * c_proj(gelu(c_fc(ln2(h))))
    for (int r = 0; r < S; r += ROWS) {
      const int m = std::min(ROWS, S - r);
      layernorm(pool, h.data() + std::size_t(r) * H, L.ln2w.data(), L.ln2b.data(), m, xn.data());
      linear(pool, xn.data(), m, H, L.fc.data(), MLP, chunk_mlp.data(), L.bfc.data(), true);
      linear(pool, chunk_mlp.data(), m, MLP, L.proj.data(), H, chunk_out.data(), L.bproj.data(), false);
      residual(pool, h.data() + std::size_t(r) * H, chunk_out.data(), L.ls2.data(), m);
    }
  }

  // merger: tile t's four patch states are the consecutive rows 1 + 4t .. 4t + 3 of h, so its 4096 values start at
  // h + H + t * MLP. proj_in, GELU, proj_out; the rows are rounded to BF16 as the GPU tower's output is
  std::vector<std::uint16_t> bits(std::size_t(Np) * OUT_DIM, 0);
  for (int r = 0; r < N; r += ROWS) {
    const int m = std::min(ROWS, N - r);
    linear(pool, h.data() + H + std::size_t(r) * MLP, m, MLP, s.proj_in.data(), MLP, chunk_mlp.data(), nullptr, true);
    linear(pool, chunk_mlp.data(), m, MLP, s.proj_out.data(), OUT_DIM, chunk_out.data(), nullptr, false);
    pool.run(m, [&](int t, int) {
      const float* y = chunk_out.data() + std::size_t(t) * OUT_DIM;
      std::uint16_t* o = bits.data() + std::size_t(r + t) * OUT_DIM;
      for (int c = 0; c < OUT_DIM; ++c) o[c] = f32_to_bf16(y[c]);
    });
  }

  DeviceBuffer out(std::size_t(Np) * OUT_DIM * 2);
  out.copy_from_host(bits.data(), bits.size() * 2);
  return out;
}

}  // namespace

VisionCpuEncoder::VisionCpuEncoder(const std::filesystem::path& model_dir, int threads)
    : state_(std::make_unique<VisionCpuState>(threads)) {
  load(model_dir, *state_);
}

VisionCpuEncoder::~VisionCpuEncoder() = default;

DeviceBuffer VisionCpuEncoder::encode(const std::uint8_t* rgb, int height, int width) {
  return encode_image(*state_, rgb, height, width);
}

std::size_t VisionCpuEncoder::host_bytes() const { return state_->host_bytes; }

}  // namespace ingine::model
