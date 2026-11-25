// Vision kernels against FP64 oracles of the transformers operations (Inkling2VisionModel): patchify (also
// against the reference processor's pixel_values), the resampled position embeddings, LayerNorm, the rotary
// attention inputs, softmax, bias + GELU, layer scale + residual and the merger rows.
#include "common/test_util.h"
#include "core/safetensors.h"
#include "kernels/vision/vision.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace ingine::test;
namespace vk = ingine::kernels::vision;

namespace {

int failures = 0;

#define CHECK(cond)                                                                 \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                                   \
    }                                                                               \
  } while (0)

std::uint16_t bits_of(float f) {
  const __nv_bfloat16 b = float_to_bf16(f);
  std::uint16_t u;
  std::memcpy(&u, &b, 2);
  return u;
}
float value_of(std::uint16_t u) {
  const std::uint32_t w = std::uint32_t(u) << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

std::vector<float> rand_f32(Rng& rng, std::size_t n, double scale, double offset = 0) {
  std::vector<float> v(n);
  for (auto& x : v) x = float(offset + rng.normal() * scale);
  return v;
}

// A BF16 result against an FP64 reference: within half a BF16 step of it (plus FP32 noise).
bool report_bf16(const char* what, const std::vector<std::uint16_t>& got, const std::vector<double>& ref,
                 double abs_slack = 1e-6) {
  double worst = 0;
  std::size_t at = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double allowed = std::fabs(ref[i]) * (1.0 / 256.0) * 1.001 + abs_slack;
    const double excess = std::fabs(double(value_of(got[i])) - ref[i]) / allowed;
    if (!(excess <= worst)) worst = excess, at = i;
  }
  const bool ok = worst <= 1.0;
  std::printf("%s %s: worst error %.3f of the BF16 half step (at %zu of %zu)\n", ok ? "PASS" : "FAIL", what, worst, at,
              ref.size());
  failures += !ok;
  return ok;
}

bool report_f32(const char* what, const std::vector<float>& got, const std::vector<double>& ref, double tol) {
  const bool ok = report(what, compare(got.data(), ref.data(), ref.size()), tol);
  failures += !ok;
  return ok;
}

void sync() { INGINE_TEST_CUDA(cudaDeviceSynchronize()); }

// Row and column (in patches) of patch p: tile raster order, then the tile's quadrants.
void patch_position(int p, int cols, int& row, int& col) {
  const int tile = p / 4, in = p % 4;
  row = (tile / cols) * 2 + in / 2;
  col = (tile % cols) * 2 + in % 2;
}

// ---- patchify

void test_patchify_random(Rng& rng, int height, int width) {
  const int rows = (height + 39) / 40, cols = (width + 39) / 40, patches = 4 * rows * cols;
  const int out_rows = patches + 5;
  std::vector<std::uint8_t> rgb(std::size_t(height) * width * 3);
  for (auto& v : rgb) v = std::uint8_t(rng.next() & 0xFF);
  for (int v = 0; v < 256 && std::size_t(v) < rgb.size(); ++v) rgb[std::size_t(v)] = std::uint8_t(v);  // every byte value
  DeviceArray<std::uint8_t> d_rgb(rgb);
  DeviceArray<std::uint16_t> out(std::size_t(out_rows) * vk::PATCH_DIM);
  std::vector<std::uint16_t> ones(out.count, 0xFFFF);
  out.upload(ones);
  vk::patchify(d_rgb.get(), height, width, rows, cols, reinterpret_cast<__nv_bfloat16*>(out.get()), out_rows, nullptr);
  sync();
  const auto got = out.download();
  std::size_t wrong = 0;
  for (int p = 0; p < out_rows; ++p) {
    int row = 0, col = 0;
    if (p < patches) patch_position(p, cols, row, col);
    for (int e = 0; e < vk::PATCH_DIM; ++e) {
      std::uint16_t want = 0;
      if (p < patches) {
        const int y = row * 20 + e / 60, x = col * 20 + (e / 3) % 20, c = e % 3;
        const float v = (y < height && x < width) ? float(rgb[(std::size_t(y) * width + x) * 3 + c]) : -1.f;
        want = bits_of((v - 127.5f) / 127.5f);  // the processor's fused rescale + normalize, in FP32
      }
      wrong += got[std::size_t(p) * vk::PATCH_DIM + e] != want;
    }
  }
  std::printf("%s patchify %dx%d (%d patches): %zu values differ\n", wrong ? "FAIL" : "PASS", width, height, patches,
              wrong);
  failures += wrong != 0;
}

// The kernel's patches are the reference processor's pixel_values, rounded to BF16.
void test_patchify_golden(const std::string& stem) {
  const std::filesystem::path path = std::filesystem::path(INGINE_GOLDEN_DIR) / "vision" / (stem + ".safetensors");
  const ingine::SafetensorsFile golden(path);
  const auto& rgb_t = golden.at("rgb");
  const int height = int(rgb_t.shape[0]), width = int(rgb_t.shape[1]);
  const int rows = (height + 39) / 40, cols = (width + 39) / 40, tiles = rows * cols;
  const std::vector<std::byte> rgb_bytes = golden.read("rgb"), pv_bytes = golden.read("pixel_values");
  CHECK(pv_bytes.size() == std::size_t(tiles) * 40 * 40 * 3 * 4);
  std::vector<std::uint8_t> rgb(rgb_bytes.size());
  std::memcpy(rgb.data(), rgb_bytes.data(), rgb.size());
  std::vector<float> pv(pv_bytes.size() / 4);
  std::memcpy(pv.data(), pv_bytes.data(), pv_bytes.size());
  DeviceArray<std::uint8_t> d_rgb(rgb);
  DeviceArray<std::uint16_t> out(std::size_t(tiles) * 4 * vk::PATCH_DIM);
  vk::patchify(d_rgb.get(), height, width, rows, cols, reinterpret_cast<__nv_bfloat16*>(out.get()), tiles * 4, nullptr);
  sync();
  const auto got = out.download();
  std::size_t wrong = 0;
  for (int t = 0; t < tiles; ++t) {
    for (int in = 0; in < 4; ++in) {
      for (int e = 0; e < vk::PATCH_DIM; ++e) {
        const int y = (in / 2) * 20 + e / 60, x = (in % 2) * 20 + (e / 3) % 20, c = e % 3;
        const float want = pv[((std::size_t(t) * 40 + y) * 40 + x) * 3 + c];  // pixel_values [N, 40, 40, 3]
        wrong += got[(std::size_t(t) * 4 + in) * vk::PATCH_DIM + e] != bits_of(want);
      }
    }
  }
  std::printf("%s patchify vs processor %s (%d tiles): %zu values differ\n", wrong ? "FAIL" : "PASS", stem.c_str(),
              tiles, wrong);
  failures += wrong != 0;
}

// ---- embeddings

void test_embed(Rng& rng, int rows, int cols) {
  const int patches = 4 * rows * cols, S = patches + 1, out_rows = ((S + 63) / 64) * 64;
  const auto x = rand_f32(rng, std::size_t(patches) * vk::HIDDEN, 1.0);
  const auto table = rand_f32(rng, std::size_t(vk::POS_GRID) * vk::POS_GRID * vk::HIDDEN, 0.5);
  const auto cls = rand_f32(rng, vk::HIDDEN, 1.0);
  DeviceArray<float> d_x(x), d_table(table), d_cls(cls), h(std::size_t(out_rows) * vk::HIDDEN);
  h.upload(std::vector<float>(h.count, 7.f));
  vk::embed_assemble(d_x.get(), d_table.get(), d_cls.get(), rows, cols, h.get(), out_rows, nullptr);
  sync();
  const auto got = h.download();
  std::vector<double> ref(got.size(), 0.0);
  for (int c = 0; c < vk::HIDDEN; ++c) ref[std::size_t(c)] = cls[std::size_t(c)];
  auto taps = [](int index, int size, int& t0, int& t1, double& w1) {
    const double src = (index + 0.5) * vk::POS_GRID / size - 0.5, low = std::floor(src);
    w1 = src - low;
    t0 = std::clamp(int(low), 0, vk::POS_GRID - 1), t1 = std::clamp(int(low) + 1, 0, vk::POS_GRID - 1);
  };
  for (int p = 0; p < patches; ++p) {
    int row, col, r0, r1, c0, c1;
    double wr, wc;
    patch_position(p, cols, row, col);
    taps(row, rows * 2, r0, r1, wr);
    taps(col, cols * 2, c0, c1, wc);
    for (int c = 0; c < vk::HIDDEN; ++c) {
      auto e = [&](int r, int cc) { return double(table[(std::size_t(r) * vk::POS_GRID + cc) * vk::HIDDEN + c]); };
      ref[(std::size_t(p) + 1) * vk::HIDDEN + c] = x[std::size_t(p) * vk::HIDDEN + c] + (1 - wr) * (1 - wc) * e(r0, c0) +
                                                   (1 - wr) * wc * e(r0, c1) + wr * (1 - wc) * e(r1, c0) +
                                                   wr * wc * e(r1, c1);
    }
  }
  char what[64];
  std::snprintf(what, sizeof what, "embed_assemble %dx%d tiles", rows, cols);
  // the kernel computes the tap weights in FP32, as the reference does: a few 1e-6 on large grids
  report_f32(what, got, ref, 1e-5);
  bool pad_zero = true;
  for (std::size_t i = std::size_t(S) * vk::HIDDEN; i < got.size(); ++i) pad_zero &= got[i] == 0.f;
  CHECK(pad_zero);
}

// ---- LayerNorm

void test_layernorm(Rng& rng) {
  const int M = 37;
  auto x = rand_f32(rng, std::size_t(M) * vk::HIDDEN, 1.0);
  for (int m = 0; m < M; ++m) {  // rows of different scale and offset
    const float scale = float(std::exp(rng.uniform(-3, 3))), offset = float(rng.normal() * 3);
    for (int c = 0; c < vk::HIDDEN; ++c) x[std::size_t(m) * vk::HIDDEN + c] = x[std::size_t(m) * vk::HIDDEN + c] * scale + offset;
  }
  const auto w = rand_f32(rng, vk::HIDDEN, 0.3, 1.0), b = rand_f32(rng, vk::HIDDEN, 0.3);
  DeviceArray<float> d_x(x), d_w(w), d_b(b), out32(x.size());
  DeviceArray<std::uint16_t> out16(x.size());
  vk::layernorm(d_x.get(), d_w.get(), d_b.get(), M, out32.get(), true, nullptr);
  vk::layernorm(d_x.get(), d_w.get(), d_b.get(), M, out16.get(), false, nullptr);
  sync();
  const auto got32 = out32.download();
  const auto got16 = out16.download();
  std::vector<double> ref(x.size());
  for (int m = 0; m < M; ++m) {
    double mean = 0, var = 0;
    for (int c = 0; c < vk::HIDDEN; ++c) mean += x[std::size_t(m) * vk::HIDDEN + c];
    mean /= vk::HIDDEN;
    for (int c = 0; c < vk::HIDDEN; ++c) var += std::pow(x[std::size_t(m) * vk::HIDDEN + c] - mean, 2);
    var /= vk::HIDDEN;
    for (int c = 0; c < vk::HIDDEN; ++c) {
      ref[std::size_t(m) * vk::HIDDEN + c] =
          (x[std::size_t(m) * vk::HIDDEN + c] - mean) / std::sqrt(var + 1e-5) * w[std::size_t(c)] + b[std::size_t(c)];
    }
  }
  report_f32("layernorm fp32", got32, ref, 5e-6);
  std::size_t wrong = 0;
  for (std::size_t i = 0; i < got32.size(); ++i) wrong += got16[i] != bits_of(got32[i]);
  CHECK(wrong == 0);  // the BF16 output is the FP32 output rounded once
  // in place
  vk::layernorm(d_x.get(), d_w.get(), d_b.get(), M, d_x.get(), true, nullptr);
  sync();
  CHECK(d_x.download() == got32);
}

// ---- rotary attention inputs

void test_qkv_post(Rng& rng, int rows, int cols) {
  const int S = 1 + 4 * rows * cols, Sp = ((S + 63) / 64) * 64;
  const auto qkv = rand_f32(rng, std::size_t(Sp) * 3 * vk::HIDDEN, 1.0);
  const auto bias = rand_f32(rng, 3 * vk::HIDDEN, 0.5);
  std::vector<float> inv_freq(vk::ROPE_FREQS);
  for (int i = 0; i < vk::ROPE_FREQS; ++i) inv_freq[std::size_t(i)] = float(1.0 / std::pow(10000.0, (2.0 * i) / 32.0));
  DeviceArray<float> d_qkv(qkv), d_bias(bias), d_freq(inv_freq);
  const std::size_t n = std::size_t(vk::HEADS) * Sp * vk::HEAD_DIM;
  DeviceArray<std::uint16_t> q(n), k(n), vt(n);
  const std::vector<std::uint16_t> ones(n, 0xFFFF);
  q.upload(ones), k.upload(ones), vt.upload(ones);
  vk::qkv_post(d_qkv.get(), d_bias.get(), d_freq.get(), rows, cols, Sp, reinterpret_cast<__nv_bfloat16*>(q.get()),
               reinterpret_cast<__nv_bfloat16*>(k.get()), reinterpret_cast<__nv_bfloat16*>(vt.get()), nullptr);
  sync();
  const auto gq = q.download(), gk = k.download(), gv = vt.download();
  std::vector<double> rq(n, 0.0), rk(n, 0.0), rv(n, 0.0);
  for (int t = 0; t < S; ++t) {
    int row = -1, col = -1;  // the class token: position (0, 0)
    if (t > 0) patch_position(t - 1, cols, row, col);
    for (int h = 0; h < vk::HEADS; ++h) {
      for (int j = 0; j < 32; ++j) {
        const double pos = j < 16 ? col + 1 : row + 1, angle = pos * double(inv_freq[std::size_t(j % 16)]);
        const double cs = std::cos(angle), sn = std::sin(angle);
        const int c = h * 64 + 2 * j;
        auto in = [&](int part, int cc) {
          return double(qkv[std::size_t(t) * 3 * vk::HIDDEN + part * vk::HIDDEN + cc]) +
                 double(bias[std::size_t(part) * vk::HIDDEN + cc]);
        };
        const std::size_t o = (std::size_t(h) * Sp + t) * 64 + 2 * j;
        rq[o] = in(0, c) * cs - in(0, c + 1) * sn, rq[o + 1] = in(0, c + 1) * cs + in(0, c) * sn;
        rk[o] = in(1, c) * cs - in(1, c + 1) * sn, rk[o + 1] = in(1, c + 1) * cs + in(1, c) * sn;
        rv[(std::size_t(h) * 64 + 2 * j) * Sp + t] = in(2, c);
        rv[(std::size_t(h) * 64 + 2 * j + 1) * Sp + t] = in(2, c + 1);
      }
    }
  }
  char what[64];
  std::snprintf(what, sizeof what, "qkv_post q %dx%d tiles", rows, cols);
  report_bf16(what, gq, rq, 2e-5);
  what[9] = 'k';
  report_bf16(what, gk, rk, 2e-5);
  what[9] = 'v';
  report_bf16(what, gv, rv, 2e-5);
  bool pad_zero = true;  // rows >= S of q and k, columns >= S of vt
  for (int h = 0; h < vk::HEADS; ++h) {
    for (int t = S; t < Sp; ++t) {
      for (int d = 0; d < 64; ++d) {
        pad_zero &= gq[(std::size_t(h) * Sp + t) * 64 + d] == 0 && gk[(std::size_t(h) * Sp + t) * 64 + d] == 0 &&
                    gv[(std::size_t(h) * 64 + d) * Sp + t] == 0;
      }
    }
  }
  CHECK(pad_zero);
}

// ---- softmax

void test_softmax(Rng& rng, int M, int ld, int S) {
  const auto scores = rand_f32(rng, std::size_t(M) * ld, 12.0);
  DeviceArray<float> d_scores(scores);
  DeviceArray<std::uint16_t> p(scores.size());
  p.upload(std::vector<std::uint16_t>(scores.size(), 0xFFFF));
  vk::softmax_rows(d_scores.get(), M, ld, S, 0.125f, reinterpret_cast<__nv_bfloat16*>(p.get()), nullptr);
  sync();
  const auto got = p.download();
  std::vector<double> ref(scores.size(), 0.0);
  bool pad_zero = true;
  for (int m = 0; m < M; ++m) {
    double mx = -INFINITY, sum = 0;
    for (int c = 0; c < S; ++c) mx = std::max(mx, 0.125 * scores[std::size_t(m) * ld + c]);
    for (int c = 0; c < S; ++c) sum += std::exp(0.125 * scores[std::size_t(m) * ld + c] - mx);
    for (int c = 0; c < S; ++c) ref[std::size_t(m) * ld + c] = std::exp(0.125 * scores[std::size_t(m) * ld + c] - mx) / sum;
    for (int c = S; c < ld; ++c) pad_zero &= got[std::size_t(m) * ld + c] == 0;
  }
  char what[64];
  std::snprintf(what, sizeof what, "softmax_rows M=%d ld=%d S=%d", M, ld, S);
  report_bf16(what, got, ref, 1e-9);
  CHECK(pad_zero);
}

// ---- bias, GELU, residual, merger

void test_bias_act(Rng& rng) {
  const int M = 13, N = vk::MLP_DIM;
  const auto x = rand_f32(rng, std::size_t(M) * N, 2.0);
  const auto bias = rand_f32(rng, N, 0.5);
  DeviceArray<float> d_x(x), d_bias(bias);
  DeviceArray<std::uint16_t> out(x.size());
  vk::bias_act(d_x.get(), d_bias.get(), M, N, true, reinterpret_cast<__nv_bfloat16*>(out.get()), nullptr);
  sync();
  std::vector<double> ref(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double v = double(x[i]) + double(bias[i % N]);
    ref[i] = 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0)));  // exact GELU
  }
  report_bf16("bias_act gelu", out.download(), ref, 2e-6);
  vk::bias_act(d_x.get(), nullptr, M, N, false, reinterpret_cast<__nv_bfloat16*>(out.get()), nullptr);
  sync();
  const auto plain = out.download();
  std::size_t wrong = 0;
  for (std::size_t i = 0; i < x.size(); ++i) wrong += plain[i] != bits_of(x[i]);
  CHECK(wrong == 0);
  vk::bias_act(d_x.get(), nullptr, M, N, true, reinterpret_cast<__nv_bfloat16*>(out.get()), nullptr);
  sync();
  for (std::size_t i = 0; i < x.size(); ++i) ref[i] = 0.5 * double(x[i]) * (1.0 + std::erf(double(x[i]) / std::sqrt(2.0)));
  report_bf16("bias_act gelu, no bias", out.download(), ref, 2e-6);
}

void test_residual(Rng& rng) {
  const int M = 29;
  const auto y = rand_f32(rng, std::size_t(M) * vk::HIDDEN, 3.0), h = rand_f32(rng, std::size_t(M) * vk::HIDDEN, 1.0);
  const auto bias = rand_f32(rng, vk::HIDDEN, 0.5), gamma = rand_f32(rng, vk::HIDDEN, 0.1);
  DeviceArray<float> d_y(y), d_h(h), d_bias(bias), d_gamma(gamma);
  vk::bias_scale_residual(d_y.get(), d_bias.get(), d_gamma.get(), M, d_h.get(), nullptr);
  sync();
  std::vector<double> ref(h.size());
  for (std::size_t i = 0; i < h.size(); ++i) {
    ref[i] = double(h[i]) + double(gamma[i % vk::HIDDEN]) * (double(y[i]) + double(bias[i % vk::HIDDEN]));
  }
  report_f32("bias_scale_residual", d_h.download(), ref, 1e-6);
}

void test_merge(Rng& rng) {
  const int N = 6, out_rows = 16;
  const auto h = rand_f32(rng, (std::size_t(N) * 4 + 1) * vk::HIDDEN, 2.0);
  DeviceArray<float> d_h(h);
  DeviceArray<std::uint16_t> out(std::size_t(out_rows) * vk::MERGED);
  out.upload(std::vector<std::uint16_t>(out.count, 0xFFFF));
  vk::merge_rows(d_h.get(), N, out_rows, reinterpret_cast<__nv_bfloat16*>(out.get()), nullptr);
  sync();
  const auto got = out.download();
  std::size_t wrong = 0;
  for (int t = 0; t < out_rows; ++t) {
    for (int c = 0; c < vk::MERGED; ++c) {
      // tile t: patches 4t .. 4t + 3 are rows 1 + 4t .. of h, concatenated
      const std::uint16_t want = t < N ? bits_of(h[(std::size_t(1) + 4 * t) * vk::HIDDEN + c]) : std::uint16_t(0);
      wrong += got[std::size_t(t) * vk::MERGED + c] != want;
    }
  }
  std::printf("%s merge_rows: %zu values differ\n", wrong ? "FAIL" : "PASS", wrong);
  failures += wrong != 0;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  Rng rng(20261122);
  test_patchify_random(rng, 40, 40);
  test_patchify_random(rng, 97, 131);   // padded right and bottom
  test_patchify_random(rng, 1, 1);
  test_patchify_random(rng, 200, 37);
  for (const char* stem : {"solid_red_40x40", "grad_100x60", "rgba_64x64", "noise_200x120"}) test_patchify_golden(stem);
  test_embed(rng, 1, 1);
  test_embed(rng, 2, 3);
  test_embed(rng, 16, 16);   // the 32 x 32 grid itself
  test_embed(rng, 31, 1);    // 62 x 2: up one way, down the other
  test_embed(rng, 43, 43);   // 86 x 86: the table is enlarged
  test_layernorm(rng);
  test_qkv_post(rng, 1, 1);
  test_qkv_post(rng, 3, 5);
  test_qkv_post(rng, 17, 18);
  test_softmax(rng, 5, 64, 5);
  test_softmax(rng, 33, 640, 613);
  test_softmax(rng, 3, 8192, 8001);
  test_bias_act(rng);
  test_residual(rng);
  test_merge(rng);
  if (failures) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
