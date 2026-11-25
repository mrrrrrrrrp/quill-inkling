// The vision tower with the checkpoint's weights against transformers (goldens of tools/vision_ref.py, FP32
// arithmetic): embeddings, the residual stream after layers 0, 11 and 22 and the projector output of every
// fixture. The tolerance of the output is transformers' own BF16 error (stored with each golden): ingine,
// with BF16 matrix operands but an FP32 residual stream, must be at least as close to FP32. Also query
// chunking and timings. Needs the model directory (skipped without it) and about 1.5 GB of GPU memory.
#include "common/test_util.h"
#include "core/safetensors.h"
#include "image/image.h"
#include "model/vision.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace ingine::test;
namespace fs = std::filesystem;
namespace image = ingine::image;
using ingine::model::VISION_OUT;
using ingine::model::VisionEncoder;

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

const fs::path kGolden = fs::path(INGINE_GOLDEN_DIR) / "vision";

template <class T>
std::vector<T> read_tensor(const ingine::SafetensorsFile& file, const char* name) {
  const std::vector<std::byte> raw = file.read(name);
  std::vector<T> out(raw.size() / sizeof(T));
  std::memcpy(out.data(), raw.data(), out.size() * sizeof(T));
  return out;
}

std::vector<std::uint8_t> read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

struct Error {
  double rel_rms = 0, cosine = 0;
};

Error error_of(const float* got, const float* ref, std::size_t n) {
  double d2 = 0, r2 = 0, g2 = 0, gr = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double g = got[i], r = ref[i];
    d2 += (g - r) * (g - r), r2 += r * r, g2 += g * g, gr += g * r;
  }
  return {std::sqrt(d2 / r2), gr / std::sqrt(g2 * r2)};
}

std::vector<float> rows_of(const ingine::DeviceBuffer& out, int tokens) {
  std::vector<std::uint16_t> bits(std::size_t(tokens) * VISION_OUT);
  out.copy_to_host(bits.data(), bits.size() * 2);
  std::vector<float> v(bits.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    const std::uint32_t w = std::uint32_t(bits[i]) << 16;
    std::memcpy(&v[i], &w, 4);
  }
  return v;
}

double seconds_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

void test_fixture(VisionEncoder& encoder, cudaStream_t stream, const fs::path& file) {
  const fs::path golden_path = kGolden / (file.stem().string() + ".safetensors");
  if (!fs::exists(golden_path)) return;
  const ingine::SafetensorsFile golden(golden_path);
  // the reference's own pixels when the golden has them (the decoder is tested in tests/image)
  std::vector<std::uint8_t> rgb;
  int height = 0, width = 0;
  if (golden.contains("rgb")) {
    const auto& t = golden.at("rgb");
    height = int(t.shape[0]), width = int(t.shape[1]);
    rgb = read_tensor<std::uint8_t>(golden, "rgb");
  } else {
    image::Limits limits;
    limits.max_tokens = image::MAX_TOKENS;
    image::Image img = image::decode(read_file(file), limits);
    height = img.height, width = img.width;
    rgb = std::move(img.rgb);
  }
  const std::vector<std::int32_t> grid = read_tensor<std::int32_t>(golden, "grid");
  const int tokens = grid[0] * grid[1];
  expect(image::tokens(height, width) == tokens, "token count");

  std::map<int, std::vector<float>> states;
  encoder.observer = [&](int stage, const float* h, int n) {
    if (stage != -1 && stage != 0 && stage != 11 && stage != 22) return;
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    std::vector<float> v(std::size_t(n) * 1024);
    INGINE_TEST_CUDA(cudaMemcpy(v.data(), h, v.size() * 4, cudaMemcpyDeviceToHost));
    states[stage] = std::move(v);
  };
  const auto start = std::chrono::steady_clock::now();
  const ingine::DeviceBuffer out = encoder.encode(rgb.data(), height, width);
  const double seconds = seconds_since(start);
  encoder.observer = nullptr;
  const std::vector<float> got = rows_of(out, tokens);

  const std::vector<std::int32_t> keep = read_tensor<std::int32_t>(golden, "rows");
  const std::vector<float> ref = read_tensor<float>(golden, "image_embeds");
  const std::vector<float> bf16 = read_tensor<float>(golden, "bf16_error");  // transformers BF16: rel rms, cosine
  std::vector<float> kept(ref.size());
  for (std::size_t r = 0; r < keep.size(); ++r) {
    std::memcpy(&kept[r * VISION_OUT], &got[std::size_t(keep[r]) * VISION_OUT], VISION_OUT * 4);
  }
  const Error e = error_of(kept.data(), ref.data(), ref.size());
  const bool ok = e.rel_rms <= bf16[0] && e.cosine >= bf16[1];
  std::printf("%s %-28s %5d tokens: rel-rms %.2e cos %.6f (transformers BF16: %.2e, %.6f)  %.0f ms\n",
              ok ? "PASS" : "FAIL", file.filename().c_str(), tokens, e.rel_rms, e.cosine, bf16[0], bf16[1],
              seconds * 1e3);
  failures += !ok;

  if (golden.contains("embeddings_in")) {  // FULL fixtures: the residual stream on the way
    const struct {
      const char* name;
      int stage;
      double max_rel_rms;
    } checks[] = {{"embeddings_in", -1, 2e-3}, {"layer_0", 0, 5e-3}, {"layer_11", 11, 1.5e-2}, {"layer_22", 22, 2e-2}};
    for (const auto& c : checks) {
      const std::vector<float> want = read_tensor<float>(golden, c.name);
      const std::vector<float>& have = states[c.stage];
      if (have.size() != want.size()) {
        expect(false, "hidden state size");
        continue;
      }
      const Error s = error_of(have.data(), want.data(), want.size());
      const bool good = s.rel_rms <= c.max_rel_rms;
      std::printf("  %s %-13s rel-rms %.2e cos %.7f (at most %.1e)\n", good ? "PASS" : "FAIL", c.name, s.rel_rms,
                  s.cosine, c.max_rel_rms);
      failures += !good;
    }
  }
}

// Attention over chunks of queries is the same computation: the residual stream after the first layer agrees
// to GEMM rounding (another shape, another summation order). The tower amplifies such rounding noise about
// a hundredfold by its last layer, so the outputs are compared through the FP32 reference: each stays as
// close to it as the other.
void test_chunking(VisionEncoder& encoder, cudaStream_t stream) {
  const ingine::SafetensorsFile golden(kGolden / "medium_720x680.safetensors");
  const std::vector<std::int32_t> keep = read_tensor<std::int32_t>(golden, "rows");
  const std::vector<float> ref = read_tensor<float>(golden, "image_embeds");
  const std::vector<float> bf16 = read_tensor<float>(golden, "bf16_error");
  const image::Image img = image::decode(read_file(kGolden / "fixtures" / "medium_720x680.png"));
  std::map<int, std::vector<float>> states;
  auto run = [&](std::size_t score_bytes) {
    encoder.set_score_bytes(score_bytes);
    states.clear();
    encoder.observer = [&](int stage, const float* h, int n) {
      if (stage != 0 && stage != 22) return;
      INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
      std::vector<float> v(std::size_t(n) * 1024);
      INGINE_TEST_CUDA(cudaMemcpy(v.data(), h, v.size() * 4, cudaMemcpyDeviceToHost));
      states[stage] = std::move(v);
    };
    std::vector<float> out = rows_of(encoder.encode(img.rgb.data(), img.height, img.width), img.tokens());
    encoder.observer = nullptr;
    return out;
  };
  auto against_reference = [&](const std::vector<float>& out) {
    std::vector<float> kept(ref.size());
    for (std::size_t r = 0; r < keep.size(); ++r) {
      std::memcpy(&kept[r * VISION_OUT], &out[std::size_t(keep[r]) * VISION_OUT], VISION_OUT * 4);
    }
    return error_of(kept.data(), ref.data(), ref.size());
  };
  const std::vector<float> whole = run(std::size_t(256) << 20);
  const std::map<int, std::vector<float>> whole_states = states;
  for (const int queries : {640, 128, 64}) {  // Sp = 1280: 2, 10 and 20 chunks
    const std::vector<float> chunked = run(std::size_t(1280) * 6 * queries);
    const Error first = error_of(states[0].data(), whole_states.at(0).data(), states[0].size());
    const Error last = error_of(states[22].data(), whole_states.at(22).data(), states[22].size());
    const Error out = error_of(chunked.data(), whole.data(), whole.size());
    const Error to_ref = against_reference(chunked);
    const bool ok = first.rel_rms <= 5e-4 && to_ref.rel_rms <= bf16[0] && to_ref.cosine >= bf16[1];
    std::printf("%s %3d queries per GEMM vs all: layer 0 %.1e, layer 22 %.1e, output %.1e; output vs reference %.2e\n",
                ok ? "PASS" : "FAIL", queries, first.rel_rms, last.rel_rms, out.rel_rms, to_ref.rel_rms);
    failures += !ok;
  }
  // the same image twice gives the same bits
  const std::vector<float> again = run(std::size_t(256) << 20);
  expect(again == whole, "the encoder is deterministic");
}

void timings(VisionEncoder& encoder) {
  Rng rng(7);
  for (const int side : {40, 320, 960, 1720, 2560}) {  // 1, 64, 576, 1849 and 4096 tokens
    std::vector<std::uint8_t> rgb(std::size_t(side) * side * 3);
    for (auto& v : rgb) v = std::uint8_t(rng.next() & 0xFF);
    std::size_t free_bytes = 0, total = 0;
    INGINE_TEST_CUDA(cudaMemGetInfo(&free_bytes, &total));
    const std::size_t need = encoder.workspace_bytes(side, side);
    if (free_bytes < need + (std::size_t(512) << 20)) {
      std::printf("info: %dx%d skipped (%zu MiB free, %zu MiB of workspace)\n", side, side, free_bytes >> 20, need >> 20);
      continue;
    }
    (void)encoder.encode(rgb.data(), side, side);  // first call of a shape picks its GEMM algorithms
    const auto start = std::chrono::steady_clock::now();
    (void)encoder.encode(rgb.data(), side, side);
    std::printf("info: %4dx%-4d %5d tokens: %7.1f ms, workspace %zu MiB\n", side, side, image::tokens(side, side),
                seconds_since(start) * 1e3, need >> 20);
  }
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  const fs::path model_dir = INGINE_MODEL_DIR;
  if (!fs::exists(model_dir / "model.safetensors.index.json")) {
    std::printf("skip: no model in %s\n", model_dir.c_str());
    return kSkip;
  }
  std::size_t free_bytes = 0, total = 0;
  INGINE_TEST_CUDA(cudaMemGetInfo(&free_bytes, &total));
  if (free_bytes < (std::size_t(2) << 30)) {
    std::printf("skip: %zu MiB of GPU memory free, 2048 needed\n", free_bytes >> 20);
    return kSkip;
  }
  cudaStream_t stream = nullptr;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  {
    const auto start = std::chrono::steady_clock::now();
    VisionEncoder encoder(model_dir, stream);
    std::printf("info: vision weights %zu MiB on the device, loaded in %.2f s\n", encoder.weight_bytes() >> 20,
                seconds_since(start));
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(kGolden / "fixtures")) files.push_back(entry.path());
    std::sort(files.begin(), files.end());
    for (const fs::path& file : files) test_fixture(encoder, stream, file);
    test_chunking(encoder, stream);
    timings(encoder);
  }
  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
  if (failures) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
