// Image decoding and preprocessing against the reference processor's pixels (goldens written by
// tools/vision_ref.py from tools/vision_fixtures.py). CPU only.

#include "core/safetensors.h"
#include "image/image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using ingine::SafetensorsFile;
namespace image = ingine::image;

int failures = 0;

#define CHECK(cond)                                                                     \
  do {                                                                                  \
    if (!(cond)) {                                                                      \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);     \
      ++failures;                                                                       \
    }                                                                                   \
  } while (0)

bool rejects(const std::function<void()>& fn) {
  try {
    fn();
  } catch (const image::Error&) { return true; }
  return false;
}

const fs::path kGolden = fs::path(INGINE_GOLDEN_DIR) / "vision";

std::vector<std::uint8_t> read_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    std::exit(1);
  }
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

std::string to_base64(const std::vector<std::uint8_t>& bytes) {
  static const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  std::size_t i = 0;
  for (; i + 2 < bytes.size(); i += 3) {
    const unsigned v = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
    out += {alphabet[v >> 18], alphabet[(v >> 12) & 63], alphabet[(v >> 6) & 63], alphabet[v & 63]};
  }
  if (i + 1 == bytes.size()) {
    const unsigned v = bytes[i] << 16;
    out += {alphabet[v >> 18], alphabet[(v >> 12) & 63], '=', '='};
  } else if (i + 2 == bytes.size()) {
    const unsigned v = (bytes[i] << 16) | (bytes[i + 1] << 8);
    out += {alphabet[v >> 18], alphabet[(v >> 12) & 63], alphabet[(v >> 6) & 63], '='};
  }
  return out;
}

void test_base64() {
  CHECK(image::decode_base64("") == "");
  CHECK(image::decode_base64("Zg==") == "f");
  CHECK(image::decode_base64("Zm8=") == "fo");
  CHECK(image::decode_base64("Zm9v") == "foo");
  CHECK(image::decode_base64("Zm9vYmFy") == "foobar");
  CHECK(image::decode_base64("Zg") == "f");                // padding is optional
  CHECK(image::decode_base64("Zm9v\r\nYm Fy\t") == "foobar");  // whitespace is skipped
  CHECK(image::decode_base64("+/8=") == std::string("\xfb\xff", 2));
  CHECK(rejects([] { (void)image::decode_base64("Z"); }));      // one symbol cannot hold a byte
  CHECK(rejects([] { (void)image::decode_base64("Zm9v!"); }));
  CHECK(rejects([] { (void)image::decode_base64("Zg==Zg"); }));  // data after padding
  CHECK(rejects([] { (void)image::decode_base64("Zm-_"); }));    // the URL alphabet is not accepted
  std::vector<std::uint8_t> all(256);
  for (int i = 0; i < 256; ++i) all[std::size_t(i)] = std::uint8_t(i);
  const std::string round = image::decode_base64(to_base64(all));
  CHECK(round.size() == 256 && std::memcmp(round.data(), all.data(), 256) == 0);
}

void test_data_uri() {
  const std::string png = to_base64(read_file(kGolden / "fixtures" / "solid_red_40x40.png"));
  const image::Image ok = image::decode_data_uri("data:image/png;base64," + png);
  CHECK(ok.width == 40 && ok.height == 40 && ok.tokens() == 1);
  CHECK(image::decode_data_uri("DATA:image/jpeg;BASE64," + png).hash == ok.hash);  // the media type is ignored
  CHECK(image::decode_data_uri("data:;base64," + png).hash == ok.hash);
  CHECK(rejects([&] { (void)image::decode_data_uri("https://example.com/a.png"); }));
  CHECK(rejects([&] { (void)image::decode_data_uri("x"); }));
  CHECK(rejects([&] { (void)image::decode_data_uri("data:image/png;base64"); }));       // no comma
  CHECK(rejects([&] { (void)image::decode_data_uri("data:image/png," + png); }));       // not base64
  CHECK(rejects([&] { (void)image::decode_data_uri("data:image/png;base64,"); }));      // empty
  CHECK(rejects([&] { (void)image::decode_data_uri("data:image/png;base64,AAAAAAAA"); }));  // not an image
  CHECK(rejects([&] { (void)image::decode_data_uri("data:image/png;base64," + png.substr(0, 60)); }));  // truncated
}

// A PNG header that claims width x height; nothing after IHDR is needed to read the size.
std::vector<std::uint8_t> png_header(std::uint32_t width, std::uint32_t height) {
  std::vector<std::uint8_t> b = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 0, 0, 0, 13, 'I', 'H', 'D', 'R'};
  for (const std::uint32_t v : {width, height}) {
    for (int s = 24; s >= 0; s -= 8) b.push_back(std::uint8_t(v >> s));
  }
  b.insert(b.end(), {8, 2, 0, 0, 0, 0, 0, 0, 0});  // 8-bit RGB, then a (wrong, unchecked) CRC
  return b;
}

void test_limits() {
  CHECK(rejects([] { (void)image::decode(png_header(16000, 16000)); }));    // above max_pixels, before decoding
  CHECK(rejects([] { (void)image::decode(png_header(100000, 100000)); }));  // above the decoder's side limit
  image::Limits small;
  small.max_pixels = 1000;
  const std::vector<std::uint8_t> red = read_file(kGolden / "fixtures" / "solid_red_40x40.png");
  CHECK(rejects([&] { (void)image::decode(red, small); }));
  small = {};
  small.max_side = 39;
  CHECK(rejects([&] { (void)image::decode(red, small); }));
  CHECK(rejects([] { (void)image::decode({}); }));
}

void test_fit() {
  CHECK(image::tokens(40, 40) == 1 && image::tokens(41, 40) == 2 && image::tokens(1, 1) == 1);
  CHECK(image::tokens(4096, 4096) == image::MAX_TOKENS);
  CHECK(image::fit(100, 60, 4096) == std::pair(100, 60));  // fits: unchanged
  const int sizes[][2] = {{3840, 2160}, {2160, 3840}, {4096, 4096}, {16384, 40}, {40, 16384}, {5000, 3}, {8000, 6000}};
  for (const auto& s : sizes) {
    for (const int max : {1, 16, 256, 1024, 4096}) {
      const auto [h, w] = image::fit(s[0], s[1], max);
      if (image::tokens(s[0], s[1]) <= max) {
        CHECK(h == s[0] && w == s[1]);
        continue;
      }
      CHECK(h >= 1 && w >= 1 && h <= s[0] && w <= s[1]);
      CHECK(image::tokens(h, w) <= max);
      if (max >= 256 && std::min(s[0], s[1]) >= 2000) {  // tight, and the aspect ratio is kept
        CHECK(image::tokens(h, w) * 10 >= max * 9);
        CHECK(std::abs(double(h) / w - double(s[0]) / s[1]) < 0.01 * double(s[0]) / s[1]);
      }
    }
  }
}

void test_reduce() {
  // 4x2 -> 2x1: each output pixel is the mean of a 2x2 block
  const std::vector<std::uint8_t> src = {0,   0,   0,   10,  20,  30,  100, 100, 100, 200, 200, 200,
                                         20,  20,  20,  30,  40,  50,  100, 100, 100, 0,   0,   0};
  const std::vector<std::uint8_t> out = image::reduce(src, 2, 4, 1, 2);
  CHECK((out == std::vector<std::uint8_t>{15, 20, 25, 100, 100, 100}));
  // a constant image stays constant under a fractional reduction
  const std::vector<std::uint8_t> flat(std::size_t(97) * 131 * 3, 177);
  const std::vector<std::uint8_t> small = image::reduce(flat, 97, 131, 31, 50);
  CHECK(small.size() == std::size_t(31) * 50 * 3);
  CHECK(std::all_of(small.begin(), small.end(), [](std::uint8_t v) { return v == 177; }));
  // 3 -> 2 along x: cells cover [0, 1.5) and [1.5, 3)
  const std::vector<std::uint8_t> row = {0, 0, 0, 90, 90, 90, 255, 255, 255};
  CHECK((image::reduce(row, 1, 3, 1, 2) == std::vector<std::uint8_t>{30, 30, 30, 200, 200, 200}));

  // an image above the token limit is reduced to fit, one below is untouched
  const std::vector<std::uint8_t> medium = read_file(kGolden / "fixtures" / "medium_720x680.png");
  image::Limits limits;
  limits.max_tokens = 64;
  const image::Image reduced = image::decode(medium, limits);
  CHECK(reduced.source_width == 720 && reduced.source_height == 680);
  CHECK(reduced.tokens() <= 64 && reduced.tokens() >= 56 && reduced.width < 720);
  CHECK(reduced.rgb.size() == std::size_t(reduced.width) * reduced.height * 3);
  const image::Image whole = image::decode(medium);
  CHECK(whole.width == 720 && whole.height == 680 && whole.tokens() == 17 * 18 && whole.hash != reduced.hash);
}

// The decoded pixels are the reference processor's, exactly (PNG, GIF, WebP: the reference decodes WebP with
// libwebp too) or within the decoder's tolerance (JPEG).
void test_reference_pixels() {
  int checked = 0;
  for (const auto& entry : fs::directory_iterator(kGolden / "fixtures")) {
    const fs::path file = entry.path();
    const fs::path golden_path = kGolden / (file.stem().string() + ".safetensors");
    if (!fs::exists(golden_path)) continue;
    const SafetensorsFile golden(golden_path);
    const image::Image img = image::decode(read_file(file));
    const std::vector<std::byte> grid_bytes = golden.read("grid");
    std::int32_t grid[2];
    std::memcpy(grid, grid_bytes.data(), sizeof grid);
    CHECK(img.rows() == grid[0] && img.cols() == grid[1]);
    if (!golden.contains("rgb")) continue;
    const auto& t = golden.at("rgb");
    CHECK(t.shape.size() == 3 && t.shape[0] == img.height && t.shape[1] == img.width && t.shape[2] == 3);
    const std::vector<std::byte> ref = golden.read("rgb");
    if (ref.size() != img.rgb.size()) {
      CHECK(ref.size() == img.rgb.size());
      continue;
    }
    int worst = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      worst = std::max(worst, std::abs(int(img.rgb[i]) - int(std::to_integer<std::uint8_t>(ref[i]))));
    }
    // stb_image and libjpeg-turbo (the reference's decoder) round the inverse DCT differently
    const int allowed = file.extension() == ".jpg" ? 3 : 0;
    std::printf("%-28s %4dx%-4d %2dx%-2d tokens, max pixel difference %d (allowed %d)\n", file.filename().c_str(),
                img.width, img.height, img.rows(), img.cols(), worst, allowed);
    CHECK(worst <= allowed);
    ++checked;
  }
  CHECK(checked >= 13);
}

void test_webp() {
  const fs::path fixtures = kGolden / "fixtures";
  const image::Image lossy = image::decode(read_file(fixtures / "lossy_160x120.webp"));
  CHECK(lossy.width == 160 && lossy.height == 120 && lossy.tokens() == 12);
  const image::Image alpha = image::decode(read_file(fixtures / "lossless_alpha_72x56.webp"));
  CHECK(alpha.width == 72 && alpha.height == 56);
  // the top row is fully transparent: the checkerboard shows (204 at the origin, 153 eight pixels on)
  CHECK(alpha.rgb[0] == 204 && alpha.rgb[1] == 204 && alpha.rgb[2] == 204 && alpha.rgb[8 * 3] == 153);
  const image::Image frame = image::decode(read_file(fixtures / "animated_80x80.webp"));  // the first frame
  CHECK(frame.width == 80 && frame.height == 80);
  // the media type of a data: URI is ignored, also for WebP
  CHECK(image::decode_data_uri("data:image/png;base64," + to_base64(read_file(fixtures / "lossy_160x120.webp"))).hash ==
        lossy.hash);
  // truncated and corrupt files are rejected, and sizes are checked before decoding
  std::vector<std::uint8_t> bytes = read_file(fixtures / "lossy_160x120.webp");
  std::vector<std::uint8_t> cut(bytes.begin(), bytes.begin() + std::ptrdiff_t(bytes.size() / 2));
  CHECK(rejects([&] { (void)image::decode(cut); }));
  CHECK(rejects([&] { (void)image::decode(std::vector<std::uint8_t>{'R', 'I', 'F', 'F', 4, 0, 0, 0, 'W', 'E', 'B', 'P'}); }));
  image::Limits small;
  small.max_side = 100;
  CHECK(rejects([&] { (void)image::decode(bytes, small); }));
  small = {};
  small.max_tokens = 4;
  const image::Image reduced = image::decode(bytes, small);
  CHECK(reduced.tokens() <= 4 && reduced.source_width == 160);
}

void test_identity() {
  const std::vector<std::uint8_t> a(std::size_t(12) * 3, 7);
  std::vector<std::uint8_t> b = a;
  b[17] = 8;
  const std::uint64_t ha = image::hash_pixels(4, 3, a);
  CHECK(ha == image::hash_pixels(4, 3, a));
  CHECK(ha != image::hash_pixels(4, 3, b));
  CHECK(ha != image::hash_pixels(3, 4, a));  // the same bytes in another shape are another image
  CHECK(ha != image::hash_pixels(6, 2, a));
  std::set<std::int32_t> ids;
  for (int k = 0; k < 4096; ++k) {
    const std::int32_t id = image::token_id(ha, k);
    CHECK(id < 0);
    CHECK(id == image::token_id(ha, k));
    ids.insert(id);
    CHECK(id != image::token_id(image::hash_pixels(4, 3, b), k));
  }
  CHECK(ids.size() >= 4095);  // tokens of one image differ from each other
  // the encoding is part of the prompt-cache identity: pin it
  CHECK(image::hash_pixels(2, 1, std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6}) == 0x4b77e93260b6a733ull);
  CHECK(image::token_id(0x0123456789abcdefull, 0) == -1193066522);
  // identical pixels from different containers are the same image
  const image::Image red = image::decode(read_file(kGolden / "fixtures" / "solid_red_40x40.png"));
  const image::Image blue = image::decode(read_file(kGolden / "fixtures" / "solid_blue_40x40.png"));
  CHECK(red.hash != blue.hash && red.hash == image::hash_pixels(40, 40, red.rgb));
}

}  // namespace

int main() {
  test_base64();
  test_data_uri();
  test_limits();
  test_fit();
  test_reduce();
  test_reference_pixels();
  test_webp();
  test_identity();
  if (failures) {
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
