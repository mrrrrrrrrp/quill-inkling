// Image input on the full model, through the request parser and the Generator: the model names the colours
// of solid images (in the order they are given), reads rendered text, and the prompt caches treat image
// tokens like any others: an identical prompt and a follow-up turn reuse the state without encoding the
// image again, another image at the same place does not, a displaced conversation comes back from the park.
// An image longer than a prefill chunk gives the same continuation as in one chunk, with and without
// speculative decoding. Needs the model (~22 GB of GPU memory).
#include "common/test_util.h"
#include "ingine/generator.h"
#include "serve/openai.h"
#include "text/chat.h"

#include <httplib.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace ingine::model;
using Json = nlohmann::ordered_json;
namespace fs = std::filesystem;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
  std::fflush(stdout);
  failures += !ok;
}

std::string data_uri(const std::string& fixture) {
  const fs::path path = fs::path(INGINE_GOLDEN_DIR) / "vision" / "fixtures" / fixture;
  std::ifstream in(path, std::ios::binary);
  const std::string bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
  return "data:image/png;base64," + httplib::detail::base64_encode(bytes);
}

Json image(const std::string& fixture) { return {{"type", "image_url"}, {"image_url", {{"url", data_uri(fixture)}}}}; }
Json text(const std::string& t) { return {{"type", "text"}, {"text", t}}; }
Json user(Json content) { return {{"role", "user"}, {"content", std::move(content)}}; }

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

struct Out {
  std::vector<std::int32_t> tokens;
  std::string text;  // lower case
  GenerateStats stats;
  ingine::serve::ChatRequest request;
};

struct Session {
  const ingine::text::ChatFormat& fmt;
  Generator& gen;
  int max_ctx;

  Out run(const Json& messages, int max_new = 48) const {
    Out o;
    const Json body{{"messages", messages}, {"enable_thinking", false}, {"temperature", 0}, {"max_tokens", max_new}};
    o.request = ingine::serve::parse_request(body, fmt, max_ctx);
    Sampling greedy;
    greedy.temperature = 0.f;
    o.stats = gen.generate(Prompt{o.request.prompt_ids, o.request.images}, o.request.max_new_tokens, greedy,
                           fmt.stop_ids(), [&](std::span<const std::int32_t> t) {
                             o.tokens.insert(o.tokens.end(), t.begin(), t.end());
                             return true;
                           });
    o.text = lower(fmt.tokenizer().decode(o.tokens));
    return o;
  }
};

bool has(const Out& o, const char* word) { return o.text.find(word) != std::string::npos; }

std::string shown(const Out& o) {
  std::string s = o.text.substr(0, 160);
  std::replace(s.begin(), s.end(), '\n', ' ');
  return "\"" + s + "\"";
}

const char* kColour = "What colour is this image? Answer with one word.";

std::vector<std::int32_t> long_image_tokens;  // the continuation after the 1849-token image, 512-token chunks

void with_spec_and_small_chunks(const ingine::text::ChatFormat& fmt, const fs::path& dir) {
  RuntimeOptions opt;
  opt.model_dir = dir;
  opt.max_ctx = 16384;
  opt.dense_fp8 = opt.kv_fp8 = opt.spec = true;
  opt.prefill_chunk = 512;
  opt.park_mb = 2048;
  opt.vision = true;
  auto gen = make_generator(opt);
  expect(gen->supports_images(), "the generator takes images");
  const Session s{fmt, *gen, opt.max_ctx};

  // colours, and the caches around one image
  const Json red = Json::array({user(Json::array({image("solid_red_40x40.png"), text(kColour)}))});
  const Json blue = Json::array({user(Json::array({image("solid_blue_40x40.png"), text(kColour)}))});
  Out a = s.run(red);
  expect(has(a, "red"), "a red image is red: " + shown(a));
  expect(a.stats.reuse == Reuse::None && a.stats.images_encoded == 1 && a.stats.image_tokens == 1,
         "first request: nothing reused, one image of one token encoded");
  const int image_pos = a.request.images.at(0).pos;
  Out again = s.run(red);
  expect(again.stats.reuse != Reuse::None && again.stats.images_encoded == 0 &&
             again.stats.reused_tokens == int(again.request.prompt_ids.size()) && again.tokens == a.tokens,
         "the same request again: the whole prompt reused, no image encoded, the same answer");
  Out b = s.run(blue);
  expect(has(b, "blue"), "a blue image at the same place is blue: " + shown(b));
  expect(b.stats.reused_tokens <= image_pos && b.stats.images_encoded == 1,
         "another image at the same place: nothing reused from the image on (" + std::to_string(b.stats.reused_tokens) +
             " tokens reused, image at " + std::to_string(image_pos) + ")");

  // a follow-up turn extends the conversation: the image is not encoded again
  Out first = s.run(red);
  const Json follow = Json::array({user(Json::array({image("solid_red_40x40.png"), text(kColour)})),
                                   {{"role", "assistant"}, {"content", "Red"}},
                                   user("Which colour do you get when you mix it with white? Answer with one word.")});
  Out second = s.run(follow);
  expect(second.stats.reuse != Reuse::None && second.stats.images_encoded == 0 &&
             second.stats.reused_tokens > image_pos,
         "a follow-up turn reuses the image's part of the prompt (" + std::to_string(second.stats.reused_tokens) +
             " tokens, " + to_string(second.stats.reuse) + ")");
  expect(has(second, "pink"), "and still knows the image was red: " + shown(second));

  // two images are told apart, in order
  const char* both = "Name the colour of the first image, then the colour of the second image. Two words.";
  Out rb = s.run(Json::array({user(Json::array({image("solid_red_40x40.png"), image("solid_blue_40x40.png"), text(both)}))}));
  Out br = s.run(Json::array({user(Json::array({image("solid_blue_40x40.png"), image("solid_red_40x40.png"), text(both)}))}));
  expect(has(rb, "red") && has(rb, "blue") && rb.text.find("red") < rb.text.find("blue"), "red then blue: " + shown(rb));
  expect(has(br, "red") && has(br, "blue") && br.text.find("blue") < br.text.find("red"), "blue then red: " + shown(br));
  expect(rb.stats.image_tokens == 2 && rb.stats.images_encoded == 2, "two images encoded");

  // text in an image (16 tokens)
  Out ocr = s.run(Json::array({user(Json::array({image("text_inkling42_320x80.png"),
                                                 text("What is written in this image? Reply with the text only.")}))}));
  expect(has(ocr, "inkling") && has(ocr, "42"), "reads the text of an image: " + shown(ocr));

  // an image of 1849 tokens spans four 512-token chunks
  const Json big = Json::array({user(Json::array({text("Here is a picture."), image("big_1720x1720.png"),
                                                  text("Describe its colours in one sentence.")}))});
  Out wide = s.run(big, 24);
  expect(wide.stats.image_tokens == 1849 && wide.stats.images_encoded == 1 && wide.tokens.size() == 24,
         "an 1849-token image over four chunks: " + shown(wide) + " (vit " +
             std::to_string(int(wide.stats.vision_seconds * 1e3)) + " ms)");
  long_image_tokens = wide.tokens;

  // parking: the conversation with the 306-token image is displaced by the big one and comes back
  const Json medium = Json::array({user(Json::array({image("medium_720x680.png"), text("Describe its colours in one sentence.")}))});
  Out m1 = s.run(medium, 16);
  Out displaced = s.run(big, 4);
  Out m2 = s.run(medium, 16);
  expect(m1.stats.images_encoded == 1 && displaced.stats.images_encoded <= 1, "the conversations take turns");
  expect(m2.stats.reuse == Reuse::Parked && m2.stats.images_encoded == 0 && m2.tokens == m1.tokens,
         std::string("a parked conversation with an image is restored without encoding it (") +
             to_string(m2.stats.reuse) + ", " + std::to_string(m2.stats.reused_tokens) + " tokens reused)");
}

void plain_with_large_chunks(const ingine::text::ChatFormat& fmt, const fs::path& dir) {
  RuntimeOptions opt;
  opt.model_dir = dir;
  opt.max_ctx = 16384;
  opt.dense_fp8 = opt.kv_fp8 = true;
  opt.spec = false;
  opt.prefill_chunk = 3072;
  opt.park_mb = 0;
  opt.vision = true;
  auto gen = make_generator(opt);
  const Session s{fmt, *gen, opt.max_ctx};
  Out a = s.run(Json::array({user(Json::array({image("solid_red_40x40.png"), text(kColour)}))}));
  expect(has(a, "red"), "without speculative decoding a red image is red: " + shown(a));
  const Json big = Json::array({user(Json::array({text("Here is a picture."), image("big_1720x1720.png"),
                                                  text("Describe its colours in one sentence.")}))});
  Out one = s.run(big, 24);
  std::size_t same = 0;
  while (same < one.tokens.size() && same < long_image_tokens.size() && one.tokens[same] == long_image_tokens[same]) ++same;
  // chunk sizes differ by rounding only; a wrong row at a chunk boundary changes the description at once
  expect(same >= 12, "the 1849-token image in one chunk continues as in four (" + std::to_string(same) +
                         " of 24 tokens equal): " + shown(one));

  // a text-only generator rejects images instead of reading placeholder embeddings
  bool rejected = false;
  try {
    std::vector<std::int32_t> ids = a.request.prompt_ids;
    (void)gen->generate(Prompt{ids, {}}, 4, Sampling{}, {}, [](std::span<const std::int32_t>) { return true; });
  } catch (const std::invalid_argument&) { rejected = true; }
  expect(rejected, "image token ids without their images are rejected");
}

}  // namespace

int main() {
  if (ingine::test::skip_without_gpu()) return ingine::test::kSkip;
  const fs::path dir = INGINE_MODEL_DIR;
  if (!fs::exists(dir / "mtp.safetensors") || !fs::exists(dir / "chat_template.jinja")) {
    std::printf("skip: no model at %s\n", dir.c_str());
    return ingine::test::kSkip;
  }
  std::size_t free_bytes = 0, total = 0;
  INGINE_TEST_CUDA(cudaMemGetInfo(&free_bytes, &total));
  if (free_bytes < (std::size_t(24) << 30)) {
    std::printf("skip: %zu MiB of GPU memory free, 24576 needed (is a server running?)\n", free_bytes >> 20);
    return ingine::test::kSkip;
  }
  const ingine::text::ChatFormat fmt = ingine::text::ChatFormat::load(dir);
  with_spec_and_small_chunks(fmt, dir);
  plain_with_large_chunks(fmt, dir);
  if (failures) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::printf("ok\n");
  return 0;
}
