// ingine-serve: the OpenAI-compatible chat completions server (flags and defaults of `python -m ingine.server`).
#include "engine/fake_generator.h"
#include "engine/log.h"
#include "image/image.h"
#include "ingine/generator.h"
#include "model/vram_meter.h"
#include "serve/chat_server.h"
#include "serve/cli.h"
#include "text/chat.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

namespace {

std::atomic<bool> g_stop{false};

void handle_signal(int) { g_stop.store(true); }

// tokenizer_config.json "model_max_length" (the Python server's fallback context limit), else 1 << 20.
int tokenizer_max_length(const std::filesystem::path& model_dir) {
  std::ifstream in(model_dir / "tokenizer_config.json");
  if (!in) return 1 << 20;
  try {
    const auto cfg = nlohmann::json::parse(in);
    if (auto it = cfg.find("model_max_length"); it != cfg.end() && it->is_number()) {
      return int(std::min<double>(it->get<double>(), 2147483647.0));
    }
  } catch (const nlohmann::json::exception&) {
  }
  return 1 << 20;
}

// Sampling of requests that omit temperature / top_p / top_k: generation_config.json's values where present,
// else ingine's defaults (temperature 1, no top-p, no top-k).
ingine::model::Sampling generation_defaults(const std::filesystem::path& model_dir) {
  ingine::model::Sampling s;
  std::ifstream in(model_dir / "generation_config.json");
  if (!in) return s;
  try {
    const auto cfg = nlohmann::json::parse(in);
    if (auto it = cfg.find("temperature"); it != cfg.end() && it->is_number()) s.temperature = it->get<float>();
    if (auto it = cfg.find("top_p"); it != cfg.end() && it->is_number()) s.top_p = it->get<float>();
    if (auto it = cfg.find("top_k"); it != cfg.end() && it->is_number_integer()) s.top_k = std::max(0, it->get<int>());
  } catch (const nlohmann::json::exception&) {
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir, host = "127.0.0.1", ablation, backend = "fast", model_id, log_level = "INFO";
  int port = 8003, max_ctx = 65536, max_queue = 4, max_model_len = 0, park_mb = -1;
  bool engine = false, fake = false, dense_fp8 = false, kv_fp8 = false, spec = false, no_vision = false;
  bool vision_cpu = false;
  int vision_threads = 8;
  int image_max_tokens = 4096, max_images = 16, max_image_tokens = 32768;
  double spec_threshold = 0.8, fake_delay = 0.02;
  // sampling defaults; negative: from generation_config.json
  double temperature = -1, top_p = -1;
  int top_k = -1;

  ingine::serve::ArgParser ap("ingine-serve", "OpenAI-compatible chat completions server for ingine.");
  ap.option("--model-dir", "", "checkpoint dir (tokenizer + chat template)", &model_dir, {}, true);
  ap.option("--host", "", "", &host);
  ap.option("--port", "", "", &port);
  ap.flag("--engine", "serve the model with ingine", &engine);
  ap.flag("--fake", "serve canned outputs (FakeGenerator)", &fake);
  ap.exclusive({"--engine", "--fake"}, true);
  ap.option("--ablation", "", "ablation file (safetensors) applied at every visit", &ablation);
  ap.option("--max-ctx", "", "positions held by the full-attention caches", &max_ctx);
  ap.option("--backend", "{fast}", "decode step implementation", &backend, {"fast"});
  ap.flag("--dense-fp8", "FP8 E4M3 dense projections and head (row scales)", &dense_fp8);
  ap.flag("--spec", "DSpark speculative decoding (mtp.safetensors drafter)", &spec);
  ap.flag("--kv-fp8", "FP8 E4M3 full-attention KV caches (half the memory; needed for 262144 positions on 32 GB)",
          &kv_fp8);
  ap.option("--spec-threshold", "", "verify the leading drafts whose draft probability exceeds this (0 = always all 7)",
            &spec_threshold);
  ap.option("--park-mb", "",
            "device memory (MiB) for conversations displaced by another one, released when memory runs short, "
            "0 = off (default: min(3072, free memory after start-up - 4096))",
            &park_mb);
  ap.flag("--no-vision", "text only: do not load the vision tower (saves 0.6 GB) and reject requests with images",
          &no_vision);
  ap.flag("--vision-cpu", "run the image tower on the CPU (host memory, no GPU memory; images take longer)", &vision_cpu);
  ap.option("--vision-threads", "", "CPU threads of the image tower (with --vision-cpu)", &vision_threads);
  ap.option("--image-max-tokens", "",
            "tokens of one image (one per 40x40 pixels); a larger image is reduced to fit (at most 10609)",
            &image_max_tokens);
  ap.option("--max-images", "", "images accepted in one request", &max_images);
  ap.option("--max-image-tokens", "", "image tokens accepted in one request", &max_image_tokens);
  ap.option("--fake-delay", "", "seconds per fake token", &fake_delay);
  ap.option("--model-id", "", "model name reported by the API (default: generator.model_id)", &model_id);
  ap.option("--max-queue", "", "requests allowed to wait behind the running one", &max_queue);
  ap.option("--max-model-len", "",
            "context limit (default: generator.max_model_len or tokenizer model_max_length)", &max_model_len);
  ap.option("--log-level", "", "", &log_level);
  ap.option("--temperature", "", "temperature of requests that omit it (default: generation_config.json, else 1)",
            &temperature);
  ap.option("--top-p", "", "top_p of requests that omit it (default: generation_config.json, else 1)", &top_p);
  ap.option("--top-k", "", "top_k of requests that omit it, 0 = off (default: generation_config.json, else 0)",
            &top_k);
  // quill (12 GB GPUs)
  double vram_cap = 11.3;
  int vram_reserve = 512, cpu_threads = 7, prefill_chunk = 2048, staging = 24;
  bool no_kv_q4 = false, keep_mxfp8 = false, no_adapt = false;
  double skip_cold = 0;
  bool skip_no_renorm = false;
  std::string draft_dir, ui_file;
  ap.option("--draft-dir", "", "quill: DFlash2 drafter directory (with --spec)", &draft_dir);
  ap.option("--ui", "", "quill: HTML page served at / (the chat app)", &ui_file);
  ap.option("--vram-cap-gib", "", "quill: total device memory in use (every process) is kept below this", &vram_cap);
  ap.option("--vram-reserve-mib", "", "quill: device memory kept free after the expert pools", &vram_reserve);
  ap.option("--cpu-threads", "", "quill: CPU expert tier worker threads", &cpu_threads);
  ap.option("--prefill-chunk", "", "quill: tokens per prefill chunk", &prefill_chunk);
  ap.option("--staging", "", "quill: prefill expert staging slots", &staging);
  ap.flag("--no-kv-q4", "quill: BF16 full-attention caches instead of IQ4_NL q4", &no_kv_q4);
  ap.flag("--keep-mxfp8", "quill: keep the MXFP8 expert layers (default: re-encoded as NVFP4)", &keep_mxfp8);
  ap.flag("--no-adapt", "quill: static expert residency (loop layers first)", &no_adapt);
  std::string verify_widths;
  ap.option("--verify-widths", "", "quill: speculative verify widths, e.g. 2,4,8", &verify_widths);
  ap.option("--skip-cold", "", "quill, LOSSY: drop non-resident experts below this fraction of the routing weight", &skip_cold);
  ap.flag("--skip-no-renorm", "quill: with --skip-cold, do not renormalise the kept weights", &skip_no_renorm);
  ap.parse(argc, argv);
  if (image_max_tokens < 1 || image_max_tokens > ingine::image::MAX_TOKENS || max_images < 0 || max_image_tokens < 0) {
    std::cerr << "ingine-serve: error: --image-max-tokens must be in [1, " << ingine::image::MAX_TOKENS
              << "] and --max-images, --max-image-tokens at least 0\n";
    return 2;
  }
  if (temperature > 2 || top_p == 0 || top_p > 1) {
    std::cerr << "ingine-serve: error: --temperature must be in [0, 2] and --top-p in (0, 1]\n";
    return 2;
  }

  auto& log = ingine::engine::server_log();
  if (!ingine::engine::set_log_level(log_level)) {
    std::cerr << "ingine-serve: error: unknown level: '" << log_level << "'\n";
    return 2;
  }
#ifdef SIGPIPE
  std::signal(SIGPIPE, SIG_IGN);
#endif

  try {
    const ingine::text::ChatFormat fmt = ingine::text::ChatFormat::load(model_dir);
    std::unique_ptr<ingine::model::Generator> generator;
    if (fake) {
      generator = std::make_unique<ingine::engine::FakeGenerator>(
          fmt.tokenizer(), "Inkling-2-Mini-fake", tokenizer_max_length(model_dir), ingine::engine::FakeGenerator::Responder{},
          fake_delay);
    } else {
#ifdef INGINE_WITH_MODEL
      ingine::model::RuntimeOptions options;
      options.model_dir = model_dir;
      options.ablation = ablation;
      options.max_ctx = max_ctx;
      options.dense_fp8 = dense_fp8;
      options.kv_fp8 = kv_fp8;
      options.spec = spec;
      options.spec_threshold = float(spec_threshold);
      options.park_mb = park_mb;
      options.vision = !no_vision;
      options.vision_cpu = vision_cpu;
      options.vision_threads = vision_threads;
      options.vram_cap_gib = vram_cap;
      options.vram_reserve_mib = vram_reserve;
      options.cpu_threads = cpu_threads;
      options.prefill_chunk = prefill_chunk;
      options.expert_staging = staging;
      options.kv_q4 = !no_kv_q4;
      options.mxfp8_as_nvfp4 = !keep_mxfp8;
      options.adaptive_experts = !no_adapt;
      if (!verify_widths.empty()) {
        options.verify_widths.clear();
        for (std::size_t a = 0; a < verify_widths.size();) {
          std::size_t b = verify_widths.find(',', a);
          if (b == std::string::npos) b = verify_widths.size();
          options.verify_widths.push_back(std::stoi(verify_widths.substr(a, b - a)));
          a = b + 1;
        }
      }
      options.skip_cold = float(skip_cold);
      options.skip_renorm = !skip_no_renorm;
      options.draft_dir = draft_dir;
      generator = ingine::model::make_generator(options);
#else
      log.critical("--engine: this ingine-serve was built without ingine_model (src/model)");
      return 1;
#endif
    }

    ingine::serve::ServerOptions server_options;
    server_options.max_queue = max_queue;
    server_options.ui_file = ui_file;
    server_options.vram_used_mib = [] { return double(ingine::model::vram_used_bytes()) / 1048576.0; };
    server_options.max_model_len = max_model_len;
    server_options.model_id = model_id;
    server_options.sampling = generation_defaults(model_dir);
    if (temperature >= 0) server_options.sampling.temperature = float(temperature);
    if (top_p >= 0) server_options.sampling.top_p = float(top_p);
    if (top_k >= 0) server_options.sampling.top_k = top_k;
    server_options.images.enabled = !no_vision;
    server_options.images.limits.max_tokens = image_max_tokens;
    server_options.images.max_images = max_images;
    server_options.images.max_total_tokens = max_image_tokens;
    log.info("default sampling: temperature {} top_p {} top_k {}", server_options.sampling.temperature,
             server_options.sampling.top_p, server_options.sampling.top_k);
    ingine::serve::ChatServer server(*generator, fmt, server_options);
    if (!server.bind(host, port)) {
      log.critical("cannot bind {}:{}", host, port);
      return 1;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);
    log.info("image input: {}", !no_vision && generator->supports_images()
                                    ? fmt::format("on (at most {} tokens per image, {} images and {} image tokens per request)",
                                                  image_max_tokens, max_images, max_image_tokens)
                                    : std::string("off"));
    log.info("serving {} on http://{}:{}/v1", server.model_id(), host, server.port());
    bool listened = true;
    std::thread listener([&] {
      listened = server.listen();
      handle_signal(0);  // the accept loop ended on its own: wake the main thread
    });
    while (!g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    server.stop();
    listener.join();
    if (!listened) {
      log.critical("listen failed on {}:{}", host, server.port());
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    log.critical("{}", e.what());
    return 1;
  }
}
