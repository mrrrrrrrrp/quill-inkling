// ingine-bench: end-to-end speed, prefill tokens/s and decode tokens/s for one request (port of
// engine/bench/bench_e2e.py; same flags, lines and JSON).
//
//     ingine-bench --tokens 512 --dense-fp8
//     ingine-bench --spec --forced agentic.jsonl   (teacher-forced replay, see forced.h)
//     ingine-bench --spec --forced agentic.jsonl --forced-free 1 --temperature 1   (free generation from each
//         item's prompt for len(output) tokens at the temperature: sampled A/B, where forced acceptance by
//         equality under-counts)
#include "forced.h"
#include "ingine/generator.h"
#include "model/cpu_tier.h"
#include "serve/cli.h"
#include "text/chat.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <utf8proc/utf8proc.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

const std::vector<std::pair<std::string, std::string>> kPrompts = {
    {"chat", "Explain how a hash map handles collisions, with a short Python example."},
    {"code", "Write a Python module implementing an LRU cache with O(1) get/put, type hints and unit tests."},
};

// Python repr(float).
std::string py_float(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
  char buf[64];
  auto [end, ec] = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
  std::string s(buf, end);  // [-]d[.ddd]e[+-]XX, shortest round-trip digits
  std::string sign;
  if (s[0] == '-') {
    sign = "-";
    s.erase(0, 1);
  }
  const size_t e = s.find('e');
  const int exp = std::stoi(s.substr(e + 1));
  std::string digits = s.substr(0, e);
  digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
  if (exp < -4 || exp >= 16) {
    std::string mant = digits.substr(0, 1);
    if (digits.size() > 1) mant += "." + digits.substr(1);
    char ebuf[16];
    std::snprintf(ebuf, sizeof ebuf, "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
    return sign + mant + ebuf;
  }
  std::string out;
  if (exp < 0) {
    out = "0." + std::string(size_t(-exp - 1), '0') + digits;
  } else if (size_t(exp) + 1 >= digits.size()) {
    out = digits + std::string(size_t(exp) + 1 - digits.size(), '0') + ".0";
  } else {
    out = digits.substr(0, size_t(exp) + 1) + "." + digits.substr(size_t(exp) + 1);
  }
  return sign + out;
}

// Python repr(str) of (possibly partial) UTF-8 text; invalid bytes decode to U+FFFD as in tokenizer.decode.
std::string py_repr(const std::string& text) {
  std::vector<int32_t> cps;
  const auto* p = reinterpret_cast<const utf8proc_uint8_t*>(text.data());
  utf8proc_ssize_t left = utf8proc_ssize_t(text.size());
  while (left > 0) {
    utf8proc_int32_t cp;
    utf8proc_ssize_t n = utf8proc_iterate(p, left, &cp);
    if (n <= 0) {
      cp = 0xFFFD;
      n = 1;
    }
    cps.push_back(cp);
    p += n;
    left -= n;
  }
  const bool has_single = std::find(cps.begin(), cps.end(), '\'') != cps.end();
  const bool has_double = std::find(cps.begin(), cps.end(), '"') != cps.end();
  const char quote = has_single && !has_double ? '"' : '\'';
  std::string out(1, quote);
  for (int32_t cp : cps) {
    if (cp == quote || cp == '\\') {
      out += '\\';
      out += char(cp);
    } else if (cp == '\n') {
      out += "\\n";
    } else if (cp == '\r') {
      out += "\\r";
    } else if (cp == '\t') {
      out += "\\t";
    } else {
      const utf8proc_category_t cat = utf8proc_category(cp);
      const bool printable =
          cp == ' ' || !(cat == UTF8PROC_CATEGORY_CC || cat == UTF8PROC_CATEGORY_CF || cat == UTF8PROC_CATEGORY_CS ||
                         cat == UTF8PROC_CATEGORY_CO || cat == UTF8PROC_CATEGORY_CN || cat == UTF8PROC_CATEGORY_ZL ||
                         cat == UTF8PROC_CATEGORY_ZP || cat == UTF8PROC_CATEGORY_ZS);
      if (printable) {
        utf8proc_uint8_t enc[4];
        const utf8proc_ssize_t n = utf8proc_encode_char(cp, enc);
        out.append(reinterpret_cast<const char*>(enc), size_t(n));
      } else {
        char buf[16];
        if (cp < 0x100) std::snprintf(buf, sizeof buf, "\\x%02x", cp);
        else if (cp < 0x10000) std::snprintf(buf, sizeof buf, "\\u%04x", cp);
        else std::snprintf(buf, sizeof buf, "\\U%08x", cp);
        out += buf;
      }
    }
  }
  out += quote;
  return out;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

std::string default_model_dir() {
  if (const char* env = std::getenv("INGINE_MODEL_DIR")) return env;
  return std::string(INGINE_REPO_DIR) + "/models/Inkling-2-Mini";
}

}  // namespace

int main(int argc, char** argv) {
  spdlog::flush_on(spdlog::level::info);
  std::string model_dir = default_model_dir(), backend = "fast", ablation, effort = "low", forced, forced_trace;
  int profile_steps = 0;
  int max_ctx = 32768, tokens = 512, runs = 3, long_prompt = 0, forced_items = 0, forced_skip = 0, forced_free = -1;
  bool dense_fp8 = false, kv_fp8 = false, spec = false;
  double spec_threshold = 0.8, temperature = 0.0;

  ingine::serve::ArgParser ap("ingine-bench",
                              "End-to-end ingine speed: prefill tokens/s and greedy decode tokens/s for one request.");
  ap.option("--model-dir", "", "", &model_dir);
  ap.option("--backend", "{fast}", "", &backend, {"fast"});
  ap.option("--ablation", "", "", &ablation);
  ap.option("--max-ctx", "", "", &max_ctx);
  ap.option("--tokens", "", "", &tokens);
  ap.option("--runs", "", "", &runs);
  ap.option("--effort", "", "", &effort);
  ap.flag("--dense-fp8", "", &dense_fp8);
  ap.flag("--kv-fp8", "FP8 full-attention KV caches", &kv_fp8);
  ap.flag("--spec", "DSpark speculative decoding", &spec);
  double vram_cap = 11.3;
  int cpu_threads = 7, prefill_chunk = 2048, vram_reserve = 512, staging = 24;
  bool no_kv_q4 = false, nv_only = false, no_adapt = false;
  double skip_cold = 0;
  bool skip_no_renorm = false;
  std::string profile_in, profile_out, draft_dir;
  ap.option("--draft-dir", "", "quill: DFlash2 drafter directory (with --spec)", &draft_dir);
  ap.option("--expert-profile", "", "quill: resident experts by these routing counts", &profile_in);
  ap.option("--save-profile", "", "quill: write the decode routing counts of this run", &profile_out);
  ap.flag("--nvfp4-only", "quill: re-encode MXFP8 experts as NVFP4", &nv_only);
  ap.flag("--no-adapt", "quill: static expert residency", &no_adapt);
  int adapt_swaps = -1, adapt_interval = -1;
  ap.option("--adapt-swaps", "", "quill: residency swaps per interval", &adapt_swaps);
  ap.option("--adapt-interval", "", "quill: decode tokens per residency interval", &adapt_interval);
  std::string verify_widths;
  ap.option("--verify-widths", "", "quill: speculative verify widths, e.g. 2,4,8", &verify_widths);
  ap.option("--skip-cold", "", "quill, LOSSY: drop non-resident experts below this fraction of the routing weight", &skip_cold);
  ap.flag("--skip-no-renorm", "quill: with --skip-cold, do not renormalise the kept weights", &skip_no_renorm);
  ap.option("--vram-cap-gib", "", "quill: total device memory cap (all processes)", &vram_cap);
  ap.option("--vram-reserve-mib", "", "quill: device memory kept for allocations after the expert pools", &vram_reserve);
  ap.option("--cpu-threads", "", "quill: CPU tier worker threads", &cpu_threads);
  ap.option("--prefill-chunk", "", "quill: tokens per prefill chunk", &prefill_chunk);
  ap.option("--staging", "", "quill: prefill expert staging slots per format", &staging);
  ap.flag("--no-kv-q4", "quill: BF16 full-attention caches", &no_kv_q4);
  ap.option("--spec-threshold", "", "verify the leading drafts whose draft probability exceeds this (0 = always all 7)",
            &spec_threshold);
  ap.option("--temperature", "", "", &temperature);
  ap.option("--long-prompt", "", "also time prefill of a synthetic prompt of N tokens", &long_prompt);
  ap.option("--profile-steps", "", "quill: after the runs, time N launches of each decode step graph width (1,2,4,8)", &profile_steps);
  ap.option("--forced", "", "teacher-forced replay of the outputs in FILE (JSON lines {name, prompt, output})",
            &forced);
  ap.option("--forced-items", "", "replay at most N items of --forced", &forced_items);
  ap.option("--forced-skip", "", "skip the first N items of --forced", &forced_skip);
  ap.option("--forced-trace", "", "append every replayed item and round to FILE (JSON lines)", &forced_trace);
  ap.option("--forced-free", "", "with --forced: sample freely from each item's prompt with this seed (no replay)", &forced_free);
  std::string record, record_prompts;
  ap.option("--record", "", "quill: write a --forced replay set to FILE (see --record-prompts)", &record);
  ap.option("--record-prompts", "", "quill: lines name<TAB>message to answer for --record", &record_prompts);
  ap.parse(argc, argv);

  try {
    using Clock = std::chrono::steady_clock;
    cudaFree(nullptr);  // create the context before measuring the model's memory
    size_t free0 = 0, total = 0;
    cudaMemGetInfo(&free0, &total);
    const auto t0 = Clock::now();
    ingine::model::RuntimeOptions options;
    options.model_dir = model_dir;
    options.ablation = ablation;
    options.max_ctx = max_ctx;
    options.dense_fp8 = dense_fp8;
    options.kv_fp8 = kv_fp8;
    options.spec = spec;
    options.spec_threshold = float(spec_threshold);
    options.vram_cap_gib = vram_cap;
    options.vram_reserve_mib = vram_reserve;
    options.cpu_threads = cpu_threads;
    options.prefill_chunk = prefill_chunk;
    options.expert_staging = staging;
    options.kv_q4 = !no_kv_q4;
    options.mxfp8_as_nvfp4 = nv_only;
    options.adaptive_experts = !no_adapt;
    if (adapt_swaps >= 0) options.adapt_swaps = adapt_swaps;
    if (adapt_interval > 0) options.adapt_interval = adapt_interval;
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
    options.expert_profile = profile_in;
    options.draft_dir = draft_dir;
    auto gen = ingine::model::make_generator(options);
    const double load_s = std::chrono::duration<double>(Clock::now() - t0).count();
    size_t free1 = 0;
    cudaMemGetInfo(&free1, &total);
    std::printf("load+capture %.1fs, GPU %.2f GB\n", load_s, double(free0 > free1 ? free0 - free1 : 0) / 1e9);
    std::fflush(stdout);
    if (!forced.empty()) {
      std::printf("%s\n", ingine::bench::run_forced(*gen, forced, temperature, forced_skip, forced_items, forced_trace, forced_free).c_str());
      return 0;
    }

    const auto fmt = ingine::text::ChatFormat::load(model_dir);
    const auto& tok = fmt.tokenizer();
    const std::vector<int32_t> stop = {*tok.token_id("<|content_model_end_sampling|>")};

    // quill: record a --forced replay set: each line of --record-prompts is "name<TAB>user message"; the model
    // answers at --temperature (seed 1000 + item) for up to --tokens tokens; JSON lines {name, prompt, output}
    if (!record.empty()) {
      std::ifstream pin(record_prompts);
      if (!pin) throw std::runtime_error("cannot read --record-prompts " + record_prompts);
      std::ofstream rout(record, std::ios::binary);
      std::string line;
      int item = 0;
      while (std::getline(pin, line)) {
        const std::size_t tab = line.find('\t');
        if (line.empty() || tab == std::string::npos) continue;
        ingine::text::ChatOptions chat;
        if (effort != "default") chat.reasoning_effort = effort;
        const nlohmann::ordered_json messages =
            nlohmann::ordered_json::array({{{"role", "user"}, {"content", line.substr(tab + 1)}}});
        const std::vector<int32_t> ids = fmt.render(messages, nlohmann::ordered_json(), chat).ids;
        ingine::model::Sampling sampling;
        sampling.temperature = float(temperature);
        sampling.seed = uint64_t(1000 + item);
        std::vector<int32_t> out;
        const auto st = gen->generate(ids, tokens, sampling, std::span<const int32_t>(stop), [&](std::span<const int32_t> t) {
          out.insert(out.end(), t.begin(), t.end());
          return true;
        });
        rout << nlohmann::json{{"name", line.substr(0, tab)}, {"prompt", ids}, {"output", out}}.dump() << "\n";
        rout.flush();
        std::printf("recorded %s: %zu prompt + %zu output tokens, %.1f tok/s\n", line.substr(0, tab).c_str(), ids.size(),
                    out.size(), (st.generated_tokens - 1) / std::max(st.decode_seconds, 1e-9));
        std::fflush(stdout);
        ++item;
      }
      return 0;
    }

    gen->reset_route_counts();
    std::string json = "{";
    for (const auto& [name, prompt] : kPrompts) {
      ingine::text::ChatOptions chat;
      chat.reasoning_effort = effort;
      const nlohmann::ordered_json messages = nlohmann::ordered_json::array({{{"role", "user"}, {"content", prompt}}});
      const std::vector<int32_t> ids = fmt.render(messages, nlohmann::ordered_json(), chat).ids;
      std::vector<double> rates;
      std::vector<int32_t> first_out;  // run 0 (greedy check of the later runs)
      ingine::model::GenerateStats st;
      for (int r = 0; r < runs; ++r) {
        std::vector<int32_t> out;
        ingine::model::cpu_tier_reset_stats();
        ingine::model::Sampling sampling;
        sampling.temperature = float(temperature);
        sampling.seed = uint64_t(r);
        // every run stops at the end-of-turn token: tokens after it are not what a request generates (and
        // speculative decoding accepts few drafts there, which made later runs look slow)
        st = gen->generate(ids, tokens, sampling, std::span<const int32_t>(stop),
                           [&](std::span<const int32_t> t) {
                             out.insert(out.end(), t.begin(), t.end());
                             return true;
                           });
        rates.push_back((st.generated_tokens - 1) / st.decode_seconds);
        {
          const auto cs = ingine::model::cpu_tier_stats();
          const double n = std::max(1, st.generated_tokens);
          std::printf("[cpu tier] %.1f calls/token, %.1f experts/token, busy %.2f ms/token of %.2f ms/token (prefill %.2fs, %d prompt tokens)"
                      " | %.1f unique/token, %.1f GB/s while busy\n",
                      cs.calls / n, cs.entries / n, cs.busy_s * 1e3 / n, st.decode_seconds * 1e3 / n, st.prefill_seconds,
                      st.prompt_tokens, cs.uniques / n, cs.busy_s > 0 ? cs.uniques * 4.5e6 / cs.busy_s / 1e9 : 0.0);
        }
        if (temperature <= 0 && r == 0) first_out = out;
        if (temperature <= 0 && r > 0) {  // later runs restore the prompt checkpoint: greedy must match run 0
          std::size_t k = 0;
          while (k < out.size() && k < first_out.size() && out[k] == first_out[k]) ++k;
          std::printf("[greedy] run %d matches run 0 for %zu of %zu tokens\n", r, k, std::min(out.size(), first_out.size()));
        }
        if (r == 0) {
          out.resize(std::min<size_t>(out.size(), 120));
          std::printf("--- %s: %s\n", name.c_str(), py_repr(tok.decode_lossy(out)).c_str());
          std::fflush(stdout);
        }
      }
      const double med = median(rates);
      std::string runs_json, runs_text;
      for (double x : rates) {
        runs_json += (runs_json.empty() ? "" : ", ") + py_float(x);
        char buf[64];
        std::snprintf(buf, sizeof buf, "%s%.1f", runs_text.empty() ? "" : ", ", x);
        runs_text += buf;
      }
      const std::string spec_json =
          spec ? "{\"rounds\": " + std::to_string(st.spec_rounds) + ", \"accepted\": " + std::to_string(st.spec_accepted) + "}"
               : "{}";
      if (json.size() > 1) json += ", ";
      json += "\"" + name + "\": {\"decode_tok_s\": " + py_float(med) + ", \"runs\": [" + runs_json +
              "], \"spec\": " + spec_json + "}";
      std::string spec_text;
      if (spec && st.spec_rounds) {
        char buf[96];
        std::snprintf(buf, sizeof buf, ", %.2f tokens/verify (last run)",
                      double(st.spec_accepted + st.spec_rounds) / st.spec_rounds);
        spec_text = buf;
      }
      std::printf("%s: decode %.1f tok/s (runs %s)%s\n", name.c_str(), med, runs_text.c_str(), spec_text.c_str());
      std::fflush(stdout);
    }
    if (long_prompt) {
      std::ifstream in(__FILE__, std::ios::binary);
      std::stringstream source;
      source << in.rdbuf();
      const std::vector<int32_t> once = tok.encode(source.str(), /*special=*/true);
      std::vector<int32_t> ids;
      for (int i = 0; i < 1000 && int(ids.size()) < long_prompt; ++i) ids.insert(ids.end(), once.begin(), once.end());
      ids.resize(std::min<size_t>(ids.size(), size_t(long_prompt)));
      ingine::model::Sampling greedy;
      greedy.temperature = 0.0f;
      std::vector<int32_t> cont;  // a greedy continuation: equal across lossless prefill changes
      const ingine::model::GenerateStats st = gen->generate(ids, 16, greedy, {}, [&](std::span<const int32_t> t) {
        cont.insert(cont.end(), t.begin(), t.end());
        return true;
      });
      std::string cs;
      for (int32_t t : cont) cs += std::to_string(t) + " ";
      std::printf("prefill continuation: %s\n", cs.c_str());
      const double tok_s = double(ids.size()) / st.prefill_seconds;
      json += ", \"prefill\": {\"tokens\": " + std::to_string(ids.size()) + ", \"tok_s\": " + py_float(tok_s) + "}";
      std::printf("prefill %zu tokens: %.0f tok/s\n", ids.size(), tok_s);
      std::fflush(stdout);
    }
    if (profile_steps > 0)  // quill: after the long prompt (if any): steps at the end of its context
      for (int m : {1, 2, 4, 8}) std::printf("profile step width %d: %.3f ms\n", m, gen->profile_steps(m, profile_steps));
    std::printf("%s}\n", json.c_str());
    if (!profile_out.empty()) {
      const auto c = gen->route_counts();
      std::FILE* f = std::fopen(profile_out.c_str(), "wb");
      if (f) std::fwrite(c.data(), 4, c.size(), f), std::fclose(f);
      std::printf("routing profile written: %s\n", profile_out.c_str());
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ingine-bench: %s\n", e.what());
    return 1;
  }
}
