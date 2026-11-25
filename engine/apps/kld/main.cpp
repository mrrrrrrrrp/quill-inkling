// quill-kld: teacher-forced accuracy of a configuration against a saved reference.
//
//   quill-kld --model-dir M --text data.txt --seqs 2 --ctx 1024 --save ref.bin [config flags]     (reference)
//   quill-kld --model-dir M --text data.txt --seqs 2 --ctx 1024 --ref ref.bin [config flags]      (test)
//
// Each sequence is `ctx` consecutive tokens of the text (non-overlapping windows from the start), run one token
// at a time through the width-1 decode step (the path generation uses). The reference stores every position's
// full log-softmax as fp16; a test run reports mean KL(ref || test), top-1 agreement with the reference, and the
// perplexity of both on the text.
#include "ingine/generator.h"
#include "model/runtime.h"
#include "model/cpu_tier.h"
#include "serve/cli.h"
#include "text/tokenizer.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr int V = ingine::model::UNPADDED_VOCAB;

void log_softmax(const float* x, std::vector<double>& out) {
  double m = -1e300;
  for (int i = 0; i < V; ++i) m = std::max(m, double(x[i]));
  double s = 0;
  for (int i = 0; i < V; ++i) s += std::exp(double(x[i]) - m);
  const double lse = m + std::log(s);
  out.resize(V);
  for (int i = 0; i < V; ++i) out[i] = double(x[i]) - lse;
}
}  // namespace

int main(int argc, char** argv) {
  spdlog::flush_on(spdlog::level::info);
  std::string model_dir, ablation, text_path, save, ref, profile_out;
  int seqs = 2, ctx = 1024, max_ctx = 8192, cpu_threads = 7, staging = 8;
  double vram_cap = 11.3;
  bool dense_bf16 = false, kv_bf16 = false, nv_only = false;
  double skip_cold = 0;
  bool skip_no_renorm = false;
  ingine::serve::ArgParser ap("quill-kld", "teacher-forced KL divergence against a saved reference");
  ap.option("--model-dir", "", "", &model_dir);
  ap.option("--ablation", "", "", &ablation);
  ap.option("--text", "", "", &text_path);
  ap.option("--save", "", "write the reference log-probs", &save);
  ap.option("--ref", "", "compare against this reference", &ref);
  ap.option("--save-profile", "", "write the routing counts of the run", &profile_out);
  ap.option("--seqs", "", "", &seqs);
  ap.option("--ctx", "", "", &ctx);
  ap.option("--max-ctx", "", "", &max_ctx);
  ap.option("--cpu-threads", "", "", &cpu_threads);
  ap.option("--staging", "", "", &staging);
  ap.option("--vram-cap-gib", "", "", &vram_cap);
  ap.flag("--dense-bf16", "BF16 dense weights (reference)", &dense_bf16);
  ap.flag("--kv-bf16", "BF16 full-attention KV (reference)", &kv_bf16);
  ap.flag("--nvfp4-only", "re-encode MXFP8 experts as NVFP4", &nv_only);
  ap.option("--skip-cold", "", "quill, LOSSY: drop non-resident experts below this fraction of the routing weight", &skip_cold);
  ap.flag("--skip-no-renorm", "quill: with --skip-cold, do not renormalise the kept weights", &skip_no_renorm);
  ap.parse(argc, argv);
  if (save.empty() == ref.empty()) {
    std::fprintf(stderr, "quill-kld: give exactly one of --save / --ref\n");
    return 2;
  }
  try {
    std::ifstream tf(text_path, std::ios::binary);
    std::stringstream ss;
    ss << tf.rdbuf();
    const auto tok = ingine::text::Tokenizer::load(model_dir);
    const std::vector<int32_t> ids = tok.encode(ss.str(), false);
    if (std::int64_t(ids.size()) < std::int64_t(seqs) * ctx) throw std::runtime_error("text too short");

    ingine::model::RuntimeOptions o;
    o.model_dir = model_dir;
    o.ablation = ablation;
    o.max_ctx = std::max(max_ctx, ctx + 16);
    o.dense_fp8 = !dense_bf16;
    o.kv_q4 = !kv_bf16;
    o.cpu_threads = cpu_threads;
    o.expert_staging = staging;
    o.vram_cap_gib = vram_cap;
    o.mxfp8_as_nvfp4 = nv_only;
    o.skip_cold = float(skip_cold);
    o.skip_renorm = !skip_no_renorm;
    ingine::model::Runtime rt(o);
    rt.finalize(std::size_t(o.vram_reserve_mib) << 20);
    // workers spin while this tool decodes (an idle tier polls with OS sleeps: ~1-15 ms per call on Windows)
    rt.cpu_tier()->activate();

    std::ofstream fo;
    std::ifstream fi;
    const bool saving = !save.empty();
    if (saving) fo.open(save, std::ios::binary);
    else fi.open(ref, std::ios::binary);
    std::vector<float> logits(V);
    std::vector<double> lp, lr;
    std::vector<__half> buf(V);
    double kl_sum = 0, nll = 0, nll_ref = 0;
    std::int64_t n = 0, agree = 0;
    std::vector<double> kls;
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < seqs; ++s) {
      ingine::model::reset_state(rt.state(), rt.stream());
      rt.ensure_kv(ctx + 64);  // shared KV/expert region: claim the rows this sequence writes
      for (int i = 0; i + 1 < ctx; ++i) {
        const int32_t t = ids[std::size_t(s) * ctx + i], next = ids[std::size_t(s) * ctx + i + 1];
        rt.set_inputs(std::span<const int32_t>(&t, 1), i);
        const float* d = rt.run_step(1);
        cudaMemcpyAsync(logits.data(), d, V * sizeof(float), cudaMemcpyDeviceToHost, rt.stream());
        cudaStreamSynchronize(rt.stream());
        log_softmax(logits.data(), lp);
        if (saving) {
          for (int v = 0; v < V; ++v) buf[v] = __float2half(float(std::max(lp[v], -60.0)));
          fo.write(reinterpret_cast<const char*>(buf.data()), V * sizeof(__half));
          nll += -lp[next];
        } else {
          fi.read(reinterpret_cast<char*>(buf.data()), V * sizeof(__half));
          lr.resize(V);
          int am_r = 0, am_t = 0;
          double kl = 0;
          for (int v = 0; v < V; ++v) {
            lr[v] = double(__half2float(buf[v]));
            if (lr[v] > lr[am_r]) am_r = v;
            if (lp[v] > lp[am_t]) am_t = v;
            const double pr = std::exp(lr[v]);
            if (pr > 1e-12) kl += pr * (lr[v] - lp[v]);
          }
          kl_sum += kl;
          kls.push_back(kl);
          agree += am_r == am_t;
          nll += -lp[next];
          nll_ref += -lr[next];
        }
        ++n;
      }
      std::printf("seq %d done (%lld tokens, %.1f s)\n", s, (long long)n,
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      std::fflush(stdout);
    }
    if (!profile_out.empty()) {
      const auto c = rt.route_counts();
      std::FILE* f = std::fopen(profile_out.c_str(), "wb");
      if (f) std::fwrite(c.data(), 4, c.size(), f), std::fclose(f);
      std::printf("routing profile written: %s\n", profile_out.c_str());
    }
    if (saving) {
      std::printf("reference saved: %lld positions, ppl %.4f\n", (long long)n, std::exp(nll / n));
    } else {
      std::sort(kls.begin(), kls.end());
      std::printf("RESULT positions %lld  mean KLD %.5f  median %.5f  p99 %.4f  top-1 agree %.2f%%  ppl %.4f (ref %.4f)\n",
                  (long long)n, kl_sum / n, kls[kls.size() / 2], kls[std::size_t(kls.size() * 0.99)],
                  100.0 * agree / n, std::exp(nll / n), std::exp(nll_ref / n));
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "quill-kld: %s\n", e.what());
    return 1;
  }
  return 0;
}
