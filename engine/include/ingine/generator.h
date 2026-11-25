// The token generator interface the server drives (mirror of engine.Engine.generate). CUDA-free so the
// scheduler, the HTTP layer and their tests build without the model; ingine_model implements make_generator.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::model {

struct Sampling {
  float temperature = 1.0f;  // 0 => greedy
  float top_p = 1.0f;
  int top_k = 0;  // 0 => disabled
  std::optional<uint64_t> seed;
  // Teacher-forced replay (test/benchmark only; empty = off): generate() emits exactly these tokens
  // (max_new_tokens is capped at their count), running its real loop along them: plain decoding samples as
  // usual and then takes the forced token; speculative decoding accepts a draft iff it equals the forced token
  // and continues with the forced token after the accepted run. Fills GenerateStats::spec_trace.
  std::span<const int32_t> forced;
};

// The token whose embedding an image token replaces (<|unused_200054|>, the chat template's image placeholder).
inline constexpr std::int32_t IMAGE_PLACEHOLDER_ID = 200054;

// One image of a prompt. The chat template marks an image with one placeholder token; the prompt holds
// tokens() ids in its place, one per 40x40-pixel tile in raster order, whose embeddings the vision tower
// computes from the pixels. These ids are negative (never a vocabulary id) and a function of the image
// content and the tile index (image::token_id), so prompts compare equal exactly when their images do.
struct PromptImage {
  int pos = 0;                                           // prompt position of the image's first token
  int height = 0, width = 0;                             // pixels
  std::shared_ptr<const std::vector<std::uint8_t>> rgb;  // [height, width, 3]
  [[nodiscard]] int rows() const { return (height + 39) / 40; }
  [[nodiscard]] int cols() const { return (width + 39) / 40; }
  [[nodiscard]] int tokens() const { return rows() * cols(); }
};

// A prompt: token ids and its images, in the order of their positions (without overlap).
struct Prompt {
  std::span<const std::int32_t> ids;
  std::span<const PromptImage> images;
};

// How the prompt reused the state of earlier requests (reused_tokens were not prefilled again).
enum class Reuse {
  None,        // empty state, the whole prompt prefilled
  Extend,      // the prompt extends the tokens in the state (previous prompt + processed generated tokens)
  Checkpoint,  // restored the checkpoint taken after the previous prompt's prefill (a prefix of this prompt)
  Parked,      // restored a parked conversation's checkpoint (a prefix of this prompt; RuntimeOptions::park_mb)
};
inline const char* to_string(Reuse reuse) {
  switch (reuse) {
    case Reuse::Extend: return "extend";
    case Reuse::Checkpoint: return "checkpoint";
    case Reuse::Parked: return "parked";
    default: return "none";
  }
}

// One speculative round of a forced replay: verify width, accepted drafts, the drafts that would have been
// accepted had all 7 been verified, and GPU time of the round's verify (+ acceptance) and of its draft.
struct SpecRoundRecord {
  int width = 0, accepted = 0, accepted_all = 0;
  float verify_ms = 0, draft_ms = 0;
  float conf[7] = {};  // quill: the drafter's confidence per draft
};

struct GenerateStats {
  int prompt_tokens = 0, reused_tokens = 0, generated_tokens = 0;
  Reuse reuse = Reuse::None;
  double prefill_seconds = 0, decode_seconds = 0;
  int spec_rounds = 0, spec_accepted = 0;
  std::vector<SpecRoundRecord> spec_trace;  // forced replay only
  // Images: tokens of the prompt's images, how many images the vision tower encoded for this request (those
  // not inside the reused prefix) and its time (part of prefill_seconds).
  int image_tokens = 0, images_encoded = 0;
  double vision_seconds = 0;
};

struct RuntimeOptions {
  std::filesystem::path model_dir, ablation;  // empty ablation: original model
  int max_ctx = 65536;
  // Tokens per prefill chunk. 3072 (the largest the sliding attention takes) reads a long prompt 6-10% faster
  // than 2048 for 0.38 GB more scratch; results differ between chunk sizes only by rounding.
  int prefill_chunk = 3072;
  // quill: the prefill scratch (~0.9 GB at 3072) lives in expert slots of the shared region: experts during
  // decode, evicted for each prefill and uploaded back after it (~40 ms)
  bool lend_prefill_scratch = true;
  // quill: the shared region in re-mappable 32 MiB chunks: chunks are released when the VRAM in use (all
  // processes, NVML) nears the cap and taken back when it falls, so the startup reserve can be small
  bool dynamic_vram = true;
  int vram_dynamic_reserve_mib = 160;
  // quill: the speculative verify widths (rows = anchor + drafts) captured as graphs; a round uses the smallest
  // width holding its confident drafts
  std::vector<int> verify_widths = {2, 4, 8};
  // quill, LOSSY (off at 0): drop non-resident experts whose routing weight is below this fraction of the
  // token's total instead of computing them on the CPU tier (kept weights renormalised unless !skip_renorm)
  float skip_cold = 0.f;
  bool skip_renorm = true;
  bool dense_fp8 = true, kv_fp8 = false, spec = false;
  float spec_threshold = 0.8f;
  // Conversation parking: device memory (MiB) for the checkpoints of conversations displaced by another
  // one, least recently used evicted, all released when a device allocation fails; 0 disables; negative:
  // min(3072, free memory after start-up - 4096), at least 0.
  int park_mb = -1;
  // Image input: load the vision tower (636 MB of the checkpoint, resident) and accept prompts with images.
  bool vision = false;
  // The tower on the CPU (model/vision_cpu.h): host memory instead of GPU memory, with this many threads.
  bool vision_cpu = false;
  int vision_threads = 8;
  // quill (12 GB GPUs): full-attention caches as IQ4_NL q4 rows; experts in host memory with a GPU slot pool
  // sized so that the device's total memory use (every process, as Task Manager counts it) stays below
  // vram_cap_gib; vram_reserve_mib is kept for what is allocated after the pools (prefill scratch, drafter).
  bool kv_q4 = true;
  double vram_cap_gib = 11.3;
  int vram_reserve_mib = 512;
  int cpu_threads = 7;
  int expert_staging = 24;  // prefill staging slots per expert format
  bool mxfp8_as_nvfp4 = false;
  bool adaptive_experts = true;
  bool shared_region = true;
  std::filesystem::path draft_dir;  // with spec: a DFlash2 drafter directory (else DSpark from model_dir)  // q4 KV and NVFP4 expert slots share one device region (short contexts get more experts)  // follow the conversation's routing with the GPU expert pool (residency.cpp)
  int adapt_interval = 16, adapt_swaps = 32, adapt_prefill_swaps = 512;
  std::filesystem::path expert_profile;  // routing counts [24][128] uint32: resident experts by use, else loop layers first  // re-encode the MXFP8 expert layers (20-23) as NVFP4 at load (lossy, half the bytes)
};

class Generator {
 public:
  virtual ~Generator() = default;
  virtual std::string model_id() const = 0;  // "inkling-2-mini-abliterated" with an ablation, else "inkling-2-mini"
  virtual int max_model_len() const = 0;     // = max_ctx
  // One request at a time. Emits sampled tokens in order through on_tokens (a speculative round emits several);
  // stops after a token in stop_ids, after max_new_tokens, or when on_tokens returns false (cancel).
  // Throws std::invalid_argument when the prompt does not fit max_ctx (reserve as engine.py does).
  // quill: decode routing counts since the last reset ([24 * 128]); empty when not available
  virtual std::vector<std::uint32_t> route_counts() { return {}; }
  virtual void reset_route_counts() {}
  // quill diagnostics: n launches of the width-m decode step graph (after a generate()), between
  // cudaProfilerStart / Stop; returns ms per step
  virtual double profile_steps(int m, int n) { (void)m, (void)n; return 0.0; }
  virtual GenerateStats generate(std::span<const int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                                 std::span<const int32_t> stop_ids,
                                 const std::function<bool(std::span<const int32_t>)>& on_tokens) = 0;
  // The same for a prompt with images. Generators without image input take prompts without images only.
  virtual GenerateStats generate(const Prompt& prompt, int max_new_tokens, const Sampling& sampling,
                                 std::span<const int32_t> stop_ids,
                                 const std::function<bool(std::span<const int32_t>)>& on_tokens) {
    if (!prompt.images.empty()) throw std::invalid_argument("this generator does not take images");
    return generate(prompt.ids, max_new_tokens, sampling, stop_ids, on_tokens);
  }
  virtual bool supports_images() const { return false; }
};

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options);  // the model (ingine_model)

}  // namespace ingine::model
