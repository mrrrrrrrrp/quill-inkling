// quill: DFlash2 drafter (dflash2.h).
#include "model/dflash2.h"

#include "core/device.h"
#include "core/safetensors.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ingine::model {

namespace df = kernels::dflash2;
namespace kd = kernels::draft;
namespace kdense = kernels::dense;

namespace {
void capture_begin(cudaStream_t s) { CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal)); }
cudaGraphExec_t capture_end(cudaStream_t s) {
  cudaGraph_t g = nullptr;
  CUDA_CHECK(cudaStreamEndCapture(s, &g));
  cudaGraphExec_t e = nullptr;
  const cudaError_t err = cudaGraphInstantiate(&e, g, 0);
  cudaGraphDestroy(g);
  CUDA_CHECK(err);
  return e;
}
constexpr std::size_t RING_BYTES = std::size_t(df::WIN) * df::KVD * 2;  // one layer's k (or v) context slots
}  // namespace

void* DFlash2::alloc(std::size_t bytes) {
  storage_.emplace_back(bytes);
  return storage_.back().p;
}

DFlash2::DFlash2(const std::filesystem::path& draft_dir, const Weights& w, cudaStream_t stream) : w_(w), stream_(stream) {
  if (!w.unembed.fp8() && !w.unembed.fp6()) throw std::runtime_error("DFlash2: the target head must be FP8 or FP6 rows");
  if (const char* e = std::getenv("QUILL_DRAFT_TS")) draft_ts_ = float(std::atof(e));  // experiment: Td / T
  if (const char* e = std::getenv("QUILL_DRAFT_FP6")) draft_fp6_ = std::atoi(e) != 0;
  if (const char* e = std::getenv("QUILL_DRAFT_FP4")) draft_fp4_ = std::atoi(e) != 0;  // default on (3 seeds: +0.7%)
  head_ = kdense::Matrix{w.unembed.w.data, static_cast<const float*>(w.unembed.scale.data), w.unembed.rows(),
                         static_cast<const std::uint8_t*>(w.unembed.bscale.data)};
  V_ = UNPADDED_VOCAB;  // quill: the head may be split (FP6 prefix + FP4 tail)
  // quill experiment (QUILL_DRAFT_VOCAB=N): the draft head reads only the first N unembedding rows (BPE ids follow
  // merge order, roughly frequency): a view, no copy; drafts stay within ids < N, verification is unchanged
  Vh_ = V_;
  head_tok_ = static_cast<const std::int32_t*>(w.head_tok.data);
  // quill: frequency-ordered head rows: the draft head reads the special tokens + the corpus-ranked shortlist
  // (41,390 tokens; 3 seeds 118.4 / 117.9 tok/s at 49k / 32k rows vs 115.3 full). QUILL_DRAFT_VOCAB=N overrides.
  if (head_tok_) Vh_ = std::min(V_, HEAD_DRAFT_ROWS);
  Vh_ = std::min(Vh_, w.unembed.rows());  // within the FP6 rows
  if (const char* e = std::getenv("QUILL_DRAFT_VOCAB")) Vh_ = std::max(1024, std::min(V_, std::atoi(e)));
  head_.rows = Vh_;
  load(draft_dir);
  auto f32 = [&](std::size_t n) { return static_cast<float*>(alloc(n * 4)); };
  auto b16 = [&](std::size_t n) { return static_cast<__nv_bfloat16*>(alloc(n * 2)); };
  auto i32 = [&](std::size_t n) { return static_cast<std::int32_t*>(alloc(n * 4)); };
  params_ = static_cast<df::Df2Params*>(alloc(sizeof(df::Df2Params)));
  CUDA_CHECK(cudaMemsetAsync(params_, 0, sizeof(df::Df2Params), stream_));
  ring_pos_ = i32(df::WIN);
  for (Layer& L : layers_) {
    L.k_ring = b16(std::size_t(df::SLOTS) * df::KVD);
    L.v_ring = b16(std::size_t(df::SLOTS) * df::KVD);
    CUDA_CHECK(cudaMemsetAsync(L.k_ring, 0, std::size_t(df::SLOTS) * df::KVD * 2, stream_));
    CUDA_CHECK(cudaMemsetAsync(L.v_ring, 0, std::size_t(df::SLOTS) * df::KVD * 2, stream_));
  }
  feats_ = b16(std::size_t(df::BLOCK) * 4 * df::H);
  g_ = b16(df::BLOCK * df::H), xn_ = b16(df::BLOCK * df::H), xc_ = b16(df::BLOCK * df::H), h_ = b16(df::BLOCK * df::H);
  q_ = b16(df::BLOCK * df::QD), attn_ = b16(df::BLOCK * df::QD), act_ = b16(std::size_t(df::BLOCK) * df::INTER);
  fcout_ = f32(df::BLOCK * df::H), kvout_ = f32(df::BLOCK * 2 * df::KVD), x_ = f32(df::BLOCK * df::H), dyn_ = f32(df::BLOCK * df::NDYN);
  qkv_out_ = f32(df::BLOCK * df::QKV), o_ = f32(df::BLOCK * df::H), gu_ = f32(std::size_t(df::BLOCK) * 2 * df::INTER);
  logits_ = f32(std::size_t(df::GAMMA) * V_), cand_val_ = f32(df::GAMMA * df::TOPK), gate_ = f32(df::GAMMA * df::RANK);
  draft_logits_ = f32(std::size_t(df::GAMMA) * V_), conf_ = f32(df::GAMMA), qstats_ = f32(2 * df::GAMMA);
  cand_id_ = i32(df::GAMMA * df::TOPK), prev_cand_ = i32(df::GAMMA * df::TOPK), tokens_ = i32(df::GAMMA);
  kd::fill(draft_logits_, std::size_t(df::GAMMA) * V_, -INFINITY, stream_);
  CUDA_CHECK(cudaMemsetAsync(prev_cand_, 0xff, df::GAMMA * df::TOPK * 4, stream_));
  CUDA_CHECK(cudaMemsetAsync(feats_, 0, std::size_t(df::BLOCK) * 4 * df::H * 2, stream_));
  auto segments = [&](int rows) {
    const int host[3] = {0, rows, 0};
    int* d = i32(3);
    CUDA_CHECK(cudaMemcpy(d, host, sizeof host, cudaMemcpyHostToDevice));
    return d;
  };
  seg_qkv_ = segments(df::QKV), seg_kv_ = segments(2 * df::KVD), seg_dyn_ = segments(df::NDYN), seg_gu_ = segments(2 * df::INTER);
  seg_head_ = segments(Vh_), seg_sel_ = segments(df::RANK);
  accept_.target = i32(df::BLOCK);
  accept_.probs = f32(std::size_t(df::BLOCK) * V_);
  accept_.resid = f32(V_);
  accept_.uniforms = f32(df::GAMMA);
  accept_.ws = f32(kd::sampled_accept_workspace_floats(V_));
  accept_.result = static_cast<kd::AcceptResult*>(alloc(sizeof(kd::AcceptResult)));
  reset();
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  spdlog::info("dflash2: drafter loaded ({:.0f} MiB on the device)", device_bytes() / 1048576.0);
}

DFlash2::~DFlash2() {
  for (cudaGraphExec_t g : {draft_graph_, commit_graph_, advance_graph_})
    if (g) cudaGraphExecDestroy(g);
}

std::size_t DFlash2::device_bytes() const {
  std::size_t n = 0;
  for (const auto& b : storage_) n += b.bytes;
  return n;
}

void DFlash2::load(const std::filesystem::path& dir) {
  SafetensorsFile f(dir / "model.safetensors");
  auto upload_bf16 = [&](const std::string& name, std::size_t numel) {
    const SafetensorsTensor& t = f.at(name);
    if (t.dtype != "BF16" || std::size_t(t.numel()) != numel)
      throw std::runtime_error("dflash2: unexpected " + name);
    auto* d = static_cast<__nv_bfloat16*>(alloc(numel * 2));
    const std::vector<std::byte> raw = f.read(name);
    CUDA_CHECK(cudaMemcpy(d, raw.data(), numel * 2, cudaMemcpyHostToDevice));
    return d;
  };
  auto base_f32 = [&](const std::string& name) {
    const std::vector<std::byte> raw = f.read(name);
    std::vector<float> v(4 * df::H);
    for (int i = 0; i < 4 * df::H; ++i) {
      std::uint16_t b;
      std::memcpy(&b, raw.data() + 2 * i, 2);
      const std::uint32_t u = std::uint32_t(b) << 16;
      std::memcpy(&v[i], &u, 4);
    }
    auto* d = static_cast<float*>(alloc(v.size() * 4));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    return static_cast<const float*>(d);
  };
  // FP8 rows of a row-concatenation of BF16 [r, K] tensors
  auto fp8 = [&](std::initializer_list<std::string> names, int K) {
    std::size_t rows = 0;
    for (const auto& n : names) rows += std::size_t(f.at(n).numel()) / K;
    DeviceBuffer tmp(rows * K * 2);
    std::size_t r0 = 0;
    for (const auto& n : names) {
      const SafetensorsTensor& t = f.at(n);
      if (t.dtype != "BF16" || t.shape.back() != K)
        throw std::runtime_error("dflash2: unexpected " + n + " (" + t.dtype + ", last dim " + std::to_string(t.shape.back()) +
                                 ", expected " + std::to_string(K) + ")");
      const std::vector<std::byte> raw = f.read(n);
      CUDA_CHECK(cudaMemcpy(static_cast<char*>(tmp.p) + r0 * K * 2, raw.data(), raw.size(), cudaMemcpyHostToDevice));
      r0 += std::size_t(t.numel()) / K;
    }
    // pageable cudaMemcpy may return before its DMA lands and the engine stream does not wait for the legacy
    // stream: without this the conversion read the tail of the last tensor before it arrived
    CUDA_CHECK(cudaDeviceSynchronize());
    if (const char* sim = std::getenv("QUILL_DRAFT_SIM")) {  // quill experiment: low-bit drafter (round trip)
      auto* wb = static_cast<__nv_bfloat16*>(tmp.p);
      if (std::string(sim) == "nvfp4") kdense::nvfp4_roundtrip(wb, int(rows), K, stream_);
      else if (std::string(sim) == "e2m2") kdense::fp6_roundtrip(wb, int(rows), K, 16, 2, stream_);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    auto* sc = static_cast<float*>(alloc(rows * 4));
    if (draft_fp4_) {  // quill: FP4 E2M1 rows (fp4.cuh)
      void* q = alloc(rows * K / 2);
      auto* bs = static_cast<std::uint8_t*>(alloc(rows * (K / 16)));
      kdense::fp4_rows(static_cast<const __nv_bfloat16*>(tmp.p), int(rows), K, q, bs, sc, stream_);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      return kdense::Matrix{q, sc, int(rows), bs, true};
    }
    if (draft_fp6_) {  // quill: FP6 E2M3 rows (fp6.cuh)
      void* q = alloc(rows * K * 3 / 4);
      auto* bs = static_cast<std::uint8_t*>(alloc(rows * (K / 32)));
      kdense::fp6_rows(static_cast<const __nv_bfloat16*>(tmp.p), int(rows), K, q, bs, sc, stream_);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      return kdense::Matrix{q, sc, int(rows), bs};
    }
    void* q = alloc(rows * K);
    kdense::fp8_rows(static_cast<const __nv_bfloat16*>(tmp.p), int(rows), K, q, sc, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return kdense::Matrix{q, sc, int(rows)};
  };
  fc_ = fp8({"fc.weight"}, 4 * df::H);
  hidden_norm_ = upload_bf16("hidden_norm.weight", df::H);
  final_norm_ = upload_bf16("norm.weight", df::H);
  for (int l = 0; l < df::LAYERS; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    Layer& L = layers_[l];
    L.in_norm = upload_bf16(p + "input_layernorm.weight", df::H);
    L.post_norm = upload_bf16(p + "post_attention_layernorm.weight", df::H);
    L.q_norm = upload_bf16(p + "self_attn.q_norm.weight", df::HD);
    L.k_norm = upload_bf16(p + "self_attn.k_norm.weight", df::HD);
    L.qkv = fp8({p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"}, df::H);
    // the context k/v projections are rows [df::QD, df::QKV) of the same matrix
    if (L.qkv.fp4)
      L.kv = kdense::Matrix{static_cast<const std::uint8_t*>(L.qkv.w) + std::size_t(df::QD) * df::H / 2, L.qkv.scale + df::QD,
                            2 * df::KVD, L.qkv.bscale + std::size_t(df::QD) * (df::H / 16), true};
    else if (L.qkv.bscale)
      L.kv = kdense::Matrix{static_cast<const std::uint8_t*>(L.qkv.w) + std::size_t(df::QD) * df::H * 3 / 4, L.qkv.scale + df::QD,
                            2 * df::KVD, L.qkv.bscale + std::size_t(df::QD) * (df::H / 32)};
    else
      L.kv = kdense::Matrix{static_cast<const std::uint8_t*>(L.qkv.w) + std::size_t(df::QD) * df::H, L.qkv.scale + df::QD, 2 * df::KVD};
    L.cpa = fp8({p + "attention_conv.kernel_projection.weight"}, df::H);
    L.cpm = fp8({p + "mlp_conv.kernel_projection.weight"}, df::H);
    L.wo = fp8({p + "self_attn.o_proj.weight"}, df::QD);
    L.gate_up = fp8({p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"}, df::H);
    L.down = fp8({p + "mlp.down_proj.weight"}, df::INTER);
    L.base_a = base_f32(p + "attention_conv.base_kernel");
    L.base_m = base_f32(p + "mlp_conv.base_kernel");
  }
  sel_ = kdense::Matrix{upload_bf16("candidate_selector.hidden_projection.weight", std::size_t(df::RANK) * df::H), nullptr, df::RANK};
  const std::size_t vcb = std::size_t(f.at("candidate_selector.predecessor_codebook").shape.at(0));
  if (vcb < std::size_t(V_)) throw std::runtime_error("dflash2: codebooks smaller than the vocabulary");
  // the codebooks (206 MB, ~17 rows read per draft step) in mapped pinned host memory instead of VRAM
  auto host_cb = [&](const std::string& name) {
    const SafetensorsTensor& t = f.at(name);
    auto buf = std::make_shared<MappedHostBuffer>(t.bytes);
    f.read(t, buf->data());
    host_bufs_.push_back(buf);
    return static_cast<const __nv_bfloat16*>(buf->device_ptr());
  };
  pred_ = host_cb("candidate_selector.predecessor_codebook");
  succ_ = host_cb("candidate_selector.successor_codebook");
  CUDA_CHECK(cudaDeviceSynchronize());
}

void DFlash2::reset() {
  CUDA_CHECK(cudaMemsetAsync(ring_pos_, 0xff, df::WIN * 4, stream_));
  length_ = 0;
}

void DFlash2::set_seed(std::uint64_t seed) { df::set_seed(params_, seed, stream_); }

// feats_ (8 staged rows) -> context k/v of all layers at p->start + r, r < p->ncommit
void DFlash2::run_inject() {
  kdense::gemv(feats_, df::BLOCK, 4 * df::H, fc_, fcout_, false, stream_);
  df::rmsnorm(fcout_, hidden_norm_, df::BLOCK, g_, stream_);
  for (int l = 0; l < df::LAYERS; ++l) {
    const Layer& L = layers_[l];
    kdense::proj(g_, df::BLOCK, L.kv, seg_kv_, 1, kvout_, false, stream_);
    df::ctx_store(kvout_, 2 * df::KVD, L.k_norm, params_, L.k_ring, L.v_ring, ring_pos_, l == 0, stream_);
  }
}

void DFlash2::run_draft(float* debug_x) {
  df::embed_block(static_cast<const __nv_bfloat16*>(w_.embed.data), static_cast<const float*>(w_.embed_norm.data), params_,
                  layers_[0].in_norm, x_, xn_, stream_);
  for (int l = 0; l < df::LAYERS; ++l) {
    const Layer& L = layers_[l];
    if (l > 0) df::rmsnorm(x_, L.in_norm, df::BLOCK, xn_, stream_);
    kdense::proj(xn_, df::BLOCK, L.cpa, seg_dyn_, 1, dyn_, false, stream_);
    df::conv_in(xn_, dyn_, L.base_a, xc_, stream_);
    kdense::proj(xc_, df::BLOCK, L.qkv, seg_qkv_, 1, qkv_out_, false, stream_);
    df::qkv_post(qkv_out_, L.q_norm, L.k_norm, params_, q_, L.k_ring, L.v_ring, stream_);
    df::attention(q_, L.k_ring, L.v_ring, ring_pos_, params_, attn_, stream_);
    kdense::gemv(attn_, df::BLOCK, df::QD, L.wo, o_, false, stream_);
    df::conv_residual(o_, dyn_, L.base_a, x_, stream_);
    df::rmsnorm(x_, L.post_norm, df::BLOCK, xn_, stream_);
    kdense::proj(xn_, df::BLOCK, L.cpm, seg_dyn_, 1, dyn_, false, stream_);
    df::conv_in(xn_, dyn_, L.base_m, xc_, stream_);
    kdense::proj(xc_, df::BLOCK, L.gate_up, seg_gu_, 1, gu_, false, stream_);
    kd::silu_mul(gu_, df::BLOCK, df::INTER, act_, stream_);
    kdense::gemv(act_, df::BLOCK, df::INTER, L.down, o_, false, stream_);
    df::conv_residual(o_, dyn_, L.base_m, x_, stream_);
    if (debug_x)
      CUDA_CHECK(cudaMemcpyAsync(debug_x + std::size_t(l) * df::BLOCK * df::H, x_, std::size_t(df::BLOCK) * df::H * 4,
                                 cudaMemcpyDeviceToDevice, stream_));
  }
  df::rmsnorm(x_, final_norm_, df::BLOCK, h_, stream_);
  const __nv_bfloat16* h1 = h_ + df::H;  // rows 1..7
  kdense::proj(h1, df::GAMMA, head_, seg_head_, 1, logits_, false, stream_);
  df::topk16(logits_, df::GAMMA, Vh_, cand_id_, cand_val_, stream_);
  if (head_tok_) df::map_ids(cand_id_, df::GAMMA * df::TOPK, head_tok_, stream_);  // quill: rows -> tokens
  kdense::proj(h1, df::GAMMA, sel_, seg_sel_, 1, gate_, false, stream_);
  df::select(cand_id_, cand_val_, gate_, pred_, succ_, params_, tokens_, draft_logits_, V_, prev_cand_, conf_, qstats_,
             stream_, draft_ts_);
}

void DFlash2::commit(const __nv_bfloat16* feats, int n, int start) {
  if (n <= 0) return;
  // only the last df::WIN positions can be attended
  const int first = std::max(0, n - df::WIN);
  for (int r0 = first; r0 < n; r0 += df::BLOCK) {
    const int k = std::min(df::BLOCK, n - r0);
    df::set_commit(params_, feats + std::size_t(r0) * 4 * df::H, k, start + r0, stream_);
    df::stage_commit(params_, feats_, stream_);
    if (commit_graph_) CUDA_CHECK(cudaGraphLaunch(commit_graph_, stream_));
    else run_inject();
    last_commit_start_ = start + r0, last_commit_n_ = k;
  }
  length_ = start + n;
}

DraftModel::Draft DFlash2::draft(int anchor, int P, float temperature) {
  if (anchor < 0 || anchor >= V_) throw std::invalid_argument("DFlash2::draft: anchor out of range");
  df::set_draft(params_, anchor, P, temperature, stream_);
  if (const char* d = std::getenv("QUILL_DF2_DUMP"); d && !dumped_ && P >= 16) {
    dump(d);
    return outputs();
  }
  if (draft_graph_) CUDA_CHECK(cudaGraphLaunch(draft_graph_, stream_));
  else run_draft();
  return outputs();
}

void DFlash2::run_advance() {
  df::round_begin(params_, round_, feats_, stream_);
  run_inject();
  run_draft();
}

void DFlash2::bind_round(const kd::RoundInputs& in) {
  if (advance_graph_) cudaGraphExecDestroy(advance_graph_), advance_graph_ = nullptr;
  round_ = in;
  round_bound_ = true;
  if (draft_graph_) {  // after capture(): one graph for the whole round transition
    capture_begin(stream_);
    run_advance();
    advance_graph_ = capture_end(stream_);
  }
}

DraftModel::Draft DFlash2::advance(int start) {
  if (!round_bound_) throw std::logic_error("DFlash2::advance: no bind_round()");
  pending_start_ = start;
  if (advance_graph_) CUDA_CHECK(cudaGraphLaunch(advance_graph_, stream_));
  else run_advance();
  return outputs();
}

void DFlash2::capture() {
  // warm-up outside capture, then capture the draft, the small commit and (if bound) the round
  df::set_draft(params_, 0, 0, 0.f, stream_);
  df::set_commit(params_, feats_, 0, 0, stream_);
  run_draft();
  run_inject();
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  capture_begin(stream_);
  run_draft();
  draft_graph_ = capture_end(stream_);
  capture_begin(stream_);
  run_inject();
  commit_graph_ = capture_end(stream_);
  if (round_bound_) bind_round(kd::RoundInputs(round_));
  reset();
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// checkpoint: the context slots of every layer (k then v) and the ring positions
std::size_t DFlash2::checkpoint_bytes() const { return std::size_t(df::LAYERS) * 2 * RING_BYTES + df::WIN * 4; }

std::array<DraftModel::Region, 3> DFlash2::checkpoint_regions(int) const {
  return {Region{0, checkpoint_bytes()}, Region{checkpoint_bytes(), 0}, Region{checkpoint_bytes(), 0}};
}

void DFlash2::save_checkpoint(void* dst) {
  auto* d = static_cast<char*>(dst);
  std::size_t off = 0;
  for (const Layer& L : layers_) {
    CUDA_CHECK(cudaMemcpyAsync(d + off, L.k_ring, RING_BYTES, cudaMemcpyDefault, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d + off + RING_BYTES, L.v_ring, RING_BYTES, cudaMemcpyDefault, stream_));
    off += 2 * RING_BYTES;
  }
  CUDA_CHECK(cudaMemcpyAsync(d + off, ring_pos_, df::WIN * 4, cudaMemcpyDefault, stream_));
  checkpoint_ = dst;
  checkpoint_length_ = length_;
}

void DFlash2::restore_checkpoint(const void* src) {
  if (src != checkpoint_) throw std::invalid_argument("DFlash2::restore_checkpoint: not the last checkpoint saved");
  const auto* s = static_cast<const char*>(src);
  std::size_t off = 0;
  for (const Layer& L : layers_) {
    CUDA_CHECK(cudaMemcpyAsync(L.k_ring, s + off, RING_BYTES, cudaMemcpyDefault, stream_));
    CUDA_CHECK(cudaMemcpyAsync(L.v_ring, s + off + RING_BYTES, RING_BYTES, cudaMemcpyDefault, stream_));
    off += 2 * RING_BYTES;
  }
  CUDA_CHECK(cudaMemcpyAsync(ring_pos_, s + off, df::WIN * 4, cudaMemcpyDefault, stream_));
  length_ = checkpoint_length_;
}

void DFlash2::adopt_checkpoint(const void* src, int length, std::span<const std::int32_t>) {
  checkpoint_ = src;
  checkpoint_length_ = length;
}

// quill diagnostics: one eager draft with every intermediate written to `dir` (see tools/df2_ref.py)
void DFlash2::dump(const std::string& dir) {
  dumped_ = true;
  DeviceBuffer xs(std::size_t(df::LAYERS) * df::BLOCK * df::H * 4);
  run_draft(static_cast<float*>(xs.p));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  auto put = [&](const std::string& name, const void* dev, std::size_t bytes) {
    std::vector<char> h(bytes);
    CUDA_CHECK(cudaMemcpy(h.data(), dev, bytes, cudaMemcpyDeviceToHost));
    std::FILE* f = std::fopen((dir + "/" + name).c_str(), "wb");
    if (f) std::fwrite(h.data(), 1, bytes, f), std::fclose(f);
  };
  df::Df2Params p{};
  CUDA_CHECK(cudaMemcpy(&p, params_, sizeof p, cudaMemcpyDeviceToHost));
  if (std::FILE* f = std::fopen((dir + "/meta.txt").c_str(), "w")) {
    std::fprintf(f, "%d %d %d %d\n", p.anchor, p.P, last_commit_start_, last_commit_n_);
    std::fclose(f);
  }
  put("ring_pos.i32", ring_pos_, df::WIN * 4);
  for (int l = 0; l < df::LAYERS; ++l) {
    put("ringk_l" + std::to_string(l) + ".bf16", layers_[l].k_ring, std::size_t(df::SLOTS) * df::KVD * 2);
    put("ringv_l" + std::to_string(l) + ".bf16", layers_[l].v_ring, std::size_t(df::SLOTS) * df::KVD * 2);
  }
  put("x_layers.f32", xs.p, xs.bytes);
  put("h.bf16", h_, std::size_t(df::BLOCK) * df::H * 2);
  put("logits.f32", logits_, std::size_t(df::GAMMA) * V_ * 4);
  put("cand_id.i32", cand_id_, df::GAMMA * df::TOPK * 4);
  put("cand_val.f32", cand_val_, df::GAMMA * df::TOPK * 4);
  put("gate.f32", gate_, df::GAMMA * df::RANK * 4);
  put("tokens.i32", tokens_, df::GAMMA * 4);
  put("feats.bf16", feats_, std::size_t(df::BLOCK) * 4 * df::H * 2);
  spdlog::info("dflash2: dumped a draft to {}", dir);
}

}  // namespace ingine::model
