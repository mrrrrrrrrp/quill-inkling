// Model runtime: weights + ablation + state + decoder, decode graphs per call width (engine.Engine._capture),
// state snapshots and speculative rollback (engine.Engine._snapshot/_restore/_rollback), pipelined width-1
// steps and the switch graphs of device-selected verify widths.
#include "model/runtime.h"

#include "core/device.h"
#include "model/cpu_tier.h"
#include "model/experts.h"
#include "model/vram_meter.h"
#include "model/vmm_region.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <array>
#include <cstdio>
#include <stdexcept>

namespace ingine::model {
namespace {

constexpr int kMaxWidth = 8;

DeviceBuffer device_copy(const Tensor& t, cudaStream_t stream) {
  DeviceBuffer b(t.bytes());
  CUDA_CHECK(cudaMemcpyAsync(b.p, t.data, t.bytes(), cudaMemcpyDeviceToDevice, stream));
  return b;
}

// Carves consecutive 256-byte aligned pieces out of one allocation.
struct Carve {
  char* p;
  template <class T>
  T* take(std::size_t count) {
    T* r = reinterpret_cast<T*>(p);
    p += (count * sizeof(T) + 255) / 256 * 256;
    return r;
  }
};
constexpr std::size_t carved(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// sizes of the step-undo and verify-backup records: one entry per visit (the loop pass count sets how many)
std::size_t undo_bytes() {
  const std::size_t nv = num_visits(), nsv = num_sliding_visits();
  return carved(nv * CONV_ROW_FLOATS * 4) + 2 * carved(nsv * KV_DIM * 2) + carved(nsv * 8);
}
std::size_t backup_bytes() {
  const std::size_t nv = num_visits(), nsv = num_sliding_visits();
  return 2 * carved(nv * 3 * KV_DIM * 4) + 2 * carved(nv * 3 * HIDDEN * 4) +
         2 * carved(nsv * kMaxWidth * KV_DIM * 2) + carved(nsv * kMaxWidth * 8);
}

}  // namespace

GraphExec::~GraphExec() {
  if (exec_) cudaGraphExecDestroy(exec_);
}

GraphExec& GraphExec::operator=(GraphExec&& o) noexcept {
  if (this != &o) {
    if (exec_) cudaGraphExecDestroy(exec_);
    exec_ = o.exec_;
    o.exec_ = nullptr;
  }
  return *this;
}

void GraphExec::launch(cudaStream_t stream) const {
  if (!exec_) throw std::logic_error("GraphExec::launch: empty graph");
  CUDA_CHECK(cudaGraphLaunch(exec_, stream));
}

Runtime::Runtime(const RuntimeOptions& options) : options_(options) {
  if (options.max_ctx < 16) throw std::invalid_argument("max_ctx must be at least 16");
  // quill: the per-thread stack reservation is stack size x every resident thread (1 KB x 86k threads by
  // default); the largest kernel stack frame is 224 bytes (cuobjdump --dump-resource-usage)
  if (const char* e = std::getenv("QUILL_STACK"); !e || std::atoi(e) > 0)
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, e ? std::size_t(std::atoi(e)) : std::size_t(256)));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  vram_phase("context");
  weights_ = load_weights(options.model_dir, options.dense_fp8, stream_, options.mxfp8_as_nvfp4);
  vram_phase("weights");
  spdlog::info("vram breakdown: weights {:.0f} MiB", device_buffer_live_bytes() / 1048576.0);
  ablation_ = load_ablation(options.ablation);
  init();
}

Runtime::Runtime(const RuntimeOptions& options, Weights weights, Ablation ablation)
    : options_(options), weights_(std::move(weights)), ablation_(std::move(ablation)) {
  if (options.max_ctx < 16) throw std::invalid_argument("max_ctx must be at least 16");
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  init();
}

void Runtime::init() {
  const RuntimeOptions& options = options_;
  spdlog::info("loop passes: {}{} -> {} visits per token ({} full-attention, {} sliding); QUILL_PASSES sets it", passes(),
               passes() == TRAINED_PASSES ? " (as trained)" : " (checkpoint trained with 3)", num_visits(),
               num_full_visits(), num_sliding_visits());
  state_ = make_state(options.max_ctx, options.kv_fp8, options.kv_q4, options.kv_q4 && options.shared_region &&
                                                                         options.mxfp8_as_nvfp4);
  const std::size_t b0 = device_buffer_live_bytes();
  decoder_ = std::make_unique<Decoder>(weights_, ablation_, state_, false);  // quill: no PDL before sm_90
  pos_ = DeviceBuffer(sizeof(std::int64_t));
  pos_.fill(0);
  token_ = DeviceBuffer(kMaxWidth * sizeof(std::int64_t));
  token_.fill(0);
  CUDA_CHECK(cudaStreamCreateWithFlags(&capture_stream_, cudaStreamNonBlocking));
  for (cudaEvent_t& e : chain_done_) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  chain_host_ = MappedHostBuffer(2 * sizeof(std::int32_t));
  for (int i = 0; i < 2; ++i) {
    undo_storage_[i] = DeviceBuffer(undo_bytes());
    Carve c{static_cast<char*>(undo_storage_[i].p)};
    undo_[i].conv = c.take<float>(std::size_t(num_visits()) * CONV_ROW_FLOATS);
    undo_[i].k = c.take<__nv_bfloat16>(std::size_t(num_sliding_visits()) * KV_DIM);
    undo_[i].v = c.take<__nv_bfloat16>(std::size_t(num_sliding_visits()) * KV_DIM);
    undo_[i].pos = c.take<std::int64_t>(num_sliding_visits());
  }

  buffers_.emplace(1, make_buffers(1));
  history_[1] = false;
  if (options.spec) {
    std::vector<int> ws = options.verify_widths;  // quill: configurable (sorted, unique, 2..8, ends at 8)
    std::sort(ws.begin(), ws.end());
    ws.erase(std::unique(ws.begin(), ws.end()), ws.end());
    if (ws.empty() || ws.front() < 2 || ws.back() != kMaxWidth)
      throw std::invalid_argument("verify_widths: values in 2..8, the largest must be 8");
    std::string wl;
    for (int w : ws) wl += (wl.empty() ? "" : ",") + std::to_string(w);
    spdlog::info("spec: verify widths {}", wl);
    for (int w : ws) {
      buffers_.emplace(w, make_buffers(w));
      history_[w] = true;
      verify_widths_.push_back(w);
    }
    backup_storage_ = DeviceBuffer(backup_bytes());
    Carve c{static_cast<char*>(backup_storage_.p)};
    for (int i = 0; i < 4; ++i) backup_.conv[i] = c.take<float>(std::size_t(num_visits()) * 3 * CONV_COLS[i]);
    backup_.k = c.take<__nv_bfloat16>(std::size_t(num_sliding_visits()) * kMaxWidth * KV_DIM);
    backup_.v = c.take<__nv_bfloat16>(std::size_t(num_sliding_visits()) * kMaxWidth * KV_DIM);
    backup_.pos = c.take<std::int64_t>(std::size_t(num_sliding_visits()) * kMaxWidth);
  }
  // every width reads its tokens from the shared buffer (set_inputs, chain_advance and the verify selector
  // write one place)
  for (auto& [m, b] : buffers_) b.token = rm(token_.p, DType::I64, {m});
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  vram_phase("ablation, state, decode bufs");
  spdlog::info("vram breakdown: ablation + state {:.0f} MiB, decoder + decode buffers {:.0f} MiB",
               (b0) / 1048576.0, (device_buffer_live_bytes() - b0) / 1048576.0);
  spdlog::info("vram: {:.0f} MiB in use after weights, state and decode buffers", device_used_bytes() / 1048576.0);
}

// quill: expert slot pools in what is left under the cap (everything else is allocated by now), the CPU tier,
// then the decode graphs.
void Runtime::finalize(std::size_t reserve_bytes) {
  const RuntimeOptions& options = options_;
  {
    CUDA_CHECK(cudaDeviceSynchronize());
    // the cap is a Task Manager number: measured with NVML (vram_meter.cpp), once the reading has settled
    const std::size_t cap = std::size_t(options.vram_cap_gib * double(1ull << 30));
    const std::size_t used = vram_used_settled_bytes();
    const std::size_t reserve = reserve_bytes;
    std::size_t budget = cap > used + reserve ? cap - used - reserve : 0;
    spdlog::info("vram: {:.0f} MiB in use (all processes, {}), cap {:.0f} MiB, reserve {:.0f} MiB -> expert pools "
                 "{:.0f} MiB", used / 1048576.0, vram_meter_nvml() ? "NVML" : "cudaMemGetInfo", cap / 1048576.0,
                 reserve / 1048576.0, budget / 1048576.0);
    std::vector<ExpertStore::Pair> priority;
    if (!options.expert_profile.empty()) {  // (layer, expert) by decode routing counts, most used first
      std::FILE* f = std::fopen(options.expert_profile.string().c_str(), "rb");
      std::vector<std::uint32_t> cnt(LAYERS * EXPERTS, 0);
      if (!f || std::fread(cnt.data(), 4, cnt.size(), f) != cnt.size())
        throw std::runtime_error("cannot read expert profile " + options.expert_profile.string());
      std::fclose(f);
      for (int l = 0; l < LAYERS; ++l)
        for (int e = 0; e < EXPERTS; ++e) priority.push_back({l, e});
      std::stable_sort(priority.begin(), priority.end(), [&](const ExpertStore::Pair& a, const ExpertStore::Pair& b) {
        return cnt[a.layer * EXPERTS + a.expert] > cnt[b.layer * EXPERTS + b.expert];
      });
      spdlog::info("experts: residency from profile {}", options.expert_profile.string());
    }
    if (state_.kv_shared && options.dynamic_vram && !std::getenv("QUILL_NO_DYN_VRAM")) {
      // quill: dynamic VRAM: the startup reserve shrinks to vram_dynamic_reserve_mib (vram_check holds the cap)
      const std::size_t dyn = std::size_t(options.vram_dynamic_reserve_mib) << 20;
      const std::size_t extra = dyn < reserve ? reserve - dyn : 0;
      vmm_ = std::make_unique<VmmRegion>();
      if (vmm_->init(budget + extra, std::size_t(32) << 20)) {
        budget = vmm_->bytes();
        spdlog::info("vram: dynamic region {:.0f} MiB in {} chunks of {:.0f} MiB (reserve {} MiB, re-checked every "
                     "residency step)", budget / 1048576.0, vmm_->chunks(), vmm_->chunk() / 1048576.0,
                     options.vram_dynamic_reserve_mib);
      } else {
        vmm_.reset();
        spdlog::warn("vram: virtual memory management unavailable; static region and reserve");
      }
    }
    if (state_.kv_shared && vmm_) {
      CUDA_CHECK(cudaMemsetAsync(vmm_->base(), 0, budget, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      char* r = vmm_->base();
      state_.k_full.data = r;
      state_.v_full.data = r + std::size_t(num_full_visits()) * 576;
      const std::size_t kv_max = std::size_t(options.max_ctx) * state_.full_stride();
      weights_.experts->build_pools_shared(r, budget, kv_max, options.expert_staging, priority, stream_);
      kv_front_ = 0;
      ensure_kv(64);
    } else if (state_.kv_shared) {
      region_ = DeviceBuffer(budget);
      // the driver may charge more than the allocation once it is resident (measured +100..+620 MiB): touch it,
      // re-measure, and give the overshoot back before anything is placed in it
      CUDA_CHECK(cudaMemsetAsync(region_.p, 0, budget, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      const std::size_t after = vram_used_settled_bytes();
      if (after + reserve > cap) {
        const std::size_t over = after + reserve - cap + (std::size_t(16) << 20);
        budget = budget > over ? budget - over : 0;
        region_ = DeviceBuffer();
        region_ = DeviceBuffer(budget);
        CUDA_CHECK(cudaMemsetAsync(region_.p, 0, budget, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        spdlog::info("vram: {:.0f} MiB in use with the region, over by {:.0f} MiB -> region {:.0f} MiB",
                     after / 1048576.0, over / 1048576.0, budget / 1048576.0);
      }
      char* r = static_cast<char*>(region_.p);
      state_.k_full.data = r;
      state_.v_full.data = r + std::size_t(num_full_visits()) * 576;
      const std::size_t kv_max = std::size_t(options.max_ctx) * state_.full_stride();
      weights_.experts->build_pools_shared(r, budget, kv_max, options.expert_staging, priority, stream_);
      kv_front_ = 0;
      ensure_kv(64);  // the decode graphs' warm-up writes a few rows
    } else {
      weights_.experts->build_pools(budget, options.expert_staging, priority, stream_);
    }
    cpu_ = std::make_unique<CpuTier>(*weights_.experts, options.cpu_threads);
    {  // quill: host copies of the routers for the tier's pre-gating prefetch
      std::vector<std::uint16_t> rw(std::size_t(LAYERS) * EXPERTS * HIDDEN);
      std::vector<float> rb(std::size_t(LAYERS) * EXPERTS);
      for (int l = 0; l < LAYERS; ++l) {
        const Layer& L = weights_.layers[l];
        CUDA_CHECK(cudaMemcpy(rw.data() + std::size_t(l) * EXPERTS * HIDDEN, L.router_bf16.data,
                              std::size_t(EXPERTS) * HIDDEN * 2, cudaMemcpyDefault));
        CUDA_CHECK(cudaMemcpy(rb.data() + std::size_t(l) * EXPERTS, L.router_bias.data, EXPERTS * 4, cudaMemcpyDefault));
      }
      cpu_->set_routers(std::move(rw), std::move(rb));
    }
    moe_ctl_ = DeviceBuffer(64);
    moe_ctl_.fill(0);
    route_counts_ = DeviceBuffer(sizeof(std::uint32_t) * LAYERS * EXPERTS);
    route_counts_.fill(0);
    CpuExchange* dx = cpu_->device_exchange();
    kernels::moe::HybridExchange hx;
    hx.req_hdr = reinterpret_cast<unsigned*>(&dx->req.seq);
    hx.req_slot = dx->req.slot;
    hx.req_expert = dx->req.expert;
    hx.req_x = &dx->req.x[0][0];
    hx.resp_ack = reinterpret_cast<unsigned*>(&dx->resp.ack);
    hx.resp_y = &dx->resp.y[0][0];
    hx.skip_below = options.skip_cold;
    hx.skip_renorm = options.skip_renorm;
    if (options.skip_cold > 0)
      spdlog::info("experts: LOSSY skip of non-resident experts below {:.3f} of the routing weight ({})",
                   options.skip_cold, options.skip_renorm ? "renormalised" : "not renormalised");
    // quill: GPU zero-copy assist of CPU-tier misses (QUILL_ASSIST = share of a call's unique misses)
    if (const char* e = std::getenv("QUILL_ASSIST")) hx.assist_share = float(std::atof(e));
    if (const char* e = std::getenv("QUILL_ASSIST_MIN")) hx.assist_min = std::atoi(e);
    if (const char* e = std::getenv("QUILL_ASSIST_BYTOK")) hx.assist_by_tokens = std::atoi(e) != 0;
    if (const char* e = std::getenv("QUILL_ASSIST_MINTOK")) hx.assist_min_tokens = std::atoi(e);
    unsigned char* aflags = nullptr;
    if (hx.assist_share > 0.f || hx.assist_min_tokens > 0) {
      assist_flags_ = DeviceBuffer(128);
      assist_flags_.fill(0);
      aflags = static_cast<unsigned char*>(assist_flags_.p);
      spdlog::info("experts: GPU zero-copy assist of {:.0f}% of each call's non-resident experts (min {} per call)",
                   hx.assist_share * 100.0, hx.assist_min);
    }
    decoder_->set_hybrid(hx, static_cast<unsigned*>(moe_ctl_.p), static_cast<unsigned*>(route_counts_.p), aflags);
  }
  vram_phase("pools + cpu tier");
  cpu_->activate();  // the eager warm-up of capture() runs the CPU tier
  for (const auto& [m, history] : history_) capture(m, history);
  vram_phase("decode graphs");
  cpu_->deactivate();
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  residency_init();
  reset_route_counts();
  if (vmm_) vram_check(true);  // quill: give back any overshoot of the startup sizing
  spdlog::info("vram: {:.0f} MiB in use after the expert pools and decode graphs (cap {:.0f} MiB)",
               device_used_bytes() / 1048576.0, options.vram_cap_gib * 1024.0);
}

std::vector<std::uint32_t> Runtime::route_counts() {
  std::vector<std::uint32_t> c(LAYERS * EXPERTS, 0);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  if (route_counts_.p) route_counts_.copy_to_host(c.data(), c.size() * 4);
  return c;
}

void Runtime::reset_route_counts() {
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  if (route_counts_.p) route_counts_.fill(0);
}

std::size_t Runtime::device_used_bytes() { return vram_used_bytes(); }

Runtime::~Runtime() {
  if (stream_) cudaStreamSynchronize(stream_);
  res_.reset();
  for (auto& [m, exec] : graphs_) cudaGraphExecDestroy(exec);
  graphs_.clear();
  decoder_.reset();
  for (cudaEvent_t e : chain_done_)
    if (e) cudaEventDestroy(e);
  if (capture_stream_) cudaStreamDestroy(capture_stream_);
  if (stream_) cudaStreamDestroy(stream_);
}

bool Runtime::has_width(int m) const { return buffers_.count(m) != 0; }

std::span<const int> Runtime::verify_widths() const { return verify_widths_; }

Buffers& Runtime::buffers(int m) {
  auto it = buffers_.find(m);
  if (it == buffers_.end()) throw std::invalid_argument("no decode buffers for width " + std::to_string(m));
  return it->second;
}

StatePtrs Runtime::state_ptrs() const {
  StatePtrs p;
  p.conv[0] = ptr<float>(state_.conv_k);
  p.conv[1] = ptr<float>(state_.conv_v);
  p.conv[2] = ptr<float>(state_.conv_attn);
  p.conv[3] = ptr<float>(state_.conv_mlp);
  p.k_ring = ptr<__nv_bfloat16>(state_.k_ring);
  p.v_ring = ptr<__nv_bfloat16>(state_.v_ring);
  p.ring_vecs = state_.ring_q8 ? 1088 / 16 : state_.ring_q4 ? 576 / 16 : 128;
  p.ring_pos = ptr<std::int64_t>(state_.ring_pos);
  return p;
}

void Runtime::capture(int m, bool history) {
  // warm up so the pointer tables and kernel attributes are set outside capture; the scratch writes to the
  // conv states and ring positions are undone afterwards (the cache rows it wrote are never valid)
  const Snapshot saved = snapshot();
  Buffers& b = buffers(m);
  for (int i = 0; i < 2; ++i) decoder_->decode(b, pos(), history, stream_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
  try {
    decoder_->decode(b, pos(), history, stream_);
  } catch (...) {
    cudaStreamEndCapture(stream_, &graph);
    if (graph) cudaGraphDestroy(graph);
    throw;
  }
  CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
  cudaGraphExec_t exec = nullptr;
  const cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
  cudaGraphDestroy(graph);
  CUDA_CHECK(err);
  CUDA_CHECK(cudaGraphUpload(exec, stream_));
  graphs_[m] = exec;
  restore(saved);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Runtime::set_inputs(std::span<const std::int32_t> tokens, int position) {
  const int m = int(tokens.size());
  buffers(m);  // the width must exist
  set_step_inputs(tokens.data(), m, position, this->tokens(), pos(), stream_);
}

const float* Runtime::run_step(int m) {
  auto it = graphs_.find(m);
  if (it == graphs_.end()) throw std::invalid_argument("no decode graph for width " + std::to_string(m));
  CUDA_CHECK(cudaGraphLaunch(it->second, stream_));
  return static_cast<const float*>(buffers(m).logits.data);
}

const float* Runtime::run_step_eager(int m) {
  Buffers& b = buffers(m);
  decoder_->decode(b, pos(), history_.at(m), stream_);
  return static_cast<const float*>(b.logits.data);
}

const float* Runtime::decode_logits(std::int32_t token, int position) {
  const std::int32_t t[1] = {token};
  set_inputs(t, position);
  return run_step(1);
}

void Runtime::chain_advance(const std::int32_t* sampled, int position) {
  const int k = position & 1;
  model::chain_advance(sampled, tokens(), pos(), position, static_cast<std::int32_t*>(chain_host_.device_ptr()) + k,
                       state_ptrs(), undo_[k], stream_);
  CUDA_CHECK(cudaEventRecord(chain_done_[k], stream_));
}

std::int32_t Runtime::chain_token(int position) {
  const int k = position & 1;
  CUDA_CHECK(cudaEventSynchronize(chain_done_[k]));
  return static_cast<const volatile std::int32_t*>(chain_host_.data())[k];
}

void Runtime::chain_undo(int position) { model::chain_undo(state_ptrs(), undo_[position & 1], position, stream_); }

Runtime::Snapshot Runtime::snapshot() {
  Snapshot s;
  for (const Tensor* t : {&state_.ring_pos, &state_.conv_k, &state_.conv_v, &state_.conv_attn, &state_.conv_mlp})
    s.tensors.push_back(device_copy(*t, stream_));
  return s;
}

void Runtime::restore(const Snapshot& snap) {
  const Tensor* ts[5] = {&state_.ring_pos, &state_.conv_k, &state_.conv_v, &state_.conv_attn, &state_.conv_mlp};
  for (int i = 0; i < 5; ++i)
    CUDA_CHECK(cudaMemcpyAsync(ts[i]->data, snap.tensors[i].p, ts[i]->bytes(), cudaMemcpyDeviceToDevice, stream_));
}

namespace {
// the state tensors a prompt checkpoint holds, in checkpoint order
std::array<const Tensor*, 7> checkpoint_tensors(const State& st) {
  return {&st.k_ring, &st.v_ring, &st.ring_pos, &st.conv_k, &st.conv_v, &st.conv_attn, &st.conv_mlp};
}
}  // namespace

std::size_t Runtime::checkpoint_bytes() const {
  std::size_t n = 0;
  for (const Tensor* t : checkpoint_tensors(state_)) n += carved(t->bytes());
  return n;
}

void Runtime::save_checkpoint(void* dst) {
  char* p = static_cast<char*>(dst);
  for (const Tensor* t : checkpoint_tensors(state_)) {
    CUDA_CHECK(cudaMemcpyAsync(p, t->data, t->bytes(), cudaMemcpyDefault, stream_));
    p += carved(t->bytes());
  }
}

void Runtime::restore_checkpoint(const void* src) {
  const char* p = static_cast<const char*>(src);
  for (const Tensor* t : checkpoint_tensors(state_)) {
    CUDA_CHECK(cudaMemcpyAsync(t->data, p, t->bytes(), cudaMemcpyDefault, stream_));
    p += carved(t->bytes());
  }
}

void Runtime::ensure_kv(int positions) {
  if (!state_.kv_shared || positions <= kv_front_) return;
  positions = std::min(positions, options_.max_ctx);
  const std::size_t front = std::size_t(positions) * state_.full_stride();
  ExpertStore& es = *weights_.experts;
  if (vmm_ && hole_lo_ < hole_hi_) {  // quill: the KV grows into released chunks: map them back first
    const std::size_t c = vmm_->chunk(), need = std::min(hole_hi_, (front + 2 * c - 1) / c * c);
    for (std::size_t b = hole_lo_; b < need; b += c)
      if (!vmm_->map(b / c)) throw std::runtime_error("vram: cannot map a region chunk back for the KV cache");
    if (need > hole_lo_) hole_lo_ = need;
  }
  if (res_ && es.first_usable(front) > es.first_usable(std::size_t(kv_front_) * state_.full_stride()))
    residency_flush();  // no copy may still be writing into a slot the KV is about to take
  es.evict_below(std::max(front, hole_hi_), stream_);
  kv_front_ = positions;
}

// ---- quill: dynamic VRAM (VmmRegion chunks) ----
bool Runtime::vram_shrink(std::size_t bytes) {
  if (!vmm_ || bytes == 0) return false;
  const std::size_t c = vmm_->chunk();
  ExpertStore& es = *weights_.experts;
  const ExpertPool& P = es.pool(ExpertFormat::NVFP4);
  // the hole starts one chunk above the KV front; it may not reach the lent prefill scratch / staging slots
  const std::size_t kv_end = (std::size_t(kv_front_) * state_.full_stride() + 2 * c - 1) / c * c;
  const std::size_t lo = hole_lo_ < hole_hi_ ? hole_lo_ : kv_end;
  const int top_slot = lent_first_ < lent_end_ ? lent_first_ : P.resident_slots();
  const std::size_t limit = std::size_t(P.base - vmm_->base()) + std::size_t(top_slot) * std::size_t(P.stride);
  const std::size_t from = std::max(hole_hi_, lo);
  const std::size_t hi = std::min(from + (bytes + c - 1) / c * c, limit / c * c);
  if (hi <= from) {
    spdlog::warn("vram: cannot release more of the region (KV front, prefill scratch)");
    return false;
  }
  residency_flush();  // no copy is still writing into the slots below
  es.evict_below(hi, stream_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  for (std::size_t b = lo; b < hi; b += c) vmm_->unmap(b / c);
  hole_lo_ = lo, hole_hi_ = hi;
  spdlog::info("vram: released {:.0f} MiB of the expert region (hole {:.0f} MiB, {} resident experts)",
               (hi - from) / 1048576.0, (hi - lo) / 1048576.0, es.resident_total());
  return true;
}

bool Runtime::vram_grow_one() {
  if (!vmm_ || hole_lo_ >= hole_hi_) return false;
  const std::size_t c = vmm_->chunk();
  if (!vmm_->map((hole_hi_ - c) / c)) return false;
  CUDA_CHECK(cudaMemsetAsync(vmm_->base() + hole_hi_ - c, 0, c, stream_));
  hole_hi_ -= c;
  if (hole_hi_ <= hole_lo_) hole_lo_ = hole_hi_ = 0;
  spdlog::info("vram: took back {:.0f} MiB of the expert region (hole {:.0f} MiB)", c / 1048576.0,
               (hole_hi_ - hole_lo_) / 1048576.0);
  return true;
}

void Runtime::vram_check(bool startup) {
  // re-entry: vram_shrink -> residency_flush -> residency_step -> vram_check (a stack overflow without this guard)
  if (!vmm_ || in_vram_check_) return;
  struct Guard {
    bool& f;
    explicit Guard(bool& b) : f(b) { f = true; }
    ~Guard() { f = false; }
  } guard(in_vram_check_);
  const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  if (!startup && now - last_vram_change_ < 1.0) return;  // NVML lags a change
  const std::size_t cap = std::size_t(options_.vram_cap_gib * double(1ull << 30));
  const std::size_t target = cap - (std::size_t(options_.vram_dynamic_reserve_mib) << 20);
  const std::size_t used = startup ? vram_used_settled_bytes() : vram_used_bytes();
  if (used + (std::size_t(48) << 20) > cap || (startup && used > target)) {
    if (vram_shrink(used - target)) last_vram_change_ = now;
  } else if (hole_lo_ < hole_hi_ && used + vmm_->chunk() + (std::size_t(96) << 20) < target &&
             now - last_vram_change_ > 2.0) {
    // take back what fits under the target (less a 96 MiB margin), up to 8 chunks per check
    std::size_t n = std::min<std::size_t>(8, (target - used - (std::size_t(96) << 20)) / vmm_->chunk());
    bool any = false;
    while (n-- > 0 && vram_grow_one()) any = true;
    if (any) last_vram_change_ = now;
  }
}

void Runtime::set_kv_length(int positions) {
  if (!state_.kv_shared) return;
  if (positions < kv_front_) kv_front_ = std::max(positions, 64);
  ensure_kv(positions);
}

std::size_t Runtime::full_rows_bytes(int n) const {
  if (state_.kv_shared) return std::size_t(std::max(n, 0)) * state_.full_stride();
  if (n < 0 || n > options_.max_ctx)
    throw std::invalid_argument("full_rows_bytes: n = " + std::to_string(n) + " outside [0, max_ctx = " +
                                std::to_string(options_.max_ctx) + "]");
  const std::size_t nfv = std::size_t(num_full_visits());
  const std::size_t row = state_.k_full.bytes() / nfv / std::size_t(options_.max_ctx);
  return 2 * nfv * row * std::size_t(n);
}

// one 2D copy per cache: num_full_visits() rows of n positions, pitch max_ctx positions in the cache, packed in
// the parked copy (k visits, then v visits)
void Runtime::save_full_rows(void* dst, int n) {
  if (state_.kv_shared) {
    CUDA_CHECK(cudaMemcpyAsync(dst, state_.k_full.data, full_rows_bytes(n), cudaMemcpyDefault, stream_));
    return;
  }
  const std::size_t nfv = std::size_t(num_full_visits());
  const std::size_t bytes = full_rows_bytes(n) / 2, width = bytes / nfv;
  if (n == 0) return;
  const std::size_t pitch = state_.k_full.bytes() / nfv;
  char* p = static_cast<char*>(dst);
  for (const Tensor* t : {&state_.k_full, &state_.v_full}) {
    CUDA_CHECK(cudaMemcpy2DAsync(p, width, t->data, pitch, width, int(nfv), cudaMemcpyDefault, stream_));
    p += bytes;
  }
}

void Runtime::restore_full_rows(const void* src, int n) {
  if (state_.kv_shared) {
    ensure_kv(n);
    CUDA_CHECK(cudaMemcpyAsync(state_.k_full.data, src, full_rows_bytes(n), cudaMemcpyDefault, stream_));
    return;
  }
  const std::size_t nfv = std::size_t(num_full_visits());
  const std::size_t bytes = full_rows_bytes(n) / 2, width = bytes / nfv;
  if (n == 0) return;
  const std::size_t pitch = state_.k_full.bytes() / nfv;
  const char* p = static_cast<const char*>(src);
  for (const Tensor* t : {&state_.k_full, &state_.v_full}) {
    CUDA_CHECK(cudaMemcpy2DAsync(t->data, pitch, p, width, width, int(nfv), cudaMemcpyDefault, stream_));
    p += bytes;
  }
}

void Runtime::save_verify(int P, int m) {
  if (verify_widths_.empty()) throw std::logic_error("save_verify needs the speculative widths (options.spec)");
  verify_save(state_ptrs(), backup_, nullptr, P, m, stream_);
}

void Runtime::rollback(int P, int commit, int m) {
  if (commit == m) return;
  if (commit < 1 || commit > m) throw std::invalid_argument("rollback: commit out of range");
  const Buffers& b = buffers(m);
  const float* hist[4] = {ptr<float>(b.hist_k), ptr<float>(b.hist_v), ptr<float>(b.hist_attn), ptr<float>(b.hist_mlp)};
  verify_rollback(state_ptrs(), backup_, hist, nullptr, P, nullptr, commit, m, stream_);
}

void Runtime::record_step(int m, cudaStream_t stream) {
  if (!graphs_.count(m)) throw std::invalid_argument("record_step: no decode graph for width " + std::to_string(m));
  decoder_->decode(buffers(m), pos(), history_.at(m), stream);
}

void Runtime::record_save_verify(int m, cudaStream_t stream) {
  if (verify_widths_.empty()) throw std::logic_error("record_save_verify needs the speculative widths (options.spec)");
  verify_save(state_ptrs(), backup_, pos(), 0, m, stream);
}

void Runtime::record_rollback(int m, const std::int32_t* accepted, cudaStream_t stream) {
  if (!history_.count(m) || !history_.at(m)) throw std::invalid_argument("record_rollback: width without history");
  const Buffers& b = buffers(m);
  const float* hist[4] = {ptr<float>(b.hist_k), ptr<float>(b.hist_v), ptr<float>(b.hist_attn), ptr<float>(b.hist_mlp)};
  verify_rollback(state_ptrs(), backup_, hist, pos(), 0, accepted, 0, m, stream);
}

GraphExec Runtime::capture_switch(const std::function<void(cudaStream_t, cudaGraphConditionalHandle)>& head,
                                  const std::function<void(cudaStream_t, int)>& branch) {
  if (verify_widths_.empty()) throw std::logic_error("capture_switch needs the speculative widths (options.spec)");
  const cudaStream_t s = capture_stream_;
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaGraphCreate(&graph, 0));
  struct Owned {
    cudaGraph_t g;
    ~Owned() { cudaGraphDestroy(g); }
  } owned{graph};
  // run `body` captured into `g` (an exception ends the capture first)
  auto captured = [&](cudaGraph_t g, const std::function<void()>& body) {
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(s, g, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
    try {
      body();
    } catch (...) {
      cudaGraph_t out = nullptr;
      cudaStreamEndCapture(s, &out);
      throw;
    }
    cudaGraph_t out = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(s, &out));
  };

  cudaGraphConditionalHandle handle;
  CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0, 0));
  cudaGraph_t* bodies = nullptr;
  captured(graph, [&] {
    head(s, handle);
    cudaStreamCaptureStatus status;
    cudaGraph_t g = nullptr;
    const cudaGraphNode_t* deps = nullptr;
    const cudaGraphEdgeData* edges = nullptr;
    std::size_t n = 0;
    CUDA_CHECK(cudaStreamGetCaptureInfo(s, &status, nullptr, &g, &deps, &edges, &n));
    cudaGraphNodeParams p{};
    p.type = cudaGraphNodeTypeConditional;
    p.conditional.handle = handle;
    p.conditional.type = cudaGraphCondTypeSwitch;
    p.conditional.size = unsigned(verify_widths_.size());
    cudaGraphNode_t node = nullptr;
    CUDA_CHECK(cudaGraphAddNode(&node, g, deps, edges, n, &p));
    bodies = p.conditional.phGraph_out;
    CUDA_CHECK(cudaStreamUpdateCaptureDependencies(s, &node, nullptr, 1, cudaStreamSetCaptureDependencies));
  });
  for (std::size_t i = 0; i < verify_widths_.size(); ++i)
    captured(bodies[i], [&] { branch(s, verify_widths_[i]); });

  cudaGraphExec_t exec = nullptr;
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  GraphExec result(exec);
  CUDA_CHECK(cudaGraphUpload(exec, stream_));
  return result;
}

}  // namespace ingine::model
