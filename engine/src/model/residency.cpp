// quill: adaptive expert residency. Which experts live in the GPU slot pool follows the conversation's routing:
// decayed per-(layer, expert) selection counts from the prompt's prefill and from the decode steps, with bounded
// swaps of the coldest resident experts for the hottest non-resident ones.
//
// A swap never stalls decoding: the victim is unmapped first (stream-ordered map update: later steps send it to
// the CPU tier), its slot is refilled on a separate copy stream that waits for that point, and the new expert is
// mapped on a later call once the copy's event has completed. Decode steps only ever see fully written slots.
#include "model/cpu_tier.h"
#include "model/experts.h"
#include "model/runtime.h"

#include "core/device.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <vector>

namespace ingine::model {

struct Runtime::Residency {
  std::vector<double> score = std::vector<double>(LAYERS * EXPERTS, 0.0);
  std::uint32_t* counts_host[2] = {nullptr, nullptr};  // pinned [24 * 128], double-buffered
  cudaEvent_t counts_ready[2] = {nullptr, nullptr};
  bool counts_pending[2] = {false, false};
  int which = 0;
  cudaStream_t copy = nullptr;
  struct Pending {
    int l, e, slot;
    cudaEvent_t done;
  };
  std::vector<Pending> pending;
  std::vector<bool> busy;  // slots with a copy in flight
  std::uint64_t swaps = 0;
  double decay = 0.96;     // per call (every 16 tokens: half-life ~270 tokens)
  double margin = 1.25, floor = 2.0;

  ~Residency() {
    for (auto& p : pending) cudaEventDestroy(p.done);
    for (int i = 0; i < 2; ++i) {
      if (counts_host[i]) cudaFreeHost(counts_host[i]);
      if (counts_ready[i]) cudaEventDestroy(counts_ready[i]);
    }
    if (copy) cudaStreamDestroy(copy);
  }
};

void Runtime::ResidencyDeleter::operator()(Residency* r) const { delete r; }

void Runtime::residency_init() {
  res_.reset(new Residency());
  if (const char* d = std::getenv("QUILL_RES_DECAY")) res_->decay = std::atof(d);    // experiment knobs
  if (const char* m = std::getenv("QUILL_RES_MARGIN")) res_->margin = std::atof(m);
  for (int i = 0; i < 2; ++i) {
    CUDA_CHECK(cudaMallocHost(&res_->counts_host[i], LAYERS * EXPERTS * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaEventCreateWithFlags(&res_->counts_ready[i], cudaEventDisableTiming));
  }
  CUDA_CHECK(cudaStreamCreateWithFlags(&res_->copy, cudaStreamNonBlocking));
  // the pool's initial set (loop layers first) gets a small prior so that it is only displaced by evidence
  const ExpertStore& es = *weights_.experts;
  for (int l = 0; l < LAYERS; ++l)
    for (int e = 0; e < EXPERTS; ++e)
      if (es.host_slot(l, e) >= 0) res_->score[l * EXPERTS + e] = 1.0;
  res_->busy.assign(std::size_t(std::max(es.pool(ExpertFormat::NVFP4).slots, es.pool(ExpertFormat::MXFP8).slots)), false);
}

void Runtime::residency_observe(const std::uint32_t* counts, double weight) {
  if (!res_) return;
  for (int i = 0; i < LAYERS * EXPERTS; ++i) res_->score[i] += weight * counts[i];
}

void Runtime::residency_step(int max_swaps, bool sample_decode) {
  vram_check();  // quill: dynamic VRAM
  if (!res_ || !options_.adaptive_experts) return;
  Residency& r = *res_;
  ExpertStore& es = *weights_.experts;
  // 1. finished copies: map their experts (stream-ordered, ahead of the next decode step)
  for (auto it = r.pending.begin(); it != r.pending.end();) {
    if (cudaEventQuery(it->done) == cudaSuccess) {
      es.set_slot(it->l, it->e, it->slot, stream_);
      r.busy[it->slot] = false;
      cudaEventDestroy(it->done);
      it = r.pending.erase(it);
    } else {
      (void)cudaGetLastError();
      ++it;
    }
  }
  // 2. decode routing counts: take the interval's counts (async D2H + reset), fold in the previous interval's
  if (sample_decode) {
    const int w = r.which;
    CUDA_CHECK(cudaMemcpyAsync(r.counts_host[w], route_counts_.p, LAYERS * EXPERTS * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemsetAsync(route_counts_.p, 0, route_counts_.bytes, stream_));
    CUDA_CHECK(cudaEventRecord(r.counts_ready[w], stream_));
    r.counts_pending[w] = true;
    r.which ^= 1;
    const int o = r.which;  // the other buffer: the previous interval
    if (r.counts_pending[o] && cudaEventQuery(r.counts_ready[o]) == cudaSuccess) {
      for (double& s : r.score) s *= r.decay;
      residency_observe(r.counts_host[o], 1.0);
      r.counts_pending[o] = false;
    }
    (void)cudaGetLastError();
  }
  if (max_swaps <= 0) return;
  // 3. plan: hottest non-resident vs coldest resident (per format pool)
  for (int f = 0; f < 2; ++f) {
    const ExpertPool& P = es.pool(ExpertFormat(f));
    const int nres = P.resident_slots();
    if (nres <= 0) continue;
    std::vector<int> victims;  // slots, coldest first (shared region: only those above the KV front + 512 rows)
    const int first = f == 0 ? es.first_usable(std::max(std::size_t(kv_front_ + 512) * state_.full_stride(), hole_hi_)) : 0;
    const int vend = (f == 0 && lent_end_ == P.slots && lent_end_ > lent_first_) ? P.slots : nres;  // + staging
    for (int s = first; s < vend; ++s)
      if (!r.busy[s]) victims.push_back(s);
    auto slot_score = [&](int s) {
      const int o = es.slot_owner(ExpertFormat(f), s);
      return o < 0 ? -1.0 : r.score[o];
    };
    std::sort(victims.begin(), victims.end(), [&](int a, int b) { return slot_score(a) < slot_score(b); });
    std::vector<int> cands;  // layer * 128 + expert, hottest first
    for (int l = 0; l < LAYERS; ++l) {
      if (int(es.formats()[l]) != f) continue;
      for (int e = 0; e < EXPERTS; ++e)
        if (es.host_slot(l, e) < 0) cands.push_back(l * EXPERTS + e);
    }
    std::partial_sort(cands.begin(), cands.begin() + std::min<std::size_t>(cands.size(), std::size_t(max_swaps)), cands.end(),
                      [&](int a, int b) { return r.score[a] > r.score[b]; });
    // candidates already being copied in are excluded
    auto in_flight = [&](int id) {
      for (const auto& p : r.pending)
        if (p.l * EXPERTS + p.e == id) return true;
      return false;
    };
    int done = 0;
    std::size_t vi = 0;
    for (std::size_t ci = 0; ci < cands.size() && done < max_swaps && vi < victims.size(); ++ci) {
      const int id = cands[ci];
      if (in_flight(id)) continue;
      const int slot = victims[vi];
      const double vs = slot_score(slot);
      if (!(r.score[id] > vs * r.margin + r.floor)) break;
      ++vi;
      const int owner = es.slot_owner(ExpertFormat(f), slot);
      if (owner >= 0) es.set_slot(owner / EXPERTS, owner % EXPERTS, -1, stream_);  // to the CPU tier from here on
      cudaEvent_t unmapped, copied;
      CUDA_CHECK(cudaEventCreateWithFlags(&unmapped, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventRecord(unmapped, stream_));
      CUDA_CHECK(cudaStreamWaitEvent(r.copy, unmapped, 0));
      es.upload(id / EXPERTS, id % EXPERTS, slot, r.copy);
      CUDA_CHECK(cudaEventRecord(copied, r.copy));
      cudaEventDestroy(unmapped);
      r.busy[slot] = true;
      r.pending.push_back({id / EXPERTS, id % EXPERTS, slot, copied});
      ++done;
      ++r.swaps;
    }
  }
}

std::uint64_t Runtime::residency_swaps() const { return res_ ? res_->swaps : 0; }

void* Runtime::lend_block(std::size_t bytes) {
  ExpertStore& es = *weights_.experts;
  const ExpertPool& P = es.pool(ExpertFormat::NVFP4);
  if (!state_.kv_shared || P.stride <= 0 || bytes == 0) return nullptr;
  const int n = int((bytes + std::size_t(P.stride) - 1) / std::size_t(P.stride));
  const int first = P.resident_slots() - n;
  const std::size_t kv_max = std::size_t(options_.max_ctx + 512) * state_.full_stride();
  if (first < 0 || es.first_usable(kv_max) > first) return nullptr;
  // quill: the staging slots above join the lent range: resident experts during decode, emptied with the scratch
  // for a prefill (its staging groups use them)
  lent_first_ = first, lent_end_ = P.slots;
  spdlog::info("experts: prefill scratch {:.0f} MiB lent from expert slots [{}, {}) (resident during decode)",
               bytes / 1048576.0, lent_first_, lent_end_);
  return P.base + std::size_t(first) * std::size_t(P.stride);
}

void Runtime::scratch_acquire() {
  if (lent_first_ >= lent_end_ || lent_acquired_) return;
  residency_flush();  // copies in flight land (and are mapped) before their slots are taken
  ExpertStore& es = *weights_.experts;
  lent_owners_.clear();
  for (int s = lent_first_; s < lent_end_; ++s) {
    const int o = es.slot_owner(ExpertFormat::NVFP4, s);
    if (o >= 0) {
      es.set_slot(o / EXPERTS, o % EXPERTS, -1, stream_);  // stream-ordered: the prefill sends it to staging
      lent_owners_.push_back({o, s});
    }
    if (res_) res_->busy[s] = true;  // no swap targets these slots while they hold the scratch
  }
  lent_acquired_ = true;
}

void Runtime::scratch_release() {
  if (!lent_acquired_) return;
  ExpertStore& es = *weights_.experts;
  for (const auto& [o, s] : lent_owners_) {
    const int l = o / EXPERTS, e = o % EXPERTS;
    if (es.host_slot(l, e) >= 0) continue;
    es.upload(l, e, s, stream_);  // after the prefill's last reads of the scratch (same stream)
    es.set_slot(l, e, s, stream_);
  }
  if (res_)
    for (int s = lent_first_; s < lent_end_; ++s) res_->busy[s] = false;
  lent_owners_.clear();
  lent_acquired_ = false;
}

void Runtime::residency_flush() {
  if (!res_) return;
  CUDA_CHECK(cudaStreamSynchronize(res_->copy));
  residency_step(0, false);
}

}  // namespace ingine::model
