// quill: hybrid routed-expert storage (experts.h).
#include "model/experts.h"

#include "core/device.h"
#include "core/safetensors.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

namespace ingine::model {

ExpertBytes expert_bytes(ExpertFormat fmt) {
  ExpertBytes b;
  const std::size_t r13 = 2 * INTER, r2 = HIDDEN;
  if (fmt == ExpertFormat::NVFP4) {
    b.w13 = r13 * HIDDEN / 2, b.w13s = r13 * HIDDEN / 16;
    b.w2 = r2 * INTER / 2, b.w2s = r2 * INTER / 16;
  } else {
    b.w13 = r13 * HIDDEN, b.w13s = r13 * HIDDEN / 32;
    b.w2 = r2 * INTER, b.w2s = r2 * INTER / 32;
  }
  return b;
}

const std::byte* HostLayerExperts::expert_w13(int e) const { return w13 + std::size_t(e) * expert_bytes(fmt).w13; }
const std::byte* HostLayerExperts::expert_w13s(int e) const { return w13s + std::size_t(e) * expert_bytes(fmt).w13s; }
const std::byte* HostLayerExperts::expert_w2(int e) const { return w2 + std::size_t(e) * expert_bytes(fmt).w2; }
const std::byte* HostLayerExperts::expert_w2s(int e) const { return w2s + std::size_t(e) * expert_bytes(fmt).w2s; }

namespace {

std::vector<float> read_f32(const SafetensorsModel& m, const std::string& name) {
  const SafetensorsRef ref = m.at(name);
  const std::vector<std::byte> raw = m.read(name);
  std::vector<float> out(std::size_t(ref->numel()));
  if (ref->dtype == "F32") {
    std::memcpy(out.data(), raw.data(), out.size() * 4);
  } else if (ref->dtype == "BF16") {
    for (std::size_t i = 0; i < out.size(); ++i) {
      std::uint16_t b;
      std::memcpy(&b, raw.data() + 2 * i, 2);
      const std::uint32_t w = std::uint32_t(b) << 16;
      std::memcpy(&out[i], &w, 4);
    }
  } else {
    throw std::runtime_error("experts: " + name + " has unsupported dtype " + ref->dtype);
  }
  return out;
}


// ---- MXFP8 -> NVFP4 re-encoding (quill option): per expert matrix, factor f = amax / (6 * 448); per block of 16 the
// E4M3 block scale (in units of f) is chosen among the three E4M3 values around amax_block / 6 by squared error,
// codes are the nearest E2M1 values. Rows keep the checkpoint's NVFP4 layout (even element in the low nibble).
float e4m3_value(int b) {
  const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  float v = e == 0 ? std::ldexp(float(m) / 8.f, -6) : std::ldexp(1.f + float(m) / 8.f, e - 7);
  if (e == 15 && m == 7) v = 0.f;
  return s ? -v : v;
}
int e4m3_nearest(float x) {  // non-negative x, saturating at 448
  static const std::vector<float> tab = [] {
    std::vector<float> t(127);
    for (int b = 0; b < 127; ++b) t[b] = e4m3_value(b);
    return t;
  }();
  auto it = std::lower_bound(tab.begin(), tab.end(), x);
  if (it == tab.end()) return 126;
  const int i = int(it - tab.begin());
  if (i > 0 && x - tab[i - 1] < tab[i] - x) return i - 1;
  return i;
}
constexpr float kE2M1v[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
inline int e2m1_nearest(float a) {  // |value| -> code 0..7
  int best = 0;
  float bd = std::fabs(a);
  for (int k = 1; k < 8; ++k) {
    const float d = std::fabs(a - kE2M1v[k]);
    if (d < bd) bd = d, best = k;
  }
  return best;
}
// rows x cols MXFP8 (codes [rows][cols], E8M0 [rows][cols/32]) -> NVFP4 codes [rows][cols/2], E4M3 [rows][cols/16]
float requant_matrix(const std::uint8_t* c8, const std::uint8_t* s8, int rows, int cols, std::uint8_t* c4, std::uint8_t* s4) {
  std::vector<float> x(std::size_t(rows) * cols);
  float amax = 0.f;
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) {
      const int se = s8[std::size_t(r) * (cols / 32) + c / 32];
      const float sc = std::ldexp(1.f, se - 127);
      const float v = e4m3_value(c8[std::size_t(r) * cols + c]) * sc;
      x[std::size_t(r) * cols + c] = v;
      amax = std::max(amax, std::fabs(v));
    }
  const float f = amax > 0.f ? amax / (6.f * 448.f) : 1.f;
  for (int r = 0; r < rows; ++r)
    for (int b = 0; b < cols / 16; ++b) {
      const float* xb = &x[std::size_t(r) * cols + 16 * b];
      float bmax = 0.f;
      for (int j = 0; j < 16; ++j) bmax = std::max(bmax, std::fabs(xb[j]));
      const int c0 = e4m3_nearest(bmax / 6.f / f);
      int best_s = c0;
      float best_err = 1e30f;
      for (int cand = std::max(0, c0 - 1); cand <= std::min(126, c0 + 1); ++cand) {
        const float s = e4m3_value(cand) * f;
        if (s <= 0.f) continue;
        float err = 0.f;
        for (int j = 0; j < 16; ++j) {
          const float q = kE2M1v[e2m1_nearest(std::fabs(xb[j]) / s)] * s;
          const float d = std::fabs(xb[j]) - q;
          err += d * d;
        }
        if (err < best_err) best_err = err, best_s = cand;
      }
      const float s = e4m3_value(best_s) * f;
      s4[std::size_t(r) * (cols / 16) + b] = std::uint8_t(best_s);
      auto code = [&](float v) {
        if (s <= 0.f) return 0;
        const int k = e2m1_nearest(std::fabs(v) / s);
        return (v < 0.f && k) ? (k | 8) : k;
      };
      for (int j = 0; j < 16; j += 2)
        c4[std::size_t(r) * (cols / 2) + 8 * b + j / 2] = std::uint8_t(code(xb[j]) | (code(xb[j + 1]) << 4));
    }
  return f;
}

}  // namespace

void ExpertStore::load_host(const std::filesystem::path& model_dir, const std::array<ExpertFormat, LAYERS>& formats,
                            bool mxfp8_as_nvfp4) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::array<ExpertFormat, LAYERS> file_formats = formats;
  formats_ = formats;
  if (mxfp8_as_nvfp4) formats_.fill(ExpertFormat::NVFP4);
  const std::array<ExpertFormat, LAYERS>& target = formats_;
  host_bytes_ = 0;
  for (int l = 0; l < LAYERS; ++l) host_bytes_ += expert_bytes(target[l]).total() * EXPERTS;
  region_ = HostRegion(host_bytes_);
  const SafetensorsModel m(model_dir);
  std::byte* p = region_.data();
  struct Job {
    const SafetensorsFile* file;
    std::uint64_t offset, bytes;
    std::byte* dst;
  };
  std::vector<Job> jobs;
  std::vector<int> convert;  // layers re-encoded after the reads
  for (int l = 0; l < LAYERS; ++l) {
    HostLayerExperts& h = host_[l];
    h.fmt = target[l];
    if (file_formats[l] != target[l]) {  // MXFP8 in the file, NVFP4 in memory: filled by the conversion below
      const ExpertBytes eb = expert_bytes(h.fmt);
      h.w13 = p, p += eb.w13 * EXPERTS;
      h.w13s = p, p += eb.w13s * EXPERTS;
      h.w2 = p, p += eb.w2 * EXPERTS;
      h.w2s = p, p += eb.w2s * EXPERTS;
      convert.push_back(l);
      continue;
    }
    const ExpertBytes eb = expert_bytes(h.fmt);
    const std::string e = "model.llm.layers." + std::to_string(l) + ".mlp.experts.";
    std::byte** dsts[4] = {&h.w13, &h.w13s, &h.w2, &h.w2s};
    const char* names[4] = {"w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"};
    const std::size_t sizes[4] = {eb.w13, eb.w13s, eb.w2, eb.w2s};
    for (int i = 0; i < 4; ++i) {
      const SafetensorsRef ref = m.at(e + names[i]);
      if (ref->bytes != sizes[i] * EXPERTS)
        throw std::runtime_error("experts: " + e + names[i] + " has " + std::to_string(ref->bytes) + " bytes, expected " +
                                 std::to_string(sizes[i] * EXPERTS));
      *dsts[i] = p;
      // split into 64 MiB reads for the reader threads
      for (std::uint64_t off = 0; off < ref->bytes; off += (64ull << 20)) {
        const std::uint64_t n = std::min<std::uint64_t>(64ull << 20, ref->bytes - off);
        jobs.push_back({ref.file, ref->offset + off, n, p + off});
      }
      p += ref->bytes;
    }
    if (h.fmt == ExpertFormat::NVFP4) {
      for (const char* w : {"w13", "w2"}) {
        const std::vector<float> amax = read_f32(m, e + w + "_weight_amax");
        const std::vector<float> den = read_f32(m, e + w + "_weight_outer_scale_denominator");
        auto& f = std::string(w) == "w13" ? h.f13 : h.f2;
        for (int i = 0; i < EXPERTS; ++i) f[i] = amax[i] / den[den.size() == 1 ? 0 : i];
      }
    } else {
      h.f13.fill(1.f), h.f2.fill(1.f);
    }
  }
  // parallel positional reads straight into the region
  std::atomic<std::size_t> next{0};
  std::vector<std::thread> th;
  for (int t = 0; t < 6; ++t)
    th.emplace_back([&] {
      for (std::size_t j; (j = next.fetch_add(1)) < jobs.size();) jobs[j].file->read_range(jobs[j].offset, jobs[j].bytes, jobs[j].dst);
    });
  for (auto& t : th) t.join();
  // MXFP8 -> NVFP4, one layer at a time (its MXFP8 tensors read into a temporary buffer), experts in parallel
  for (int l : convert) {
    HostLayerExperts& h = host_[l];
    const std::string e = "model.llm.layers." + std::to_string(l) + ".mlp.experts.";
    const std::vector<std::byte> w13 = m.read(e + "w13_weight"), w13s = m.read(e + "w13_weight_scale");
    const std::vector<std::byte> w2 = m.read(e + "w2_weight"), w2s = m.read(e + "w2_weight_scale");
    const ExpertBytes src = expert_bytes(ExpertFormat::MXFP8), dst = expert_bytes(ExpertFormat::NVFP4);
    std::atomic<int> nx{0};
    std::vector<std::thread> ct;
    auto u8 = [](const std::vector<std::byte>& v, std::size_t off) {
      return reinterpret_cast<const std::uint8_t*>(v.data()) + off;
    };
    for (int t = 0; t < 7; ++t)
      ct.emplace_back([&] {
        for (int x; (x = nx.fetch_add(1)) < EXPERTS;) {
          h.f13[x] = requant_matrix(u8(w13, x * src.w13), u8(w13s, x * src.w13s), 2 * INTER, HIDDEN,
                                    reinterpret_cast<std::uint8_t*>(h.w13 + x * dst.w13),
                                    reinterpret_cast<std::uint8_t*>(h.w13s + x * dst.w13s));
          h.f2[x] = requant_matrix(u8(w2, x * src.w2), u8(w2s, x * src.w2s), HIDDEN, INTER,
                                   reinterpret_cast<std::uint8_t*>(h.w2 + x * dst.w2),
                                   reinterpret_cast<std::uint8_t*>(h.w2s + x * dst.w2s));
        }
      });
    for (auto& t : ct) t.join();
  }
  if (!convert.empty()) spdlog::info("experts: {} MXFP8 layers re-encoded as NVFP4", convert.size());
  const double read_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const bool pinned = region_.register_with_cuda();
  spdlog::info("experts: {:.2f} GiB in host memory ({:.1f} s read, {} in {:.1f} s)", host_bytes_ / double(1ull << 30),
               read_s, pinned ? "pinned" : "NOT pinned", region_.register_seconds());
}

std::vector<ExpertStore::Pair> ExpertStore::default_priority(const std::array<ExpertFormat, LAYERS>& formats) {
  std::vector<Pair> out;
  // loop layers (three visits per token) first, NVFP4 before MXFP8, then the rest by layer
  for (int pass = 0; pass < 4; ++pass)
    for (int l = 0; l < LAYERS; ++l) {
      const bool loop = is_loop_layer(l), nv = formats[l] == ExpertFormat::NVFP4;
      const int cls = loop ? (nv ? 0 : 1) : (nv ? 2 : 3);
      if (cls != pass) continue;
      for (int e = 0; e < EXPERTS; ++e) out.push_back({l, e});
    }
  return out;
}

void ExpertStore::build_pools(std::size_t budget_bytes, int staging, std::vector<Pair> priority, cudaStream_t stream) {
  if (priority.empty()) priority = default_priority(formats_);
  for (auto& row : host_map_) row.fill(-1);
  // greedy fill by priority within the byte budget
  std::array<int, 2> resident{0, 0};
  std::vector<std::pair<Pair, int>> placed;
  std::size_t used = 0;
  std::array<bool, 2> has_fmt{false, false};
  for (int l = 0; l < LAYERS; ++l) has_fmt[int(formats_[l])] = true;
  std::size_t staging_bytes = 0;
  for (int f = 0; f < 2; ++f)
    if (has_fmt[f]) staging_bytes += std::size_t(staging) * expert_bytes(ExpertFormat(f)).total();
  for (const Pair& pr : priority) {
    const ExpertFormat f = formats_[pr.layer];
    const std::size_t b = expert_bytes(f).total();
    if (used + b + staging_bytes > budget_bytes) continue;  // smaller formats may still fit
    used += b;
    host_map_[pr.layer][pr.expert] = resident[int(f)]++;
    placed.push_back({pr, host_map_[pr.layer][pr.expert]});
  }
  for (int f = 0; f < 2; ++f) {
    ExpertPool& P = pools_[f];
    P.fmt = ExpertFormat(f);
    P.staging = has_fmt[f] ? staging : 0;
    P.slots = resident[f] + P.staging;
    const ExpertBytes eb = expert_bytes(P.fmt);
    const std::size_t n = std::max(P.slots, 1);
    P.w13 = DeviceBuffer(n * eb.w13);
    P.w13s = DeviceBuffer(n * eb.w13s);
    P.w2 = DeviceBuffer(n * eb.w2);
    P.w2s = DeviceBuffer(n * eb.w2s);
    P.f13 = DeviceBuffer(n * 4);
    P.f2 = DeviceBuffer(n * 4);
  }
  for (const auto& [pr, slot] : placed) upload(pr.layer, pr.expert, slot, stream);
  for (int f = 0; f < 2; ++f) owners_[f].assign(std::size_t(std::max(pools_[f].slots, 1)), -1);
  for (const auto& [pr, slot] : placed) owners_[int(formats_[pr.layer])][slot] = pr.layer * EXPERTS + pr.expert;
  if (!ring_) CUDA_CHECK(cudaMallocHost(&ring_, 4096 * sizeof(std::int32_t)));
  slot_map_ = DeviceBuffer(sizeof(std::int32_t) * LAYERS * EXPERTS);
  CUDA_CHECK(cudaMemcpyAsync(slot_map_.p, host_map_.data(), slot_map_.bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  spdlog::info("experts: {} resident on the GPU ({:.2f} GiB; NVFP4 {} + MXFP8 {} slots, {} staging each), {} on the CPU",
               resident[0] + resident[1], used / double(1ull << 30), resident[0], resident[1], staging,
               LAYERS * EXPERTS - resident[0] - resident[1]);
}

void ExpertStore::build_pools_shared(char* region, std::size_t bytes, std::size_t kv_max, int staging,
                                     std::vector<Pair> priority, cudaStream_t stream) {
  for (int l = 0; l < LAYERS; ++l)
    if (formats_[l] != ExpertFormat::NVFP4) throw std::runtime_error("experts: a shared region needs NVFP4 experts only");
  if (priority.empty()) priority = default_priority(formats_);
  dev_region_ = region;
  dev_region_bytes_ = bytes;
  ExpertPool& P = pools_[int(ExpertFormat::NVFP4)];
  const ExpertBytes eb = expert_bytes(ExpertFormat::NVFP4);
  P.fmt = ExpertFormat::NVFP4;
  P.stride = (long long)((eb.total() + 255) / 256 * 256);
  P.off13s = eb.w13, P.off2 = eb.w13 + eb.w13s, P.off2s = eb.w13 + eb.w13s + eb.w2;
  P.slots = int(bytes / std::size_t(P.stride));
  P.staging = staging;
  // slot records start 256-byte aligned (the budget is not a multiple of the stride)
  P.base = region + ((bytes - std::size_t(P.slots) * std::size_t(P.stride)) & ~std::size_t(255));
  // the staging slots (top of the region) must stay above the largest KV front
  if (P.base + std::size_t(P.slots - staging) * std::size_t(P.stride) < region + kv_max)
    throw std::runtime_error("experts: the shared region cannot hold the KV cache of max_ctx and the staging slots");
  P.f13 = DeviceBuffer(std::size_t(P.slots) * 4);
  P.f2 = DeviceBuffer(std::size_t(P.slots) * 4);
  pools_[int(ExpertFormat::MXFP8)] = ExpertPool{};
  pools_[int(ExpertFormat::MXFP8)].fmt = ExpertFormat::MXFP8;
  for (auto& row : host_map_) row.fill(-1);
  owners_[0].assign(std::size_t(P.slots), -1);
  owners_[1].assign(1, -1);
  int n = 0;
  for (const Pair& pr : priority) {
    if (n >= P.resident_slots()) break;
    host_map_[pr.layer][pr.expert] = n;
    owners_[0][n] = pr.layer * EXPERTS + pr.expert;
    upload(pr.layer, pr.expert, n, stream);
    ++n;
  }
  slot_map_ = DeviceBuffer(sizeof(std::int32_t) * LAYERS * EXPERTS);
  CUDA_CHECK(cudaMemcpyAsync(slot_map_.p, host_map_.data(), slot_map_.bytes, cudaMemcpyHostToDevice, stream));
  if (!ring_) CUDA_CHECK(cudaMallocHost(&ring_, 4096 * sizeof(std::int32_t)));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  spdlog::info("experts: shared KV/expert region {:.2f} GiB: {} slots ({} staging), {} resident, KV up to {:.2f} GiB",
               bytes / double(1ull << 30), P.slots, staging, n, kv_max / double(1ull << 30));
}

int ExpertStore::first_usable(std::size_t front_bytes) const {
  const ExpertPool& P = pools_[int(ExpertFormat::NVFP4)];
  if (!dev_region_ || P.stride == 0) return 0;
  const char* front = dev_region_ + front_bytes;
  if (P.base >= front) return 0;
  return int((front - P.base + P.stride - 1) / P.stride);
}

int ExpertStore::evict_below(std::size_t front_bytes, cudaStream_t stream) {
  if (!dev_region_) return 0;
  const int first = std::min(first_usable(front_bytes), pools_[0].slots);
  int n = 0;
  for (int s = 0; s < first; ++s) {
    const int o = owners_[0][s];
    if (o >= 0) set_slot(o / EXPERTS, o % EXPERTS, -1, stream), ++n;
  }
  return n;
}

// the per-expert NVFP4 factors as pinned sources: a 4-byte copy from pageable memory makes cudaMemcpyAsync
// synchronous and costs driver memory per copy
const float* ExpertStore::pinned_factors(int l, int e) {
  if (!fac_) {
    CUDA_CHECK(cudaMallocHost(&fac_, sizeof(float) * 2 * LAYERS * EXPERTS));
    for (int ll = 0; ll < LAYERS; ++ll)
      for (int ee = 0; ee < EXPERTS; ++ee) {
        fac_[(ll * EXPERTS + ee) * 2] = host_[ll].f13[ee];
        fac_[(ll * EXPERTS + ee) * 2 + 1] = host_[ll].f2[ee];
      }
  }
  return fac_ + (l * EXPERTS + e) * 2;
}

const float* ExpertStore::pinned_factors_device(int l) {
  float* h = const_cast<float*>(pinned_factors(l, 0));
  void* d = nullptr;
  if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) return nullptr;
  return static_cast<const float*>(d);
}

void ExpertStore::upload(int l, int e, int slot, cudaStream_t stream, bool factors) const {
  const HostLayerExperts& h = host_[l];
  const float* fac = const_cast<ExpertStore*>(this)->pinned_factors(l, e);
  const ExpertPool& P = pools_[int(h.fmt)];
  const ExpertBytes eb = expert_bytes(h.fmt);
  if (P.stride) {  // shared region: one record per slot
    char* d = P.base + std::size_t(slot) * std::size_t(P.stride);
    CUDA_CHECK(cudaMemcpyAsync(d, h.expert_w13(e), eb.w13, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d + P.off13s, h.expert_w13s(e), eb.w13s, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d + P.off2, h.expert_w2(e), eb.w2, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d + P.off2s, h.expert_w2s(e), eb.w2s, cudaMemcpyHostToDevice, stream));
    if (factors) {
      CUDA_CHECK(cudaMemcpyAsync(static_cast<float*>(P.f13.p) + slot, fac, 4, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(static_cast<float*>(P.f2.p) + slot, fac + 1, 4, cudaMemcpyHostToDevice, stream));
    }
    return;
  }
  auto cp = [&](const DeviceBuffer& dst, std::size_t sz, const std::byte* src) {
    CUDA_CHECK(cudaMemcpyAsync(static_cast<char*>(dst.p) + std::size_t(slot) * sz, src, sz, cudaMemcpyHostToDevice, stream));
  };
  cp(P.w13, eb.w13, h.expert_w13(e));
  cp(P.w13s, eb.w13s, h.expert_w13s(e));
  cp(P.w2, eb.w2, h.expert_w2(e));
  cp(P.w2s, eb.w2s, h.expert_w2s(e));
  cp(P.f13, 4, reinterpret_cast<const std::byte*>(fac));
  cp(P.f2, 4, reinterpret_cast<const std::byte*>(fac + 1));
}

ExpertStore::~ExpertStore() {
  if (ring_) cudaFreeHost(ring_);
  if (fac_) cudaFreeHost(fac_);
}

void ExpertStore::set_slot(int l, int e, int slot, cudaStream_t stream) {
  const int f = int(formats_[l]);
  const int old = host_map_[l][e];
  if (old >= 0 && owners_[f][old] == l * EXPERTS + e) owners_[f][old] = -1;
  host_map_[l][e] = slot;
  if (slot >= 0) owners_[f][slot] = l * EXPERTS + e;
  std::int32_t* src = ring_ + ring_pos_;
  ring_pos_ = (ring_pos_ + 1) & 4095;
  *src = slot;
  CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(slot_map_.p) + l * EXPERTS + e, src, sizeof(std::int32_t),
                             cudaMemcpyHostToDevice, stream));
}

int ExpertStore::resident_count(int l) const {
  int n = 0;
  for (int e = 0; e < EXPERTS; ++e) n += host_map_[l][e] >= 0;
  return n;
}

int ExpertStore::resident_total() const {
  int n = 0;
  for (int l = 0; l < LAYERS; ++l) n += resident_count(l);
  return n;
}

std::size_t ExpertStore::device_bytes() const {
  std::size_t b = slot_map_.bytes;
  for (const ExpertPool& P : pools_) b += P.w13.bytes + P.w13s.bytes + P.w2.bytes + P.w2s.bytes + P.f13.bytes + P.f2.bytes;
  return b;
}

}  // namespace ingine::model
