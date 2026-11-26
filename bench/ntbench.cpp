// ntbench.cpp: SPEEDUP-IDEAS.md item 3 gate microbenchmark. No CUDA, no GPU, no engine.
//
// Question: on a Ryzen 7 5800X3D (Zen 3, 96 MiB L3 that is a VICTIM cache of L2), can the
// CPU tier stop non-reusable DRAM streams from evicting a small "keep set" (the misses of
// loop layers that passes 2/3 will re-read) out of L3, using x86 non-temporal load hints
// on ordinary write-back memory?
//
//   A  baseline: 60 MiB keep-set read, 7 threads, 3 reads          -> DRAM speed vs L3 speed
//   B  evict control: warm keep -> ~300 MiB plain stream -> re-read keep
//   C  like B, stream read via _mm_stream_load_si128 (MOVNTDQA on WB memory)
//   D  like B, stream preceded by _mm_prefetch(_MM_HINT_NTA) per 64B (real loads stay plain)
//   E  like B, but the keep set is re-touched between 7 MiB stream slices (fallback design):
//      after each 7 MiB granule, ONE thread re-reads a 4 MiB keep window (15 windows tile it)
//   F  mismatched-DIMM range scan: 12 GiB VirtualAlloc'ed, committed and touched, read in
//      2 MiB slices with 4 threads (disjoint quarters of each slice), per-slice GB/s stats.
//
// Consumer everywhere: AVX2 loads accumulated to one uint64 per phase (the MOVNTDQA path
// zero-extend adds the 4 dwords of each 128-bit result), folded into a global sink that is
// printed at the end, so the optimizer cannot dead-code-eliminate any read. The load
// primitive is _mm256_loadu_ps reinterpreted as integer (identical 32-byte load encoding),
// per the task's "_mm256_loadu_ps / uint64 sums" consumer spec.
//
// Timing: std::chrono::steady_clock. Workers: 7 threads pinned to logical cores
// 2,4,6,8,10,12,14 (the CPU tier's cores) at THREAD_PRIORITY_HIGHEST; main pins to logical 0.
// All per-thread byte splits below are 64B-granular, adjacent and non-overlapping, and every
// buffer is 64B-aligned, so MOVNTDQA's 16B alignment requirement always holds.
//
// Item-3 gate: keep-set re-read in C or D at least 30% faster than in B.
//
// Build: bench\ntbench.cmd (MSVC x64 /O2 /arch:AVX2). Run: ntbench.exe [--reps 5] [--only abcdef]
// F writes per-slice data to ntbench_f_slices.csv next to the exe.
#define NOMINMAX
#include <immintrin.h>
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#ifndef __AVX2__
#error "ntbench.cpp must be built with /arch:AVX2 (bench\\ntbench.cmd does)"
#endif

namespace {

// ---------------------------------------------------------------- configuration
using clk = std::chrono::steady_clock;

constexpr int NW = 7;                                     // CPU-tier worker threads
constexpr int NW_LOGICAL[NW] = {2, 4, 6, 8, 10, 12, 14};  // the tier's logical cores
constexpr int FT = 4;                                     // experiment-F scan threads
constexpr int FT_LOGICAL[FT] = {2, 6, 10, 14};            // 4 of the same physical cores

constexpr std::uint64_t KBYTES = 60ull << 20;              // keep-set: 60 MiB
constexpr std::uint64_t GSIZE = 7ull << 20;                // stream granule for E: 7 MiB
constexpr int NGRAN = 43;                                  // 43 x 7 MiB = 300.75 MiB ("300 MB")
constexpr std::uint64_t SBYTES = std::uint64_t(NGRAN) * GSIZE;
constexpr std::uint64_t WBYTES = 4ull << 20;               // E refresh window: 4 MiB
constexpr int NWIN = int(KBYTES / WBYTES);                 // 15 windows tile the keep-set
constexpr std::uint64_t FBYTES = 12ull << 30;             // F: 12 GiB
constexpr std::uint64_t FSLICE = 2ull << 20;              // 2 MiB slices -> 6144
constexpr int FNSLICE = int(FBYTES / FSLICE);
constexpr std::uint64_t PROBE_BYTES = 1ull << 20;         // MOVNTDQA behaviour probe size

enum Cmd { KEEP_READ = 0, S_PLAIN, S_NT, S_NTA, S_MIX };
enum FCmd { F_TOUCH = 0, F_SLICE };

// ---------------------------------------------------------------- state
std::uint8_t* g_keep = nullptr;    // 60 MiB keep-set, 4K-aligned
std::uint8_t* g_stream = nullptr;  // ~300 MiB stream, 4K-aligned
std::uint8_t* g_fbuf = nullptr;    // F: 12 GiB via VirtualAlloc

std::atomic<int> g_cmd{0}, g_fcmd{0}, g_fslice{0};
std::atomic<std::uint32_t> g_seq{0}, g_ack{0}, g_fseq{0}, g_fack{0};
std::atomic<bool> g_stop{false}, g_fstop{false};
std::atomic<std::int64_t> g_e_refresh_ns{0};  // experiment E: thread-0 refresh time

struct Barrier {
  const int n;
  std::atomic<std::uint32_t> gen{0}, arrive{0};
  explicit Barrier(int n_) : n(n_) {}
  void wait() {
    const std::uint32_t g = gen.load(std::memory_order_acquire);
    if (arrive.fetch_add(1, std::memory_order_acq_rel) == std::uint32_t(n - 1)) {
      arrive.store(0, std::memory_order_relaxed);
      gen.store(g + 1, std::memory_order_release);
    } else {
      while (gen.load(std::memory_order_acquire) == g) _mm_pause();
    }
  }
};
Barrier g_bar{NW}, g_fbar{FT};

std::uint64_t g_sink[NW + FT];         // anti-DCE sink, printed at exit
std::uint64_t g_phase_sum[NW];         // per-thread sums, read by main right after an ack
std::uint64_t g_expected_nt[NW];       // per-thread dword-style sums of the stream (C check)

int g_reps = 5;
std::string g_only = "abcdef";
bool wants(char c) { return g_only.find(c) != std::string::npos; }

// ---------------------------------------------------------------- helpers
// Split [0,bytes) evenly across T threads at 64B granularity. Ranges are adjacent,
// non-overlapping and cover [0,bytes) exactly; requires bytes % 64 == 0.
struct Range { std::uint64_t lo, hi; };
inline Range split(std::uint64_t bytes, int t, int T) {
  Range r;
  r.lo = (bytes * std::uint64_t(t) / T) & ~std::uint64_t(63);
  r.hi = (bytes * std::uint64_t(t + 1) / T) & ~std::uint64_t(63);
  if (t == T - 1) r.hi = bytes;
  return r;
}

// The consumer load: _mm256_loadu_ps (PS spelling per the task), reinterpreted as integer.
inline __m256i loadu32(const void* p) {
  return _mm256_castps_si256(_mm256_loadu_ps(reinterpret_cast<const float*>(p)));
}
inline std::uint64_t hsum64(__m256i v) {
  const __m128i s = _mm_add_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  return std::uint64_t(_mm_cvtsi128_si64(s)) + std::uint64_t(_mm_extract_epi64(s, 1));
}
// Sum all uint64s of [p, p+bytes) with AVX2 loads + 64-bit lane adds. p 64B-aligned, bytes%64==0.
inline std::uint64_t sum64_av(const std::uint8_t* p, std::uint64_t bytes) {
  __m256i a = _mm256_setzero_si256();
  for (std::uint64_t o = 0; o < bytes; o += 64) {
    a = _mm256_add_epi64(a, loadu32(p + o));
    a = _mm256_add_epi64(a, loadu32(p + o + 32));
  }
  return hsum64(a);
}
// Experiment C stream: MOVNTDQA loads on ordinary WB memory, 4 NT loads per 64B line so every
// byte is touched; each 128-bit result is zero-extend added (its 4 dwords) into the consumer.
inline std::uint64_t sum_nt_stream(const std::uint8_t* p, std::uint64_t bytes) {
  __m256i a = _mm256_setzero_si256();
  for (std::uint64_t o = 0; o < bytes; o += 64) {
    for (int j = 0; j < 64; j += 16) {
      const __m128i v = _mm_stream_load_si128(reinterpret_cast<const __m128i*>(p + o + j));
      a = _mm256_add_epi64(a, _mm256_cvtepu32_epi64(v));  // zero-extend add the 4 dwords
    }
  }
  return hsum64(a);
}
// Same consumer style with plain 128-bit loads (reference for the C probe / per-rep check).
inline std::uint64_t sum_dwords_plain(const std::uint8_t* p, std::uint64_t bytes) {
  __m256i a = _mm256_setzero_si256();
  for (std::uint64_t o = 0; o < bytes; o += 64) {
    for (int j = 0; j < 64; j += 16) {
      const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + o + j));
      a = _mm256_add_epi64(a, _mm256_cvtepu32_epi64(v));
    }
  }
  return hsum64(a);
}
// Experiment D stream: _MM_HINT_NTA prefetch every 64B, the real load stays PLAIN.
inline std::uint64_t sum_nta_stream(const std::uint8_t* p, std::uint64_t bytes) {
  __m256i a = _mm256_setzero_si256();
  for (std::uint64_t o = 0; o < bytes; o += 64) {
    _mm_prefetch(reinterpret_cast<const char*>(p + o), _MM_HINT_NTA);
    a = _mm256_add_epi64(a, loadu32(p + o));
    a = _mm256_add_epi64(a, loadu32(p + o + 32));
  }
  return hsum64(a);
}

double median_of(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// ---------------------------------------------------------------- worker pool (A..E)
void handle(int id) {
  const int c = g_cmd.load(std::memory_order_relaxed);
  std::uint64_t sum = 0;
  switch (c) {
    case KEEP_READ: {  // read the whole keep-set (warm / baseline / re-read)
      const Range r = split(KBYTES, id, NW);
      sum = sum64_av(g_keep + r.lo, r.hi - r.lo);
      break;
    }
    case S_PLAIN:
    case S_NT:
    case S_NTA: {
      const Range r = split(SBYTES, id, NW);
      const std::uint8_t* p = g_stream + r.lo;
      const std::uint64_t n = r.hi - r.lo;
      sum = (c == S_PLAIN) ? sum64_av(p, n) : (c == S_NT) ? sum_nt_stream(p, n) : sum_nta_stream(p, n);
      break;
    }
    case S_MIX: {  // experiment E: stream granule-wise, keep-set re-touched between slices
      for (int g = 0; g < NGRAN; ++g) {
        const Range r = split(GSIZE, id, NW);
        sum += sum64_av(g_stream + std::uint64_t(g) * GSIZE + r.lo, r.hi - r.lo);
        g_bar.wait();  // all 7 threads finished this 7 MiB of streaming
        if (g + 1 < NGRAN && id == 0) {
          // ONE thread re-reads the next 4 MiB keep window (rough LRU-refresh defense)
          const std::uint8_t* w = g_keep + std::uint64_t(g % NWIN) * WBYTES;
          const auto t0 = clk::now();
          sum += sum64_av(w, WBYTES);
          g_e_refresh_ns.fetch_add(
              std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count());
        }
        g_bar.wait();  // refresh done before the next slice streams
      }
      break;
    }
    default:
      break;
  }
  g_phase_sum[id] = sum;  // consumed by main right after the ack (also anti-DCE)
  g_sink[id] = g_sink[id] * 1315423911ull + sum;
  g_bar.wait();  // handler-terminating barrier -> ack time == phase completion
}

void worker(int id) {
  SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << NW_LOGICAL[id]);
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
  std::uint32_t done = 0;
  for (;;) {
    const std::uint32_t s = g_seq.load(std::memory_order_acquire);
    if (s == done) {
      if (g_stop.load(std::memory_order_relaxed)) break;
      _mm_pause();
      continue;
    }
    if (g_stop.load(std::memory_order_relaxed)) break;  // main's stop bump: do not run
    handle(id);
    done = s;
    if (id == 0) g_ack.store(s, std::memory_order_release);
  }
}

double run_cmd_timed(int cmd, double* refresh_s = nullptr) {
  const std::uint32_t s = g_seq.load(std::memory_order_relaxed) + 1;
  g_ack.store(0, std::memory_order_relaxed);
  g_cmd.store(cmd, std::memory_order_relaxed);
  const auto t0 = clk::now();
  g_seq.store(s, std::memory_order_release);
  while (g_ack.load(std::memory_order_acquire) != s) _mm_pause();
  const auto t1 = clk::now();
  if (refresh_s) *refresh_s = double(g_e_refresh_ns.exchange(0)) * 1e-9;
  return std::chrono::duration<double>(t1 - t0).count();
}

// one experiment group: warm keep, stream variant, re-read keep; reps times; medians.
struct GrpRes { double stream{0.0}, keep{0.0}; };
GrpRes run_group(char letter, int cmd, const char* desc) {
  std::printf("\n%c. %s\n", letter ? letter : '?', desc);
  std::vector<double> streams, keeps, refreshs;
  for (int rep = 1; rep <= g_reps; ++rep) {
    const double wsec = run_cmd_timed(KEEP_READ);  // warm the keep-set into L3
    double rfs = 0.0;
    const double ssec = run_cmd_timed(cmd, cmd == S_MIX ? &rfs : nullptr);
    const std::uint64_t phase_sink = g_sink[0] ^ g_sink[1] ^ g_sink[2] ^ g_sink[3] ^ g_sink[4] ^
                                     g_sink[5] ^ g_sink[6];  // keeps every rep's sums live
    char ntmsg[96] = "";
    if (cmd == S_NT) {  // does MOVNTDQA on WB memory behave (per-thread sums vs plain)?
      int bad = 0;
      for (int t = 0; t < NW; ++t)
        if (g_phase_sum[t] != g_expected_nt[t]) ++bad;
      std::snprintf(ntmsg, sizeof(ntmsg), " | MOVNTDQA sums vs plain: %s",
                    bad == 0 ? "MATCH" : "MISMATCH!!");
    }
    const double ksec = run_cmd_timed(KEEP_READ);  // keep-set re-read right after the stream
    if (cmd == S_MIX)
      std::printf(
          "  rep %d/%d: warm %6.2f GB/s | stream(incl refresh) %6.2f GB/s (%6.2f ms; refresh "
          "%5.2f ms, pure ~%6.2f GB/s) | keep re-read %6.2f GB/s ^%04llx\n",
          rep, g_reps, KBYTES / wsec / 1e9, SBYTES / ssec / 1e9, ssec * 1e3, rfs * 1e3,
          rfs < ssec ? SBYTES / (ssec - rfs) / 1e9 : 0.0, KBYTES / ksec / 1e9,
          (unsigned long long)(phase_sink & 0xffff));
    else
      std::printf("  rep %d/%d: warm %6.2f GB/s | stream %6.2f GB/s (%6.2f ms) | keep re-read "
                  "%6.2f GB/s (%5.2f ms)%s %04llx\n",
                  rep, g_reps, KBYTES / wsec / 1e9, SBYTES / ssec / 1e9, ssec * 1e3,
                  KBYTES / ksec / 1e9, ksec * 1e3, ntmsg,
                  (unsigned long long)(phase_sink & 0xffff));
    if (cmd == S_MIX) refreshs.push_back(rfs);
    streams.push_back(ssec);
    keeps.push_back(ksec);
  }
  GrpRes r;
  r.stream = SBYTES / median_of(streams) / 1e9;
  r.keep = KBYTES / median_of(keeps) / 1e9;
  std::printf("  median: stream %6.2f GB/s | keep re-read %6.2f GB/s", r.stream, r.keep);
  if (cmd == S_MIX && !refreshs.empty())
    std::printf(" | median refresh overhead %.2f ms", median_of(refreshs) * 1e3);
  std::printf("\n");
  return r;
}

// ---------------------------------------------------------------- experiment F pool
void f_handle(int id) {
  const int c = g_fcmd.load(std::memory_order_relaxed);
  std::uint64_t sum = 0;
  if (c == F_TOUCH) {  // commit + touch: non-zero writes force real physical pages
    const Range r = split(FBYTES, id, FT);
    std::memset(g_fbuf + r.lo, 0x5A, r.hi - r.lo);
  } else if (c == F_SLICE) {  // each thread reads its own disjoint quarter of the 2 MiB slice
    const std::uint64_t i = std::uint64_t(g_fslice.load(std::memory_order_relaxed));
    const Range r = split(FSLICE, id, FT);
    sum = sum64_av(g_fbuf + i * FSLICE + r.lo, r.hi - r.lo);
  }
  g_sink[NW + id] = g_sink[NW + id] * 1315423911ull + sum;
  g_fbar.wait();
}

void f_worker(int id) {
  SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << FT_LOGICAL[id]);
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
  std::uint32_t done = 0;
  for (;;) {
    const std::uint32_t s = g_fseq.load(std::memory_order_acquire);
    if (s == done) {
      if (g_fstop.load(std::memory_order_relaxed)) break;
      _mm_pause();
      continue;
    }
    if (g_fstop.load(std::memory_order_relaxed)) break;
    f_handle(id);
    done = s;
    if (id == 0) g_fack.store(s, std::memory_order_release);
  }
}

double f_run(int fcmd, int slice) {
  const std::uint32_t s = g_fseq.load(std::memory_order_relaxed) + 1;
  g_fack.store(0, std::memory_order_relaxed);
  g_fslice.store(slice, std::memory_order_relaxed);
  g_fcmd.store(fcmd, std::memory_order_relaxed);
  const auto t0 = clk::now();
  g_fseq.store(s, std::memory_order_release);
  while (g_fack.load(std::memory_order_acquire) != s) _mm_pause();
  return std::chrono::duration<double>(clk::now() - t0).count();
}

void fill(std::uint8_t* p, std::uint64_t bytes, std::uint64_t seed) {
  std::uint64_t x = seed;
  for (std::uint64_t o = 0; o < bytes; o += 8) {  // splitmix64 pattern; also first-touches pages
    x += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);
    std::memcpy(p + o, &z, 8);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--reps")) g_reps = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--only")) g_only = argv[++i];
    else {
      std::printf("usage: ntbench.exe [--reps N=5] [--only abcdef]\n"
                  "  a..e = experiments A..E, f = mismatched-DIMM scan\n");
      return 0;
    }
  }
  for (char& c : g_only) c = char(std::tolower((unsigned char)c));
  SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << 0);  // main out of the workers' way

  std::printf("ntbench: SPEEDUP-IDEAS.md item 3 - NT hints vs keep-set L3 residency (5800X3D)\n");
  std::printf("  workers %d pinned to logical cores", NW);
  for (int t = 0; t < NW; ++t) std::printf(" %d", NW_LOGICAL[t]);
  std::printf(", THREAD_PRIORITY_HIGHEST; main on logical 0; steady_clock\n");
  std::printf("  keep-set %.1f MiB | stream %.2f MiB (%d x %.1f MiB) | reps %d | experiments %s\n",
              double(KBYTES) / 1048576.0, double(SBYTES) / 1048576.0, NGRAN,
              double(GSIZE) / 1048576.0, g_reps, g_only.c_str());

  g_keep = static_cast<std::uint8_t*>(_aligned_malloc(KBYTES, 4096));
  g_stream = static_cast<std::uint8_t*>(_aligned_malloc(SBYTES, 4096));
  if (!g_keep || !g_stream) {
    std::printf("alloc failed\n");
    return 1;
  }
  std::printf("filling keep-set + stream arrays (%.0f MiB, first touch)...\n",
              double(KBYTES + SBYTES) / 1048576.0);
  fill(g_keep, KBYTES, 0x1234567);
  fill(g_stream, SBYTES, 0x89abcdef);

  // Does _mm_stream_load_si128 exist (it compiled -> yes) and behaves on WB memory?
  const std::uint64_t nt = sum_nt_stream(g_stream, PROBE_BYTES);
  const std::uint64_t pl = sum_dwords_plain(g_stream, PROBE_BYTES);
  std::printf("probe: _mm_stream_load_si128 (SSE4.1 MOVNTDQA) compiled ok | WB-memory data "
              "correct: %s (nt %016llx vs plain %016llx)\n",
              nt == pl ? "YES" : "NO", (unsigned long long)nt, (unsigned long long)pl);
  if (wants('c')) {  // per-thread reference sums over the stream (plain loads, same consumer)
    for (int t = 0; t < NW; ++t) {
      const Range r = split(SBYTES, t, NW);
      g_expected_nt[t] = sum_dwords_plain(g_stream + r.lo, r.hi - r.lo);
    }
    std::printf("probe: per-thread MOVNTDQA reference sums computed\n");
  }

  std::vector<std::thread> th;
  for (int i = 0; i < NW; ++i) th.emplace_back([i] { worker(i); });

  // -------------------------------------------------------------- A: baseline
  if (wants('a')) {
    std::printf("\nA. baseline: 60 MiB keep-set reads, 7 threads, uint64 sums\n");
    for (int k = 1; k <= 3; ++k) {
      const double s = run_cmd_timed(KEEP_READ);
      std::printf("  read %d: %7.2f GB/s (%6.2f ms)%s\n", k, double(KBYTES) / s / 1e9, s * 1e3,
                  k == 1 ? "  <- cold: the DRAM rate"
                         : (k == 2 ? "  <- L3-resident rate if the warm read stuck in the L3"
                                   : ""));
    }
  }

  // -------------------------------------------------------------- B..E groups
  const GrpRes resB = wants('b') ? run_group('B', S_PLAIN,
                                   "evict control: warm keep, stream with PLAIN loads, re-read")
                                 : GrpRes{};
  const GrpRes resC =
      wants('c') ? run_group('C', S_NT, "stream via _mm_stream_load_si128 (MOVNTDQA)") : GrpRes{};
  const GrpRes resD =
      wants('d')
          ? run_group('D', S_NTA, "stream preceded by _mm_prefetch(_MM_HINT_NTA) per 64B, plain loads")
          : GrpRes{};
  const GrpRes resE =
      wants('e')
          ? run_group('E', S_MIX,
                      "fallback: keep re-touched between 7 MiB stream slices (one thread, 4 MiB)")
          : GrpRes{};
  (void)resE;

  // -------------------------------------------------------------- item-3 gate
  if (wants('b') && (wants('c') || wants('d')) && resB.keep > 0.0) {
    const double b = resB.keep;
    double best = 0.0;
    const char* name = "";
    std::printf("\nITEM-3 GATE: keep-set re-read in C or D at least 30%% faster than in B?\n");
    std::printf("  B control keep re-read median: %.2f GB/s\n", b);
    if (wants('c')) {
      std::printf("  C (MOVNTDQA):  %6.2f GB/s  (%+.1f%% vs B)\n", resC.keep,
                  100.0 * (resC.keep / b - 1.0));
      if (resC.keep > best) {
        best = resC.keep;
        name = "C MOVNTDQA";
      }
    }
    if (wants('d')) {
      std::printf("  D (PREFETCHNTA): %6.2f GB/s  (%+.1f%% vs B)\n", resD.keep,
                  100.0 * (resD.keep / b - 1.0));
      if (resD.keep > best) {
        best = resD.keep;
        name = "D PREFETCHNTA";
      }
    }
    const bool ok = best >= 1.30 * b;
    std::printf("  best = %s: %.2f GB/s vs B %.2f GB/s -> %+.1f%% => %s\n", name, best, b,
                100.0 * (best / b - 1.0),
                ok ? "GATE PASSED: NT hints keep the keep-set in L3 on this machine"
                   : "GATE FAILED: hints do not preserve the keep set (E re-touch fallback)");
    if (wants('e'))
      std::printf("  (E fallback context: keep re-read after re-touch defense: %.2f GB/s)\n",
                  resE.keep);
  }

  // -------------------------------------------------------------- stop the 7 idlers
  g_stop.store(true, std::memory_order_relaxed);
  g_seq.fetch_add(1, std::memory_order_release);
  for (auto& t : th) t.join();

  // -------------------------------------------------------------- F: DIMM range scan
  if (wants('f')) {
    std::printf("\nF. mismatched-DIMM range scan: 12 GiB, 2 MiB slices, 4 threads\n");
    g_fbuf = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, FBYTES, MEM_RESERVE | MEM_COMMIT,
                                                     PAGE_READWRITE));
    if (!g_fbuf) {
      std::printf("  VirtualAlloc of 12 GiB failed ( GetLastError %lu ) - F skipped\n",
                  (unsigned long)GetLastError());
    } else {
      std::vector<std::thread> fth;
      for (int i = 0; i < FT; ++i) fth.emplace_back([i] { f_worker(i); });
      const double tw = f_run(F_TOUCH, 0);
      std::printf("  committed + touched 12 GiB in %.2f s\n", tw);

      // 2 MiB per slice, each thread its own disjoint 512 KiB quarter of the slice
      std::vector<double> s_sec(FNSLICE);
      const auto ts0 = clk::now();
      double sum_slice_s = 0.0;
      for (int i = 0; i < FNSLICE; ++i) {
        const double t = f_run(F_SLICE, i);
        s_sec[i] = t;
        sum_slice_s += t;
      }
      const double wall = std::chrono::duration<double>(clk::now() - ts0).count();

      g_fstop.store(true, std::memory_order_relaxed);
      g_fseq.fetch_add(1, std::memory_order_release);
      for (auto& t : fth) t.join();

      std::vector<double> gbps(FNSLICE);
      for (int i = 0; i < FNSLICE; ++i) gbps[i] = double(FSLICE) / s_sec[i] / 1e9;
      std::vector<int> order(FNSLICE);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(),
                [&](int a, int b) { return gbps[a] < gbps[b]; });
      const double p = double(FNSLICE - 1);
      std::printf("  scan wall %.2f s (sum of slices %.2f s, per-slice sync overhead %.3f s)\n",
                  wall, sum_slice_s, wall - sum_slice_s);
      std::printf("  per-slice GB/s: min %.2f | p5 %.2f | median %.2f | p95 %.2f | max %.2f\n",
                  gbps[order[0]], gbps[order[int(std::llround(0.05 * p))]],
                  gbps[order[FNSLICE / 2]], gbps[order[int(std::llround(0.95 * p))]],
                  gbps[order[FNSLICE - 1]]);
      std::printf("  slowest 10 slices (offset / GB/s):\n");
      for (int k = 0; k < 10; ++k) {
        const std::uint64_t off = std::uint64_t(order[k]) * FSLICE;
        std::printf("    [%.3f GB / %5.2f GiB] %.2f GB/s\n", double(off) / 1e9,
                    double(off) / 1073741824.0, gbps[order[k]]);
      }
      std::printf("  fastest 10 slices (offset / GB/s):\n");
      for (int k = 0; k < 10; ++k) {
        const int idx = order[FNSLICE - 1 - k];
        const std::uint64_t off = std::uint64_t(idx) * FSLICE;
        std::printf("    [%.3f GB / %5.2f GiB] %.2f GB/s\n", double(off) / 1e9,
                    double(off) / 1073741824.0, gbps[idx]);
      }
      std::FILE* fcsv = std::fopen("ntbench_f_slices.csv", "w");
      if (fcsv) {
        std::fprintf(fcsv, "slice,offset_bytes,offset_GB,gbps\n");
        for (int i = 0; i < FNSLICE; ++i)
          std::fprintf(fcsv, "%d,%llu,%.3f,%.3f\n", i,
                       (unsigned long long)(std::uint64_t(i) * FSLICE),
                       double(std::uint64_t(i) * FSLICE) / 1e9, gbps[i]);
        std::fclose(fcsv);
        std::printf("  full per-slice dump: ntbench_f_slices.csv\n");
      }
      VirtualFree(g_fbuf, 0, MEM_RELEASE);
      g_fbuf = nullptr;
    }
  }

  _aligned_free(g_stream);
  _aligned_free(g_keep);
  std::printf("\nanti-DCE sinks (workers):");
  for (int i = 0; i < NW + FT; ++i) std::printf(" %016llx", (unsigned long long)g_sink[i]);
  std::printf("\nntbench done\n");
  return 0;
}
