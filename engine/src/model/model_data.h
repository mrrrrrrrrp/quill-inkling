// C++ mirrors of the Python engine's model data: weights.py (constants, visit plan, Weights/Layer/Experts),
// model.py (State, Ablation, TAP_VISITS) and fast.py (projection layout, Buffers per call width).
//
// Tensors are ingine::Tensor views over device memory. Shape convention: Tensor::ne[] lists dimensions
// innermost first (ggml order), so a Python tensor of shape [a, b, c] has ne = {c, b, a}; every field
// comment below gives the Python (row-major) shape. All tensors are contiguous unless noted.
// `rm(data, dtype, {a, b, c})` builds a view from a Python-order shape.
#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::model {

class ExpertStore;  // model/experts.h

// ------------------------------------------------------------------------------------ weights.py constants

inline constexpr int HIDDEN = 2304;
inline constexpr int HEADS = 24, KV_HEADS = 8, HEAD_DIM = 128;
inline constexpr int D_REL = 16, REL_EXTENT = 1024, WINDOW = 1024;
inline constexpr int INTER = 1152, EXPERTS = 128, TOP_K = 8;
inline constexpr int HEAD_DRAFT_ROWS = 41456;  // quill: frequency-ordered head rows the drafter reads
inline constexpr int CONV_K = 4;
inline constexpr int LAYERS = 24, BLOCK = 6;
inline constexpr int LOOP_START = 6, LOOP_END = 18;
inline constexpr int LOOP_LEN = LOOP_END - LOOP_START;
// Loop passes: the checkpoint (config.json loop_config.num_passes) was trained with 3. quill runs any count in
// 1..MAX_PASSES: set the environment variable QUILL_PASSES=N (read once, before the model is built). Everything
// that depends on the count (visit plan, caches, conv states, ablation rows, taps, expert reads) is derived from
// visit_plan() below, and N = 3 is exactly the checkpoint's plan.
inline constexpr int TRAINED_PASSES = 3;
inline constexpr int MAX_PASSES = 8;
inline constexpr int VOCAB = 201024, UNPADDED_VOCAB = 200058;
inline constexpr int GATE_RANK = 64;
inline constexpr float EPS = 1e-6f;
inline constexpr double QK_CLIP = 22.627416997969522, REL_CLIP = 8.0;
inline constexpr double LOG_ALPHA = 0.1, LOG_FLOOR = 8192.0;
inline constexpr float GATE_SOFTCAP = 4.0f, UP_SOFTCAP = 25.0f, LOGIT_SOFTCAP = 30.0f;
inline constexpr float ROUTE_SCALE = 8.0f;
inline constexpr float FP8_MAX = 448.0f;  // largest finite float8_e4m3fn (model.FP8_MAX)
inline constexpr int KV_DIM = KV_HEADS * HEAD_DIM;  // 1024

inline constexpr bool is_full_layer(int l) { return l == 5 || l == 11 || l == 17 || l == 23; }
inline constexpr bool is_loop_layer(int l) { return LOOP_START <= l && l < LOOP_END; }

// --------------------------------------------------------------------------------------------- visit plan

struct Visit {
  int layer;
  int pass;  // loop pass (0 outside the loop)
};

// weights.visit_plan(): layers 0-5, layers 6-17 `passes` times, layers 18-23 (48 visits for the trained 3 passes)
inline constexpr int MAX_VISITS = LAYERS - LOOP_LEN + LOOP_LEN * MAX_PASSES;

struct VisitPlan {
  int passes = TRAINED_PASSES;
  int num_visits = 0, num_full = 0, num_sliding = 0;
  std::array<Visit, MAX_VISITS> visits{};
  std::array<int, MAX_VISITS> ring{}, full{};  // model.State.ring_index / full_index: visit -> cache slot (-1: other kind)
  // model.TAP_VISITS (5, 23, 41 for 3 passes): DSpark features = the partial block entering these visits, then the
  // final normed hidden. Layer 5, layer 11 of pass 1, layer 17 of the LAST pass (the drafter reads the block ends
  // of the first block, the middle and the end of the loop).
  std::array<int, 3> taps{};
};

inline VisitPlan make_visit_plan(int passes) {
  if (passes < 1 || passes > MAX_PASSES)
    throw std::invalid_argument("QUILL_PASSES must be in 1.." + std::to_string(MAX_PASSES) + ", got " +
                                std::to_string(passes));
  VisitPlan P;
  P.passes = passes;
  int v = 0;
  for (int l = 0; l < LOOP_START; ++l) P.visits[v++] = {l, 0};
  for (int p = 0; p < passes; ++p)
    for (int l = LOOP_START; l < LOOP_END; ++l) P.visits[v++] = {l, p};
  for (int l = LOOP_END; l < LAYERS; ++l) P.visits[v++] = {l, 0};
  P.num_visits = v;
  int nr = 0, nf = 0;
  for (int i = 0; i < v; ++i) {
    const bool full = is_full_layer(P.visits[i].layer);
    P.ring[i] = full ? -1 : nr++;
    P.full[i] = full ? nf++ : -1;
  }
  P.num_full = nf, P.num_sliding = nr;
  auto visit_of = [&](int layer, int pass) {
    for (int i = 0; i < v; ++i)
      if (P.visits[i].layer == layer && P.visits[i].pass == (is_loop_layer(layer) ? pass : 0)) return i;
    return -1;
  };
  P.taps = {visit_of(5, 0), visit_of(11, std::min(1, passes - 1)), visit_of(17, passes - 1)};
  return P;
}

// The process-wide plan (QUILL_PASSES, default TRAINED_PASSES), built at first use.
inline int env_passes() {
  const char* e = std::getenv("QUILL_PASSES");
  if (!e || !*e) return TRAINED_PASSES;
  char* end = nullptr;
  const long n = std::strtol(e, &end, 10);
  if (end == e || *end != '\0') throw std::invalid_argument(std::string("QUILL_PASSES is not an integer: ") + e);
  return int(n);
}
inline const VisitPlan& visit_plan() {
  static const VisitPlan plan = make_visit_plan(env_passes());
  return plan;
}
inline int passes() { return visit_plan().passes; }
inline int num_visits() { return visit_plan().num_visits; }
inline int num_full_visits() { return visit_plan().num_full; }
inline int num_sliding_visits() { return visit_plan().num_sliding; }
inline const Visit& visit_at(int v) { return visit_plan().visits[v]; }
inline int ring_index(int v) { return visit_plan().ring[v]; }
inline int full_index(int v) { return visit_plan().full[v]; }
inline int tap_visit(int i) { return visit_plan().taps[i]; }

// Per-visit files (the ablation file) hold the trained plan's TRAINED_VISITS rows. Row of (layer, pass) in them: the
// loop's extra passes reuse the last trained pass (a plan with fewer passes uses the first ones).
inline constexpr int TRAINED_VISITS = LAYERS - LOOP_LEN + LOOP_LEN * TRAINED_PASSES;  // 48
inline int trained_row(const Visit& vis) {
  if (vis.layer < LOOP_START) return vis.layer;
  if (vis.layer < LOOP_END) return LOOP_START + LOOP_LEN * std::min(vis.pass, TRAINED_PASSES - 1) + (vis.layer - LOOP_START);
  return LOOP_START + LOOP_LEN * TRAINED_PASSES + (vis.layer - LOOP_END);
}
inline constexpr int NUM_TAPS = 4;

// ----------------------------------------------------------------------------- fast.py projection layout

// SEG_ROWS order (KERNELS.md): q | gate | rel | sink | k | v | (full layers) k_dyn | v_dyn
enum Seg : int { SEG_Q = 0, SEG_GATE, SEG_REL, SEG_SINK, SEG_K, SEG_V, SEG_K_DYN, SEG_V_DYN, NUM_SEGS };
inline constexpr std::array<int, NUM_SEGS> SEG_ROWS = {HEADS * HEAD_DIM, HEADS * HEAD_DIM, HEADS * D_REL, HEADS,
                                                       KV_DIM,           KV_DIM,           KV_HEADS * CONV_K,
                                                       KV_HEADS * CONV_K};
inline constexpr std::array<int, NUM_SEGS> SEG_INPUT = {0, 0, 0, 0, 1, 2, 1, 2};  // xin index (q, k, v inputs)
inline constexpr std::array<int, NUM_SEGS + 1> SEG_BEGIN = [] {
  std::array<int, NUM_SEGS + 1> b{};
  for (int s = 0; s < NUM_SEGS; ++s) b[s + 1] = b[s] + SEG_ROWS[s];
  return b;
}();
inline constexpr int PROJ_ROWS_SLIDING = SEG_BEGIN[SEG_K_DYN];  // 8600
inline constexpr int PROJ_ROWS_FULL = SEG_BEGIN[NUM_SEGS];      // 8664
inline constexpr int num_segs(bool full) { return full ? NUM_SEGS : SEG_K_DYN; }
inline constexpr int proj_rows(bool full) { return full ? PROJ_ROWS_FULL : PROJ_ROWS_SLIDING; }

// --------------------------------------------------------------------------------------------- helpers

// View with a Python-order (row-major) shape.
inline Tensor rm(void* data, DType dtype, std::initializer_list<std::int32_t> python_shape) {
  std::int32_t s[4] = {1, 1, 1, 1};
  const int n = static_cast<int>(python_shape.size());
  int i = n - 1;
  for (std::int32_t d : python_shape) s[i--] = d;
  switch (n) {
    case 1: return Tensor(data, dtype, {s[0]});
    case 2: return Tensor(data, dtype, {s[0], s[1]});
    case 3: return Tensor(data, dtype, {s[0], s[1], s[2]});
    default: return Tensor(data, dtype, {s[0], s[1], s[2], s[3]});
  }
}

template <class T>
inline T* ptr(const Tensor& t) {
  return static_cast<T*>(t.data);
}

// ----------------------------------------------------------------------------------------------- weights

// A dense matrix as the decoder stores it: BF16 rows, or float8_e4m3fn rows with FP32 per-row scales
// (`scale.data != nullptr`; value = fp8 * scale, see weights.dequant / dense fp8_rows).
struct Dense {
  Tensor w;      // [rows, cols] bf16 or fp8; quill FP6: U8 [rows, cols * 3 / 4] (kernels/dense/fp6.cuh)
  Tensor scale;  // [rows] fp32, or empty (data == nullptr) for BF16
  Tensor bscale; // quill FP6: E4M3 block scales [rows, cols / 32], else empty
  int ncols = 0; // quill FP6: K (w.ne[0] holds the packed bytes)
  bool fp4 = false;  // quill: FP4 E2M1 rows (fp4.cuh): w U8 [rows, K/2], bscale E4M3 [rows, K/16]
  bool fp6() const { return bscale.data != nullptr && !fp4; }
  bool fp8() const { return scale.data != nullptr && bscale.data == nullptr; }
  int rows() const { return w.ne[1]; }
  int cols() const { return fp6() ? ncols : w.ne[0]; }
};

enum class ExpertFormat : int { NVFP4 = 0, MXFP8 = 1 };  // the moe kernels' `fmt`

// weights.Experts (checkpoint encoding; see weights.py docstring)
struct Experts {
  ExpertFormat fmt = ExpertFormat::NVFP4;
  // NVFP4: w13 u8 [128, 2304, 1152], w13_scale e4m3 [128, 2304, 144], w2 u8 [128, 2304, 576],
  //        w2_scale e4m3 [128, 2304, 72], factors fp32 [128] (amax / outer_scale_denominator)
  // MXFP8: w13 e4m3 [128, 2304, 2304], w13_scale u8 (E8M0) [128, 2304, 72], w2 e4m3 [128, 2304, 1152],
  //        w2_scale u8 [128, 2304, 36], factors empty
  Tensor w13, w13_scale, w2, w2_scale;
  Tensor w13_factor, w2_factor;
};

// weights.Layer plus the per-layer derived tensors of fast.FastDecoder (_concat_projections,
// _stack_layer_params, _fp8_dense). Absent fields have data == nullptr.
struct Layer {
  int idx = 0;
  bool full = false, loop = false;

  // norms (fp32 [H])
  Tensor q_norm, k_norm, v_norm;  // full layers
  Tensor attn_norm;               // sliding layers
  Tensor mlp_norm;

  // attention projections: one [R, H] matrix in SEG order (R = proj_rows(full)); the per-projection
  // entries are row views of it (with the matching rows of its scales when FP8)
  Dense wproj;
  Dense wq, wgate, wr, wsink, wk, wv, wk_dyn, wv_dyn;
  Dense wo;  // [H, 3072]

  Tensor q_scale;   // fp32 [24, 128]
  Tensor r_scale;   // fp32 [24, 16]
  Tensor rel_proj;  // fp32 [16, 1024]
  Tensor k_static, v_static;  // full: fp32 [4, 1024] by lag (row 0 = current token)
  Tensor k_conv, v_conv;      // sliding: fp32 [1024, 4] conv1d order (tap 3 = current token)
  Tensor attn_conv, mlp_conv;  // fp32 [H, 4]
  Tensor router;       // fp32 [128, H]
  Tensor router_bias;  // fp32 [128]
  Tensor router_bf16;  // bf16 [128, H] (exact copy of the checkpoint values)
  Tensor attn_query;   // fp32 [Q, H] (Q = 3 full, 1 sliding), layers > 0
  Tensor mlp_query;    // fp32 [1, H]

  // loop layers: message gates. Stacks the kernels take, with per-gate views into them.
  Tensor qkv_gate_down;                      // bf16 [3, 64, 2H]
  Tensor qkv_gate_forget, qkv_gate_gain;     // fp32 [3, H, 64]
  Tensor qkv_gate_forget16, qkv_gate_gain16; // bf16 [3, H, 64]
  Tensor mlp1_gate_down;                     // bf16 [1, 64, 2H] (= mlp_gate_down)
  Tensor mlp1_gate_forget, mlp1_gate_gain;   // fp32 [1, H, 64]
  Tensor mlp1_gate_forget16, mlp1_gate_gain16;
  Tensor q_gate_down, k_gate_down, v_gate_down, mlp_gate_down;          // bf16 [64, 2H] views
  Tensor q_gate_forget, k_gate_forget, v_gate_forget, mlp_gate_forget;  // fp32 [H, 64] views
  Tensor q_gate_gain, k_gate_gain, v_gate_gain, mlp_gate_gain;          // fp32 [H, 64] views

  // stacked norm weights (fast._stack_layer_params)
  Tensor attn_norm3;     // fp32 [3, H]: (q, k, v) norms, or attn_norm three times
  Tensor attn_norm_mix;  // fp32 [3, H] (full) or [1, H] (sliding)
  Tensor mlp_norm1;      // fp32 [1, H]

  Experts experts;
};

// weights.Weights (+ fast.FastDecoder.final_norm1 / head_w). Owns its device memory.
struct Weights {
  std::vector<Layer> layers;  // LAYERS
  Tensor embed;               // bf16 [VOCAB, H] (padded rows kept, as weights.load)
  Tensor embed_norm;          // fp32 [H]
  Tensor final_query;         // fp32 [1, H]
  Tensor final_norm;          // fp32 [H] (also final_norm1 = [1, H] view of the same memory)
  Dense unembed;              // [UNPADDED_VOCAB, H] bf16, or FP8 rows + scales (dense_fp8)
  Tensor head_tok;            // quill: int32 [UNPADDED_VOCAB] token of each FP6 head row (frequency order), or empty
  Dense unembed_tail;         // quill: FP4 rows head_split.. of the frequency-ordered head (unembed: the FP6 prefix)
  int head_split = 0;         // quill: rows in unembed when unembed_tail is set
  Tensor message_queries;     // fp32 [12, 4, H]: q, k, v, mlp per loop layer
  std::array<ExpertFormat, LAYERS> expert_formats{};
  bool dense_fp8 = false;     // wproj (+ views), wo and unembed are FP8 rows

  std::vector<DeviceBuffer> storage;  // device allocations behind the views
  std::shared_ptr<ExpertStore> experts;  // quill: routed experts (host memory + GPU slot pools)
  std::shared_ptr<void> host_embed;       // quill: the embedding table lives in mapped pinned host memory
};

// model.Ablation: per visit y <- y - w * r * max(r . y - offset, floor) on the attention / MLP writes; offset 0
// with floor -FLT_MAX is the plain projection y - w * r * (r . y), floor 0 removes only what exceeds the
// offset. With detectors a write is changed only where det . y exceeds the visit's threshold. Without an
// ablation file (Ablation.none) directions and weights are zero.
struct Ablation {
  Tensor attn_dirs;  // fp32 [96, H]: rows 0..47 unit directions, rows 48..95 detectors (zero without them)
  Tensor attn_w;     // fp32 [48, 4] rows: weight, offset, floor, detector threshold
  Tensor mlp_dirs;   // fp32 [96, H]
  Tensor mlp_w;      // fp32 [48, 4]
  bool active = false;  // loaded from a file
  bool detect = false;  // the file has detectors

  std::vector<DeviceBuffer> storage;
};

// ------------------------------------------------------------------------------------------------- state

// model.State: per-request caches.
struct State {
  int max_ctx = 0;
  bool kv_fp8 = false;
  bool kv_q4 = false;  // quill: full caches are u8 [8, max_ctx, 576] IQ4_NL rows
  bool kv_shared = false;  // quill: q4 rows position-major [max_ctx][k 8 visits | v 8 visits][576] in the
                           // runtime's shared KV/expert region (k_full/v_full point into it after finalize)
  std::size_t full_stride() const {
    return kv_shared ? std::size_t(2 * num_full_visits()) * 576 : kv_q4 ? 576 : std::size_t(KV_DIM) * (kv_fp8 ? 1 : 2);
  }
  std::size_t full_row_bytes() const { return kv_q4 ? 576 : std::size_t(KV_DIM) * (kv_fp8 ? 1 : 2); }
  // visit counts below are for the trained 3 passes: 40 sliding, 8 full, 48 in all (see visit_plan())
  Tensor k_ring, v_ring;      // bf16 [40, 1024, 8, 128] (sliding visits, ring_index); quill ring_q4: u8 [40, 1024, 576]
  bool ring_q4 = false, ring_q8 = false;
  Tensor ring_pos;            // int64 [40, 1024], absolute position per slot, -1 = empty
  Tensor k_full, v_full;      // bf16 or fp8 e4m3 [8, max_ctx, 8, 128] (full visits, full_index)
  Tensor conv_k, conv_v;      // fp32 [48, 3, 1024] previous 3 key/value conv inputs (oldest first)
  Tensor conv_attn, conv_mlp; // fp32 [48, 3, H]
  int length = 0;             // tokens in the context (host copy; decode steps do not advance it)

  std::vector<DeviceBuffer> storage;

  // per-visit views
  Tensor k_ring_at(int ring) const;
  Tensor v_ring_at(int ring) const;
  std::int64_t* ring_pos_at(int ring) const;
  Tensor k_full_at(int full) const;
  Tensor v_full_at(int full) const;
  float* conv_k_at(int v) const { return ptr<float>(conv_k) + std::size_t(v) * 3 * KV_DIM; }
  float* conv_v_at(int v) const { return ptr<float>(conv_v) + std::size_t(v) * 3 * KV_DIM; }
  float* conv_attn_at(int v) const { return ptr<float>(conv_attn) + std::size_t(v) * 3 * HIDDEN; }
  float* conv_mlp_at(int v) const { return ptr<float>(conv_mlp) + std::size_t(v) * 3 * HIDDEN; }
};

// ----------------------------------------------------------------------------------------------- buffers

// fast.Buffers: fixed device buffers of one call width M (one set per captured CUDA graph).
//   B0 = embedding; B1 = hidden entering layer 6 (pass 0); B2[p] = hidden entering layer 12 in pass p;
//   E[p] = hidden after layer 17 in pass p (E[passes() - 1] is also the block entering layer 18).
struct Buffers {
  int m = 0;
  Tensor token;                     // int64 [m] input token ids (read by embed)
  Tensor B0, B1;                    // fp32 [m, H]
  std::array<Tensor, MAX_PASSES> B2, E; // fp32 [m, H]; the first passes() entries are allocated
  Tensor hid, part, y, y2;          // fp32 [m, H]
  Tensor xin;                       // bf16 [3, m, H]
  Tensor mixf;                      // fp32 [3, m, H]
  Tensor msgs;                      // fp32 [4, m, H]
  Tensor xm;                        // bf16 [1, m, H]
  Tensor mixm;                      // fp32 [1, m, H]
  Tensor proj_sliding, proj_full;   // fp32 [m, 8600], [m, 8664] (fast: proj_out keyed by row count)
  Tensor attn_out;                  // bf16 [m, 3072]
  Tensor idx;                       // int32 [m, 8]
  Tensor wts;                       // fp32 [m, 8]
  Tensor act;                       // bf16 [m, 8, I]
  Tensor ymoe;                      // fp32 [m * 8, H] per-slot expert outputs (quill hybrid MoE)
  Tensor xf;                        // bf16 [1, m, H]
  Tensor logits;                    // fp32 [m, UNPADDED_VOCAB]
  Tensor taps;                      // bf16 [m, 4, H]: partial block entering TAP_VISITS + final normed hidden
  // convolution inputs of each row per visit, for rolling back rejected rows (history = true)
  Tensor hist_attn, hist_mlp;       // fp32 [num_visits(), m, H]
  Tensor hist_k, hist_v;            // fp32 [num_visits(), m, 1024]

  std::vector<DeviceBuffer> storage;

  // fast.Buffers.blocks: completed blocks visible to layer l's attention read in pass p
  std::vector<const float*> blocks(int l, int p) const;
};

}  // namespace ingine::model
