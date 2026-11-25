// quill: DFlash2 drafter kernels (incoai/Inkling-2-Mini-DFlash2). Raw-pointer launchers, graph capturable: every
// per-call value comes from the device-resident Df2Params.
//
// Architecture (llama.cpp src/models/dflash*.cpp, SGLang dflash.py with incoai's Inkling patch):
// * context: g = rms(fc . concat(4 target taps)) * hidden_norm per committed position; per layer
//   k = rope(rms_head(Wk g) * k_norm), v = Wv g, kept in a ring of the last WIN = 2048 positions.
// * block of 8 rows (anchor embedding + 7 mask-token embeddings, raw rows of the target's table), 5 layers:
//   n = rms(x) * in_norm; dyn_a = Wca n; n' = conv(n, dyn_a, side 0); q, k, v = W n'; q/k head RMS norms, NeoX
//   rope (theta 1e4); bidirectional attention over the ring (window 2048) + the 8 block rows; o = Wo a;
//   x += conv(o, dyn_a, side 1); f = rms(x) * post_norm; dyn_m = Wcm f; f' = conv(f, dyn_m, side 0);
//   d = Wd (silu(Wg f') * Wu f'); x += conv(d, dyn_m, side 1). h = rms(x) * norm.
// * conv(x, dyn, side)[t, c] = (base[side, 0, c] + dyn[t, side*288 + g]) x[t, c]
//                            + (base[side, 1, c] + dyn[t, side*288 + 144 + g]) x[t-1, c]   (g = c / 16, x[-1] = 0)
// * logits = target head (raw, before the softcap) of rows 1..7; per row the top-16 candidates; gate = Wsel h;
//   step i: s_c = 30 tanh((succ[c] . (pred[prev] * gate_i) + logit_c) / 30) (softcap after the correction),
//   prev = anchor (i = 0) or the token picked at i - 1; greedy argmax or Gumbel-max at the temperature.
#pragma once

#include "kernels/draft/draft.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ingine::kernels::dflash2 {

inline constexpr int H = 2304, NH = 32, NKV = 8, HD = 128, QD = NH * HD, KVD = NKV * HD, QKV = QD + 2 * KVD;
inline constexpr int INTER = 6912, LAYERS = 5, WIN = 2048, SLOTS = WIN + 8, BLOCK = 8, GAMMA = 7;
inline constexpr int TOPK = 16, RANK = 256, GROUPS = H / 16, NDYN = 4 * GROUPS;  // 576 dynamic coefficients
inline constexpr int MASK_TOKEN = 200056;
inline constexpr float THETA = 10000.f, EPS = 1e-6f, SOFTCAP = 30.f;

struct Df2Params {
  std::int32_t anchor = 0, P = 0;
  float temperature = 0.f;
  std::int32_t start = 0, ncommit = 0;  // commit rows: positions start..start+ncommit-1
  std::int32_t round_start = 0;         // P of the last draft
  std::uint64_t seed = 0, draws = 0;
  const __nv_bfloat16* commit_feats = nullptr;  // [ncommit, 4 * H]
};

void set_draft(Df2Params* p, int anchor, int P, float temperature, cudaStream_t s);
void set_commit(Df2Params* p, const __nv_bfloat16* feats, int n, int start, cudaStream_t s);
void set_seed(Df2Params* p, std::uint64_t seed, cudaStream_t s);
// Round transition (SpecLoop): commit = the verify call's first *accepted + 1 rows at p->round_start, next anchor
// at P = round_start + n (next_greedy if temperature <= 0 else next_sampled). Copies the committed taps into
// feats_scratch [8, 4H] (zero rows past n) and points p->commit_feats at it.
void round_begin(Df2Params* p, const draft::RoundInputs& in, __nv_bfloat16* feats_scratch, cudaStream_t s);
// Copies p->commit_feats rows (< ncommit) to scratch [8, 4H], zero rows above (fixed-shape GEMV input).
void stage_commit(const Df2Params* p, __nv_bfloat16* scratch, cudaStream_t s);

// out[r] = bf16(rms(in[r]) * w), in fp32 [rows, H]
void rmsnorm(const float* in, const __nv_bfloat16* w, int rows, __nv_bfloat16* out, cudaStream_t s);
// x[0] = float(embed[anchor]), x[1..7] = float(embed[MASK_TOKEN]); xn = bf16(rms(x) * w)
// x[t] = bf16(rms(embed[token]) * embed_norm) (anchor, then MASK_TOKEN); xn = bf16(rms(x) * w)
void embed_block(const __nv_bfloat16* embed, const float* embed_norm, const Df2Params* p, const __nv_bfloat16* w,
                 float* x, __nv_bfloat16* xn, cudaStream_t s);
// side 0: out bf16 [8, H] = conv(in bf16, dyn); side 1: x += conv(in fp32, dyn)
void conv_in(const __nv_bfloat16* in, const float* dyn, const float* base, __nv_bfloat16* out, cudaStream_t s);
void conv_residual(const float* in, const float* dyn, const float* base, float* x, cudaStream_t s);
// q/k/v of the block: qkv fp32 [8, QKV]; q -> q_out bf16 [8, NH, HD]; k, v -> ring slots WIN + t
void qkv_post(const float* qkv, const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm, const Df2Params* p,
              __nv_bfloat16* q_out, __nv_bfloat16* k_ring, __nv_bfloat16* v_ring, cudaStream_t s);
// context k/v: kv fp32 [8, 2 * KVD] (k | v) of committed rows r < p->ncommit at p->start + r -> slot pos % WIN
void ctx_store(const float* kv, int ld, const __nv_bfloat16* k_norm, const Df2Params* p, __nv_bfloat16* k_ring,
               __nv_bfloat16* v_ring, std::int32_t* ring_pos, bool write_pos, cudaStream_t s);
// bidirectional block attention, out bf16 [8, QD]
void attention(const __nv_bfloat16* q, const __nv_bfloat16* k_ring, const __nv_bfloat16* v_ring,
               const std::int32_t* ring_pos, const Df2Params* p, __nv_bfloat16* out, cudaStream_t s);
// top-16 of each row of logits fp32 [rows, V]: cand_id int32 [rows, 16], cand_val fp32 [rows, 16]
void topk16(const float* logits, int rows, int V, std::int32_t* cand_id, float* cand_val, cudaStream_t s);
// quill: ids[i] = map[ids[i]] for i < n <= 128 (head rows in frequency order -> token ids)
void map_ids(std::int32_t* ids, int n, const std::int32_t* map, cudaStream_t s);
// the selector walk and the SpecLoop outputs (see the header comment). pred/succ bf16 [Vcb, RANK] (may be host
// mapped), gate fp32 [7, RANK]. draft_logits fp32 [7, V] stays -inf outside each row's candidates (prev_cand
// tracks the last ones). conf [7] = max softmax(s); qstats [7, 2] for sampled acceptance; p->draws += 1.
void select(const std::int32_t* cand_id, const float* cand_val, const float* gate, const __nv_bfloat16* pred,
            const __nv_bfloat16* succ, Df2Params* p, std::int32_t* tokens, float* draft_logits, int V,
            std::int32_t* prev_cand, float* conf, float* qstats, cudaStream_t s, float dts = 1.f);

}  // namespace ingine::kernels::dflash2
