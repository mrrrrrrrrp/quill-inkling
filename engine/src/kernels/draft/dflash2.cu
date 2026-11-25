// quill: DFlash2 drafter kernels (dflash2.h).
#include "kernels/draft/dflash2.h"
#include "kernels/draft/rng.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace ingine::kernels::dflash2 {
namespace {

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }

__device__ __forceinline__ float block_sum(float v, float* red) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = (blockDim.x + 31) >> 5;
  __syncthreads();
  if (l == 0) red[w] = v;
  __syncthreads();
  float t = 0.f;
  for (int i = 0; i < nw; ++i) t += red[i];
  return t;
}

// ---- params
__global__ void set_draft_k(Df2Params* p, int anchor, int P, float t) {
  p->anchor = anchor, p->P = P, p->round_start = P, p->temperature = t;
}
__global__ void set_commit_k(Df2Params* p, const __nv_bfloat16* f, int n, int start) {
  p->commit_feats = f, p->ncommit = n, p->start = start;
}
__global__ void set_seed_k(Df2Params* p, std::uint64_t seed) { p->seed = seed, p->draws = 0; }

// copy n rows of 4H into scratch [8][4H], zero above
__global__ void stage_k(const Df2Params* p, __nv_bfloat16* scratch) {
  const int r = blockIdx.x, n = p->ncommit;
  const uint4* src = reinterpret_cast<const uint4*>(p->commit_feats + std::size_t(r) * 4 * H);
  uint4* dst = reinterpret_cast<uint4*>(scratch + std::size_t(r) * 4 * H);
  for (int i = threadIdx.x; i < 4 * H / 8; i += blockDim.x) dst[i] = r < n ? src[i] : make_uint4(0, 0, 0, 0);
}

__global__ void round_k(Df2Params* p, draft::RoundInputs in, __nv_bfloat16* scratch) {
  const int m = *in.m, n = *in.accepted + 1, start = p->round_start;
  const __nv_bfloat16* feats = in.taps[0];
  for (int i = 0; i < in.nwidths; ++i)
    if (in.widths[i] == m) feats = in.taps[i];
  const int r = blockIdx.x;
  const uint4* src = reinterpret_cast<const uint4*>(feats + std::size_t(r) * 4 * H);
  uint4* dst = reinterpret_cast<uint4*>(scratch + std::size_t(r) * 4 * H);
  for (int i = threadIdx.x; i < 4 * H / 8; i += blockDim.x) dst[i] = r < n && r < m ? src[i] : make_uint4(0, 0, 0, 0);
  if (r == 0 && threadIdx.x == 0) {
    p->commit_feats = scratch;
    p->ncommit = n;
    p->start = start;
    p->anchor = p->temperature > 0.f ? *in.next_sampled : *in.next_greedy;
    p->P = start + n;
  }
}

// ---- norms / embedding
__global__ void __launch_bounds__(256) rmsnorm_k(const float* in, const __nv_bfloat16* w, __nv_bfloat16* out) {
  __shared__ float red[8];
  const int r = blockIdx.x;
  const float* x = in + std::size_t(r) * H;
  float ss = 0.f;
  for (int c = threadIdx.x; c < H; c += blockDim.x) ss += x[c] * x[c];
  const float inv = rsqrtf(block_sum(ss, red) / H + EPS);
  for (int c = threadIdx.x; c < H; c += blockDim.x) out[std::size_t(r) * H + c] = __float2bfloat16_rn(x[c] * inv * bf(w[c]));
}

// the target's input embedding (Inkling: bf16(rms(table row) * embed_norm), as SGLang's get_input_embeddings)
__global__ void __launch_bounds__(256) embed_k(const __nv_bfloat16* embed, const float* embed_norm, const Df2Params* p,
                                               const __nv_bfloat16* w, float* x, __nv_bfloat16* xn) {
  __shared__ float red[8];
  const int r = blockIdx.x;
  const std::size_t tok = std::size_t(r == 0 ? p->anchor : MASK_TOKEN);
  float s0 = 0.f;
  for (int c = threadIdx.x; c < H; c += blockDim.x) {
    const float v = bf(embed[tok * H + c]);
    s0 += v * v;
  }
  const float inv0 = rsqrtf(block_sum(s0, red) / H + EPS);
  float ss = 0.f;
  for (int c = threadIdx.x; c < H; c += blockDim.x) {
    const float v = bf(__float2bfloat16_rn(bf(embed[tok * H + c]) * inv0 * embed_norm[c]));
    x[std::size_t(r) * H + c] = v;
    ss += v * v;
  }
  const float inv = rsqrtf(block_sum(ss, red) / H + EPS);
  for (int c = threadIdx.x; c < H; c += blockDim.x)
    xn[std::size_t(r) * H + c] = __float2bfloat16_rn(x[std::size_t(r) * H + c] * inv * bf(w[c]));
}

// ---- dynamic two-tap convolutions over the block's 8 rows
__device__ __forceinline__ void conv_w(const float* dyn, const float* base, int t, int side, int c, float& w0, float& w1) {
  const int g = c >> 4;
  w0 = base[(side * 2 + 0) * H + c] + dyn[std::size_t(t) * NDYN + side * 288 + g];
  w1 = base[(side * 2 + 1) * H + c] + dyn[std::size_t(t) * NDYN + side * 288 + 144 + g];
}
__global__ void conv_in_k(const __nv_bfloat16* in, const float* dyn, const float* base, __nv_bfloat16* out) {
  const int t = blockIdx.x;
  for (int c = threadIdx.x; c < H; c += blockDim.x) {
    float w0, w1;
    conv_w(dyn, base, t, 0, c, w0, w1);
    float v = w0 * bf(in[std::size_t(t) * H + c]);
    if (t > 0) v += w1 * bf(in[std::size_t(t - 1) * H + c]);
    out[std::size_t(t) * H + c] = __float2bfloat16_rn(v);
  }
}
__global__ void conv_res_k(const float* in, const float* dyn, const float* base, float* x) {
  const int t = blockIdx.x;
  for (int c = threadIdx.x; c < H; c += blockDim.x) {
    float w0, w1;
    conv_w(dyn, base, t, 1, c, w0, w1);
    float v = w0 * in[std::size_t(t) * H + c];
    if (t > 0) v += w1 * in[std::size_t(t - 1) * H + c];
    x[std::size_t(t) * H + c] += v;
  }
}

// ---- rope (NeoX: rotate halves), position pos, dims d < 64 pair with d + 64
__device__ __forceinline__ void rope_pair(float& a, float& b, int d, int pos) {
  const float inv = powf(THETA, -float(2 * d) / float(HD));
  float sn, cs;
  sincosf(float(pos) * inv, &sn, &cs);
  const float x = a, y = b;
  a = x * cs - y * sn;
  b = y * cs + x * sn;
}

// one warp per (row, head): 32 q heads + 8 k heads + 8 v heads = 48 per row
__global__ void __launch_bounds__(256) qkv_post_k(const float* qkv, const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm,
                                                  const Df2Params* p, __nv_bfloat16* q_out, __nv_bfloat16* k_ring,
                                                  __nv_bfloat16* v_ring) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= BLOCK * 48) return;
  const int t = warp / 48, h = warp % 48, pos = p->P + t;
  const float* src = qkv + std::size_t(t) * QKV + (h < 32 ? h * HD : h < 40 ? QD + (h - 32) * HD : QD + KVD + (h - 40) * HD);
  float v[4];  // dims lane, lane + 32, lane + 64, lane + 96
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = src[lane + 32 * i];
  if (h < 40) {  // q / k: head RMS norm, rope
    float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float inv = rsqrtf(ss / HD + EPS);
    const __nv_bfloat16* nw = h < 32 ? q_norm : k_norm;
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = bf(__float2bfloat16_rn(v[i] * inv * bf(nw[lane + 32 * i])));
    rope_pair(v[0], v[2], lane, pos);
    rope_pair(v[1], v[3], lane + 32, pos);
  }
  __nv_bfloat16* dst = h < 32 ? q_out + (std::size_t(t) * NH + h) * HD
                              : (h < 40 ? k_ring : v_ring) + (std::size_t(WIN + t) * NKV + (h < 40 ? h - 32 : h - 40)) * HD;
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[lane + 32 * i] = __float2bfloat16_rn(v[i]);
}

// context rows: one warp per (row, k|v head)
__global__ void __launch_bounds__(256) ctx_store_k(const float* kv, int ld, const __nv_bfloat16* k_norm, const Df2Params* p,
                                                   __nv_bfloat16* k_ring, __nv_bfloat16* v_ring, std::int32_t* ring_pos,
                                                   bool write_pos) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (warp >= BLOCK * 16) return;
  const int r = warp / 16, h = warp % 16;
  if (r >= p->ncommit) return;
  const int pos = p->start + r, slot = pos % WIN;
  const float* src = kv + std::size_t(r) * ld + h * HD;  // k heads 0..7 then v heads
  float v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = src[lane + 32 * i];
  if (h < 8) {
    float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float inv = rsqrtf(ss / HD + EPS);
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = bf(__float2bfloat16_rn(v[i] * inv * bf(k_norm[lane + 32 * i])));
    rope_pair(v[0], v[2], lane, pos);
    rope_pair(v[1], v[3], lane + 32, pos);
  }
  __nv_bfloat16* dst = (h < 8 ? k_ring : v_ring) + (std::size_t(slot) * NKV + (h & 7)) * HD;
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[lane + 32 * i] = __float2bfloat16_rn(v[i]);
  if (write_pos && h == 0 && lane == 0) ring_pos[slot] = pos;
}

// ---- attention: grid (kv head g, query row t); 16 half-warps over the keys; lane c holds dims 8c..8c+7
__device__ __forceinline__ float hsum16(float x) {
#pragma unroll
  for (int o = 1; o < 16; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
  return x;
}
__device__ __forceinline__ void load8(const __nv_bfloat16* p, float (&f)[8]) {
  const uint4 u = *reinterpret_cast<const uint4*>(p);
  f[0] = __uint_as_float(u.x << 16), f[1] = __uint_as_float(u.x & 0xffff0000u);
  f[2] = __uint_as_float(u.y << 16), f[3] = __uint_as_float(u.y & 0xffff0000u);
  f[4] = __uint_as_float(u.z << 16), f[5] = __uint_as_float(u.z & 0xffff0000u);
  f[6] = __uint_as_float(u.w << 16), f[7] = __uint_as_float(u.w & 0xffff0000u);
}
constexpr int GQA = NH / NKV;  // 4
__global__ void __launch_bounds__(256) attn_k(const __nv_bfloat16* q, const __nv_bfloat16* kr, const __nv_bfloat16* vr,
                                              const std::int32_t* ring_pos, const Df2Params* p, __nv_bfloat16* out) {
  constexpr int NHW = 16;
  constexpr float SCALE = 0.08838834764831845f * 1.4426950408889634f;  // log2 domain
  const int g = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, hw = tid >> 4, c = tid & 15;
  const int P = p->P, qpos = P + t;
  __shared__ float red_m[NHW][GQA], red_l[NHW][GQA];
  __shared__ float red_o[NHW][GQA][HD];
  float qv[GQA][8];
#pragma unroll
  for (int i = 0; i < GQA; ++i) {
    load8(q + (std::size_t(t) * NH + g * GQA + i) * HD + c * 8, qv[i]);
#pragma unroll
    for (int e = 0; e < 8; ++e) qv[i][e] *= SCALE;
  }
  float mr[GQA], lr[GQA], acc[GQA][8];
#pragma unroll
  for (int i = 0; i < GQA; ++i) {
    mr[i] = -1e30f, lr[i] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
  }
  for (int base = 0; base < SLOTS; base += NHW) {
    const int j = base + hw;  // all lanes iterate together (full-mask shuffles)
    bool ok = j < SLOTS;
    if (ok && j < WIN) {
      const int kp = ring_pos[j];
      ok = kp >= 0 && kp < P && qpos - kp < WIN;
    }
    float kf[8], vf[8];
    const int js = j < SLOTS ? j : 0;
    load8(kr + (std::size_t(js) * NKV + g) * HD + c * 8, kf);
    load8(vr + (std::size_t(js) * NKV + g) * HD + c * 8, vf);
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
      float s = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) s = fmaf(qv[i][e], kf[e], s);
      s = hsum16(s);
      s = ok ? s : -INFINITY;
      const float mn = fmaxf(mr[i], s), a = exp2f(mr[i] - mn), pe = exp2f(s - mn);
      lr[i] = lr[i] * a + pe;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][e] = fmaf(acc[i][e], a, pe * vf[e]);
      mr[i] = mn;
    }
  }
  if (c == 0)
#pragma unroll
    for (int i = 0; i < GQA; ++i) red_m[hw][i] = mr[i], red_l[hw][i] = lr[i];
#pragma unroll
  for (int i = 0; i < GQA; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) red_o[hw][i][c * 8 + e] = acc[i][e];
  __syncthreads();
  for (int it = tid; it < GQA * HD; it += blockDim.x) {
    const int i = it / HD, d = it % HD;
    float M = -1e30f;
    for (int w = 0; w < NHW; ++w) M = fmaxf(M, red_m[w][i]);
    float l = 0.f, o = 0.f;
    for (int w = 0; w < NHW; ++w) {
      const float f = exp2f(red_m[w][i] - M);
      l = fmaf(red_l[w][i], f, l);
      o = fmaf(red_o[w][i][d], f, o);
    }
    out[std::size_t(t) * QD + (g * GQA + i) * HD + d] = __float2bfloat16_rn(o / l);
  }
}

// ---- top-16 per row (exact), two stages. Stage 1: blocks of 256 threads x 32 values (8192 per block, 25 per
// row); each warp takes its 1024 values' top 16 by 16 shuffle argmax rounds (values stay in registers, static
// indexing), warp 0 merges the 8 warps' lists into the block's top 16. Stage 2: one warp per row merges the
// blocks' lists. Ties go to the lower index. (Was one block per row with spilled insertion lists: 466 us.)
constexpr int TK_PER = 32, TK_THREADS = 256, TK_CHUNK = TK_PER * TK_THREADS, TK_MAXB = 32, TK_MAXROWS = 8;
__device__ float g_tk_val[TK_MAXROWS * TK_MAXB * TOPK];
__device__ int g_tk_idx[TK_MAXROWS * TK_MAXB * TOPK];

// (v, i) of the larger, ties to the lower index
__device__ __forceinline__ void tk_better(float& v, int& i, float ov, int oi) {
  if (ov > v || (ov == v && unsigned(oi) < unsigned(i))) v = ov, i = oi;
}
__device__ __forceinline__ void tk_warp_argmax(float& v, int& i) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) tk_better(v, i, __shfl_xor_sync(0xffffffffu, v, o), __shfl_xor_sync(0xffffffffu, i, o));
}
// top 16 of the warp's N values per lane (vals[k] at index idx(k)); lane 0 gets them in out_v/out_i (descending)
template <int N, class Idx>
__device__ __forceinline__ void tk_warp_top16(float (&vals)[N], Idx idx, float* out_v, int* out_i) {
  const int lane = threadIdx.x & 31;
  for (int it = 0; it < TOPK; ++it) {
    float m = -FLT_MAX;
    int mk = 0, mi = -1;  // mi: the index of the lane's best (static k: no local-memory indexing)
#pragma unroll
    for (int k = 0; k < N; ++k)
      if (vals[k] > m) m = vals[k], mk = k, mi = idx(k);  // first (lowest k) on ties
    float v = m;
    int i = m == -FLT_MAX ? -1 : mi;
    if (i < 0) v = -FLT_MAX, i = 0x7fffffff;
    tk_warp_argmax(v, i);
    if (lane == 0) out_v[it] = v, out_i[it] = i == 0x7fffffff ? -1 : i;
    if (m != -FLT_MAX && mi == i) {
#pragma unroll
      for (int k = 0; k < N; ++k)
        if (k == mk) vals[k] = -FLT_MAX;
    }
  }
}

__global__ void __launch_bounds__(TK_THREADS) topk16_stage1(const float* logits, int V) {
  const int r = blockIdx.y, b = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const float* x = logits + std::size_t(r) * V;
  // warp w owns the block's elements [w * 1024, (w + 1) * 1024): lane l holds w * 1024 + k * 32 + l
  const int base = b * TK_CHUNK + warp * (TK_PER * 32) + lane;
  float vals[TK_PER];
#pragma unroll
  for (int k = 0; k < TK_PER; ++k) {
    const int i = base + k * 32;
    vals[k] = i < V ? x[i] : -FLT_MAX;
  }
  __shared__ float wv[TK_THREADS / 32][TOPK];
  __shared__ int wi[TK_THREADS / 32][TOPK];
  tk_warp_top16(vals, [&](int k) { return base + k * 32; }, wv[warp], wi[warp]);
  __syncthreads();
  if (warp == 0) {
    float cv[TK_THREADS / 32 * TOPK / 32];  // 128 candidates, 4 per lane
    int ci[TK_THREADS / 32 * TOPK / 32];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int c = k * 32 + lane;
      cv[k] = wi[c / TOPK][c % TOPK] < 0 ? -FLT_MAX : wv[c / TOPK][c % TOPK];
      ci[k] = wi[c / TOPK][c % TOPK];
    }
    float* ov = g_tk_val + (std::size_t(r) * TK_MAXB + b) * TOPK;
    int* oi = g_tk_idx + (std::size_t(r) * TK_MAXB + b) * TOPK;
    tk_warp_top16(cv, [&](int k) { return ci[k]; }, ov, oi);
  }
}

__global__ void __launch_bounds__(32) topk16_stage2(int nblocks, std::int32_t* cand_id, float* cand_val) {
  const int r = blockIdx.x, lane = threadIdx.x;
  constexpr int N = TK_MAXB * TOPK / 32;  // 16 per lane
  float cv[N];
  int ci[N];
#pragma unroll
  for (int k = 0; k < N; ++k) {
    const int c = k * 32 + lane;  // block c / 16, rank c % 16
    const bool ok = c < nblocks * TOPK;
    const int id = ok ? g_tk_idx[std::size_t(r) * TK_MAXB * TOPK + c] : -1;
    ci[k] = id;
    cv[k] = id < 0 ? -FLT_MAX : g_tk_val[std::size_t(r) * TK_MAXB * TOPK + c];
  }
  __shared__ float fv[TOPK];
  __shared__ int fi[TOPK];
  tk_warp_top16(cv, [&](int k) { return ci[k]; }, fv, fi);
  __syncwarp();
  if (lane < TOPK) cand_id[r * TOPK + lane] = fi[lane], cand_val[r * TOPK + lane] = fv[lane];
}

// ---- selector walk (one block of RANK threads)
__global__ void __launch_bounds__(RANK) select_k(const std::int32_t* cand_id, const float* cand_val, const float* gate,
                                                 const __nv_bfloat16* pred, const __nv_bfloat16* succ, Df2Params* p,
                                                 std::int32_t* tokens, float* draft_logits, int V, std::int32_t* prev_cand,
                                                 float* conf, float* qstats, float dts) {
  const int tid = threadIdx.x;
  __shared__ float part[RANK / 32][TOPK];
  __shared__ float s[TOPK];
  __shared__ int prev_tok;
  const float T = p->temperature;
  if (tid == 0) prev_tok = p->anchor;
  __syncthreads();
  for (int i = 0; i < GAMMA; ++i) {
    // reset the row's previous candidates
    if (tid < TOPK) {
      const int pc = prev_cand[i * TOPK + tid];
      if (pc >= 0 && pc < V) draft_logits[std::size_t(i) * V + pc] = -INFINITY;
    }
    const float cond = bf(pred[std::size_t(prev_tok) * RANK + tid]) * gate[i * RANK + tid];
#pragma unroll
    for (int c = 0; c < TOPK; ++c) {
      const int cid = cand_id[i * TOPK + c];
      float v = cid >= 0 ? bf(succ[std::size_t(cid) * RANK + tid]) * cond : 0.f;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if ((tid & 31) == 0) part[tid >> 5][c] = v;
    }
    __syncthreads();
    if (tid < TOPK) {
      float d = 0.f;
#pragma unroll
      for (int w = 0; w < RANK / 32; ++w) d += part[w][tid];
      const int cid = cand_id[i * TOPK + tid];
      s[tid] = cid >= 0 ? SOFTCAP * tanhf((d + cand_val[i * TOPK + tid]) / SOFTCAP) : -INFINITY;
    }
    __syncthreads();
    if (tid == 0) {
      float mx = -INFINITY;
      int best = 0;
      for (int c = 0; c < TOPK; ++c)
        if (s[c] > mx) mx = s[c], best = c;
      // confidence: max softmax probability (T = 1)
      float z = 0.f;
      for (int c = 0; c < TOPK; ++c) z += expf(s[c] - mx);
      conf[i] = 1.f / z;
      if (T > 0.f) {  // Gumbel-max at the draft temperature Tq = T * dts (quill: dts = 1 by default)
        const float Tq = T * dts;
        float bm = -INFINITY, qm = -INFINITY;
        for (int c = 0; c < TOPK; ++c) {
          const float g = s[c] / Tq + draft::gumbel(p->seed, p->draws, std::uint32_t(i), std::uint32_t(c));
          if (g > bm) bm = g, best = c;
          qm = fmaxf(qm, s[c] / Tq);
        }
        float zq = 0.f;
        for (int c = 0; c < TOPK; ++c) zq += expf(s[c] / Tq - qm);
        qstats[2 * i] = qm, qstats[2 * i + 1] = zq;
      } else {
        qstats[2 * i] = mx, qstats[2 * i + 1] = z;
      }
      prev_tok = cand_id[i * TOPK + best];
      tokens[i] = prev_tok;
    }
    if (tid < TOPK) {
      const int cid = cand_id[i * TOPK + tid];
      // the acceptance reads q = softmax(draft_logits / T): logits / dts make that the sampled softmax(s / Tq)
      if (cid >= 0 && cid < V) draft_logits[std::size_t(i) * V + cid] = T > 0.f ? s[tid] / dts : s[tid];
      prev_cand[i * TOPK + tid] = cid;
    }
    __syncthreads();
  }
  if (tid == 0) {
    p->draws += 1;
    p->round_start = p->P;
  }
}

void check() { CUDA_CHECK(cudaGetLastError()); }

}  // namespace

void set_draft(Df2Params* p, int anchor, int P, float t, cudaStream_t s) { set_draft_k<<<1, 1, 0, s>>>(p, anchor, P, t), check(); }
void set_commit(Df2Params* p, const __nv_bfloat16* f, int n, int start, cudaStream_t s) {
  set_commit_k<<<1, 1, 0, s>>>(p, f, n, start), check();
}
void set_seed(Df2Params* p, std::uint64_t seed, cudaStream_t s) { set_seed_k<<<1, 1, 0, s>>>(p, seed), check(); }
void round_begin(Df2Params* p, const draft::RoundInputs& in, __nv_bfloat16* scratch, cudaStream_t s) {
  round_k<<<BLOCK, 256, 0, s>>>(p, in, scratch), check();
}
void stage_commit(const Df2Params* p, __nv_bfloat16* scratch, cudaStream_t s) { stage_k<<<BLOCK, 256, 0, s>>>(p, scratch), check(); }
void rmsnorm(const float* in, const __nv_bfloat16* w, int rows, __nv_bfloat16* out, cudaStream_t s) {
  rmsnorm_k<<<rows, 256, 0, s>>>(in, w, out), check();
}
void embed_block(const __nv_bfloat16* embed, const float* embed_norm, const Df2Params* p, const __nv_bfloat16* w,
                 float* x, __nv_bfloat16* xn, cudaStream_t s) {
  embed_k<<<BLOCK, 256, 0, s>>>(embed, embed_norm, p, w, x, xn), check();
}
void conv_in(const __nv_bfloat16* in, const float* dyn, const float* base, __nv_bfloat16* out, cudaStream_t s) {
  conv_in_k<<<BLOCK, 256, 0, s>>>(in, dyn, base, out), check();
}
void conv_residual(const float* in, const float* dyn, const float* base, float* x, cudaStream_t s) {
  conv_res_k<<<BLOCK, 256, 0, s>>>(in, dyn, base, x), check();
}
void qkv_post(const float* qkv, const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm, const Df2Params* p,
              __nv_bfloat16* q_out, __nv_bfloat16* k_ring, __nv_bfloat16* v_ring, cudaStream_t s) {
  qkv_post_k<<<BLOCK * 48 / 8, 256, 0, s>>>(qkv, q_norm, k_norm, p, q_out, k_ring, v_ring), check();
}
void ctx_store(const float* kv, int ld, const __nv_bfloat16* k_norm, const Df2Params* p, __nv_bfloat16* k_ring,
               __nv_bfloat16* v_ring, std::int32_t* ring_pos, bool write_pos, cudaStream_t s) {
  ctx_store_k<<<BLOCK * 16 / 8, 256, 0, s>>>(kv, ld, k_norm, p, k_ring, v_ring, ring_pos, write_pos), check();
}
void attention(const __nv_bfloat16* q, const __nv_bfloat16* k_ring, const __nv_bfloat16* v_ring,
               const std::int32_t* ring_pos, const Df2Params* p, __nv_bfloat16* out, cudaStream_t s) {
  attn_k<<<dim3(NKV, BLOCK), 256, 0, s>>>(q, k_ring, v_ring, ring_pos, p, out), check();
}
void topk16(const float* logits, int rows, int V, std::int32_t* cand_id, float* cand_val, cudaStream_t s) {
  const int nb = (V + TK_CHUNK - 1) / TK_CHUNK;
  if (nb > TK_MAXB || rows > TK_MAXROWS) throw std::invalid_argument("topk16: vocabulary or rows above the stage-1 scratch");
  topk16_stage1<<<dim3(nb, rows), TK_THREADS, 0, s>>>(logits, V), check();
  topk16_stage2<<<rows, 32, 0, s>>>(nb, cand_id, cand_val), check();
}
void select(const std::int32_t* cand_id, const float* cand_val, const float* gate, const __nv_bfloat16* pred,
            const __nv_bfloat16* succ, Df2Params* p, std::int32_t* tokens, float* draft_logits, int V,
            std::int32_t* prev_cand, float* conf, float* qstats, cudaStream_t s, float dts) {
  select_k<<<1, RANK, 0, s>>>(cand_id, cand_val, gate, pred, succ, p, tokens, draft_logits, V, prev_cand, conf, qstats, dts),
      check();
}

__global__ void map_ids_kernel(std::int32_t* ids, int n, const std::int32_t* __restrict__ map) {
  const int i = threadIdx.x;
  if (i < n) ids[i] = map[ids[i]];
}

void map_ids(std::int32_t* ids, int n, const std::int32_t* map, cudaStream_t s) {
  map_ids_kernel<<<1, 128, 0, s>>>(ids, n, map);
}

}  // namespace ingine::kernels::dflash2
