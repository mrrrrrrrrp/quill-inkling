#include "model/model_kernels.h"

#include "core/device.h"

#include <stdexcept>

namespace ingine::model {
namespace {

__global__ void f32_to_bf16_rows_kernel(const float* __restrict__ src, int rows, int cols,
                                        __nv_bfloat16* __restrict__ dst, long long dst_stride) {
  const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= (long long)rows * cols) return;
  const long long r = i / cols, c = i % cols;
  dst[r * dst_stride + c] = __float2bfloat16_rn(src[i]);
}

constexpr int kThreads = 256;
constexpr int kMaxRows = 8;                      // widest decode call
constexpr int kRowVec = CONV_ROW_FLOATS / 4;     // float4 per visit across the four conv states (1664)
// float4 per conv-state row of all visits; the visit counts come with the state (checked() fills them in)
__host__ __device__ inline int conv_vecs(const StatePtrs& st) { return st.visits * kRowVec; }
constexpr int kRingVec = KV_DIM * 2 / 16;        // uint4 per ring slot row (128)

// float4 q of [visit][k | v | attn | mlp] -> (tensor, visit, float4 column, float4 per row of the tensor)
struct ConvAt {
  int t, v, c, cols;
};
__device__ __forceinline__ ConvAt conv_at(int q) {
  const int v = q / kRowVec, c = q % kRowVec;
  constexpr int k4 = KV_DIM / 4, h4 = HIDDEN / 4;
  if (c < k4) return {0, v, c, k4};
  if (c < 2 * k4) return {1, v, c - k4, k4};
  if (c < 2 * k4 + h4) return {2, v, c - 2 * k4, h4};
  return {3, v, c - 2 * k4 - h4, h4};
}

__device__ __forceinline__ float4* conv_row(float* const* conv, const ConvAt& a, int row) {
  return reinterpret_cast<float4*>(conv[a.t]) + (std::size_t(a.v) * 3 + row) * a.cols + a.c;
}

// quill: rows of rv 16-byte vectors (128 BF16, 36 q4)
__device__ __forceinline__ uint4* ring_row(__nv_bfloat16* ring, int r, int slot, int rv) {
  return reinterpret_cast<uint4*>(ring) + (std::size_t(r) * WINDOW + slot) * std::size_t(rv);
}

unsigned blocks_for(long long items) { return unsigned((items + kThreads - 1) / kThreads); }

struct StepInputs {
  long long tokens[kMaxRows];
  long long pos;
  int m;
};

__global__ void set_step_inputs_kernel(StepInputs in, long long* __restrict__ token, long long* __restrict__ pos) {
  const int i = threadIdx.x;
  if (i < in.m) token[i] = in.tokens[i];
  if (i == 0) *pos = in.pos;
}

// items: conv row 0 (conv_vecs(st) float4), ring k and v slots (40 * 2 * kRingVec uint4), ring positions (40)
__global__ void chain_advance_kernel(const int* __restrict__ sampled, long long* __restrict__ token,
                                     long long* __restrict__ pos, int position, volatile int* host_token,
                                     StatePtrs st, StepUndo undo) {
  const int kRingVecR = st.ring_vecs;  // quill: 128 (BF16 rows) or 36 (q4 rows)
  const int i = blockIdx.x * kThreads + threadIdx.x;
  if (i == 0) {
    const int t = *sampled;
    *token = t;
    *pos = position;
    *host_token = t;
  }
  const int slot = position % WINDOW;
  if (i < conv_vecs(st)) {
    const ConvAt a = conv_at(i);
    reinterpret_cast<float4*>(undo.conv)[i] = *conv_row(st.conv, a, 0);
    return;
  }
  int j = i - conv_vecs(st);
  if (j < st.sliding * 2 * kRingVecR) {
    const int r = j / (2 * kRingVecR), k = j % (2 * kRingVecR);
    if (k < kRingVecR)
      reinterpret_cast<uint4*>(undo.k)[r * kRingVecR + k] = ring_row(st.k_ring, r, slot, kRingVecR)[k];
    else
      reinterpret_cast<uint4*>(undo.v)[r * kRingVecR + k - kRingVecR] = ring_row(st.v_ring, r, slot, kRingVecR)[k - kRingVecR];
    return;
  }
  j -= st.sliding * 2 * kRingVecR;
  if (j < st.sliding) undo.pos[j] = st.ring_pos[std::size_t(j) * WINDOW + slot];
}

__global__ void chain_undo_kernel(StatePtrs st, StepUndo undo, int position) {
  const int kRingVecR = st.ring_vecs;  // quill: 128 (BF16 rows) or 36 (q4 rows)
  const int i = blockIdx.x * kThreads + threadIdx.x;
  const int slot = position % WINDOW;
  if (i < conv_vecs(st)) {
    const ConvAt a = conv_at(i);
    float4* r0 = conv_row(st.conv, a, 0);
    float4* r1 = conv_row(st.conv, a, 1);
    float4* r2 = conv_row(st.conv, a, 2);
    const float4 x0 = *r0, x1 = *r1;
    *r2 = x1;
    *r1 = x0;
    *r0 = reinterpret_cast<const float4*>(undo.conv)[i];
    return;
  }
  int j = i - conv_vecs(st);
  if (j < st.sliding * 2 * kRingVecR) {
    const int r = j / (2 * kRingVecR), k = j % (2 * kRingVecR);
    if (k < kRingVecR)
      ring_row(st.k_ring, r, slot, kRingVecR)[k] = reinterpret_cast<const uint4*>(undo.k)[r * kRingVecR + k];
    else
      ring_row(st.v_ring, r, slot, kRingVecR)[k - kRingVecR] = reinterpret_cast<const uint4*>(undo.v)[r * kRingVecR + k - kRingVecR];
    return;
  }
  j -= st.sliding * 2 * kRingVecR;
  if (j < st.sliding) st.ring_pos[std::size_t(j) * WINDOW + slot] = undo.pos[j];
}

// items: conv rows 1, 2 (2 * conv_vecs(st)), ring k/v rows r < m (40 * m * 2 * kRingVec), positions (40 * m)
__global__ void verify_save_kernel(StatePtrs st, VerifyBackup bk, const long long* __restrict__ pos_dev,
                                   int pos_host, int m) {
  const int kRingVecR = st.ring_vecs;  // quill: 128 (BF16 rows) or 36 (q4 rows)
  const int i = blockIdx.x * kThreads + threadIdx.x;
  if (i < 2 * conv_vecs(st)) {
    const int row = 1 + i / conv_vecs(st);
    const ConvAt a = conv_at(i % conv_vecs(st));
    *conv_row(bk.conv, a, row) = *conv_row(st.conv, a, row);
    return;
  }
  const int P = pos_dev ? int(*pos_dev) : pos_host;
  int j = i - 2 * conv_vecs(st);
  const int per_ring = m * 2 * kRingVecR;
  if (j < st.sliding * per_ring) {
    const int r = j / per_ring, k = j % per_ring;
    const int row = k / (2 * kRingVecR), c = k % (2 * kRingVecR);
    const int slot = (P + row) % WINDOW;
    const std::size_t b = (std::size_t(r) * kMaxRows + row) * kRingVecR;
    if (c < kRingVecR)
      reinterpret_cast<uint4*>(bk.k)[b + c] = ring_row(st.k_ring, r, slot, kRingVecR)[c];
    else
      reinterpret_cast<uint4*>(bk.v)[b + c - kRingVecR] = ring_row(st.v_ring, r, slot, kRingVecR)[c - kRingVecR];
    return;
  }
  j -= st.sliding * per_ring;
  if (j < st.sliding * m) {
    const int r = j / m, row = j % m;
    bk.pos[r * kMaxRows + row] = st.ring_pos[std::size_t(r) * WINDOW + (P + row) % WINDOW];
  }
}

struct Hist {
  const float* p[4];
};

// items: conv rows 0..2 (3 * conv_vecs(st)), ring k/v rows r < m (only r >= commit restored), positions (40 * m)
__global__ void verify_rollback_kernel(StatePtrs st, VerifyBackup bk, Hist hist, const long long* __restrict__ pos_dev,
                                       int pos_host, const int* __restrict__ accepted_dev, int commit_host, int m) {
  const int kRingVecR = st.ring_vecs;  // quill: 128 (BF16 rows) or 36 (q4 rows)
  const int commit = accepted_dev ? *accepted_dev + 1 : commit_host;
  if (commit >= m) return;
  const int i = blockIdx.x * kThreads + threadIdx.x;
  if (i < 3 * conv_vecs(st)) {
    const int j = i / conv_vecs(st), src = commit + j;
    const ConvAt a = conv_at(i % conv_vecs(st));
    const float4 x = src < 3 ? *conv_row(bk.conv, a, src)
                             : reinterpret_cast<const float4*>(hist.p[a.t])[(std::size_t(a.v) * m + src - 3) * a.cols + a.c];
    *conv_row(st.conv, a, j) = x;
    return;
  }
  const int P = pos_dev ? int(*pos_dev) : pos_host;
  int j = i - 3 * conv_vecs(st);
  const int per_ring = m * 2 * kRingVecR;
  if (j < st.sliding * per_ring) {
    const int r = j / per_ring, k = j % per_ring;
    const int row = k / (2 * kRingVecR), c = k % (2 * kRingVecR);
    if (row < commit) return;
    const int slot = (P + row) % WINDOW;
    const std::size_t b = (std::size_t(r) * kMaxRows + row) * kRingVecR;
    if (c < kRingVecR)
      ring_row(st.k_ring, r, slot, kRingVecR)[c] = reinterpret_cast<const uint4*>(bk.k)[b + c];
    else
      ring_row(st.v_ring, r, slot, kRingVecR)[c - kRingVecR] = reinterpret_cast<const uint4*>(bk.v)[b + c - kRingVecR];
    return;
  }
  j -= st.sliding * per_ring;
  if (j < st.sliding * m) {
    const int r = j / m, row = j % m;
    if (row >= commit) st.ring_pos[std::size_t(r) * WINDOW + (P + row) % WINDOW] = bk.pos[r * kMaxRows + row];
  }
}

__device__ __forceinline__ unsigned width_index(const Widths& widths, int m) {
  unsigned idx = 0;
  while (int(idx) < widths.n && widths.w[idx] != m) ++idx;
  return idx;  // widths.n (no body) if m is not a width
}

__global__ void verify_select_kernel(const float* __restrict__ conf, const int* __restrict__ drafts, float threshold,
                                     Widths widths, long long* __restrict__ token, int* __restrict__ m_out,
                                     cudaGraphConditionalHandle handle) {
  const int i = threadIdx.x;
  if (i < 7) token[1 + i] = drafts[i];
  if (i == 0) {
    int lead = 0;
    while (lead < 7 && conf[lead] > threshold) ++lead;
    int m = widths.w[widths.n - 1];
    for (int k = widths.n - 1; k >= 0; --k)
      if (widths.w[k] >= lead + 1) m = widths.w[k];
    *m_out = m;
    cudaGraphSetConditional(handle, width_index(widths, m));
  }
}

__global__ void width_select_kernel(const int* __restrict__ m, Widths widths, cudaGraphConditionalHandle handle) {
  cudaGraphSetConditional(handle, width_index(widths, *m));
}

__global__ void verify_publish_kernel(const int* __restrict__ m, const int* __restrict__ accepted,
                                      const int* __restrict__ next, const long long* __restrict__ token,
                                      volatile VerifyRecord* rec, const float* __restrict__ conf) {
  const int i = threadIdx.x;
  if (i < 7) rec->drafts[i] = int(token[1 + i]), rec->conf[i] = conf ? conf[i] : 0.f;
  if (i == 0) {
    rec->m = *m;
    rec->accepted = *accepted;
    if (next) rec->next = *next;
  }
}

StatePtrs checked(const StatePtrs& in) {
  StatePtrs st = in;
  st.visits = num_visits(), st.sliding = num_sliding_visits();
  for (float* c : st.conv)
    if (!c) throw std::invalid_argument("model kernels: null conv state");
  if (!st.k_ring || !st.v_ring || !st.ring_pos) throw std::invalid_argument("model kernels: null ring");
  return st;
}

}  // namespace

namespace {
__global__ void set_stage_factors_kernel(const int* __restrict__ slot_map, int first, const float* __restrict__ fac,
                                         float* __restrict__ f13, float* __restrict__ f2) {
  const int e = threadIdx.x;
  const int slot = slot_map[e];
  if (slot >= first) f13[slot] = fac[2 * e], f2[slot] = fac[2 * e + 1];
}
}  // namespace

void set_stage_factors(const int* slot_map, int first_staging, const float* fac, float* f13, float* f2, cudaStream_t stream) {
  set_stage_factors_kernel<<<1, 128, 0, stream>>>(slot_map, first_staging, fac, f13, f2);
  CUDA_CHECK(cudaGetLastError());
}

namespace {
__global__ void gather_rows_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, const int* __restrict__ idx,
                                   int vecs) {
  const uint4* s = src + size_t(idx[blockIdx.x]) * vecs;
  uint4* d = dst + size_t(blockIdx.x) * vecs;
  for (int i = threadIdx.x; i < vecs; i += blockDim.x) d[i] = s[i];
}
}  // namespace

void gather_rows(const void* src, void* dst, const int* idx, int rows, std::size_t row_bytes, cudaStream_t stream) {
  if (rows <= 0) return;
  gather_rows_kernel<<<rows, 128, 0, stream>>>(static_cast<const uint4*>(src), static_cast<uint4*>(dst), idx,
                                              int(row_bytes / 16));
  CUDA_CHECK(cudaGetLastError());
}

void f32_to_bf16_rows(const float* src, int rows, int cols, __nv_bfloat16* dst, std::int64_t dst_stride,
                      cudaStream_t stream) {
  const long long n = (long long)rows * cols;
  if (n == 0) return;
  f32_to_bf16_rows_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(src, rows, cols, dst, dst_stride);
  CUDA_CHECK(cudaGetLastError());
}

void set_step_inputs(const std::int32_t* tokens, int m, int position, std::int64_t* token, std::int64_t* pos,
                     cudaStream_t stream) {
  if (m < 1 || m > kMaxRows) throw std::invalid_argument("set_step_inputs: width out of range");
  StepInputs in{};
  for (int i = 0; i < m; ++i) in.tokens[i] = tokens[i];
  in.pos = position;
  in.m = m;
  set_step_inputs_kernel<<<1, 32, 0, stream>>>(in, reinterpret_cast<long long*>(token),
                                               reinterpret_cast<long long*>(pos));
  CUDA_CHECK(cudaGetLastError());
}

void chain_advance(const std::int32_t* sampled, std::int64_t* token, std::int64_t* pos, int position,
                   std::int32_t* host_token, const StatePtrs& st_in, const StepUndo& undo, cudaStream_t stream) {
  const StatePtrs st = checked(st_in);  // with the visit counts of the loop pass plan
  const long long items = conv_vecs(st) + st.sliding * (2 * kRingVec + 1);
  chain_advance_kernel<<<blocks_for(items), kThreads, 0, stream>>>(sampled, reinterpret_cast<long long*>(token),
                                                                   reinterpret_cast<long long*>(pos), position,
                                                                   host_token, checked(st), undo);
  CUDA_CHECK(cudaGetLastError());
}

void chain_undo(const StatePtrs& st_in, const StepUndo& undo, int position, cudaStream_t stream) {
  const StatePtrs st = checked(st_in);
  const long long items = conv_vecs(st) + st.sliding * (2 * kRingVec + 1);
  chain_undo_kernel<<<blocks_for(items), kThreads, 0, stream>>>(checked(st), undo, position);
  CUDA_CHECK(cudaGetLastError());
}

void verify_save(const StatePtrs& st_in, const VerifyBackup& bk, const std::int64_t* pos_dev, int pos_host, int m,
                 cudaStream_t stream) {
  if (m < 1 || m > kMaxRows) throw std::invalid_argument("verify_save: width out of range");
  const StatePtrs st = checked(st_in);
  const long long items = 2LL * conv_vecs(st) + st.sliding * m * (2 * kRingVec + 1);
  verify_save_kernel<<<blocks_for(items), kThreads, 0, stream>>>(
      checked(st), bk, reinterpret_cast<const long long*>(pos_dev), pos_host, m);
  CUDA_CHECK(cudaGetLastError());
}

void verify_rollback(const StatePtrs& st_in, const VerifyBackup& bk, const float* const hist[4],
                     const std::int64_t* pos_dev, int pos_host, const std::int32_t* accepted_dev, int commit_host,
                     int m, cudaStream_t stream) {
  if (m < 1 || m > kMaxRows) throw std::invalid_argument("verify_rollback: width out of range");
  if (!accepted_dev && (commit_host < 1 || commit_host > m))
    throw std::invalid_argument("verify_rollback: commit out of range");
  const StatePtrs st = checked(st_in);
  const long long items = 3LL * conv_vecs(st) + st.sliding * m * (2 * kRingVec + 1);
  const Hist h{{hist[0], hist[1], hist[2], hist[3]}};
  verify_rollback_kernel<<<blocks_for(items), kThreads, 0, stream>>>(
      checked(st), bk, h, reinterpret_cast<const long long*>(pos_dev), pos_host, accepted_dev, commit_host, m);
  CUDA_CHECK(cudaGetLastError());
}

void verify_select(const float* conf, const std::int32_t* drafts, float threshold, const Widths& widths,
                   std::int64_t* token, std::int32_t* m_out, cudaGraphConditionalHandle handle, cudaStream_t stream) {
  if (widths.n < 1 || widths.n > 8) throw std::invalid_argument("verify_select: 1 to 8 widths");
  verify_select_kernel<<<1, 32, 0, stream>>>(conf, drafts, threshold, widths, reinterpret_cast<long long*>(token),
                                             m_out, handle);
  CUDA_CHECK(cudaGetLastError());
}

void width_select(const std::int32_t* m, const Widths& widths, cudaGraphConditionalHandle handle,
                  cudaStream_t stream) {
  width_select_kernel<<<1, 1, 0, stream>>>(m, widths, handle);
  CUDA_CHECK(cudaGetLastError());
}

void verify_publish(const std::int32_t* m, const std::int32_t* accepted, const std::int32_t* next,
                    const std::int64_t* token, VerifyRecord* rec, cudaStream_t stream, const float* conf) {
  verify_publish_kernel<<<1, 32, 0, stream>>>(m, accepted, next, reinterpret_cast<const long long*>(token), rec, conf);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::model
