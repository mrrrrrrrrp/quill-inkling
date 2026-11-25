# quill: Inkling 2 Mini on RTX 4070 SUPER 12 GB (cap 11.3 GiB total) + Ryzen 7 5800X3D + 32 GB DDR4-3000

## Targets and constraints (from the user)
- Inkling 2 Mini, abliterated/heretic preferred; experts >= 3-bit, accuracy close to the full-precision model.
- Fastest decode and prefill; speculative decoding (incoai DFlash2 drafter preferred over the official DSpark).
- >= 300k context, q4 KV cache.
- Device memory (all processes, Task Manager "dedicated") <= 11.3 GiB = 11571 MiB.
- Purpose-built engine (user rejected llama.cpp, NInfer, Strata). Everything downloaded under the install folder.

## Machine (measured 2026-11-24, engine/tools/hwbench.cu)
- GPU read bandwidth 414 GB/s (1 GiB), 468 GB/s (64 MiB, L2 48 MB); D2D copy 298 GB/s (r+w).
- FP16 GEMM (fp32 acc) 71.5-75.5 TFLOPS; fp16 acc 126-146; int8 252-256 TOPS.
- PCIe 4.0 x16: H2D 26.2 GB/s, D2H 26.2 GB/s; synchronous 9 KB H2D+D2H round trip 107 us (WDDM).
- CPU DRAM read 33.5 GB/s (4-16 threads), 29.4 GB/s single thread.
- Windows desktop + apps use ~0.9-1.0 GiB of device memory at idle.

## Pins
- thinkingmachines/Inkling-2-Mini 7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6
- satellitedown/Inkling-2-Mini-abliterated-ingine cea73d67 (ablation file + ingine engine source @556acc7c, the base of quill)
- incoai/Inkling-2-Mini-DFlash2 4f9088ea66c21f2b99ed657d1941c306b03e84d2
- CUDA 13.3, MSVC 19.44, driver 591.86

## Log
### 2026-11-24 06:42 baseline B0
- Config: native experts (NVFP4 L0-19, MXFP8 L20-23), dense FP8 rows, q4 (IQ4_NL) full-attn KV, max_ctx 32768,
  1444 NVFP4 experts resident (loop layers first), 1628 on the CPU tier (7 AVX2 threads), no ablation, no spec.
- ingine-bench --tokens 128 --runs 1 --effort none: chat 19.4 tok/s, code 19.4 tok/s. Output coherent.
- Peak device memory 11133 MiB (watchdog, nvidia-smi), engine reported 11398 MiB after pools (cudaMemGetInfo).
- Expected ~40 tok/s from bytes (626 MB/token on the CPU tier @ 33.5 GB/s + ~5 ms GPU): profile next.

### 06:45 B1: CPU tier dot kernels templated on the token count (accumulators in registers)  -> ACCEPT
- B0 19.0/18.4 -> 29.6/31.1 tok/s (chat/code). CPU tier busy 42.6 -> 22.2 ms/token for 114 experts/token
  (625 MB): 14.7 -> 28.2 GB/s. Cause: runtime-indexed __m256 acc[8] spilled to the stack (MSVC).

### 06:49-07:03 accuracy harness (quill-kld): WikiText-2 test head, 2 x 1024 tokens, teacher-forced decode path
- Reference R0: dense BF16 rows, BF16 full KV, native experts (NVFP4 L0-19, MXFP8 L20-23): ppl 6.4784.
- T1 (B1 config: dense FP8 rows, IQ4_NL q4 KV, native experts): mean KLD 0.00561, median 0.00253, p99 0.0503,
  top-1 96.87%, ppl 6.4787.
- T2 (T1 + MXFP8 L20-23 re-encoded as NVFP4 at load): KLD 0.00719, median 0.00338, p99 0.0632, top-1 96.58%,
  ppl 6.4862 (+0.12%).
- (A first attempt reported KLD 10.4: tool bug, an unopened ofstream tested true and skewed the reference by one
  position. Fixed; reference file was valid.)

### 07:06 B2: B1 + --nvfp4-only (T2 accuracy)  -> ACCEPT as default
- 37.3 / 37.2 tok/s (3 runs each, spread 0.1). CPU tier 106 experts/token, 16.2 ms/token (29.3 GB/s);
  rest (GPU + sync) 10.4 ms/token. 1488 experts resident. Peak device memory 11307 MiB.

### 07:10 B3: static residency from a WikiText routing profile  -> REJECT
- Routing is very skewed per layer (top 64 of 128 experts = 79-97% of selections) but domain dependent:
  chat 30.5 / code 25.7 tok/s (CPU experts/token 164 / 206) vs 37.3 with loop-layers-first.
- In-domain profile (chat/code counts, same prompts): 79.0 / 69.2 tok/s, CPU experts/token 20 / 33: the ceiling
  for adapting residency to the conversation.

### 07:20 B5: adaptive residency (prefill routing seeds, decode counts every 16 tokens, <= 32 async swaps)  -> ACCEPT
- chat 70.6 tok/s (runs 54.9 cold, 75.5, 70.6), code 69.1 (60.4, 69.1, 69.3); 256 tokens, 3 runs.
- CPU tier 22-33 experts/token, 3.5-5 ms/token; GPU + sync now ~9.5 ms/token (the larger share).

### 07:21 B6: 300k context (max_ctx 300032, q4 KV 2.77 GB)  -> baseline for the target config
- 937 experts resident (vs 1488 at 32k). chat 50.6 tok/s (37.3 cold, 50.6, 53.5), code 43.9 (43.0, 43.9, 45.6).
- Prefill 32768-token synthetic prompt: 1178 tok/s. Peak device memory 11429 MiB.
- Prefill is PCIe-bound and serialised: ~15 GB of staged experts per 2048-token chunk (loop layers re-staged on
  each of their 3 passes), uploaded on the compute stream with no overlap.

### 07:24 abliteration (satellitedown ingine runtime ablation, sha256 45686f11...)
- KLD vs R0 0.00729 (vs 0.00719 without), top-1 96.53%, ppl 6.4887: the detector-gated ablation is nearly free
  on harmless text.
- Server check (ingine-serve, 300k, q4 KV, ablation): the "angry rant full of swear words" borderline prompt,
  refused by the original in the ablation author's eval, is answered. 272 tokens in 7.5 s wall (cold pool).

### 07:40 B7: shared KV/expert region (position-major q4 KV grows from the bottom, NVFP4 slots from the top)  -> ACCEPT
- 300k config: 1555 experts resident at short context (vs 937). chat 73.1 tok/s (53.7 cold, 73.1, 75.2),
  code 68.6 (61.0, 68.6, 74.0); prefill 32k 1570 tok/s (vs 1178). Peak 11317 MiB.
- KLD (300k config, 2 x 1024 WikiText): 0.00667, top-1 96.97%, ppl 6.4949 (T2 0.00719 / 96.58% / 6.4862: same
  quantisation, differences are CPU/GPU placement rounding).
- First attempt: cudaErrorMisalignedAddress, slot base not 256-aligned (budget not a multiple of the stride). Fixed.

### 07:50 accuracy attribution (vs R0) and the user's quality bar
- A1 BF16 dense + q4 KV + NVFP4-only: KLD 0.00542, top-1 96.82%, ppl 6.4755.
- A2 FP8 dense + BF16 KV + NVFP4-only: KLD 0.00571, top-1 97.02%, ppl 6.4765.
- So FP8 dense ~0.002, q4 KV ~0.0015, MXFP8->NVFP4 ~0.0016 of the 0.0072 total; none dominates.
- User: quality at or better than Unsloth's 4-bit level (KLD ~0.011-0.017, top-1 96-98% vs their Q8) is enough;
  after that only speed and context. Current default (KLD 0.0073) meets it: quality settled, no lossy speed levers
  needed; next levers are lossless (DFlash2, residency, kernels) and context capacity.

### 07:55-10:53 DFlash2 drafter (incoai) in quill
- Implemented as a DraftModel next to DSpark: 5 FP8-row layers, dynamic two-tap convs, ring of 2048, codebook
  selector, one CUDA graph per draft / commit / round. 631 MiB on the device.
- Bugs found with a NumPy reference (tools/df2_ref.py, engine dump QUILL_DF2_DUMP):
  1. block embeddings must be the target's input embeddings, embed_norm(rms(row)) (SGLang get_input_embeddings);
     raw rows gave garbage drafts (0 accepted).
  2. pageable cudaMemcpy + conversion on the non-blocking engine stream: the FP8 conversion read the tail of each
     fused matrix (V head 7) before its DMA landed. Fixed with a device sync.
  After the fixes: per-layer x within ~2%, 14-16/16 top-16 candidates equal to the reference.
- B8 (300k config, greedy, spec_threshold 0.8): chat 80.1 tok/s (48.2 cold, 80.1, 89.5), code 72.3 (67.9, 72.3,
  81.0); 2.45-2.53 tokens/verify. vs B7 73.1 / 68.6.
- Threshold 0.5: 66.9 / 67.6 (2.84-3.15 tokens/verify); 0.2: 56.0 / 52.8 (3.14-3.56). Wider verifies cost more
  than they gain: the verify rows' expert union lands on the CPU tier. Keep 0.8.
- Also: other apps' device memory rose 990 -> 1450 MiB during the session (Oculus runtime): default reserve 512 MiB.

### 10:55-11:45 VRAM meter, host embedding, prefill overlap, chat app
- Watchdog kills at 11752 / 11679 MiB: the pools were sized with cudaMemGetInfo, which on WDDM reads 50-660 MiB
  below NVML (nvidia-smi / Task Manager) and moves as allocations become resident (standalone test: context
  1232 vs 1889 MiB). Not first-touch (6.7 GB malloc+memset costs exactly 6.7 GB), not kernel stacks (max 224 B),
  not the pageable 4-byte factor copies (pinned now anyway: pool fill 0.69 -> 0.47 s).
  Fix: vram_meter.cpp (NVML via nvml.dll, max with cudaMemGetInfo), settled reading; the region is touched,
  re-measured and shrunk once by any overshoot. Watchdog now logs a timeline (<log>.mem).
- Token embedding (0.93 GB, one row/token) and DFlash2 codebooks (206 MB) -> mapped pinned host memory.
- B12 (no spec, greedy, 300k): 1560 resident experts (= B7's 1555: the freed GB paid for honest NVML sizing and
  other apps 990 -> 1635 MiB). chat 70.4 / code 71.9 tok/s (B7 73.1 / 68.6: noise). Peak nvidia-smi 10775 MiB.
- Prefill: staging double-buffered on a copy stream. Greedy continuation after a 32k prompt identical with
  overlap on/off (lossless), 1855 tok/s both ways (vs 1570 earlier). Probe without uploads: 3105 tok/s -> uploads
  cost ~7 s of 17.7 s but do not overlap on WDDM as issued. OPEN: find why (WDDM batching? one copy engine busy?).
- Temp 1, spec thr 0.8: chat 68.6, code 86.9 (2.35 / 2.79 tok/verify); thr 0.9: 49.0 / 77.8. No-spec temp-1
  control was killed by the old sizing (rerun pending).
- Server: per-request engine timings in usage.timings (prefill/decode/spec/ttft), VRAM + max_model_len in
  /health, --ui serves app/chat.html at /. Launcher: Start Inkling Chat.cmd (now Inkling.exe).
  Smoke test: 54 tok/s, 2.68 tok/verify, TTFT 0.39 s, VRAM 11082 MiB (NVML) idle with spec.

## Decode push (user: decode is the focus; prefill an afterthought). User config = abliterated + DFlash2 + T 1.0
### 12:00 profile (nsys, cuda-graph-trace=node)
- Plain decode, per token: moe_join (GPU waiting on the CPU tier) 4.76 ms | resident experts up+down 4.09 (at
  bandwidth) | dense GEMVs 3.08 (proj 19.8 MB @ 422 GB/s, wo @ 407: at bandwidth) | LM head 1.03 (441 GB/s) |
  attention ~1.1 | glue ~1.3.
- Spec (greedy): moe_join = 51% of GPU time (8.0 of 15.7 ms/token): verify rows -> union of experts -> misses.
- Ceiling: 1.75 GB dense (looped layers re-read 3x) + 1.72 GB experts per token -> ~8.4 ms + attention:
  ~100 tok/s without spec even with every expert resident. The gap is the CPU tier.
- CPU workers spin while active (no wake latency). CPU tier ~26-29 GB/s vs 37-38 GB/s CPU read ceiling.
### REJECTED: GPU zero-copy assist for misses
- Old note (CPU 36 + GPU 17.5 concurrently) was a measurement bug (GPU active 0.25 s of the CPU's 3 s window).
  Fully overlapped: CPU 22.0 + GPU 18.4 = 40.4 GB/s vs CPU alone 38.3: DDR4 is the shared ceiling. +5% max.
- CPU read BW by threads: 4 threads 37.8, 7: 36-37.6, 14 (SMT): 34.7-35 GB/s. Pinning irrelevant.
### ACCEPT: prefill scratch lent from expert slots (lossless)
- VRAM audit: weights 1212 MiB, drafter 434, prompt checkpoint 205, state 166, decode bufs 50, taps 36, prefill
  scratch 738 (idle during decode), CUDA context/modules ~640.
- Prefiller lays out its scratch in one block; Runtime::lend_block puts it in the expert slots just below the
  staging slots (last ones the KV could reach; checked against max_ctx). scratch_acquire() evicts those experts
  and locks the slots for a prefill (memset zero), scratch_release() uploads them back (~30 ms).
- Greedy 32k continuation identical. Resident 1560 -> 1722 (no spec, no ablation). Warm CPU misses -20%.
  Prefill 32k 1760 tok/s (1855 before: re-upload + memset + noise).
- E1 (chunk 1024 instead, user config): resident 1404 -> 1480, misses -10%; tok/s within T=1 noise (76-100).
### Lossy candidate: skip non-resident experts below a routing-weight fraction (--skip-cold, off by default)
- KLD (WikiText 2x1024 vs BF16, ablation on): base 0.00757 / 96.73% | 0.02: 0.00730 / 96.24% |
  0.04: 0.00805 / 95.80% | 0.06: 0.01349 / 94.97% (out). User bar: no worse than UD-Q4_K_XL (KLD 0.0107,
  98.0% on Unsloth's set; our harness scores ~96.7-97% top-1 even near-lossless) -> KLD <= 0.0107 and <= ~1 pt
  top-1 loss vs our baseline.
- Speed (greedy no spec, 512 tok, same resident 1728): base 73.7/76.6, skip .02 74.4/77.6, skip .04 75.2/79.3
  (misses 26.8 -> 23.8). REJECT: ~3% for -0.9 pt top-1. Flag stays, off.
- Natural experiment: resident 1637 -> 1728 (+91) gave misses -13% and decode +8%: ~+2% decode per 100 MiB.
### ACCEPT: prompt checkpoint (205 MiB) in pinned host memory (copies via cudaMemcpyDefault)
- Resident 1728 -> 1766, misses 26.8 -> 24.1, greedy 73.7/76.6 -> 75.2/78.8.
- Restore exactness: with --no-adapt every repeat greedy run matches run 0 for 256/256 tokens (spec on).
  (With adaptive residency repeats diverge at ~token 68-73: CPU-vs-GPU rounding of a near-tie.)
- Static residency costs 2x: 112.7 misses/token, 49.5/53.5 tok/s spec greedy.
### REJECT: 14 CPU-tier threads (SMT siblings): 75.5/79.2 vs 75.2/78.8 (memory-bound, as the BW test said).
- Greedy 256-token runs now: no spec 87.4/78.2; spec 103.8/94.4 (2.42/2.30 t/v) vs B8 80.1/72.3.
### 13:00-14:10 (user config T=1 from now on; user: only ever temperature 1)
- Threshold sweep T=1 (512 tok x 4 warm): 0.6 104.0/103.3, 0.7 96.0/106.5, 0.8 103.7/98.8, 0.9 106.9/104.9: all
  within noise. Keep 0.8. User config jumped from c1/c2 80-85/90-92 to ~100-107 (lending + host checkpoint).
- Verify widths (greedy spec): 2,4,8 109.3/103.3 & 108.6/107.2; 2,3,4,5,6,8 107.5/106.4; 2,3,4,6,8 105.8/104.7;
  3,5,8 106.7/102.7. REJECT finer widths (noise, +75 MiB). --verify-widths option kept.
- Replay set: 12 everyday prompts x 768 tokens recorded at T=1 (replay_t1.jsonl, --record). Free sampled replay
  (seed 7): 80.7 tok/s, 2.34 tok/verify (content range 70-110).
- Drafter top-16 rewritten (exact two-stage, warp argmax; was 466 us/round, spilled insertion lists): replay
  80.7 tok/s, 2.32 t/v: correct, gain inside noise. KEEP.
- Spec profile now: moe_join 47% (4.8 ms/token), experts 2.3, dense+heads ~1.8, attention 0.3.
- NV UEFI Unlocker (user question): lifts GeForce tensor-rate caps (dev claims ~2.6x FP16/BF16, ~1.9x FP8).
  Decode is memory-bound -> no gain expected; prefill GEMMs run at the capped 71.5 TFLOPS -> ~+30-50% prefill.
- RAM: two mixed kits (DDR4-3333 + DDR4-3200) running at 3000. A matched DDR4-3600 kit ~+20% CPU-tier BW.
- NVFP4 dense (fake-quant round trip at load, --dense-bf16, ablation on): BF16 baseline KLD 0.00607 / 96.63%;
  wproj in NVFP4: 0.01096 / 95.31% -> already over the 0.0107 bar. REJECT plain NVFP4 for wproj. wo / unembed
  runs pending (stopped: user wanted the GPU).

## Overnight (goal: KLD <= UD-Q4_K_M 0.0108 on our harness; decode first)
### VRAM phase accounting (NVML vs DeviceBuffer bytes, other apps 1064 MiB at night)
- CUDA context +467 MiB hidden (bare context ~250), weights +70 hidden, cublasLt +33, drafter +50, graphs +38.
- ACCEPT: cudaLimitStackSize 256 B (max kernel stack 224 B): context 467 -> 405 MiB, +14 resident experts.
### REJECT: NVFP4 dense (fake-quant, BF16-dense baseline 0.00607): wo 0.01127, unembed 0.01023, both 0.0160.
  On top of the FP8 config (0.0076) the head alone would be ~0.0118 > 0.0108.
### CPU tier is COMPUTE-bound (microbench tierbench.cpp, real decode/dot code)
- fp32 path: 5.5 GB/s per core even from L2; 7 cores 38.6 (L2/L3) vs 36.0 (DRAM): prefetch alone can't help.
- Real tier while busy: 22.7-26.3 GB/s (per-call overheads on top).
### ACCEPT: int8 CPU-tier dots (x and activations quantized per 32 to int8; E2M1x2 -> int8 by pshufb;
  maddubs/madd; split >4 rows): 10 GB/s per core, 69 GB/s 7 cores from L3, DRAM-bound with 4 threads.
- KLD 0.00741 / 96.48% vs fp32 tier 0.00726 / 96.38% (noise).
- Replay T=1 (12 prompts, seed 7, 1735 resident): int8 96.1 tok/s (2.26 t/v, 23.5 ms/round) vs fp32 88.5
  (2.35 t/v, 26.5 ms/round). +8.6% tok/s, -11% round time. (Afternoon baseline 80.7 at ~1650 resident.)
### Replay A/B is deterministic for numerics-neutral changes (same text, same tok/verify): controls 96.0 / 95.9 / 96.0.
### REJECT (replay, user config): CPU threads 4: 82.0, 5: 87.9, 8: 91.8 (SMT sibling shares worker 0's core:
  barrier imbalance), 10: 95.7, 14: 95.8 vs 7: 96.0. Residency: swaps 64 93.9; interval 8 96.0; decay 0.90 97.5 but
  per round slower (text changed); margin 1.10 95.9. Draft temperature x0.8 97.0 / x1.25 95.5 (text noise, per
  round not faster).
### REJECT: L3 prefetch of predicted misses: trace of 30.7k CPU-tier requests: only 5.4% of accesses within 81 MB
  LRU distance; best predictor (same layer, previous 2 rounds) recall 41.8% at precision 24.5%: est. +3-5%.
### Dense format sweep (fake-quant round trip, all dense, --dense-bf16; BF16 base 0.00607):
  FP6 E2M3 blk32 0.00734 | E2M3 blk16 0.00739 | E3M2 blk32 0.01153 | FP5 E2M2 blk16 0.00977 | E2M2 blk32 0.0136.
### ACCEPT: FP6 E2M3 dense rows (6.25 bits: [K/2 nibbles | K/4 e1+sign words] + E4M3 scale per 32 + FP32 row
  scale x64). Decode: s<<7 | (c & 0x1f) read as E4M3 = value x 2^-6 (hardware E4M3->F16 cvt); bit-spread multiplies
  for e1/sign. M=1 kernel 16 lanes x 16-value chunks, tensor-core kernel 32 cols/lane/step. Real-kernel KLD 0.00739
  (FP8 rows 0.00741). Replay: 104.3 tok/s (2.34 t/v, 22.4 ms/round) vs 96.0 (2.26, 23.5 ms/round), resident
  1735 -> 1789. Default on with dense_fp8 (QUILL_DENSE_FP6=0 for FP8 rows).
### CPU tier phase timing (FP6 base, per call, 2.86 unique experts): input+quant 5 us, w13 267 us (32 GB/s),
  actq 1.5 us, w2 138 us (31 GB/s). The tier is purely DRAM-bound at ~85% of the synthetic 36-38 GB/s.
### REJECT: draft head over the first N vocab ids: 100k -> 1.63 t/v (from 2.34), 64k 1.58: ids are not frequency order.
### ACCEPT: FP6 drafter matrices (fc/o/down need SPLITK 18 for 128-col steps). 3 seeds (7/11/23): base 104.7/101.3/102.2
  (mean 102.7, t/v 2.32) vs FP6 104.5/103.3/102.4 (mean 103.4, t/v 2.30); +19 resident experts.
### VRAM value (cap sweep, FP6 drafter, 2 seeds): 11.3 -> 11.1 -> 10.9 GiB: resident 1808/1761/1713, time per
  round s7 21.6/22.7/23.8 ms, s11 22.6/23.1/24.6 ms: ~2.5% per 100 MiB.
### ACCEPT: dynamic VRAM (VmmRegion: CUDA VMM, 32 MiB chunks). Startup reserve 512 -> 160 MiB; vram_check every
  residency step: used + 48 MiB > cap -> release chunks just above the KV front (evict_below, sync, unmap);
  well under target -> take back up to 8 chunks per check (2 s spacing). KV growth maps hole chunks back first.
- Replay 3 seeds: 107.3 / 104.9 / 108.6 (mean 106.9, t/v 2.32/2.31/2.38) vs 103.4 mean before: +3.4%.
- Stress (vramhog 600 MiB mid-run): released 480 MiB within ~0.2 s (peak 11591 MiB by nvidia-smi), took it all
  back within ~2 s after the hog freed it; generation continued (102.8 tok/s). First version recursed
  (vram_shrink -> residency_flush -> residency_step -> vram_check): stack overflow; re-entrancy guard added.
### Concurrency microbench for GPU assist: int8 tier 7 thr 22.8 GB/s + GPU zero-copy 17.9 GB/s = 40.7 GB/s
  (tier alone in the engine 31.5): est. -13..18% CPU-tier time if a share of each call's misses runs on the GPU.
### ACCEPT: GPU zero-copy assist: dispatch flags the first round(0.34 U) of a call's U unique misses (U >= 3, >= 1
  left to the CPU); moe_up/moe_down read those experts from the host arrays (device-mapped pinned memory).
  Seed 7 (old build): ctrl 107.9 (22.1 ms/round) | 34%/min2 108.3 | 50%/min2 100.4 (PCIe-bound, worse) |
  34%/min3 112.0 (21.6 ms/round). New build: s11 110.1 -> 112.1, s23 107.5 -> 111.3; 25%/min3 110.6. ~+3%.
### ACCEPT: step-graph prefill for new tails <= 64 tokens (no images): set_inputs + run_step(8/4/2/1), drafter
  commit from the step taps, last row's logits. Greedy outputs identical to the Prefiller; cold TTFT 28-token
  prompt 0.39 -> 0.24 s. Decode after short prompts also faster (no 512-swap storm / scratch re-upload at the
  start of decode): new-build controls s11 104.9 -> 110.1.
### ACCEPT: CPU tier dynamic work split (atomic 16-row chunks in both weight phases): identical text,
  112.5 vs 110.3 tok/s static (+2.0%).
### KLD of the full config (FP6 + int8 tier + assist 34%/min3 + dynamic VRAM): 0.00817 (deterministic, 3 repeats).
  Assist sweep KLD: none 0.00738 | 34%/min1-2 0.00777 | 34%/min3 0.00817 | 50%/min3 0.00738 | 90%/min1 0.00720:
  non-monotonic = sensitivity to which (token, expert) products take the int8 CPU path, not an assist bug (90%
  assist is the best). All far under the 0.0108 bar; 34%/min3 kept for speed (50%/min3: 103.7/101.9 vs 112.6/114.3).
### REJECT: verify threshold 0.65 (110.8/107.8) and 0.9 (114.6/110.1) vs 0.8 (112.6/114.3).
### REJECT: draft temperature x0.3 (110.4/108.5, t/v 2.26/2.27), x0.05 (106.4/107.0, t/v 2.12/2.16).
### REJECT: prefill swaps 512/128/0 after a Prefiller prompt: 114.2/114.2/114.3 (identical text): no effect.
### Step-prefill crossover: 96 tok: steps 147 tok/s vs Prefiller 271; 192: 142 vs 461 -> threshold 48 tokens.
### CUDA_MODULE_LOADING: default = LAZY (identical); EAGER +130 MiB. Context ~405 MiB is intrinsic.
### Profile (nsys): inter-kernel gaps negligible (0.18 us mean); kernels inside graph conditional bodies are not
  traced (the verify forward shows as a ~24 ms gap), so per-token kernel totals from such profiles are partial.
### REJECT: sliding-window rings in IQ4_NL q4 (+121 MiB): KLD 0.01135 vs 0.00817 (over the bar). Code kept, off.
### Robustness: 64k-token prompt (KV through the dynamic region's hole): prefill 2022 tok/s, decode continues,
  peak 11153 MiB. Server smoke test (launcher flags): 118.8 / 136.9 tok/s decode, TTFT 0.29 / 0.18 s, 11.4 GB.
### REJECT (again, current build): skip-cold 0.02 / 0.03: 115.1/113.4, 115.0/115.2 vs 112.6/114.3: +0.7..1.5%, noise level.
### ACCEPT: cuBLASLt workspace (32 MiB) + drafter prefill taps (36 MiB) inside the lent prefill scratch (806 MiB
  lent; resident 1908 -> 1916). 32k prefill 2175 tok/s.
### Long context (replay_long.jsonl: 22.3k-token code prompt, 768 output tokens): 84.7 tok/s, 2.21 t/v, 26.1 ms/
  round vs ~20.5 ms short. Not residency (prefill swaps 0/128: 84.8 both). CPU tier per token lower (4.1 vs 5.1 ms),
  GPU side +2.9 ms/token. Plain-decode profile at 22k: full attn89_split 137.9 us/visit (25 MB KV at ~181 GB/s).
### ACCEPT: attention split grid (M, splits, groups) instead of (splits, groups, M): the M verify rows of a split
  read the same KV chunk back to back (L2) instead of M separate passes with evict-first loads. Same text:
  long 84.8 -> 88.4 tok/s (+4.2%), short 113.2 -> 113.4.
### REJECT: attention 128 splits x 128-key chunks: short 117.2/113.0/113.8 vs 113.2/114.1/113.4; long (22k) 92.5/86.1
  vs 88.2/89.1 (per-round time tracks t/v): neutral, reverted (saves ~10 MB workspace).
### ACCEPT: attention grid (rows, head groups, splits): same text, long 88.2 -> 88.9, 89.1 -> 89.6 (+0.7%).
### REJECT: prompt-lookup n-gram drafting: offline on the 13 recorded outputs it could supply 9.1% of tokens
  (1-22% per item, DFlash2 already covers much of it).
### ACCEPT: the 24 staging slots join the lent range (resident experts during decode, emptied for a Prefiller
  prompt). s7 113.2 -> 115.5 (20.85 -> 20.35 ms/round), s11 114.1 -> 115.3 (21.03 -> 20.56). 32k prefill unchanged
  (2175 tok/s, identical greedy continuation).
### REJECT: CPU threads 8 / 10 with work stealing: 115.4 / 115.2 vs 115.5 (same text): DRAM-bound.
### ACCEPT: sliding rings in int8 (fp16 scale per 32; 1088 B rows): KLD 0.00778 (BF16 0.00817), resident 1916 ->
  1931, round time s7 -1.7%, s11 +-0. 32k prefill greedy continuation identical, 2192 tok/s; long replay 88.3.
### ACCEPT: prefill staging factors by one kernel per group (no 2 x 4-byte H2D copies per staged expert): 32k 2192 -> 2211 tok/s, identical greedy continuation.
### Tier timeline (QUILL_TIER_TRACE with timestamps, 4 replay items): CPU tier busy only 49.5% of the active time.
  Calls 381 us mean (232 median, U 2.39, M 4.17); gaps 414 us mean (206 median; ~120 us at layers 1-5/19-23 where
  every visit calls, 300-420 us in the loop where many visits are fully resident).
### REJECT: pre-gating L3 prefetch (QUILL_PREFETCH, code kept, off): after each call the workers apply visit v+1's
  router to the call's inputs (top-8 per token, non-resident), then load those experts while the GPU runs the next
  attention. Hint quality: 66% of calls hinted, recall 70%, precision 48% -> 54% of CPU expert reads covered.
  _mm_prefetch is dropped on Zen 3 (165 "GB/s"); real loads reach ~42 GB/s, ~4 MB per gap. Hinted calls get faster
  (U=2 both hinted 278 -> 248 us, U=3 507 -> 460) but gaps grow 11-22 us (DRAM contention with the GPU assist's
  PCIe reads): replay 119.0 -> 119.3..119.8 (+0.3-0.6%, noise). Ranked/capped variants the same.
### Draft head vocabulary: the BPE ids are not frequency ordered (replay outputs: 50% of tokens have id < 100k), so the
  earlier prefix test was meaningless. FP6 head rows now permuted at load (specials, the draft_shortlist.bin corpus
  ranking, rest by id); target logits scatter back through head_tok (KLD identical 0.00778); DFlash2 reads the
  first QUILL_DRAFT_VOCAB rows and maps candidates to ids. Shortlist coverage of replay outputs: 32k 97.0%, 49k 97.9%.
### Draft vocab default 41,456 rows (specials + all corpus-ranked tokens). 3 seeds: 118.9 / 118.7 / 121.1 (2.24/2.30/2.44 t/v).
### REJECT (code kept, QUILL_DRAFT_FP4=1): FP4 E2M1 drafter (fp4.cuh, E4M3 per 16): drafter 353 -> 271 MiB, resident +22.
  Sampled 3 seeds 117.6/119.5/119.8 vs 118.9/118.7/121.1; teacher-forced (identical text) 109.2 vs 108.2 (+0.9%,
  t/v 2.08 vs 2.09). Marginal residency is worth much less now (~1931 of 3072 resident).
### NOTE: a background A/B (FP4 seeds 23/31) was killed by Claude Code for low system memory while idle; s23 had
  finished. Runs now go in the foreground, one per call. Teacher-forced replay (rp.ps1 -Seed -1) = identical text.
### Step profile (--profile-steps, nsys per width; repeated-token inputs so the MoE part is not representative):
  per visit at M=4 outside the experts ~126 us: proj 39 (at bandwidth), wo 16, attn split 24 (latency: 2.2 MB ring),
  gates89 14, router 7, prep 6, dispatch 6, combines 6, mix ~6. ncu has no counter permission (non-admin).
### Attention: %globaltimer phase timers per block (QUILL_ATTN_DBG): staging 2.7-5 us, key loop 2.2-2.6 us (8 iter,
  latency chain ~0.25 us each, not throughput: mantissa-trick int8 + lazy rescale did not move it), merge 1 us; kernel
  time = waves x block time (2 blocks/SM by registers). launch_bounds(256,3) spills: 2-3x slower (reverted).
  ACCEPT: 128-key splits staged by cp.async + all rows per block (attn89_slide8, 64 blocks = 1 wave): M=8 36 -> ~27 us,
  M=2/4 ~even; teacher-forced 108.4 (2.08) vs 108.2. KLD 0.00804 (split order).
### Real decode timeline (QUILL_JOIN_DBG, s7): 41% of visits call the CPU tier; per call doorbell -> join 309 us (GPU
  experts incl. PCIe assist), GPU waits for the CPU 48 us mean (45% of calls), 4.0 s of 78 s. The GPU side, not the
  CPU tier, was the long pole -> assist re-swept (teacher-forced, identical text): 0: 109.2, 0.2: 108.8, 0.34: 108.4.
  ACCEPT assist off (KLD 0.00755). With assist off the GPU waits 209 us per call -> idle VRAM time.
### ACCEPT: L2 prefetch of the next visit's proj/wo FP6 rows, router and q8 ring on a side stream forked after moe_down,
  gated on a pending CPU call (one ack read per block; reading it from every thread floods PCIe: 98.4 tok/s).
  CPU-idle gap before the next visit 163 -> 145 us mean. Teacher-forced 109.2 -> 109.7. Sampled 3 seeds (assist off +
  prefetch) 122.4 / 119.4 / 118.9 vs 118.9 / 118.7 / 121.1.
### REJECT: verify-width policies (offline on the teacher-forced trace with per-round draft confidences): threshold
  0.9 +1.3%, calibrated expected-value policy +0.8% (cross-validated), oracle +9% (unreachable from confidence);
  width 1 for low first-draft confidence: +0.7-2.3% at the measured ~11.5-12 ms step cost (spec loop needs widths >= 2).
### REJECT: CPU pre-gating prefetch again with assist off: 109.9 vs 109.7 (noise).
### Long context (22k, profile steps): full attn89_split per visit 108 / 173 / 315 / 604 us at M = 1/2/4/8 (each row
  re-reads ~25 MB). ACCEPT attn89_full8 (all rows per block, 128-key tiles staged by cp.async, IQ4_NL LUT in smem):
  90 / 139 / 232 / 423 us; teacher-forced long replay 82.5 -> 84.2 tok/s (verify 20.5 -> 20.1 ms). 32k greedy
  continuation identical to the per-row kernel. (Bench long prompt = its own main.cpp: edited tonight, so the old
  continuation tokens no longer apply.) Sampled single-item long replay is too noisy (88.7 vs 93.1, other text).
### ACCEPT: tensor-core full attention (attn89_full_mma; q4 tiles -> f16 per warp via a byte->f16x2 table x scale,
  S = Q K^T and O += P V with mma.m16n8k16, online softmax on the fragments, P reused as A; rows 3M in 1-2 m16 tiles).
  22k per visit: M=1 85.7 / M=2 86.6 / M=4 98.1 / M=8 125.4 us (all-rows 90/139/232/423, per-row 108/173/315/604).
  Long teacher-forced 89.3 tok/s (per-row 82.5, all-rows 84.2). KLD 0.00720, 32k greedy continuation identical. Short
  teacher-forced 109.3 (2.07 t/v) vs 109.7 (2.09): per round 18.94 vs 19.05 ms.
### ACCEPT: tensor-core sliding attention (attn89_slide_mma, q8 ring; int8 -> f16 1024+u by PRMT, (h-1152)*s by HFMA2): per visit M=1/2/4/8 12.1/12.8/14.7/16.5 us (all-rows 12.3/15.3/22.3/36.8). Teacher-forced 109.7 (2.08) ~ same per round (width-2 rounds dominate). KLD 0.00706.
### Real-decode nsys (conditional bodies traced now): with assist off moe_join = 38% of decode (GPU waits for the CPU
  tier). Tier efficiency falls with call size (U=1 36.7 GB/s, U=2 33.3, U=3 26.4, U=4 25.3: multi-token experts).
### REJECT: assist by token count (QUILL_ASSIST_MINTOK 3/2: 109.1/109.2 vs 109.7): a PCIe expert costs the GPU ~250 us,
  no faster than the CPU, and the same host DRAM bytes. CPU threads 14: 107.2.
### ACCEPT (re-test, CPU-bound now): FP4 drafter default: teacher-forced 111.2 (2.07) vs 109.7 (2.08); sampled s7 120.6 ->
  123.2, s11 118.2 -> 119.8. Resident 1931 -> 1953.
### Split head (QUILL_HEAD_TAIL4, default on in the current build, under review): FP6 prefix 41,456 rows + FP4 tail.
  KLD 0.00706 -> 0.00792; resident +15; teacher-forced 111.3 (2.06) vs 111.2 (2.07): rounds -0.6%; sampled s7 121.2 vs
  123.2 (resident 1953 both: other apps took the freed VRAM). ~0-0.7 tok/s: recommended off (user to decide).

## STATUS (2026-11-25 morning, user awake and chatting: GPU work paused)
- Replay (12 prompts x 768 tok, T=1, sampled): afternoon 80.7 -> bedtime ~115 -> now ~120-123 (s7 123.2, s11 119.8).
- Long context 22k (teacher-forced): 82.5 -> 89.3 tok/s. Prefill 32k: 2175 -> 2217 tok/s. KLD 0.00792 (bar 0.0108).
- Bottleneck: the CPU tier reading non-resident experts from DDR4 (GPU waits ~38% of decode in moe_join); host DRAM
  ~36-40 GB/s is the ceiling for misses. Levers left: faster RAM (user hardware), more VRAM for experts (FP5 attention:
  KLD 0.00933 sim, ~2-3%), CPU tier multi-token efficiency, server smoke test on this build (not done).
- A/B method that works: teacher-forced replay (rp.ps1 -Seed -1) = identical text, +-0.2 tok/s; sampled seeds vary
  +-1.5% from text alone. Real-decode nsys traces conditional bodies (cuda-graph-trace=node).
