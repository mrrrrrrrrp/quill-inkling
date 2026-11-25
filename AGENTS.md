# AGENTS.md: setting up Inkling + quill on hardware other than the reference

This guide is for a coding agent (Claude Code, Codex, Cursor, …) helping someone run this project on their own
machine. The engine was built and tuned on one machine:

> **Reference:** RTX 4070 SUPER 12 GB (sm_89) · Ryzen 7 5800X3D (8 cores, AVX2) · 32 GB DDR4-3000 · Windows 11
> **Result:** 100+ tok/s decode (≈120 median, temperature 1, DFlash2), 600k context, KLD 0.0071 vs BF16

On RTX 40/50 cards with 32 GB of RAM, the installer (`Inkling-Setup.exe`) already scales the settings. Use this file
when the machine differs more than that, or when the user wants it tuned. Read sections 0–2 before changing anything.

**Use `inkling-cli.exe` from a terminal.** It is attached to every release and placed in the install folder. It is
the same program as `Inkling.exe` / `Inkling-Setup.exe`, built as a console program, so PowerShell and cmd wait for
it and capture its output. They do neither for the windowed `Inkling.exe`.

---

## 0. Ground rules

These are the project's own requirements. Keep them unless the user explicitly says otherwise.

1. **`QUILL_PASSES` stays 3.** Inkling-2-Mini is trained for exactly 3 passes through its loop layers. At 5 it
   produces coherent text that dissolves into word salad within a few hundred tokens.
   - The claim online that it is "trained up to 5, works up to 10" is about a different checkpoint,
     `thinkingmachines/TML-d24-varloop-research`. That checkpoint is not wired up here, and the drafter and
     abliteration were built for Inkling-2-Mini.
2. **The quality bar is a KLD of at most 0.0107 against the BF16 model** on the WikiText harness (section 6), and
   at most ~1 point of top-1 loss.
   - The current default scores 0.0071.
   - Do not trade quality for a percent of speed. Lossy switches stay opt-in (`--skip-cold`, `QUILL_HEAD_TAIL4`).
3. **Measure the way it is used: temperature 1 with speculative decoding.**
   - Sampled runs at T=1 are noisy (single runs spread ~±15%).
   - For A/B tests use the teacher-forced replay (`rp.ps1 -Seed -1`, identical text in every configuration,
     ±0.2 tok/s), or the median of several seeds.
   - Greedy decoding is only for diagnostics.
4. **VRAM is counted for all processes**, as Task Manager and NVML/`nvidia-smi` show it.
   - `cudaMemGetInfo` reads up to ~650 MiB low under Windows WDDM. Do not size anything with it.
5. **Check what was already tried.** `engine/HISTORY.txt` holds the engine's commit messages, with the measured
   numbers of every kept change. `bench/rp_results.txt` holds the reference machine's benchmark history.
6. **Don't benchmark while the user is chatting with the model.** A benchmark loads the model on the same GPU.
   Ask first.

---

## 1. Find out what you have

```powershell
inkling-cli detect --json             # from a checkout: cargo run --release --manifest-path installer\Cargo.toml -- detect --json
Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity, ConfiguredClockSpeed, PartNumber   # RAM speed
nvidia-smi -q -d PCIE | Select-String "Generation|Width"                                           # PCIe link
```

The JSON includes:
- `hardware`: the GPUs (VRAM, compute capability), the CUDA version the driver supports, RAM, CPU cores, AVX2
- `assessment`: `support` (Confirmed / Expected / Experimental / Unsupported) with the reasons in `notes`, the
  recommended `settings`, `max_ctx_fit` (the estimated context ceiling), whether the prebuilt engine covers the GPU
  (`prebuilt`), and the `cuda_arch` for a source build
- `toolchain`: what a source build is missing, each item with its winget command

Map the machine onto the sections in §3.

---

## 2. How the engine uses the machine

Reason from this before changing settings.

**The model.** Inkling 2 Mini is a 26.4B-parameter MoE:
- 24 layers, hidden size 2304, 24 attention heads / 8 KV heads × 128.
- 128 experts per layer, top-8, so 3,072 experts in all.
- Layers 6–17 are a loop that runs 3 times, giving 48 layer visits per token.
- Full attention at layers 5, 11, 17 and 23, which is 8 full-attention visits at 3 passes. Every other layer uses
  a 1,024-token sliding window.
- The tokenizer has 201k entries.

**Formats at run time:**

| Part | Format |
|---|---|
| Dense weights | FP6 E2M3 with E4M3 scales per 32 (`QUILL_DENSE_FP6`) |
| Experts | NVFP4. Layers 20–23 ship as MXFP8 and are re-encoded at load, which is part of the ~1 minute start. |
| Output head | FP6, with rows in token-frequency order so the drafter reads only the first 41,456 |
| Full-attention KV | q4 (IQ4_NL), 9 KiB per token at 3 passes |
| Sliding-window KV | int8 |
| Drafter (DFlash2) | FP4 |

**System RAM: why 32 GB is needed.** About 14 GiB is pinned (`cudaHostRegister`):
- All 3,072 experts, ~12.8 GiB, loaded into one host region and pinned (`src/model/experts.cpp`)
- The embedding table, 0.93 GB, mapped pinned memory
- The drafter's codebooks (206 MB) and a prompt checkpoint (205 MB)
- With `vision = "cpu"`, the image tower also runs from RAM.

If pinning fails, the log says `NOT pinned` and transfers slow down.

**VRAM:**
- ~2.1 GiB is fixed: dense weights, the drafter, the CUDA context and buffers.
- Everything else under `vram_cap_gib`, minus what other programs use, becomes one **shared region**:
  - The q4 KV cache grows from the bottom as a chat gets longer.
  - Expert slots (~4.3 MiB each) fill from the top.
  - 24 staging slots, and an 806 MiB prefill scratch that is lent to experts between prompts.
- Dynamic VRAM (CUDA VMM, 32 MiB chunks) gives chunks back when another program needs memory, and takes them back
  later.

On the reference machine:
- Cap 11.3 GiB → a ~8.2 GiB region → ~1,800 of the 3,072 experts on the GPU at short context.
- Context ceiling ≈ (region − 0.95 GiB) / 9 KiB ≈ 850k tokens. `max_ctx` itself costs nothing until a chat fills
  it.
- If the region cannot hold the KV cache for `max_ctx` plus the staging slots, the engine refuses to start: *"the
  shared region cannot hold the KV cache of max_ctx and the staging slots"*.

**Decode:**
- Each token visits 48 layers × 8 experts. Experts on the GPU run there.
- Misses run on the **CPU tier**: `cpu_threads` workers with int8 AVX2 dot products, reading NVFP4 from pinned RAM.
- Adaptive residency counts routing every 16 tokens and swaps hot experts in asynchronously.
- On the reference machine the GPU waits on the CPU tier for ~38% of the decode time.

**So the speed levers, in order:**
1. More experts on the GPU (more VRAM)
2. Faster RAM (DDR4 gives ~32–40 GB/s for misses)
3. More physical cores (the tier is limited by per-core compute as well as RAM bandwidth)
4. GPU-side kernel speed, last: it mostly turns into longer waits for the CPU

**Speculative decoding.** DFlash2 (`incoai/Inkling-2-Mini-DFlash2`) drafts up to 7 tokens. The model verifies the
leading drafts whose probability is above 0.8, at widths 2/4/8, accepting ~2.1–2.3 tokens per verification at
T=1. It is a large share of the speed: in a mid-project greedy test it took decode from 87 to 104 tok/s.
Measure before turning it off.

---

## 3. Per-setup guidance

### RTX 40, 12 GB, 32 GB RAM (the reference)
Install and you're done. Nothing to tune.

### RTX 40/50 with 16 GB (4060 Ti 16 GB, 4070 Ti SUPER, 4080, 5060 Ti 16 GB, 5070 Ti, 5080)
- **What the installer sets:**
  - `vram_cap_gib` = card size − 0.7 (15.3 on a 4080 SUPER), which leaves room for ~900 more experts on the GPU.
  - A tester's RTX 4080 SUPER reached ~200 tok/s decode with a raised VRAM cap. Use that as the target for 16 GB
    Ada cards.
  - `vision = "gpu"`, and 1M context is offered.
- **Check the start-up log** for `experts: shared KV/expert region … N resident`. N should be well above 1,800.
- **Then A/B, one at a time** (§6):
  - `QUILL_ASSIST`: the share of CPU misses that the GPU reads over PCIe instead. Default 0, because on PCIe 4 it
    stopped helping once the int8 tier landed. Try 0.2–0.4 on PCIe 5.
  - `cpu_threads`
  - `extra_args = ["--verify-widths", "2,4,8"]` variants
  - `--spec-threshold` 0.7–0.9
- **Laptop cards** have the same VRAM sizes but lower clocks and power limits, so expect less.

### RTX 4090 / 5090 (24 / 32 GB): every expert may fit
- **Every expert fits.** 3,072 × ~4.3 MiB ≈ 12.8 GiB, and a 23.3 GiB cap gives a ~20 GiB region. The pool builder
  stops placing experts when there are none left, so the extra slots stay empty.
  - **This path is untested.** Check the log, and check that adaptive residency doesn't misbehave with empty slots
    (`src/model/residency.cpp`). If it does, run with `extra_args = ["--no-adapt"]`.
- **The CPU tier then sits idle.** Decode becomes GPU-bound, so GPU kernel work starts paying off:
  - `src/kernels/attn/attn89.cu`, `src/kernels/moe/moe.cu`, `src/kernels/dense/gemv.cu`
  - The verify widths and the draft threshold
  - Long-context attention, now the largest cost (`bench/replay_long.jsonl`)
- **RAM win (needs a code change).** When every expert is resident, the 12.8 GiB pinned host copy is only an
  upload source.
  - Release or unpin it after `build_pools_shared` uploads everything (`src/model/experts.cpp`).
  - That cuts the pinned RAM from ~14 GiB to ~1.5 GiB and makes 16 GB machines work on these cards.
  - Keep it behind a check that every expert really is resident.

### RTX 50 (Blackwell, sm_120)
- **The prebuilt engine includes `sm_120a` code** (`release.yml` builds `89;120a`). It compiles, but has not been
  confirmed on an RTX 50.
  - First run: check the log and a chat request, then the KLD check (§6), then the speed (§6).
- **Where the Blackwell headroom is.** The kernels came from upstream ingine, which targeted RTX 5090 (`sm_120a`).
  quill replaced Blackwell-only instructions with sm_89 versions. Search for `quill (sm_89)` and `sm_89 has no` in
  `src/`:
  - **FP4 conversion.** `src/kernels/moe/moe.cu` builds E2M1 → f16 by placing bits, where Blackwell has
    `cvt.rn.f16x2.e2m1x2`. Also check `dense/fp4.cuh` and `draft/draft_device.cuh`.
  - **Programmatic dependent launch** is disabled (`numAttrs = 0`) in `src/kernels/common/pdl.cuh`,
    `src/core/pdl.cuh` and `moe.cu`.
  - **Thread-block clusters.** `src/kernels/dense/sm89.cu` replaces two cluster kernels; see
    `dense/cluster.cuh`.

  Restore each one behind `#if __CUDA_ARCH__ >= 1000` (some, like `dense/gemv.cu`, already are), then A/B it and
  check the KLD.
- **Re-tune the platform knobs.** Blackwell machines usually have PCIe 5 and DDR5: re-tune `QUILL_ASSIST` and
  `cpu_threads`.

### RTX 30 or older (compute ≤ 8.6): not supported as is
The kernels use FP8 E4M3 conversion instructions that need compute 8.9:
- in the scales of FP6, NVFP4 and FP4
- in the drafter
- in prefill

These appear in 14 files: `src/kernels/{attn/attn.cu, dense/{dense.cuh,fp4.cuh,fp6.cuh,gemv.cu},
draft/{attention.cu,draft_device.cuh,fold.cu}, moe/{dequant.cu,moe.cu},
prefill/{attention.cu,common.cuh,rows.cu,wgemm.cu}}`.

A port would replace every E4M3 → f16/bf16/f32 conversion with a software decode (a 256-entry table in shared
memory, or bit manipulation), build with `--arch 86`, and pass the KLD bar. It is real work; tell the user before
starting. An RTX 3090's 24 GB would hold every expert, so the CPU-tier caveats above also apply.

### 8 GB cards (4060, 4060 Ti 8 GB, laptops)
- **Settings:** the installer sets a ~7.2 GiB cap and a 262k context. Few experts stay on the GPU, so decode
  leans on the CPU tier and is much slower.
- **Lower context frees room for experts:** use 131072 or 65536 when long chats aren't needed.
- **Keep `vision = "cpu"`.**
- **Measure before promising anything.**

### Less than 32 GB of RAM
- **24 GB:** it may load with nothing else running. If the log says `NOT pinned`, it still runs, but slower.
- **16 GB:** the pinned host region doesn't fit. Options, best first:
  1. **A 24 GB+ GPU** holds every expert. Free the host copy after upload (above).
  2. **Keep only the non-resident experts in host memory.** Use static residency (`--no-adapt`), size the host
     region to the misses (`ExpertStore` load in `src/model/experts.cpp`), and disable swaps. That needs about
     (3,072 − resident) × 4.3 MiB of RAM.
  3. **Map the safetensors file and let Windows page it.** This works but is slow: misses become disk reads. Not
     recommended.

### Fewer cores, other CPUs
- **Threads:** `cpu_threads` = physical cores − 1. On the reference 8-core CPU, 7 threads measured best (96.0 tok/s);
  8, 10 and 14 (SMT siblings) were no faster. With 6 cores, use 5.
- **AVX2 is required.** The engine is compiled with `/arch:AVX2`, and the tier uses AVX2 intrinsics.
- **AVX-512 is unexplored** and could help the int8 dots (`src/model/cpu_tier.cpp`).
- **Cache:** the reference CPU has a 96 MB L3 (X3D). Prefetching predicted misses into L3 was rejected: only 5.4% of
  accesses would have hit. Smaller caches should matter little.

### Linux
- **The installer is Windows-only.**
- **Upstream ingine built on Linux with GCC**, and `CMakeLists.txt` still finds `nvcc` in `/opt/cuda` or a private
  toolkit. quill has only been built with MSVC, so expect small fixes: host memory and VMM code in `src/core/`,
  `/arch:AVX2` → `-mavx2 -mfma`.
- **Running it:** start `ingine-serve` with the arguments from `inkling-cli command` (or `server_args` in
  `installer/src/config.rs`). Set `QUILL_PASSES=3`.

---

## 4. Configure: `inkling.toml`

The launcher reads `<install>\inkling.toml` at every start. `inkling-cli command` prints the exact command line.
`inkling-cli launch --no-browser` runs the server in your terminal, so you see the log there (Ctrl+C stops it).

| Key | Default (reference) | Meaning |
|---|---|---|
| `server.host`, `server.port` | 127.0.0.1, 8003 | Where the API and chat page listen |
| `server.max_ctx` | 600064 | Context window (tokens). Must fit the region (§2). The model's maximum is 1048576. |
| `server.vram_cap_gib` | 11.3 | All-process VRAM cap, card size − 0.7. Lower it if the user games or renders at the same time. |
| `server.cpu_threads` | 7 | CPU-tier workers: physical cores − 1 |
| `server.vision` | "cpu" | `cpu` (no VRAM), `gpu` (0.6 GB VRAM, faster images), `off` |
| `server.vision_threads`, `server.image_max_tokens` | 8, 1024 | The image tower: threads, and tokens per image (one per 40×40 px, at most 10609) |
| `server.spec` | true | DFlash2 speculative decoding |
| `server.extra_args` | [] | More `ingine-serve` flags (`ingine-serve.exe --help`) |
| `[env]` | `QUILL_PASSES = "3"` | Environment of the server: the switches below. With more than one GPU the installer adds `CUDA_DEVICE_ORDER` / `CUDA_VISIBLE_DEVICES`. |

Useful `ingine-serve` flags for `extra_args`:

| Flag | Effect |
|---|---|
| `--verify-widths 2,4,8` | Speculative verify widths |
| `--spec-threshold 0.8` | Verify drafts above this probability |
| `--staging 24` | Prefill expert staging slots |
| `--prefill-chunk 2048` | Tokens per prefill chunk |
| `--no-adapt` | Static residency |
| `--vram-reserve-mib 512` | Memory kept free after the expert pools |
| `--park-mb` | Memory for a displaced conversation |
| `--skip-cold X` | **Lossy.** Drops non-resident experts below fraction X of the routing weight. Off. |

Engine switches, the `QUILL_*` environment variables (defaults in bold):

| Variable | Default | Effect |
|---|---|---|
| `QUILL_PASSES` | **3** | Loop passes. **Keep 3** (§0). |
| `QUILL_ASSIST` | **0** | Share of a call's CPU misses that the GPU reads over PCIe (zero-copy). `QUILL_ASSIST_MIN` (3) is the minimum number of misses. Re-tune on PCIe 5 / DDR5. |
| `QUILL_L2PF` | **1** | Prefetches the next layer's weights into L2 while waiting for the CPU tier |
| `QUILL_DRAFT_FP4` | **1** | 4-bit drafter matrices (0: FP6) |
| `QUILL_DRAFT_VOCAB` | **41456** | Head rows the drafter reads (frequency-ordered) |
| `QUILL_HEAD_TAIL4` | **0** | 4-bit rare head rows: +0.0009 KLD for ~1 tok/s. The user chose quality, so off. |
| `QUILL_DENSE_FP6` | **1** | FP6 dense weights (0: FP8) |
| `QUILL_RING_Q8` | **1** | int8 sliding-window KV (0: BF16). `QUILL_RING_Q4=1` is over the quality bar. |
| `QUILL_ATTN_SMMA`, `QUILL_ATTN_FMMA` | **1** | Tensor-core sliding and full attention |
| `QUILL_STEP_PREFILL` | **48** | Prompts up to N new tokens are prefilled through the decode graphs |
| `QUILL_RES_DECAY`, `QUILL_RES_MARGIN` | 0.96, 1.25 | Adaptive residency: count decay and swap margin |
| `QUILL_NO_DYN_VRAM` | unset | Disables dynamic VRAM (static region) |
| `QUILL_TIER_TRACE=file`, `QUILL_JOIN_DBG=1`, `QUILL_ATTN_DBG=1` | | Diagnostics |

---

## 5. Build from source

You need Visual Studio 2022 or Build Tools 2022 with "Desktop development with C++", and the CUDA Toolkit 13.x.
CMake and Ninja come with Visual Studio. `detect --json` lists anything missing with its winget command.

```powershell
engine\build.cmd 89                                        # RTX 40 -> engine\build\apps\serve\ingine-serve.exe
$env:QUILL_CUDA_ARCH = "120a"; engine\build.cmd "" build-120a   # an arch with a semicolon is easiest through the variable
inkling-cli install --from-source --source . --arch 120a        # build and install into the install folder
```

Arch values:
- `89` = RTX 40
- `120a` = RTX 50
- `90a` = H100 (untested)

A source install copies `cublasLt64_13.dll` from your toolkit. Rebuild into a fresh build folder after changing the
arch. Keep a build of the unchanged source for A/B comparisons.

---

## 6. Verify and measure

**Smoke test.** Start Inkling, then:

```powershell
curl http://127.0.0.1:8003/health
curl http://127.0.0.1:8003/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"inkling-2-mini-abliterated","messages":[{"role":"user","content":"hi"}],"max_tokens":50}'
```

**Start-up log lines to read:**
- `experts: 12.81 GiB in host memory (… pinned …)`
- `vram: dynamic region … MiB`
- `experts: shared KV/expert region X GiB: N slots (24 staging), M resident, KV up to Y GiB`

**Speed.** Stop the server first: the benchmark loads its own copy of the model. `bench/rp.ps1` replays 12 real
prompts (768 tokens each) under a VRAM watchdog. It takes the install folder's models, `bin\ingine-bench.exe` and
the cap and threads from `inkling.toml`:

```powershell
cd bench
.\rp.ps1 -Root $env:LOCALAPPDATA\Inkling -Label base -Seed -1          # teacher-forced: same text in every config, +-0.2 tok/s
.\rp.ps1 -Root $env:LOCALAPPDATA\Inkling -Label assist02 -Seed -1 -Env @{QUILL_ASSIST="0.2"}
.\rp.ps1 -Root $env:LOCALAPPDATA\Inkling -Label s7 -Seed 7             # sampled, real-world (+-1.5% from the text alone)
```

Results are appended to `bench/rp_results.txt`. The reference machine's history is already in that file.

For long context, copy `rp.ps1` and point its `--forced` at `replay_long.jsonl` (a 22k-token code prompt). For
prefill, use `ab.ps1 -Label x -Extra "--long-prompt","32768"`.

**Quality.** Generate the BF16 reference once (slow), then compare each configuration against it:

```powershell
$I = "$env:LOCALAPPDATA\Inkling"; $M = "$I\models"
& "$I\bin\quill-kld.exe" --model-dir $M\Inkling-2-Mini --ablation $M\ablation-ingine\inkling-2-mini-abliterated.safetensors --text data\wikitext2_test_head.txt --seqs 2 --ctx 1024 --max-ctx 300032 --vram-cap-gib 11.3 --dense-bf16 --kv-bf16 --save bench\kld\ref_wt2.bin
& "$I\bin\quill-kld.exe" --model-dir $M\Inkling-2-Mini --ablation $M\ablation-ingine\inkling-2-mini-abliterated.safetensors --text data\wikitext2_test_head.txt --seqs 2 --ctx 1024 --max-ctx 300032 --vram-cap-gib 11.3 --nvfp4-only --ref bench\kld\ref_wt2.bin
```

Pass the machine's own `--vram-cap-gib` and `--cpu-threads`. A change ships only if the KLD stays ≤ 0.0107 and
top-1 drops by at most ~1 point.

---

## 7. Code map

| Path | What |
|---|---|
| `engine/src/model/runtime.cpp` | Start-up: VRAM budget, shared region, dynamic VRAM, CPU tier, assist |
| `engine/src/model/experts.cpp`, `residency.cpp` | Host expert store, GPU pools, adaptive residency |
| `engine/src/model/cpu_tier.cpp` | CPU expert tier (int8 AVX2) |
| `engine/src/model/decoder.cpp`, `generator.cpp`, `spec.*` | Decode step, CUDA graphs per verify width, speculative loop |
| `engine/src/model/dflash2.cpp` | DFlash2 drafter |
| `engine/src/model/prefill.cpp` | Chunked prefill, expert staging |
| `engine/src/kernels/{attn,moe,dense,draft,prefill,head}` | CUDA kernels (`attn89.cu`: sm_89 attention) |
| `engine/apps/{serve,bench,kld}` | `ingine-serve`, `ingine-bench`, `quill-kld` |
| `engine/DESIGN.md`, `KERNELS.md` | Upstream design and kernel contracts. Paths there say `native/`; here it's `engine/`. |
| `installer/src/plan.rs` | Hardware → verdict and settings (the VRAM model from §2 is in code here) |
| `installer/src/config.rs` | `inkling.toml` and the server command line |

---

## 8. Report back

If you get it running on new hardware, ask the user whether they want to share the result as an issue or pull
request. Include:
- the hardware (`inkling-cli detect --json`)
- `inkling.toml`
- teacher-forced and sampled `rp.ps1` numbers
- the KLD
- any engine changes

That way the README's hardware table can grow from measurements instead of estimates.
