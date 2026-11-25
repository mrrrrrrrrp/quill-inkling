# Inkling 2 Mini, local: 100+ tok/s with a 600k context on a 12 GB RTX 40 card

**quill** is an inference engine (C++/CUDA, no Python at run time) written for one model: **Inkling 2 Mini**
(abliterated), with **DFlash2** speculative decoding. The experts that fit in VRAM run on the GPU, and the rest run
on the CPU. It serves an **OpenAI-compatible API** and a **chat page** in your browser.

A Windows installer, written in Rust, checks your hardware, picks the settings, downloads and verifies everything,
and sets up a Start-menu launcher.

## Confirmed setup

This is the machine the engine was built and tuned on:

| | Confirmed working |
|---|---|
| GPU | **NVIDIA RTX 40 series with 12 GB** (RTX 4070 SUPER) |
| System memory | **32 GB** (DDR4-3000) |
| CPU | 8 cores with AVX2 (Ryzen 7 5800X3D) |
| OS / driver | Windows 11, NVIDIA driver R580+ (CUDA 13) |
| Disk | ~22 GB |

Results on it, at temperature 1 with DFlash2:

| | |
|---|---|
| Decode, chat and code | **100+ tok/s**: ~120 median on a 12-prompt replay set, 105–150 depending on content |
| Decode at 22k context | ~89 tok/s |
| Decode with a ~300k-token chat (code) | ~100 tok/s |
| Context window | **600k tokens** (600,064), q4 KV cache |
| Prefill | ~2,200 tok/s (32k-token prompt) |
| Quality | mean KLD 0.0071 vs the BF16 model (WikiText-2, 2 × 1,024 tokens). The bar for every change was 0.0107, the level of Unsloth's UD-Q4_K_XL |
| GPU memory | stays under 11.3 GB in total, counting every program (Task Manager's number), and gives memory back when other apps need it |

**Different hardware?** For an RTX 50, a 16–32 GB card, less RAM or another CPU, the installer scales the settings
(see below). To tune further, point your coding agent (Claude Code, Codex, …) at **[AGENTS.md](AGENTS.md)**. It
explains how the engine uses VRAM, RAM and the CPU, and walks the agent through setting it up and measuring it on
your machine.

## Install

1. Download **`Inkling-Setup.exe`** from the [latest release](../../releases/latest) and run it. Windows SmartScreen
   may warn about an unsigned program: click **More info → Run anyway**.
2. The window shows your hardware, a verdict and the settings it picked. Choose a folder (default
   `%LOCALAPPDATA%\Inkling`) and click **Install**.
   - Models: ~20 GB from Hugging Face at pinned revisions, no login needed.
   - Engine: ~160 MB prebuilt.
   - cuBLAS: ~400 MB from NVIDIA.

   Every file is checksum-verified. An interrupted install resumes where it stopped.
3. Start **Inkling** from the Start menu or the desktop. A server window opens and loads the model (about a minute),
   then the chat opens at http://127.0.0.1:8003/. Close the server window to stop it and free the GPU.

**Inkling Settings** in the Start menu changes the settings, repairs, or uninstalls. Inkling also appears in
*Apps & features*.

### What the installer picks for other hardware

| Your machine | Verdict | What changes |
|---|---|---|
| RTX 40, 12 GB, 32 GB RAM | **Confirmed** | Nothing: the reference settings |
| RTX 40 with 16–24 GB | Should work | VRAM cap = card size − 0.7 GB, so more experts stay on the GPU (expect it to be faster). Images on the GPU. 1M context available. |
| RTX 50 (Blackwell) | Should work | The prebuilt engine includes `sm_120a` code. Not confirmed on an RTX 50 yet. |
| 8 GB RTX 40/50 | Experimental | 262k context, few experts on the GPU, much slower |
| 24 GB RAM | Experimental | ~16 GB is pinned for the experts, so loading may fail |
| 16 GB RAM, RTX 30 or older, no AVX2 | Not supported as is | The engine needs changes; AGENTS.md describes them |

Every setting is in `inkling.toml` in the install folder, with a comment on each line. Edit it and restart Inkling.

### Command line

`inkling-cli.exe` is the same program built for terminals, scripts and coding agents: shells wait for it and
capture its output. It is attached to every release and also placed in the install folder.

```powershell
inkling-cli detect --json          # hardware, verdict, recommended settings, toolchain
inkling-cli install --dir D:\Inkling --max-ctx 262144 --env QUILL_ASSIST=0.2
inkling-cli install --from-source --arch 120a     # build the engine yourself (VS 2022 + CUDA 13)
inkling-cli install --models-from D:\old\models    # reuse downloaded models (hard links on the same drive)
inkling-cli command                # the exact server command line the launcher runs
inkling-cli launch --no-browser    # run the server in this terminal (Ctrl+C stops it)
inkling-cli uninstall --yes --keep-models
```

## Using it from other apps

The server is OpenAI-compatible:

| | |
|---|---|
| Base URL | `http://127.0.0.1:8003/v1` |
| Model id | `inkling-2-mini-abliterated` |
| Context | 600064 |
| Tool calls | supported |
| Thinking | `reasoning_effort`: none / minimal / low / medium / high / xhigh / max (default high) |

The thinking effort is a hint inside the prompt, not an on/off switch: "none" still produces a short thought, and
xhigh and max are the same level.

<details><summary>OpenCode (<code>~/.config/opencode/opencode.json</code>, in <code>"provider"</code>)</summary>

```json
"quill_inkling": {
  "name": "Inkling 2 Mini abliterated (quill, local)",
  "npm": "@ai-sdk/openai-compatible",
  "options": { "baseURL": "http://127.0.0.1:8003/v1" },
  "models": { "inkling-2-mini-abliterated": {
    "name": "Inkling 2 Mini abliterated (quill, 600K, DFlash2)",
    "reasoning": true, "tool_call": true, "temperature": true,
    "limit": { "context": 600064, "output": 65536 },
    "variants": { "off": {"reasoning_effort": "none"}, "minimal": {"reasoning_effort": "minimal"},
      "low": {"reasoning_effort": "low"}, "medium": {"reasoning_effort": "medium"}, "high": {"reasoning_effort": "high"},
      "xhigh": {"reasoning_effort": "xhigh"}, "max": {"reasoning_effort": "max"} } } }
}
```
</details>

<details><summary>pi (<code>~/.pi/agent/models.json</code>, in <code>"providers"</code>)</summary>

```json
"quill-inkling": {
  "baseUrl": "http://127.0.0.1:8003/v1", "api": "openai-completions", "apiKey": "none",
  "compat": { "supportsStore": false, "supportsDeveloperRole": false, "supportsReasoningEffort": true,
    "thinkingFormat": "reasoning_effort", "supportsUsageInStreaming": true, "supportsFinishReason": true,
    "maxTokensField": "max_tokens" },
  "models": [ { "id": "inkling-2-mini-abliterated", "name": "Inkling 2 Mini abliterated (quill, local)",
    "reasoning": true, "input": ["text"], "contextWindow": 600064, "maxTokens": 65536,
    "thinkingLevelMap": { "off": "none", "minimal": "minimal", "low": "low", "medium": "medium",
      "high": "high", "xhigh": "xhigh", "max": "max" },
    "samplingParams": { "temperature": 1.0 },
    "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 } } ]
}
```
</details>

## How it is fast

- **Speculative decoding:** the DFlash2 drafter proposes up to 7 tokens, and the model verifies 2–8 of them per
  step: ~2.2 tokens per verification at temperature 1.
- **Hybrid MoE:** the model has 3,072 experts (24 layers × 128, top-8). At short context ~1,800 of them live in
  VRAM as NVFP4. The rest run on the CPU with int8 AVX2 kernels reading from pinned RAM. Residency adapts to what
  the conversation routes to.
- **One shared VRAM region:** the KV cache grows from one end and expert slots fill from the other, so a long chat
  takes VRAM from the experts only as it grows. Unused context costs nothing.
- **Compact formats:** FP6 dense weights, a q4 full-attention KV cache (9 KiB per token), and an int8
  sliding-window cache. Each one was kept only if it stayed under the quality bar.

[`bench/PLAN.md`](bench/PLAN.md) logs every experiment, with numbers, kept or rejected.
[`engine/DESIGN.md`](engine/DESIGN.md) and [`engine/KERNELS.md`](engine/KERNELS.md) describe the engine.

## Repository layout

| | |
|---|---|
| `installer/` | The installer and launcher (Rust): `cargo build --release` gives `inkling.exe` |
| `engine/` | quill (C++20/CUDA 13): `engine\build.cmd 89` builds it with Visual Studio 2022 |
| `app/chat.html` | The chat page the server serves at `/` |
| `bench/` | Benchmark scripts, replay sets, the experiment log, results |
| `data/` | WikiText-2 excerpt for the quality (KLD) check |
| `models/PINS.txt` | Pinned Hugging Face revisions of the three model repositories |

### Building a release

```powershell
$env:QUILL_CUDA_ARCH = "89;120a"; engine\build.cmd "" build-release   # one engine binary for RTX 40 and RTX 50
$env:INKLING_REPO = "<owner>/<repo>"; cargo build --release --manifest-path installer\Cargo.toml
.\tools\package-release.ps1          # dist\: Inkling-Setup.exe, inkling-cli.exe, engine zip, SHA256SUMS.txt
```

Then attach the files in `dist\` to a GitHub release tagged `v<version>`. The version comes from
`installer/Cargo.toml`. The workflow in `.github/workflows/release.yml` does the same on a tag push.

No CUDA installed? `tools\fetch-cuda.ps1 -Dest C:\cuda` assembles the exact toolkit from NVIDIA's redistributable
archives. Then set `CUDA_PATH` to that folder and `CUDACXX` to its `bin\nvcc.exe`.

## Credits and licenses

- **quill** is a fork of **ingine** by satellitedown, which adapts fngine (Apache-2.0). See
  [`engine/NOTICE`](engine/NOTICE) for the full chain and the vendored libraries in `engine/third_party`, which keep
  their own licenses.
- **Models**, downloaded from Hugging Face and not included here, all Apache-2.0:
  - [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
  - [incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) (the drafter)
  - [satellitedown/Inkling-2-Mini-abliterated-ingine](https://huggingface.co/satellitedown/Inkling-2-Mini-abliterated-ingine)
    (the abliteration)
- **cuBLAS** is downloaded from NVIDIA's CUDA redistributables, under the CUDA Toolkit license.
- `data/wikitext2_test_head.txt` is an excerpt of WikiText-2 (Merity et al., 2016), CC BY-SA 3.0.
- Everything else in this repository is under the [Apache License 2.0](LICENSE).

The model is abliterated, which means its refusals are removed. You are responsible for how you use it.
