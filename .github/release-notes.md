## New in v0.1.2

- **Faster:** about **10% faster decode** in the author's everyday use. The controlled A/B runs were too noisy to
  confirm the figure or to say which change it comes from.
- **Chat page:** the live tok/s figure is more accurate. It read about 6–7% low, and lower early in a reply. This
  changes the display, not the decode speed, so don't compare its figures with the old counter's.
- **Chat page:** streaming replies repaint at most about 12 times a second, to take less GPU time from the model.
- **Engine, on by default:** non-temporal loads in the CPU expert tier (`QUILL_NT_STREAM=0` turns it off) and a
  higher Windows GPU scheduling priority (`QUILL_GPU_PRIORITY=0` turns it off). Neither changes the model's output
  math.
- **For tuning:** a routing trace (`QUILL_TIER_TRACE`), an offline simulator, and the results of the speed ideas
  tested so far in [`bench/FINDINGS.md`](https://github.com/mrrrrrrrrp/quill-inkling/blob/main/bench/FINDINGS.md).

**Updating:** close the Inkling server window, run the new `Inkling-Setup.exe`, keep the same folder and click
**Update / repair**. Downloaded models are kept.

---

**Inkling 2 Mini, local, on Windows + NVIDIA.** Confirmed on an RTX 40 with 12 GB + 32 GB RAM: 100+ tok/s decode,
600k context.

1. Download **Inkling-Setup.exe** and run it. SmartScreen may warn about an unsigned program: choose
   *More info → Run anyway*.
2. It checks your hardware, picks the settings, and downloads the models (~20 GB, verified), the engine and cuBLAS.
3. Start **Inkling** from the Start menu. The chat opens at http://127.0.0.1:8003/.

Different hardware (RTX 50, 16–32 GB GPUs, less RAM)? See the verdict in the installer, and point a coding agent
at AGENTS.md.

| File | What |
|---|---|
| `Inkling-Setup.exe` | Installer, launcher, settings and uninstaller (one program) |
| `inkling-cli.exe` | The same program for terminals, scripts and coding agents (`inkling-cli detect --json`) |
| `quill-engine-win64-cuda13-sm89-sm120a.zip` | The engine for RTX 40 and RTX 50. The installer downloads it for you. |
| `SHA256SUMS.txt` | Checksums. The installer verifies the engine against them. |
