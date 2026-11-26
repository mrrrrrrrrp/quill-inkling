## New in v0.1.3

- **Added an unabliterated option.** `Inkling-Setup-Original.exe` installs the original model (API model id
  `inkling-2-mini`). **Choose it for the model's original quality:** abliteration can worsen overall model
  performance.
- The installer has a "Remove refusals (abliteration)" checkbox with that warning. It is also `abliterated` in
  `inkling.toml` and `--abliteration on|off` on the command line.
- Existing installs stay abliterated unless you change it. The engine's speed and math are unchanged.

**Updating:** close the Inkling server window, run an installer, keep the same folder, set the checkbox and click
**Update / repair**. Downloaded models are kept.

---

**Inkling 2 Mini, local, on Windows + NVIDIA.** Confirmed on an RTX 40 with 12 GB + 32 GB RAM: 100+ tok/s decode,
600k context.

1. Download **Inkling-Setup-Original.exe** or **Inkling-Setup.exe** and run it. SmartScreen may warn about an unsigned program: choose
   *More info → Run anyway*.
2. It checks your hardware, picks the settings, and downloads the models (~20 GB, verified), the engine and cuBLAS.
3. Start **Inkling** from the Start menu. The chat opens at http://127.0.0.1:8003/.

Different hardware (RTX 50, 16–32 GB GPUs, less RAM)? See the verdict in the installer, and point a coding agent
at AGENTS.md.

| File | What |
|---|---|
| `Inkling-Setup.exe` | Installer, launcher, settings and uninstaller (one program) |
| `Inkling-Setup-Original.exe` | The same program with abliteration switched off |
| `inkling-cli.exe` | The same program for terminals, scripts and coding agents (`inkling-cli detect --json`) |
| `quill-engine-win64-cuda13-sm89-sm120a.zip` | The engine for RTX 40 and RTX 50. The installer downloads it for you. |
| `SHA256SUMS.txt` | Checksums. The installer verifies the engine against them. |
