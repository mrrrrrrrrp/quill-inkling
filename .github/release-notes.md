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
