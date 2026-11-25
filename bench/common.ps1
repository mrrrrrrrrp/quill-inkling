# Shared by rp.ps1 / ab.ps1: finds the models, the bench binary and this machine's settings under an install folder.
function Resolve-Quill([string]$Root) {
  $Root = (Resolve-Path $Root).Path
  $models = Join-Path $Root "models"
  if (-not (Test-Path "$models\Inkling-2-Mini\config.json")) {
    throw "No models under $models. Pass -Root <install folder> or set INKLING_HOME (the installer's folder)."
  }
  $bench = @("$Root\bin\ingine-bench.exe", "$Root\engine\build\apps\bench\ingine-bench.exe", "$Root\engine\build-release\apps\bench\ingine-bench.exe") |
    Where-Object { Test-Path $_ } | Select-Object -First 1
  if (-not $bench) { throw "ingine-bench.exe not found under $Root\bin or $Root\engine\build*" }
  # inkling.toml (written by the installer): same VRAM cap and CPU threads as the chat server
  $cap = 11.3; $threads = 7
  $toml = Join-Path $Root "inkling.toml"
  if (Test-Path $toml) {
    $t = Get-Content $toml -Raw
    if ($t -match '(?m)^\s*vram_cap_gib\s*=\s*([0-9.]+)') { $cap = [double]$Matches[1] }
    if ($t -match '(?m)^\s*cpu_threads\s*=\s*(\d+)') { $threads = [int]$Matches[1] }
  }
  [pscustomobject]@{
    Models = $models
    Bench  = $bench
    HwArgs = @("--vram-cap-gib", "$cap", "--cpu-threads", "$threads")
    CapMiB = [int]($cap * 1024 + 300)   # watchdog: the engine's own cap plus headroom for its transient peaks
  }
}
