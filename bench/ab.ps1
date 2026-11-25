# A/B decode harness: the reference configuration (abliterated, DFlash2 speculation, temperature 1, 300k context)
# plus any extra flags; writes ab_<label>.log and prints the summary (tools: ab_summary.py).
# Usage: .\ab.ps1 -Label ctrl1 [-Root D:\Inkling] [-Extra "--prefill-chunk","1024"] [-NoSpec] [-Temp 0] [-Tokens 512] [-Runs 4]
#   -Root: the install folder (holds models\ and bin\); default $env:INKLING_HOME, else this repo.
param(
  [Parameter(Mandatory = $true)][string]$Label,
  [string]$Root = $(if ($env:INKLING_HOME) { $env:INKLING_HOME } else { Split-Path $PSScriptRoot -Parent }),
  [string[]]$Extra = @(),
  [switch]$NoSpec,
  [string]$Temp = "1",
  [int]$Tokens = 512,
  [int]$Runs = 4,
  [int]$Cap = 0
)
. (Join-Path $PSScriptRoot "common.ps1")
$q = Resolve-Quill $Root
if ($Cap -le 0) { $Cap = $q.CapMiB }
Set-Location $PSScriptRoot
$a = @("--model-dir", "$($q.Models)\Inkling-2-Mini", "--ablation", "$($q.Models)\ablation-ingine\inkling-2-mini-abliterated.safetensors",
       "--max-ctx", "300032", "--tokens", "$Tokens", "--runs", "$Runs", "--dense-fp8", "--effort", "none", "--nvfp4-only",
       "--temperature", $Temp) + $q.HwArgs
if (-not $NoSpec) { $a += @("--spec", "--draft-dir", "$($q.Models)\Inkling-2-Mini-DFlash2") }
$a += $Extra
$o = & .\run_guarded.ps1 -CapMiB $Cap -Log "ab_$Label.log" -- $q.Bench @a
($o | Select-String "WATCHDOG") | ForEach-Object { $_.Line } | Out-File -Append -Encoding utf8 "ab_$Label.log"
python -I "$PSScriptRoot\ab_summary.py" "ab_$Label.log" $Label
