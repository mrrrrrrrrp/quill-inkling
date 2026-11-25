# Replay A/B on the reference configuration (abliterated, DFlash2, T=1, 300k ctx): the 12-prompt T=1 replay set in
# free sampled mode (seed 7; -Seed -1 = teacher-forced, identical text in every config, +-0.2 tok/s).
# Appends "label | resident | tok/s | tok/verify | rounds" to rp_results.txt.
# Usage: .\rp.ps1 -Label x [-Root D:\Inkling] [-Extra "--cpu-threads","5"] [-Env @{QUILL_ASSIST="0.2"}] [-Items 12]
#   -Root: the install folder (holds models\ and bin\); default $env:INKLING_HOME, else this repo.
#   -Cap:  VRAM watchdog in MiB (all processes); default: the vram_cap_gib in <Root>\inkling.toml + 300 MiB, else 11650.
param(
  [Parameter(Mandatory = $true)][string]$Label,
  [string]$Root = $(if ($env:INKLING_HOME) { $env:INKLING_HOME } else { Split-Path $PSScriptRoot -Parent }),
  [string[]]$Extra = @(),
  [hashtable]$Env = @{},
  [int]$Items = 12,
  [int]$Seed = 7,
  [int]$Cap = 0
)
. (Join-Path $PSScriptRoot "common.ps1")
$q = Resolve-Quill $Root
if ($Cap -le 0) { $Cap = $q.CapMiB }
Set-Location $PSScriptRoot
foreach ($k in $Env.Keys) { Set-Item -Path "Env:$k" -Value $Env[$k] }
$a = @("--model-dir", "$($q.Models)\Inkling-2-Mini", "--ablation", "$($q.Models)\ablation-ingine\inkling-2-mini-abliterated.safetensors",
       "--max-ctx", "300032", "--dense-fp8", "--nvfp4-only", "--temperature", "1", "--spec", "--draft-dir",
       "$($q.Models)\Inkling-2-Mini-DFlash2", "--forced", "$PSScriptRoot\replay_t1.jsonl", "--forced-free", "$Seed",
       "--forced-items", "$Items") + $q.HwArgs + $Extra
$o = & .\run_guarded.ps1 -CapMiB $Cap -Log "rp_$Label.log" -- $q.Bench @a
foreach ($k in $Env.Keys) { Remove-Item -Path "Env:$k" -ErrorAction SilentlyContinue }
$res = ($o | Select-String "(\d+) resident" | Select-Object -First 1)
$r = if ($res) { $res.Matches[0].Groups[1].Value } else { "?" }
$tot = $o | Select-String "^TOTAL" | Select-Object -First 1
$wd = $o | Select-String "WATCHDOG" | Select-Object -First 1
$line = "{0,-16} | resident {1} | {2} | {3}" -f $Label, $r, ($(if ($tot) { ($tot.Line -replace '\s+', ' ') } else { "NO RESULT" })), ($(if ($wd) { $wd.Line } else { "" }))
$line | Out-File -Append -Encoding utf8 rp_results.txt
$line
