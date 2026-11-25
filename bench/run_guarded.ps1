# Runs a command under a VRAM watchdog: polls total device memory in use (all processes, as Task Manager's
# "Dedicated GPU memory") every 100 ms, records the peak, and kills the process if it exceeds the cap.
# A timeline (elapsed ms, MiB, whenever the total moves by >= 32 MiB) goes to <Log>.mem.
# Usage: run_guarded.ps1 -CapMiB 11571 -Log out.txt -- exe args...
param(
  [int]$CapMiB = 11571,
  [string]$Log = "run.log",
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$Cmd
)
$exe = $Cmd[0]
$cargs = if ($Cmd.Count -gt 1) { $Cmd[1..($Cmd.Count - 1)] } else { @() }
$argline = ($cargs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
$errLog = "$Log.err"
$p = Start-Process -FilePath $exe -ArgumentList $argline -RedirectStandardOutput $Log -RedirectStandardError $errLog -PassThru -NoNewWindow
$peak = 0; $killed = $false; $last = 0; $sw = [Diagnostics.Stopwatch]::StartNew()
"t_ms used_mib" | Out-File "$Log.mem"
while (-not $p.HasExited) {
  $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if ($used -gt $peak) { $peak = $used }
  if ([Math]::Abs($used - $last) -ge 32) { "$($sw.ElapsedMilliseconds) $used" | Out-File -Append "$Log.mem"; $last = $used }
  if ($used -gt $CapMiB) {
    $killed = $true
    "WATCHDOG: device memory $used MiB > cap $CapMiB MiB, killing" | Out-File -Append "$Log.wd"
    $p.Kill(); break
  }
  Start-Sleep -Milliseconds 100
}
$p.WaitForExit()
"WATCHDOG: peak device memory $peak MiB (cap $CapMiB MiB), exit $($p.ExitCode)$(if ($killed) {' KILLED'})" | Out-File -Append "$Log.wd"
Get-Content $errLog -ErrorAction SilentlyContinue
Get-Content $Log -ErrorAction SilentlyContinue
Get-Content "$Log.wd"
Remove-Item "$Log.wd"
