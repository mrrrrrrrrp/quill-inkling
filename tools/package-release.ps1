# Packages a release into dist\: Inkling-Setup.exe, the engine zip, SHA256SUMS.txt. Attach all three to a GitHub
# release tagged v<version> (installer\Cargo.toml). Build first:
#   $env:QUILL_CUDA_ARCH = "89;120a"; engine\build.cmd "" build-release
#   $env:INKLING_REPO = "<owner>/<repo>"; cargo build --release --manifest-path installer\Cargo.toml
param(
  [string]$EngineBuild = "engine\build-release",
  [string]$Installer = "installer\target\release\inkling.exe"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$ver = (Select-String -Path installer\Cargo.toml -Pattern '^version = "(.+)"').Matches[0].Groups[1].Value
$asset = "quill-engine-win64-cuda13-sm89-sm120a.zip"

# the installer must know its repository, or it cannot find the engine zip
$info = & $Installer detect --json | Out-String | ConvertFrom-Json
if ($info.version -ne $ver) { throw "installer reports version $($info.version), Cargo.toml says ${ver}: rebuild it" }
if ($info.repo -like "OWNER/*") { throw "the installer was built without INKLING_REPO; set it to <owner>/<repo> and rebuild" }

# the engine must hold code for both RTX 40 (sm_89) and RTX 50 (sm_120a)
$serve = Join-Path $EngineBuild "apps\serve\ingine-serve.exe"
$cuobjdump = @("$env:CUDA_PATH\bin\cuobjdump.exe", "cuobjdump") | Where-Object { Get-Command $_ -ErrorAction SilentlyContinue } | Select-Object -First 1
if ($cuobjdump) {
  $elf = & $cuobjdump --list-elf $serve | Out-String
  foreach ($sm in "sm_89", "sm_120a") { if ($elf -notmatch $sm) { throw "$serve has no $sm code: build with QUILL_CUDA_ARCH=89;120a" } }
} else { Write-Warning "cuobjdump not found: architectures not checked" }

$dist = "dist"
if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
$stage = New-Item -ItemType Directory -Force "$dist\stage\bin"
foreach ($e in "serve\ingine-serve.exe", "bench\ingine-bench.exe", "kld\quill-kld.exe") {
  Copy-Item (Join-Path $EngineBuild "apps\$e") $stage
}
Compress-Archive -Path "$dist\stage\bin" -DestinationPath "$dist\$asset" -CompressionLevel Optimal
Remove-Item -Recurse -Force "$dist\stage"
Copy-Item $Installer "$dist\Inkling-Setup.exe"
# inkling-cli.exe: the same program with the console subsystem (PE optional header + 68), for terminals and agents:
# PowerShell and cmd neither wait for a windowed program nor capture its output
$b = [IO.File]::ReadAllBytes((Resolve-Path $Installer).Path)
$sub = [BitConverter]::ToInt32($b, 0x3c) + 24 + 68
if ($b[$sub] -ne 2) { throw "unexpected PE subsystem $($b[$sub]) in $Installer (expected 2, windowed)" }
$b[$sub] = 3; $b[$sub + 1] = 0
[IO.File]::WriteAllBytes((Join-Path (Resolve-Path $dist).Path "inkling-cli.exe"), $b)
Get-ChildItem $dist -File | ForEach-Object { "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name } |
  Set-Content -Encoding ascii "$dist\SHA256SUMS.txt"
Get-ChildItem $dist | Select-Object Name, @{ n = "MB"; e = { [math]::Round($_.Length / 1MB, 1) } } | Format-Table -AutoSize
"release v$ver for $($info.repo): attach every file in $dist to the GitHub release v$ver"
