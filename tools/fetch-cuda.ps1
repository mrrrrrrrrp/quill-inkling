# Assembles a private CUDA toolkit (compiler, runtime, cuBLAS, headers) from NVIDIA's redistributable archives:
# enough to build the engine with Ninja + MSVC, without the CUDA installer or admin rights. Used by the release
# workflow; works locally too.
#   .\tools\fetch-cuda.ps1 -Dest C:\cuda-13.3.1          then: $env:CUDACXX = "C:\cuda-13.3.1\bin\nvcc.exe"; $env:CUDA_PATH = "C:\cuda-13.3.1"
param(
  [string]$Version = "13.3.1",
  [Parameter(Mandatory = $true)][string]$Dest,
  [string[]]$Components = @("cuda_nvcc", "cuda_crt", "libnvvm", "cuda_cudart", "cccl", "cuda_nvtx", "cuda_profiler_api", "libcublas", "cuda_cuobjdump")
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$base = "https://developer.download.nvidia.com/compute/cuda/redist"
$manifest = Invoke-RestMethod "$base/redistrib_$Version.json"
New-Item -ItemType Directory -Force $Dest | Out-Null
$tmp = Join-Path ([IO.Path]::GetTempPath()) "cuda-redist-$Version"
New-Item -ItemType Directory -Force $tmp | Out-Null
foreach ($c in $Components) {
  $w = $manifest.$c."windows-x86_64"
  if (-not $w) { throw "$c has no windows-x86_64 archive in CUDA $Version" }
  $zip = Join-Path $tmp ([IO.Path]::GetFileName($w.relative_path))
  if (-not (Test-Path $zip) -or (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower() -ne $w.sha256) {
    Write-Host "downloading $c $($manifest.$c.version) ($([math]::Round([int64]$w.size / 1MB)) MB)"
    Invoke-WebRequest "$base/$($w.relative_path)" -OutFile $zip
    if ((Get-FileHash $zip -Algorithm SHA256).Hash.ToLower() -ne $w.sha256) { throw "checksum mismatch: $c" }
  }
  $x = Join-Path $tmp "x-$c"
  if (Test-Path $x) { Remove-Item -Recurse -Force $x }
  Expand-Archive $zip -DestinationPath $x
  # each archive holds one top folder (<name>-windows-x86_64-<ver>-archive) with bin\, include\, lib\, ...
  $top = Get-ChildItem $x -Directory | Select-Object -First 1
  Copy-Item -Recurse -Force (Join-Path $top.FullName "*") $Dest
  Remove-Item -Recurse -Force $x
}
"CUDA $Version assembled in $Dest"
