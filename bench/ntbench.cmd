@echo off
rem ntbench: SPEEDUP-IDEAS.md item 3 microbenchmark (no CUDA, no engine). MSVC x64 + AVX2, Release, /O2.
rem Question: can NT load/prefetch hints stop a 300 MB non-reused stream from evicting a 60 MiB
rem keep-set out of the 5800X3D's 96 MiB victim L3? Plus a 12 GiB mismatched-DIMM range scan.
rem This script only BUILDS ntbench.exe (into this bench dir); it does not run the benchmark.
pushd "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /MT /arch:AVX2 /fp:precise /EHsc /std:c++17 /Fe:ntbench.exe ntbench.cpp /link /SUBSYSTEM:CONSOLE
popd
