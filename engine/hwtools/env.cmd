@rem Developer environment (MSVC x64 + CUDA) for building the hardware microbenchmarks, e.g.
@rem   call env.cmd ^& nvcc -O3 -arch=sm_89 -o hwbench.exe hwbench.cu
@for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do @call "%%i\VC\Auxiliary\Build\vcvars64.bat" >nul
