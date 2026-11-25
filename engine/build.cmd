@echo off
rem Builds quill (ingine-serve, ingine-bench, quill-kld) with MSVC x64 + CUDA 13 + Ninja.
rem
rem   build.cmd [arch] [build-dir]
rem
rem   arch       CUDA architectures, quoted when it holds a semicolon. Default 89.
rem                89         RTX 40 (Ada)            - the tuned, confirmed target
rem                120a       RTX 50 (Blackwell)
rem                "89;120a"  both (what the release binary is built with)
rem   build-dir  Default: build
rem
rem The arch can also come from the QUILL_CUDA_ARCH environment variable (no quoting needed), e.g. from PowerShell:
rem   $env:QUILL_CUDA_ARCH = "89;120a"; .\build.cmd "" build-release
rem
rem Needs Visual Studio 2022 (or Build Tools 2022) with "Desktop development with C++" and the CUDA Toolkit 13.x.
rem CMake and Ninja come from Visual Studio when they are not on PATH.
setlocal
set "ARCH=%~1"
if "%ARCH%"=="" set "ARCH=%QUILL_CUDA_ARCH%"
if "%ARCH%"=="" set "ARCH=89"
set "BDIR=%~2"
if "%BDIR%"=="" set "BDIR=build"
set "SRC=%~dp0"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto novs
set "VS="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%i"
if not defined VS goto novs
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
rem vcvars adds VS's CMake and Ninja when the "C++ CMake tools" component is installed; add them explicitly otherwise
set "PATH=%PATH%;%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
where nvcc >nul 2>&1 || if defined CUDA_PATH set "PATH=%CUDA_PATH%\bin;%PATH%"

if not exist "%SRC%\%BDIR%\build.ninja" (
  cmake -S "%SRC%" -B "%SRC%\%BDIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=%ARCH%" -DINGINE_BUILD_TESTS=OFF || exit /b 1
)
cmake --build "%SRC%\%BDIR%" -j %NUMBER_OF_PROCESSORS% 2>&1 || exit /b 1
echo Built: %SRC%\%BDIR%\apps\serve\ingine-serve.exe
exit /b 0

:novs
echo Visual Studio 2022 or Build Tools 2022 with "Desktop development with C++" was not found.
echo   winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --wait"
exit /b 1
