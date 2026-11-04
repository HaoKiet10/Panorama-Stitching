@echo off
rem ====================================================================================
rem build.bat - build the stitcher on Windows (produces stitch.exe next to this file).
rem Authors: <Member A>, <Member B>, <Member C>
rem
rem Needs: the CUDA Toolkit (nvcc on PATH) and Visual Studio C++ Build Tools, because nvcc
rem uses Microsoft's cl.exe as the host compiler on Windows.
rem
rem The script finds Visual Studio with vswhere, loads its 64-bit compiler environment
rem (vcvars64.bat), then runs the same nvcc command as the Makefile.
rem When you add a version (e.g. src\v2_tiled.cu), add it to the nvcc line at the bottom.
rem ====================================================================================
setlocal

rem ---- find the newest Visual Studio installation that has the C++ tools
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set VSPATH=
for /f "usebackq delims=" %%i in (`call "%VSWHERE%" -products * -latest -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH (
    echo Visual Studio C++ Build Tools not found.
    exit /b 1
)

rem ---- load the compiler environment (vcvars64 itself looks for vswhere on PATH)
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul

rem ---- compile from the folder this script is in
cd /d "%~dp0"
nvcc -O3 -std=c++17 -arch=native -o stitch.exe src\main.cu src\v0_sequential.cpp src\v1_basic.cu src\stb_impl.cpp
