@echo off
setlocal
cd /d "%~dp0"
title IMAGE_GEN - txt2img

rem -----------------------------------------------------------------------------
rem Optional startup arguments
rem Set COMMANDLINE_ARGS here, or define it before launching this BAT.
rem Examples:
rem   set "COMMANDLINE_ARGS=--xformers --medvram"
rem   set "COMMANDLINE_ARGS=--attention-backend xformers --memory-policy low_vram"
rem Explicit arguments passed to this BAT still override COMMANDLINE_ARGS.
rem -----------------------------------------------------------------------------
if not defined COMMANDLINE_ARGS set "COMMANDLINE_ARGS="

call "%~dp0scripts\resolve_python.bat" "%~dp0"
if errorlevel 1 (
    pause
    exit /b 1
)
set "PYTHON_EXE=%IMAGE_GEN_PYTHON%"
rem Preserve values supplied by the caller; otherwise use the requested defaults.
if not defined MSLK_FMHA_POLICY set "MSLK_FMHA_POLICY=env"
if not defined MSLK_FMHA_DEBUG set "MSLK_FMHA_DEBUG="
if not defined MSLK_FMHA_BLOCK_N set "MSLK_FMHA_BLOCK_N=32"
if not defined MSLK_FMHA_BLOCK_M set "MSLK_FMHA_BLOCK_M="
if not defined MSLK_FMHA_NUM_WARPS set "MSLK_FMHA_NUM_WARPS=2"
if not defined MSLK_FMHA_NUM_STAGES set "MSLK_FMHA_NUM_STAGES="

echo.
echo COMMANDLINE_ARGS=%COMMANDLINE_ARGS%

if "%~1"=="" (
    "%PYTHON_EXE%" -m modules.txt2img.cli run --interactive --save %COMMANDLINE_ARGS%
) else (
    "%PYTHON_EXE%" -m modules.txt2img.cli run %COMMANDLINE_ARGS% %*
)
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" echo IMAGE_GEN exited with code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
