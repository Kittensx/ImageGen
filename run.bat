@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0user_config\runtime_environment.bat" call "%~dp0user_config\runtime_environment.bat"
title IMAGE_GEN - txt2img

rem -----------------------------------------------------------------------------
rem Optional startup arguments
rem Set COMMANDLINE_ARGS here, or define it before launching this BAT.
rem Examples:
rem   set "COMMANDLINE_ARGS=--xformers --medvram"
rem   set "COMMANDLINE_ARGS=--attention-backend xformers --memory-policy low_vram"
rem Explicit arguments passed to this BAT still override COMMANDLINE_ARGS.
rem -----------------------------------------------------------------------------
if not defined COMMANDLINE_ARGS set "COMMANDLINE_ARGS=--xformers"

call "%~dp0scripts\resolve_python.bat" "%~dp0"
if errorlevel 1 (
    pause
    exit /b 1
)
set "PYTHON_EXE=%IMAGE_GEN_PYTHON%"
rem Preserve values supplied by the caller; otherwise use the requested defaults.
if not defined MSLK_FMHA_POLICY set "MSLK_FMHA_POLICY=blackwell_safe"
if not defined MSLK_FMHA_DEBUG set "MSLK_FMHA_DEBUG="
if not defined MSLK_FMHA_BLOCK_N set "MSLK_FMHA_BLOCK_N="
if not defined MSLK_FMHA_BLOCK_M set "MSLK_FMHA_BLOCK_M="
if not defined MSLK_FMHA_NUM_WARPS set "MSLK_FMHA_NUM_WARPS="
if not defined MSLK_FMHA_NUM_STAGES set "MSLK_FMHA_NUM_STAGES="

echo.
echo COMMANDLINE_ARGS=%COMMANDLINE_ARGS%

set "LORA_SCAN_MODE=%IMAGE_GEN_SCAN_LORAS%"
if /I "%LORA_SCAN_MODE%"=="1" set "LORA_SCAN_MODE=missing"
if /I "%LORA_SCAN_MODE%"=="true" set "LORA_SCAN_MODE=missing"
if /I "%LORA_SCAN_MODE%"=="yes" set "LORA_SCAN_MODE=missing"
if /I "%LORA_SCAN_MODE%"=="on" set "LORA_SCAN_MODE=missing"
if /I "%LORA_SCAN_MODE%"=="all" goto :scan_loras
if /I "%LORA_SCAN_MODE%"=="missing" goto :scan_loras
goto :run_image_gen

:scan_loras
echo.
echo Scanning LoRAs before launch (mode=%LORA_SCAN_MODE%)...
"%PYTHON_EXE%" "%~dp0test_validations\scan_loras.py" --project-root "%~dp0" --mode "%LORA_SCAN_MODE%"
if errorlevel 1 (
    set "EXIT_CODE=%ERRORLEVEL%"
    echo.
    echo LoRA scan failed with code %EXIT_CODE%.
    if not defined IMAGE_GEN_NO_PAUSE pause
    exit /b %EXIT_CODE%
)

goto :run_image_gen

:run_image_gen
if "%~1"=="" (
    "%PYTHON_EXE%" -m modules.txt2img.cli run --interactive --save %COMMANDLINE_ARGS%
) else (
    "%PYTHON_EXE%" -m modules.txt2img.cli run %COMMANDLINE_ARGS% %*
)
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" echo IMAGE_GEN exited with code %EXIT_CODE%.
if not defined IMAGE_GEN_NO_PAUSE pause
exit /b %EXIT_CODE%
