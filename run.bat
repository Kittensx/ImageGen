@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if exist "%~dp0user_config\runtime_environment.bat" call "%~dp0user_config\runtime_environment.bat"
title IMAGE_GEN - txt2img

rem -----------------------------------------------------------------------------
rem Optional startup arguments
rem Set COMMANDLINE_ARGS here, or define it before launching this BAT.
rem Examples:
rem   set "COMMANDLINE_ARGS=--xformers --medvram"
rem   set "COMMANDLINE_ARGS=--attention-backend xformers --memory-policy low_vram"
rem Explicit generation arguments passed to this BAT still override defaults.
rem
rem Console progress memory display:
rem   compact - one in-place sampling line with compact VRAM figures (default)
rem   off     - hide memory figures
rem   json    - restore MEMORY_STATUS_JSON lines for diagnostics/integration
rem Normal compact mode suppresses STEP_PREVIEW_JSON console transport. WebUI
rem workers and explicit verbose diagnostics retain structured progress events.
rem Example:
rem   set "IMAGE_GEN_CONSOLE_MEMORY=off"
rem
rem Interactive launch modes:
rem   run.bat                 - choose Standard or Hires interactively
rem   run.bat standard        - standard interactive txt2img
rem   run.bat hires           - interactive neural-.pth hires txt2img
rem   set IMAGE_GEN_RUN_MODE=hires before launch to select hires without a menu
rem -----------------------------------------------------------------------------
if not defined COMMANDLINE_ARGS set "COMMANDLINE_ARGS=--attention-backend auto"
if not defined IMAGE_GEN_CONSOLE_MEMORY set "IMAGE_GEN_CONSOLE_MEMORY=compact"

set "IMAGE_GEN_SELECTED_RUN_MODE=%IMAGE_GEN_RUN_MODE%"
if /I "%~1"=="standard" set "IMAGE_GEN_SELECTED_RUN_MODE=standard"
if /I "%~1"=="--standard" set "IMAGE_GEN_SELECTED_RUN_MODE=standard"
if /I "%~1"=="hires" set "IMAGE_GEN_SELECTED_RUN_MODE=hires"
if /I "%~1"=="--hires" set "IMAGE_GEN_SELECTED_RUN_MODE=hires"

call "%~dp0scripts\resolve_python.bat" "%~dp0"
if errorlevel 1 (
    pause
    exit /b 1
)
set "PYTHON_EXE=%IMAGE_GEN_PYTHON%"

rem Preserve values supplied by the caller; otherwise use the requested defaults.
if not defined MSLK_FMHA_POLICY (
    if /I "%IMAGE_GEN_HARDWARE_QUALIFICATION%"=="community_unverified" (
        set "MSLK_FMHA_POLICY=auto"
    ) else (
        set "MSLK_FMHA_POLICY=blackwell_safe"
    )
)
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
goto :select_or_run

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

goto :select_or_run

:select_or_run
rem Any argument other than an explicit interactive mode remains a normal CLI run.
if not "%~1"=="" (
    if /I not "%~1"=="standard" if /I not "%~1"=="--standard" if /I not "%~1"=="hires" if /I not "%~1"=="--hires" goto :run_explicit_arguments
)

if /I "%IMAGE_GEN_SELECTED_RUN_MODE%"=="standard" goto :run_interactive_standard
if /I "%IMAGE_GEN_SELECTED_RUN_MODE%"=="hires" goto :run_interactive_hires

:choose_interactive_mode
echo.
echo === IMAGE_GEN Manual Runner ===
echo 1. Standard txt2img
echo 2. Hires / second-pass txt2img
echo.
set "IMAGE_GEN_MODE_CHOICE="
set /p "IMAGE_GEN_MODE_CHOICE=Choose run mode [1]: "
if not defined IMAGE_GEN_MODE_CHOICE set "IMAGE_GEN_MODE_CHOICE=1"
if "%IMAGE_GEN_MODE_CHOICE%"=="1" goto :run_interactive_standard
if "%IMAGE_GEN_MODE_CHOICE%"=="2" goto :run_interactive_hires
echo Invalid selection. Enter 1 or 2.
goto :choose_interactive_mode

:run_interactive_standard
echo.
echo Launching standard interactive txt2img...
"%PYTHON_EXE%" -m modules.txt2img.cli run --interactive --save --console-memory "%IMAGE_GEN_CONSOLE_MEMORY%" %COMMANDLINE_ARGS%
goto :capture_exit

:run_interactive_hires
echo.
echo Launching interactive neural-.pth hires txt2img on the configured runtime...
echo The launcher will prompt for a supported .pth upscaler and can save both low-res and pre-denoise high-res artifacts.
"%PYTHON_EXE%" -m modules.txt2img.cli run --interactive-hires --save --console-memory "%IMAGE_GEN_CONSOLE_MEMORY%" %COMMANDLINE_ARGS%
goto :capture_exit

:run_explicit_arguments
"%PYTHON_EXE%" -m modules.txt2img.cli run --console-memory "%IMAGE_GEN_CONSOLE_MEMORY%" %COMMANDLINE_ARGS% %*

:capture_exit
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" echo IMAGE_GEN exited with code %EXIT_CODE%.
if not defined IMAGE_GEN_NO_PAUSE pause
exit /b %EXIT_CODE%
