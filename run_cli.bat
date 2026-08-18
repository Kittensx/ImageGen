@echo off
setlocal
cd /d "%~dp0"
echo NOTE: run_cli.bat is retained for compatibility. run.bat is the canonical launcher.

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

echo.
echo COMMANDLINE_ARGS=%COMMANDLINE_ARGS%

if "%~1"=="" (
    "%PYTHON_EXE%" -m modules.txt2img.cli run --interactive %COMMANDLINE_ARGS%
) else (
    "%PYTHON_EXE%" -m modules.txt2img.cli run %COMMANDLINE_ARGS% %*
)
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" echo IMAGE_GEN exited with code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
