@echo off
setlocal
cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"
title IMAGE_GEN - Config Run

rem -----------------------------------------------------------------------------
rem Optional startup arguments
rem Set COMMANDLINE_ARGS here, or define it before launching this BAT.
rem Examples:
rem   set "COMMANDLINE_ARGS=--xformers --medvram"
rem   set "COMMANDLINE_ARGS=--attention-backend xformers --memory-policy low_vram"
rem Explicit arguments passed to this BAT still override COMMANDLINE_ARGS.
rem -----------------------------------------------------------------------------
if not defined COMMANDLINE_ARGS set "COMMANDLINE_ARGS="

call "%PROJECT_ROOT%\scripts\resolve_python.bat" "%PROJECT_ROOT%"
if errorlevel 1 (
    pause
    exit /b 1
)
set "PYTHON_EXE=%IMAGE_GEN_PYTHON%"

set "CONFIG_PATH="
if "%~1"=="" goto config_path_ready
set "FIRST_ARG=%~1"
if "%FIRST_ARG:~0,2%"=="--" goto config_path_ready
set "CONFIG_PATH=%~1"
shift

:config_path_ready
if not defined CONFIG_PATH set "CONFIG_PATH=%PROJECT_ROOT%\configs\generation_config.yml"
if not exist "%CONFIG_PATH%" (
    color 4F
    echo ERROR: Generation config was not found:
    echo   %CONFIG_PATH%
    pause
    exit /b 2
)

echo Running the canonical run.bat pipeline from config:
echo   %CONFIG_PATH%
echo COMMANDLINE_ARGS=%COMMANDLINE_ARGS%
echo.
"%PYTHON_EXE%" -m modules.txt2img.cli run --project-root "%PROJECT_ROOT%" --config "%CONFIG_PATH%" --save %COMMANDLINE_ARGS% %*
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" echo IMAGE_GEN config run exited with code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
