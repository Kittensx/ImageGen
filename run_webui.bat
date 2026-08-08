@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if exist "%~dp0user_config\runtime_environment.bat" call "%~dp0user_config\runtime_environment.bat"
title IMAGE_GEN WebUI Server

rem -----------------------------------------------------------------------------
rem Optional startup arguments
rem Set COMMANDLINE_ARGS here, or define it before launching this BAT.
rem Examples:
rem   set "COMMANDLINE_ARGS=--xformers --medvram"
rem   set "COMMANDLINE_ARGS=--attention-backend xformers --memory-policy low_vram"
rem Explicit arguments passed to this BAT still override COMMANDLINE_ARGS.
rem -----------------------------------------------------------------------------
if not defined COMMANDLINE_ARGS set "COMMANDLINE_ARGS=--attention-backend auto"

rem Resolve the project virtual environment. Prefer .venv, then legacy venv.
set "VENV_DIR=%CD%\.venv"
if not exist "%VENV_DIR%\Scripts\python.exe" set "VENV_DIR=%CD%\venv"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo.
    echo ERROR: IMAGE_GEN virtual environment was not found.
    echo Expected one of:
    echo   %CD%\venv\Scripts\python.exe
    echo   %CD%\.venv\Scripts\python.exe
    echo.
    pause
    exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate the IMAGE_GEN virtual environment.
    echo   %VENV_DIR%
    echo.
    pause
    exit /b 1
)

set "WEBUI_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PYTHONPATH=%CD%\src;%CD%;%PYTHONPATH%"
set "IMAGE_GEN_WEBUI_START_PORT=7860"
set "IMAGE_GEN_WEBUI_URL_FILE=%TEMP%\image_gen_webui_%RANDOM%_%RANDOM%.url"
set "IMAGE_GEN_WEBUI_RESTART_FILE=%TEMP%\image_gen_webui_%RANDOM%_%RANDOM%.restart"
if exist "%IMAGE_GEN_WEBUI_URL_FILE%" del /q "%IMAGE_GEN_WEBUI_URL_FILE%" >nul 2>&1
if exist "%IMAGE_GEN_WEBUI_RESTART_FILE%" del /q "%IMAGE_GEN_WEBUI_RESTART_FILE%" >nul 2>&1

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
echo IMAGE_GEN WebUI launcher
echo ------------------------
echo Project root: %CD%
echo Virtual env:  %VENV_DIR%
echo Python:       %WEBUI_PYTHON%
echo Command args: %COMMANDLINE_ARGS%
"%WEBUI_PYTHON%" --version
if errorlevel 1 (
    echo.
    echo ERROR: The venv Python executable could not be started.
    pause
    exit /b 1
)

rem Always use the venv interpreter for pip. A visible ^(venv^) prompt is not
rem sufficient proof that the bare pip command points at this environment.
"%WEBUI_PYTHON%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo The venv does not currently have pip. Bootstrapping pip...
    "%WEBUI_PYTHON%" -m ensurepip --upgrade
    if errorlevel 1 (
        echo.
        echo ERROR: Unable to bootstrap pip inside the IMAGE_GEN venv.
        pause
        exit /b 1
    )
)

rem Install only the lightweight dependencies required to start the WebUI.
"%WEBUI_PYTHON%" -c "import fastapi, uvicorn, yaml" >nul 2>&1
if errorlevel 1 (
    echo.
    echo WebUI dependencies are missing from the Python 3.10 venv.
    echo Installing FastAPI, Uvicorn, and PyYAML into:
    echo   %VENV_DIR%
    echo.
    "%WEBUI_PYTHON%" -m pip install --disable-pip-version-check "fastapi>=0.115,<1" "uvicorn[standard]>=0.30,<1" "PyYAML>=6"
    if errorlevel 1 (
        echo.
        echo ERROR: WebUI dependency installation failed.
        echo You can retry manually with:
        echo   "%WEBUI_PYTHON%" -m pip install "fastapi>=0.115,<1" "uvicorn[standard]>=0.30,<1" "PyYAML>=6"
        echo.
        pause
        exit /b 1
    )
)

rem Print the actual import locations so environment mistakes are visible.
"%WEBUI_PYTHON%" -c "import sys, fastapi, uvicorn; print('FastAPI:', fastapi.__file__); print('Uvicorn:', uvicorn.__file__); print('Interpreter:', sys.executable)"
if errorlevel 1 (
    echo.
    echo ERROR: WebUI dependencies still cannot be imported from the venv.
    pause
    exit /b 1
)

echo.
echo MSLK_FMHA_POLICY=%MSLK_FMHA_POLICY%
echo MSLK_FMHA_DEBUG=%MSLK_FMHA_DEBUG%
echo MSLK_FMHA_BLOCK_N=%MSLK_FMHA_BLOCK_N%
echo MSLK_FMHA_BLOCK_M=%MSLK_FMHA_BLOCK_M%
echo MSLK_FMHA_NUM_WARPS=%MSLK_FMHA_NUM_WARPS%
echo MSLK_FMHA_NUM_STAGES=%MSLK_FMHA_NUM_STAGES%
echo.
echo Starting server on the first available port beginning with 127.0.0.1:%IMAGE_GEN_WEBUI_START_PORT%
echo If that port is occupied, IMAGE_GEN will use the next available port.
echo The browser will open after the selected health endpoint responds.
echo Press Ctrl+C in this window to stop the server.
echo.

rem Wait for the server to publish the selected URL, then open it after health succeeds.
start "" /b powershell.exe -NoProfile -NonInteractive -WindowStyle Hidden -Command ^
  "$path='%IMAGE_GEN_WEBUI_URL_FILE%';" ^
  "for($i=0; $i -lt 240; $i++){" ^
  "  if(Test-Path -LiteralPath $path){" ^
  "    $url=(Get-Content -LiteralPath $path -Raw -ErrorAction SilentlyContinue).Trim();" ^
  "    if($url){" ^
  "      $health=$url + '/api/health';" ^
  "      try { Invoke-WebRequest -UseBasicParsing -Uri $health -TimeoutSec 1 | Out-Null; Start-Process $url; exit 0 } catch {}" ^
  "    }" ^
  "  }" ^
  "  Start-Sleep -Milliseconds 500" ^
  "}"

rem Keep this BAT as the persistent supervisor. The Python backend can request
rem a full restart while the browser tab and launcher console remain open.
:launch_server
"%WEBUI_PYTHON%" -m image_gen.webui.server --project-root "%CD%" --host 127.0.0.1 --port %IMAGE_GEN_WEBUI_START_PORT% --url-file "%IMAGE_GEN_WEBUI_URL_FILE%" --restart-file "%IMAGE_GEN_WEBUI_RESTART_FILE%" %COMMANDLINE_ARGS% %*
set "SERVER_EXIT_CODE=%ERRORLEVEL%"

if exist "%IMAGE_GEN_WEBUI_RESTART_FILE%" (
    del /q "%IMAGE_GEN_WEBUI_RESTART_FILE%" >nul 2>&1
    echo.
    echo Restart requested by the WebUI. Starting a clean backend process...
    echo.
    goto launch_server
)

if exist "%IMAGE_GEN_WEBUI_URL_FILE%" del /q "%IMAGE_GEN_WEBUI_URL_FILE%" >nul 2>&1

if not "%SERVER_EXIT_CODE%"=="0" (
    echo.
    echo ERROR: The IMAGE_GEN WebUI server exited with code %SERVER_EXIT_CODE%.
    echo Review the messages above for the startup failure.
    echo.
    pause
)

exit /b %SERVER_EXIT_CODE%
