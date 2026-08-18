@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo IMAGE_GEN SD3 support requires the main IMAGE_GEN environment first.
  echo Run the normal IMAGE_GEN installer, then rerun this file.
  exit /b 2
)

"%PYTHON%" "scripts\setup\install_sd3_support.py" --project-root "%CD%" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" pause
exit /b %RC%
