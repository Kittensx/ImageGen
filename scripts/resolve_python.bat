@echo off
rem Resolve IMAGE_GEN Python without requiring the environment to be activated.
rem Usage: call scripts\resolve_python.bat "<project-root>"
rem IMAGE_GEN_PYTHON is always resolved to a concrete python.exe path so callers
rem can safely invoke it as "%IMAGE_GEN_PYTHON%".
set "IMAGE_GEN_PYTHON="
set "IMAGE_GEN_VENV_DIR="
set "_IMAGE_GEN_ROOT=%~1"
if not defined _IMAGE_GEN_ROOT set "_IMAGE_GEN_ROOT=%CD%"

if exist "%_IMAGE_GEN_ROOT%\.venv\Scripts\python.exe" (
    set "IMAGE_GEN_VENV_DIR=%_IMAGE_GEN_ROOT%\.venv"
    set "IMAGE_GEN_PYTHON=%_IMAGE_GEN_ROOT%\.venv\Scripts\python.exe"
    exit /b 0
)
if exist "%_IMAGE_GEN_ROOT%\venv\Scripts\python.exe" (
    set "IMAGE_GEN_VENV_DIR=%_IMAGE_GEN_ROOT%\venv"
    set "IMAGE_GEN_PYTHON=%_IMAGE_GEN_ROOT%\venv\Scripts\python.exe"
    exit /b 0
)

where py >nul 2>&1
if not errorlevel 1 (
    for /f "usebackq delims=" %%P in (`py -3.10 -c "import sys; print(sys.executable)" 2^>nul`) do if not defined IMAGE_GEN_PYTHON set "IMAGE_GEN_PYTHON=%%P"
    if defined IMAGE_GEN_PYTHON exit /b 0
)

where python >nul 2>&1
if not errorlevel 1 (
    for /f "usebackq delims=" %%P in (`python -c "import sys; print(sys.executable)" 2^>nul`) do if not defined IMAGE_GEN_PYTHON set "IMAGE_GEN_PYTHON=%%P"
    if defined IMAGE_GEN_PYTHON exit /b 0
)

echo ERROR: No IMAGE_GEN Python interpreter was found.
echo Checked:
echo   %_IMAGE_GEN_ROOT%\.venv\Scripts\python.exe
echo   %_IMAGE_GEN_ROOT%\venv\Scripts\python.exe
echo   py -3.10
echo   python
exit /b 1
