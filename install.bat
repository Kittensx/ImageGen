@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title IMAGE_GEN Installer

echo.
echo IMAGE_GEN hardware-aware installer
echo ==================================
echo This installer detects NVIDIA GPUs, installed CUDA toolkits, and the
echo NVIDIA driver's supported CUDA level. It installs only a published,
echo validated PyTorch + Triton + custom MSLK + custom xFormers profile.
echo A machine-specific user lock is written only after validation succeeds.
echo.

set "PYTHON_EXE="
set "PYTHON_ARGS="

:detect_python
where py.exe >nul 2>&1
if not errorlevel 1 (
    py -3.10 -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) and struct.calcsize('P')*8==64 else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=py.exe"
        set "PYTHON_ARGS=-3.10"
        goto python_ready
    )
)

where python.exe >nul 2>&1
if not errorlevel 1 (
    python -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) and struct.calcsize('P')*8==64 else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=python.exe"
        set "PYTHON_ARGS="
        goto python_ready
    )
)

for %%P in (
    "%LocalAppData%\Programs\Python\Python310\python.exe"
    "%ProgramFiles%\Python310\python.exe"
    "%ProgramFiles(x86)%\Python310\python.exe"
) do (
    if exist "%%~P" (
        "%%~P" -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) and struct.calcsize('P')*8==64 else 1)" >nul 2>&1
        if not errorlevel 1 (
            set "PYTHON_EXE=%%~P"
            set "PYTHON_ARGS="
            goto python_ready
        )
    )
)

if defined IMAGE_GEN_PYTHON_INSTALL_ATTEMPTED (
    echo ERROR: Python 3.10 x64 was installed but could not be located.
    echo Close this window, open a new Command Prompt, and rerun install.bat.
    pause
    exit /b 1
)

echo ERROR: Python 3.10 x64 was not found.
echo.
where winget.exe >nul 2>&1
if errorlevel 1 (
    echo Install Python 3.10 x64, enable the Python launcher, then rerun install.bat.
    pause
    exit /b 1
)
set /p "INSTALL_PYTHON=Install Python 3.10 x64 with winget now? [Y/N]: "
if /I not "%INSTALL_PYTHON%"=="Y" exit /b 1
winget install --id Python.Python.3.10 --exact --source winget --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo ERROR: Python installation failed.
    pause
    exit /b 1
)
set "IMAGE_GEN_PYTHON_INSTALL_ATTEMPTED=1"
goto detect_python

:python_ready
if not exist "%SystemRoot%\System32\vcruntime140.dll" (
    echo.
    echo Microsoft Visual C++ Runtime was not detected.
    where winget.exe >nul 2>&1
    if errorlevel 1 (
        echo Install the Microsoft Visual C++ 2015-2022 x64 Runtime, then rerun install.bat.
        pause
        exit /b 1
    )
    set /p "INSTALL_VC=Install the Microsoft Visual C++ x64 Runtime with winget? [Y/N]: "
    if /I not "%INSTALL_VC%"=="Y" (
        echo The Visual C++ x64 Runtime is required.
        pause
        exit /b 1
    )
    winget install --id Microsoft.VCRedist.2015+.x64 --exact --source winget --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo ERROR: Visual C++ Runtime installation failed.
        pause
        exit /b 1
    )
)

echo.
"%PYTHON_EXE%" %PYTHON_ARGS% "%CD%\scripts\setup\install_image_gen.py" --project-root "%CD%" %*
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
    echo IMAGE_GEN installation failed with code %EXIT_CODE%.
) else (
    echo IMAGE_GEN installation completed successfully.
)
if not defined IMAGE_GEN_NO_PAUSE pause
exit /b %EXIT_CODE%
