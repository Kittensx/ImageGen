@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title IMAGE_GEN Python 3.10.20 Installer

set "PYTHON_VERSION=3.10.20"
set "PYTHON_ASSET=python-3.10.20-amd64.exe"
set "PYTHON_URL=https://github.com/Kittensx/CPython_Installer/releases/download/python-3.10.20-amd64/python-3.10.20-amd64.exe"
set "PYTHON_SHA256=a658798f56ab0c54b482c1a706f327cc01691bb5106089da79984d0ec7d60909"
set "DOWNLOAD_DIR=%TEMP%\IMAGE_GEN\python"
set "INSTALLER_PATH=%DOWNLOAD_DIR%\%PYTHON_ASSET%"

echo.
echo IMAGE_GEN Python prerequisite installer
echo =======================================
echo Required runtime: Python %PYTHON_VERSION% x64
echo.
echo This installer downloads the IMAGE_GEN validated Python 3.10.20 x64
echo release, verifies its SHA-256, launches the Python installer, then
echo confirms that the exact interpreter is available before returning.
echo.

call :has_exact_python
if not errorlevel 1 (
    echo Python %PYTHON_VERSION% x64 is already available. Nothing to do.
    goto success
)

where powershell.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: Windows PowerShell is required to download and verify Python.
    goto failure
)

if not exist "%DOWNLOAD_DIR%" mkdir "%DOWNLOAD_DIR%"
if errorlevel 1 (
    echo ERROR: Could not create download directory:
    echo   %DOWNLOAD_DIR%
    goto failure
)

echo Downloading %PYTHON_ASSET%...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -Uri '%PYTHON_URL%' -OutFile '%INSTALLER_PATH%'"
if errorlevel 1 (
    echo ERROR: Python download failed.
    goto failure
)

echo Verifying SHA-256...
set "ACTUAL_SHA256="
for /f "usebackq delims=" %%H in (`powershell.exe -NoProfile -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath '%INSTALLER_PATH%').Hash.ToLowerInvariant()"`) do set "ACTUAL_SHA256=%%H"

if not defined ACTUAL_SHA256 (
    echo ERROR: Could not calculate the downloaded installer's SHA-256.
    goto failure
)

if /I not "%ACTUAL_SHA256%"=="%PYTHON_SHA256%" (
    echo ERROR: Python installer SHA-256 mismatch.
    echo Expected: %PYTHON_SHA256%
    echo Actual:   %ACTUAL_SHA256%
    del /q "%INSTALLER_PATH%" >nul 2>&1
    goto failure
)

echo SHA-256 verified.
echo.
echo The Python installer will now open.
echo Complete the installation, then close the Python installer.
echo IMAGE_GEN will verify Python %PYTHON_VERSION% before continuing.
echo.
start "" /wait "%INSTALLER_PATH%"
set "PYTHON_INSTALL_EXIT=%ERRORLEVEL%"

if not "%PYTHON_INSTALL_EXIT%"=="0" (
    echo ERROR: Python installer exited with code %PYTHON_INSTALL_EXIT%.
    goto failure
)

call :has_exact_python
if errorlevel 1 (
    echo.
    echo ERROR: The Python installer completed, but Python %PYTHON_VERSION% x64
    echo could not be located through the Python launcher, PATH, or the standard
    echo per-user/system Python310 installation folders.
    echo.
    echo If you selected a custom installation directory, either enable the
    echo Python launcher/PATH option or rerun the Python installer using a
    echo standard installation location, then rerun this installer.
    goto failure
)

echo.
echo Python %PYTHON_VERSION% x64 was installed and verified.

:success
if not defined IMAGE_GEN_PARENT_INSTALL pause
exit /b 0

:failure
if not defined IMAGE_GEN_PARENT_INSTALL pause
exit /b 1

:has_exact_python
where py.exe >nul 2>&1
if not errorlevel 1 (
    py -3.10 -c "import struct,sys; raise SystemExit(0 if sys.version_info[:3]==(3,10,20) and struct.calcsize('P')*8==64 else 1)" >nul 2>&1
    if not errorlevel 1 exit /b 0
)

where python.exe >nul 2>&1
if not errorlevel 1 (
    python -c "import struct,sys; raise SystemExit(0 if sys.version_info[:3]==(3,10,20) and struct.calcsize('P')*8==64 else 1)" >nul 2>&1
    if not errorlevel 1 exit /b 0
)

for %%P in (
    "%LocalAppData%\Programs\Python\Python310\python.exe"
    "%ProgramFiles%\Python310\python.exe"
    "%ProgramFiles(x86)%\Python310\python.exe"
) do (
    if exist "%%~P" (
        "%%~P" -c "import struct,sys; raise SystemExit(0 if sys.version_info[:3]==(3,10,20) and struct.calcsize('P')*8==64 else 1)" >nul 2>&1
        if not errorlevel 1 exit /b 0
    )
)

exit /b 1
