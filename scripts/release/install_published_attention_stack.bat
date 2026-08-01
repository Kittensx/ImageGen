@echo off
setlocal EnableExtensions
cd /d "%~dp0\..\.."
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)
set "PYTHONPATH=%CD%;%CD%\src"
"%PY%" scripts\release\install_published_attention_stack.py --project-root "%CD%" --python "%PY%" %*
exit /b %ERRORLEVEL%
