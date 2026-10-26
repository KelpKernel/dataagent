@echo off
setlocal
set "PYTHONPATH=%~dp0src"
set "PYTHONIOENCODING=utf-8"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Run these first:
  echo   uv venv .venv --python 3.13
  echo   uv pip install --python .venv\Scripts\python.exe -r requirements.txt
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -m litmus.cli %*
