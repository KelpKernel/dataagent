$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
$env:PYTHONIOENCODING = "utf-8"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "venv not found. Run: uv venv .venv --python 3.13; uv pip install --python .venv\Scripts\python.exe -r requirements.txt"
    exit 1
}
& $python -m dataagent.cli @args
