param([string]$PythonPath = '')
$ErrorActionPreference = 'Stop'
$flysocRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $flysocRoot

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if ($PythonPath) {
        & $PythonPath -m venv .venv
    } else {
        py -3.12 -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Venv creation failed. Supply -PythonPath for a Python 3.11/3.12 executable.' }
}

$env:PIP_NO_CACHE_DIR = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
& .\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& .\.venv\Scripts\python.exe -m pip install --no-cache-dir --no-build-isolation --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Editable package installation failed' }
& .\.venv\Scripts\python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
& .\.venv\Scripts\python.exe --version
