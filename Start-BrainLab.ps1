param([int]$Port = 8765, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$flysocRoot = $PSScriptRoot
$flysocPython = Join-Path $flysocRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $flysocPython)) {
    throw 'Run scripts\bootstrap.ps1 first to create the Python environment.'
}
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Choose a port between 1024 and 65535.' }
$flysocUrl = "http://127.0.0.1:$Port"
$flysocExisting = $null
try { $flysocExisting = Invoke-RestMethod "$flysocUrl/api/status" -TimeoutSec 2 } catch { }
if ($flysocExisting -and $flysocExisting.kc_dim -and $flysocExisting.connectome) {
    Write-Host "Brain Lab is already running: $flysocUrl"
    if (-not $NoBrowser) { Start-Process $flysocUrl }
    return
}
Push-Location -LiteralPath $flysocRoot
try {
    Write-Host "Starting Brain Lab at $flysocUrl. Keep this terminal open; Ctrl+C stops the server."
    $flysocArguments = @('scripts/brain_lab.py', '--port', "$Port")
    if (-not $NoBrowser) { $flysocArguments += '--open-browser' }
    & $flysocPython @flysocArguments
    if ($LASTEXITCODE -ne 0) { throw 'Brain Lab stopped with an error. See the message above.' }
} finally { Pop-Location }
