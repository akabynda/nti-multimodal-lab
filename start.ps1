$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw 'Create .venv and install requirements first, as described in README.md'
}
& $taskPython -m uvicorn nti_lab.app:app --host 127.0.0.1 --port 8765
