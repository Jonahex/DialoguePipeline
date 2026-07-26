$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$uiExe = Join-Path $repoRoot ".venv\Scripts\dialogue-review.exe"

if (-not (Test-Path -LiteralPath $uiExe)) {
    throw "Virtual environment is missing. Run .\setup_venv.ps1 first."
}

& $uiExe
exit $LASTEXITCODE
