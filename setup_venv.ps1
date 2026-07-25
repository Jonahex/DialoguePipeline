param(
    [string]$Python = "python",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if ($Recreate -and (Test-Path -LiteralPath $venvPath)) {
    $resolvedRepo = [System.IO.Path]::GetFullPath($repoRoot)
    $resolvedVenv = [System.IO.Path]::GetFullPath($venvPath)
    if (-not $resolvedVenv.StartsWith($resolvedRepo, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a virtual environment outside the repository."
    }
    Remove-Item -LiteralPath $resolvedVenv -Recurse -Force
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv $venvPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the virtual environment."
    }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}
& $venvPython -m pip install -r (Join-Path $repoRoot "requirements-dev.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install Python requirements."
}
& $venvPython -m pip install --no-build-isolation --no-deps -e $repoRoot
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the dialogue pipeline package."
}

Write-Host ""
Write-Host "Virtual environment ready: $venvPath"
Write-Host "Run: .\.venv\Scripts\dialogue-pipeline.exe --help"
