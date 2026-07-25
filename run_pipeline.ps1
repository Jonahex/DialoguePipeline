param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PipelineArgs
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pipelineExe = Join-Path $repoRoot ".venv\Scripts\dialogue-pipeline.exe"

if (-not (Test-Path -LiteralPath $pipelineExe)) {
    throw "Virtual environment is missing. Run .\setup_venv.ps1 first."
}

& $pipelineExe @PipelineArgs
exit $LASTEXITCODE

