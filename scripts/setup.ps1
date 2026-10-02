param([switch]$WithLogistic)
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$researchEnv = Join-Path $project '.venv-research'
$previousEnv = $env:UV_PROJECT_ENVIRONMENT
try {
    $env:UV_PROJECT_ENVIRONMENT = $researchEnv
    Push-Location $project
    $extras = @('--extra','experiments','--extra','dev')
    if ($WithLogistic) { $extras += @('--extra','logistic') }
    uv sync --locked @extras --python 3.12
    if ($LASTEXITCODE -ne 0) { throw 'Environment synchronization failed.' }
} finally {
    Pop-Location
    $env:UV_PROJECT_ENVIRONMENT = $previousEnv
}
Write-Output "Research environment ready: $researchEnv"
