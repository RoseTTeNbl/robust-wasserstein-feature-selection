param([string]$Environment = '.venv-research')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $env:UV_PROJECT_ENVIRONMENT = $Environment
    uv sync --locked --extra experiments --extra dev --python 3.12
    if ($LASTEXITCODE -ne 0) { throw 'Environment installation failed' }
} finally {
    Pop-Location
}
