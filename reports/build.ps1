param(
    [string]$Results = '',
    [switch]$Draft
)
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Results)) {
    $Results = Join-Path $projectDir 'results/research_20261003'
}
$pythonPath = Join-Path $projectDir '.venv-research/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'The isolated research environment is missing; run scripts/setup.ps1 first.'
}
$buildDir = Join-Path ([System.IO.Path]::GetTempPath()) ('robusthw-report-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $buildDir | Out-Null
$reportName = 'experiment_report'
$sourcePath = if ($Draft) { Join-Path $buildDir ($reportName + '.tex') } else { Join-Path $PSScriptRoot ($reportName + '.tex') }
$generatorArgs = @((Join-Path $projectDir 'scripts/build_report.py'), '--project-root', $projectDir,
                   '--root', $Results, '--template', (Join-Path $PSScriptRoot 'report_template.tex'),
                   '--output', $sourcePath)
if ($Draft) { $generatorArgs += '--draft' }
& $pythonPath @generatorArgs
if ($LASTEXITCODE -ne 0) { throw 'Report input validation or source generation failed.' }

if (-not $Draft) {
    Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $buildDir ($reportName + '.tex'))
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'figures') -Destination (Join-Path $buildDir 'figures') -Recurse
Push-Location -LiteralPath $buildDir
try {
    foreach ($pass in 1..2) {
        & xelatex -interaction=nonstopmode -halt-on-error ($reportName + '.tex') > (Join-Path $buildDir "pass-$pass.txt")
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath (Join-Path $buildDir "pass-$pass.txt") -Tail 35
            throw "XeLaTeX failed; diagnostics are in $buildDir"
        }
    }
    if ($Draft) {
        Write-Output "Built draft $(Join-Path $buildDir ($reportName + '.pdf'))"
    } else {
        Copy-Item -LiteralPath (Join-Path $buildDir ($reportName + '.pdf')) -Destination (Join-Path $PSScriptRoot ($reportName + '.pdf'))
        Write-Output "Built $(Join-Path $PSScriptRoot ($reportName + '.pdf'))"
    }
    Write-Output "Intermediate files: $buildDir"
} finally {
    Pop-Location
}
