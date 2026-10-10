param(
    [string]$Results = '',
    [string]$Validation = '',
    [switch]$Draft,
    [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Results)) {
    $Results = Join-Path $projectDir 'results/research_20261003'
}
if ([string]::IsNullOrWhiteSpace($Validation)) {
    $candidatePath = Join-Path $PSScriptRoot 'verification/20261010/next_selection'
    if (Test-Path -LiteralPath (Join-Path $candidatePath 'current/confirmation/summary.json')) {
        $Validation = $candidatePath
    } elseif (-not $Draft) {
        throw 'Final reports require frozen development and confirmation evidence. Use -Draft for a reference-only preview.'
    }
}
$pythonPath = Join-Path $projectDir '.venv-research/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'The isolated research environment is missing; run scripts/setup.ps1 first.'
}
if (-not (Get-Command xelatex -ErrorAction SilentlyContinue)) {
    throw 'XeLaTeX is unavailable. No report source or stable PDF was replaced.'
}
$buildDir = Join-Path ([System.IO.Path]::GetTempPath()) ('robusthw-report-pair-' + [guid]::NewGuid().ToString('N'))
$sourceDir = Join-Path $buildDir 'source'
New-Item -ItemType Directory -Path $sourceDir | Out-Null
$sourceInputs = @{}
foreach ($relative in @('scripts/build_report.py','scripts/report_evidence.py','scripts/plot_results.py',
                       'reports/report_template.tex','reports/report_template_zh.tex','reports/build.ps1')) {
    $original = Join-Path $projectDir $relative
    $sourceInputs[$relative] = (Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash.ToLowerInvariant()
    $snapshot = Join-Path $sourceDir $relative
    New-Item -ItemType Directory -Path (Split-Path -Parent $snapshot) -Force | Out-Null
    Copy-Item -LiteralPath $original -Destination $snapshot
    if ((Get-FileHash -LiteralPath $snapshot -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sourceInputs[$relative]) {
        throw "Source changed during snapshot: $relative. Diagnostics: $buildDir"
    }
}
$sourceInputs | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $buildDir 'source.inputs.json') -Encoding UTF8
function Assert-ReportSources {
    foreach ($relative in $sourceInputs.Keys) {
        if ((Get-FileHash -LiteralPath (Join-Path $projectDir $relative) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sourceInputs[$relative]) {
            throw "Source changed during paired build: $relative. Products retained in $buildDir"
        }
    }
}
$figureDir = Join-Path $buildDir 'figures' 
New-Item -ItemType Directory -Path $figureDir | Out-Null
$commonArgs = @('--root', $Results)
if (-not [string]::IsNullOrWhiteSpace($Validation)) { $commonArgs += @('--validation-root', $Validation) }
if ($Draft) { $commonArgs += '--draft' }
& $pythonPath (Join-Path $sourceDir 'scripts/plot_results.py') @commonArgs --project-root $projectDir --output $figureDir --language both
if ($LASTEXITCODE -ne 0) { throw "Figure input validation failed. Diagnostics: $buildDir" }
$reportNames = @('experiment_report', 'experiment_report_zh')
foreach ($language in @('en', 'zh')) {
    $name = if ($language -eq 'zh') { $reportNames[1] } else { $reportNames[0] }
    & $pythonPath (Join-Path $sourceDir 'scripts/build_report.py') @commonArgs --project-root $projectDir `
        --template (Join-Path $sourceDir ('reports/report_template' + $(if ($language -eq 'zh') { '_zh' } else { '' }) + '.tex')) `
        --figure-root $figureDir --language $language --output (Join-Path $buildDir ($name + '.tex'))
    if ($LASTEXITCODE -ne 0) { throw "Report validation or $language generation failed. Diagnostics: $buildDir" }
}
Push-Location -LiteralPath $buildDir
try {
    foreach ($name in $reportNames) {
        foreach ($pass in 1..3) {
            $passLog = Join-Path $buildDir ($name + '-pass-' + $pass + '.txt')
            & xelatex -interaction=nonstopmode -halt-on-error ($name + '.tex') > $passLog
            if ($LASTEXITCODE -ne 0) {
                Get-Content -LiteralPath $passLog -Tail 35
                throw "XeLaTeX failed; neither stable report was replaced. Diagnostics: $buildDir"
            }
        }
        $logText = Get-Content -LiteralPath (Join-Path $buildDir ($name + '.log')) -Raw
        if ($logText -match 'Overfull|Missing character|undefined|LaTeX Warning') {
            throw "The $name log needs review; neither stable report was replaced. Diagnostics: $buildDir"
        }
    }
} finally {
    Pop-Location
}
Assert-ReportSources
if ($Draft -or $PrepareOnly) {
    Write-Output "Both reports built in $buildDir; stable artifacts were not replaced. Render and inspect the pair before publication."
    exit 0
}
# Publish only after both languages complete. Backups permit rollback if a
# destination is locked; figure copies do not replace unrelated theory assets.
$publications = @()
foreach ($name in $reportNames) {
    foreach ($extension in @('.tex', '.pdf', '.manifest.json')) {
        $publications += @{ Source = (Join-Path $buildDir ($name + $extension)); Target = (Join-Path $PSScriptRoot ($name + $extension)) }
    }
}
foreach ($asset in Get-ChildItem -LiteralPath $figureDir -File) {
    $publications += @{ Source = $asset.FullName; Target = (Join-Path $PSScriptRoot ('figures/' + $asset.Name)) }
}
$backupDir = Join-Path $buildDir 'previous'
New-Item -ItemType Directory -Path $backupDir | Out-Null
foreach ($index in 0..($publications.Count - 1)) {
    $item = $publications[$index]
    $item.Existed = Test-Path -LiteralPath $item.Target
    $item.Backup = Join-Path $backupDir ($index.ToString() + '.bak')
    if ($item.Existed) { Copy-Item -LiteralPath $item.Target -Destination $item.Backup }
}
Assert-ReportSources
$copied = @()
try {
    foreach ($item in $publications) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $item.Target) -Force | Out-Null
        $copied += $item
        Copy-Item -LiteralPath $item.Source -Destination $item.Target -Force
    }
} catch {
    foreach ($item in $copied) {
        if ($item.Existed) { Copy-Item -LiteralPath $item.Backup -Destination $item.Target -Force }
        elseif (Test-Path -LiteralPath $item.Target) { Remove-Item -LiteralPath $item.Target }
    }
    throw "Paired publication failed and previous files were restored. Diagnostics: $buildDir. $($_.Exception.Message)"
}
Write-Output "Built both stable reports: $(Join-Path $PSScriptRoot 'experiment_report.pdf') and $(Join-Path $PSScriptRoot 'experiment_report_zh.pdf')"
Write-Output "Intermediate files: $buildDir"
