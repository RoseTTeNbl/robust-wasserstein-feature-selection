$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectDir '.venv-research/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pythonPath = (Get-Command python -ErrorAction Stop).Source
}
$checker = Join-Path $projectDir 'scripts/check_note_sync.py'
$manifestPath = Join-Path $PSScriptRoot 'bilingual_sync.json'
& $pythonPath $checker
if ($LASTEXITCODE -ne 0) {
    throw 'Bilingual source review is missing or stale. No PDF was replaced.'
}
$manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash
$approved = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$buildDir = Join-Path ([System.IO.Path]::GetTempPath()) ('robusthw-note-pair-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $buildDir | Out-Null
$jobs = @(
    [pscustomobject]@{ Language = 'english'; Source = 'main.tex'; Pdf = 'robust_coordinate_selection.pdf'; Hash = $approved.sources.english.sha256 },
    [pscustomobject]@{ Language = 'chinese'; Source = 'main_zh.tex'; Pdf = 'robust_coordinate_selection_zh.pdf'; Hash = $approved.sources.chinese.sha256 }
)

# Compile immutable source snapshots; neither stable PDF is touched here.
foreach ($job in $jobs) {
    $jobDir = Join-Path $buildDir $job.Language
    New-Item -ItemType Directory -Path $jobDir | Out-Null
    $snapshot = Join-Path $jobDir $job.Source
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $job.Source) -Destination $snapshot
    if ((Get-FileHash -LiteralPath $snapshot -Algorithm SHA256).Hash -ne $job.Hash) {
        throw "The $($job.Language) source changed while preparing the build. No PDF was replaced."
    }
    Push-Location -LiteralPath $jobDir
    try {
        # A third pass stabilizes references used in appendix headings.
        foreach ($pass in 1..3) {
            $logPath = Join-Path $jobDir "pass-$pass.txt"
            & xelatex -interaction=nonstopmode -halt-on-error $job.Source > $logPath
            if ($LASTEXITCODE -ne 0) {
                Get-Content -LiteralPath $logPath -Tail 35
                throw "$($job.Language) compilation failed. Previous PDFs remain previous builds; diagnostics: $buildDir"
            }
        }
    } finally {
        Pop-Location
    }
}

# An edit or a new review during compilation invalidates these snapshots.
& $pythonPath $checker
if ($LASTEXITCODE -ne 0 -or (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash -ne $manifestHash) {
    throw "Sources or their review changed during compilation. No PDF was replaced; diagnostics: $buildDir"
}
foreach ($job in $jobs) {
    $built = Join-Path (Join-Path $buildDir $job.Language) ([System.IO.Path]::ChangeExtension($job.Source, '.pdf'))
    if (-not (Test-Path -LiteralPath $built) -or (Get-Item -LiteralPath $built).Length -eq 0) {
        throw "The $($job.Language) PDF is missing. No PDF was replaced."
    }
    $destination = Join-Path $PSScriptRoot $job.Pdf
    if (Test-Path -LiteralPath $destination) {
        Copy-Item -LiteralPath $destination -Destination (Join-Path $buildDir ($job.Language + '.previous.pdf'))
    }
}

# Publish only after both succeeded; restore the previous pair on copy failure.
try {
    foreach ($job in $jobs) {
        $built = Join-Path (Join-Path $buildDir $job.Language) ([System.IO.Path]::ChangeExtension($job.Source, '.pdf'))
        $destination = Join-Path $PSScriptRoot $job.Pdf
        Copy-Item -LiteralPath $built -Destination $destination
        if ((Get-FileHash -LiteralPath $built -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash) {
            throw "Published $($job.Language) PDF differs from the completed build."
        }
    }
} catch {
    foreach ($job in $jobs) {
        $destination = Join-Path $PSScriptRoot $job.Pdf
        $backup = Join-Path $buildDir ($job.Language + '.previous.pdf')
        if (Test-Path -LiteralPath $backup) {
            Copy-Item -LiteralPath $backup -Destination $destination
        } elseif (Test-Path -LiteralPath $destination) {
            Remove-Item -LiteralPath $destination
        }
    }
    throw "Pair publication failed; prior PDFs restored. No current bilingual pair was published. Diagnostics: $buildDir"
}
Write-Output "Published the reviewed English/Chinese PDF pair: $($jobs.Pdf -join ', ')"
Write-Output "Review: $($approved.reviewed_by), $($approved.reviewed_at_utc)"
Write-Output "Intermediate files: $buildDir"
