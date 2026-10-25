param([switch]$Bilingual)

$ErrorActionPreference = 'Stop'
$projectDir = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectDir '.venv-research/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $pythonPath = (Get-Command python -ErrorAction Stop).Source
}
$checker = Join-Path $projectDir 'scripts/check_note_sync.py'
$checkerArgs = @()
$manifestName = 'english_review.json'
$buildMode = 'english'
if ($Bilingual) {
    $checkerArgs += '--bilingual'
    $manifestName = 'bilingual_sync.json'
    $buildMode = 'pair'
}
$manifestPath = Join-Path $PSScriptRoot $manifestName
& $pythonPath $checker @checkerArgs
if ($LASTEXITCODE -ne 0) {
    throw 'Source review is missing or stale. No PDF was replaced.'
}
$manifestHash = (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash
$approved = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$bibliographyPath = Join-Path $PSScriptRoot 'references.bib'
if (-not (Test-Path -LiteralPath $bibliographyPath)) {
    throw 'The shared references.bib is missing. No PDF was replaced.'
}
$bibliographyHash = (Get-FileHash -LiteralPath $bibliographyPath -Algorithm SHA256).Hash
if (-not $approved.bibliography -or $approved.bibliography.path -ne 'references.bib' -or $approved.bibliography.sha256 -ne $bibliographyHash) {
    throw 'The shared bibliography differs from its reviewed version. No PDF was replaced.'
}
$buildDir = Join-Path ([System.IO.Path]::GetTempPath()) ('robusthw-note-' + $buildMode + '-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $buildDir | Out-Null
$jobs = @(
    [pscustomobject]@{ Language = 'english'; Source = 'main.tex'; Pdf = 'robust_coordinate_selection.pdf'; Hash = $approved.sources.english.sha256 }
)
if ($Bilingual) {
    $jobs += [pscustomobject]@{ Language = 'chinese'; Source = 'main_zh.tex'; Pdf = 'robust_coordinate_selection_zh.pdf'; Hash = $approved.sources.chinese.sha256 }
}

# Compile immutable source snapshots; no stable PDF is touched here.
foreach ($job in $jobs) {
    $jobDir = Join-Path $buildDir $job.Language
    New-Item -ItemType Directory -Path $jobDir | Out-Null
    $snapshot = Join-Path $jobDir $job.Source
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $job.Source) -Destination $snapshot
    $bibliographySnapshot = Join-Path $jobDir 'references.bib'
    Copy-Item -LiteralPath $bibliographyPath -Destination $bibliographySnapshot
    if ((Get-FileHash -LiteralPath $snapshot -Algorithm SHA256).Hash -ne $job.Hash) {
        throw "The $($job.Language) source changed while preparing the build. No PDF was replaced."
    }
    if ((Get-FileHash -LiteralPath $bibliographySnapshot -Algorithm SHA256).Hash -ne $bibliographyHash) {
        throw 'The bibliography changed while preparing the build. No PDF was replaced.'
    }
    Push-Location -LiteralPath $jobDir
    try {
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($job.Source)
        $steps = @('pdflatex-1', 'bibtex', 'pdflatex-2', 'pdflatex-3')
        foreach ($step in $steps) {
            $logPath = Join-Path $jobDir "$step.txt"
            if ($step -eq 'bibtex') {
                & bibtex $stem > $logPath
            } else {
                & pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape $job.Source > $logPath
            }
            if ($LASTEXITCODE -ne 0) {
                Get-Content -LiteralPath $logPath -Tail 35
                throw "$($job.Language) $step failed. Previous PDFs remain previous builds; diagnostics: $buildDir"
            }
        }
        $finalLog = Get-Content -LiteralPath (Join-Path $jobDir ($stem + '.log')) -Raw
        if ($finalLog -match 'Missing character:|undefined references|undefined citations|(?:Reference|Citation) .+ undefined|Overfull |Label\(s\) may have changed') {
            throw "$($job.Language) has unresolved text, citations, references or layout overflow. No PDF was replaced; diagnostics: $buildDir"
        }
    } finally {
        Pop-Location
    }
}

# An edit or a new review during compilation invalidates these snapshots.
& $pythonPath $checker @checkerArgs
if ($LASTEXITCODE -ne 0 -or (Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash -ne $manifestHash -or (Get-FileHash -LiteralPath $bibliographyPath -Algorithm SHA256).Hash -ne $bibliographyHash) {
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

# Publish only after all selected sources succeed; roll back on copy failure.
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
    throw "Publication failed; prior selected PDFs restored. Diagnostics: $buildDir"
}
Write-Output "Published reviewed PDF(s): $($jobs.Pdf -join ', ')"
Write-Output "Review: $($approved.reviewed_by), $($approved.reviewed_at_utc)"
Write-Output "Intermediate files: $buildDir"
