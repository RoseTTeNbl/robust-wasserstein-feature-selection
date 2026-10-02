$ErrorActionPreference = 'Stop'
$buildDir = Join-Path ([System.IO.Path]::GetTempPath()) ('robusthw-note-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $buildDir | Out-Null
Push-Location -LiteralPath $PSScriptRoot
try {
    foreach ($pass in 1..2) {
        & xelatex -interaction=nonstopmode -halt-on-error "-output-directory=$buildDir" main.tex > (Join-Path $buildDir "pass-$pass.txt")
        if ($LASTEXITCODE -ne 0) {
            Get-Content -LiteralPath (Join-Path $buildDir "pass-$pass.txt") -Tail 35
            throw "XeLaTeX failed; diagnostics are in $buildDir"
        }
    }
    Copy-Item -LiteralPath (Join-Path $buildDir 'main.pdf') -Destination (Join-Path $PSScriptRoot 'robust_coordinate_selection.pdf')
    Write-Output "Built $(Join-Path $PSScriptRoot 'robust_coordinate_selection.pdf')"
    Write-Output "Intermediate files: $buildDir"
} finally {
    Pop-Location
}
