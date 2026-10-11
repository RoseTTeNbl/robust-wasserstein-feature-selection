# Overleaf project

The theory note and experimental report share one Overleaf project. They remain
four independent manuscripts; choose the main document to switch between them.
The published project is [selection hy testing](https://www.overleaf.com/project/6ab9dbd92cb870b10bf20b1c).
All four entries were compiled online on 2026-10-11; the current preview is the
Chinese experimental report. The downloaded cloud sources match all 16 packaged
file contents. The online editor normalized the two experimental TeX entries from
CRLF to LF; their text and mathematical content are identical.
Set the project compiler to **XeLaTeX**. **TeX Live 2025** is recommended for the
current cloud deployment; cloud compilation is checked separately from the local
build.

| Canonical source in this repository | Overleaf entry | Content |
| --- | --- | --- |
| `note/main.tex` | `main.tex` | English theory note; default entry |
| `note/main_zh.tex` | `main_zh.tex` | Complete Chinese theory note |
| `reports/experiment_report.tex` | `experiment_report.tex` | English experimental report |
| `reports/experiment_report_zh.tex` | `experiment_report_zh.tex` | Complete Chinese experimental report |

The repository sources remain authoritative. Update the complete English and
Chinese pair, finish the normal review and publication checks, then regenerate
the upload package. A cloud edit does not automatically update this repository.

## Rebuild the package

From the repository root, run:

```powershell
.venv-research/Scripts/python.exe scripts/package_overleaf.py --verify
```

This creates `reports/overleaf/robusthw_overleaf.zip` and
`reports/overleaf/robusthw_overleaf.manifest.json`. The ZIP contains the four
entries, a shared `figures/` directory, `README.md`, and `latexmkrc`. Figures are
collected from the current manuscripts' actual references, rather than from the
whole local figures directory. No experiment records, source-control files,
compilation caches, or proprietary fonts are included.

The command compiles all four entries three times in a unique system TEMP
directory using the packaged entries' default open fonts. The optional
`--force-fallback` flag also sets the same font switch in a temporary verification
wrapper. Compilation rejects missing characters and unresolved references, and
records overfull-box counts. Layout review and compilation in Overleaf remain
part of publication checks.

For a temporary check that does not replace the saved ZIP, add `--check-only`.
For packaging without local compilation, omit `--verify`.
The printed TEMP path contains the exact upload files and compilation logs.

## Font and content portability

The conversion adds a first-line font switch, changes existing font declarations
in the preamble and, when
necessary, normalizes referenced graphics to `figures/<filename>`. It does not
edit the manuscripts' mathematical content, prose, results, or canonical files.
Each body's identity is checked after the permitted figure-path normalization.

The entries default to TeX Gyre Termes, TeX Gyre Cursor, TeX Gyre Termes Math, and
Fandol Chinese fonts by their OTF filenames. This skips repeated system-font
probes in the cloud preamble. The first line of each entry is:

```tex
\def\RobustHWForceFallbackFonts{1}
```

Delete that line to prefer Times New Roman, Consolas, Cambria Math, and SimSun
when they are available; the existing branches retain open fonts as the fallback.
Native XeTeX Chinese line breaking is preserved. The open fonts belong to the TeX
distribution; they are not uploaded.

The build retains the two `unicode-math`/`mathtools` compatibility warnings about
bracket and colon commands. The four manuscripts do not use the affected
commands. Their summaries appear in the manifest, and the complete messages
remain in the TEMP logs; the package order and mathematical expressions are
unchanged.
The Chinese entries also retain `microtype` notices about unknown character slots
in the open font's protrusion configuration. All four cloud compilations report
zero errors. Local compilation reports zero missing characters, unresolved
references, and overfull boxes. Full publication checks are recorded in
`reports/verification/20261011/readability/overleaf_publication_audit.json`.

The manifest records every canonical input hash, every packaged file hash, body
identity checks, compilation results, and the ZIP hash. Inputs are checked again
before publication, so a concurrent manuscript or figure edit stops the build
instead of publishing a mixed snapshot.

## Upload and compile

For a new project, import the ZIP. For the existing project, upload the four TeX
entries, `README.md`, and `latexmkrc` at the project root, then upload the referenced
figures into `figures/`. Leave unrelated existing files intact. Temporary
`verify_*.tex` wrappers and build logs are not upload files.

Select `main.tex` as the default main document and compile with XeLaTeX under the
selected TeX Live version. Select
each of the other three entries in turn to verify the complete bilingual theory
and report pair. The packaging script itself performs no upload and changes no
cloud sharing settings.
