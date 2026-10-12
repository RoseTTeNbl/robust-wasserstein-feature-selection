# Overleaf project

The online package contains only the **English theory note and English experimental
report**, with their referenced English figures. The complete Chinese sources and
PDFs remain in the local repository for reading and review; they are not uploaded.
Choose the main document to switch between the two English manuscripts.

Current English project: [Robust Feature Selection (English)](https://www.overleaf.com/project/6acc3aafcb36ee55cea8de47).
The reading link is [English manuscripts](https://www.overleaf.com/read/tzfnycpfrqbh#a40c25).
The 12 October source upload contains the deterministic global-master method,
without GP or Appendix E, and the revised experimental report. Its publication
checks are recorded in `reports/verification/20261012/overleaf_publication_audit.json`.

The English manuscripts use a separate project. The earlier bilingual project is
retained as a historical backup; it is not replaced or stripped of Chinese files.
Use **XeLaTeX** with **TeX Live 2025**, which was tested successfully for the
project's font configuration. Cloud compilation of the current manuscript version
is checked separately from the local build.

| Canonical source in this repository | Overleaf entry | Content |
| --- | --- | --- |
| `note/main.tex` | `main.tex` | English theory note; default entry |
| `reports/experiment_report.tex` | `experiment_report.tex` | English experimental report |

The repository sources remain authoritative. Update and review the complete local
English/Chinese pair before regenerating the English upload package. A cloud edit
does not automatically update this repository.

## Rebuild the package

After the revised manuscripts have passed their normal review and publication
checks, run this command from the repository root:

```powershell
.venv-research/Scripts/python.exe scripts/package_overleaf.py --verify
```

This creates `reports/overleaf/robusthw_overleaf.zip` and
`reports/overleaf/robusthw_overleaf.manifest.json`. The ZIP contains `main.tex`,
`experiment_report.tex`, `README.md`, `latexmkrc`, and a shared `figures/` directory.
The current report references five English figure PDFs. Graphics are collected
from the manuscripts' actual references. Chinese entries and figures, experiment
records, compilation caches, and proprietary fonts are excluded.

Both entries are compiled three times in a unique system TEMP directory using
their default open fonts. The optional `--force-fallback` flag also sets the same
font switch in a temporary verification wrapper. Compilation rejects missing
characters and unresolved references, and records overfull-box counts and package
warnings. Layout review and compilation in Overleaf remain publication checks.

Add `--check-only` for a temporary check that does not replace the saved ZIP. Omit
`--verify` to package without compiling. The printed TEMP path contains the exact
upload files and compilation logs.

## Font and content portability

The conversion adds a first-line font switch, changes existing font declarations
in the preamble, and normalizes graphics paths to `figures/<filename>` when needed.
It does not edit the canonical sources, mathematical content, prose, or results.
Each body's identity is checked after the permitted figure-path normalization.

The entries default to TeX Gyre Termes, TeX Gyre Cursor, and TeX Gyre Termes Math by
their OTF filenames. This skips repeated system-font probes in the cloud preamble.
Each entry starts with:

```tex
\def\RobustHWForceFallbackFonts{1}
```

Delete that line to prefer available Times New Roman, Consolas, and Cambria Math;
the existing branches retain open fonts as the fallback. These open fonts belong
to the TeX distribution and are not uploaded.

The build retains the `unicode-math`/`mathtools` compatibility warnings about
bracket and colon commands. The manuscripts do not use the affected commands.
Their summaries appear in the manifest; complete messages remain in the TEMP
logs. Package order and mathematical expressions are unchanged.

The manifest records the two canonical English source hashes, referenced figure
hashes, every packaged file hash, body identity checks, compilation results, and
the ZIP hash. Inputs are checked again before publication, so a concurrent source
or figure edit stops the build rather than publishing a mixed snapshot.

## Upload and compile

Import the ZIP into a new project, or upload its four root files and five English
figure PDFs into the existing project's matching paths. The online project uses
only the English entries; retained local Chinese files are outside this package.
Temporary `verify_*.tex` wrappers and build logs are not upload files.

Select `main.tex` as the default main document, compile with XeLaTeX under TeX Live
2025, then select and compile `experiment_report.tex`. The packaging script
performs no upload and changes no cloud sharing settings.

The earlier four-entry bilingual project,
[selection hy testing](https://www.overleaf.com/project/6ab9dbd92cb870b10bf20b1c),
was verified on 2026-10-11 in
`reports/verification/20261011/readability/overleaf_publication_audit.json`. That
record documents the previous scope; it is not verification of the current
English-only package or later manuscript revisions.
