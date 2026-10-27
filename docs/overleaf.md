# Overleaf theory note

Target project in the new account:
[Variable Selection for Wasserstein Robust Hypothesis Testing](https://www.overleaf.com/project/6ad4c10310f26580d0959bbc).

The prepared online update contains the **English theory note only**. English is
the maintained source. Experimental design and code are separate from the online manuscript.

## Canonical sources and package

`note/main.tex` is the English source. Complete its review with
`scripts/check_note_sync.py --record --reviewed-by reviewer` and build with
`note/build.ps1`. Packaging checks `note/english_review.json` and does not read
the Chinese source or bilingual review. `note/references.bib` is the bibliography; it includes
the robust hypothesis-testing and mixed-integer optimization papers used in
the derivation. The reviewed bibliography is copied into each immutable build
snapshot and checked again before the selected PDF outputs are replaced.

From the repository root:

```powershell
python scripts/package_overleaf.py --verify
```

The command prints the ZIP and diagnostics paths in a unique system TEMP folder.
Use `--output <path.zip>` to copy the package and its manifest elsewhere. It never
uploads or changes the online project.

The ZIP contains only `main.tex`, `references.bib`, `README.md` and `latexmkrc`.
The source and bibliography are byte-identical to their reviewed canonical files.
The manuscript uses the newtx text and mathematics fonts distributed with TeX Live.
No figures, fonts, experimental outputs or compilation caches are uploaded.
The manifest records source/review hashes, exact ZIP contents and any local build.

## Online revision

Use **pdfLaTeX**, TeX Live 2025 or later, and `main.tex` as the main document.
Upload all four package files to this theory-only online project. Recompile
and check the online PDF and log; a successful local package build alone is
not cloud verification.

`latexmkrc` selects pdfLaTeX and permits BibTeX. A manual build is:

```text
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Cloud upload is pending a tool-accessible editable browser session. The
user-supplied project has not been verified online because browser URL
verification stopped automation. Its online revision must not be described
as current until upload and compilation are verified. Consult the newly
generated package manifest for the current local compilation result.
