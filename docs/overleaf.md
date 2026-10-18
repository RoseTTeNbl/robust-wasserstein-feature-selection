# Overleaf theory note

Target project in the new account:
[Variable Selection for Wasserstein Robust Hypothesis Testing](https://www.overleaf.com/project/6ad4c10310f26580d0959bbc).

The prepared online update contains the **English theory note only**. Complete English
and Chinese sources, the English reading copy, and both PDFs are maintained in
`note/`. Experimental design and code are separate from the online manuscript.

## Canonical sources and package

`note/main.tex` is the English source; `note/main_zh.tex` is its complete Chinese
counterpart. Both must pass `scripts/check_note_sync.py` and `note/build.ps1`
before packaging. The current pair has two propositions and 28 numbered
equations. Appendix B restates the finite and extended primal models, derives
the normalized Lagrangian dual, and proves the affine representation and exact
contact of its supporting hyperplanes. The algorithm section retains only
the upper/lower bounds and stopping rule. The convergence proposition,
its auxiliary definitions and Appendix C have been removed.

From the repository root:

```powershell
python scripts/package_overleaf.py --verify
```

The command prints the ZIP and diagnostics paths in a unique system TEMP folder.
Use `--output <path.zip>` to copy the package and its manifest elsewhere. It never
uploads or changes the online project.

The ZIP contains only `main.tex`, `README.md` and `latexmkrc`. The source body is
byte-identical to the reviewed English manuscript. Its preamble uses TeX Gyre
Termes and Termes Math, distributed with TeX Live, in place of Windows fonts.
No figures, fonts, experimental outputs or compilation caches are uploaded.
The manifest records source/review hashes, exact ZIP contents and any local build.

## Online revision

Use **XeLaTeX**, TeX Live 2025 or later, and `main.tex` as the main document.
Upload the three package files to this theory-only online project. Recompile
and check the online PDF and log; a successful local package build alone is
not cloud verification.

The 18 October portable package compiled locally in three passes, with no
missing characters, undefined references or overflow. Cloud upload
is pending a tool-accessible editable browser session. The user-supplied new
project has not been inspected or repaired online because browser URL
verification stopped automation. Its online revision must not be described
as current until upload and compilation are verified.
