# Overleaf theory note

Current project: [Robust Feature Selection (English)](https://www.overleaf.com/project/6acc3aafcb36ee55cea8de47).
The existing [reading link](https://www.overleaf.com/read/tzfnycpfrqbh#a40c25) remains the same.

The prepared online update contains the **English theory note only**. Complete English
and Chinese sources, the English reading copy, and both PDFs are maintained in
`note/`. Experimental design and code are separate from the online manuscript.

## Canonical sources and package

`note/main.tex` is the English source; `note/main_zh.tex` is its complete Chinese
counterpart. Both must pass `scripts/check_note_sync.py` and `note/build.ps1`
before packaging. The current pair has seven propositions, all their proofs,
and 32 numbered equations. The proofs retain their assumptions and intermediate
steps while replacing repeated explanations with explicit mathematical bounds.

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
Before replacing the earlier project contents, download its source ZIP as a
backup. Upload the three package files and remove the obsolete experiment entry
and figures from this theory-only online project. Recompile and check the online
PDF and log; a successful local package build alone is not cloud verification.

The 18 October portable package compiled locally in three passes, with nine
pages and no missing characters, undefined references or overflow. Cloud upload
is pending the editable Overleaf browser session; the earlier reading link
currently shows the previous manuscript. It must not be cited as the current
revision until upload and compilation are verified.
