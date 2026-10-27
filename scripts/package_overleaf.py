"""Build the reviewed English theory note as a portable Overleaf ZIP.

The English source and its shared BibTeX database are copied byte for byte.
This command never contacts Overleaf or changes cloud files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from check_note_sync import read_manifest, source_info, uncomment

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "note/main.tex"
REVIEW = ROOT / "note/english_review.json"
BIBLIOGRAPHY = ROOT / "note/references.bib"
README = """# Wasserstein robust variable selection: theory note

Compile `main.tex` using **pdfLaTeX** and BibTeX (TeX Live 2025 or later).
This package contains only the current English theory note. Experimental
design and code are maintained separately.

`main.tex` and `references.bib` exactly match the reviewed canonical files.
The source uses the newtx text and mathematics fonts included in TeX Live;
no proprietary fonts or external figures are needed. `latexmkrc` selects
pdfLaTeX; latexmk runs BibTeX when required. For a manual build, run
`pdflatex main`, `bibtex main`, then `pdflatex main` twice.

The packaging manifest is saved beside the ZIP, outside the uploaded
manuscript. Packaging does not upload or publish files.
"""
LATEXMKRC = (
    "$pdf_mode = 1;\n"
    "$pdflatex = 'pdflatex %O -interaction=nonstopmode -halt-on-error -no-shell-escape %S';\n"
    "$bibtex_use = 2;\n"
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reviewed_source() -> tuple[bytes, bytes, dict]:
    """Require a complete, current English review before packaging English."""
    english, structure, _ = source_info(SOURCE)
    review = read_manifest(REVIEW, languages=("english",))
    if english != review["sources"]["english"]:
        raise ValueError("Stale English review")
    if structure["counts"] != review["structure_counts"]:
        raise ValueError("The review does not cover the current full document")
    source = SOURCE.read_bytes()
    bibliography = BIBLIOGRAPHY.read_bytes()
    if review.get("bibliography") != {
        "path": BIBLIOGRAPHY.name,
        "sha256": digest(bibliography),
    }:
        raise ValueError("The shared bibliography differs from its reviewed version")
    if digest(source) != english["sha256"]:
        raise RuntimeError("Source changed during review validation")
    return source, bibliography, review


def portable_source(source: bytes) -> bytes:
    """Validate the pdfLaTeX source without rewriting its preamble or body."""
    text = uncomment(source.decode("utf-8"))
    preamble, separator, body = text.partition(r"\begin{document}")
    if not separator:
        raise ValueError("Missing document body")
    if re.search(r"\\(?:setmainfont|setmathfont|XeTeX\w*)\b", preamble) or re.search(
        r"\\usepackage(?:\[[^\]]*\])?\{[^}]*\b(?:fontspec|unicode-math)\b", preamble
    ):
        raise ValueError("The canonical English preamble still requires XeLaTeX")
    if not re.search(r"\\bibliography\s*\{references\}", body):
        raise ValueError("The English source must use the shared references.bib")
    return source


def compile_snapshot(folder: Path) -> dict:
    engine, bibliography_engine = shutil.which("pdflatex"), shutil.which("bibtex")
    if not engine or not bibliography_engine:
        raise RuntimeError("pdfLaTeX or BibTeX is unavailable")
    latex_command = [
        engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-no-shell-escape",
        "main.tex",
    ]
    steps = [
        ("pdflatex-1", latex_command),
        ("bibtex", [bibliography_engine, "main"]),
        ("pdflatex-2", latex_command),
        ("pdflatex-3", latex_command),
    ]
    for name, command in steps:
        completed = subprocess.run(
            command,
            cwd=folder,
            capture_output=True,
            timeout=90,
            check=False,
        )
        (folder / f"{name}.txt").write_bytes(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"{name} failed; diagnostics: {folder}")
    log = (folder / "main.log").read_text(encoding="utf-8", errors="replace")
    failures = (
        "Missing character:",
        "undefined references",
        "undefined citations",
        "Overfull ",
        "Label(s) may have changed",
    )
    if any(message in log for message in failures) or re.search(
        r"(?:Reference|Citation) .* undefined", log
    ):
        raise RuntimeError(f"Unresolved text, references or layout overflow; diagnostics: {folder}")
    return {
        "success": True,
        "passes": 3,
        "bibliography_passes": 1,
        "pdf_sha256": digest((folder / "main.pdf").read_bytes()),
        "pages": int(re.search(r"Output written on .*?\((\d+) pages?(?:,|\))", log).group(1)),
        "overfull_boxes": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Optional ZIP destination; default is unique system TEMP"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Build the snapshot with pdfLaTeX and BibTeX"
    )
    args = parser.parse_args()
    source, bibliography, review = reviewed_source()
    review_hash = digest(REVIEW.read_bytes())
    folder = Path(tempfile.mkdtemp(prefix="robusthw-theory-overleaf-"))
    payload = {
        "main.tex": portable_source(source),
        "references.bib": bibliography,
        "README.md": README.encode(),
        "latexmkrc": LATEXMKRC.encode(),
    }
    for name, data in payload.items():
        (folder / name).write_bytes(data)
    compilation = compile_snapshot(folder) if args.verify else None
    if (
        SOURCE.read_bytes() != source
        or BIBLIOGRAPHY.read_bytes() != bibliography
        or digest(REVIEW.read_bytes()) != review_hash
    ):
        raise RuntimeError("Canonical source, bibliography or review changed; no package published")
    archive = folder / "robusthw_theory_overleaf.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, data in sorted(payload.items()):
            item = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o644 << 16
            zipped.writestr(item, data)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "english_theory_only",
        "main_document": "main.tex",
        "canonical_source": "note/main.tex",
        "canonical_sha256": digest(source),
        "bibliography_sha256": digest(bibliography),
        "english_review_sha256": review_hash,
        "reviewed_by": review["reviewed_by"],
        "body_unchanged": True,
        "compiler": "pdfLaTeX",
        "bibliography_engine": "BibTeX",
        "open_fonts": True,
        "files": {name: digest(data) for name, data in sorted(payload.items())},
        "zip_sha256": digest(archive.read_bytes()),
        "compilation": compilation,
        "remote_uploaded": False,
    }
    manifest_path = archive.with_suffix(".manifest.json")
    manifest_path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, args.output)
        shutil.copy2(manifest_path, args.output.with_suffix(".manifest.json"))
    print(
        json.dumps(
            {
                "zip": str(args.output or archive),
                "temp": str(folder),
                "verified": bool(compilation),
                "pages": (compilation or {}).get("pages"),
            }
        )
    )


if __name__ == "__main__":
    main()
