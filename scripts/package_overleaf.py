"""Build the reviewed English theory note as a portable Overleaf ZIP.

Only preamble font declarations change. The document body is byte-identical
to note/main.tex. This command never contacts Overleaf or changes cloud files.
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

from check_note_sync import check_pair, read_manifest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "note/main.tex"
REVIEW = ROOT / "note/bilingual_sync.json"
FONT_REPLACEMENTS = {
    r"\setmainfont{Times New Roman}": (
        r"\setmainfont{texgyretermes-regular.otf}"
        r"[BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf,"
        r"BoldItalicFont=texgyretermes-bolditalic.otf]"
    ),
    r"\setmathfont{Cambria Math}": r"\setmathfont{texgyretermes-math.otf}",
}
README = """# Wasserstein robust variable selection: theory note

Compile `main.tex` using **XeLaTeX** and TeX Live 2025 or later.
This package contains only the current English theory note. The complete
Chinese counterpart and experimental design remain in the local repository.

The document body exactly matches the reviewed canonical source. Only the
preamble replaces Windows fonts with TeX Gyre Termes and Termes Math, which
are included in TeX Live. No proprietary fonts or external figures are needed.
`latexmkrc` selects XeLaTeX. The packaging manifest is saved beside the ZIP,
outside the uploaded manuscript. Packaging does not upload or publish files.
"""
LATEXMKRC = "$pdf_mode = 5;\n$xelatex = 'xelatex %O -interaction=nonstopmode -halt-on-error %S';\n"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reviewed_source() -> tuple[bytes, dict]:
    """Require a complete, current bilingual review before packaging English."""
    english, chinese, structure = check_pair(SOURCE, ROOT / "note/main_zh.tex")
    review = read_manifest(REVIEW)
    for language, actual in (("english", english), ("chinese", chinese)):
        if actual != review["sources"][language]:
            raise ValueError(f"Stale bilingual review: {language}")
    if structure["counts"] != review["structure_counts"]:
        raise ValueError("The review does not cover the current full document")
    source = SOURCE.read_bytes()
    if digest(source) != english["sha256"]:
        raise RuntimeError("Source changed during review validation")
    return source, review


def portable_source(source: bytes) -> bytes:
    text = source.decode("utf-8")
    preamble, separator, body = text.partition(r"\begin{document}")
    if not separator:
        raise ValueError("Missing document body")
    for old, new in FONT_REPLACEMENTS.items():
        if preamble.count(old) != 1:
            raise ValueError(f"Expected one font declaration: {old}")
        preamble = preamble.replace(old, new)
    portable = (preamble + separator + body).encode("utf-8")
    if portable.split(b"\\begin{document}", 1)[1] != source.split(b"\\begin{document}", 1)[1]:
        raise AssertionError("Portability conversion changed the document body")
    return portable


def compile_snapshot(folder: Path) -> dict:
    engine = shutil.which("xelatex")
    if not engine:
        raise RuntimeError("XeLaTeX is unavailable")
    for run in range(1, 4):
        completed = subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape", "main.tex"],
            cwd=folder,
            capture_output=True,
            timeout=90,
            check=False,
        )
        (folder / f"pass-{run}.txt").write_bytes(completed.stdout + completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"Compilation failed; diagnostics: {folder}")
    log = (folder / "main.log").read_text(encoding="utf-8", errors="replace")
    failures = ("Missing character:", "undefined references", "Overfull ")
    if any(message in log for message in failures) or re.search(r"Reference .* undefined", log):
        raise RuntimeError(f"Unresolved text, references or layout overflow; diagnostics: {folder}")
    return {
        "success": True,
        "passes": 3,
        "pdf_sha256": digest((folder / "main.pdf").read_bytes()),
        "pages": int(re.search(r"Output written on .*?\((\d+) pages?\)", log).group(1)),
        "overfull_boxes": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Optional ZIP destination; default is unique system TEMP"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Compile the portable snapshot three times"
    )
    args = parser.parse_args()
    source, review = reviewed_source()
    review_hash = digest(REVIEW.read_bytes())
    folder = Path(tempfile.mkdtemp(prefix="robusthw-theory-overleaf-"))
    payload = {
        "main.tex": portable_source(source),
        "README.md": README.encode(),
        "latexmkrc": LATEXMKRC.encode(),
    }
    for name, data in payload.items():
        (folder / name).write_bytes(data)
    compilation = compile_snapshot(folder) if args.verify else None
    if SOURCE.read_bytes() != source or digest(REVIEW.read_bytes()) != review_hash:
        raise RuntimeError("Canonical source or review changed; no package published")
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
        "bilingual_review_sha256": review_hash,
        "reviewed_by": review["reviewed_by"],
        "body_unchanged": True,
        "compiler": "XeLaTeX",
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
