"""Package the four reviewed manuscripts for one portable XeLaTeX project.

No remote service is contacted. Figures are copied only when referenced by an
entry. Optional builds run on immutable snapshots in a unique system TEMP.
"""
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

ROOT = Path(__file__).resolve().parents[1]
ENTRIES = {
    "main.tex": "note/main.tex",
    "main_zh.tex": "note/main_zh.tex",
    "experiment_report.tex": "reports/experiment_report.tex",
    "experiment_report_zh.tex": "reports/experiment_report_zh.tex",
}
FONT_FLAG = r"\def\RobustHWForceFallbackFonts{1}" + "\n"
LATIN = (r"{texgyretermes-regular.otf}"
         r"[BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf,"
         r"BoldItalicFont=texgyretermes-bolditalic.otf]")
CHINESE = (r"{FandolSong-Regular.otf}"
           r"[BoldFont=FandolHei-Regular.otf,ItalicFont=FandolSong-Regular.otf,"
           r"BoldItalicFont=FandolHei-Regular.otf]")
FONT_COMMANDS = {
    r"\setmainfont{Times New Roman}": ("Times New Roman", r"\setmainfont" + LATIN),
    r"\setmainfont{SimSun}[BoldFont=SimHei,ItalicFont=SimSun,BoldItalicFont=SimHei]":
        ("SimSun", r"\setmainfont" + CHINESE),
    r"\newfontfamily\LatinText{Times New Roman}":
        ("Times New Roman", r"\newfontfamily\LatinText" + LATIN),
    r"\setmonofont{Consolas}[Scale=0.9]":
        ("Consolas", r"\setmonofont{texgyrecursor-regular.otf}[Scale=0.9]"),
    r"\setmathfont{Cambria Math}":
        ("Cambria Math", r"\setmathfont{texgyretermes-math.otf}"),
}
README = """# Robust feature selection: four manuscript entries

Choose **XeLaTeX** as the compiler. **TeX Live 2025** is recommended for the
current cloud deployment. The default main document is `main.tex`.
Change the main-document setting to compile another entry:

| Entry | Manuscript |
| --- | --- |
| main.tex | English theory note |
| main_zh.tex | Complete Chinese theory note |
| experiment_report.tex | English experimental report |
| experiment_report_zh.tex | Complete Chinese experimental report |

Each entry is standalone. Its bibliography, if present, is inside the TeX file.
The shared figures/ directory contains only graphics referenced by these entries.
Local equivalent command: latexmk -xelatex main.tex (or another entry).

The entries default to TeX Live's TeX Gyre Termes, Cursor and Termes Math OTF files,
and Fandol OTF files for Chinese. The first line defines
\\RobustHWForceFallbackFonts, which avoids repeated system-font probes on the
cloud service. Delete that first line to restore the native-font preference;
the existing branches then use available Windows fonts or the open-font fallback.
No proprietary font files are included. Native XeTeX Chinese line breaking is
preserved. The body, mathematical statements and numerical results are copied
from their canonical manuscripts without editorial changes.

The unicode-math/mathtools compatibility warnings about bracket and colon
commands are retained in the build logs. The manuscripts do not use the affected
commands; no package import order or mathematical expression is changed.

The external packaging manifest records canonical hashes, packaged-file hashes,
body identity checks and any optional local compilation. It is provenance rather
than a manuscript entry. Uploading or replacing an Overleaf project is a separate
explicit operation; the packaging command itself does not publish anything.
"""
LATEXMKRC = "$pdf_mode = 5;\n$xelatex = 'xelatex %O -interaction=nonstopmode -halt-on-error %S';\n"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def portable_fonts(text):
    """Change only existing preamble font commands; leave the body untouched."""
    head, separator, body = text.partition(r"\begin{document}")
    if not separator:
        raise ValueError("Entry has no document body")
    changed = []
    for original, (name, fallback) in FONT_COMMANDS.items():
        if original not in head:
            continue
        if head.count(original) != 1:
            raise ValueError("Ambiguous font command: " + original)
        choice = (r"\ifdefined\RobustHWForceFallbackFonts" + "\n  " + fallback
                  + "\n" + r"\else" + "\n  " + r"\IfFontExistsTF{" + name + "}\n    {"
                  + original + "}\n    {" + fallback + "}\n" + r"\fi")
        head = head.replace(original, choice)
        changed.append(name)
    if not {"Consolas", "Cambria Math"}.issubset(changed):
        raise ValueError("Expected monospace/math font setup was not found")
    return head + separator + body, changed


def graphics_for(text, source):
    """Resolve literal includegraphics and the project's ResearchFigure macro."""
    without_comments = re.sub(r"(?<!\\)%[^\r\n]*", "", text)
    references = set(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",
                                without_comments))
    references = {name for name in references if "#" not in name}
    references.update("figures/" + stem + ".pdf" for stem in
                      re.findall(r"\\ResearchFigure\{([^}]+)\}", without_comments))
    replacements, images = {}, {}
    for reference in sorted(references):
        candidates = [source.parent / reference]
        if not Path(reference).suffix:
            candidates = [Path(str(candidates[0]) + suffix)
                          for suffix in (".pdf", ".png", ".jpg", ".jpeg")]
        found = next((path.resolve() for path in candidates if path.is_file()), None)
        if found is None or not found.is_relative_to(ROOT):
            raise ValueError("Missing or external figure: " + reference)
        target = "figures/" + found.name
        replacements[reference] = target
        images[target] = (found, found.read_bytes())
    # Paths may be normalized; names used by ResearchFigure are already shared.
    for old, new in replacements.items():
        if old != new:
            text = text.replace("{" + old + "}", "{" + new + "}")
    return text, replacements, images


def build_payload():
    payload, provenance, inputs = {}, {}, {}
    for entry, relative in ENTRIES.items():
        source = ROOT / relative
        original = source.read_bytes()
        text = original.decode("utf-8")
        portable, fonts = portable_fonts(text)
        portable, paths, images = graphics_for(portable, source)
        expected = text
        for old, new in paths.items():
            expected = expected.replace("{" + old + "}", "{" + new + "}")
        body = portable.partition(r"\begin{document}")[2]
        if body != expected.partition(r"\begin{document}")[2]:
            raise ValueError("Body content changed during portability conversion")
        payload[entry] = (FONT_FLAG + portable).encode("utf-8")
        inputs[relative] = digest(original)
        provenance[entry] = dict(canonical=relative, canonical_sha256=digest(original),
                                 packaged_sha256=digest(payload[entry]), font_choices=fonts,
                                 open_fonts_default=True,
                                 body_sha256=digest(body.encode("utf-8")), body_unchanged=True,
                                 graphics=list(images))
        for target, (path, data) in images.items():
            if target in payload and payload[target] != data:
                raise ValueError("Different figures share a packaged basename: " + target)
            payload[target] = data
            inputs[path.relative_to(ROOT).as_posix()] = digest(data)
    payload["README.md"] = README.encode("utf-8")
    payload["latexmkrc"] = LATEXMKRC.encode("utf-8")
    return payload, provenance, inputs


def verify_entries(folder, force_fallback):
    engine = shutil.which("xelatex")
    if engine is None:
        raise RuntimeError("XeLaTeX is unavailable; sources remain packaged in TEMP")
    results = {}
    for entry in ENTRIES:
        target = entry
        if force_fallback:
            target = "verify_" + entry
            (folder / target).write_text(r"\def\RobustHWForceFallbackFonts{1}" + "\n"
                                         + r"\input{" + entry + "}\n", encoding="utf-8")
        job = Path(target).stem
        for run in range(1, 4):
            completed = subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error",
                                        "-no-shell-escape", target], cwd=folder,
                                       capture_output=True, timeout=90)
            (folder / f"{job}.pass-{run}.txt").write_bytes(completed.stdout + completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"Compilation failed: {entry}; diagnostics: {folder}")
        log = (folder / (job + ".log")).read_text(encoding="utf-8", errors="replace")
        if "Missing character:" in log or re.search(r"(?:undefined references|Reference .* undefined)", log):
            raise RuntimeError(f"Unresolved text or references in {entry}; diagnostics: {folder}")
        results[entry] = dict(success=True, fallback_forced=True,
                              verification_wrapper=force_fallback, passes=3,
                              pdf_sha256=digest((folder / (job + ".pdf")).read_bytes()),
                              overfull_boxes=len(re.findall(r"Overfull \\[hv]box", log)),
                              package_warnings=[dict(package=package, summary=summary)
                                                for package, summary in re.findall(
                                                    r"^Package (\S+) Warning: ([^\r\n]+)",
                                                    log, re.MULTILINE)])
        print(json.dumps(dict(verified=entry, **results[entry])), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/overleaf/robusthw_overleaf.zip")
    parser.add_argument("--verify", action="store_true", help="Compile all four TEMP entry snapshots")
    parser.add_argument("--force-fallback", action="store_true",
                        help="Also force open fonts through a temporary verification wrapper")
    parser.add_argument("--check-only", action="store_true", help="Verify without publishing a ZIP")
    args = parser.parse_args()
    if args.force_fallback and not args.verify:
        parser.error("--force-fallback requires --verify")
    folder = Path(tempfile.mkdtemp(prefix="robusthw-overleaf-"))
    payload, entries, inputs = build_payload()
    for name, data in payload.items():
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    compiled = verify_entries(folder, args.force_fallback) if args.verify else {}
    if any(digest((ROOT / name).read_bytes()) != sha for name, sha in inputs.items()):
        raise RuntimeError("A canonical manuscript or figure changed; no ZIP was published")
    archive = folder / "robusthw_overleaf.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name in sorted(payload):
            item = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o644 << 16
            zipped.writestr(item, payload[name])
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), compiler="XeLaTeX",
                    main_document="main.tex", recommended_tex_live="2025",
                    font_policy="open fonts by default; remove first-line flag for native preference",
                    entries=entries, canonical_inputs=inputs,
                    packaged_files={name: digest(data) for name, data in sorted(payload.items())},
                    zip_sha256=digest(archive.read_bytes()), compilation=compiled,
                    verification_temp=str(folder), remote_uploaded=False)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.check_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, args.output)
        shutil.copy2(folder / "manifest.json", args.output.with_suffix(".manifest.json"))
    print(json.dumps(dict(output=None if args.check_only else str(args.output),
                         zip_sha256=manifest["zip_sha256"], files=len(payload),
                         temp=str(folder), verified=bool(compiled))), flush=True)


if __name__ == "__main__":
    main()
