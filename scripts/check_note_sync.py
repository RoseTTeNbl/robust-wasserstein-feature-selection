"""Check approved English/Chinese theory-note sources using only the stdlib.

--record attests that a reviewer checked the complete translation and formulas.
Structural checks support that review; they are not machine translation or a
proof of semantic equivalence. PDF currency requires a successful paired build.
"""

import argparse
import bisect
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MATH_ENVS = (
    "equation",
    "align",
    "alignat",
    "flalign",
    "gather",
    "multline",
    "displaymath",
)
NESTED_MATH_ENVS = (
    "aligned",
    "alignedat",
    "gathered",
    "split",
    "cases",
    "array",
    "matrix",
    "pmatrix",
    "bmatrix",
    "vmatrix",
    "Vmatrix",
)
ALGORITHM_STEPS = (
    "Require",
    "Ensure",
    "State",
    "Statex",
    "If",
    "ElsIf",
    "Else",
    "EndIf",
    "For",
    "ForAll",
    "EndFor",
    "While",
    "EndWhile",
    "Repeat",
    "Until",
    "Return",
    "Procedure",
    "EndProcedure",
    "Function",
    "EndFunction",
)
TEXT_COMMANDS = ("text", "textrm", "textnormal", "textsf", "textit", "textbf", "mbox")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def uncomment(source):
    """Ignore TeX comments, respecting escaped percent signs."""
    lines = []
    for line in source.splitlines():
        for i, character in enumerate(line):
            if character != "%":
                continue
            backslashes, j = 0, i - 1
            while j >= 0 and line[j] == "\\":
                backslashes += 1
                j -= 1
            if backslashes % 2 == 0:
                line = line[:i]
                break
        lines.append(line)
    return "\n".join(lines)


def argument_end(source, start):
    """Return the end of a balanced brace argument."""
    depth, i = 1, start + 1
    while i < len(source):
        if source[i] == "\\":
            i += 2
            continue
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("Unbalanced brace argument in a mathematical expression")


def normalized_math(expression):
    """Allow translated natural-language math annotations, not changed symbols."""
    pattern = re.compile(r"\\(?:" + "|".join(TEXT_COMMANDS) + r")\s*\{")
    parts, position = [], 0
    for_match = pattern.search(expression, position)
    while for_match is not None:
        parts.append(expression[position : for_match.start()])
        end = argument_end(expression, for_match.end() - 1)
        parts.append(r"\text{ANNOTATION}")
        position = end
        for_match = pattern.search(expression, position)
    parts.append(expression[position:])
    return re.sub(r"\s+", "", "".join(parts))


def mathematical_expressions(body):
    environments = "|".join(MATH_ENVS)
    pattern = re.compile(
        r"\\begin\{((?:" + environments + r")\*?)\}(.*?)\\end\{\1\}"
        r"|\\\[(.*?)\\\]|\\\((.*?)\\\)"
        r"|(?<!\\)\$\$(.*?)(?<!\\)\$\$|(?<!\\)\$(.*?)(?<!\\)\$",
        re.DOTALL,
    )
    separators = re.compile(r"\n[ \t]*\n|(?=\\(?:" + "|".join(ALGORITHM_STEPS) + r")\b)")
    boundaries = sorted({0, *(match.end() for match in separators.finditer(body))})
    inline_blocks = [Counter() for _ in boundaries]
    displays = []
    count = 0
    for match in pattern.finditer(body):
        if match.group(1):
            kind, content = match.group(1), match.group(2)
        else:
            group = next(i for i in range(3, 7) if match.group(i) is not None)
            kind = {
                3: "bracket-display",
                4: "paren-inline",
                5: "dollar-display",
                6: "dollar-inline",
            }[group]
            content = match.group(group)
        expression = (kind, normalized_math(content))
        if kind in ("paren-inline", "dollar-inline"):
            block = bisect.bisect_right(boundaries, match.start()) - 1
            inline_blocks[block][expression] += 1
        else:
            displays.append(expression)
        count += 1
    return dict(
        display=displays,
        inline_blocks=[
            sorted((kind, value, n) for (kind, value), n in block.items())
            for block in inline_blocks
        ],
    ), count


def structure(source):
    source = uncomment(source)
    if source.count(r"\begin{document}") != 1 or source.count(r"\end{document}") != 1:
        raise ValueError("Each note must contain exactly one complete document")
    preamble, remaining = source.split(r"\begin{document}", 1)
    body = remaining.split(r"\end{document}", 1)[0]
    if re.search(r"\\(?:input|include)\s*[{ ]", body):
        raise ValueError(
            "The sync check requires the complete note body in each source, without external TeX input"
        )
    body = re.sub(
        r"\\begin\{(verbatim\*?|lstlisting|minted)\}.*?\\end\{\1\}", "", body, flags=re.DOTALL
    )
    declarations = re.findall(r"\\newtheorem\*?\{([^{}]+)\}", preamble)
    semantic_envs = set(
        MATH_ENVS
        + NESTED_MATH_ENVS
        + (
            "abstract",
            "algorithm",
            "algorithmic",
            "proof",
            "thebibliography",
        )
        + tuple(declarations)
    )
    pattern = re.compile(
        r"\\(section|subsection|subsubsection)(\*)?\s*(?:\[[^\]]*\])?\s*\{"
        r"|\\(appendix)\b"
        r"|\\(label)\s*\{([^{}]+)\}"
        r"|\\(begin|end)\s*\{([^{}]+)\}"
        r"|\\(" + "|".join(ALGORITHM_STEPS) + r")\b"
        r"|\\(eqref|ref|autoref|cref|Cref)\s*\{([^{}]+)\}"
        r"|\\(cite\w*)\s*(?:\[[^\]]*\]\s*)*\{([^{}]+)\}"
        r"|\\(bibitem)\s*(?:\[[^\]]*\])?\s*\{([^{}]+)\}",
    )
    tokens, labels, counts, stack = [], [], Counter(), []
    appendix = False
    main_sections = appendix_sections = 0
    for match in pattern.finditer(body):
        if match.group(1):
            kind = match.group(1)
            tokens.append([kind, bool(match.group(2)), "appendix" if appendix else "main"])
            counts[("appendix_" if appendix else "main_") + kind] += 1
            if kind == "section":
                if appendix:
                    appendix_sections += 1
                else:
                    main_sections += 1
        elif match.group(3):
            appendix = True
            tokens.append(["appendix"])
            counts["appendix"] += 1
        elif match.group(4):
            labels.append(match.group(5))
            tokens.append(["label", match.group(5)])
        elif match.group(6):
            action, environment = match.group(6), match.group(7)
            if environment.rstrip("*") not in semantic_envs:
                continue
            tokens.append([action, environment])
            if action == "begin":
                counts["environment:" + environment] += 1
                stack.append(environment)
            elif not stack or stack.pop() != environment:
                raise ValueError(f"Unbalanced semantic environment: {environment}")
        elif match.group(8):
            tokens.append(["algorithm_step", match.group(8)])
            counts["algorithm_step:" + match.group(8)] += 1
        elif match.group(9):
            tokens.append([match.group(9), match.group(10)])
        elif match.group(11):
            tokens.append(["cite", [x.strip() for x in match.group(12).split(",")]])
        else:
            tokens.append(["bibitem", match.group(14)])
    if stack:
        raise ValueError("A semantic environment is not closed")
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate mathematical/structural label")
    if counts["appendix"] != 1 or not main_sections or not appendix_sections:
        raise ValueError(
            "The complete note must contain both main sections and one appendix boundary"
        )
    if counts["environment:abstract"] != 1:
        raise ValueError("The complete note must include one abstract")
    counts["labels"] = len(labels)
    formulas, formula_count = mathematical_expressions(body)
    counts["math_expressions"] = formula_count
    signature = dict(
        tokens=tokens,
        counts=dict(sorted(counts.items())),
        theorem_environments=sorted(set(declarations)),
        numbering=re.findall(r"\\numberwithin\{([^{}]+)\}\{([^{}]+)\}", preamble),
        math=formulas,
    )
    return signature, body


def source_info(path):
    data = path.read_bytes()
    signature, body = structure(data.decode("utf-8-sig"))
    encoded = json.dumps(
        signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return (
        dict(path=path.name, sha256=sha256_bytes(data), structure_sha256=sha256_bytes(encoded)),
        signature,
        body,
    )


def check_pair(english, chinese):
    en, en_structure, _ = source_info(english)
    zh, zh_structure, zh_body = source_info(chinese)
    if not re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", zh_body):
        raise ValueError("The Chinese document contains no Chinese prose")
    if en_structure != zh_structure:
        for field in ("counts", "theorem_environments", "numbering", "tokens", "math"):
            if en_structure[field] == zh_structure[field]:
                continue
            if field == "math":
                for part in ("display", "inline_blocks"):
                    left, right = en_structure[field][part], zh_structure[field][part]
                    if left == right:
                        continue
                    position = next(
                        (i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
                        min(len(left), len(right)),
                    )
                    raise ValueError(
                        f"English/Chinese mathematical {part} differ at item {position + 1}; formulas must remain in corresponding paragraphs or algorithm steps"
                    )
            if field == "tokens":
                left, right = en_structure[field], zh_structure[field]
                position = next(
                    (i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
                    min(len(left), len(right)),
                )
                raise ValueError(
                    f"English/Chinese {field} differ at item {position + 1}; compare the complete main text and appendices"
                )
            raise ValueError(
                f"English/Chinese {field} differ: {en_structure[field]} versus {zh_structure[field]}"
            )
    return en, zh, en_structure


def read_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("reviewed_by"):
        raise ValueError("Sync manifest has no supported schema or review attestation")
    if not all(language in manifest.get("sources", {}) for language in ("english", "chinese")):
        raise ValueError("Sync manifest must record both reviewed source versions")
    return manifest


def bibliography_info(english, chinese):
    """Fingerprint the shared database used by both complete TeX sources."""
    databases = []
    for source in (english, chinese):
        active = uncomment(source.read_text(encoding="utf-8-sig"))
        names = re.findall(r"\\bibliography\{([^{}]+)\}", active)
        if not names:
            databases.append(None)
            continue
        if names != ["references"]:
            raise ValueError("The theory pair must use one shared references.bib")
        path = source.parent / "references.bib"
        databases.append(dict(path=path.name, sha256=sha256_bytes(path.read_bytes())))
    if databases[0] != databases[1]:
        raise ValueError("English and Chinese must use the same bibliography")
    return databases[0]


def main(argv=None):
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english", type=Path, default=project / "note/main.tex")
    parser.add_argument("--chinese", type=Path, default=project / "note/main_zh.tex")
    parser.add_argument("--manifest", type=Path, default=project / "note/bilingual_sync.json")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Attest completed full bilingual review and record the approved hashes",
    )
    parser.add_argument("--reviewed-by", help="Reviewer identity required for --record")
    parser.add_argument(
        "--review-note", default="", help="Optional concise record of the completed review"
    )
    args = parser.parse_args(argv)
    try:
        en, zh, signature = check_pair(args.english, args.chinese)
        bibliography = bibliography_info(args.english, args.chinese)
        previous = read_manifest(args.manifest) if args.manifest.exists() else None
        if args.record:
            if not args.reviewed_by or not args.reviewed_by.strip():
                raise ValueError(
                    "--record requires --reviewed-by after complete semantic and formula review"
                )
            if previous:
                old = previous["sources"]
                if (
                    en["sha256"] != old["english"]["sha256"]
                    and zh["sha256"] == old["chinese"]["sha256"]
                ):
                    raise ValueError(
                        "English changed but Chinese was not updated; complete synchronization before recording review"
                    )
            manifest = dict(
                schema_version=1,
                reviewed_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                reviewed_by=args.reviewed_by.strip(),
                review_note=args.review_note,
                sources=dict(english=en, chinese=zh),
                structure_counts=signature["counts"],
                bibliography=bibliography,
                review_scope="Complete title, abstract, main text, appendices, formulas, theorem/proof and algorithm content",
            )
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
            temporary.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            temporary.replace(args.manifest)
            status = "review_recorded"
        else:
            if previous is None:
                raise ValueError(
                    "No reviewed bilingual sync manifest; finish the complete translation and explicitly record its review"
                )
            for language, actual in (("english", en), ("chinese", zh)):
                approved = previous["sources"][language]
                if any(
                    actual[field] != approved.get(field)
                    for field in ("path", "sha256", "structure_sha256")
                ):
                    raise ValueError(
                        f"{language.capitalize()} source differs from its reviewed version; update both notes and review before building"
                    )
            if signature["counts"] != previous.get("structure_counts"):
                raise ValueError(
                    "The approved manifest does not cover the complete current document structure"
                )
            if bibliography != previous.get("bibliography"):
                raise ValueError(
                    "The shared bibliography differs from its reviewed version; review its citations before building"
                )
            status = "in_sync"
        print(
            json.dumps(
                dict(
                    status=status,
                    english=en,
                    chinese=zh,
                    counts=signature["counts"],
                    note="Approved source synchronization does not certify PDF currency; use the paired build.",
                ),
                ensure_ascii=False,
            )
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Bilingual note check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
