"""Load sealed subset-search evidence without mixing development and confirmation."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MEASUREMENTS = ("completed_evaluations", "physical_calls", "continuous_calls",
                "masters", "master_seconds", "seconds")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_number(saved, actual):
    if isinstance(actual, (str, bool)) or actual is None:
        return saved == actual
    return math.isclose(float(saved), float(actual), rel_tol=1e-9, abs_tol=1e-8)


def checked_path(root, relative):
    path = (root / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evidence path leaves its declared directory")
    return path


def measurements(result):
    measured, diagnostics = result.get("measured", {}), result.get("diagnostics", {})
    masters = measured.get("master_measurements")
    return {"status": result["status"], "completed_evaluations": result.get("visited"),
                "physical_calls": result.get("oracle_calls"), "seconds": result.get("seconds"),
                "continuous_calls": measured.get("continuous_calls"),
                "second_stage_calls": measured.get("second_stage_calls"),
                "masters": diagnostics.get("master_calls", sum(r["phase"] == "master"
                                                            for r in result.get("trace", []))),
                "master_seconds": (sum(r["seconds"] for r in masters) if masters is not None else
                                diagnostics.get("master_seconds")),
                "master_nodes": (sum(r.get("nodes", 0) or 0 for r in masters)
                              if masters is not None else None),
                "lower": result.get("lower"), "upper": result.get("upper"), "gap": result.get("gap")}


def totals(rows):
    answer = {}
    for method in dict.fromkeys(r["method"] for r in rows):
        group = [r for r in rows if r["method"] == method]
        result = {"cases": len(group), "statuses": dict(Counter(r["status"] for r in group)),
                      "all_pass": all(r["all_checks"] for r in group)}
        for field in (*MEASUREMENTS, "master_nodes", "second_stage_calls"):
            values = [float(r[field]) for r in group if r.get(field) is not None]
            result[field] = {"total": sum(values), "median": float(np.median(values))} if values else None
        answer[method] = result
    return answer


class SearchEvidence:
    """Numbers come from frozen raw runs; summary files are checked, never trusted."""

    def __init__(self, root, draft=False):
        self.root, self.draft = Path(root).resolve(), draft
        self.input_files = []
        self.protocol = self._read(self.root / "protocol.json")
        self._check_development_seal()
        self.development = [self._development(path) for path in sorted(
            (self.root / "development").glob("phase*/development/summary.json"))]
        integrated = self.root / "current/development"
        self.integration = self._repeated_stage(integrated) if (integrated / "summary.json").exists() else None
        self.confirmation = None
        confirmation = self.root / "current/confirmation"
        if (confirmation / "summary.json").exists():
            self.confirmation = self._repeated_stage(confirmation)
        self.confirmation_cases = len(self.confirmation["raw"]) if self.confirmation else 0
        self.input_files = list(dict.fromkeys(self.input_files))
        planned = int(self.protocol["confirmation"]["cases"])
        if not draft and self.confirmation_cases != planned:
            raise ValueError(f"Final report needs {planned} confirmation cases; found {self.confirmation_cases}")

    def _read(self, path):
        self.input_files.append(path)
        return read_json(path)

    def _check_development_seal(self):
        manifest = self._read(self.root / "development/evidence_manifest.json")
        for phase, record in manifest.items():
            directory = self.root / "development" / phase
            for relative, expected in record["sha256"].items():
                path = checked_path(directory, relative)
                if file_hash(path) != expected:
                    raise ValueError(f"Frozen development input changed: {phase}/{relative}")
                self.input_files.append(path)

    def _data(self, record, fields=("x0", "x1", "q0", "q1")):
        arrays = [np.asarray(record["data"][name], dtype=float).ravel() for name in fields]
        digest = hashlib.sha256(np.concatenate(arrays).tobytes()).hexdigest()
        if digest != record["case"]["data_sha256"]:
            raise ValueError(f"Generated data changed: {record['case']['id']}")

    def _development(self, path):
        saved = self._read(path)
        raw, rows = {}, []
        for filename in sorted((path.parent / "raw").glob("*.json")):
            record = self._read(filename)
            self._data(record)
            case = record["case"]["id"]
            if case in raw:
                raise ValueError("Duplicate development case")
            raw[case] = record
            for method, result in record["methods"].items():
                row = dict(case=case, method=method, **measurements(result),
                           all_checks=all(record["checks"][method].values()))
                rows.append(row)
        actual = {(r["case"], r["method"]): r for r in rows}
        expected = {(r["case"], r["method"]): r for r in saved["rows"]}
        if len(expected) != len(saved["rows"]) or actual.keys() != expected.keys():
            raise ValueError(f"Development summary has missing or duplicate rows: {path}")
        for identity, row in expected.items():
            for field in set(row) & set(actual[identity]):
                if not same_number(row[field], actual[identity][field]):
                    raise ValueError(f"Development raw value differs: {identity}, {field}")
        recomputed = totals(rows)
        for method, result in saved["totals"].items():
            computed = recomputed[method]
            for field in ("cases", "statuses", "all_pass"):
                if result[field] != computed[field]:
                    raise ValueError(f"Development total differs: {method}, {field}")
            for field in (*MEASUREMENTS, "master_nodes", "second_stage_calls"):
                if field not in result:
                    continue
                for aggregate, value in result[field].items():
                    if not same_number(value, computed[field][aggregate]):
                        raise ValueError(f"Development total differs: {method}, {field}")
        return {"name": path.parents[1].name, "raw": raw, "rows": rows, "totals": recomputed}

    def _repeated_stage(self, directory):
        protocol = self._read(directory / "protocol.json")
        seal = self._read(directory / "seal.json")
        sealed = {relative.replace("\\", "/") for relative in seal["files"]}
        required = {"protocol.json", "summary.json"} | {path.relative_to(directory).as_posix()
                    for path in (directory / "raw").glob("*.json")}
        if not required <= sealed:
            raise ValueError("Confirmation inputs are missing from their seal")
        for relative, expected in seal["files"].items():
            path = checked_path(directory, relative)
            if file_hash(path) != expected:
                raise ValueError(f"Frozen confirmation input changed: {relative}")
            self.input_files.append(path)
        if seal["sources"] != protocol["sources"]:
            raise ValueError("Protocol and seal source versions differ")
        for relative, expected in protocol["sources"].items():
            path = checked_path(directory / "source", relative)
            if file_hash(path) != expected:
                raise ValueError(f"Frozen source snapshot changed: {relative}")
            self.input_files.append(path)
        raw, rows, repeat_rows = {}, [], []
        methods, repeat_count = protocol["methods"], protocol["timing_repeats"]
        for path in sorted((directory / "raw").glob("*.json")):
            record = self._read(path)
            self._data(record, protocol["data_hash_fields"])
            case = record["case"]["id"]
            if case in raw:
                raise ValueError("Duplicate confirmation case")
            raw[case] = record
            repeats = record["repeats"]
            if len(repeats) != repeat_count or {r["repeat"] for r in repeats} != set(range(repeat_count)):
                raise ValueError(f"Missing confirmation repetitions: {case}")
            for repeated in repeats:
                if set(repeated["methods"]) != set(methods):
                    raise ValueError(f"Confirmation methods differ: {case}")
                if len(repeated["method_order"]) != len(methods) or set(repeated["method_order"]) != set(methods):
                    raise ValueError(f"Invalid method order: {case}")
                for method, result in repeated["methods"].items():
                    repeat_rows.append(dict(case=case, repeat=repeated["repeat"], method=method,
                                            **measurements(result),
                                            all_checks=all(repeated["checks"][method].values())))
            for method in methods:
                runs = [r for r in repeat_rows if r["case"] == case and r["method"] == method]
                row = {"case": case, "method": method, "all_checks": all(r["all_checks"] for r in runs),
                           "status": "optimal" if all(r["status"] == "optimal" for r in runs) else "incomplete",
                           "statuses": dict(Counter(r["status"] for r in runs))}
                for field in MEASUREMENTS:
                    values = [r[field] for r in runs if r[field] is not None]
                    row[field] = float(np.median(values)) if len(values) == repeat_count else None
                queries = [r["completed_evaluations"] for r in runs
                           if r["completed_evaluations"] is not None]
                row["query_range"] = [min(queries), max(queries)] if queries else None
                rows.append(row)
        declared = {case["id"]: case for case in protocol["cases"]}
        actual_cases = {identity: record["case"] for identity, record in raw.items()}
        if len(declared) != len(protocol["cases"]) or actual_cases != declared:
            raise ValueError("Recorded cases differ from the frozen stage protocol")
        summary = self._read(directory / "summary.json")
        expected_rows = {(r["case"], r["repeat"], r["method"]): r for r in summary["rows"]}
        actual_rows = {(r["case"], r["repeat"], r["method"]): r for r in repeat_rows}
        if len(expected_rows) != len(summary["rows"]) or expected_rows.keys() != actual_rows.keys():
            raise ValueError("Repeated-stage summary has duplicate or missing rows")
        for identity, saved in expected_rows.items():
            for field in set(saved) & set(actual_rows[identity]):
                if not same_number(saved[field], actual_rows[identity][field]):
                    raise ValueError(f"Repeated-stage raw value differs: {identity}, {field}")
        if summary["case_count"] != len(raw) or summary["timing_repeats"] != repeat_count:
            raise ValueError("Repeated-stage summary counts differ")
        if bool(summary["all_pass"]) != all(r["all_checks"] for r in repeat_rows):
            raise ValueError("Repeated-stage summary checks differ")
        return {"name": directory.name, "raw": raw, "rows": rows, "repeat_rows": repeat_rows,
                    "totals": totals(rows), "run_totals": totals(repeat_rows), "protocol": protocol}

    def paired(self, selected="curvature", baseline="baseline"):
        by_case = defaultdict(dict)
        for row in self.confirmation["rows"]:
            by_case[row["case"]][row["method"]] = row
        return [(case, group[selected], group[baseline]) for case, group in sorted(by_case.items())]
