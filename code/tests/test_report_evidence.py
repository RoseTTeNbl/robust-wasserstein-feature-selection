"""Independent evidence-reader regressions: frozen files, weights, repeat costs."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT / "scripts/report_evidence.py"
SPEC = importlib.util.spec_from_file_location("report_evidence", MODULE_PATH)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def result(visited=3, seconds=1., status="optimal"):
    return {"status": status, "visited": visited, "oracle_calls": visited,
                "seconds": seconds, "diagnostics": {"master_calls": 1, "master_seconds": .2},
                "measured": {"continuous_calls": 2 * visited, "second_stage_calls": 0,
                          "master_measurements": [{"seconds": .2, "nodes": 1}]}}


def fixture(root):
    save(root / "protocol.json", {"confirmation": {"cases": 1}})
    data = {"x0": [[0., 1.]], "x1": [[1., 0.]], "q0": [1.], "q1": [1.]}
    digest = hashlib.sha256(np.r_[0., 1., 1., 0., 1., 1.].tobytes()).hexdigest()
    record = {"case": {"id": "one", "data_sha256": digest}, "data": data,
                  "methods": {"baseline": result(status="master_failure")},
                  "checks": {"baseline": {"optimal": False, "binary_only": True}}}
    row = dict(case="one", method="baseline", **REPORT.measurements(record["methods"]["baseline"]),
               all_checks=False)
    stage = root / "development/phase1/development"
    raw = stage / "raw/one.json"
    save(raw, record)
    summary = stage / "summary.json"
    save(summary, {"rows": [row], "totals": REPORT.totals([row])})
    phase = stage.parent
    save(root / "development/evidence_manifest.json", {"phase1": {"sha256": {
        path.relative_to(phase).as_posix(): REPORT.file_hash(path) for path in (raw, summary)}}})
    return data, record


def test_complete_existing_development_is_auditable():
    project = PROJECT / "reports/verification/20261010/next_selection"
    evidence = REPORT.SearchEvidence(project, draft=True)
    assert [(s["name"], len(s["raw"])) for s in evidence.development] == [
        ("phase1", 26), ("phase2", 26), ("phase3", 26), ("phase4", 26)]
    assert evidence.development[-1]["totals"]["curvature"]["statuses"] == {
        "master_failure": 1, "optimal": 25}


def test_failed_development_remains_in_denominator(tmp_path):
    fixture(tmp_path)
    evidence = REPORT.SearchEvidence(tmp_path, draft=True)
    total = evidence.development[0]["totals"]["baseline"]
    assert total["cases"] == 1 and total["statuses"] == {"master_failure": 1}
    assert total["completed_evaluations"]["total"] == 3
    assert not total["all_pass"]


def test_frozen_raw_change_is_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "development/phase1/development/raw/one.json"
    record = REPORT.read_json(path)
    record["methods"]["baseline"]["visited"] = 0
    save(path, record)
    with pytest.raises(ValueError, match="Frozen development input changed"):
        REPORT.SearchEvidence(tmp_path, draft=True)


def test_data_digest_includes_mass_weights(tmp_path):
    _, record = fixture(tmp_path)
    evidence = REPORT.SearchEvidence(tmp_path, draft=True)
    record["data"]["q0"] = [.5]
    with pytest.raises(ValueError, match="Generated data changed"):
        evidence._data(record)


def test_final_requires_declared_confirmation_count(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError, match="needs 1 confirmation cases"):
        REPORT.SearchEvidence(tmp_path)


def test_three_timing_repeats_are_not_three_independent_datasets(tmp_path):
    data, development = fixture(tmp_path)
    confirmation = tmp_path / "current/confirmation"
    protocol = {"methods": ["baseline", "curvature", "master"], "timing_repeats": 3,
                    "data_hash_fields": ["x0", "x1", "q0", "q1"], "cases": [development["case"]], "sources": {}}
    save(confirmation / "protocol.json", protocol)
    repeated = []
    for index, seconds in enumerate((1., 7., 3.)):
        methods = {"baseline": result(visited=6, seconds=seconds),
                   "curvature": result(visited=4, seconds=2. * seconds),
                   "master": result(visited=7, seconds=.5 * seconds)}
        repeated.append({"repeat": index, "method_order": list(methods), "methods": methods,
                             "checks": {method: {"optimal": True} for method in methods}})
    raw = confirmation / "raw/one.json"
    save(raw, {"case": development["case"], "data": data, "repeats": repeated})
    summary_rows = [dict(case="one", repeat=r["repeat"], method=method, **REPORT.measurements(result), all_checks=True)
                    for r in repeated for method, result in r["methods"].items()]
    save(confirmation / "summary.json", {"rows": summary_rows, "case_count": 1, "timing_repeats": 3, "all_pass": True})
    save(confirmation / "seal.json", {"sources": {}, "files": {
        path.relative_to(confirmation).as_posix(): REPORT.file_hash(path)
        for path in (confirmation / "protocol.json", raw, confirmation / "summary.json")}})
    evidence = REPORT.SearchEvidence(tmp_path)
    assert evidence.confirmation_cases == 1
    group = evidence.confirmation["totals"]["curvature"]
    assert group["cases"] == 1
    assert group["completed_evaluations"]["total"] == 4
    assert group["seconds"]["total"] == 6.
    assert evidence.confirmation["run_totals"]["curvature"]["cases"] == 3
    assert evidence.confirmation["run_totals"]["curvature"]["seconds"]["total"] == 22.
    assert len(evidence.paired()) == 1
