"""Report checks protect trial pairing, denominators, and generated TeX."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("report_builder", PROJECT / "scripts/build_report.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def trial(method="risk", seed=0, suite="statistical", value=.5):
    return dict(suite=suite, dataset="mean", dimension=12, budget=3, theta=.25,
                method=method, seed=seed, power=value, run_status="complete",
                selection_status="optimal")


def test_frozen_design_has_exact_requested_trial_inventory():
    expected, _ = REPORT.expected_trials(PROJECT)
    counts = {suite: sum(k[0] == suite for k in expected) for suite in REPORT.SUITES}
    assert counts == dict(statistical=960, computational=320, real=240, sensitivity=60, ratio=60)
    assert len(expected) == 1640


def test_pairing_keeps_common_seeds_and_the_supplement_separate():
    rows = [trial("risk", 0, "ratio", 5), trial("risk_ratio", 0, "ratio", 3),
            trial("risk", 1, "ratio", 100), trial("risk_ratio", 2, "ratio", 100),
            trial("risk", 0, "computational", 999)]
    n, interval = REPORT.paired(rows, "ratio", "mean", "power", a="risk_ratio", b="risk")
    assert n == 1
    assert interval == (-2, -2, -2)


def test_stale_summary_or_effective_count_is_rejected():
    rows = [trial(value=.25), trial(seed=1, value=.75)]
    summary = {**trial(), "n_total": 2, "n_complete": 2, "status_counts": '{"optimal": 2}',
               "power_mean": .5, "power_n": 2}
    REPORT.check_summary(rows, [summary])
    with pytest.raises(ValueError, match="effective metric count"):
        REPORT.check_summary(rows, [{**summary, "power_n": 1}])
    with pytest.raises(ValueError, match="mean"):
        REPORT.check_summary(rows, [{**summary, "power_mean": .6}])
    with pytest.raises(ValueError, match="status counts"):
        REPORT.check_summary(rows, [{**summary, "status_counts": '{"time_limit": 2}'}])


def test_missing_metrics_are_dashes_not_zero_and_real_sd_is_distinct():
    assert REPORT.estimate({}, "power") == "---"
    assert REPORT.estimate({"power_mean": .4, "power_sd": .1}, "power", real=True) == r"0.400\(\pm\)0.100"
    assert "[" not in REPORT.estimate({"power_mean": .4, "power_sd": .1}, "power", real=True)


def test_generated_table_rejects_bad_row_length():
    with pytest.raises(ValueError, match="row length"):
        REPORT.longtable("Title", ["A", "B"], [["only one"]])
    tex = REPORT.longtable("Title", ["A", "B"], [["x", "1"]])
    assert r"\endfirsthead" in tex and r"\endlastfoot" in tex
    assert "resizebox" not in tex


def test_text_escaping_is_single_pass():
    assert REPORT.escaped(r"x\_y 50% &") == r"x\textbackslash{}\_y 50\% \&"


def test_interval_brackets_are_not_parsed_as_linebreak_spacing():
    cell = REPORT.estimate({"power_mean": .4, "power_lo": .3, "power_hi": .5}, "power")
    paired_cell = REPORT.interval_cell((.4, .3, .5))
    assert cell == paired_cell
    assert r"\\[" not in cell
    assert r"\\{[" in cell
    assert cell == r"\shortstack{0.400\\{[0.300, 0.500]}}"


def test_incomplete_final_build_is_rejected_and_draft_is_marked(tmp_path):
    (tmp_path / "per_trial.csv").write_text(
        "suite,dataset,dimension,budget,theta,method,seed,run_status,selection_status\n",
        encoding="utf-8")
    (tmp_path / "summary.json").write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="Final build rejected"):
        REPORT.Evidence(PROJECT, tmp_path, draft=False)
    evidence = REPORT.Evidence(PROJECT, tmp_path, draft=True)
    template = "\n".join("%%" + m + "%%" for m in REPORT.MARKERS)
    source = REPORT.render(evidence, template)
    assert "DRAFT SNAPSHOT" in source
    assert "1640" in source
    assert all("%%" + m + "%%" not in source for m in REPORT.MARKERS)


def failure_fixture(tmp_path, monkeypatch):
    row = {k: v for k, v in trial().items() if k != "power"}
    row.update(run_status="failure", selection_status="failure", config_sha256="frozen",
               record_file="statistical/records/failed.json")
    original = {k: v for k, v in row.items() if k not in ("suite", "record_file", "selection_status")}
    original.update(error_type="RuntimeError", error="solver raised", total_seconds=2.5)
    path = tmp_path / row["record_file"]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(original), encoding="utf-8")
    with (tmp_path / "per_trial.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    summary = {**row, "n_total": 1, "n_complete": 0, "status_counts": '{"failure": 1}'}
    (tmp_path / "summary.json").write_text(json.dumps([summary]), encoding="utf-8")
    monkeypatch.setattr(REPORT, "expected_trials", lambda _: ({REPORT.key(row): "frozen"}, {}))
    return row, original


def test_actual_runner_exception_without_selection_is_reportable(tmp_path, monkeypatch):
    row, original = failure_fixture(tmp_path, monkeypatch)
    evidence = REPORT.Evidence(PROJECT, tmp_path, draft=True)
    assert evidence.raw[REPORT.key(row)] == original
    text = REPORT.overview(evidence)
    assert "Selection-only timing is unavailable for 1 records" in text
    assert evidence.summaries[0]["n_complete"] == 0


def test_software_line_uses_bundled_backend_not_independent_highspy(tmp_path, monkeypatch):
    failure_fixture(tmp_path, monkeypatch)
    environment = dict(packages={"scipy": "1.18.1", "highspy": "1.15.1",
                                 "cvxpy": "1.9.3", "clarabel": "0.11.1"},
                       linear_integer_backend={"highs_version": "1.12.0"})
    (tmp_path / "environment.json").write_text(json.dumps(environment), encoding="utf-8")
    text = REPORT.overview(REPORT.Evidence(PROJECT, tmp_path, draft=True))
    assert "SciPy-backed HiGHS 1.12.0" in text
    assert "1.15.1" not in text
    assert "CVXPY 1.9.3" in text and "Clarabel 0.11.1" in text


def test_final_requires_published_initial_failure_provenance(tmp_path, monkeypatch):
    failure_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="requires published remediation"):
        REPORT.Evidence(PROJECT, tmp_path, draft=False)
    provenance = tmp_path / "remediation/initial_failures.json"
    provenance.parent.mkdir()
    provenance.write_text("[]", encoding="utf-8")
    REPORT.Evidence(PROJECT, tmp_path, draft=False)


def test_stale_raw_metric_is_rejected_even_when_csv_and_summary_agree():
    row = {**trial(value=.4), "config_sha256": "same"}
    record = {k: v for k, v in row.items() if k not in ("suite", "selection_status", "power")}
    record.update(selection={"status": "optimal"}, metrics={"power": .8})
    with pytest.raises(ValueError, match="numeric value.*power"):
        REPORT.check_raw_record(row, record)
    with pytest.raises(ValueError, match="identity/configuration"):
        REPORT.check_raw_record(row, {**record, "seed": 999})


def test_mechanism_readout_reports_observed_probabilities_and_paired_points():
    evidence = REPORT.Evidence.__new__(REPORT.Evidence)
    evidence.missing = set()
    evidence.rows = [trial("risk", 0, value=.2), trial("risk", 1, value=.4),
                     trial("mmd", 0, value=.6), trial("mmd", 1, value=.8)]
    evidence.summaries = [
        {**trial("risk"), "power_mean": .3, "power_n": 2, "recall_mean": .2, "recall_n": 2},
        {**trial("mmd"), "power_mean": .7, "power_n": 2, "recall_mean": .8, "recall_n": 2},
    ]
    text = REPORT.mechanism_readout(evidence, "mean")
    assert "0.300" in text and "0.700" in text
    assert "-40.0 percentage points" in text and "2 paired seeds" in text
    assert "0.200" in text and "0.800" in text


def test_computational_readout_uses_all_outcomes_and_observed_timeouts():
    evidence = REPORT.Evidence.__new__(REPORT.Evidence)
    evidence.summaries = [
        {**trial(method, suite="computational"), "n_total": 2,
         "selection_seconds_mean": seconds, "status_counts": statuses}
        for method, seconds, statuses in [
            ("risk", 10, '{"optimal": 1, "time_limit": 1}'),
            ("triangle", 2, '{"optimal": 2}'),
            ("box", 1, '{"optimal": 2}'),
        ]
    ]
    evidence.rows = [{**trial(seed=i, suite="computational"), "dimension": 20,
                      "selection_status": "time_limit" if i < 3 else "optimal"} for i in range(4)]
    text = REPORT.computational_readout(evidence, "computational", "mean")
    assert "Global risk solves 1/2" in text and "charged time 10.00 s" in text
    assert "Triangle solves 2/2" in text and "Box solves 2/2" in text
    assert "3/4" in text and r"75.0\%" in text
