"""Report checks protect trial pairing, denominators, and generated TeX."""

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
SPEC = importlib.util.spec_from_file_location("report_builder", PROJECT / "scripts/build_report.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def trial(method="risk", seed=0, suite="statistical", value=.5):
    return {"suite": suite, "dataset": "mean", "dimension": 12, "budget": 3, "theta": .25,
                "method": method, "seed": seed, "power": value, "run_status": "complete",
                "selection_status": "optimal"}


def test_frozen_design_has_exact_requested_trial_inventory():
    expected, _ = REPORT.expected_trials(PROJECT)
    counts = {suite: sum(k[0] == suite for k in expected) for suite in REPORT.SUITES}
    assert counts == {"statistical": 960, "computational": 320, "real": 240, "sensitivity": 60, "ratio": 60}
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
    interval = {"power_sd": np.std([.25, .75], ddof=1), "power_lo": .25, "power_hi": .75}
    REPORT.check_summary(rows, [{**summary, **interval}])
    with pytest.raises(ValueError, match="summary interval"):
        REPORT.check_summary(rows, [{**summary, **interval, "power_hi": .9}])
    with pytest.raises(ValueError, match="summary SD"):
        REPORT.check_summary(rows, [{**summary, **interval, "power_sd": .9}])


def test_missing_metrics_are_dashes_not_zero_and_real_sd_is_distinct():
    assert REPORT.estimate({}, "power") == "---"
    assert REPORT.estimate({"power_mean": .4, "power_sd": .1}, "power", real=True) == r"0.400\(\pm\)0.100"
    assert "[" not in REPORT.estimate({"power_mean": .4, "power_sd": .1}, "power", real=True)


def test_generated_table_rejects_bad_row_length():
    with pytest.raises(ValueError, match="row length"):
        REPORT.report_table("Title", ["A", "B"], [["only one"]], label="tab:test")
    tex = REPORT.report_table("Title", ["A", "B"], [["x", "1"]], label="tab:test")
    assert r"\begin{tabularx}" in tex and r"\label{tab:test}" in tex
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
    text = REPORT.concise_records(evidence, "en")
    assert "Selection-only timing is unavailable for 1 records" in text
    assert evidence.summaries[0]["n_complete"] == 0


def test_software_line_uses_bundled_backend_not_independent_highspy(tmp_path, monkeypatch):
    failure_fixture(tmp_path, monkeypatch)
    environment = {"packages": {"scipy": "1.18.1", "highspy": "1.15.1",
                                 "cvxpy": "1.9.3", "clarabel": "0.11.1"},
                       "linear_integer_backend": {"highs_version": "1.12.0"}}
    (tmp_path / "environment.json").write_text(json.dumps(environment), encoding="utf-8")
    text = REPORT.concise_records(REPORT.Evidence(PROJECT, tmp_path, draft=True), "en")
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
    text = REPORT.statistical_report(evidence, "en")
    assert "0.300" in text and "0.700" in text
    assert "-40.0 percentage points" in text and "$n=2$" in text


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
    text = REPORT.reference_computation(evidence, "en")
    assert "Global risk & 1/2 & 10.00" in text
    assert "Triangle & 2/2" in text and "Box & 2/2" in text
    assert "3/4" in text


def test_head_bound_excess_is_paired_by_trial_not_by_group_mean():
    evidence = REPORT.Evidence.__new__(REPORT.Evidence)
    evidence.summaries = []
    evidence.configs = {}
    evidence.rows = [
        {**trial(seed=0), "head_robust_upper": .4, "robust_randomized_balanced_error": .3,
         "empirical_certificate_violation": 0},
        {**trial(seed=1), "head_robust_upper": .8, "robust_randomized_balanced_error": .2,
         "empirical_certificate_violation": 0},
        {**trial("greedy", seed=0), "head_robust_upper": .1,
         "robust_randomized_balanced_error": .99, "empirical_certificate_violation": 1},
    ]
    text = REPORT.classifier_report(evidence, "en")
    assert "1/2" in text
    assert "0.300" in text and "0.250" in text
    assert "not a certificate failure" in text


def test_native_wang_and_greedy_readouts_keep_their_different_targets():
    evidence = REPORT.Evidence.__new__(REPORT.Evidence)
    evidence.missing = set()
    evidence.rows = [trial("risk", value=.3), trial("mmd", value=.4)]
    evidence.summaries = [
        {**trial("risk"), "power_mean": .3, "power_n": 1, "recall_mean": .2,
         "recall_n": 1, "n_total": 1, "status_counts": '{"optimal": 1}'},
        {**trial("mmd"), "power_mean": .4, "power_n": 1, "recall_mean": .6, "recall_n": 1},
        {**trial("greedy"), "power_mean": .5},
        {**trial("wang"), "native_additive_power_mean": .8, "native_additive_null_rejection_mean": .05,
         "native_additive_power_n": 1, "power_mean": .6},
    ]
    text = REPORT.statistical_report(evidence, "en")
    assert "All 1/1 mean-shift global searches meet the optimization tolerance" in text
    assert "greedy power is 0.500" in text
    for row in evidence.rows:
        row["dataset"] = "variance"
    for row in evidence.summaries:
        row["dataset"] = "variance"
    native = REPORT.statistical_report(evidence, "en")
    assert "native power retains the signed additive weights" in native
    assert "0.600 & 0.800 & 0.050" in native


def test_radius_readout_describes_joint_refits_and_observed_nonmonotonicity():
    evidence = REPORT.Evidence.__new__(REPORT.Evidence)
    evidence.configs = {"sensitivity": {"radii": [.1, .25, .5]}}
    evidence.summaries = [
        {**trial(suite="sensitivity"), "theta": radius, "power_mean": power,
         "robust_randomized_balanced_error_mean": .4,
         "robust_worst_stress_balanced_error_mean": .5}
        for radius, power in [(0.1, .6), (.25, .3), (.5, .5)]
    ]
    evidence.rows = []
    text = REPORT.sensitivity_report(evidence, "en")
    assert all(value in text for value in ("0.600", "0.300", "0.500"))
    assert "Joint radius sensitivity" in text


def test_replay_inventory_does_not_include_later_unarchived_failures(tmp_path):
    evidence = REPORT.Evidence.__new__(REPORT.Evidence)
    evidence.root = evidence.project = tmp_path
    original = {**trial(), "selection": {"status": "master_failure", "seconds": .5}, "total_seconds": .6}
    folder = tmp_path / "remediation"
    folder.mkdir()
    (folder / "initial_failures.json").write_text(json.dumps([original]), encoding="utf-8")
    evidence.raw = {REPORT.key(original): {**original, "selection": {"status": "optimal", "seconds": 1.0}}}
    for seed in (1, 2):
        later = {**trial(seed=seed), "selection": {"status": "master_failure", "seconds": .2}}
        evidence.raw[REPORT.key(later)] = later
    evidence.rows = [{**r, "selection_status": r["selection"]["status"]} for r in evidence.raw.values()]
    text = REPORT.concise_records(evidence, "en")
    assert "1 original failures" in text
    assert "final collection has 2 failures" in text


def test_bilingual_templates_are_complete_symmetric_and_without_appendices():
    import re
    evidence = REPORT.Evidence(PROJECT, PROJECT / "results/research_20261003", draft=False)
    templates = [(PROJECT / "reports" / name).read_text(encoding="utf-8")
                 for name in ("report_template.tex", "report_template_zh.tex")]
    rendered = [REPORT.render(evidence, source, language=language)
                for source, language in zip(templates, ("en", "zh"), strict=True)]
    for template, source in zip(templates, rendered, strict=True):
        assert all(template.count("%%" + marker + "%%") == 1 for marker in REPORT.MARKERS)
        assert r"\begin{abstract}" in source
        assert r"\appendix" not in source
        for label in ("eq:permutation", "eq:recovery", "eq:errors"):
            assert source.count(r"\label{" + label + "}") == 1
        assert "resizebox" not in source and "scriptsize" not in source
        assert "tab:search-development" not in source and "tab:reference-ratio" not in source
        assert "tab:null" in source and "fig:prediction" in source
    for pattern in (r"\\label\{([^{}]+)\}", r"\\cite\{([^{}]+)\}"):
        assert re.findall(pattern, rendered[0]) == re.findall(pattern, rendered[1])
    assert "FDP" in rendered[1] and "NDP" in rendered[1] and "Wang" in rendered[1]
    assert r"\operatorname{NDP}=\frac{|T_*\setminus S|}{|T_*|}" in rendered[0]
    assert r"\operatorname{NDP}=\frac{|T_*\setminus S|}{|T_*|}" in rendered[1]


def test_new_bilingual_renderer_keeps_paired_outcomes_and_missing_times(tmp_path, monkeypatch):
    _row, _ = failure_fixture(tmp_path, monkeypatch)
    evidence = REPORT.Evidence(PROJECT, tmp_path, draft=True)
    en = REPORT.concise_records(evidence, "en")
    zh = REPORT.concise_records(evidence, "zh")
    assert "Selection-only timing is unavailable for 1 records" in en
    assert "1 条记录没有单独选择时间" in zh
    assert r"\label{tab:outcomes}" in en and r"\label{tab:outcomes}" in zh
    assert "1 & 0 & 0 & 1 & 0" in en and "1 & 0 & 0 & 1 & 0" in zh



def test_documented_permutation_tolerance_and_recovery_scope():
    for name in ("report_template.tex", "report_template_zh.tex"):
        source = (PROJECT / "reports" / name).read_text(encoding="utf-8")
        assert r"T_{\rm observed}-10^{-14}" in source
        assert r"\max\{1,|S|\}" in source
        assert "$|S|=|T_*|=3$" in source


def test_rendered_search_reports_cost_tradeoff_and_run_count():
    evidence = REPORT.SearchEvidence(PROJECT / "reports/verification/20261010/next_selection")
    english = REPORT.search_report(evidence, "en")
    chinese = REPORT.search_report(evidence, "zh")
    for source in (english, chinese):
        for number in ("598", "453", "24.2", "+4.3", "144/144", "15/0/1"):
            assert number in source
        assert not any(ord(c) < 32 and c not in "\n\r\t" for c in source)
    assert "unchanged GP cannot be credited" in english
    assert "总体提速" not in chinese


@pytest.mark.parametrize("canonical", ["reports/report_template.tex", "scripts/build_report.py"])
def test_identical_known_source_snapshot_has_portable_label(tmp_path, canonical):
    project = tmp_path / "project"
    target = project / canonical
    target.parent.mkdir(parents=True)
    target.write_bytes(b"same frozen source")
    snapshot = tmp_path / "immutable" / target.name
    snapshot.parent.mkdir()
    snapshot.write_bytes(target.read_bytes())
    label, digest = REPORT.input_fingerprint(snapshot, project, canonical)
    assert label == canonical
    assert digest == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    snapshot.write_bytes(b"different custom source with same filename")
    assert REPORT.input_fingerprint(snapshot, project, canonical)[0] == str(snapshot.resolve())
    assert REPORT.input_fingerprint(target, project)[0] == canonical


def test_unrecognized_external_source_is_not_rebased_by_filename(tmp_path):
    project = tmp_path / "project"
    target = project / "scripts/build_report.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"same")
    external = tmp_path / target.name
    external.write_bytes(b"same")
    assert REPORT.input_fingerprint(external, project)[0] == str(external.resolve())


def test_questions_precede_parameter_details_and_metrics_are_not_conflated():
    for name in ("report_template.tex", "report_template_zh.tex"):
        source = (PROJECT / "reports" / name).read_text(encoding="utf-8")
        positions = [source.index(r"\label{" + label + "}") for label in
                     ("sec:design", "sec:statistics", "sec:recovery", "sec:prediction",
                      "sec:real", "sec:search", "sec:parameters")]
        assert positions == sorted(positions)
        assert source.index("%%SETUP_TABLE%%") > positions[-1]
        assert "L0" in source and "L1" in source
        assert "2026" in source
    evidence = REPORT.Evidence(PROJECT, PROJECT / "results/research_20261003", draft=False)
    power = REPORT.statistical_report(evidence, "en")
    recovery = REPORT.recovery_report(evidence, "en")
    assert "known-signal power is 0.218" in power
    assert "0.957" in power and "0.138" in power
    assert "false alarms" in power and "Native power" in power
    assert "0.467" in recovery and "0.244" in recovery
    assert "0.367" in recovery and "0.289" in recovery
    real = REPORT.real_report(evidence, "en")
    assert "Robust AUC" not in real
    assert "40/40" in real and "certified optima" in real
