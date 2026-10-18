"""Aggregation units and failure denominators, using fabricated raw records."""

import csv
from dataclasses import replace

import numpy as np
import pytest

from robusthw.core import Cut, Problem
from robusthw.experiments import analysis
from robusthw.experiments.storage import fingerprint, read_json, write_json
from robusthw.experiments.verification import transport_value, verify_small
from robusthw.search import select


def _method(method="global", budget=4, ber=0.2, runtime=1.0, repeat=0, status="optimal", **extra):
    return {
        "method": method,
        "budget": budget,
        "role": "comparison",
        "radius_factor": 1,
        "initialization": "marginal_mmd",
        "timing_repeat": repeat,
        "status": status,
        "runtime": runtime,
        "upper": 0.5,
        "lower": 0.49 if method == "global" else None,
        "gap": 0.01 if method == "global" else None,
        "fit_count": 2,
        "master_count": 1,
        "master_seconds": 0.1,
        "oracle_seconds": 0.9,
        "metrics": {"ber_prob": ber},
        "trace": [],
        **extra,
    }


def _record(
    root,
    identity,
    results,
    *,
    kind="synthetic",
    D=50,
    n=20,
    s=4,
    status="completed",
    filename=None,
    **extra,
):
    record = {
        "schema_version": 1,
        "manifest_hash": "fixture-protocol",
        "spec": {
            "independent_data_id": identity,
            "kind": kind,
            "model": "mean" if kind != "har" else "har",
            "D": D,
            "n": n,
            "s": s,
            "seed": 1,
            "budgets": [4],
            "metadata": {"suite": "main" if kind == "synthetic" else "public"},
        },
        "status": status,
        "data": {},
        "radius_selection": {},
        "results": results,
        **extra,
    }
    record["record_hash"] = fingerprint(record)
    write_json(root / "records" / (filename or f"{identity}.json"), record)


def _summary(root, monkeypatch):
    monkeypatch.setattr(analysis, "_figures", lambda *args: [])
    result = analysis.summarize(root)
    return read_json(result["summary"]), result


def _group(summary, method="global"):
    return next(item for item in summary["groups"] if item["group"]["method"] == method)


def test_timing_repetitions_are_medians_not_extra_independent_data(tmp_path, monkeypatch):
    for identity, runtimes, primary_ber in [("a", [1, 100, 3], 0.1), ("b", [2, 4, 50], 0.2)]:
        results = [
            _method(
                runtime=runtime,
                repeat=repeat,
                ber=primary_ber if repeat == 0 else 0.9,
                status="time_limit" if identity == "a" and repeat == 1 else "optimal",
            )
            for repeat, runtime in enumerate(runtimes)
        ]
        _record(tmp_path, identity, results)
    summary, paths = _summary(tmp_path, monkeypatch)
    group = _group(summary)
    assert group["n_instances"] == 2 and group["n_timing_runs"] == 6
    assert group["metrics"]["runtime"]["mean"] == 3.5
    assert group["metrics"]["runtime"]["n"] == 2
    assert group["metrics"]["ber_prob"]["mean"] == pytest.approx(0.15)
    assert group["timing_status_counts"]["time_limit"] == 1
    with open(paths["tables"]["timing_runs"], encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 6


def test_har_averages_training_repetitions_inside_people_and_bootstraps_people(
    tmp_path, monkeypatch
):
    for repeat, (person1, person2) in enumerate(zip([0.1, 0.2, 0.3], [0.6, 0.8, 1.0])):
        global_people = {"1": {"ber_prob": person1}, "2": {"ber_prob": person2}}
        baseline_people = {
            person: {"ber_prob": value["ber_prob"] - 0.05}
            for person, value in global_people.items()
        }
        _record(
            tmp_path,
            f"har{repeat}",
            [
                _method(metrics={"ber_prob": 0.99, "by_subject": global_people}),
                _method("random", metrics={"ber_prob": 0.01, "by_subject": baseline_people}),
            ],
            kind="har",
            D=561,
            n=40,
            s=0,
        )
    summary, _ = _summary(tmp_path, monkeypatch)
    group = _group(summary)
    metric = group["metrics"]["ber_prob"]
    assert group["n_instances"] == 3
    assert metric["n"] == 2 and metric["unit"] == "held_out_person"
    assert metric["mean"] == pytest.approx(0.5)
    assert metric["person_values"] == pytest.approx({"1": 0.2, "2": 0.8})
    assert metric["ci95"] is not None
    comparison = next(
        item for item in summary["paired_comparisons"] if item["metric"] == "ber_prob"
    )
    assert comparison["n"] == 2 and comparison["paired_training_instances"] == 3
    assert comparison["mean"] == pytest.approx(-0.05)
    assert comparison["ci95"] == pytest.approx([-0.05, -0.05])


def test_failed_and_timed_out_methods_remain_in_denominators(tmp_path, monkeypatch):
    _record(tmp_path, "ok", [_method(ber=0.1)])
    _record(tmp_path, "timeout", [_method(ber=0.5, status="time_limit")])
    _record(
        tmp_path,
        "bad",
        [_method(status="solver_error", metrics={})],
        status="failed",
        error="simulated failure",
    )
    summary, _ = _summary(tmp_path, monkeypatch)
    group = _group(summary)
    assert group["n_instances"] == 3
    assert group["metrics"]["ber_prob"]["n"] == 2
    assert group["metrics"]["ber_prob"]["missing_instances"] == 1
    assert group["metrics"]["ber_prob"]["mean"] == pytest.approx(0.3)
    assert group["status_counts"] == {"optimal": 1, "time_limit": 1, "solver_error": 1}
    assert group["failed_record_instances"] == 1
    assert len(summary["failed_instances"]) == 1


def test_conditions_and_paired_ids_are_not_pooled(tmp_path, monkeypatch):
    for D in (20, 50):
        for repeat in range(2):
            _record(
                tmp_path,
                f"D{D}-{repeat}",
                [_method(ber=0.1 + repeat * 0.1), _method("random", ber=0.2 + repeat * 0.3)],
                D=D,
            )
    summary, _ = _summary(tmp_path, monkeypatch)
    assert len(summary["groups"]) == 4
    comparisons = [item for item in summary["paired_comparisons"] if item["metric"] == "ber_prob"]
    assert len(comparisons) == 2
    for comparison in comparisons:
        assert comparison["n"] == 2
        assert comparison["mean"] == pytest.approx(0.2)
        assert comparison["ci95"] == pytest.approx([0.1, 0.3])


def test_tabular_split_statistics_do_not_get_independent_split_bootstrap(tmp_path, monkeypatch):
    for repeat in range(3):
        _record(
            tmp_path,
            f"split{repeat}",
            [_method(ber=0.1), _method("random", ber=0.2)],
            kind="tabular",
            s=0,
        )
    summary, _ = _summary(tmp_path, monkeypatch)
    comparison = next(
        item for item in summary["paired_comparisons"] if item["metric"] == "ber_prob"
    )
    assert comparison["unit"] == "reused_dataset_split"
    assert comparison["ci95"] is None


def test_shift_metrics_expose_missing_shift_results(tmp_path, monkeypatch):
    shift = {"kind": "translation", "delta_factor": 1, "delta": 0.3, "metrics": {"ber_prob": 0.4}}
    _record(tmp_path, "shift-ok", [_method(shifts=[shift])], kind="shift")
    _record(tmp_path, "shift-bad", [_method(status="solver_error")], kind="shift")
    summary, _ = _summary(tmp_path, monkeypatch)
    group = summary["shift_groups"][0]
    assert group["n_instances"] == 2 and group["n_shift_records"] == 1
    assert group["metrics"]["ber_prob"]["n"] == 1
    assert group["metrics"]["ber_prob"]["missing_instances"] == 1


def test_malformed_records_and_duplicate_attempts_are_explicit(tmp_path, monkeypatch):
    _record(tmp_path, "same", [_method()], filename="first.json")
    _record(tmp_path, "same", [_method()], filename="second.json")
    (tmp_path / "records" / "broken.json").write_text("broken", encoding="utf-8")
    summary, _ = _summary(tmp_path, monkeypatch)
    assert summary["record_files"] == 3
    assert len(summary["load_errors"]) == len(summary["duplicate_attempts"]) == 1
    assert _group(summary)["n_instances"] == 1


def test_frozen_plan_exposes_unrun_instances_and_missing_methods_without_imputation(
    tmp_path, monkeypatch
):
    _record(tmp_path, "planned-a", [_method()])
    path = tmp_path / "records" / "planned-a.json"
    record = read_json(path)
    spec_a = record["spec"]
    spec_b = {**spec_a, "independent_data_id": "planned-b", "seed": 2}
    manifest = {
        "schema_version": 1,
        "suite": "main",
        "stage": "confirmation",
        "methods": ["global", "random"],
        "specs": [spec_a, spec_b],
        "budget_tasks": 2,
    }
    manifest["manifest_hash"] = fingerprint(manifest)
    write_json(tmp_path / "protocol.json", manifest)
    record["manifest_hash"] = manifest["manifest_hash"]
    record.pop("record_hash")
    record["record_hash"] = fingerprint(record)
    write_json(path, record)
    summary, paths = _summary(tmp_path, monkeypatch)
    assert paths["planned_missing_records"] == 1
    assert paths["planned_missing_outcomes"] == 3
    assert summary["protocol_coverage"][0]["planned_instances"] == 2
    group = _group(summary)
    assert group["n_instances"] == 1 and group["planned_instances"] == 2
    assert group["missing_outcome_instances"] == 1
    absent = _group(summary, "random")
    assert absent["n_instances"] == 0 and absent["planned_instances"] == 2
    assert absent["metrics"]["ber_prob"]["mean"] is None
    assert absent["metrics"]["ber_prob"]["missing_planned_instances"] == 2


def test_record_hash_rejects_tampered_values_instead_of_aggregating_them(tmp_path, monkeypatch):
    _record(tmp_path, "changed", [_method(ber=0.2)])
    path = tmp_path / "records" / "changed.json"
    record = read_json(path)
    record["results"][0]["metrics"]["ber_prob"] = 0
    write_json(path, record)
    summary, paths = _summary(tmp_path, monkeypatch)
    assert paths["load_errors"] == 1
    assert summary["observed_method_attempts"] == 0 and summary["groups"] == []


def test_summary_builds_condition_specific_figures(tmp_path):
    for budget in (2, 4):
        for repeat in range(2):
            _record(
                tmp_path,
                f"d{budget}-r{repeat}",
                [
                    _method(budget=budget, metrics={"ber_prob": 0.2, "recall": budget / 4}),
                    _method("random", budget=budget, metrics={"ber_prob": 0.3, "recall": 0.1}),
                ],
            )
    shifts = [
        {
            "kind": kind,
            "delta_factor": delta,
            "delta": delta * 0.3,
            "metrics": {"ber_prob": 0.2 + delta * 0.1},
        }
        for kind in ("translation", "noise")
        for delta in (0, 1)
    ]
    _record(tmp_path, "shift", [_method(shifts=shifts)], kind="shift")
    _record(
        tmp_path,
        "small",
        [_method(), _method("forward", runtime=0.5)],
        kind="small",
        verification={"status": "passed", "independent_optimum": 0.4},
    )
    for dimension in (20, 50):
        _record(
            tmp_path,
            f"scale{dimension}",
            [_method(runtime=dimension / 20)],
            kind="scaling",
            D=dimension,
        )
    result = analysis.summarize(tmp_path)
    summary = read_json(result["summary"])
    assert not summary["figure_errors"]
    assert any("E1-error" in path for path in result["figures"])
    assert any("E1-recall" in path for path in result["figures"])
    assert any("E1-paired-baselines" in path for path in result["figures"])
    assert any("E2-shift" in path for path in result["figures"])
    assert any("E3-bound-trace" in path for path in result["figures"])
    assert any("E3-excess-risk-time" in path for path in result["figures"])
    assert any("E3-time" in path for path in result["figures"])
    assert all(__import__("pathlib").Path(path).stat().st_size > 1000 for path in result["figures"])


def test_independent_verification_uses_all_small_subsets_and_checks_own_bound():
    problem = Problem([[0, 0]], [[2, 1]], [0.2, 0.2])
    assert transport_value(problem, ()) == pytest.approx(1.0)
    assert transport_value(problem, (0,)) == pytest.approx(0.2)
    assert transport_value(problem, (1,)) == pytest.approx(0.4)
    assert transport_value(problem, (0, 1)) == pytest.approx(0.4 / 3)
    result = select(problem, 1)
    verified = verify_small(problem, 1, [result])
    assert verified["enumerated_subsets"] == 3
    assert verified["status"] == "passed"
    with pytest.raises(AssertionError, match="Invalid affine cut"):
        verify_small(problem, 1, [replace(result, cuts=(Cut(2.0, np.zeros(2)),))])
    # This upper bound is above the global optimum .2, but below the chosen
    # subset's true value .4. A mere global-interval check would miss the error.
    invalid = replace(result, support=(1,), upper=0.3, lower=None)
    with pytest.raises(AssertionError, match="own subset"):
        verify_small(problem, 1, [invalid])


def test_first_best_time_and_later_certificate_are_distinct(tmp_path, monkeypatch):
    trace = [
        {"seconds": 1, "support": [0], "fit_upper": 0.1, "upper": 1.0, "lower": 0.0},
        {"seconds": 2, "support": [0, 1], "fit_upper": 0.2, "upper": 0.8, "lower": 0.0},
        {"seconds": 3, "upper": 0.2, "lower": 0.0},
        {"seconds": 42, "upper": 0.2, "lower": 0.1995},
    ]
    _record(
        tmp_path,
        "late-proof",
        [
            _method(
                budget=2,
                runtime=50,
                upper=0.2,
                lower=0.1999,
                gap=0.0001,
                trace=trace,
                target_tolerance=1e-3,
            )
        ],
    )
    summary, paths = _summary(tmp_path, monkeypatch)
    metrics = _group(summary)["metrics"]
    assert metrics["first_best_time"]["mean"] == 2
    assert metrics["time_to_gap"]["mean"] == 42
    assert metrics["time_to_gap"]["target_tolerances"] == [1e-3]
    assert metrics["gap_at_10s"]["mean"] == pytest.approx(0.2)
    assert metrics["gap_at_60s"]["mean"] == pytest.approx(0.0001)
    with open(paths["tables"]["instances"], encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["first_best_time"]) == 2 and float(row["time_to_gap"]) == 42
    assert int(row["time_to_gap_valid_timing_runs"]) == 1


def test_timeout_with_invalid_lower_bound_never_creates_a_certificate_time(tmp_path, monkeypatch):
    trace = [
        {"seconds": 2, "upper": 0.2, "lower": 0.0},
        {"seconds": 40, "upper": 0.2, "lower": 0.3},
        {"seconds": 45, "upper": 0.2, "lower": None},
    ]
    _record(
        tmp_path,
        "unproved",
        [_method(runtime=50, upper=0.2, lower=0.1, gap=0.1, trace=trace, status="time_limit")],
    )
    summary, paths = _summary(tmp_path, monkeypatch)
    metrics = _group(summary)["metrics"]
    assert metrics["first_best_time"]["mean"] == 2
    assert metrics["time_to_gap"]["n"] == 0
    assert metrics["time_to_gap"]["mean"] is None
    assert metrics["time_to_gap"]["valid_timing_runs"] == 0
    assert metrics["time_to_gap"]["missing_timing_runs"] == 1
    with open(paths["tables"]["instances"], encoding="utf-8", newline="") as handle:
        assert next(csv.DictReader(handle))["time_to_gap"] == ""


def test_search_times_use_within_instance_timing_medians_and_reached_counts(tmp_path, monkeypatch):
    repetitions = []
    for repeat, first, proof in [(0, 2, 50), (1, 4, 20), (2, 8, None)]:
        trace = [{"seconds": first, "upper": 0.2, "lower": 0.0}]
        if proof is not None:
            trace.append({"seconds": proof, "upper": 0.2, "lower": 0.1999})
        repetitions.append(
            _method(
                repeat=repeat,
                runtime=60,
                upper=0.2,
                lower=0.1999 if proof is not None else 0.1,
                trace=trace,
                status="optimal" if proof is not None else "time_limit",
            )
        )
    _record(tmp_path, "one-data-set", repetitions, kind="scaling")
    summary, paths = _summary(tmp_path, monkeypatch)
    group = _group(summary)
    assert group["n_instances"] == 1 and group["n_timing_runs"] == 3
    assert group["metrics"]["first_best_time"]["mean"] == 4
    metric = group["metrics"]["time_to_gap"]
    assert metric["n"] == 1 and metric["mean"] == 35
    assert metric["valid_timing_runs"] == 2 and metric["missing_timing_runs"] == 1
    with open(paths["tables"]["timing_runs"], encoding="utf-8", newline="") as handle:
        times = [row["time_to_gap"] for row in csv.DictReader(handle)]
    assert times == ["50.0", "20.0", ""]


@pytest.mark.parametrize(
    "kind,explicit,expected",
    [
        ("small", None, None),
        ("synthetic", None, 5),
        ("small", 1e-5, 5),
    ],
)
def test_certificate_time_respects_the_declared_or_suite_tolerance(
    tmp_path, monkeypatch, kind, explicit, expected
):
    fields = {"target_tolerance": explicit} if explicit is not None else {}
    trace = [{"seconds": 5, "upper": 0.2, "lower": 0.199995}]
    _record(
        tmp_path,
        "precision",
        [_method("enumeration", runtime=10, upper=0.2, lower=0.199995, trace=trace, **fields)],
        kind=kind,
    )
    summary, _ = _summary(tmp_path, monkeypatch)
    assert _group(summary, "enumeration")["metrics"]["time_to_gap"]["mean"] == expected


def test_constant_incumbent_starts_at_zero_but_heuristics_do_not_certify(tmp_path, monkeypatch):
    trace = [{"seconds": 7, "upper": 1.0, "lower": 0.9999999999}]
    _record(
        tmp_path,
        "constant",
        [
            _method(runtime=10, upper=1.0, lower=0.9999999999, trace=trace),
            _method("forward", runtime=10, upper=1.0, lower=None, trace=trace),
        ],
    )
    summary, _ = _summary(tmp_path, monkeypatch)
    assert _group(summary)["metrics"]["first_best_time"]["mean"] == 0
    assert _group(summary)["metrics"]["time_to_gap"]["mean"] == 7
    assert _group(summary, "forward")["metrics"]["time_to_gap"]["mean"] is None


@pytest.mark.parametrize(
    "status", ["verification_failure", "numerical_failure", "instance_failure", "solver_error"]
)
def test_invalid_certificate_status_cannot_report_trace_certification(
    tmp_path, monkeypatch, status
):
    trace = [{"seconds": 2, "upper": 0.2, "lower": 0.1999999999}]
    _record(
        tmp_path,
        "invalid-certificate",
        [_method(runtime=5, upper=0.2, lower=0.1999999999, trace=trace, status=status)],
    )
    summary, _ = _summary(tmp_path, monkeypatch)
    metrics = _group(summary)["metrics"]
    assert metrics["time_to_gap"]["n"] == 0 and metrics["time_to_gap"]["mean"] is None
    for seconds in (10, 60, 300):
        for bound in ("upper", "lower", "gap"):
            assert metrics[f"{bound}_at_{seconds}s"]["mean"] is None
    # The observed incumbent time remains an auditable reported result; its
    # failed status is kept, rather than silently promoting it to a proof.
    assert metrics["first_best_time"]["mean"] == 2


def test_inconsistent_timeout_interval_is_not_a_negative_gap_checkpoint(tmp_path, monkeypatch):
    trace = [{"seconds": 2, "upper": 0.2, "lower": 0.3}]
    _record(
        tmp_path,
        "inconsistent-timeout",
        [_method(runtime=5, upper=0.2, lower=None, trace=trace, status="time_limit")],
    )
    summary, _ = _summary(tmp_path, monkeypatch)
    metrics = _group(summary)["metrics"]
    assert metrics["time_to_gap"]["mean"] is None
    for seconds in (10, 60, 300):
        assert metrics[f"gap_at_{seconds}s"]["mean"] is None
