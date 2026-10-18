"""Presentation summaries retain pairing, conditions and independent data units."""

import csv
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "report_results.py"
SPEC = importlib.util.spec_from_file_location("report_results", SCRIPT)
report_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report_results)


def presentation():
    return {
        "tables": {
            name: {"rows": [], "columns": ["condition", "a", "b", "c"]}
            for name in ("generality", "shift", "radius", "initialization")
        },
        "commentary": {},
    }


def group(
    suite,
    model,
    method="global",
    *,
    radius=1,
    start="marginal_mmd",
    error=0.2,
    observations=2,
    **condition,
):
    return {
        "group": {
            "manifest_hash": "frozen-" + suite,
            "suite": suite,
            "kind": "synthetic" if suite == "main" else suite,
            "model": model,
            "D": 50,
            "n": 20,
            "s": 4,
            "budget": 4,
            "role": "comparison",
            "method": method,
            "radius_factor": radius,
            "initialization": start,
            **condition,
        },
        "n_instances": observations,
        "n_timing_runs": observations,
        "planned_instances": 30 if suite == "main" else 10,
        "planned_timing_runs": 30 if suite == "main" else 10,
        "status_counts": {"optimal": observations},
        "metrics": {
            metric: {"n": observations, "mean": value, "sd": 0.01 if observations > 1 else None}
            for metric, value in (("ber_prob", error), ("upper", 0.4), ("runtime", 2), ("gap", 0))
        },
    }


def shift_record(identity, zero, two, *, joint_zero=None, joint_two=None):
    def row(method, start, end):
        shifts = [{"kind": "translation", "delta_factor": 0, "metrics": {"ber_prob": start}}]
        if end is not None:
            shifts.append({"kind": "translation", "delta_factor": 2, "metrics": {"ber_prob": end}})
        return {"method": method, "shifts": shifts}

    rows = [row("global", zero, two)]
    if joint_zero is not None:
        rows.append(row("joint_mmd", joint_zero, joint_two))
    return {
        "manifest_hash": "frozen-shift",
        "spec": {"independent_data_id": identity, "model": "mean", "kind": "shift"},
        "results": rows,
    }


def test_shift_sd_is_computed_from_within_record_differences_and_retains_improvement():
    records = [
        shift_record("a", 0.1, 0.3, joint_zero=0.4, joint_two=0.2),
        shift_record("b", 0.4, 0.5, joint_zero=0.2, joint_two=0.1),
        shift_record("missing-endpoint", 0.6, None),
    ]
    data = report_results.enrich(presentation(), {"groups": []}, records)
    row = data["tables"]["shift"]["rows"][0]
    assert row[1]["mean"] == pytest.approx(0.15)
    assert row[1]["sd"] == pytest.approx(0.1 / 2**0.5)
    assert row[1]["n"] == 2
    assert row[3]["mean"] == pytest.approx(-0.15)
    assert row[3]["sd"] == pytest.approx(0.1 / 2**0.5)
    assert len(data["tables"]["shift"]["rows"]) == 1
    counts = data["result_counts"]["shift"][0]["methods"]["global"]
    assert counts == {"paired": 2, "observed": 3}


def test_radius_and_initialization_keep_mean_and_mixture_separate():
    groups = []
    for model, offset in (("mean", 0), ("mixture", 0.5)):
        for method in ("global", "forward_swap"):
            for factor, error in ((0.5, 0.1), (1, 0.2), (2, 0.3)):
                groups.append(group("ablation", model, method, radius=factor, error=error + offset))
        groups.append(group("ablation", model, start="random", error=0.99))
    data = report_results.enrich(presentation(), {"groups": groups}, [])
    radius_rows = data["tables"]["radius"]["rows"]
    assert len(radius_rows) == 4
    mean_rows = [row for row in radius_rows if row[0]["en"].startswith("Mean /")]
    mixture_rows = [row for row in radius_rows if row[0]["en"].startswith("Mixture /")]
    assert all([cell["mean"] for cell in row[1:]] == [0.1, 0.2, 0.3] for row in mean_rows)
    for row in mixture_rows:
        assert [cell["mean"] for cell in row[1:]] == pytest.approx([0.6, 0.7, 0.8])
    assert all(cell["n"] == 2 for row in radius_rows for cell in row[1:])
    init_rows = data["tables"]["initialization"]["rows"]
    assert len(init_rows) == 4
    assert {row[0]["en"] for row in init_rows} == {
        "Mean / MMD start",
        "Mean / Random start",
        "Mixture / MMD start",
        "Mixture / Random start",
    }


def test_generality_uses_only_equal_signal_budget_and_does_not_pool_scales():
    groups = [
        group("main", "mean", D=20, n=10, error=0.1),
        group("main", "mean", error=0.3),
        group("main", "mean", budget=8, error=0.99),
        group("main", "variance", observations=0, error=0.2),
    ]
    data = report_results.enrich(presentation(), {"groups": groups}, [])
    rows = data["tables"]["generality"]["rows"]
    assert len(rows) == 2
    assert [row[1]["mean"] for row in rows] == [0.1, 0.3]
    assert all(row[2:] == [None, None] for row in rows)
    assert all(row[1]["n"] == 2 for row in rows)


def test_scaling_csv_exports_instance_medians_without_counting_timing_repeats_as_data(tmp_path):
    item = group("scaling", "mean")
    item.update(n_timing_runs=6, planned_timing_runs=30)
    item["metrics"]["runtime"] = {"n": 2, "mean": 10, "sd": 3}
    item["metrics"]["time_to_gap"] = {
        "n": 1,
        "mean": 8,
        "sd": None,
        "valid_timing_runs": 3,
        "missing_timing_runs": 3,
    }
    missing = group("scaling", "mixture", observations=0)
    for measure in missing["metrics"].values():
        measure.update(mean=None, sd=None)
    path = report_results.write_scaling_csv({"groups": [item, missing]}, tmp_path / "scaling.csv")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["observed_datasets"] == "2"
    assert rows[0]["observed_timing_runs"] == "6"
    assert rows[0]["runtime_mean"] == "10"
    assert rows[0]["runtime_n"] == "2"
    assert rows[0]["certificate_unreached_timing_runs"] == "3"
    assert rows[1]["observed_datasets"] == "0" and rows[1]["runtime_mean"] == ""
