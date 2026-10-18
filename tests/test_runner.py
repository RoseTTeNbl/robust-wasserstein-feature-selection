"""Orchestration regressions: information barriers, accounting and immutable runs."""

from dataclasses import replace
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest
from numpy.testing import assert_allclose

from robusthw.core import SolveOptions, constant_fit
from robusthw.experiments import runner
from robusthw.experiments.baselines import SelectionTimeout
from robusthw.experiments.data import synthetic_split
from robusthw.experiments.protocol import ExperimentSpec
from robusthw.experiments.storage import array_hash, fingerprint, read_json, write_json


@pytest.fixture
def data():
    return synthetic_split("mean", 6, 5, 4, 71, validation_per_class=5, test_per_class=8)


@pytest.fixture
def spec():
    return ExperimentSpec(
        "development/fixture/r000",
        "development",
        "mean",
        6,
        5,
        4,
        71,
        (1, 2),
        {"suite": "development", "repeat": 0},
    )


def fake_fit(problem, support, *, validation_data=None, error=0.5, runtime=0.25, status="optimal"):
    def predict(points):
        if validation_data is not None:
            if points is validation_data.val0:
                return np.full(len(points), 1 - 2 * error)
            if points is validation_data.val1:
                return np.full(len(points), 2 * error - 1)
            raise AssertionError("Tuning inspected observations outside the validation sample")
        return np.zeros(len(points))

    return SimpleNamespace(
        problem=problem,
        support=tuple(support),
        scores=np.zeros(problem.size),
        source_bounds=np.full(problem.size, 0.5),
        prices=np.zeros(2),
        upper=1.0,
        lower=1.0,
        status=status,
        runtime=runtime,
        diagnostics={},
        predict_score=predict,
    )


def test_common_radius_uses_only_validation_and_ties_choose_larger_radius(data, spec, monkeypatch):
    seen = []

    def selector(method, left, right, budget, **kwargs):
        assert left is data.train0 and right is data.train1
        seen.append((method, budget))
        return tuple(range(budget))

    monkeypatch.setattr(runner, "select_features", selector)
    monkeypatch.setattr(
        runner,
        "fit_fixed",
        lambda problem, support, *args, **kwargs: fake_fit(problem, support, validation_data=data),
    )
    radii, selection = runner._radius_selection(data, spec, SolveOptions())
    assert_allclose(radii, 2 * runner._scales(data))
    assert selection["factor"] == 2
    assert seen == [("marginal_mmd", 4), ("mrmr", 4), ("joint_mmd", 4)]
    assert len(selection["candidates"]) == 5
    assert all(len(item["references"]) == 3 for item in selection["candidates"])


def test_one_radius_selection_is_shared_across_budgets_and_methods(
    tmp_path, data, spec, monkeypatch
):
    calls, applied = [], []
    radii = np.array([0.15, 0.3])
    monkeypatch.setattr(runner, "_prepare", lambda *args: data)

    def choose(*args):
        calls.append(True)
        return radii, {"rule": "shared_validation", "runtime": 0.1}

    def compare(specification, prepared, shared, method, budget, *args, **kwargs):
        assert prepared is data
        applied.append((method, budget, shared.copy()))
        return {
            "method": method,
            "budget": budget,
            "status": "complete",
            "upper": 1.0,
            "role": "comparison",
            "runtime": 0.2,
        }, None

    monkeypatch.setattr(runner, "_radius_selection", choose)
    monkeypatch.setattr(runner, "_single_record", compare)
    monkeypatch.setattr(
        runner, "fit_fixed", lambda problem, support, *args, **kwargs: fake_fit(problem, support)
    )
    result = runner._execute(spec, tmp_path, None, ("random", "mrmr"))
    assert len(calls) == 1 and len(applied) == 4
    assert all(np.array_equal(shared, radii) for _, _, shared in applied)
    assert result["radius_selection"]["runtime"] == 0.1
    compared = [row for row in result["results"] if row["role"] == "comparison"]
    assert all(row["radius_selection_seconds"] == 0.1 for row in compared)
    assert all(row["total_runtime"] == pytest.approx(0.3) for row in compared)


def test_anytime_trace_clock_includes_initial_feature_ranking(data, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(runner, "perf_counter", lambda: clock[0])

    def ranking(*args):
        clock[0] += 2
        return tuple(range(data.dimension))

    def search(problem, budget, method, options, **kwargs):
        assert options.time_limit == 8
        clock[0] += 3
        fitted = constant_fit(problem, tuple(range(budget)))
        return SimpleNamespace(
            status="time_limit",
            upper=1.0,
            lower=0.0,
            gap=1.0,
            fit_count=1,
            master_count=1,
            master_seconds=1.0,
            oracle_seconds=2.0,
            trace=[{"seconds": 1.0}, {"seconds": 3.0}],
            diagnostics={},
            cuts=(),
            fit=fitted,
        )

    monkeypatch.setattr(runner, "rank_features", ranking)
    monkeypatch.setattr(runner, "select", search)
    record, _, _ = runner._fit_method(
        data, np.array([0.1, 0.1]), 2, "global", SolveOptions(time_limit=10), seed=7
    )
    assert record["initialization_seconds"] == 2
    assert record["runtime"] == 5
    assert [row["seconds"] for row in record["trace"]] == [3, 5]


def test_l1_parameters_are_chosen_on_validation_with_stronger_penalty_ties(data, monkeypatch):
    parameters = []
    clock = [0.0]

    def candidate(prepared, radii, budget, method, options, seed, *, C):
        assert prepared is data and method == "l1_logistic"
        parameters.append(C)
        clock[0] += len(parameters)
        fit = fake_fit(
            runner.Problem(*data.train, radii),
            (0, 1),
            validation_data=data,
            error=0.1 if C <= 0.1 else 0.4,
        )
        return (
            {
                "status": "complete",
                "runtime": float(len(parameters)),
                "fit_count": 1,
                "oracle_seconds": 0.25,
            },
            fit,
            None,
        )

    monkeypatch.setattr(runner, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(runner, "_fit_method", candidate)
    result, _, _ = runner._fit_l1(data, np.array([0.2, 0.2]), 2, SolveOptions(), 71)
    assert parameters == [0.01, 0.1, 1.0, 10.0, 100.0]
    assert result["validation"]["selected_C"] == 0.01
    assert result["runtime"] == 15
    assert result["fit_count"] == 5 and result["oracle_seconds"] == 1.25
    assert result["validation"]["candidates"][0]["runtime"] == 1


def test_anchor_fits_all_variables_without_invoking_any_selector(tmp_path, data, spec, monkeypatch):
    fitted_supports = []
    anchor = replace(spec, kind="anchor", budgets=(data.dimension,))
    monkeypatch.setattr(runner, "_prepare", lambda *args: data)
    monkeypatch.setattr(
        runner, "_radius_selection", lambda *args: (np.array([0.2, 0.2]), {"runtime": 0.0})
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A full-variable anchor does not run feature selection")

    def fit(problem, support, *args, **kwargs):
        fitted_supports.append(tuple(support))
        return fake_fit(problem, support)

    monkeypatch.setattr(runner, "select_features", forbidden)
    monkeypatch.setattr(runner, "select", forbidden)
    monkeypatch.setattr(runner, "fit_fixed", fit)
    result = runner._execute(anchor, tmp_path, None, runner.METHODS)
    assert fitted_supports == [tuple(range(data.dimension))]
    assert len(result["results"]) == 1
    record = result["results"][0]
    assert record["method"] == "all_variables" and record["role"] == "reference"
    assert "precision" not in record["metrics"]
    assert record["status"] in ("optimal", "complete")


def test_baseline_timeout_fills_budget_and_uses_only_remaining_time(data, monkeypatch):
    received = []
    monkeypatch.setattr(runner, "rank_features", lambda *args: (0, 1, 2, 3, 4, 5))

    def timeout(*args, **kwargs):
        raise SelectionTimeout((3,))

    def fit(problem, support, options, **kwargs):
        received.append((support, options.time_limit))
        return constant_fit(problem, support, "time_limit")

    clock = iter([0.0, 6.0, 6.0])
    monkeypatch.setattr(runner, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(runner, "select_features", timeout)
    monkeypatch.setattr(runner, "fit_fixed", fit)
    record, fitted, _ = runner._fit_method(
        data, np.array([0.2, 0.2]), 2, "joint_mmd", SolveOptions(time_limit=5), 71
    )
    assert received == [((0, 3), 0.0)]
    assert record["status"] == "incomplete" and record["runtime"] == 6
    assert_allclose(fitted.predict_score(data.test0), 0)


def test_method_exception_is_charged_and_keeps_constant_prediction(data, spec, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("deliberate candidate failure")

    clock = iter([10.0, 17.0])
    monkeypatch.setattr(runner, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(runner, "_fit_method", fail)
    result, _ = runner._single_record(spec, data, np.array([0.2, 0.2]), "random", 2, SolveOptions())
    assert result["status"] == "solver_error"
    assert result["runtime"] == 7
    assert "deliberate candidate failure" in result["error"]
    assert result["metrics"]["ber_prob"] == 0.5
    assert len(result["support"]) == 2


def frozen_fixture(tmp_path, spec, monkeypatch):
    monkeypatch.setattr(runner, "_sources", lambda: {})
    monkeypatch.setattr(runner, "build_plan", lambda *args: [spec])
    path = tmp_path / "protocol.json"
    runner.freeze("development", path, methods=("random", "marginal_mmd"))
    return path


def test_manifest_and_snapshot_tampering_are_rejected(tmp_path, spec, monkeypatch):
    path = frozen_fixture(tmp_path, spec, monkeypatch)
    manifest = read_json(path)
    manifest["specs"][0]["n"] += 1
    write_json(path, manifest)
    with pytest.raises(ValueError, match="[Ff]rozen|[Pp]rotocol"):
        runner.run(path)


def test_preparation_failure_preserves_all_method_budget_predictions(tmp_path, spec, monkeypatch):
    path = frozen_fixture(tmp_path, spec, monkeypatch)

    def fail(*args):
        raise ValueError("deliberate data/preprocessing failure")

    monkeypatch.setattr(runner, "_execute", fail)
    outcome = runner.run(path)
    assert outcome["failed"] == 1
    record = read_json(tmp_path / "records" / (spec.independent_data_id + ".json"))
    expected = {(method, budget) for method in ("random", "marginal_mmd") for budget in (1, 2)}
    assert {(row["method"], row["budget"]) for row in record["results"]} == expected
    assert all(row["metrics"]["ber_prob"] == 0.5 for row in record["results"])
    assert "deliberate data/preprocessing failure" in record["error"]


def test_resume_rejects_changed_raw_outcome(tmp_path, data, spec, monkeypatch):
    path = frozen_fixture(tmp_path, spec, monkeypatch)
    monkeypatch.setattr(runner, "_prepare", lambda *args: data)
    monkeypatch.setattr(
        runner, "_radius_selection", lambda *args: (np.array([0.2, 0.2]), {"runtime": 0.0})
    )
    monkeypatch.setattr(
        runner, "fit_fixed", lambda problem, support, *args, **kwargs: fake_fit(problem, support)
    )
    runner.run(path)
    destination = tmp_path / "records" / (spec.independent_data_id + ".json")
    record = read_json(destination)
    record["results"][0]["metrics"]["ber_prob"] = 0.123456
    write_json(destination, record)
    with pytest.raises(ValueError, match="[Cc]hanged|[Cc]hecksum|[Tt]amper|[Ff]ingerprint"):
        runner.run(path)


def source_fixture(tmp_path, data, spec, monkeypatch):
    source = replace(spec, budgets=(4,))
    path = frozen_fixture(tmp_path, source, monkeypatch)
    monkeypatch.setattr(runner, "_prepare", lambda *args: data)
    monkeypatch.setattr(
        runner, "_radius_selection", lambda *args: (np.array([0.2, 0.4]), {"runtime": 0.0})
    )
    monkeypatch.setattr(
        runner, "fit_fixed", lambda problem, support, *args, **kwargs: fake_fit(problem, support)
    )
    runner.run(path)
    shifted = replace(
        source,
        kind="shift",
        metadata={
            "suite": "shift",
            "source_data_id": source.independent_data_id,
            "shift_types": ["translation", "gaussian_noise"],
            "delta_factors": [0.0, 0.5, 1.0, 2.0],
            "refit": False,
        },
    )
    return path, shifted


def test_shift_reuses_matching_fits_without_selection_or_training(
    tmp_path, data, spec, monkeypatch
):
    _, shifted = source_fixture(tmp_path, data, spec, monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("E2 must reuse E1 fits")

    monkeypatch.setattr(runner, "fit_fixed", forbidden)
    monkeypatch.setattr(runner, "select_features", forbidden)
    result = runner._shift(shifted, tmp_path, methods=("random", "marginal_mmd"))
    assert result["refitted"] is False and len(result["results"]) == 2
    assert all(len(row["shifts"]) == 8 for row in result["results"])
    left, right = result["results"]
    assert [item["test_hashes"] for item in left["shifts"]] == [
        item["test_hashes"] for item in right["shifts"]
    ]
    no_change = left["shifts"][0]
    assert no_change["test_hashes"] == [array_hash(value) for value in data.test]
    changed = data.test0.copy()
    changed[:, list(data.signal_indices)] += 0.5 * float(np.mean(result["radii"])) / 4
    assert left["shifts"][1]["test_hashes"][0] == array_hash(changed)
    with pytest.raises(ValueError, match="missing"):
        runner._shift(shifted, tmp_path, methods=("global",))


def test_shift_rejects_foreign_setting_and_changed_samples(tmp_path, data, spec, monkeypatch):
    path, shifted = source_fixture(tmp_path, data, spec, monkeypatch)
    record_path = path.parent / "records" / (spec.independent_data_id + ".json")
    original = read_json(record_path)
    foreign = {**original, "spec": {**original["spec"], "seed": 999}}
    foreign["record_hash"] = fingerprint(
        {key: value for key, value in foreign.items() if key != "record_hash"}
    )
    write_json(record_path, foreign)
    with pytest.raises(ValueError, match="mismatch"):
        runner._shift(shifted, tmp_path, methods=("random",))
    write_json(record_path, original)
    data_path = path.parent / "data" / (spec.independent_data_id + ".npz")
    with data_path.open("ab") as handle:
        handle.write(b"altered bytes")
    with pytest.raises(ValueError, match="data file changed"):
        runner._shift(shifted, tmp_path, methods=("random",))


def test_resume_rejects_conflicting_manifest_even_if_record_hash_is_recomputed(
    tmp_path, data, spec, monkeypatch
):
    path, _ = source_fixture(tmp_path, data, spec, monkeypatch)
    record_path = path.parent / "records" / (spec.independent_data_id + ".json")
    record = read_json(record_path)
    record["manifest_hash"] = "different frozen run"
    record["record_hash"] = fingerprint(
        {key: value for key, value in record.items() if key != "record_hash"}
    )
    write_json(record_path, record)
    with pytest.raises(ValueError, match="Conflicting"):
        runner.run(path)


def test_frozen_source_snapshot_is_verified_independently_of_live_sources(tmp_path, monkeypatch):
    project = tmp_path / "repository"
    source = project / "src" / "robusthw" / "model.py"
    source.parent.mkdir(parents=True)
    (project / "configs").mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(runner, "PROJECT", project)
    monkeypatch.setattr(runner, "build_plan", lambda *args: [])
    path = tmp_path / "protocol.json"
    runner.freeze("development", path, methods=("random",))
    archive_path = path.with_suffix(".sources.zip")
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("src/robusthw/model.py", "VALUE = 999\n")
    with pytest.raises(ValueError, match="[Ss]napshot|[Ss]ource"):
        runner.run(path)


def test_failed_shift_keeps_every_planned_curve_point(data, spec):
    shifted = replace(
        spec,
        kind="shift",
        budgets=(4,),
        metadata={
            "suite": "shift",
            "source_data_id": spec.independent_data_id,
            "shift_types": ["translation", "gaussian_noise"],
            "delta_factors": [0.0, 0.5, 1.0, 2.0],
            "refit": False,
        },
    )
    record = runner._failed_instance(shifted, ("random",), "missing E1 fit", None, 1.0)
    assert len(record["results"]) == 1
    curves = record["results"][0]["shifts"]
    assert len(curves) == 8
    assert all(point["metrics"]["ber_prob"] == 0.5 for point in curves)


def test_small_verification_failure_retains_actual_method_records(
    tmp_path, data, spec, monkeypatch
):
    small = replace(spec, kind="small", budgets=(2,), metadata={"validation_tolerance": 1e-8})
    monkeypatch.setattr(runner, "_prepare", lambda *args: data)
    monkeypatch.setattr(
        runner, "_radius_selection", lambda *args: (np.array([0.1, 0.1]), {"runtime": 0.0})
    )
    monkeypatch.setattr(
        runner,
        "_single_record",
        lambda spec, data, radii, method, budget, *args, **kw: (
            {
                "method": method,
                "budget": budget,
                "status": "optimal",
                "runtime": 0.1,
                "upper": 0.4,
                "metrics": {"ber_prob": 0.23},
                "raw_witness": [1, 2, 3],
            },
            None,
        ),
    )

    def invalid(*args, **kwargs):
        raise AssertionError("Independent check rejected a cut")

    monkeypatch.setattr(runner, "verify_small", invalid)
    record = runner._execute(small, tmp_path, None, runner.METHODS)
    assert record["status"] == "failed"
    assert "rejected a cut" in record["verification"]["error"]
    assert len(record["results"]) == 4
    assert all(row["status"] == "verification_failure" for row in record["results"])
    assert all(
        row["metrics"]["ber_prob"] == 0.23 and row["raw_witness"] == [1, 2, 3]
        for row in record["results"]
    )


def test_fixed_scale_and_precision_use_the_frozen_spec_values(data, spec):
    small = replace(
        spec,
        kind="small",
        metadata={"radius_factor": 0.5, "outer_tolerance": 2e-5, "validation_tolerance": 3e-7},
    )
    options = runner._options(small)
    assert options.absolute_tolerance == 2e-5 and options.inner_tolerance == 3e-7
    radii, selection = runner._radius_selection(data, small, options)
    assert_allclose(radii, 0.5 * runner._scales(data))
    assert selection["factor"] == 0.5


def test_reference_failure_does_not_discard_completed_comparison(tmp_path, data, spec, monkeypatch):
    spec = replace(spec, budgets=(1,))
    monkeypatch.setattr(runner, "_prepare", lambda *args: data)
    monkeypatch.setattr(
        runner, "_radius_selection", lambda *args: (np.array([0.2, 0.2]), {"runtime": 0.0})
    )

    def fit(problem, support, *args, **kwargs):
        if len(support) > 1:
            raise RuntimeError("deliberate reference-only fit failure")
        return fake_fit(problem, support)

    monkeypatch.setattr(runner, "fit_fixed", fit)
    record = runner._execute(spec, tmp_path, None, ("random",))
    comparison = [row for row in record["results"] if row["role"] == "comparison"]
    references = [row for row in record["results"] if row["role"] == "reference"]
    assert len(comparison) == 1 and comparison[0]["status"] == "complete"
    assert len(references) == 2
    assert all(row["status"] == "solver_error" for row in references)
    assert all(row["metrics"]["ber_prob"] == 0.5 for row in references)


def test_l1_failed_parameter_is_retained_while_other_candidates_are_evaluated(data, monkeypatch):
    attempted = []

    def candidate(prepared, radii, budget, method, options, seed, *, C):
        attempted.append(C)
        if C == 0.01:
            raise RuntimeError("failed C=0.01")
        fitted = fake_fit(
            runner.Problem(*data.train, radii),
            (0, 1),
            validation_data=data,
            error=0.2 if C == 1 else 0.4,
        )
        return (
            {"status": "complete", "runtime": 0.1, "fit_count": 1, "oracle_seconds": 0.1},
            fitted,
            None,
        )

    monkeypatch.setattr(runner, "_fit_method", candidate)
    record, _, _ = runner._fit_l1(data, np.array([0.2, 0.2]), 2, SolveOptions(), 71)
    assert attempted == [0.01, 0.1, 1.0, 10.0, 100.0]
    assert record["validation"]["selected_C"] == 1
    failure = record["validation"]["candidates"][0]
    assert failure["status"] == "solver_error" and "failed C=0.01" in failure["error"]
    assert record["fit_count"] == 4


def test_failed_radius_reference_is_explicit_in_every_candidate(data, spec, monkeypatch):
    def selector(method, left, right, budget, **kwargs):
        if method == "mrmr":
            raise RuntimeError("reference selection failed")
        return tuple(range(budget))

    monkeypatch.setattr(runner, "select_features", selector)
    monkeypatch.setattr(
        runner,
        "fit_fixed",
        lambda problem, support, *args, **kwargs: fake_fit(
            problem, support, validation_data=data, error=0.2
        ),
    )
    _, selection = runner._radius_selection(data, spec, SolveOptions())
    for candidate in selection["candidates"]:
        reference = next(row for row in candidate["references"] if row["method"] == "mrmr")
        assert reference["validation_error"] == 0.5
        assert "reference selection failed" in reference["error"]


def test_all_l1_candidates_failed_retains_five_attempts_and_constant_metrics(data, monkeypatch):
    elapsed = [0.0]
    attempts = []

    def fail(prepared, radii, budget, method, options, seed, *, C):
        attempts.append(C)
        elapsed[0] += 2
        raise RuntimeError(f"deliberate C={C} failure")

    monkeypatch.setattr(runner, "perf_counter", lambda: elapsed[0])
    monkeypatch.setattr(runner, "_fit_method", fail)
    record, fitted, _ = runner._fit_l1(data, np.array([0.2, 0.2]), 2, SolveOptions(), 71)
    assert attempts == [0.01, 0.1, 1.0, 10.0, 100.0]
    assert record["status"] == "solver_error" and record["runtime"] == 10
    assert record["validation"]["selected_C"] == 0.01
    candidates = record["validation"]["candidates"]
    assert len(candidates) == 5
    assert all(row["status"] == "solver_error" and row["runtime"] == 2 for row in candidates)
    assert all(row["error"] and row["validation_error"] == 0.5 for row in candidates)
    assert runner._evaluate(fitted, data, 2)["ber_prob"] == 0.5
