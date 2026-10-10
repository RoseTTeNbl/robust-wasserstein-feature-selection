"""Full-inner adaptive selection checked against independent support risks."""
from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
from test_oracle_new import original_transport_value

import robusthw.adaptive as adaptive
import robusthw.solver as outer
from robusthw.history import EvaluationHistory
from robusthw.oracle import GateOracle
from robusthw.problem import Problem
from robusthw.solver import enumerate_supports, solve


def small_problem(loss="overlap", seed=7):
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=(3, 5)), rng.normal(size=(4, 5))
    x1[:, :2] += 1.1
    return Problem(x0, x1, 2, .17, .31, loss=loss,
                   q0=np.array([.2, .3, .5]), q1=np.array([.1, .2, .3, .4]))


def assert_evaluation_ownership(problem, result, tolerance):
    records = result.diagnostics["evaluations"]
    assert result.diagnostics["cut_count"] == result.visited == len(records)
    assert len({tuple(row["support"]) for row in records}) == len(records)
    for row in records:
        z = np.asarray(row["point"])
        assert np.all((z == 0) | (z == 1)) and z.sum() == problem.budget
        assert tuple(row["support"]) == problem.support(z)
        actual = row["intercept"] + np.asarray(row["gradient"]) @ z
        assert actual == pytest.approx(row["anchor_lower"], abs=1e-12)
        assert row["anchor_gap"] <= tolerance / 4
        assert row["accurate"]
    completed = [entry for entry in result.trace if entry["phase"] in ("initial", "adaptive")]
    assert len(completed) == len(records)
    assert all(entry["cut_added"] for entry in completed)
    observed_upper = problem.B
    for entry in result.trace:
        if entry["phase"] in ("initial", "adaptive"):
            observed_upper = min(observed_upper, entry["oracle_upper"])
        assert entry["upper"] == pytest.approx(observed_upper, abs=1e-12)
    acquisitions = [entry for entry in result.trace if entry["phase"] == "acquisition"]
    for entry in acquisitions:
        assert entry["model_lower"] <= entry["upper"]-entry["improvement_margin"]
        assert entry["improvement_margin"] == pytest.approx(tolerance / 2)
        assert entry["score"] == pytest.approx(entry["mean"]-entry["std"])
        assert entry["std_range"][0] <= entry["std"] <= entry["std_range"][1]
        assert entry["mean_range"][0] <= entry["mean"] <= entry["mean_range"][1]
        assert entry["candidate_count"] >= entry["eligible"] > 0
        assert entry["candidate_count"] <= result.diagnostics["proposal_count"] <= 8
    assert result.diagnostics["forecast_model"] == "fixed_hamming_gp_v1"
    assert result.diagnostics["prediction_training_points"] == len(records)
    assert result.diagnostics["prediction_factor_updates"] == len(records)
    assert result.diagnostics["prediction_batches"] == len(acquisitions)
    assert result.diagnostics["transferred_tests"] == 0
    assert result.diagnostics["reused_fraction_cuts"] == 0
    assert result.diagnostics["fractional_root_calls"] == 0
    assert result.diagnostics["predicted_upper_updates"] == 0


@pytest.mark.parametrize("loss", ["overlap", "logistic"])
def test_full_inner_adaptive_matches_independent_enumeration(loss):
    p = small_problem(loss)
    tolerance = 2e-5
    if loss == "logistic":
        pytest.importorskip("cvxpy")
    result = solve(p, search="adaptive", root_rounds=20, time_limit=90.,
                   tolerance=tolerance, relative_tolerance=0, proposal_count=8)
    if loss == "overlap":
        true_values = [original_transport_value(p, p.mask(s))
                       for s in combinations(range(p.p), p.budget)]
        reference_lower = reference_upper = min(true_values)
    else:
        reference = enumerate_supports(p, time_limit=90., tolerance=tolerance)
        assert reference.status == "optimal"
        reference_lower, reference_upper = reference.lower, reference.upper
    assert result.status == "optimal", result.to_dict()
    assert result.lower <= reference_upper + 2e-7
    assert result.upper >= reference_lower - 2e-7
    assert result.upper-reference_lower <= tolerance+2e-7
    assert result.test is not None
    assert np.isfinite(result.test.prob_class0(p.points)).all()
    assert result.diagnostics["search"] == "full_inner_uncertainty_v3"
    assert result.diagnostics["requested_root_rounds"] == 20
    assert result.diagnostics["root_rounds"] == 0
    assert_evaluation_ownership(p, result, tolerance)


def test_scalar_candidate_bounds_cover_independent_original_risks():
    p = small_problem()
    oracle = GateOracle(p)
    history = EvaluationHistory(p)
    anchor = p.mask((0, 2))
    result = oracle.solve(anchor)
    history.add(anchor, test=result.test, intercept=result.intercept,
                gradient=result.gradient, accurate=True)
    tests_before = dict(history.tests)
    for support in combinations(range(p.p), p.budget):
        z = p.mask(support)
        actual = original_transport_value(p, z)
        lower, predicted, width = history.bounds(z)
        assert lower <= actual+1e-7 <= predicted+1e-7
        assert isinstance(predicted, float)
        assert width >= 0
        assert predicted <= p.B
    assert history.tests == tests_before
    assert len(history.cuts) == len(history.points) == 1
    assert not hasattr(history, "reprice")
    assert not hasattr(history, "densify")


def test_history_rejects_fractional_evaluation_ownership():
    p = small_problem()
    history = EvaluationHistory(p)
    with pytest.raises(ValueError, match="binary"):
        history.add(np.full(p.p, p.budget/p.p), test=None,
                    intercept=0., gradient=np.zeros(p.p), accurate=False)


def test_no_transfers_recentering_or_fractional_oracle_calls(monkeypatch):
    p = small_problem()
    calls, owned = [], {}
    actual_solve = GateOracle.solve

    def forbidden(*args, **kwargs):
        raise AssertionError("this operation is forbidden in the full-inner path")

    def binary_solve(self, z, **kwargs):
        z = np.asarray(z)
        assert np.all((z == 0) | (z == 1)) and z.sum() == p.budget
        calls.append(z.copy())
        answer = actual_solve(self, z, **kwargs)
        owned.setdefault(p.support(z), []).append(answer)
        return answer

    monkeypatch.setattr(GateOracle, "solve", binary_solve)
    monkeypatch.setattr(GateOracle, "reprice", forbidden)
    monkeypatch.setattr(EvaluationHistory, "densify", forbidden, raising=False)
    monkeypatch.setattr(EvaluationHistory, "reprice", forbidden, raising=False)
    monkeypatch.setattr(EvaluationHistory, "scalar_upper", forbidden)
    monkeypatch.setattr(EvaluationHistory, "bounds", forbidden)
    monkeypatch.setattr(outer, "_ratio_master", forbidden)
    result = solve(p, root_rounds=30, tolerance=2e-5, relative_tolerance=0)
    assert result.status == "optimal", result.to_dict()
    assert len(calls) == result.oracle_calls
    for row in result.diagnostics["evaluations"]:
        candidates = owned[tuple(row["support"])]
        assert any(np.allclose(answer.gradient, row["gradient"], atol=0, rtol=0)
                   and answer.intercept == row["intercept"] for answer in candidates)
        assert any(answer.test.upper == row["actual_upper"] for answer in candidates)
    assert_evaluation_ownership(p, result, 2e-5)


def test_forecast_does_not_update_the_incumbent(monkeypatch):
    p = Problem([[0., 0.]], [[2., 0.]], 1, .1, .1)
    good = p.mask((0,)).astype(float)
    bad = p.mask((1,)).astype(float)
    monkeypatch.setattr(adaptive, "proposal_pool", lambda *args, **kwargs: [good])
    monkeypatch.setattr(EvaluationHistory, "predict",
                        lambda self, points: (np.full(len(points), -.3),
                                              np.full(len(points), .2)))
    result = solve(p, warm_support=(1,), tolerance=1e-5, relative_tolerance=0,
                   root_rounds=99, max_iterations=1, certify_every=10)
    assert result.status == "iteration_limit"
    acquisition = next(entry for entry in result.trace if entry["phase"] == "acquisition")
    assert acquisition["mean"] == pytest.approx(-.3)
    assert acquisition["std"] == pytest.approx(.2)
    assert acquisition["score"] == pytest.approx(-.5)
    assert acquisition["upper"] == pytest.approx(p.B)
    assert result.oracle_calls == 2
    assert result.upper == pytest.approx(.1, abs=1e-8)
    np.testing.assert_equal(result.diagnostics["evaluations"][0]["point"], bad)
    np.testing.assert_equal(result.diagnostics["evaluations"][1]["point"], good)
    assert result.diagnostics["predicted_upper_updates"] == 0


def test_refinements_keep_one_final_owned_cut_per_support(monkeypatch):
    p = small_problem()
    actual_solve, attempts = GateOracle.solve, {}

    def first_coarse(self, z, **kwargs):
        answer = actual_solve(self, z, **kwargs)
        key = p.support(z)
        attempts[key] = attempts.get(key, 0)+1
        if attempts[key] == 1:
            return replace(answer, value=answer.value-.001,
                           intercept=answer.intercept-.001, status="accuracy_limit")
        return answer

    monkeypatch.setattr(GateOracle, "solve", first_coarse)
    result = solve(p, tolerance=2e-5, relative_tolerance=0, max_refinements=1)
    assert result.status == "optimal", result.to_dict()
    assert result.oracle_calls == 2*result.visited
    assert result.diagnostics["refinements"] == result.visited
    assert result.diagnostics["cut_count"] == result.visited
    assert_evaluation_ownership(p, result, 2e-5)


def test_unresolved_inner_accuracy_never_claims_convergence(monkeypatch):
    p = small_problem()
    actual_solve = GateOracle.solve

    def always_coarse(self, z, **kwargs):
        answer = actual_solve(self, z, **kwargs)
        return replace(answer, value=answer.value-.001,
                       intercept=answer.intercept-.001, status="accuracy_limit")

    monkeypatch.setattr(GateOracle, "solve", always_coarse)
    result = solve(p, tolerance=1e-5, relative_tolerance=0, max_refinements=1)
    assert result.status == "oracle_accuracy_limit"
    assert result.oracle_calls == 2
    assert result.diagnostics["cut_count"] == result.visited == 1
    assert result.diagnostics["accurate_support_evaluations"] == 0
    assert not result.diagnostics["count_assumptions_met"]


def test_global_guard_escapes_uninformative_proposals(monkeypatch):
    p = Problem(np.zeros((2, 2)), np.array([[2., 0.], [2., 0.]]), 1, .1, .1)
    bad = p.mask((1,)).astype(float)
    monkeypatch.setattr(adaptive, "proposal_pool", lambda *args, **kwargs: [bad])
    result = solve(p, warm_support=(1,), root_rounds=10, tolerance=1e-5,
                   relative_tolerance=0, proposal_count=2)
    assert result.status == "optimal", result.to_dict()
    assert result.support == (0,)
    assert result.upper < .2
    assert result.diagnostics["master_calls"] > 0


def test_inaccurate_global_candidate_is_not_evaluated(monkeypatch):
    p = Problem([[0., 0.]], [[2., 0.]], 1, .1, .1)
    bad = p.mask((1,)).astype(float)
    monkeypatch.setattr(adaptive, "proposal_pool", lambda *args, **kwargs: [bad])
    monkeypatch.setattr(outer, "_master", lambda *args, **kwargs:
                        SimpleNamespace(status=1, x=np.r_[bad, 1.],
                                        fun=1., mip_dual_bound=0.))
    result = solve(p, warm_support=(1,), tolerance=1e-5, relative_tolerance=0)
    assert result.status == "time_limit"
    assert result.oracle_calls == 1
    assert result.lower == 0
    assert result.diagnostics["candidate_gap"] > result.diagnostics["master_target"]


def test_constant_bound_can_stop_without_an_oracle_call():
    p = small_problem()
    result = solve(p, tolerance=p.B, relative_tolerance=0)
    assert result.status == "optimal"
    assert result.oracle_calls == 0
    assert result.diagnostics["cut_count"] == 0


def test_accurate_initial_test_can_stop_before_proposals_or_master(monkeypatch):
    p = Problem([[0., 0.]], [[2., 0.]], 1, .1, .1)

    def forbidden(*args, **kwargs):
        raise AssertionError("search must stop after the sufficient initial test")

    monkeypatch.setattr(adaptive, "proposal_pool", forbidden)
    monkeypatch.setattr(outer, "_master", forbidden)
    result = solve(p, warm_support=(0,), tolerance=.2, relative_tolerance=0,
                   root_rounds=50)
    assert result.status == "optimal"
    assert result.oracle_calls == result.visited == 1
    assert result.upper == pytest.approx(.1, abs=1e-8)
    assert result.upper-result.lower < .2
    assert result.diagnostics["master_calls"] == 0
    assert result.diagnostics["accurate_support_evaluations"] == 1
    assert result.diagnostics["cut_count"] == 1


@pytest.mark.parametrize("keyword", ["proposal_count", "certify_every"])
def test_invalid_search_counts_are_rejected(keyword):
    with pytest.raises(ValueError):
        solve(small_problem(), **{keyword: 0})


def test_zero_movement_coordinates_control_the_number_of_evaluations():
    rng = np.random.default_rng(32)
    x0, x1 = np.zeros((3, 7)), np.zeros((4, 7))
    x0[:, :2], x1[:, :2] = rng.normal(size=(3, 2)), rng.normal(size=(4, 2)) + 1.
    p = Problem(x0, x1, 2, .18, .27)
    tolerance = 2e-5
    result = solve(p, root_rounds=40, tolerance=tolerance, relative_tolerance=0)
    assert result.status == "optimal", result.to_dict()
    weights = np.asarray(result.diagnostics["metric_weights"])
    assert np.count_nonzero(weights) == 2
    assert result.diagnostics["precision_coordinates"] <= 2
    assert result.diagnostics["accurate_evaluation_bound"] <= 4
    assert result.diagnostics["accurate_support_evaluations"] <= 4
    radius = result.diagnostics["separation_radius"]
    points = [np.asarray(row["point"]) for row in result.diagnostics["evaluations"]]
    for i, point in enumerate(points):
        for earlier in points[:i]:
            assert weights @ np.abs(point-earlier) >= radius-1e-8
    assert_evaluation_ownership(p, result, tolerance)
