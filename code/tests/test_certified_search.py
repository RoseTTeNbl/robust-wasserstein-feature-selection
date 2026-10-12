"""Current default: complete binary solves and certified global masters only."""
from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
from test_oracle_new import original_transport_value

import robusthw.history as history_module
import robusthw.solver as outer
from robusthw import cli
from robusthw.oracle import GateOracle
from robusthw.problem import Problem


def problem(seed=7, loss="overlap"):
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=(3, 5)), rng.normal(size=(4, 5))
    x1[:, :2] += 1.1
    return Problem(x0, x1, 2, .17, .31, loss=loss,
                   q0=[.2, .3, .5], q1=[.1, .2, .3, .4])


@pytest.mark.parametrize("seed", [7, 13])
@pytest.mark.parametrize("use_fractions", [False, True])
def test_current_default_brackets_independent_original_transport_enumeration(seed, use_fractions):
    p = problem(seed)
    result = outer.solve(p, tolerance=2e-5, relative_tolerance=0,
                         use_fractions=use_fractions)
    # Initialize the shared HiGHS scheduler through the single-thread solver;
    # the independent reference changes no production constraints or bounds.
    risks = {support: original_transport_value(p, p.mask(support))
             for support in combinations(range(p.p), p.budget)}
    optimum = min(risks.values())
    assert result.status == "optimal", result.to_dict()
    assert result.lower <= optimum+2e-7 <= result.upper+2e-7
    assert result.upper-optimum <= 2e-5
    assert result.diagnostics["search"] == "certified_global_cuts_v1"
    rows = result.diagnostics["evaluations"]
    assert result.visited == result.diagnostics["cut_count"] == len(rows)
    assert len({tuple(row["support"]) for row in rows}) == len(rows)
    for row in rows:
        assert row["accurate"] and row["anchor_gap"] <= 2e-5/4
        assert row["actual_upper"] >= risks[tuple(row["support"])]-2e-7
        for support, risk in risks.items():
            z = p.mask(support)
            assert row["intercept"]+np.asarray(row["gradient"]) @ z <= risk+2e-7
            fraction = row["owned_fraction"]
            if fraction is not None:
                value = fraction["numerator"]/(fraction["tau"]+np.asarray(fraction["mu"]) @ z)
                assert value <= risk+2e-7
        if not use_fractions:
            assert row["owned_fraction"] is None
    for row in result.trace:
        if row["phase"] == "integer":
            assert row["model_lower"] <= row["pre_upper"]-2e-5/2
        if row["phase"] == "master" and "candidate_gap" in row:
            assert row["candidate_gap"] <= 2e-5/4


@pytest.mark.parametrize("use_fractions", [False, True])
def test_logistic_current_search_brackets_full_support_enumeration(use_fractions):
    pytest.importorskip("cvxpy")
    p = problem(loss="logistic")
    reference = outer.enumerate_supports(p, tolerance=2e-5, time_limit=90.)
    result = outer.solve(p, tolerance=2e-5, relative_tolerance=0,
                         time_limit=90., use_fractions=use_fractions)
    assert reference.status == result.status == "optimal"
    assert result.lower <= reference.upper+2e-7
    assert result.upper >= reference.lower-2e-7
    assert result.upper-reference.lower <= 2e-5
    assert result.diagnostics["model_reuse"]["models_built"] == 2


def test_default_has_no_forecast_pool_random_or_fractional_inner_solve(monkeypatch):
    p = problem()
    actual_solve, actual_master = GateOracle.solve, outer._ratio_master
    oracle_points, master_sizes = [], []

    def forbidden(*args, **kwargs):
        raise AssertionError("the certified default must not use a forecast or proposal pool")

    def binary_solve(self, z, **kwargs):
        assert not self.core_point
        assert np.all((z == 0) | (z == 1)) and np.sum(z) == p.budget
        oracle_points.append(p.support(z))
        return actual_solve(self, z, **kwargs)

    def master(problem, cuts, fractions, remaining, upper, **kwargs):
        master_sizes.append(len(cuts))
        return actual_master(problem, cuts, fractions, remaining, upper, **kwargs)

    monkeypatch.setattr(history_module.FixedKernelPredictor, "__init__", forbidden)
    monkeypatch.setattr(history_module, "proposal_pool", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(GateOracle, "reprice", forbidden)
    monkeypatch.setattr(GateOracle, "solve", binary_solve)
    monkeypatch.setattr(outer, "_ratio_master", master)
    result = outer.solve(p, root_rounds=100, tolerance=2e-5, relative_tolerance=0)
    assert result.status == "optimal", result.to_dict()
    assert len(oracle_points) == result.oracle_calls == result.visited
    assert len(set(oracle_points)) == result.visited
    assert master_sizes == list(range(1, len(master_sizes)+1))
    assert len(master_sizes) == result.diagnostics["master_calls"]
    assert result.diagnostics["forecast_model"] is None
    assert result.diagnostics["prediction_training_points"] == 0
    assert result.diagnostics["predicted_upper_updates"] == 0
    assert result.diagnostics["transferred_tests"] == 0
    assert result.diagnostics["fractional_root_calls"] == 0
    assert not {"acquisition", "root"} & {row["phase"] for row in result.trace}


def test_initial_mean_screen_uses_full_budget_and_stable_ties():
    p = Problem([[0., 0., 0.]], [[2., 0., 0.]], 2, .1, .1)
    result = outer.solve(p, tolerance=1e-5, relative_tolerance=0)
    assert result.status == "optimal"
    assert result.diagnostics["evaluations"][0]["support"] == [0, 1]
    assert all(sum(row["point"]) == 2 for row in result.diagnostics["evaluations"])


@pytest.mark.parametrize("recover", [False, True])
def test_refinement_stores_only_one_certificate_from_a_complete_solve(monkeypatch, recover):
    p = problem()
    actual_solve, attempts = GateOracle.solve, {}

    def coarse(self, z, **kwargs):
        answer = actual_solve(self, z, **kwargs)
        key = p.support(z)
        attempts[key] = attempts.get(key, 0)+1
        if not recover or attempts[key] == 1:
            return replace(answer, value=answer.value-.001, intercept=answer.intercept-.001,
                           status="accuracy_limit", certificate={**answer.certificate,
                           "margin": answer.certificate["margin"]+.001})
        return answer

    monkeypatch.setattr(GateOracle, "solve", coarse)
    result = outer.solve(p, tolerance=2e-5, relative_tolerance=0, max_refinements=1)
    assert result.oracle_calls == 2*result.visited
    assert result.diagnostics["cut_count"] == result.visited
    assert result.diagnostics["refinements"] == result.visited
    if recover:
        assert result.status == "optimal"
        assert all(row["accurate"] for row in result.diagnostics["evaluations"])
    else:
        assert result.status == "oracle_accuracy_limit"
        assert result.visited == 1 and result.lower == 0
        assert result.diagnostics["accurate_support_evaluations"] == 0
        assert not result.diagnostics["count_assumptions_met"]


@pytest.mark.parametrize("status,candidate", [(0, (0,)), (0, (1,)), (1, (0,))])
def test_inaccurate_or_repeated_master_candidate_is_not_evaluated(monkeypatch, status, candidate):
    p = Problem([[0., 0.]], [[2., 0.]], 1, .1, .1)
    z = p.mask(candidate)
    monkeypatch.setattr(outer, "_ratio_master", lambda *args, **kwargs:
                        SimpleNamespace(status=status, x=np.r_[z, 0., 0., 0.],
                                        fun=0., mip_dual_bound=0.))
    result = outer.solve(p, warm_support=(1,), tolerance=1e-5, relative_tolerance=0)
    assert result.status == ("time_limit" if status == 1 else "master_accuracy_limit")
    assert result.oracle_calls == result.visited == 1
    assert result.lower == 0
    assert result.diagnostics["failed_support"] == list(candidate)
    assert result.diagnostics["candidate_gap"] > result.diagnostics["master_target"]


def test_failed_master_cannot_supply_polluted_bound(monkeypatch):
    p = Problem([[0., 0.]], [[2., 0.]], 1, .1, .1)
    monkeypatch.setattr(outer, "_ratio_master", lambda *args, **kwargs:
                        SimpleNamespace(status=4, x=np.array([1., 0., 99.]),
                                        fun=99., mip_dual_bound=99.))
    result = outer.solve(p, warm_support=(1,), tolerance=1e-5, relative_tolerance=0)
    assert result.status == "master_failure"
    assert result.lower == 0 and result.oracle_calls == 1


def test_inconsistent_success_bound_is_rejected_without_corrupting_the_interval(monkeypatch):
    p = Problem([[0., 0.]], [[2., 0.]], 1, .1, .1)
    monkeypatch.setattr(outer, "_ratio_master", lambda *args, **kwargs:
                        SimpleNamespace(status=0, x=np.array([1., 0., 2.]),
                                        fun=2., mip_dual_bound=2.))
    result = outer.solve(p, warm_support=(1,), tolerance=1e-5)
    assert result.status == "numerical_inconsistency"
    assert result.lower == 0 and result.upper == p.B
    assert result.diagnostics["rejected_master_bound"] > result.upper


def test_nonbinary_master_point_is_rejected_without_inner_solve(monkeypatch):
    p = Problem([[0., 0.]], [[2., 0.]], 1, .1, .1)
    monkeypatch.setattr(outer, "_ratio_master", lambda *args, **kwargs:
                        SimpleNamespace(status=0, x=np.array([.2, .8, 0.]),
                                        fun=0., mip_dual_bound=0.))
    result = outer.solve(p, warm_support=(1,), tolerance=1e-5, relative_tolerance=0)
    assert result.status == "master_failure"
    assert result.oracle_calls == 1


def test_cli_selects_the_certified_default(tmp_path, monkeypatch):
    seen = []
    result = SimpleNamespace(status="optimal", support=(0, 1), lower=0., upper=0., seconds=0.,
                             to_dict=lambda: {"status": "optimal"})
    monkeypatch.setattr(cli, "solve", lambda *args, **kwargs: seen.append(kwargs) or result)
    assert cli.main(["--output", str(tmp_path/"record.json")]) == 0
    assert seen[0]["search"] == "cuts" and seen[0]["use_fractions"]
