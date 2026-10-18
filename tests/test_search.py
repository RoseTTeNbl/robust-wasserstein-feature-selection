"""Global certificates and heuristic bookkeeping, independently of search internals."""

from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

import robusthw.search as search
from robusthw.core import Cut, Problem, SolveOptions, constant_fit
from robusthw.experiments.data import synthetic_split
from robusthw.experiments.storage import array_hash
from robusthw.experiments.verification import transport_value as transport_reference


@pytest.mark.parametrize("seed", [3, 7, 11])
def test_global_and_enumeration_bracket_independent_transport_optimum(seed):
    rng = np.random.default_rng(seed)
    problem = Problem(
        rng.normal(size=(3, 5)),
        rng.normal(size=(4, 5)) + 0.7,
        [0.13, 0.19],
        q0=[0.2, 0.3, 0.5],
        q1=[0.1, 0.2, 0.3, 0.4],
    )
    truth = {
        support: transport_reference(problem, support)
        for size in range(3)
        for support in combinations(range(5), size)
    }
    optimum = min(truth.values())
    global_result = search.select(problem, 2, initial_support=(3, 4))
    enumeration = search.select(problem, 2, method="enumeration")
    for result in [global_result, enumeration]:
        assert result.status == "optimal", result.diagnostics
        assert result.lower <= optimum + 1e-8 <= result.upper + 1e-8
        assert result.gap <= SolveOptions().absolute_tolerance
        assert len(result.support) == 2
        assert result.fit.support == result.support
        for cut in result.cuts:
            for support, value in truth.items():
                assert cut.evaluate(problem.mask(support)) <= value + 1e-8
        assert result.oracle_seconds <= result.runtime
        assert result.master_seconds <= result.runtime
    lower_bounds = [row["lower"] for row in global_result.trace]
    assert all(left <= right for left, right in zip(lower_bounds, lower_bounds[1:]))
    upper_bounds = [row["upper"] for row in global_result.trace]
    assert all(left >= right for left, right in zip(upper_bounds, upper_bounds[1:]))
    assert enumeration.fit_count == 10
    assert enumeration.master_count == 0


@pytest.mark.parametrize("budget", [0, 3])
@pytest.mark.parametrize("method", ["global", "enumeration", "forward", "forward_swap"])
def test_empty_and_all_coordinate_budgets(budget, method):
    problem = Problem([[0, 0, 0]], [[2, 1, 0]], [0.1, 0.1])
    result = search.select(problem, budget, method)
    assert result.status in ("optimal", "complete")
    assert len(result.support) == budget
    assert result.fit_count >= 1
    if method in ("forward", "forward_swap"):
        assert result.lower is None and result.gap is None
    if budget == 0:
        assert result.upper == 1
        np.testing.assert_array_equal(result.fit.predict_score([[10, 20, 30]]), [0])


@pytest.mark.parametrize("method", ["global", "enumeration", "forward", "forward_swap"])
def test_zero_deadline_performs_no_post_timeout_fit(method):
    problem = Problem([[0, 0, 0]], [[2, 1, 0]], [0.1, 0.1])
    result = search.select(
        problem, 2, method, SolveOptions(time_limit=0), ranked_features=[2, 1, 0]
    )
    assert result.status in ("time_limit", "incomplete")
    assert result.fit_count == result.master_count == 0
    assert result.support == (1, 2)
    assert result.upper == 1


def _affine_fit(problem, support, options, with_cut=True):
    z = problem.mask(support)
    value = 0.6 - 0.4 * z[0]
    cut = Cut(0.6, np.array([-0.4, 0.0]))
    return replace(
        constant_fit(problem, support), upper=value, lower=value, status="optimal", cut=cut
    )


def test_time_limited_master_incumbent_fun_is_not_a_lower_bound(monkeypatch):
    problem = Problem([[0, 0]], [[2, 0]], [0.1, 0.1])
    monkeypatch.setattr(search, "fit_fixed", _affine_fit)
    monkeypatch.setattr(
        search,
        "milp",
        lambda *args, **kwargs: SimpleNamespace(
            status=1, x=np.array([0.0, 1.0, 0.55]), fun=0.55, mip_dual_bound=0.1
        ),
    )
    result = search.select(problem, 1, initial_support=(1,))
    assert result.status == "accuracy_limit"
    assert result.lower <= 0.1
    assert result.lower < 0.55
    assert result.fit_count == 1


@pytest.mark.parametrize("status", [2, 3, 4])
def test_failed_master_cannot_contribute_polluted_bound_or_incumbent(monkeypatch, status):
    problem = Problem([[0, 0]], [[2, 0]], [0.1, 0.1])
    monkeypatch.setattr(search, "fit_fixed", _affine_fit)
    monkeypatch.setattr(
        search,
        "milp",
        lambda *args, **kwargs: SimpleNamespace(
            status=status,
            x=np.array([1.0, 0.0, 99.0]),
            fun=99.0,
            mip_dual_bound=99.0,
            message="deliberate failure",
        ),
    )
    result = search.select(problem, 1, initial_support=(1,))
    assert result.status == "solver_error"
    assert result.lower == 0
    assert result.upper == pytest.approx(0.6)
    assert result.fit_count == 1
    assert result.master_count == (2 if status == 4 else 1)
    assert result.diagnostics["master_retries"] == (1 if status == 4 else 0)


def test_inconsistent_master_bound_preserves_previous_valid_interval(monkeypatch):
    problem = Problem([[0, 0]], [[2, 0]], [0.1, 0.1])
    monkeypatch.setattr(search, "fit_fixed", _affine_fit)
    monkeypatch.setattr(
        search,
        "milp",
        lambda *args, **kwargs: SimpleNamespace(
            status=0, x=np.array([1.0, 0.0, 2.0]), fun=2.0, mip_dual_bound=2.0
        ),
    )
    result = search.select(problem, 1, initial_support=(1,))
    assert result.status == "numerical_failure"
    assert result.lower == 0 and result.upper == pytest.approx(0.6)


def test_missing_global_bound_does_not_fall_back_to_master_fun(monkeypatch):
    problem = Problem([[0, 0]], [[2, 0]], [0.1, 0.1])
    monkeypatch.setattr(search, "fit_fixed", _affine_fit)
    monkeypatch.setattr(
        search,
        "milp",
        lambda *args, **kwargs: SimpleNamespace(status=0, x=np.array([0.0, 1.0, 0.6]), fun=0.6),
    )
    result = search.select(problem, 1, initial_support=(1,))
    assert result.status == "accuracy_limit" and result.lower == 0


def test_best_single_swap_improves_forward_and_cache_avoids_duplicate_fits(monkeypatch):
    problem = Problem([[0, 0, 0, 0]], [[1, 1, 1, 1]], [0.1, 0.1])
    risks = {
        (0,): 0.6,
        (1,): 0.7,
        (2,): 0.8,
        (3,): 0.9,
        (0, 1): 0.5,
        (0, 2): 0.55,
        (0, 3): 0.6,
        (1, 2): 0.1,
        (1, 3): 0.6,
        (2, 3): 0.2,
    }
    visited = []

    def fit(problem, support, options, with_cut=True):
        visited.append(support)
        return replace(
            constant_fit(problem, support),
            upper=risks[support],
            lower=risks[support],
            status="optimal",
        )

    monkeypatch.setattr(search, "fit_fixed", fit)
    forward = search.select(problem, 2, "forward")
    assert forward.support == (0, 1)
    visited.clear()
    swapped = search.select(problem, 2, "forward_swap")
    assert swapped.status == "complete" and swapped.support == (1, 2)
    assert swapped.upper < forward.upper
    assert swapped.diagnostics["accepted_swaps"] == 1
    assert len(visited) == len(set(visited)) == swapped.fit_count
    assert swapped.diagnostics["cache_hits"] > 0
    assert swapped.lower is None and swapped.gap is None


def test_forward_timeout_completes_budget_without_accepting_partial_candidate_pass(monkeypatch):
    problem = Problem([[0, 0, 0, 0]], [[1, 1, 1, 1]], [0.1, 0.1])
    clock, allowances = SimpleNamespace(now=0.0), []

    def fit(problem, support, options, with_cut=True):
        allowances.append(options.time_limit)
        clock.now += min(0.3, options.time_limit)
        return replace(constant_fit(problem, support), upper=0.6, lower=0.6, status="optimal")

    monkeypatch.setattr(search, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(search, "fit_fixed", fit)
    result = search.select(
        problem, 2, "forward", SolveOptions(time_limit=1), ranked_features=[3, 2, 1, 0]
    )
    assert result.status == "incomplete"
    assert result.support == (2, 3)
    assert result.upper == 1
    assert result.fit_count == 4
    assert result.diagnostics["completed_passes"] == 0
    np.testing.assert_allclose(allowances, [1.0, 0.7, 0.4, 0.1], atol=1e-12)


def test_enumeration_cannot_claim_optimality_with_an_unresolved_fit(monkeypatch):
    problem = Problem([[0, 0]], [[1, 1]], [0.1, 0.1])

    def fit(problem, support, options, with_cut=True):
        return replace(
            constant_fit(problem, support), upper=0.5, lower=0.49, status="accuracy_limit"
        )

    monkeypatch.setattr(search, "fit_fixed", fit)
    result = search.select(problem, 1, "enumeration")
    assert result.status == "accuracy_limit"
    assert result.diagnostics["enumeration_complete"]
    assert result.fit_count == 2 and result.lower == 0.49


@pytest.mark.parametrize("budget", [-1, 4, True, 1.5])
def test_invalid_budgets_are_rejected(budget):
    with pytest.raises(ValueError):
        search.select(Problem([[0, 0, 0]], [[1, 1, 1]], [0.1, 0.1]), budget)


def test_development_presolve_failure_regression_matches_original_data_and_enumeration():
    """Frozen development r000, retained from v1 before the numerical fix.

    The four-cut master produced HiGHS status 4 with presolve enabled. These
    are the original generation seed and validation-selected radii, rather
    than values tuned to make the corrected algorithm succeed.
    """
    data = synthetic_split("mean", 6, 5, 4, 3745672870)
    assert (
        array_hash(data.train0)
        == "9a3588064cddfb86cb42ac73f080729d15fc3f3801987aefdc562f6bd3a4e40b"
    )
    assert (
        array_hash(data.train1)
        == "6d56adb44526ab81cc6a3140bab9844e90060e539b9676605f26918a01807043"
    )
    problem = Problem(*data.train, [0.9968682458188717, 1.307542434238851])
    result = search.select(problem, 2, initial_support=(2, 4))
    enumerated = search.select(problem, 2, "enumeration")
    assert result.status == enumerated.status == "optimal"
    assert result.lower <= 1 <= result.upper
    assert enumerated.lower <= 1 <= enumerated.upper
    assert result.gap <= SolveOptions().absolute_tolerance
    assert (
        result.master_count
        == result.diagnostics["master_problem_count"] + result.diagnostics["master_retries"]
    )
    truth = {
        support: transport_reference(problem, support)
        for size in range(3)
        for support in combinations(range(6), size)
    }
    for cut in result.cuts:
        for support, value in truth.items():
            assert cut.evaluate(problem.mask(support)) <= value + 1e-8


def test_numerical_solve_error_retries_identical_master_once_and_records_actual_calls(monkeypatch):
    problem = Problem([[0, 0]], [[2, 0]], [0.1, 0.1])
    calls = []
    first = SimpleNamespace(
        status=4, x=np.ones(3), fun=99.0, mip_dual_bound=99.0, message="solve error"
    )

    def solve(objective, **kwargs):
        calls.append((objective, kwargs))
        return (
            first
            if len(calls) == 1
            else SimpleNamespace(
                status=0,
                x=np.array([1.0, 0.0, 0.2]),
                fun=0.2,
                mip_dual_bound=0.2,
                message="optimal",
            )
        )

    monkeypatch.setattr(search, "fit_fixed", _affine_fit)
    monkeypatch.setattr(search, "milp", solve)
    result = search.select(problem, 1, initial_support=(1,))
    assert result.status == "optimal" and result.support == (0,)
    assert result.master_count == 2 and result.diagnostics["master_problem_count"] == 1
    assert result.diagnostics["master_retries"] == 1
    assert len(calls) == 2 and calls[0][0] is calls[1][0]
    for name in ("integrality", "bounds", "constraints"):
        assert calls[0][1][name] is calls[1][1][name]
    assert "presolve" not in calls[0][1]["options"]
    assert calls[1][1]["options"]["presolve"] is False
    assert first.x is first.fun is first.mip_dual_bound is None
    log = result.diagnostics["master_attempts"][0]["attempts"]
    assert [item["status"] for item in log] == [4, 0]
    assert [item["presolve"] for item in log] == [True, False]
    assert all(item["seconds"] >= 0 for item in log)


@pytest.mark.parametrize("exhausted", [False, True])
def test_master_construction_and_retry_share_the_original_deadline(monkeypatch, exhausted):
    problem = Problem([[0, 0]], [[2, 0]], [0.1, 0.1])
    clock, calls = SimpleNamespace(now=0.0), []
    original_matrix = search.sparse.csr_matrix

    def matrix(*args, **kwargs):
        clock.now += 0.15
        return original_matrix(*args, **kwargs)

    def solve(objective, **kwargs):
        calls.append(kwargs)
        clock.now += 0.9 if exhausted else 0.2
        return SimpleNamespace(
            status=4 if len(calls) == 1 else 0,
            x=np.array([1.0, 0.0, 0.2]),
            fun=0.2,
            mip_dual_bound=0.2,
            message="test",
        )

    monkeypatch.setattr(search, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(search.sparse, "csr_matrix", matrix)
    monkeypatch.setattr(search, "milp", solve)
    result = search._master(problem, 1, [Cut(0.6, np.array([-0.4, 0.0]))], SolveOptions(), 1.0)
    assert calls[0]["options"]["time_limit"] == pytest.approx(0.85)
    if exhausted:
        assert len(calls) == result.master_solver_calls == 1
        assert result.status == 1 and result.mip_dual_bound is None
        assert result.master_retry_count == 0
    else:
        assert len(calls) == result.master_solver_calls == 2
        assert calls[1]["options"]["time_limit"] == pytest.approx(0.65)
        assert result.master_retry_count == 1


def test_forward_timeout_retains_best_already_evaluated_full_budget_fit(monkeypatch):
    problem = Problem([[0, 0, 0]], [[1, 1, 1]], [0.1, 0.1])
    clock = SimpleNamespace(now=0.0)

    def fit(problem, support, options, with_cut=True):
        clock.now += min(0.6, options.time_limit)
        risk = 0.4 if support == (0,) else 0.2
        return replace(constant_fit(problem, support), upper=risk, lower=risk, status="optimal")

    monkeypatch.setattr(search, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(search, "fit_fixed", fit)
    result = search.select(problem, 1, "forward", SolveOptions(time_limit=1))
    assert result.status == "incomplete" and result.support == (1,)
    assert result.upper == 0.2 and result.fit_count == 2
    assert result.diagnostics["completed_passes"] == 0
    assert result.diagnostics["returned_best_evaluated_full_support"]


def test_completed_swap_does_not_accept_improvement_below_declared_threshold(monkeypatch):
    problem = Problem([[0, 0]], [[1, 1]], [0.1, 0.1])

    def fit(problem, support, options, with_cut=True):
        risk = 0.5 if support == (0,) else 0.499999
        return replace(constant_fit(problem, support), upper=risk, lower=risk, status="optimal")

    monkeypatch.setattr(search, "fit_fixed", fit)
    result = search.select(
        problem, 1, "forward_swap", SolveOptions(swap_improvement=1e-4), initial_support=(0,)
    )
    assert result.status == "complete" and result.support == (0,)
    assert result.diagnostics["accepted_swaps"] == 0
