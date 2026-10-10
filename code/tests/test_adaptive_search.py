"""Acquisition and history bounds checked against independent exact support solves."""
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

import robusthw.adaptive as adaptive
import robusthw.solver as outer
from robusthw.history import SearchHistory
from robusthw.oracle import GateOracle
from robusthw.problem import Problem
from robusthw.solver import enumerate_supports, solve


def small_problem(loss="overlap", seed=7):
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=(3, 5)), rng.normal(size=(4, 5))
    x1[:, :2] += 1.1
    return Problem(x0, x1, 2, .17, .31, loss=loss,
                   q0=np.array([.2, .3, .5]), q1=np.array([.1, .2, .3, .4]))


@pytest.mark.parametrize("loss", ["overlap", "logistic"])
def test_adaptive_selection_matches_independent_enumeration(loss):
    p = small_problem(loss)
    reference = enumerate_supports(p, time_limit=90., tolerance=2e-5)
    result = solve(p, search="adaptive", root_rounds=0, time_limit=90.,
                   tolerance=2e-5, relative_tolerance=0, proposal_count=8)
    assert result.status == "optimal", result.to_dict()
    assert result.lower <= reference.upper + 2e-7
    assert result.upper >= reference.lower - 2e-7
    assert result.upper-reference.lower <= 2.1e-5
    assert result.test is not None
    assert np.isfinite(result.test.prob_class0(p.points)).all()
    acquisitions = [entry for entry in result.trace if entry["phase"] == "acquisition"]
    for entry in acquisitions:
        assert entry["width"] > entry["information_threshold"]
        assert entry["model_lower"] < entry["upper"]
    assert result.diagnostics["transferred_tests"] > 0
    assert max(result.diagnostics["anchor_gaps"]) <= result.diagnostics["anchor_target"]


def test_history_width_satisfies_quantitative_geometry_bound():
    p = small_problem()
    oracle = GateOracle(p)
    history = SearchHistory(p, oracle)
    anchor = p.mask((0, 2))
    history.add(anchor, oracle.solve(anchor))
    delta = history.upper(anchor).upper-history.lower(anchor)
    for support in combinations(range(p.p), p.budget):
        z = p.mask(support)
        width = history.upper(z).upper-history.lower(z)
        assert width <= delta+p.B*history.distance(z, anchor)+1e-8


def test_cached_fraction_creates_valid_new_cuts_without_an_oracle():
    p = Problem(np.zeros((2, 3)), np.ones((2, 3)), 1, .2, .2)
    history = SearchHistory(p, object())
    history.add(p.mask((0,)), SimpleNamespace(
        intercept=.8, gradient=np.array([0., -.8, -1.6]), test=None,
        certificate={"entropy": .4, "tau": .5, "mu": np.array([0., .5, 1.])}))
    history.densify([p.mask((1,))])
    assert history.extra_cuts == 1
    assert history.lower(p.mask((1,))) == pytest.approx(.4, abs=1e-8)
    rng = np.random.default_rng(8)
    for z in rng.uniform(size=(100, 3)):
        actual = .4/(.5+np.array([0., .5, 1.]) @ z)
        assert history.lower(z) <= actual+1e-10


def test_global_guard_escapes_an_uninformative_proposal_pool(monkeypatch):
    p = Problem(np.zeros((2, 2)), np.array([[2., 0.], [2., 0.]]), 1, .1, .1)
    bad = p.mask((1,)).astype(float)
    monkeypatch.setattr(adaptive, "proposal_pool", lambda *args, **kwargs: [bad])
    result = solve(p, warm_support=(1,), root_rounds=0, tolerance=1e-5,
                   relative_tolerance=0, proposal_count=2)
    assert result.status == "optimal"
    assert result.support == (0,)
    assert result.upper < .2
    assert result.diagnostics["master_calls"] > 0


def test_invalid_acquisition_controls_are_rejected():
    p = small_problem()
    with pytest.raises(ValueError):
        solve(p, information_fraction=1.)
    with pytest.raises(ValueError):
        solve(p, proposal_count=0)


def test_interrupted_root_lp_objective_never_becomes_a_lower_bound(monkeypatch):
    p = small_problem()
    actual_master = outer._master

    def interrupted_root(problem, cuts, remaining, integer):
        if not integer:
            return SimpleNamespace(status=1, x=None, fun=problem.B)
        return actual_master(problem, cuts, remaining, integer)

    monkeypatch.setattr(outer, "_master", interrupted_root)
    result = solve(p, root_rounds=2, tolerance=2e-5, relative_tolerance=0)
    assert result.diagnostics["root_bound"] == 0
    root_records = [entry for entry in result.trace if entry["phase"] == "root_master"]
    assert root_records and root_records[0]["lower"] == 0


def test_constant_certificate_can_stop_without_an_oracle_call():
    p = small_problem()
    result = solve(p, tolerance=p.B, relative_tolerance=0)
    assert result.status == "optimal"
    assert result.oracle_calls == 0
    assert result.upper-result.lower == p.B


def test_root_binary_candidates_cannot_bypass_acquisition(monkeypatch):
    p = small_problem()
    actual_master = outer._master

    def binary_root(problem, cuts, remaining, integer):
        if not integer:
            return SimpleNamespace(status=0, x=np.r_[problem.mask((0, 1)), 0.], fun=0.)
        return actual_master(problem, cuts, remaining, integer)

    monkeypatch.setattr(outer, "_master", binary_root)
    result = solve(p, root_rounds=3, tolerance=2e-5, relative_tolerance=0)
    assert all(entry["support"] is None for entry in result.trace if entry["phase"] == "root")
    binary_calls = [entry for entry in result.trace if entry["phase"] in ("initial", "adaptive")]
    assert len(result.diagnostics["anchor_gaps"]) <= len(binary_calls)


def test_zero_movement_coordinates_give_a_nontrivial_evaluation_bound():
    rng = np.random.default_rng(32)
    x0, x1 = np.zeros((3, 7)), np.zeros((4, 7))
    x0[:, :2], x1[:, :2] = rng.normal(size=(3, 2)), rng.normal(size=(4, 2)) + 1.
    p = Problem(x0, x1, 2, .18, .27)
    result = solve(p, root_rounds=2, tolerance=2e-5, relative_tolerance=0)
    assert result.status == "optimal"
    assert np.count_nonzero(result.diagnostics["metric_weights"]) == 2
    # The 21 full-budget supports have only four patterns on the two
    # coordinates with positive movement weights.
    assert result.diagnostics["accurate_support_evaluations"] <= 4
