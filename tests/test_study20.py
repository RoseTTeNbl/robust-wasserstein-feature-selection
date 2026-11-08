from itertools import combinations

import numpy as np
import pytest

from robusthw.algorithm import Problem, SolveOptions, fit_fixed
from robusthw.experiments.study20 import (
    Matching,
    compact_program,
    matching_lp,
    pair_costs,
    witness,
)


def _data(seed=0, n=7, dimension=5):
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=(n, dimension)), rng.normal(size=(n, dimension))
    x1[:, :2] += 1.0
    return x0, x1


@pytest.mark.parametrize("rho", [0.05, 0.4, 1.3, 50.0])
def test_cardinality_matchings_reproduce_the_matching_lp(rho):
    x0, x1 = _data()
    costs = pair_costs(x0, x1)
    for support in [(0,), (1, 3), (0, 2, 4)]:
        cost = costs[:, :, list(support)].sum(axis=2)
        assert Matching(cost).value(rho) == pytest.approx(matching_lp(cost, rho)[0], abs=1e-8)


def test_budget_inverts_the_value():
    x0, x1 = _data(1)
    matching = Matching(pair_costs(x0, x1)[:, :, [0, 3]].sum(axis=2))
    for mass in [0.2, 0.5, 6 / 7, 0.99]:
        assert matching.value(matching.budget(mass)) == pytest.approx(mass, abs=1e-9)


def test_compact_program_matches_enumeration():
    x0, x1 = _data(2, n=6)
    costs, rho = pair_costs(x0, x1), 0.6
    best = min(
        2 * Matching(costs[:, :, list(s)].sum(axis=2)).value(rho) for s in combinations(range(5), 2)
    )
    value, support, _ = compact_program(costs, rho, 2)
    assert value == pytest.approx(best, abs=1e-6)
    assert len(support) == 2
    relaxed, _, _ = compact_program(costs, rho, 2, relax=True)
    assert relaxed <= value + 1e-7


def test_matching_value_equals_core_fit():
    x0, x1 = _data(3, n=6)
    options = SolveOptions(fit_time_limit=60)
    for support in [(0,), (1, 4)]:
        fit = fit_fixed(Problem(x0, x1, [0.2, 0.3]), support, options, with_cut=False)
        cost = pair_costs(x0, x1)[:, :, list(support)].sum(axis=2)
        assert fit.upper == pytest.approx(2 * Matching(cost).value(0.5), abs=1e-6)


def test_witness_detector_meets_its_tangent():
    x0, x1 = _data(4, n=8)
    support, rho = (0, 1), 0.4
    m, u, v, lam = matching_lp(pair_costs(x0, x1)[:, :, list(support)].sum(axis=2), rho)
    score = witness(x1, support, v, lam)
    empirical = np.mean(1 - score(x0)) + np.mean(1 + score(x1))
    # Unperturbed risk is at most the tangent intercept 2(mean u + mean v) = V(S) - 2*lambda*rho.
    assert empirical <= 2 * (m - lam * rho) + 1e-7
    assert np.all(np.abs(score(x0)) <= 1)
