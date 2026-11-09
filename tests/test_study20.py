import numpy as np
import pytest

from robusthw.algorithm import Problem, SolveOptions, fit_fixed
from robusthw.experiments.study20 import (
    BranchAndBound,
    compact_program,
    joint,
    log_ratio,
    marginal,
    robust_values,
    sample,
)


def _data(seed=0, n=7, dimension=5):
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=(n, dimension)), rng.normal(size=(n, dimension))
    x1[:, :2] += 1.0
    return x0, x1


def test_matching_values_equal_the_transport_lp_of_the_note():
    x0, x1 = _data()
    radii = np.array([0.05, 0.4, 1.3, 50.0])
    subsets, values = robust_values(x0, x1, 2, radii)
    for support, row in list(zip(subsets, values))[::3]:
        for rho, value in zip(radii, row):
            fit = fit_fixed(Problem(x0, x1, [rho / 2, rho / 2]), support, SolveOptions(),
                            with_cut=False)
            assert value == pytest.approx(fit.upper, abs=1e-6)


def test_joint_selection_is_the_minimum_and_never_worse_than_marginal():
    x0, x1 = _data(1)
    radii = np.array([0.2, 1.0])
    table = {d: robust_values(x0, x1, d, radii) for d in (1, 2)}
    subsets, values = table[2]
    for column in range(len(radii)):
        best = joint(table, 2, column)
        assert values[subsets.index(best), column] == values[:, column].min()
        assert values[subsets.index(best), column] <= values[subsets.index(
            marginal(table, 2, column)), column]


def test_compact_program_and_branch_and_bound_match_enumeration():
    x0, x1 = _data(2, n=6)
    costs, rho = np.abs(x0[:, None, :] - x1[None, :, :]), 0.6
    subsets, values = robust_values(x0, x1, 2, np.array([rho]))
    best = values[:, 0].min()
    value, support, _ = compact_program(costs, rho, 2)
    assert value == pytest.approx(best, abs=1e-6) and len(support) == 2
    assert compact_program(costs, rho, 2, relax=True)[0] <= value + 1e-7
    value, support, root = BranchAndBound(x0, x1, rho).run(2)
    assert value == pytest.approx(best, abs=1e-6) and root <= value + 1e-7


def test_sparse_mixture_classes_share_their_one_coordinate_marginals():
    x0, x1 = sample("interaction", 200_000, np.random.default_rng(0))
    levels = np.linspace(0.05, 0.95, 19)
    np.testing.assert_allclose(np.quantile(x0, levels, axis=0), np.quantile(x1, levels, axis=0),
                               atol=0.03)


@pytest.mark.parametrize("case", ["mean_shift", "mixture", "interaction"])
def test_bayes_rule_beats_a_coordinate_threshold(case):
    rng = np.random.default_rng(1)
    x0, x1 = sample(case, 20_000, rng)
    bayes = 0.5 * (np.mean(log_ratio(case, x0) > 0) + np.mean(log_ratio(case, x1) <= 0))
    single = min(0.5 * (np.mean(s * x0[:, 0] > 0) + np.mean(s * x1[:, 0] <= 0)) for s in (1, -1))
    assert bayes <= single + 0.01


@pytest.mark.parametrize("case", ["mean_shift", "mixture", "interaction"])
def test_distractors_are_the_same_in_both_classes_and_carry_no_information(case):
    x0, x1 = sample(case, 100_000, np.random.default_rng(2))
    levels = np.linspace(0.05, 0.95, 19)
    np.testing.assert_allclose(np.quantile(x0[:, 4:], levels, axis=0),
                               np.quantile(x1[:, 4:], levels, axis=0), atol=0.05)
    np.testing.assert_allclose(np.corrcoef(x0.T)[:4, 4:], 0, atol=0.02)
    # The Bayes rule ignores the distractors.
    shuffled = x0.copy()
    shuffled[:, 4:] = x1[:, 4:]
    np.testing.assert_allclose(log_ratio(case, shuffled), log_ratio(case, x0))
