import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.distance import pdist

from robusthw.experiments.baselines import (
    SelectionTimeout,
    mmd_squared,
    rank_features,
    select_features,
)


def test_mmd_matches_independent_scalar_formula_with_all_cross_pairs():
    left = np.array([[0.0], [0.4], [1.2]])
    right = np.array([[1.0], [2.0], [3.0], [4.0]])
    h = np.median(pdist(np.vstack((left, right))))

    def kernel(x, y):
        return np.exp(-float((x - y) @ (x - y)) / (2 * h * h))

    within0 = (
        sum(kernel(x, y) for i, x in enumerate(left) for j, y in enumerate(left) if i != j) / 6
    )
    within1 = (
        sum(kernel(x, y) for i, x in enumerate(right) for j, y in enumerate(right) if i != j) / 12
    )
    cross = sum(kernel(x, y) for x in left for y in right) / 12
    assert_allclose(mmd_squared(left, right), within0 + within1 - 2 * cross)


@pytest.mark.parametrize("method", ["marginal_mmd", "l1_logistic", "mrmr", "joint_mmd", "random"])
def test_each_method_fills_budget_deterministically(method):
    rng = np.random.default_rng(3)
    x0, x1 = rng.normal(size=(15, 6)), rng.normal(size=(15, 6))
    x1[:, 0] += 3
    a = select_features(method, x0, x1, 3, seed=11)
    b = select_features(method, x0, x1, 3, seed=11)
    assert a == b and len(a) == len(set(a)) == 3
    assert a == tuple(sorted(a))


def test_constant_data_bandwidth_and_ties_are_defined():
    x0 = x1 = np.ones((6, 4))
    assert mmd_squared(x0, x1) == 0
    assert rank_features(x0, x1) == (0, 1, 2, 3)
    for method in ("marginal_mmd", "mrmr", "joint_mmd", "l1_logistic"):
        assert select_features(method, x0, x1, 2) == (0, 1)


def test_joint_mmd_detects_sign_correlation_with_equal_marginals():
    values = np.tile(np.array([-2.0, -1.0, 1.0, 2.0]), 8)
    x0 = np.column_stack((values, values, np.zeros(len(values))))
    x1 = np.column_stack((values, -values, np.zeros(len(values))))
    assert mmd_squared(x0[:, [0]], x1[:, [0]]) < 0
    assert mmd_squared(x0[:, [0, 1]], x1[:, [0, 1]]) > 0.05
    assert select_features("joint_mmd", x0, x1, 2) == (0, 1)


def test_mrmr_merged_quantiles_do_not_fail_on_ties():
    x0, x1 = np.zeros((6, 3)), np.ones((6, 3))
    x0[:, 2] = x1[:, 2] = 0
    assert select_features("mrmr", x0, x1, 1) == (0,)


def test_exhausted_shared_budget_has_explicit_partial_support():
    with pytest.raises(SelectionTimeout) as caught:
        select_features("joint_mmd", np.zeros((5, 4)), np.ones((5, 4)), 2, time_limit=0)
    assert caught.value.partial_support == ()


def test_timeout_preserves_last_complete_joint_selection_stage(monkeypatch):
    from robusthw.experiments import baselines

    elapsed = [0.0]
    monkeypatch.setattr(baselines, "perf_counter", lambda: elapsed[0])

    def score(left, right):
        elapsed[0] += 1
        return float(np.mean(right - left))

    monkeypatch.setattr(baselines, "mmd_squared", score)
    with pytest.raises(SelectionTimeout) as caught:
        select_features("joint_mmd", np.zeros((4, 3)), np.ones((4, 3)), 2, time_limit=3.5)
    assert caught.value.partial_support == (0,)
