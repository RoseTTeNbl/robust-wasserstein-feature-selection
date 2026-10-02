"""Independent checks of data isolation, literature coefficients, and metrics."""

from itertools import combinations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from robusthw.experiments.baselines import (
    coordinate_contrasts,
    screen_mmd,
    select,
    sparse_trust_region,
    trust_region_maximum,
    wang_coefficients,
)
from robusthw.experiments.data import Standardizer, real_trial, synthetic_trial
from robusthw.experiments.metrics import (
    permutation_mmd,
    probability_metrics,
    rbf_gram,
    stability_jaccard,
    support_metrics,
)


@pytest.mark.parametrize("mechanism", ["mean", "variance", "dependence", "mixture"])
def test_selection_only_standardization_and_seed_replay(mechanism):
    trial = synthetic_trial(mechanism, 7, n_select=8, n_head=5, n_classification=9)
    replay = synthetic_trial(mechanism, 7, n_select=8, n_head=5, n_classification=500)
    assert_allclose(trial.selection[0], replay.selection[0])
    assert_allclose(trial.head[0], replay.head[0])
    pooled = np.vstack(trial.selection)
    assert_allclose(pooled.mean(0), 0, atol=1e-14)
    assert_allclose(pooled.std(0, ddof=1), 1, atol=1e-14)
    assert trial.semantic_support(trial.true_support) == [0, 1, 2]
    assert not np.array_equal(trial.selection[0][:5], trial.head[0])
    assert_allclose(trial.sample_testing_pair(3, 20)[0], replay.sample_testing_pair(3, 20)[0])


def test_real_splits_are_disjoint_and_keep_constant_columns(tmp_path):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 5))
    x[:, 0] = 2
    path = tmp_path / "fixture.npz"
    np.savez(path, x=x, y=np.r_[np.zeros(40), np.ones(40)])
    trial = real_trial(path, 2, n_select=8, n_head=6)
    groups = [set(sum(trial.split_indices[key], [])) for key in ("selection", "head", "testing")]
    assert all(not left & right for left, right in combinations(groups, 2))
    assert len(set.union(*groups)) == 80
    assert_allclose(trial.selection[0][:, 0], 0)
    assert np.isfinite(trial.testing[1]).all()


def test_wang_coefficients_match_independent_row_covariance_and_revision_sign():
    x = np.array([[0.2, 1, 0], [1, 0.4, 2], [-0.3, 0, 1], [0.7, 1.8, -0.4]])
    y = np.array([[0.7, 0.5, 1], [1.2, 0.9, 3], [-0.1, 0.1, 1.4], [1.4, 2.1, 0.2]])
    a, t, width, revision = wang_coefficients(x, y, 0.1)
    # Scalar kernel evaluation independent of the production vectorized assembly.
    h = np.empty((4, 4, 3))
    for i in range(4):
        for j in range(4):
            for k in range(3):
                def kernel(u, v):
                    return np.exp(-(u - v) ** 2 / (2 * width[k]))
                h[i, j, k] = (kernel(x[i, k], x[j, k]) + kernel(y[i, k], y[j, k])
                              - kernel(x[i, k], y[j, k]) - kernel(y[i, k], x[j, k]))
    expected_variance = 4 * np.cov(h.mean(1), rowvar=False, bias=True)
    expected_t = np.mean([h[i, j] for i in range(4) for j in range(4) if i != j], axis=0)
    assert_allclose(a, -0.1 * expected_variance, atol=1e-14)
    assert_allclose(t, expected_t, atol=1e-14)
    # Author revision's formula is +4 lambda Cov, then a harmless eigen shift.
    assert_allclose(revision, 0.1 * expected_variance, atol=1e-12)
    assert np.linalg.eigvalsh(a).max() <= 1e-12


@pytest.mark.parametrize("seed", range(5))
def test_global_trust_region_against_dense_two_dimensional_search(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(2, 2))
    a = (a + a.T) / 2
    t = rng.normal(size=2)
    x, value = trust_region_maximum(a, t)
    angles = np.linspace(0, 2 * np.pi, 100001)
    candidates = np.c_[np.cos(angles), np.sin(angles)]
    dense_maximum = np.max(np.einsum("ij,jk,ik->i", candidates, a, candidates) + candidates @ t)
    assert_allclose(np.linalg.norm(x), 1, atol=1e-12)
    assert value >= dense_maximum - 1e-10
    assert value <= dense_maximum + 1e-7


def test_trust_region_hard_case_and_sparse_support():
    a = np.diag([3, 1, -2])
    t = np.array([0, 1, 0])
    x, value = trust_region_maximum(a, t)
    assert_allclose(x[1], 0.25, atol=1e-12)
    assert_allclose(value, 3.125, atol=1e-12)
    support, weights, objective, status, _ = sparse_trust_region(a, t, 2)
    assert support == (0, 1)
    assert status == "exact_support_enumeration"
    assert_allclose(objective, value)
    assert_allclose(np.linalg.norm(weights), 1)


def test_budget_and_zero_bandwidth_baselines():
    x, y = np.ones((5, 4)), np.ones((5, 4))
    h, widths = coordinate_contrasts(x, y)
    assert_allclose(h, 0)
    assert (widths > 0).all()
    assert screen_mmd(x, y, 2) == (0, 1)
    result = select("wang", x, y, 2)
    assert len(result["support"]) <= 2
    assert_allclose(np.linalg.norm(result["weights"]), 1)
    assert select("random", x, y, 2, seed=7)["support"] == select("random", x, y, 2, seed=7)["support"]


def test_unbiased_mmd_observed_statistic_and_plus_one_calibration():
    x = np.array([[0.0], [0.4], [1.2], [2], [2.7], [3]])
    gram = rbf_gram(x, 1)
    result = permutation_mmd(gram, 3, permutations=199, seed=4)
    direct = ((gram[:3, :3].sum() - np.trace(gram[:3, :3])) / 6
              + (gram[3:, 3:].sum() - np.trace(gram[3:, 3:])) / 6
              - 2 * gram[:3, 3:].mean())
    assert_allclose(result["statistic"], direct)
    assert 0 < result["pvalue"] <= 1
    assert_allclose(result["pvalue"] * 200, round(result["pvalue"] * 200))
    assert permutation_mmd(np.ones((6, 6)), 3, permutations=199, seed=8)["pvalue"] == 1


def test_support_and_probability_metrics_are_distinct():
    metrics = support_metrics((0, 2), (0, 1))
    assert metrics["fdp"] == metrics["ndp"] == metrics["f1"] == 0.5
    assert stability_jaccard([(0, 1), (1, 2)]) == 1 / 3
    prediction = probability_metrics(np.array([0.9, 0.5]), np.array([0.1, 0.5]))
    assert prediction["false_positive_rate"] == 0
    assert prediction["false_negative_rate"] == 0.5
    assert_allclose(prediction["randomized_balanced_error"], 0.3)
    assert_allclose(prediction["half_hinge_surrogate_sum"], 0.6)


def test_standardizer_fit_does_not_use_later_samples():
    transform = Standardizer.fit(np.array([[0, 2], [1, 2]]), np.array([[2, 2], [3, 2]]))
    assert_allclose(transform.center, [1.5, 2])
    assert np.isfinite(transform.transform(np.array([[1000, 2]]))).all()


def test_trial_record_resume_requires_the_same_configuration(tmp_path, monkeypatch):
    import json

    from robusthw.experiments import runner

    trial = synthetic_trial("mean", 1, dimension=3, signal_size=1,
                            n_select=3, n_head=3, n_classification=3)
    monkeypatch.setattr(runner, "_trial_jobs", lambda config, root: iter([trial]))
    monkeypatch.setattr(runner, "_select", lambda *args: {
        "support": (0,), "status": "marginal_screening", "seconds": 0.0, "weights": None})
    monkeypatch.setattr(runner, "evaluate", lambda *args: {"metrics": {"power": 0.5}})
    config = {"methods": ["mmd"], "budget": 1}
    first = runner.run_statistical(config, tmp_path, tmp_path)
    second = runner.run_statistical(config, tmp_path, tmp_path)
    assert first == {"completed": 1, "skipped": 0, "failures": 0}
    assert second == {"completed": 0, "skipped": 1, "failures": 0}
    saved = json.loads(next((tmp_path / "records").glob("*.json")).read_text())
    assert saved["metrics"]["power"] == 0.5
    assert saved["preprocessing"]["center"] == trial.standardizer.center.tolist()
    with pytest.raises(ValueError, match="Resume configuration"):
        runner.run_statistical({**config, "theta": 0.5}, tmp_path, tmp_path)


def test_trial_failure_is_saved_not_discarded(tmp_path, monkeypatch):
    import json

    from robusthw.experiments import runner

    trial = synthetic_trial("mean", 1, dimension=3, signal_size=1,
                            n_select=3, n_head=3, n_classification=3)
    monkeypatch.setattr(runner, "_trial_jobs", lambda config, root: iter([trial]))

    def fail(*args):
        raise RuntimeError("deliberate solver failure")

    monkeypatch.setattr(runner, "_select", fail)
    summary = runner.run_statistical({"methods": ["risk"]}, tmp_path, tmp_path)
    record = json.loads(next((tmp_path / "records").glob("*.json")).read_text())
    assert summary["failures"] == 1
    assert record["run_status"] == "failure"
    assert record["error_type"] == "RuntimeError"


def test_greedy_timeout_returns_labeled_budget_feasible_fallback(monkeypatch):
    from robusthw import oracle
    from robusthw.experiments.runner import robust_greedy
    from robusthw.problem import Problem

    class TimeoutOracle:
        def __init__(self, problem):
            self.problem = problem

        def fixed_support(self, *args, **kwargs):
            raise TimeoutError("deliberate timeout")

    monkeypatch.setattr(oracle, "GateOracle", TimeoutOracle)
    x0, x1 = np.zeros((3, 4)), np.ones((3, 4))
    result = robust_greedy(Problem(x0, x1, 2, 0.25, 0.25, "overlap"))
    assert result["status"] == "greedy_time_limit"
    assert len(result["support"]) == 2
    assert result["oracle_calls"] == 1
