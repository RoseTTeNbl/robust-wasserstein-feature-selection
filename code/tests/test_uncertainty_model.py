"""Forecast conditioning and its separation from certified search bounds."""
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

from robusthw.history import EvaluationHistory, FixedKernelPredictor
from robusthw.oracle import GateOracle
from robusthw.problem import Problem


def forecast_problem(scale=1.):
    return SimpleNamespace(p=4, budget=2, B=scale)


def dense_conditioning(points, responses, queries, scale, noise, length):
    """Independent dense calculation, without the incremental factor API."""
    def covariance(left, right):
        return .25 * np.exp(-np.sum(np.abs(left[:, None]-right[None, :]), axis=2)/length)

    matrix = covariance(points, points)+noise*np.eye(len(points))
    cross = covariance(points, queries)
    mean = scale*(.5 + cross.T @ np.linalg.solve(matrix, responses/scale-.5))
    variance = .25-np.sum(cross*np.linalg.solve(matrix, cross), axis=0)
    return mean, scale*np.sqrt(np.maximum(0., variance)), matrix


def test_fixed_prior_and_hamming_kernel():
    predictor = FixedKernelPredictor(forecast_problem(2.), .02)
    points = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]])
    mean, sigma = predictor.predict(points)
    np.testing.assert_equal(mean, np.ones(3))
    np.testing.assert_equal(sigma, np.ones(3))
    np.testing.assert_allclose(predictor.kernel(points[:1], points),
                               [[.25, .25*np.exp(-1), .25*np.exp(-2)]])
    assert predictor.noise_variance == pytest.approx(.0001)
    assert predictor.factor_updates == predictor.factor_reuses == 0
    assert predictor.prediction_batches == 1


@pytest.mark.parametrize("scale", [1., 2.7])
def test_incremental_factor_and_batches_match_dense_conditioning(scale):
    predictor = FixedKernelPredictor(forecast_problem(scale), .013*scale)
    points = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 1]])
    responses = scale*np.array([.1, .3, .8, .2])
    queries = np.array([np.isin(np.arange(4), support).astype(float)
                        for support in combinations(range(4), 2)])
    for n, (point, response) in enumerate(zip(points, responses), start=1):
        previous = predictor.factor.copy()
        predictor.append(point, response)
        np.testing.assert_equal(predictor.factor[:n-1, :n-1], previous)
        mean, sigma = predictor.predict(queries)
        ref_mean, ref_sigma, matrix = dense_conditioning(
            points[:n], responses[:n], queries, scale,
            predictor.noise_variance, predictor.length_scale)
        np.testing.assert_allclose(predictor.factor @ predictor.factor.T, matrix,
                                   rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(predictor.alpha,
                                   np.linalg.solve(matrix, responses[:n]/scale-.5),
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(mean, ref_mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(sigma, ref_sigma, rtol=1e-11, atol=1e-12)
    assert predictor.factor_updates == predictor.factor_reuses == len(points)
    assert predictor.predicted_points == len(points)*len(queries)


def test_append_uses_cross_column_not_full_covariance_rebuild(monkeypatch):
    predictor = FixedKernelPredictor(forecast_problem(), .01)
    shapes = []
    kernel = predictor.kernel

    def watched(left, right):
        shapes.append((len(left), len(right)))
        return kernel(left, right)

    def forbidden(*args, **kwargs):
        raise AssertionError("the incremental update must not refactor a dense covariance")

    monkeypatch.setattr(predictor, "kernel", watched)
    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    for support, risk in [((0, 1), .1), ((0, 2), .2), ((2, 3), .8)]:
        predictor.append(np.isin(np.arange(4), support), risk)
    assert shapes == [(1, 1), (2, 1)]
    predictor.predict([[0, 1, 1, 0], [1, 0, 0, 1]])
    assert shapes[-1] == (3, 2)


def test_response_and_adjacency_produce_informative_candidate_ranking():
    low = FixedKernelPredictor(forecast_problem(), .01)
    high = FixedKernelPredictor(forecast_problem(), .01)
    observation = np.array([1, 1, 0, 0])
    queries = np.array([[1, 0, 1, 0], [0, 0, 1, 1]])
    low.append(observation, .1)
    high.append(observation, .9)
    means, std = low.predict(queries)
    other_means, other_std = high.predict(queries)
    assert means[0] < means[1]
    assert std[0] < std[1]
    assert other_means[0] > other_means[1]
    np.testing.assert_equal(other_std, std)
    # Both candidates pass a U=.1, epsilon=.04 screen for these cut values.
    # Pure cut minimization selects the far candidate; the forecast selects near.
    cut_lower = np.array([.03, 0.])
    assert np.all(cut_lower <= .1-.04/2)
    assert np.argmin(cut_lower) == 1
    assert np.argmin(means-std) == 0
    expected = np.sqrt(.25-.25**2*np.exp(-np.array([2., 4.]))/.2501)
    np.testing.assert_allclose(std, expected, atol=1e-14)
    old_std = std.copy()
    low.append(queries[1], .7)
    _, std = low.predict(queries)
    assert std[1] < .011
    assert np.all(std <= old_std)


def test_forecast_means_are_not_clipped_into_certified_risk_range():
    predictor = FixedKernelPredictor(SimpleNamespace(p=6, budget=3, B=1.), .01)
    points = np.array([[0, 1, 0, 0, 1, 1], [0, 0, 1, 1, 1, 0],
                       [1, 0, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 0, 1]])
    responses = np.array([0., 0., 1., 0., 1.])
    for point, risk in zip(points, responses):
        predictor.append(point, risk)
    queries = np.array([[0, 0, 1, 0, 1, 1]])
    mean, sigma = predictor.predict(queries)
    reference, ref_sigma, _ = dense_conditioning(points, responses, queries,
                                                1., predictor.noise_variance, 3.)
    assert mean[0] < 0
    np.testing.assert_allclose(mean, reference, atol=1e-12)
    np.testing.assert_allclose(sigma, ref_sigma, atol=1e-12)


def test_history_forecast_uses_only_completed_accurate_actual_evaluations():
    problem = Problem([[0., 0., 0.]], [[2., 1., 0.]], 1, .1, .1)
    history = EvaluationHistory(problem, anchor_target=1e-5)
    oracle = GateOracle(problem)
    point = problem.mask((0,))
    answer = oracle.solve(point)
    history.add(point, test=answer.test, intercept=answer.intercept,
                gradient=answer.gradient, accurate=True)
    np.testing.assert_equal(history.predictor.points, [point])
    np.testing.assert_equal(history.predictor.responses, [answer.test.upper])
    count = len(history.records)
    history.predict([problem.mask((1,)), problem.mask((2,))])
    assert len(history.records) == len(history.cuts) == count == 1
    assert history.predictor.factor_updates == 1
    uncertain = problem.mask((1,))
    answer = oracle.solve(uncertain)
    history.add(uncertain, test=answer.test, intercept=answer.intercept-.1,
                gradient=answer.gradient, accurate=False)
    assert len(history.records) == 2
    assert history.predictor.factor_updates == history.accurate_count == 1
    np.testing.assert_equal(history.predictor.responses, [history.uppers[0]])


@pytest.mark.parametrize("bad_point", [[.5, .5, 1, 0], [1, 0, 1], [1, 1, 0, np.nan]])
def test_predictor_rejects_nonbinary_or_mismatched_inputs(bad_point):
    predictor = FixedKernelPredictor(forecast_problem(), .01)
    with pytest.raises(ValueError, match="binary"):
        predictor.predict(bad_point)


@pytest.mark.parametrize("response", [-.1, 1.1, np.nan])
def test_predictor_rejects_invalid_actual_responses(response):
    predictor = FixedKernelPredictor(forecast_problem(), .01)
    with pytest.raises(ValueError, match="actual risk"):
        predictor.append([1, 1, 0, 0], response)
