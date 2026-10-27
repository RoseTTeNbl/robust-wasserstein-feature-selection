"""Check the homogeneous selection function against an independent epigraph LP.

The reference below keeps every score unbounded and writes the manuscript's
constraints one at a time. It neither calls the production matrix builder nor
uses its gamma/mu dual, so these checks catch changes of formulation or scale.
"""

import warnings
from itertools import combinations

import numpy as np
import pytest
from scipy.optimize import linprog

from robusthw.core import Problem, SolveOptions, fit_fixed, fit_relaxed
from robusthw.experiments.verification import transport_value


def selection_primal_reference(problem, z):
    """Dense epigraph: min eta with objective<=eta and rho^T*w_m<=eta*z_m."""
    n, dimension = problem.size, problem.dimension
    a_start, lambda_start, w_start = n, 2 * n, 2 * n + 2
    eta_start = w_start + 2 * dimension
    count = eta_start + 1
    objective = np.zeros(count)
    objective[eta_start] = 1
    rows, rhs = [], []
    for source in range(n):
        k = int(problem.y[source])
        for target in range(n):
            row = np.zeros(count)
            row[target] = -1 if k == 0 else 1
            row[a_start + source] = -1
            row[w_start + k * dimension : w_start + (k + 1) * dimension] = -np.abs(
                problem.x[source] - problem.x[target]
            )
            rows.append(row)
            rhs.append(-1)
    for k in range(2):
        for m in range(dimension):
            row = np.zeros(count)
            row[w_start + k * dimension + m] = 1
            row[lambda_start + k] = -1
            rows.append(row)
            rhs.append(0)
    for m in range(dimension):
        row = np.zeros(count)
        for k in range(2):
            row[w_start + k * dimension + m] = problem.radii[k]
        row[eta_start] = -z[m]
        rows.append(row)
        rhs.append(0)
    row = np.zeros(count)
    row[a_start:lambda_start] = problem.q
    row[lambda_start:w_start] = problem.radii
    row[eta_start] = -1
    rows.append(row)
    rhs.append(0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*threads")
        result = linprog(
            objective,
            A_ub=rows,
            b_ub=rhs,
            bounds=[(None, None)] * n + [(0, None)] * (count - n),
            method="highs",
            options={"threads": 1},
        )
    assert result.success, result.message
    return float(result.fun)


def assert_direct_dual_certificate(problem, fitted, z):
    """Verify gamma and mu against D, without trusting solver residual fields."""
    dual = fitted.diagnostics["dual"]
    gamma = [np.asarray(dual[f"gamma{k}"]) for k in range(2)]
    mu = np.asarray(dual["mu"])
    alpha = dual["alpha"]
    assert mu.shape == (problem.dimension,)
    assert np.min(mu) >= 0
    target_masses = []
    for k, (source, q) in enumerate(((problem.x0, problem.q0), (problem.x1, problem.q1))):
        assert gamma[k].shape == (len(source), problem.size)
        assert np.min(gamma[k]) >= 0
        np.testing.assert_allclose(gamma[k].sum(axis=1), alpha * q, atol=1e-10, rtol=1e-10)
        distances = np.abs(source[:, None, :] - problem.x[None, :, :])
        cost = np.sum(gamma[k][:, :, None] * distances, axis=(0, 1))
        assert np.maximum(cost - problem.radii[k] * mu, 0).sum() <= problem.radii[k] * alpha + 1e-8
        target_masses.append(gamma[k].sum(axis=0))
    envelope = 2 * np.minimum(*target_masses).sum()
    assert dual["envelope"] == pytest.approx(envelope, abs=1e-10)
    np.testing.assert_allclose(fitted.cut.slope, -mu, atol=1e-12, rtol=1e-12)
    assert fitted.cut.intercept == pytest.approx(
        2 * np.sqrt(envelope) - alpha - dual["cut_margin"], abs=1e-10
    )
    assert fitted.lower == pytest.approx(max(0, fitted.cut.evaluate(z)), abs=1e-10)
    assert alpha >= 0


@pytest.mark.parametrize("seed", range(3))
def test_fractional_selection_matches_independent_primal_and_direct_dual(seed):
    rng = np.random.default_rng(6100 + seed)
    problem = Problem(
        rng.normal(size=(2, 3)),
        rng.normal(size=(3, 3)) + 0.7,
        [0.17, 0.29],
        q0=[0.25, 0.75],
        q1=[0.2, 0.3, 0.5],
    )
    selections = [np.zeros(3), np.ones(3), np.array([0, 0.35, 1]), rng.uniform(size=3)]
    truths = [selection_primal_reference(problem, z) for z in selections]
    for z, truth in zip(selections, truths):
        fitted = fit_relaxed(problem, z)
        assert fitted.status == "optimal", fitted.diagnostics
        assert fitted.lower <= truth + 1e-8 <= fitted.upper + 1e-8
        assert fitted.gap <= 1e-7
        assert_direct_dual_certificate(problem, fitted, z)
        for other, value in zip(selections, truths):
            assert fitted.cut.evaluate(other) <= value + 1e-8
        w = np.asarray(fitted.coordinate_weights)
        assert w.shape == (2, problem.dimension)
        assert np.min(w) >= 0
        assert np.max(w - fitted.prices[:, None]) <= 1e-9
        assert np.max(problem.radii @ w - fitted.upper * z) <= 1e-9
        for k in range(2):
            source_rows = problem.y == k
            distances = np.abs(problem.x[source_rows, None, :] - problem.x[None, :, :])
            costs = np.sum(distances * w[k], axis=2)
            losses = 1 + (-fitted.scores if k == 0 else fitted.scores)
            residual = losses - fitted.source_bounds[source_rows, None] - costs
            assert np.max(residual) <= 1e-9
        primal_value = problem.q @ fitted.source_bounds + problem.radii @ fitted.prices
        assert primal_value <= fitted.upper + 1e-10


def test_every_binary_point_matches_both_finite_transport_and_continuous_extension():
    problem = Problem([[0, 1, 2], [1, 0, 2]], [[2, 2, 2], [-1, 1, 2], [3, 0, 2]], [0.12, 0.23])
    supports = [s for size in range(4) for s in combinations(range(3), size)]
    values = {s: transport_value(problem, s) for s in supports}
    for support in supports:
        z = problem.mask(support)
        fixed = fit_fixed(problem, support)
        relaxed = fit_relaxed(problem, z)
        truth = values[support]
        for fitted in (fixed, relaxed):
            assert fitted.status == "optimal", fitted.diagnostics
            assert fitted.lower <= truth + 1e-8 <= fitted.upper + 1e-8
            assert fitted.gap < 1e-7
            assert_direct_dual_certificate(problem, fitted, z)
            for other, value in values.items():
                assert fitted.cut.evaluate(problem.mask(other)) <= value + 1e-8


def test_observed_projected_duplicates_receive_the_stored_common_score():
    problem = Problem([[0, 1], [0, -1]], [[0, 3], [2, 2], [2, 4]], [0.11, 0.19])
    fitted = fit_fixed(problem, (0,))
    assert fitted.status == "optimal", fitted.diagnostics
    predicted = fitted.predict_score(problem.x)
    np.testing.assert_allclose(predicted, fitted.scores, atol=1e-9)
    for coordinate in np.unique(problem.x[:, 0]):
        group = problem.x[:, 0] == coordinate
        assert np.ptp(predicted[group]) <= 1e-12
    changed = problem.x.copy()
    changed[:, 1] += np.arange(problem.size) * 1000
    np.testing.assert_array_equal(fitted.predict_score(changed), predicted)


@pytest.mark.parametrize("z", [[0.1], [0.1, 1.1], [-0.01, 0.2], [0.5, np.nan]])
def test_relaxed_selection_rejects_invalid_coordinates(z):
    problem = Problem([[0, 0]], [[1, 2]], [0.1, 0.2])
    with pytest.raises(ValueError):
        fit_relaxed(problem, z)


def test_relaxed_timeout_reports_constant_feasible_upper_without_false_cut():
    problem = Problem([[0, 0]], [[1, 2]], [0.1, 0.2])
    fitted = fit_relaxed(problem, [0.25, 0.75], SolveOptions(time_limit=0))
    assert fitted.status == "time_limit"
    assert fitted.lower == 0 and fitted.upper == 2
    assert fitted.cut is None


def test_fractional_value_is_convex_and_cuts_touch_at_the_evaluated_points():
    problem = Problem([[0, 1], [1, 0]], [[2, 2], [-1, 1], [3, 0]], [0.12, 0.23])
    left, right = np.array([0.1, 0.8]), np.array([0.7, 0.2])
    alpha = 0.37
    middle = alpha * left + (1 - alpha) * right
    results = [fit_relaxed(problem, z) for z in (left, right, middle)]
    assert all(result.status == "optimal" for result in results)
    assert results[2].upper <= alpha * results[0].upper + (1 - alpha) * results[1].upper + 1e-8
    for z, result in zip((left, right, middle), results):
        assert abs(result.cut.evaluate(z) - result.upper) < 1e-7
        for other, target in zip((left, right, middle), results):
            assert result.cut.evaluate(other) <= target.upper + 1e-8


def test_inexact_normalized_plan_is_repaired_to_a_global_cone_cut():
    from robusthw.core import Cut
    from robusthw.core._certificates import _repair_dual

    problem = Problem([[0, 1], [1, 0]], [[2, 2], [-1, 1], [3, 0]], [0.12, 0.23])
    n, d = problem.size, problem.dimension
    rng = np.random.default_rng(2726)
    raw = rng.normal(size=n * n + 3 * d + n + 1)
    raw[-1] = 0.4
    z = np.array([0.3, 0.7])
    repaired, info = _repair_dual(problem, raw, {"status": "time_limit"}, SolveOptions(), z)
    assert repaired is not None and info["repair_status"] == "accepted"
    h, mu, alpha, margin = repaired
    cut = Cut(2 * np.sqrt(h) - alpha - margin, -mu)
    for other in ([0, 0], [0, 1], [1, 0], [1, 1], [0.2, 0.4], [0.9, 0.6]):
        assert cut.evaluate(other) <= selection_primal_reference(problem, np.array(other)) + 1e-8
    raw[-1] = 0
    repaired, info = _repair_dual(problem, raw, {"status": "time_limit"}, SolveOptions(), z)
    assert repaired[0] == repaired[2] == repaired[3] == 0
    assert not np.any(repaired[1])
