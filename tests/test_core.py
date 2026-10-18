"""Mathematical cross-checks independent of the normalized-dual implementation."""

from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog

from robusthw.core import Problem, SolveOptions, constant_fit, fit_fixed
from robusthw.core import oracle as oracle_module


def transport_reference(problem, support):
    """Original transport LP: source masses and TWO total-cost constraints.

    No normalized masses, coordinate prices, source potentials or affine
    extension occur here. Its transported overlap equals the fixed-set risk.
    """
    n = problem.size
    count = n * n + n
    objective = np.r_[np.zeros(n * n), -np.ones(n)]
    equalities, equality_rhs, inequalities, rhs = [], [], [], []
    selected = list(support)
    distance = np.abs(problem.x[:, None, selected] - problem.x[None, :, selected]).sum(axis=2)
    for i in range(n):
        row = np.zeros(count)
        row[i * n : (i + 1) * n] = 1
        equalities.append(row)
        equality_rhs.append(problem.q[i])
    for k in range(2):
        sources = np.flatnonzero(problem.y == k)
        row = np.zeros(count)
        for i in sources:
            row[i * n : (i + 1) * n] = distance[i]
        inequalities.append(row)
        rhs.append(problem.radii[k])
        for r in range(n):
            row = np.zeros(count)
            row[n * n + r] = 1
            row[sources * n + r] = -1
            inequalities.append(row)
            rhs.append(0)
    result = linprog(
        objective, A_ub=inequalities, b_ub=rhs, A_eq=equalities, b_eq=equality_rhs, method="highs"
    )
    assert result.success, result.message
    return -result.fun


@pytest.mark.parametrize(
    "distance,radii", [(1.0, [0.2, 0.2]), (2.0, [0.1, 0.3]), (1.0, [0.8, 0.6])]
)
def test_two_atoms_have_analytic_robust_risk(distance, radii):
    problem = Problem([[0]], [[distance]], radii)
    fitted = fit_fixed(problem, [0])
    expected = min(1.0, sum(radii) / distance)
    assert fitted.status == "optimal"
    assert fitted.lower <= expected + 1e-9
    assert fitted.upper >= expected - 1e-9
    assert fitted.gap < 1e-8
    assert fitted.cut.evaluate([1]) == pytest.approx(fitted.lower, abs=1e-9)
    assert fitted.cut.evaluate([0]) <= 1 + 1e-9
    assert np.all(-fitted.cut.slope <= problem.price_bounds + 1e-9)


@pytest.mark.parametrize("seed", range(4))
def test_every_subset_and_every_cross_subset_cut_matches_original_transport(seed):
    rng = np.random.default_rng(720 + seed)
    problem = Problem(
        rng.normal(size=(2, 3)),
        rng.normal(size=(3, 3)),
        [0.15, 0.27],
        q0=[0.3, 0.7],
        q1=[0.2, 0.3, 0.5],
    )
    supports = [support for size in range(4) for support in combinations(range(3), size)]
    truth = {support: transport_reference(problem, support) for support in supports}
    for support in supports:
        fitted = fit_fixed(problem, support)
        assert fitted.status == "optimal", fitted.diagnostics
        assert fitted.lower <= truth[support] + 1e-8 <= fitted.upper + 1e-8
        assert fitted.gap < 1e-7
        assert fitted.cut is not None
        for other in supports:
            assert fitted.cut.evaluate(problem.mask(other)) <= truth[other] + 1e-8
        assert fitted.diagnostics["dual"]["scaled_violation"] < 1e-8


def test_duplicates_unequal_weights_zero_mass_and_uninformative_coordinates():
    problem = Problem(
        [[0, 2, 5], [0, 2, 5], [99, 99, 99]],
        [[0, 2, 5], [2, 2, 5]],
        [0.1, 0.3],
        q0=[0.2, 0.8, 0],
        q1=[0.7, 0.3],
    )
    assert problem.size == 4
    assert problem.price_bounds[1] == problem.price_bounds[2] == 0
    for support in [(), (0,), (1,), (0, 1, 2)]:
        fitted = fit_fixed(problem, support)
        expected = transport_reference(problem, support)
        assert fitted.status == "optimal", fitted.diagnostics
        assert fitted.lower <= expected + 1e-8 <= fitted.upper + 1e-8
        assert fitted.gap < 1e-7


def test_recovered_score_preserves_both_loss_envelopes_everywhere():
    rng = np.random.default_rng(918)
    problem = Problem(rng.normal(size=(3, 4)), rng.normal(size=(4, 4)) + 0.6, [0.14, 0.21])
    fitted = fit_fixed(problem, [0, 2])
    queries = rng.normal(size=(500, 4)) * 12
    scores = fitted.predict_score(queries)
    assert np.all(np.abs(scores) <= 1)
    for k in range(2):
        rows = problem.y == k
        distance = np.abs(problem.x[rows][:, None, [0, 2]] - queries[None, :, [0, 2]]).sum(axis=2)
        envelope = (fitted.source_bounds[rows, None] + fitted.prices[k] * distance).min(axis=0)
        loss = 0.5 * (1 + (-scores if k == 0 else scores))
        assert np.max(loss - envelope) <= 1e-9
    changed = queries.copy()
    changed[:, [1, 3]] += 10000
    np.testing.assert_array_equal(fitted.predict_score(queries), fitted.predict_score(changed))
    # Weighted source translations give adversaries obeying the original
    # class transport budgets. Risk must remain below the stored upper bound.
    risk = 0.0
    for k, source in enumerate((problem.x0, problem.x1)):
        shift = rng.normal(size=source.shape)
        q = problem.q0 if k == 0 else problem.q1
        shift *= problem.radii[k] / (q @ np.abs(shift).sum(axis=1))
        moved_score = fitted.predict_score(source + shift)
        risk += q @ (0.5 * (1 + (-moved_score if k == 0 else moved_score)))
    assert risk <= fitted.upper + 1e-9


def test_prediction_is_the_specified_distance_interpolation():
    problem = Problem([[0]], [[1]], [0.2, 0.2])
    fitted = fit_fixed(problem, [0])
    np.testing.assert_allclose(
        fitted.predict_score([[0], [0.25], [0.5], [0.75], [1], [3]]),
        [1, 0.5, 0, -0.5, -1, -1],
        atol=1e-9,
    )


def test_zero_time_retains_feasible_constant_test_and_explicit_status():
    problem = Problem([[0]], [[1]], [0.2, 0.2])
    fitted = fit_fixed(problem, [0], SolveOptions(time_limit=0))
    assert fitted.status == "time_limit"
    assert fitted.lower == 0 and fitted.upper == 1 and fitted.cut is None
    np.testing.assert_array_equal(fitted.predict_score([[0], [1], [200]]), [0, 0, 0])
    fallback = constant_fit(problem, (), "selection_timeout")
    assert fallback.status == "selection_timeout"
    assert fallback.support == ()


def test_suppressing_a_cut_preserves_the_same_numerical_fit():
    problem = Problem([[0]], [[1]], [0.2, 0.2])
    fitted = fit_fixed(problem, [0], with_cut=False)
    assert fitted.status == "optimal" and fitted.cut is None
    assert fitted.lower == pytest.approx(0.4, abs=1e-8)


@pytest.mark.parametrize("support", [(True,), (0.5,), (0, 0), (-1,), (2,)])
def test_invalid_supports_are_rejected(support):
    problem = Problem([[0, 0]], [[1, 1]], [0.2, 0.2])
    with pytest.raises(ValueError):
        fit_fixed(problem, support)


@pytest.mark.parametrize("radii", [[0, 0.2], [-1, 0.2], [np.nan, 0.2], [0.2]])
def test_radii_must_be_two_positive_finite_values(radii):
    with pytest.raises(ValueError):
        Problem([[0]], [[1]], radii)


def test_data_and_result_certificates_are_read_only():
    problem = Problem([[0]], [[1]], [0.2, 0.2])
    fitted = fit_fixed(problem, [0])
    for array in [
        problem.x0,
        problem.x,
        problem.q,
        problem.delta,
        fitted.scores,
        fitted.source_bounds,
        fitted.prices,
        fitted.cut.slope,
    ]:
        assert not array.flags.writeable


def test_failed_lp_results_do_not_supply_stale_primal_or_dual_certificates(monkeypatch):
    problem = Problem([[0]], [[1]], [0.2, 0.2])
    monkeypatch.setattr(
        oracle_module,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(
            status=4,
            x=np.full(len(args[0]), 99.0),
            fun=99.0,
            nit=0,
            message="simulated solve failure",
        ),
    )
    fitted = fit_fixed(problem, [0])
    assert fitted.status == "solver_error"
    assert fitted.lower == 0 and fitted.upper == 1 and fitted.cut is None
    np.testing.assert_array_equal(fitted.predict_score([[0], [1]]), [0, 0])
