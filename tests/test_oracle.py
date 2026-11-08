"""The oracle Eq. (4) and the ratio cut Eq. (8) against an independent transport LP (Eq. (3))."""

from itertools import combinations

import numpy as np
import pytest

from robusthw.algorithm import Problem, SolveOptions, constant_fit, fit_fixed
from robusthw.experiments.verification import transport_value


@pytest.mark.parametrize(
    "distance,radii", [(1.0, [0.2, 0.2]), (2.0, [0.1, 0.3]), (1.0, [0.8, 0.6])]
)
def test_two_atoms_have_the_analytic_value(distance, radii):
    problem = Problem([[0]], [[distance]], radii)
    fitted = fit_fixed(problem, [0])
    expected = 2 * min(1.0, sum(radii) / distance)
    assert fitted.status == "optimal"
    assert fitted.lower == pytest.approx(expected, abs=1e-8)
    assert fitted.upper == pytest.approx(expected, abs=1e-8)
    assert fitted.cut.evaluate([1]) == pytest.approx(expected, abs=1e-8)
    assert fitted.cut.evaluate([0]) <= 2 + 1e-9


@pytest.mark.parametrize("seed", range(4))
def test_values_and_ratio_cuts_match_the_transport_lp_on_every_subset(seed):
    rng = np.random.default_rng(720 + seed)
    problem = Problem(rng.normal(size=(2, 3)), rng.normal(size=(3, 3)), [0.15, 0.27],
                      q0=[0.3, 0.7], q1=[0.2, 0.3, 0.5])
    supports = [s for size in range(4) for s in combinations(range(3), size)]
    truth = {s: transport_value(problem, s) for s in supports}
    for support in supports:
        fitted = fit_fixed(problem, support)
        assert fitted.status == "optimal"
        assert fitted.lower - 1e-8 <= truth[support] <= fitted.upper + 1e-8
        # Eq. (8): tight at its own selection and a lower bound on V everywhere else.
        assert fitted.cut.evaluate(problem.mask(support)) == pytest.approx(truth[support], abs=1e-7)
        for other in supports:
            assert fitted.cut.evaluate(problem.mask(other)) <= truth[other] + 1e-8


def test_duplicates_unequal_weights_and_zero_mass_rows():
    problem = Problem([[0, 2, 5], [0, 2, 5], [99, 99, 99]], [[0, 2, 5], [2, 2, 5]], [0.1, 0.3],
                      q0=[0.2, 0.8, 0], q1=[0.7, 0.3])
    assert problem.size == 4
    for support in [(), (0,), (1,), (0, 1, 2)]:
        fitted = fit_fixed(problem, support)
        assert fitted.lower - 1e-8 <= transport_value(problem, support) <= fitted.upper + 1e-8


def test_extension_keeps_both_loss_envelopes_everywhere():
    rng = np.random.default_rng(918)
    problem = Problem(rng.normal(size=(3, 4)), rng.normal(size=(4, 4)) + 0.6, [0.14, 0.21])
    fitted = fit_fixed(problem, [0, 2])
    queries = rng.normal(size=(500, 4)) * 12
    scores = fitted.predict_score(queries)
    for k in range(2):  # Eq. (5): ell(-g) <= E_0 and ell(g) <= E_1
        rows = problem.y == k
        distance = np.abs(problem.x[rows][:, None, [0, 2]] - queries[None, :, [0, 2]]).sum(axis=2)
        envelope = (fitted.source_bounds[rows, None] + fitted.prices[k] * distance).min(axis=0)
        assert np.max(np.maximum(0, 1 + (-scores if k == 0 else scores)) - envelope) <= 1e-9
    moved = queries.copy()
    moved[:, [1, 3]] += 1e4
    np.testing.assert_array_equal(scores, fitted.predict_score(moved))


def test_extension_interpolates_between_the_two_anchors():
    fitted = fit_fixed(Problem([[0]], [[1]], [0.2, 0.2]), [0])
    np.testing.assert_allclose(fitted.predict_score([[0], [0.25], [0.5], [0.75], [1], [3]]),
                               [1, 0.5, 0, -0.5, -1, -1], atol=1e-9)


def test_zero_time_returns_the_constant_detector():
    problem = Problem([[0]], [[1]], [0.2, 0.2])
    fitted = fit_fixed(problem, [0], SolveOptions(time_limit=0))
    assert (fitted.status, fitted.lower, fitted.upper, fitted.cut) == ("time_limit", 0, 2, None)
    np.testing.assert_array_equal(fitted.predict_score([[0], [1], [200]]), [0, 0, 0])
    assert constant_fit(problem, ()).support == ()


@pytest.mark.parametrize("support", [(0, 0), (-1,), (2,)])
def test_invalid_supports_are_rejected(support):
    with pytest.raises(ValueError):
        fit_fixed(Problem([[0, 0]], [[1, 1]], [0.2, 0.2]), support)


@pytest.mark.parametrize("radii", [[0, 0.2], [-1, 0.2], [0.2]])
def test_radii_must_be_two_positive_values(radii):
    with pytest.raises(ValueError):
        Problem([[0]], [[1]], radii)
