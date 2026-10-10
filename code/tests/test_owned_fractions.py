"""Same-evaluation fractions strengthen search without new inner solves."""
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
from test_oracle_new import original_transport_value

import robusthw.adaptive as adaptive
import robusthw.solver as outer
from robusthw.history import EvaluationHistory
from robusthw.oracle import GateOracle
from robusthw.problem import Problem


def safe_fraction(answer):
    certificate = answer.certificate
    return (max(0., certificate["entropy"]-certificate["margin"]),
            certificate["tau"], certificate["mu"])


def problem():
    rng = np.random.default_rng(13)
    x0, x1 = rng.normal(size=(3, 5)), rng.normal(size=(4, 5))
    x1[:, :2] += 1.2
    return Problem(x0, x1, 2, .18, .27)


def test_owned_fraction_and_tangent_cover_independent_original_risks():
    p = problem()
    history = EvaluationHistory(p)
    point = p.mask((3, 4))
    answer = GateOracle(p).solve(point)
    history.add(point, test=answer.test, intercept=answer.intercept,
                gradient=answer.gradient, accurate=True, fraction=safe_fraction(answer))
    numerator, tau, mu = history.fractions[0]
    assert history.metric_weights is p.price_bounds
    assert not history.metric_weights.flags.writeable
    assert numerator / (tau+mu @ point) == pytest.approx(answer.value, abs=1e-9)
    # The original transport LP does not use the gate or stored lower model.
    for support in combinations(range(p.p), p.budget):
        z = p.mask(support)
        actual = original_transport_value(p, z)
        fraction = numerator / (tau+mu @ z)
        affine = max(0., answer.intercept+answer.gradient @ z)
        assert fraction <= actual+1e-7
        assert history.lower(z) <= actual+1e-7
        assert history.lower(z) >= affine
    assert len(history.points) == len(history.cuts) == len(history.fractions) == 1
    np.testing.assert_equal(history.cuts[0][1], answer.gradient)
    saved_prices = mu.copy()
    answer.certificate["mu"][:] = 99.  # Retained prices must not alias the caller.
    np.testing.assert_equal(history.fractions[0][2], saved_prices)


def test_downward_margin_requires_retaining_the_affine_bound_too():
    p = Problem([[0., 0.]], [[0., 0.]], 1, .1, .1)
    history = EvaluationHistory(p)
    owner, query = p.mask((0,)), p.mask((1,))
    h, margin, tau, mu = .8, .1, .9375, np.array([.0625, 0.])
    assert tau+mu @ owner == 1
    gradient = -h*mu
    lower = h-margin
    history.add(owner, test=SimpleNamespace(support=(0,), upper=.8),
                intercept=lower-gradient @ owner, gradient=gradient,
                accurate=False, fraction=(lower, tau, mu))
    affine = lower+gradient @ (query-owner)
    fraction = lower/(tau+mu @ query)
    assert fraction < affine
    assert history.lower(query) == pytest.approx(affine)
    # Changing the old gradient to -(h-margin)*mu would lose its exact provenance.
    np.testing.assert_equal(history.cuts[0][1], -h*mu)


@pytest.mark.parametrize("fraction", [(-.1, 1., [0., 0.]),
                                       (.5, 0., [0., 0.]),
                                       (.5, np.nan, [0., 0.]),
                                       (.5, 1., [-1., 0.]),
                                       (.5, 1., [np.nan, 0.]),
                                       (.5, 1., [0.]),
                                       (.7, 1., [0., 0.])])
def test_invalid_owned_fraction_does_not_partially_mutate_history(fraction):
    p = Problem([[0., 0.]], [[0., 0.]], 1, .1, .1)
    history = EvaluationHistory(p)
    with pytest.raises(ArithmeticError, match="fraction"):
        history.add(p.mask((0,)), test=SimpleNamespace(support=(0,), upper=.5),
                    intercept=.5, gradient=np.zeros(2), accurate=True, fraction=fraction)
    assert not history.records and not history.cuts and not history.fractions
    assert not history.tests and not history.points
    assert history.predictor.factor_updates == 0


def test_full_fraction_storage_does_not_change_the_predictor():
    p = problem()
    plain, strengthened = EvaluationHistory(p), EvaluationHistory(p)
    oracle = GateOracle(p)
    for support in [(3, 4), (0, 1)]:
        z = p.mask(support)
        answer = oracle.solve(z)
        for history, fraction in [(plain, None), (strengthened, safe_fraction(answer))]:
            history.add(z, test=answer.test, intercept=answer.intercept,
                        gradient=answer.gradient, accurate=True, fraction=fraction)
    queries = np.array([p.mask(s) for s in combinations(range(p.p), p.budget)])
    for left, right in zip(plain.predict(queries), strengthened.predict(queries)):
        np.testing.assert_equal(left, right)
    np.testing.assert_equal(plain.predictor.factor, strengthened.predictor.factor)
    np.testing.assert_equal(plain.predictor.responses, strengthened.predictor.responses)
    assert len(plain.cuts) == len(strengthened.cuts) == 2


def test_periodic_ratio_master_receives_the_true_incumbent_witness(monkeypatch):
    p = Problem([[0., 0.]], [[2., 0.]], 1, .1, .1)
    bad = p.mask((1,))
    monkeypatch.setattr(adaptive, "proposal_pool", lambda *args, **kwargs: [bad])
    real_master, seen = outer._ratio_master, []

    def watched(problem, cuts, fractions, remaining, upper, *, witness):
        seen.append((np.asarray(witness).copy(), upper, list(fractions)))
        return real_master(problem, cuts, fractions, remaining, upper, witness=witness)

    monkeypatch.setattr(outer, "_ratio_master", watched)
    result = outer.solve(p, warm_support=(1,), tolerance=1e-5, relative_tolerance=0)
    assert result.status == "optimal", result.to_dict()
    assert result.support == (0,) and result.oracle_calls == result.visited == 2
    assert seen and np.array_equal(seen[0][0], bad)
    assert seen[0][1] == pytest.approx(p.B)
    assert all(np.all((point == 0) | (point == 1)) and point.sum() == p.budget
               for point, _, _ in seen)
    assert result.diagnostics["fraction_count"] == result.visited


def test_unresolved_ratio_master_error_cannot_become_a_global_bound(monkeypatch):
    p = Problem([[0., 0.]], [[2., 0.]], 1, .1, .1)
    monkeypatch.setattr(adaptive, "proposal_pool", lambda *args, **kwargs: [p.mask((1,))])
    monkeypatch.setattr(outer, "_ratio_master", lambda *args, **kwargs:
                        SimpleNamespace(status=4, x=None, fun=99., mip_dual_bound=99.))
    result = outer.solve(p, warm_support=(1,), tolerance=1e-5, relative_tolerance=0)
    assert result.status == "master_failure"
    assert result.lower == 0 and result.oracle_calls == 1
    assert result.diagnostics["fraction_count"] == result.visited == 1
