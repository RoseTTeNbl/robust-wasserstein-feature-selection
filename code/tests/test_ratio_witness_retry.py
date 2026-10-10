"""An infeasible status permits retry only when the exact model has a witness."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult

import robusthw.solver as outer


@pytest.fixture
def saved_model():
    return json.loads((Path(__file__).parent / 'fixtures/ratio_incumbent_boundary.json').read_text())


def invoke(saved, witness=None, remaining=10.):
    p = SimpleNamespace(p=saved['dimension'], budget=saved['budget'], B=saved['B'])
    cuts = [(intercept, np.asarray(gradient)) for intercept, gradient in saved['cuts']]
    ratios = [(h, tau, np.asarray(mu)) for h, tau, mu in saved['ratios']]
    return outer._ratio_master(p, cuts, ratios, remaining, saved['upper'], witness=witness)


def failed(status=2):
    # Deliberately polluted solver fields must never become a master bound.
    return OptimizeResult(status=status, success=False, message='simulated solver failure',
                          x=np.ones(13), fun=99., mip_dual_bound=99.)


def test_saved_boundary_model_retries_identical_model_with_real_incumbent(monkeypatch, saved_model):
    real_milp, calls = outer.milp, []
    first = failed()

    def solve(c, **kwargs):
        calls.append((c, kwargs))
        return first if len(calls) == 1 else real_milp(c, **kwargs)

    monkeypatch.setattr(outer, 'milp', solve)
    result = invoke(saved_model, saved_model['witness'])
    assert result.status == 0
    assert result.fun == pytest.approx(saved_model['lower_model_at_witness'], abs=1e-8)
    assert np.flatnonzero(np.rint(result.x[:6])).tolist() == [0, 4]
    assert len(calls) == 2 and result.master_retries == 1
    assert calls[0][0] is calls[1][0]
    for key in ('bounds', 'constraints', 'integrality'):
        assert calls[0][1][key] is calls[1][1][key]
    assert calls[1][1]['options']['presolve'] is False
    proof = result.master_retry_log[0]['initial']['witness']
    assert proof['verified'] and proof['row_violation'] == 0
    assert proof['bound_violation'] == proof['integrality_violation'] == 0
    assert first.x is None and first.fun is None and first.mip_dual_bound is None


@pytest.mark.parametrize('witness', [None, [1, 1, 0, 0, 0, 0], [.5, .5, 0, 0, 1, 0],
                                    [2, 0, 0, 0, 0, 0], [np.nan, 0, 0, 0, 1, 0]])
def test_absent_or_invalid_witness_does_not_authorize_retry(monkeypatch, saved_model, witness):
    calls, first = [], failed()
    monkeypatch.setattr(outer, 'milp', lambda *args, **kwargs: calls.append(kwargs) or first)
    result = invoke(saved_model, witness)
    assert result.status == 2 and len(calls) == 1
    assert result.x is None and result.fun is None and result.mip_dual_bound is None
    if witness is not None:
        assert result.master_retries == 0
        assert not result.master_retry_log[0]['witness']['verified']


def test_genuine_infeasible_model_stays_infeasible():
    p = SimpleNamespace(p=2, budget=1, B=1.)
    result = outer._ratio_master(p, [(.8, np.zeros(2))], [(.8, 1., np.zeros(2))],
                                 10., .2, witness=[1, 0])
    assert result.status == 2 and result.master_retries == 0
    assert result.mip_dual_bound is None
    assert result.master_retry_log[0]['witness']['row_violation'] > .5


def test_witness_check_does_not_use_solver_tolerance():
    bounds = Bounds(np.zeros(1), np.ones(1))
    tiny_violation = LinearConstraint([[1e-300]], -np.inf, 0.)
    proof = outer._check_milp_witness([1.], bounds, tiny_violation, [1])
    assert not proof['verified'] and proof['row_violation'] > 0
    # Exact product equalities should not acquire floating summation residuals.
    product_sum = LinearConstraint([[-3., 1., 1., 1.]], 0., 0.)
    proof = outer._check_milp_witness([.1]*4, Bounds(np.zeros(4), np.ones(4)),
                                     product_sum, np.zeros(4))
    assert proof['verified'] and proof['row_violation'] == 0


@pytest.mark.parametrize('exhausted', [False, True])
def test_witness_retry_shares_remaining_budget(monkeypatch, saved_model, exhausted):
    clock, calls = SimpleNamespace(now=0.), []

    def solve(*args, **kwargs):
        calls.append(kwargs)
        clock.now += 2. if exhausted else .2
        return failed() if len(calls) == 1 else OptimizeResult(
            status=1, success=False, message='time limit', x=None, fun=None, mip_dual_bound=None)

    monkeypatch.setattr(outer, 'perf_counter', lambda: clock.now)
    monkeypatch.setattr(outer, 'milp', solve)
    result = invoke(saved_model, saved_model['witness'], remaining=1.)
    assert result.status == 1
    if exhausted:
        assert len(calls) == 1 and result.master_retries == 0
        assert not result.master_retry_log[0]['retried']
    else:
        assert len(calls) == 2 and result.master_retries == 1
        assert calls[1]['options']['time_limit'] == pytest.approx(.8)
    assert result.mip_dual_bound is None
    assert result.master_retry_log[0]['initial']['witness']['verified']


@pytest.mark.parametrize('retry_status', [2, 4])
def test_failed_witness_retry_is_terminal_and_has_no_bound(monkeypatch, saved_model, retry_status):
    calls = []
    monkeypatch.setattr(outer, 'milp', lambda *args, **kwargs:
                        calls.append(kwargs) or failed(2 if len(calls) == 1 else retry_status))
    result = invoke(saved_model, saved_model['witness'])
    assert len(calls) == 2 and result.status == retry_status
    assert result.x is None and result.fun is None and result.mip_dual_bound is None
    assert result.master_retries == 1


class FractionOracle:
    def __init__(self, *args, **kwargs):
        pass

    def solve(self, z, time_limit=None):
        denominator = 1+2*z[0]
        h, tau, mu = .6/denominator, 1/denominator, np.array([2/denominator, 0.])
        gradient = -h*mu
        return SimpleNamespace(value=h, upper=h, gradient=gradient,
            intercept=h-gradient @ z, test=outer._ConstantTest(np.flatnonzero(z), h),
            certificate=dict(entropy=h, tau=tau, mu=mu, margin=0.))


def test_legacy_ratio_search_supplies_actual_incumbent_without_polluted_lb(monkeypatch):
    real_milp, calls = outer.milp, []

    def solve(c, **kwargs):
        calls.append(kwargs)
        return failed() if len(calls) == 1 else real_milp(c, **kwargs)

    monkeypatch.setattr(outer, 'milp', solve)
    monkeypatch.setattr(outer, 'GateOracle', FractionOracle)
    result = outer.solve(SimpleNamespace(p=2, budget=1, B=1.), warm_support=(1,),
                         root_rounds=0, cut_mode='ratio')
    assert result.status == 'optimal' and result.support == (0,)
    assert result.lower <= .2 <= result.upper+1e-12
    assert result.diagnostics['master_retries'] == 1
    assert result.diagnostics['master_retry_log'][0]['initial']['witness']['verified']
