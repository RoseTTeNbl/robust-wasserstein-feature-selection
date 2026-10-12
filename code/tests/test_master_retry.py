"""HiGHS solve errors must not invalidate bounds or consume a fresh budget."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import robusthw.solver as outer


def result(status=0, *, bound=.2):
    return OptimizeResult(status=status, success=status == 0, message=f"status {status}",
                          x=np.array([1., 0., .2]), fun=.2, mip_dual_bound=bound)


@pytest.fixture
def problem():
    return SimpleNamespace(p=2, budget=1, B=1.)


def invoke(problem, mode, remaining=1.):
    cuts = [(.6, np.array([-.4, 0.]))]
    if mode == "tangent":
        return outer._master(problem, cuts, remaining, integer=True)
    return outer._ratio_master(problem, cuts, [(.2, .5, np.array([.5, 0.]))],
                               remaining, .6)


@pytest.mark.parametrize("mode", ["tangent", "ratio"])
def test_status_four_retries_identical_model_once_and_shares_time(monkeypatch, problem, mode):
    clock = SimpleNamespace(now=0.)
    calls = []
    failed, repaired = result(4, bound=100.), result()

    def milp(c, **kwargs):
        calls.append((c, kwargs))
        clock.now += .2
        return failed if len(calls) == 1 else repaired

    monkeypatch.setattr(outer, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(outer, "milp", milp)
    actual = invoke(problem, mode)
    assert actual is repaired
    assert actual.master_retries == 1
    assert len(calls) == 2
    first, second = calls[0][1], calls[1][1]
    assert calls[0][0] is calls[1][0]
    for key in ("integrality", "bounds", "constraints"):
        assert first[key] is second[key]
    assert "presolve" not in first["options"]
    assert second["options"]["presolve"] is False
    assert second["options"]["time_limit"] == pytest.approx(.8)
    assert second["options"]["threads"] == 1
    log = actual.master_retry_log[0]
    assert log["initial"]["status"] == 4
    assert log["initial"]["seconds"] == pytest.approx(.2)
    assert log["retry"]["status"] == 0


@pytest.mark.parametrize("status", [0, 1, 2, 3])
def test_other_statuses_do_not_trigger_retry(monkeypatch, problem, status):
    calls = []
    first = result(status)
    monkeypatch.setattr(outer, "milp", lambda *args, **kwargs: calls.append(kwargs) or first)
    actual = invoke(problem, "tangent")
    assert actual is first
    assert len(calls) == 1
    if status in (0, 1):
        assert actual.mip_dual_bound == .2
    else:
        assert actual.mip_dual_bound is None
        assert actual.x is None


def test_no_retry_after_budget_exhausted_and_error_has_no_bound(monkeypatch, problem):
    clock = SimpleNamespace(now=0.)
    calls = []

    def milp(*args, **kwargs):
        calls.append(kwargs)
        clock.now = 2.
        return result(4, bound=100.)

    monkeypatch.setattr(outer, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(outer, "milp", milp)
    actual = invoke(problem, "tangent")
    assert len(calls) == 1
    assert actual.status == 1
    assert actual.x is None and actual.fun is None and actual.mip_dual_bound is None
    assert actual.master_retries == 0
    assert actual.master_retry_log[0]["retried"] is False


def test_failed_retry_is_terminal_and_cannot_supply_bound(monkeypatch, problem):
    calls = []
    monkeypatch.setattr(outer, "milp", lambda *args, **kwargs:
                        calls.append(kwargs) or result(4, bound=100.))
    actual = invoke(problem, "ratio")
    assert len(calls) == 2
    assert actual.status == 4
    assert actual.mip_dual_bound is None and actual.fun is None and actual.x is None
    assert actual.master_retries == 1
    assert actual.master_retry_log[0]["retry"]["status"] == 4


class AffineOracle:
    def __init__(self, problem, **kwargs):
        pass

    def solve(self, z, time_limit=None):
        value = .6 - .4 * z[0]
        support = tuple(np.flatnonzero(z))
        return SimpleNamespace(value=value, upper=value,
                               gradient=np.array([-.4, 0.]), intercept=.6,
                               test=outer._ConstantTest(support, value))


def test_outer_solver_retains_retry_provenance(monkeypatch, problem):
    original = outer.milp
    calls = []

    def fail_first(c, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return result(4, bound=100.)
        return original(c, **kwargs)

    monkeypatch.setattr(outer, "GateOracle", AffineOracle)
    monkeypatch.setattr(outer, "milp", fail_first)
    actual = outer.solve(problem, search="master", warm_support=(1,), root_rounds=0)
    assert actual.status == "optimal"
    assert actual.lower <= .2 <= actual.upper + 1e-8
    assert actual.diagnostics["master_retries"] == 1
    assert actual.diagnostics["master_retry_log"][0]["initial"]["status"] == 4
    assert actual.trace[1]["master_retries"] == 1


def test_outer_solver_never_uses_error_result_bound(monkeypatch, problem):
    monkeypatch.setattr(outer, "GateOracle", AffineOracle)
    monkeypatch.setattr(outer, "_master", lambda *args, **kwargs: result(4, bound=100.))
    actual = outer.solve(problem, search="master", warm_support=(1,), root_rounds=0)
    assert actual.status == "master_failure"
    assert actual.lower == 0.
