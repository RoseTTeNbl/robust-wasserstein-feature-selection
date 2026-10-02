"""Outer-loop checks independent of the core oracle implementation."""
from types import SimpleNamespace

import numpy as np
import pytest

import robusthw.solver as outer


class AffineOracle:
    def __init__(self, problem, **kwargs):
        self.problem = problem

    def solve(self, z, time_limit=None):
        # Exact convex extension; observing coordinate 0 gives risk 0.2.
        value = .6 - .4 * z[0]
        support = tuple(np.flatnonzero(z))
        test = outer._ConstantTest(support, value)
        return SimpleNamespace(value=value, upper=value, gradient=np.array([-.4, 0.]),
                               intercept=.6, test=test, status="optimal")


@pytest.fixture
def p():
    return SimpleNamespace(p=2, budget=1, B=1.)


def test_global_search_moves_from_bad_initial_support(monkeypatch, p):
    monkeypatch.setattr(outer, "GateOracle", AffineOracle)
    result = outer.solve(p, warm_support=(1,), root_rounds=0)
    assert result.status == "optimal"
    assert result.support == (0,)
    assert result.lower <= .2 <= result.upper + 1e-8
    assert result.oracle_calls == 2


def test_root_and_binary_bounds_bracket_exact_value(monkeypatch, p):
    monkeypatch.setattr(outer, "GateOracle", AffineOracle)
    result = outer.solve(p, warm_support=(1,), root_rounds=3)
    assert result.status == "optimal"
    assert result.lower <= .2 <= result.upper + 1e-8


def test_master_incumbent_is_not_used_as_lower_bound(monkeypatch, p):
    monkeypatch.setattr(outer, "GateOracle", AffineOracle)
    # A feasible master incumbent of .55 does not certify .55: global bound .1.
    monkeypatch.setattr(outer, "_master", lambda *args, **kwargs:
                        SimpleNamespace(status=1, x=np.array([0., 1., .55]),
                                        fun=.55, mip_dual_bound=.1, message="time limit"))
    result = outer.solve(p, warm_support=(1,), root_rounds=0)
    assert result.lower <= .1
    assert result.status != "optimal"


def test_enumeration_reference_is_independent_search(monkeypatch, p):
    monkeypatch.setattr(outer, "GateOracle", AffineOracle)
    result = outer.enumerate_supports(p)
    assert result.status == "optimal"
    assert result.support == (0,)
    assert result.visited == 2


def test_invalid_warm_support_is_rejected(monkeypatch, p):
    monkeypatch.setattr(outer, "GateOracle", AffineOracle)
    with pytest.raises(ValueError):
        outer.solve(p, warm_support=(1, 1))
