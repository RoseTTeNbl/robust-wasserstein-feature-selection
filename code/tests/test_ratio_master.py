"""Product-lifted ratio master checked against explicit binary enumeration."""
from types import SimpleNamespace

import numpy as np
import pytest

from robusthw.problem import Problem
from robusthw.solver import _ratio_master, enumerate_supports, solve


@pytest.mark.parametrize('upper',[1., .23])
def test_complete_ratio_master_equals_binary_enumeration(upper):
    p = SimpleNamespace(p=4, budget=1, B=1.)
    h, tau, mu = .8, .5, np.array([.5,1.,2.,3.])
    # Same fraction tangent at support {0}; the value there is .8.
    gradient = -h * mu
    cuts = [(h-gradient[0], gradient)]
    master = _ratio_master(p, cuts, [(h,tau,mu)], 10., upper)
    reference = min(h/(tau+mu[j]) for j in range(4))
    assert master.status == 0
    assert master.fun == pytest.approx(reference, abs=1e-8)
    assert int(np.argmax(master.x[:4])) == 3
    eta, y = master.x[4], master.x[5:]
    assert np.max(abs(y-eta*np.rint(master.x[:4]))) < 1e-8


@pytest.mark.parametrize('seed',[3,10,19])
def test_ratio_search_brackets_independent_support_enumeration(seed):
    rng = np.random.default_rng(seed)
    p = Problem(rng.normal(size=(3,6)),rng.normal(size=(4,6)),
                budget=2,theta0=.23,theta1=.37)
    ref = enumerate_supports(p)
    ratio = solve(p,search='master',cut_mode='ratio')
    assert ratio.status == 'optimal'
    assert ratio.lower <= ref.upper + 1e-8
    assert ratio.upper >= ref.lower - 1e-8
    assert ratio.upper == pytest.approx(ref.upper,abs=1e-6)


def test_ratio_mode_rejects_other_gate():
    p = Problem(np.zeros((2,2)),np.ones((2,2)),1,.2,.2)
    with pytest.raises(ValueError):
        solve(p,search='master',gate='box',cut_mode='ratio')
