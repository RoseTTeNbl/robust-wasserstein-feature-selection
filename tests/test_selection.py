"""Algorithm 1 and the baseline searches against independent enumeration of Eq. (3)."""

import subprocess
import sys
from itertools import combinations

import numpy as np
import pytest

from robusthw.algorithm import Problem, SolveOptions, select
from robusthw.experiments.verification import transport_value


def test_identical_classes_have_value_two():
    result = select(Problem([[0, 0]], [[0, 0]], [0.1, 0.2]), 1, "oa")
    assert result.status == "optimal"
    assert result.lower <= 2.0 + 1e-9 and result.upper == pytest.approx(2.0)


@pytest.mark.parametrize("seed", [3, 7, 11])
def test_outer_approximation_and_enumeration_bracket_the_optimum(seed):
    rng = np.random.default_rng(seed)
    problem = Problem(rng.normal(size=(3, 5)), rng.normal(size=(4, 5)) + 0.7, [0.13, 0.19],
                      q0=[0.2, 0.3, 0.5], q1=[0.1, 0.2, 0.3, 0.4])
    subsets = [s for size in range(3) for s in combinations(range(5), size)]
    truth = {s: transport_value(problem, s) for s in subsets}
    optimum = min(v for s, v in truth.items() if len(s) == 2)
    for result in [select(problem, 2, initial_support=(3, 4)), select(problem, 2, "enumeration")]:
        assert result.status == "optimal"
        assert result.lower - 1e-8 <= optimum <= result.upper + 1e-8
        assert result.gap <= SolveOptions().absolute_tolerance
        for cut in result.cuts:
            for support, value in truth.items():
                assert cut.evaluate(problem.mask(support)) <= value + 1e-8
    lowers = [row["lower"] for row in result.trace]
    assert all(a <= b for a, b in zip(lowers, lowers[1:]))


@pytest.mark.parametrize("budget", [0, 3])
@pytest.mark.parametrize("method", ["global", "enumeration", "forward", "forward_swap"])
def test_empty_and_full_budgets(budget, method):
    result = select(Problem([[0, 0, 0]], [[2, 1, 0]], [0.1, 0.1]), budget, method)
    assert result.status in ("optimal", "complete")
    assert len(result.support) == budget
    if method.startswith("forward"):
        assert result.lower is None and result.gap is None
    if budget == 0:
        assert result.upper == 2


def test_forward_swap_never_ends_worse_than_forward():
    rng = np.random.default_rng(5)
    problem = Problem(rng.normal(size=(6, 6)), rng.normal(size=(6, 6)) + 0.5, [0.2, 0.2])
    forward = select(problem, 2, "forward")
    swapped = select(problem, 2, "forward_swap", initial_support=forward.support)
    assert swapped.upper <= forward.upper + 1e-12


@pytest.mark.parametrize("method", ["global", "enumeration", "forward", "forward_swap"])
def test_zero_time_limit_fits_nothing(method):
    problem = Problem([[0, 0, 0]], [[2, 1, 0]], [0.1, 0.1])
    result = select(problem, 2, method, SolveOptions(time_limit=0), ranked_features=[2, 1, 0])
    assert result.status in ("time_limit", "incomplete")
    assert result.fit_count == result.master_count == 0
    assert result.support == (1, 2) and result.upper == 2


@pytest.mark.parametrize("budget", [-1, 4, True])
def test_invalid_budgets_are_rejected(budget):
    with pytest.raises(ValueError):
        select(Problem([[0, 0, 0]], [[2, 1, 0]], [0.1, 0.1]), budget)


def test_master_reuses_a_scheduler_started_with_another_thread_count():
    script = """
import numpy as np
from scipy.optimize import milp
from robusthw.algorithm import Problem, SolveOptions, select
milp(np.ones(2), integrality=np.ones(2))
rng = np.random.default_rng(0)
x0, x1 = rng.normal(size=(6, 4)), rng.normal(size=(6, 4)) + 0.8
print(select(Problem(x0, x1, [0.3, 0.3]), 2, "oa", SolveOptions(time_limit=60)).status)
"""
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "optimal"
