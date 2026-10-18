"""Independent finite transport LP for the exhaustive small-instance check.

This formulation has transport masses and one target-overlap variable, with
no a/lambda source bounds, selector prices, normalized dual or outer cuts.
It checks the implemented fixed-set formulation through a separate matrix.
"""

import warnings
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def transport_value(problem, support):
    n = problem.size
    support = problem.support(support)
    distance = np.abs(problem.x[:, None, list(support)] - problem.x[None, :, list(support)]).sum(
        axis=2
    )
    eq = sparse.lil_matrix((n, n * n + n))
    ub = sparse.lil_matrix((2 + 2 * n, n * n + n))
    for i in range(n):
        eq[i, i * n : (i + 1) * n] = 1
        ub[problem.y[i], i * n : (i + 1) * n] = distance[i]
        for r in range(n):
            ub[2 + problem.y[i] * n + r, i * n + r] = -1
    for k in (0, 1):
        for r in range(n):
            ub[2 + k * n + r, n * n + r] = 1
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*threads")
        result = linprog(
            np.r_[np.zeros(n * n), -np.ones(n)],
            A_ub=ub.tocsr(),
            b_ub=np.r_[problem.radii, np.zeros(2 * n)],
            A_eq=eq.tocsr(),
            b_eq=problem.q,
            bounds=(0, None),
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-10,
                "dual_feasibility_tolerance": 1e-10,
                "threads": 1,
            },
        )
    if not result.success:
        raise RuntimeError(f"Independent transport LP failed: {result.message}")
    return float(-result.fun)


def verify_small(problem, budget, results, tolerance=1e-8):
    """Complete every <=d subset; reject invalid cuts or reported intervals."""
    if (
        isinstance(budget, (bool, np.bool_))
        or not isinstance(budget, (int, np.integer))
        or not 0 <= budget <= problem.dimension
    ):
        raise ValueError("budget must be an integer in the coordinate range")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    values = {
        support: transport_value(problem, support)
        for size in range(budget + 1)
        for support in combinations(range(problem.dimension), size)
    }
    optimum = min(value for support, value in values.items() if len(support) == budget)
    checks, maximum = 0, 0.0
    for result in results:
        support = problem.support(result.support)
        if len(support) != budget:
            raise AssertionError("A comparison result does not use the declared budget")
        if not np.isfinite(result.upper) or result.upper + tolerance < values[support]:
            raise AssertionError(
                "Reported feasible risk bound is below its own subset's independent optimum"
            )
        if (
            result.lower is not None
            and not result.lower - tolerance <= optimum <= result.upper + tolerance
        ):
            raise AssertionError("Returned global interval misses the independent optimum")
        for cut in result.cuts:
            for support, value in values.items():
                violation = cut.evaluate(problem.mask(support)) - value
                maximum = max(maximum, violation)
                checks += 1
                if violation > tolerance:
                    raise AssertionError(f"Invalid affine cut at {support}: {violation}")
    return {
        "status": "passed",
        "independent_optimum": optimum,
        "enumerated_subsets": len(values),
        "cut_checks": checks,
        "maximum_cut_violation": maximum,
        "tolerance": tolerance,
        "values": [{"support": support, "risk": value} for support, value in values.items()],
    }
