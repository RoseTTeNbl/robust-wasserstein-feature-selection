"""Lines 4-5 of Algorithm 1: one linear program per selection S.

Equation numbers refer to note/main.tex. For the hinge loss with scores in
[-1, 1], Eq. (4) is the linear program

    min  sum_i q_i a_i + rho_0 lambda_0 + rho_1 lambda_1
    s.t. 1 + sigma_k g_r <= a_i + lambda_k ||x_i[S] - x_r[S]||_1   (source i of class k, target r)

with sigma_0 = -1, sigma_1 = +1. The multipliers of these rows form a least
favorable transport gamma of Eq. (3), so a single solve gives the detector, the
value V(S), a lower bound and the ratio cut Eq. (8).
"""

import warnings
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .model import MODEL_VERSION, ZERO_SCORE_RISK, Cut, FitResult, Problem, SolveOptions


def highs(solver, *args, threads=1, **kwargs):
    """Call a scipy HiGHS solver, working around two HiGHS solve errors (status 4).

    HiGHS keeps one process-wide scheduler, so a call that asks for another thread
    count fails with "Not Set": retry without a thread count. Presolve can also
    fail on small well-posed master problems: retry without presolve.
    """
    options = kwargs.pop("options", {})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected")
        result = solver(*args, options={**options, "threads": threads}, **kwargs)
        if result.status == 4 and "Not Set" in str(result.message):
            result = solver(*args, options=options, **kwargs)
        if result.status == 4:
            result = solver(*args, options={**options, "presolve": False}, **kwargs)
    return result


def constant_fit(problem: Problem, support, status="constant_fallback"):
    """The zero score: a = 1, lambda = 0, risk ZERO_SCORE_RISK = V(empty)."""
    n = problem.size
    return FitResult(
        problem, problem.support(support), np.zeros(n), np.ones(n), np.zeros(2),
        ZERO_SCORE_RISK, 0.0, status, 0.0, None, {"model_version": MODEL_VERSION},
    )  # fmt: skip


def hinge_value(problem, plan):
    """H(gamma) = sum_r psi(p_0(r), p_1(r)) with psi(a, b) = 2 min(a, b) for the hinge."""
    mass = [plan[problem.y == k].sum(axis=0) for k in range(2)]
    return 2.0 * float(np.minimum(mass[0], mass[1]).sum())


def feasible_plan(problem, plan, distance):
    """Make a transport feasible in Eq. (3) without lowering H much.

    Rows are completed to q_i with the stay-in-place transport iota (cost 0), and
    each class is mixed with iota until its cost meets rho_k (Appendix A.1).
    """
    plan = np.maximum(plan, 0)
    total = plan.sum(axis=1)
    plan *= np.minimum(1.0, problem.q / np.maximum(total, 1e-300))[:, None]
    plan[np.diag_indices_from(plan)] += problem.q - plan.sum(axis=1)
    for k in range(2):
        rows = problem.y == k
        cost = float((plan[rows] * distance[rows]).sum())
        if cost > problem.radii[k]:
            t = problem.radii[k] / cost
            iota = np.diag(problem.q)[rows]
            plan[rows] = t * plan[rows] + (1 - t) * iota
    return plan


def ratio_cut(problem, plan, support):
    """Eq. (8) at z = 1_S from a feasible plan of Eq. (3) (Appendix A.1).

    mu_m = max_k c_k[m] / rho_k off S and 0 on S; scaled so that H = 1, this gives
    1/F(z) <= (1 + mu @ z) / H(plan) for every z in [0, 1]^D.
    """
    h = hinge_value(problem, plan)
    if h <= 0:
        return None
    cost = np.array([np.einsum("ir,irm->m", plan[problem.y == k], problem.delta[problem.y == k])
                     for k in range(2)])  # fmt: skip
    mu = np.max(cost / problem.radii[:, None], axis=0)
    mu[list(support)] = 0
    return Cut(1.0 / h, mu / h)


def fit_fixed(problem: Problem, support, options=None, with_cut=True):
    """Solve Eq. (4) at S. Returns the detector, lower <= V(S) <= upper and the cut."""
    options = options or SolveOptions()
    support = problem.support(support)
    started = perf_counter()
    seconds = min(options.time_limit, options.fit_time_limit)
    if seconds <= 0:
        return constant_fit(problem, support, "time_limit")
    n = problem.size
    distance = problem.distance(support)
    sigma = np.where(problem.y == 0, -1.0, 1.0)
    rows = np.arange(n * n)  # row (i, r) -> i * n + r
    # Variables [a (n), lambda (2), g (n)]; row (i, r): -a_i - lambda_k d_ir + sigma_k g_r <= -1.
    matrix = sparse.csr_matrix(
        (
            np.r_[-np.ones(n * n), -distance.ravel(), np.repeat(sigma, n)],
            (np.tile(rows, 3), np.r_[rows // n, n + np.repeat(problem.y, n), n + 2 + rows % n]),
        ),
        shape=(n * n, 2 * n + 2),
    )
    result = highs(
        linprog,
        np.r_[problem.q, problem.radii, np.zeros(n)],
        A_ub=matrix,
        b_ub=-np.ones(n * n),
        bounds=[(0, None)] * (n + 2) + [(-1, 1)] * n,
        method="highs",
        threads=options.threads,
        options={"time_limit": seconds},
    )
    if result.x is None:
        status = "time_limit" if result.status == 1 else "solver_error"
        return constant_fit(problem, support, status)

    # Upper bound: Eq. (4) at the returned (g, lambda) with the smallest feasible a.
    # Coincident projected targets get one common score, so Eq. (5) is well defined.
    lam = np.maximum(result.x[n : n + 2], 0)
    g = np.clip(result.x[n + 2 :], -1, 1)
    _, first, inverse = np.unique(distance, axis=1, return_index=True, return_inverse=True)
    g = g[first[inverse.ravel()]]
    a = np.maximum(0.0, np.max(1 + sigma[:, None] * g[None, :] - lam[problem.y, None] * distance,
                               axis=1))  # fmt: skip
    upper = float(problem.q @ a + problem.radii @ lam)

    # Lower bound: the row multipliers are a least favorable transport of Eq. (3).
    duals = -result.ineqlin.marginals if result.status == 0 else np.zeros(n * n)
    plan = feasible_plan(problem, duals.reshape(n, n), distance)
    lower = min(hinge_value(problem, plan), upper)
    gap = upper - lower
    status = "optimal" if gap <= options.inner_tolerance else (
        "time_limit" if result.status == 1 else "accuracy_limit")  # fmt: skip
    return FitResult(
        problem, support, g, a, lam, upper, lower, status, perf_counter() - started,
        ratio_cut(problem, plan, support) if with_cut else None,
        {"model_version": MODEL_VERSION, "lp_status": int(result.status), "gap": gap},
    )  # fmt: skip
