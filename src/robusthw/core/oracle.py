"""The fixed-subset LP and its normalized dual, without search heuristics.

Half-hinge scores are restricted to [-1,1] without loss of optimality. The
dual is the linear form of Appendix B's normalized model: its Bayes envelope
is H(Pi)=sum_r min(sum_{i in I0} Pi_ir, sum_{i in I1} Pi_ir).
Multiplying a normalized certificate by H gives the affine optimizer's scale;
in particular the cut slope is -H*mu, rather than -mu.
"""

import warnings
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .model import Cut, FitResult, Problem, SolveOptions


def constant_fit(problem: Problem, support, status="constant_fallback"):
    """An always-feasible zero score, with risk bound one and lower bound zero."""
    return FitResult(
        problem,
        problem.support(support),
        np.zeros(problem.size),
        np.full(problem.size, 0.5),
        np.zeros(2),
        1.0,
        0.0,
        status,
        0.0,
        None,
        {"fallback": True, "reason": status},
    )


def _matrix(rows, columns, values, shape):
    return sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=shape
    ).tocsr()


def _solve(c, ub, rhs, options, deadline, *, eq=None, eqrhs=None, bounds=None):
    remaining = deadline - perf_counter()
    if remaining <= 0:
        return None, {"status": "time_limit", "message": "No fit time remains"}
    # HiGHS' documented minimum feasibility tolerance is 1e-10.
    tolerance = max(1e-10, min(1e-9, options.constraint_tolerance / 10))
    settings = {
        "time_limit": remaining,
        "threads": options.threads,
        "primal_feasibility_tolerance": tolerance,
        "dual_feasibility_tolerance": tolerance,
    }
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*threads")
            result = linprog(
                c,
                A_ub=ub,
                b_ub=rhs,
                A_eq=eq,
                b_eq=eqrhs,
                bounds=bounds,
                method="highs",
                options=settings,
            )
    except Exception as error:
        return None, {"status": "solver_error", "message": str(error)}
    status = {0: "optimal", 1: "time_limit", 2: "infeasible", 3: "unbounded", 4: "solver_error"}
    info = {
        "status": status.get(result.status, "solver_error"),
        "message": str(result.message),
        "solver_objective": None if result.fun is None else float(result.fun),
        "iterations": int(result.nit),
        "feasibility_tolerance": tolerance,
    }
    if result.status not in (0, 1) or result.x is None or not np.isfinite(result.x).all():
        return None, info
    return result.x, info


def _primal(problem, support, options, deadline):
    """Equation (4): loss_r <= a_i + lambda_k*c_ir^S."""
    n = problem.size
    count, score_start = 2 * n + 2, n + 2
    c = np.r_[problem.q, problem.radii, np.zeros(n)]
    distance = problem.delta[:, :, list(support)].sum(axis=2)
    rr = np.arange(n * n)
    rows = [rr, rr, rr]
    columns = [
        np.repeat(np.arange(n), n),
        np.repeat(n + problem.y, n),
        score_start + np.tile(np.arange(n), n),
    ]
    data = [-np.ones(n * n), -distance.ravel(), np.repeat(np.where(problem.y == 0, -0.5, 0.5), n)]
    matrix = _matrix(rows, columns, data, (n * n, count))
    bounds = [(0, None)] * (n + 2) + [(-1, 1)] * n
    raw, info = _solve(c, matrix, np.full(n * n, -0.5), options, deadline, bounds=bounds)
    if raw is None:
        return None, info
    scores = np.clip(raw[score_start:], -1, 1)
    prices = np.maximum(raw[n : n + 2], 0)
    losses = 0.5 * (1 + np.where(problem.y[:, None] == 0, -scores[None, :], scores[None, :]))
    # Keep the score and prices, then recompute every source allowance. Solver
    # primal tolerances can therefore never be mistaken for feasible bounds.
    with np.errstate(over="ignore", invalid="ignore"):
        allowances = np.maximum(0.0, np.max(losses - prices[problem.y, None] * distance, axis=1))
    if not np.isfinite(allowances).all():
        return None, {**info, "repair_status": "invalid_certificate"}
    allowances = np.nextafter(allowances, np.inf)
    upper = float(problem.q @ allowances + problem.radii @ prices)
    if not np.isfinite(upper):
        return None, {**info, "repair_status": "invalid_certificate"}
    margin = 64 * np.finfo(float).eps * (1 + upper + float(np.max(losses)))
    upper = float(np.nextafter(upper + margin, np.inf))
    residual = float(
        max(0.0, np.max(losses - allowances[:, None] - prices[problem.y, None] * distance))
    )
    info.update(repaired_primal_residual=residual, upper_margin=margin)
    return (scores, allowances, prices, upper), info


def _dual(problem, z, options, deadline):
    """LP for H under row masses, coordinate budgets and normalization."""
    n, d = problem.size, problem.dimension
    beta_start, mu_start = n * n, n * n + 2 * d
    alpha_index, h_start = mu_start + d, mu_start + d + 1
    count = h_start + n
    c = np.r_[np.zeros(h_start), -np.ones(n)]
    eq_rows = [np.repeat(np.arange(n), n), np.arange(n), np.full(d + 1, n)]
    eq_cols = [
        np.arange(n * n),
        np.full(n, alpha_index),
        np.r_[alpha_index, mu_start + np.arange(d)],
    ]
    eq_values = [np.ones(n * n), -problem.q, np.r_[1.0, z]]
    eq = _matrix(eq_rows, eq_cols, eq_values, (n + 1, count))
    rows, columns, values = [], [], []
    for k in range(2):
        source = np.flatnonzero(problem.y == k)
        plan_indices = (source[:, None] * n + np.arange(n)).ravel()
        coordinate_rows = k * d + np.arange(d)
        rows.extend(
            [np.repeat(coordinate_rows, len(plan_indices)), coordinate_rows, coordinate_rows]
        )
        columns.extend(
            [np.tile(plan_indices, d), beta_start + k * d + np.arange(d), mu_start + np.arange(d)]
        )
        values.extend(
            [
                problem.delta[source].reshape(-1, d).T.ravel(),
                -np.ones(d),
                -np.full(d, problem.radii[k]),
            ]
        )
        rows.extend([np.full(d, 2 * d + k), np.array([2 * d + k])])
        columns.extend([beta_start + k * d + np.arange(d), np.array([alpha_index])])
        values.extend([np.ones(d), np.array([-problem.radii[k]])])
        target_rows = 2 * d + 2 + k * n + np.arange(n)
        rows.extend([target_rows, np.tile(target_rows, len(source))])
        columns.extend([h_start + np.arange(n), plan_indices])
        values.extend([np.ones(n), -np.ones(len(plan_indices))])
    # Appendix C permits bounded coordinate multipliers without changing the
    # optimum. It also removes unbounded directions on constant coordinates.
    bound_rows = 2 * d + 2 + 2 * n + np.arange(d)
    rows.extend([bound_rows, bound_rows])
    columns.extend([mu_start + np.arange(d), np.full(d, alpha_index)])
    values.extend([np.ones(d), -problem.price_bounds])
    matrix = _matrix(rows, columns, values, (3 * d + 2 + 2 * n, count))
    raw, info = _solve(
        c,
        matrix,
        np.zeros(matrix.shape[0]),
        options,
        deadline,
        eq=eq,
        eqrhs=np.r_[np.zeros(n), 1.0],
    )
    if raw is None:
        return None, info
    alpha = max(0.0, float(raw[alpha_index]))
    plan = np.maximum(raw[: n * n].reshape(n, n), 0)
    for i in range(n):
        total = plan[i].sum()
        if total <= 0:
            plan[i, i] = 1
            total = 1
        plan[i] *= alpha * problem.q[i] / total
    beta = np.maximum(raw[beta_start:mu_start].reshape(2, d), 0)
    for k in range(2):
        total = beta[k].sum()
        if total > alpha * problem.radii[k]:
            beta[k] *= alpha * problem.radii[k] / total
    cost = np.array(
        [
            np.einsum("ir,irj->j", plan[problem.y == k], problem.delta[problem.y == k])
            for k in range(2)
        ]
    )
    if not np.isfinite(cost).all():
        return None, {**info, "repair_status": "invalid_certificate"}
    mu = np.maximum(0.0, np.max((cost - beta) / problem.radii[:, None], axis=0))
    # Recompute beta after deriving minimal coordinate prices; the new beta is
    # no greater than the repaired allocation used above.
    beta = np.maximum(cost - problem.radii[:, None] * mu, 0)
    normalization = alpha + float(mu @ z)
    if normalization <= 0 or not np.isfinite(normalization):
        return None, {**info, "repair_status": "zero_certificate"}
    plan, beta, mu, alpha, cost = (
        plan / normalization,
        beta / normalization,
        mu / normalization,
        alpha / normalization,
        cost / normalization,
    )
    masses = [plan[problem.y == k].sum(axis=0) for k in range(2)]
    h = float(np.minimum(*masses).sum())
    slope = -h * mu
    intercept = h - float(slope @ z)
    # All constraints are checked after repair and rescaling. The interval is
    # a floating-point certificate with an explicit downward cut margin.
    row_residual = float(np.max(np.abs(plan.sum(axis=1) - alpha * problem.q)))
    budget_residual = float(max(0.0, np.max(beta.sum(axis=1) - alpha * problem.radii)))
    coordinate_residual = float(max(0.0, np.max(cost - beta - problem.radii[:, None] * mu)))
    norm_residual = abs(alpha + float(mu @ z) - 1)
    bound_residual = float(max(0.0, np.max(mu - alpha * problem.price_bounds)))
    scales = (
        max(1.0, alpha),
        max(1.0, float(np.max(alpha * problem.radii))),
        max(1.0, float(np.max(cost))),
        1.0,
        max(1.0, float(np.max(mu))),
    )
    residuals = (row_residual, budget_residual, coordinate_residual, norm_residual, bound_residual)
    violation = max(r / s for r, s in zip(residuals, scales))
    info.update(
        row_residual=row_residual,
        budget_residual=budget_residual,
        coordinate_residual=coordinate_residual,
        normalization_residual=norm_residual,
        price_bound_residual=bound_residual,
        scaled_violation=violation,
        alpha=alpha,
        mu=mu.tolist(),
        envelope=h,
        affine_price_bound=float(np.max(np.maximum(-slope - problem.price_bounds, 0))),
    )
    if not np.isfinite([h, intercept, *slope]).all() or violation > options.constraint_tolerance:
        return None, {**info, "repair_status": "invalid_certificate"}
    margin = max(1e-10, 128 * np.finfo(float).eps * (1 + abs(intercept) + np.abs(slope).sum()))
    cut = Cut(float(intercept - margin), slope)
    lower = max(0.0, cut.evaluate(z))
    info.update(cut_margin=margin, repair_status="accepted")
    return (lower, cut), info


def fit_fixed(problem: Problem, support, options=None, with_cut=True):
    """Fit one subset; preserve feasible bounds on timeout or numerical failure.

    Both LPs share min(time_limit, fit_time_limit). The primal gets at most
    half of that allowance so the dual can still return a lower certificate.
    ``with_cut=False`` omits the returned cut but keeps the same numerical
    lower/upper assessment for fair comparisons between selection methods.
    """
    options = SolveOptions() if options is None else options
    support = problem.support(support)
    started = perf_counter()
    allowance = min(options.time_limit, options.fit_time_limit)
    fallback = constant_fit(problem, support, "time_limit" if allowance == 0 else "solver_error")
    if allowance == 0:
        return fallback
    deadline = started + allowance
    primal, primal_info = _primal(problem, support, options, started + allowance / 2)
    fitted = fallback
    if (
        primal is not None
        and primal[3] < 1
        and primal_info["repaired_primal_residual"] <= options.constraint_tolerance
    ):
        fitted = FitResult(problem, support, *primal[:3], primal[3], 0.0, "fitted", 0.0, None)
    dual, dual_info = _dual(problem, problem.mask(support), options, deadline)
    lower, cut = (0.0, None) if dual is None else dual
    if lower > fitted.upper + options.constraint_tolerance:
        status, lower, cut = "numerical_failure", 0.0, None
    elif fitted.upper - lower <= options.inner_tolerance:
        status = "optimal"
    elif perf_counter() >= deadline or "time_limit" in (primal_info["status"], dual_info["status"]):
        status = "time_limit"
    elif {primal_info["status"], dual_info["status"]} & {"solver_error", "infeasible", "unbounded"}:
        status = "solver_error"
    elif "invalid_certificate" in (
        primal_info.get("repair_status"),
        dual_info.get("repair_status"),
    ):
        status = "numerical_failure"
    else:
        status = "accuracy_limit"
    return FitResult(
        problem,
        support,
        fitted.scores,
        fitted.source_bounds,
        fitted.prices,
        fitted.upper,
        lower,
        status,
        perf_counter() - started,
        cut if with_cut else None,
        {
            "primal": primal_info,
            "dual": dual_info,
            "fallback": bool(fitted.upper == 1),
            "gap": fitted.upper - lower,
            "requested_fit_seconds": allowance,
            "certificate_arithmetic": "floating point with explicit repairs and margins",
        },
    )
