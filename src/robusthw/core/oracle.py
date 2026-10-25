"""Sparse LP oracles for the standard hinge ell(t) = max(0, 1 + t).

This is a tractable specialization of the manuscript's general convex loss.
Clipping scores to [-1, 1] does not increase either hinge loss. The fixed-set
problem uses the finite primal; fractional selections use the lifted primal.
The direct selection dual has row masses q, H = 2*sum min(p0, p1), and
objective H - 2*mu@z. There is no normalization or rescaled dual certificate.

The internal N-row target array contains n0+n1 observations, with class 0
followed by class 1; it does not assume equal sample sizes. Repeated target
labels are harmless. All optimization matrices are sparse.
"""

import warnings
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .model import (
    MODEL_VERSION,
    ZERO_SCORE_RISK,
    Cut,
    FitResult,
    Problem,
    RelaxedResult,
    SolveOptions,
)


def _metadata():
    return {
        "model_version": MODEL_VERSION,
        "loss": "standard_hinge",
        "standard_hinge": True,
        "zero_score_risk": ZERO_SCORE_RISK,
        "certificate_arithmetic": "floating point with explicit repairs and margins",
    }


def constant_fit(problem: Problem, support, status="constant_fallback"):
    """An always-feasible zero score: a=1, lambda=0 and risk at most two."""
    return FitResult(
        problem,
        problem.support(support),
        np.zeros(problem.size),
        np.ones(problem.size),
        np.zeros(2),
        ZERO_SCORE_RISK,
        0.0,
        status,
        0.0,
        None,
        {**_metadata(), "fallback": True, "reason": status},
    )


def _matrix(rows, columns, values, shape):
    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))), shape=shape
    ).tocsr()
    matrix.eliminate_zeros()
    return matrix


def _solve(c, ub, rhs, options, deadline, *, eq=None, eqrhs=None, bounds=None):
    remaining = deadline - perf_counter()
    if remaining <= 0:
        return None, {"status": "time_limit", "message": "No fit time remains"}
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
            # HiGHS keeps a process-wide scheduler. Another scipy.linprog call
            # may already have initialized it with a different thread count.
            # Reuse that scheduler rather than returning a spurious solve error.
            if result.status == 4 and "Not Set" in str(result.message):
                settings.pop("threads", None)
                settings["time_limit"] = deadline - perf_counter()
                if settings["time_limit"] <= 0:
                    return None, {"status": "time_limit", "message": "No fit time remains"}
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
        "variables": len(c),
        "inequalities": ub.shape[0],
        "nonzeros": ub.nnz,
        "requested_threads": options.threads,
        "thread_limit_applied": "threads" in settings,
    }
    if result.status not in (0, 1) or result.x is None or not np.isfinite(result.x).all():
        return None, info
    return result.x, info


def _distance(problem, support):
    selected = problem.x[:, list(support)]
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.abs(selected[:, None, :] - selected[None, :, :]).sum(axis=2)
    if not np.isfinite(result).all():
        raise ValueError("Selected distances overflow; rescale the data")
    return result


def _consistent_scores(problem, scores, support):
    """Identical constraint columns receive the first stored target's score."""
    selected = problem.x[:, list(support)]
    _, first, inverse = np.unique(selected, axis=0, return_index=True, return_inverse=True)
    return scores[first[inverse]]


def _repair_primal(problem, raw, info, *, distance=None, z=None):
    n, d = problem.size, problem.dimension
    scores = np.clip(raw[n + 2 : 2 * n + 2], -1, 1)
    prices = np.maximum(raw[n : n + 2], 0)
    if z is None:
        # distance contains the selected coordinates only.
        _, first, inverse = np.unique(distance, axis=1, return_index=True, return_inverse=True)
        scores = scores[first[inverse]]
        costs = prices[problem.y, None] * distance
        weights = None
    else:
        scores = _consistent_scores(problem, scores, range(d))
        weights = np.minimum(np.maximum(raw[2 * n + 2 :].reshape(2, d), 0), prices[:, None])
        capacity = problem.radii @ weights
        scale = np.ones(d)
        excessive = capacity > 2 * z
        scale[excessive] = np.nextafter(2 * z[excessive] / capacity[excessive], 0)
        weights *= np.maximum(scale, 0)
        costs = np.empty((n, n))
        for k in range(2):
            rows = problem.y == k
            costs[rows] = np.einsum("irm,m->ir", problem.delta[rows], weights[k])
    losses = 1 + np.where(problem.y[:, None] == 0, -scores[None, :], scores[None, :])
    # Recompute a from repaired scores/prices, rather than trusting LP tolerances.
    with np.errstate(over="ignore", invalid="ignore"):
        allowances = np.maximum(0.0, np.max(losses - costs, axis=1))
    if not np.isfinite(allowances).all() or not np.isfinite(costs).all():
        return None, {**info, "repair_status": "invalid_certificate"}
    allowances = np.nextafter(allowances, np.inf)
    upper = float(problem.q @ allowances + problem.radii @ prices)
    if not np.isfinite(upper):
        return None, {**info, "repair_status": "invalid_certificate"}
    margin = 64 * np.finfo(float).eps * (1 + upper + float(np.max(losses)))
    upper = float(np.nextafter(upper + margin, np.inf))
    residual = max(0.0, float(np.max(losses - allowances[:, None] - costs)))
    capacity_residual = (
        0.0
        if weights is None
        else max(
            0.0,
            float(np.max(problem.radii @ weights - 2 * z)),
            float(np.max(weights - prices[:, None])),
        )
    )
    info.update(
        repaired_primal_residual=residual,
        capacity_residual=capacity_residual,
        upper_margin=margin,
        repair_status="accepted",
    )
    return (scores, allowances, prices, weights, upper), info


def _primal(problem, support, options, deadline):
    """Fixed-set finite primal: 1 +/- g_y <= a_i + lambda_k*c_iy."""
    n, score_start = problem.size, problem.size + 2
    count = 2 * n + 2
    distance = _distance(problem, support)
    rr = np.arange(n * n)
    matrix = _matrix(
        [rr, rr, rr],
        [
            np.repeat(np.arange(n), n),
            np.repeat(n + problem.y, n),
            score_start + np.tile(np.arange(n), n),
        ],
        [-np.ones(n * n), -distance.ravel(), np.repeat(np.where(problem.y == 0, -1.0, 1.0), n)],
        (n * n, count),
    )
    raw, info = _solve(
        np.r_[problem.q, problem.radii, np.zeros(n)],
        matrix,
        -np.ones(n * n),
        options,
        deadline,
        bounds=[(0, None)] * (n + 2) + [(-1, 1)] * n,
    )
    if raw is None:
        return None, info
    return _repair_primal(problem, raw, info, distance=distance)


def _primal_relaxed(problem, z, options, deadline):
    """Lifted selection primal, including rho-weighted coordinate capacities."""
    n, d = problem.size, problem.dimension
    w_start = 2 * n + 2
    count = w_start + 2 * d
    rr = np.arange(n * n)
    rows = [rr, rr]
    columns = [np.repeat(np.arange(n), n), n + 2 + np.tile(np.arange(n), n)]
    values = [-np.ones(n * n), np.repeat(np.where(problem.y == 0, -1.0, 1.0), n)]
    for k in range(2):
        sources = np.flatnonzero(problem.y == k)
        pairs = (sources[:, None] * n + np.arange(n)).ravel()
        rows.append(np.repeat(pairs, d))
        columns.append(w_start + k * d + np.tile(np.arange(d), len(pairs)))
        values.append(-problem.delta[sources].ravel())
        bound_rows = n * n + k * d + np.arange(d)
        rows.extend([bound_rows, bound_rows, n * n + 2 * d + np.arange(d)])
        columns.extend(
            [
                w_start + k * d + np.arange(d),
                np.full(d, n + k),
                w_start + k * d + np.arange(d),
            ]
        )
        values.extend([np.ones(d), -np.ones(d), np.full(d, problem.radii[k])])
    matrix = _matrix(rows, columns, values, (n * n + 3 * d, count))
    raw, info = _solve(
        np.r_[problem.q, problem.radii, np.zeros(n + 2 * d)],
        matrix,
        np.r_[-np.ones(n * n), np.zeros(2 * d), 2 * z],
        options,
        deadline,
        bounds=[(0, None)] * (n + 2) + [(-1, 1)] * n + [(0, None)] * (2 * d),
    )
    if raw is None:
        return None, info
    return _repair_primal(problem, raw, info, z=z)


def _plan_rows(problem, raw):
    """Repair source marginals without changing their class normalization."""
    n = problem.size
    plan = np.maximum(raw[: n * n].reshape(n, n), 0).copy()
    for i in range(n):
        total = plan[i].sum()
        if total <= 0:
            plan[i, i], total = 1.0, 1.0
        plan[i] *= problem.q[i] / total
        largest = int(np.argmax(plan[i]))
        plan[i, largest] += problem.q[i] - plan[i].sum()
    return plan


def _plan_equalities(problem, count):
    n = problem.size
    return _matrix([np.repeat(np.arange(n), n)], [np.arange(n * n)], [np.ones(n * n)], (n, count))


def _envelope_rows(problem, h_start, start):
    n = problem.size
    rows, columns, values = [], [], []
    for k in range(2):
        sources = np.flatnonzero(problem.y == k)
        pairs = (sources[:, None] * n + np.arange(n)).ravel()
        rr = start + k * n + np.arange(n)
        rows.extend([rr, np.tile(rr, len(sources))])
        columns.extend([h_start + np.arange(n), pairs])
        values.extend([np.ones(n), -np.ones(len(pairs))])
    return rows, columns, values


def _evidence(problem, plan):
    n0 = len(problem.x0)
    masses = np.array([plan[:n0].sum(axis=0), plan[n0:].sum(axis=0)])
    h = 2 * float(np.minimum(masses[0], masses[1]).sum())
    return h, {
        "gamma0": plan[:n0].tolist(),
        "gamma1": plan[n0:].tolist(),
        "target_masses": masses.tolist(),
        "radii": problem.radii.tolist(),
        "envelope": h,
    }


def _dual(problem, z, options, deadline):
    """Direct gamma/mu/beta LP: maximize 2*sum h_y - 2*mu@z."""
    n, d = problem.size, problem.dimension
    beta_start, mu_start = n * n, n * n + 2 * d
    h_start = mu_start + d
    count = h_start + n
    rows, columns, values = _envelope_rows(problem, h_start, 2 * d + 2)
    rhs = np.r_[np.zeros(2 * d), problem.radii, np.zeros(2 * n)]
    for k in range(2):
        sources = np.flatnonzero(problem.y == k)
        pairs = (sources[:, None] * n + np.arange(n)).ravel()
        rr = k * d + np.arange(d)
        rows.extend([np.repeat(rr, len(pairs)), rr, rr, np.full(d, 2 * d + k)])
        columns.extend(
            [
                np.tile(pairs, d),
                beta_start + k * d + np.arange(d),
                mu_start + np.arange(d),
                beta_start + k * d + np.arange(d),
            ]
        )
        values.extend(
            [
                problem.delta[sources].reshape(-1, d).T.ravel(),
                -np.ones(d),
                -np.full(d, problem.radii[k]),
                np.ones(d),
            ]
        )
    matrix = _matrix(rows, columns, values, (2 * d + 2 + 2 * n, count))
    raw, info = _solve(
        np.r_[np.zeros(mu_start), 2 * z, -2 * np.ones(n)],
        matrix,
        rhs,
        options,
        deadline,
        eq=_plan_equalities(problem, count),
        eqrhs=problem.q,
    )
    info["formulation"] = "direct_selection_dual"
    if raw is None:
        return None, info
    plan = _plan_rows(problem, raw)
    beta = np.maximum(raw[beta_start:mu_start].reshape(2, d), 0)
    for k in range(2):
        allowed = np.nextafter(problem.radii[k], 0)
        total = beta[k].sum()
        if total > allowed:
            beta[k] *= np.nextafter(allowed / total, 0)
    with np.errstate(over="ignore", invalid="ignore"):
        cost = np.array(
            [
                np.einsum("ir,irm->m", plan[problem.y == k], problem.delta[problem.y == k])
                for k in range(2)
            ]
        )
        mu = np.maximum(0.0, np.max((cost - beta) / problem.radii[:, None], axis=0))
        mu = np.nextafter(mu, np.inf)
    if not np.isfinite(cost).all() or not np.isfinite(mu).all():
        return None, {**info, "repair_status": "invalid_certificate"}
    h, evidence = _evidence(problem, plan)
    row_error = np.abs(plan.sum(axis=1) - problem.q)
    budget_error = np.maximum(beta.sum(axis=1) - problem.radii, 0)
    coordinate_error = np.maximum(cost - beta - problem.radii[:, None] * mu, 0)
    violation = max(
        float(np.max(row_error)),
        float(np.max(budget_error / np.maximum(1, problem.radii))),
        float(np.max(coordinate_error / np.maximum(1, cost))),
    )
    # Each optimal primal can be restricted to J<=2. Thus a_i<=2/q_i and
    # lambda_k,w_km<=2/rho_k bound the effect of residual floating-point errors.
    margin = max(
        1e-10,
        128 * np.finfo(float).eps * (1 + abs(h) + 2 * float(mu.sum()))
        + 2 * float(np.sum(row_error / problem.q))
        + 2 * float(np.sum(budget_error / problem.radii))
        + 2 * float(np.sum(coordinate_error / problem.radii[:, None])),
    )
    info.update(
        **evidence,
        mu=mu.tolist(),
        beta=beta.tolist(),
        coordinate_costs=cost.tolist(),
        row_residual=float(np.max(row_error)),
        budget_residual=float(np.max(budget_error)),
        coordinate_residual=float(np.max(coordinate_error)),
        scaled_violation=violation,
        cut_margin=margin,
        repair_status="accepted",
    )
    if not np.isfinite([h, margin]).all() or violation > options.constraint_tolerance:
        return None, {**info, "repair_status": "invalid_certificate"}
    cut = Cut(h - margin, -2 * mu)
    return (max(0.0, cut.evaluate(z)), cut), info


def _dual_fixed(problem, support, options, deadline):
    """Original finite LFD LP, avoiding D-dimensional cut costs when unneeded."""
    n = problem.size
    count, h_start = n * n + n, n * n
    rows, columns, values = _envelope_rows(problem, h_start, 2)
    distance = _distance(problem, support)
    for k in range(2):
        sources = np.flatnonzero(problem.y == k)
        pairs = (sources[:, None] * n + np.arange(n)).ravel()
        rows.append(np.full(len(pairs), k))
        columns.append(pairs)
        values.append(distance[sources].ravel())
    matrix = _matrix(rows, columns, values, (2 + 2 * n, count))
    raw, info = _solve(
        np.r_[np.zeros(n * n), -2 * np.ones(n)],
        matrix,
        np.r_[problem.radii, np.zeros(2 * n)],
        options,
        deadline,
        eq=_plan_equalities(problem, count),
        eqrhs=problem.q,
    )
    info["formulation"] = "fixed_selection_lfd"
    if raw is None:
        return None, info
    plan = _plan_rows(problem, raw)
    # Mixing with the zero-cost diagonal plan repairs each transport budget.
    for k in range(2):
        sources = np.flatnonzero(problem.y == k)
        cost = float(np.sum(plan[sources] * distance[sources]))
        if cost > np.nextafter(problem.radii[k], 0):
            scale = max(0.0, np.nextafter(np.nextafter(problem.radii[k], 0) / cost, 0))
            plan[sources] *= scale
            plan[sources, sources] += (1 - scale) * problem.q[sources]
    h, evidence = _evidence(problem, plan)
    row_error = np.abs(plan.sum(axis=1) - problem.q)
    costs = np.array([np.sum(plan[problem.y == k] * distance[problem.y == k]) for k in range(2)])
    excess = np.maximum(costs - problem.radii, 0)
    violation = max(float(np.max(row_error)), float(np.max(excess / np.maximum(1, costs))))
    margin = max(
        1e-10,
        128 * np.finfo(float).eps * (1 + abs(h))
        + 2 * float(np.sum(row_error / problem.q))
        + 2 * float(np.sum(excess / problem.radii)),
    )
    info.update(
        **evidence,
        row_residual=float(np.max(row_error)),
        budget_residual=float(np.max(excess)),
        scaled_violation=violation,
        cut_margin=margin,
        repair_status="accepted",
    )
    if not np.isfinite([h, margin]).all() or violation > options.constraint_tolerance:
        return None, {**info, "repair_status": "invalid_certificate"}
    return (max(0.0, h - margin), None), info


def _finish(upper, dual, primal_info, dual_info, options, deadline):
    lower, cut = (0.0, None) if dual is None else dual
    if lower > upper + options.constraint_tolerance:
        return 0.0, None, "numerical_failure"
    if upper - lower <= options.inner_tolerance:
        return min(lower, upper), cut, "optimal"
    if perf_counter() >= deadline or "time_limit" in (primal_info["status"], dual_info["status"]):
        return lower, cut, "time_limit"
    if {primal_info["status"], dual_info["status"]} & {"solver_error", "infeasible", "unbounded"}:
        return lower, cut, "solver_error"
    if "invalid_certificate" in (primal_info.get("repair_status"), dual_info.get("repair_status")):
        return lower, cut, "numerical_failure"
    return lower, cut, "accuracy_limit"


def fit_fixed(problem: Problem, support, options=None, with_cut=True):
    """Fit a subset with repaired bounds; both LPs share one fit allowance.

    A requested cut uses the new direct selection dual. Without a cut, the
    original fixed-support LFD gives a cheaper independent lower certificate.
    Solver status alone never certifies optimality: the returned gap must pass.
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
        and primal[4] < ZERO_SCORE_RISK
        and primal_info["repaired_primal_residual"] <= options.constraint_tolerance
    ):
        fitted = FitResult(problem, support, *primal[:3], primal[4], 0, "fitted", 0, None)
    if with_cut:
        dual, dual_info = _dual(problem, problem.mask(support), options, deadline)
    else:
        dual, dual_info = _dual_fixed(problem, support, options, deadline)
    lower, cut, status = _finish(fitted.upper, dual, primal_info, dual_info, options, deadline)
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
            **_metadata(),
            "primal": primal_info,
            "dual": dual_info,
            "fallback": fitted is fallback,
            "gap": fitted.upper - lower,
            "requested_fit_seconds": allowance,
        },
    )


def fit_relaxed(problem: Problem, z, options=None, with_cut=True):
    """Evaluate the lifted convex value F(z) for any z in [0,1]^D.

    This returns a RelaxedResult, not a detector. The primal weights w and
    class-indexed direct-dual evidence allow independent feasibility checks.
    """
    options = SolveOptions() if options is None else options
    z = problem.relaxation(z)
    started = perf_counter()
    allowance = min(options.time_limit, options.fit_time_limit)
    deadline = started + allowance
    primal_info = dual_info = {"status": "time_limit"}
    fitted = (
        np.zeros(problem.size),
        np.ones(problem.size),
        np.zeros(2),
        np.zeros((2, problem.dimension)),
        ZERO_SCORE_RISK,
    )
    fallback, dual = True, None
    if allowance > 0:
        primal, primal_info = _primal_relaxed(problem, z, options, started + allowance / 2)
        if (
            primal is not None
            and primal[4] < ZERO_SCORE_RISK
            and max(primal_info["repaired_primal_residual"], primal_info["capacity_residual"])
            <= options.constraint_tolerance
        ):
            fitted, fallback = primal, False
        dual, dual_info = _dual(problem, z, options, deadline)
    lower, cut, status = _finish(fitted[4], dual, primal_info, dual_info, options, deadline)
    return RelaxedResult(
        problem,
        z,
        *fitted[:4],
        fitted[4],
        lower,
        status,
        perf_counter() - started,
        cut if with_cut else None,
        {
            **_metadata(),
            "primal": primal_info,
            "dual": dual_info,
            "fallback": fallback,
            "gap": fitted[4] - lower,
            "requested_fit_seconds": allowance,
        },
    )
