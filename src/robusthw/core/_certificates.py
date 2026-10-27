"""Numerical feasibility repairs, bound margins, and solver diagnostics."""

from time import perf_counter

import numpy as np

from .model import MODEL_VERSION, ZERO_SCORE_RISK, FitResult, Problem


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
        # Coincident projections: identical distance columns, common score g_r=g_t.
        _, first, inverse = np.unique(distance, axis=1, return_index=True, return_inverse=True)
        scores = scores[first[inverse]]
        costs = prices[problem.y, None] * distance
        weights = None
    else:
        scores = _consistent_scores(problem, scores, range(d))
        weights = np.minimum(np.maximum(raw[2 * n + 2 : -1].reshape(2, d), 0), prices[:, None])
        # z_m=0 implies w_0m=w_1m=0. Positive coordinates need no fixed cap.
        weights[:, z == 0] = 0
        costs = np.empty((n, n))
        for k in range(2):
            rows = problem.y == k
            costs[rows] = np.einsum("irm,m->ir", problem.delta[rows], weights[k])
    losses = 1 + np.where(problem.y[:, None] == 0, -scores[None, :], scores[None, :])
    # Feasible source bounds: a_i=max(0,max_r{ell(sigma_k*g_r)-cost_ir}).
    with np.errstate(over="ignore", invalid="ignore"):
        allowances = np.maximum(0.0, np.max(losses - costs, axis=1))
    if not np.isfinite(allowances).all() or not np.isfinite(costs).all():
        return None, {**info, "repair_status": "invalid_certificate"}
    allowances = np.nextafter(allowances, np.inf)
    upper = float(problem.q @ allowances + problem.radii @ prices)
    if weights is not None:
        # Smallest feasible epigraph eta=max(objective,max_{m:z_m>0} rho^T*w_m/z_m).
        active = z > 0
        with np.errstate(over="ignore", invalid="ignore"):
            capacity_value = (problem.radii @ weights[:, active]) / z[active]
        upper = max(upper, float(np.max(capacity_value, initial=0)))
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
            float(np.max(problem.radii @ weights - upper * z)),
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


def _plan_rows(problem, raw):
    """Repair source marginals without changing their class normalization."""
    n = problem.size
    plan = np.maximum(raw[: n * n].reshape(n, n), 0).copy()
    for i in range(n):
        total = plan[i].sum()
        if total <= 0:
            plan[i, i], total = 1.0, 1.0
        # Source normalization: gamma_ir <- q_i*gamma_ir/sum_t gamma_it.
        plan[i] *= problem.q[i] / total
        largest = int(np.argmax(plan[i]))
        plan[i, largest] += problem.q[i] - plan[i].sum()
    return plan


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


def _repair_dual(problem, raw, info, options, z):
    """Repair the normalized LP and return a homogeneous cone certificate.

    Row-mass errors are covered by explicit transport corrections; their
    coordinate costs enlarge mu and their hinge loss reduces the intercept.
    No bound on the primal objective or transport prices is used.
    The returned alpha is the common class mass M(gamma), defined as half
    the total mass of both classes and fixed by the repaired row equations.
    """
    n, d = problem.size, problem.dimension
    beta_start, mu_start = n * n, n * n + 2 * d
    raw_alpha = max(0.0, float(raw[-1]))
    if raw_alpha == 0:
        # alpha=0 forces gamma=0. The zero cone certificate always gives c(z)=0.
        plan, beta, mu = np.zeros((n, n)), np.zeros((2, d)), np.zeros(d)
        h, evidence = _evidence(problem, plan)
        info.update(
            **evidence,
            mu=mu.tolist(),
            beta=beta.tolist(),
            alpha=0.0,
            coordinate_costs=beta.tolist(),
            row_residual=0.0,
            budget_residual=0.0,
            coordinate_residual=0.0,
            scaled_violation=0.0,
            cut_margin=0.0,
            repair_status="accepted",
        )
        return (h, mu, 0.0, 0.0), info
    plan = _plan_rows(problem, raw)
    # Normalize to unit class mass before repairing; then beta has budget rho_k.
    with np.errstate(over="ignore", invalid="ignore"):
        beta = np.maximum(raw[beta_start:mu_start].reshape(2, d), 0) / raw_alpha
    if not np.isfinite(beta).all():
        return None, {**info, "repair_status": "invalid_certificate"}
    eps = np.finfo(float).eps
    for k in range(2):
        allowed = problem.radii[k] * (1 - 64 * eps)
        total = beta[k].sum()
        if total > allowed:
            beta[k] *= np.nextafter(allowed / total, 0)
    # An exact row correction has L1 norm at most row_error[i]. Its possible
    # increase in C_km is bounded by sum_i row_error[i]*max_y distance[i,y,m].
    row_error = np.abs(plan.sum(axis=1) - problem.q)
    with np.errstate(over="ignore", invalid="ignore"):
        cost = np.array(
            [
                np.einsum("ir,irm->m", plan[problem.y == k], problem.delta[problem.y == k])
                for k in range(2)
            ]
        )
        correction = np.array(
            [
                row_error[problem.y == k] @ problem.delta[problem.y == k].max(axis=1)
                for k in range(2)
            ]
        )
        cost_error = correction + (16 * n * n * eps) * np.abs(cost)
        # mu makes every coordinate inequality feasible after the row correction.
        mu = np.maximum(0.0, np.max((cost + cost_error - beta) / problem.radii[:, None], axis=0))
        mu = np.nextafter(mu, np.inf)
    if not np.isfinite(cost).all() or not np.isfinite(mu).all():
        return None, {**info, "repair_status": "invalid_certificate"}
    h, _ = _evidence(problem, plan)
    # H=2*sum min(p0,p1) is 2-Lipschitz in total transported mass (L1).
    h_safe = max(0.0, h - 2 * float(row_error.sum()) - 32 * n * n * eps * h)
    denominator = 1 + float(mu @ z)
    # max_r 2*sqrt(r*H)-r*(1+mu^T*z): r=H/(1+mu^T*z)^2.
    # Unit class mass before scaling implies M(gamma)=alpha afterwards.
    alpha = (h_safe / denominator) / denominator
    plan, beta, mu, cost = alpha * plan, alpha * beta, alpha * mu, alpha * cost
    h_cone, evidence = _evidence(problem, plan)
    row_error = np.abs(plan.sum(axis=1) - alpha * problem.q)
    budget_error = np.maximum(beta.sum(axis=1) - alpha * problem.radii, 0)
    coordinate_error = np.maximum(cost - beta - problem.radii[:, None] * mu, 0)
    violation = max(
        float(np.max(row_error)),
        float(np.max(budget_error / np.maximum(1, problem.radii))),
        float(np.max(coordinate_error / np.maximum(1, cost))),
    )
    # The square-root difference covers the row correction and H arithmetic.
    # The remaining margin covers evaluation of the affine cut on [0,1]^D.
    envelope_margin = max(0.0, 2 * (np.sqrt(h_cone) - np.sqrt(alpha * h_safe)))
    margin = envelope_margin + max(
        1e-10, 128 * eps * (1 + 2 * np.sqrt(h_cone) + alpha + float(mu.sum()))
    )
    info.update(
        **evidence,
        mu=mu.tolist(),
        beta=beta.tolist(),
        alpha=alpha,
        normalized_scale=raw_alpha,
        coordinate_costs=cost.tolist(),
        row_residual=float(np.max(row_error)),
        budget_residual=float(np.max(budget_error)),
        coordinate_residual=float(np.max(coordinate_error)),
        scaled_violation=violation,
        cut_margin=margin,
        repair_status="accepted",
    )
    if not np.isfinite([h_cone, alpha, margin]).all() or violation > options.constraint_tolerance:
        return None, {**info, "repair_status": "invalid_certificate"}
    return (h_cone, mu, alpha, margin), info


def _repair_fixed_dual(problem, raw, info, distance, options):
    plan = _plan_rows(problem, raw)
    # gamma <- t*gamma+(1-t)*diag(q), classwise t<=min(1,rho_k/cost_k).
    # Source masses preserved; transport cost scaled by t (diagonal cost is zero).
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
