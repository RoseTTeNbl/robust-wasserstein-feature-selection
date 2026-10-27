"""Evaluation oracle of Algorithm 1; equation numbers refer to note/main.tex.

fit_fixed(S)   lines 3-4: Eq. (5) gives a detector and upper >= V(S);
               Eq. (9) at z=1_S gives lower <= V(S) and the cut Eq. (11).
fit_relaxed(z) F(z) at fractional z from Eq. (10) (upper) and Eq. (9) (lower).
"""

from time import perf_counter

import numpy as np

from ._certificates import _finish, _metadata, _repair_dual, _repair_primal
from ._certificates import constant_fit as constant_fit
from ._evaluation import _dual_fixed, _envelope_rows, _primal_relaxed
from ._linear_program import _distance, _matrix, _solve
from .model import ZERO_SCORE_RISK, Cut, FitResult, Problem, RelaxedResult, SolveOptions


def _primal(problem, support, options, deadline):
    """Eq. (5) at S: detector certificate (g, a, lambda) and upper bound on V(S)."""
    # Sources i and targets r both run over the pooled samples x_1..x_N;
    # problem.y[i] is the class k of source i and q_i replaces 1/n_k.
    n, score_start = problem.size, problem.size + 2
    # Variable blocks: [a (N), lambda (2), g (N)].
    count = 2 * n + 2
    # distance[i,r] = 1_S^T Delta_ir = ||x_i[S]-x_r[S]||_1.
    distance = _distance(problem, support)
    rr = np.arange(n * n)
    # Hinge with g in [-1,1]: ell(sigma_k*g_r)=1+sigma_k*g_r, so row (i,r) reads
    # -a_i - lambda_k*distance[i,r] + sigma_k*g_r <= -1.
    matrix = _matrix(
        [rr, rr, rr],
        [
            np.repeat(np.arange(n), n),  # Source a_i, repeated over targets.
            np.repeat(n + problem.y, n),  # Class-specific lambda_k.
            score_start + np.tile(np.arange(n), n),  # Target score g_r.
        ],
        [-np.ones(n * n), -distance.ravel(), np.repeat(np.where(problem.y == 0, -1.0, 1.0), n)],
        (n * n, count),
    )
    raw, info = _solve(
        # Objective: sum_i q_i*a_i+sum_k rho_k*lambda_k.
        np.r_[problem.q, problem.radii, np.zeros(n)],
        matrix,
        -np.ones(n * n),
        options,
        deadline,
        bounds=[(0, None)] * (n + 2) + [(-1, 1)] * n,
    )
    if raw is None:
        return None, info
    # Reset a_i=max_r[ell(sigma_k*g_r)-lambda_k*distance[i,r]]_+ (exactly feasible).
    return _repair_primal(problem, raw, info, distance=distance)


def _dual(problem, z, options, deadline):
    """Eq. (9) at z: Charnes-Cooper LP of F(z); returns (lower bound, cut)."""
    n, d = problem.size, problem.dimension
    # Variables [gamma_hat (N*N), beta (2*D), mu_hat (D), h (N), alpha (1)], where
    # (gamma_hat, mu_hat, alpha) = (gamma, mu, 1)/(1+mu^T z) and gamma is in Pi.
    beta_start, mu_start = n * n, n * n + 2 * d
    h_start = mu_start + d
    alpha_start = h_start + n
    count = alpha_start + 1
    # Hinge: H(gamma_hat)=2*sum_r h_r subject to h_r <= p_0(r), h_r <= p_1(r).
    rows, columns, values = _envelope_rows(problem, h_start, 2 * d + 2)
    rhs = np.zeros(2 * d + 2 + 2 * n)
    for k in range(2):
        sources = np.flatnonzero(problem.y == k)
        pairs = (sources[:, None] * n + np.arange(n)).ravel()
        rr = k * d + np.arange(d)
        # Coupling rows (k,m): c_k(gamma_hat)[m] - beta_km - rho_k*mu_hat_m <= 0,
        # and the budget row k: sum_m beta_km - rho_k*alpha <= 0.
        rows.extend(
            [np.repeat(rr, len(pairs)), rr, rr, np.full(d, 2 * d + k), np.array([2 * d + k])]
        )
        columns.extend(
            [
                np.tile(pairs, d),
                beta_start + k * d + np.arange(d),
                mu_start + np.arange(d),
                beta_start + k * d + np.arange(d),
                np.array([alpha_start]),
            ]
        )
        values.extend(
            [
                problem.delta[sources].reshape(-1, d).T.ravel(),  # c_k(gamma_hat)[m].
                -np.ones(d),  # -beta_km.
                -np.full(d, problem.radii[k]),  # -rho_k*mu_m.
                np.ones(d),
                np.array([-problem.radii[k]]),  # sum_m beta_km-rho_k*alpha<=0.
            ]
        )
    matrix = _matrix(rows, columns, values, (2 * d + 2 + 2 * n, count))
    # sum_r gamma_hat_ir = q_i*alpha and alpha + mu_hat^T z = 1.
    equalities = _matrix(
        [np.repeat(np.arange(n), n), np.arange(n), np.full(d, n), np.array([n])],
        [
            np.arange(n * n),
            np.full(n, alpha_start),
            mu_start + np.arange(d),
            np.array([alpha_start]),
        ],
        [np.ones(n * n), -problem.q, z, np.ones(1)],
        (n + 1, count),
    )
    raw, info = _solve(
        # Maximize H(gamma_hat)=2*sum_r h_r; the optimal value is F(z).
        np.r_[np.zeros(h_start), -2 * np.ones(n), 0.0],
        matrix,
        rhs,
        options,
        deadline,
        eq=equalities,
        eqrhs=np.r_[np.zeros(n), 1.0],
    )
    info["formulation"] = "normalized_homogeneous_transport"
    if raw is None:
        return None, info
    # Repair: gamma=gamma_hat/alpha rescaled into Pi, mu raised until Eq. (6) holds,
    # H lowered by an error bound; then everything is scaled by s=v/(1+mu^T z).
    repaired, info = _repair_dual(problem, raw, info, options, z)
    if repaired is None:
        return None, info
    h, mu, alpha, margin = repaired
    # Eq. (8) piece with h=H(s*gamma), alpha=s, mu=s*mu:
    # 2*sqrt(h)-alpha-mu^T z' = v*(1-mu_bar^T(z'-z)), i.e. the cut Eq. (11).
    cut = Cut(2 * np.sqrt(h) - alpha - margin, -mu)
    # The cut at z itself is a certified lower bound on F(z).
    return (max(0.0, cut.evaluate(z)), cut), info


def fit_fixed(problem: Problem, support, options=None, with_cut=True):
    """Lines 3-4 of Algorithm 1 at S: detector, lower <= V(S) <= upper, cut."""
    options = SolveOptions() if options is None else options
    support = problem.support(support)
    started = perf_counter()
    allowance = min(options.time_limit, options.fit_time_limit)
    fallback = constant_fit(problem, support, "time_limit" if allowance == 0 else "solver_error")
    if allowance == 0:
        return fallback
    deadline = started + allowance
    # Eq. (5): upper bound and detector; half the time allowance.
    # Fallback g=0, a=1, lambda=0 has risk V(empty)=2.
    primal, primal_info = _primal(problem, support, options, started + allowance / 2)
    fitted = fallback
    if (
        primal is not None
        and primal[4] < ZERO_SCORE_RISK
        and primal_info["repaired_primal_residual"] <= options.constraint_tolerance
    ):
        fitted = FitResult(problem, support, *primal[:3], primal[4], 0, "fitted", 0, None)
    if with_cut:
        # Eq. (9) at z=1_S: lower bound on F(1_S)=V(S) and the cut Eq. (11).
        dual, dual_info = _dual(problem, problem.mask(support), options, deadline)
    else:
        # Evaluation only: the fixed-S LFD Eq. (4), lower bound without a cut.
        dual, dual_info = _dual_fixed(problem, support, options, deadline)
    # Status from the inner gap upper-lower and the solver states.
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
    """Evaluate F(z) for z in [0,1]^D by Eq. (10) (upper) and Eq. (9) (lower).

    This returns a RelaxedResult, not a detector. The weights w of Eq. (10)
    and the scaled transport certificate allow independent feasibility checks.
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
        # Eq. (10): min eta with coupling rho_0*w_0m+rho_1*w_1m <= eta*z_m.
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
