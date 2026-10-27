"""Core reading route: fixed-selection risk, selection dual, and affine cut."""

from time import perf_counter

import numpy as np

from ._certificates import _finish, _metadata, _repair_dual, _repair_primal
from ._certificates import constant_fit as constant_fit
from ._evaluation import _dual_fixed, _envelope_rows, _primal_relaxed
from ._linear_program import _distance, _matrix, _solve
from .model import ZERO_SCORE_RISK, Cut, FitResult, Problem, RelaxedResult, SolveOptions


def _primal(problem, support, options, deadline):
    """Eq. (4): fixed-selection primal and feasible upper bound on V(S)."""
    # N=n0+n1; source class k=problem.y[i], target r indexes X.
    # q_i=1/n_k by default; each class has total mass 1.
    n, score_start = problem.size, problem.size + 2
    # Variable blocks: [a (N), lambda (2), g (N)].
    count = 2 * n + 2
    # c_ir(S)=||x_i[S]-x_r[S]||_1; g_r=g(x_r[S]).
    distance = _distance(problem, support)
    rr = np.arange(n * n)
    # Hinge ell(t)=(1+t)_+: clipping g to [-1,1] preserves optimality.
    # Row (i,r): -a_i-c_ir(S)*lambda_k+sigma_k*g_r <= -1.
    # sigma_0=-1, sigma_1=+1; ell(-g_r)=1-g_r, ell(g_r)=1+g_r.
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
    # Feasible a_i=max_r{ell(sigma_k*g_r)-lambda_k*c_ir(S)}; rounded upward.
    return _repair_primal(problem, raw, info, distance=distance)


def _dual(problem, z, options, deadline):
    """Normalized LP for Eq. (5); alpha stores the dependent mass M(gamma)."""
    n, d = problem.size, problem.dimension
    # Variable blocks: [gamma (N*N), beta (2*D), mu (D), h (N), alpha (1)].
    # M(gamma)=0.5*sum_{k=0}^1 sum_{i,y} gamma^(k)_iy.
    # Row equations give each class total mass alpha, hence alpha=M(gamma).
    # alpha+mu^T*z=1 is the LP normalization from Appendix B.
    beta_start, mu_start = n * n, n * n + 2 * d
    h_start = mu_start + d
    alpha_start = h_start + n
    count = alpha_start + 1
    # Hinge H: inf_{|v|<=1}[p_0r*(1-v)+p_1r*(1+v)]=2*min(p_0r,p_1r).
    # p_kr=sum_{i:class(i)=k} gamma_ir; h_r<=p_0r,p_1r linearizes max H.
    rows, columns, values = _envelope_rows(problem, h_start, 2 * d + 2)
    rhs = np.zeros(2 * d + 2 + 2 * n)
    for k in range(2):
        sources = np.flatnonzero(problem.y == k)
        pairs = (sources[:, None] * n + np.arange(n)).ravel()
        rr = k * d + np.arange(d)
        # c_k(gamma)[m]=sum_{i,y} gamma^(k)_iy*|x_i^(k)[m]-y[m]|.
        # t_+=max(t,0); sum_m (c_k(gamma)[m]-rho_k*mu_m)_+ <= rho_k*M(gamma).
        # beta>=0, c_k(gamma)[m]<=beta_km+rho_k*mu_m, sum_m beta_km<=rho_k*alpha.
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
                problem.delta[sources].reshape(-1, d).T.ravel(),  # c_k(gamma)[m] coefficients.
                -np.ones(d),  # -beta_km.
                -np.full(d, problem.radii[k]),  # -rho_k*mu_m.
                np.ones(d),
                np.array([-problem.radii[k]]),  # sum_m beta_km-rho_k*alpha<=0.
            ]
        )
    matrix = _matrix(rows, columns, values, (2 * d + 2 + 2 * n, count))
    # sum_r gamma_ir=q_i*alpha; alpha+sum_m mu_m*z_m=1.
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
        # H(gamma)=2*sum_r h_r. The normalized maximum equals F(z).
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
    # If the LP value is v, (v*gamma,v*mu) solves Eq. (5), with M(v*gamma)=v*alpha.
    # Numerical repair returns a feasible cone certificate without a risk cap.
    repaired, info = _repair_dual(problem, raw, info, options, z)
    if repaired is None:
        return None, info
    h, mu, alpha, margin = repaired
    # Eq. (7): c(z')=2*sqrt(H(gamma))-M(gamma)-mu^T*z'; alpha=M(gamma).
    cut = Cut(2 * np.sqrt(h) - alpha - margin, -mu)
    # Pointwise lower bound max(0,c(z)); retain the affine c for the master.
    return (max(0.0, cut.evaluate(z)), cut), info


def fit_fixed(problem: Problem, support, options=None, with_cut=True):
    """Fixed S: detector, bounds on V(S), and optional selection cut."""
    options = SolveOptions() if options is None else options
    support = problem.support(support)
    started = perf_counter()
    allowance = min(options.time_limit, options.fit_time_limit)
    fallback = constant_fit(problem, support, "time_limit" if allowance == 0 else "solver_error")
    if allowance == 0:
        return fallback
    deadline = started + allowance
    # Eq. (4): feasible objective >=V(S); reserve half the allowance for the primal.
    # Fallback g=0, a=1, lambda=0 gives risk 2.
    primal, primal_info = _primal(problem, support, options, started + allowance / 2)
    fitted = fallback
    if (
        primal is not None
        and primal[4] < ZERO_SCORE_RISK
        and primal_info["repaired_primal_residual"] <= options.constraint_tolerance
    ):
        fitted = FitResult(problem, support, *primal[:3], primal[4], 0, "fitted", 0, None)
    if with_cut:
        # Eq. (6): F(1_S)=V(S). The dual supplies a lower bound and a cut
        # for the selection problem, Eq. (8).
        dual, dual_info = _dual(problem, problem.mask(support), options, deadline)
    else:
        # Evaluation only: fixed-selection LFD, Eq. (3), without a selection cut.
        dual, dual_info = _dual_fixed(problem, support, options, deadline)
    # Inner accuracy: upper-lower. Checked bounds and cuts feed the outer search.
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

    This returns a RelaxedResult, not a detector. The epigraph weights w and
    scaled transport certificate allow independent feasibility checks.
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
        # Fractional z: min eta, objective<=eta, rho^T*w[:,m]<=eta*z_m.
        # The auxiliary w values do not define a fixed-coordinate detector.
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
