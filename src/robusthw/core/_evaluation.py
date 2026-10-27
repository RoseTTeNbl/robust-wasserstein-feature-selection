"""Optional evaluators and the two shared transport-constraint blocks."""

import numpy as np

from ._certificates import _repair_fixed_dual, _repair_primal
from ._linear_program import _distance, _matrix, _solve


def _plan_equalities(problem, count):
    n = problem.size
    # Row i: sum_r gamma_ir=q_i.
    return _matrix([np.repeat(np.arange(n), n)], [np.arange(n * n)], [np.ones(n * n)], (n, count))


def _envelope_rows(problem, h_start, start):
    n = problem.size
    rows, columns, values = [], [], []
    for k in range(2):
        sources = np.flatnonzero(problem.y == k)
        pairs = (sources[:, None] * n + np.arange(n)).ravel()
        rr = start + k * n + np.arange(n)
        # Row (k,r): h_r-sum_{i:class(i)=k} gamma_ir<=0.
        rows.extend([rr, np.tile(rr, len(sources))])
        columns.extend([h_start + np.arange(n), pairs])
        values.extend([np.ones(n), -np.ones(len(pairs))])
    return rows, columns, values


def _primal_relaxed(problem, z, options, deadline):
    """Exact epigraph problem for optional fractional-z evaluation."""
    n, d = problem.size, problem.dimension
    w_start = 2 * n + 2
    eta_start = w_start + 2 * d
    count = eta_start + 1
    rr = np.arange(n * n)
    rows = [rr, rr]
    columns = [np.repeat(np.arange(n), n), n + 2 + np.tile(np.arange(n), n)]
    values = [-np.ones(n * n), np.repeat(np.where(problem.y == 0, -1.0, 1.0), n)]
    for k in range(2):
        sources = np.flatnonzero(problem.y == k)
        pairs = (sources[:, None] * n + np.arange(n)).ravel()
        # Loss row: -a_i+sigma_k*g_r-sum_m w_km*|x_i[m]-x_r[m]|<=-1.
        rows.append(np.repeat(pairs, d))
        columns.append(w_start + k * d + np.tile(np.arange(d), len(pairs)))
        values.append(-problem.delta[sources].ravel())
        bound_rows = n * n + k * d + np.arange(d)
        # w_km<=lambda_k; rho_0*w_0m+rho_1*w_1m<=eta*z_m.
        rows.extend([bound_rows, bound_rows, n * n + 2 * d + np.arange(d)])
        columns.extend(
            [
                w_start + k * d + np.arange(d),
                np.full(d, n + k),
                w_start + k * d + np.arange(d),
            ]
        )
        values.extend([np.ones(d), -np.ones(d), np.full(d, problem.radii[k])])
    objective_row = n * n + 3 * d
    rows.extend(
        [n * n + 2 * d + np.arange(d), np.full(n + 2, objective_row), np.array([objective_row])]
    )
    columns.extend([np.full(d, eta_start), np.arange(n + 2), np.array([eta_start])])
    # sum_i q_i*a_i+sum_k rho_k*lambda_k <= eta.
    values.extend([-z, np.r_[problem.q, problem.radii], -np.ones(1)])
    matrix = _matrix(rows, columns, values, (objective_row + 1, count))
    raw, info = _solve(
        np.r_[np.zeros(eta_start), 1.0],
        matrix,
        np.r_[-np.ones(n * n), np.zeros(3 * d + 1)],
        options,
        deadline,
        bounds=[(0, None)] * (n + 2) + [(-1, 1)] * n + [(0, None)] * (2 * d + 1),
    )
    info["formulation"] = "selection_epigraph"
    if raw is None:
        return None, info
    return _repair_primal(problem, raw, info, z=z)


def _dual_fixed(problem, support, options, deadline):
    """Eq. (3): hinge LFD for fixed-selection evaluation without cuts."""
    n = problem.size
    count, h_start = n * n + n, n * n
    rows, columns, values = _envelope_rows(problem, h_start, 2)
    distance = _distance(problem, support)
    for k in range(2):
        sources = np.flatnonzero(problem.y == k)
        pairs = (sources[:, None] * n + np.arange(n)).ravel()
        # Transport budget: sum_{i:class(i)=k,r} gamma_ir*c_ir(S)<=rho_k.
        rows.append(np.full(len(pairs), k))
        columns.append(pairs)
        values.append(distance[sources].ravel())
    matrix = _matrix(rows, columns, values, (2 + 2 * n, count))
    raw, info = _solve(
        # max H=2*sum_r min(p_0r,p_1r), with h_r<=p_kr.
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
    return _repair_fixed_dual(problem, raw, info, distance, options)
