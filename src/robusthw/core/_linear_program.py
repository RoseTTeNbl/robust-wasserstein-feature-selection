"""Sparse matrix assembly and the shared, deadline-aware HiGHS interface."""

import warnings
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


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
