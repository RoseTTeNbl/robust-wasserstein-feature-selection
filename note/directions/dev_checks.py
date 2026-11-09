"""Two checks of the hinge example in directions.tex, Section 2; not experiments.

The experiments of the note are in src/robusthw/experiments/study20.py (results/study20-v3).
Run from the repository root: PYTHONPATH=src python3 note/directions/dev_checks.py
  --hinge      the solver of the theory note equals 2 m_S (Proposition 2)
  --witness    m_S equals its Lipschitz-witness form (Proposition 3)
"""
import argparse

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, eye, hstack, kron, vstack


def matching(x0, x1, S, rho, price=False):
    """m_S(rho) of Eq. (1); with price=True also the budget price lambda (the slope in rho)."""
    n0, n1, S = len(x0), len(x1), list(S)
    C = np.abs(x0[:, None, S] - x1[None, :, S]).sum(-1).ravel() if S else np.zeros(n0 * n1)
    A = vstack([coo_matrix(np.kron(np.eye(n0), np.ones(n1))),
                coo_matrix(np.kron(np.ones(n0), np.eye(n1))), coo_matrix(C[None, :])])
    b = np.r_[np.full(n0, 1 / n0), np.full(n1, 1 / n1), rho]
    result = linprog(-np.ones(n0 * n1), A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    return (-result.fun, -result.ineqlin.marginals[-1]) if price else -result.fun


def lipschitz_witness(x0, x1, S, lam):
    """beta_lambda of Section 2.3 on the projected samples: values h in [-1/2, 1/2]."""
    n0, n1, S = len(x0), len(x1), list(S)
    C = np.abs(x0[:, None, S] - x1[None, :, S]).sum(-1).ravel()
    A = hstack([kron(eye(n0), np.ones((n1, 1))), -kron(np.ones((n0, 1)), eye(n1))])
    c = np.r_[np.full(n0, -1 / n0), np.full(n1, 1 / n1)]
    return -linprog(c, A_ub=A, b_ub=lam * C, bounds=(-0.5, 0.5), method="highs").fun


def shifted(D, n, seed):
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=(n, D)), rng.normal(size=(n, D))
    x1[:, :3] += 0.8
    x1[:, 3] *= 1.8
    return x0, x1


def check_hinge():
    from robusthw.algorithm import Problem, SolveOptions, fit_fixed
    x0, x1 = shifted(8, 10, 0)
    rng, worst = np.random.default_rng(5), 0.0
    options = SolveOptions(time_limit=60, fit_time_limit=60)
    for radii in [(0.3, 0.3), (0.5, 0.1), (0.05, 0.55), (1.0, 0.2)]:
        problem = Problem(x0, x1, np.array(radii))
        for _ in range(6):
            S = tuple(sorted(rng.choice(8, size=rng.integers(1, 5), replace=False)))
            fit = fit_fixed(problem, S, options, with_cut=False)
            target = 2 * matching(x0, x1, S, sum(radii))
            worst = max(worst, abs(fit.upper - target), abs(fit.lower - target))
    print(f"max |V(S) - 2 m_S(rho0+rho1)| = {worst:.1e}")


def check_witness():
    from scipy.optimize import minimize_scalar
    worst = 0.0
    for seed in range(4):
        x0, x1 = shifted(6, 8 + seed, seed)
        for S in [(0,), (0, 4), (1, 2, 5)]:
            for rho in [0.1, 0.4, 0.9]:
                best = minimize_scalar(lambda lam: rho * lam - lipschitz_witness(x0, x1, S, lam),
                                       bounds=(0, 1 / rho), method="bounded",
                                       options={"xatol": 1e-9})
                worst = max(worst, abs(matching(x0, x1, S, rho) - (1 + best.fun)))
    print(f"max |m_S - (1 - max_lambda(beta_lambda - lambda rho))| = {worst:.1e}")


if __name__ == "__main__":
    checks = {"hinge": check_hinge, "witness": check_witness}
    parser = argparse.ArgumentParser()
    for flag in checks:
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    for flag, check in checks.items():
        if getattr(args, flag):
            check()
