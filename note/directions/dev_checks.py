"""Development checks behind the tables of directions.tex (not formal results).

Run from the repository root: PYTHONPATH=src python3 note/directions/dev_checks.py
  --hinge   compare the solver of the theory note with 2*m_S (Proposition 2.2)
  --gap     Table 1: compact program (2), its relaxation, and enumeration
  --noise   Table 2: robust values when both classes have the same distribution
"""
import argparse
import itertools
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, kron, vstack


def matching(x0, x1, S, rho):
    """m_S(rho) of Eq. (1): largest cross-class mass matched within budget rho."""
    n0, n1, S = len(x0), len(x1), list(S)
    C = np.abs(x0[:, None, S] - x1[None, :, S]).sum(-1).ravel() if S else np.zeros(n0 * n1)
    A = vstack([coo_matrix(np.kron(np.eye(n0), np.ones(n1))),
                coo_matrix(np.kron(np.ones(n0), np.eye(n1))), coo_matrix(C[None, :])])
    b = np.r_[np.full(n0, 1 / n0), np.full(n1, 1 / n1), rho]
    return -linprog(-np.ones(n0 * n1), A_ub=A, b_ub=b, bounds=(0, None), method="highs").fun


def selection(x0, x1, rho, d, relax=False, time_limit=600):
    """Program (2) with variables [w, z, lambda, u, v]; returns V-scale value and S."""
    n0, n1, D = len(x0), len(x1), x0.shape[1]
    C = np.abs(x0[:, None, :] - x1[None, :, :]).reshape(n0 * n1, D)
    cost = np.r_[np.zeros(2 * D), rho, np.full(n0, 1 / n0), np.full(n1, 1 / n1)]
    A = vstack([
        hstack([csr_matrix(C), csr_matrix((n0 * n1, D + 1)),
                kron(eye(n0), np.ones((n1, 1))), kron(np.ones((n0, 1)), eye(n1))]),
        hstack([eye(D), csr_matrix((D, D)), -np.ones((D, 1)), csr_matrix((D, n0 + n1))]),
        hstack([rho * eye(D), -eye(D), csr_matrix((D, 1 + n0 + n1))]),
        csr_matrix(np.r_[np.ones(D), np.zeros(D), -d, np.zeros(n0 + n1)][None, :]),
        csr_matrix(np.r_[np.zeros(D), np.ones(D), 0, np.zeros(n0 + n1)][None, :]),
    ]).tocsr()
    lower = np.r_[np.ones(n0 * n1), np.full(2 * D + 1, -np.inf), d]
    upper = np.r_[np.full(n0 * n1, np.inf), np.zeros(2 * D + 1), d]
    integrality = np.r_[np.zeros(D), np.full(D, 0 if relax else 1), np.zeros(1 + n0 + n1)]
    bounds = Bounds(np.zeros(len(cost)),
                    np.r_[np.full(D, np.inf), np.ones(D), 1 / rho, np.full(n0 + n1, np.inf)])
    start = time.perf_counter()
    result = milp(cost, constraints=LinearConstraint(A, lower, upper), integrality=integrality,
                  bounds=bounds, options={"time_limit": time_limit})
    support = None if relax else tuple(int(m) for m in np.flatnonzero(np.rint(result.x[D:2 * D])))
    return 2 * result.fun, support, time.perf_counter() - start, result


def shifted(D, n, seed):
    rng = np.random.default_rng(seed)
    x0, x1 = rng.normal(size=(n, D)), rng.normal(size=(n, D))
    x1[:, :3] += 0.8
    x1[:, 3] *= 1.8
    return x0, x1


def check_hinge():
    from robusthw.core import Problem, SolveOptions, fit_fixed
    x0, x1 = shifted(8, 10, 0)
    rng, options, worst = np.random.default_rng(5), SolveOptions(time_limit=60, fit_time_limit=60), 0.0
    for radii in [(0.3, 0.3), (0.5, 0.1), (0.05, 0.55), (1.0, 0.2)]:
        problem = Problem(x0, x1, np.array(radii))
        for _ in range(6):
            S = tuple(sorted(rng.choice(8, size=rng.integers(1, 5), replace=False)))
            fit = fit_fixed(problem, S, options, with_cut=False)
            target = 2 * matching(x0, x1, S, sum(radii))
            worst = max(worst, abs(fit.upper - target), abs(fit.lower - target))
    print(f"max |V(S) - 2 m_S(rho0+rho1)| = {worst:.1e}")


def check_gap(rho=0.6):
    for D, n, d, seed, enumerate_all in [(10, 15, 2, 0, True), (15, 15, 2, 1, True),
                                         (12, 15, 3, 3, True), (30, 20, 3, 4, True),
                                         (50, 20, 3, 5, False), (100, 20, 3, 6, False)]:
        x0, x1 = shifted(D, n, seed)
        relaxed = selection(x0, x1, rho, d, relax=True)[0]
        value, S, seconds, result = selection(x0, x1, rho, d)
        line = (f"D={D} N={2 * n} d={d} relaxation={relaxed:.4f} program={value:.4f} S={S} "
                f"{seconds:.1f}s nodes={getattr(result, 'mip_node_count', None)} status={result.status}")
        if enumerate_all:
            start = time.perf_counter()
            best = min((2 * matching(x0, x1, T, rho), T) for T in itertools.combinations(range(D), d))
            line += f" enumeration={best[0]:.4f} S={best[1]} {time.perf_counter() - start:.1f}s"
        print(line, flush=True)


def check_noise(rho=0.6, n=15, d=2):
    for D in [1, 2, 3, 5, 10, 20, 50]:
        full, best = [], []
        for seed in range(5):
            rng = np.random.default_rng(100 + seed)
            x0, x1 = rng.normal(size=(n, D)), rng.normal(size=(n, D))
            full.append(2 * matching(x0, x1, range(D), rho))
            if d <= D <= 20:
                best.append(min(2 * matching(x0, x1, S, rho)
                                for S in itertools.combinations(range(D), d)))
        print(f"D={D} V([D])={np.mean(full):.3f} min_|S|={d} V(S)="
              f"{np.mean(best) if best else float('nan'):.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for flag in ("hinge", "gap", "noise"):
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    if args.hinge:
        check_hinge()
    if args.gap:
        check_gap()
    if args.noise:
        check_noise()
