"""Development checks behind the tables of directions.tex (Chinese note); not formal results.

All checks use the hinge loss, where V(S) = 2 m_S(rho_0 + rho_1) is a budgeted matching.
Run from the repository root: PYTHONPATH=src python3 note/directions/dev_checks.py
  --hinge      the solver of the theory note equals 2 m_S
  --witness    m_S equals its Lipschitz-witness form
  --gap        compact program, its relaxation and enumeration
  --oa         outer approximation with tangent cuts and with ratio cuts (about 10 minutes)
  --search     branch and bound with the commitment bound (about 40 minutes)
  --tolerance  the same search stopped at a statistical tolerance
  --noise      robust values when both classes have the same distribution
  --random     achievable value of randomized selection
"""
import argparse
import heapq
import itertools
import time
from math import comb

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, kron, vstack

from robusthw.algorithm.oracle import highs  # retries the HiGHS solve errors on small MILPs


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


def selection(x0, x1, rho, d, relax=False, time_limit=600, gap=None):
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
    options = {"time_limit": time_limit} | ({"mip_rel_gap": gap} if gap else {})
    start = time.perf_counter()
    result = milp(cost, constraints=LinearConstraint(A, lower, upper), integrality=integrality,
                  bounds=bounds, options=options)
    support = None if relax else tuple(int(m) for m in np.flatnonzero(np.rint(result.x[D:2 * D])))
    return 2 * result.fun, support, time.perf_counter() - start, result


def plan_costs(x0, x1, S, rho):
    """m_S(rho) and the costs kappa_m = sum_ij pi_ij |x0_i[m] - x1_j[m]| of an optimal plan pi."""
    n0, n1 = len(x0), len(x1)
    C = np.abs(x0[:, None, :] - x1[None, :, :]).reshape(n0 * n1, -1)
    A = vstack([coo_matrix(np.kron(np.eye(n0), np.ones(n1))),
                coo_matrix(np.kron(np.ones(n0), np.eye(n1))),
                coo_matrix(C[:, list(S)].sum(1)[None, :])])
    b = np.r_[np.full(n0, 1 / n0), np.full(n1, 1 / n1), rho]
    result = linprog(-np.ones(n0 * n1), A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    return -result.fun, np.maximum(result.x, 0) @ C


def outer_approximation(x0, x1, rho, d, ratio):
    """Algorithm 1 with tangent cuts V(T) >= V(S)(1 - pi(T)) or ratio cuts V(T) >= V(S)/(1 + pi(T)).

    pi_m = kappa_m / rho off S and 0 on S. Returns the optimum and the number of evaluated subsets.
    """
    D = x0.shape[1]
    support, cuts, best, seen = tuple(range(d)), [], np.inf, set()
    while support not in seen:
        seen.add(support)
        value, kappa = plan_costs(x0, x1, support, rho)
        best = min(best, value)
        pi = kappa / rho
        pi[list(support)] = 0
        cuts.append((value, pi))
        # Variables [z, theta]. Ratio: max theta <= (1 + pi z)/V. Tangent: min theta >= V(1 - pi z).
        sign = 1.0 if ratio else -1.0
        rows = [np.r_[-sign * pi / v, sign] if ratio else np.r_[-v * pi, -1.0] for v, pi in cuts]
        rhs = [1 / v if ratio else -v for v, _ in cuts]
        result = highs(milp, np.r_[np.zeros(D), -sign], integrality=np.r_[np.ones(D), 0],
                      bounds=Bounds(np.zeros(D + 1), np.r_[np.ones(D), np.inf]),
                      constraints=[LinearConstraint(np.array(rows), -np.inf, rhs),
                                   LinearConstraint(np.r_[np.ones(D), 0][None, :], d, d)])
        lower = 1 / result.x[-1] if ratio else result.x[-1]
        if best - lower <= 1e-7:
            break
        support = tuple(int(m) for m in np.flatnonzero(np.rint(result.x[:D])))
    return 2 * best, len(seen)


def check_oa(rho=0.6):
    for D, n, d, seed in [(10, 15, 2, 0), (15, 15, 2, 1), (12, 15, 3, 3), (20, 15, 2, 1)]:
        x0, x1 = shifted(D, n, seed)
        tangent, ratio = (outer_approximation(x0, x1, rho, d, flag) for flag in (False, True))
        print(f"D={D} N={2 * n} d={d} subsets={comb(D, d)} OPT={ratio[0]:.4f} "
              f"tangent cuts evaluate {tangent[1]}, ratio cuts evaluate {ratio[1]}", flush=True)


class Search:
    """Branch and bound for min_{|S|=d} m_S(rho) with the commitment bound at every node."""

    def __init__(self, x0, x1, rho):
        n0, n1 = len(x0), len(x1)
        self.C = np.abs(x0[:, None, :] - x1[None, :, :]).reshape(n0 * n1, -1)
        self.rows = vstack([coo_matrix(np.kron(np.eye(n0), np.ones(n1))),
                            coo_matrix(np.kron(np.ones(n0), np.eye(n1)))]).tocsr()
        self.b = np.r_[np.full(n0, 1 / n0), np.full(n1, 1 / n1)]
        self.rho, self.lps = rho, 0

    def bound(self, inside, free=(), r=0):
        """Largest matched mass over plans with kappa(inside) + (r largest kappa_m in free) <= rho,
        and the plan's kappa. Without free coordinates this is m_S(rho) for S = inside."""
        P, F = self.C.shape[0], len(free)
        cost = self.C[:, list(inside)].sum(1)
        if F == 0:
            A, b = vstack([self.rows, csr_matrix(cost[None, :])]), np.r_[self.b, self.rho]
        else:
            # Variables [pi, t, s]: r largest kappa_m = min r t + sum s, s_m >= kappa_m - t.
            top = hstack([csr_matrix(self.C[:, list(free)].T), -np.ones((F, 1)), -eye(F)])
            A = vstack([hstack([self.rows, csr_matrix((len(self.b), 1 + F))]),
                        csr_matrix(np.r_[cost, r, np.ones(F)][None, :]), top])
            b = np.r_[self.b, self.rho, np.zeros(F)]
        result = linprog(np.r_[-np.ones(P), np.zeros(A.shape[1] - P)], A_ub=A, b_ub=b,
                         bounds=(0, None), method="highs")
        self.lps += 1
        return -result.fun, np.maximum(result.x[:P], 0) @ self.C

    def local(self, d):
        """Swap search guided by the prices kappa: bring in costly coordinates, drop cheap ones."""
        everything = tuple(range(self.C.shape[1]))
        support = tuple(sorted(int(m) for m in np.argsort(-self.bound(everything)[1])[:d]))
        value, kappa = self.bound(support)
        seen = {support}
        while True:
            outside = sorted(set(range(self.C.shape[1])) - set(support), key=lambda m: -kappa[m])
            for m in outside[:3]:
                for j in sorted(support, key=lambda j: kappa[j])[:2]:
                    other = tuple(sorted(set(support) - {j} | {m}))
                    if other in seen:
                        continue
                    seen.add(other)
                    other_value, other_kappa = self.bound(other)
                    if other_value < value - 1e-9:
                        break
                else:
                    continue
                break
            else:
                return value, support
            support, value, kappa = other, other_value, other_kappa

    def run(self, d, tolerance=1e-6):
        """Best-first search; stops when no node can beat the incumbent by more than `tolerance`."""
        upper, best = self.local(d)
        self.start = (2 * upper, self.lps)              # the local-search incumbent and its LPs
        root, kappa = self.bound((), tuple(range(self.C.shape[1])), d)
        heap = [(root, (), tuple(range(self.C.shape[1])), tuple(kappa))]
        while heap and heap[0][0] < upper - tolerance:
            _, inside, free, kappa = heapq.heappop(heap)
            m = max(free, key=lambda j: kappa[j])       # the coordinate the adversary uses most
            rest = tuple(j for j in free if j != m)
            for child in (inside + (m,), inside):
                r = d - len(child)
                if r > len(rest):
                    continue
                if r in (0, len(rest)):                  # a leaf: one subset
                    leaf = tuple(sorted(child + (rest if r else ())))
                    value = self.bound(leaf)[0]
                    upper, best = min((upper, best), (value, leaf))
                    continue
                value, child_kappa = self.bound(child, rest, r)
                if value < upper - tolerance:
                    heapq.heappush(heap, (value, child, rest, tuple(child_kappa)))
        return 2 * upper, best, 2 * root


def check_search(cases=((12, 15, 3, 3), (20, 15, 3, 2), (30, 20, 3, 4), (50, 20, 3, 5),
                        (100, 20, 3, 6), (30, 25, 3, 8), (30, 15, 4, 9)), rho=0.6):
    for D, n, d, seed in cases:
        x0, x1 = shifted(D, n, seed)
        search, start = Search(x0, x1, rho), time.perf_counter()
        value, S, root = search.run(d)
        print(f"D={D} N={2 * n} d={d}: local search {search.start[0]:.4f} ({search.start[1]} LPs), "
              f"optimum {value:.4f} S={S}, root bound {root:.4f}, LPs {search.lps} of "
              f"{comb(D, d)} subsets, {time.perf_counter() - start:.0f}s", flush=True)


def check_tolerance(cases=((50, 20, 3, 5, 0.6), (50, 20, 3, 5, 2.0))):
    for D, n, d, seed, rho in cases:
        x0, x1 = shifted(D, n, seed)
        for tolerance in (0.0, 0.05, 0.1):              # on the scale of V
            search, start = Search(x0, x1, rho), time.perf_counter()
            value, S, _ = search.run(d, tolerance / 2 + 1e-6)
            print(f"D={D} N={2 * n} d={d} rho={rho} tolerance={tolerance}: value {value:.4f} "
                  f"S={S}, LPs {search.lps}, {time.perf_counter() - start:.0f}s", flush=True)


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


def check_gap(rho=0.6):
    for D, n, d, seed, enumerate_all in [(10, 15, 2, 0, True), (15, 15, 2, 1, True),
                                         (12, 15, 3, 3, True), (30, 20, 3, 4, True),
                                         (50, 20, 3, 5, False), (100, 20, 3, 6, False)]:
        x0, x1 = shifted(D, n, seed)
        relaxed = selection(x0, x1, rho, d, relax=True)[0]
        value, S, seconds, result = selection(x0, x1, rho, d)
        nodes = getattr(result, "mip_node_count", None)
        line = (f"D={D} N={2 * n} d={d} relaxation={relaxed:.4f} program={value:.4f} S={S} "
                f"{seconds:.1f}s nodes={nodes} status={result.status}")
        if enumerate_all:
            start = time.perf_counter()
            best = min((2 * matching(x0, x1, T, rho), T)
                       for T in itertools.combinations(range(D), d))
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


def randomized(x0, x1, rho, d, radii=24):
    """W_T of Theorem 18, with tangents of each V_S at radii in (0, rho]."""
    D = x0.shape[1]
    rows, rhs = [], []
    for S in itertools.combinations(range(D), d):
        for r in np.linspace(0, rho, radii)[1:]:
            m, lam = matching(x0, x1, S, r, price=True)
            row = np.zeros(D + 1)                       # t <= A + Lambda * r(S)
            row[-1], row[list(S)] = 1, -2 * lam
            rows.append(row)
            rhs.append(2 * (m - lam * r))
    rows.append(np.r_[np.ones(D), 0.0])
    rhs.append(rho)
    result = linprog(np.r_[np.zeros(D), -1.0], A_ub=np.array(rows), b_ub=np.array(rhs),
                     bounds=[(0, None)] * D + [(None, None)], method="highs")
    return -result.fun


def check_random(rho=0.6):
    for D, n, d, seed in [(10, 15, 1, 0), (10, 15, 2, 0), (15, 15, 2, 1), (12, 15, 3, 3)]:
        x0, x1 = shifted(D, n, seed)
        fixed = min(2 * matching(x0, x1, S, rho) for S in itertools.combinations(range(D), d))
        relaxed = selection(x0, x1, rho, d, relax=True)[0]
        value = randomized(x0, x1, rho, d)
        print(f"D={D} N={2 * n} d={d} fixed={fixed:.4f} randomized={value:.4f} "
              f"relaxation={relaxed:.4f}", flush=True)


if __name__ == "__main__":
    checks = {"hinge": check_hinge, "witness": check_witness, "gap": check_gap, "oa": check_oa,
              "search": check_search, "tolerance": check_tolerance, "noise": check_noise,
              "random": check_random}
    parser = argparse.ArgumentParser()
    for flag in checks:
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    for flag, check in checks.items():
        if getattr(args, flag):
            check()
