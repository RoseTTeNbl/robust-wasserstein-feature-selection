"""Twenty-dimensional checks of the research-directions note (standard hinge loss).

The robust value is computed through its matching form V(S)=2*m_S(rho_0+rho_1)
(directions note, Proposition 2); the deployed detector and its certificate come from
the core oracle fit_fixed at the selected subset. These are development checks of
the research directions, not formal performance results.

Experiments (configs/study20.json):
  null          identical classes: radius needed to avoid false separation
  signal        mean shift on two coordinates: validity, recovery, choice of d
  computation   compact program (2) against enumeration and outer approximation
  randomized    randomized d-sparse tests against fixed selection
"""

import json
import os
import platform
import subprocess
from itertools import combinations
from math import e, pi
from multiprocessing import Pool
from pathlib import Path
from statistics import NormalDist
from time import perf_counter

import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, linprog, milp

from robusthw.algorithm import MODEL_VERSION, Problem, SolveOptions, fit_fixed
from robusthw.algorithm import select

from .storage import json_value

PURPOSE = {"null": 1, "signal": 2, "test": 3}


# ---------------------------------------------------------------- data


def sample(config, seed, n, purpose, shift):
    """Class 0 ~ N(0, I_D); class 1 ~ N(shift * 1_signal, I_D); n rows each."""
    rng = np.random.default_rng([seed, n, PURPOSE[purpose]])
    x0 = rng.normal(size=(n, config["dimension"]))
    x1 = rng.normal(size=(n, config["dimension"]))
    x1[:, config["signal"]] += shift
    return x0, x1


def pair_costs(x0, x1):
    """C[i, j, m] = |x_i^(0)[m] - x_j^(1)[m]|, so c_ij(S) = C[i, j, S].sum()."""
    return np.abs(x0[:, None, :] - x1[None, :, :])


def subsets(dimension, budget):
    return list(combinations(range(dimension), budget))


def bayes_hinge(config, support):
    """2(1 - TV) between the projected classes: the Bayes hinge risk B_S."""
    active = len(set(support) & set(config["signal"]))
    tv = 2 * NormalDist().cdf(config["shift"] * np.sqrt(active) / 2) - 1
    return 2 * (1 - tv)


# ---------------------------------------------------------------- matching form


class Matching:
    """Budgeted cross-class matchings of one pair-cost matrix (n_0 = n_1 = n).

    cost(k) is the smallest total cost of k disjoint cross-class pairs, divided
    by n (each pair carries mass 1/n). It is convex in k, so m_S(rho), the
    largest mass matchable within budget rho, interpolates its inverse.
    """

    def __init__(self, cost):
        if cost.shape[0] != cost.shape[1]:
            raise ValueError("Matching needs equal class sizes")
        self.matrix, self.n = cost, len(cost)
        self.big = float(cost.sum()) + 1.0
        self.cache = {0: 0.0}

    def cost(self, k):
        if k not in self.cache:
            # Pad with n-k dummy rows and columns: a perfect assignment that never
            # pairs two dummies matches exactly k real pairs.
            n, size = self.n, 2 * self.n - k
            padded = np.zeros((size, size))
            padded[:n, :n] = self.matrix
            padded[n:, n:] = self.big
            rows, columns = linear_sum_assignment(padded)
            real = (rows < n) & (columns < n)
            self.cache[k] = float(self.matrix[rows[real], columns[real]].sum()) / n
        return self.cache[k]

    def budget(self, mass):
        """Smallest rho with m_S(rho) >= mass (mass in [0, 1])."""
        position = mass * self.n
        k = min(int(np.floor(position)), self.n - 1)
        if position == k:
            return self.cost(k)
        return self.cost(k) + (position - k) * (self.cost(k + 1) - self.cost(k))

    def value(self, rho):
        """m_S(rho) of Eq. (1) in the directions note."""
        n = self.n
        if self.cost(n) <= rho:
            return 1.0
        low, high = 0, n  # cost(low) <= rho < cost(high)
        while high - low > 1:
            middle = (low + high) // 2
            if self.cost(middle) <= rho:
                low = middle
            else:
                high = middle
        start, stop = self.cost(low), self.cost(high)
        return (low + (rho - start) / (stop - start)) / n


def matching_lp(cost, rho):
    """m_S(rho) and its dual (u, v, lambda) by linear programming (any n_0, n_1)."""
    n0, n1 = cost.shape
    matrix = sparse.vstack(
        [
            sparse.kron(sparse.eye(n0), np.ones((1, n1))),
            sparse.kron(np.ones((1, n0)), sparse.eye(n1)),
            sparse.csr_matrix(cost.ravel()[None, :]),
        ]
    ).tocsr()
    rhs = np.r_[np.full(n0, 1 / n0), np.full(n1, 1 / n1), rho]
    result = linprog(-np.ones(n0 * n1), A_ub=matrix, b_ub=rhs, bounds=(0, None), method="highs")
    duals = -result.ineqlin.marginals
    # Dual of Eq. (1): min rho*lambda + mean(u) + mean(v) s.t. u_i + v_j + lambda*c_ij >= 1.
    return -result.fun, duals[:n0], duals[n0 : n0 + n1], duals[-1]


def witness(x1, support, v, lam):
    """Detector f = 2h of Proposition 3 built from a matching dual (v, lambda).

    h = v_j - 1/2 on class-1 points, lowered to at most 1/2, is extended by the
    clipped McShane formula; on class-0 points it is then at least 1/2 - u_i.
    """
    selected = list(support)
    anchors = x1[:, selected]
    values = np.minimum(v - 0.5, 0.5)

    def score(x):
        distances = np.abs(x[:, None, selected] - anchors[None, :, :]).sum(axis=2)
        h = np.min(values[None, :] + lam * distances, axis=1)
        return 2 * np.clip(h, -0.5, 0.5)

    return score


def compact_program(costs, rho, budget, relax=False, time_limit=600.0):
    """Program (2) of the directions note, variables [w (D), z (D), lambda, u (n0), v (n1)].

    Returns the value on the V scale (twice the program value), the support and
    the HiGHS result.
    """
    n0, n1, d = costs.shape
    pairs = costs.reshape(n0 * n1, d)
    objective = np.r_[np.zeros(2 * d), rho, np.full(n0, 1 / n0), np.full(n1, 1 / n1)]
    matrix = sparse.vstack(
        [
            # u_i + v_j + sum_m w_m C^m_ij >= 1.
            sparse.hstack(
                [
                    sparse.csr_matrix(pairs),
                    sparse.csr_matrix((n0 * n1, d + 1)),
                    sparse.kron(sparse.eye(n0), np.ones((n1, 1))),
                    sparse.kron(np.ones((n0, 1)), sparse.eye(n1)),
                ]
            ),
            # w_m <= lambda.
            sparse.hstack(
                [sparse.eye(d), sparse.csr_matrix((d, d)), -np.ones((d, 1)),
                 sparse.csr_matrix((d, n0 + n1))]
            ),
            # rho * w_m <= z_m.
            sparse.hstack(
                [rho * sparse.eye(d), -sparse.eye(d), sparse.csr_matrix((d, 1 + n0 + n1))]
            ),
            # sum_m w_m <= budget * lambda, and sum_m z_m = budget.
            sparse.csr_matrix(np.r_[np.ones(d), np.zeros(d), -budget, np.zeros(n0 + n1)][None, :]),
            sparse.csr_matrix(np.r_[np.zeros(d), np.ones(d), 0, np.zeros(n0 + n1)][None, :]),
        ]
    ).tocsr()
    lower = np.r_[np.ones(n0 * n1), np.full(2 * d + 1, -np.inf), budget]
    upper = np.r_[np.full(n0 * n1, np.inf), np.zeros(2 * d + 1), budget]
    integrality = np.r_[np.zeros(d), np.full(d, 0 if relax else 1), np.zeros(1 + n0 + n1)]
    bounds = Bounds(np.zeros(len(objective)),
                    np.r_[np.full(d, np.inf), np.ones(d), 1 / rho, np.full(n0 + n1, np.inf)])
    result = milp(objective, constraints=LinearConstraint(matrix, lower, upper),
                  integrality=integrality, bounds=bounds,
                  options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if result.x is None:
        return None, None, result
    support = None if relax else tuple(int(m) for m in np.flatnonzero(np.rint(result.x[d : 2 * d])))
    return 2 * result.fun, support, result


# ---------------------------------------------------------------- experiment 1: null


def radius_lower_bound(d, n, mass):
    """Proposition 8 for N(0, I_d): V(S) >= 2*mass needs rho_0+rho_1 >= this."""
    density = (2 * pi) ** (-d / 2)
    c = 2 * (1 - mass)
    return (1 - c) * d / (4 * e) * (4 * density * n) ** (-1 / d)


def null_task(config, seed, n):
    """Radii that keep identical classes unseparated, for every subset of each size."""
    x0, x1 = sample(config, seed, n, "null", 0.0)
    costs = pair_costs(x0, x1)
    mass, record = config["null_mass"], {"seed": seed, "n": n, "sizes": {}}
    for d in [*config["budgets"], config["dimension"]]:
        needed, fixed = [], []
        for support in subsets(config["dimension"], d):
            matching = Matching(costs[:, :, list(support)].sum(axis=2))
            # V(S; rho) >= 2*mass  <=>  rho >= needed radius of S.
            needed.append(matching.budget(mass))
            fixed.append(2 * matching.value(config["fixed_radius"]))
        needed = np.array(needed)
        bound = radius_lower_bound(d, n, mass)
        record["sizes"][str(d)] = {
            "subsets": len(needed),
            "needed_max": float(needed.max()),
            "needed_median": float(np.median(needed)),
            "needed_min": float(needed.min()),
            "lower_bound": bound,
            "below_lower_bound": int(np.sum(needed < bound)),
            "fixed_radius_best_value": float(min(fixed)),
            "fixed_radius_median_value": float(np.median(fixed)),
        }
    return record


def honest_radii(records, config):
    """rho_h(d, n): quantile over development draws of the largest needed radius."""
    table = {}
    for d in [*config["budgets"], config["dimension"]]:
        for n in config["train_sizes"]:
            values = [r["sizes"][str(d)]["needed_max"] for r in records if r["n"] == n]
            table.setdefault(str(d), {})[str(n)] = float(
                np.quantile(values, config["honest_quantile"])
            )
    return table


# ---------------------------------------------------------------- experiment 2: signal


def _radius_levels(config, radii, n):
    levels = {"fixed": {d: config["fixed_radius"] for d in config["budgets"]}}
    for factor in config["radius_factors"]:
        levels[f"honest_x{factor:g}"] = {
            d: factor * radii[str(d)][str(n)] for d in config["budgets"]
        }
    return levels


def _detector(config, x0, x1, support, rho, tests):
    started = perf_counter()
    options = SolveOptions(fit_time_limit=config["fit_time_limit"], threads=1)
    fit = fit_fixed(Problem(x0, x1, [rho / 2, rho / 2]), support, options, with_cut=False)
    score0, score1 = fit.predict_score(tests[0]), fit.predict_score(tests[1])
    losses = np.r_[1 - score0, 1 + score1]
    m = len(score0)
    return {
        "support": list(support),
        "core_upper": fit.upper,
        "core_lower": fit.lower,
        "core_status": fit.status,
        "test_hinge": float(np.mean(1 - score0) + np.mean(1 + score1)),
        # Standard error of the two-class Monte Carlo sum.
        "test_hinge_se": float(np.sqrt((np.var(losses[:m]) + np.var(losses[m:])) / m)),
        "test_errors": float(np.mean(score0 < 0) + np.mean(score1 >= 0)),
        "bayes_hinge": bayes_hinge(config, support),
        "fit_seconds": perf_counter() - started,
    }


def signal_task(config, radii, seed, n, shift, levels=None):
    """Select at each radius level and budget; certify and test the detectors."""
    config = {**config, "shift": shift}
    x0, x1 = sample(config, seed, n, "signal", shift)
    tests = sample(config, seed, config["test_size"], "test", shift)
    costs = pair_costs(x0, x1)
    matchers = {
        support: Matching(costs[:, :, list(support)].sum(axis=2))
        for d in config["budgets"]
        for support in subsets(config["dimension"], d)
    }
    signal, record = tuple(config["signal"]), {"seed": seed, "n": n, "shift": shift, "levels": {}}
    for level, radius in _radius_levels(config, radii, n).items():
        if levels is not None and level not in levels:
            continue
        chosen = {}
        for d in config["budgets"]:
            values = {s: 2 * matchers[s].value(radius[d]) for s in subsets(config["dimension"], d)}
            best = min(values, key=lambda s: (values[s], s))
            ordered = sorted(values.values())
            chosen[d] = {
                "support": list(best),
                "value": values[best],
                "runner_up": ordered[1] if len(ordered) > 1 else None,
                "radius": radius[d],
                "signal_hits": len(set(best) & set(signal)),
            }
        # Corollary 11: the budget with the smallest certificate.
        adaptive = min(config["budgets"], key=lambda d: (chosen[d]["value"], d))
        entry = {"budgets": {str(d): chosen[d] for d in config["budgets"]}, "adaptive": adaptive}
        true_size = len(signal)
        entry["detector_true_size"] = _detector(
            config, x0, x1, tuple(chosen[true_size]["support"]), radius[true_size], tests
        )
        entry["detector_true_size"]["certificate"] = chosen[true_size]["value"]
        entry["detector_adaptive"] = _detector(
            config, x0, x1, tuple(chosen[adaptive]["support"]), radius[adaptive], tests
        )
        entry["detector_adaptive"]["certificate"] = chosen[adaptive]["value"]
        record["levels"][level] = entry
    return record


# ---------------------------------------------------------------- experiment 3: computation


def computation_task(config, seed, n, budget):
    rho = config["fixed_radius"]
    x0, x1 = sample(config, seed, n, "signal", config["shift"])
    costs = pair_costs(x0, x1)
    record = {"seed": seed, "n": n, "budget": budget, "radius": rho}
    started = perf_counter()
    values = {
        s: 2 * Matching(costs[:, :, list(s)].sum(axis=2)).value(rho)
        for s in subsets(config["dimension"], budget)
    }
    best = min(values, key=lambda s: (values[s], s))
    record["enumeration"] = {"value": values[best], "support": list(best),
                             "seconds": perf_counter() - started, "subsets": len(values)}
    started = perf_counter()
    relaxed, _, _ = compact_program(costs, rho, budget, relax=True)
    record["relaxation"] = {"value": relaxed, "seconds": perf_counter() - started}
    started = perf_counter()
    value, support, result = compact_program(costs, rho, budget,
                                             time_limit=config["search_time_limit"])
    record["compact_program"] = {
        "value": value,
        "support": None if support is None else list(support),
        "status": int(result.status),
        "message": str(result.message),
        "nodes": getattr(result, "mip_node_count", None),
        "dual_bound": None if getattr(result, "mip_dual_bound", None) is None
        else 2 * float(result.mip_dual_bound),
        "seconds": perf_counter() - started,
    }
    options = SolveOptions(time_limit=config["search_time_limit"],
                           fit_time_limit=config["fit_time_limit"], threads=1)
    search = select(Problem(x0, x1, [rho / 2, rho / 2]), budget, "oa", options)
    record["outer_approximation"] = {
        "upper": search.upper,
        "lower": search.lower,
        "support": list(search.support),
        "status": search.status,
        "evaluated_subsets": search.fit_count,
        "seconds": search.runtime,
        "oracle_seconds": search.oracle_seconds,
        "master_seconds": search.master_seconds,
    }
    return record


# ---------------------------------------------------------------- experiment 4: randomized


def _water_filling(matchers, rho, tolerance=1e-10):
    """Theorem 19 for singletons: max{t : sum_m V_m^{-1}(t) <= rho}."""
    low, high = 0.0, 2.0
    while high - low > tolerance:
        t = (low + high) / 2
        if sum(m.budget(t / 2) for m in matchers) <= rho:
            low = t
        else:
            high = t
    return low


def randomized_task(config, seed, n, budget):
    rho, grid = config["fixed_radius"], config["tangent_radii"]
    x0, x1 = sample(config, seed, n, "signal", config["shift"])
    tests = sample(config, seed, config["test_size"], "test", config["shift"])
    costs, d = pair_costs(x0, x1), config["dimension"]
    tangents, rows, rhs = [], [], []
    for support in subsets(d, budget):
        cost = costs[:, :, list(support)].sum(axis=2)
        for r in np.linspace(0, rho, grid)[1:]:
            m, _, v, lam = matching_lp(cost, r)
            # V_S(r') <= A + Lambda*r' with A = 2(m - lambda*r), Lambda = 2*lambda.
            tangents.append((support, r, v, lam))
            row = np.zeros(d + 1)
            row[-1], row[list(support)] = 1.0, -2 * lam
            rows.append(row)
            rhs.append(2 * (m - lam * r))
    rows.append(np.r_[np.ones(d), 0.0])
    rhs.append(rho)
    result = linprog(np.r_[np.zeros(d), -1.0], A_ub=np.array(rows), b_ub=np.array(rhs),
                     bounds=[(0, None)] * d + [(None, None)], method="highs")
    # Multipliers of the tangent rows are the defender's mixture p (Theorem 18).
    mixture = np.maximum(-result.ineqlin.marginals[:-1], 0)
    fixed_values = {s: 2 * matching_lp(costs[:, :, list(s)].sum(axis=2), rho)[0]
                    for s in subsets(d, budget)}
    best = min(fixed_values, key=lambda s: (fixed_values[s], s))

    def test_hinge(score):
        return float(np.mean(1 - score(tests[0])) + np.mean(1 + score(tests[1])))

    _, _, v, lam = matching_lp(costs[:, :, list(best)].sum(axis=2), rho)
    components = []
    for weight, (support, r, v_t, lam_t) in zip(mixture, tangents):
        if weight > 1e-9:
            components.append({"support": list(support), "radius": r, "weight": float(weight),
                               "test_hinge": test_hinge(witness(x1, support, v_t, lam_t))})
    record = {
        "seed": seed, "n": n, "budget": budget, "radius": rho,
        "fixed_value": fixed_values[best],
        "fixed_support": list(best),
        "fixed_test_hinge": test_hinge(witness(x1, best, v, lam)),
        "randomized_value": -result.fun,
        "budget_split": (result.x[:d] / rho).tolist(),
        "relaxation": compact_program(costs, rho, budget, relax=True)[0],
        "mixture": components,
        "mixture_test_hinge": float(sum(c["weight"] * c["test_hinge"] for c in components)
                                    / sum(c["weight"] for c in components)),
    }
    if budget == 1:
        matchers = [Matching(costs[:, :, m]) for m in range(d)]
        record["water_filling"] = _water_filling(matchers, rho)
    return record


# ---------------------------------------------------------------- driver


def _environment():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"model_version": MODEL_VERSION, "git_commit": commit, "git_dirty": dirty,
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), indent=1, sort_keys=True) + "\n")


def load(path):
    return json.loads(Path(path).read_text())


def run_parallel(function, tasks, workers):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with Pool(min(workers, len(tasks))) as pool:
        return pool.starmap(function, tasks)


def develop(config, output, workers):
    """Development: honest radii from null draws, then the signal strength."""
    seeds = config["development_seeds"]
    nulls = run_parallel(null_task, [(config, s, n) for s in seeds for n in config["train_sizes"]],
                         workers)
    radii = honest_radii(nulls, config)
    trials = run_parallel(
        signal_task,
        [(config, radii, s, n, shift, {"honest_x1"}) for s in seeds[: config["shift_seeds"]]
         for n in config["train_sizes"] for shift in config["shift_grid"]],
        workers,
    )
    rates = {}
    for shift in config["shift_grid"]:
        chosen = [t for t in trials if t["shift"] == shift and t["n"] == max(config["train_sizes"])]
        hits = [t["levels"]["honest_x1"]["budgets"][str(len(config["signal"]))]["signal_hits"]
                for t in chosen]
        rates[str(shift)] = float(np.mean(np.array(hits) == len(config["signal"])))
    passing = [s for s in config["shift_grid"] if rates[str(s)] >= config["shift_recovery_target"]]
    shift = min(passing) if passing else max(config["shift_grid"])
    save(Path(output) / "development.json", {
        "environment": _environment(), "config": config, "honest_radii": radii,
        "shift_recovery": rates, "chosen_shift": shift, "null_records": nulls,
        "signal_records": trials,
    })
    return radii, shift


def confirm(config, output, workers):
    development = load(Path(output) / "development.json")
    radii, shift = development["honest_radii"], development["chosen_shift"]
    config = {**config, "shift": shift}
    seeds = config["confirmation_seeds"]
    started = perf_counter()
    nulls = run_parallel(null_task, [(config, s, n) for s in seeds for n in config["train_sizes"]],
                         workers)
    save(Path(output) / "null.json", {"environment": _environment(), "records": nulls,
                                      "seconds": perf_counter() - started})
    started = perf_counter()
    signals = run_parallel(signal_task, [(config, radii, s, n, shift) for s in seeds
                                         for n in config["train_sizes"]], workers)
    save(Path(output) / "signal.json", {"environment": _environment(), "honest_radii": radii,
                                        "records": signals, "seconds": perf_counter() - started})
    computations = run_parallel(
        computation_task,
        [(config, seeds[0], n, b) for n in config["computation_sizes"]
         for b in config["computation_budgets"]],
        workers,
    )
    save(Path(output) / "computation.json",
         {"environment": _environment(), "records": computations})
    randomized = run_parallel(
        randomized_task,
        [(config, s, config["randomized_size"], b) for s in seeds[: config["randomized_seeds"]]
         for b in config["randomized_budgets"]],
        workers,
    )
    save(Path(output) / "randomized.json", {"environment": _environment(), "records": randomized})
