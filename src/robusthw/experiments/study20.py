"""Twenty-dimensional experiments: four ground-truth coordinates among sixteen distractors.

Each case draws two classes that differ only on four ground-truth coordinates; the other
sixteen coordinates are distractors, independent of everything and identically distributed
in both classes, so the likelihood ratio depends on the ground-truth coordinates alone. The
question is whether a test built from a few training points per class detects the
difference, and whether it finds the right coordinates.

Protocol of Xie, Gao and Xie (2021, Sec. 5.1) [XGX]: n training points per class, 1000 fresh
test points per class, the radius chosen by cross-validation, errors averaged over
independent trials. The signal types come from [XGX] and from Wang, Dey and Xie (2023,
Sec. 6) [WDX]. Hinge loss throughout, so V(S) = 2 m_S(rho) with rho = rho_0 + rho_1
(directions note, Proposition 2).

  accuracy      test errors, certificate and recovered coordinates of each method
  computation   exact selection at the cross-validated radius: enumeration, the compact
                program Eq. (4) of the directions note (HiGHS), outer approximation, branch and bound
"""

import heapq
import json
import os
import platform
import subprocess
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, linprog, milp
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from robusthw.algorithm import MODEL_VERSION, Problem, SolveOptions, fit_fixed, select

from .storage import json_value

D = 20
CASES = ("null", "mean_shift", "mixture", "interaction")
SUPPORT = (0, 1, 2, 3)  # ground-truth coordinates before shuffling: D/5, as d_true/D in [WDX]
TRUTH = {"null": (), "mean_shift": SUPPORT, "mixture": SUPPORT, "interaction": SUPPORT}
OPTIONS = SolveOptions(fit_time_limit=60, threads=1)

# ---------------------------------------------------------------- data

AR = 0.5 ** np.abs(np.subtract.outer(np.arange(4), np.arange(4)))  # [WDX] case I covariance
SHIFT = 2 / np.arange(1, 5)  # [WDX] case I, mu_s = tau/s with tau = 2
SPREAD = 2 / np.arange(1, 5)  # [WDX] case IV, tau = 2
CENTRE0, CENTRE1 = np.ones(4), np.array([1.0, 1.0, -1.0, -1.0])  # [XGX] Sec. 5.1 centres


def _gmm(rng, n, centre):
    """1/2 N(centre, I) + 1/2 N(-centre, I)."""
    return rng.normal(size=(n, len(centre))) + rng.choice([-1.0, 1.0], size=(n, 1)) * centre


def sample(case, n, rng):
    """n points of each class: ground-truth coordinates 0-3, distractors 4-19.

    mean_shift   [WDX] case I on the ground truth: N(0, Sigma) against N(mu, Sigma),
                 Sigma_ab = 0.5^|a-b|; distractors N(0, 1)
    mixture      [WDX] case IV: N(0, I) against the symmetric mixture N(+-mu, I), same mean,
                 larger spread along mu; distractors N(0, 1)
    interaction  [XGX] Sec. 5.1 mixtures on the ground truth, centres (1, 1, 1, 1) and
                 (1, 1, -1, -1); distractors 1/2 N(1, 1) + 1/2 N(-1, 1), so every coordinate
                 has the same one-coordinate distribution in both classes and the difference
                 lies only in how the ground-truth coordinates move together
    null         no ground truth: twenty N(0, 1) distractors in both classes
    """
    noise = (lambda size: rng.normal(size=size) + rng.choice([-1.0, 1.0], size=size)) \
        if case == "interaction" else (lambda size: rng.normal(size=size))
    x0, x1 = noise((n, D)), noise((n, D))
    if case == "mean_shift":
        root = np.linalg.cholesky(AR)
        x0[:, :4], x1[:, :4] = (rng.normal(size=(n, 4)) @ root.T,
                                rng.normal(size=(n, 4)) @ root.T + SHIFT)
    elif case == "mixture":
        x1[:, :4] += rng.choice([-1.0, 1.0], size=(n, 1)) * SPREAD
    elif case == "interaction":
        x0[:, :4], x1[:, :4] = _gmm(rng, n, CENTRE0), _gmm(rng, n, CENTRE1)
    return x0, x1


def log_ratio(case, x):
    """log p_1(x) / p_0(x); the distractors cancel. The Bayes test decides class 1 when > 0."""
    y = x[:, :4]
    if case == "null":
        return np.zeros(len(x))
    if case == "mean_shift":
        return (y - SHIFT / 2) @ np.linalg.solve(AR, SHIFT)
    if case == "mixture":
        return np.log(np.cosh(y @ SPREAD)) - SPREAD @ SPREAD / 2

    def log_density(c):
        return logsumexp([-((y - c) ** 2).sum(1) / 2, -((y + c) ** 2).sum(1) / 2], axis=0)

    return log_density(CENTRE1) - log_density(CENTRE0)


def draw(case, n, trial, test_size):
    """Training and test samples of one trial, coordinates in a random order, and the Bayes error.

    The order hides the true coordinates from every tie-break by index (when many subsets have
    the same value, the first one in index order is taken). The stream depends only on
    (case, n, trial).
    """
    rng = np.random.default_rng([CASES.index(case), n, trial])
    train, tests = sample(case, n, rng), sample(case, test_size, rng)
    bayes = test_errors(-log_ratio(case, tests[0]), -log_ratio(case, tests[1]), split=True)
    order = rng.permutation(D)
    truth = tuple(sorted(int(m) for m in np.argsort(order)[list(TRUTH[case])]))
    return [x[:, order] for x in train], [x[:, order] for x in tests], truth, bayes


# ---------------------------------------------------------------- robust values (hinge)


def matching_curve(cost):
    """cost_k = smallest total cost of k disjoint cross-class pairs, divided by n; k = 0..n.

    Pad with n - k dummy rows and columns: an assignment that never pairs two dummies
    matches exactly k real pairs.
    """
    n, big = len(cost), cost.sum() + 1.0
    curve = [0.0]
    for k in range(1, n + 1):
        padded = np.zeros((2 * n - k, 2 * n - k))
        padded[:n, :n], padded[n:, n:] = cost, big
        rows, columns = linear_sum_assignment(padded)
        real = (rows < n) & (columns < n)
        curve.append(cost[rows[real], columns[real]].sum() / n)
    return np.array(curve)


def robust_values(x0, x1, budget, radii):
    """V(S; rho) = 2 m_S(rho) for every |S| = budget (rows) and radius (columns).

    cost_k is convex in k, so m_S(rho), the largest mass matchable within budget rho,
    interpolates its inverse.
    """
    costs = np.abs(x0[:, None, :] - x1[None, :, :])
    subsets = list(combinations(range(x0.shape[1]), budget))
    grid = np.arange(len(x0) + 1) / len(x0)
    values = [2 * np.interp(radii, matching_curve(costs[:, :, s].sum(2)), grid) for s in subsets]
    return subsets, np.array(values)


def joint(table, d, column):
    """argmin_{|S| = d} V(S; rho); ties go to the first subset in lexicographic order."""
    subsets, values = table[d]
    return subsets[int(np.argmin(values[:, column]))]


def marginal(table, d, column):
    """The d coordinates with the smallest one-coordinate values V({m}; rho)."""
    singles = table[1][1][:, column]
    return tuple(sorted(int(m) for m in np.lexsort((np.arange(len(singles)), singles))[:d]))


def test_errors(score0, score1, split=False):
    """Average of the two error probabilities; a nonnegative score decides class 0.

    With split=True: the type I error (a class-0 point declared class 1), the type II error
    (a class-1 point declared class 0) and their average.
    """
    first, second = float(np.mean(score0 < 0)), float(np.mean(score1 >= 0))
    average = 0.5 * (first + second)
    return {"error": average, "type1": first, "type2": second} if split else average


def robust_fit(x0, x1, support, rho):
    return fit_fixed(Problem(x0, x1, [rho / 2, rho / 2]), support, OPTIONS, with_cut=False)


def validation(fit, x0, x1):
    """Error rate of the detector, then its hinge risk on the V scale (for ties)."""
    score0, score1 = fit.predict_score(x0), fit.predict_score(x1)
    return np.array([test_errors(score0, score1), np.mean(1 - score0) + np.mean(1 + score1)])


# ---------------------------------------------------------------- experiment 1: accuracy


def folds(n, count, rng):
    """Stratified folds: the same random split of the indices of each class."""
    return np.array_split(rng.permutation(n), count)


def cross_validate(x0, x1, truth, config, rng):
    """Choose (d, rho) for each robust method by validation error, as the baselines are chosen.

    Ties go to the smaller hinge risk, then the smaller d, then the larger radius.
    """
    radii, budgets = np.geomspace(*config["radii"]), config["budgets"]
    risk = defaultdict(lambda: np.zeros((len(radii), 2)))
    for held in folds(len(x0), config["folds"], rng):
        keep = np.setdiff1d(np.arange(len(x0)), held)
        tr0, tr1, va0, va1 = x0[keep], x1[keep], x0[held], x1[held]
        table = {d: robust_values(tr0, tr1, d, radii) for d in budgets}
        fitted = {}

        def validate(support, column):
            if (support, column) not in fitted:
                fit = robust_fit(tr0, tr1, support, radii[column])
                fitted[support, column] = validation(fit, va0, va1)
            return fitted[support, column]

        for column in range(len(radii)):
            for d in budgets:
                risk["select", d][column] += validate(joint(table, d, column), column)
                risk["marginal", d][column] += validate(marginal(table, d, column), column)
            risk["full", D][column] += validate(tuple(range(D)), column)
            if 0 < len(truth) < D:
                risk["oracle", len(truth)][column] += validate(truth, column)
    chosen = {}
    for (method, d), values in risk.items():
        for column, (error, hinge) in enumerate(values):
            key = (error, hinge, d, -column)
            if method not in chosen or key < chosen[method][0]:
                chosen[method] = (key, d, column)
    return {method: (d, column) for method, (_, d, column) in chosen.items()}


def recovery(support, truth):
    """False-discovery and non-discovery proportions of [WDX] Eq. (24)."""
    support, truth = set(support), set(truth)
    return {"fdp": len(support - truth) / max(len(support), 1),
            "ndp": len(truth - support) / len(truth)}


def baselines(x0, x1, tests, config):
    """[XGX] baselines (logistic regression, RBF SVM) and the sparse logistic of [WDX].

    Regularization by the same stratified cross-validation, over sklearn's default grid
    10^-4 .. 10^4 of LogisticRegressionCV.
    """
    x, y = np.vstack([x0, x1]), np.r_[np.zeros(len(x0)), np.ones(len(x1))]
    cv = StratifiedKFold(config["folds"], shuffle=True, random_state=0)
    grid = np.logspace(-4, 4, 10)
    models = {
        "logistic": LogisticRegressionCV(Cs=grid, cv=cv, l1_ratios=(0,), scoring="accuracy",
                                         max_iter=10000, use_legacy_attributes=False),
        "sparse_logistic": LogisticRegressionCV(Cs=grid, cv=cv, l1_ratios=(1,), solver="liblinear",
                                                scoring="accuracy", max_iter=10000,
                                                use_legacy_attributes=False),
        "svm_rbf": GridSearchCV(SVC(kernel="rbf", gamma="scale"), {"C": grid}, cv=cv),
    }
    records = {}
    for name, model in models.items():
        model.fit(x, y)
        # predict 1 = class 1; as a score, nonnegative decides class 0.
        records[name] = test_errors(-model.predict(tests[0]), -model.predict(tests[1]), split=True)
        if name == "sparse_logistic":
            records[name]["support"] = [int(m) for m in np.flatnonzero(model.coef_[0])]
    return records


def accuracy_task(config, case, n, trial):
    started = perf_counter()
    (x0, x1), tests, truth, bayes = draw(case, n, trial, config["test_size"])
    radii = np.geomspace(*config["radii"])
    chosen = cross_validate(x0, x1, truth, config, np.random.default_rng([trial, n]))
    table = {d: robust_values(x0, x1, d, radii) for d in config["budgets"]}
    methods = {}
    for method, (d, column) in chosen.items():
        if method == "select":
            support = joint(table, d, column)
        elif method == "marginal":
            support = marginal(table, d, column)
        else:
            support = truth if method == "oracle" else tuple(range(D))
        fit = robust_fit(x0, x1, support, radii[column])
        score0, score1 = fit.predict_score(tests[0]), fit.predict_score(tests[1])
        methods[method] = {
            "support": list(support), "radius": float(radii[column]), "radius_index": column,
            **test_errors(score0, score1, split=True),
            "randomized_error": (np.mean(1 - score0) + np.mean(1 + score1)) / 4,
            # Worst-case error of the randomized test over the two Wasserstein balls.
            "certificate": fit.upper / 4,
        }
    methods.update(baselines(x0, x1, tests, config))
    if 0 < len(truth) < D:
        for record in methods.values():
            if "support" in record:
                record.update(recovery(record["support"], truth))
    return {"case": case, "n": n, "trial": trial, "truth": list(truth), "bayes": bayes,
            "methods": methods,
            "seconds": perf_counter() - started}


# ---------------------------------------------------------------- experiment 2: computation


def compact_program(costs, rho, budget, relax=False, time_limit=600.0):
    """Eq. (4) of the directions note, variables [w (D), z (D), lambda, u (n0), v (n1)].

    Returns the value on the V scale (twice the program value), the support and the HiGHS
    result. With relax=True, z is continuous and the value is the commitment bound.
    """
    n0, n1, d = costs.shape
    pairs = costs.reshape(n0 * n1, d)
    objective = np.r_[np.zeros(2 * d), rho, np.full(n0, 1 / n0), np.full(n1, 1 / n1)]
    matrix = sparse.vstack([
        # u_i + v_j + sum_m w_m C^m_ij >= 1: each cross-class pair is paid for or priced apart.
        sparse.hstack([sparse.csr_matrix(pairs), sparse.csr_matrix((n0 * n1, d + 1)),
                       sparse.kron(sparse.eye(n0), np.ones((n1, 1))),
                       sparse.kron(np.ones((n0, 1)), sparse.eye(n1))]),
        # w_m <= lambda.
        sparse.hstack([sparse.eye(d), sparse.csr_matrix((d, d)), -np.ones((d, 1)),
                       sparse.csr_matrix((d, n0 + n1))]),
        # rho * w_m <= z_m: only selected coordinates carry a price.
        sparse.hstack([rho * sparse.eye(d), -sparse.eye(d), sparse.csr_matrix((d, 1 + n0 + n1))]),
        # sum_m w_m <= budget * lambda, and sum_m z_m = budget.
        sparse.csr_matrix(np.r_[np.ones(d), np.zeros(d), -budget, np.zeros(n0 + n1)][None, :]),
        sparse.csr_matrix(np.r_[np.zeros(d), np.ones(d), 0, np.zeros(n0 + n1)][None, :]),
    ]).tocsr()
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
    support = None if relax else tuple(int(m) for m in np.flatnonzero(np.rint(result.x[d:2 * d])))
    return 2 * result.fun, support, result


class BranchAndBound:
    """Exact min_{|S|=d} m_S(rho) with the commitment bound at every node (directions Sec. 3.5)."""

    def __init__(self, x0, x1, rho):
        n0, n1 = len(x0), len(x1)
        self.C = np.abs(x0[:, None, :] - x1[None, :, :]).reshape(n0 * n1, -1)
        self.rows = sparse.vstack([sparse.kron(sparse.eye(n0), np.ones((1, n1))),
                                   sparse.kron(np.ones((1, n0)), sparse.eye(n1))]).tocsr()
        self.b = np.r_[np.full(n0, 1 / n0), np.full(n1, 1 / n1)]
        self.rho, self.lps = rho, 0

    def bound(self, inside, free=(), r=0):
        """Largest matched mass over plans with kappa(inside) + (r largest kappa_m on free) <= rho,
        and the plan's coordinate costs kappa. Without free coordinates this is m_S(rho)."""
        P, F = self.C.shape[0], len(free)
        cost = self.C[:, list(inside)].sum(1)
        if F == 0:
            A, b = sparse.vstack([self.rows, sparse.csr_matrix(cost[None, :])]), np.r_[self.b, self.rho]
        else:
            # Variables [pi, t, s]: the r largest kappa_m = min r t + sum s, s_m >= kappa_m - t.
            top = sparse.hstack([sparse.csr_matrix(self.C[:, list(free)].T), -np.ones((F, 1)),
                                 -sparse.eye(F)])
            A = sparse.vstack([sparse.hstack([self.rows, sparse.csr_matrix((len(self.b), 1 + F))]),
                               sparse.csr_matrix(np.r_[cost, r, np.ones(F)][None, :]), top])
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
            outside = sorted(set(everything) - set(support), key=lambda m: -kappa[m])
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
        """Best-first search; returns V-scale optimum, its subset and the root bound."""
        upper, best = self.local(d)
        root, kappa = self.bound((), tuple(range(self.C.shape[1])), d)
        heap = [(root, (), tuple(range(self.C.shape[1])), tuple(kappa))]
        while heap and heap[0][0] < upper - tolerance:
            _, inside, free, kappa = heapq.heappop(heap)
            m = max(free, key=lambda j: kappa[j])  # the coordinate the adversary uses most
            rest = tuple(j for j in free if j != m)
            for child in (inside + (m,), inside):
                r = d - len(child)
                if r > len(rest):
                    continue
                if r in (0, len(rest)):  # a leaf: one subset
                    leaf = tuple(sorted(child + (rest if r else ())))
                    upper, best = min((upper, best), (self.bound(leaf)[0], leaf))
                    continue
                value, child_kappa = self.bound(child, rest, r)
                if value < upper - tolerance:
                    heapq.heappush(heap, (value, child, rest, tuple(child_kappa)))
        return 2 * upper, best, 2 * root


def computation_task(config, case, n, trial, budget, rho):
    (x0, x1), _, _, _ = draw(case, n, trial, config["test_size"])
    record = {"case": case, "n": n, "trial": trial, "budget": budget, "radius": rho}
    started = perf_counter()
    subsets, values = robust_values(x0, x1, budget, np.array([rho]))
    best = int(np.argmin(values[:, 0]))
    record["enumeration"] = {"value": float(values[best, 0]), "support": list(subsets[best]),
                             "subsets": len(subsets), "seconds": perf_counter() - started}
    costs = np.abs(x0[:, None, :] - x1[None, :, :])
    seconds = config["computation"]["time_limit"]
    record["commitment_bound"] = compact_program(costs, rho, budget, relax=True)[0]
    started = perf_counter()
    value, support, result = compact_program(costs, rho, budget, time_limit=seconds)
    record["compact_program"] = {
        "value": value, "support": None if support is None else list(support),
        "status": int(result.status), "nodes": getattr(result, "mip_node_count", None),
        "seconds": perf_counter() - started,
    }
    options = SolveOptions(time_limit=seconds, fit_time_limit=60, threads=1)
    search = select(Problem(x0, x1, [rho / 2, rho / 2]), budget, "oa", options)
    record["outer_approximation"] = {
        "upper": search.upper, "lower": search.lower, "support": list(search.support),
        "status": search.status, "subsets": search.fit_count, "seconds": search.runtime,
    }
    started = perf_counter()
    tree = BranchAndBound(x0, x1, rho)
    value, support, _ = tree.run(budget)
    record["branch_and_bound"] = {"value": value, "support": list(support), "lps": tree.lps,
                                  "seconds": perf_counter() - started}
    return record


# ---------------------------------------------------------------- driver


def environment():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
    return {"model_version": MODEL_VERSION, "git_commit": commit.strip(),
            "git_dirty": bool(dirty.strip()), "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), indent=1, sort_keys=True) + "\n")


def run_parallel(function, tasks, workers):
    os.environ["OMP_NUM_THREADS"] = "1"
    with Pool(min(workers, len(tasks))) as pool:
        return pool.starmap(function, tasks, chunksize=1)


def accuracy(config, output, workers):
    tasks = [(config, case, n, trial) for n in sorted(config["train_sizes"], reverse=True)
             for case in CASES for trial in range(config["trials"])]  # longest first
    started = perf_counter()
    records = run_parallel(accuracy_task, tasks, workers)
    save(Path(output) / "accuracy.json", {"environment": environment(), "config": config,
                                          "records": records, "seconds": perf_counter() - started})


def computation(config, output, workers):
    """Exact selection at the radius that cross-validation chose for `select` in experiment 1."""
    records = json.loads((Path(output) / "accuracy.json").read_text())["records"]
    radius = {(r["case"], r["n"], r["trial"]): r["methods"]["select"]["radius"] for r in records}
    setting = config["computation"]
    tasks = [(config, case, n, trial, budget, radius[case, n, trial])
             for case in setting["cases"] for n in setting["train_sizes"]
             for trial in range(setting["trials"]) for budget in setting["budgets"]]
    save(Path(output) / "computation.json",
         {"environment": environment(), "config": config,
          "records": run_parallel(computation_task, tasks, workers)})
