"""Algorithm 1 (outer approximation) and three baseline searches over the same oracle.

Equation numbers refer to note/main.tex. Every evaluated selection S gives an
upper bound V(S) and the ratio cut Eq. (8), 1/F(z) <= c(z). The master Eq. (9)
maximizes theta <= c_s(z) over |S| = d; L = 1/theta is a lower bound on OPT.
"""

from dataclasses import dataclass, field, replace
from itertools import combinations
from math import comb
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .model import MODEL_VERSION, Cut, FitResult, Problem, SolveOptions
from .oracle import constant_fit, fit_fixed, highs


@dataclass(frozen=True)
class SearchResult:
    fit: FitResult
    support: tuple[int, ...]
    upper: float
    lower: float | None  # only outer approximation and enumeration certify a lower bound
    gap: float | None
    runtime: float
    status: str
    trace: list[dict]
    fit_count: int
    master_count: int
    master_seconds: float
    oracle_seconds: float
    cuts: tuple[Cut, ...]
    diagnostics: dict = field(default_factory=dict)


def master(cuts, dimension, budget, options, seconds):
    """Eq. (9): max theta s.t. theta <= intercept_s + slope_s @ z, 1^T z = d, z binary.

    Returns (z, theta_bound); theta_bound is HiGHS's global upper bound on theta.
    1^T z = d suffices because F is nonincreasing (Section 3).
    """
    rows = [np.r_[np.ones(dimension), 0.0]] + [np.r_[-cut.slope, 1.0] for cut in cuts]
    lows = np.r_[budget, np.full(len(cuts), -np.inf)]
    highs_ = np.r_[budget, [cut.intercept for cut in cuts]]
    result = highs(
        milp,
        np.r_[np.zeros(dimension), -1.0],  # maximize theta
        integrality=np.r_[np.ones(dimension), 0],
        bounds=Bounds(np.zeros(dimension + 1), np.r_[np.ones(dimension), np.inf]),
        constraints=LinearConstraint(np.array(rows), lows, highs_),
        threads=options.threads,
        options={"time_limit": max(seconds, 1e-3), "mip_rel_gap": 0.0},
    )
    if result.status not in (0, 1) or result.x is None:
        return None, None
    bound = getattr(result, "mip_dual_bound", None)
    return np.rint(result.x[:dimension]), (None if bound is None else -float(bound))


class _Search:
    """Shared clock, cache of fitted selections, cuts, incumbent and trace."""

    def __init__(self, problem, budget, method, options, ranked):
        self.problem, self.budget, self.method, self.options = problem, budget, method, options
        self.ranked, self.started = ranked, perf_counter()
        self.cache, self.cuts, self.trace = {}, [], []
        self.lower = 0.0
        self.master_count, self.master_seconds, self.oracle_seconds = 0, 0.0, 0.0
        self.fit = constant_fit(problem, ranked[:budget])

    def remaining(self):
        return self.options.time_limit - (perf_counter() - self.started)

    def evaluate(self, support, keep_best=True):
        """Lines 3-8 of Algorithm 1: fit S once, store its cut, update the incumbent U."""
        support = self.problem.support(support)
        if support not in self.cache:
            if self.remaining() <= 0:
                raise TimeoutError
            tick = perf_counter()
            fitted = fit_fixed(
                self.problem, support, replace(self.options, time_limit=self.remaining())
            )
            self.oracle_seconds += perf_counter() - tick
            self.cache[support] = fitted
            if fitted.cut is not None:
                self.cuts.append(fitted.cut)
            self.log("fit", support=list(support), fit_upper=fitted.upper, fit_status=fitted.status)
        fitted = self.cache[support]
        if keep_best and fitted.upper < self.fit.upper:
            self.fit = fitted
        return fitted

    def log(self, phase, **fields):
        lower = self.lower if self.method in ("global", "enumeration") else None
        self.trace.append(
            {"phase": phase, "seconds": perf_counter() - self.started,
             "upper": self.fit.upper, "lower": lower, **fields}
        )  # fmt: skip

    def finish(self, status, **diagnostics):
        certified = self.method in ("global", "enumeration")
        self.log("finish", status=status)
        unresolved = [list(s) for s, f in self.cache.items() if f.status != "optimal"]
        return SearchResult(
            self.fit, self.fit.support, self.fit.upper,
            self.lower if certified else None,
            max(0.0, self.fit.upper - self.lower) if certified else None,
            perf_counter() - self.started, status, self.trace, len(self.cache),
            self.master_count, self.master_seconds, self.oracle_seconds, tuple(self.cuts),
            {"method": self.method, "model_version": MODEL_VERSION, "budget": self.budget,
             "evaluated_supports": len(self.cache), "unresolved_fits": unresolved,
             **diagnostics},
        )  # fmt: skip


def _outer_approximation(run, initial):
    """Algorithm 1 with ratio cuts."""
    problem, budget, tolerance = run.problem, run.budget, run.options.absolute_tolerance
    support = initial if initial is not None else run.ranked[:budget]
    try:
        first = run.evaluate(support)
        if budget in (0, problem.dimension):  # a single feasible selection
            run.lower = first.lower
            return run.finish("optimal" if first.gap <= tolerance else first.status)
        while run.fit.upper - run.lower > tolerance:
            if run.remaining() <= 0:
                return run.finish("time_limit")
            tick = perf_counter()
            z, theta = master(run.cuts, problem.dimension, budget, run.options, run.remaining())
            run.master_count += 1
            run.master_seconds += perf_counter() - tick
            if theta is not None and theta > 0:
                run.lower = max(run.lower, 1.0 / theta)  # line 9: L_t = 1 / theta_t
            run.log("master", master_bound=theta)
            if z is None:
                return run.finish("solver_error")
            if run.fit.upper - run.lower <= tolerance:
                break
            support = tuple(int(m) for m in np.flatnonzero(z))
            if support in run.cache:
                # Exact solves would give L >= U here (Section 3): numerical accuracy limit.
                return run.finish("accuracy_limit", repeated_support=list(support))
            run.evaluate(support)
        return run.finish("optimal")
    except TimeoutError:
        return run.finish("time_limit")


def _enumeration(run):
    total = comb(run.problem.dimension, run.budget)
    try:
        for support in combinations(range(run.problem.dimension), run.budget):
            run.evaluate(support)
    except TimeoutError:
        return run.finish("time_limit", total_supports=total, enumeration_complete=False)
    # Every selection is evaluated: min_S lower_S <= OPT <= min_S upper_S.
    run.lower = min(fitted.lower for fitted in run.cache.values())
    status = "optimal" if run.fit.upper - run.lower <= run.options.absolute_tolerance else (
        "accuracy_limit")  # fmt: skip
    return run.finish(status, total_supports=total, enumeration_complete=True)


def _best_neighbor(run, candidates):
    """Evaluate a whole candidate pass, then return the best (upper, support)."""
    fits = [run.evaluate(s, keep_best=False) for s in candidates]
    return min((f.upper, f.support) for f in fits)


def _forward(run, initial, swap):
    """Forward selection (add the best coordinate) and optional best-swap improvement."""
    problem, budget = run.problem, run.budget
    selected = tuple(initial) if swap and initial is not None else ()
    swaps = 0
    try:
        while len(selected) < budget:
            free = [j for j in range(problem.dimension) if j not in selected]
            _, selected = _best_neighbor(run, [(*selected, j) for j in free])
            run.log("forward_accept", support=list(selected))
        run.fit = run.evaluate(selected, keep_best=False)
        while swap and swaps < run.options.swap_limit and 0 < budget < problem.dimension:
            free = [j for j in range(problem.dimension) if j not in selected]
            neighbors = [(*[s for s in selected if s != out], new) for out in selected
                         for new in free]  # fmt: skip
            upper, best = _best_neighbor(run, neighbors)
            if run.fit.upper - upper < run.options.swap_improvement:
                break
            selected, run.fit, swaps = best, run.cache[best], swaps + 1
            run.log("swap_accept", support=list(selected), accepted_swaps=swaps)
        return run.finish("complete", accepted_swaps=swaps)
    except TimeoutError:
        # Keep the best full-size selection already evaluated, else fill from the ranking.
        full = [f for s, f in run.cache.items() if len(s) == budget]
        if full:
            run.fit = min(full, key=lambda f: (f.upper, f.support))
        else:
            filled = list(selected) + [j for j in run.ranked if j not in selected]
            run.fit = constant_fit(problem, filled[:budget], "incomplete")
        return run.finish("incomplete" if len(selected) < budget else "time_limit",
                          accepted_swaps=swaps)  # fmt: skip


def select(problem: Problem, budget, method="global", options=None, initial_support=None,
           ranked_features=None):  # fmt: skip
    """Choose `budget` coordinates.

    global (Algorithm 1) and enumeration return a certified interval lower <= OPT <= upper;
    forward and forward_swap are heuristics without a lower bound.
    """
    method = {"oa": "global", "outer_approximation": "global", "exhaustive": "enumeration",
              "greedy": "forward", "greedy_swap": "forward_swap"}.get(method, method)  # fmt: skip
    if method not in ("global", "enumeration", "forward", "forward_swap"):
        raise ValueError("method must be global, enumeration, forward or forward_swap")
    if not 0 <= int(budget) <= problem.dimension or isinstance(budget, bool):
        raise ValueError("budget must be an integer between 0 and the dimension")
    ranked = tuple(range(problem.dimension)) if ranked_features is None else tuple(ranked_features)
    if sorted(ranked) != list(range(problem.dimension)):
        raise ValueError("ranked_features must be a permutation of the coordinates")
    initial = None if initial_support is None else problem.support(initial_support)
    run = _Search(problem, int(budget), method, options or SolveOptions(), ranked)
    if method == "global":
        return _outer_approximation(run, initial)
    if method == "enumeration":
        return _enumeration(run)
    return _forward(run, initial, swap=method == "forward_swap")
