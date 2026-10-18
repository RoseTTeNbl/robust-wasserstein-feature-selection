"""Coordinate selection with one common robust fitting oracle.

Global search uses only the affine cuts from the current theory note. A
feasible fitted test supplies U; a binary master's global dual bound supplies
L. A master incumbent's objective is never a lower bound. Since an additional
observed coordinate cannot increase optimal risk, optimizing exactly d
coordinates has the same optimum as optimizing at most d coordinates.
"""

import warnings
from dataclasses import dataclass, field, replace
from itertools import combinations
from math import comb
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, milp

from .core import Cut, FitResult, Problem, SolveOptions, constant_fit, fit_fixed


@dataclass(frozen=True)
class SearchResult:
    fit: FitResult
    support: tuple[int, ...]
    upper: float
    lower: float | None
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


def _master(problem, budget, cuts, options, seconds):
    """Solve one unchanged master, retrying a solve error once without presolve.

    A HiGHS numerical solve error (status 4) can arise in presolve even for a
    feasible bounded model. Disabling presolve changes only the solver path.
    Both calls and matrix construction share the supplied wall-time allowance;
    infeasible/unbounded statuses never trigger this numerical-error retry.
    """
    started = perf_counter()
    d = problem.dimension
    rows = [np.r_[np.ones(d), 0.0]]
    rows.extend(np.r_[-cut.slope, 1.0] for cut in cuts)
    lows = np.r_[budget, [cut.intercept for cut in cuts]]
    highs = np.r_[budget, np.full(len(cuts), np.inf)]
    matrix = sparse.csr_matrix(np.vstack(rows))
    objective = np.r_[np.zeros(d), 1.0]
    arguments = {
        "integrality": np.r_[np.ones(d), 0],
        "bounds": Bounds(np.zeros(d + 1), np.ones(d + 1)),
        "constraints": LinearConstraint(matrix, lows, highs),
    }
    attempts = []
    result = None
    for presolve in (True, False):
        remaining = seconds - (perf_counter() - started)
        if remaining <= 0:
            result = OptimizeResult(
                status=1,
                message="Master deadline exhausted before the next solve",
                x=None,
                fun=None,
                mip_dual_bound=None,
            )
            break
        settings = {"time_limit": remaining, "mip_rel_gap": 0.0, "threads": options.threads}
        if not presolve:
            settings["presolve"] = False
        tick = perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Unrecognized options detected.*threads")
                result = milp(objective, **arguments, options=settings)
        except Exception as error:
            result = OptimizeResult(
                status=4, message=str(error), x=None, fun=None, mip_dual_bound=None
            )
        attempts.append(
            {
                "status": int(result.status),
                "message": str(getattr(result, "message", "")),
                "presolve": presolve,
                "seconds": perf_counter() - tick,
                "time_limit": remaining,
            }
        )
        if result.status not in (0, 1):
            result.x, result.fun, result.mip_dual_bound = None, None, None
        if result.status != 4 or not presolve:
            break
    result.master_attempts = attempts
    result.master_solver_calls = len(attempts)
    result.master_retry_count = max(0, len(attempts) - 1)
    return result


class _Run:
    """One method's deadline, cache and accounting; no cross-method free fits."""

    def __init__(self, problem, budget, method, options, ranked, started):
        self.problem, self.budget, self.method, self.options = problem, budget, method, options
        self.ranked, self.started = ranked, started
        self.cache, self.trace, self.cuts = {}, [], []
        self.fit_count = self.master_count = 0
        self.oracle_seconds = self.master_seconds = 0.0
        self.lower = 0.0
        self.fit = constant_fit(problem, tuple(sorted(ranked[:budget])))
        self.diagnostics = {
            "method": method,
            "budget": budget,
            "cardinality": "exact d, equivalent optimum to at most d by monotonicity",
            "time_limit": options.time_limit,
            "fit_time_limit": options.fit_time_limit,
            "cache_hits": 0,
            "unresolved_fits": [],
            "completed_passes": 0,
            "master_problem_count": 0,
            "master_retries": 0,
            "master_attempts": [],
        }

    def elapsed(self):
        return perf_counter() - self.started

    def remaining(self):
        return max(0.0, self.options.time_limit - self.elapsed())

    def event(self, phase, **fields):
        lower = self.lower if self.method in ("global", "enumeration") else None
        self.trace.append(
            {
                "phase": phase,
                "seconds": self.elapsed(),
                "upper": self.fit.upper,
                "lower": lower,
                **fields,
            }
        )

    def evaluate(self, support, phase, *, update_incumbent=False):
        support = self.problem.support(support)
        if support in self.cache:
            self.diagnostics["cache_hits"] += 1
            return self.cache[support]
        if self.remaining() <= 0:
            raise TimeoutError("Selection deadline reached before a fixed-subset fit")
        tick = perf_counter()
        # Both LPs, their construction and score recovery belong to this same
        # method's allowance. No post-timeout final fit is performed.
        options = replace(self.options, time_limit=self.remaining())
        fitted = fit_fixed(self.problem, support, options, with_cut=True)
        self.oracle_seconds += perf_counter() - tick
        self.fit_count += 1
        self.cache[support] = fitted
        if fitted.cut is not None:
            self.cuts.append(fitted.cut)
        if fitted.status != "optimal":
            self.diagnostics["unresolved_fits"].append(
                {
                    "support": list(support),
                    "status": fitted.status,
                    "lower": fitted.lower,
                    "upper": fitted.upper,
                }
            )
        if update_incumbent and fitted.upper < self.fit.upper:
            self.fit = fitted
        self.event(
            "fit",
            evaluation_phase=phase,
            support=list(support),
            fit_lower=fitted.lower,
            fit_upper=fitted.upper,
            fit_status=fitted.status,
            fit_seconds=fitted.runtime,
            cut_added=fitted.cut is not None,
        )
        return fitted

    def fill(self, selected, status):
        """Complete a partial choice from the fixed MMD ranking, then fit if time remains."""
        partial = tuple(selected)
        selected = set(selected)
        for j in self.ranked:
            if len(selected) == self.budget:
                break
            selected.add(j)
        support = tuple(sorted(selected))
        self.diagnostics["budget_filled_after_incomplete_search"] = True
        self.diagnostics["partial_support"] = list(partial)
        if support in self.cache:
            self.fit = self.cache[support]
        elif self.remaining() > 0:
            try:
                self.fit = self.evaluate(support, "incomplete_budget_fill")
            except TimeoutError:
                self.fit = constant_fit(self.problem, support, status)
        else:
            self.fit = constant_fit(self.problem, support, status)

    def finish(self, status, **fields):
        if self.method in ("forward", "forward_swap") and status in ("incomplete", "time_limit"):
            # An interrupted candidate pass cannot accept a greedy move, but
            # its already verified full-budget fits must still be retained.
            full_fits = [fit for support, fit in self.cache.items() if len(support) == self.budget]
            if full_fits:
                best = min(full_fits, key=lambda fit: (fit.upper, fit.support))
                if best.upper < self.fit.upper:
                    self.fit = best
                    self.diagnostics["returned_best_evaluated_full_support"] = True
        certified = self.method in ("global", "enumeration")
        lower = float(self.lower) if certified else None
        gap = max(0.0, self.fit.upper - lower) if certified else None
        self.diagnostics.update(fields)
        self.diagnostics["evaluated_supports"] = len(self.cache)
        self.event("finish", status=status)
        return SearchResult(
            self.fit,
            self.fit.support,
            float(self.fit.upper),
            lower,
            gap,
            self.elapsed(),
            status,
            self.trace,
            self.fit_count,
            self.master_count,
            self.master_seconds,
            self.oracle_seconds,
            tuple(self.cuts),
            dict(self.diagnostics),
        )


def _global(run, initial_support):
    problem, budget = run.problem, run.budget
    support = tuple(sorted(run.ranked[:budget])) if initial_support is None else initial_support
    run.fit = constant_fit(problem, support)
    try:
        first = run.evaluate(support, "initial", update_incumbent=True)
        if budget in (0, problem.dimension):
            run.lower = first.lower
            return run.finish(
                "optimal"
                if run.fit.upper - run.lower <= run.options.absolute_tolerance
                else first.status
            )
        if run.fit.upper - run.lower <= run.options.absolute_tolerance:
            return run.finish("optimal")
        if first.cut is None:
            return run.finish(first.status if first.status != "optimal" else "accuracy_limit")
        while run.remaining() > 0:
            tick = perf_counter()
            try:
                master = _master(problem, budget, run.cuts, run.options, run.remaining())
            except Exception as error:
                run.master_seconds += perf_counter() - tick
                run.diagnostics["master_problem_count"] += 1
                return run.finish("solver_error", master_error=str(error))
            run.master_seconds += perf_counter() - tick
            solver_calls = int(getattr(master, "master_solver_calls", 1))
            retries = int(getattr(master, "master_retry_count", 0))
            attempts = getattr(master, "master_attempts", [])
            run.master_count += solver_calls
            run.diagnostics["master_problem_count"] += 1
            run.diagnostics["master_retries"] += retries
            run.diagnostics["master_attempts"].append(
                {"problem": run.diagnostics["master_problem_count"], "attempts": attempts}
            )
            # A failed solver result cannot contribute any bound, even if its
            # stale fields happen to contain finite numbers.
            bound = getattr(master, "mip_dual_bound", None) if master.status in (0, 1) else None
            if bound is not None and np.isfinite(bound):
                margin = 1e-9 * max(1.0, abs(float(bound)))
                candidate_lower = max(run.lower, 0.0, float(bound) - margin)
                if candidate_lower > run.fit.upper:
                    return run.finish("numerical_failure", rejected_master_bound=candidate_lower)
                run.lower = candidate_lower
            run.event(
                "master",
                master_status=int(master.status),
                master_bound=None if bound is None else float(bound),
                master_incumbent=getattr(master, "fun", None),
                master_solver_calls=solver_calls,
                master_retries=retries,
                attempts=attempts,
            )
            if run.fit.upper - run.lower <= run.options.absolute_tolerance:
                return run.finish("optimal")
            if master.status not in (0, 1):
                return run.finish(
                    "solver_error", master_message=str(getattr(master, "message", ""))
                )
            if run.remaining() <= 0:
                return run.finish("time_limit")
            raw = getattr(master, "x", None)
            if raw is None:
                return run.finish("time_limit" if master.status == 1 else "solver_error")
            raw = np.asarray(raw, dtype=float)[: problem.dimension]
            z = np.rint(raw)
            if (
                raw.shape != (problem.dimension,)
                or not np.isfinite(raw).all()
                or np.max(abs(raw - z)) > 1e-5
                or not np.all((z == 0) | (z == 1))
                or z.sum() != budget
            ):
                return run.finish(
                    "numerical_failure", master_message="No feasible binary master incumbent"
                )
            support = tuple(int(j) for j in np.flatnonzero(z))
            if support in run.cache:
                return run.finish("accuracy_limit", repeated_support=list(support))
            fitted = run.evaluate(support, "global", update_incumbent=True)
            if run.fit.upper - run.lower <= run.options.absolute_tolerance:
                return run.finish("optimal")
            if fitted.cut is None:
                return run.finish(fitted.status if fitted.status != "optimal" else "accuracy_limit")
        return run.finish("time_limit")
    except TimeoutError:
        return run.finish("time_limit")


def _enumeration(run):
    total = comb(run.problem.dimension, run.budget)
    try:
        for support in combinations(range(run.problem.dimension), run.budget):
            run.evaluate(support, "enumeration", update_incumbent=True)
        run.lower = min(fitted.lower for fitted in run.cache.values())
        if run.diagnostics["unresolved_fits"]:
            status = "accuracy_limit"
        else:
            status = (
                "optimal"
                if run.fit.upper - run.lower <= run.options.absolute_tolerance
                else "accuracy_limit"
            )
        return run.finish(status, total_supports=total, enumeration_complete=True)
    except TimeoutError:
        return run.finish("time_limit", total_supports=total, enumeration_complete=False)


def _forward(run, initial_support, swap):
    selected = ()
    if swap and initial_support is not None:
        selected = initial_support
        run.fit = constant_fit(run.problem, selected)
        try:
            run.fit = run.evaluate(selected, "swap_initial")
        except TimeoutError:
            return run.finish("time_limit")
    try:
        while len(selected) < run.budget:
            candidates = []
            for j in range(run.problem.dimension):
                if j not in selected:
                    support = tuple(sorted((*selected, j)))
                    fitted = run.evaluate(support, "forward_candidate")
                    candidates.append((fitted.upper, support, fitted))
                    if run.remaining() <= 0:
                        raise TimeoutError("Forward candidate pass did not finish in time")
            _, selected, fitted = min(
                candidates, key=lambda candidate: (candidate[0], candidate[1])
            )
            run.diagnostics["completed_passes"] += 1
            if len(selected) == run.budget:
                run.fit = fitted
            run.event("forward_accept", support=list(selected), selected_upper=fitted.upper)
        if run.budget == 0:
            run.fit = run.evaluate((), "empty_budget")
        accepted_swaps = 0
        while (
            swap
            and accepted_swaps < run.options.swap_limit
            and 0 < run.budget < run.problem.dimension
        ):
            candidates = []
            for removed in selected:
                for added in range(run.problem.dimension):
                    if added not in selected:
                        support = tuple(sorted((*[j for j in selected if j != removed], added)))
                        fitted = run.evaluate(support, "swap_candidate")
                        candidates.append((fitted.upper, support, fitted))
                        if run.remaining() <= 0:
                            raise TimeoutError("Swap candidate pass did not finish in time")
            run.diagnostics["completed_passes"] += 1
            _, candidate_support, candidate_fit = min(
                candidates, key=lambda candidate: (candidate[0], candidate[1])
            )
            if run.fit.upper - candidate_fit.upper < run.options.swap_improvement:
                break
            previous = selected
            selected, run.fit = candidate_support, candidate_fit
            accepted_swaps += 1
            run.event(
                "swap_accept",
                previous_support=list(previous),
                support=list(selected),
                accepted_swaps=accepted_swaps,
            )
        run.diagnostics["accepted_swaps"] = accepted_swaps
        return run.finish("accuracy_limit" if run.diagnostics["unresolved_fits"] else "complete")
    except TimeoutError:
        if len(selected) < run.budget:
            run.fill(selected, "incomplete")
            return run.finish("incomplete")
        return run.finish("time_limit")


def select(
    problem: Problem,
    budget,
    method="global",
    options=None,
    initial_support=None,
    ranked_features=None,
):
    """Run one of the report's four outer-search controls.

    ``global`` and ``enumeration`` return a certified numerical interval.
    ``forward`` and ``forward_swap`` return a feasible test and no global lower
    bound; status ``complete`` means the declared heuristic finished. Every
    candidate pass completes before an addition or swap is accepted. A timeout
    during an unfinished forward pass fills the remaining budget using the
    supplied marginal-MMD ranking and records status ``incomplete``.
    """
    started = perf_counter()
    options = SolveOptions() if options is None else options
    if (
        isinstance(budget, (bool, np.bool_))
        or not isinstance(budget, (int, np.integer))
        or not 0 <= budget <= problem.dimension
    ):
        raise ValueError("budget must be an integer between zero and the coordinate dimension")
    aliases = {
        "enumerate": "enumeration",
        "exhaustive": "enumeration",
        "greedy": "forward",
        "greedy_swap": "forward_swap",
        "forward+swap": "forward_swap",
    }
    method = aliases.get(method, method)
    if method not in ("global", "enumeration", "forward", "forward_swap"):
        raise ValueError("method must be global, enumeration, forward, or forward_swap")
    ranked = tuple(range(problem.dimension)) if ranked_features is None else tuple(ranked_features)
    if len(ranked) != problem.dimension or set(problem.support(ranked)) != set(
        range(problem.dimension)
    ):
        raise ValueError("ranked_features must be a permutation of all coordinate indices")
    ranked = tuple(int(j) for j in ranked)
    if initial_support is not None:
        initial_support = problem.support(initial_support)
        if method == "forward":
            if initial_support:
                raise ValueError("forward selection starts from the empty set")
        elif len(initial_support) != budget:
            raise ValueError("initial_support must contain exactly budget coordinates")
    run = _Run(problem, int(budget), method, options, ranked, started)
    if method == "global":
        return _global(run, initial_support)
    if method == "enumeration":
        return _enumeration(run)
    return _forward(run, initial_support, swap=method == "forward_swap")
