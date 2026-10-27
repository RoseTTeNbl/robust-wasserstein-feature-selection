"""Coordinate selection with one common robust fitting oracle."""

import warnings
from dataclasses import dataclass, field, replace
from itertools import combinations
from math import comb
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, milp

from .model import MODEL_VERSION, ZERO_SCORE_RISK, Cut, FitResult, Problem, SolveOptions
from .oracle import constant_fit, fit_fixed


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
    # Eq. (8), variable order [z_1,...,z_D,eta]; local d=D, budget=|S|.
    # sum_m z_m=budget; exact and at-most budgets share the optimum by monotonicity.
    rows = [np.r_[np.ones(d), 0.0]]
    # Retained cuts: [-slope_s,1] @ [z,eta] >= intercept_s, i.e. eta>=c_s(z).
    rows.extend(np.r_[-cut.slope, 1.0] for cut in cuts)
    lows = np.r_[budget, [cut.intercept for cut in cuts]]
    highs = np.r_[budget, np.full(len(cuts), np.inf)]
    matrix = sparse.csr_matrix(np.vstack(rows))
    # min eta, binary z, continuous eta>=0; no artificial upper constraint.
    objective = np.r_[np.zeros(d), 1.0]
    arguments = {
        "integrality": np.r_[np.ones(d), 0],
        "bounds": Bounds(np.zeros(d + 1), np.r_[np.ones(d), np.inf]),
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
        # Initial bounds: L_0=0, U_0=R(g=0)=2.
        self.lower = 0.0
        self.fit = constant_fit(problem, tuple(sorted(ranked[:budget])))
        self.diagnostics = {
            "method": method,
            "model_version": MODEL_VERSION,
            "zero_score_risk": ZERO_SCORE_RISK,
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
        # Eqs. (4)--(6): lower_s <= F(1_S)=V(S) <= upper_s, plus c_s(z)<=F(z).
        # The pointwise lower_s is not a global search bound. Shared remaining deadline.
        options = replace(self.options, time_limit=self.remaining())
        fitted = fit_fixed(self.problem, support, options, with_cut=True)
        self.oracle_seconds += perf_counter() - tick
        self.fit_count += 1
        self.cache[support] = fitted
        if fitted.cut is not None:
            # Retain every dual-feasible cut, including those from inexact inner solves.
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
        # U_t=min_{s<=t} upper_s; retain the corresponding selection and detector.
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
        """Complete a partial choice from the fixed training-only ranking, then fit if time remains."""
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
        # Global methods provide L<=OPT; heuristics return no global lower bound or gap.
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
        # Initial evaluation: feasible upper bound U_1 and cut c_1(z)<=F(z).
        first = run.evaluate(support, "initial", update_incumbent=True)
        if budget in (0, problem.dimension):
            # A singleton selection domain makes the inner lower bound global.
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
                # Master: min_z m_t(z), m_t(z)=max(0,c_1(z),...,c_t(z))<=F(z).
                # Its solution proposes z^{t+1}; the oracle evaluates the true risk.
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
            # mip_dual_bound <= min_z m_t(z) <= OPT; fun is only a master incumbent.
            # Bounds are accepted only from optimal or time-limited solver states.
            bound = getattr(master, "mip_dual_bound", None) if master.status in (0, 1) else None
            if bound is not None and np.isfinite(bound):
                # Monotone global bound: L_t=max(L_{t-1},0,bound-margin).
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
            # U_t-L_t<=epsilon certifies the incumbent's global gap.
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
            # Validate integrality and cardinality before extracting S={m:z_m=1}.
            # Rounding removes solver tolerances, not a fractional relaxation.
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
                # A repeated point with an open gap indicates limited numerical accuracy.
                return run.finish("accuracy_limit", repeated_support=list(support))
            # Evaluate z^{t+1}, append its cut, and update U before the next master.
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
        # Complete enumeration: min_S lower_S <= OPT <= min_S upper_S.
        # This lower bound requires every feasible selection to have been evaluated.
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
            # Forward step: argmin_{j not in S} upper(S union {j}), using the same oracle.
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
            # Best improving swap over (i,j) in S x S^c: (S minus {i}) union {j}.
            # A complete neighborhood pass is required; no global certificate.
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
    supplied training-only ranking and records status ``incomplete``.
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
        "oa": "global",
        "outer_approximation": "global",
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
