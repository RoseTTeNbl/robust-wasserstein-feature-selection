"""Global search: full convex oracles, valid tangents, and a binary master.

A feasible test supplies UB. Only a globally valid master bound supplies LB.
Full-budget supports suffice because observing an extra coordinate cannot
increase the optimal robust risk. Timeouts retain their numerical bounds.
"""
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .oracle import GateOracle
from .problem import Problem, json_value


@dataclass
class SelectionResult:
    support: tuple
    lower: float
    upper: float
    status: str
    seconds: float
    oracle_calls: int
    visited: int
    trace: list = field(default_factory=list)
    test: object = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def gap(self):
        return max(0., self.upper - self.lower)

    def to_dict(self):
        return json_value(dict(support=list(self.support), lower=self.lower, upper=self.upper,
                    gap=self.gap, status=self.status, seconds=self.seconds,
                    oracle_calls=self.oracle_calls, visited=self.visited,
                    trace=self.trace, diagnostics=self.diagnostics))


class _ConstantTest:
    def __init__(self, support, value):
        self.support, self.primal_objective = tuple(support), float(value)

    def prob_class0(self, x):
        return np.full(len(np.atleast_2d(x)), .5)

    def predict(self, x):
        return np.zeros(len(np.atleast_2d(x)), dtype=int)


def _master(p, cuts, remaining, integer):
    # SciPy forwards this valid HiGHS option but warns because its public
    # options allowlist omits it. Keep single-thread control without log spam.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*threads")
        return _master_impl(p, cuts, remaining, integer)


def _master_impl(p, cuts, remaining, integer):
    """Solve the current outer approximation. Cuts are (intercept, gradient)."""
    started = perf_counter()
    rows = np.array([np.r_[gradient, -1.] for _, gradient in cuts])
    rhs = np.array([-intercept for intercept, _ in cuts])
    objective = np.r_[np.zeros(p.p), 1.]
    if integer:
        matrix = np.vstack([np.r_[np.ones(p.p), 0.], -rows])
        return _milp_with_retry(objective, integrality=np.r_[np.ones(p.p), 0],
                    bounds=Bounds(np.zeros(p.p + 1), np.r_[np.ones(p.p), p.B]),
                    constraints=LinearConstraint(matrix, np.r_[p.budget, -rhs],
                                                 np.r_[p.budget, np.full(len(cuts), np.inf)]),
                    remaining=remaining, started=started)
    return linprog(objective, A_ub=rows, b_ub=rhs,
                   A_eq=[np.r_[np.ones(p.p), 0.]], b_eq=[p.budget],
                   bounds=[(0, 1)] * p.p + [(0, p.B)], method="highs",
                   options={"time_limit": max(remaining, 1e-5), "threads": 1})


def _milp_with_retry(objective, *, integrality, bounds, constraints, remaining, started):
    """Retry only HiGHS solve errors, with the identical model and no presolve.

    An error result is never a source of incumbents or lower bounds. The
    second call shares the first call's wall-time allowance and is attempted
    at most once. Successful and time-limited first calls are unchanged.
    """
    arguments = dict(integrality=integrality, bounds=bounds, constraints=constraints)
    options = dict(time_limit=max(remaining - (perf_counter() - started), 1e-5),
                   mip_rel_gap=0., threads=1)
    tick = perf_counter()
    first = milp(objective, **arguments, options=options)
    first_seconds = perf_counter() - tick
    if first.status != 4:
        # Other failure statuses also do not certify a useful master bound.
        if first.status not in (0, 1):
            first.x, first.fun, first.mip_dual_bound = None, None, None
        return first
    failure = dict(status=int(first.status), message=str(first.message),
                   seconds=first_seconds, presolve=True)
    left = remaining - (perf_counter() - started)
    if left <= 0:
        first.status, first.success = 1, False
        first.message = "Wall-time budget exhausted after HiGHS solve error; no retry attempted"
        first.x, first.fun, first.mip_dual_bound = None, None, None
        first.master_retries = 0
        first.master_retry_log = [dict(initial=failure, retried=False,
                                       reason="wall-time budget exhausted")]
        return first
    tick = perf_counter()
    retry = milp(objective, **arguments, options={**options, "presolve": False,
                                                 "time_limit": left})
    retry_seconds = perf_counter() - tick
    retry.master_retries = 1
    retry.master_retry_log = [dict(initial=failure, retried=True,
                                   retry=dict(status=int(retry.status),
                                              message=str(retry.message),
                                              seconds=retry_seconds, presolve=False))]
    if retry.status not in (0, 1):
        retry.x, retry.fun, retry.mip_dual_bound = None, None, None
    return retry


def _ratio_master(p, cuts, ratios, remaining, upper):
    """Retain complete fractional certificates in an exact binary MILP.

    Shared y_j=eta*z_j needs only D extra variables for every certificate.
    The incumbent is a feasible upper bound, hence a valid product bound.
    """
    started = perf_counter()
    d, count = p.p, 2 * p.p + 1
    rows, lows, highs = [], [], []

    def row(z=None, eta=0., y=None, low=-np.inf, high=np.inf):
        vector = np.zeros(count)
        if z is not None:
            vector[:d] = z
        vector[d] = eta
        if y is not None:
            vector[d+1:] = y
        rows.append(vector)
        lows.append(low)
        highs.append(high)

    row(z=np.ones(d), low=p.budget, high=p.budget)
    # Cardinality multiplied by eta: valid at every binary solution and useful
    # for tightening the continuous product relaxation.
    row(eta=-p.budget, y=np.ones(d), low=0., high=0.)
    for j in range(d):
        unit = np.eye(1, d, j)[0]
        row(z=-upper * unit, y=unit, high=0.)
        row(eta=-1., y=unit, high=0.)
        row(z=upper * unit, eta=1., y=-unit, high=upper)
    for intercept, gradient in cuts:
        row(z=gradient, eta=-1., high=-intercept)
    for value, tau, mu in ratios:
        row(eta=-tau, y=-mu, high=-value)
        if upper > 0:
            row(z=-mu, high=tau-value/upper)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*threads")
        return _milp_with_retry(np.r_[np.zeros(d), 1., np.zeros(d)],
                    integrality=np.r_[np.ones(d), np.zeros(d+1)],
                    bounds=Bounds(np.zeros(count), np.r_[np.ones(d), upper,
                                                        np.full(d, upper)]),
                    constraints=LinearConstraint(np.asarray(rows), lows, highs),
                    remaining=remaining, started=started)


def _solve_master(problem: Problem, *, gate="risk", time_limit=30., root_rounds=5,
          core_point=False, warm_support=None, tolerance=1e-6,
          relative_tolerance=1e-5, max_iterations=10000, cut_mode="tangent"):
    """Return a support, feasible test, and certified numerical interval.

    All stages share one wall-time budget. A solver may spend a little extra
    time starting/finalizing a call. Repeated supports with an unresolved gap
    are an accuracy limit, never a claim of optimality.
    """
    if time_limit <= 0 or tolerance <= 0 or relative_tolerance < 0:
        raise ValueError("time_limit/tolerance must be positive")
    if root_rounds < 0 or max_iterations < 1:
        raise ValueError("invalid iteration count")
    if cut_mode not in ("tangent", "ratio") or (cut_mode == "ratio" and gate != "risk"):
        raise ValueError("ratio cuts require the risk gate")
    p = problem
    oracle = GateOracle(p, gate=gate, core_point=core_point)
    started = perf_counter()
    cache, cuts, ratios, trace = {}, [], [], []
    lower, upper, calls = 0., float(p.B), 0
    support = tuple(range(p.budget)) if warm_support is None else tuple(sorted(warm_support))
    if len(support) != p.budget or len(set(support)) != len(support) or any(
            j < 0 or j >= p.p for j in support):
        raise ValueError("warm_support must contain budget distinct valid coordinates")
    best_support, best_test = support, _ConstantTest(support, p.B)
    oracle_seconds, master_seconds, root_bound = 0., 0., 0.
    master_retries, master_retry_log = 0, []

    def remaining():
        return max(0., time_limit - (perf_counter() - started))

    def done():
        return upper - lower <= tolerance + relative_tolerance * abs(upper)

    def finish(status, **detail):
        return SelectionResult(best_support, float(lower), float(upper), status,
                               perf_counter() - started, calls, len(cache), trace, best_test,
                               dict(gate=gate, cut_mode=cut_mode, root_rounds=root_rounds,
                                    core_point=core_point,
                                    oracle_seconds=oracle_seconds, master_seconds=master_seconds,
                                    root_bound=root_bound, master_retries=master_retries,
                                    master_retry_log=master_retry_log, **detail))

    def evaluate(z, phase):
        nonlocal calls, upper, best_support, best_test, oracle_seconds
        z = np.asarray(z, dtype=float)
        binary = bool(np.all((z == 0) | (z == 1)))
        key = tuple(int(j) for j in np.flatnonzero(z)) if binary else None
        if key is not None and key in cache:
            return cache[key]
        if remaining() <= 0:
            raise TimeoutError("wall-time budget exhausted")
        calls += 1
        tick = perf_counter()
        result = oracle.solve(z, time_limit=remaining())
        oracle_seconds += perf_counter() - tick
        gradient = np.asarray(result.gradient, dtype=float)
        if gradient.shape != (p.p,) or not np.all(np.isfinite(gradient)):
            raise ArithmeticError("oracle returned an invalid gradient")
        if not np.isfinite(result.value) or not np.isfinite(result.upper):
            raise ArithmeticError("oracle returned nonfinite bounds")
        if result.value > result.upper + 1e-7 * max(1., p.B):
            raise ArithmeticError("oracle lower certificate exceeds feasible upper bound")
        cuts.append((float(result.intercept), gradient.copy()))
        if cut_mode == "ratio":
            cert = result.certificate
            tau = float(cert.get("tau", 0.))
            value = max(0., float(cert.get("entropy", 0.)) - float(cert.get("margin", 0.)))
            if tau > 0 and value > 0:
                ratios.append((value, tau, np.asarray(cert["mu"], float).copy()))
        if binary:
            cache[key] = result
            if result.test is not None and result.upper <= upper:
                upper, best_support, best_test = float(result.upper), key, result.test
        trace.append(dict(phase=phase, seconds=perf_counter() - started,
                          support=list(key) if binary else None,
                          point=z.tolist(), oracle_lower=float(result.value),
                          oracle_upper=float(result.upper), lower=float(lower), upper=float(upper)))
        return result

    try:
        z = np.zeros(p.p)
        z[list(support)] = 1
        initial = evaluate(z, "initial")
        if p.budget in (0, p.p):
            lower = max(lower, float(initial.value) - 1e-9)
            return finish("optimal" if done() else "oracle_accuracy_limit")
        # Begin at an interior feasible point; every root tangent remains valid
        # for the later binary master. Root LP optima also give global lower bounds.
        if root_rounds:
            evaluate(np.full(p.p, p.budget / p.p), "root")
        for _ in range(max(0, root_rounds - 1)):
            tick = perf_counter()
            master = _master(p, cuts, remaining(), integer=False)
            master_seconds += perf_counter() - tick
            if master.status != 0 or master.x is None:
                break
            root_bound = max(root_bound, float(master.fun) - 1e-9)
            lower = max(lower, root_bound)
            if done():
                return finish("optimal")
            evaluate(np.clip(master.x[:p.p], 0, 1), "root")
        for _ in range(max_iterations):
            if remaining() <= 0:
                return finish("time_limit")
            tick = perf_counter()
            master = (_ratio_master(p, cuts, ratios, remaining(), upper)
                      if cut_mode == "ratio" else _master(p, cuts, remaining(), integer=True))
            master_seconds += perf_counter() - tick
            retries = int(getattr(master, "master_retries", 0))
            master_retries += retries
            retry_log = getattr(master, "master_retry_log", [])
            master_retry_log.extend(dict(iteration=len([x for x in trace if x["phase"] == "master"]),
                                         **entry) for entry in retry_log)
            bound = getattr(master, "mip_dual_bound", None) if master.status in (0, 1) else None
            if bound is None and master.status == 0:
                bound = master.fun
            if bound is not None and np.isfinite(bound):
                lower = max(lower, float(bound) - 1e-9 * max(1., abs(bound), p.B))
            trace.append(dict(phase="master", seconds=perf_counter() - started,
                              lower=float(lower), upper=float(upper), status=int(master.status),
                              master_retries=retries, retry_log=retry_log))
            if lower > upper + 1e-7 * max(1., p.B):
                return finish("numerical_inconsistency")
            if done():
                return finish("optimal")
            if master.status not in (0, 1) or master.x is None:
                return finish("time_limit" if master.status == 1 else "master_failure",
                              message=str(master.message))
            if remaining() <= 0:
                return finish("time_limit")
            raw = master.x[:p.p]
            z = np.rint(raw)
            if np.max(abs(raw - z)) > 1e-5 or z.sum() != p.budget:
                return finish("master_failure", message="no feasible binary incumbent")
            key = tuple(int(j) for j in np.flatnonzero(z))
            if key in cache:
                return finish("oracle_accuracy_limit" if cache[key].upper-cache[key].value >
                              tolerance else "master_accuracy_limit", repeated_support=list(key))
            evaluate(z, "integer")
        return finish("iteration_limit")
    except TimeoutError as error:
        return finish("time_limit", message=str(error))
    except (ArithmeticError, RuntimeError) as error:
        return finish("oracle_failure", message=str(error))


def solve(problem: Problem, *, gate="risk", time_limit=30., root_rounds=5,
          core_point=False, warm_support=None, tolerance=1e-6,
          relative_tolerance=1e-5, max_iterations=10000, cut_mode="tangent",
          search="adaptive", proposal_count=8,
          certify_every=3, max_refinements=2, random_state=0):
    """Full binary inner solves with uncertainty-guided selection and owned cuts."""
    if search not in ("adaptive", "master"):
        raise ValueError("search must be 'adaptive' or 'master'")
    if time_limit <= 0 or tolerance <= 0 or relative_tolerance < 0:
        raise ValueError("time_limit/tolerance must be positive")
    if root_rounds < 0 or max_iterations < 1:
        raise ValueError("invalid iteration count")
    if cut_mode not in ("tangent", "ratio") or (cut_mode == "ratio" and gate != "risk"):
        raise ValueError("ratio cuts require the risk gate")
    common = dict(gate=gate, time_limit=time_limit, root_rounds=root_rounds,
                  core_point=core_point, warm_support=warm_support, tolerance=tolerance,
                  relative_tolerance=relative_tolerance, max_iterations=max_iterations,
                  cut_mode=cut_mode)
    # Alternative gates, ratio masters and mock oracles retain their original path.
    if (search == "master" or gate != "risk" or cut_mode == "ratio"
            or not isinstance(problem, Problem)):
        return _solve_master(problem, **common)
    from .adaptive import solve_adaptive
    return solve_adaptive(problem, oracle_class=GateOracle, **common,
                          proposal_count=proposal_count,
                          certify_every=certify_every, max_refinements=max_refinements,
                          random_state=random_state)


def enumerate_supports(problem, *, time_limit=300., tolerance=1e-6):
    """Reference enumeration; not the proposed selection algorithm."""
    oracle, started = GateOracle(problem), perf_counter()
    total = comb(problem.p, problem.budget)
    values, trace = [], []
    upper = problem.B
    support = tuple(range(problem.budget))
    test = _ConstantTest(support, problem.B)
    for candidate in combinations(range(problem.p), problem.budget):
        left = time_limit - (perf_counter() - started)
        if left <= 0:
            break
        z = np.zeros(problem.p)
        z[list(candidate)] = 1
        try:
            r = oracle.solve(z, time_limit=left)
        except TimeoutError:
            break
        values.append(r.value)
        trace.append(dict(support=list(candidate), lower=r.value, upper=r.upper))
        if r.upper <= upper:
            upper, support, test = r.upper, candidate, r.test
    complete = len(values) == total
    lower = max(0., min(values) - 1e-9) if complete else 0.
    status = "optimal" if complete and upper-lower <= tolerance else (
        "oracle_accuracy_limit" if complete else "time_limit")
    return SelectionResult(support, lower, upper, status, perf_counter()-started,
                           len(values), len(values), trace, test,
                           dict(total_supports=total, completed_supports=len(values)))
