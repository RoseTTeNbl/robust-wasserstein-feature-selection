"""Global binary masters followed by complete primal and dual evaluations."""
from time import perf_counter

import numpy as np

from .history import EvaluationHistory


def solve_cuts(p, *, oracle_class, time_limit, warm_support, tolerance,
               relative_tolerance, max_iterations, max_refinements, use_fractions):
    from .solver import SelectionResult, _master, _ratio_master

    if max_refinements < 0:
        raise ValueError("max_refinements must be nonnegative")
    started = perf_counter()
    oracle = oracle_class(p, gate="risk", core_point=False)
    delta = tolerance / 4
    history = EvaluationHistory(p, anchor_target=delta, forecast=False)
    if warm_support is None:
        means = [q @ x for q, x in zip(p.weights, p.samples)]
        scales = [np.sqrt(q @ (x-mean)**2)
                  for q, x, mean in zip(p.weights, p.samples, means)]
        scores = np.abs(means[0]-means[1]) / (scales[0]+scales[1]+1e-12)
        warm_support = tuple(sorted(np.argsort(-scores, kind="stable")[:p.budget]))
    support = tuple(sorted(warm_support))
    initial = p.mask(support).astype(float)
    if len(support) != p.budget:
        raise ValueError("warm_support must contain budget distinct valid coordinates")
    best_support, best_test = support, oracle._constant(support)
    lower, upper = 0., float(p.B)
    calls = refinements = master_calls = retries = 0
    oracle_seconds = master_seconds = 0.
    trace, retry_log = [], []

    def remaining():
        return max(0., time_limit-(perf_counter()-started))

    def done():
        return upper-lower <= tolerance+relative_tolerance*abs(upper)

    def finish(status, **detail):
        return SelectionResult(best_support, float(lower), float(upper), status,
                    perf_counter()-started, calls, len(history.points), trace, best_test,
                    dict(search="certified_global_cuts_v1", gate="risk", cut_mode="tangent",
                         use_fractions=bool(use_fractions), root_rounds=0, core_point=False,
                         fractional_root_calls=0, transferred_tests=0, predicted_upper_updates=0,
                         forecast_model=None, prediction_training_points=0,
                         oracle_seconds=oracle_seconds, master_seconds=master_seconds,
                         master_calls=master_calls, master_retries=retries,
                         master_retry_log=retry_log, refinements=refinements,
                         cut_count=len(history.cuts), cuts_added=len(history.cuts),
                         fraction_count=len(history.fractions), evaluations=history.records,
                         accurate_support_evaluations=history.accurate_count,
                         anchor_target=delta, master_target=delta, screening_margin=tolerance/2,
                         anchor_gaps=history.anchor_gaps, metric_weights=history.metric_weights.tolist(),
                         model_reuse=getattr(oracle, "reuse_statistics", {}),
                         **history.quantitative_bound(tolerance, delta), **detail))

    def evaluate(z, phase):
        nonlocal calls, refinements, oracle_seconds, upper, best_support, best_test
        key = p.support(z)
        if len(key) != p.budget or key in history.tests:
            raise ArithmeticError("evaluation requires a new full-budget binary support")
        before_upper, model_lower = upper, history.lower(z)
        local_test, cut, fraction = oracle._constant(key), None, None
        owned_lower, last_status, attempts = -np.inf, "time_limit", 0
        for attempt in range(max_refinements+1):
            if remaining() <= 0:
                break
            calls += 1
            refinements += int(attempt > 0)
            attempts += 1
            tick = perf_counter()
            result = oracle.solve(z, time_limit=remaining())
            oracle_seconds += perf_counter()-tick
            gradient = np.asarray(result.gradient, dtype=float)
            if gradient.shape != (p.p,) or not np.isfinite(gradient).all() or not np.isfinite(
                    [result.value, result.upper, result.intercept]).all():
                raise ArithmeticError("nonfinite or invalid oracle certificate")
            anchor = float(result.intercept+gradient @ z)
            if abs(anchor-result.value) > 1e-7*max(1., p.B):
                raise ArithmeticError("oracle value and owned tangent disagree")
            if result.test is not None:
                if tuple(result.test.support) != key or not np.isfinite(result.test.upper) or result.test.upper < 0:
                    raise ArithmeticError("invalid actual test for the evaluated support")
                if result.test.upper <= local_test.upper:
                    local_test = result.test
            if anchor > local_test.upper+1e-7*max(1., p.B):
                raise ArithmeticError("owned cut exceeds an actual feasible test")
            if cut is None or anchor > owned_lower:
                cut, owned_lower, fraction = (float(result.intercept), gradient.copy()), anchor, None
                if use_fractions:
                    cert = getattr(result, "certificate", {})
                    numerator = max(0., float(cert.get("entropy", 0.))-float(cert.get("margin", 0.)))
                    tau = float(cert.get("tau", 0.))
                    if numerator > 0 and tau > 0:
                        fraction = (numerator, tau, np.asarray(cert["mu"], dtype=float).copy())
            last_status = result.status
            gap = max(0., float(local_test.upper-owned_lower))
            trace.append(dict(phase="inner_attempt", evaluation_phase=phase,
                        seconds=perf_counter()-started, support=list(key), point=z.tolist(),
                        retained_lower=owned_lower, retained_actual_upper=float(local_test.upper),
                        lower=float(lower), upper=float(upper), anchor_gap=gap,
                        refinement=attempt, status=last_status, cut_added=False))
            if gap <= delta:
                break
            if hasattr(oracle, "tolerance"):
                oracle.tolerance *= .1
        if not attempts:
            raise TimeoutError("wall-time budget exhausted before the inner evaluation")
        gap = max(0., float(local_test.upper-owned_lower))
        accurate = gap <= delta
        history.add(z, test=local_test, intercept=cut[0], gradient=cut[1],
                    accurate=accurate, fraction=fraction)
        if local_test.upper <= upper:
            upper, best_support, best_test = float(local_test.upper), key, local_test
        trace.append(dict(phase=phase, seconds=perf_counter()-started,
                    support=list(key), point=z.tolist(), oracle_lower=owned_lower,
                    oracle_upper=float(local_test.upper), lower=float(lower), upper=float(upper),
                    pre_upper=float(before_upper), model_lower=model_lower,
                    selection_model_lower=model_lower, screening_threshold=before_upper-tolerance/2,
                    anchor_gap=gap, attempts=attempts, status=last_status, cut_added=True))
        if accurate:
            return None
        return ("time_limit" if remaining() <= 0 or last_status == "time_limit" else
                "oracle_failure" if last_status == "solver_error" else "oracle_accuracy_limit")

    try:
        if done():
            return finish("optimal")
        failure = evaluate(initial, "initial")
        if failure:
            return finish(failure, failed_support=list(support))
        for _ in range(max_iterations):
            if done():
                return finish("optimal")
            if remaining() <= 0:
                return finish("time_limit")
            tick = perf_counter()
            master = (_ratio_master(p, history.cuts, history.fractions, remaining(), upper,
                                    witness=p.mask(best_support)) if use_fractions else
                      _master(p, history.cuts, remaining(), integer=True))
            master_seconds += perf_counter()-tick
            master_calls += 1
            retry_count = int(getattr(master, "master_retries", 0))
            retries += retry_count
            retry_log.extend(getattr(master, "master_retry_log", []))
            bound = getattr(master, "mip_dual_bound", None) if master.status in (0, 1) else None
            if master.status == 0 and bound is None:
                bound = master.fun
            if bound is not None and np.isfinite(bound):
                candidate_lower = max(lower, float(bound)-1e-9*max(1., abs(bound), p.B))
                if candidate_lower > upper:
                    return finish("numerical_inconsistency", rejected_master_bound=candidate_lower)
                lower = candidate_lower
            trace.append(dict(phase="master", seconds=perf_counter()-started,
                        lower=float(lower), upper=float(upper), status=int(master.status),
                        master_retries=retry_count))
            if done():
                return finish("optimal")
            if master.status not in (0, 1) or master.x is None:
                return finish("time_limit" if master.status == 1 else "master_failure")
            raw = np.asarray(master.x[:p.p], dtype=float)
            z = np.rint(raw)
            if raw.shape != (p.p,) or not np.isfinite(raw).all() or np.max(abs(raw-z)) > 1e-5 or not np.all(
                    (z == 0) | (z == 1)) or z.sum() != p.budget:
                return finish("master_failure", message="no full-budget binary master candidate")
            model_lower = history.lower(z)
            candidate_gap = max(0., model_lower-lower)
            trace[-1].update(support=list(p.support(z)), point=z.tolist(), pre_upper=float(upper),
                             model_lower=model_lower, candidate_gap=candidate_gap,
                             screening_threshold=upper-tolerance/2)
            key = p.support(z)
            if candidate_gap > delta or model_lower > upper-tolerance/2 or key in history.tests:
                return finish("time_limit" if master.status == 1 else "master_accuracy_limit",
                              message="the master candidate fails accuracy, improvement, or uniqueness",
                              candidate_gap=candidate_gap, failed_support=list(key))
            failure = evaluate(z, "integer")
            if failure:
                return finish(failure, failed_support=list(key))
        return finish("optimal" if done() else "iteration_limit")
    except TimeoutError as error:
        return finish("time_limit", message=str(error))
    except (ArithmeticError, RuntimeError) as error:
        return finish("oracle_failure", message=str(error))
