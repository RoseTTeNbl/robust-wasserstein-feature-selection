"""History-guided acquisition with a global certification guard.

The acquisition ranks possible evaluations; it never supplies a lower bound.
Every transfer is a feasible original test, and every expensive evaluation
refits the full continuous problem. A verified width threshold connects the
search policy to the quantitative packing bound in the theory note.
"""
from time import perf_counter

import numpy as np

from .history import SearchHistory, proposal_pool


def solve_adaptive(p, *, oracle_class, gate, time_limit, root_rounds, core_point,
                   warm_support, tolerance, relative_tolerance, max_iterations,
                   cut_mode, proposal_count, information_fraction, certify_every,
                   max_refinements, random_state):
    from .solver import SelectionResult, _master, _ratio_master

    if not 0 < information_fraction < 1 or proposal_count < 1 or certify_every < 1:
        raise ValueError("invalid acquisition controls")
    if max_refinements < 0:
        raise ValueError("max_refinements must be nonnegative")
    started = perf_counter()
    oracle = oracle_class(p, gate=gate, core_point=core_point)
    history = SearchHistory(p, oracle, ratios=cut_mode == "ratio")
    rng = np.random.default_rng(random_state)
    if warm_support is None:
        means = [q @ x for q, x in zip(p.weights, p.samples)]
        scales = [np.sqrt(q @ ((x - mean)**2))
                  for q, x, mean in zip(p.weights, p.samples, means)]
        screening = np.abs(means[0] - means[1]) / (scales[0] + scales[1] + 1e-12)
        warm_support = tuple(sorted(np.argsort(-screening, kind="stable")[:p.budget]))
    support = tuple(sorted(warm_support))
    if len(support) != p.budget or len(set(support)) != len(support) or any(
            j < 0 or j >= p.p for j in support):
        raise ValueError("warm_support must contain budget distinct valid coordinates")
    best_test, best_support = oracle._constant(support), support
    lower, upper = 0., p.B
    calls = refinements = master_calls = 0
    oracle_seconds = master_seconds = root_bound = 0.
    retries, retry_log, trace = 0, [], []
    guard = None
    anchor_target = information_fraction * tolerance / 2

    def remaining():
        return max(0., time_limit - (perf_counter() - started))

    def target():
        return tolerance + relative_tolerance * abs(upper)

    def done():
        return upper - lower <= target()

    def consider(test):
        nonlocal upper, best_test, best_support
        if test is not None and test.upper < upper:
            upper, best_test, best_support = float(test.upper), test, test.support

    def finish(status, **detail):
        return SelectionResult(best_support, float(lower), float(upper), status,
                    perf_counter() - started, calls, len(history.tests), trace, best_test,
                    dict(gate=gate, cut_mode=cut_mode, search="adaptive_history_v1",
                         root_rounds=root_rounds, core_point=core_point, root_bound=root_bound,
                         oracle_seconds=oracle_seconds, master_seconds=master_seconds,
                         master_retries=retries, master_retry_log=retry_log,
                         master_calls=master_calls, proposal_count=proposal_count,
                         information_fraction=information_fraction,
                         transferred_tests=history.transfer_calls,
                         reused_fraction_cuts=history.extra_cuts, refinements=refinements,
                         accurate_support_evaluations=len(history.points),
                         anchor_target=anchor_target, anchor_gaps=history.anchor_gaps,
                         metric_weights=history.metric_weights.tolist(),
                         model_reuse=getattr(oracle, "reuse_statistics", {}), **detail))

    def evaluate(z, phase):
        nonlocal calls, refinements, oracle_seconds
        binary = bool(np.all((z == 0) | (z == 1)))
        attempts = max_refinements + 1 if binary else 1
        for attempt in range(attempts):
            if remaining() <= 0:
                raise TimeoutError("wall-time budget exhausted")
            calls += 1
            refinements += int(attempt > 0)
            tick = perf_counter()
            result = oracle.solve(z, time_limit=remaining())
            oracle_seconds += perf_counter() - tick
            gradient = np.asarray(result.gradient)
            if gradient.shape != (p.p,) or not np.isfinite(gradient).all():
                raise ArithmeticError("invalid oracle gradient")
            if not np.isfinite(result.value) or not np.isfinite(result.upper):
                raise ArithmeticError("nonfinite oracle bounds")
            if result.value > result.upper + 1e-7 * max(1., p.B):
                raise ArithmeticError("oracle lower certificate exceeds its feasible upper bound")
            history.add(z, result)
            consider(result.test)
            gap = None
            if binary:
                transferred = history.upper(z, remaining)
                consider(transferred)
                if history.lower(z) > transferred.upper + 1e-7 * max(1., p.B):
                    raise ArithmeticError("history lower model exceeds a transferred feasible bound")
                gap = max(0., transferred.upper - history.lower(z))
            trace.append(dict(phase=phase, seconds=perf_counter() - started,
                        support=list(p.support(z)) if binary else None, point=z.tolist(),
                        oracle_lower=float(result.value), oracle_upper=float(result.upper),
                        lower=float(lower), upper=float(upper), anchor_gap=gap,
                        refinement=attempt, status=result.status))
            if not binary or gap <= anchor_target:
                if binary:
                    # This evaluation may pair the new certificate with an
                    # even better historical feasible score at the same anchor.
                    history.tests[p.support(z)] = transferred
                    history.upper_cache.clear()
                    history.completed(z, gap)
                return True
            # Reusing the parameterized model makes this a real refinement.
            oracle.tolerance *= .1
        return False

    def master(integer):
        nonlocal lower, master_seconds, master_calls, retries, root_bound
        if remaining() <= 0:
            raise TimeoutError("wall-time budget exhausted")
        tick = perf_counter()
        result = (_ratio_master(p, history.cuts, history.ratios, remaining(), upper)
                  if integer and cut_mode == "ratio" else
                  _master(p, history.cuts, remaining(), integer=integer))
        master_seconds += perf_counter() - tick
        master_calls += 1
        count = int(getattr(result, "master_retries", 0))
        retries += count
        retry_log.extend(getattr(result, "master_retry_log", []))
        bound = (getattr(result, "mip_dual_bound", None) if integer else
                 result.fun if result.status == 0 else None)
        if result.status == 0 and bound is None:
            bound = result.fun
        if result.status in (0, 1) and bound is not None and np.isfinite(bound):
            valid_bound = float(bound) - 1e-9 * max(1., abs(bound), p.B)
            lower = max(lower, valid_bound)
            if not integer:
                root_bound = max(root_bound, valid_bound)
        trace.append(dict(phase="master" if integer else "root_master",
                          seconds=perf_counter() - started, lower=float(lower),
                          upper=float(upper), status=int(result.status), master_retries=count))
        if lower > upper + 1e-7 * max(1., p.B):
            raise ArithmeticError("master lower bound exceeds a feasible upper bound")
        return result

    try:
        if done():
            return finish("optimal")
        initial = p.mask(support).astype(float)
        accurate = evaluate(initial, "initial")
        if done():
            return finish("optimal")
        if p.budget in (0, p.p):
            lower = max(lower, history.lower(initial))
            return finish("optimal" if done() else "oracle_accuracy_limit")
        if not accurate:
            return finish("oracle_accuracy_limit", failed_support=list(support))
        if root_rounds:
            evaluate(np.full(p.p, p.budget / p.p), "root")
        for _ in range(max(0, root_rounds - 1)):
            result = master(False)
            if done():
                return finish("optimal")
            if result.status != 0 or result.x is None:
                break
            evaluate(np.clip(result.x[:p.p], 0, 1), "root")
        for iteration in range(max_iterations):
            if remaining() <= 0:
                return finish("time_limit")
            points = proposal_pool(p, history, best_support, count=proposal_count,
                                   rng=rng, guard=guard)
            history.densify(points)
            choices = []
            for z in points:
                test = history.upper(z, remaining)
                consider(test)
                model = history.lower(z)
                width = max(0., test.upper - model)
                choices.append((z, model, width))
            if done():
                return finish("optimal")
            gap = upper - lower
            eligible = [item for item in choices if item[1] < upper and
                        item[2] > information_fraction * gap]
            if iteration % certify_every == 0 or not eligible:
                result = master(True)
                if done():
                    return finish("optimal")
                if result.status not in (0, 1) or result.x is None:
                    return finish("time_limit" if result.status == 1 else "master_failure")
                raw = result.x[:p.p]
                z_guard = np.rint(raw)
                if np.max(abs(raw-z_guard)) > 1e-5 or z_guard.sum() != p.budget:
                    return finish("master_failure", message="no feasible binary master candidate")
                guard = p.support(z_guard)
                test = history.upper(z_guard, remaining)
                consider(test)
                if done():
                    return finish("optimal")
                choices.append((z_guard, history.lower(z_guard),
                                max(0., test.upper-history.lower(z_guard))))
                gap = upper - lower
                eligible = [item for item in choices if item[1] < upper and
                            item[2] > information_fraction * gap]
            if not eligible:
                return finish("master_accuracy_limit", message="no certified information progress")
            chosen, model, width = max(eligible, key=lambda item:
                                      ((upper-item[1])*item[2], -sum(item[0])))
            distances = [history.distance(chosen, old) for old in history.points]
            trace.append(dict(phase="acquisition", seconds=perf_counter()-started,
                        support=list(p.support(chosen)), lower=float(lower), upper=float(upper),
                        model_lower=model, model_upper=model+width, width=width,
                        information_threshold=information_fraction*gap,
                        acquisition=(upper-model)*width,
                        nearest_history_distance=min(distances, default=None),
                        candidates=len(choices), eligible=len(eligible)))
            accurate = evaluate(chosen, "adaptive")
            if done():
                return finish("optimal")
            if not accurate:
                return finish("oracle_accuracy_limit", failed_support=list(p.support(chosen)))
        return finish("iteration_limit")
    except TimeoutError as error:
        return finish("time_limit", message=str(error))
    except (ArithmeticError, RuntimeError) as error:
        return finish("oracle_failure", message=str(error))
